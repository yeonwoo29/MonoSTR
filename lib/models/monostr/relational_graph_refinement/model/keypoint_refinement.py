import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ============================================================
# 1. IoU 계산 (2D 바운딩 박스)
# ============================================================
def iou_2d(boxA, boxB):
    """
    boxA, boxB: [x1,y1,x2,y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# ============================================================
# 2. KITTI 파일 로더 (prediction & label 공통)
# ============================================================
def load_kitti_file(path, with_score=True):
    """
    path: KITTI label or prediction txt
    return: list of dicts [{'box2d','box3d','score'}]
    """
    dets = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:  # invalid line
                continue
            cls = parts[0]
            if cls != 'Car':  # Car만 처리
                continue
            x1, y1, x2, y2 = map(float, parts[4:8])
            h, w, l = map(float, parts[8:11])
            x, y, z, ry = map(float, parts[11:15])
            score = float(parts[15]) if with_score and len(parts) > 15 else 1.0
            dets.append({
                "box2d":[x1,y1,x2,y2],
                "box3d":[x,y,z,w,h,l,ry],
                "score":score
            })
    return dets


# ============================================================
# 3. Keypoint set 로더 (예시: json or npy → KITTI와 동일 구조로 변환했다고 가정)
# ============================================================
import json
def load_keypoint_json(json_path, image_id):
    """
    json_path: keypoints_with_theta_pred_train.json
    image_id:  "000123" (확장자 제거)
    return: [{'box2d','keypoints','ry'}]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    kp_sets = []
    for obj in data:
        # image_id 매칭
        if obj["image_id"].split('.')[0] != image_id:
            continue
        kp_sets.append({
            "box2d": obj["crop_bbox"],                # [x1,y1,x2,y2]
            "keypoints": np.array(obj["keypoints"]),  # (K,2)
            "ry": obj["theta"]
        })
    return kp_sets

# ============================================================
# 4. 매칭 함수
# ============================================================
def build_samples(output_dir, kp_json_path, label_dir, iou_thresh=0.8):
    samples = []
    out_files = sorted(glob.glob(os.path.join(output_dir, "*.txt")))

    for out_path in out_files:
        image_id = os.path.splitext(os.path.basename(out_path))[0]
        label_path = os.path.join(label_dir, image_id + ".txt")

        if not os.path.exists(label_path):
            continue

        outputs = load_kitti_file(out_path, with_score=True)
        kp_sets = load_keypoint_json(kp_json_path, image_id)  # JSON에서 해당 image_id 가져오기
        labels  = load_kitti_file(label_path, with_score=False)

        for out in outputs:
            for kp in kp_sets:
                iou = iou_2d(out["box2d"], kp["box2d"])
                if iou >= iou_thresh:
                    # label에서 가장 IoU 큰 것 선택
                    gt = max(labels, key=lambda g: iou_2d(out["box2d"], g["box2d"]))
                    samples.append({
                        "init_3d": np.array(out["box3d"], dtype=np.float32),
                        "keypoints": np.array(kp["keypoints"], dtype=np.float32).flatten(),
                        "ry_keypoint": np.array([kp["ry"]], dtype=np.float32),
                        "gt_3d": np.array(gt["box3d"], dtype=np.float32),
                    })
    return samples

# ============================================================
# 5. Dataset & Model (앞에서 정의한 그대로)
# ============================================================
class RefineDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "init_3d": torch.tensor(s["init_3d"]),
            "keypoints": torch.tensor(s["keypoints"]),
            "ry_keypoint": torch.tensor(s["ry_keypoint"]),
            "gt_3d": torch.tensor(s["gt_3d"]),
        }

class OutputKeypointRegressor(nn.Module):
    def __init__(self, kp_dim, hidden_dim=128):
        super().__init__()
        input_dim = 7 + kp_dim + 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 7)

    def forward(self, init_3d, keypoints, yaw):
        x = torch.cat([init_3d, keypoints, yaw], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        delta = self.fc_out(x)
        return delta

def train_regressor(samples, kp_dim=24, epochs=10, batch_size=16, lr=1e-3,
                    save_path="../outputs/regressor_best.pth"):
    dataset = RefineDataset(samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = OutputKeypointRegressor(kp_dim=kp_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            init_3d = batch["init_3d"]
            keypoints = batch["keypoints"]
            ry_keypoint = batch["ry_keypoint"]
            gt_3d = batch["gt_3d"]

            delta_gt = gt_3d - init_3d
            delta_pred = model(init_3d, keypoints, ry_keypoint)

            loss = F.smooth_l1_loss(delta_pred, delta_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        # ✅ Best 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ↳ 새로운 best 모델 저장됨: {save_path} (loss={best_loss:.4f})")

    print("학습 완료. Best Loss =", best_loss)
    return model



def resume_training(samples, kp_dim, epochs=10, batch_size=16, lr=1e-3,
                    resume_path="../outputs/regressor_best.pth",
                    save_path="../outputs/regressor_best.pth"):
    dataset = RefineDataset(samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = OutputKeypointRegressor(kp_dim=kp_dim)

    # ✅ 기존 checkpoint 불러오기
    if os.path.exists(resume_path):
        model.load_state_dict(torch.load(resume_path))
        print(f"기존 모델 로드 완료: {resume_path}")
    else:
        print("⚠️ resume_path 에 파일이 없어 새로 학습을 시작합니다.")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            init_3d = batch["init_3d"]
            keypoints = batch["keypoints"]
            ry_keypoint = batch["ry_keypoint"]
            gt_3d = batch["gt_3d"]

            delta_gt = gt_3d - init_3d
            delta_pred = model(init_3d, keypoints, ry_keypoint)

            loss = F.smooth_l1_loss(delta_pred, delta_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Resume Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        # ✅ best 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ↳ 새로운 best 모델 저장됨: {save_path} (loss={best_loss:.4f})")

    print("추가 학습 완료. Best Loss =", best_loss)
    return model




# 실행 예시
output_dir = "../dataset/merge_output_train"         # detector 결과 (KITTI txt)
kp_json_path = "../dataset/keypoints_with_theta_pred_train.json"  # keypoint + theta JSON
label_dir    = "../dataset/label_2_train"         # KITTI GT labels

samples = build_samples(output_dir, kp_json_path, label_dir, iou_thresh=0.75)
print(f"총 {len(samples)} 개의 학습 샘플 생성됨")





#학습 & best.pth 저장
model = train_regressor(
    samples,
    kp_dim=len(samples[0]["keypoints"]),
    epochs=10,
    save_path="../outputs/regressor_best.pth"
)





# # 기존 모델 이어서 학습
# model = resume_training(
#     samples,
#     kp_dim=len(samples[0]["keypoints"]),
#     epochs=500,   # 추가 학습 epoch
#     resume_path="../outputs/regressor_best.pth",
#     save_path="../outputs/regressor_best.pth"
# )



# def refine_outputs(model, output_dir, kp_json_path, save_dir, iou_thresh=0.5):
#     os.makedirs(save_dir, exist_ok=True)
#     out_files = sorted(glob.glob(os.path.join(output_dir, "*.txt")))

#     for out_path in out_files:
#         image_id = os.path.splitext(os.path.basename(out_path))[0]

#         # 원래 detection 결과
#         outputs = load_kitti_file(out_path, with_score=True)
#         if len(outputs) == 0:
#             # 비어있으면 빈 txt 저장
#             open(os.path.join(save_dir, image_id + ".txt"), "w").close()
#             continue

#         # keypoints set
#         kp_sets = load_keypoint_json(kp_json_path, image_id)

#         refined_dets = []
#         for out in outputs:
#             best_match = None
#             best_iou = 0.0
#             for kp in kp_sets:
#                 iou = iou_2d(out["box2d"], kp["box2d"])
#                 if iou > best_iou:
#                     best_iou = iou
#                     best_match = kp

#             if best_match is not None and best_iou >= iou_thresh:
#                 # refine 적용
#                 init_3d = torch.tensor(out["box3d"], dtype=torch.float32).unsqueeze(0)
#                 keypoints = torch.tensor(best_match["keypoints"], dtype=torch.float32).flatten().unsqueeze(0)
#                 ry_keypoint = torch.tensor([best_match["ry"]], dtype=torch.float32).unsqueeze(0)

#                 delta = model(init_3d, keypoints, ry_keypoint).detach().cpu().numpy()[0]
#                 refined_box = init_3d.numpy()[0] + delta*0.5
#             else:
#                 # 매칭 안되면 그대로 사용
#                 refined_box = out["box3d"]

#             refined_dets.append({
#                 "cls": "Car",
#                 "box2d": out["box2d"],
#                 "box3d": refined_box,
#                 "score": out["score"]
#             })

#         # KITTI 형식으로 저장
#         save_path = os.path.join(save_dir, image_id + ".txt")
#         with open(save_path, "w") as f:
#             for det in refined_dets:
#                 x1, y1, x2, y2 = det["box2d"]
#                 x, y, z, w, h, l, ry = det["box3d"]
#                 score = det["score"]
#                 line = f"Car 0.00 0 -1.67 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} " \
#                        f"{h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f} {score:.3f}\n"
                # f.write(line)

    # print(f"✅ Refined 결과 저장 완료: {save_dir}")




def refine_outputs(model, output_dir, kp_json_path, save_dir, iou_thresh=0.5):
    os.makedirs(save_dir, exist_ok=True)
    out_files = sorted(glob.glob(os.path.join(output_dir, "*.txt")))

    for out_path in out_files:
        image_id = os.path.splitext(os.path.basename(out_path))[0]

        # 원래 detection 결과
        outputs = load_kitti_file(out_path, with_score=True)
        if len(outputs) == 0:
            # 비어있으면 빈 txt 저장
            open(os.path.join(save_dir, image_id + ".txt"), "w").close()
            continue

        # keypoints set
        kp_sets = load_keypoint_json(kp_json_path, image_id)

        refined_dets = []
        for out in outputs:
            best_match = None
            best_iou = 0.0
            for kp in kp_sets:
                iou = iou_2d(out["box2d"], kp["box2d"])
                if iou > best_iou:
                    best_iou = iou
                    best_match = kp

            if best_match is not None and best_iou >= iou_thresh:
                # refine 적용
                init_3d = torch.tensor(out["box3d"], dtype=torch.float32).unsqueeze(0)
                keypoints = torch.tensor(best_match["keypoints"], dtype=torch.float32).flatten().unsqueeze(0)
                ry_keypoint = torch.tensor([best_match["ry"]], dtype=torch.float32).unsqueeze(0)

                delta = model(init_3d, keypoints, ry_keypoint).detach().cpu().numpy()[0]
                refined_box = init_3d.numpy()[0] + delta*0.3
            else:
                # 매칭 안되면 그대로 사용
                refined_box = out["box3d"]

            refined_dets.append({
                "cls": "Car",
                "box2d": out["box2d"],
                "box3d": refined_box,
                "score": out["score"]
            })

        # KITTI 형식으로 저장
        save_path = os.path.join(save_dir, image_id + ".txt")
        with open(save_path, "w") as f:
            for det in refined_dets:
                x1, y1, x2, y2 = det["box2d"]
                x, y, z, w, h, l, ry = det["box3d"]
                score = det["score"]
                line = f"Car 0.00 0 -1.67 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} " \
                       f"{h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f} {score:.3f}\n"
                f.write(line)

    print(f"✅ Refined 결과 저장 완료: {save_dir}")




    # 1. 모델 불러오기
model = OutputKeypointRegressor(kp_dim=24)
model.load_state_dict(torch.load("../outputs/regressor_best.pth"))
model.eval()

# 2. refinement 실행
refine_outputs(
    model,
    output_dir="../dataset/merge_output_val",
    kp_json_path="../dataset/keypoints_with_theta_pred_val.json",
    save_dir="../dataset/keypoint_refined_output_val",
    iou_thresh=0.5
)
