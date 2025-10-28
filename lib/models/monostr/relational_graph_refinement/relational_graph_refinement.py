import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
import json
from torch.utils.data import Dataset

class GraphRefineDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.samples = []
        for obj in self.data:
            bbox3d = torch.tensor(obj["bbox3d"], dtype=torch.float32)
            keypoints = torch.tensor(obj["keypoints"], dtype=torch.float32).flatten()
            theta = torch.tensor([obj["theta"]], dtype=torch.float32)
            crop_bbox = torch.tensor(obj["crop_bbox"], dtype=torch.float32)

            feature = torch.cat([bbox3d, keypoints, theta], dim=0)  # (32,)
            box_center = bbox3d[:3]  # x, y, z

            self.samples.append({
                "feature": feature,
                "center": box_center,
                "original_bbox3d": bbox3d,
                "crop_bbox": crop_bbox,
                "image_id": obj["image_id"]
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    

def save_kitti_format(refined_bbox, crop_bbox, image_id, save_dir):
    """
    Save one object in KITTI label format
    """
    # class truncation occlusion alpha x1 y1 x2 y2 h w l x y z ry
    obj_class = "Car"
    truncated = 0.00
    occluded = 0
    alpha = -1.67  # 임시값. 추후 orientation으로 계산 가능
    x1, y1, x2, y2 = crop_bbox.tolist()
    x, y, z, w, h, l, ry = refined_bbox.tolist()

    line = f"{obj_class} {truncated:.2f} {occluded} {alpha:.2f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f}"

    save_path = os.path.join(save_dir, image_id.replace(".png", ".txt"))
    with open(save_path, 'a') as f:
        f.write(line + "\n")



# GCN Layer 정의
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # adj: (N, N)
        agg = torch.matmul(adj, x)  # (N, in_dim)
        out = self.linear(agg)
        return F.relu(out)

def build_distance_adj_matrix(pos, threshold=3.0):
    """
    pos: (N, 3) - 노드 위치 (x, y, z)
    return: (N, N) adjacency matrix where edge exists if dist < threshold
    """
    N = pos.size(0)
    dists = torch.cdist(pos, pos)  # (N, N)
    adj = (dists < threshold).float()  # threshold 이내는 1, 외는 0
    adj.fill_diagonal_(0)  # self-loop 제거
    deg = adj.sum(1, keepdim=True)
    adj = adj / (deg + 1e-6)  # Normalize
    return adj

# 전체 Graph 기반 Refinement 모듈
class GraphRefinementModule(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, out_dim=7):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, out_dim)  # predict residual box delta

    def forward(self, box_feat_32d, box_center_xyz):
        """
        box_feat_32d: (N, 32) - [bbox7d, keypoints24d, angle1d]
        box_center_xyz: (N, 3) - used to build graph edges (e.g., [x, y, z])
        """
        # Build Graph
        adj = build_distance_adj_matrix(box_center_xyz, threshold=3.0)  

        # Node feature transform
        x = F.relu(self.linear1(box_feat_32d))  # (N, hidden_dim)
        x = F.relu(self.linear2(x))             # (N, hidden_dim)

        # Graph Interaction
        x = self.gcn1(x, adj)                   # (N, hidden_dim)
        x = self.gcn2(x, adj)                   # (N, hidden_dim)

        # Predict delta
        delta = self.regressor(x)               # (N, 7)
        return delta
    





dataset = GraphRefineDataset("/path/to/your.json")
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

model = GraphRefinementModule()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    for batch in loader:
        feat = batch["feature"]         # (B, 32)
        center = batch["center"]        # (B, 3)
        gt_box = batch["original_bbox3d"]  # (B, 7)

        delta = model(feat, center)
        refined = gt_box + delta

        loss = F.smooth_l1_loss(refined, gt_box)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


model.eval()
with torch.no_grad():
    for sample in dataset:
        delta = model(sample["feature"].unsqueeze(0), sample["center"].unsqueeze(0))
        refined = sample["original_bbox3d"] + delta.squeeze(0)
        save_kitti_format(refined, sample["crop_bbox"], sample["image_id"], "./output_labels")