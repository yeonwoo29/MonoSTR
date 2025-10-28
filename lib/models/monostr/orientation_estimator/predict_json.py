# predict_json.py
import torch
import json
import math
from torch.utils.data import DataLoader
from theta_regression import KeypointThetaDataset, MLPThetaPredictor

def denormalize_theta(theta):
    # 모델 출력 [-1, 1] → 라디안 [-pi, pi]
    return theta * math.pi

def predict_and_save(model_path, input_json, output_json, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = KeypointThetaDataset(input_json)
    loader = DataLoader(dataset, batch_size=batch_size)

    # Load model
    model = MLPThetaPredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    total_samples = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            pred = model(x)
            pred_angle = denormalize_theta(pred).cpu().numpy()  # 라디안 단위 복원

            for i in range(len(pred_angle)):
                item = dataset.data[total_samples + i].copy()
                # theta를 라디안 단위 예측값으로 교체
                item["theta"] = float(pred_angle[i])
                results.append(item)

            total_samples += len(pred_angle)

    # JSON 저장
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved predicted JSON to {output_json}")


if __name__ == "__main__":
    model_path = "mlp_theta_predictor.pth"   # 학습된 모델 경로
    input_json = "keypoints_with_theta_train.json"  # 입력 JSON
    output_json = "keypoints_with_theta_pred_train.json"  # 예측 JSON 저장 경로

    predict_and_save(model_path, input_json, output_json)
