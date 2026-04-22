import cv2
import torch
from lightglue import SuperPoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = "../images/correspondences_matching/000000.png"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
assert img is not None, f"{image_path} not found"
tensor = torch.from_numpy(img).float()[None, None] / 255.0
tensor = tensor.to(device)

superpoint = SuperPoint(max_num_keypoints=2048).eval().to(device)

with torch.inference_mode():
    out = superpoint({"image": tensor})

keypoints = out["keypoints"][0].cpu().numpy()
scores = out["keypoint_scores"][0].cpu().numpy()
descriptors = out["descriptors"][0].cpu().numpy()

print(f"keypoints: {keypoints.shape}")
print(f"scores:    {scores.shape}")
print(f"desc:      {descriptors.shape}")

vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for x, y in keypoints.astype(int):
    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
cv2.imshow("superpoint_keypoints.jpg", vis)
cv2.waitKey(0)
