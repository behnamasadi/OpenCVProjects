import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lightglue import SuperPoint

device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = "../images/correspondences_matching/000000.png"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
assert img is not None, f"{image_path} not found"
tensor = torch.from_numpy(img).float()[None, None].to(device) / 255.0

sp = SuperPoint(max_num_keypoints=2048).eval().to(device)

# Capture the 65-channel detector-head logits (pre-softmax) via a forward hook.
logits = {}
sp.convPb.register_forward_hook(lambda m, i, o: logits.setdefault("x", o))

with torch.inference_mode():
    out = sp({"image": tensor})

# Reconstruct the dense H x W keypoint heatmap (see SUPERPOINT.md §2.2):
# softmax over 65 -> drop dustbin -> depth-to-space with stride 8.
probs = F.softmax(logits["x"], dim=1)[:, :-1]            # [1, 64, H/8, W/8]
heatmap = F.pixel_shuffle(probs, 8)[0, 0].cpu().numpy()  # [H, W]

keypoints = out["keypoints"][0].cpu().numpy()
descriptors = out["descriptors"][0].cpu().numpy()
print(f"heatmap:     {heatmap.shape}  max={heatmap.max():.4f}")
print(f"keypoints:   {keypoints.shape}")
print(f"descriptors: {descriptors.shape}")

heat_vis = cv2.applyColorMap(
    (heatmap / heatmap.max() * 255).astype(np.uint8), cv2.COLORMAP_JET
)
kp_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for x, y in keypoints.astype(int):
    cv2.circle(kp_vis, (x, y), 2, (0, 255, 0), -1)

cv2.imshow("network output: keypoint heatmap", heat_vis)
cv2.imshow("after NMS + threshold: keypoints", kp_vis)
cv2.waitKey(0)
