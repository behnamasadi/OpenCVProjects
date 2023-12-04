import cv2
import numpy as np

# # Load two consecutive frames
# prev_img = cv2.imread('frame1.jpg', cv2.IMREAD_GRAYSCALE)
# next_img = cv2.imread('frame2.jpg', cv2.IMREAD_GRAYSCALE)


prev_img = cv2.imread(
    '/home/behnam/Pictures/colmap_projects/GroupeE_Maigrauge/images/00002.png', cv2.IMREAD_GRAYSCALE)
next_img = cv2.imread(
    '/home/behnam/Pictures/colmap_projects/GroupeE_Maigrauge/images/00003.png', cv2.IMREAD_GRAYSCALE)

# Create the DualTVL1 optical flow object
dualTVL1 = cv2.optflow.DualTVL1OpticalFlow_create()

# Calculate optical flow
flow = dualTVL1.calc(prev_img, next_img, None)

# Visualizing the optical flow
hsv = np.zeros((prev_img.shape[0], prev_img.shape[1], 3), dtype=np.uint8)
hsv[..., 1] = 255

mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('Optical Flow', rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # Load two consecutive frames
# prev_img = cv2.imread('frame1.jpg', cv2.IMREAD_GRAYSCALE)
# next_img = cv2.imread('frame2.jpg', cv2.IMREAD_GRAYSCALE)

# # Create the DualTVL1 optical flow object
# dualTVL1 = cv2.DualTVL1OpticalFlow_create()

# # Calculate optical flow
# flow = dualTVL1.calc(prev_img, next_img, None)

# # Extract corresponding points from the optical flow
# h, w = flow.shape[:2]
# flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h))) + flow.reshape(-1, 2)
# mask = (0 <= flow_map[:, 0]) & (flow_map[:, 0] < h) & (0 <= flow_map[:, 1]) & (flow_map[:, 1] < w)
# pts_prev = np.column_stack((np.tile(np.arange(w), h), np.repeat(np.arange(h), w)))[mask]
# pts_next = flow_map[mask]

# # Assume camera intrinsics are known. If not, you'll need them for accurate pose estimation.
# focal_length = 1.0  # this is a placeholder value, you should use the actual value
# principal_point = (w / 2, h / 2)  # center of the image

# # Estimate the Essential matrix
# E, mask = cv2.findEssentialMat(pts_next, pts_prev, focal=focal_length, pp=principal_point, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# # Decompose the Essential matrix to get the rotation and translation
# _, R, t, mask = cv2.recoverPose(E, pts_next, pts_prev, focal=focal_length, pp=principal_point)

# print("Rotation matrix:")
# print(R)
# print("Translation vector:")
# print(t)
