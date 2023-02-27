import numpy as np
import cv2

# Load the calibration parameters
K = np.loadtxt('camera_matrix.txt', delimiter=',')
D = np.loadtxt('distortion_coefficients.txt', delimiter=',')

# Load the correspondences
pts1 = np.loadtxt('image_points_1.txt', delimiter=',')
pts2 = np.loadtxt('image_points_2.txt', delimiter=',')

# Undistort the correspondences
pts1_undist = cv2.fisheye.undistortPoints(pts1, K, D)
pts2_undist = cv2.fisheye.undistortPoints(pts2, K, D)

# Compute the essential matrix
F, mask = cv2.findFundamentalMat(pts1_undist, pts2_undist, cv2.FM_RANSAC, 0.1)

# Compute the camera matrices
E = K.T @ F @ K
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P2s = cv2.decomposeEssentialMat(E)

# Choose the correct camera matrix
for i in range(4):
    R = P2s[i][:3, :3]
    t = P2s[i][:, 3]
    if np.linalg.det(R) > 0:
        P2 = np.hstack((R, t.reshape(3, 1)))
        break

# Print the results
print('Essential matrix:\n', E)
print('Camera matrix 1:\n', K @ P1)
print('Camera matrix 2:\n', K @ P2)

