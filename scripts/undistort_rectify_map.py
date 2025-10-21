import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Image/camera setup
camera_width = 640
camera_height = 480
fx, fy, cx, cy = 100, 100, camera_width / 2, camera_height / 2

# Camera intrinsic matrix
cameraMatrix = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype=np.float64)

# Distortion coefficients: [k1, k2, p1, p2, k3]
# Try increasing these to exaggerate the effect
distCoeffs = np.array([-0.3, 0.1, 0.0, 0.0, 0.0], dtype=np.float64)

# Create a simple 3D grid of world points (Z = constant)
nx, ny = 10, 8
x = np.linspace(-3, 3, nx)
y = np.linspace(-2, 2, ny)
X, Y = np.meshgrid(x, y)
Z = np.ones_like(X) * 5.0  # all points at Z=5 meters
points_3D = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

# Define camera pose (rotation + translation)
R_camera_world = np.eye(3)
t_camera_world = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

# Convert to Rodrigues rotation vector
rvec, _ = cv.Rodrigues(R_camera_world)
tvec = t_camera_world.flatten()

# Project *without* distortion
points_undistorted, _ = cv.projectPoints(
    objectPoints=points_3D,
    rvec=rvec,
    tvec=tvec,
    cameraMatrix=cameraMatrix,
    distCoeffs=None
)

# Project *with* distortion
points_distorted, _ = cv.projectPoints(
    objectPoints=points_3D,
    rvec=rvec,
    tvec=tvec,
    cameraMatrix=cameraMatrix,
    distCoeffs=distCoeffs
)

# Convert to integer pixel coordinates
points_undistorted = np.squeeze(points_undistorted)
points_distorted = np.squeeze(points_distorted)

# Create blank canvas
img_distorted = np.ones((camera_height, camera_width, 3), np.uint8) * 255
img_undistorted = np.ones_like(img_distorted) * 255

# Draw grid points
for p in points_distorted.astype(int):
    cv.circle(img_distorted, tuple(p), 3, (0, 0, 255), -1)  # red distorted

for p in points_undistorted.astype(int):
    cv.circle(img_undistorted, tuple(p), 3,
              (0, 255, 0), -1)  # green undistorted

# Undistort the distorted image using OpenCV
undistorted_from_opencv = cv.undistort(img_distorted, cameraMatrix, distCoeffs)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Projected with Distortion")
plt.imshow(cv.cvtColor(img_distorted, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Projected without Distortion")
plt.imshow(cv.cvtColor(img_undistorted, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Corrected using cv.undistort()")
plt.imshow(cv.cvtColor(undistorted_from_opencv, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
