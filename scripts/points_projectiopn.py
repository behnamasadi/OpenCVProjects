import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from plot_frame_poses_in_3d import plot_frame
from quaternion_utils import quaternion_to_rotation_matrix


point_in_world = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [-1, 0, 0], [0, 2, 0], [0, -1, 0]])


# Example usage:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# world
length = 1.0
x_world, y_world, z_world = 0, 0, 0
q_world = [1, 0, 0, 0]  # q_w, q_x, q_y, q_z

plot_frame(ax, x_world, y_world, z_world, q_world, length, "World")

# Camera
x_camera_1, y_camera_1, z_camera_1 = 0, 0, -2
q_camera_1 = [1, 0, 0, 0]  # Identity quaternion, no rotation

plot_frame(ax, x_camera_1, y_camera_1, z_camera_1,
           q_camera_1, length, "Camera")

# Plot the world points
ax.scatter(point_in_world[:, 0], point_in_world[:, 1], point_in_world[:, 2],
           c='red', marker='o', s=50, label='World Points')

# Optionally, add labels for each point
for i, point in enumerate(point_in_world):
    ax.text(point[0], point[1], point[2], f'  P{i}', fontsize=8)


# Pose of camera in world frame (T_world_camera)
# Camera position in world coordinates
# Translation: (0, 0, -2)
t_world_camera = np.array([[x_camera_1], [y_camera_1], [z_camera_1]])

# Camera orientation in world - convert quaternion to rotation matrix
R_world_camera = quaternion_to_rotation_matrix(q_camera_1)

# 4x4 transformation matrix: camera pose in world frame
T_world_camera = np.eye(4)
T_world_camera[0:3, 0:3] = R_world_camera
T_world_camera[0:3, 3:4] = t_world_camera

print("Pose of Camera in World Frame (T_world_camera):")
print(T_world_camera)
print()

# Pose of world in camera frame (T_camera_world)
# This is the inverse of T_world_camera
# For rigid body transformations: T_camera_world = [R^T | -R^T * t]
R_camera_world = R_world_camera.T
t_camera_world = -R_camera_world @ t_world_camera

T_camera_world = np.eye(4)
T_camera_world[0:3, 0:3] = R_camera_world
T_camera_world[0:3, 3:4] = t_camera_world

print("Pose of World in Camera Frame (T_camera_world):")
print(T_camera_world)
print()

# Transform points from world to camera frame
# Add homogeneous coordinate (append 1 to each point)
points_homogeneous = np.hstack(
    [point_in_world, np.ones((point_in_world.shape[0], 1))])

# Transform: P_camera = T_camera_world * P_world
points_in_camera = (T_camera_world @ points_homogeneous.T).T

print("Points in World Frame:")
print(point_in_world)
print()
print("Points in Camera Frame:")
print(points_in_camera[:, 0:3])  # Remove homogeneous coordinate for display
print()


# Project points onto camera image plane
camera_width = 640
camera_height = 480
fx, fy, cx, cy = 100, 100, camera_width/2, camera_height/2
cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Convert rotation matrix to rotation vector (Rodrigues formula)
rvec, _ = cv.Rodrigues(R_camera_world)

# Flatten tvec to 1D array for projectPoints
tvec = t_camera_world.flatten()

# Ensure points are in float32 or float64 format and proper shape
objectPoints = point_in_world.astype(np.float64)

# Project points: transforms world points to image plane
# cv.projectPoints needs rvec (3x1), tvec (3x1), and expects (N,3) or (N,1,3) points
imagePoints, _ = cv.projectPoints(
    objectPoints=objectPoints,
    rvec=rvec,
    tvec=tvec,
    cameraMatrix=cameraMatrix,
    distCoeffs=None
)

# Reshape imagePoints from (N, 1, 2) to (N, 2)
imagePoints = imagePoints.reshape(-1, 2)

print("Projected Image Points (pixels):")
for i, pt in enumerate(imagePoints):
    print(f"P{i}: ({pt[0]:.2f}, {pt[1]:.2f})")
print()


# --- Backproject 2D image points to 3D rays ---
# Inverse of camera matrix (intrinsic matrix)
K_inv = np.linalg.inv(cameraMatrix)

# Convert image points to homogeneous coordinates
# Method 1: Manual (using numpy)
imagePoints_homogeneous = np.hstack(
    [imagePoints, np.ones((imagePoints.shape[0], 1))])

# Method 2: Using OpenCV function (alternative)
# imagePoints_homogeneous = cv.convertPointsToHomogeneous(imagePoints).reshape(-1, 3)

# Backproject to normalized image coordinates (rays in camera frame)
# rays = K^(-1) * [u, v, 1]^T
rays_camera = (K_inv @ imagePoints_homogeneous.T).T

print("Backprojected Rays in Camera Frame (normalized):")
print(rays_camera)
print()

# --- Recover 3D points at depth Z = 1 ---
# For each ray, scale it so that Z = 1
# ray = [X/Z, Y/Z, 1], to get point at Z=1: [X/Z, Y/Z, 1]
# The rays are already normalized, so we just use them directly
points_3d_recovered_z1 = rays_camera.copy()

print("Recovered 3D Points at Z = 1 in Camera Frame:")
print(points_3d_recovered_z1)
print()

# --- Recover 3D points at their actual depth ---
# We need the actual Z values from the original points in camera frame
actual_z_values = points_in_camera[:, 2]

# Scale each ray by its actual Z value to recover the original 3D point
points_3d_recovered_actual = rays_camera * actual_z_values[:, np.newaxis]

print("Recovered 3D Points at Actual Depth in Camera Frame:")
print(points_3d_recovered_actual)
print()

print("Original Points in Camera Frame (for comparison):")
print(points_in_camera[:, 0:3])
print()

# Verify recovery accuracy
error = np.abs(points_3d_recovered_actual - points_in_camera[:, 0:3])
print("Recovery Error (should be close to zero):")
print(error)
print(f"Max error: {np.max(error):.6e}")
print()

# Note: OpenCV also provides functions for homogeneous coordinate conversion:
# - cv.convertPointsToHomogeneous(points) : converts (N, D) to (N, 1, D+1)
# - cv.convertPointsFromHomogeneous(points) : converts (N, 1, D+1) to (N, D)
# Example:
# points_3d_homogeneous = cv.convertPointsToHomogeneous(points_3d_recovered_actual)
# points_3d_back = cv.convertPointsFromHomogeneous(points_3d_homogeneous)
print("OpenCV provides cv.convertPointsToHomogeneous() and cv.convertPointsFromHomogeneous()")
print("for converting between homogeneous and non-homogeneous coordinates.")
print()

# Visualize the 3D scene
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Frame Plot')
ax.legend()

# Create a new figure for the 2D projected points
fig2, ax2 = plt.subplots(figsize=(8, 6))

# Draw the image plane
ax2.set_xlim(0, camera_width)
ax2.set_ylim(camera_height, 0)  # Invert y-axis (image coordinates)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('u (pixels)')
ax2.set_ylabel('v (pixels)')
ax2.set_title('Projected Points on Image Plane')

# Plot projected points
ax2.scatter(imagePoints[:, 0], imagePoints[:, 1], c='red',
            s=100, marker='o', label='Projected Points')

# Add labels for each point
for i, pt in enumerate(imagePoints):
    ax2.text(pt[0] + 10, pt[1] + 10, f'P{i}', fontsize=10)

# Draw principal point (optical center)
ax2.scatter(cx, cy, c='blue', s=100, marker='+',
            linewidths=3, label='Principal Point')

ax2.legend()

plt.show()
