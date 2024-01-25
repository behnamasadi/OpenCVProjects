import numpy as np
import cv2
np.set_printoptions(suppress=True)

# Define the object points (corners of a square on the plane Z=0)
object_points = np.array([
    [0, 0, 0],  # Point 1 (origin)
    [1, 0, 0],  # Point 2 (1 unit along the X-axis)
    [1, 1, 0],  # Point 3 (1 unit up along the Y-axis)
    [0, 1, 0]   # Point 4 (back to Y-axis)
], dtype=np.float32)

# Camera intrinsic parameters (assuming a simple pinhole camera model)
focal_length = 1.0  # Example focal length
principal_point = (0.5, 0.5)  # Assuming the principal point is at the center

camera_matrix = np.array([
    [focal_length, 0, principal_point[0]],
    [0, focal_length, principal_point[1]],
    [0, 0, 1]
], dtype=np.float32)

# Camera 1 extrinsic parameters (position and orientation)
# Assume camera 1 is at origin looking along the Z-axis
rotation_vector_1 = np.array([0, 0, 0], dtype=np.float32)  # No rotation
translation_vector_1 = np.array([0, 0, 1], dtype=np.float32)  # 1 unit away from the origin along Z-axis

# Camera 2 extrinsic parameters (position and orientation)
# Assume camera 2 is positioned differently
rotation_vector_2 = np.array([0, np.deg2rad(45), 0], dtype=np.float32)  # Rotated 45 degrees around Y-axis
translation_vector_2 = np.array([1, 0, 0.5], dtype=np.float32)  # Different position

# Project points onto each camera's image plane
rotation_matrix_1, _ = cv2.Rodrigues(rotation_vector_1)
rotation_matrix_2, _ = cv2.Rodrigues(rotation_vector_2)

projection_matrix_1 = np.hstack((rotation_matrix_1, translation_vector_1.reshape(-1, 1)))
projection_matrix_2 = np.hstack((rotation_matrix_2, translation_vector_2.reshape(-1, 1)))

# Project points using camera matrices
points_2d_camera_1 = cv2.projectPoints(object_points, rotation_vector_1, translation_vector_1, camera_matrix, None)[0]
points_2d_camera_2 = cv2.projectPoints(object_points, rotation_vector_2, translation_vector_2, camera_matrix, None)[0]

points_2d_camera_1, points_2d_camera_2, projection_matrix_1, projection_matrix_2


# Triangulating the points to reconstruct the 3D positions
# Convert the 2D points for triangulation (removing the unnecessary third dimension)
points_2d_camera_1 = points_2d_camera_1.reshape(-1, 2)
points_2d_camera_2 = points_2d_camera_2.reshape(-1, 2)

# Create the full projection matrices for both cameras
full_projection_matrix_1 = camera_matrix @ projection_matrix_1
full_projection_matrix_2 = camera_matrix @ projection_matrix_2

# Perform triangulation
homogeneous_3d_points = cv2.triangulatePoints(full_projection_matrix_1, full_projection_matrix_2, 
                                              points_2d_camera_1.T, points_2d_camera_2.T)

# Convert from homogeneous coordinates to 3D
triangulated_3d_points = homogeneous_3d_points[:3] / homogeneous_3d_points[3]

print(triangulated_3d_points.T)  # Transposed for better readability

print(object_points)
