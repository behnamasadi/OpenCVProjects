import numpy as np
import cv2
np.set_printoptions(suppress=True)


def simple():
    # Define the 3D points of the square (10x10 units)
    square_size = 10
    object_points = np.array([
        [0, 0, 0],
        [square_size, 0, 0],
        [square_size, square_size, 0],
        [0, square_size, 0]
    ], dtype=np.float32)

    # Define the camera intrinsic parameters (Assuming a 1080p camera)
    focal_length = 1000
    center = (540, 960)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Simulated 2D points (as if they are projected by a camera)
    # Adjust these values to simulate different camera positions
    image_points = np.array([
        [500, 500],
        [600, 500],
        [600, 600],
        [500, 600]
    ], dtype=np.float32)

    # Solving the PnP problem
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs)

    # Convert the rotation vector to a rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    print("Rotation Matrix:\n", rotation_matrix)
    print("Translation Vector:\n", translation_vector)



def semi():
    # Define the 3D points of the square (10x10 units)
    square_size = 10
    object_points = np.array([
        [0, 0, 0],
        [square_size, 0, 0],
        [square_size, square_size, 0],
        [0, square_size, 0]
    ], dtype=np.float32)

    # Position of the object in the world (e.g., translate by 20 units along x-axis)
    world_translation = np.array([20, 0, 0], dtype=np.float32)
    object_points_world = object_points + world_translation


    # Define the camera intrinsic parameters (Assuming a 1080p camera)
    focal_length = 1000
    center = (540, 960)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Camera extrinsic parameters (position and orientation in the world)
    camera_translation = np.array([0, 0, 50], dtype=np.float32) # Camera 50 units away from origin
    camera_rotation = np.eye(3) # Assuming camera is looking straight forward initially

    # We simulate the camera view by transforming the object points to the camera coordinate system
    # and then projecting them onto the camera's image plane.

    # Transform object points to camera coordinate system
    object_points_camera = np.dot(camera_rotation, (object_points_world - camera_translation).T).T

    # Project points onto camera plane
    image_points = np.array([np.dot(camera_matrix, pt)[:2] / np.dot(camera_matrix, pt)[2] 
                            for pt in object_points_camera], dtype=np.float32)


    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs)

    print("Estimated Rotation Vector:\n", rotation_vector)
    print("Estimated Translation Vector:\n", translation_vector)









import numpy as np
import cv2

# Step 1: Define the Object in the World
# Define the 3D points of the square (10x10 units)
square_size = 10
object_points = np.array([
    [0, 0, 0],
    [square_size, 0, 0],
    [square_size, square_size, 0],
    [0, square_size, 0]
], dtype=np.float32)

# Define rotation angle (in degrees) around the Y-axis
rotation_angle_deg = 30  # for example, 30 degrees
rotation_angle_rad = np.radians(rotation_angle_deg)

# Rotation matrix around Y-axis
rotation_matrix_y = np.array([
    [np.cos(rotation_angle_rad), 0, np.sin(rotation_angle_rad)],
    [0, 1, 0],
    [-np.sin(rotation_angle_rad), 0, np.cos(rotation_angle_rad)]
])

# Apply rotation to object points
object_points_rotated = np.dot(object_points, rotation_matrix_y.T)

# Position of the object in the world (e.g., translate by 20 units along x-axis)
world_translation = np.array([20, 0, 0], dtype=np.float32)
object_points_world = object_points_rotated + world_translation

# Step 2: Camera Parameters and Position
# Define the camera intrinsic parameters (Assuming a 1080p camera)
focal_length = 1000
center = (540, 960)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float32)

# Assuming no lens distortion
dist_coeffs = np.zeros((4, 1))

# Camera extrinsic parameters (position and orientation in the world)
camera_translation = np.array([0, 0, 50], dtype=np.float32) # Camera 50 units away from origin
camera_rotation = np.eye(3) # Assuming camera is looking straight forward initially

# Step 3: Project the Object onto the Camera Plane
# Transform object points to camera coordinate system
object_points_camera = np.dot(camera_rotation, (object_points_world - camera_translation).T).T

# Project points onto camera plane
image_points = np.array([np.dot(camera_matrix, pt)[:2] / np.dot(camera_matrix, pt)[2] 
                         for pt in object_points_camera], dtype=np.float32)

# Step 4: Solve for Object Position
# Using OpenCV's solvePnP to estimate the pose of the object
success, rotation_vector, translation_vector = cv2.solvePnP(
    object_points, image_points, camera_matrix, dist_coeffs)

print(rotation_vector,"\n" ,translation_vector)
