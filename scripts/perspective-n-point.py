import matplotlib.pyplot as plt
from  plot_frame_poses_in_3d import *
from euler_quaternions import *
import numpy as np
import cv2

np.set_printoptions(suppress=True)

# Step 1: Define the Object in the World
# Define the 3D points of the square (10x10 units)
square_size = 10
object_points = np.array([
    [0, 0, 0],
    [square_size, 0, 0],
    [square_size, square_size, 0],
    [0, square_size, 0]
], dtype=np.float32)


print("object_points:\n",object_points)

# Define rotation angle (in degrees) around the Y-axis
rotation_angle_deg = 30  # for example, 30 degrees
rotation_angle_rad = np.radians(rotation_angle_deg)
rotation_matrix=rotation_matrix_from_roll_pitch_yaw(roll=0,pitch=rotation_angle_rad,yaw=0)

# Apply the rotation matrix to each of the object points
rotated_object_points = np.dot(object_points, rotation_matrix.T)
#print(rotation_matrix)

# Position of the object in the world (e.g., translate by 20 units along x-axis)
world_translation = np.array([20, 0, 0], dtype=np.float32)
object_points_world = rotated_object_points + world_translation

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

print("rotation of object in camera: \n", rotation_vector,"\n","translation of object in camera:\n",translation_vector)





fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# world
length = 10.0
x, y, z = 0, 0, 0
q = [1, 0, 0, 0]

plot_frame(ax, x, y, z, q, length, "world")

# First frame
plot_frame(ax, *object_points_world[0], q, length, "object frame")


#length = 1
plot_frame(ax, *object_points_world[1], q, length, "object_points_1")
plot_frame(ax, *object_points_world[2], q, length, "object_points_2")
plot_frame(ax, *object_points_world[3], q, length, "object_points_3")


length = 10.0
plot_frame(ax, *camera_translation, q, length, "camera")



ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Frame Plot')
ax.legend()

plt.show()
