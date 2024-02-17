

import pandas as pd


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

# Load the CSV file into a DataFrame and specify the 'ts' column to be skipped

file = '/home/behnam/Desktop/20231025_pgm_pictures/pictures/pgm_proper_rotated/camera_trajectory.csv'


df = pd.read_csv(file, usecols=lambda col: col != 'ts')

# Extract relevant columns
x = df['x'].values
y = df['y'].values
z = df['z'].values
q1 = df['q1'].values
q2 = df['q2'].values
q3 = df['q3'].values
q4 = df['q4'].values
roll = df['roll'].values
pitch = df['pitch'].values
yaw = df['yaw'].values

# Plot the camera trajectory in 2D
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Camera Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Camera Trajectory')
plt.show()

# Plot the camera trajectory in 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Camera Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Camera Trajectory (3D)')
plt.show()

# Create a visualizer for 3D visualization with Open3D
vis = o3d.visualization.Visualizer()
vis.create_window()

# Iterate through camera poses and create pyramids
for i in range(len(x)):
    # Create a pyramid representing the camera pose
    camera_pose = o3d.geometry.TriangleMesh.create_cone(
        radius=0.0002, height=0.0004, resolution=4)

    # Set the position of the pyramid
    camera_pose.translate([x[i], y[i], z[i]])

    # Calculate the rotation matrix from Euler angles (roll, pitch, yaw)
    R = o3d.geometry.get_rotation_matrix_from_xyz(
        [np.radians(roll[i]), np.radians(pitch[i]), np.radians(yaw[i])])

    # Rotate the pyramid using the calculated rotation matrix
    camera_pose.rotate(R)

    # Set the visualization style of the camera pyramid to wireframe
    # camera_pose.paint_uniform_color([0.8, 0.8, 0.8])  # Set color to gray
    camera_pose.paint_uniform_color([0.8, 0.0, 0.0])  # Set color to gray
    camera_pose.compute_vertex_normals()
    vis.add_geometry(camera_pose)

    # Set rendering option for wireframe
    vis.get_render_option().line_width = 2  # Adjust line width for wireframe

# View the scene
vis.run()
vis.destroy_window()
