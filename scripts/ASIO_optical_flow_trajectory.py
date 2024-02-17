

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = '/home/behnam/Desktop/20231025_pgm_pictures/pictures/pgm_proper_rotated/camera_trajectory.csv'

# Load the CSV file into a DataFrame and specify the 'ts' column to be skipped
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

# Convert to NED convention
z = -z  # Reverse the Z-axis

# Plot the camera trajectory in 3D (NED convention)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Camera Trajectory')
ax.set_xlabel('Forward (X)')
ax.set_ylabel('Right (Y)')
ax.set_zlabel('Down (Z)')
ax.legend()
plt.title('Camera Trajectory (3D, NED)')

# Add X, Y, Z axes representing camera orientation (NED convention)
for i in range(len(x)):
    R = np.array([
        [1 - 2 * (q2[i]**2 + q3[i]**2), 2 * (q1[i]*q2[i] - q4[i]
                                             * q3[i]), 2 * (q1[i]*q3[i] + q4[i]*q2[i])],
        [2 * (q1[i]*q2[i] + q4[i]*q3[i]), 1 - 2 *
         (q1[i]**2 + q3[i]**2), 2 * (q2[i]*q3[i] - q4[i]*q1[i])],
        [2 * (q1[i]*q3[i] - q4[i]*q2[i]), 2 * (q2[i]*q3[i] +
                                               q4[i]*q1[i]), 1 - 2 * (q1[i]**2 + q2[i]**2)]
    ])

    # Length of axes
    length = 0.002

    # Add text labels for every k-th frame (e.g., k=5)
    k = 4
    if i % k == 0:

        # X-axis (forward, red)
        ax.plot([x[i], x[i] + R[0, 0] * length], [y[i], y[i] + R[1, 0]
                * length], [z[i], z[i] + R[2, 0] * length], color='red')

        # Y-axis (right, green)
        ax.plot([x[i], x[i] + R[0, 1] * length], [y[i], y[i] + R[1, 1] *
                length], [z[i], z[i] + R[2, 1] * length], color='green')

        # Z-axis (down, blue)
        ax.plot([x[i], x[i] + R[0, 2] * length], [y[i], y[i] + R[1, 2]
                * length], [z[i], z[i] + R[2, 2] * length], color='blue')

    # Add text labels for every k-th frame (e.g., k=3)
    if i % k == 0:
        ax.text(x[i], y[i], z[i], f'Frame {i}', fontsize=12,
                color='black', ha='center', va='bottom')


# Add the world coordinate axis (XYZ) at the origin
world_axis_length = 1.0
ax.quiver(0, 0, 0, world_axis_length, 0, 0, color='red', label='X (World)')
ax.quiver(0, 0, 0, 0, world_axis_length, 0, color='green', label='Y (World)')
ax.quiver(0, 0, 0, 0, 0, world_axis_length, color='blue', label='Z (World)')

ax.legend()

plt.show()
