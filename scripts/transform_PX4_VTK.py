import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Function to draw a vector
def draw_vector(ax, start, end, color):
    arrow = np.array(end) - np.array(start)
    ax.quiver(start[0], start[1], start[2], arrow[0], arrow[1], arrow[2],
              arrow_length_ratio=0.1, color=color)

# Set up the figure and 3D axis for PX4 and VTK with transformations
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Limits and labels for PX4 Autopilot axis (right-handed coordinate system)
ax1.set_xlim([-1, 1])
ax1.set_ylim([-1, 1])
ax1.set_zlim([-1, 1])
ax1.set_xlabel('X (forward)')
ax1.set_ylabel('Y (right)')
ax1.set_zlabel('Z (downward)')
ax1.set_title('PX4 Coordinate System')

# Limits and labels for VTK axis (right-handed coordinate system)
ax2.set_xlim([-1, 1])
ax2.set_ylim([-1, 1])
ax2.set_zlim([-1, 1])
ax2.set_xlabel('X (right)')
ax2.set_ylabel('Y (up)')
ax2.set_zlabel('Z (out of screen)')
ax2.set_title('VTK Coordinate System')

# Draw PX4 Autopilot coordinate frame
draw_vector(ax1, [0, 0, 0], [0.5, 0, 0], 'r')  # X-axis in red
draw_vector(ax1, [0, 0, 0], [0, 0.5, 0], 'g')  # Y-axis in green
draw_vector(ax1, [0, 0, 0], [0, 0, -0.5], 'b')  # Z-axis in blue (pointing down)

# Draw VTK coordinate frame after transformation
# PX4 X-axis becomes VTK Z-axis
draw_vector(ax2, [0, 0, 0], [0, 0, 0.5], 'r')  # Z-axis in red
# PX4 Y-axis becomes VTK X-axis
draw_vector(ax2, [0, 0, 0], [0.5, 0, 0], 'g')  # X-axis in green
# PX4 Z-axis (inverted) becomes VTK Y-axis
draw_vector(ax2, [0, 0, 0], [0, -0.5, 0], 'b')  # Y-axis in blue

# Show plot
plt.show()
