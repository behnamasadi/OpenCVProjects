"""
Example usage of quaternion_utils module for relative pose computation.

This demonstrates computing the pose of frame C in frame A, given:
- Pose of frame B in frame A
- Pose of frame C in frame B
"""
import numpy as np
from quaternion_utils import relative_pose

# Define poses in format: [x, y, z, q_w, q_x, q_y, q_z]

# Pose of B in A: position [1, 2, 3], rotation 45 degrees around y-axis
# q = [cos(22.5°), 0, sin(22.5°), 0] = [0.9239, 0, 0.3827, 0]
pose_B_in_A = [1, 2, 3, 0.9239, 0, 0.3827, 0]

# Pose of C in B: position [2, 0, 1], rotation 30 degrees around x-axis
# q = [cos(15°), sin(15°), 0, 0] = [0.9659, 0.2588, 0, 0]
pose_C_in_B = [2, 0, 1, 0.9659, 0.2588, 0, 0]

# Compute combined pose
pose_C_in_A = relative_pose(pose_B_in_A, pose_C_in_B)

print(f"Pose of B in A: {pose_B_in_A}")
print(f"Pose of C in B: {pose_C_in_B}")
print(f"Pose of C in A: {pose_C_in_A}")
print()
print(
    f"Position of C in A: [{pose_C_in_A[0]:.4f}, {pose_C_in_A[1]:.4f}, {pose_C_in_A[2]:.4f}]")
print(
    f"Orientation of C in A (quaternion): [{pose_C_in_A[3]:.4f}, {pose_C_in_A[4]:.4f}, {pose_C_in_A[5]:.4f}, {pose_C_in_A[6]:.4f}]")
