"""
Example usage of quaternion_utils module for inverse pose computation.
"""
from quaternion_utils import inverse_pose

# Test
# Pose format: [x, y, z, q_w, q_x, q_y, q_z]
pose = [1, 2, 3, 0.7071, 0, 0, 0.7071]
pose_inv = inverse_pose(pose)

print(f"Original pose: {pose}")
print(f"Inverse pose: {pose_inv}")
