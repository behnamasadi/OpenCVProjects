import numpy as np
import matplotlib.pyplot as plt
from euler_quaternions import *
from  plot_frame_poses_in_3d import *


QB_C = np.array([0.707, 0.707, 0, 0])  # Example quaternion
PB_C = np.array([0, 1, 0])             # Example position


QA_B = np.array([0.707, 0, 0.707, 0])  # Example quaternion
PA_B = np.array([1, 0, 0])             # Example position

R_A_B=quaternion_to_rotation_matrix(QA_B)
T_A_B=np.eye(4)
T_A_B[:3, :3] = R_A_B  # Replace the top-left 3x3 section with the rotation matrix
T_A_B[:3, 3] = PA_B  # Set the top-right 3x1 section as the translation vector
T_A_B[3,3]=1

print(T_A_B)


print( rotation_matrix_to_quaternion(R_A_B)   )


roll, pitch, yaw = rotation_matrix_to_roll_pitch_yaw(R_A_B)
print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")



QA_C, PA_C = compute_transformations(QA_B, QB_C, PA_B, PB_C)

print("QA_C:", QA_C)
print("PA_C:", PA_C)


# Example usage:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# A
length = 0.2
x, y, z = [0,0,0]
q=[1,0,0,0]

plot_frame(ax, x, y, z, q, length, "Frame A")

# T_A_B
x1, y1, z1 = PA_B
plot_frame(ax, x1, y1, z1, QA_B, length, "Frame B")


# T_A_C
x1, y1, z1=PA_C
plot_frame(ax, x1, y1, z1, QA_C, length, "Frame C")



ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Frame Plot')
ax.legend()

plt.show()