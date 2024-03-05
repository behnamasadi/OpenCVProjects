import numpy as np
import matplotlib.pyplot as plt
from euler_quaternions import *
from  plot_frame_poses_in_3d import *

np.set_printoptions(suppress = True)

# pose of C expressed in B
QB_C = np.array([0.707, 0.707, 0, 0])
PB_C = np.array([-1, -1, 1])

# Rotation and corresponding 4x4 transformation
R_B_C=quaternion_to_rotation_matrix(QB_C)
T_B_C=np.eye(4)
T_B_C[:3, :3] = R_B_C   # Replace the top-left 3x3 section with the rotation matrix
T_B_C[:3, 3] = PB_C     # Set the top-right 3x1 section as the translation vector
#print("T_B_C \n:", T_B_C)

# pose of B expressed in A
QA_B = np.array([0.707, 0, 0.707, 0])
PA_B = np.array([1, -1, 1])

# Rotation and corresponding 4x4 transformation
R_A_B=quaternion_to_rotation_matrix(QA_B)
T_A_B=np.eye(4)
T_A_B[:3, :3] = R_A_B
T_A_B[:3, 3] = PA_B
# print("T_A_B \n:", T_A_B)


# calculating pose of C in A using chain rule
QA_C, PA_C = compute_transformations(QA_B, QB_C, PA_B, PB_C)

print("QA_C:", QA_C)
print("PA_C:", PA_C)

T_A_C=np.eye(4)
R_A_C=quaternion_to_rotation_matrix(QA_C)
T_A_C[:3, :3] = R_A_C 
T_A_C[:3, 3] = PA_C  


# print("T_A_C calculated from quaternions:\n",T_A_C.round(decimals=3))

T_A_B_T_B_C=T_A_B@T_B_C
print("T_A_B_T_B_C:\n",T_A_B_T_B_C.round(decimals=3))


diff=T_A_C -T_A_B_T_B_C

# print("diff: \n",diff.round(decimals=3))
print( "sum of diff elements: ", diff.sum())



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# A
length = 0.2
x, y, z = [0,0,0]
q=[1,0,0,0]

plot_frame(ax, x, y, z, q, length, "Frame A")

# F_A_B
x1, y1, z1 = PA_B
plot_frame(ax, x1, y1, z1, QA_B, length, "Frame B")


# F_A_C
x1, y1, z1=PA_C
plot_frame(ax, x1, y1, z1, QA_C, length, "Frame C")




# print("PB_C: ", PB_C)
# print(T_A_B_T_B_C[0:3,3])
# print( rotation_matrix_to_quaternion(R_B_C)   )
roll, pitch, yaw = rotation_matrix_to_roll_pitch_yaw(R_B_C)
# print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Frame Plot')
ax.legend()

plt.show()