# import numpy as np
# from pyquaternion import Quaternion


# def relative_pose(q_IMU_C0, t_IMU_C0, q_IMU_C1, t_IMU_C1):
#     # Calculate relative quaternion
#     q_C0_C1 = q_IMU_C1 * q_IMU_C0.inverse

#     # Calculate relative translation
#     t_diff = np.array(t_IMU_C1) - np.array(t_IMU_C0)
#     t_C0_C1 = q_IMU_C1.rotate(t_diff)

#     return q_C0_C1, t_C0_C1.tolist()


# # Define quaternions and translations for Camera0 and Camera1 w.r.t IMU
# q_IMU_C0 = Quaternion(w=0.6328142, x=0.3155095, y=-0.3155095, z=0.6328142)
# t_IMU_C0 = [0.234508, 0.028785, 0.039920]

# q_IMU_C1 = Quaternion(w=0.3155095, x=-0.6328142, y=-0.6328142, z=-0.3155095)
# t_IMU_C1 = [0.234508, 0.028785, -0.012908]

# q_C0_C1, t_C0_C1 = relative_pose(q_IMU_C0, t_IMU_C0, q_IMU_C1, t_IMU_C1)
# print("Quaternion of Camera1 w.r.t Camera0:", q_C0_C1)
# print("Translation of Camera1 w.r.t Camera0:", t_C0_C1)


import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plot_frame_poses_in_3d import plot_frame


def relative_pose(q_C0_IMU, t_C0_IMU, q_C1_IMU, t_C1_IMU):
    q_C0_C1 = q_C0_IMU * q_C1_IMU.inverse
    t_diff = np.array(t_C1_IMU) - np.array(t_C0_IMU)
    t_C0_C1 = q_C0_IMU.rotate(t_diff)
    return q_C0_C1, t_C0_C1.tolist()


def plot_camera_pose(ax, t, q, color):
    R = q.rotation_matrix
    origin = np.array(t)

    # plot the camera center
    ax.scatter(*origin, color=color)

    # plot the camera coordinate axes
    for i, c in enumerate(['r', 'g', 'b']):
        axis = R[:, i]
        ax.quiver(*origin, *axis, length=0.1, color=c)


q_C0_IMU = Quaternion(w=0.6328142, x=0.3155095, y=-0.3155095, z=0.6328142)
t_C0_IMU = [0.234508, 0.028785, 0.039920]

q_C1_IMU = Quaternion(w=0.3155095, x=-0.6328142, y=-0.6328142, z=-0.3155095)
t_C1_IMU = [0.234508, 0.028785, -0.012908]

q_C0_C1, t_C0_C1 = relative_pose(q_C0_IMU, t_C0_IMU, q_C1_IMU, t_C1_IMU)

print("Quaternion of Camera1 w.r.t Camera0:", q_C0_C1)
print("Translation of Camera1 w.r.t Camera0:", t_C0_C1)


# Now, plot the cameras in 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

length = 0.020

# cam 1
x1, y1, z1 = 0, 0, 0
q1 = [1, 0, 0, 0]  # Identity quaternion, no rotation

plot_frame(ax, x1, y1, z1, q1, length, "cam 1")


# cam 2

x2, y2, z2 = [0.0, 0.042190314999783624, -0.031792686332221626]
q2 = [0.000, +0.000, +0.602, +0.799]

plot_frame(ax, x2, y2, z2, q2, length, "cam 2")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Frame Plot')
ax.legend()

plt.show()
