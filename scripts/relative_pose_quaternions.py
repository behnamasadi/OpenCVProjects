import numpy as np
import matplotlib.pyplot as plt
from euler_quaternions import *
from  plot_frame_poses_in_3d import *


QB_C = np.array([0.707, 0.707, 0, 0])  # Example quaternion
PB_C = np.array([0, 1, 0])             # Example position


QA_B = np.array([0.707, 0, 0.707, 0])  # Example quaternion
PA_B = np.array([1, 0, 0])             # Example position


QA_C, PA_C = compute_transformations(QA_B, QB_C, PA_B, PB_C)

print("QA_C:", QA_C)
print("PA_C:", PA_C)


# Example usage:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# T_B_C
length = 0.2
x, y, z = PB_C

plot_frame(ax, x, y, z, QB_C, length, "T_B_C")

# T_A_B
x1, y1, z1 = PA_B
plot_frame(ax, x1, y1, z1, QA_B, length, "T_A_B")



x1, y1, z1=PA_C
plot_frame(ax, x1, y1, z1, QA_C, length, "T_A_C")



ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Frame Plot')
ax.legend()

plt.show()