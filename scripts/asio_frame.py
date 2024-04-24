import numpy as np
import matplotlib.pyplot as plt
from euler_quaternions import *
from  plot_frame_poses_in_3d import *


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# body
length = 0.01
x, y, z = 0, 0, 0
q = [1, 0, 0, 0]

plot_frame(ax, x, y, z, q, length, "body")

# navcam0
x= 0.1942
y= 0.048
z= 0.022

qx= -0.3155095
qy= 0.3155095
qz= -0.6328142
qw= 0.6328142

q = [qw, qx, qy, qz] 
plot_frame(ax, x, y, z, q, length, "navcam0")





# navcam1

x= 0.1942
y= 0.048
z= -0.03

qx= 0.6328142
qy= 0.6328142
qz= 0.3155095
qw= 0.3155095


q = [qw, qx, qy, qz] 

plot_frame(ax, x, y, z, q, length, "navcam1")



# gimbal
x_gimbal= 0.188
y_gimbal= 0.03285
z_gimbal= 0.0019
# for the rest pose but could change during flight and it is dynamic
q_gimbal = [1, 0, 0, 0] 


plot_frame(ax, x_gimbal, y_gimbal, z_gimbal, q_gimbal, length, "gimbal")




# tof, expressed in gimbal frame
x_tof= 0.008997
y_tof= -0.0234
z_tof= -0.016215

qx_tof= 0.7071068
qy_tof= 0.0
qz_tof= 0.7071068
qw_tof= 0.0

# gimbal=B in body=A
QA_B= q_gimbal
PA_B=np.array([x_gimbal,y_gimbal,z_gimbal])


# tof=C in gimbal=A
QB_C=np.array([qw_tof,qx_tof,qy_tof,qz_tof])
PB_C=np.array([x_tof,y_tof,z_tof])


#compute_transformations(QA_B, QB_C, PA_B, PB_C)
# pose of tof in bdy
QA_C, PA_C=compute_transformations(QA_B, QB_C, PA_B, PB_C)




x_body_gimbal, y_body_gimbal, z_body_gimbal =PA_C
q_body_gimbal=QA_C


plot_frame(ax, x_body_gimbal, y_body_gimbal, z_body_gimbal, q_body_gimbal, length, "tof")



ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Frame Plot')
ax.legend()

plt.show()