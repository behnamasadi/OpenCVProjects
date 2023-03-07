import numpy as np


R_cam_imu=np.array([ [0  , 1 ,  0],
                     [-1 , 0 ,  0] ,
                     [0  , 0 , -1]])

t_cam_imu=np.array([0.2,-0.4,0.3])

T_cam_imu=np.empty((4, 4))
T_cam_imu[:3, :3] = R_cam_imu
T_cam_imu[:3, 3] = t_cam_imu
T_cam_imu[3, :] = [0, 0, 0, 1]



# or

T_imu_cam = np.vstack((np.hstack((R_cam_imu, t_cam_imu[:, None])), [0, 0, 0 ,1]))

