import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


import numpy as np
import matplotlib.pyplot as plt


# http://www.cvlibs.net/download.php?file=data_odometry_poses.zip
ground_truth_poses = '/home/behnam/workspace/OpenCVProjects/data/kitti/odometry/05/poses/05.txt'

# http://www.cvlibs.net/download.php?file=data_odometry_gray.zip
left_images_path = '/home/behnam/workspace/OpenCVProjects/data/kitti/odometry/05/image_0/'
right_images_path = '/home/behnam/workspace/OpenCVProjects/data/kitti/odometry/05/image_1/'
time_stamp_path = '/home/behnam/Downloads/data_odometry_gray/dataset/sequences/00/times.txt'


# http://www.cvlibs.net/download.php?file=data_odometry_calib.zip
cameras_file_path = '/home/behnam/Downloads/data_odometry_gray/dataset/sequences/00/calib.txt'

poses = pd.read_csv(
    ground_truth_poses, delimiter=' ', header=None)
# print('Shape of position dataframe:', poses.shape)


calib = pd.read_csv(cameras_file_path,
                    delimiter=' ', header=None, index_col=0)


P0 = np.array(calib.loc['P0:']).reshape((3, 4))
print("cam0: left gray camera projection matrix:\n", P0)

P1 = np.array(calib.loc['P1:']).reshape((3, 4))
print("cam1: right gray camera projection matrix:\n", P1)


# decomposition of a projection matrix into a calibration and a rotation matrix and the position of a camera.
# It optionally returns three rotation matrices, one for each axis,

cameraMatrix_cam0, rotMatrix_cam0, transVect_cam0, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv2.decomposeProjectionMatrix(
    P0)


print("""\n

Schematic of cam0 and cam1
        ^ x
        |
        |
        |
z <-----⦻ y  cam1



        ^ x
        |
        |
        |
z <-----⦻ y  cam0
\n""")


print("cam0 cameraMatrix:\n", cameraMatrix_cam0)

print("cam0 rotMatrix:\n", rotMatrix_cam0)

transVect = transVect_cam0/transVect_cam0[3]

print("cam0 transVect:\n", transVect_cam0)


cameraMatrix_cam1, rotMatrix_cam1, transVect_cam1, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv2.decomposeProjectionMatrix(
    P1)


print("cam1 cameraMatrix:\n", cameraMatrix_cam1)

print("cam1 rotMatrix:\n", rotMatrix_cam1)

transVect_cam1 = transVect_cam1/transVect_cam1[3]

print("cam1 transVect:\n", transVect_cam1.round(4))

# The R and t from decomposition of projection matrix are R_world_in_cam and t_world_in_cam.
# The decomposition of cam0 shows that the world coordinate is located at cam0.
# According to the schematic, the cam1 is 0.54m from cam0 on the X axis, so we should expect the t of cam1 (the t_world_in_cam1) to be [-0.54,0,0]
# but it is [-0.54,0,0], the reason is the matrix that we is NOT the projection matrix to bring point from world coordinate to image plane.
# This is is a rectified projection matrix for a stereo rig, which is intended to project points from the coordinate frames of multiple
# cameras onto the SAME image plane (left cam),

# Rectification matrix: A rotation matrix aligning the camera coordinate system to the ideal stereo image plane so that epipolar lines in both stereo images are parallel.


def compute_disparity(img_left, img_right, matcher):
    '''
    matcher -- (str) can be 'bm' for StereoBM or 'sgbm' for StereoSGBM matching
    '''
    sad_window = 6
    num_disparities = sad_window*16
    block_size = 11
    matcher_name = matcher

    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                      blockSize=block_size
                                      )

    elif matcher_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1=8 * 3 * sad_window ** 2,
                                        P2=32 * 3 * sad_window ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                        )
    disp_left = matcher.compute(img_left, img_right).astype(np.float32)

    return disp_left


left_image_files = [os.path.abspath(os.path.join(
    left_images_path, p)) for p in os.listdir(left_images_path)]
left_image_files.sort()


right_image_files = [os.path.abspath(os.path.join(
    right_images_path, p)) for p in os.listdir(right_images_path)]
right_image_files.sort()


for l, r in zip(left_image_files, right_image_files):
    # print(r)
    # print(l)
    img_left = cv2.imread(r, cv2.COLOR_BGR2GRAY)
    img_right = cv2.imread(l, cv2.COLOR_BGR2GRAY)
    disp_left = compute_disparity(img_left, img_right, matcher='sgbm')
    cv2.imshow('disp_left', disp_left)
    k = cv2.waitKey(1)
