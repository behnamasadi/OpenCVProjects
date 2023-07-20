import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


import numpy as np
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager


# http://www.cvlibs.net/download.php?file=data_odometry_poses.zip
ground_truth_poses = '/home/behnam/Downloads/data_odometry_gray/data_odometry_poses/dataset/poses/00.txt'

# http://www.cvlibs.net/download.php?file=data_odometry_gray.zip
left_images_path = '/home/behnam/Downloads/data_odometry_gray/dataset/sequences/00/image_0'
time_stamp_path = '/home/behnam/Downloads/data_odometry_gray/dataset/sequences/00/times.txt'


# http://www.cvlibs.net/download.php?file=data_odometry_calib.zip
cameras_file_path = '/home/behnam/Downloads/data_odometry_gray/dataset/sequences/00/calib.txt'


######################### Ground Truth Poses #########################
# r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
# 12 comes from flattening a 3x4 transformation matrix of the left
# stereo camera with respect to the global coordinate frame.
# Ref: https://stackoverflow.com/questions/60639665/visual-odometry-kitti-dataset

poses = pd.read_csv(
    ground_truth_poses, delimiter=' ', header=None)
print('Shape of position dataframe:', poses.shape)

print('First position:')
first_pose = np.array(poses.iloc[0]).reshape((3, 4)).round(2)
print(first_pose)


gt = np.zeros((len(poses), 3, 4))
for i in range(len(poses)):
    gt[i] = np.array(poses.iloc[i]).reshape((3, 4))

gt[1].dot(np.array([0, 0, 0, 1]))


fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(gt[:, :, 3][:, 0], gt[:, :, 3][:, 1], gt[:, :, 3][:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=-40, azim=270)
# plt.show()


left_image_files = [os.path.abspath(os.path.join(
    left_images_path, p)) for p in os.listdir(left_images_path)]
# sorted(os.listdir(left_images_path))
left_image_files.sort()


# left_image_files = os.listdir(left_images_path)


######################### Time Stamp #########################


times = pd.read_csv(time_stamp_path,
                    delimiter=' ', header=None)


######################### Projection Matrices/ LIDAR #########################
# Matrices for 4 cameras projection,  3x4 projection matrices, P0, P1, P2, P3, Tr(LIDAR)

calib = pd.read_csv(cameras_file_path,
                    delimiter=' ', header=None, index_col=0)

P0 = np.array(calib.loc['P0:']).reshape((3, 4))
print(P0)

P1 = np.array(calib.loc['P1:']).reshape((3, 4))
print(P1)


# decomposition of a projection matrix into a calibration and a rotation matrix and the position of a camera.
# It optionally returns three rotation matrices, one for each axis,
cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv2.decomposeProjectionMatrix(
    P1)


print(rotMatrix)
transVect = transVect/transVect[3]
print(transVect)


# Rectification matrix (stereo cameras only) A rotation matrix aligning the camera coordinate system to the ideal stereo image plane so that epipolar lines in both stereo images are parallel.
Rt = np.hstack([rotMatrix, transVect[:3]])
print(Rt)


# detector_name = 'orb'

# if detector_name == 'sift':
#     detector = cv2.SIFT_create()
# elif detector_name == 'orb':
#     detector = cv2.ORB_create()
# elif detector_name == 'surf':
#     detector = cv2.xfeatures2d.SURF_create()


detector = cv2.FastFeatureDetector_create(
    threshold=25, nonmaxSuppression=True)

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


lk_params = dict(winSize=(21, 21), criteria=(
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# lk_params = dict(winSize=(21, 21), criteria=(
#                      cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

previous_frame = None


cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv2.decomposeProjectionMatrix(
    P0)

# np.set_printoptions(suppress=True)

np.set_printoptions(suppress=True,  formatter={'float_kind': '{:f}'.format})

T_cam_new_in_cam_previous = np.eye(4)
T_cam_previous_in_world = np.eye(4)


traj = np.zeros(shape=(600, 800, 3))


for i, img_name in enumerate(left_image_files):
    # print(img_name)
    # if (i % 20 == 0):
    #     print(i)
    current_frame = cv2.imread(img_name, cv2.COLOR_BGR2GRAY)

    keypoints_current_frame = cv2.KeyPoint_convert(
        detector.detect(current_frame))

    # keypoints_current_frame = detector.detect(current_frame)

    # keypoints_current_frame = np.array(
    #     [x.pt for x in keypoints_current_frame], dtype=np.float32).reshape(-1, 1, 2)

    # keypoints_current_frame = cv2.goodFeaturesToTrack(
    #     current_frame, mask=None,  **feature_params)

    if previous_frame is None:
        keypoints_previous_frame = keypoints_current_frame
        previous_frame = current_frame
        continue

    # cv2.imshow('previous_frame', previous_frame)
    # cv2.waitKey(5000)
    # cv2.imshow('current_frame', current_frame)
    # cv2.waitKey(5000)

    # print("keypoints_previous_frame:", keypoints_previous_frame)
    # print("keypoints_current_frame:", keypoints_current_frame)

    # opticalFlowNextPts, status, err = cv2.calcOpticalFlowPyrLK(
    #     previous_frame, current_frame, keypoints_previous_frame, keypoints_current_frame, **lk_params)

    opticalFlowNextPts, status, err = cv2.calcOpticalFlowPyrLK(
        previous_frame, current_frame, keypoints_previous_frame, None, **lk_params)

    # print("status.shape: ", status.shape)
    # print("keypoints_previous_frame.shape: ", keypoints_previous_frame.shape)
    # print("opticalFlowNextPts.shape: ", opticalFlowNextPts.shape)

    keypoints_previous_frame = keypoints_previous_frame.reshape(-1, 1, 2)
    opticalFlowNextPts = opticalFlowNextPts.reshape(-1, 1, 2)

    good_previous = keypoints_previous_frame[status == 1]
    good_new = opticalFlowNextPts[status == 1]

    previous_frame = current_frame
    keypoints_previous_frame = keypoints_current_frame

    # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga13f7e34de8fa516a686a56af1196247f
    E, mask = cv2.findEssentialMat(
        good_new, good_previous, cameraMatrix, cv2.RANSAC, 0.999, 1.0, None)
    # print("Essential Matrix:", E)

    # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gadb7d2dfcc184c1d2f496d8639f4371c0
    retval, R, t, mask = cv2.recoverPose(
        E, good_previous, good_new, cameraMatrix)

    # print("t:", t)
    # # print("R:", R @ np.transpose(R))
    # print("R:", R)

    T_cam_new_in_cam_previous = np.zeros([4, 4])
    T_cam_new_in_cam_previous[:3, :3] = R
    T_cam_new_in_cam_previous[:3, 3] = t.ravel()
    T_cam_new_in_cam_previous[3, 3] = 1
    # print(T_cam_new_in_cam_previous[:3, :3])
    # print("T_cam_new_in_cam_previous:", T_cam_new_in_cam_previous)
    # print("T_cam_previous_in_world:", T_cam_previous_in_world)

    T_cam_new_in_world = T_cam_previous_in_world@T_cam_new_in_cam_previous
    x, y, z = T_cam_new_in_world[:3, 3]
    print("x,y,z:\n", x, y, z)
    T_cam_previous_in_world = T_cam_new_in_world

    cv2.imshow('current_frame', current_frame)
    k = cv2.waitKey(1)

    center = (round(x + 400), round(z + 500))
    print("center: ", center)
    center_coordinates = (120, 50)

    color = (255, 0, 0)
    radius = 1
    thickness = 2

    traj = cv2.circle(traj, center, radius, color, thickness)

    cv2.imshow('trajectory', traj)
    k = cv2.waitKey(1)
