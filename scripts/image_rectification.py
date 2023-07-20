from pytransform3d.transform_manager import TransformManager
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
from pytransform3d import batch_rotations as br
from pytransform3d import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as R


# https://en.wikipedia.org/wiki/Image_rectification
# https://www.sci.utah.edu/~gerig/CS6320-S2013/Materials/CS6320-CV-F2012-Rectification.pdf
# https://www.cs.cmu.edu/~16385/s17/Slides/13.1_Stereo_Rectification.pdf
# https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
# https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6


#                 Z                        Z
#                 ▲                         ▲
#                /                           \
#               /                             \
#              /1 2 3 4     X                  \ 1 2 3 4
# Left Cam   |------------ ⯈                   |------------ ⯈Right cam
#           1|                               1 |
#           2|                               2 |
#           3|                               3 |
#          Y |                               Y |
#            ⯆                                ⯆


#                                Z ▲
#                                  |    ▲ Y
#                                 4|   /3
#                                 3|  /2
#                                 2| /1
#                       world     1|------------ ⯈ X
#                                   1 2 3 4 5
#
#
#
#


def rotationMatrix(roll, pitch, yaw):
    yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    R = yawMatrix * pitchMatrix * rollMatrix
    return R


def creatingEllipsoidInWorldCoordinate(center_x, center_y, center_z, a=2,   b=3,   c=1.6):
    phiStepSize = 0.10
    thetaStepSize = 0.20
    objectPointsInWorldCoordinate = []

    for phi in np.arange(-np.pi,  np.pi,  phiStepSize):
        for theta in np.arange(-np.pi / 2, np.pi / 2, thetaStepSize):
            X = a * np.cos(theta) * np.cos(phi)+center_x
            Y = b * np.cos(theta) * np.sin(phi)+center_y
            Z = c * np.sin(theta)+center_z
            objectPointsInWorldCoordinate.append(np.array([X, Y, Z]))

    return np.array(objectPointsInWorldCoordinate)


roll_cam0 = -np.pi / 2
pitch_cam0 = +np.pi / 36
yaw_cam0 = 0.0
rotationMatrix_cam0 = rotationMatrix(roll_cam0, pitch_cam0, yaw_cam0)

t0_x = -0.75
t0_y = -1.0
t0_z = 1.0


t_cam0 = np.array([[t0_x],
                   [t0_y],
                   [t0_z]])


cam0_in_world = pt.transform_from(rotationMatrix_cam0, t_cam0.ravel())

print("cam0_in_world\n:", cam0_in_world)

r = R.from_matrix([[1,  0,  0],
                   [0,  0,  1],
                   [0, -1, 0]])

# print("yaw, pitch, roll: ", r.as_euler('zyx', degrees=True))
# print("-----------------------")


roll_cam1 = -np.pi / 2
pitch_cam1 = -np.pi / 36
yaw_cam1 = 0.0

rotationMatrix_cam1 = rotationMatrix(roll_cam1, pitch_cam1, yaw_cam1)

t1_x = +0.75
t1_y = -1.0
t1_z = 1.0


t_cam1 = np.array([[t1_x],
                   [t1_y],
                   [t1_z]])


cam1_in_world = pt.transform_from(rotationMatrix_cam1, t_cam1.ravel())

print("cam1_in_world\n:", cam1_in_world)

tm = TransformManager()

tm.add_transform("cam0", "world", cam0_in_world)
tm.add_transform("cam1", "world", cam1_in_world)

ax = tm.plot_frames_in("world", s=.5)


ax.set_xlim((-5, 5))
ax.set_ylim((-5, 5))
ax.set_zlim((0, 5))

objectPointsInWorldCoordinate = creatingEllipsoidInWorldCoordinate(
    center_x=0, center_y=4, center_z=1.1, a=2,   b=3/5,   c=1.6/5)

ax.scatter(objectPointsInWorldCoordinate[:, 0],
           objectPointsInWorldCoordinate[:, 1], objectPointsInWorldCoordinate[:, 2])


# camera intrinsic parameters


focalLength = 2.0
numberOfPixelInHeight = 600
numberOfPixelInWidth = 600

heightOfSensor = 10
widthOfSensor = 10

my = (numberOfPixelInHeight) / heightOfSensor
U0 = (numberOfPixelInHeight) / 2

mx = (numberOfPixelInWidth) / widthOfSensor
V0 = (numberOfPixelInWidth) / 2


K = np.array([
             [focalLength * mx, 0, V0],
             [0, focalLength * my, U0],
             [0, 0, 1]
             ])


print("K:\n", K)

distCoeffs = np.array([0.0, 0.0, 0.0, 0.0])


imagePoints_cam0, jacobian = cv2.projectPoints(
    objectPointsInWorldCoordinate, np.linalg.inv(rotationMatrix_cam0), -t_cam0, K, distCoeffs)


leftImage = np.zeros([numberOfPixelInHeight, numberOfPixelInWidth])
for pixel_coordinate in imagePoints_cam0:
    U = int(pixel_coordinate[0, 0])
    V = int(pixel_coordinate[0, 1])
    leftImage[V, U] = 1


imagePoints_cam1, jacobian = cv2.projectPoints(
    objectPointsInWorldCoordinate, np.linalg.inv(rotationMatrix_cam1), -t_cam1, K, distCoeffs)

rightImage = np.zeros([numberOfPixelInHeight, numberOfPixelInWidth])
for pixel_coordinate in imagePoints_cam1:
    U = int(pixel_coordinate[0, 0])
    V = int(pixel_coordinate[0, 1])
    rightImage[V, U] = 1


cam1_in_cam0 = tm.get_transform("cam1", "cam0")
print("cam1_in_cam0:\n", cam1_in_cam0)

R = np.zeros([3, 3])
T = np.zeros([3, 1])
print("T:\n", t_cam1-t_cam0)

R = cam1_in_cam0[0:3, 0:3]
T = cam1_in_cam0[0:3, 3]

print(R)
print(T)


######################## stereoCalibrate ########################

# cv.stereoCalibrate(	objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize[, R[, T[, E[, F[, flags[, criteria]]]]]]	) ->	retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F


######################## stereoRectify ########################


R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrix1=K, distCoeffs1=distCoeffs, cameraMatrix2=K, distCoeffs2=distCoeffs, imageSize=(600, 600), R=R, T=T)

# print("R1, R2, P1, P2, Q, validPixROI1, validPixROI2:\n",
#       R1, R2, P1, P2, Q, validPixROI1, validPixROI2)


print("R1: (rotation matrix) for the first camera, brings points given in the unrectified first camera's coordinate system to points in the rectified first camera's coordinate system:\n", R1)

print("R2: (rotation matrix) for the second camera, brings points given in the unrectified second camera's coordinate system to points in the rectified second camera's coordinate system:\n", R2)

print("P1: projects points given in the rectified first camera coordinate system into the rectified first camera's image:\n", P1)

print("P2: projects points given in the rectified first camera coordinate system into the rectified second camera's image.:\n", P2)


# essential_matrix_estimation
cv2.imshow('leftImage', leftImage)
cv2.imshow('rightImage', rightImage)
plt.show()


######################## initUndistortRectifyMap ########################


# The function computes the joint undistortion and rectification transformation and represents the result in the form of maps for remap.
map1, map2 = cv2.initUndistortRectifyMap(
    K, distCoeffs, R1, P1, size=(600, 600), m1type=cv2.CV_32FC1)

print("map1 :\n", map1.shape)

print("map2 :\n", map2.shape)

cv2.waitKey(0)
