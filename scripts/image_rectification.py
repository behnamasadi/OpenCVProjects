# from pytransform3d.transform_manager import TransformManager
# from pytransform3d import transformations as pt
# from pytransform3d import rotations as pr
# from pytransform3d import batch_rotations as br
# from pytransform3d import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from euler_quaternions import *

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




def createChessBoardInWorldCoordinate(center_x, center_y, center_z, squareSize=0.2, numberOfRows=6, numberOfCols=7):
    objectPointsInObjectCoordinate = []
    objectPointsInWorldCoordinate = []

    for r in range(numberOfRows):
        for c in range(numberOfCols):
            # Points in object coordinate system (origin at one corner of the chessboard)
            objectPoint = np.array(
                [r * squareSize, 0.0, c * squareSize], dtype=np.float32)
            objectPointsInObjectCoordinate.append(objectPoint)

            # Points in world coordinate system (offset by center_x, center_y, center_z)
            worldPoint = objectPoint + \
                np.array([center_x, center_y, center_z], dtype=np.float32)
            objectPointsInWorldCoordinate.append(worldPoint)

    return np.array(objectPointsInObjectCoordinate), np.array(objectPointsInWorldCoordinate)


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


# camera extrinsic parameters


roll_cam0 = -np.pi / 2
pitch_cam0 = +np.pi / 36
yaw_cam0 = 0.0
rotationMatrix_cam0 = rotation_matrix_from_roll_pitch_yaw(roll_cam0, pitch_cam0, yaw_cam0)

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

rotationMatrix_cam1 = rotation_matrix_from_roll_pitch_yaw(roll_cam1, pitch_cam1, yaw_cam1)

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


centers_x = [0, 0.5]
centers_y = [4, 4.2]
centers_z = [1.1, 1.3]


objectPoints = []
imagePoints1 = []
imagePoints2 = []

for center_x, center_y, center_z in zip(centers_x, centers_y, centers_z):
    # Generate object points
    print(center_x, center_y, center_z)

    objectPointsInObjectCoordinate, objectPointsInWorldCoordinate = createChessBoardInWorldCoordinate(
        center_x, center_y, center_z, squareSize=0.2, numberOfRows=5, numberOfCols=7)

    ax.scatter(objectPointsInWorldCoordinate[:, 0],
               objectPointsInWorldCoordinate[:, 1], objectPointsInWorldCoordinate[:, 2])

    ############################## projecting points into cam0 and cam1 to find the corresponding projection point   ##############################

    imagePoints_cam0, jacobian = cv2.projectPoints(
        objectPointsInWorldCoordinate, np.linalg.inv(rotationMatrix_cam0), -t_cam0, K, distCoeffs)

    imagePoints_cam1, jacobian = cv2.projectPoints(
        objectPointsInWorldCoordinate, np.linalg.inv(rotationMatrix_cam1), -t_cam1, K, distCoeffs)

    ###################################################################################################
    objectPoints.append(objectPointsInObjectCoordinate)
    imagePoints1.append(imagePoints_cam0.reshape(-1, 1, 2))
    imagePoints2.append(imagePoints_cam1.reshape(-1, 1, 2))

    ############################## creating image from projected points in  ##############################

    leftImage = np.zeros([numberOfPixelInHeight, numberOfPixelInWidth])
    for pixel_coordinate in imagePoints_cam0:
        U = int(pixel_coordinate[0, 0])
        V = int(pixel_coordinate[0, 1])
        leftImage[V, U] = 1

    rightImage = np.zeros([numberOfPixelInHeight, numberOfPixelInWidth])
    for pixel_coordinate in imagePoints_cam1:
        U = int(pixel_coordinate[0, 0])
        V = int(pixel_coordinate[0, 1])
        rightImage[V, U] = 1

    # essential_matrix_estimation
    cv2.imshow('leftImage', leftImage)
    cv2.imshow('rightImage', rightImage)

    # cv2.imwrite('leftImage_'+str(center_x)+".jpg", leftImage)
    # plt.show()
    # Wait for a key press and check if it is the Escape key
    key = cv2.waitKey(0)
    if key == 27:  # ASCII value of Escape key is 27
        cv2.destroyAllWindows()  # Close all OpenCV windows
        continue  # Continue to the next iteration of the loop

# since we know exact pose of camera we can find the pose of cam1 in cam0

cam1_in_cam0 = tm.get_transform("cam1", "cam0")
print("cam1_in_cam0:\n", cam1_in_cam0)

R = np.zeros([3, 3])
T = np.zeros([3, 1])
print("T:\n", t_cam1-t_cam0)

R = cam1_in_cam0[0:3, 0:3]
T = cam1_in_cam0[0:3, 3]

print("######################## ground truth ########################")


print(R)
print(T)


######################## we use stereoCalibrate to find  the pose of cam1 in cam0 ########################

# objectPoints = [objectPointsInObjectCoordinate]
# imagePoints1 = [imagePoints_cam0.reshape(-1, 1, 2)]
# imagePoints2 = [imagePoints_cam1.reshape(-1, 1, 2)]


# Camera matrices and distortion coefficients (assuming no distortion in simulation)
cameraMatrix1 = K.copy()
cameraMatrix2 = K.copy()
distCoeffs1 = np.zeros(4)
distCoeffs2 = np.zeros(4)

# Image size
imageSize = (numberOfPixelInWidth, numberOfPixelInHeight)

# Perform stereo calibration
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    objectPoints,
    imagePoints1,
    imagePoints2,
    cameraMatrix1, distCoeffs1,
    cameraMatrix2, distCoeffs2,
    imageSize,
    flags=cv2.CALIB_FIX_INTRINSIC
)

print("########################  stereoCalibrate ########################")
print("Stereo Calibration Output:")
print("Rotation matrix (R):\n", R)
print("Translation vector (T):\n", T)
print("Essential matrix (E):\n", E)
print("Fundamental matrix (F):\n", F)


print("######################## stereoRectify ########################")


R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrix1=K, distCoeffs1=distCoeffs, cameraMatrix2=K, distCoeffs2=distCoeffs, imageSize=(600, 600), R=R, T=T)

# print("R1, R2, P1, P2, Q, validPixROI1, validPixROI2:\n",
#       R1, R2, P1, P2, Q, validPixROI1, validPixROI2)


print("R1: (rotation matrix) for the first camera, brings points given in the unrectified first camera's coordinate system to points in the rectified first camera's coordinate system:\n", R1)

print("R2: (rotation matrix) for the second camera, brings points given in the unrectified second camera's coordinate system to points in the rectified second camera's coordinate system:\n", R2)

print("P1: projects points given in the rectified first camera coordinate system into the rectified first camera's image:\n", P1)

print("P2: projects points given in the rectified first camera coordinate system into the rectified second camera's image.:\n", P2)


######################## initUndistortRectifyMap ########################


# The function computes the joint undistortion and rectification transformation and represents the result in the form of maps for remap.
map1, map2 = cv2.initUndistortRectifyMap(
    K, distCoeffs, R1, P1, size=(600, 600), m1type=cv2.CV_32FC1)

print("map1 :\n", map1.shape)

print("map2 :\n", map2.shape)
