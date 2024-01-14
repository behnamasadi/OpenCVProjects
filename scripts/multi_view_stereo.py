import numpy as np
import cv2
np.set_printoptions(suppress=True)


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


def createChessboardCorners(squareSize, numberOfRows, numberOfCols):
    objp = np.zeros((numberOfRows*numberOfCols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:numberOfCols, 0:numberOfRows].T.reshape(-1, 2)
    objp *= squareSize
    return objp


def simulateStereoCamerasAndProjectPoints(objPoints, cameraMatrix, distCoeffs, rotationMatrix, translationVector):
    # Ensure translation vector is a 3x1 array
    translationVector = np.array(
        translationVector, dtype=np.float32).reshape(3, 1)

    # Project points
    imagePoints, _ = cv2.projectPoints(
        objPoints, rotationMatrix, translationVector, cameraMatrix, distCoeffs)
    return imagePoints


# Define different chessboard center positions
chessboard_positions = [
    (0, 4, 1.1),
    (0.5, 4, 1.1),
    (-0.5, 4, 1.1)
]

# Other necessary parameters
squareSize = 0.2
numberOfRows = 6
numberOfCols = 7


# Define camera parameters (assuming same for both cameras)
focal_length = 800
center = (400, 400)
cameraMatrix = np.array([[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype=np.float32)
distCoeffs = np.zeros(4)  # Assuming no lens distortion

# Define stereo camera positions and orientations
# Assuming first camera looks straight at the chessboard
rotationMatrix1 = np.eye(3)
# Assuming first camera is 10 units away
translationVector1 = np.array([0, 0, -10])

# Assuming second camera also looks straight but from a different angle
rotationMatrix2 = np.eye(3)
# Slightly to the right of the first camera
translationVector2 = np.array([0.5, 0, -10])

# Generate chessboard corners
# Square size of 0.2 units, 6x7 chessboard
objPoints = createChessboardCorners(0.2, 6, 7)
print("objPoints:\n", objPoints)

# Project points onto each camera's image plane
imagePoints1 = simulateStereoCamerasAndProjectPoints(
    objPoints, cameraMatrix, distCoeffs, rotationMatrix1, translationVector1)

print("imagePoints1:\n", imagePoints1)
imagePoints2 = simulateStereoCamerasAndProjectPoints(
    objPoints, cameraMatrix, distCoeffs, rotationMatrix2, translationVector2)
print("imagePoints2:\n", imagePoints1)
# # Prepare data for stereoCalibrate
# objectPoints = [objPoints]
# imagePoints1 = [imagePoints1]
# imagePoints2 = [imagePoints2]
imageSize = (800, 800)  # Assuming an 800x800 image size

# # Perform stereo calibration
# retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
#     objectPoints, imagePoints1, imagePoints2, cameraMatrix, distCoeffs, cameraMatrix, distCoeffs, imageSize
# )
# Prepare data for stereoCalibrate
objectPoints = [objPoints]
imagePoints1 = [imagePoints1.reshape(-1, 1, 2)]
imagePoints2 = [imagePoints2.reshape(-1, 1, 2)]

# Perform stereo calibration
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    objectPoints, imagePoints1, imagePoints2, cameraMatrix, distCoeffs, cameraMatrix, distCoeffs, imageSize
)


print("########################  stereoCalibrate ########################")
print("Stereo Calibration Output:")
print("Rotation matrix (R):\n", R)
print("Translation vector (T):\n", T)
print("Essential matrix (E):\n", E)
print("Fundamental matrix (F):\n", F)


print("########################  ground truth ########################")

print(translationVector2-translationVector1)


R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1,
    cameraMatrix2, distCoeffs2,
    imageSize, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
)

map1x, map1y = cv2.initUndistortRectifyMap(
    cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv2.CV_16SC2)

map2x, map2y = cv2.initUndistortRectifyMap(
    cameraMatrix2, distCoeffs2, R2, P2, imageSize, cv2.CV_16SC2)


# # Applying remap to the left and right images (assuming you have leftImage and rightImage)
# rectified_leftImage = cv2.remap(leftImage, map1x, map1y, cv2.INTER_LINEAR)
# rectified_rightImage = cv2.remap(rightImage, map2x, map2y, cv2.INTER_LINEAR)
