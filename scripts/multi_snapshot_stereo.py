import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(suppress=True)
# np.set_printoptions(threshold=np.inf)


def plotCameraAndChessboard(objectPoints, cameraPoses, cameraOrientation):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the chessboard
    for point in objectPoints:
        ax.scatter(point[0], point[1], point[2], color='black')

    # Plotting the cameras
    for pose, orientation in zip(cameraPoses, cameraOrientation):
        # Each camera is represented as a set of axes
        # The camera position is 'pose' and orientation is 'orientation'
        # Creating axes for the camera
        x_axis = orientation @ np.array([0.5, 0, 0])
        y_axis = orientation @ np.array([0, 0.5, 0])
        z_axis = orientation @ np.array([0, 0, 0.5])
        ax.quiver(*pose, *x_axis, color='red', length=0.5)
        ax.quiver(*pose, *y_axis, color='green', length=0.5)
        ax.quiver(*pose, *z_axis, color='blue', length=0.5)
    # Setting labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Poses and Chessboard Visualization')
    plt.show()

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

def simulateStereoCamerasAndProjectPoints(objPoints, cameraMatrix, distCoeffs, rotationMatrix, translationVector):
    # Ensure translation vector is a 3x1 array
    translationVector = np.array(
        translationVector, dtype=np.float32).reshape(3, 1)

    # Project points
    imagePoints, _ = cv2.projectPoints(
        objPoints, rotationMatrix, translationVector, cameraMatrix, distCoeffs)
    return imagePoints

def createImageFromPoints(imagePoints, imageSize):
    image = np.zeros((imageSize[1], imageSize[0]), dtype=np.uint8)
    for p in imagePoints:
        x, y = int(p[0][0]), int(p[0][1])
        cv2.circle(image, (x, y), radius=2,
                   color=(255, 255, 255), thickness=-1)
    return image


# Define camera parameters (assuming same for both cameras)
focal_length = 400
imageSize = (800, 800)  # Assuming an 800x800 image size

center = (imageSize[0]/2, imageSize[1]/2)
cameraMatrix = np.array([[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype=np.float32)
distCoeffs = np.zeros(4)  # Assuming no lens distortion


# Define stereo camera positions and orientations
# Assuming first camera looks straight at the chessboard
rotationMatrix_cam0 = np.eye(3)
# Assuming first camera is 10 units away
t_cam0 = np.array([0, 0, -10])

# Assuming second camera also looks straight but from a different angle
rotationMatrix_cam1 = np.eye(3)
# Slightly to the right of the first camera
t_cam1 = np.array([0.5, 0, -10])

# Define different chessboard center positions
chessboard_positions = [
    (0, +4, 1.1),
    (0.5, +4, 1.1),
    (-0.5, +4, 1.1)
]

# Other necessary parameters
squareSize = 0.4
numberOfRows = 6
numberOfCols = 7


# Lists to hold all projected points for each camera
all_imagePoints_cam0 = []
all_imagePoints_cam1 = []

for center_x, center_y, center_z in chessboard_positions:
    # Generate object points for this chessboard position
    _, objectPointsInWorldCoordinate = createChessBoardInWorldCoordinate(
        center_x, center_y, center_z, squareSize, numberOfRows, numberOfCols)

    # Project points onto each camera's image plane
    imagePoints_cam0 = simulateStereoCamerasAndProjectPoints(
        objectPointsInWorldCoordinate, cameraMatrix, distCoeffs, rotationMatrix_cam0, -t_cam0)

    leftImage = createImageFromPoints(imagePoints_cam0, imageSize)

    # print("imagePoints_cam0:\n", imagePoints_cam0)

    imagePoints_cam1 = simulateStereoCamerasAndProjectPoints(
        objectPointsInWorldCoordinate, cameraMatrix, distCoeffs, rotationMatrix_cam1, -t_cam1)

    rightImage = createImageFromPoints(imagePoints_cam1, imageSize)

    cv2.imshow('Left Image', leftImage)
    cv2.imshow('Right Image', rightImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Add to lists
    all_imagePoints_cam0.append(imagePoints_cam0.reshape(-1, 1, 2))
    all_imagePoints_cam1.append(imagePoints_cam1.reshape(-1, 1, 2))


# Prepare object points for stereoCalibrate
objectPoints = [createChessBoardInWorldCoordinate(
    0, 0, 0, squareSize, numberOfRows, numberOfCols)[0]] * len(chessboard_positions)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    objectPoints, all_imagePoints_cam0, all_imagePoints_cam1, cameraMatrix, distCoeffs, cameraMatrix, distCoeffs, imageSize)


print("########################  stereoCalibrate ########################")
print("Stereo Calibration Output:")
print("Rotation matrix (R):\n", R)
print("Translation vector (T):\n", T)
print("Essential matrix (E):\n", E)
print("Fundamental matrix (F):\n", F)


print("########################  ground truth ########################")

print(t_cam0-t_cam1)


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


# Applying remap to the left and right images (assuming you have leftImage and rightImage)
left_image_rectified = cv2.remap(leftImage, map1x, map1y, cv2.INTER_LINEAR)
right_image_rectified = cv2.remap(rightImage, map2x, map2y, cv2.INTER_LINEAR)


# _, objectPointsInWorldCoordinate = createChessBoardInWorldCoordinate(
#     0, 0, 0, squareSize, numberOfRows, numberOfCols)

# cameraPoses = [t_cam0, t_cam1]
# cameraOrientations = [rotationMatrix_cam0, rotationMatrix_cam1]

# plotCameraAndChessboard(objectPointsInWorldCoordinate,
#                         cameraPoses, cameraOrientations)


# Create a stereo matcher object
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Compute the disparity map
disparity = stereo.compute(left_image_rectified, right_image_rectified)

# Normalize the disparity map (for visualization)
disparity_visual = cv2.normalize(
    disparity, None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
disparity_visual = np.uint8(disparity_visual)

# Compute the 3D points
points_3D = cv2.reprojectImageTo3D(disparity, Q)

# Filter points based on disparity map
mask = disparity > disparity.min()
points_3D = points_3D[mask]

# Now, points_3D contains the 3D coordinates of the points
print(points_3D)
