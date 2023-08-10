import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import cv2


def matrix_to_quat_and_position(rotation_matrix):
    """Converts a rotation matrix and translation vector to quaternion and position."""

    # Extract the rotation components
    r11, r12, r13 = rotation_matrix[0]
    r21, r22, r23 = rotation_matrix[1]
    r31, r32, r33 = rotation_matrix[2]

    # Calculate the trace of the matrix
    trace = r11 + r22 + r33

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r32 - r23) * s
        y = (r13 - r31) * s
        z = (r21 - r12) * s
    else:
        if r11 > r22 and r11 > r33:
            s = 2.0 * np.sqrt(1.0 + r11 - r22 - r33)
            w = (r32 - r23) / s
            x = 0.25 * s
            y = (r12 + r21) / s
            z = (r13 + r31) / s
        elif r22 > r33:
            s = 2.0 * np.sqrt(1.0 + r22 - r11 - r33)
            w = (r13 - r31) / s
            x = (r12 + r21) / s
            y = 0.25 * s
            z = (r23 + r32) / s
        else:
            s = 2.0 * np.sqrt(1.0 + r33 - r11 - r22)
            w = (r21 - r12) / s
            x = (r13 + r31) / s
            y = (r23 + r32) / s
            z = 0.25 * s

    quat = np.array([w, x, y, z])

    return quat


def rotation_matrix_to_quaternion(matrix):
    """Convert a rotation matrix to quaternion."""
    q0 = np.sqrt(1 + matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) / 2
    q1 = (matrix[2, 1] - matrix[1, 2]) / (4 * q0)
    q2 = (matrix[0, 2] - matrix[2, 0]) / (4 * q0)
    q3 = (matrix[1, 0] - matrix[0, 1]) / (4 * q0)
    return [q0, q1, q2, q3]


base_path = '/home/behnam/workspace/OpenCVProjects/'
ground_truth_poses = "/data/kitti/poses/05.txt"
ground_truth_poses_abs_path = base_path+ground_truth_poses


if not os.path.exists(base_path+ground_truth_poses):
    print("no file")
    exit()

poses = pd.read_csv(ground_truth_poses_abs_path, sep=" ", header=None)

# r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz

# colmap format:
# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

CAMERA_ID = 1
NAME = ""
length = 6
gt = np.zeros((len(poses), 3, 4))
for i in range(len(poses)):
    gt[i] = np.array(poses.iloc[i]).reshape((3, 4))

    rotation_matrix = np.array(poses.iloc[i]).reshape((3, 4))[:, 0:3]
    translation_vector = np.array(poses.iloc[i]).reshape((3, 4))[:, 3]

    # -R^t * T
    translation_vector = np.dot(- np.transpose(rotation_matrix),
                                translation_vector)

    rotation_matrix = np.transpose(rotation_matrix)

    # print(rotation_matrix.shape)

    quat = rotation_matrix_to_quaternion(rotation_matrix)

    QW, QX, QY, QZ = quat
    TX, TY, TZ = translation_vector
    IMAGE_ID = i+1
    # IMAGE_ID = i

    # This creates a format string like "{:0>5}" for length = 5
    format_string = f"{{:0>{length}}}"
    NAME = format_string.format(i)+".png"

    str = f"{IMAGE_ID} {QW} {QX} {QY} {QZ} {TX} {TY} {TZ} {CAMERA_ID} {NAME} \n".format(
        IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME)
    print(str)


fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(gt[:, :, 3][:, 0], gt[:, :, 3][:, 1], gt[:, :, 3][:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# ax.view_init(elev=-40, azim=270)
plt.show()


# # Example usage:
# rotation_matrix = np.array([
#     [0.8660254, -0.5,       0],
#     [0.5,       0.8660254, 0],
#     [0,         0,         1]
# ])
# translation_vector = np.array([1, 2, 3])

# quat, position = matrix_to_quat_and_position(
#     rotation_matrix, translation_vector)
# print("Quaternion:", quat)
# print("Position:", position)


######################### Projection Matrices/ LIDAR #########################
# Matrices for 4 cameras projection,  3x4 projection matrices, P0, P1, P2, P3, Tr(LIDAR)
cameras_file_path = "/data/kitti/05/calib.txt"
cameras_file_path_abs_path = base_path+cameras_file_path


calib = pd.read_csv(cameras_file_path_abs_path,
                    delimiter=' ', header=None, index_col=0)

P0 = np.array(calib.loc['P0:']).reshape((3, 4))
# print(P0)


# decomposition of a projection matrix into a calibration and a rotation matrix and the position of a camera.
# It optionally returns three rotation matrices, one for each axis,
cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv2.decomposeProjectionMatrix(
    P0)

# print(cameraMatrix)
