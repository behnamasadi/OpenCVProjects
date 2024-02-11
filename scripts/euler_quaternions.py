import numpy as np


def rotation_matrix_from_roll_pitch_yaw(roll, pitch, yaw):
    yawMatrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    pitchMatrix = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    rollMatrix = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R = yawMatrix @ pitchMatrix @ rollMatrix
    return R

def get_quaternion_from_euler(roll, pitch, yaw):
    """quaternion_to_rot_matrix
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return np.array([qw, qx, qy, qz])

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion into a rotation matrix.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def rotation_matrix_to_quaternion(rotation_matrix):
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

def rotation_matrix_to_quaternion_simple(rotation_matrix):
    """Convert a rotation matrix to quaternion."""
    q0 = np.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2
    q1 = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * q0)
    q2 = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * q0)
    q3 = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * q0)
    return [q0, q1, q2, q3]


if __name__ == "__main__":

    np.set_printoptions(suppress=True)
    roll=np.pi/2
    pitch= np.pi/4
    yaw=np.pi/3

    rotation_matrix=rotation_matrix_from_roll_pitch_yaw(roll, pitch, yaw)
    print(rotation_matrix)
    translation_vector = np.array([1, 2, 3])

    quat = rotation_matrix_to_quaternion_simple(rotation_matrix)
    print("Quaternion:", quat)
    # print("Position:", position)


    print("Quaternion:",rotation_matrix_to_quaternion(rotation_matrix))

