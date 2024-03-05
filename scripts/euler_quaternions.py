import numpy as np
import math

def rotation_matrix_to_roll_pitch_yaw(R):
    """
    Convert a rotation matrix to Euler angles.
    
    Parameters:
    R (numpy.ndarray): A 3x3 transformation matrix.
    
    Returns:
    tuple: A tuple containing the roll, pitch, and yaw angles.
    """
    # Ensure the input is a 3x3 matrix
    assert(R.shape == (3, 3))

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return x,y,z
    #return np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)  # Convert to degrees

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
    Converts a quaternion into a rotation matrix.
    """
    w, x, y, z = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),       2 * (xz + wy)],
        [2 * (xy + wz),         1 - 2 * (xx + zz),   2 * (yz - wx)],
        [2 * (xz - wy),         2 * (yz + wx),       1 - 2 * (xx + yy)]
    ])
    return R

def rotation_matrix_to_quaternion(R):
    """
    Converts a rotation matrix to a quaternion.
    """
    w = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z])

def quaternion_multiplication(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    q = np.array([w, x, y, z])
    #return q / np.linalg.norm(q)  # Normalize the quaternion
    return q

def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_rotate(q, v):
    qv = np.array([0.0] + v.tolist())
    return quaternion_multiplication(quaternion_multiplication(q, qv), quaternion_conjugate(q))[1:]

def compute_transformations(QA_B, QB_C, PA_B, PB_C):
    """
    Computes the combined rotation and translation.
    """
    QA_C = quaternion_multiplication(QA_B, QB_C)
    PA_C = PA_B + quaternion_rotate(QA_B, PB_C)
    return QA_C, PA_C


if __name__ == "__main__":

    np.set_printoptions(suppress=True)
    roll=np.pi/2
    pitch= np.pi/4
    yaw=np.pi/3


    print(f"ground truth Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")


    rotation_matrix=rotation_matrix_from_roll_pitch_yaw(roll, pitch, yaw)
    print("rotation_matrix: ",rotation_matrix)




    roll, pitch, yaw = rotation_matrix_to_roll_pitch_yaw(rotation_matrix)
    print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")



    translation_vector = np.array([1, 2, 3])

    quat = rotation_matrix_to_quaternion_simple(rotation_matrix)
    print("Quaternion:", quat)
    # print("Position:", position)


    print("Quaternion:",rotation_matrix_to_quaternion(rotation_matrix))



    # Example usage:
    QA_B = np.array([0.707, 0, 0.707, 0])  # Example quaternion
    QB_C = np.array([0.707, 0.707, 0, 0])  # Example quaternion
    PA_B = np.array([1, 0, 0])             # Example position
    PB_C = np.array([0, 1, 0])             # Example position

    QA_C, PA_C = compute_transformations(QA_B, QB_C, PA_B, PB_C)

    print("QA_C:", QA_C)
    print("PA_C:", PA_C)

