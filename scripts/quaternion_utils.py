"""
Quaternion and Pose Transformation Utilities

This module provides functions for working with quaternions and pose transformations,
including conversions between quaternions and rotation matrices, pose inversion,
and relative pose computation.

Quaternion convention: [q_w, q_x, q_y, q_z] where q_w is the scalar part
"""

import numpy as np


def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Args:
        q: quaternion [q_w, q_x, q_y, q_z]

    Returns:
        R: 3x3 rotation matrix
    """
    q_w, q_x, q_y, q_z = q

    # Normalize quaternion
    norm = np.sqrt(q_w**2 + q_x**2 + q_y**2 + q_z**2)
    q_w, q_x, q_y, q_z = q_w/norm, q_x/norm, q_y/norm, q_z/norm

    # Compute rotation matrix
    R = np.array([
        [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_w*q_z), 2*(q_x*q_z + q_w*q_y)],
        [2*(q_x*q_y + q_w*q_z), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_w*q_x)],
        [2*(q_x*q_z - q_w*q_y), 2*(q_y*q_z + q_w*q_x), 1 - 2*(q_x**2 + q_y**2)]
    ])

    return R


def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion.

    Args:
        R: 3x3 rotation matrix

    Returns:
        q: quaternion [q_w, q_x, q_y, q_z]
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q_w = 0.25 / s
        q_x = (R[2, 1] - R[1, 2]) * s
        q_y = (R[0, 2] - R[2, 0]) * s
        q_z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q_w = (R[2, 1] - R[1, 2]) / s
        q_x = 0.25 * s
        q_y = (R[0, 1] + R[1, 0]) / s
        q_z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q_w = (R[0, 2] - R[2, 0]) / s
        q_x = (R[0, 1] + R[1, 0]) / s
        q_y = 0.25 * s
        q_z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q_w = (R[1, 0] - R[0, 1]) / s
        q_x = (R[0, 2] + R[2, 0]) / s
        q_y = (R[1, 2] + R[2, 1]) / s
        q_z = 0.25 * s

    return np.array([q_w, q_x, q_y, q_z])


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions: q1 * q2

    Args:
        q1: first quaternion [q_w, q_x, q_y, q_z]
        q2: second quaternion [q_w, q_x, q_y, q_z]

    Returns:
        q: resulting quaternion [q_w, q_x, q_y, q_z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])


def quaternion_conjugate(q):
    """
    Compute the conjugate of a quaternion.

    Args:
        q: quaternion [q_w, q_x, q_y, q_z]

    Returns:
        q_conj: conjugate quaternion [q_w, -q_x, -q_y, -q_z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_rotate_vector(q, v):
    """
    Rotate a 3D vector using a quaternion.

    Args:
        q: rotation quaternion [q_w, q_x, q_y, q_z]
        v: 3D vector [x, y, z]

    Returns:
        v_rotated: rotated 3D vector [x, y, z]
    """
    # Convert vector to quaternion form [0, x, y, z]
    v_quat = np.array([0, v[0], v[1], v[2]])

    # Compute q * v * q_conjugate
    q_conj = quaternion_conjugate(q)
    result = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)

    # Return the vector part
    return result[1:4]


def pose_to_transformation_matrix(position, quaternion):
    """
    Convert position and quaternion to a 4x4 transformation matrix.

    Args:
        position: 3D position [x, y, z]
        quaternion: rotation quaternion [q_w, q_x, q_y, q_z]

    Returns:
        T: 4x4 transformation matrix
    """
    R = quaternion_to_rotation_matrix(quaternion)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = position

    return T


def transformation_matrix_to_pose(T):
    """
    Extract position and quaternion from a 4x4 transformation matrix.

    Args:
        T: 4x4 transformation matrix

    Returns:
        position: 3D position [x, y, z]
        quaternion: rotation quaternion [q_w, q_x, q_y, q_z]
    """
    position = T[0:3, 3]
    R = T[0:3, 0:3]
    quaternion = rotation_matrix_to_quaternion(R)

    return position, quaternion


def inverse_pose(pose):
    """
    Compute the inverse of a pose.

    If pose represents frame B in frame A, the inverse represents frame A in frame B.

    Args:
        pose: [x, y, z, q_w, q_x, q_y, q_z] - position and orientation

    Returns:
        inverse_pose: [x_inv, y_inv, z_inv, q_w_inv, q_x_inv, q_y_inv, q_z_inv]
    """
    # Extract position and orientation from the pose
    x, y, z, q_w, q_x, q_y, q_z = pose
    position = np.array([x, y, z])
    quaternion = np.array([q_w, q_x, q_y, q_z])

    # Compute the conjugate of the orientation quaternion
    q_inv = quaternion_conjugate(quaternion)

    # Rotate the negated position by the conjugate quaternion
    position_inv = quaternion_rotate_vector(q_inv, -position)

    return [position_inv[0], position_inv[1], position_inv[2],
            q_inv[0], q_inv[1], q_inv[2], q_inv[3]]


def relative_pose(pose_B_in_A, pose_C_in_B):
    """
    Compute the relative pose of frame C in frame A given:
    - Pose of frame B in frame A
    - Pose of frame C in frame B

    Args:
        pose_B_in_A: [x, y, z, q_w, q_x, q_y, q_z] - pose of B in A
        pose_C_in_B: [x, y, z, q_w, q_x, q_y, q_z] - pose of C in B

    Returns:
        pose_C_in_A: [x, y, z, q_w, q_x, q_y, q_z] - pose of C in A
    """
    # Extract positions and orientations
    P_B_A = np.array(pose_B_in_A[0:3])
    Q_B_A = np.array(pose_B_in_A[3:7])

    P_C_B = np.array(pose_C_in_B[0:3])
    Q_C_B = np.array(pose_C_in_B[3:7])

    # Combined rotation: Q_C_A = Q_B_A * Q_C_B
    Q_C_A = quaternion_multiply(Q_B_A, Q_C_B)

    # Combined translation: P_C_A = P_B_A + Q_B_A.rotate(P_C_B)
    P_C_A = P_B_A + quaternion_rotate_vector(Q_B_A, P_C_B)

    return [P_C_A[0], P_C_A[1], P_C_A[2],
            Q_C_A[0], Q_C_A[1], Q_C_A[2], Q_C_A[3]]


def normalize_quaternion(q):
    """
    Normalize a quaternion to unit length.

    Args:
        q: quaternion [q_w, q_x, q_y, q_z]

    Returns:
        q_normalized: normalized quaternion
    """
    norm = np.sqrt(np.sum(q**2))
    return q / norm


if __name__ == "__main__":
    # Test quaternion_to_rotation_matrix and rotation_matrix_to_quaternion
    print("Test 1: Quaternion <-> Rotation Matrix Conversion")
    q_test = np.array([0.7071, 0, 0.7071, 0])  # 90° rotation around Y-axis
    R = quaternion_to_rotation_matrix(q_test)
    print(f"Original quaternion: {q_test}")
    print(f"Rotation matrix:\n{R}")
    q_back = rotation_matrix_to_quaternion(R)
    print(f"Quaternion from matrix: {q_back}")
    print()

    # Test inverse_pose
    print("Test 2: Inverse Pose")
    pose = [1, 2, 3, 0.7071, 0, 0, 0.7071]
    pose_inv = inverse_pose(pose)
    print(f"Original pose: {pose}")
    print(f"Inverse pose: {pose_inv}")
    print()

    # Test relative_pose
    print("Test 3: Relative Pose")
    # 45 degrees around y-axis
    pose_B_A = [1, 2, 3, 0.9239, 0, 0.3827, 0]
    # 30 degrees around x-axis
    pose_C_B = [2, 0, 1, 0.9659, 0.2588, 0, 0]
    pose_C_A = relative_pose(pose_B_A, pose_C_B)
    print(f"Pose of B in A: {pose_B_A}")
    print(f"Pose of C in B: {pose_C_B}")
    print(f"Pose of C in A: {pose_C_A}")
