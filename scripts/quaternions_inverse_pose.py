def inverse_pose(pose):
    # Extract position and orientation from the pose
    x, y, z, q1, q2, q3, q4 = pose

    # Compute the conjugate of the orientation quaternion
    q1_inv, q2_inv, q3_inv, q4_inv = q1, -q2, -q3, -q4

    # Express the negated position as a quaternion
    q_pos = [0, -x, -y, -z]

    # Multiply the quaternions: q_inv * q_pos * q
    # We'll break this down into two quaternion multiplications:
    # First, q_inv * q_pos
    a1, b1, c1, d1 = q1_inv, q2_inv, q3_inv, q4_inv
    a2, b2, c2, d2 = q_pos

    q_temp = [
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2
    ]

    # Next, multiply q_temp by q
    a1, b1, c1, d1 = q_temp
    a2, b2, c2, d2 = q1, q2, q3, q4

    q_result = [
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2
    ]

    # Extract the transformed position of the world in frame A
    x_inv, y_inv, z_inv = q_result[1], q_result[2], q_result[3]

    return [x_inv, y_inv, z_inv, q1_inv, q2_inv, q3_inv, q4_inv]

# Test
pose = [1, 2, 3, 0.7071, 0, 0, 0.7071]
print(inverse_pose(pose))
