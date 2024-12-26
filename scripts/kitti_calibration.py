import numpy as np
import cv2


def parse_kitti_calibration(file_path):
    """
    Parses a KITTI calibration file and returns a dictionary of projection matrices.

    :param file_path: Path to the KITTI calibration file.
    :return: Dictionary of projection matrices (P0, P1, P2, P3).
    """
    calibration_data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        key, values = line.split(':', 1)
        matrix = np.array([float(v) for v in values.split()]).reshape(3, 4)
        calibration_data[key.strip()] = matrix

    return calibration_data


def decompose_projection_matrix(P):
    """
    Decomposes a 3x4 projection matrix into intrinsic and extrinsic parameters using OpenCV.

    :param P: 3x4 projection matrix.
    :return: Tuple (K, R, T) where:
             K is the 3x3 intrinsic matrix,
             R is the 3x3 rotation matrix,
             T is the 3x1 translation vector.
    """
    # Use OpenCV's decomposeProjectionMatrix
    P = np.array(P, dtype=np.float64)
    K, R, T, _, _, _, _ = cv2.decomposeProjectionMatrix(P)

    # Normalize K to ensure K[2, 2] = 1
    K /= K[2, 2]

    # Convert T to a 3x1 vector
    T = (T[:3] / T[3]).reshape(3, 1)

    return K, R, T


def load_and_decompose_calibration(file_path):
    """
    Loads a KITTI calibration file and decomposes each projection matrix.

    :param file_path: Path to the KITTI calibration file.
    :return: Dictionary with decomposed matrices for each camera (P0, P1, P2, P3).
    """
    calibration_data = parse_kitti_calibration(file_path)
    decomposed_data = {}

    for key, P in calibration_data.items():
        K, R, T = decompose_projection_matrix(P)
        decomposed_data[key] = {
            "Projection Matrix": P,
            "Intrinsic Matrix": K,
            "Rotation Matrix": R,
            "Translation Vector": T
        }

    return decomposed_data


if __name__ == "__main__":
    # Example usage

    calibration_file = "/home/behnam/workspace/OpenCVProjects/data/kitti/05/calib.txt"

    decomposed_data = load_and_decompose_calibration(calibration_file)

    for camera, data in decomposed_data.items():
        print(f"Camera: {camera}")
        print("Projection Matrix:")
        print(data["Projection Matrix"])
        print("Intrinsic Matrix:")
        print(data["Intrinsic Matrix"])
        print("Rotation Matrix:")
        print(data["Rotation Matrix"])
        print("Translation Vector:")
        print(data["Translation Vector"])
        print("-")
