import numpy as np
import cv2

# Load the calibration parameters
K = np.loadtxt('camera_matrix.txt', delimiter=',')
D = np.loadtxt('distortion_coefficients.txt', delimiter=',')

# Set the initial pose to the identity matrix
T = np.eye(4)

# Initialize the feature detector and matcher
detector = cv2.FastFeatureDetector_create(threshold=25)
matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

# Process the first frame
prev_img = cv2.imread('frame_0.png', cv2.IMREAD_GRAYSCALE)
prev_pts = cv2.fisheye.undistortPoints(detector.detect(prev_img), K, D)
prev_desc = cv2.xfeatures2d.FREAK_create().compute(prev_img, prev_pts)[1]

for i in range(1, 100):
    # Load the current frame
    curr_img = cv2.imread(f'frame_{i}.png', cv2.IMREAD_GRAYSCALE)

    # Undistort the image
    curr_img_undist = cv2.fisheye.undistort(curr_img, K, D)

    # Detect and describe features in the current frame
    curr_pts = cv2.fisheye.undistortPoints(detector.detect(curr_img), K, D)
    curr_desc = cv2.xfeatures2d.FREAK_create().compute(curr_img, curr_pts)[1]

    # Match the features between the frames
    matches = matcher.match(prev_desc, curr_desc)

    # Calculate the relative pose using RANSAC
    prev_pts_ransac = np.array([prev_pts[m.queryIdx].pt for m in matches], dtype=np.float32)
    curr_pts_ransac = np.array([curr_pts[m.trainIdx].pt for m in matches], dtype=np.float32)
    E, mask = cv2.findEssentialMat(prev_pts_ransac, curr_pts_ransac, K, cv2.RANSAC, 0.999, 1.0)

    _, R, t, mask = cv2.recoverPose(E, prev_pts_ransac, curr_pts_ransac, K)

    # Update the pose
    T[:3, :3] = T[:3, :3] @ R
    T[:3, 3] += T[:3, :3] @ t.reshape(3)

    # Save the current frame as the previous frame
    prev_img = curr_img
    prev_pts = curr_pts
    prev_desc = curr_desc

    # Print the current pose
    print(f'Pose {i}:')
    print(T)

