import cv2
import numpy as np

# Load two consecutive frames

prevImg_path = "/home/behnam/workspace/OpenCVProjects/images/opticalflow/bt_0.png"
nextImg_path = "/home/behnam/workspace/OpenCVProjects/images/opticalflow/bt_1.png"

prev_img = cv2.imread(prevImg_path, cv2.IMREAD_GRAYSCALE)
next_img = cv2.imread(nextImg_path, cv2.IMREAD_GRAYSCALE)

# Detect Shi-Tomasi corners on the base image
prev_pts = cv2.goodFeaturesToTrack(
    prev_img, maxCorners=100, qualityLevel=0.3, minDistance=7)

# Build optical flow pyramids for both frames
max_level = 3
win_size = (15, 15)
prev_pyramid = cv2.buildOpticalFlowPyramid(
    prev_img, winSize=win_size, maxLevel=max_level, withDerivatives=False)[1]
next_pyramid = cv2.buildOpticalFlowPyramid(
    next_img, winSize=win_size, maxLevel=max_level, withDerivatives=False)[1]

# Calculate optical flow using cv2.calcOpticalFlowPyrLK for each pyramid level
for level in range(max_level + 1):
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_pyramid[level],
        next_pyramid[level],
        prev_pts,
        None,
        winSize=win_size,
        maxLevel=0  # We are manually handling pyramid levels
    )
    # print(level)

    # Process or store the resulting 'next_pts' as required

    # Scale points for the next pyramid level (if not the last level)
    if level < max_level:
        prev_pts = 2.0 * prev_pts

# Visualization: Draw the optical flow results on the base image for visualization
vis_img = cv2.cvtColor(prev_img, cv2.COLOR_GRAY2BGR)
for i, (new, old) in enumerate(zip(next_pts, prev_pts)):
    print(i)
    print(status[i])

    if status[i]:
        a, b = new.ravel()
        c, d = old.ravel()
        print(a, b, c, d)
        cv2.line(vis_img, (a, b), (c, d), (0, 255, 0), 2)
        cv2.circle(vis_img, (a, b), 5, (0, 255, 0), -1)

cv2.imshow("Optical Flow", vis_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
