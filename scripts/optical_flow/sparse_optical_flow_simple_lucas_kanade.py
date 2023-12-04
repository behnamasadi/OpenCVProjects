import cv2
from matplotlib import pyplot as plt
import numpy as np


########################## reading images ##########################
prevImg_path = "/home/behnam/workspace/OpenCVProjects/images/opticalflow/bt_0.png"
nextImg_path = "/home/behnam/workspace/OpenCVProjects/images/opticalflow/bt_1.png"

prevImg = cv2.imread(nextImg_path, cv2.IMREAD_COLOR)
nextImg = cv2.imread(nextImg_path, cv2.IMREAD_COLOR)


prevImg_gray = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
nextImg_gray = cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)

########################## Shi-Tomasi corners ##########################

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

prevPts = cv2.goodFeaturesToTrack(prevImg_gray, mask=None,
                                  **feature_params)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

nextPts, st, err = cv2.calcOpticalFlowPyrLK(
    prevImg_gray, nextImg_gray, prevPts, None, **lk_params)

# size of nextPts is N x 1 x 2

color = (0, 255, 0)  # Green color

for i, (new, old) in enumerate(zip(nextPts, prevPts)):
    a, b = new.ravel()
    c, d = old.ravel()

    # Draw a line between old and new position
    cv2.line(nextImg, (int(a), int(b)), (int(c), int(d)), color, 2)
    cv2.circle(nextImg, (int(a), int(b)), 5, color, -1)

cv2.imshow('Optical Flow', nextImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
