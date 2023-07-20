import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img_path = "/home/behnam/workspace/OpenCVProjects/images/opticalflow/bt_0.png"
img = cv.imread(img_path)


############################ ORB ############################
detector = cv.ORB_create()
img_pts = detector.detect(img, None)
img_pts, img_descriptor = detector.compute(img, img_pts)


# draw only keypoints location,not size and orientation
prevImg_marked = cv.drawKeypoints(
    img, img_pts, None, color=(0, 255, 0), flags=0)

plt.imshow(prevImg_marked), plt.show()


print(type(img_pts))
for i in img_pts:
    print("point: ", i.pt)
    print("angle: ", i.angle)
    print("class_id: ", i.class_id)
    print("octave: ", i.octave)
    print("response: ", i.response)
    print("size: ", i.size)


############################ FastFeatureDetector ############################

img = cv.imread(img_path)
detector = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_pts = detector.detect(img, None)


img_marked = cv.drawKeypoints(
    img, img_pts, None, color=(0, 0, 255), flags=0)

plt.imshow(img_marked), plt.show()


############################ convert vector of keypoints to vector of points  ############################

# This method converts vector of keypoints to vector of points -> Array of (x,y) coordinates of each keypoint


pts = cv2.KeyPoint_convert(kp)

pts = np.float([key_point.pt for key_point in kp]).reshape(-1, 1, 2)


############################ goodFeaturesToTrack ############################


# params for corner detection

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

keypoints = cv.goodFeaturesToTrack(img_gray, mask=None,
                                   **feature_params)

print(type(keypoints))
for i in keypoints:
    print("point: ", i)
