import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


# Use relative path based on the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = "../images/feature_detection_description/000000.png"
img_path_abs_path = os.path.join(script_dir, img_path)
print("reading calibration file from: ", img_path_abs_path)


img = cv2.imread(img_path_abs_path)


############################ ORB ############################
detector = cv2.ORB_create()
img_pts = detector.detect(img, None)
img_pts, img_descriptor = detector.compute(img, img_pts)


# draw only keypoints location,not size and orientation
prevImg_marked = cv2.drawKeypoints(
    img, img_pts, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

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

img = cv2.imread(img_path)
# detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
detector = cv2.ORB_create()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_pts = detector.detect(img, None)


# cv2.DRAW_MATCHES_FLAGS_DEFAULT,
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
# cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
# cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS


img_marked = cv2.drawKeypoints(
    img, img_pts, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(img_marked), plt.show()


############################ convert vector of keypoints to vector of points  ############################

point = img_pts[0]

x = point.pt[0]
y = point.pt[1]
(x, y) = point.pt


# This method converts vector of keypoints to vector of points -> Array of (x,y) coordinates of each keypoint


pts = cv2.KeyPoint_convert(img_pts)
print(pts)

pts = np.float64([key_point.pt for key_point in img_pts]).reshape(-1, 1, 2)


############################ goodFeaturesToTrack ############################


# params for corner detection

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

keypoints = cv2.goodFeaturesToTrack(img_gray, mask=None,
                                    **feature_params)

print(type(keypoints))
for i in keypoints:
    print("point: ", i)
