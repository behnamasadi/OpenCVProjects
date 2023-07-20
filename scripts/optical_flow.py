import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

# OPTFLOW_USE_INITIAL_FLOW
# https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#
prevImg_path = "/home/behnam/workspace/OpenCVProjects/images/opticalflow/bt_0.png"
nextImg_path = "/home/behnam/workspace/OpenCVProjects/images/opticalflow/bt_1.png"

prevImg_gray = cv.imread(prevImg_path, cv.COLOR_BGR2GRAY)
nextImg_gray = cv.imread(nextImg_path, cv.COLOR_BGR2GRAY)


# params for corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


p0 = cv.goodFeaturesToTrack(prevImg_gray, mask=None,
                            **feature_params)

p1 = cv.goodFeaturesToTrack(nextImg_gray, mask=None,
                            **feature_params)


lk_params = dict(winSize=(21, 21), criteria=(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)),


# print(type(prevPts))
# for i in prevPts:
#     print(i.pt)
#     print(i.angle)
#     print(i.class_id)
#     print(i.octave)
#     print(i.response)
#     print(i.size)


opticalFlowNextPts, status, err = cv.calcOpticalFlowPyrLK(
    prevImg_gray, nextImg_gray, p0, p1)


print(status)

print(type(opticalFlowNextPts))
print(opticalFlowNextPts.shape)

print(len(opticalFlowNextPts))
print(len(status))


detector = cv.ORB_create()
prevPts = detector.detect(prevImg_gray, None)
nextPts = detector.detect(nextImg_gray, None)


good_new = opticalFlowNextPts[status == 1]
# good_old = prevPts[status == 1]


prevPts2f = cv.KeyPoint.convert(prevPts)
nextPts2f = cv.KeyPoint.convert(nextPts)


print("prevPts2f:", type(prevPts2f))
print("nextPts2f:", type(nextPts2f))


# OPTFLOW_USE_INITIAL_FLOW uses initial estimations, stored in nextPts; if the flag is not set, then prevPts is copied to nextPts and is considered the initial estimate.
opticalFlowNextPts, status, err = cv.calcOpticalFlowPyrLK(
    prevImg_gray, nextImg_gray, prevPts2f, nextPts2f, cv.OPTFLOW_USE_INITIAL_FLOW)


print("opticalFlowNextPts:", type(opticalFlowNextPts))


opticalFlowNextPts = opticalFlowNextPts.reshape(-1, 1, 2)
print("opticalFlowNextPts:", opticalFlowNextPts.shape)

# # print(err)


# # print(len(nextPts))


# print("len(status):", len(status))
# print("status:", status)

# # print(len(prevPts))

# print(len(opticalFlowNextPts))


# foo = cv.KeyPoint_convert(opticalFlowNextPts)

# print("foo:", foo)


good_new = opticalFlowNextPts[status == 1]

prevPts2f = prevPts2f.reshape(-1, 1, 2)

print("prevPts2f.shape:", prevPts2f.shape)


good_old = prevPts2f[status == 1]


# Create some random colors
color = np.random.randint(0, 255, (1000, 3))

# Create a mask image for drawing purposes
mask = np.zeros_like(prevImg_gray)
# draw the tracks
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()

    print(a, b, c, d)
    mask = cv.line(mask, (int(a), int(b)),
                   (int(c), int(d)), color[i].tolist(), 2)
    nextImg_gray = cv.circle(
        nextImg_gray, (int(a), int(b)), 5, color[i].tolist(), -1)
img = cv.add(nextImg_gray, mask)

cv.imshow('frame', img)
k = cv.waitKey(0) & 0xff


# # # pts = cv2.KeyPoint_convert(kp)
# # # import numpy as np

# # # pts = np.float([key_point.pt for key_point in kp]).reshape(-1, 1, 2)
# # # p1, st, err = cv.calcOpticalFlowPyrLK(prevImg, nextImg, p0, None, **lk_params)


# # # prevPts, prevDes = detector.compute(prevImg, prevPts)
# # # nextPts, nextDes = detector.compute(prevImg, prevPts)


# # # # draw only keypoints location,not size and orientation
# # # prevImg_marked = cv.drawKeypoints(
# # #     prevImg, prevPts, None, color=(0, 255, 0), flags=0)

# # # # plt.imshow(prevImg_marked), plt.show()


# # nextImg_marked = cv.drawKeypoints(
# #     nextImg, nextPts, None, color=(0, 255, 0), flags=0)

# # # plt.imshow(nextImg_marked), plt.show()


# # # detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)):
