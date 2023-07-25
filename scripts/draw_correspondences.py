import cv2
import numpy as np
import argparse


def drawCorrespondences(img1, img2, detector, matcher, top_matches=50):
    # Find the keypoints and descriptors with ORB, SIFT, etc
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    matches = []
    print("matcher type:", type(matcher).__name__)

    if (type(matcher).__name__ == "BFMatcher"):
        # Match descriptors
        matches = matcher.match(des1, des2)

        # Sort them based on the distance
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:top_matches]

    elif type(matcher).__name__ == "FlannBasedMatcher":

        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = []

        # Ratio Test:  For each keypoint, if the distance ratio between the best and the second-best match is below a
        # threshold (usually around 0.7 to 0.8), the match is retained.
        ratio_rest = 0.7
        for m, n in matches:
            if m.distance < ratio_rest * n.distance:
                good_matches.append(m)
        matches = good_matches

    # Draw the top matches correspondences
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

    title = f"Detector Name: {detector.getDefaultName()}, Detector Type:{detector.descriptorType()}, Matcher: {type(matcher).__name__ }"
    cv2.imshow(title, img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Load the images

img1_file_path = '/home/behnam/workspace/OpenCVProjects/images/opticalflow/bt_0.png'
img2_file_path = '/home/behnam/workspace/OpenCVProjects/images/opticalflow/bt_1.png'


img1 = cv2.imread(img1_file_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_file_path, cv2.IMREAD_GRAYSCALE)

assert img1 is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"

w = int(1.5*640)
h = int(1.5*480)
img1 = cv2.resize(img1, (w, h))
img2 = cv2.resize(img2, (w, h))

#  detectors
orb = cv2.ORB_create()
sift = cv2.SIFT_create()


# Matcher
# BFMatcher
# For binary string based descriptors like ORB, BRIEF, BRISK etc, cv.NORM_HAMMING should be used, which used Hamming distance as measurement. If ORB is using WTA_K == 3 or 4, cv.NORM_HAMMING2 should be used.
orb_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Distance measurement by default, is cv.NORM_L2. It is good for SIFT, SURF etc. cv.NORM_L1 is also available
sift_bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


# FlannBasedMatcher
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)


drawCorrespondences(img1, img2, sift, flann_matcher)
