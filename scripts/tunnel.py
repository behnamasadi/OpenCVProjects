import os
import cv2
import numpy as np
import argparse


def drawCorrespondences(img1, img2, detector, matcher, top_matches=50):
    # Find the keypoints and descriptors with ORB, SIFT, etc
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    print("kp1:", kp1)
    print("des1:", des1)
    matches = []
    # print("matcher type:", type(matcher).__name__)

    if (type(matcher).__name__ == "BFMatcher"):
        # Match descriptors
        matches = matcher.match(des1, des2)

        # Sort them based on the distance
        matches = sorted(matches, key=lambda x: x.distance)

        print(len(matches))
        # matches = matches[:top_matches]
        matches = matches[:int(len(matches)/10)]

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


directory_path = "/home/behnam/Pictures/colmap_projects/GroupeE_Maigrauge/images/"
if not os.path.exists(directory_path):
    print(f"Directory '{directory_path}' does not exist.")

file_list = os.listdir(directory_path)
file_list.sort()


img1_file_path = directory_path+file_list[0]

img1 = cv2.imread(img1_file_path, cv2.IMREAD_GRAYSCALE)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

orb_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


orb = cv2.ORB_create()
#  detectors
sift = cv2.SIFT_create()


for index, file_name in enumerate(file_list, start=1):

    img2_file_path = directory_path + file_name
    img2 = cv2.imread(img2_file_path, cv2.IMREAD_GRAYSCALE)

    assert img1 is not None, "file could not be read, check with os.path.exists()"
    assert img2 is not None, "file could not be read, check with os.path.exists()"

    w = int(1.5*640)
    h = int(1.5*480)
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))

    drawCorrespondences(img1, img2, orb, orb_bf)
    img1 = img2
