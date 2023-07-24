import cv2
import numpy as np


def drawCorrespondences(img1, img2, detector, matcher, top_matches=50):
    # Find the keypoints and descriptors with ORB, SIFT, etc
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # Match descriptors
    matches = matcher.match(des1, des2)

    # Sort them based on the distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the top matches correspondences
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:top_matches], None, flags=2)

    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Load the images

img1_file_path = '/home/behnam/Pictures/colmap_projects/GroupeE_Maigrauge/images/00002.png'
img2_file_path = '/home/behnam/Pictures/colmap_projects/GroupeE_Maigrauge/images/00003.png'

img1 = cv2.imread(img1_file_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_file_path, cv2.IMREAD_GRAYSCALE)

assert img1 is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"

w = 640
h = 480
img1 = cv2.resize(img1, (w, h))
img2 = cv2.resize(img2, (w, h))

# Initialize ORB detector
orb = cv2.ORB_create()
print(orb.descriptorType())
# Use the BFMatcher (Brute Force Matcher) and Hamming distance
# Basics of Brute-Force Matcher

# distance measurement to be used. By default, it is cv.NORM_L2. It is good for SIFT, SURF etc (cv.NORM_L1 is also there). For binary string based descriptors like ORB, BRIEF, BRISK etc, cv.NORM_HAMMING should be used, which used Hamming distance as measurement. If ORB is using WTA_K == 3 or 4, cv.NORM_HAMMING2 should be used.

# BFMatcher


# FlannBasedMatcher

# KNN


# GMS Matcher


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


# Initialize SIFT detector
sift = cv2.SIFT_create()


print(sift.getDefaultName())
print(sift.descriptorType())


bf = cv2.BFMatcher()


cv2.FlannBasedMatcher()

drawCorrespondences(img1, img2, sift, bf)
