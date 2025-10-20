import os
import cv2
import sys
from utils.file_utils import resource_path


def drawCorrespondences(img1, img2, detector, matcher, top_k_matches=10):
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

        print(len(matches))
        matches = matches[:int(len(matches)/top_k_matches)]

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

    title = f"Detector Name: {detector.getDefaultName()}, Detector Type:{
        detector.descriptorType()}, Matcher: {type(matcher).__name__}"
    cv2.imshow(title, img_matches)
    cv2.imwrite(title+".png", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # Check if a relative path is provided as a command-line argument
    if len(sys.argv) > 1:
        relative_path = sys.argv[1]
    else:
        relative_path = "../images/00/image_2/"

    directory_path_abs_path = resource_path(relative_path)

    # Check if the directory exists
    if not os.path.exists(directory_path_abs_path):
        print(f"Directory '{directory_path_abs_path}' does not exist.")
        sys.exit(1)
    print("loading images in:", directory_path_abs_path)

    # List and sort the files in the directory
    file_list = os.listdir(directory_path_abs_path)
    file_list.sort()

    # Ensure the file list is not empty
    if len(file_list) == 0:
        print(f"No files found in directory '{directory_path_abs_path}'.")
        sys.exit(1)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    orb_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    orb = cv2.ORB_create()
    #  detectors
    sift = cv2.SIFT_create()

    for i in range(len(file_list)-1):

        img1 = cv2.imread(os.path.join(directory_path_abs_path,
                          file_list[i]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(directory_path_abs_path,
                          file_list[i+1]), cv2.IMREAD_GRAYSCALE)

        assert img1 is not None, "file could not be read,"
        assert img2 is not None, "file could not be read,"

        w = int(1.5*640)
        h = int(1.5*480)
        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))

        drawCorrespondences(img1, img2, orb, orb_bf)
