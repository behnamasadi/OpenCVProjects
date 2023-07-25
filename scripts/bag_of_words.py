import cv2
import numpy as np

# Step 1: Feature Extraction
sift = cv2.SIFT_create()
train_images = ["path_to_image1.jpg", "path_to_image2.jpg", ...]
all_descriptors = []

for img_path in train_images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)

    # Stacking all descriptors
    if len(all_descriptors) == 0:
        all_descriptors = des
    else:
        all_descriptors = np.vstack((all_descriptors, des))

# Step 2: Clustering
k = 100  # Number of clusters or "visual words"
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, vocab = cv2.kmeans(
    all_descriptors, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Step 3: Histogram Construction
bow_train_data = []

for img_path in train_images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)

    # Compute histogram for the current image
    hist, _ = np.histogram(labels, bins=k, range=(0, k))
    bow_train_data.append(hist)

# The variable `bow_train_data` now contains the Bag of Words representation for each training image.
# You can then use this data for image classification tasks.
