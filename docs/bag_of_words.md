#  Bag of Words (BoW) 

The Bag of Words (BoW) model is a technique to represent images a collection of "words" without considering the order or structure of these words. In the context of computer vision, "words" are actually **visual features** or clusters of similar features in an image. The process typically involves:

1. **Feature Extraction:** Extracting features from each image using feature detectors such as SIFT, SURF, or ORB.

2. **Clustering:** Grouping all the features from all the images into clusters. Each cluster represents a "visual word". This is often done using the K-means clustering algorithm.

3. **Histogram Construction:** For each image, constructing a histogram that counts the number of features in each cluster.

4. **Classification:** Using the histogram as a feature vector for further tasks such as image classification.

```
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
_, labels, vocab = cv2.kmeans(all_descriptors, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

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
```

This is a basic implementation and the actual process can be more intricate. For instance, OpenCV provides the `BOWKMeansTrainer` and `BOWImgDescriptorExtractor` classes, which offer a more structured way to build and use the BoW model. Additionally, for real-world tasks, you might want to incorporate other steps like normalization of histograms, using the TF-IDF (term frequency–inverse document frequency) weighting scheme, or employing SVM for classification.



Refs: [1](https://nicolovaligi.com/articles/bag-of-words-loop-closure-visual-slam/), [2](https://www.youtube.com/watch?v=a4cFONdc6nc)
