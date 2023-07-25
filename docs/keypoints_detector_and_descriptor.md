# 1. Image Feature Detectors and Descriptor 

## 1.1. Gradient-based Descriptors:

These descriptors use image gradients or patterns of intensity change in images. They are typically robust to lighting changes and slight transformations.
Examples:
- SIFT (Scale-Invariant Feature Transform): Computes a descriptor based on the local gradient histograms.
- SURF (Speeded-Up Robust Features): An optimized and faster version of SIFT.

## 1.2. Binary Descriptors:

These descriptors produce binary strings. They are computationally efficient and require less memory.
Examples:
- ORB (Oriented FAST and Rotated BRIEF): Combines the FAST keypoint detector and the BRIEF descriptor.
- BRIEF (Binary Robust Independent Elementary Features): Produces a binary string descriptor by efficiently comparing pixel intensities.
- BRISK (Binary Robust Invariant Scalable Keypoints): A fast keypoint detector with a binary descriptor.
- FREAK (Fast Retina Keypoint): A binary descriptor inspired by the human visual system.

## 1.3. Intensity-based Descriptors

These are simpler and based directly on pixel intensities in the image region.
Example:
- DAISY: It's a dense descriptor often used in stereo matching.

## 1.4. Compact Descriptors

These are designed to provide a more compact representation of the image or region. They are typically used in large-scale image retrieval scenarios where memory is a concern.
Example:
-  VGG (VGG-like Descriptor in OpenCV): A compact descriptor derived from the CNN's (Convolutional Neural Networks) architecture.

## 1.5. 3D Descriptors (For 3D Data)

OpenCV also provides tools for working with 3D data, such as point clouds. Some descriptors are designed to work with this kind of data.
Example:
-  SHOT (Signature of Histograms of OrienTations): Describes the local appearance of 3D point cloud data around a point of interest.

## 1.6 Color Descriptors

While many of the popular descriptors focus on shape or structure, some can incorporate or be extended to use color information.
Example:
-  Color-invariant versions of SIFT and SURF.

## 1.7. Region-based Descriptors

These descriptors work on a segmented region or shape in the image, rather than on keypoints.
Example:
- Hu Moments: Descriptive statistics that can describe the shape of a binary image or region.


[List of all available 2D image feature detectors and descriptor](https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html#a40182e88bf6aa2c74311c9927ba056dc)


## 1.8. Getting Name of The Detector and Type of Descriptor

```
detector = cv2.SIFT_create()
print(detector.getDefaultName()) 
print(detector.descriptorType())

detector = cv2.ORB_create()
print(detector.getDefaultName()) 
print(detector.descriptorType())
```
## 1.9. Example of Feature Detectors 
## 1.9.1 ORB

```python
detector = cv.ORB_create()
img_pts = detector.detect(img, None)
img_pts, img_descriptor = detector.compute(img, img_pts)
```

properties of ORB keypoints

```python
print(type(img_pts))
for i in img_pts:
    print("point: ", i.pt)
    print("angle: ", i.angle)
    print("class_id: ", i.class_id)
    print("octave: ", i.octave)
    print("response: ", i.response)
    print("size: ", i.size)

```

## 1.9.2. goodFeaturesToTrack

The goodFeaturesToTrack function in OpenCV is a method to detect the most prominent corners or features in an image. This function is based on the Shi-Tomasi corner detection method, which is a modification of the Harris corner detection. While Harris scores corners based on a combination of the eigenvalues of the corner's covariance matrix, the Shi-Tomasi method simply considers the minimum of these eigenvalues.


params for corner detection: 
If `useHarrisDetector` set to True, Harris corner detection is used instead of `Shi-Tomasi`.
```python
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7,useHarrisDetector=False)

keypoints = cv.goodFeaturesToTrack(img_gray, mask=None,
                                   **feature_params)

print(type(keypoints))
for i in keypoints:
    print("point: ", i)
```




## 1.9.3. FastFeatureDetector 

The FAST in FastFeatureDetector stands for "Features from Accelerated Segment Test." FAST is primarily used to detect corners in an image. Corners are typically regions in an image with high variation in pixel values in multiple directions.



### How It Works

A circle of sixteen pixels is used to inspect around a candidate pixel. The intensity of the center pixel is noted.
If there's a set of n contiguous pixels in the circle (where typically n can be 12, 11, 10, or 9) which are all brighter than the center pixel intensity plus a threshold, or all darker than the center pixel intensity minus a threshold, then that center pixel is considered as a corner.

### Non-maximal Suppression

Once potential keypoints are found, FAST uses non-maximal suppression to pick the most prominent ones. This means if there are multiple keypoints detected close to each other, only the strongest one is retained.

### Limitations

While FAST is very efficient in detecting corners, it does not compute any descriptor for the keypoints. Thus, to describe the keypoints, you often need to use another descriptor method, such as BRIEF, ORB, etc.
FAST does not provide any information about the orientation of the keypoint.
- Parameters:
  - threshold: The intensity difference threshold to decide if a pixel is brighter or darker than the center pixel.
  - nonmaxSuppression: Boolean value indicating whether to use non-maximal suppression to filter the keypoints.
  - type: FAST detection type, like TYPE_9_16, TYPE_7_12, or TYPE_5_8.


```python
img = cv.imread(img_path)
detector = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_pts = detector.detect(img, None)

img_marked = cv.drawKeypoints(
    img, img_pts, None, color=(0, 0, 255), flags=0)

plt.imshow(img_marked), plt.show()
```

# 2. Drawing Keypoints

In the drawKeypoints function, using the flag DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS will ensure that both the location and the orientation of the keypoints are drawn.

```python
prevImg_marked = cv.drawKeypoints(
    img, img_pts, None, color=(0, 255, 0), flags=0)

plt.imshow(prevImg_marked)
plt.show()
```


# 3. Convert vector of keypoints to vector of points

This method converts **vector of keypoints** to **vector of points** meaning Array of (x,y) coordinates of each keypoint

```python
img_pts = detector.detect(img, None)

point = img_pts[0]

x = point.pt[0]
y = point.pt[1]
(x, y) = point.pt
```
This method converts vector of keypoints to vector of points -> Array of (x,y) coordinates of each keypoint


```python
pts = cv2.KeyPoint_convert(img_pts)
print(pts)
pts = np.float64([key_point.pt for key_point in img_pts]).reshape(-1, 1, 2)
```

# 5. Matching Keypoints

once keypoints are detected and described, we must match these keypoints between images. Here are some of the most widely-used keypoint feature matchers:

## 5.1 BFMatcher (Brute-Force Matcher):

The brute-force matcher compares each descriptor in one image with every descriptor in the other image, computing a distance measure to determine the closest match.

## 5.1.1 Distance Types
- `L2 Norm` : `cv.NORM_L2` Euclidean distance, used mainly for floating-point descriptors like SIFT and SURF.
By default, it is `cv.NORM_L2`. It is good for SIFT, SURF etc (`cv.NORM_L1 is` also there). 

```
bf = cv2.BFMatcher(cv2.cv.NORM_L2, crossCheck=True)
```


- `Hamming Distance`: `cv.NORM_HAMMING` Used for binary string-based descriptors like `ORB, BRIEF, and BRISK`. It counts the number of differing bits between two binary strings. If ORB is using `WTA_K == 3` or `4`, which takes 3 or 4 points to produce BRIEF descriptor, `cv.NORM_HAMMING2` should be used.


```
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
```

- `Cross Check`: An option in the BFMatcher that ensures mutual matching. For two keypoints to be considered a match, the keypoint in the first image must match the keypoint in the second image, and vice-versa.

## 5.2. FlannBasedMatcher (Fast Library for Approximate Nearest Neighbors)

- How it Works: Instead of brute force searching, FLANN uses hierarchical algorithms to speed up the matching process, making it more suitable for large datasets.

- Parameters: FLANN requires configuration parameters, which can be determined automatically using an algorithm provided by FLANN. The parameters mainly dictate the type of trees and branching factors.

Applicability: It's used mainly for floating-point descriptors like **SIFT and SURF**. However, it can be used with binary descriptors after converting them into floating-point.

## 5.3. KNN (K-Nearest Neighbors) Matching:

While not a distinct matcher on its own, both BFMatcher and FlannBasedMatcher can perform KNN matching. Instead of finding the single best match for each descriptor, they find the k best matches.

- Ratio Test: A common technique used in conjunction with KNN matching. For each keypoint, if the distance ratio between the best and the second-best match is below a threshold (usually around 0.7 to 0.8), the match is retained. This helps in removing ambiguous matches and is especially useful in applications like object recognition.

## 5.4. GMS Matcher (Grid-based Motion Statistics):

- How it Works: It uses a grid-based motion statistics approach to improve feature matching results. After initial matching using other matchers, GMS removes incorrect matches based on consistent motion statistics.
- Use Case: Particularly useful in video-related applications where object motion is involved.
When choosing a matcher, consider the nature of your application, the type of descriptors you're using, and the size of your dataset. For example, while brute-force matching might be suitable for small datasets or real-time applications, FLANN might be a better choice for large datasets where computational efficiency is a concern.

# 6. Drawing  Matches



