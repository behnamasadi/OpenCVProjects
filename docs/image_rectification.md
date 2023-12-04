# Image Rectification
Image rectification is transforming an image of a scene into a view that is aligned with a desired coordinate system. The goal of rectification is to remove the effects of camera perspective, rotation, and lens distortion, so that the resulting image has a uniform scale and appears to be captured from a front-facing perspective. 


In the following:
 
- The camera rotating around the `z` axis.
- The virtual image plane at `5°` degree is red and at `90°` is green. 
- The rectified images are in the blue virtual image plane. 
- The virtual plane must be parallel to the stereo baseline (orange). 


|   |   |
|---|---|
|<img src="images/image_rectification_1.png" alt="" />    |<img src="images/image_rectification_8.png" alt="" />  |
|<img src="images/image_rectification_20.png" alt="" />   | <img src="images/image_rectification_30.png" alt="" />  |


# Image Rectification Algorithms
All rectified images satisfy the following two properties:
- All epipolar lines are parallel to the horizontal axis.
- Corresponding points have identical vertical coordinates.

## Projective rectification
Projective rectification is the process of transforming an image so that all parallel lines in the image are transformed to be parallel in the new image. The goal of projective rectification is to obtain a view of the scene that is orthographic or fronto-parallel. Projective rectification can be performed using a homography matrix, which maps points from one image to the other.

```
import cv2
import numpy as np

# Load the two images
img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')

# Find the homography matrix using the findHomography function
homography, _ = cv2.findHomography(src_points, dst_points)

# Use the perspectiveTransform function to project the second image onto the first image
img2_rectified = cv2.warpPerspective(img2, homography, (img1.shape[1], img1.shape[0]))

# Show the rectified images
cv2.imshow("Rectified Image 1", img1)
cv2.imshow("Rectified Image 2", img2_rectified)
cv2.waitKey(0)
cv2.destroyAllWindows()
```


## Epipolar Rectification
Epipolar rectification, on the other hand, is the process of rectifying two images such that the epipolar lines in the two images are aligned. The epipolar lines are the lines that intersect the two images and correspond to a single 3D point in the scene. The goal of epipolar rectification is to simplify the problem of finding corresponding points in two images, as the epipolar lines provide a unique constraint on the corresponding points.


##  Computing The Rectification Matrices






A simple way to rectify the two images is to first rotate both cameras so that they are
looking perpendicular to the line joining the camera centers c 0 and c 1 . Since there is a de-
gree of freedom in the tilt, the smallest rotations that achieve this should be used. Next, to
determine the desired twist around the optical axes, make the up vector (the camera y axis)


[code](../scripts/image_rectification.py)

Refs [1](https://en.wikipedia.org/wiki/Image_rectification), [2](https://www.cs.cmu.edu/~16385/s17/Slides/13.1_Stereo_Rectification.pdf)
