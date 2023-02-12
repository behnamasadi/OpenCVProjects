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


## Epipolar rectification.
Refs [1](https://en.wikipedia.org/wiki/Image_rectification), [2](https://www.cs.cmu.edu/~16385/s17/Slides/13.1_Stereo_Rectification.pdf)
