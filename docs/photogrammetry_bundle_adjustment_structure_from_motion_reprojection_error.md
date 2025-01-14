photogrammetry is the broad field encompassing techniques for extracting measurements from photographs. Structure from Motion is a specific approach within photogrammetry that uses image sequences to build 3D models. Bundle Adjustment is a mathematical technique used within both photogrammetry and SfM to refine these models and improve their accuracy.


## Photogrammetry

   - **Definition**: Photogrammetry is a technique used to measure and interpret features from photographs. It's broadly used in various fields such as surveying, mapping, engineering, and archaeology.
   - **Application**: It involves taking measurements from photographs to create maps, 3D models, or drawings of real-world objects or scenes.
   - **Process**: This process can use either single or multiple photographs and relies on the principles of perspective and projective geometry.
   - **Tools & Techniques**: It might involve various software and hardware tools, like drones for aerial photography, and specialized software for image processing and 3D reconstruction.

## Structure from Motion (SfM)

   - **Definition**: SfM is a photogrammetric method used to create 3D structures from 2D image sequences. It's a subset of computer vision and photogrammetry.
   - **Application**: Primarily used to reconstruct 3D models from photographs, particularly in applications where traditional surveying methods are impractical.
   - **Process**: The process involves identifying and matching features across a series of images, estimating camera positions and angles, and then reconstructing a 3D structure based on these estimations.
   - **Relation to Photogrammetry**: SfM is more automated and software-driven compared to traditional photogrammetry and is considered a modern advancement in the field.

## Bundle Adjustment
   - **Definition**: Bundle Adjustment is a mathematical optimization technique used in computer vision and photogrammetry.
   - **Application**: It's used to refine a visual reconstruction to produce jointly optimal 3D structure and viewing parameter (camera position, orientation) estimates.
   - **Process**: It adjusts the 3D coordinates of the point cloud and camera parameters by minimizing the re-projection error, which is the difference between the observed image points and projected points from the 3D model.
   - **Relation to SfM and Photogrammetry**: Bundle Adjustment is often the final step in the SfM process to refine the model. It's a critical part of both SfM and photogrammetry when accuracy is essential.


## Noah Snavely reprojection error

Ref [1](https://www.eecs.umich.edu/courses/eecs442-ahowens/fa21/slides/lec22-sfm.pdf), [2](https://ceres-solver.googlesource.com/ceres-solver/+/1.12.0/examples/snavely_reprojection_error.h)
,[3](http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment), [4](https://homes.cs.washington.edu/~sagarwal/bal.pdf)
## CMVS-PMVS
Refs: [1](https://github.com/pmoulon/CMVS-PMVS)


## openMVG
Refs: [1](https://opensourcephotogrammetry.blogspot.com/)




##  Pipeline for SFM

**Feature Detection and Matching:**
First, features are detected in each image. Common algorithms include SIFT, SURF, or ORB. 
These features are then matched between images to find correspondences. OpenCV's cv::BFMatcher or cv::FlannBasedMatcher can be used for this.

**Pose Estimation:**
With matched features, you estimate the relative pose between two images. cv::recoverPose is often used here. It computes the rotation and translation between two views given corresponding points and the essential matrix. The essential matrix can be calculated using `cv::findEssentialMat`.

```cpp
cv::Mat E = cv::findEssentialMat(points1, points2, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
cv::recoverPose(E, points1, points2, R, t, focal, pp, mask);
```


**Incremental or Global Structure from Motion:**
- Incremental SfM: Start with an initial pair of images, recover pose, and then add images one by one, adjusting the structure and camera poses with each addition. Here, after using cv::recoverPose, you would typically use a bundle adjustment step to refine the structure and camera poses.
- Global SfM: All images are considered at once. 












