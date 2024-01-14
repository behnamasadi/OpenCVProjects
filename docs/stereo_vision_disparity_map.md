# Stereo Calibration

The following API computes the 

Poses of an object relative to the first camera: 


<img src="https://latex.codecogs.com/svg.latex?R_1%2CT_1%20%3D%20R_w%5Ec%2C%20T_w%5Ec" alt="https://latex.codecogs.com/svg.latex?R_1,T_1 = R_w^c, T_w^c" />


Poses of an object relative to the second camera:

<img src="https://latex.codecogs.com/svg.latex?R_2%2CT_2%20%3D%20R2_w%5Ec%2C%20T2_w%5Ec" alt="https://latex.codecogs.com/svg.latex?R_2,T_2 = R2_w^c, T2_w^c" />

<br/>
<br/>

It computes <img src="https://latex.codecogs.com/svg.latex?R%2CT" alt="https://latex.codecogs.com/svg.latex?R,T" /> such that:


<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20R_2%3DR%20R_1%20%5C%5C%20T_2%3DR%20T_1%20&plus;%20T." alt="https://latex.codecogs.com/svg.latex?\\
R_2=R R_1
\\
T_2=R T_1 + T." />



<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7D%20X_2%20%5C%5C%20Y_2%20%5C%5C%20Z_2%20%5C%5C%201%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20R%20%26%20T%20%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%20X_1%20%5C%5C%20Y_1%20%5C%5C%20Z_1%20%5C%5C%201%20%5Cend%7Bbmatrix%7D." alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix} X_2 \\ Y_2 \\ Z_2 \\ 1 \end{bmatrix} = \begin{bmatrix} R & T \\ 0 & 1 \end{bmatrix} \begin{bmatrix} X_1 \\ Y_1 \\ Z_1 \\ 1 \end{bmatrix}." />



<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20E%3D%5Cbegin%7Bbmatrix%7D%200%20%26%20-T_2%20%26T_1%20%5C%5C%20T_2%20%26%200%20%26%20-T_0%5C%5C%20-T_1%20%26%20-T_0%20%26%200%20%5Cend%7Bbmatrix%7D%20%5C%5C%20%5C%5C%20%5C%5C%20T%3D%20%5Cbegin%7Bbmatrix%7D%20T_0%5C%5C%20T_1%5C%5C%20T_2%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\\
E=\begin{bmatrix}
0 & -T_2 &T_1 \\ 
T_2 & 0 & -T_0\\ 
-T_1 & -T_0 & 0
\end{bmatrix}
\\
\\
\\
T=
\begin{bmatrix}
T_0\\ 
T_1\\ 
T_2
\end{bmatrix}
" />


<img src="https://latex.codecogs.com/svg.latex?F%20%3D%20cameraMatrix2%5E%7B-T%7D%5Ccdot%20E%20%5Ccdot%20cameraMatrix1%5E%7B-1%7D" alt="https://latex.codecogs.com/svg.latex?F = cameraMatrix2^{-T}\cdot E \cdot cameraMatrix1^{-1}" />

<br/>
<br/>

```cpp
cv::stereoCalibrate	(	InputArrayOfArrays 	objectPoints,
InputArrayOfArrays 	imagePoints1,
InputArrayOfArrays 	imagePoints2,
InputOutputArray 	cameraMatrix1,
InputOutputArray 	distCoeffs1,
InputOutputArray 	cameraMatrix2,
InputOutputArray 	distCoeffs2,
Size 	imageSize,
InputOutputArray 	R,
InputOutputArray 	T,
OutputArray 	E,
OutputArray 	F,
OutputArray 	perViewErrors,
int 	flags = CALIB_FIX_INTRINSIC,
TermCriteria 	criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6) 
)		
```
Refs: [1](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d)

Refs: [1](https://www.cs.cmu.edu/~16385/s17/Slides/13.1_Stereo_Rectification.pdf), [3](https://www.allaboutvision.com/eye-care/measure-pupillary-distance/), [4](https://www.mathworks.com/matlabcentral/answers/1451509-what-should-be-the-distance-between-the-two-cameras)

# Stereo Rectification
If cameras are calibrated:
```cpp
cv::stereoRectify	(	InputArray 	cameraMatrix1,
InputArray 	distCoeffs1,
InputArray 	cameraMatrix2,
InputArray 	distCoeffs2,
Size 	imageSize,
InputArray 	R,
InputArray 	T,
OutputArray 	R1,
OutputArray 	R2,
OutputArray 	P1,
OutputArray 	P2,
OutputArray 	Q,
int 	flags = CALIB_ZERO_DISPARITY,
double 	alpha = -1,
Size 	newImageSize = Size(),
Rect * 	validPixROI1 = 0,
Rect * 	validPixROI2 = 0 
)		
```

Refs: [1](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6)

If cameras are not calibrated:
```cpp
cv::stereoRectifyUncalibrated	(	InputArray 	points1,
InputArray 	points2,
InputArray 	F,
Size 	imgSize,
OutputArray 	H1,
OutputArray 	H2,
double 	threshold = 5 
)		
```

Refs: [1](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gaadc5b14471ddc004939471339294f052)




# Distance Between the two cameras and stereo angle
Refs: [1](https://ch.mathworks.com/matlabcentral/answers/1451509-what-should-be-the-distance-between-the-two-cameras#answer_786164), [2](https://correlated.kayako.com/article/78-lens-selection-and-stereo-angle#:~:text=For%20shorter%20focal%20length%20lenses,a%2025%20degree%20stereo%20angle)



# Stereo Camera Simulation:

### 1. `cv2.stereoCalibrate`

This function performs stereo calibration. It estimates the parameters of two cameras and how they relate to each other. Here's what each parameter and return value represents:

- **Input Parameters**:
  - `objectPoints`: 3D points in the real world. These are typically corners of a chessboard or similar calibration pattern, observed in multiple images.
  - `all_imagePoints_cam0`: 2D points in the first camera's image plane corresponding to `objectPoints`.
  - `all_imagePoints_cam1`: 2D points in the second camera's image plane corresponding to `objectPoints`.
  - `cameraMatrix`: The intrinsic camera matrix for the initial guess of both cameras (assumed to be the same here).
  - `distCoeffs`: The distortion coefficients for the initial guess of both cameras (assumed to be the same here).
  - `imageSize`: Size of the image used for calibration.

- **Return Values**:
  - `retval`: A boolean flag indicating if the calibration was successful.
  - `cameraMatrix1`, `cameraMatrix2`: Optimized intrinsic camera matrices for each camera.
  - `distCoeffs1`, `distCoeffs2`: Optimized distortion coefficients for each camera.
  - `R`: Rotation matrix describing the rotation from the first to the second camera.
  - `T`: Translation vector describing the translation from the first to the second camera.
  - `E`: Essential matrix.
  - `F`: Fundamental matrix.

The essential matrix (`E`) relates corresponding points in stereo images considering the internal parameters of the cameras, while the fundamental matrix (`F`) relates these points without considering internal parameters.

### 2. `cv2.stereoRectify`

This function is used to compute the rotation matrices for each camera that bring the corresponding points into the same line, facilitating easier computation of disparity:

- **Input Parameters**:
  - `cameraMatrix1`, `cameraMatrix2`: Intrinsic parameters of each camera obtained from stereo calibration.
  - `distCoeffs1`, `distCoeffs2`: Distortion coefficients of each camera.
  - `imageSize`: Size of the image.
  - `R`, `T`: Rotation matrix and translation vector obtained from stereo calibration.
  - `flags`, `alpha`: Additional parameters to control the rectification process.

- **Return Values**:
  - `R1`, `R2`: Rectification transforms (rotation matrices) for each camera.
  - `P1`, `P2`: Projection matrices in the new (rectified) coordinate systems for each camera.
  - `Q`: Disparity-to-depth mapping matrix.
  - `validPixROI1`, `validPixROI2`: Valid pixel ROI (Region Of Interest) within the rectified images for each camera.

### 3. `cv2.initUndistortRectifyMap`

This function computes the undistortion and rectification transformation map for each camera:

- **Input Parameters for Each Camera**:
  - `cameraMatrix`, `distCoeffs`: Intrinsic parameters and distortion coefficients of the camera.
  - `R

`: The rectification transform (rotation matrix) obtained from `cv2.stereoRectify`.
  - `P`: The projection matrix in the new (rectified) coordinate system for the camera.
  - `imageSize`: Size of the image.
  - `cv2.CV_16SC2`: Specifies the type of the map to be returned, which is a 16-bit signed two-channel image.

- **Return Values for Each Camera**:
  - `map1x`, `map1y` (for the first camera): The x and y pixel coordinate remapping arrays. These are used to perform the undistortion and rectification transformation.
  - `map2x`, `map2y` (for the second camera): Similarly, the x and y pixel coordinate remapping arrays for the second camera.

In essence, the process works as follows:

1. **Stereo Calibration (`cv2.stereoCalibrate`)**: Determine the relationship between the two cameras (their relative position and orientation) and refine the individual intrinsic parameters of each camera.

2. **Stereo Rectification (`cv2.stereoRectify`)**: Compute the rectification transformations. After rectification, the epipolar lines in stereo images are aligned horizontally, which is essential for stereo correspondence and depth estimation algorithms.

3. **Undistort and Rectify Map Creation (`cv2.initUndistortRectifyMap`)**: Generate maps to efficiently rectify and undistort images captured from each camera. These maps are used to transform the captured images into a common plane, aligning them so that corresponding points in the stereo images are on the same row.




To obtain the 3D position of points from a stereo camera setup, you typically follow these steps after setting up and calibrating your stereo cameras:

1. **Capture Stereo Images**: Capture images from both the left and right cameras simultaneously.

2. **Rectify the Images**: Use the rectification maps (`map1x`, `map1y`, `map2x`, `map2y`) obtained from `cv2.initUndistortRectifyMap` to rectify the images from both cameras. This aligns the images such that their corresponding epipolar lines become co-planar and horizontal.

3. **Compute Disparity Map**: Use a stereo matching algorithm like Block Matching to compute the disparity map. The disparity map indicates the pixel distance between corresponding points in the left and right rectified images.

4. **Calculate 3D Coordinates**: Once you have the disparity map, you can calculate the 3D coordinates of each point using the disparity-to-depth mapping matrix (`Q`) obtained from `cv2.stereoRectify`.

Here's a simplified outline of how this can be done in code, using OpenCV functions:

```python
import cv2
import numpy as np

# Assuming you have already captured or loaded rectified images
left_image_rectified = cv2.remap(left_image, map1x, map1y, cv2.INTER_LINEAR)
right_image_rectified = cv2.remap(right_image, map2x, map2y, cv2.INTER_LINEAR)

# Create a stereo matcher object
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Compute the disparity map
disparity = stereo.compute(left_image_rectified, right_image_rectified)

# Normalize the disparity map (for visualization)
disparity_visual = cv2.normalize(disparity, None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
disparity_visual = np.uint8(disparity_visual)

# Compute the 3D points
points_3D = cv2.reprojectImageTo3D(disparity, Q)

# Filter points based on disparity map
mask = disparity > disparity.min()
points_3D = points_3D[mask]

# Now, points_3D contains the 3D coordinates of the points
```

### Important Considerations:
- **Disparity Map**: The disparity map obtained from block matching represents the differences in horizontal coordinates of corresponding points in the left and right rectified images. It's crucial for calculating depth.
  
- **Depth Calculation**: The `cv2.reprojectImageTo3D` function transforms the disparity map into a 3D representation. The resulting array (`points_3

D`) contains the 3D coordinates (X, Y, Z) of each pixel in the stereo image pair. The Z value represents the depth information.

- **Quality of Disparity Map**: The accuracy of the 3D reconstruction heavily depends on the quality of the disparity map. Factors like the number of disparities, block size, and the uniqueness and texture of the scene can affect this.

- **Filtering and Masking**: It's common to apply a mask to the disparity map to filter out unreliable values. For example, disparity values that are too small might correspond to infinite distances and are typically not reliable.

- **Normalization for Visualization**: The disparity map is often normalized for visualization purposes. This doesn't affect the 3D reconstruction but makes it easier to analyze the disparity visually.

- **Camera Calibration and Rectification**: This process assumes that the stereo camera system is well-calibrated and the images are correctly rectified. Errors in calibration or rectification can lead to inaccuracies in the 3D reconstruction.

### Final Note:
The 3D positions obtained this way are in the coordinate system of the cameras. To interpret these positions in a real-world context, you may need to consider additional transformations, especially if you're integrating this data with other spatial data or sensors.

Refs: [1](https://towardsdatascience.com/a-comprehensive-tutorial-on-stereo-geometry-and-stereo-rectification-with-python-7f368b09924a), [2](https://people.scs.carleton.ca/~c_shu/Courses/comp4900d/notes/rectification.pdf), [3](https://www.cs.cmu.edu/~16385/s17/Slides/13.1_Stereo_Rectification.pdf), [4](https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/)
