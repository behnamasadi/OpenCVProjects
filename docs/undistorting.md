- [Undistorting Points](#undistorting-points)
  * [initUndistortRectifyMap](#initundistortrectifymap)
  * [undistort](#undistort)


## Undistorting Points

### initUndistortRectifyMap
The following function computes the `undistortion` and `rectification transformation`. The undistorted is image that has been captured with a camera using the `camera matrix =newCameraMatrix` and zero distortion.

1. In case of a monocular camera, `newCameraMatrix` is usually equal to `cameraMatrix` or it can be computed by getOptimalNewCameraMatrix for a better control over scaling.

2. In case of a stereo camera, `newCameraMatrix` is normally set to `P1` or `P2` computed by `stereoRectify` .



```cpp
void cv::initUndistortRectifyMap	(	InputArray 	cameraMatrix,
InputArray 	distCoeffs,
InputArray 	R,
InputArray 	newCameraMatrix,
Size 	size,
int 	m1type,
OutputArray 	map1,
OutputArray 	map2 
)	
```



**Parameters:**

1. `cameraMatrix`:
<br/>
 <img src="https://latex.codecogs.com/svg.latex?A%3D%5Cbegin%7Bbmatrix%7D%20f_x%20%26%200%20%26%20c_x%5C%5C%200%20%26%20f_y%20%26%20c_y%5C%5C%200%20%26%200%20%26%201%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?A=\begin{bmatrix}
f_x & 0 & c_x\\ 
0 & f_y & c_y\\ 
0 & 0 & 1
\end{bmatrix}" />

2. `distCoeffs`: input vector of distortion coefficients 4, 5, 8, 12 or 14 elements:

 <img src="https://latex.codecogs.com/svg.latex?%28k_1%2C%20k_2%2C%20p_1%2C%20p_2%5B%2C%20k_3%5B%2C%20k_4%2C%20k_5%2C%20k_6%5B%2C%20s_1%2C%20s_2%2C%20s_3%2C%20s_4%5B%2C%20%5Ctau_x%2C%20%5Ctau_y%5D%5D%5D%5D%29" alt="https://latex.codecogs.com/svg.latex?" />

3. `R`: Optional rectification transformation in the object space (3x3 matrix). `R1 `or `R2` , computed by `stereoRectify` can be passed here. If the matrix is empty, the identity transformation is assumed. In `initUndistortRectifyMap` R is assumed to be an identity matrix.


4. `newCameraMatrix`: New camera matrix


<img src="https://latex.codecogs.com/svg.latex?A%3D%5Cbegin%7Bbmatrix%7D%20f%27_x%20%26%200%20%26%20c%27_x%5C%5C%200%20%26%20f%27_y%20%26%20c%27_y%5C%5C%200%20%26%200%20%26%201%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?A=\begin{bmatrix}
f'_x & 0 & c'_x\\ 
0 & f'_y & c'_y\\ 
0 & 0 & 1
\end{bmatrix}" />


For each observed point coordinate (u,v) the function computes:

<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Bl%7D%20x%20%5Cleftarrow%20%28u%20-%20%7Bc%27%7D_x%29/%7Bf%27%7D_x%20%5C%5C%20y%20%5Cleftarrow%20%28v%20-%20%7Bc%27%7D_y%29/%7Bf%27%7D_y%20%5C%5C%20%7B%5BX%5C%2CY%5C%2CW%5D%7D%20%5ET%20%5Cleftarrow%20R%5E%7B-1%7D*%5Bx%20%5C%2C%20y%20%5C%2C%201%5D%5ET%20%5C%5C%20x%27%20%5Cleftarrow%20X/W%20%5C%5C%20y%27%20%5Cleftarrow%20Y/W%20%5C%5C%20r%5E2%20%5Cleftarrow%20x%27%5E2%20&plus;%20y%27%5E2%20%5C%5C%20x%27%27%20%5Cleftarrow%20x%27%20%5Cfrac%7B1%20&plus;%20k_1%20r%5E2%20&plus;%20k_2%20r%5E4%20&plus;%20k_3%20r%5E6%7D%7B1%20&plus;%20k_4%20r%5E2%20&plus;%20k_5%20r%5E4%20&plus;%20k_6%20r%5E6%7D%20&plus;%202p_1%20x%27%20y%27%20&plus;%20p_2%28r%5E2%20&plus;%202%20x%27%5E2%29%20&plus;%20s_1%20r%5E2%20&plus;%20s_2%20r%5E4%5C%5C%20y%27%27%20%5Cleftarrow%20y%27%20%5Cfrac%7B1%20&plus;%20k_1%20r%5E2%20&plus;%20k_2%20r%5E4%20&plus;%20k_3%20r%5E6%7D%7B1%20&plus;%20k_4%20r%5E2%20&plus;%20k_5%20r%5E4%20&plus;%20k_6%20r%5E6%7D%20&plus;%20p_1%20%28r%5E2%20&plus;%202%20y%27%5E2%29%20&plus;%202%20p_2%20x%27%20y%27%20&plus;%20s_3%20r%5E2%20&plus;%20s_4%20r%5E4%20%5Cend%7Barray%7D" alt="https://latex.codecogs.com/svg.latex?\begin{array}{l} x \leftarrow (u - {c'}_x)/{f'}_x \\ y \leftarrow (v - {c'}_y)/{f'}_y \\ {[X\,Y\,W]} ^T \leftarrow R^{-1}*[x \, y \, 1]^T \\ x' \leftarrow X/W \\ y' \leftarrow Y/W \\ r^2 \leftarrow x'^2 + y'^2 \\ x'' \leftarrow x' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + 2p_1 x' y' + p_2(r^2 + 2 x'^2) + s_1 r^2 + s_2 r^4\\ y'' \leftarrow y' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + p_1 (r^2 + 2 y'^2) + 2 p_2 x' y' + s_3 r^2 + s_4 r^4  \end{array}" />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?s%5Cbegin%7Bbmatrix%7D%20%7Bx%7D%27%27%27%5C%5C%20%7B%7Dy%27%27%27%5C%5C%201%20%5Cend%7Bbmatrix%7D%3D%20%5Cbegin%7Bbmatrix%7D%20R_%7B33%7D%28%5Ctau_x%2C%20%5Ctau_y%29%20%26%200%20%26%7B-R_%7B13%7D%28%28%5Ctau_x%2C%20%5Ctau_y%29%7D%20%5C%5C%200%20%26%20R_%7B33%7D%28%5Ctau_x%2C%20%5Ctau_y%29%20%26%20-R_%7B23%7D%28%5Ctau_x%2C%20%5Ctau_y%29%5C%5C%200%20%26%200%20%26%201%20%5Cend%7Bbmatrix%7DR%28%5Ctau_x%2C%20%5Ctau_y%29%5Cbegin%7Bbmatrix%7D%20%7Bx%7D%27%27%5C%5C%20%7By%7D%27%27%5C%5C%201%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?s\begin{bmatrix}
{x}'''\\ 
{}y'''\\ 
1
\end{bmatrix}=
\begin{bmatrix}
R_{33}(\tau_x, \tau_y) & 0 &{-R_{13}((\tau_x, \tau_y)}  \\ 
0 & R_{33}(\tau_x, \tau_y)  & -R_{23}(\tau_x, \tau_y)\\ 
0 & 0 & 1 
\end{bmatrix}R(\tau_x, \tau_y)\begin{bmatrix}
{x}''\\ 
{y}''\\ 
1
\end{bmatrix}" />
<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?%5C%5Cmap_x%28u%2Cv%29%20%5Cleftarrow%20x%27%27%27%20f_x%20&plus;%20c_x%20%5C%5C%20map_y%28u%2Cv%29%20%5Cleftarrow%20y%27%27%27%20f_y%20&plus;%20c_y" alt="https://latex.codecogs.com/svg.latex?\\map_x(u,v) \leftarrow x''' f_x + c_x \\ map_y(u,v) \leftarrow y''' f_y + c_y" />


where <img src="https://latex.codecogs.com/svg.latex?%28k_1%2C%20k_2%2C%20p_1%2C%20p_2%5B%2C%20k_3%5B%2C%20k_4%2C%20k_5%2C%20k_6%5B%2C%20s_1%2C%20s_2%2C%20s_3%2C%20s_4%5B%2C%20%5Ctau_x%2C%20%5Ctau_y%5D%5D%5D%5D%29" alt="https://latex.codecogs.com/svg.latex?" /> are the distortion coefficients.



In case of a stereo camera, this function is called twice: once for each camera head, after `stereoRectify`, which in its turn is called after `stereoCalibrate`. But if the stereo camera was not calibrated, it is still possible to compute the rectification transformations directly from the fundamental matrix using `stereoRectifyUncalibrated`. For each camera, the function computes `homography H` as the rectification transformation in a pixel domain, not a rotation matrix` R` in 3D space. `R` can be computed from H as


<img src="https://latex.codecogs.com/svg.latex?%5Ctext%7BR%7D%20%3D%20%5Ctext%7BcameraMatrix%7D%20%5E%7B-1%7D%20%5Ccdot%20%5Ctext%7BH%7D%20%5Ccdot%20%5Ctext%7BcameraMatrix%7D" alt="https://latex.codecogs.com/svg.latex?\text{R} = \text{cameraMatrix} ^{-1} \cdot \text{H} \cdot \text{cameraMatrix}" />



Refs: [1](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a), [2](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#initundistortrectifymap)




### cv::undistortPoints
This function is similar to `initUndistortRectifyMap` but it operates on a sparse set of points
```cpp
void cv::undistortPoints	(	InputArray 	src,
OutputArray 	dst,
InputArray 	cameraMatrix,
InputArray 	distCoeffs,
InputArray 	R = noArray(),
InputArray 	P = noArray() 
)
```






In OpenCV `cv::undistort`does the followings:
For each pixel of the destination lens-corrected image do:

- Convert the pixel coordinates `(u_dst, v_dst)` to normalized coordinates `(x_d, y_d`) using the inverse of the calibration matrix `K`,
- Apply the lens-distortion model, as displayed above, to obtain the distorted normalized coordinates `(x_u, y_u)`,
- Convert `(x_u, y_u)` to distorted pixel coordinates using the calibration matrix `K`,
- Use the interpolation method of your choice to find the intensity/depth associated with the pixel coordinates `(u_src, v_src)` in the source image, and assign this intensity/depth to the current destination pixel.


Refs: [1](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#camera-calibration-and-3d-reconstruction), [2](https://stackoverflow.com/questions/21958521/understanding-of-opencv-undistortion), [3](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga69f2545a8b62a6b0fc2ee060dc30559d)






##  cv::UndistortTypes in OpenCV:

**`cv::UndistortTypes`** is an enumeration (enum) in OpenCV's C++ API that specifies different distortion correction models used in the `undistort` function. This function is essential for rectifying images captured with lenses that introduce geometric distortions, such as barrel or pincushion distortion.

**Available options in `cv::UndistortTypes`:**

- **`cv::PROJ_SPHERICAL_ORTHO` (value: 0):** This model assumes a spherical projection with orthographic rectification. It's suitable for images captured with fisheye lenses that have a very wide field of view. In this model, straight lines in the real world may appear curved in the distorted image, but after undistortion using `cv::PROJ_SPHERICAL_ORTHO`, they will be represented as straight lines.

- **`cv::PROJ_SPHERICAL_EQRECT` (value: 1):** This model also assumes a spherical projection, but with equirectangular rectification. It's appropriate for panoramic images where the goal is to create a rectangular image with minimal distortion. Straight lines in the real world may be slightly bent after undistortion, but the overall distortion is reduced.

**Choosing the appropriate model:**

The choice between these models depends on the type of lens distortion present in your image and the desired outcome.

- If you have a fisheye image and want to preserve straight lines, use `cv::PROJ_SPHERICAL_ORTHO`.
- If you have a panoramic image and want a rectangular representation with minimal distortion, use `cv::PROJ_SPHERICAL_EQRECT`.

**Additional considerations:**

- To use these models effectively, you'll need the camera calibration parameters (camera matrix and distortion coefficients) obtained through a calibration process. These parameters are typically used as input to the `undistort` function.
- OpenCV provides other distortion correction models beyond these two, which might be more suitable for specific lens types or applications. Refer to the OpenCV documentation for a comprehensive list.




## fisheye::undistortPoints

Refs: [1](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#fisheye-undistortpoints)



## fisheye::initUndistortRectifyMap

Refs: [1](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#fisheye-initundistortrectifymap)


## fisheye::undistortImage

Refs: [1](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#fisheye-undistortimage)


## cv::undistort

The function transforms an image to compensate radial and tangential lens distortion.

The function is simply a combination of `initUndistortRectifyMap` (with unity `R` ) and `remap` (with bilinear interpolation). 


```cpp
void cv::undistort	(	InputArray 	src,
OutputArray 	dst,
InputArray 	cameraMatrix,
InputArray 	distCoeffs,
InputArray 	newCameraMatrix = noArray() 
)
```


Refs: [1](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga69f2545a8b62a6b0fc2ee060dc30559d)




## cv::undistortImagePoints

```cpp

void cv::undistortImagePoints	(	InputArray 	src,
OutputArray 	dst,
InputArray 	cameraMatrix,
InputArray 	distCoeffs,
TermCriteria 	= TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 5, 0.01) 
)
```
- `src`: Observed points position, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel (CV_32FC2 or CV_64FC2) (or vector<Point2f> ).

Compute undistorted image points position

