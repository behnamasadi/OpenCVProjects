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


