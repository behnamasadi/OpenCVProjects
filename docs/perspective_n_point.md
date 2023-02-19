# Perspective-n-Point
Perspective-n-Point is the problem to estimating the pose of a **calibrated** camera given a set of n 3D points in the world and their correspondence projection point in camera.

# P3P

<img src="images/p3p.svg" alt="images/p3p.svg" />

First step: using cosine to relate the length in 4 triangle in the tetrahedron:

<img src="images/p3p_length.jpg" alt="images/p3p_length.jpg" />
 


second step:

<img src=" images/p3p_polynominals.jpg" width="860" height="540" alt=" images/p3p_polynominals.jpg" />


which gives us for possible solution so we need a 4th point or initial guess.



The solution look like these 4 tetrahedron:


<img src="images/p3p_results.png" alt="images/p3p_results.png" />

<br/>
<br/>

```cpp
solveP3P	(	InputArray 	objectPoints,
InputArray 	imagePoints,
InputArray 	cameraMatrix,
InputArray 	distCoeffs,
OutputArrayOfArrays 	rvecs,
OutputArrayOfArrays 	tvecs,
int 	flags 
)	
```

## Critical Cylinder

<img src="images/critical_cylinder.png" alt="images/critical_cylinder.png" />

The solution get in-stable


Refs: [1](https://www.youtube.com/watch?v=N1aCvzFll6Q), [2](https://www.cis.upenn.edu/~cis580/Spring2015/Lectures/cis580-13-LeastSq-PnP.pdf), [3](https://www.youtube.com/watch?v=xdlLXEyCoJY)






# PnP

```cpp
bool cv::solvePnP	(	InputArray 	objectPoints,
InputArray 	imagePoints,
InputArray 	cameraMatrix,
InputArray 	distCoeffs,
OutputArray 	rvec,
OutputArray 	tvec,
bool 	useExtrinsicGuess = false,
int 	flags = SOLVEPNP_ITERATIVE 
)	
```


`useExtrinsicGuess`:	Parameter used for `SOLVEPNP_ITERATIVE`. If true (1), the function uses the provided `rvec` and `tvec` values as initial


With SOLVEPNP_ITERATIVE method and useExtrinsicGuess=true, the minimum number of points is 3 (3 points are sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the global solution to converge.


Refs: [1](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html), [2](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)


