# Recovery of R,T from Essential Matrix


```cpp
recoverPose() [1/4]
int cv::recoverPose	(	InputArray 	points1,
InputArray 	points2,
InputArray 	cameraMatrix1,
InputArray 	distCoeffs1,
InputArray 	cameraMatrix2,
InputArray 	distCoeffs2,
OutputArray 	E,
OutputArray 	R,
OutputArray 	t,
int 	method = cv::RANSAC,
double 	prob = 0.999,
double 	threshold = 1.0,
InputOutputArray 	mask = noArray() 
)	
```


Refs: [1](https://www.ece.ualberta.ca/~lcheng5/papers/XuEtAl_TPAMI17.pdf), [2](https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture10-3-decomposing-F-matrix-into-Rotation-and-Translation.pdf), [3](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga1b2f149ee4b033c4dfe539f87338e243)
