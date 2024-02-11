# 1. Hand Eye Calibration



## 1.1 Eye-in-Hand

<img src="images/Eye-in-Hand.png" />

<br/>
<br/>


From robot kinematic we have the position of griper in robot base at time `i,j`: 



<img src="https://latex.codecogs.com/svg.latex?A_i%3D%7B%7D%5E%7Bb%7DT_%7Bg%7D%5E%7B%28i%29%7D" alt="https://latex.codecogs.com/svg.latex?A_i={}^{b}T_{g}^{(i)}" />
<br/>
<br/>
<img src="https://latex.codecogs.com/svg.latex?A_j%3D%7B%7D%5E%7Bb%7DT_%7Bg%7D%5E%7B%28j%29%7D" alt="https://latex.codecogs.com/svg.latex?A_j={}^{b}T_{g}^{(j)}" />




From the camera we have the position of target in camera at time `i,j`: 

<img src="https://latex.codecogs.com/svg.latex?B_i%3D%7B%7D%5E%7Bc%7DT_%7Bt%7D%5E%7B%28i%29%7D" alt="https://latex.codecogs.com/svg.latex?B_i={}^{c}T_{t}^{(i)}" />
<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?B_j%3D%7B%7D%5E%7Bc%7DT_%7Bt%7D%5E%7B%28j%29%7D" alt="https://latex.codecogs.com/svg.latex?B_j={}^{c}T_{t}^{(j)}" />




<br/>
<br/>
Now

<br/>
<br/>
<img src="https://latex.codecogs.com/svg.latex?B%3DB_iB_j%5E%7B-1%7D" alt="https://latex.codecogs.com/svg.latex?B=B_iB_j^{-1}" />



<img src="https://latex.codecogs.com/svg.latex?T_%7Bc_j%7D%5E%7Bc_i%7D%3D%7B%7D%5E%7Bc%7DT_%7Bt%7D%5E%7B%28i%29%7D%20%5Cleft%20%28%20%7B%7D%5E%7Bc%7DT_%7Bt%7D%5E%7B%28j%29%7D%20%5Cright%20%29%5E%7B-1%7D%3D%20%7B%7D%5E%7Bc%7DT_%7Bt%7D%5E%7B%28i%29%7D%20%7B%7D%5E%7Bt%7DT_%7Bc%7D%5E%7B%28j%29%7D
" alt="https://latex.codecogs.com/svg.latex?T_{c_j}^{c_i}={}^{c}T_{t}^{(i)}  \left ( {}^{c}T_{t}^{(j)} \right )^{-1}=
{}^{c}T_{t}^{(i)}   {}^{t}T_{c}^{(j)}" />


and:
<br/>
<br/>
<img src="https://latex.codecogs.com/svg.latex?A%3DA_j%5E%7B-1%7D%20B_i" alt="https://latex.codecogs.com/svg.latex?A=A_j^{-1} A_i" />





<img src="https://latex.codecogs.com/svg.latex?T_%7Bg_j%7D%5E%7Bg_i%7D%3D%5Cleft%20%28%7B%7D%5E%7Bb%7DT_%7Bg%7D%5E%7B%28i%29%7D%20%5Cright%20%29%5E%7B-1%7D%20%7B%7D%5E%7Bb%7DT_%7Bg%7D%5E%7B%28j%29%7D%3D%20%7B%7D%5E%7Bg%7DT_%7Bb%7D%5E%7B%28i%29%7D%20%7B%7D%5E%7Bb%7DT_%7Bg%7D%5E%7B%28j%29%7D" alt="https://latex.codecogs.com/svg.latex?T_{g_j}^{g_i}=\left ({}^{b}T_{g}^{(i)}  \right )^{-1}  {}^{b}T_{g}^{(j)}=
{}^{g}T_{b}^{(i)}   {}^{b}T_{g}^{(j)}" />

Now if we solve a set of equation in the from of:

<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20AX%3DXB%20%5C%5C%20T_%7Bg_i%7D%5E%7Bg_j%7DX%3DXT_%7Bc_i%7D%5E%7Bc_j%7D%20%5C%5C%20X%3DT_%7Bc%7D%5E%7Bg%7D" alt="https://latex.codecogs.com/svg.latex?\\
AX=XB
\\
T_{g_i}^{g_j}X=XT_{c_i}^{c_j}
\\
X=T_{c}^{g}" />

<br/>
<br/>
<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20AT_c%5E%7Bg%7D%3DT_c%5E%7Bg%7DB%20%5C%5C%20%5C%5C%20T_%7Bg_j%7D%5E%7Bg_i%7DT_c%5E%7Bg%7D%3DT_c%5E%7Bg%7DT_%7Bc_j%7D%5E%7Bc_i%7D%20%5C%5C%20%5C%5C%20T%5E%7Bg_i%7D_c%3DT%5Eg_c_j" alt="https://latex.codecogs.com/svg.latex?\\
AT_c^{g}=T_c^{g}B
\\
\\
T_{g_j}^{g_i}T_c^{g}=T_c^{g}T_{c_j}^{c_i}
\\
\\
T^{g_i}_c=T^g_c_j" />


since the relative position of griper and camera is always fixed at time `i` and `j` we can drop them. 



Refs: [1](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b), [2](https://www.youtube.com/watch?v=xQ79ysnrzUk), [3](https://campar.in.tum.de/Chair/HandEyeCalibration)

## 1.2 Eye-to-Hand

## 1.3 OpenCV Eye-in-Hand API

The following API computes Hand-Eye calibration:  <img src="https://latex.codecogs.com/svg.latex?_%7B%7D%5E%7Bg%7D%5Ctextrm%7BT%7D_c" alt="_{}^{g}\textrm{T}_c" />







```cpp
calibrateHandEye	(	InputArrayOfArrays 	R_gripper2base,
InputArrayOfArrays 	t_gripper2base,
InputArrayOfArrays 	R_target2cam,
InputArrayOfArrays 	t_target2cam,
OutputArray 	R_cam2gripper,
OutputArray 	t_cam2gripper,
HandEyeCalibrationMethod 	method = CALIB_HAND_EYE_TSAI 
)	
```

# 2. Robot-World/Hand-Eye Calibration


<img src="images/robot-world_hand-eye_figure.png" alt="robot-world_hand-eye_figure" />
<br/>
<br/>
<br/>
<br/>
This API computes Robot-World/Hand-Eye calibration:
<img src="https://latex.codecogs.com/svg.latex?_%7B%7D%5E%7Bw%7D%5Ctextrm%7BT%7D_b%20%5Ctext%7B%20and%20%7D%20_%7B%7D%5E%7Bc%7D%5Ctextrm%7BT%7D_g" alt="https://latex.codecogs.com/svg.latex?_{}^{w}\textrm{T}_b \text{ and } _{}^{c}\textrm{T}_g" />



## 1.3 OpenCV Robot-World/Hand-Eye API


```cpp
cv::calibrateRobotWorldHandEye	(	InputArrayOfArrays 	R_world2cam,
InputArrayOfArrays 	t_world2cam,
InputArrayOfArrays 	R_base2gripper,
InputArrayOfArrays 	t_base2gripper,
OutputArray 	R_base2world,
OutputArray 	t_base2world,
OutputArray 	R_gripper2cam,
OutputArray 	t_gripper2cam,
RobotWorldHandEyeCalibrationMethod 	method = CALIB_ROBOT_WORLD_HAND_EYE_SHAH 
)	
```

Refs: [1](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga41b1a8dd70eae371eba707d101729c36), [2](https://support.zivid.com/en/latest/academy/applications/hand-eye.html), [3](https://wiki.ros.org/ensenso_driver/Tutorials/HandEyeCalibration)
