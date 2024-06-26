
# Graphical Projection

There are two graphical projection categories:

- parallel projection
- perspective projection


<img src="images/comparison_of_graphical_projections.svg" width="500" height="500" />



# Pinhole Camera Model

the coordinates  of point <img src="https://latex.codecogs.com/svg.image?Q(x,y)" title="https://latex.codecogs.com/svg.image?Q(x,y)" /> depend on the coordinates of point <img src="https://latex.codecogs.com/svg.image?P(X_w,Y_w,Z_w)" title="https://latex.codecogs.com/svg.image?P(X_w,Y_w,Z_w)" /> 

- <img src="https://latex.codecogs.com/svg.image?\frac{-y}{Y_w}=\frac{f}{Z_w}" title="https://latex.codecogs.com/svg.image?\frac{-y}{Y_w}=\frac{f}{Z_w}" />

- <img src="https://latex.codecogs.com/svg.image?\frac{-x}{X_w}=\frac{f}{Z_w}" title="https://latex.codecogs.com/svg.image?\frac{-x}{X_w}=\frac{f}{Z_w}" />


<img src="images/Pinhole.svg" />
<img src="images/Pinhole2.svg" />



Refs: [1](https://en.wikipedia.org/wiki/Pinhole_camera_model#Geometry),
[2](https://ksimek.github.io/2013/08/13/intrinsic/),


## Rotated Image and the Virtual Image Plane

The mapping from 3D to 2D coordinates described by a pinhole camera is a perspective projection followed by a `180°` rotation in the image plane. This corresponds to how a real pinhole camera operates; the resulting image is rotated `180°` and the relative size of projected objects depends on their distance to the focal point and the overall size of the image depends on the distance f between the image plane and the focal point. In order to produce an unrotated image, which is what we expect from a camera we Place the image plane so that it intersects the <img src="https://latex.codecogs.com/svg.image?Z" title="https://latex.codecogs.com/svg.image?Z" /> axis at `f` instead of at `-f` and rework the previous calculations. This would generate a virtual (or front) image plane which cannot be implemented in practice, but provides a theoretical camera which may be simpler to analyse than the real one.

<img src="images/pinhole_camera_model.png" />


<img src="https://latex.codecogs.com/svg.image?mx=\frac{&space;\text{Number&space;of&space;Pixel&space;In&space;Width}}{\text{Width&space;of&space;Sensor}}=\frac{1}{\text{Width&space;of&space;Pixel}" title="https://latex.codecogs.com/svg.image?mx=\frac{ \text{Number of Pixel In Width}}{\text{Width of Sensor}}=\frac{1}{\text{Width of Pixel}" />

<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?my=\frac{&space;\text{Number&space;of&space;Pixel&space;In&space;Heigh}}{\text{Height&space;of&space;Sensor}}=\frac{1}{\text{Height&space;of&space;Pixel}" title="https://latex.codecogs.com/svg.image?my=\frac{ \text{Number of Pixel In Heigh}}{\text{Height of Sensor}}=\frac{1}{\text{Height of Pixel}" />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?cy=\frac{&space;\text{Number&space;of&space;Pixel&space;In&space;Height}}{\text{2}}" title="https://latex.codecogs.com/svg.image?cy=\frac{ \text{Number of Pixel In Height}}{\text{2}}" />    


<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?cx=\frac{&space;\text{Number&space;of&space;Pixel&space;In&space;Width}}{\text{2}}" title="https://latex.codecogs.com/svg.image?cx=\frac{ \text{Number of Pixel In Width}}{\text{2}}" />    

<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?f_x=f\times&space;m_x" title="https://latex.codecogs.com/svg.image?f_x=f\times m_x" />


<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?f_y=f\times&space;m_y" title="https://latex.codecogs.com/svg.image?f_y=f\times m_y" />



<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?\text{column=u}&space;=f_x\frac{X}{Z}&space;&plus;cx" title="https://latex.codecogs.com/svg.image?\text{column=u} =f_x\frac{X}{Z} +cx" />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?\text{row=v}&space;=f_y\frac{Y}{Z}&space;&plus;cy" title="https://latex.codecogs.com/svg.image?\text{row=v} =f_y\frac{Y}{Z} +cy" />
<br/>
<br/>

so the projection of the point is at `(u,v)`, Please note that `u` will increase from left to right and `v` will increase from top to bottom 



<br/>
<br/>


```cpp
              u                      
    ------------------------------------------►
    | (0,0) (1,0) (2,0) (3,0) (u,v) (u+1,v)
    | (0,1) (1,1) (2,1) (3,1)
    | (0,2) (1,2) (2,2) (3,2)
  v | (u,v)
    | (u,v+1)
    |
    |
    ▼

```


## Example of Projection 


<img src="https://latex.codecogs.com/svg.image?\text{Number&space;of&space;Pixel&space;In&space;Width=640}" title="https://latex.codecogs.com/svg.image?\text{Number of Pixel In Width=640}" />
<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?\text{Number&space;of&space;Pixel&space;In&space;Height=480}&space;" title="https://latex.codecogs.com/svg.image?\text{Number of Pixel In Height=480} " />



<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?\text{Height%20of%20Sensor=10%20mm}" title="https://latex.codecogs.com/svg.image?\text{Height of Sensor=10 mm}" />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?\text{Width&space;of&space;Sensor=10&space;mm}&space;" title="https://latex.codecogs.com/svg.image?\text{Width of Sensor=10 mm} " />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?\text{f=0.1}&space;" title="https://latex.codecogs.com/svg.image?\text{f=0.1} " />
<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?\text{Points&space;in&space;Camera&space;Coordinate=}\begin{bmatrix}&space;0&&space;2&space;&&space;1&space;&&space;&space;2&&space;3&space;&&space;2&&space;2\\&space;0&&space;1&space;&&space;2&space;&&space;&space;2&space;&&space;2&&space;3&space;&&space;4\\&space;0&&space;1&space;&&space;1&space;&&space;&space;1&&space;1&&space;1&space;&&space;1\\\end{bmatrix}&space;&space;" title="https://latex.codecogs.com/svg.image?\text{Points in Camera Coordinate=}\begin{bmatrix} 0& 2 & 1 & 2& 3 & 2& 2\\ 0& 1 & 2 & 2 & 2& 3 & 4\\ 0& 1 & 1 & 1& 1& 1 & 1\\\end{bmatrix} " />


<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?cy=\frac{480}{2}=240" title="https://latex.codecogs.com/svg.image?cy=\frac{480}{2}=240" />
<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?cx=\frac{640}{2}=320" title="https://latex.codecogs.com/svg.image?cx=\frac{640}{2}=320" />
<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?fx=0.1&space;\times&space;\frac{640}{10}=6.4" title="https://latex.codecogs.com/svg.image?fx=0.1 \times \frac{640}{10}=6.4" />

<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?fy=0.1&space;\times&space;\frac{240}{10}=2.4" title="https://latex.codecogs.com/svg.image?fy=0.1 \times \frac{240}{10}=2.4" />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?K=\begin{bmatrix}6.4&space;&&space;0&space;&space;&&space;320&space;&space;\\0,&space;&&space;4.8&space;&&space;240&space;\\0&space;&space;&&space;0&space;&space;&&space;1&space;\\\end{bmatrix}&space;" title="https://latex.codecogs.com/svg.image?K=\begin{bmatrix}6.4 & 0 & 320 \\0, & 4.8 & 240 \\0 & 0 & 1 \\\end{bmatrix} " />


<br/>
<br/>


projected pints in camera:

<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?\text{Projectd&space;Points&space;Camera&space;Plane=}\begin{bmatrix}\text{column&space;}\\\text{row&space;}\\1\end{bmatrix}&space;=\begin{bmatrix}320&space;&&space;332.8&space;&&space;326.4&space;&&space;332.8&space;&&space;339.2&space;&&space;332.8&space;&&space;332.8\\&space;240&space;&&space;244.8&space;&&space;249.6&space;&&space;249.6&space;&&space;249.6&space;&&space;254.4&space;&&space;259.2\\&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1&space;&&space;1\\\end{bmatrix}&space;&space;" title="https://latex.codecogs.com/svg.image?\text{Projectd Points Camera Plane=}\begin{bmatrix}\text{column }\\\text{row }\\1\end{bmatrix} =\begin{bmatrix}320 & 332.8 & 326.4 & 332.8 & 339.2 & 332.8 & 332.8\\ 240 & 244.8 & 249.6 & 249.6 & 249.6 & 254.4 & 259.2\\ 1 & 1 & 1 & 1 & 1 & 1 & 1\\\end{bmatrix} " />

<br/>
<br/>








<br/>
<br/>

<img src="images/image_0.100000_.jpg">


## Image coordinate and Matrix coordinate

In OpenCV, `Point(x=u=column,y=v=row)`. For instance the point in the following image can be accessed with

```cpp
              x=u                      
    --------column---------►
    | Point(0,0) Point(1,0) Point(2,0) Point(3,0)
    | Point(0,1) Point(1,1) Point(2,1) Point(3,1)
    | Point(0,2) Point(1,2) Point(2,2) Point(3,2)
 y=v|
   row
    |
    |
    ▼

```
However if you access an image directly, the access is matrix based index, the order is <img src="https://latex.codecogs.com/svg.image?\text{(row,&space;column)}" title="https://latex.codecogs.com/svg.image?\text{(row, column)}" />

```cpp
    X                      
    --------column---------►
    | mat.at<type>(0,0) mat.at<type>(0,1) mat.at<type>(0,2) mat.at<type>(0,3)
    | mat.at<type>(1,0) mat.at<type>(1,1) mat.at<type>(1,2) mat.at<type>(1,3)
    | mat.at<type>(2,0) mat.at<type>(2,1) mat.at<type>(2,2) mat.at<type>(2,3)
  y |
   row
    |
    |
    ▼
```    


So the following will return the same value:



```cpp
mat.at<type>(row,column) 
mat.at<type>(cv::Point(column,row))
```
For instance:
```cpp
std::cout<<static_cast<unsigned>(img.at<uchar>(row,column))    <<std::endl;
std::cout<<static_cast<unsigned>(img.at<uchar>( cv::Point(column,row))     )<<std::endl;
```




## Projection Matrix and Frame Transformation

Projection matrix refers to the pinhole camera model, a camera matrix <img src="https://latex.codecogs.com/svg.image?\text{P}_{3%20\times%204}" alt="https://latex.codecogs.com/svg.image?\text{P}_{3 \times 4}" /> is used to denote a projective mapping from world coordinates to pixel coordinates. If the camera and world share the **same coordinate system**:


<br/>
<br/>


<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20K%3D%7B%5Cbegin%7Bbmatrix%7Df_%7Bx%7D%26%5Cgamma%20%26c_%7Bx%7D%5C%5C0%26f_%7By%7D%26c_%7By%7D%5C%5C0%260%261%5Cend%7Bbmatrix%7D%7D%7D" alt="https://latex.codecogs.com/svg.image?{\displaystyle K={\begin{bmatrix}f_{x}&\gamma &c_{x}\\0&f_{y}&c_{y}\\0&0&1\end{bmatrix}}}" />
<br/>
<br/>


<img src="https://latex.codecogs.com/svg.latex?%5Cmathbf%7BP%7D%3D%20%7B%5Cdisplaystyle%20%7B%5Cbegin%7Bbmatrix%7Df_%7Bx%7D%26%5Cgamma%20%26c_%7Bx%7D%5C%5C0%26f_%7By%7D%26c_%7By%7D%5C%5C0%260%261%5Cend%7Bbmatrix%7D%7D%7D%20%5Cbegin%7Bbmatrix%7D%201%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%201%20%26%200%20%26%200%5C%5C%200%20%26%200%20%26%201%20%26%200%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\mathbf{P}= {\displaystyle {\begin{bmatrix}f_{x}&\gamma &c_{x}\\0&f_{y}&c_{y}\\0&0&1\end{bmatrix}}} \begin{bmatrix}  1& 0 & 0 & 0 \\ 0 & 1 & 0 & 0\\ 0 & 0 & 1 & 0 \end{bmatrix}" />



<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?%5Cmathbf%7BP_%7B3%5Ctimes%204%7D%3DK_%7B3%5Ctimes%203%7D%5BI%7C0%5D_%7B3%5Ctimes%204%7D%7D" alt="https://latex.codecogs.com/svg.latex?\mathbf{P_{3\times 4}=K_{3\times 3}[I|0]_{3\times 4}}" />
<br/>
<br/>


## World to Camera Transformation

What if the camera and world are in **different coordinate system**, 


<img src="images/world_to_camera1.jpg" width="50%" height="50%" alt="" />

First subtract the position of camera (origin of camera expressed in world coordinate) from the point, Then rotate it with matrix, <img src="https://latex.codecogs.com/svg.latex?R_W%5EC" alt="https://latex.codecogs.com/svg.latex?R_W^C" />, to get this matrix just express the axis of world coordinate in camera coordinate.



<img src="images/world_to_camera2.jpg" width="50%" height="50%" alt="" />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?P_C%20%3D%20R_W%5EC%20%28%20P_W%20-%20C%29" alt="https://latex.codecogs.com/svg.latex?P_C = R_W^C ( P_W - C)" />

<br/>
<br/>

<img src="images/world_to_camera2.jpg" width="50%" height="50%" alt="" />

### Figuring out Rotations

Example

<img src="images/world_to_camera4.jpg" width="50%" height="50%" alt="" />



<img src="https://latex.codecogs.com/svg.latex?R_%7Btrain%7D%5E%7Bc%7D%3D%5Cbegin%7Bbmatrix%7D%200%20%26%200%20%26%201%5C%5C%200%20%26%20-1%20%26%200%5C%5C%201%20%26%200%20%26%200%20%5Cend%7Bbmatrix%7D"  alt="https://latex.codecogs.com/svg.latex?R_{train}^{c}=\begin{bmatrix}
0 & 0 & 1\\ 
0 & -1 & 0\\ 
1 & 0  & 0
\end{bmatrix}" />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?R_%7Bfly%7D%5E%7Bc%7D%3D%5Cbegin%7Bbmatrix%7D%200%20%26%20-1%20%26%20%5C%5C%200%20%26%200%20%26%201%5C%5C%20-1%20%26%200%20%26%200%20%5Cend%7Bbmatrix%7D"  alt="https://latex.codecogs.com/svg.latex?R_{fly}^{c}=\begin{bmatrix}
0 & -1 & \\ 
0 & 0 & 1\\ 
-1 & 0  & 0
\end{bmatrix}" />

Now lets do it algebraically, First let's revisit some basics. Inverse of `4x4` transformation matrix is: 

<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20A%3D%5Cbegin%7Bbmatrix%7D%20R_c%5Ew%20%26%20t_c%5Ew%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D%20%5C%5C%20A%5E%7B-1%7D%3D%5Cbegin%7Bbmatrix%7D%20R_c%5Ew%5ET%20%26%20-R_c%5Ew%5ETt_c%5Ew%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\\
A=\begin{bmatrix}
R_c^w & t_c^w\\ 
0 & 1
\end{bmatrix}
\\ A^{-1}=\begin{bmatrix} R_c^w^T & -R_c^w^Tt_c^w\\  0 & 1 \end{bmatrix} " />




<img src="https://latex.codecogs.com/svg.latex?X%5Ec%3DA%5E%7B-1%7DX%5Ew%3D%5Cbegin%7Bbmatrix%7D%20Rc%5Ew%5ET%20%26%20-Rc%5Ew%5ETtc%5Ew%5C%5C%200%20%26%201%20%5Cend%7Bbmatrix%7D%20X%5Ew%3D%5Cbegin%7Bbmatrix%7D%20Rc%5Ew%5ET%28X%5Ew-t_c%5Ew%29%20%5C%5C%201%20%5Cend%7Bbmatrix%7D_%7B4%5Ctimes1%7D" alt="https://latex.codecogs.com/svg.latex?X^c=A^{-1}X^w=\begin{bmatrix} Rc^w^T & -Rc^w^Ttc^w\\  0 & 1 \end{bmatrix} X^w=\begin{bmatrix}
Rc^w^T(X^w-t_c^w)
\\ 
1
\end{bmatrix}_{4\times1}" />



<br/>
<br/>







<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20%5Cmathbf%7BP%3DK%5BR_w%5Ec%7Ct_w%5Ec%5D%7D%20%5C%5C%20%5Cmathbf%7BP%3DKR_w%5Ec%5BI-C%5D%7D" alt="https://latex.codecogs.com/svg.latex?\\
\mathbf{P=K[R_w^c|t_w^c]}
\\
\mathbf{P=KR_w^c[I-C]}" />



Where <img src="https://latex.codecogs.com/svg.latex?%5Cmathbf%7BC%7D" alt="https://latex.codecogs.com/svg.latex?\mathbf{C}" /> is the location of camera in world coordinate, therefore:




<img src="https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bt_w%5Ec%3D-R_w%5EcC%7D" alt="https://latex.codecogs.com/svg.latex?\mathbf{t_w^c=-R_w^cC}" />





<br/>
<br/>






Ref: [1](https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf), [2](https://stackoverflow.com/questions/73340550/how-does-opencv-projectpoints-perform-transformations-before-projecting), [3](https://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf)

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20z_%7Bc%7D%7B%5Cbegin%7Bbmatrix%7Du%5C%5Cv%5C%5C1%5Cend%7Bbmatrix%7D%7D%3DK%5C%2C%7B%5Cbegin%7Bbmatrix%7DR_%7Bw%7D%5E%7Bc%7D%20%26T_%7Bw%7D%5E%7Bc%7D%5Cend%7Bbmatrix%7D%7D%7B%5Cbegin%7Bbmatrix%7DX_%7Bw%7D%5C%5CY_%7Bw%7D%5C%5CZ_%7Bw%7D%5C%5C1%5Cend%7Bbmatrix%7D%7D%3DP%7B%5Cbegin%7Bbmatrix%7DX_%7Bw%7D%5C%5CY_%7Bw%7D%5C%5CZ_%7Bw%7D%5C%5C1%5Cend%7Bbmatrix%7D%7D%7D" alt="{\displaystyle z_{c}{\begin{bmatrix}u\\v\\1\end{bmatrix}}=K\,{\begin{bmatrix}R_{w}^{c} &T_{w}^{c}\end{bmatrix}}{\begin{bmatrix}X_{w}\\Y_{w}\\Z_{w}\\1\end{bmatrix}}=P{\begin{bmatrix}X_{w}\\Y_{w}\\Z_{w}\\1\end{bmatrix}}}" />





<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?[u\&space;v\&space;1]^{T}" title="https://latex.codecogs.com/svg.image?[u\ v\ 1]^{T}" /> represent a 2D point position in pixel coordinates and <img src="https://latex.codecogs.com/svg.image?[X_{w}\&space;Y_{w}\&space;Z_{w}\&space;1]^{T}" title="https://latex.codecogs.com/svg.image?[X_{w}\ Y_{w}\ Z_{w}\ 1]^{T}" /> represent a 3D point position in world coordinates.



<br/>
<br/>





```cpp

void cv::projectPoints	(	InputArray 	objectPoints,
InputArray 	rvec,
InputArray 	tvec,
InputArray 	cameraMatrix,
InputArray 	distCoeffs,
OutputArray 	imagePoints,
OutputArray 	jacobian = noArray(),
double 	aspectRatio = 0 
)	
```

`rvec` and `tvec` are the rotation and translation that transform object pose from world coordinate into 
camera's coordinate, namely `R_c_w=R_w_c.inv()` and `T_c_w=-T_w_c`



<br/>
<br/>


# Decompose Projection Matrix

Refs: [1](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaaae5a7899faa1ffdf268cd9088940248)



# OpenCV, OpenGL, VTK, Drones, and Cars Camera Coordinate

## OpenCV
<img src="images/opencv_coordinate.png" heigh="333" width="466" />

<br/>
<br/>

## OpenGL

<img src="images/opengl_coordinate.png" heigh="333" width="466" />

<br/>
<br/>

## VTK

<img src="images/vtk.png"  />

<br/>
<br/>

## For Airplanes/ Drones


<img src="images/frame_heading.px4.png" width="90%" height="90%"  />


Refs: [1](https://docs.px4.io/main/en/config/flight_controller_orientation.html) 
<br/>
<br/>

<img src="images/RPY_angles_of_airplanes.png"  />

<br/>
<br/>

## For Cars



<img src="images/RPY_angles_of_cars.png"  />


<br/>
<br/>


## Representing Robot Pose


<img src="images/representing_robot_pose1.png" width="50%" height="50%" />

Refs: [1](https://web.archive.org/web/20161029231029/https://paulfurgale.info/news/2014/6/9/representing-robot-pose-the-good-the-bad-and-the-ugly)

<br/>
<br/>



# Global Shutter vs  Rolling Shutter

<img src="images/Rolling-Blue.gif" heigh="333" width="333" />

[image courtesy](https://www.photometrics.com/learn/advanced-imaging/rolling-vs-global-shutter)


