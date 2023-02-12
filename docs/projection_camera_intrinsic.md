# 1. Graphical Projection

There are two graphical projection categories:

- parallel projection
- perspective projection


<img src="images/comparison_of_graphical_projections.svg" width="500" height="500" />



# 2. Pinhole Camera Model

the coordinates  of point <img src="https://latex.codecogs.com/svg.image?Q(x,y)" title="https://latex.codecogs.com/svg.image?Q(x,y)" /> depend on the coordinates of point <img src="https://latex.codecogs.com/svg.image?P(X_w,Y_w,Z_w)" title="https://latex.codecogs.com/svg.image?P(X_w,Y_w,Z_w)" /> 

- <img src="https://latex.codecogs.com/svg.image?\frac{-y}{Y_w}=\frac{f}{Z_w}" title="https://latex.codecogs.com/svg.image?\frac{-y}{Y_w}=\frac{f}{Z_w}" />

- <img src="https://latex.codecogs.com/svg.image?\frac{-x}{X_w}=\frac{f}{Z_w}" title="https://latex.codecogs.com/svg.image?\frac{-x}{X_w}=\frac{f}{Z_w}" />


![Pinhole](images/Pinhole.svg)
![Pinhole2](images/Pinhole2.svg)



Refs: [1](https://en.wikipedia.org/wiki/Pinhole_camera_model#Geometry),
[2](https://ksimek.github.io/2013/08/13/intrinsic/),


## 2.1 Rotated Image and the Virtual Image Plane

The mapping from 3D to 2D coordinates described by a pinhole camera is a perspective projection followed by a `180°` rotation in the image plane. This corresponds to how a real pinhole camera operates; the resulting image is rotated `180°` and the relative size of projected objects depends on their distance to the focal point and the overall size of the image depends on the distance f between the image plane and the focal point. In order to produce an unrotated image, which is what we expect from a camera we Place the image plane so that it intersects the <img src="https://latex.codecogs.com/svg.image?Z" title="https://latex.codecogs.com/svg.image?Z" /> axis at `f` instead of at `-f` and rework the previous calculations. This would generate a virtual (or front) image plane which cannot be implemented in practice, but provides a theoretical camera which may be simpler to analyse than the real one.


## 2.2 Camera Resectioning and Projection Matrix 

Projection refers to the pinhole camera model, a camera matrix <img src="https://latex.codecogs.com/svg.image?\text{P}" title="https://latex.codecogs.com/svg.image?\text{P}" /> is used to denote a projective mapping from world coordinates to pixel coordinates.

<br/>
<br/>
Assuming that the camera and world share the same coordinate system:

<br/>
<br/>
<img src="https://latex.codecogs.com/svg.latex?%5Cmathbf%7BP%3DK%5BI%7C0%5D%7D" alt="https://latex.codecogs.com/svg.latex?\mathbf{P=K[I|0]} " />
<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?%5Cmathbf%7BP%7D%3D%20%7B%5Cdisplaystyle%20%7B%5Cbegin%7Bbmatrix%7Df_%7Bx%7D%26%5Cgamma%20%26c_%7Bx%7D%5C%5C0%26f_%7By%7D%26c_%7By%7D%5C%5C0%260%261%5Cend%7Bbmatrix%7D%7D%7D%20%5Cbegin%7Bbmatrix%7D%201%26%200%20%26%200%20%26%200%20%5C%5C%200%20%26%201%20%26%200%20%26%200%5C%5C%200%20%26%200%20%26%201%20%26%200%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\mathbf{P}= {\displaystyle {\begin{bmatrix}f_{x}&\gamma &c_{x}\\0&f_{y}&c_{y}\\0&0&1\end{bmatrix}}} \begin{bmatrix}  1& 0 & 0 & 0 \\ 0 & 1 & 0 & 0\\ 0 & 0 & 1 & 0 \end{bmatrix}" />

<br/>
<br/>
If they are different:

<br/>
<br/>
<img src="images/projection_in_ex.jpg" height="520" width="500" alt="" />

[image courtesy](https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf)
 

<br/>
<br/>


<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20%5Cmathbf%7BP%3DKR%5BI-C%5D%7D%20%5C%5C%20%5Cmathbf%7BP%3DK%5BR%7Ct%5D%7D%20%5C%5C%20%5Ctext%7Bwhere%3A%7D%20%5C%5C%20%5Cmathbf%7Bt%3D-RC%7D%20%5C%5C%20%5Ctext%7Band%20C%20is%20Coordinate%20of%20the%20camera%20center%20in%20the%20world%20coordinate%20frame%7D" alt="https://latex.codecogs.com/svg.latex?\\
\mathbf{P=KR[I-C]}
\\
\mathbf{P=K[R|t]}
\\ \text{where:}
\\
\mathbf{t=-RC}
\\
\text{and C is Coordinate of the
camera center in the
world coordinate frame} " />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?{\displaystyle&space;z_{c}{\begin{bmatrix}u\\v\\1\end{bmatrix}}=K\,{\begin{bmatrix}R&T\end{bmatrix}}{\begin{bmatrix}X_{w}\\Y_{w}\\Z_{w}\\1\end{bmatrix}}=P{\begin{bmatrix}X_{w}\\Y_{w}\\Z_{w}\\1\end{bmatrix}}}" title="https://latex.codecogs.com/svg.image?{\displaystyle z_{c}{\begin{bmatrix}u\\v\\1\end{bmatrix}}=K\,{\begin{bmatrix}R&T\end{bmatrix}}{\begin{bmatrix}X_{w}\\Y_{w}\\Z_{w}\\1\end{bmatrix}}=P{\begin{bmatrix}X_{w}\\Y_{w}\\Z_{w}\\1\end{bmatrix}}}" />





<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?[u\&space;v\&space;1]^{T}" title="https://latex.codecogs.com/svg.image?[u\ v\ 1]^{T}" /> represent a 2D point position in pixel coordinates and <img src="https://latex.codecogs.com/svg.image?[X_{w}\&space;Y_{w}\&space;Z_{w}\&space;1]^{T}" title="https://latex.codecogs.com/svg.image?[X_{w}\ Y_{w}\ Z_{w}\ 1]^{T}" /> represent a 3D point position in world coordinates.


<img src="https://latex.codecogs.com/svg.image?{\displaystyle&space;K={\begin{bmatrix}f_{x}&\gamma&space;&c_{x}&0\\0&f_{y}&c_{y}&0\\0&0&1&0\end{bmatrix}}}" title="https://latex.codecogs.com/svg.image?{\displaystyle K={\begin{bmatrix}f_{x}&\gamma &c_{x}&0\\0&f_{y}&c_{y}&0\\0&0&1&0\end{bmatrix}}}" />

<br/>
<br/>


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


## 2.3 Example of Projection 


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





![pinhole_camera_model](images/pinhole_camera_model.png)


<br/>
<br/>

<img src="images/image_0.100000_.jpg">


## 2.4 Image coordinate and Matrix coordinate

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



## 2.5 Projection with Lens Distortion


<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7D%20x%5C%5C%20y%5C%5C%20z%20%5Cend%7Bbmatrix%7D%3D%20R%5Cbegin%7Bbmatrix%7D%20X%5C%5C%20Y%5C%5C%20Z%20%5Cend%7Bbmatrix%7D&plus;t" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix} x\\ y\\ z \end{bmatrix}= R\begin{bmatrix} X\\ Y\\ Z \end{bmatrix}+t" />



<img src="https://latex.codecogs.com/svg.latex?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20%7Bx%7D%27%3D%5Cfrac%7Bx%7D%7Bz%7D%20%5C%5C%20%7By%7D%27%3D%5Cfrac%7By%7D%7Bz%7D%20%5Cend%7Bmatrix%7D%5Cright." alt="https://latex.codecogs.com/svg.latex?\left\{\begin{matrix}
{x}'=\frac{x}{z} 
\\ 
{y}'=\frac{y}{z} 
\end{matrix}\right." />


<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7BBmatrix%7D%20%7Bx%7D%27%27%3D%7Bx%7D%27%20%5Cfrac%7B1&plus;k_1r%5E2&plus;%20k_2r%5E4&plus;%20k_3r%5E6%7D%7B1&plus;k_4r%5E2%20&plus;k_5r%5E4%20&plus;%20k_6r6%20%7D%20&plus;2p_1%7Bx%7D%27%7By%7D%27&plus;p_2%28r%5E2&plus;2%7Bx%7D%27%5E2%29%20%5C%5C%20%7By%7D%27%27%3D%7By%7D%27%20%5Cfrac%7B1&plus;k_1r%5E2&plus;%20k_2r%5E4&plus;%20k_3r%5E6%7D%7B1&plus;k_4r%5E2%20&plus;k_5r%5E4%20&plus;%20k_6r6%20%7D&plus;p_1%28r%5E2&plus;2%7Bx%7D%27%5E2%29%20&plus;2p_2%7Bx%7D%27%7By%7D%27%20%5Cend%7BBmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\begin{Bmatrix}
{x}''={x}' \frac{1+k_1r^2+ k_2r^4+ k_3r^6}{1+k_4r^2 +k_5r^4 + k_6r6 } +2p_1{x}'{y}'+p_2(r^2+2{x}'^2) \\ {y}''={y}' \frac{1+k_1r^2+ k_2r^4+ k_3r^6}{1+k_4r^2 +k_5r^4 + k_6r6 }+p_1(r^2+2{x}'^2) +2p_2{x}'{y}' \end{Bmatrix}" />






<img src="https://latex.codecogs.com/svg.latex?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20u%3Df_x%20%5Ctimes%20%7Bx%7D%27%27&plus;c_x%20%5C%5C%20v%3Df_y%20%5Ctimes%20%7By%7D%27%27&plus;c_y%20%5Cend%7Bmatrix%7D%5Cright." alt="https://latex.codecogs.com/svg.latex?\left\{\begin{matrix} u=f_x \times {x}''+c_x \\ v=f_y \times {y}''+c_y 
\end{matrix}\right." />


<img src="https://latex.codecogs.com/svg.latex?%5Ctext%7Bwhere%3A%20%7D%20r%5E2%3D%7Bx%7D%27%5E2%20&plus;%20%7By%7D%27%5E2" alt="https://latex.codecogs.com/svg.latex?\text{where: } r^2={x}'^2 + {y}'^2" />



## 2.6 Undistorting Points

### 2.6.1 initUndistortRectifyMap
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



Refs: [1](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a)


### 2.6.2 undistort
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
For each observed point coordinate (u,v) the function computes:

Refs: [1](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga69f2545a8b62a6b0fc2ee060dc30559d)



# 3D World Unit Vector




Refs: [1](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga69f2545a8b62a6b0fc2ee060dc30559d)

# 3D World Unit Vector

Refs: [1](https://stackoverflow.com/questions/12977980/in-opencv-converting-2d-image-point-to-3d-world-unit-vector),
[2](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html),
[3](https://stackoverflow.com/questions/44888119/c-opencv-calibration-of-the-camera-with-different-resolution),
[4](https://docs.opencv.org/3.2.0/da/d54/group__imgproc__transform.html#ga55c716492470bfe86b0ee9bf3a1f0f7e),
[5](https://www.mathematik.uni-marburg.de/~thormae/lectures/graphics1/graphics_6_1_eng_web.html#1)






# Resizing Image Effect on the Camera Intrinsic Matrix







