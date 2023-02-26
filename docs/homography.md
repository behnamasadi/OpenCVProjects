# 1. Homography
Any two images of the same planar surface in space are related by a homography. 
The planar homography relates the transformation between two planes (up to a scale factor):


<img src="https://latex.codecogs.com/svg.latex?s%20%5Cbegin%7Bbmatrix%7D%20x%5E%7B%27%7D%20%5C%5C%20y%5E%7B%27%7D%20%5C%5C%201%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cmathbf%7BH%7D%20%5Cbegin%7Bbmatrix%7D%20x%20%5C%5C%20y%20%5C%5C%201%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20h_%7B11%7D%20%26%20h_%7B12%7D%20%26%20h_%7B13%7D%20%5C%5C%20h_%7B21%7D%20%26%20h_%7B22%7D%20%26%20h_%7B23%7D%20%5C%5C%20h_%7B31%7D%20%26%20h_%7B32%7D%20%26%20h_%7B33%7D%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%20x%20%5C%5C%20y%20%5C%5C%201%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?s \begin{bmatrix} x^{'} \\ y^{'} \\ 1 \end{bmatrix} = \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}" />




The homography matrix is a `3x3` matrix with 8 DoF as it is estimated up to a scale. It is generally normalized:
 
 
 <img src="https://latex.codecogs.com/svg.latex?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20h_%7B33%7D%20%3D%201%20%5C%5C%20or%20%5C%5C%20h_%7B11%7D%5E2%20&plus;%20h_%7B12%7D%5E2%20&plus;%20h_%7B13%7D%5E2%20&plus;%20h_%7B21%7D%5E2%20&plus;%20h_%7B22%7D%5E2%20&plus;%20h_%7B23%7D%5E2%20&plus;%20h_%7B31%7D%5E2%20&plus;%20h_%7B32%7D%5E2%20&plus;%20h_%7B33%7D%5E2%20%3D%201%20%5Cend%7Bmatrix%7D%5Cright." alt="https://latex.codecogs.com/svg.latex?\left\{\begin{matrix} h_{33} = 1 \\ or \\ h_{11}^2 + h_{12}^2 + h_{13}^2 + h_{21}^2 + h_{22}^2 + h_{23}^2 + h_{31}^2 + h_{32}^2 + h_{33}^2 = 1
\end{matrix}\right." />
 
 
 

Refs: [1](https://www.cse.psu.edu/~rtc12/CSE486/lecture16.pdf), [2](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)

# 2. Different Kinds of Transformation Related By Homography 
The transformations shown in the followings instances are all related to transformations between two planes.


## 2.1 Planar Surface And The Image Plane


<img src="images/homography_transformation_example1.jpg" alt="" />



## 2.2 Planar Surface Viewed By Two Cameras


<img src="images/homography1.svg" />   




## 2.3 Rotating Camera Around Its Axis of Projection, 

A rotating camera around its axis of projection, equivalent to consider that the points are on a plane at infinity (image taken from

<img src="images/homography_transformation_example3.jpg" />

# 3. Calculating Homography Matrix
For any point in the world the projection of the point on the camera plan would be:
<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%7B%5Cbegin%7Bbmatrix%7Du%5C%5Cv%5C%5Cw%5Cend%7Bbmatrix%7D%7D%3D%20K%5C%2C%20%7B%5Cbegin%7Bbmatrix%7D%20r_%7B11%7D%20%26%20r_%7B12%7D%20%26%20r_%7B13%7D%20%26t_%7B1%7D%20%5C%5C%20r_%7B21%7D%20%26%20r_%7B22%7D%20%26%20r_%7B23%7D%20%26t_%7B2%7D%20%5C%5C%20r_%7B31%7D%20%26%20r_%7B31%7D%20%26%20r_%7B31%7D%20%26t_%7B3%7D%20%5Cend%7Bbmatrix%7D%7D%7B%5Cbegin%7Bbmatrix%7DX_%7Bw%7D%5C%5CY_%7Bw%7D%5C%5CZ_%7Bw%7D%5C%5CW_%7Bw%7D%5Cend%7Bbmatrix%7D%7D%20%3DK%5C%2C%7B%5Cbegin%7Bbmatrix%7DR_%7Bw%7D%5E%7Bc%7D%20%26T_%7Bw%7D%5E%7Bc%7D%5Cend%7Bbmatrix%7D%7D%7B%5Cbegin%7Bbmatrix%7DX_%7Bw%7D%5C%5CY_%7Bw%7D%5C%5CZ_%7Bw%7D%5C%5CW_%7Bw%7D%5Cend%7Bbmatrix%7D%7D%3DP%7B%5Cbegin%7Bbmatrix%7DX_%7Bw%7D%5C%5CY_%7Bw%7D%5C%5CZ_%7Bw%7D%5C%5CW_%7Bw%7D%5Cend%7Bbmatrix%7D%7D%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle  {\begin{bmatrix}u\\v\\w\end{bmatrix}}=K\,{\begin{bmatrix}r_{11} & r_{12} & r_{13} &t_{1} \\ r_{21} & r_{22} & r_{23} &t_{2} \\ r_{31} & r_{31} & r_{31} &t_{3} \end{bmatrix}}{\begin{bmatrix}X_{w}\\Y_{w}\\Z_{w}\\W_{w}\end{bmatrix}} =K\,{\begin{bmatrix}R_{w}^{c} &T_{w}^{c}\end{bmatrix}}{\begin{bmatrix}X_{w}\\Y_{w}\\Z_{w}\\W_{w}\end{bmatrix}}=P{\begin{bmatrix}X_{w}\\Y_{w}\\Z_{w}\\W_{w}\end{bmatrix}}}" />

<br/>
<br/>

now if we put world reference frame on the plane such that X and Y axis lays on the plane:



<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7Du%5C%5Cv%5C%5Cw%5Cend%7Bbmatrix%7D%3D%20%5Cbegin%7Bbmatrix%7D%20p_%7B11%7D%20%26%20p_%7B12%7D%20%26%20p_%7B13%7D%20%26p_%7B14%7D%20%5C%5C%20p_%7B21%7D%20%26%20p_%7B22%7D%20%26%20p_%7B23%7D%20%26p_%7B24%7D%20%5C%5C%20p_%7B31%7D%20%26%20p_%7B32%7D%20%26%20p_%7B33%7D%20%26p_%7B34%7D%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7DX_%7Bw%7D%5C%5CY_%7Bw%7D%5C%5CZ_%7Bw%7D%5C%5CW_%7Bw%7D%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix}u\\v\\w\end{bmatrix}=\begin{bmatrix} p_{11} & p_{12} & p_{13} &p_{14}
\\p_{21} & p_{22} & p_{23} &p_{24} \\ p_{31} & p_{32} & p_{33} &p_{34} \end{bmatrix} \begin{bmatrix}X_{w}\\Y_{w}\\Z_{w}\\W_{w}\end{bmatrix}"/>


so all point the `Z` will be zero: 


<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7Du%5C%5Cv%5C%5Cw%5Cend%7Bbmatrix%7D%3D%20%5Cbegin%7Bbmatrix%7D%20p_%7B11%7D%20%26%20p_%7B12%7D%20%26%200%20%26p_%7B14%7D%20%5C%5C%20p_%7B21%7D%20%26%20p_%7B22%7D%20%26%200%20%26p_%7B24%7D%20%5C%5C%20p_%7B31%7D%20%26%20p_%7B32%7D%20%26%200%20%26p_%7B34%7D%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7DX_%7Bw%7D%5C%5CY_%7Bw%7D%5C%5C0%5C%5CW_%7Bw%7D%5Cend%7Bbmatrix%7D"
  alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix}u\\v\\w\end{bmatrix}=\begin{bmatrix} p_{11} & p_{12} & 0 &p_{14}
\\ p_{21} & p_{22} & 0 &p_{24} \\ p_{31} & p_{32} & 0 &p_{34} \end{bmatrix}\begin{bmatrix}X_{w}\\Y_{w}\\0\\W_{w}\end{bmatrix}" />


<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7Du%5C%5Cv%5C%5Cw%5Cend%7Bbmatrix%7D%3D%20%5Cbegin%7Bbmatrix%7D%20p_%7B11%7D%20%26%20p_%7B12%7D%20%26p_%7B14%7D%20%5C%5C%20p_%7B21%7D%20%26%20p_%7B22%7D%20%26p_%7B24%7D%20%5C%5C%20p_%7B31%7D%20%26%20p_%7B32%7D%20%26p_%7B34%7D%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7DX_%7Bw%7D%5C%5CY_%7Bw%7D%5C%5CW_%7Bw%7D%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix}u\\v\\w\end{bmatrix}= \begin{bmatrix} p_{11} & p_{12}  &p_{14} \\p_{21} & p_{22}  &p_{24} \\ p_{31} & p_{32}  &p_{34} \end{bmatrix} \begin{bmatrix}X_{w}\\Y_{w}\\W_{w}\end{bmatrix}" />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7Du%5C%5Cv%5C%5Cw%5Cend%7Bbmatrix%7D%3D%20%5Cbegin%7Bbmatrix%7D%20H_%7B11%7D%20%26%20H_%7B12%7D%20%26H_%7B13%7D%20%5C%5C%20H_%7B21%7D%20%26%20H_%7B22%7D%20%26H_%7B23%7D%20%5C%5C%20H_%7B31%7D%20%26%20H_%7B32%7D%20%26H_%7B33%7D%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7DX_%7Bw%7D%5C%5CY_%7Bw%7D%5C%5CW_%7Bw%7D%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix}u\\v\\w\end{bmatrix}=\begin{bmatrix} H_{11} & H_{12}  &H_{13} \\ H_{21} & H_{22}  &H_{23} \\ H_{31} & H_{32}  &H_{33} \end{bmatrix} \begin{bmatrix}X_{w}\\Y_{w}\\W_{w}\end{bmatrix}" />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7Du_1%5C%5Cv_1%5C%5Cw_1%5Cend%7Bbmatrix%7D%3D%20H_%7B1%7D%20%5Cbegin%7Bbmatrix%7DX_%7Bw%7D%5C%5CY_%7Bw%7D%5C%5CW_%7Bw%7D%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix}u_1\\v_1\\w_1\end{bmatrix}=
H_{1}
\begin{bmatrix}X_{w}\\Y_{w}\\W_{w}\end{bmatrix}" />


<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7Du_2%5C%5Cv_2%5C%5Cw_2%5Cend%7Bbmatrix%7D%3D%20H_%7B2%7DH%5E%7B-1%7D_%7B1%7D%20%5Cbegin%7Bbmatrix%7Du_%7B1%7D%5C%5Cv_%7B1%7D%5C%5Cw_%7B1%7D%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix}u_2\\v_2\\w_2\end{bmatrix}=H_{2}H^{-1}_{1}\begin{bmatrix}u_{1}\\v_{1}\\w_{1}\end{bmatrix}" />


<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7Du_2%5C%5Cv_2%5C%5Cw_2%5Cend%7Bbmatrix%7D%3D%20H%20%5Cbegin%7Bbmatrix%7Du_%7B1%7D%5C%5Cv_%7B1%7D%5C%5Cw_%7B1%7D%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix}u_2\\v_2\\w_2\end{bmatrix}= H \begin{bmatrix}u_{1}\\v_{1}\\w_{1}\end{bmatrix}" />


<br/>
<br/>




<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20%5Ctext%7Bif%20we%20set%3A%7D%20%5C%5C%20w_1%3D1%20%5C%5C%20w_2%3D1%20%5C%5C%20%5Ctext%7Bwhich%20gives%20us%3A%7D%20%5C%5C%20%5C%5C%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20u_1%20%3D%20%5Cfrac%7BH_%7B11%7D%20u_1%20&plus;%20H_%7B12%7D%20v_1%20&plus;%20H_%7B13%7D%7D%7BH_%7B31%7D%20u_1%20&plus;%20H_%7B32%7D%20v_1%20&plus;%20H_%7B33%7D%7D%20%5C%5C%20%5C%5C%20v_1%20%3D%20%5Cfrac%7BH_%7B21%7D%20u_1%20&plus;%20H_%7B22%7D%20v_1%20&plus;%20H_%7B23%7D%7D%7BH_%7B31%7D%20u_1%20&plus;%20H_%7B32%7D%20v_1%20&plus;%20H_%7B33%7D%7D%20%5Cend%7Bmatrix%7D%5Cright." alt="\\ \text{if we set:} \\ w_1=1 \\ w_2=1 \\ \text{which gives us:} \\ \\ \left\{\begin{matrix} u_1 =   \frac{H_{11} u_1 + H_{12} v_1 + H_{13}}{H_{31} u_1 + H_{32} v_1 + H_{33}}  \\ \\ v_1 =    \frac{H_{21} u_1 + H_{22} v_1 + H_{23}}{H_{31} u_1 + H_{32} v_1 + H_{33}} \end{matrix}\right." />


<br/>
<br/>

or:

<br/>
<br/>
<img src="https://latex.codecogs.com/svg.latex?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20xp_n%20%3D%20%5Cfrac%7BH_%7B11%7D%20x_n%20&plus;%20H_%7B12%7D%20y_n%20&plus;%20H_%7B13%7D%7D%7BH_%7B31%7D%20x_n%20&plus;%20H_%7B32%7D%20y_n%20&plus;%20H_%7B33%7D%7D%20%5C%5C%20%5C%5C%20yp_n%20%3D%20%5Cfrac%7BH_%7B21%7D%20x_n%20&plus;%20H_%7B22%7D%20y_n%20&plus;%20H_%7B23%7D%7D%7BH_%7B31%7D%20x_n%20&plus;%20H_%7B32%7D%20y_n%20&plus;%20H_%7B33%7D%7D%20%5Cend%7Bmatrix%7D%5Cright." alt="https://latex.codecogs.com/svg.image?\left\{\begin{matrix}
xp_n =   \frac{H_{11} x_n + H_{12} y_n + H_{13}}{H_{31} x_n + H_{32} y_n + H_{33}} 
\\ 
\\ 
yp_n =    \frac{H_{21} x_n + H_{22} y_n + H_{23}}{H_{31} x_n + H_{32} y_n + H_{33}}  
\end{matrix}\right." />


<br/>
<br/>

if we write the upper equation for 4 points and rewrite them we would get the following linear equation to solve:

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;\left(&space;\begin{array}{ccccccccc}&space;-x_1&space;&&space;-y_1&space;&&space;-1&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;x_1*xp_1&space;&&space;y_1*xp_1&space;&&space;xp_1\\0&space;&&space;0&space;&&space;0&space;&&space;-x_1&space;&&space;-y_1&space;&&space;-1&space;&&space;x_1*yp_1&space;&&space;y_1*yp_1&space;&&space;yp_1\\-x_2&space;&&space;-y_2&space;&&space;-1&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;x_2*xp_2&space;&&space;y_2*xp_2&space;&&space;xp_2\\0&space;&&space;0&space;&&space;0&space;&&space;-x_2&space;&&space;-y_2&space;&&space;-1&space;&&space;x_2*yp_2&space;&&space;y_2*yp_2&space;&&space;yp_2\\-x_3&space;&&space;-y_3&space;&&space;-1&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;x_3*xp_3&space;&&space;y_3*xp_3&space;&&space;xp_3\\0&space;&&space;0&space;&&space;0&space;&&space;-x_3&space;&&space;-y_3&space;&&space;-1&space;&&space;x_3*yp_3&space;&&space;y_3*yp_3&space;&&space;yp_3\\-x_4&space;&&space;-y_4&space;&&space;-1&space;&&space;0&space;&&space;0&space;&&space;0&space;&&space;x_4*xp_4&space;&&space;y_4*xp_4&space;&&space;xp_4\\0&space;&&space;0&space;&&space;0&space;&&space;-x_4&space;&&space;-y_4&space;&&space;-1&space;&&space;x_4*yp_4&space;&&space;y_4*yp_4&space;&&space;yp_4\\\end{array}&space;\right)&space;*H=0&space;\end{equation}" title="https://latex.codecogs.com/svg.image?\begin{equation} \left( \begin{array}{ccccccccc} -x_1 & -y_1 & -1 & 0 & 0 & 0 & x_1*xp_1 & y_1*xp_1 & xp_1\\0 & 0 & 0 & -x_1 & -y_1 & -1 & x_1*yp_1 & y_1*yp_1 & yp_1\\-x_2 & -y_2 & -1 & 0 & 0 & 0 & x_2*xp_2 & y_2*xp_2 & xp_2\\0 & 0 & 0 & -x_2 & -y_2 & -1 & x_2*yp_2 & y_2*yp_2 & yp_2\\-x_3 & -y_3 & -1 & 0 & 0 & 0 & x_3*xp_3 & y_3*xp_3 & xp_3\\0 & 0 & 0 & -x_3 & -y_3 & -1 & x_3*yp_3 & y_3*yp_3 & yp_3\\-x_4 & -y_4 & -1 & 0 & 0 & 0 & x_4*xp_4 & y_4*xp_4 & xp_4\\0 & 0 & 0 & -x_4 & -y_4 & -1 & x_4*yp_4 & y_4*yp_4 & yp_4\\\end{array} \right) *H=0 \end{equation}" />

<br/>
<br/>
<br/>



<img src="https://latex.codecogs.com/svg.image?H^{*}&space;\underset{H}{\mathrm{argmin}}=&space;\|AH\|^{2}" title="https://latex.codecogs.com/svg.image?H^{*} \underset{H}{\mathrm{argmin}}= \|AH\|^{2}" />




Singular-value Decomposition (SVD) of any given matrix <img src="https://latex.codecogs.com/svg.image?A_{M{\times}N}" title="https://latex.codecogs.com/svg.image?A_{M{\times}N}" />

<br/>
<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;\underbrace{\mathbf{A}}_{M&space;\times&space;N}&space;=&space;\underbrace{\mathbf{U}}_{M&space;\times&space;M}&space;\times&space;\underbrace{\mathbf{\Sigma}}_{M\times&space;N}&space;\times&space;\underbrace{\mathbf{V}^{\text{T}}}_{N&space;\times&space;N}&space;\end{equation}" title="https://latex.codecogs.com/svg.image?\begin{equation} \underbrace{\mathbf{A}}_{M \times N} = \underbrace{\mathbf{U}}_{M \times M} \times \underbrace{\mathbf{\Sigma}}_{M\times N} \times \underbrace{\mathbf{V}^{\text{T}}}_{N \times N} \end{equation}" />



<img src="https://latex.codecogs.com/svg.image?H^{*}" title="https://latex.codecogs.com/svg.image?H^{*}" /> is the last column of <img src="https://latex.codecogs.com/svg.image?V" title="https://latex.codecogs.com/svg.image?V" />



# OpenCV API

To find homography Matrix from 4 Corresponding Points:


```cpp
cv::Mat homographyMatrix= cv::getPerspectiveTransform(point_on_plane1,point_on_plane2);
cv::Mat H = cv::findHomography(  point_on_plane1,point_on_plane2,0 );

```


If you need to perform the Homography matrix transformation on points:
```cpp
cv::perspectiveTransform 
```

If you want to transform an image using perspective transformation, use:


```cpp
cv::warpPerspective
```

The function `cv::warpPerspective` transforms the source image using the specified matrix:


<img src="https://latex.codecogs.com/svg.image?\texttt{dst}&space;(x,y)&space;=&space;\texttt{src}&space;\left&space;(&space;\frac{M_{11}&space;x&space;&plus;&space;M_{12}&space;y&space;&plus;&space;M_{13}}{M_{31}&space;x&space;&plus;&space;M_{32}&space;y&space;&plus;&space;M_{33}}&space;,&space;\frac{M_{21}&space;x&space;&plus;&space;M_{22}&space;y&space;&plus;&space;M_{23}}{M_{31}&space;x&space;&plus;&space;M_{32}&space;y&space;&plus;&space;M_{33}}&space;\right&space;)" title="https://latex.codecogs.com/svg.image?\texttt{dst} (x,y) = \texttt{src} \left ( \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} , \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \right )" />

<br/>
<br/>

[Apply homography on image](../src/apply_homography_on_image.cpp), 
<br/>
[Finding homography matrix from 4 corresponding points](../src/finding_homography_matrix_4_corresponding_points.cpp)
<br/>
[Finding homography Matrix between two images using keypoints and RANSAC](../src/finding_homography_using_keypoints_RANSAC.cpp)




# Decompose Homography Matrix
Refs: [1](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga7f60bdff78833d1e3fd6d9d0fd538d92)


