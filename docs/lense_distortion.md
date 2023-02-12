# 1. Camera Lenses



## 1.1 Tilt Shift Lenses

Refs: [1](https://en.wikipedia.org/wiki/Tilt%E2%80%93shift_photography) 


## 1.2 Fisheye Lenses


## 1.3 Rectilinear Lenses
A rectilinear lens produces images in which straight features, such as the edges of building walls, appear with straight lines rather than curves. In other words, it is a lens that exhibits little or no pincushion or barrel distortion.
As an object gets closer to the edge of the frame, the rectilinear perspective will make it appear as though it is being stretched and enlarged. Lenses of this kind are frequently employed to produce forced perspective effects.


# 2. Lenses Distortion

## 2.1 Radial Distortions

The lens isn't a perfect pinhole, which has the side consequence of causing symmetric radial distortion. Outside of the perspective center, light enters the lens and bends toward the image plane. The best way to understand symmetric radial distortion could be to imagine that the concavity or convexity of the lens is being used to map the image plane. Because all light passes through a single point in a pinhole camera, there would be no distortion.


Because it solely models distortion as a function of distance from the center of the image plane, this distortion is described as being symmetric. Radial distortion only has a geometric impact in the radial direction,




## 2.1 Pincushion Distortion

<img src="images/Pincushion_distortion.svg" height="250" width="250" />


## 2.2 Barrel Distortion
<img src="images/Barrel_distortion.svg" height="250" width="250" />


## 2.3 Mustache Distortion
<img src="images/Mustache_distortion.svg" height="250" width="250"/>




## 2.4 Tangential Distortions
Decentering distortion is a result of the lens assembly not being centered over and parallel to the image plane as the main reason.


|   |   |
|---|---|
|<img src="images/tangential_distortions.svg" height="250" width="250"/>   | <img src="images/radial-and-tangential-distortion.png" height="270" width="350"/>   |
|[image courtesy](https://www.tangramvision.com/blog/camera-modeling-exploring-distortion-and-distortion-models-part-i)   |      [image courtesy](https://www.researchgate.net/publication/260728375_Laboratory_calibration_of_star_sensor_with_installation_error_using_a_nonlinear_distortion_model)  |





# 3. Lens Chromatic Aberration


<img src="images/lens_chromatic_aberration.png" height="250" width="250"/>


# 4. Distortion Models

## 4.1  Brown-Conrady

The Brown-Conrady model corrects both for radial distortion and for tangential distortion as a series of higher order polynomial. In the following all points are n the image plane with Cartesian coordinate and not pixel based coordinate.

### 4.1.1 Radial Distortion
<img src="images/radial_distortion_image_plane.svg" height="250" width="250" />

<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?\delta&space;r=&space;k_1&space;r^3&space;&plus;&space;k_2&space;r^5&space;&plus;&space;k_3&space;r^7&space;&plus;&space;...&space;&plus;&space;k_n&space;r^{n&plus;2}" title="https://latex.codecogs.com/svg.image?\delta r= k_1 r^3 + k_2 r^5 + k_3 r^7 + ... + k_n r^{n+2}" />


<br/>
<br/>



<img src="https://latex.codecogs.com/svg.image?\delta&space;x_r&space;=&space;\sin(\psi)&space;\delta&space;r&space;=&space;\frac{x}{r}&space;(k_1r^3&space;&plus;&space;k_2r^5&space;&plus;&space;k_3r^7)" title="https://latex.codecogs.com/svg.image?\delta x_r = \sin(\psi) \delta r = \frac{x}{r} (k_1r^3 + k_2r^5 + k_3r^7)" />

<br/>
<img src="https://latex.codecogs.com/svg.image?\delta&space;y_r&space;=&space;\cos(\psi)&space;\delta&space;r&space;=&space;\frac{y}{r}&space;(k_1r^3&space;&plus;&space;k_2r^5&space;&plus;&space;k_3r^7)" title="https://latex.codecogs.com/svg.image?\delta y_r = \cos(\psi) \delta r = \frac{y}{r} (k_1r^3 + k_2r^5 + k_3r^7)" />



<img src="https://latex.codecogs.com/svg.image?(x_{\mathrm&space;{d}&space;},\&space;y_{\mathrm&space;{d}&space;})" title="https://latex.codecogs.com/svg.image?(x_{\mathrm {d} },\ y_{\mathrm {d} })" /> is the distorted image point

<br/>

<img src="https://latex.codecogs.com/svg.image?(x_{\mathrm&space;{u}&space;},\&space;y_{\mathrm&space;{u}&space;})" title="https://latex.codecogs.com/svg.image?(x_{\mathrm {u} },\ y_{\mathrm {u} })" /> is the undistorted image point

<br/>

<img src="https://latex.codecogs.com/svg.image?(x_{\mathrm&space;{c}&space;},\&space;y_{\mathrm&space;{c}&space;})" title="https://latex.codecogs.com/svg.image?(x_{\mathrm {c} },\ y_{\mathrm {c} })" /> is the distortion center

<br/>

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?r={\displaystyle&space;{\sqrt&space;{(x_{\mathrm&space;{d}&space;}-x_{\mathrm&space;{c}&space;})^{2}&plus;(y_{\mathrm&space;{d}&space;}-y_{\mathrm&space;{c}&space;})^{2}}}}" title="https://latex.codecogs.com/svg.image?r={\displaystyle {\sqrt {(x_{\mathrm {d} }-x_{\mathrm {c} })^{2}+(y_{\mathrm {d} }-y_{\mathrm {c} })^{2}}}}" />



<br/>
<br/>

### 4.1.2 Tangential distortion:
<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?\delta&space;x_t&space;=&space;p_1(r^2&space;&plus;&space;2x^2)&space;&plus;&space;2p_2xy" title="https://latex.codecogs.com/svg.image?\delta x_t = p_1(r^2 + 2x^2) + 2p_2xy" />
<br/>

<img src="https://latex.codecogs.com/svg.image?\delta&space;y_t&space;=&space;p_2(r^2&space;&plus;&space;2y^2)&space;&plus;&space;2p_1xy" title="https://latex.codecogs.com/svg.image?\delta y_t = p_2(r^2 + 2y^2) + 2p_1xy" />

<br/>
<br/>

### 4.1.3  Both Distortion Together

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?{\displaystyle&space;{\begin{alignedat}{}x_{\mathrm&space;{u}&space;}=x_{\mathrm&space;{d}&space;}&&plus;(x_{\mathrm&space;{d}&space;}-x_{\mathrm&space;{c}&space;})(K_{1}r^{2}&plus;K_{2}r^{4}&plus;\cdots&space;)&plus;(P_{1}(r^{2}&plus;2(x_{\mathrm&space;{d}&space;}-x_{\mathrm&space;{c}&space;})^{2})\\&&plus;2P_{2}(x_{\mathrm&space;{d}&space;}-x_{\mathrm&space;{c}&space;})(y_{\mathrm&space;{d}&space;}-y_{\mathrm&space;{c}&space;}))(1&plus;P_{3}r^{2}&plus;P_{4}r^{4}\cdots&space;)\\y_{\mathrm&space;{u}&space;}=y_{\mathrm&space;{d}&space;}&&plus;(y_{\mathrm&space;{d}&space;}-y_{\mathrm&space;{c}&space;})(K_{1}r^{2}&plus;K_{2}r^{4}&plus;\cdots&space;)&plus;(2P_{1}(x_{\mathrm&space;{d}&space;}-x_{\mathrm&space;{c}&space;})(y_{\mathrm&space;{d}&space;}-y_{\mathrm&space;{c}&space;})\\&&plus;P_{2}(r^{2}&plus;2(y_{\mathrm&space;{d}&space;}-y_{\mathrm&space;{c}&space;})^{2}))(1&plus;P_{3}r^{2}&plus;P_{4}r^{4}\cdots&space;),\end{alignedat}}}" title="https://latex.codecogs.com/svg.image?{\displaystyle {\begin{alignedat}{}x_{\mathrm {u} }=x_{\mathrm {d} }&+(x_{\mathrm {d} }-x_{\mathrm {c} })(K_{1}r^{2}+K_{2}r^{4}+\cdots )+(P_{1}(r^{2}+2(x_{\mathrm {d} }-x_{\mathrm {c} })^{2})\\&+2P_{2}(x_{\mathrm {d} }-x_{\mathrm {c} })(y_{\mathrm {d} }-y_{\mathrm {c} }))(1+P_{3}r^{2}+P_{4}r^{4}\cdots )\\y_{\mathrm {u} }=y_{\mathrm {d} }&+(y_{\mathrm {d} }-y_{\mathrm {c} })(K_{1}r^{2}+K_{2}r^{4}+\cdots )+(2P_{1}(x_{\mathrm {d} }-x_{\mathrm {c} })(y_{\mathrm {d} }-y_{\mathrm {c} })\\&+P_{2}(r^{2}+2(y_{\mathrm {d} }-y_{\mathrm {c} })^{2}))(1+P_{3}r^{2}+P_{4}r^{4}\cdots ),\end{alignedat}}}" />


<br/>

<img src="https://latex.codecogs.com/svg.image?K_{n}&space;" title="https://latex.codecogs.com/svg.image?K_{n} " /> is the <img src="https://latex.codecogs.com/svg.image?n^{\mathrm&space;{th}&space;}" title="https://latex.codecogs.com/svg.image?n^{\mathrm {th} }" /> radial distortion coefficient.

<br/>


<img src="https://latex.codecogs.com/svg.image?P_{n}" title="https://latex.codecogs.com/svg.image?P_{n}" /> is the <img src="https://latex.codecogs.com/svg.image?n^{\mathrm&space;{th}&space;}" title="https://latex.codecogs.com/svg.image?n^{\mathrm {th} }" /> tangential distortion coefficient.

In practice, only the <img src="https://latex.codecogs.com/svg.image?k_1" title="https://latex.codecogs.com/svg.image?k_1" />, <img src="https://latex.codecogs.com/svg.image?k_2" title="https://latex.codecogs.com/svg.image?k_1" /> and <img src="https://latex.codecogs.com/svg.image?k_3" title="https://latex.codecogs.com/svg.image?k_3" /> and <img src="https://latex.codecogs.com/svg.image?p_1" title="https://latex.codecogs.com/svg.image?p_1" />, <img src="https://latex.codecogs.com/svg.image?p_2" title="https://latex.codecogs.com/svg.image?p_2" /> 
terms are typically used



- Barrel distortion typically will have a negative term for <img src="https://latex.codecogs.com/svg.image?K_{1}&space;" title="https://latex.codecogs.com/svg.image?K_{1} " /> 
- Pincushion distortion will have a positive value for <img src="https://latex.codecogs.com/svg.image?K_{1}&space;" title="https://latex.codecogs.com/svg.image?K_{1} " /> . 
- Moustache distortion will have a non-monotonic radial geometric series where for some <img src="https://latex.codecogs.com/svg.image?r" title="https://latex.codecogs.com/svg.image?r" /> the sequence will change sign.



## 4.2  Division Model


provides a more accurate approximation than Brown-Conrady's even-order polynomial model. For radial distortion, this division model is often preferred over the Brown–Conrady model, as it requires fewer terms to more accurately describe severe distortion


<img src="https://latex.codecogs.com/svg.image?{\begin{aligned}x_{\mathrm&space;{u}&space;}&=x_{\mathrm&space;{c}&space;}&plus;{\frac&space;{x_{\mathrm&space;{d}&space;}-x_{\mathrm&space;{c}&space;}}{1&plus;K_{1}r^{2}&plus;K_{2}r^{4}&plus;\cdots&space;}}\\y_{\mathrm&space;{u}&space;}&=y_{\mathrm&space;{c}&space;}&plus;{\frac&space;{y_{\mathrm&space;{d}&space;}-y_{\mathrm&space;{c}&space;}}{1&plus;K_{1}r^{2}&plus;K_{2}r^{4}&plus;\cdots&space;}},\end{aligned}}" title="https://latex.codecogs.com/svg.image?{\begin{aligned}x_{\mathrm {u} }&=x_{\mathrm {c} }+{\frac {x_{\mathrm {d} }-x_{\mathrm {c} }}{1+K_{1}r^{2}+K_{2}r^{4}+\cdots }}\\y_{\mathrm {u} }&=y_{\mathrm {c} }+{\frac {y_{\mathrm {d} }-y_{\mathrm {c} }}{1+K_{1}r^{2}+K_{2}r^{4}+\cdots }},\end{aligned}}" />


Refs: [1](https://www.tangramvision.com/blog/camera-modeling-exploring-distortion-and-distortion-models-part-i),
[2](https://www.robots.ox.ac.uk/~vgg/publications/2001/Fitzgibbon01b/fitzgibbon01b.pdf), [3](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf), [4](https://ori.codes/artificial-intelligence/camera-calibration/camera-distortions/), [5](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)



# 4.3 OpenCV Lens Distortion Model





<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7D%20x%5C%5C%20y%5C%5C%20z%20%5Cend%7Bbmatrix%7D%3D%20R%5Cbegin%7Bbmatrix%7D%20X%5C%5C%20Y%5C%5C%20Z%20%5Cend%7Bbmatrix%7D&plus;t
" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix}
x\\ 
y\\ 
z
\end{bmatrix}=
R\begin{bmatrix}
X\\ 
Y\\ 
Z
\end{bmatrix}+t" />



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



# Image Resolution and Distortion Coefficient
The distortion coefficients <img src="https://latex.codecogs.com/svg.image?k_1,k_2,p_1,p_2,k_3,k_4,k_5,k_6" title="https://latex.codecogs.com/svg.image?k_1,k_2,p_1,p_2,k_3,k_4,k_5,k_6" /> 
do not depend on the scene viewed and they remain the **same** regardless of image resolution. If, for example, a camera has been calibrated on images of 320 x 240 resolution, absolutely the same distortion coefficients can be used for 640 x 480 images from the same camera

However, <img src="https://latex.codecogs.com/svg.image?f_x" title="https://latex.codecogs.com/svg.image?f_x" />, <img src="https://latex.codecogs.com/svg.image?f_y" title="https://latex.codecogs.com/svg.image?f_y" />, <img src="https://latex.codecogs.com/svg.image?c_x" title="https://latex.codecogs.com/svg.image?c_x" />, and <img src="https://latex.codecogs.com/svg.image?c_y" title="https://latex.codecogs.com/svg.image?c_y" /> need to be scaled appropriately.


In OpenCV `cv::undistort`does the followings:
For each pixel of the destination lens-corrected image do:

- Convert the pixel coordinates `(u_dst, v_dst)` to normalized coordinates `(x_d, y_d`) using the inverse of the calibration matrix `K`,
- Apply the lens-distortion model, as displayed above, to obtain the distorted normalized coordinates `(x_u, y_u)`,
- Convert `(x_u, y_u)` to distorted pixel coordinates using the calibration matrix `K`,
- Use the interpolation method of your choice to find the intensity/depth associated with the pixel coordinates `(u_src, v_src)` in the source image, and assign this intensity/depth to the current destination pixel.


Refs: [1](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#camera-calibration-and-3d-reconstruction), [2](https://stackoverflow.com/questions/21958521/understanding-of-opencv-undistortion)



