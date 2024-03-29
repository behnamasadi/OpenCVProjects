## Taylor of a function around point a:
First lets review some math:
<br/>


<img src="https://latex.codecogs.com/svg.latex?f%28x%29%3Df%28a%29&plus;f%27%28a%29%28x-a%29&plus;%5Cfrac%7B1%7D%7B2%7D%28x-a%29f%27%27%28a%29%28x-a%29&plus;%5Cfrac%7B1%7D%7B6%7Df%27%27%27%28a%29%28x-a%29%5E3&plus;..." alt="https://latex.codecogs.com/svg.latex?f(x)=f(a)+f'(a)(x-a)+\frac{1}{2}(x-a)f''(a)(x-a)+\frac{1}{6}f'''(a)(x-a)^3+..." />


If we substitute

<img src="https://latex.codecogs.com/svg.latex?x%20%3D%20a&plus;%5CDelta%20x" alt="https://latex.codecogs.com/svg.latex?x = a+\Delta x" />

<br/>
<br/>


<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20f%28x%29%20%3D%20f%28a&plus;%5CDelta%20x%29%20%3D%20f%28a%29%20&plus;%20f%5E%7B%27%7D%28a%29%28a%20&plus;%20%5CDelta%20x-a%29%20&plus;%20%5Cfrac%7Bf%5E%7B%27%27%7D%28a%29%28a%20&plus;%20%5CDelta%20x-a%29%5E2%7D%7B2%21%7D%20&plus;%20%5Ctext%7B...%7D%20%5C%5C%20%3D%20f%28a%29%20&plus;f%5E%7B%27%7D%28a%29%5CDelta%20x&plus;%20%5Cfrac%7Bf%5E%7B%27%27%7D%28a%29%7D%7B2%21%7D%28%5CDelta%20x%29%5E2%20&plus;%20%5Ctext%7B...%7D" alt="https://latex.codecogs.com/svg.latex?\\
f(x) = f(a+\Delta x) = f(a) + f^{'}(a)(a + \Delta x-a) + \frac{f^{''}(a)(a + \Delta x-a)^2}{2!} + \text{...}
\\
= f(a) +f^{'}(a)\Delta x+ \frac{f^{''}(a)}{2!}(\Delta x)^2 + \text{...}" />



## Multivariable Taylor 


<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20p_2%28x%2Cy%29%20%26%3D%20f%28a%2Cb%29%20&plus;%20D%20f%28a%2Cb%29_%7B1%5Ctimes2%7D%20%5Cleft%5B%20%5Cbegin%7Barray%7D%7Bc%7D%20x-a%20%5C%5C%20y-b%20%5Cend%7Barray%7D%20%5Cright%5D%20&plus;%20%5Cfrac%7Ba%7D%7B2%7D%20%5Cleft%5B%20%5Cbegin%7Barray%7D%7Bcc%7D%20x-a%20%26y-b%20%5Cend%7Barray%7D%20%5Cright%5D%20Hf%28a%2Cb%29_%7B2%5Ctimes2%7D%20%5Cleft%5B%20%5Cbegin%7Barray%7D%7Bc%7D%20x-a%20%5C%5C%20y-b%20%5Cend%7Barray%7D%20%5Cright%5D%20%5Cend%7Balign*%7D" alt="https://latex.codecogs.com/svg.latex?\begin{align*}   p_2(x,y) &= f(a,b) +  D f(a,b)_{1\times2}   \left[     \begin{array}{c}       x-a \\ y-b     \end{array}   \right]   + \frac{a}{2}   \left[     \begin{array}{cc}       x-a &y-b      \end{array}   \right]   Hf(a,b)_{2\times2}    \left[     \begin{array}{c}       x-a \\ y-b     \end{array}   \right]
\end{align*}" />


<br/>
<br/>

## Optical Flow

Between two image frames which are taken at times <img src="https://latex.codecogs.com/svg.latex?t" alt="https://latex.codecogs.com/svg.latex?t" /> and  <img src="https://latex.codecogs.com/svg.latex?t&plus;%5CDelta%20t" alt="https://latex.codecogs.com/svg.latex?t+\Delta t" /> at every position following brightness constancy constraint:

<img src="https://latex.codecogs.com/svg.latex?I%28x%2Cy%2Ct%29%20%3D%20I%28x&plus;%5CDelta%20x%2C%20y%20&plus;%20%5CDelta%20y%2C%20t%20&plus;%20%5CDelta%20t%29" alt="https://latex.codecogs.com/svg.latex?I(x,y,t) = I(x+\Delta x, y + \Delta y, t + \Delta t)" />



Assuming the movement to be small, Taylor series of:

<img src="https://latex.codecogs.com/svg.latex?I%28x&plus;%5CDelta%20x%2C%20y%20&plus;%20%5CDelta%20y%2C%20t%20&plus;%20%5CDelta%20t%29" alt="https://latex.codecogs.com/svg.latex?I(x+\Delta x, y + \Delta y, t + \Delta t)" /> 
<br/>
<br/>

is:

<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20I%28x&plus;%5CDelta%20x%2Cy&plus;%5CDelta%20y%2Ct&plus;%5CDelta%20t%29%3DI%28x%2Cy%2Ct%29&plus;%7B%5Cfrac%20%7B%5Cpartial%20I%7D%7B%5Cpartial%20x%7D%7D%5C%2C%5CDelta%20x&plus;%7B%5Cfrac%20%7B%5Cpartial%20I%7D%7B%5Cpartial%20y%7D%7D%5C%2C%5CDelta%20y&plus;%7B%5Cfrac%20%7B%5Cpartial%20I%7D%7B%5Cpartial%20t%7D%7D%5C%2C%5CDelta%20t&plus;%7B%7D%7D%20%5Ctext%7Bhigher-order%20terms%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle I(x+\Delta x,y+\Delta y,t+\Delta t)=I(x,y,t)+{\frac {\partial I}{\partial x}}\,\Delta x+{\frac {\partial I}{\partial y}}\,\Delta y+{\frac {\partial I}{\partial t}}\,\Delta t+{}} \text{higher-order terms}" />


<br/>
<br/>




<img src="https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20I%7D%7B%5Cpartial%20x%7D%5CDelta%20x&plus;%5Cfrac%7B%5Cpartial%20I%7D%7B%5Cpartial%20y%7D%5CDelta%20y&plus;%5Cfrac%7B%5Cpartial%20I%7D%7B%5Cpartial%20t%7D%5CDelta%20t%20%3D%200" alt="https://latex.codecogs.com/svg.latex?\frac{\partial I}{\partial x}\Delta x+\frac{\partial I}{\partial y}\Delta y+\frac{\partial I}{\partial t}\Delta t = 0" />

<br/>
<br/>

dividing by <img src="https://latex.codecogs.com/svg.latex?%5CDelta%20t" alt="https://latex.codecogs.com/svg.latex?\Delta t" />:


<br/>
<br/>



<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%7B%5Cfrac%20%7B%5Cpartial%20I%7D%7B%5Cpartial%20x%7D%7D%7B%5Cfrac%20%7B%5CDelta%20x%7D%7B%5CDelta%20t%7D%7D&plus;%7B%5Cfrac%20%7B%5Cpartial%20I%7D%7B%5Cpartial%20y%7D%7D%7B%5Cfrac%20%7B%5CDelta%20y%7D%7B%5CDelta%20t%7D%7D&plus;%7B%5Cfrac%20%7B%5Cpartial%20I%7D%7B%5Cpartial%20t%7D%7D%7B%5Cfrac%20%7B%5CDelta%20t%7D%7B%5CDelta%20t%7D%7D%3D0%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\frac {\partial I}{\partial x}}{\frac {\Delta x}{\Delta t}}+{\frac {\partial I}{\partial y}}{\frac {\Delta y}{\Delta t}}+{\frac {\partial I}{\partial t}}{\frac {\Delta t}{\Delta t}}=0}" />


<br/>
<br/>


<img src="https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20I%7D%7B%5Cpartial%20x%7DV_x&plus;%5Cfrac%7B%5Cpartial%20I%7D%7B%5Cpartial%20y%7DV_y&plus;%5Cfrac%7B%5Cpartial%20I%7D%7B%5Cpartial%20t%7D%20%3D%200" alt="https://latex.codecogs.com/svg.latex?\frac{\partial I}{\partial x}V_x+\frac{\partial I}{\partial y}V_y+\frac{\partial I}{\partial t} = 0" />

<br/>
<br/>





<img src="https://latex.codecogs.com/svg.latex?I_xV_x&plus;I_yV_y%3D-I_t" alt="https://latex.codecogs.com/svg.latex?I_xV_x+I_yV_y=-I_t" />


<br/>
<br/>


<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%5Cnabla%20I%5Ccdot%20%7B%5Cvec%20%7BV%7D%7D%3D-I_%7Bt%7D%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \nabla I\cdot {\vec {V}}=-I_{t}}" />

<br/>
<br/>

## Aperture Problem

This equation: 
<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%5Cnabla%20I%5Ccdot%20%7B%5Cvec%20%7BV%7D%7D%3D-I_%7Bt%7D%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \nabla I\cdot {\vec {V}}=-I_{t}}" />


is known as the aperture problem of the optical flow, has two unknowns.  To find the optical flow another set of equations is needed, given by some additional constraint. All optical flow methods introduce additional conditions for estimating the actual flow.

## Lucas-Kanade Method

Lucas-Kanade method takes a `3x3` patch around the point. So all the 9 points have the same motion. So now our problem becomes solving 9 equations with two unknown variables which is over-determined.


<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%7B%5Cbegin%7Baligned%7DI_%7Bx%7D%28q_%7B1%7D%29V_%7Bx%7D&plus;I_%7By%7D%28q_%7B1%7D%29V_%7By%7D%26%3D-I_%7Bt%7D%28q_%7B1%7D%29%5C%5CI_%7Bx%7D%28q_%7B2%7D%29V_%7Bx%7D&plus;I_%7By%7D%28q_%7B2%7D%29V_%7By%7D%26%3D-I_%7Bt%7D%28q_%7B2%7D%29%5C%5C%26%5C%3B%5C%20%5Cvdots%20%5C%5CI_%7Bx%7D%28q_%7Bn%7D%29V_%7Bx%7D&plus;I_%7By%7D%28q_%7Bn%7D%29V_%7By%7D%26%3D-I_%7Bt%7D%28q_%7Bn%7D%29%5Cend%7Baligned%7D%7D%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{aligned}I_{x}(q_{1})V_{x}+I_{y}(q_{1})V_{y}&=-I_{t}(q_{1})\\I_{x}(q_{2})V_{x}+I_{y}(q_{2})V_{y}&=-I_{t}(q_{2})\\&\;\ \vdots \\I_{x}(q_{n})V_{x}+I_{y}(q_{n})V_{y}&=-I_{t}(q_{n})\end{aligned}}}" />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?Av%3Db" alt="https://latex.codecogs.com/svg.latex?Av=b" />


<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20A%3D%7B%5Cbegin%7Bbmatrix%7DI_%7Bx%7D%28q_%7B1%7D%29%26I_%7By%7D%28q_%7B1%7D%29%5C%5C%5B10pt%5DI_%7Bx%7D%28q_%7B2%7D%29%26I_%7By%7D%28q_%7B2%7D%29%5C%5C%5B10pt%5D%5Cvdots%20%26%5Cvdots%20%5C%5C%5B10pt%5DI_%7Bx%7D%28q_%7Bn%7D%29%26I_%7By%7D%28q_%7Bn%7D%29%5Cend%7Bbmatrix%7D%7D%5Cquad%20%5Cquad%20%5Cquad%20v%3D%7B%5Cbegin%7Bbmatrix%7DV_%7Bx%7D%5C%5C%5B10pt%5DV_%7By%7D%5Cend%7Bbmatrix%7D%7D%5Cquad%20%5Cquad%20%5Cquad%20b%3D%7B%5Cbegin%7Bbmatrix%7D-I_%7Bt%7D%28q_%7B1%7D%29%5C%5C%5B10pt%5D-I_%7Bt%7D%28q_%7B2%7D%29%5C%5C%5B10pt%5D%5Cvdots%20%5C%5C%5B10pt%5D-I_%7Bt%7D%28q_%7Bn%7D%29%5Cend%7Bbmatrix%7D%7D%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle A={\begin{bmatrix}I_{x}(q_{1})&I_{y}(q_{1})\\[10pt]I_{x}(q_{2})&I_{y}(q_{2})\\[10pt]\vdots &\vdots \\[10pt]I_{x}(q_{n})&I_{y}(q_{n})\end{bmatrix}}\quad \quad \quad v={\begin{bmatrix}V_{x}\\[10pt]V_{y}\end{bmatrix}}\quad \quad \quad b={\begin{bmatrix}-I_{t}(q_{1})\\[10pt]-I_{t}(q_{2})\\[10pt]\vdots \\[10pt]-I_{t}(q_{n})\end{bmatrix}}}" />


<br/>
<br/>


to solve this problem:

<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20%7B%5Cdisplaystyle%20A%5E%7BT%7DAv%3DA%5E%7BT%7Db%7D%5C%5C%20%5C%5C%20%7B%5Cdisplaystyle%20%5Cmathrm%20%7Bv%7D%20%3D%28A%5E%7BT%7DA%29%5E%7B-1%7DA%5E%7BT%7Db%7D" alt="https://latex.codecogs.com/svg.latex?\\
{\displaystyle A^{T}Av=A^{T}b}\\
\\
{\displaystyle \mathrm {v} =(A^{T}A)^{-1}A^{T}b}" />



That is, it computes

<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%7B%5Cbegin%7Bbmatrix%7DV_%7Bx%7D%5C%5C%5B10pt%5DV_%7By%7D%5Cend%7Bbmatrix%7D%7D%3D%7B%5Cbegin%7Bbmatrix%7D%5Csum%20_%7Bi%7DI_%7Bx%7D%28q_%7Bi%7D%29%5E%7B2%7D%26%5Csum%20_%7Bi%7DI_%7Bx%7D%28q_%7Bi%7D%29I_%7By%7D%28q_%7Bi%7D%29%5C%5C%5B10pt%5D%5Csum%20_%7Bi%7DI_%7By%7D%28q_%7Bi%7D%29I_%7Bx%7D%28q_%7Bi%7D%29%26%5Csum%20_%7Bi%7DI_%7By%7D%28q_%7Bi%7D%29%5E%7B2%7D%5Cend%7Bbmatrix%7D%7D%5E%7B-1%7D%7B%5Cbegin%7Bbmatrix%7D-%5Csum%20_%7Bi%7DI_%7Bx%7D%28q_%7Bi%7D%29I_%7Bt%7D%28q_%7Bi%7D%29%5C%5C%5B10pt%5D-%5Csum%20_%7Bi%7DI_%7By%7D%28q_%7Bi%7D%29I_%7Bt%7D%28q_%7Bi%7D%29%5Cend%7Bbmatrix%7D%7D%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{bmatrix}V_{x}\\[10pt]V_{y}\end{bmatrix}}={\begin{bmatrix}\sum _{i}I_{x}(q_{i})^{2}&\sum _{i}I_{x}(q_{i})I_{y}(q_{i})\\[10pt]\sum _{i}I_{y}(q_{i})I_{x}(q_{i})&\sum _{i}I_{y}(q_{i})^{2}\end{bmatrix}}^{-1}{\begin{bmatrix}-\sum _{i}I_{x}(q_{i})I_{t}(q_{i})\\[10pt]-\sum _{i}I_{y}(q_{i})I_{t}(q_{i})\end{bmatrix}}}" />

<br/>
<br/>

# OpenCV OpticalFlow API

## SparseOpticalFlow 
Abstract base class for calculating sparse optical-flow.

## SparsePyrLKOpticalFlow

`buildOpticalFlowPyramid`: Constructs the image pyramid which can be passed to `calcOpticalFlowPyrLK`

<img src="images/Pyramid_Level_0.png" />
<br/>  
<img src="images/Pyramid_Level_1.png" />  
<br/>  
<img src="images/Pyramid_Level_2.png" />  
<br/>  
<img src="images/Pyramid_Level_3.png" />  


```
ret, pyramid = cv2.buildOpticalFlowPyramid(
    img, winSize, maxLevel, withDerivatives=False, pyrBorder=cv2.BORDER_REFLECT)
```

`withDerivatives`:set to precompute gradients for the every pyramid level. If pyramid is constructed without the gradients then calcOpticalFlowPyrLK will calculate them internally.





The class can calculate an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.


- `nextPts`: output vector of 2D points (floating-point coordinates) containing the calculated new positions of input features in the second image;
when OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.


- `status`: output status vector (of unsigned chars); each element of the vector is set to 1
if the flow for the corresponding features has been found, otherwise, it is set to 0.


- `err`: output vector of errors; each element of the vector is set to an error for the corresponding feature,
type of the error measure can be set in flags parameter; if the flow wasn't found then the error is not defined (use the status parameter to find such cases).

- `maxLevel`: 0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on;
if pyramids are passed to input then algorithm will use as many levels as pyramids have but no more than maxLevel.


Refs: [1](https://github.com/npinto/opencv/blob/master/samples/python2/lk_track.py), [2](https://github.com/npinto/opencv/blob/master/samples/python2/lk_homography.py)

## DenseOpticalFlow
Abstract base class for calculating dense optical-flow.

### FarnebackOpticalFlow 
Farnebäck Optical Flow (`cv2.FarnebackOpticalFlow_create()`):

Algorithm developed by Gunnar Farnebäck.
Based on polynomial expansion to approximate neighboring pixel displacements.
Typically faster than more recent methods, but may not be as accurate for complex scenes or large displacements.
Good compromise between accuracy and speed for real-time applications.

### DualTVL1OpticalFlow

Dual TV L1 Optical Flow (`cv2.DualTVL1OpticalFlow_create()`):

Based on the Total Variation (TV) regularization, which tends to produce cleaner flow fields.
Often provides smoother and more coherent flow than Farnebäck, but at the cost of computational efficiency.
Can handle larger displacements than Farnebäck.
Has several parameters that can be tweaked, which may require tuning for specific applications.

### DenseOpticalFlow
DeepFlow (`cv2.optflow.createOptFlow_DeepFlow()`):

Uses smoothness assumptions as in classical methods but also takes advantage of descriptor matching, which can be seen as a regularized, dense version of sparse matching techniques.
Generally provides high-quality results but can be slower.

### SimpleFlow
SimpleFlow (`cv2.optflow.createOptFlow_SimpleFlow()`):

Designed to be a more straightforward, non-regularized version of the DeepFlow algorithm.
It might be faster than DeepFlow but could provide less accurate results due to the absence of regularization.

### PCA-Flow, PCA-Layers, and SPARSE-PCA

PCA-Flow, PCA-Layers, and SPARSE-PCA (`cv2.optflow.createOptFlow_PCAFlow()` and related methods):

Uses principal component analysis to compute optical flow.
Tends to be experimental and might be better suited for specific scenes or scenarios.

### DIS Optical Flow
DIS Optical Flow (`cv2.optflow.createOptFlow_DIS()`):

Stands for Dense Inverse Search.
Designed to be faster and more efficient, especially with parallel processing capabilities.
Provides competitive accuracy with other state-of-the-art methods but with faster performance.



Each of these algorithms has its strengths and weaknesses, and the choice depends on the application's specific requirements, such as computational efficiency, accuracy, and the nature of the video sequences.

For a comparative analysis tailored to a particular scenario or application, one would typically evaluate each method on sample data to measure factors like accuracy, computational time, robustness to noise, and visual quality of the flow fields.
