# Optical Flow

Between two image frames which are taken at times <img src="https://latex.codecogs.com/svg.latex?t" alt="https://latex.codecogs.com/svg.latex?t" /> and  <img src="https://latex.codecogs.com/svg.latex?t&plus;%5CDelta%20t" alt="https://latex.codecogs.com/svg.latex?t+\Delta t" /> at every position following brightness constancy constraint:

<img src="https://latex.codecogs.com/svg.latex?I%28x%2Cy%2Ct%29%20%3D%20I%28x&plus;%5CDelta%20x%2C%20y%20&plus;%20%5CDelta%20y%2C%20t%20&plus;%20%5CDelta%20t%29" alt="https://latex.codecogs.com/svg.latex?I(x,y,t) = I(x+\Delta x, y + \Delta y, t + \Delta t)" />



Assuming the movement to be small, Taylor series of:

<img src="https://latex.codecogs.com/svg.latex?I%28x&plus;%5CDelta%20x%2C%20y%20&plus;%20%5CDelta%20y%2C%20t%20&plus;%20%5CDelta%20t%29" alt="https://latex.codecogs.com/svg.latex?I(x+\Delta x, y + \Delta y, t + \Delta t)" /> 

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



This equation is known as the aperture problem of the optical flow, has two unknowns.  To find the optical flow another set of equations is needed, given by some additional constraint. All optical flow methods introduce additional conditions for estimating the actual flow.


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




<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%7B%5Cbegin%7Bbmatrix%7DV_%7Bx%7D%5C%5C%5B10pt%5DV_%7By%7D%5Cend%7Bbmatrix%7D%7D%3D%7B%5Cbegin%7Bbmatrix%7D%5Csum%20_%7Bi%7DI_%7Bx%7D%28q_%7Bi%7D%29%5E%7B2%7D%26%5Csum%20_%7Bi%7DI_%7Bx%7D%28q_%7Bi%7D%29I_%7By%7D%28q_%7Bi%7D%29%5C%5C%5B10pt%5D%5Csum%20_%7Bi%7DI_%7By%7D%28q_%7Bi%7D%29I_%7Bx%7D%28q_%7Bi%7D%29%26%5Csum%20_%7Bi%7DI_%7By%7D%28q_%7Bi%7D%29%5E%7B2%7D%5Cend%7Bbmatrix%7D%7D%5E%7B-1%7D%7B%5Cbegin%7Bbmatrix%7D-%5Csum%20_%7Bi%7DI_%7Bx%7D%28q_%7Bi%7D%29I_%7Bt%7D%28q_%7Bi%7D%29%5C%5C%5B10pt%5D-%5Csum%20_%7Bi%7DI_%7By%7D%28q_%7Bi%7D%29I_%7Bt%7D%28q_%7Bi%7D%29%5Cend%7Bbmatrix%7D%7D%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle {\begin{bmatrix}V_{x}\\[10pt]V_{y}\end{bmatrix}}={\begin{bmatrix}\sum _{i}I_{x}(q_{i})^{2}&\sum _{i}I_{x}(q_{i})I_{y}(q_{i})\\[10pt]\sum _{i}I_{y}(q_{i})I_{x}(q_{i})&\sum _{i}I_{y}(q_{i})^{2}\end{bmatrix}}^{-1}{\begin{bmatrix}-\sum _{i}I_{x}(q_{i})I_{t}(q_{i})\\[10pt]-\sum _{i}I_{y}(q_{i})I_{t}(q_{i})\end{bmatrix}}}" />

<br/>
<br/>

<img src="" alt="https://latex.codecogs.com/svg.latex?" />




