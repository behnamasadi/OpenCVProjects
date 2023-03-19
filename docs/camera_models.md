# Camera Models
The camera models is the projection of 3D points from camera coordinates `x, y, z` into points `u, v` in normalized image coordinates.

## Perspective Camera


<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Bl%7D%20x_n%20%3D%20%5Cfrac%7Bx%7D%7Bz%7D%20%5C%5C%20y_n%20%3D%20%5Cfrac%7By%7D%7Bz%7D%20%5C%5C%20r%5E2%20%3D%20x_n%5E2%20&plus;%20y_n%5E2%20%5C%5C%20d%20%3D%201%20&plus;%20k_1%20r%5E2%20&plus;%20k_2%20r%5E4%20%5C%5C%20u%20%3D%20f%5C%20d%5C%20x_n%20%5C%5C%20v%20%3D%20f%5C%20d%5C%20y_n%20%5Cend%7Barray%7D" alt="https://latex.codecogs.com/svg.latex?\begin{array}{l}
x_n = \frac{x}{z} \\
y_n = \frac{y}{z} \\
r^2 = x_n^2 + y_n^2 \\
d = 1 + k_1 r^2 + k_2 r^4 \\
u = f\ d\ x_n \\
v = f\ d\ y_n
\end{array}
" />


## Fisheye Camera


<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Bl%7D%20r%5E2%20%3D%20x%5E2%20&plus;%20y%5E2%20%5C%5C%20%5Ctheta%20%3D%20%5Carctan%28r%20/%20z%29%20%5C%5C%20d%20%3D%201%20&plus;%20k_1%20%5Ctheta%5E2&plus;%20k_2%20%5Ctheta%5E4%20%5C%5C%20u%20%3D%20f%5C%20d%5C%20%5Ctheta%5C%20%5Cfrac%7Bx%7D%7Br%7D%20%5C%5C%20v%20%3D%20f%5C%20d%5C%20%5Ctheta%5C%20%5Cfrac%7By%7D%7Br%7D%20%5Cend%7Barray%7D" alt="https://latex.codecogs.com/svg.latex?\begin{array}{l}
r^2 = x^2 + y^2 \\
\theta = \arctan(r / z) \\
d = 1 +  k_1 \theta^2+  k_2 \theta^4 \\
u = f\ d\ \theta\ \frac{x}{r} \\
v = f\ d\ \theta\ \frac{y}{r}
\end{array}
" />


<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20L%28%5Ctilde%7Bx%7D%2C%5Ctilde%7By%7D%29%20%26%3D%20%5Cfrac%7Br_d%7D%7Br%7D%20%5Cbegin%7Bbmatrix%7D%20%5Ctilde%7Bx%7D%20%5C%5C%20%5Ctilde%7By%7D%20%5Cend%7Bbmatrix%7D%20%5C%5C%20r_d%20%26%3D%20M_1%28%5Ctheta_d%29%20%5C%5C%20%5Ctheta_d%20%26%3D%20%5Ctheta%281&plus;%20k_1%5Ctheta%5E2%20&plus;%20k_2%5Ctheta%5E4%20&plus;%20k_3%5Ctheta%5E6%20&plus;%20k_4%5Ctheta%5E8%29%20%5C%5C%20%5Ctheta%20%26%3D%20%5Carctan%28r%29%20%5C%5C%20r%20%26%3D%20%5Csqrt%7B%5Ctilde%7Bx%7D%5E2%20&plus;%20%5Ctilde%7By%7D%5E2%7D%20%5Cend%7Balign*%7D" alt="https://latex.codecogs.com/svg.latex?\begin{align*} L(\tilde{x},\tilde{y}) &= \frac{r_d}{r} \begin{bmatrix} \tilde{x} \\ \tilde{y} \end{bmatrix} \\ r_d &= M_1(\theta_d) \\ \theta_d &= \theta(1+ k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8) \\ \theta &= \arctan(r) \\ r &= \sqrt{\tilde{x}^2 + \tilde{y}^2} \end{align*}" />


Refs: [1](https://docs.nvidia.com/vpi/algo_ldc.html)



## Spherical Camera

<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Bl%7D%20%5Cmathrm%7Blon%7D%20%3D%20%5Carctan%5Cleft%28%5Cfrac%7Bx%7D%7Bz%7D%5Cright%29%20%5C%5C%20%5Cmathrm%7Blat%7D%20%3D%20%5Carctan%5Cleft%28%5Cfrac%7B-y%7D%7B%5Csqrt%7Bx%5E2%20&plus;%20z%5E2%7D%7D%5Cright%29%20%5C%5C%20u%20%3D%20%5Cfrac%7B%5Cmathrm%7Blon%7D%7D%7B2%20%5Cpi%7D%20%5C%5C%20v%20%3D%20-%5Cfrac%7B%5Cmathrm%7Blat%7D%7D%7B2%20%5Cpi%7D%20%5Cend%7Barray%7D" alt="https://latex.codecogs.com/svg.latex?\begin{array}{l}
\mathrm{lon} = \arctan\left(\frac{x}{z}\right) \\
\mathrm{lat} = \arctan\left(\frac{-y}{\sqrt{x^2 + z^2}}\right) \\
u = \frac{\mathrm{lon}}{2 \pi} \\
v = -\frac{\mathrm{lat}}{2 \pi}
\end{array} 
" />



Refs: [1](https://opensfm.readthedocs.io/en/latest/geometry.html)
