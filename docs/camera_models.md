# Camera Models
The camera model is the projection of 3D points from camera coordinates `x, y, z` into points `u, v` in normalized image coordinates.

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


## Kannala-Brandt Fisheye Camera


<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Bl%7D%20r%5E2%20%3D%20x%5E2%20&plus;%20y%5E2%20%5C%5C%20%5Ctheta%20%3D%20%5Carctan%28r%20/%20z%29%20%5C%5C%20d%20%3D%201%20&plus;%20k_1%20%5Ctheta%5E2&plus;%20k_2%20%5Ctheta%5E4%20%5C%5C%20u%20%3D%20f%5C%20d%5C%20%5Ctheta%5C%20%5Cfrac%7Bx%7D%7Br%7D%20%5C%5C%20v%20%3D%20f%5C%20d%5C%20%5Ctheta%5C%20%5Cfrac%7By%7D%7Br%7D%20%5Cend%7Barray%7D" alt="https://latex.codecogs.com/svg.latex?\begin{array}{l}
r^2 = x^2 + y^2 \\
\theta = \arctan(r / z) \\
d = 1 +  k_1 \theta^2+  k_2 \theta^4 \\
u = f\ d\ \theta\ \frac{x}{r} \\
v = f\ d\ \theta\ \frac{y}{r}
\end{array}
" />


<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20L%28%5Ctilde%7Bx%7D%2C%5Ctilde%7By%7D%29%20%26%3D%20%5Cfrac%7Br_d%7D%7Br%7D%20%5Cbegin%7Bbmatrix%7D%20%5Ctilde%7Bx%7D%20%5C%5C%20%5Ctilde%7By%7D%20%5Cend%7Bbmatrix%7D%20%5C%5C%20r_d%20%26%3D%20M_1%28%5Ctheta_d%29%20%5C%5C%20%5Ctheta_d%20%26%3D%20%5Ctheta%281&plus;%20k_1%5Ctheta%5E2%20&plus;%20k_2%5Ctheta%5E4%20&plus;%20k_3%5Ctheta%5E6%20&plus;%20k_4%5Ctheta%5E8%29%20%5C%5C%20%5Ctheta%20%26%3D%20%5Carctan%28r%29%20%5C%5C%20r%20%26%3D%20%5Csqrt%7B%5Ctilde%7Bx%7D%5E2%20&plus;%20%5Ctilde%7By%7D%5E2%7D%20%5Cend%7Balign*%7D" alt="https://latex.codecogs.com/svg.latex?\begin{align*} L(\tilde{x},\tilde{y}) &= \frac{r_d}{r} \begin{bmatrix} \tilde{x} \\ \tilde{y} \end{bmatrix} \\ r_d &= M_1(\theta_d) \\ \theta_d &= \theta(1+ k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8) \\ \theta &= \arctan(r) \\ r &= \sqrt{\tilde{x}^2 + \tilde{y}^2} \end{align*}" />

Read the original paper [A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses](https://oulu3dvision.github.io/calibgeneric/Kannala_Brandt_calibration.pdf)


Refs: [1](https://docs.nvidia.com/vpi/algo_ldc.html)


The Kannala-Brandt fisheye model is a mathematical model used to describe fisheye lenses, which are lenses that have a wide field of view, typically greater than 180 degrees. This model was introduced by Juho Kannala and Sami S. Brandt in their paper "A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses."

The model represents the mapping between the coordinates in the camera's image plane and the direction of incoming light rays. The model uses the radial distortion coefficient and the tangential distortion coefficient to describe the distortion in the image caused by the lens. The radial distortion coefficient represents the amount of radial distortion in the image, which is the stretching or compression of the image along radial lines from the center of the image. The tangential distortion coefficient represents the amount of tangential distortion in the image, which is the shift of the image along the tangential direction.

The Kannala-Brandt fisheye model uses a parameter called the distortion center to describe the position of the center of distortion in the image. The model also includes a set of coefficients that describe the radial distortion as a polynomial of the radius from the distortion center. The coefficients can be estimated using calibration techniques, such as using a checkerboard pattern to determine the correspondence between the image coordinates and the world coordinates.

One advantage of the Kannala-Brandt fisheye model is that it is a generic model that can be used to describe both conventional and fisheye lenses. This means that the same calibration techniques can be used for both types of lenses, making the model useful for a wide range of applications.

Overall, the Kannala-Brandt fisheye model is an important tool for camera calibration and for understanding the properties of fisheye lenses. It provides a way to model the distortion caused by the lens and to correct for that distortion in the resulting image.




** Math behind Kannala-Brandt fisheye model **

The Kannala-Brandt fisheye model represents the mapping between the image coordinates (u, v) in the image plane and the direction of incoming light rays (x, y, z) in the world coordinates. The mapping is described by the following equation:

(x, y, z) = (x', y', z') / ||(x', y', z')||, where

x' = r * (u - u0) / f
y' = r * (v - v0) / f
z' = z0

and

r = rd(||(u - u0, v - v0)||),

where (u0, v0) is the distortion center in the image plane, f is the focal length of the lens, rd(r) is the radial distortion function, and z0 is a constant. The radial distortion function, rd(r), models the stretching or compression of the image along radial lines from the center of the image. It is described as a polynomial of the radius r, where the coefficients of the polynomial are estimated using calibration techniques.

The tangential distortion component is modeled as a linear combination of the image coordinates:

x'' = x' + (2 * p1 * x' * y' + p2 * (r^2 + 2 * x'^2))
y'' = y' + (p1 * (r^2 + 2 * y'^2) + 2 * p2 * x' * y')

where p1 and p2 are the tangential distortion coefficients.

The mapping between the image coordinates and the world coordinates can be inverted to correct for the distortion in the image. The correction process involves transforming the distorted image coordinates into undistorted coordinates by using the inverse of the mapping described by the radial and tangential distortion components.

In summary, the mathematics behind the Kannala-Brandt fisheye model involves representing the mapping between the image coordinates and the direction of incoming light rays using radial and tangential distortion components, and using polynomials and linear combinations to describe the distortion. The coefficients of the radial and tangential distortion components are estimated using calibration techniques, and the correction for distortion in the image is performed by using the inverse of the mapping.

```
import numpy as np

def kannala_brandt_model(image_coordinates, distortion_center, focal_length, radial_coefficients, tangential_coefficients):
    u, v = image_coordinates
    u0, v0 = distortion_center
    f = focal_length
    p1, p2 = tangential_coefficients

    r = np.sqrt((u - u0)**2 + (v - v0)**2)
    theta = np.arctan2(v - v0, u - u0)

    radial_distortion = np.polyval(radial_coefficients, r)
    x = radial_distortion * np.cos(theta)
    y = radial_distortion * np.sin(theta)

    x_prime = x / f
    y_prime = y / f

    x_double_prime = x_prime + (2 * p1 * x_prime * y_prime + p2 * (r**2 + 2 * x_prime**2))
    y_double_prime = y_prime + (p1 * (r**2 + 2 * y_prime**2) + 2 * p2 * x_prime * y_prime)

    return x_double_prime, y_double_prime
```


In this example, image_coordinates is a tuple (u, v) representing the image coordinates in the image plane, distortion_center is a tuple (u0, v0) representing the position of the center of distortion in the image, focal_length is the focal length of the lens, radial_coefficients is a list of coefficients for the radial distortion function represented as a polynomial, and tangential_coefficients is a tuple (p1, p2) representing the tangential distortion coefficients.

The function kannala_brandt_model takes the image coordinates, distortion center, focal length, radial coefficients, and tangential coefficients as input and returns the undistorted image coordinates (x'', y'').



    



Refs: [1](https://oulu3dvision.github.io/calibgeneric/Kannala_Brandt_calibration.pdf)




## Spherical Camera

<img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Barray%7D%7Bl%7D%20%5Cmathrm%7Blon%7D%20%3D%20%5Carctan%5Cleft%28%5Cfrac%7Bx%7D%7Bz%7D%5Cright%29%20%5C%5C%20%5Cmathrm%7Blat%7D%20%3D%20%5Carctan%5Cleft%28%5Cfrac%7B-y%7D%7B%5Csqrt%7Bx%5E2%20&plus;%20z%5E2%7D%7D%5Cright%29%20%5C%5C%20u%20%3D%20%5Cfrac%7B%5Cmathrm%7Blon%7D%7D%7B2%20%5Cpi%7D%20%5C%5C%20v%20%3D%20-%5Cfrac%7B%5Cmathrm%7Blat%7D%7D%7B2%20%5Cpi%7D%20%5Cend%7Barray%7D" alt="https://latex.codecogs.com/svg.latex?\begin{array}{l}
\mathrm{lon} = \arctan\left(\frac{x}{z}\right) \\
\mathrm{lat} = \arctan\left(\frac{-y}{\sqrt{x^2 + z^2}}\right) \\
u = \frac{\mathrm{lon}}{2 \pi} \\
v = -\frac{\mathrm{lat}}{2 \pi}
\end{array} 
" />



Refs: [1](https://opensfm.readthedocs.io/en/latest/geometry.html)
