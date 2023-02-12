The Kannala-Brandt fisheye model is a mathematical model used to describe fisheye lenses, which are lenses that have a wide field of view, typically greater than 180 degrees. This model was introduced by Juho Kannala and Sami S. Brandt in their paper "A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses."

The model represents the mapping between the coordinates in the camera's image plane and the direction of incoming light rays. The model uses the radial distortion coefficient and the tangential distortion coefficient to describe the distortion in the image caused by the lens. The radial distortion coefficient represents the amount of radial distortion in the image, which is the stretching or compression of the image along radial lines from the center of the image. The tangential distortion coefficient represents the amount of tangential distortion in the image, which is the shift of the image along the tangential direction.

The Kannala-Brandt fisheye model uses a parameter called the distortion center to describe the position of the center of distortion in the image. The model also includes a set of coefficients that describe the radial distortion as a polynomial of the radius from the distortion center. The coefficients can be estimated using calibration techniques, such as using a checkerboard pattern to determine the correspondence between the image coordinates and the world coordinates.

One advantage of the Kannala-Brandt fisheye model is that it is a generic model that can be used to describe both conventional and fisheye lenses. This means that the same calibration techniques can be used for both types of lenses, making the model useful for a wide range of applications.

Overall, the Kannala-Brandt fisheye model is an important tool for camera calibration and for understanding the properties of fisheye lenses. It provides a way to model the distortion caused by the lens and to correct for that distortion in the resulting image.




# math behind Kannala-Brandt fisheye model
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
