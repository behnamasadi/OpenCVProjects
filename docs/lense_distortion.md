# 1. Lens Distortion

A real lens is not a perfect pinhole, so the mapping from 3D rays to image pixels departs from the ideal perspective projection. Because every ray passes through a single point in an ideal pinhole camera, a pinhole produces no distortion. A real lens bends light, and the two dominant effects are **radial** distortion (from the shape of the lens elements) and **tangential** distortion (from the lens assembly not being perfectly parallel to and centered on the sensor).

> See also the companion notebook [`undistortion.ipynb`](undistortion.ipynb) for a worked example of removing these effects with OpenCV.

## 1.1 Radial Distortion

Radial distortion is caused by the lens bending light more or less strongly away from the optical axis. It is called **symmetric** because it depends only on the radial distance `r` from the distortion center, not on direction. It therefore only displaces points along the radial direction.

Radial distortion is modeled by the even-order polynomial factor

$$1 + k_1 r^2 + k_2 r^4 + k_3 r^6$$

applied to the (normalized) point coordinates. The sign and magnitude of the coefficients determine the type of distortion.

### 1.1.1 Barrel Distortion (Negative Radial Distortion)

Magnification **decreases** with distance from the optical axis, so the radial factor $1 + k_1 r^2 + k_2 r^4 + k_3 r^6$ is monotonically **decreasing** (dominant $k_1 < 0$). Straight lines near the edges bow **outward**, like the sides of a barrel. This is typical of wide-angle and fisheye lenses.

For example, $k_1 = -1.5$.

<img src="images/Barrel_distortion.svg" height="250" width="250" />

### 1.1.2 Pincushion Distortion (Positive Radial Distortion)

Magnification **increases** with distance from the optical axis, so the radial factor $1 + k_1 r^2 + k_2 r^4 + k_3 r^6$ is monotonically **increasing** (dominant $k_1 > 0$). Straight lines near the edges bow **inward**, toward the center, like the seams of a pincushion. This is typical of telephoto lenses.

For example, $k_1 = +1.5$.

<img src="images/Pincushion_distortion.svg" height="250" width="250" />

### 1.1.3 Mustache Distortion

Also called *complex* or *wavy* distortion: a mixture of the two above, where the radial factor is **non-monotonic** and changes sign for some value of `r` (e.g. barrel near the center transitioning to pincushion toward the edges). It requires the higher-order terms ($k_2$, $k_3$) to be modeled.

<img src="images/Mustache_distortion.svg" height="250" width="250"/>

## 1.2 Tangential Distortion

Tangential (decentering) distortion arises when the lens assembly is not perfectly centered over and parallel to the image plane. Unlike radial distortion it is **not** symmetric about the center; it shifts points perpendicular to the radial direction as well.

|   |   |
|---|---|
|<img src="images/tangential_distortions.svg" height="250" width="250"/>   | <img src="images/radial-and-tangential-distortion.png" height="270" width="350"/>   |
|[image courtesy](https://www.tangramvision.com/blog/camera-modeling-exploring-distortion-and-distortion-models-part-i)   |      [image courtesy](https://www.researchgate.net/publication/260728375_Laboratory_calibration_of_star_sensor_with_installation_error_using_a_nonlinear_distortion_model)  |

# 2. The Brown–Conrady Model (Core Equations)

The Brown–Conrady model combines radial and tangential distortion. **All coordinates below are in the normalized image plane** (i.e. after dividing by $z$), *not* in pixels. Given a normalized point $(x, y)$ with $r^2 = x^2 + y^2$, the distorted point $(x_d, y_d)$ is

$$x_d = x\left(1 + k_1 r^2 + k_2 r^4 + k_3 r^6\right) + \underbrace{2 p_1 x y + p_2\left(r^2 + 2x^2\right)}_{\text{tangential}}$$

$$y_d = y\left(1 + k_1 r^2 + k_2 r^4 + k_3 r^6\right) + \underbrace{p_1\left(r^2 + 2y^2\right) + 2 p_2 x y}_{\text{tangential}}$$

Only after distortion is applied are the points mapped to pixels by the intrinsic matrix $K$:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} x_d \\ y_d \\ 1 \end{bmatrix}, \qquad K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

that is $u = f_x x_d + c_x$ and $v = f_y y_d + c_y$.

The distortion parameters are:

1. **Radial** coefficients: $k_1, k_2, k_3$ (and, in the extended rational model, $k_4, k_5, k_6$).
2. **Tangential** coefficients: $p_1, p_2$.
3. **Thin-prism** coefficients (extended model only): $s_1, s_2, s_3, s_4$.

In practice $k_1, k_2, k_3, p_1, p_2$ are the ones most commonly estimated.

- Barrel distortion typically has a negative $k_1$.
- Pincushion distortion typically has a positive $k_1$.
- Mustache distortion has a non-monotonic radial series that changes sign for some $r$.

# 3. OpenCV Lens Distortion Model

OpenCV uses an extended Brown–Conrady model. Starting from a 3D point, it applies the extrinsics, normalizes, distorts, and finally applies the intrinsics.

$$\begin{bmatrix} x \\ y \\ z \end{bmatrix} = R \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} + t$$

$$x' = \frac{x}{z}, \qquad y' = \frac{y}{z}$$

The full model adds a rational radial term (denominator with $k_4, k_5, k_6$) and thin-prism terms ($s_1 \dots s_4$):

$$x'' = x'\,\frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + 2 p_1 x' y' + p_2\left(r^2 + 2 x'^2\right) + s_1 r^2 + s_2 r^4$$

$$y'' = y'\,\frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + p_1\left(r^2 + 2 y'^2\right) + 2 p_2 x' y' + s_3 r^2 + s_4 r^4$$

$$\text{where } r^2 = x'^2 + y'^2$$

The point is then mapped to pixels:

$$u = f_x x'' + c_x, \qquad v = f_y y'' + c_y$$

If $k_4 = k_5 = k_6 = 0$ and $s_1 = s_2 = s_3 = s_4 = 0$, this reduces exactly to the core Brown–Conrady model in Section 2.

> **Calibration note.** Radial distortion is always monotonic for real lenses. If the estimator produces a non-monotonic result (outside a genuine mustache lens), this should be treated as a calibration failure: it can look deceptively good near the image center yet perform poorly toward the edges, which hurts AR / SfM applications. OpenCV's optimizer does not enforce monotonicity constraints (that would require integer programming and polynomial inequalities the framework does not support). See OpenCV issue [#15992](https://github.com/opencv/opencv/issues/15992) for details.

## 3.1 Relevant OpenCV Functions

- **`cv2.undistort(src, K, distCoeffs[, newCameraMatrix])`** — one-shot undistortion of a whole image.
- **`cv2.initUndistortRectifyMap(K, distCoeffs, R, newCameraMatrix, size, m1type)`** — precomputes the `(map1, map2)` remap tables; combined with `cv2.remap`, this is the efficient way to undistort a stream of images from a fixed camera.
- **`cv2.undistortPoints(src, K, distCoeffs[, R, P])`** — undistorts a sparse set of points (e.g. feature correspondences) rather than a full image; returns normalized coordinates unless a projection matrix `P` is supplied.
- **`cv2.getOptimalNewCameraMatrix(K, distCoeffs, size, alpha)`** — computes a new camera matrix that controls how much of the (possibly black-bordered) undistorted image to keep. `alpha = 0` crops to remove all invalid pixels; `alpha = 1` retains all source pixels.

See [`undistortion.ipynb`](undistortion.ipynb) for these in action.

## 3.2 Fisheye Model

Wide field-of-view (fisheye) lenses distort too strongly for the polynomial Brown–Conrady model to fit well. OpenCV provides a dedicated **`cv2.fisheye`** module based on the Kannala–Brandt model, which parameterizes distortion as a polynomial in the incidence angle $\theta$ (the angle between the incoming ray and the optical axis) rather than in the radius $r$:

$$\theta_d = \theta\left(1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8\right)$$

The counterparts are `cv2.fisheye.undistortImage`, `cv2.fisheye.initUndistortRectifyMap`, `cv2.fisheye.undistortPoints`, and `cv2.fisheye.calibrate`.

# 4. Image Resolution and Distortion Coefficients

The distortion coefficients $k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6$ do not depend on the scene and remain the **same** regardless of image resolution, because $r$ is measured in normalized coordinates. If a camera is calibrated at $320 \times 240$, the same distortion coefficients apply to $640 \times 480$ images from the same camera.

The intrinsics $f_x, f_y, c_x, c_y$, however, must be scaled with resolution:

```
fx_new = (new_width  / old_width ) * fx_old
fy_new = (new_height / old_height) * fy_old

cx_new = (new_width  / old_width ) * cx_old
cy_new = (new_height / old_height) * cy_old
```

Ref: [C++ OpenCV: calibration of the camera with different resolution](https://stackoverflow.com/questions/44888119/c-opencv-calibration-of-the-camera-with-different-resolution)

# 5. Distortion Models (Alternative Formulations)

## 5.1 Brown–Conrady (Radial Displacement Form)

The same radial distortion can be written as a radial displacement $\delta r$ as a function of the (image-plane, Cartesian, not pixel) radius. All points below are in the image plane.

### 5.1.1 Radial Distortion

<img src="images/radial_distortion_image_plane.svg" height="250" width="250" />

$$\delta r = k_1 r^3 + k_2 r^5 + k_3 r^7 + \cdots + k_n r^{\,n+2}$$

Projected onto the axes (with $\sin\psi = x/r$, $\cos\psi = y/r$):

$$\delta x_r = \sin(\psi)\,\delta r = \frac{x}{r}\left(k_1 r^3 + k_2 r^5 + k_3 r^7\right) = x\left(k_1 r^2 + k_2 r^4 + k_3 r^6\right)$$

$$\delta y_r = \cos(\psi)\,\delta r = \frac{y}{r}\left(k_1 r^3 + k_2 r^5 + k_3 r^7\right) = y\left(k_1 r^2 + k_2 r^4 + k_3 r^6\right)$$

where

- $(x_d,\ y_d)$ is the distorted image point,
- $(x_u,\ y_u)$ is the undistorted image point,
- $(x_c,\ y_c)$ is the distortion center,
- $r = \sqrt{(x_d - x_c)^2 + (y_d - y_c)^2}$.

### 5.1.2 Tangential Distortion

Using the OpenCV / standard Brown–Conrady convention:

$$\delta x_t = 2 p_1 x y + p_2\left(r^2 + 2x^2\right)$$

$$\delta y_t = p_1\left(r^2 + 2y^2\right) + 2 p_2 x y$$

### 5.1.3 Both Together (Distorted → Undistorted)

Combining radial and tangential terms to recover the undistorted point:

$$
\begin{aligned}
x_u = x_d &+ (x_d - x_c)(K_1 r^2 + K_2 r^4 + \cdots) \\
          &+ \left(P_1\left(r^2 + 2(x_d - x_c)^2\right) + 2 P_2 (x_d - x_c)(y_d - y_c)\right)(1 + P_3 r^2 + P_4 r^4 + \cdots) \\
y_u = y_d &+ (y_d - y_c)(K_1 r^2 + K_2 r^4 + \cdots) \\
          &+ \left(2 P_1 (x_d - x_c)(y_d - y_c) + P_2\left(r^2 + 2(y_d - y_c)^2\right)\right)(1 + P_3 r^2 + P_4 r^4 + \cdots)
\end{aligned}
$$

where $K_n$ is the $n^\text{th}$ radial coefficient and $P_n$ is the $n^\text{th}$ tangential coefficient.

## 5.2 Division Model

For radial distortion the division model is often preferred over Brown–Conrady, since it needs fewer terms to accurately describe severe distortion:

$$
\begin{aligned}
x_u &= x_c + \frac{x_d - x_c}{1 + K_1 r^2 + K_2 r^4 + \cdots} \\
y_u &= y_c + \frac{y_d - y_c}{1 + K_1 r^2 + K_2 r^4 + \cdots}
\end{aligned}
$$

# References

1. [Camera Modeling: Exploring Distortion and Distortion Models (Tangram Vision)](https://www.tangramvision.com/blog/camera-modeling-exploring-distortion-and-distortion-models-part-i)
2. [Fitzgibbon, *Simultaneous Linear Estimation of Multiple View Geometry and Lens Distortion* (division model)](https://www.robots.ox.ac.uk/~vgg/publications/2001/Fitzgibbon01b/fitzgibbon01b.pdf)
3. [Zhang, *A Flexible New Technique for Camera Calibration* (Microsoft Research)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)
4. [Camera Distortions (ori.codes)](https://ori.codes/artificial-intelligence/camera-calibration/camera-distortions/)
5. [OpenCV camera calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
</content>
</invoke>
