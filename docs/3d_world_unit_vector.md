# 3D World Unit Vector / Camera Direction Vectors / Camera Projection Rays

This note explains how to turn a **2D pixel** `(u, v)` into the **3D ray**
(direction / bearing vector) it corresponds to, first in the **camera frame**
and then in the **world frame**. This is the back-projection step used in
triangulation, PnP sanity checks, ray casting, and bearing-only SLAM.

## Convention (read this first)

Extrinsics here are **world-to-camera**:

$$
X_{\text{cam}} = R \, X_{\text{world}} + t
$$

so `[R | t]` maps a world point into the camera frame. Consequently the
**camera-to-world** map uses the transpose (rotations are orthonormal,
$R^{-1} = R^\top$):

$$
X_{\text{world}} = R^\top \left( X_{\text{cam}} - t \right) = R^\top X_{\text{cam}} - R^\top t
$$

The **camera center** in world coordinates is therefore

$$
C = -R^\top t
$$

This matches OpenCV's `cv::projectPoints`, whose `rvec`/`tvec` are exactly the
world-to-camera $R$ (via Rodrigues) and $t$. If instead your $[R|t]$ is
camera-to-world (some SfM tools store poses this way), **drop all the
transposes** below and use $R$ directly, with $C = t$.

## The pinhole model and why a pixel is a ray, not a point

The full projection of a world point $X = (X, Y, Z, 1)^\top$ is

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \, [R \mid t] \, X,
\qquad
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

The scalar $s$ is the (unknown) **depth** along the optical axis. Projection
*divides it out*, so when we go backwards from a pixel we cannot recover $s$:
one pixel is consistent with an entire **line of 3D points** through the camera
center. That line is the **projection ray**. All we can recover from a single
image is its **direction**, not a point on it.

## Normalized (calibrated) coordinates → camera-frame ray

Multiplying the pixel by $K^{-1}$ removes the intrinsics and gives
**normalized image coordinates** — the point where the ray pierces the plane
$Z = 1$ in the camera frame:

$$
\text{ray}_{\text{cam}} = K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
= \begin{bmatrix} (u - c_x)/f_x \\ (v - c_y)/f_y \\ 1 \end{bmatrix}
$$

This vector *is* the ray direction in the camera frame (it points from the
camera center out through the pixel). Its third component is fixed at 1, so to
get a proper **unit bearing vector** we normalize:

$$
d_{\text{cam}} = \frac{\text{ray}_{\text{cam}}}{\lVert \text{ray}_{\text{cam}} \rVert}
$$

$d_{\text{cam}}$ is dimensionless and depth-free — exactly the "3D unit vector"
that a pixel encodes.

## Handle lens distortion first

The $K^{-1}$ formula above assumes an **ideal pinhole**. With real lenses,
radial/tangential distortion means the pixel is *not* where the linear model
predicts, so you must **undistort before** back-projecting. `cv::undistortPoints`
does both steps at once: it removes distortion **and** applies $K^{-1}$,
returning normalized coordinates directly (pass `P = noArray()` / no new camera
matrix so the output stays normalized rather than being re-projected to pixels).

## Transforming the ray into the world frame

Rotate the camera-frame direction into world coordinates with $R^\top$ (a
direction has no translation component):

$$
d_{\text{world}} = R^\top \, d_{\text{cam}}
$$

The ray is then the set of world points

$$
X(\lambda) = C + \lambda \, d_{\text{world}},
\qquad C = -R^\top t, \quad \lambda \ge 0
$$

Two such rays from two cameras are what triangulation intersects (in practice,
least-squares closest approach, since they rarely meet exactly).

## OpenCV snippet

### Python

```python
import cv2
import numpy as np

K    = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
dist = np.array([-0.28, 0.10, 0.0, 0.0, 0.0])   # k1 k2 p1 p2 k3
R, t = R_w2c, t_w2c                              # world->camera extrinsics

uv = np.array([[450.0, 190.0]], dtype=np.float64).reshape(-1, 1, 2)

# Undistort + K^-1 in one call -> normalized coords (x, y) on the Z=1 plane
norm = cv2.undistortPoints(uv, K, dist)          # shape (N,1,2)
x, y = norm[0, 0]
ray_cam = np.array([x, y, 1.0])
d_cam   = ray_cam / np.linalg.norm(ray_cam)      # unit bearing in camera frame

d_world = R.T @ d_cam                             # rotate into world frame
C       = -R.T @ t.reshape(3)                     # camera center in world
# world ray: X(lambda) = C + lambda * d_world,  lambda >= 0
print("unit dir (cam):", d_cam, " (world):", d_world, " center:", C)
```

### C++

```cpp
std::vector<cv::Point2f> pix{ {450.f, 190.f} }, norm;
cv::undistortPoints(pix, norm, K, dist);          // -> normalized coords
cv::Vec3d rayCam(norm[0].x, norm[0].y, 1.0);
cv::Vec3d dCam = cv::normalize(rayCam);           // unit bearing, camera frame

cv::Mat dWorld = R.t() * cv::Mat(dCam);           // R is world->camera
cv::Mat C      = -R.t() * t;                       // camera center in world
```

Without distortion you can skip `undistortPoints` and use `K.inv()` directly:
`cv::Mat rayCam = K.inv() * (cv::Mat_<double>(3,1) << u, v, 1);`.

## Related code and cross-links

- **`src/camera_projection_matrix.cpp`** in this repo builds $K$, $[R|t]$ and
  calls `cv::projectPoints` (the **forward** direction, world → pixel). The ray
  computation here is the **inverse** of that pipeline; the same $R$, $t$, and
  $K$ conventions apply (`R_c_w`, `T_c_w` there are the world-to-camera pose fed
  to `projectPoints`).
- **Normalized image coordinates** are the shared concept with the epipolar /
  essential-matrix material: see
  [`docs/epipolar_geometry_essential_matrix_fundamental_matrix.ipynb`](./epipolar_geometry_essential_matrix_fundamental_matrix.ipynb).
  The essential matrix $E$ relates the *normalized* rays of two views
  ($\hat{x}'^\top E \, \hat{x} = 0$), i.e. the same $K^{-1}[u,v,1]^\top$ bearings
  derived above.
- See also `docs/undistortion.md`/`undistortion.ipynb` and
  `docs/pinhole_camera_model_projection_intrinsic.ipynb` for the forward model.

Refs:
- [OpenCV `undistortPoints`](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga55c716492470bfe86b0ee9bf3a1f0f7e)
- [OpenCV camera calibration & 3D reconstruction (pinhole model)](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
