# Snavely Reprojection Error
The Noah Snavely reprojection error is a common metric used in the field of computer vision and 3D reconstruction to evaluate the accuracy of a camera pose estimation or 3D point triangulation. It is named after Noah Snavely, a researcher known for his contributions to computer vision and structure from motion.

The reprojection error measures the discrepancy between the observed 2D image points and their corresponding 3D points projected back onto the image plane using the estimated camera pose and intrinsic parameters. The goal is to minimize this error to achieve accurate camera pose estimates and 3D reconstructions.

For a single image, let's denote:

- Observed 2D image point as (u, v).
- Corresponding 3D point in the scene as (X, Y, Z).
- Estimated camera intrinsic parameters as focal length (f) and principal point (c_x, c_y).
- Estimated camera extrinsic parameters (pose) as rotation matrix (R) and translation vector (t).

The reprojection error is then calculated as follows:

1. Project the 3D point (X, Y, Z) into the image plane using the estimated camera pose and intrinsic parameters:

   ```
   x_proj = f * (R[0, 0] * X + R[0, 1] * Y + R[0, 2] * Z + t[0]) / (R[2, 0] * X + R[2, 1] * Y + R[2, 2] * Z + t[2])
   y_proj = f * (R[1, 0] * X + R[1, 1] * Y + R[1, 2] * Z + t[1]) / (R[2, 0] * X + R[2, 1] * Y + R[2, 2] * Z + t[2])
   ```

2. Compute the difference between the observed image point (u, v) and the projected image point (x_proj, y_proj):

   ```
   dx = u - x_proj
   dy = v - y_proj
   ```

3. Calculate the squared reprojection error as the sum of squared differences:

   ```
   reprojection_error = dx^2 + dy^2
   ```

The Noah Snavely reprojection error is then the sum of squared reprojection errors across all observed image points for a given camera pose or 3D point. The goal during camera pose estimation or bundle adjustment is to find the camera poses and 3D point positions that minimize this reprojection error.

Minimizing the reprojection error is crucial for accurate 3D reconstruction and computer vision tasks, such as structure from motion, simultaneous localization and mapping (SLAM), and visual odometry. Various optimization techniques, such as Levenberg-Marquardt or Gauss-Newton, are commonly used to find the optimal camera poses and 3D points that minimize the reprojection error in these applications.

Refs: [1](https://ceres-solver.googlesource.com/ceres-solver/+/1.12.0/examples/snavely_reprojection_error.h), [2](https://www.eecs.umich.edu/courses/eecs442-ahowens/fa21/slides/lec22-sfm.pdf)
