# Euler Angles, Roll/Pitch/Yaw, and Quaternions

A concise reference on 3D rotation representations: rotation matrices, Euler
angles, axis-angle/Rodrigues, and unit quaternions — with the conversions and
pitfalls that matter for OpenCV, Eigen, and ROS code.

## Rotation Matrices: SO(3)

A rotation in 3D is an element of the *special orthogonal group* $SO(3)$: the
set of real $3\times3$ matrices $R$ satisfying

$$R^\top R = R R^\top = I, \qquad \det(R) = +1.$$

Consequences:

- **Orthonormal**: columns (and rows) are mutually orthogonal unit vectors —
  they are the images of the world axes expressed in the new frame.
- **Inverse is transpose**: $R^{-1} = R^\top$. Cheap and numerically stable.
- **$\det = +1$** excludes reflections (those have $\det = -1$ and are not
  rotations). Together, orthonormality + unit determinant = proper rotation.
- **Composition**: $R = R_2 R_1$ applies $R_1$ first, then $R_2$. Matrix
  multiplication is not commutative, so **rotation order matters**.
- A rotation matrix has 9 entries but only **3 degrees of freedom** (the 6
  orthonormality constraints remove the rest).

## Euler Angles: Roll, Pitch, Yaw

Any rotation can be decomposed into three successive rotations about coordinate
axes. In the robotics/aerospace convention:

- **Roll** $\phi$ — rotation about the $x$ axis.
- **Pitch** $\theta$ — rotation about the $y$ axis.
- **Yaw** $\psi$ — rotation about the $z$ axis.

### Elementary rotation matrices

$$
R_x(\phi) =
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos\phi & -\sin\phi \\
0 & \sin\phi & \cos\phi
\end{bmatrix},\quad
R_y(\theta) =
\begin{bmatrix}
\cos\theta & 0 & \sin\theta \\
0 & 1 & 0 \\
-\sin\theta & 0 & \cos\theta
\end{bmatrix},\quad
R_z(\psi) =
\begin{bmatrix}
\cos\psi & -\sin\psi & 0 \\
\sin\psi & \cos\psi & 0 \\
0 & 0 & 1
\end{bmatrix}.
$$

### Intrinsic vs. extrinsic

- **Extrinsic**: each rotation is about the axes of the *fixed* world frame.
- **Intrinsic**: each rotation is about the axes of the *moving* body frame,
  which have already been rotated by the previous steps.

These are duals: an intrinsic sequence about axes $Z\!-\!Y'\!-\!X''$ equals the
extrinsic sequence about $x\!-\!y\!-\!z$ read in reverse. The common
**yaw–pitch–roll (ZYX)** rotation from body to world is

$$
R = R_z(\psi)\, R_y(\theta)\, R_x(\phi).
$$

Written out:

$$
R =
\begin{bmatrix}
c_\psi c_\theta & c_\psi s_\theta s_\phi - s_\psi c_\phi & c_\psi s_\theta c_\phi + s_\psi s_\phi \\
s_\psi c_\theta & s_\psi s_\theta s_\phi + c_\psi c_\phi & s_\psi s_\theta c_\phi - c_\psi s_\phi \\
-s_\theta & c_\theta s_\phi & c_\theta c_\phi
\end{bmatrix},
$$

where $c_\bullet = \cos$ and $s_\bullet = \sin$.

### The 12 conventions

An Euler sequence picks 3 axes where no two *consecutive* axes are equal. That
gives **12 valid orderings**, split into two families:

- **Proper/classic Euler** (first = last axis): `ZXZ, XYX, YZY, ZYZ, XZX, YXY`.
- **Tait–Bryan / Cardan** (all three axes distinct): `XYZ, YZX, ZXY, XZY, ZYX, YXZ`.

Roll/pitch/yaw is the Tait–Bryan `ZYX` (or `XYZ`, depending on library). Because
each convention can also be intrinsic or extrinsic, always state both the axis
order *and* the frame — this is the single most common source of bugs when
moving angles between OpenCV, Eigen, SciPy, and ROS.

### Gimbal lock

When pitch reaches $\theta = \pm 90^\circ$, the roll and yaw axes align and one
rotational degree of freedom is lost — different $(\phi, \psi)$ pairs produce the
same orientation, so the decomposition is no longer unique. In the `ZYX` matrix
above, $\theta = 90^\circ$ makes $c_\theta = 0$ and the top-left / bottom-left
terms collapse, leaving only $\phi + \psi$ (or their difference) determinable.
Gimbal lock is a property of the *3-angle representation*, not of the physical
rotation, and it is the primary reason to prefer quaternions for integration and
interpolation.

## Axis-Angle and Rodrigues

Euler's rotation theorem: every rotation is a single rotation by angle $\alpha$
about a unit axis $\hat{n}$. The **axis-angle** (rotation) vector packs both into
$\mathbf{r} = \alpha\,\hat{n}$, with $\|\mathbf{r}\| = \alpha$. The matrix is
given by **Rodrigues' formula**:

$$
R = I + \sin\alpha\,[\hat{n}]_\times + (1-\cos\alpha)\,[\hat{n}]_\times^2,
\qquad
[\hat{n}]_\times =
\begin{bmatrix}
0 & -n_z & n_y \\
n_z & 0 & -n_x \\
-n_y & n_x & 0
\end{bmatrix}.
$$

OpenCV exposes this both ways via `cv::Rodrigues`, which converts between a
$3\times1$ rotation vector and a $3\times3$ matrix (and returns the Jacobian for
optimization). See `src/rodrigue_rotation_matrices.cpp` in this repo for a
worked C++ example.

## Quaternions

A quaternion is $q = w + x\,i + y\,j + k\,z$ with
$i^2 = j^2 = k^2 = ijk = -1$. A **unit quaternion** ($\|q\| = 1$) represents a
rotation of angle $\alpha$ about unit axis $\hat{n}$ as

$$
q = \left(\cos\tfrac{\alpha}{2},\; \hat{n}\,\sin\tfrac{\alpha}{2}\right)
  = \left(w,\; x,\; y,\; z\right).
$$

Key properties:

- **Double cover**: $q$ and $-q$ represent the *same* rotation (the map
  $S^3 \to SO(3)$ is 2-to-1). Normalize signs before comparing or averaging.
- **No gimbal lock**: quaternions parameterize $SO(3)$ smoothly with only a
  single redundancy (the unit-norm constraint), so there is no singular
  orientation.
- **Composition** via the **Hamilton product** — 16 multiplies, cheaper and
  more stable than multiplying $3\times3$ matrices, and easy to renormalize:
  $q_{\text{total}} = q_2 \otimes q_1$ applies $q_1$ first.
- **Inverse** of a unit quaternion is its conjugate:
  $q^{-1} = q^* = (w, -x, -y, -z)$.
- **Interpolation** via **SLERP** (spherical linear interpolation) gives
  constant-angular-velocity, shortest-arc blends between orientations —
  something Euler angles cannot do cleanly.

The Hamilton product of $q_1 = (w_1, \mathbf{v}_1)$ and
$q_2 = (w_2, \mathbf{v}_2)$:

$$
q_2 \otimes q_1 =
\big(w_2 w_1 - \mathbf{v}_2\cdot\mathbf{v}_1,\;\;
w_2 \mathbf{v}_1 + w_1 \mathbf{v}_2 + \mathbf{v}_2 \times \mathbf{v}_1\big).
$$

### Quaternion → rotation matrix

For a unit quaternion $q = (w, x, y, z)$:

$$
R =
\begin{bmatrix}
1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
\end{bmatrix}.
$$

### Rotation matrix → quaternion

Using the trace $t = R_{00} + R_{11} + R_{22}$ (when $t > 0$):

$$
w = \tfrac{1}{2}\sqrt{1+t},\quad
x = \frac{R_{21}-R_{12}}{4w},\quad
y = \frac{R_{02}-R_{20}}{4w},\quad
z = \frac{R_{10}-R_{01}}{4w}.
$$

For numerical stability when $t \le 0$, pick the largest diagonal element and
use the corresponding branch (this is what SciPy and Eigen do internally).

### Hamilton vs. JPL — a common pitfall

Two incompatible conventions exist:

- **Hamilton** (used by **Eigen**, **ROS/tf2**, **SciPy**, most math texts):
  right-handed $ijk$, $ij = k$.
- **JPL** (common in aerospace/filtering literature, e.g. some MEKF papers):
  left-handed $ijk$, $ij = -k$.

They differ by the sign of the vector part and by the order of composition, so a
quaternion copied from a JPL paper into Eigen will rotate the wrong way. Also
watch **storage order**: Eigen's `Quaterniond(w, x, y, z)` constructor takes $w$
first but stores $x, y, z, w$ internally, while ROS messages and SciPy's
`as_quat()` use $[x, y, z, w]$. Always confirm scalar-first vs. scalar-last.

This repo's helpers assume the Hamilton convention: see
`scripts/quaternion_utils.py` (basic ops and conversions),
`scripts/quaternions_relative_pose.py` (relative pose $q_{rel} = q_2 \otimes q_1^{-1}$),
and `scripts/quaternions_inverse_pose.py` (pose inversion).

## Code: converting between representations

### Python (SciPy)

```python
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# Build a rotation from yaw-pitch-roll (ZYX intrinsic), degrees.
# Capital letters => intrinsic; lowercase => extrinsic.
r = Rot.from_euler("ZYX", [90, 30, 10], degrees=True)

R = r.as_matrix()          # 3x3 rotation matrix (SO(3))
q = r.as_quat()            # [x, y, z, w]  (scalar-LAST, Hamilton)
rotvec = r.as_rotvec()     # axis * angle (Rodrigues vector)

# Round-trip back to Euler angles:
ypr = r.as_euler("ZYX", degrees=True)

# Compose (apply r1 first, then r2) and SLERP:
r2 = Rot.from_rotvec([0, 0, np.pi / 4])
r_total = r2 * r           # note operator order: r2 after r
from scipy.spatial.transform import Slerp
slerp = Slerp([0, 1], Rot.concatenate([r, r_total]))
r_mid = slerp(0.5)         # halfway orientation
```

### C++ (Eigen)

```cpp
#include <Eigen/Geometry>
using namespace Eigen;

// ZYX yaw-pitch-roll -> quaternion (Eigen is Hamilton, scalar-first ctor).
double yaw = M_PI / 2, pitch = 0.5, roll = 0.2;
Quaterniond q = AngleAxisd(yaw,   Vector3d::UnitZ())
              * AngleAxisd(pitch, Vector3d::UnitY())
              * AngleAxisd(roll,  Vector3d::UnitX());
q.normalize();

Matrix3d R = q.toRotationMatrix();        // quaternion -> matrix
Quaterniond q2(R);                        // matrix -> quaternion
Vector3d ypr = R.eulerAngles(2, 1, 0);    // back to yaw, pitch, roll
Quaterniond q_rel = q2 * q.inverse();     // relative rotation (Hamilton)
```

`cv::Rodrigues(rvec, R)` (and its inverse) bridges OpenCV's axis-angle `rvec`
used by `solvePnP`/`calibrateCamera` to and from the $3\times3$ matrix above.

## Refs

- [SciPy `Rotation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html)
- [Eigen: Space transformations](https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html)
- [J. Sola, *Quaternion kinematics for the error-state Kalman filter*](https://arxiv.org/abs/1711.02508) (Hamilton vs. JPL, conventions in depth)
