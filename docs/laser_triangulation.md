# Laser Triangulation

Laser triangulation is an *active* range-sensing technique: a known light source
(a laser point or a laser line) and a camera are mounted a fixed distance apart.
The laser projects onto the scene, the camera observes where the spot/stripe
lands, and because the geometry of the emitter–camera pair is known, the range
to the illuminated surface point is recovered by solving a triangle. It is the
optical analogue of stereo vision, but one of the two "cameras" is replaced by a
controlled emitter, which removes the correspondence problem — you always know
which feature in the image is "the laser."

## Principle

<img src="images/configuration_of_a_Laser_Triangulation_System.png" />

The camera and the laser are separated by a fixed **baseline** $b$. The laser
ray leaves the emitter at a known angle and strikes the object; the reflected
spot is imaged by the camera at some pixel. As the object moves closer or
farther, the imaged spot slides across the sensor. That image displacement (a
*disparity*) is the measurement, and it maps monotonically to range $Z$. Because
the light source is controlled, detection is robust: the brightest peak along
the relevant image direction is the laser, so no descriptor matching is needed.

Two common configurations exist:

- **Fixed (parallel) laser**, ray parallel to the optical axis or at a fixed
  known angle — depth comes directly from the horizontal offset of the spot.
- **Steered laser**, where a mirror/galvo sweeps the beam and the angle $\theta$
  is commanded and known — depth comes from the angle plus the imaged position.

## Geometry and the triangulation equation

Consider a pinhole camera with focal length $f$ (in pixels) and the laser
emitter offset by baseline $b$ along the camera's $x$-axis. Let a scene point be
at range $Z$ (distance along the optical axis).

**Disparity form.** If the laser ray is parallel to the optical axis, an object
at range $Z$ produces a spot whose image offset $x$ (in pixels, measured from
the position it would occupy at infinity, i.e. the principal point for a
parallel ray) obeys the same relation as stereo disparity:

$$Z = \frac{b \, f}{x}$$

where

- $Z$ — range to the surface point (same length units as $b$),
- $b$ — baseline, the emitter–camera separation,
- $f$ — camera focal length in **pixels**,
- $x$ — image displacement (disparity) of the laser spot in pixels.

Small $x$ (spot near the vanishing position) means large $Z$; the sensitivity
$\partial Z / \partial x = -b f / x^2 = -Z^2/(b f)$ grows with $Z^2$, so
precision degrades quadratically with distance and improves with a larger
baseline or longer focal length.

**Angular form.** If instead the laser is steered to a known angle $\theta$
(measured from the baseline) and the camera observes the spot along a line of
sight at angle $\alpha$, the two rays and the baseline form a triangle. For the
simple case where the camera looks along the optical axis and the laser makes
angle $\theta$ with the baseline,

$$Z = b \, \tan(\theta)$$

for a laser mounted perpendicular-referenced to the baseline, or, using both
observed angles via the law of sines,

$$Z = b \, \frac{\tan\theta \, \tan\alpha}{\tan\theta + \tan\alpha}.$$

Here $\theta$ is the (known, commanded) laser angle and $\alpha$ is the camera
bearing to the spot, obtained from the imaged pixel and the intrinsics via
$\alpha = \arctan\!\big((x - c_x)/f\big)$, with $c_x$ the principal point.

In practice one fits/absorbs all of this into a calibrated model rather than
using the idealized formula directly, but the equations show the essential
$Z \propto b$ scaling and the $Z^2$ error growth.

## Sheet-of-light (laser-stripe) scanning

A single laser *point* yields one range sample per frame. To digitize a surface
efficiently, the point is spread into a **line** with a cylindrical lens,
producing a *sheet of light* — a plane in 3D. The intersection of this light
plane with the object is a bright **stripe**, and the camera images a whole
profile (one range value per image column) in a single exposure.

Reconstruction is a **plane–ray intersection**:

1. Each illuminated pixel $(u, v)$ back-projects to a camera ray
   $\mathbf{r}(t) = t \, K^{-1} [u, v, 1]^\top$, where $K$ is the intrinsic
   matrix.
2. The calibrated laser plane is $\mathbf{n}^\top \mathbf{X} + d = 0$ in the
   camera frame.
3. Substituting the ray into the plane equation solves for the unique $t$,
   giving the 3D point $\mathbf{X} = \mathbf{r}(t)$.

Sweeping the sheet across the object (by moving the object on a conveyor/stage,
or by panning the laser/sensor) and stacking the per-frame profiles yields a
full 3D point cloud or range image. This is the workhorse of industrial 3D
inspection and many commercial "profile" sensors.

## Calibration

Two calibrations are required:

- **Camera intrinsics** — $f$, principal point $(c_x, c_y)$, and lens
  distortion, obtained with a standard checkerboard / `cv::calibrateCamera`.
  Depth accuracy depends directly on these.
- **Laser plane calibration** — the plane $(\mathbf{n}, d)$ of the light sheet
  in the camera frame. A common method: place a planar target at several known
  poses, detect the laser stripe on it, back-project the stripe pixels onto the
  (known) target plane to get 3D points that lie in the laser plane, then fit a
  plane by least squares (SVD) to the accumulated points across all poses.

For a steered-point system, the analogue is calibrating the angle-to-range
mapping (or the emitter pose and per-command angle).

## Practical notes

- **Sub-pixel stripe localization.** The stripe has finite width; taking the
  single brightest pixel is noisy. Estimate the peak per column with a
  **center-of-mass (intensity centroid)** of the profile,
  $\hat{v} = \dfrac{\sum_i v_i \, I(v_i)}{\sum_i I(v_i)}$, or a Gaussian /
  parabolic fit to the intensity around the peak. This routinely gives
  1/10–1/50 pixel precision and is the single biggest lever on accuracy.
- **Exposure and laser power.** Set exposure so the stripe is bright but not
  clipped/saturated — a saturated (flat-topped) profile destroys the centroid
  estimate. Match power to surface reflectance; specular and very dark surfaces
  are the hard cases.
- **Ambient light rejection.** Use a narrow **band-pass optical filter** matched
  to the laser wavelength (e.g. 650 nm) plus short exposure, so the sensor sees
  mostly laser light. This is why the technique works on a factory floor.
- **Occlusion / shadowing.** Because emitter and camera are separated by $b$,
  some surfaces the laser hits are not visible to the camera (and vice versa),
  leaving gaps — a larger baseline improves accuracy but worsens occlusion.
- **Speckle** from coherent laser light adds noise to the stripe location; it
  sets a floor on achievable precision.

## Relationship to structured light

Laser triangulation with a swept stripe is really the single-line member of the
broader **structured light** family. Structured-light scanners project a *coded*
2D pattern (stripes, gray codes, phase-shifted sinusoids) so that many surface
points are triangulated per frame without mechanical sweeping — trading a
projector and pattern-decoding for the laser's mechanical scan. The triangulation
math (baseline, plane/ray or pattern-code correspondence, then intersection) is
the same. See the repo example
[`src/structured_light_range_finding.cpp`](../src/structured_light_range_finding.cpp).

## Refs

- [Wikipedia — Laser triangulation / 3D scanner](https://en.wikipedia.org/wiki/3D_scanner#Triangulation)
- [HALCON sheet-of-light 3D reconstruction (MVTec docs)](https://www.mvtec.com/doc/halcon/1911/en/toc_sheet_of_light.html)
- [Steger, Ulrich, Wiedemann, *Machine Vision Algorithms and Applications* — sheet-of-light chapter](https://onlinelibrary.wiley.com/doi/book/10.1002/9783527802096)
