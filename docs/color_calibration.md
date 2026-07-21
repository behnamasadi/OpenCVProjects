# Color Calibration

Color calibration (a.k.a. color correction / color constancy) maps the **measured**
RGB values produced by a camera to a **reference** color space so that the same
physical surface produces the same numbers regardless of the sensor and the
illumination under which it was captured.

## Why we need it

The raw response of a camera to a scene depends on three things that have nothing
to do with the "true" color of the object:

- **Spectral sensitivity of the sensor.** Each camera's R, G, B filters integrate
  the incoming spectrum differently, so two cameras rarely agree on the same patch.
- **Illuminant.** A surface lit by tungsten (~3000 K) reflects a very different
  spectrum than the same surface under daylight (~6500 K).
- **Pipeline nonlinearities.** Gamma encoding, tone curves, and white-balance gains
  applied in the ISP further distort the numbers.

The goal is a **device-independent, illuminant-normalized** representation: feed in
measured RGB, get out RGB (or XYZ/Lab) that matches a standard reference.

## Color checker charts

The ground truth comes from a physical target with patches of *known* reference
values. The standard is the **X-Rite / Macbeth ColorChecker** with **24 patches**
(18 colored + a 6-step neutral gray ramp). The manufacturer publishes the reference
sRGB / XYZ / Lab values of each patch under a standard illuminant (commonly D50 or
D65).

Workflow: photograph the chart under the target illumination, detect the 24 patches,
average the pixels inside each patch to get 24 measured RGB triplets, and pair them
with the 24 published reference triplets. Those correspondences drive the fit.

## The correction model (CCM)

The simplest and most common model is a **linear 3×3 Color Correction Matrix (CCM)**
mapping measured RGB to reference RGB:

$$
\mathbf{r}_\text{ref} = M^\top \, \mathbf{r}_\text{meas}, \qquad M \in \mathbb{R}^{3\times3}
$$

Stack the $N$ patch measurements as rows of $R_\text{meas} \in \mathbb{R}^{N\times3}$
and the references as rows of $R_\text{ref} \in \mathbb{R}^{N\times3}$. The CCM is the
least-squares solution

$$
M = \arg\min_{M} \; \lVert R_\text{ref} - R_\text{meas}\,M \rVert_F^2
$$

with the closed form (normal equations)

$$
M = \left(R_\text{meas}^\top R_\text{meas}\right)^{-1} R_\text{meas}^\top R_\text{ref}.
$$

### Adding an affine offset

A pure linear map forces black to map to black. To also absorb a bias (e.g. sensor
black level, veiling flare), append a constant column of ones to the measurements and
solve for a $4\times3$ matrix:

$$
\hat{R}_\text{meas} = \begin{bmatrix} R_\text{meas} & \mathbf{1} \end{bmatrix}
\in \mathbb{R}^{N\times4}, \qquad
M = \left(\hat R_\text{meas}^\top \hat R_\text{meas}\right)^{-1}\hat R_\text{meas}^\top R_\text{ref}
\in \mathbb{R}^{4\times3}.
$$

In Python this is just `M, *_ = np.linalg.lstsq(R_meas, R_ref, rcond=None)`.

### Higher-order / polynomial correction

When the sensor response cannot be captured by a linear map (metamerism, strong
nonlinearity), extend the feature vector with polynomial terms before the same
least-squares fit. For example a degree-2 model expands each RGB triplet to

$$
[\,R,\;G,\;B,\;R^2,\;G^2,\;B^2,\;RG,\;GB,\;RB,\;1\,]
$$

and solves for a $10\times3$ matrix. This is "root-polynomial" / polynomial color
correction. It reduces residual error at the cost of more parameters and a higher
risk of overfitting or extrapolating badly outside the fitted gamut — keep the degree
low (2–3) and validate on held-out patches.

## Linearize first (gamma)

**The CCM is a linear operation and must be fit in linear light.** Camera output is
usually gamma-encoded (e.g. sRGB's ~2.2 transfer function). If you fit the matrix on
gamma-encoded values the linear model is applied to a nonlinear signal and the fit is
biased. The correct order is:

1. **Linearize** the measured values (invert the camera/sRGB transfer curve, ideally
   from a *raw* or de-gamma'd image) so intensities are proportional to scene
   radiance.
2. Fit and apply the CCM in linear space.
3. **Re-encode** (re-apply gamma / output transfer function) for display.

The neutral gray patches of the chart are useful for estimating/verifying the
linearization curve.

## White balance: the simple special case

White balance is a **diagonal** special case of the CCM: instead of a full $3\times3$
mix it applies one **per-channel gain** so that a neutral surface reads equal in R, G,
B.

$$
M_\text{wb} = \operatorname{diag}(g_R,\, g_G,\, g_B)
$$

The classic **gray-world** assumption sets the gains so the average of each channel is
equal; **white-patch / max-RGB** normalizes by the brightest values. This corrects the
*illuminant* cast but not the sensor's cross-channel mixing, which is why a full CCM
does better on saturated colors. See [`scripts/wb.py`](../scripts/wb.py) in this repo
for gray-world / white-patch implementations.

## OpenCV support (`cv::ccm`, opencv_contrib)

OpenCV ships color-calibration tooling in the **`opencv_contrib`** extra modules
(not in the core build — you must build/install `opencv-contrib-python` or compile
contrib):

- **`cv::mcc`** — Macbeth ColorChecker detection (`CCheckerDetector`) that locates the
  chart and extracts the 24 patch colors from an image.
- **`cv::ccm`** — `ColorCorrectionModel`, which fits a CCM from measured patch colors
  against a built-in reference chart (e.g. `COLORCHECKER_Macbeth`), supports linear/
  polynomial models, choice of linearization (gamma / polynomial), the working color
  space, and a distance metric for the fit.

Minimal Python sketch (API names may shift between versions; treat as illustrative):

```python
import cv2
import numpy as np

# src: Nx1x3 float array of measured patch colors in [0,1] (e.g. from mcc detector)
src = measured_patches.astype(np.float64) / 255.0

model = cv2.ccm.ColorCorrectionModel(src, cv2.ccm.COLORCHECKER_Macbeth)
model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
model.setCCM_TYPE(cv2.ccm.CCM_3x3)          # or CCM_4x3 for the affine offset
model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
model.setLinearGamma(2.2)
model.run()

ccm = model.getCCM()                        # the fitted 3x3 (or 4x3) matrix

# apply to a linearized, RGB, float image in [0,1]
img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
calibrated = model.infer(img)
```

> Requires `opencv_contrib`. Because the exact enum/method names have changed across
> releases, check `dir(cv2.ccm.ColorCorrectionModel)` for your installed version.

## Evaluating the result: ΔE in CIELAB

RMS error in RGB is a poor proxy for perceived color error. Evaluate in **CIELAB**,
which is approximately perceptually uniform, using the **color difference ΔE**:

$$
\Delta E^*_{ab} = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}
$$

between each calibrated patch and its reference (newer metrics ΔE94 / ΔE2000 weight
the channels more perceptually). A rule of thumb: $\Delta E \lesssim 1$ is
imperceptible, $\Delta E \approx 2\text{–}3$ is good for most work. Report the mean
and max ΔE over the 24 patches, and ideally over patches **not** used in the fit to
detect overfitting.

## Practical checklist

- Shoot the chart flat, evenly lit, no glare, filling a good part of the frame.
- Use raw or properly de-gamma'd data; **fit in linear light**, re-encode for output.
- Start with a linear 3×3 (or 4×3 affine); only go polynomial if ΔE demands it.
- Refit whenever the illuminant or camera settings change.
- Validate with ΔE (CIELAB), ideally on held-out patches.

## Refs

- OpenCV `ccm` (Color Correction Model): https://docs.opencv.org/4.x/d1/dc4/tutorial_ccm_color_correction_model.html
- OpenCV `mcc` (Macbeth ColorChecker detection): https://docs.opencv.org/4.x/dd/d19/group__mcc.html
- `ColorCorrectionModel` API reference: https://docs.opencv.org/4.x/d0/d6b/classcv_1_1ccm_1_1ColorCorrectionModel.html
