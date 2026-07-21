# Dynamic Range and HDR Imaging

## Definition

The **dynamic range** of an imaging system (or a scene) is the ratio between
the largest and smallest measurable light intensity:

$$
\text{DR} = \frac{I_\text{max}}{I_\text{min}}
$$

where $I_\text{max}$ is the brightest signal before saturation (the *full-well
capacity* of a sensor pixel) and $I_\text{min}$ is the smallest signal
distinguishable from noise (the *noise floor*). Because this ratio spans many
orders of magnitude it is quoted logarithmically:

- **Decibels:** $\text{DR}_{\text{dB}} = 20 \log_{10}\!\left(\dfrac{I_\text{max}}{I_\text{min}}\right)$
- **Stops (EV):** $\text{DR}_{\text{stops}} = \log_2\!\left(\dfrac{I_\text{max}}{I_\text{min}}\right)$

One *stop* is a factor of 2, so $6.02\ \text{dB} \approx 1\ \text{stop}$.

We distinguish two quantities:

- **Sensor dynamic range** — a property of the hardware. A typical consumer
  CMOS sensor delivers roughly $60$–$70\ \text{dB}$ (≈ 10–12 stops); a good
  DSLR reaches ~14 stops.
- **Scene dynamic range** — a property of the world. A sunlit outdoor scene
  with deep shadows can span $>20$ stops ($>120\ \text{dB}$), far exceeding
  what any single exposure of the sensor can capture.

<img src="images/dynamic_range.png" />

## Why 8-bit LDR Images Clip

A standard **Low Dynamic Range (LDR)** image stores 8 bits per channel, i.e.
only $2^8 = 256$ levels — about $48\ \text{dB}$ or 8 stops of *encoded* range,
and gamma-encoded rather than linear. When a real scene exceeds the exposure
window chosen by the camera:

- Pixels brighter than $I_\text{max}$ **saturate** and are written as `255`
  (blown highlights — a white sky with no cloud detail).
- Pixels darker than the noise floor collapse to `0` (crushed shadows).

Once clipped, the information is *gone*: no amount of post-processing recovers
detail from a region of solid `255` or `0`. A single exposure simply slides an
8-stop window up or down the scene's 20-stop range. **HDR imaging** widens that
window by combining several exposures.

## Building an HDR Radiance Map from Exposure Bracketing

The idea is to shoot the same static scene at several **exposure times**
$\Delta t_j$ (a *bracket*), then fuse them into a single linear **radiance
map** $E_i$ (irradiance per pixel $i$).

### The Camera Response Function (CRF)

A camera does not record scene radiance linearly. The measured pixel value is

$$
Z_{ij} = f\big(E_i\,\Delta t_j\big)
$$

where $E_i \Delta t_j$ is the *exposure* at pixel $i$ for shot $j$, and $f$ is
the nonlinear, monotonic **Camera Response Function** (sensor response ×
gamma × tone curve). To recover linear radiance we must invert $f$.

### Debevec & Malik recovery

Taking $g = \ln f^{-1}$, the model becomes linear in logs:

$$
g(Z_{ij}) = \ln E_i + \ln \Delta t_j
$$

Debevec & Malik (1997) recover the discrete curve $g(z)$ for all
$z \in [0,255]$ and the radiances $\ln E_i$ simultaneously by least squares,
minimizing

$$
\mathcal{O} =
\sum_{i}\sum_{j} \Big\{ w(Z_{ij})\big[g(Z_{ij}) - \ln E_i - \ln \Delta t_j\big]\Big\}^2
\;+\; \lambda \sum_{z} \big[w(z)\,g''(z)\big]^2
$$

- The first term enforces the model over all pixels and exposures.
- The second term ($\lambda$) is a **smoothness** prior on $g''$.
- $w(z)$ is a **hat weighting** that down-weights values near 0 and 255, which
  are unreliable (near clipping). The system is fixed up to a scale by
  anchoring the middle value $g(127)=0$.

Once $g$ is known, radiance is obtained by a weighted average over exposures:

$$
\ln E_i =
\frac{\sum_j w(Z_{ij})\,\big[g(Z_{ij}) - \ln \Delta t_j\big]}
     {\sum_j w(Z_{ij})}
$$

The result is a **32-bit float radiance map** (`CV_32FC3`) that is linear and
proportional to true scene irradiance.

## Tone Mapping

A radiance map holds far more range than any monitor (~8 stops) can show, so
**tone mapping** compresses HDR luminance back to a displayable [0,1] LDR image
while preserving local contrast. Common **tone-mapping operators (TMOs)**:

- **Reinhard** — photographic global/local operator; simple, natural results.
- **Drago** — adaptive logarithmic mapping, good for very high contrast.
- **Mantiuk** — perceptually-based contrast-domain operator; vivid, detailed.

The simplest Reinhard global form illustrates the compression:

$$
L_\text{display} = \frac{L}{1 + L}
$$

which maps $[0,\infty)$ into $[0,1)$. A **gamma** parameter then encodes the
result for the display.

## OpenCV API and Pipeline

OpenCV's `photo` module implements the full pipeline:

| Stage | OpenCV factory |
|-------|----------------|
| Recover CRF | `cv::createCalibrateDebevec` (also `...Robertson`) |
| Merge to radiance | `cv::createMergeDebevec` (also `...Robertson`) |
| Tone map | `cv::createTonemap`, `createTonemapReinhard`, `createTonemapDrago`, `createTonemapMantiuk` |
| Exposure fusion | `cv::createMergeMertens` (no CRF needed) |

### Python: bracket → CRF → HDR → tonemap

```python
import cv2
import numpy as np

# Bracketed exposures of a static scene + their shutter times (seconds)
files = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]
images = [cv2.imread(f) for f in files]
times  = np.array([1/30., 0.25, 2.5, 15.0], dtype=np.float32)

# 1. Recover the Camera Response Function (CRF)
calibrate = cv2.createCalibrateDebevec()
response  = calibrate.process(images, times)

# 2. Merge exposures into a 32-bit linear radiance map
merge_debevec = cv2.createMergeDebevec()
hdr = merge_debevec.process(images, times, response)   # CV_32FC3
cv2.imwrite("scene.hdr", hdr)                           # Radiance .hdr

# 3. Tone map to a displayable LDR image
tonemap = cv2.createTonemapReinhard(gamma=2.2)
ldr = tonemap.process(hdr)                              # float [0,1]
cv2.imwrite("ldr_reinhard.jpg", np.clip(ldr * 255, 0, 255).astype("uint8"))
```

### C++ equivalent (core calls)

```cpp
#include <opencv2/photo.hpp>

std::vector<cv::Mat> images;          // loaded brackets
std::vector<float>   times;           // matching exposure times

cv::Mat response, hdr, ldr;
cv::createCalibrateDebevec()->process(images, response, times);
cv::createMergeDebevec()->process(images, hdr, times, response);
cv::createTonemapReinhard(2.2f)->process(hdr, ldr);   // ldr is CV_32FC3 in [0,1]
```

## Exposure Fusion (Mertens) — the Simpler Alternative

If you only want a good-looking LDR result and do **not** need a physically
linear radiance map, **exposure fusion** (Mertens et al.) skips CRF recovery
and tone mapping entirely. It blends the brackets directly in the LDR domain
using per-pixel quality weights (contrast, saturation, well-exposedness) in a
Laplacian-pyramid, so **exposure times are not even required**:

```python
merge_mertens = cv2.createMergeMertens()
fusion = merge_mertens.process(images)                 # float [0,1], display-ready
cv2.imwrite("fusion.jpg", np.clip(fusion * 255, 0, 255).astype("uint8"))
```

This is faster, needs no calibration, and is robust to unknown exposure
metadata — at the cost of not producing a reusable HDR radiance map.

## Refs

- OpenCV HDR imaging tutorial: <https://docs.opencv.org/4.x/d2/df0/tutorial_py_hdr.html>
- P. Debevec, J. Malik, *"Recovering High Dynamic Range Radiance Maps from Photographs"*, SIGGRAPH 1997: <https://www.pauldebevec.com/Research/HDR/>
- T. Mertens, J. Kautz, F. Van Reeth, *"Exposure Fusion"*, Pacific Graphics 2007: <https://mericam.github.io/papers/exposure_fusion_reduced.pdf>
