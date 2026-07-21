# Color Spaces

A color space is a coordinate system for representing color numerically. The
same physical color can be encoded in many spaces; each makes a different
property (device output, perceptual distance, luma/chroma separation) easy to
work with. This note covers the spaces you meet most often in OpenCV.

## RGB / BGR

RGB stores a color as additive **R**ed, **G**reen, **B**lue channels, matching
how displays emit light. Each channel is typically an 8-bit value in
$[0, 255]$.

**Gotcha:** OpenCV stores color images as **BGR**, not RGB. `cv::imread`,
`cv::imwrite`, and the drawing functions all assume BGR memory layout, whereas
Matplotlib, PIL, and most of the rest of the world assume RGB. Displaying a
BGR array with Matplotlib swaps red and blue (skies turn orange). Convert
explicitly when crossing that boundary:

```python
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # for Matplotlib / PIL
```

BGR/RGB are device-dependent: the same triple can look different on different
monitors. The spaces below trade that away for other useful properties.

## Grayscale

Grayscale collapses color to a single intensity channel. OpenCV uses the
**ITU-R BT.601** luma weights, which approximate human luminance sensitivity
(the eye is most sensitive to green, least to blue):

$$
Y = 0.299\,R + 0.587\,G + 0.114\,B
$$

A plain channel average ($\tfrac{R+G+B}{3}$) ignores this and looks wrong —
skin and foliage come out muddy. Use `COLOR_BGR2GRAY`, which applies the
weights above.

## HSV and HSL

Both separate **color** from **brightness**, which makes them far more robust
than RGB for color-based segmentation under changing illumination.

- **Hue** — the color itself, an angle on the color wheel ($0^\circ$ red,
  $120^\circ$ green, $240^\circ$ blue).
- **Saturation** — colorfulness / purity, from gray ($0$) to vivid.
- **Value** (HSV) — brightness of the brightest channel.
- **Lightness** (HSL) — midpoint brightness; pure colors sit at $L = 0.5$.

**OpenCV H range:** for 8-bit images Hue is stored as $0$–$179$ (degrees
halved so the angle fits in a byte), while S and V span $0$–$255$. For 32-bit
float images H is the full $0$–$360$. This halved range trips up thresholds
copied from other tools — a "red at 0–360" reference must be scaled.

Typical use: threshold in HSV to isolate a colored object. Note that red wraps
around the hue circle, so it needs two ranges:

```python
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
lower1, upper1 = (0, 120, 70),   (10, 255, 255)
lower2, upper2 = (170, 120, 70), (179, 255, 255)
mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
```

## YCrCb / YUV

These split a **luma** channel (Y, brightness) from two **chroma** channels
(Cr = red-difference, Cb = blue-difference). YUV is the analog-broadcast
sibling; YCbCr is its digital form and what JPEG/MPEG use.

$$
Y = 0.299\,R + 0.587\,G + 0.114\,B, \quad
C_r = R - Y, \quad C_b = B - Y
$$

(scaled and offset in practice). Two reasons this matters:

- **Chroma subsampling.** Human vision resolves brightness detail far better
  than color detail, so codecs keep Y at full resolution and downsample the
  chroma planes (4:2:0, 4:2:2). This is a large, nearly-free compression win
  and the main reason YCbCr exists in image/video formats.
- **Skin detection.** Skin tones cluster tightly in the Cr–Cb plane roughly
  independently of brightness, so a fixed chroma box is a cheap, effective
  skin classifier.

Use `COLOR_BGR2YCrCb` (note OpenCV's Cr-before-Cb channel order).

## Lab (CIELAB)

CIELAB is a **perceptually uniform** space: a fixed Euclidean distance
corresponds to a roughly constant perceived color difference anywhere in the
space — something RGB and HSV badly fail at. Three channels:

- **L\*** — lightness, $0$ (black) to $100$ (white).
- **a\*** — green ($-$) to red ($+$).
- **b\*** — blue ($-$) to yellow ($+$).

Because distance $\approx$ perceived difference, color difference is just a
Euclidean norm (CIE76 $\Delta E$):

$$
\Delta E_{76} = \sqrt{(\Delta L^{\ast})^2 + (\Delta a^{\ast})^2 + (\Delta b^{\ast})^2}
$$

$\Delta E \approx 2.3$ is the just-noticeable-difference threshold. (Later
CIEDE2000 refines this with weighting terms.) Lab is the space of choice for
quality-control color matching and, because L\* isolates lightness, for
**color-constancy** and white-balance work. Use `COLOR_BGR2Lab`.

> Scaling note: for 8-bit images OpenCV packs L\* into $0$–$255$
> ($L \leftarrow L \times \tfrac{255}{100}$) and a\*, b\* into $0$–$255$ with a
> $+128$ offset. For 32-bit float they keep their native $[0,100]$ /
> $[-127,127]$ ranges.

## XYZ (CIE)

The **CIE 1931 XYZ** space is the device-independent master reference from
which the others are derived. Its primaries are imaginary (chosen so all real
colors have non-negative coordinates), Y is defined to equal luminance, and it
models the response of a standard human observer. You rarely display XYZ, but
it is the mathematical hub: RGB→Lab conversions, for instance, route
$\text{RGB} \rightarrow \text{XYZ} \rightarrow \text{Lab}$. OpenCV exposes it
as `COLOR_BGR2XYZ`.

## Converting in OpenCV

All conversions go through a single function, `cv::cvtColor(src, dst, code)`,
with a `COLOR_<from>2<to>` enum. Codes are directional and reversible
(`COLOR_BGR2HSV` / `COLOR_HSV2BGR`).

**C++:**

```cpp
#include <opencv2/imgproc.hpp>

cv::Mat bgr = cv::imread("img.png");   // BGR by default
cv::Mat gray, hsv, lab, ycrcb;
cv::cvtColor(bgr, gray,  cv::COLOR_BGR2GRAY);
cv::cvtColor(bgr, hsv,   cv::COLOR_BGR2HSV);    // H in 0..179 (8-bit)
cv::cvtColor(bgr, lab,   cv::COLOR_BGR2Lab);
cv::cvtColor(bgr, ycrcb, cv::COLOR_BGR2YCrCb);
```

**Python:**

```python
import cv2

bgr = cv2.imread("img.png")            # BGR by default
gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)    # H in 0..179 (uint8)
lab   = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
```

## Summary table

| Space   | Channels        | Property                              | Typical use case                                   |
|---------|-----------------|---------------------------------------|----------------------------------------------------|
| RGB/BGR | R, G, B         | Additive, device-dependent (OpenCV=BGR)| Storage, display, drawing                          |
| Gray    | Y               | Single intensity (BT.601 luma)        | Edges, features, thresholding, most classical CV   |
| HSV     | H, S, V         | Color separated from brightness       | Color segmentation / masking (H = 0–179 in 8-bit)  |
| HSL     | H, S, L         | Color separated, symmetric lightness  | Color pickers, palette work                        |
| YCrCb   | Y, Cr, Cb       | Luma + chroma                         | JPEG/video compression, skin detection             |
| Lab     | L\*, a\*, b\*   | Perceptually uniform                  | Color difference (ΔE), color constancy, QC         |
| XYZ     | X, Y, Z         | Device-independent reference          | Conversion hub, colorimetry                        |

## Refs

- [OpenCV `cvtColor` and color conversion codes](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)
- [Wikipedia: Color space](https://en.wikipedia.org/wiki/Color_space)
- [Wikipedia: CIELAB color space](https://en.wikipedia.org/wiki/CIELAB_color_space)
