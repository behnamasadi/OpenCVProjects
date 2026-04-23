# SuperPoint

**Paper:** DeTone, Malisiewicz, Rabinovich (Magic Leap, CVPRW 2018) — *SuperPoint: Self-Supervised Interest Point Detection and Description.*

A single CNN that, in one forward pass, both **finds keypoints** and **describes them** — trained **without any human-labeled keypoints**.

Architecture diagram: [viz_out/superpoint_layered.svg](viz_out/superpoint_layered.svg).

---

## 1. Why it exists

Classical detectors (SIFT, SURF, ORB, Harris) use hand-crafted rules for "what is a corner." Deep learning swept most of vision, but keypoint detection lagged because there is **no natural ground truth** — humans don't label corners on photographs the way they label objects. SuperPoint's contribution is a training recipe that generates its own keypoint labels, plus an architecture that makes detection and description share the same backbone.

Two practical wins:

- **One network, one pass.** Detection and description share ~90% of the compute via a shared encoder.
- **Full-resolution heatmap without transposed convs.** A depth-to-space trick on an 8× downsampled grid reconstructs a per-pixel keypoint probability map cheaply.

---

## 2. Architecture

Input: a grayscale image of shape `1 × H × W`.

Three pieces:

### 2.1 Shared VGG-style encoder

Four stages, each two `Conv3×3 → ReLU`, with `MaxPool2d(2)` between the first three stages (none after stage 4). Channel widths: `64, 64, 128, 128`.

| Stage | Convs       | After pool  | Output        |
|-------|-------------|-------------|---------------|
| 1     | `1→64→64`   | pool /2     | `64 × H/2 × W/2`   |
| 2     | `64→64→64`  | pool /2     | `64 × H/4 × W/4`   |
| 3     | `64→128→128`| pool /2     | `128 × H/8 × W/8`  |
| 4     | `128→128→128` | —         | `128 × H/8 × W/8`  |

Final feature map is **128 channels at 1/8 resolution**. This tensor is the shared input for both heads.

### 2.2 Detector head — the 65-channel depth-to-space trick

Two convs on the shared feature map:

```
128 → Conv3×3 → 256 → ReLU → Conv1×1 → 65
```

Output shape: `65 × H/8 × W/8`.

Why **65**? Treat each pixel of the coarse 1/8 grid as summarizing an **8×8 block** of the full-resolution image. The 65 channels are:
- **64 channels** = probability that the keypoint (if any) sits at position `(i, j)` within that 8×8 block, for each of the 64 possible positions.
- **1 "dustbin" channel** = probability that **no keypoint** exists in this block at all.

Inference:
1. `softmax` across the 65 channels → probability distribution per block.
2. **Drop the dustbin** channel → 64 channels.
3. **Depth-to-space** reshape: `64 × H/8 × W/8 → 1 × H × W`. Each 8×8 block of the output gets filled from the 64 probabilities of the corresponding coarse pixel.

Result: a full-resolution **`1 × H × W` keypoint probability heatmap** — no transposed conv needed. That's cheap and avoids checkerboard artifacts that plague transposed-conv upsampling.

Final keypoints are picked by **non-maximum suppression** (NMS) on this heatmap with a probability threshold.

### 2.3 Descriptor head — semi-dense descriptors

Two convs on the same shared feature map:

```
128 → Conv3×3 → 256 → ReLU → Conv1×1 → 256
```

Output shape: `256 × H/8 × W/8` — **semi-dense**: one 256-D descriptor per 8×8 block, not per pixel.

Inference:
1. **Bicubic upsample** `256 × H/8 × W/8 → 256 × H × W` (done on-demand, only at keypoint pixels).
2. **L2-normalize** each 256-D vector.
3. Sample at each keypoint location from the detector head.

Keeping descriptors semi-dense saves roughly **64×** the compute vs producing a dense `256 × H × W` descriptor map everywhere.

---

## 3. Self-supervised training — where the labels come from

SuperPoint can't be trained directly on natural images because there are no keypoint labels. The paper gets around this with a two-stage bootstrap:

### 3.1 MagicPoint — the synthetic pretrain

Render a huge dataset of **synthetic shapes** (cubes, checkerboards, stars, polygons) where the corners are **known by construction**. Train a CNN (the detector head on the shared encoder) on these synthetic images with the exact corner labels.

The resulting model ("MagicPoint") is great at finding corners of geometric shapes but **generalizes poorly** to real-world photos.

### 3.2 Homographic Adaptation — pseudo-labels on real images

The self-supervised trick. For each real image `I`:

1. Sample `N` random homographies `H_1, ..., H_N` (perspective warps, rotations, scales).
2. For each `H_i`: warp `I` by `H_i`, run MagicPoint on the warped image, then unwarp the detections back to the original image's coordinates.
3. **Aggregate** all N detection sets. Keep points that are found **consistently across many warps** — they're likely real corners, not noise.

This aggregated set of points is the **pseudo-label** for image `I`. Because a true keypoint should be invariant under reasonable viewpoint changes, this robustness filter does a remarkable job of surfacing stable interest points without any human input.

### 3.3 Final joint training

Now retrain the full SuperPoint (both heads) on real images (e.g. MS-COCO), using the Homographic-Adaptation pseudo-labels for the detector head and **pairs of images related by a known homography** for the descriptor head (contrastive: matching points = positive, non-matching = negative).

---

## 4. Loss function

Two terms, summed:

- **Detector loss** `L_p`: cross-entropy over the 65-way classification at each coarse cell, against the pseudo-label position (or "dustbin" if no keypoint in that cell).
- **Descriptor loss** `L_d`: a **hinge loss** over descriptor pairs between image `A` and its warped version `A' = H(A)`.
  - For every pair of coarse cells `(cell in A, cell in A')`, if they correspond under `H`, push their descriptors **together** (cosine similarity ≥ positive margin).
  - If they don't correspond, push them **apart** (cosine similarity ≤ negative margin).

The detector and descriptor losses are computed simultaneously on each image pair; gradients flow through the shared encoder from both.

---

## 5. Inference pipeline in one picture

```
grayscale image (1 × H × W)
        │
        ▼
┌──────────────────────┐
│   shared encoder     │   4 stages, 3 maxpools → 128 × H/8 × W/8
└──────────────────────┘
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  
┌──────────────┐   ┌───────────────┐
│ detector head│   │ descriptor hd │
│ 2 conv → 65  │   │ 2 conv → 256  │
└──────────────┘   └───────────────┘
        │                  │
  softmax(65)        bicubic upsample
   drop dustbin        L2-normalize
  depth-to-space
        │                  │
        ▼                  ▼
  1 × H × W           sample at each
  heatmap             keypoint location
        │
     NMS + threshold
        │
        ▼
  keypoint (x, y) list + matching 256-D descriptors
```

---

## 6. What's strong and what's weak

**Strong:**
- Real-time: one forward pass for both tasks; runs at 70+ FPS on a GPU for VGA-sized images.
- Repeatable: keypoints are consistent across viewpoint, illumination, and small non-rigid distortion.
- Self-supervised: no human keypoint labels anywhere in the pipeline.
- Dense on-demand: descriptors can be sampled anywhere bilinearly, not just at detected locations.

**Weak:**
- **Assumes rigid/homographic relationships** for descriptor training. Works well on structured scenes (buildings, documents, AR markers), less well on non-rigid scenes (people, animals).
- Grayscale-only by design — color information is discarded.
- **Per-pixel precision is bounded by the 8×8 block structure.** The depth-to-space trick gives full-res probabilities but the spatial "resolution" of the information is still at the 1/8 grid level; extreme sub-pixel accuracy relies on downstream refinement.
- Descriptors are only 256-D and trained contrastively on synthetic warps, so extreme out-of-distribution appearance (e.g. day↔night, different seasons) can still break matching. Successor works (e.g. LoFTR, DISK, DeDoDe, LightGlue for matching) target these cases.

---

## 7. Influence

SuperPoint became the de-facto learnable drop-in for SIFT in many SLAM/SfM pipelines: COLMAP, ORB-SLAM variants, and mobile AR stacks. Its **shared-encoder-plus-two-heads** pattern and **depth-to-space** upsampling were both reused heavily — LightGlue (matcher), SuperPoint + SuperGlue → SuperPoint + LightGlue, and descendants like DISK and DeDoDe all build on the same shared-encoder scaffolding.

---

## References

- DeTone, D., Malisiewicz, T., & Rabinovich, A. (2018). *SuperPoint: Self-Supervised Interest Point Detection and Description.* CVPR Workshops. [arXiv:1712.07629](https://arxiv.org/abs/1712.07629)
- Magic Leap official PyTorch release: https://github.com/magicleap/SuperPointPretrainedNetwork
- Related follow-ups: SuperGlue (CVPR 2020), LightGlue (ICCV 2023), DISK (NeurIPS 2020), LoFTR (CVPR 2021), DeDoDe (3DV 2024).
