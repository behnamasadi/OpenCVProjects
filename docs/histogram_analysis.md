# Normalization
The linear normalization of a gray-scale digital image:

<!-- 
<img src="https://latex.codecogs.com/svg.latex?I_{N}=( {\text{newMax}}-{\text{newMin}}   ){\frac  {   I-{\text{Min}}  }{{\text{Max}}-{\text{Min}}}}+{\text{newMin}}" />
-->

<img src="https://latex.codecogs.com/svg.image?I_{N}=(&space;{\text{newMax}}-{\text{newMin}}&space;&space;&space;){\frac&space;&space;{&space;&space;&space;I-{\text{Min}}&space;&space;}{{\text{Max}}-{\text{Min}}}}&plus;{\text{newMin}}" title="https://latex.codecogs.com/svg.image?I_{N}=( {\text{newMax}}-{\text{newMin}} ){\frac { I-{\text{Min}} }{{\text{Max}}-{\text{Min}}}}+{\text{newMin}}" />


# Contrast Stretching
Purpose: To stretch the range of pixel values of an image to span the entire 0 to 255 scale (for 8-bit images) or a desired range.
Method: The minimum pixel value in the image is mapped to 0 (or a desired min), and the maximum pixel value is mapped to 255 (or a desired max). Intermediate values are linearly stretched between these extremes.
Usage: Particularly useful for images that have pixel values in a limited range due to issues like underexposure or overexposure. By stretching this range, the contrast of the image is improved.

Refs [1](https://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm)




# Histogram Normalization

Purpose: To adjust the pixel values of an image such that they conform to a desired histogram shape, often a uniform or normal distribution.
Method: This involves mapping the cumulative distribution function (CDF) of the image's histogram to the CDF of the desired histogram.
Usage: Used when we want an image to have a specific distribution of pixel values. For instance, if we want an image to have a uniform distribution of pixel values, histogram normalization can achieve this.

# Histogram Stretching

Purpose: This term is often used interchangeably with Contrast Stretching. However, if a distinction is made, Histogram Stretching typically refers to a technique where the aim isn't necessarily to stretch the histogram to the full available range but to a desired range.
Method: Similar to contrast stretching, but instead of mapping to the full 0 to 255 range, you might map the image pixel values to a specific desired range.
Usage: Useful when the aim isn't to maximize contrast but to ensure the image pixel values lie within a specific range.
In essence, while all three methods aim to adjust the range and distribution of pixel values in an image, they do so based on different principles:

**Contrast Stretching**: focuses on maximizing the contrast by using the full intensity scale.
**Histogram Normalization** tries to shape the histogram of the image to a desired distribution.
**Histogram Stretching** adjusts the pixel values to lie within a desired range, not necessarily the full scale.
In many image processing contexts, Contrast Stretching and Histogram Stretching might be used interchangeably, but it's always good to clarify the specific method and purpose when discussing or implementing these techniques.


# Histogram Matching: 
Creating new image which has new distribution function (pdf)


# Histogram Equalization:
Creating new image which has new uniform distribution. The primary idea behind HE is to adjust the intensity values of an image such that they're uniformly distributed across the entire range. **HE** works well for images where the subject and background are both underexposed or overexposed. But for images with varying brightness levels, applying HE can over-amplify the contrast in some regions, making details hard to discern.


Refs: [1](https://automaticaddison.com/difference-between-histogram-equalization-and-histogram-matching/)

# Adaptive Histogram Equalization (AHE):
To counter the issues of standard HE, AHE was introduced.
Instead of applying HE on the whole image, AHE divides the image into smaller, non-overlapping regions or tiles. HE is then applied to each of these small tiles.
This approach ensures that the equalization is adaptive and local, taking into consideration only the data in that tile. Thus, it can bring out details in regions with different brightness levels.
However, AHE has a drawback: If there's noise in the image, it can get amplified in the tiles, since noise can produce local intensity peaks.
Contrast Limited Adaptive Histogram Equalization (CLAHE):

# Contrast Limited Adaptive Histogram Equalization (CLAHE) 
Contrast Limited Adaptive Histogram Equalization (CLAHE) is an advanced method for improving the contrast of images. It's an extension of the standard histogram equalization (HE) method, which aims to enhance image contrast by redistributing the intensities of pixels. Here's a step-by-step breakdown of CLAHE:

1. Histogram Equalization (HE):
2. Adaptive Histogram Equalization (AHE):

CLAHE is an improvement over AHE.
The main difference is that in CLAHE, the histogram equalization is applied with a contrast limit. If any histogram bin is above this specified limit, the excess is clipped off and distributed among other bins.
This contrast limiting reduces the amplification of noise in the image.
After the contrast-limiting step, the neighboring tiles are combined using bilinear interpolation to remove any artificially induced boundaries, making the enhancement smooth.


## Histogram Equalization Vs Histogram Normalization
Numbers are scaled such that either the maximum
count / bin or cumulative bin count equals a specific number
(usually 1). Keeps the shape of the histogram, but modifies
the numbers / bin.

Equalization: The bins are re-distributed such that the
histogram becomes flatter. Changes the shape of the histogram.


# Gamma Correction

the formula for normalization sometime is written as:
new_x=a+ (b-a)*[(x-A)/(B-A)]^Gamma

because always 
0< (x-A)/(B-A) < 1 
so if Gamma is bigger than one then the result became smaller and if Gamma if smaller than 1 it became bigger


