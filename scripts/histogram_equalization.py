import numpy as np
import cv2
import matplotlib.pyplot as plt


def histogram_equalization(image):
    # Compute the histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Mask for pixels with zero probability
    cdf_masked = np.ma.masked_equal(cdf, 0)

    # Normalize the CDF to stretch its values to the full 0-255 range
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / \
        (cdf_masked.max() - cdf_masked.min())

    # Fill in the masked values with 0
    cdf = np.ma.filled(cdf_masked, 0).astype('uint8')

    # Map the original pixel values to the equalized values using the CDF
    equalized_image = cdf[image]

    return equalized_image, cdf


path_base = "/home/behnam/workspace/OpenCVProjects"
# image_path = "/images/lena.jpg"
# image_path = "/images/Unequalized_Hawkes_Bay_NZ.jpg"
image_path = "/images/00001.png"


image = cv2.imread(path_base+image_path,  cv2.IMREAD_GRAYSCALE)


equalized_image, cdf = histogram_equalization(image)


hist_equalized_image, bins = np.histogram(
    equalized_image.flatten(), 256, [0, 256])

# Compute the cumulative distribution function (CDF)
cdf_hist_equalized_image = hist_equalized_image.cumsum()


plt.figure(figsize=(18, 6))

plt.subplot(3, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.autoscale()


plt.subplot(3, 2, 2)
plt.imshow(equalized_image, cmap="gray")
plt.title("Equalized Image")
plt.autoscale()


plt.subplot(3, 2, 3)
plt.plot(cdf, color='black')
plt.title("CDF")
plt.xlim([0, 255])
plt.ylim([0, cdf[-1]])  # Maximum value of CDF for the y limit
plt.xlabel('Pixel Intensity')
plt.ylabel('Cumulative Count')


plt.subplot(3, 2, 4)
plt.plot(cdf_hist_equalized_image, color='black')
plt.title("cdf_hist_equalized_image")
plt.xlim([0, 255])
# Maximum value of cdf_hist_equalized_image for the y limit
plt.ylim([0, cdf_hist_equalized_image[-1]])
plt.xlabel('Pixel Intensity')
plt.ylabel('Cumulative Count')


plt.subplot(3, 2, 5)
plt.hist(image.ravel(), 256, [0, 256])


plt.subplot(3, 2, 6)
plt.hist(equalized_image.ravel(), 256, [0, 256])


plt.tight_layout()
plt.show()
