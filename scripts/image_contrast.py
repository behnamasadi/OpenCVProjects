import cv2
import numpy as np
import matplotlib.pyplot as plt

# https://pixelcraft.photo.blog/2022/06/01/the-image-histogram-vi-contrast-and-clipping/
# https://docs.opencv.org/3.4/d1/db7/tutorial_py_histogram_begins.html


# Read the image


path_base = "/home/behnam/workspace/OpenCVProjects"
# image_path = "/images/lena.jpg"
image_path = "/images/Unequalized_Hawkes_Bay_NZ.jpg"
# image_path = "/images/pca_rectangle.png"


# Convert the image to grayscale
image = cv2.imread(path_base+image_path, cv2.IMREAD_GRAYSCALE)

# Calculate the histogram
histogram, bins = np.histogram(image.ravel(), 256, [0, 256])

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.plot(histogram, color='black')
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
