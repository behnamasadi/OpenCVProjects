import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path_base = "/home/behnam/workspace/OpenCVProjects"
# image_path = "/images/lena.jpg"
image_path = "/images/Unequalized_Hawkes_Bay_NZ.jpg"
# image_path = "/images/pca_rectangle.png"


image = cv.imread(path_base+image_path,  cv.IMREAD_GRAYSCALE)

images = [image]
channels = [0]
mask = None
histSize = [256]
ranges = [0, 256]

hist = cv.calcHist(images, channels, mask, histSize, ranges)
cv.imshow("Histogram", image)

cv.waitKey(0)


# hist = np.linalg.norm(hist)
cdf = np.cumsum(hist)

print(type(hist))
print(hist.shape)


for bin in hist:
    print(bin.shape)
    print(type(bin))


plt.plot(hist)
plt.plot(cdf)

plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()


# Normalize the image
normalized_image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)

# Display the original and normalized images
cv.imshow("Original Image", image)
cv.imshow("Normalized Image", normalized_image)
cv.waitKey(0)
cv.destroyAllWindows()


equalizeHist_image = cv.equalizeHist(image)
cv.imshow("Original Image", image)
cv.imshow("equalized histogram  Image", equalizeHist_image)
cv.waitKey(0)
cv.destroyAllWindows()

# https://docs.opencv.org/3.4/d1/db7/tutorial_py_histogram_begins.html
