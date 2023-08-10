import cv2
import numpy as np
from matplotlib import pyplot as plt


def is_blurry(image, threshold=300):
    """
    Determine if the image is blurry based on the variance of the Laplacian.
    """

    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    print(laplacian)
    # Compute the variance
    laplacian_variance = np.var(laplacian)

    print(laplacian_variance)
    return laplacian_variance < threshold


project_root = "/home/behnam/workspace/OpenCVProjects/"

image_path = '/images/lena.jpg'

# Read image in grayscale
image = cv2.imread(project_root+image_path, cv2.IMREAD_GRAYSCALE)

assert image is not None, "check image path"


image_gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
image_median = cv2.medianBlur(image, 5)
image_bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)

mean = np.ones((5, 5), np.float32)/25
image_mean = cv2.filter2D(image, -1, mean)


plt.subplot(321), plt.imshow(image), plt.title('Original')
plt.xticks([]), plt.yticks([])


plt.subplot(322), plt.imshow(image_mean), plt.title(
    'mean, image is: ' + ("blury" if is_blurry(image_mean) else "sharp"))
plt.xticks([]), plt.yticks([])


plt.subplot(323), plt.imshow(image_median), plt.title(
    'median, image is: ' + ("blury" if is_blurry(image_median) else "sharp"))
plt.xticks([]), plt.yticks([])


plt.subplot(324), plt.imshow(
    image_bilateral_filter), plt.title('bilateral filter, image is: ' + ("blury" if is_blurry(image_bilateral_filter) else "sharp"))
plt.xticks([]), plt.yticks([])


plt.subplot(325), plt.imshow(
    image_gaussian_blur), plt.title('gaussian blur, image is: ' + ("blury" if is_blurry(image_gaussian_blur) else "sharp"))
plt.xticks([]), plt.yticks([])


plt.show()

if is_blurry(image):
    print("The image is blurry.")
else:
    print("The image is not blurry.")


if is_blurry(image_mean):
    print("The image is blurry.")
else:
    print("The image is not blurry.")

if is_blurry(image_gaussian_blur):
    print("The image is blurry.")
else:
    print("The image is not blurry.")
