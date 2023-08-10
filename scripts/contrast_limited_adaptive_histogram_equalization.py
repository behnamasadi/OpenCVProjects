import numpy as np
import cv2
import matplotlib.pyplot as plt


path_base = "/home/behnam/workspace/OpenCVProjects"
image_path = "/images/Unequalized_Hawkes_Bay_NZ.jpg"
# image_path = "/images/00001.png"


image = cv2.imread(path_base+image_path,  cv2.IMREAD_GRAYSCALE)

image_equalize_hist = cv2.equalizeHist(image)


# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
image_clahe = clahe.apply(image)


plt.figure(figsize=(18, 6))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")


plt.subplot(2, 3, 2)
plt.imshow(image_equalize_hist, cmap="gray")
plt.title("Equalized Image")


plt.subplot(2, 3, 3)
plt.imshow(image_clahe, cmap='gray')
plt.title("image_clahe")


plt.subplot(2, 3, 4)
plt.hist(image.ravel(), 256, [0, 256])
plt.title("image hist")


plt.subplot(2, 3, 5)
plt.hist(image_equalize_hist.ravel(), 256, [0, 256])
plt.title("image_equalize_hist")

plt.subplot(2, 3, 6)
plt.hist(image_clahe.ravel(), 256, [0, 256])
plt.title("image_clahe")

plt.tight_layout()
plt.show()
