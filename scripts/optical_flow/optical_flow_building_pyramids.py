import cv2
import numpy as np

# Constructs the image pyramid which can be passed to calcOpticalFlowPyrLK.

# Load an image

image_path = "/home/behnam/workspace/OpenCVProjects/images/lena.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Parameters for pyramid
maxLevel = 3   # Number of pyramid layers including the initial image
winSize = (15, 15)  # Window size

# Create an empty pyramid (using list comprehension)
# pyramid = [np.zeros_like(img) for _ in range(maxLevel + 1)]


# withDerivatives:set to precompute gradients for the every pyramid level. If pyramid is constructed without the gradients then calcOpticalFlowPyrLK will calculate them internally.


########################################## examples how to use it properly: ##########################################
# https://cpp.hotexamples.com/examples/-/-/calcOpticalFlowPyrLK/cpp-calcopticalflowpyrlk-function-examples.html
################################################################################################
# Build the pyramid
ret, pyramid = cv2.buildOpticalFlowPyramid(
    img, winSize, maxLevel, withDerivatives=False, pyrBorder=cv2.BORDER_REFLECT)

# Display the pyramid layers
for level, img_pyramid in enumerate(pyramid):
    cv2.imshow(f'Pyramid Level {level}', img_pyramid)
    cv2.waitKey(0)

cv2.destroyAllWindows()
