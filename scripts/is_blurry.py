import cv2
import numpy as np


def is_blurry(image_path, threshold=300):
    """
    Determine if the image is blurry based on the variance of the Laplacian.

    Parameters:
    - image_path (str): Path to the image.
    - threshold (float): Variance threshold below which the image is considered blurry.

    Returns:
    - bool: True if the image is blurry, False otherwise.
    """
    image = cv2.imread(
        image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    print(laplacian_var)
    return laplacian_var < threshold


image_path = '/home/behnam/Pictures/colmap_projects/GroupeE_Maigrauge/images/9395.0.png'
if is_blurry(image_path):
    print("The image is blurry.")
else:
    print("The image is not blurry.")
