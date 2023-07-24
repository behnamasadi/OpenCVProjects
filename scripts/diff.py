import cv2
import numpy as np


def is_blurry(image, threshold=100):
    """
    Determine if the image is blurry based on the variance of the Laplacian.

    Parameters:
    - image_path (str): Path to the image.
    - threshold (float): Variance threshold below which the image is considered blurry.

    Returns:
    - bool: True if the image is blurry, False otherwise.
    """

    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var


frame_index = 0
w = 1920
h = 1080

threshold = 4.0

clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))


# Load the video
cap = cv2.VideoCapture(
    '/home/behnam/Pictures/colmap_projects/GroupeE_Maigrauge/video/video_1.mp4')


path = "/home/behnam/Pictures/colmap_projects/GroupeE_Maigrauge/images/"
print(f"number of frames:  ", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"frame rate: ", cap.get(cv2.CAP_PROP_FPS))
print(f"resolution is:  ", cap.get(
    cv2.CAP_PROP_FRAME_WIDTH),     cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)


# Read the first frame
ret, prev_frame = cap.read()


prev_frame = cv2.resize(prev_frame, (w, h))

if not ret:
    print("Error reading the video.")
    exit()

while True:
    # Read the next frame
    ret, frame = cap.read()

    # print(cap.get(cv2.CAP_PROP_POS_MSEC))

    frame = cv2.resize(frame, (w, h))

    if not ret:
        break

    # Compute the absolute difference between the current and the previous frame
    diff = cv2.absdiff(prev_frame, frame)

    # Convert the difference to grayscale
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Compute the score as the mean of the differences
    score = np.mean(gray_diff)

    # print(f"Change Score: {score}")

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (00, 100)

    # fontScale
    fontScale = 2

    # Red color in BGR
    color = (255, 255, 255)

    # Line thickness of 2 px
    thickness = 2

    gray_diff = cv2.putText(gray_diff, str(score), org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

    # Display the difference for visualization
    cv2.imshow('Difference', gray_diff)
    frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if (score > threshold):
        gray_image = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        image_clahe = clahe.apply(gray_image)

        # cv2.imwrite(path+str(frame_id)+".png", image_clahe)
        blur, val = is_blurry(image_clahe, threshold=300)

        if (not blur):
            file_extension = ".png"
            new_file_name = f"{frame_index:05d}{file_extension}"
            cv2.imwrite(path+new_file_name, image_clahe)
            frame_index = frame_index+1
        else:
            print(f"The image is blurry: ", val)

    # Use the current frame as the previous frame for the next iteration
    prev_frame = frame.copy()

    # Exit if 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
