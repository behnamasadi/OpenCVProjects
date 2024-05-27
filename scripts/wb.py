












input='/media/behnam/behnam/video.mp4'
output='/media/behnam/behnam/output_video.mp4'



import cv2
import numpy as np

# Function to adjust white balance
def simple_white_balance(frame):
    result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

# Function to apply median filter
def apply_median_filter(frame, kernel_size=5):
    return cv2.medianBlur(frame, kernel_size)

# Function to resize frame with maintaining the aspect ratio
def resize_frame(image, max_width=1920, max_height=1080):
    height, width = image.shape[:2]
    scale = min(max_width/width, max_height/height)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Load your video
cap = cv2.VideoCapture(input)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


# Define the codec and create VideoWriter object using H.265
fourcc = cv2.VideoWriter_fourcc(*'H', 'E', 'V', 'C')
out = cv2.VideoWriter(output, fourcc, fps, (width * 2, height))  # Double the width for side-by-side
frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    print(f"Processing frame number: {frame_number}")

    # Adjust white balance
    frame_wb = simple_white_balance(frame)

    # Apply median filter
    frame_filtered = apply_median_filter(frame_wb)

    # Concatenate the original and processed frames
    comparison = np.hstack((frame, frame_filtered))

    # Resize for display
    comparison_display = resize_frame(comparison)

    # Show side by side
    cv2.imshow('Before and After', comparison_display)

        # Write the frame to the output file
    out.write(comparison)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the video display before it ends
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
