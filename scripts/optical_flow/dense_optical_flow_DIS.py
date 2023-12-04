import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

# pip install opencv-python
try:
    vid_source = sys.argv[1]
except:
    vid_source = "/dev/video0"


cap = cv2.VideoCapture(vid_source)

if not cap.isOpened():
    print("Error opening video file!")
    exit()

ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)


dis = cv2.DISOpticalFlow_create()

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = dis.calc(prev_frame_gray, frame_gray, None)

    # Visualization of the flow
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("Optical Flow", bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if not ret:
        break
    cv2.imshow('', frame_gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    prev_frame = frame_gray.copy()
