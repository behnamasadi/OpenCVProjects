import cv2
import numpy as np
from utils.file_utils import resource_path


frame1_path = resource_path("../images/correspondences_matching/000000.png")
frame2_path = resource_path("../images/correspondences_matching/000001.png")

prev = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
next = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

# --- 2. Compute dense optical flow ---
flow = cv2.calcOpticalFlowFarneback(
    prev=prev,
    next=next,
    # output flow array (None -> allocate automatically)
    flow=None,
    pyr_scale=0.5,          # image scale (<1) for pyramid
    levels=3,               # number of pyramid levels
    winsize=15,             # averaging window size
    iterations=3,           # iterations per pyramid level
    poly_n=5,               # size of pixel neighborhood
    poly_sigma=1.2,         # std of Gaussian for derivatives
    flags=0                 # usually 0 or cv2.OPTFLOW_FARNEBACK_GAUSSIAN
)


# --- 3. flow is an array of shape (H, W, 2) ---
# flow[...,0] = horizontal displacement (u)
# flow[...,1] = vertical displacement   (v)

print("images shape:", prev.shape)
print("Flow shape:", flow.shape)
print("Example flow vector at center (u,v):", flow[50, 50])


# --- 4. Visualize optical flow using hue-saturation mapping ---
h, w = prev.shape
flow_magnitude, flow_angle = cv2.cartToPolar(
    flow[..., 0], flow[..., 1], angleInDegrees=True)

hsv = np.zeros((h, w, 3), np.uint8)
hsv[..., 0] = flow_angle / 2                     # Hue represents direction
hsv[..., 1] = 255                               # Full saturation
hsv[..., 2] = cv2.normalize(flow_magnitude, None, 0,
                            255, cv2.NORM_MINMAX)  # Value = magnitude
flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# The ellipsis (...) is shorthand for “fill in all remaining dimensions.” flow[:, :, 0]
map_x = flow[..., 0]
map_y = flow[..., 1]

cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR)

# --- 5. Display results ---
cv2.imshow('Previous frame', prev)
cv2.imshow('Next frame', next)
cv2.imshow('Optical flow (HSV)', flow_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
