import cv2
import numpy as np
from utils.file_utils import resource_path


frame1_path = resource_path("../images/00/image_2/000000.png")
frame2_path = resource_path("../images/00/image_2/000001.png")

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


u = flow[..., 0]
v = flow[..., 1]


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


# --- 5. Visualize dense flow with arrows ---
# Sample every k-th pixel to avoid clutter
step = 20  # sample every 20 pixels
frame_with_arrows = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)

for y in range(0, h, step):
    for x in range(0, w, step):
        # Get flow vector at this pixel
        fx = u[y, x]
        fy = v[y, x]

        # Skip if flow is too small (noise)
        if np.sqrt(fx**2 + fy**2) < 1.0:
            continue

        # Calculate end point
        x_end = int(x + fx)
        y_end = int(y + fy)

        # Draw arrow from current position to flow destination
        cv2.arrowedLine(frame_with_arrows, (x, y), (x_end, y_end),
                        (0, 255, 0), 1, tipLength=0.3)
        # Draw circle at starting point
        cv2.circle(frame_with_arrows, (x, y), 2, (0, 0, 255), -1)


# --- 6. Warp prev frame using the flow ---
# Create meshgrid of coordinates
h, w = prev.shape
y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

# Add flow to coordinates to get new positions
map_x = x_coords + u
map_y = y_coords + v

# Warp prev frame: move each pixel to its new location according to flow
warped_prev = cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR)

# Calculate difference to see warping error
diff = cv2.absdiff(next, warped_prev)

# --- 7. Display results ---
cv2.imshow('1. Previous frame', prev)
cv2.imshow('2. Next frame', next)
cv2.imshow('3. Optical flow (HSV)', flow_bgr)
cv2.imshow('4. Optical Flow (Dense - Arrows)', frame_with_arrows)
cv2.imshow('5. Warped Previous frame', warped_prev)
cv2.imshow('6. Difference (Next - Warped Prev)', diff)
cv2.waitKey(0)
cv2.destroyAllWindows()
