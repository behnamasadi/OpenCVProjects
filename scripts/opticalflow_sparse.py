import cv2
import numpy as np
from utils.file_utils import resource_path


frame1_path = resource_path("../images/00/image_2/000000.png")
frame2_path = resource_path("../images/00/image_2/000001.png")

prev = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
next = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)


# --- 2. Detect feature points in the first frame ---
# Parameters for Shi-Tomasi corner detection
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Detect good features to track in the previous frame
p0 = cv2.goodFeaturesToTrack(prev, mask=None, **feature_params)

# --- 3. Compute sparse optical flow using Lucas-Kanade ---
# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Calculate optical flow
p1, status, error = cv2.calcOpticalFlowPyrLK(
    prev, next, p0, None, **lk_params
)

# Select good points (status == 1 means point was found)
if p1 is not None:
    good_new = p1[status == 1]
    good_old = p0[status == 1]
else:
    good_new = []
    good_old = []

print("Images shape:", prev.shape)
print("Number of initial feature points:", len(p0))
print("Number of tracked points:", len(good_new))

# --- 4. Visualize the optical flow ---
# Create a color image to draw on (using previous frame)
frame_with_flow = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)

# Draw the tracks
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    a, b, c, d = int(a), int(b), int(c), int(d)

    # Draw arrowed line from old to new position
    cv2.arrowedLine(frame_with_flow, (c, d), (a, b),
                    (0, 255, 0), 2, tipLength=0.3)
    # Draw circle at old position (starting point)
    cv2.circle(frame_with_flow, (c, d), 5, (0, 0, 255), -1)

# --- 5. Display results ---
cv2.imshow('Optical Flow (Sparse - Lucas-Kanade)', frame_with_flow)

cv2.waitKey(0)
cv2.destroyAllWindows()
