
import cv2
from utils.file_utils import resource_path


relative_path = "../images/00/image_2/000000.png"

frame1_path = resource_path(relative_path)

image = cv2.imread(frame1_path, cv2.IMREAD_COLOR)


H, W, C = image.shape


#
#         (x = column= u) ->  x is  [0, width-1]
#     ------------------------------------------►u
#     | (0,0) (1,0) (2,0) (3,0)
#     | (0,1) (1,1) (2,1) (3,1)
#     | (0,2) (1,2) (2,2) (3,2)
#     |
#     |
#     ▼
#     v  (y = row=v) -> y is  [0, height-1]
#


x_start, y_start = int(W/2), int(H/2)
x_end, y_end = W, 0

pt1 = (x_start, y_start)
pt2 = (x_end, y_end)


red, green, blue = 0, 255, 255
color = (red, green, blue)
thickness = 2
line_type = cv2.LINE_AA  # Anti-aliased line
shift = 0
tipLength = 0.1  # Length of the arrow tip in relation to the arrow length


img = cv2.arrowedLine(image, pt1, pt2, color, thickness,
                      line_type, shift, tipLength)


cv2.imshow('Arrow from center to top right', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
