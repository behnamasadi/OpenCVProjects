import cv2
import os


def image_stream_from_directory(directory_path):
    """
    Generator to yield images from a directory in order
    """
    for filename in sorted(os.listdir(directory_path)):
        # add more formats if required
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(directory_path, filename)
            yield cv2.imread(filepath)


directory_path = '/home/behnam/workspace/OpenCVProjects/data/kitti/05/image_0'

for image in image_stream_from_directory(directory_path):
    if image is not None:
        cv2.imshow('Image Stream', image)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
