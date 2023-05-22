# OpenCV Projects



![alt text](https://img.shields.io/badge/license-BSD-blue.svg) ![build workflow](https://github.com/behnamasadi/OpenCVProjects/actions/workflows/docker-image.yml/badge.svg)  

This project contains my Computer Vinson Projects with OpenCV.



## Building and Installation
### 1. Building the Image
There is docker file for this project where contains all dependencies and you build the image with :   

`docker build -t myopencv_image:latest .`

### 2. Creating the container
Create a container where you mount the checkout code into your container: 

`docker run --name <continer-name> -v <checked-out-path-on-host>:<path-in-the-container> -it <docker-image-name>`

for instance:

`docker run --name myopencv_container -v /home/behnam/workspace/OpenCVProjects:/OpenCVProjects -it myopencv_image`

### 3. Starting an existing container
If you have already created a container from the docker image, you can start it with:

`docker start -i myopencv_container`

### 4. Removing  unnecessary images and containers
You can remove unnecessary images and containers by:

`docker image prune -a`

`docker container prune` 


### GUI application with docker
1. You need to run:

`docker run --name myopencv_container -v /home/behnam/workspace/OpenCVProjects:/OpenCVProjects --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"  -it myopencv_image`

2. On the host run the following (every time you run your container):

<code>export containerId=$(docker ps -l -q)  
<code>  xhost +local: docker inspect --format='{{ .Config.Hostname }}' $containerId </code>


read more [here](https://ros-developer.com/2017/11/08/docker/)

# Using OpenCV Python API
Install the conda packages:

`conda install -c conda-forge opencv`


# [Computer Vision](#)
   * [Edge Detection](docs/edge_detection.md)
   * [Histogram Analysis](docs/histogram_analysis.md)
   * [Pinhole Camera Model and Projection](docs/projection_camera_intrinsic.md)
   * [Direct Linear Transformation](docs/direct_linear_transformation.md)
   * [Zhang's Camera Calibration Algorithm](docs/zhang_camera_calibration_algorithm.md)
   * [Image Resolution and Calibration Parameter](docs/image_resolution_and_calibration_parameter.md)
   * [Affine Transformation](docs/affine_transformation.md)
   * [Perspective Transformation](docs/perspective_transform.md)
   * [Homography Transformation](docs/homography.md)
   * [Lense Distortion Modeling, Image Undistortion](docs/lense_distortion.md)
   * [Triangulation](docs/triangulation.md)
   * [Epipolar Geometry](docs/epipolar_geometry.md)
   * [Essential Matrix](docs/essential_matrix.md)
   * [Extraction of Camera Motion](docs/extraction_of_camera_motion.md)
   * [Fundamental Matrix](docs/fundamental_matrix.md)
   * [Image Rectification](docs/image_rectification.md)
   * [Correspondence Problem](docs/correspondence_problem.md)
   * [Stereo Vision](docs/stereo_vision.md)
   * [Photogrammetry](docs/photogrammetry.md)
   * [Structure From Motion](docs/structure_from_motion.md)
   * [Parallax](docs/parallax.md)
   * [Optical Flow](docs/optical_flow.md)
   * [Visual Odometry](docs/visual_odometry.md)
   * [Ego Motion](docs/ego-motion.md)
   * [Recover Pose and Pose Estimation](docs/recover_pose_pose_estimation.md)
   * [Laser Triangulation](docs/laser_triangulation.md)
   * [Estate Estimation with Kalman Filter](docs/kalman_filter.md)
   * [Bag of Words](docs/bag_of_words.md)
   * [Perspective-n-point](docs/perspective_n_point.md)
   * [3D World Unit Vector Camera Direction Vectors Camera Projection Rays](docs/3d_world_unit_vector.md)
   * [Undistorting Points and Images](docs/undistorting.md)
   * [Hand Eye Calibration](docs/hand_eye_calibration.md)
   * [Normalized Image Coordinates](docs/normalized_image_coordinates.md)
   * [Camera Models](docs/camera_models.md#camera-models)
     * [Perspective Camera](docs/camera_models.md#perspective-camera)
     * [Kannala-Brandt Fisheye Camera](docs/camera_models.md#kannala-brandt-fisheye-camera)
     * [Spherical Camera](docs/camera_models.md#spherical-camera)
   * [Procrustes Analysis](docs/shape_analysis.md#procrustes-analysis)
      * [Wahba's problem](docs/shape_analysis.md#wahba-s-problem)
      * [Quaternion estimator algorithm (QUEST)](docs/shape_analysis.md#quaternion-estimator-algorithm--quest-)
      * [Kabsch Algorithm](docs/shape_analysis.md#kabsch-algorithm)
      * [Umeyama Algorithm](docs/shape_analysis.md#umeyama-algorithm)
      * [Iterative Closest Point (ICP)](docs/shape_analysis.md#iterative-closest-point--icp-)
      * [Difference between Kabsch, Procrustes, Umeyama and ICP Algorithm](docs/shape_analysis.md#difference-between-kabsch--procrustes--umeyama-and-icp-algorithm)
    *[Color Space CIE 1931](https://en.wikipedia.org/wiki/CIE_1931_color_space)  

# [OpenCV API](#)
   * [Basic Operations](src/basic_operations.cpp)
   * [Coordinate System and Points](src/coordinate_system_points.cpp)
   * [File Storage IO](src/file_storage_io.cpp)
   * [Blob Detection](src/blob_detection.cpp)
   * [Corner Detection](src/corner_detection.cpp)
   * [Feature Detection](src/feature_detection.cpp)
   * [Feature Description](src/feature_description.cpp)
   * [Image Moments](src/image_moments.cpp)
   * [PCA Principal Component Analysis](src/pca.cpp)
   * [Morphological Transformation](src/morphological_transformation.cpp)
   * [Hough Transform](src/hough_transform.cpp)
   * [Homogeneous Conversion](src/homogeneous_conversion.cpp)
   * [Camera Calibration](src/virtual_camera_calibration.cpp)
   * [Projection Matrix, Camera Intrinsic](src/camera_projection_matrix.cpp)
   * [Perspective-n-point](src/perspective-n-point.cpp)
   * [3D World Unit Vector](src/camera_projection_matrix.cpp)
   * [Essential Matrix Estimation](src/essential_matrix_estimation.cpp)
   * [Fundamental Matrix Estimation](src/fundamental_matrix_estimation.cpp)
   * [Rodrigue, Rotation Matrices](src/rodrigue_rotation_matrices.cpp)
   * [Tracking Objects by Color](src/tracking_objects_by_color.cpp)
   * [Image Remaping](src/remap.cpp)
   * [Undistorting Images](src/undistorting_images.cpp)
   * [Rectifying Images](src/rectifying_images.cpp)
   * [Triangulation](src/triangulation.cpp)
   * [ICP Iterative Closest Point](src/icp.cpp)
   * [Structured Light Range Finding](src/structured_light_range_finding.cpp)
   * [Estate Estimation with Kalman Filter](src/kalman_filter.cpp)
   * [Drawing Frame Axes](src/drawing_drame_axes.cpp)
   * [Writing video with choosing encoder using ffmpeg and libav](https://github.com/dataplayer12/video-writer)

# [FFmpeg Commands](#)
- [Available encoders and decoders](docs/ffmpeg.md#available-encoders-and-decoders)
- [Conversion between codecs](docs/ffmpeg.md#conversion-between-codecs)
- [Change the output resolution](docs/ffmpeg.md#change-the-output-resolution)
- [ffmpeg metadata](docs/ffmpeg.md#ffmpeg-metadata)
  * [Using ffmpeg to copy metadata](docs/ffmpeg.md#using-ffmpeg-to-copy-metadata)
  * [Image metadata exif](docs/ffmpeg.md#image-metadata-exif)
- [Extracting key frames](docs/ffmpeg.md#extracting-key-frames)
- [Extracting scene-changing frames](docs/ffmpeg.md#extracting-scene-changing-frames)
- [Rotating video](docs/ffmpeg.md#rotating-video)

# [Virtual Camera](#)
   * [Virtual Camera](scripts/virtual_camera.py)

# [Bundle adjustment and Structure from motion](#)
- [Colmap](colmap/colmap.md)

Refs: [1](https://www.youtube.com/channel/UCf0WB91t8Ky6AuYcQV0CcLw/videos),[2](https://github.com/spmallick/learnopencv/blob/master/README.md),[3](http://graphics.cs.cmu.edu/courses/15-463/),[4](https://www.tangramvision.com/blog/camera-modeling-exploring-distortion-and-distortion-models-part-i)  
