# OpenCV Projects


![alt text](https://img.shields.io/badge/license-BSD-blue.svg) ![build workflow](https://github.com/behnamasadi/OpenCVProjects/actions/workflows/docker-build.yml/badge.svg)  

This project contains my Computer Vinson Projects with OpenCV.

## Building and Installation
### 1. Building the Image
There is docker file for this project that contains all dependencies and you build the image with :   

`docker build -t myopencv_image:latest .`

### 2. Creating the container
Create a container where you mount the checkout code into your container: 

`docker run --name <continer-name> -v <checked-out-path-on-host>:<path-in-the-container> -it <docker-image-name>`

for instance:

`docker run --name myopencv_container -v /home/$USER/workspace/OpenCVProjects:/OpenCVProjects -it myopencv_image`

### 3. Starting an existing container
If you have already created a container from the docker image, you can start it with:

`docker start -i myopencv_container`

### 4. Removing  unnecessary images and containers
You can remove unnecessary images and containers by:

`docker image prune -a`

`docker container prune` 

### GUI application with docker
1. You need to run:

`xhost +local:docker `  

and then: 

`docker run -v <path-on-host>:<path-in-the-container> -v /tmp/.X11-unix:/tmp/.X11-unix --name <container-name> -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 --network=host --privileged -it <image-name:tag>  bash  `

read more [here](https://ros-developer.com/2017/11/08/docker/)


# How to build on your machine 
configure it:

`cmake -G "Ninja Multi-Config"  -S . -B build -DOpenCV_DIR="/home/$USER/usr/lib/cmake/opencv4"`

build it:

`cmake --build build --config Release`

or 

`cmake --build build --config Debug`

or be more specific:

`cmake --build build --target all --config Release`

If you prefer `preset` use:

`cmake --preset ninja-multi`

and 

`cmake --build --preset ninja-multi-debug`

or 

`cmake --build --preset ninja-multi-release`

# Installing OpenCV Python API

Install the conda packages:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
conda create -n OpenCVProjects
conda activate OpenCVProjects
conda install conda-forge::opencv
conda install scipy numpy matplotlib scikit-image  
conda install -c conda-forge jupyterlab
conda install anaconda::ipywidgets
conda install -c conda-forge rerun-sdk
```

create softlinks: 

```
cd $HOME/anaconda3/envs/OpenCVProjects/
ln -s $HOME/workspace/OpenCVProjects/scripts .
ln -s $HOME/workspace/OpenCVProjects/images/ .
```

# [Computer Vision](#)
[Edge Detection](docs/edge_detection.md)  
[Image Filtering Convolution](docs/image_filtering_convolution.md)  
[Keypoints (Feature2D) Detector and Descriptor](docs/keypoints_detector_and_descriptor.ipynb)  
[Correspondence Problem, Keypoints Matching](docs/correspondences_matching.ipynb)  
[Image Matching WebUI](https://huggingface.co/spaces/Realcat/image-matching-webui)  
[Histogram Analysis](docs/histogram_analysis.md)  
[Pinhole Camera Model and Projection](docs/pinhole_camera_model_projection_intrinsic.md)  
[Direct Linear Transformation](docs/direct_linear_transformation.md)  
[Zhang's Camera Calibration Algorithm](docs/camera_calibration_algorithm.ipynb)  
[Extrinsic Calibration of Camera to LIDAR](https://www.dropbox.com/s/arhpp59d502fuks/2023IROS_fu.pdf?dl=0)   
[Image Resolution and Calibration Parameter](docs/image_resolution_and_calibration_parameter.md)  
[Affine Transformation](docs/affine_transformation.md)  
[Perspective Transformation](docs/perspective_transform.md)  
[Homography Transformation](docs/homography.md)  
[Lense Distortion Modeling, Image Undistortion](docs/lense_distortion.md)  
[Triangulation](docs/triangulation.md)  
[Epipolar Geometry, Essential Matrix, Fundamental Matrix](docs/epipolar_geometry_essential_matrix_fundamental_matrix.ipynb)  
[Extraction of Camera Motion](docs/extraction_of_camera_motion.md)  
[Image Rectification](docs/image_rectification.md)  
[Image Registration](docs/image_registration.md)  
[Stereo Vision, Disparity Map](docs/stereo_vision_disparity_map.md)  
[Stereo Calibration](docs/stereo_calibration.md)  
[Parallax](docs/parallax.md)  
[Sparse Optical Flow, Dense Optical Flow](docs/opticalflow.ipynb)
[Visual Odometry](docs/visual_odometry.md)  
[KITTI Visual Odometry](docs/kitti_visual_odometry.md)      
[Ego Motion Estimation](docs/ego_motion.md)  
[Recover Pose and Pose Estimation](docs/recover_pose_pose_estimation.md)  
[Laser Triangulation](docs/laser_triangulation.md)  
[Bayes Filter, Estate Estimation with Kalman Filter, Extended Kalman Filter](docs/kalman_filter.md)  
[Bag of Words](docs/bag_of_words.md)   
[Vocabulary Tree](docs/vocabulary_tree.md)    
[Perspective-n-point](docs/perspective_n_point.md)  
[3D World Unit Vector Camera Direction Vectors Camera Projection Rays](docs/3d_world_unit_vector.md)  
[Undistorting Points and Images](docs/undistorting.md)  
[Hand Eye Calibration](docs/hand_eye_calibration.md)  
[Normalized Image Coordinates](docs/normalized_image_coordinates.md)  
[Camera Models](docs/camera_models.md#camera-models)  
[Perspective Camera](docs/camera_models.md#perspective-camera)  
[Kannala-Brandt Fisheye Camera](docs/camera_models.md#kannala-brandt-fisheye-camera)  
[Spherical Camera](docs/camera_models.md#spherical-camera)  
[Color Space](docs/color_spaces.md)   
[Color Calibration](docs/color_calibration.md)  
[Dynamic Range](docs/dynamic_range.md)    
[White Balance](srcipts/white_balance.ipynb)    
[Decompose Projection Matrix](docs/decomposition_matrix.ipynb#Decompose-Essential-Matrix)  
[Decompose Homography Matrix](docs/decomposition_matrix.ipynb#decomposition_matrix.ipynb#Decompose-Homography-Matrix)  
[Decompose Essential Matrix](docs/decomposition_matrix.ipynb#Decompose-Projection-Matrix)  

# [OpenCV API](#)
[Basic Operations](src/basic_operations.cpp)  
[Coordinate System and Points](src/coordinate_system_points.cpp)  
[File Storage IO](src/file_storage_io.cpp)  
[Feature Detection, Feature Description](src/feature_detection_description.cpp)
[Correspondence Problem, Keypoints Matching](src/correspondences_matching.cpp)    
[Image Moments](src/image_moments.cpp)  
[PCA Principal Component Analysis](src/pca.cpp)  
[Morphological Transformation](src/morphological_transformation.cpp)  
[Hough Transform](src/hough_transform.cpp)  
[Homogeneous Conversion](src/homogeneous_conversion.cpp)  
[Camera Calibration](src/virtual_camera_calibration.cpp)  
[Projection Matrix, Camera Intrinsic](src/camera_projection_matrix.cpp)  
[Perspective-n-point](src/perspective-n-point.cpp)  
[3D World Unit Vector](src/camera_projection_matrix.cpp)  
[Essential Matrix Estimation](src/essential_matrix_estimation.cpp)  
[Fundamental Matrix Estimation](src/fundamental_matrix_estimation.cpp)  
[Rodrigue, Rotation Matrices](src/rodrigue_rotation_matrices.cpp)  
[Tracking Objects by Color](src/tracking_objects_by_color.cpp)  
[Image Remaping](src/remap.cpp)  
[Undistorting Images](src/undistorting_images.cpp)  
[Rectifying Images](src/rectifying_images.cpp)  
[Triangulation](src/triangulation.cpp)  
[ICP Iterative Closest Point](src/icp.cpp)  
[Structured Light Range Finding](src/structured_light_range_finding.cpp)  
[Estate Estimation with Kalman Filter](src/kalman_filter.cpp)  
[Drawing Frame Axes](src/drawing_frame_axes.cpp)  
[Writing video by choosing encoder using ffmpeg and libav](https://github.com/dataplayer12/video-writer)  
[Plot Frame poses in 3d](docs/plot_frame_poses_in_3d.md)  
[Euler Angle, Roll,Pitch, Yaw, Quaternion](docs/euler_quaternions.md)  


# [FFmpeg Commands](#)  
[Available encoders and decoders](docs/ffmpeg.md#available-encoders-and-decoders)  
[1. FFmpeg Common Options](docs/ffmpeg.md#1-ffmpeg-common-options)  
[2. FFmpeg Filters](docs/ffmpeg.md#2-ffmpeg-filters)  
  [2.1 Available Filters](docs/ffmpeg.md#21-available-filters)  
  [2.2 Send Output of FFmpeg Directly to FFplay](docs/ffmpeg.md#22-send-output-of-ffmpeg-directly-to-ffplay)  
  [2.3 Apply a Filter](docs/ffmpeg.md#23-apply-a-filter)  
  [2.4 Filter Graph](#docs/ffmpeg.md24-filter-graph)  
  [2.5 Filter Complex](docs/ffmpeg.md#25-filter-complex)  
  [2.6 Filter Complex vs Filter Graph](docs/ffmpeg.md#26-filter-complex-vs-filter-graph)  
  [2.7 Setting Number of Threads](docs/ffmpeg.md#27-setting-number-of-threads)  
  [2.8 Commonly Used FFmpeg Filters](docs/ffmpeg.md#28-commonly-used-ffmpeg-filters)  
[3. ffprobe](docs/ffmpeg.md#3-ffprobe)  
[4. ffmpeg metadata](docs/ffmpeg.md#4-ffmpeg-metadata)  
[5. Setting encoder for a specific codec](docs/ffmpeg.md#5-setting-encoder-for-a-specific-codec)  
[6. Set the format (container) and codec for the output](docs/ffmpeg.md#6-set-the-format--container--and-codec-for-the-output)  
[7. map](docs/ffmpeg.md#7-map)  
[8. graph2dot](docs/ffmpeg.md#graph2dot)  
[9. Determining Pixel Format](docs/ffmpeg.md#8-determining-pixel-format)  

# [Virtual Camera](#)
[Virtual Camera](scripts/virtual_camera.py)

# [Bundle adjustment and Structure from motion](#)
[Colmap](docs/colmap.md)  
[Photogrammetry](docs/photogrammetry_bundle_adjustment_structure_from_motion_reprojection_error.md#photogrammetry)  
[Structure from Motion (SfM)](docs/photogrammetry_bundle_adjustment_structure_from_motion_reprojection_error.md#structure-from-motion--sfm-)  
[Bundle Adjustment](docs/photogrammetry_bundle_adjustment_structure_from_motion_reprojection_error.md#bundle-adjustment)  
[Noah Snavely reprojection error](docs/photogrammetry_bundle_adjustment_structure_from_motion_reprojection_error.md#noah-snavely-reprojection-error)  

  
Refs: [1](https://www.youtube.com/channel/UCf0WB91t8Ky6AuYcQV0CcLw/videos),[2](https://github.com/spmallick/learnopencv/blob/master/README.md),[3](http://graphics.cs.cmu.edu/courses/15-463/),[4](https://www.tangramvision.com/blog/camera-modeling-exploring-distortion-and-distortion-models-part-i)
