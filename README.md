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
   * [Image Filtering Convolution](docs/image_filtering_convolution.md)
   * [Histogram Analysis](docs/histogram_analysis.md)
   * [Keypoints Detector and Descriptor](docs/keypoints_detector_and_descriptor.md)
   * [Pinhole Camera Model and Projection](docs/projection_camera_intrinsic.md)
   * [Direct Linear Transformation](docs/direct_linear_transformation.md)
   * [Zhang's Camera Calibration Algorithm](docs/zhang_camera_calibration_algorithm.md)
   * [Extrinsic Calibration of Camera to LIDAR](https://www.dropbox.com/s/arhpp59d502fuks/2023IROS_fu.pdf?dl=0)   
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
   * [Image Registration](docs/image_registration.md)
   * [Correspondence Problem](docs/correspondence_problem.md)
   * [Stereo Vision, Disparity Map](docs/stereo_vision_disparity_map.md)
   * [Parallax](docs/parallax.md)
   * [Optical Flow](docs/optical_flow.md)
   * [Visual Odometry](docs/visual_odometry.md)
   * [KITTI Visual Odometry](docs/kitti_visual_odometry.md)     
   * [Visual Inertial Odometry](docs/VIO.md)   
   * [Ego Motion](docs/ego_motion.md)
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
   * [Color Space](docs/color_spaces.md)  
   * [Color Calibration](docs/color_calibration.md)
   * [Dynamic Range](docs/dynamic_range.md)  
   * [White Balance](docs/white_balance.md)  
   * [Signal To Noise Ratio](docs/signal_to_noise_ratio.md)  


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
- [1. FFmpeg Common Options](docs/ffmpeg.md#1-ffmpeg-common-options)  
- [2. FFmpeg Filters](docs/ffmpeg.md#2-ffmpeg-filters)  
  * [2.1 Available Filters](docs/ffmpeg.md#21-available-filters)  
  * [2.2 Send Output of FFmpeg Directly to FFplay](docs/ffmpeg.md#22-send-output-of-ffmpeg-directly-to-ffplay)  
  * [2.3 Apply a Filter](docs/ffmpeg.md#23-apply-a-filter)  
  * [2.4 Filter Graph](#docs/ffmpeg.md24-filter-graph)  
  * [2.5 Filter Complex](docs/ffmpeg.md#25-filter-complex)  
  * [2.6 Filter Complex vs Filter Graph](docs/ffmpeg.md#26-filter-complex-vs-filter-graph)  
  * [2.7 Setting Number of Threads](docs/ffmpeg.md#27-setting-number-of-threads)  
  * [2.8 Commonly Used FFmpeg Filters](docs/ffmpeg.md#28-commonly-used-ffmpeg-filters)  
- [3. ffprobe](docs/ffmpeg.md#3-ffprobe)  
- [4. ffmpeg metadata](docs/ffmpeg.md#4-ffmpeg-metadata)  
- [5. Setting encoder for a specific codec](docs/ffmpeg.md#5-setting-encoder-for-a-specific-codec)  
- [6. Set the format (container) and codec for the output](docs/ffmpeg.md#6-set-the-format--container--and-codec-for-the-output)  
- [7. map](docs/ffmpeg.md#7-map)  
- [graph2dot](docs/ffmpeg.md#graph2dot)  
- [8. Determining Pixel Format](docs/ffmpeg.md#8-determining-pixel-format) 


# [Virtual Camera](#)
- [Virtual Camera](scripts/virtual_camera.py)

# [Bundle adjustment and Structure from motion](#)
- [Colmap](docs/colmap.md)  
- [Photogrammetry](docs/photogrammetry_bundle_adjustment_structure_from_motion_reprojection_error.md#photogrammetry)  
- [Structure from Motion (SfM)](docs/photogrammetry_bundle_adjustment_structure_from_motion_reprojection_error.md#structure-from-motion--sfm-)  
- [Bundle Adjustment](docs/photogrammetry_bundle_adjustment_structure_from_motion_reprojection_error.md#bundle-adjustment)  
- [Noah Snavely reprojection error](docs/photogrammetry_bundle_adjustment_structure_from_motion_reprojection_error.md#noah-snavely-reprojection-error)  


  

# [SLAM](#)
[SLAM](docs/slam.md)  
[Kinematics of Differential Drive Robots and Wheel odometry](docs/differential_drive_robots_kinematics.md)  
[Graph SLAM](docs/graph_slam.md)  
[g2o](docs/g2o.md)  
[NeRF-SLAM](docs/NeRF-SLAM.md)  
[Factor Graph](docs/factor_graph.md)  
[GTSAM](docs/GTSAM.md)  
[Active Exposure Control for Robust Visual Odometry in HDR Environments](docs/active_exposure_control_HDR_environments.md)  
[Resilient Autonomy in Perceptually-degraded Environments](https://www.youtube.com/watch?v=L0PQKxU8cps)  
[A visual introduction to Gaussian Belief Propagation](https://gaussianbp.github.io/)  
[Lidar and IMU ](docs/lidar_and_imu.md)  
[IMU Propagation Derivations](https://docs.openvins.com/propagation.html)  
[Open Keyframe-based Visual-Inertial SLAM](https://github.com/ethz-asl/okvis)  
[HBA Large-Scale LiDAR Mapping Module](https://github.com/hku-mars/HBA)  
[Gaussian Splatting](docs/gaussian_splatting.md)  
[GANeRF](https://github.com/barbararoessle/ganerf)  
[Hierarchical, multi-resolution volumetric mapping](https://github.com/ethz-asl/wavemap)  


# [Lie Groups](#)
- [Matrix Lie Groups for Robotics](docs/matrix_lie_groups.md)


# [Instant NGP](#)
- [instant-ngp](docs/instant_ngp.md)
  
Refs: [1](https://www.youtube.com/channel/UCf0WB91t8Ky6AuYcQV0CcLw/videos),[2](https://github.com/spmallick/learnopencv/blob/master/README.md),[3](http://graphics.cs.cmu.edu/courses/15-463/),[4](https://www.tangramvision.com/blog/camera-modeling-exploring-distortion-and-distortion-models-part-i)  
