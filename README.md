# OpenCV Projects

![Docker CI](https://github.com/behnamasadi/OpenCVProjects/actions/workflows/docker.yml/badge.svg)
![alt text](https://img.shields.io/badge/license-BSD-blue.svg)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/behnamasadi/OpenCVProjects)
![GitHub Release](https://img.shields.io/github/v/release/behnamasadi/OpenCVProjects)
![GitHub Repo stars](https://img.shields.io/github/stars/behnamasadi/OpenCVProjects)
![GitHub forks](https://img.shields.io/github/forks/behnamasadi/OpenCVProjects)

This project contains my **Computer Vision Projects** with OpenCV.

## Building and Installation

> **TL;DR** — the project runs entirely inside a Docker image (OpenCV 4.6, Eigen, Ceres, etc. all pre-installed). The host never needs any C++ dependency installed. Every build step is a one-shot `docker run` with the repo bind-mounted, so your editor sees every file instantly and `build/` stays on the host.

### Quickstart — build & run everything inside the image

From the repo root:

```bash
# 1. Pull the pre-built image (or see §1 below to build it locally)
docker pull ghcr.io/behnamasadi/opencvprojects:master
docker tag  ghcr.io/behnamasadi/opencvprojects:master myopencv_image:latest

# 2. Shorthand for every subsequent command — same flags, different cmd at the end
RUN='docker run --rm --user $(id -u):$(id -g) -v "$(pwd)":/OpenCVProjects -w /OpenCVProjects myopencv_image'

# 3. Configure, build, test
eval $RUN cmake --preset release
eval $RUN cmake --build --preset release
eval $RUN ctest --preset release            # no-op until tests/ is populated

# 4. Run a binary (add X11 flags if the demo uses cv::imshow)
eval $RUN ./build/release/basic_operations

# 5. Run a GUI demo with X11 forwarding
xhost +local:docker
docker run --rm --user $(id -u):$(id -g) \
    -v "$(pwd)":/OpenCVProjects -w /OpenCVProjects \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 --network=host \
    myopencv_image ./build/release/edge_detection
```

**Why the indirection via `$RUN`**: it keeps the long mount+user flags in one place while the actual cmake/ctest/binary invocations stay visible — no hidden wrapper script.

**Caveat** — binaries under `build/release/` link against the container's OpenCV, so they will **not** run on the bare host. Always launch them through `docker run ... myopencv_image ./build/release/<exe>` (or from an interactive shell inside the container; see §2 below).

**VSCode users** — open the project and choose *Reopen in Container*. The ships-with `.devcontainer/devcontainer.json` builds the same image, mounts the source, wires up X11, and points C++ IntelliSense at `build/release/compile_commands.json`. (Inspect the JSON to see exactly what it does — nothing hidden.)

The remaining subsections document additional Docker flows (local build, persistent container, disposable container with GUI, cleanup).

### 1. Getting the Image

#### Option A — Pull the pre-built image (fastest)

CI publishes the image to GitHub Container Registry on every push to `master` and every tag. Just pull it:

```bash
docker pull ghcr.io/behnamasadi/opencvprojects:master
docker tag  ghcr.io/behnamasadi/opencvprojects:master myopencv_image:latest
```

#### Option B — Build it locally

The Dockerfile uses Ubuntu 24.04 with OpenCV from official repositories. Build it with:

```bash
docker build -t myopencv_image:latest .
```

**Note:** This is much faster than building OpenCV from scratch because it uses pre-built OpenCV packages (version 4.6.0) from the Ubuntu archive instead of compiling from source.

### 2. Creating the container

From the project root (the directory containing the `Dockerfile`), create a container that mounts the checkout into the container:

```bash
docker run --name myopencv_container \
    --user $(id -u):$(id -g) \
    -v "$(pwd)":/OpenCVProjects \
    -w /OpenCVProjects \
    -it myopencv_image bash
```

**Notes:**

- `--user $(id -u):$(id -g)` runs the container with your host user's UID and GID, so files created in the build directory are owned by your user on the host, not root. This lets you delete the build directory without `sudo`.
- `-w /OpenCVProjects` drops you into the mounted project directory (the Dockerfile's default `WORKDIR` is `/`).
- `-v "$(pwd)":/OpenCVProjects` avoids hardcoding a path — run the command from the repo root.
- The trailing `bash` makes the shell explicit regardless of the image's default `CMD`.

### 3. Starting an existing container

If you have already created a container from the docker image, you can start it with:

```bash
docker start -i myopencv_container
```

### 4. Removing unused images and containers

To reclaim disk space, remove stopped containers and dangling/unused images. Both commands prompt for confirmation unless you pass `-f`:

```bash
docker container prune          # remove all stopped containers
docker image prune -a           # remove all images not referenced by any container
```

**Warning:** `docker image prune -a` removes every image that isn't currently in use by a container, not just intermediate/dangling ones. Use plain `docker image prune` if you only want to drop dangling layers.

### GUI application with docker

This is an **alternative to step 2** when you need X11 forwarding (e.g., `cv::imshow`, `highgui` windows). If you already created `myopencv_container` in step 2, either pick a different `--name`, remove the old container with `docker rm myopencv_container`, or use the disposable workflow below.

1. Allow local Docker clients to talk to your X server (once per login session):

```bash
xhost +local:docker
```

2. Then, from the project root, run:

```bash
docker run --name myopencv_container_gui \
    --user $(id -u):$(id -g) \
    -v "$(pwd)":/OpenCVProjects \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -w /OpenCVProjects \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    --network=host \
    -it myopencv_image bash
```

**Note on `--privileged`:** It is **not** required for X11 forwarding and is omitted above because it grants the container broad host access. Add `--privileged` (or a narrower `--device=...`) only if you need direct access to host devices such as a webcam (`/dev/video*`) or GPU.

read more [here](https://ros-developer.com/2017/11/08/docker/)

### Using Docker Containers as Disposable

Some developers prefer using Docker containers as disposable, meaning they create a fresh container each time instead of reusing existing ones. This approach ensures a clean environment every time.

To use this workflow:

1. First, allow Docker to access your X server (only needed once per session):

```bash
xhost +local:docker
```

2. From the project root, run a new container with the `--rm` flag (auto-removes the container when it exits):

```bash
docker run --rm \
    --user $(id -u):$(id -g) \
    -v "$(pwd)":/OpenCVProjects \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -w /OpenCVProjects \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    --network=host \
    -it myopencv_image bash
```

**Key differences from persistent containers:**

- The `--rm` flag automatically removes the container when you exit
- No `--name` flag needed since the container is temporary
- Any changes made inside the container (outside mounted volumes) are lost when you exit
- Your code in `/OpenCVProjects` persists because it's mounted from your host machine
- Build artifacts in `/OpenCVProjects/build` persist because they're in the mounted directory

**When to use disposable containers:**

- Testing configurations without affecting a persistent environment
- Ensuring reproducible builds from a clean state
- Avoiding container name conflicts
- Keeping your Docker environment clean without manual pruning

**When to use persistent containers:**

- You've installed additional packages and want to keep them
- You want to maintain shell history and configurations
- You prefer faster startup times (no need to recreate the container)

# Installing OpenCV Python API

All Python dependencies live in `environment.yml` (single source of truth for the conda env). Create and activate the environment with:

```bash
conda env create -f environment.yml
conda activate OpenCVProjects
```

To update after editing `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

Optional — symlink the project's `scripts/` and `images/` into the env directory so they're on Jupyter's default path:

```bash
cd "$HOME/anaconda3/envs/OpenCVProjects/"
ln -s "$HOME/workspace/OpenCVProjects/scripts" .
ln -s "$HOME/workspace/OpenCVProjects/images"  .
```

## Deep image matching toolkit (hloc)

The env already includes `colmap`, `pycolmap`, `pytorch`, and `kornia`. To add [hloc](https://github.com/cvg/Hierarchical-Localization) (SuperPoint, DISK, ALIKED, LightGlue, LoFTR, NetVLAD, vocab-tree retrieval — all wrapped around COLMAP):

```bash
conda activate OpenCVProjects
git clone --recursive https://github.com/cvg/Hierarchical-Localization.git thirdparty/hloc
python -m pip install -e thirdparty/hloc
```

Pipeline example: [docs/hloc.md](docs/hloc.md).

## Pre-trained COLMAP vocabulary trees

The `vocab_tree/` directory holds FAISS-based vocabulary trees (SIFT and ALIKED variants) for COLMAP image retrieval, tracked via **Git LFS** so `git clone` is cheap when you don't need them. To fetch only what you want:

```bash
git lfs pull --include "vocab_tree/vocab_tree_faiss_flickr100K_words256K.bin"
```

Which file is which, and how to fetch selectively: [docs/vocab_tree.md](docs/vocab_tree.md).

# [Computer Vision](#)

[Edge Detection](docs/edge_detection.ipynb)  
[Image Filtering Convolution](docs/image_filtering_convolution.md)  
[Keypoints (Feature2D) Detector and Descriptor](docs/keypoints_detector_and_descriptor.ipynb)  
[Keypoints matching, Image Registration](docs/correspondences_matching.ipynb)  
[Correspondence Problem, Keypoints Matching](docs/correspondences_matching.ipynb)  
[Image Matching WebUI](https://huggingface.co/spaces/Realcat/image-matching-webui)  
[Histogram Analysis](docs/histogram_analysis.ipynb)  
[Pinhole Camera Model and Projection](docs/pinhole_camera_model_projection_intrinsic.ipynb)  
[Direct Linear Transformation](docs/pinhole_camera_model_projection_intrinsic.ipynb#Direct-Linear-Transform)  
[Zhang's Camera Calibration Algorithm](docs/pinhole_camera_model_projection_intrinsic.ipynb#Zhang's-Algorithm)  
[Extrinsic Calibration of Camera to LIDAR](https://www.dropbox.com/s/arhpp59d502fuks/2023IROS_fu.pdf?dl=0)  
[Affine Transformation](docs/affine_transformation.ipynb)  
[Image Warping](docs/image_warping.ipynb)  
[Perspective Transformation](docs/perspective_transform.ipynb)  
[Homography Transformation](docs/homography.ipynb)  
[Image Remaping](docs/remap.ipynb)  
[Lense Distortion Modeling, Undistorting Points and Images](docs/undistortion.ipynb)  
[Triangulation](docs/triangulation.ipynb)  
[Epipolar Geometry, Essential Matrix, Fundamental Matrix](docs/epipolar_geometry_essential_matrix_fundamental_matrix.ipynb)  
[Ego Motion Estimation, Recover Pose and Pose Estimation, R,t from Essential Matrix](docs//epipolar_geometry_essential_matrix_fundamental_matrix.ipynb#Recovery-of-R,T-from-Essential-Matrix)  
[Sparse Optical Flow, Dense Optical Flow](docs/opticalflow.ipynb)  
[KITTI Visual Odometry](docs/kitti.ipynb)  
[Stereo Calibration](docs/stereo_calibration_disparity_map.ipynb#Stereo-Calibration)  
[Disparity Map](docs/stereo_calibration_disparity_map.ipynb#Disparity-Map)  
[Stereo Rectification](docs/stereo_calibration_disparity_map.ipynb#Stereo-Rectification)  
[Parallax](docs/parallax.ipynb)  
[Laser Triangulation](docs/laser_triangulation.md)  
[Bayes Filter, Estate Estimation with Kalman Filter, Extended Kalman Filter](docs/kalman_filter.md)  
[Bag of Words, Vocabulary Tree](docs/bag_of_words_vocabulary_tree.ipynb)  
[PnP,P3P, Perspective-n-point](docs/perspective_n_point_pnp.ipynb)  
[3D World Unit Vector Camera Direction Vectors Camera Projection Rays](docs/3d_world_unit_vector.md)  
[Hand Eye Calibration](docs/hand_eye_calibration.ipynb)  
[Normalized Image Coordinates](docs/epipolar_geometry_essential_matrix_fundamental_matrix.ipynb#Normalized-Image-Coordinates)  
[Camera Models](docs/camera_models.ipynb)  
[Color Space](docs/color_spaces.md)  
[Color Calibration](docs/color_calibration.md)  
[Dynamic Range](docs/dynamic_range.md)  
[White Balance](srcipts/white_balance.ipynb)  
[Decompose Projection Matrix](docs/decomposition_matrix.ipynb#Decompose-Essential-Matrix)  
[Decompose Homography Matrix](docs/decomposition_matrix.ipynb#decomposition_matrix.ipynb#Decompose-Homography-Matrix)  
[Decompose Essential Matrix](docs/epipolar_geometry_essential_matrix_fundamental_matrix.ipynb#Decompose-Essential-Matrix)  
[Laplacian Variance Blur Detection](docs/laplacian_variance_blur_detection.ipynb)  
[Image-Based Rendering](https://staff.aist.go.jp/naoyuki.ichimura/research/IBR/ibr.htm)

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
[Photogrammetry](docs/photogrammetry_bundle_adjustment_structure_from_motion_reprojection_error.ipynb#photogrammetry)  
[Structure from Motion (SfM)](docs/photogrammetry_bundle_adjustment_structure_from_motion_reprojection_error.md#structure-from-motion--sfm-)  
[Bundle Adjustment](docs/photogrammetry_bundle_adjustment_structure_from_motion_reprojection_error.md#bundle-adjustment)  
[Noah Snavely reprojection error](docs/photogrammetry_bundle_adjustment_structure_from_motion_reprojection_error.md#noah-snavely-reprojection-error)  
[Hierarchical Localization / deep image matching (hloc)](docs/hloc.md)  
[Pre-trained COLMAP vocabulary trees](docs/vocab_tree.md)

Refs: [1](https://www.youtube.com/channel/UCf0WB91t8Ky6AuYcQV0CcLw/videos),[2](https://github.com/spmallick/learnopencv/blob/master/README.md),[3](http://graphics.cs.cmu.edu/courses/15-463/),[4](https://www.tangramvision.com/blog/camera-modeling-exploring-distortion-and-distortion-models-part-i)
