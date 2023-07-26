# COLMAP installation

## Installation using docker
### 1. Building the Image
There is docker file for this project where contains all dependencies and you build the image with :   
`docker build -t sfm .`

to quickly check your nvidia docker, run:

`docker run --gpus all nvidia/cuda:11.6.0-devel-ubuntu20.04 nvidia-smi`


### 2. Creating the container
Create a container where you mount you image dataset into your container: 

`docker run --gpus all --name <continer-name> -v <image-dataset-path-on-host>:<path-in-the-container> -it <docker-image-name>`

for instance:

`docker run --gpus all --name sfm_container -v /home/behnam/workspace/sfm:/sfm -it sfm`

### 3. Starting an existing container
If you have already created a container from the docker image, you can start it with:

`docker  start -i sfm_container`


Refs: [1](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker), [2](https://github.com/colmap/colmap/blob/dev/docker/Dockerfile), [3](https://github.com/NVIDIA/nvidia-docker)


## Direct installation on your machine
### CUDA Installation

1. Verify You Have a CUDA-Capable GPU

`lspci | grep -i nvidia`

2. The kernel headers and development packages

`apt-get install linux-headers-$(uname -r)`

3. Remove Outdated Signing Key
`apt-key del 7fa2af80`

4. Install the new cuda-keyring package:

where `$distro/$arch` is `ubuntu2004/x86_64`

`wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb`

`wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb`

`dpkg -i cuda-keyring_1.0-1_all.deb`

5. Update the apt repository cache

`apt-get update`

6. Install CUDA SDK:

These two commands must be executed separately:
- `apt-get install cuda`

- `apt-get install nvidia-gds`

Refs: [1](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation)

## CUDA settings
To see the current default version of installed CUDA:
```
sudo update-alternatives --display cuda
```
To change the default version pf CUDA:
```
sudo update-alternatives --config cuda
```
to see the version of the CUDA compiler:
```
 /usr/local/cuda/bin/nvcc --version
 ```
to set the prefered CUDA version:
```
to set the preferred executable for compiling CUDA language files
```
CUDACXX=/usr/local/cuda-12.1/bin/nvcc
```
export PATH="/usr/local/<cuda-version>/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/<cuda-version>/lib64:$LD_LIBRARY_PATH"
```
for instance:
```
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
```
## COLMAP Installation

Set the compilers:

```
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDAHOSTCXX=/usr/bin/g++-9
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
```

Dependencies:
 
```
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev
```

Download and build COLMAP

``` 
git clone https://github.com/colmap/colmap

cd colmap

cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_CUDA_STANDARD=14  -DCMAKE_CUDA_STANDARD_REQUIRED=TRUE -DCMAKE_CXX_STANDARD_REQUIRED=TRUE -DCMAKE_INSTALL_PREFIX=~/usr

cmake --build build -j18

cmake --install build
```

# Running COLMAP via Command-line Interface

create a project folder i.e. `south-building`. This directory must contain a folder `images` with all the images. If you need to resize your images run the following, it will resize your images `20%` and put them in the `_resized` directory:

`find . -maxdepth 1 -iname "*.jpg" | xargs -L1 -I{} convert -resize 20% "{}" _resized/"{}"`

First set the path:

`DATASET_PATH=/sfm/image_dataset/south-building`

and create dense and sparse directory:

```
mkdir $DATASET_PATH/sparse
mkdir $DATASET_PATH/dense
```

## Feature Extraction
Now run the feature extractor:

```   
colmap feature_extractor  \
--database_path $DATASET_PATH/database.db  \
--image_path $DATASET_PATH/images  \
--ImageReader.single_camera true  \
--SiftExtraction.use_gpu 1 
```

If you have your camera parameter you specify them:

set you camera parameters:
```
CAM=fx,fy,cx,cy,k1,k2,k3,k4
```
Then 
```
colmap feature_extractor  \
--database_path $DATASET_PATH/database.db  \
--image_path $DATASET_PATH/images  \
--ImageReader.single_camera=true --ImageReader.camera_model=OPENCV_FISHEYE --ImageReader.camera_params=$CAM \ --SiftExtraction.use_gpu 1
```

[List of camera models in COLMAP](https://colmap.github.io/cameras.html)


To increase the number of matches, you should use the more discriminative DSP-SIFT features instead of plain SIFT and also estimate the affine feature shape using the options:
```
--SiftExtraction.estimate_affine_shape=true \
--SiftExtraction.domain_size_pooling=true\
```
Also, you should enable guided feature matching in the matching step (exhaustive_matcher, sequential_matcher, etc) using: `--SiftMatching.guided_matching=true`
so your extraction would be:


```
colmap feature_extractor  \
--database_path $DATASET_PATH/database.db  \
--image_path $DATASET_PATH/images  \
--ImageReader.single_camera=true --ImageReader.camera_model=OPENCV_FISHEYE --ImageReader.camera_params=$CAM \ --SiftExtraction.use_gpu 1 \
--SiftExtraction.estimate_affine_shape=true \
--SiftExtraction.domain_size_pooling=true
```

Warning: The GPU implementation of SIFT simply does not support estimating **affine shapes**. Thus, Colmap falls back to a **CPU** implementation of SIFT that supports this feature.

Refs: [1](https://colmap.github.io/faq.html#increase-number-of-matches-sparse-3d-points)

## Feature Matching

### Exhaustive Matching   
Then run the matcher: 
```
colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db \
   --SiftMatching.use_gpu 1
```
### Sequential Matching

If your images are in a sequential order and consecutive frames have visual overlap, consecutively captured images can be matched against each other. This matching mode has built-in loop detection based on a vocabulary tree.
Every N-th image (`loop_detection_period`) is matched against its visually most similar images (`loop_detection_num_images`). Image file names must be ordered sequentially (e.g., `image0001.jpg, image0002.jpg`, etc.), images are explicitly ordered according to their file names. Note that loop detection requires a pre-trained vocabulary tree, that can be downloaded from [Here](https://demuc.de/colmap/).

```
colmap sequential_matcher \
   --database_path $DATASET_PATH/database.db \
   --SequentialMatching.overlap=3 \
   --SequentialMatching.loop_detection=true \
   --SequentialMatching.loop_detection_period=2 \
   --SequentialMatching.loop_detection_num_images=50 \
   --SequentialMatching.vocab_tree_path="../vocab_tree/vocab_tree_flickr100K_words32K.bin" \
   --SiftMatching.use_gpu 1 --SiftMatching.gpu_index=-1  --SiftMatching.guided_matching=true  
```

Refs: [1](https://colmap.github.io/tutorial.html#feature-matching-and-geometric-verification)

### Hierarchical Mapper
`hierarchical_mapper`: Sparse 3D reconstruction / mapping of the dataset using hierarchical SfM after performing feature extraction and matching. This parallelizes the reconstruction process by partitioning the scene into overlapping submodels and then reconstructing each submodel independently. Finally, the overlapping submodels are merged into a single reconstruction. It is recommended to run a few rounds of point triangulation and bundle adjustment after this step.


## Sparse 3D Reconstruction

Then run the mapper:

```
colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse
```

## Undistortion
Then run the image undistorter:

```
colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000
```

##  Dense 3D Reconstruction

Then run the match stereo(warning this will take a while):

```
colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true
```

## Fusion Into Point Cloud

Then run the stereo fusion:

```
colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply
```

## Meshing of Fused Point Cloud

Now your point cloud is ready, you can create mesh file from that:

```
colmap poisson_mesher \
    --input_path $DATASET_PATH/dense/fused.ply \
    --output_path $DATASET_PATH/dense/meshed-poisson.ply
```

and optionally a delaunay mesh:

```
colmap delaunay_mesher \
    --input_path $DATASET_PATH/dense \
    --output_path $DATASET_PATH/dense/meshed-delaunay.ply    
```

Ref: [1](https://colmap.github.io/cli.html)

# Running COLMAP GUI

1. You need to run:

`docker run --gpus all --name sfm_container_gui -v /home/behnam/workspace/sfm:/sfm --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -it sfm`


2. On the host run the following (every time you run your container):

`export containerId=$(docker ps -l -q)`

<code>  xhost +local: docker inspect --format='{{ .Config.Hostname }}' $containerId </code>

Refs: [1](https://github.com/jamesbrink/docker-opengl)

# Tips and Utility Tools

## Reconstruct Sparse/Dense Model From Known Camera Poses

Refs: [1](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses)

## Merge Disconnected Models

```
colmap model_merger \
    --input_path1 /path/to/sub-model1 \
    --input_path2 /path/to/sub-model2 \
    --output_path /path/to/merged-model
```
To improve the quality of the alignment between the two sub-models, it is recommended to run another global bundle adjustment after the merge: 

```
colmap bundle_adjuster \
    --input_path /path/to/merged-model \
    --output_path /path/to/refined-merged-model
```
  

## Extending COLMAP



## COLMAP and Visual Odometry

Refs: [1](https://github.com/colmap/colmap/issues/568)

## COLMAP Parameters
Refs: [1](https://github.com/mwtarnowski/colmap-parameters)


## Loop Closure
Colmap can add images in arbitrary order. Colmap prefers to use image pairs that are not pure forward motion for **initialization**. 

Try skipping frames, eg, use only a frame every 1 or 2 seconds. This should improve reduce drift for video data. You could also try to I crease the size of the local bundle adjustment window.  Otherwise, COLMAP probably detects the loops successfully but cannot optimize them as well as ORBSLAM, which is specifically optimized for the SLAM scenario. It also does pose graph optimization if I remember correctly, which is better in correcting for large drift than pure bundle adjustment used in COLMAP.

Refs: [1](https://github.com/colmap/colmap/issues/1521), [2](https://github.com/colmap/colmap/issues/254)


## Train the Vocabulary Tree
Refs: [1](https://github.com/colmap/colmap/issues/866)


## External Dense Reconstruction Libraries
If you do not have a CUDA-enabled GPU but some other GPU, you can use all COLMAP functionality except the dense reconstruction part. You can use external dense reconstruction software as an alternative. COLMAP exports to several other dense reconstruction libraries, .

- CMVS/PMVS [furukawa10]
- CMP-MVS [jancosek11]
- Line3D++ [hofer16].


Refs: [1](https://colmap.github.io/tutorial.html#dense-reconstruction)


## Improving Dense Reconstruction of Weakly Textured Surfaces

Refs: [1](https://colmap.github.io/faq.html#improving-dense-reconstruction-results-for-weakly-textured-surfaces)


## Speedup Dense Reconstruction


Refs: [1](https://colmap.github.io/faq.html#speedup-dense-reconstruction)


## Importing and Exporting

output type could be `{BIN, TXT, NVM, Bundler, VRML, PLY, R3D, CAM}`

```
colmap model_converter --input_path $DATASET_PATH/sparse/0 --output_path /home/$USER/ --output_type TXT
```
which will give you `points3D.txt`, `images.txt`, `cameras.txt`

The output could be used for [Instant-ngp: Instant neural graphics primitives](https://nvlabs.github.io/instant-ngp/)


## Instant-ngp

```
python3 /home/$USER/workspace/instant-ngp/scripts/colmap2nerf.py --text /home/$USER/<above-output> --images /home/$USER/<images>
```


## Register/Localize New Images Into an Existing Reconstruction

If you have an existing reconstruction of images and want to register/localize new images within this reconstruction, you can follow these steps:
create an image list text file contains a list of images to get extracted and matched, specified as one image file name per line. 

```
colmap feature_extractor \
    --database_path $PROJECT_PATH/database.db \
    --image_path $PROJECT_PATH/images \
    --image_list_path /path/to/image-list.txt

colmap vocab_tree_matcher \
    --database_path $PROJECT_PATH/database.db \
    --VocabTreeMatching.vocab_tree_path /path/to/vocab-tree.bin \
    --VocabTreeMatching.match_list_path /path/to/image-list.txt

colmap image_registrator \
    --database_path $PROJECT_PATH/database.db \
    --input_path /path/to/existing-model \
    --output_path /path/to/model-with-new-images

colmap bundle_adjuster \
    --input_path /path/to/model-with-new-images \
    --output_path /path/to/model-with-new-images
```


If you need a more accurate image registration with triangulation, then you should restart or continue the reconstruction process rather than just registering the images to the model. Instead of running the `image_registrator`, you should run the mapper to continue the reconstruction process from the existing model:

```
colmap mapper \
    --database_path $PROJECT_PATH/database.db \
    --image_path $PROJECT_PATH/images \
    --input_path /path/to/existing-model \
    --output_path /path/to/model-with-new-images
```
    
Or, alternatively, you can start the reconstruction from scratch:

```
colmap mapper \
    --database_path $PROJECT_PATH/database.db \
    --image_path $PROJECT_PATH/images \
    --output_path /path/to/model-with-new-images
```

Note that dense reconstruction must be re-run from scratch after running the `mapper` or the `bundle_adjuster`, as the coordinate frame of the model can change during these steps.    


Refs: [1](https://colmap.github.io/faq.html#register-localize-new-images-into-an-existing-reconstruction)



## Manhattan World Alignment

Insert this after the mapper:
```
colmap model_orientation_aligner \
    --input_images $DATASET_PATH/images  \
    --input_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse_aligned
```
and then use the "sparse_aligned" as the input for subsequent steps.


Manhattan world alignment is a computer vision concept used in scene understanding and 3D reconstruction. It is based on the assumption that the world is composed of a set of dominant, orthogonal (at right angles to each other) surfaces, much like the layout of buildings and streets in Manhattan, New York City. This assumption simplifies the understanding and reconstruction of scenes, making it easier for computer algorithms to interpret and analyze the environment.

In a Manhattan world, the three main orthogonal directions are typically aligned with the three axes of a Cartesian coordinate system:

- X-axis: Horizontal direction (left-right).
- Y-axis: Vertical direction (up-down).
- Z-axis: Depth direction (front-back).


In this world, the surfaces are assumed to be planar (flat), and objects are placed parallel or perpendicular to each other, forming right angles. Due to this assumption, scenes can be described more easily, and object orientations and locations can be estimated with greater accuracy.

Manhattan world alignment is often used in computer vision tasks like:

1. Vanishing point detection: Detecting the points where parallel lines in a scene converge (vanish) due to perspective projection. In a Manhattan world, the vanishing points often align with the X, Y, and Z axes.

2. 3D scene reconstruction: Inferring the 3D structure of a scene from 2D images. Assuming a Manhattan world simplifies the process of reconstructing 3D geometry from 2D images.

3. Object detection and orientation estimation: Understanding the layout and orientation of objects in a scene, assuming they are aligned with the dominant surfaces.

By leveraging the Manhattan world assumption, computer vision algorithms can efficiently reason about scene layout, estimate camera poses, and make accurate predictions about the positions and orientations of objects in a 3D environment. However, it's essential to note that not all real-world scenes adhere strictly to the Manhattan world assumption, especially in outdoor environments or complex indoor scenes. Nevertheless, the concept provides a useful starting point for many computer vision applications, and more advanced techniques can be applied to handle non-Manhattan scenes.

Refs: [1](https://colmap.github.io/faq.html#manhattan-world-alignment), [2](https://github.com/colmap/colmap/issues/1743), [3](https://grail.cs.washington.edu/projects/manhattan/manhattan.pdf)


## Multiple View Triangulation

`feature_importer` and `matches_importer`

Refs: [1](https://robotics.stackexchange.com/questions/16132/multiple-view-triangulation-method-used-by-colmap), [2](https://github.com/colmap/colmap/issues/688)
