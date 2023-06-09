# Colmap installation

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

## Colmap Installation

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

Download and build colmap

``` 
git clone https://github.com/colmap/colmap

cd colmap

cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_CUDA_STANDARD=14  -DCMAKE_CUDA_STANDARD_REQUIRED=TRUE -DCMAKE_CXX_STANDARD_REQUIRED=TRUE -DCMAKE_INSTALL_PREFIX=~/usr

cmake --build build -j6

cmake --install build
```

# Running colmap

create a project folder i.e. `south-building`. This directory must contain a folder `images` with all the images. If you need to resize your images run the following, it will resize your images `20%` and put them in the `_resized` directory:

`find . -maxdepth 1 -iname "*.jpg" | xargs -L1 -I{} convert -resize 20% "{}" _resized/"{}"`

First set the path:

`DATASET_PATH=/sfm/image_dataset/south-building`

and create dense and sparse directory:

```
mkdir $DATASET_PATH/sparse
mkdir $DATASET_PATH/dense
```


Now run the feature extractor:

```   
colmap feature_extractor  \
--database_path $DATASET_PATH/database.db  \
--image_path $DATASET_PATH/images  \
--ImageReader.single_camera true  \
--SiftExtraction.use_gpu 1 
```
   
Then run the matcher: 
```
colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db \
   --SiftMatching.use_gpu 1
```

Then run the mapper:

```
colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse
```


Then run the image undistorter:

```
colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000
```


Then run the match stereo:

```
colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true
```

Then run the stereo fusion (warning this will take a while):

```
colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply
```

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

# Running colmap gui

1. You need to run:

`docker run --gpus all --name sfm_container_gui -v /home/behnam/workspace/sfm:/sfm --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -it sfm`


2. On the host run the following (every time you run your container):

`export containerId=$(docker ps -l -q)`

<code>  xhost +local: docker inspect --format='{{ .Config.Hostname }}' $containerId </code>

Refs: [1](https://github.com/jamesbrink/docker-opengl)



# Reconstruct sparse/dense model from known camera poses

Refs: [1](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses)



# Extending COLMAP



# Colmap and Visual Odometry

Refs: [1](https://github.com/colmap/colmap/issues/568)

