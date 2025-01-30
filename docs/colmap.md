# 1. COLMAP Installation

## 1.1 ssh/ scp set up (optional)
If you running colmap on the remote machine first:


```
MACHINE_ADDRESS=<ip-address>
```
Then tunnel the localhost for jupyter:

```
ssh -i /home/$USER/.ssh/<user>.pem -L 10000:localhost:10000 <user-name>@MACHINE_ADDRESS
```

Install jupyter
```
pip install --upgrade jupyter
pip install --upgrade ipywidgets
```

So you can run jupyter
```
jupyter notebook --ip 0.0.0.0 --port 10000
```
-----

## 1.2 Installation using docker

Now on your machine run `nvidia-smi` and check the cuda supported version:

```
NVIDIA-SMI 550.127.08             Driver Version: 550.127.08     CUDA Version: 12.4
```

since we have `12.4`, download and install the cuda toolkit 12.4, and set the pass:

```
export PATH="/usr/local/cuda-12.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"
```

check the version:

to see the version of the CUDA compiler:
```
 /usr/local/cuda/bin/nvcc --version
```

Now docker part:

- `docker pull colmap/colmap:20240212.4` or `docker pull colmap/colmap:20240112.3`

- `xhost +local:`
- Now run: `docker run --gpus all --name <continer-name> -v <image-dataset-path-on-host>:<path-in-the-container> -it <docker-image-name>` 

for instance`docker run --gpus all  -v /home/$USER/:/home/$USER/ -v /tmp/.X11-unix:/tmp/.X11-unix --name colmap_container -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 --network=host --privileged -it colmap/colmap:20240212.4 bash`


If you have already created a container from the docker image, you can start it with:

`docker  start -i colmap_container`



[Full list of available Nvidia tags on docker hub](https://hub.docker.com/r/nvidia/cuda/tags)  
[Full list of available Colmap tags on docker hub](https://hub.docker.com/r/colmap/colmap/tags)  


# 2. Running colmap

## 2.1 GUI

On the host run the following (every time you run your container):

`export containerId=$(docker ps -l -q)`

<code>  xhost +local: docker inspect --format='{{ .Config.Hostname }}' $containerId </code>



## 2.2 Command-line Interface
create a project folder i.e. `south-building`, set the path:

`DATASET_PATH=/sfm/image_dataset/south-building`

and create dense and sparse directories:

```
mkdir $DATASET_PATH/sparse
mkdir $DATASET_PATH/dense
```

project directory structure should look like this:

```
.
├── database.db
├── dense
├── images
│   ├── 00000.png
│   ├── 00001.png
│   └── 00002.png
├── sparse
├── video
│   └── video.mp4
└── vocab_tree
    └── vocab_tree_flickr100K_words256K.bin
```


## 2.3 Create image sequence from video

set the file name:
```
FILE=video.mkv
FILE=video_0.mp4
```


To select 4 images per second, we need to select every 1/4th of a second. Assuming a typical frame rate of 25 frames per second, we need to select every 6th frame.

```
ffmpeg -i $DATASET_PATH/video/$FILE -ss 00:00:59  -to 00:07:00  -vf "select=not(mod(n\,6)),scale=1920:-1" -vsync vfr $DATASET_PATH/images/%05d.png 
```

If you set the `select=not(mod(n\,25))` it select every 25th frame.


## 2.4 Feature Extraction

First set your camera parameter

### Camera Models
- SIMPLE_PINHOLE: `f,cx,cy`
- PINHOLE: `fx,fy,cx,cy`
- SIMPLE_RADIAL: `f,cx,cy,k`
- SIMPLE_RADIAL_FISHEYE: `f,cx,cy,k`
- RADIAL: `f,cx,cy,k1,k2`
- RADIAL_FISHEYE: `f,cx,cy,k1,k2`
- OPENCV: `fx,fy,cx,cy,k1,k2,p1,p2`
- OPENCV_FISHEYE:`fx,fy,cx,cy,k1,k2,k3` 
- FULL_OPENCV: `fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6` 
- FOV:`fx,fy,cx,cy,omega`
- THIN_PRISM_FISHEYE: `fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,sx1,sy1`

Refs: [1](https://colmap.github.io/cameras.html)


If you have your camera parameter you specify them:

set your camera parameters(fish eye):
```
CAM=fx,fy,cx,cy,k1,k2,k3,k4
```

for normal pinhole:
```
CAM=fx,fy,cx,cy,k1,k2,p1,p2
```
for instance:
```
CAM=848.53117539872062,848.53117539872062,639.5,359.5,0.15971715887123378,-0.61045779426294378,0,0
```

Then 

```
colmap feature_extractor  \
--database_path $DATASET_PATH/database.db  \
--image_path $DATASET_PATH/images  \
--ImageReader.single_camera=true --ImageReader.camera_model=OPENCV_FISHEYE --ImageReader.camera_params=$CAM \ 
--SiftExtraction.use_gpu 1 \
--SiftExtraction.estimate_affine_shape=true \
--SiftExtraction.domain_size_pooling=true
```


To increase the number of matches, you should use the more discriminative DSP-SIFT features instead of plain SIFT and also estimate the affine feature shape using the options:
```
--SiftExtraction.estimate_affine_shape=true 
--SiftExtraction.domain_size_pooling=true 
```

The GPU implementation of SIFT simply does not support estimating **affine shapes**. Thus, Colmap falls back to a **CPU** implementation of SIFT that supports this feature.

Refs: [1](https://colmap.github.io/faq.html#increase-number-of-matches-sparse-3d-points)


## 2.5 Feature Matching

### 2.5.1 Sequential Matching

If your images are in sequential order and consecutive frames have visual overlap, consecutively captured images can be matched against each other. This matching mode has built-in loop detection **based on a vocabulary tree** that can be downloaded from [Here](https://demuc.de/colmap/). Every N-th image (`loop_detection_period`) is matched against its visually most similar images (`loop_detection_num_images`). Image file names must be ordered sequentially (e.g., `image0001.jpg, image0002.jpg`, etc.), and images are explicitly ordered according to their file names. 


```bash
colmap sequential_matcher \
   --database_path $DATASET_PATH/database.db \
   --SequentialMatching.overlap=10 \
   --SequentialMatching.loop_detection=true \
   --SequentialMatching.loop_detection_period=1 \
   --SequentialMatching.loop_detection_num_images=200 \
   --SequentialMatching.vocab_tree_path="$DATASET_PATH/../vocab_tree/vocab_tree_flickr100K_words256K.bin" \
   --SiftMatching.use_gpu 1 \
   --SiftMatching.gpu_index=-1 \
   --SiftMatching.guided_matching=true \
   --SiftMatching.max_num_matches=100000 \
   --SiftMatching.max_ratio=0.7 \
   --SiftMatching.max_distance=0.6
```

`--SequentialMatching.loop_detection_num_images`: This expands the search space, allowing the matcher to consider more images as potential loop closure candidates. Increase this value to a higher number (e.g., `100` or `200`).


`--SequentialMatching.loop_detection_period`: Set this to `1`. This ensures loop detection is attempted for every image in the sequence, increasing the chances of detecting loops.


`--SequentialMatching.overlap`: This extends the range of images considered for sequential matching, which can help capture more spatially adjacent loop closures. Increase this to `5` or `10`


**Trade-offs:** Increasing parameters like `loop_detection_num_images` and `overlap` will make the process more computationally expensive and slower.


`--SiftMatching.guided_matching`: Keep this enabled. Guided matching ensures that matches respect geometric constraints, reducing false positives and increasing the reliability of loop closure.


Given an image pair and the essential matrix `E` that relates them, the epipolar constraint states that for any point in the first image, its corresponding match in the second image should lie on a specific line known as the epipolar line. This constraint can be derived from the essential matrix `E`. Guided matching leverages this epipolar constraint to refine feature matches. Here's what happens when you enable `SiftMatching.guided_matching`:

1. Initial matches between two images are found based on the SIFT descriptors.
2. Using the matches, an essential matrix `E` is estimated.
3. With the estimated `E`, the epipolar lines in the second image for each feature in the first image are computed.
4. Matches that don't lie close to these epipolar lines are considered mismatches and are discarded.
5. The result is a refined set of matches that adhere better to the epipolar geometry.


`--SiftMatching.max_num_matches`: Allowing more matches can improve the chances of detecting loops.

`--SiftMatching.max_ratio`: default is `0.8`.  If you reduce it (e.g., `0.7` or `0.6`), you make the ratio test stricter, so fewer matches survive. and it reduces false matches but may also remove some true matches if you go too low.


`--SiftMatching.max_error`: (during geometric verification) Controls the maximum reprojection error in pixels for inlier correspondences during geometric verification (the default is usually `4.0` pixels). Lower it to be stricter (e.g., `2.0` or `1.5`).
Effect: In geometric verification, only correspondences that reproject within this tighter pixel error will remain.


`--SiftMatching.cross_check`: If you enable cross-check (true), a match must be mutual: A → B is best match and B → A is best match. This can eliminate unidirectional false matches. Effect: Tends to produce fewer but more reliable matches, at some computational cost.


`--SiftMatching.min_num_inliers`: Default is usually `15`. If you increase this number, you will discard image pairs that do not meet the minimum inlier count after geometric verification.
Effect: Eliminates pairs that only have a small number of inliers, which can often be false or too weak to help.


`--SiftMatching.confidence`: The default is usually `0.9999`, which influences how many RANSAC iterations are done internally to find inliers.If you reduce it (e.g., `0.999` or `0.99`), it might cause RANSAC to use fewer iterations. Usually, you’d increase it to be sure you don’t skip inliers, so be careful here.
Effect: Lower confidence means you might terminate RANSAC earlier; in practice, it rarely is used for controlling pickiness as directly as the others.


`--SiftMatching.max_distance` default: is `0.7`,  Any SIFT match with descriptor distance >0.7 is discarded.

---

### 2.5.2  Vocab tree matcher

```
colmap vocab_tree_matcher \
    --database_path $DATASET_PATH/database.db \
    --VocabTreeMatching.vocab_tree_path="$DATASET_PATH/../vocab_tree/vocab_tree_flickr100K_words256K.bin" \
    --SiftMatching.use_gpu 1 \
    --SiftMatching.gpu_index=-1 \
    --SiftMatching.guided_matching=true \ 
    --SiftMatching.cross_check=true \ # Enable cross-checking
    --SiftMatching.max_ratio=0.6 \  # Stricter ratio test
    --SiftMatching.max_distance=0.4  # Stricter max distance
```

###  2.5.3 Exhaustive Matching   

```
colmap exhaustive_matcher \
    --database_path $DATASET_PATH/database.db \
    --SiftMatching.use_gpu 1 \
    --SiftMatching.gpu_index=-1 \
    --SiftMatching.guided_matching=true \  
    --SiftMatching.cross_check=true \ # Enable cross-checking
    --SiftMatching.max_ratio=0.5 \  # Stricter ratio test
    --SiftMatching.max_distance=0.4  # Stricter max distance    
```

Refs: [1](https://colmap.github.io/tutorial.html#feature-matching-and-geometric-verification)

###  2.5.4 Hierarchical Mapper
`hierarchical_mapper`: Sparse 3D reconstruction / mapping of the dataset using hierarchical SfM after performing feature extraction and matching. This parallelizes the reconstruction process by partitioning the scene into overlapping submodels and then reconstructing each submodel independently. Finally, the overlapping submodels are merged into a single reconstruction. It is recommended to run a few rounds of point triangulation and bundle adjustment after this step.


## 2.6 Sparse 3D Reconstruction

Then run the mapper:

```
colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse
```

## 2.7 Undistortion
Then run the image undistorter:

```
colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000
```

## 2.8 Dense 3D Reconstruction

Then run the match stereo(warning this will take a while):

```
colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true
```

## 2.9 Fusion Into Point Cloud

Then run the stereo fusion:

```
colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply
```

## 2.10 Meshing of Fused Point Cloud

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



## 2.11 COLMAP Parameters

Full list of colmap parameters
Refs: [1](https://github.com/mwtarnowski/colmap-parameters)


# 3. Tips and Utility Tools


## 3.1 Importing and Exporting

output type could be `{BIN, TXT, NVM, Bundler, VRML, PLY, R3D, CAM}`

```
colmap model_converter --input_path $DATASET_PATH/sparse/0 --output_path /home/$USER/ --output_type TXT
```
which will give you `points3D.txt`, `images.txt`, `cameras.txt`

The output could be used for [Instant-ngp: Instant neural graphics primitives](https://nvlabs.github.io/instant-ngp/)




Refs: [1](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses)


## 3.2 Reconstruct Sparse/Dense Model From Known Camera Poses


Your data should have the following structure: 

```
├── database.db
├── dense
│   └── sparse
│       └── model
│           └── 0
├── images
│   ├── 00000.png
│   ├── 00001.png
│   ├── 00002.png
│   └── 00003.png
└── sparse
    └── model
        └── 0
            ├── cameras.txt
            ├── images.txt
            └── points3D.txt
```

1. `cameras.txt`: should be like this:

```
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 3
1 SIMPLE_PINHOLE 3072 2304 2559.81 1536 1152
2 PINHOLE 3072 2304 2560.56 2560.56 1536 1152
3 SIMPLE_RADIAL 3072 2304 2559.69 1536 1152 -0.0218531
```

2. `images.txt`: should be like this:

```
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 2, mean observations per image: 2
1 0.695104 0.718385 -0.024566 0.012285 -0.046895 0.005253 -0.199664 1 00000.png

2 0.696445 0.717090 -0.023185 0.014441 -0.041213 0.001928 -0.134851 1 00001.png

3 0.697457 0.715925 -0.025383 0.018967 -0.054056 0.008579 -0.378221 1 00002.png

4 0.698777 0.714625 -0.023996 0.021129 -0.048184 0.004529 -0.313427 1 00003.png
```

3. `points3D.txt`: This file should be empty.



Full example [here](kitti.ipynb#Reconstruct-Sparse/Dense-Model-From-Known-Camera-Poses-with-Colmap)

## 3.3 Merging Disconnected Models

```
colmap model_merger \
    --input_path1 /path/to/sub-model1 \
    --input_path2 /path/to/sub-model2 \
    --output_path /path/to/merged-model
```


The two folders should contain the output of the reconstruction process, i.e., either the `cameras.bin, images.bin, points3D.bin `files or the `cameras.txt, images.txt, points3D.txt` files.


To improve the quality of the alignment between the two sub-models, it is recommended to run another global bundle adjustment after the merge: 

```
colmap bundle_adjuster \
    --input_path /path/to/merged-model \
    --output_path /path/to/refined-merged-model
```

## 3.4 Merge two Colmap Databases

Merging two COLMAP databases can be useful when you have processed different subsets of a dataset separately and want to bring the results together into a single database. To merge two COLMAP databases, follow these steps:

COLMAP provides a tool named `database_merger` specifically for merging two databases.

```bash
colmap database_merger \
	--database_path1 path/to/database1.db \
	--database_path2 path/to/database2.db \
	--output_path path/to/merged_database.db
```

Replace `path/to/database1.db`, `path/to/database2.db`, and `path/to/merged_database.db` with your actual paths. 

This command will merge the contents of `database1.db` and `database2.db` into `merged_database.db`.


## 3.5 Rig bundle Adjuster
What the rig bundle adjuster does is it takes a 3D reconstruction as input and performs some form of constrained bundle adjustment. The constraint here comes from using a multi-camera rig. During reconstruction, Colmap does not enforce relative pose constraints between images taken at the same point in time by the different cameras in the multi-camera rig. This is done as a post-processing step by the rig bundle adjuster.
To this end, you first need to define the multi-camera rig (as explained in the documentation starting in the above code snippet).

Note that the purpose of the rig bundle adjuster is not pose graph optimization. Rather it tries to ensure a rigid movement of the multi-camera rig, i.e., the relative pose between two cameras in the rig should stay the same over all snapshots (images taken by the multi-camera rig at the same point in time).
This is not the same as pose graph optimization, where relative poses between images are used as measurements.

Refs: [1](https://github.com/colmap/colmap/issues/891)

<img src="images/camera_rig.png" alt="camera_rig" width="40%" height="40%" />


An example configuration of a single camera rig:

```
 [
   {
     "ref_camera_id": 1,
     "cameras":
     [
       {
           "camera_id": 1,
           "image_prefix": "left1_image"
           "rel_tvec": [0, 0, 0],
           "rel_qvec": [1, 0, 0, 0]
       },
       {
           "camera_id": 2,
           "image_prefix": "left2_image"
           "rel_tvec": [0, 0, 0],
           "rel_qvec": [0, 1, 0, 0]
       },
       {
           "camera_id": 3,
           "image_prefix": "right1_image"
           "rel_tvec": [0, 0, 0],
           "rel_qvec": [0, 0, 1, 0]
       },
       {
           "camera_id": 4,
           "image_prefix": "right2_image"
           "rel_tvec": [0, 0, 0],
           "rel_qvec": [0, 0, 0, 1]
       }
     ]
   }
 ]
```

The "camera_id" and "image_prefix" fields are required, whereas the
"rel_tvec" and "rel_qvec" fields optionally specify the relative
extrinsics of the camera rig in the form of a translation vector and a
rotation quaternion. The relative extrinsics rel_qvec and rel_tvec transform
coordinates from rig to camera coordinate space. If the relative extrinsics
are not provided then they are automatically inferred from the
reconstruction.

This file specifies the configuration for a single camera rig and that you
could potentially define multiple camera rigs. The rig is composed of 4
cameras: all images of the first camera must have "left1_image" as a name
prefix, e.g., "left1_image_frame000.png" or "left1_image/frame000.png".
Images with the same suffix ("_frame000.png" and "/frame000.png") are
assigned to the same snapshot, i.e., they are assumed to be captured at the
same time. Only snapshots with the reference image registered will be added
to the bundle adjustment problem. The remaining images will be added with
independent poses to the bundle adjustment problem. The above configuration
could have the following input image file structure:

```
    /path/to/images/...
        left1_image/...
            frame000.png
            frame001.png
            frame002.png
            ...
        left2_image/...
            frame000.png
            frame001.png
            frame002.png
            ...
        right1_image/...
            frame000.png
            frame001.png
            frame002.png
            ...
        right2_image/...
            frame000.png
            frame001.png
            frame002.png
            ...
```

you can call `rig_bundle_adjuster` to run the bundle adjuster for a known rig mode:

```bash
colmap rig_bundle_adjuster --input_path $DATASET_PATH/sparse/0 --output_path $DATASET_PATH/sparse/rig --rig_config_path $DATASET_PATH/rig_config.json
```

Example in the [documentation](https://github.com/colmap/colmap/blob/main/src/colmap/exe/sfm.cc)  

Example of parameters in the  [test file](https://github.com/colmap/colmap/blob/main/src/colmap/scene/camera_rig_test.cc)  



Refs: [1](https://github.com/colmap/colmap/issues/1624), [2](https://pdfs.semanticscholar.org/b01d/3c3cd7b43e58a344c8ea40d08aa87d63b13f.pdf)





## 3.6 Register/Localize New Images Into an Existing Reconstruction

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



## 3.7 Manhattan World Alignment

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





## 3.8 Loop Closure
Colmap can add images in arbitrary order. Colmap prefers to use image pairs that are not pure forward motion for **initialization**. 

Try skipping frames, eg, use only a frame every 1 or 2 seconds. This should improve reduce drift for video data. You could also try to I crease the size of the local bundle adjustment window.  Otherwise, COLMAP probably detects the loops successfully but cannot optimize them as well as ORBSLAM, which is specifically optimized for the SLAM scenario. It also does pose graph optimization if I remember correctly, which is better in correcting for large drift than pure bundle adjustment used in COLMAP.

Refs: [1](https://github.com/colmap/colmap/issues/1521), [2](https://github.com/colmap/colmap/issues/254)



## External Dense Reconstruction Libraries
If you do not have a CUDA-enabled GPU but some other GPU, you can use all COLMAP functionality except the dense reconstruction part. You can use external dense reconstruction software as an alternative. COLMAP exports to several other dense reconstruction libraries, .

- CMVS/PMVS [furukawa10]
- CMP-MVS [jancosek11]
- Line3D++ [hofer16].

Refs: [1](https://colmap.github.io/tutorial.html#dense-reconstruction)

**Dense Point-Cloud with OpenMVS**

```
DATASET_PATH=<you-dataset>
```


```
./InterfaceCOLMAP -i $DATASET_PATH/dense/ -o scene.mvs --image-folder $DATASET_PATH/dense/images
```



Refs: [1](https://github.com/cdcseacave/openMVS/wiki/Usage#convert-sfm-scene-from-colmap), [2](https://github.com/cdcseacave/openMVS_sample)


## Improving Dense Reconstruction of Weakly Textured Surfaces

Refs: [1](https://colmap.github.io/faq.html#improving-dense-reconstruction-results-for-weakly-textured-surfaces)


## Speedup Dense Reconstruction


Refs: [1](https://colmap.github.io/faq.html#speedup-dense-reconstruction)





## Instant-ngp

```
python3 /home/$USER/workspace/instant-ngp/scripts/colmap2nerf.py --text /home/$USER/<above-output> --images /home/$USER/<images>
```


## Multiple View Triangulation

`feature_importer` and `matches_importer`

Refs: [1](https://robotics.stackexchange.com/questions/16132/multiple-view-triangulation-method-used-by-colmap), [2](https://github.com/colmap/colmap/issues/688)






## Colmap SLAM


Paper: [COLMAP-SLAM: A FRAMEWORK FOR VISUAL ODOMETRY](https://isprs-archives.copernicus.org/articles/XLVIII-1-W1-2023/317/2023/isprs-archives-XLVIII-1-W1-2023-317-2023.pdf)  
[code](https://github.com/3DOM-FBK/COLMAP_SLAM)








