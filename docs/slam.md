# SLAM


# GraphSLAM


1. <img src="https://latex.codecogs.com/svg.latex?%5E%7Bi%7D_jR%3D%5E%7Bw%7D_iR%5ET%20%7B%5E%7Bw%7D_%7Bj%7DR%7D" alt="https://latex.codecogs.com/svg.latex?{^{w}_{j}R}" />

2. <img src="https://latex.codecogs.com/svg.latex?%5E%7Bi%7D_jP%3D%5E%7Bw%7D_iR%5ET%20%28%20%7B%5E%7Bw%7D_%7Bj%7DP%7D%20-%7B%5E%7Bw%7D_%7Bi%7DP%7D%20%29" alt="https://latex.codecogs.com/svg.latex?^{i}_jP=^{w}_iR^T (  {^{w}_{j}P} -{^{w}_{i}P} ) " />

which gives us: 
- <img src="https://latex.codecogs.com/svg.latex?%5E%7Bi%7D_j%20%5Ctheta%20%3D%5E%7Bw%7D_j%5Ctheta%20-%7B%5E%7Bw%7D_%7Bi%7D%5Ctheta%7D" alt="https://latex.codecogs.com/svg.latex?^{i}_j \theta =^{w}_j\theta -{^{w}_{i}\theta}" />





- <img src="https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7D%20%5E%7Bi%7D_j%20x%20%5C%5C%20%5E%7Bi%7D_j%20y%20%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7D%20cos%28%5Ctheta1%29%20%26%20sin%28%5Ctheta1%29%20%5C%5C%20-sin%28%5Ctheta1%29%20%26%20cos%28%5Ctheta1%29%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%20%5E%7Bw%7D_jx%20-%20%5E%7Bw%7D_ix%20%5C%5C%20%5E%7Bw%7D_jy%20-%20%5E%7Bw%7D_iy%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\begin{bmatrix} ^{i}_j x \\  ^{i}_j y \end{bmatrix}=\begin{bmatrix} cos(\theta1) & sin(\theta1) \\ -sin(\theta1) & cos(\theta1) \end{bmatrix} \begin{bmatrix} ^{w}_jx - ^{w}_ix \\  ^{w}_jy - ^{w}_iy \end{bmatrix}" />


our state is:
<br/>

<img src="https://latex.codecogs.com/svg.latex?%5Cmathbb%7BX%7D%3D%28x_1%2Cy_1%2C%5Ctheta_1%2C%20x_2%2Cy_2%2C%20%5Ctheta_2%2C%20...%20%2Cx_n%2Cy_n%2C%5Ctheta_n%29" alt="https://latex.codecogs.com/svg.latex?\mathbb{X}=(x_1,y_1,\theta_1, x_2,y_2, \theta_2, ... ,x_n,y_n,\theta_n)" />

<br/>

our error function is:
<br/>

<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20%5Ctextbf%7Be%7D_%7Bij%7D%28%5Cmathbb%7BX%7D%29%3D%5Ctextbf%7Be%7D_%7Bij%7D%28%5Ctextbf%7BX%7D_%7Bi%7D%2C%5Ctextbf%7BX%7D_%7Bj%7D%29%20%5C%5C%20%5Ctextbf%7BX%7D_%7Bi%7D%3D%28x_i%2Cy_i%2C%5Ctheta_i%29%20%5C%5C%20%5Ctextbf%7BX%7D_%7Bj%7D%3D%28x_j%2Cy_j%2C%5Ctheta_j%29" alt="\\\textbf{e}_{ij}(\mathbb{X})=\textbf{e}_{ij}(\textbf{X}_{i},\textbf{X}_{j})  \\ \textbf{X}_{i}=(x_i,y_i,\theta_i) \\ \textbf{X}_{j}=(x_j,y_j,\theta_j) " />


<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20%5Ctextbf%7Be%7D_%7Bij%7D%28%5Ctextbf%7BX%7D_%7Bi%7D%2C%5Ctextbf%7BX%7D_%7Bj%7D%29%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20%5Ei_jx%3Dcos%28%5Ctheta_i%29%28x_j-x_i%29%20&plus;sin%28%5Ctheta_i%29%28y_j-y_i%29%20%5C%5C%20%5Ei_jy%3D-sin%28%5Ctheta_i%29%28x_j-x_i%29&plus;cos%28%5Ctheta_i%29%28y_j-y_i%29%20%5C%5C%20%5E%7Bi%7D_j%20%5Ctheta%20%3D%5E%7Bw%7D_j%5Ctheta%20-%7B%5E%7Bw%7D_%7Bi%7D%5Ctheta%7D%20%5Cend%7Bmatrix%7D%5Cright." alt="https://latex.codecogs.com/svg.latex?\\
\textbf{e}_{ij}(\textbf{X}_{i},\textbf{X}_{j})= \left\{\begin{matrix}  
^i_jx=cos(\theta_i)(x_j-x_i) +sin(\theta_i)(y_j-y_i) \\  ^i_jy=-sin(\theta_i)(x_j-x_i)+cos(\theta_i)(y_j-y_i) \\ ^{i}_j \theta =^{w}_j\theta -{^{w}_{i}\theta} \end{matrix}\right." />



<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20J_%7Bij%7D%3D%5Cbegin%7Bbmatrix%7D%200%20%26%20...%20%26%200%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_jx%7D%7B%20%5Cpartial%20_%7Bi%7D%20x%7D%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_jx%7D%7B%20%5Cpartial%20_%7Bi%7D%20y%7D%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_j%20x%7D%7B%20%5Cpartial%20_%7Bi%7D%20%5Ctheta%7D%20%260%20%26...%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_jx%7D%7B%20%5Cpartial%20_%7Bj%7D%20x%7D%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_jx%7D%7B%20%5Cpartial%20_%7Bj%7D%20y%7D%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_j%20x%7D%7B%20%5Cpartial%20_%7Bj%7D%20%5Ctheta%7D%20%26...%20%260%5C%5C%200%20%26%20...%20%26%200%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_jy%7D%7B%20%5Cpartial%20_%7Bi%7D%20x%7D%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_jy%7D%7B%20%5Cpartial%20_%7Bi%7D%20y%7D%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_j%20y%7D%7B%20%5Cpartial%20_%7Bi%7D%20%5Ctheta%7D%20%260%20%26...%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_jy%7D%7B%20%5Cpartial%20_%7Bj%7D%20x%7D%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_jy%7D%7B%20%5Cpartial%20_%7Bj%7D%20y%7D%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_j%20y%7D%7B%20%5Cpartial%20_%7Bj%7D%20%5Ctheta%7D%26%20...%20%26%200%5C%5C%200%20%26%20...%20%26%200%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_j%5Ctheta%7D%7B%20%5Cpartial%20_%7Bi%7D%20x%7D%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_j%5Ctheta%7D%7B%20%5Cpartial%20_%7Bi%7D%20y%7D%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_j%20%5Ctheta%7D%7B%20%5Cpartial%20_%7Bi%7D%20%5Ctheta%7D%20%260%20%26...%26%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_j%5Ctheta%7D%7B%20%5Cpartial%20_%7Bj%7D%20x%7D%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_j%5Ctheta%7D%7B%20%5Cpartial%20_%7Bj%7D%20y%7D%20%26%20%5Cfrac%7B%5Cpartial%20%5E%7Bi%7D_j%20%5Ctheta%7D%7B%20%5Cpartial%20_%7Bj%7D%20%5Ctheta%7D%20%26...%20%26%200%5C%5C%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?\\ J_{ij}=\begin{bmatrix} 0 & ... & 0 & \frac{\partial ^{i}_jx}{ \partial _{i} x}      &  \frac{\partial ^{i}_jx}{ \partial _{i} y}  & \frac{\partial ^{i}_j x}{ \partial _{i} \theta} &0 &... & \frac{\partial ^{i}_jx}{ \partial _{j} x}      &  \frac{\partial ^{i}_jx}{ \partial _{j} y}  & \frac{\partial ^{i}_j x}{ \partial _{j} \theta}  &... &0\\ 0 & ... & 0 & \frac{\partial ^{i}_jy}{ \partial _{i} x} &  \frac{\partial ^{i}_jy}{ \partial _{i} y}  & \frac{\partial ^{i}_j y}{ \partial _{i} \theta} &0   &...& \frac{\partial ^{i}_jy}{ \partial _{j} x} &  \frac{\partial ^{i}_jy}{ \partial _{j} y}  & \frac{\partial ^{i}_j y}{ \partial _{j} \theta}& ...     & 0\\ 0 & ... & 0 & \frac{\partial ^{i}_j\theta}{ \partial _{i} x}      &  \frac{\partial ^{i}_j\theta}{ \partial _{i} y}  & \frac{\partial ^{i}_j \theta}{ \partial _{i} \theta} &0 &...&\frac{\partial ^{i}_j\theta}{ \partial _{j} x}      &  \frac{\partial ^{i}_j\theta}{ \partial _{j} y}  & \frac{\partial ^{i}_j \theta}{ \partial _{j} \theta} &...  & 0\\ \end{bmatrix}" />


<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?A_%7Bi%2Cj%7D%3D%5Cbegin%7Bbmatrix%7D%20-cos%28%5Ctheta_i%29%20%26%20-sin%28%5Ctheta_i%29%20%26%20-sin%28%5Ctheta_i%29%28x_j-x_i%29&plus;cos%28%5Ctheta_i%29%28y_j-y_i%29%5C%5C%20sin%28%5Ctheta_i%29%20%26%20-cos%28%5Ctheta_i%29%20%26%20-cos%28%5Ctheta_i%29%28x_j-x_i%29%20-%20sin%28%5Ctheta_i%29%28y_j-y_i%29%5C%5C%200%20%26%200%20%26%20-1%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?A_{i,j}=\begin{bmatrix} -cos(\theta_i) & -sin(\theta_i)  & -sin(\theta_i)(x_j-x_i)+cos(\theta_i)(y_j-y_i)\\ sin(\theta_i) & -cos(\theta_i) & -cos(\theta_i)(x_j-x_i) - sin(\theta_i)(y_j-y_i)\\ 0 & 0 & -1 \end{bmatrix}" />

<br/>
<br/>


<img src="https://latex.codecogs.com/svg.latex?B_%7Bi%2Cj%7D%3D%5Cbegin%7Bbmatrix%7D%20cos%28%5Ctheta_i%29%20%26%20-sin%28%5Ctheta_i%29%20%26%200%5C%5C%20-sin%28%5Ctheta_i%29%20%26%20cos%28%5Ctheta_i%29%20%26%200%5C%5C%200%20%26%200%20%26%201%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?B_{i,j}=\begin{bmatrix} cos(\theta_i) & -sin(\theta_i)  & 0\\  -sin(\theta_i) & cos(\theta_i) & 0\\  0 & 0 & 1 \end{bmatrix}" />

<br/>
<br/>




<img src="images/J_i_j.png"  width= "50%"  height= "50%"/>

<br/>
<br/>
<img src="images/b_h.png" width= "50%"  height= "50%"/>

<br/>
<br/>
<img src="images/b_i_j_h_i_j.png" width= "50%"  height= "50%" />

<br/>
<br/>
<img src="images/building_the_linear_system.png" width= "50%"  height= "50%" />






Refs: [1](https://python-graphslam.readthedocs.io/en/stable/)


# g2o 
g2o is a C++ framework for optimizing graph-based nonlinear error functions
```
git clone https://github.com/gabime/spdlog.git
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/home/behnam/usr -DSPDLOG_BUILD_SHARED=ON
cmake --build build
cmake --install build
```

`sudo apt install libsuitesparse-dev qtdeclarative5-dev qt5-qmake libqglviewer-dev-qt5`


```
git clone https://github.com/RainerKuemmerle/g2o/
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/home/behnam/usr -DSPDLOG_BUILD_SHARED=ON
cmake --build build
cmake --install build
```

## g2o-python


```
pip install -U g2o-python
```
Refs: [1](https://github.com/miquelmassot/g2o-python)


## File Format SLAM 2D

In a graph-based optimization problem, you typically have a set of variables (also known as nodes) and constraints (also known as edges) between these variables. The g2o format provides a way to express these variables and constraints in a text-based file.

The g2o file consists of several sections, each indicated by a specific keyword:


## Vertices
syntax: 

`TAG ID CURRENT_ESTIMATE`

Examples:


### 2D Robot Pose

`VERTEX_SE2 i x y theta`

`VERTEX_SE2 4 0.641008 -0.011200 -0.007444`

### 2D Landmarks / Features

`VERTEX_XY i x y`

### Edges / Constraints

`TAG ID_SET MEASUREMENT INFORMATION_MATRIX`

The odometry of a robot connects subsequent vertices with a **relative transformation** which specifies how the robot moved according to its measurements. For a compact documentation we employ the following helper function.

`EDGE_SE2 i j x y theta info(x, y, theta)`

Where <img src="https://latex.codecogs.com/svg.latex?z_%7Bij%7D%20%3D%20%5Bx%2C%20y%2C%20%5Ctheta%5D%5ET" alt="https://latex.codecogs.com/svg.latex?z_{ij} = [x, y, \theta]^T" /> is the measurement moving from <img src="https://latex.codecogs.com/svg.latex?x_i" alt="https://latex.codecogs.com/svg.latex?x_i" />  to <img src="https://latex.codecogs.com/svg.latex?x_j" alt="https://latex.codecogs.com/svg.latex?x_j" />, i.e. <img src="https://latex.codecogs.com/svg.latex?x_j%20%3D%20x_i%20%5Coplus%20z_ij" alt="https://latex.codecogs.com/svg.latex?x_j = x_i \oplus z_ij" />



`EDGE_SE2 24 25 0.645593 0.014612 0.008602 11.156105 -3.207460 0.000000 239.760661 0.000000 2457.538661`



## File Format SLAM 3D

### 3D Robot Pose

`VERTEX_SE3:QUAT i x y z qx qy qz qw`

`VERTEX_SE3:QUAT 0 0 0 0 0 0 1`

### 3D Point


`VERTEX_TRACKXYZ i x y z`

### Edges / Constraints


`EDGE_SE3:QUAT 0 1 0 0 0 0 0 0 0 1 0 0 0 0.1 0 0 0 0.1 0 0.1 0.1 0.1 0.1`

`ID_SET`: is a list of vertex IDs which specifies to which vertices the edge is connected.

`MEASUREMENT` : The information matrix or precision matrix which represent the uncertainty of the measurement error is the inverse of the covariance matrix. Hence, it is symmetric and positive semi-definite. We typically only store the upper-triangular block of the matrix in row-major order. For example, if the information matrix <img src="https://latex.codecogs.com/svg.latex?%5Cmathbf%7B%5COmega%7D" alt="https://latex.codecogs.com/svg.latex?\mathbf{\Omega}"  /> is a `3x3`  


<img src="https://latex.codecogs.com/svg.latex?q_%7B11%7D%20%5C%3B%20q_%7B12%7D%20%5C%3B%20q_%7B13%7D%20%5C%3B%20q_%7B22%7D%20%5C%3B%20q_%7B23%7D%20%5C%3B%20q_%7B33%7D" alt="https://latex.codecogs.com/svg.latex?q_{11} \; q_{12} \; q_{13} \; q_{22} \; q_{23} \; q_{33}" />

[Examples](https://github.com/RainerKuemmerle/g2o/tree/pymem/python/examples)


Refs: [1](https://github.com/RainerKuemmerle/g2o/wiki/File-Format-SLAM-2D)






## Multivariate Gaussians

### Moments parameterization
<img src="https://latex.codecogs.com/svg.latex?p%28%5Cmathbf%7Bx%7D%20%29%3D%7B%5Cdisplaystyle%20%282%5Cpi%20%29%5E%7B-k/2%7D%5Cdet%28%7B%5Cboldsymbol%20%7B%5CSigma%20%7D%7D%29%5E%7B-1/2%7D%5C%2C%5Cexp%20%5Cleft%28-%7B%5Cfrac%20%7B1%7D%7B2%7D%7D%28%5Cmathbf%20%7Bx%7D%20-%7B%5Cboldsymbol%20%7B%5Cmu%20%7D%7D%29%5E%7B%5Cmathsf%20%7BT%7D%7D%7B%5Cboldsymbol%20%7B%5CSigma%20%7D%7D%5E%7B-1%7D%28%5Cmathbf%20%7Bx%7D%20-%7B%5Cboldsymbol%20%7B%5Cmu%20%7D%7D%29%5Cright%29%2C%7D" alt="https://latex.codecogs.com/svg.latex?p(\mathbf{x} )={\displaystyle (2\pi )^{-k/2}\det({\boldsymbol {\Sigma }})^{-1/2}\,\exp \left(-{\frac {1}{2}}(\mathbf {x} -{\boldsymbol {\mu }})^{\mathsf {T}}{\boldsymbol {\Sigma }}^{-1}(\mathbf {x} -{\boldsymbol {\mu }})\right)}" />

## Canonical Parameterization
Alternative representation for Gaussians

<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20%5COmega%20%3D%5CSigma%5E%7B-1%7D%20%5C%5C%20%5Cxi%20%3D%5CSigma%5E%7B-1%7D%5Cmu" alt="https://latex.codecogs.com/svg.latex?\\
\Omega =\Sigma^{-1}
\\
\xi =\Sigma^{-1}\mu" />


<img src="https://latex.codecogs.com/svg.latex?p%28%5Cmathbf%7Bx%7D%20%29%3D%5Cfrac%7Bexp%28-%5Cfrac%7B1%7D%7B2%7D%5Cmu%5ET%5Cxi%20%29%7D%7Bdet%282%5Cpi%5COmega%5E%7B-1%7D%29%5E%7B%5Cfrac%7B1%7D%7B2%7D%7D%20%7Dexp%28-%5Cfrac%7B1%7D%7B2%7D%5Cmathbf%7Bx%7D%5ET%5COmega%20%5Cmathbf%7Bx%7D&plus;%5Cmathbf%7Bx%7D%5ET%5Cxi%20%29" alt="https://latex.codecogs.com/svg.latex?p(\mathbf{x} )=\frac{exp(-\frac{1}{2}\mu^T\xi )}{det(2\pi\Omega^{-1})^{\frac{1}{2}} }exp(-\frac{1}{2}\mathbf{x}^T\Omega \mathbf{x}+\mathbf{x}^T\xi  )" />


<br/>

<img src="images/towards_the_information_form.jpg" alt="images/towards_the_information_form.jpg" width= "50%"  height= "50%" />

<br/>

## Further reading


- simple_2d_slam
Refs: [1](https://github.com/goldbattle/simple_2d_slam)

- Robust Pose-graph Optimization
Refs: [1](https://www.youtube.com/watch?v=zOr9HreMthY)

- Awesome Visual Odometry
Refs: [1](https://github.com/chinhsuanwu/awesome-visual-odometry)

- Monocular Video Odometry Using OpenCV
Refs: [1](https://github.com/alishobeiri/Monocular-Video-Odometery)

- modern-slam-tutorial-python
Refs: [1](https://github.com/gisbi-kim/modern-slam-tutorial-python)

- Monocular-Video-Odometery
Refs: [1](https://github.com/alishobeiri/Monocular-Video-Odometery/blob/master/monovideoodometery.py)
  
- Matrix Lie Groups for Robotics
Refs: [1](https://www.youtube.com/watch?v=NHXAnvv4mM8&list=PLdMorpQLjeXmbFaVku4JdjmQByHHqTd1F&index=8)   

- Factor Graph - 5 Minutes with Cyrill
Refs: [1](https://www.youtube.com/watch?v=uuiaqGLFYa4&t=145s)

- GTSAM: Georgia Tech Smoothing and Mapping Library  
Refs [1](https://gtsam.org/), [2](https://github.com/borglab/gtsam)

- DBoW2: library for indexing and converting images into a bag-of-word representation  
Refs: [1](https://github.com/dorian3d/DBoW2)

- iSAM: Incremental Smoothing and Mapping  
Refs: [1](https://openslam-org.github.io/iSAM)



## python-graphslam

Refs: [1](https://github.com/JeffLIrion/python-graphslam), [2](https://python-graphslam.readthedocs.io/en/stable/)



## add apriltag to loop closure

Refs: [1](https://berndpfrommer.github.io/tagslam_web/)



## DROID-SLAM

## Hierarchical-Localization

## image-matching-webui

## LightGlue

## Nerf-SLAM

## DenseSFM

Refs: [1](https://github.com/tsattler/visuallocalizationbenchmark)

## Pixel-Perfect Structure-from-Motion
Refs: [1](https://github.com/cvg/pixel-perfect-sfm)

## ODM
```
docker run -ti --rm -v /home/$USER/workspace/odm_projects/datasets/code/:/datasets/code opendronemap/odm --project-path /datasets
```
[Datasets](https://www.opendronemap.org/odm/datasets/)

## GTSAM
Refs: [1](https://www.youtube.com/watch?v=zOr9HreMthY)
