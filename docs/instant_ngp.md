export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"


python3 ./scripts/colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 32 --images=/home/behnam/Pictures/south-building/images/_resized



export PATH="/usr/local/cuda-11.3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"


export PATH="/usr/local/cuda-12.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"





/home/behnam/usr/bin/cmake -S . -B build -DCMAKE_INSTALL_PREFIX=~/usr
/home/behnam/usr/bin/cmake --build build --config RelWithDebInfo -j






export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"


export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDAHOSTCXX=/usr/bin/g++-9

-Dglog_DIR=/home/behnam/usr/lib/cmake/glog 

/home/behnam/usr/bin/cmake --build build -j16
/home/behnam/usr/bin/cmake --install build


to set the CMake:

export PATH="/home/behnam/usr/bin:$PATH"
export LD_LIBRARY_PATH="/home/behnam/usr/lib:$LD_LIBRARY_PATH"











ffmpeg -i video.mp4 -r 1 -vf scale=960:-1  image%05d.jpg


467.6804713692255,350.30927995207475,479.875,269.875,-0.04404631655181317, 0.05060128895587718, -0.07711300524376409, 0.03136288703337228




--cmake-args -DCeres_DIR=/home/behnam/USR/lib/cmake/Ceres

catkin config --cmake-args -Dglog_DIR=/home/behnam/usr/lib/cmake/glog




catkin config --cmake-args  -DCMAKE_CXX_STANDARD=17  -DCMAKE_CXX_FLAGS='-std=c++17' -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_EXTENSIONS=OFF  
























sudo update-alternatives --display cuda

sudo update-alternatives --config cuda



/home/behnam/usr/bin/cmake ./thirdparty/gtsam -DGTSAM_BUILD_PYTHON=1 -B build_gtsam
/home/behnam/usr/bin/cmake --build build_gtsam --config RelWithDebInfo -j18



cmake -DPYTHON_LIBRARY=/home/behnam/anaconda3/envs/NeRF-SLAM/lib/libpython3.8.so \
-DPYTHON_INCLUDE_DIR=/home/behnam/anaconda3/envs/NeRF-SLAM/include/python3.8 \
-DPYTHON_EXECUTABLE=/home/behnam/anaconda3/envs/NeRF-SLAM/bin/python3



/home/behnam/usr/bin/cmake ./thirdparty/gtsam  -DGTSAM_PYTHON_VERSION=3.8.10 -DGTSAM_BUILD_PYTHON=1 -DPython3_LIBRARY=/home/behnam/anaconda3/envs/NeRF-SLAM/lib/libpython3.8.so \
-DPython3_INCLUDE_DIR=/home/behnam/anaconda3/envs/NeRF-SLAM/include/python3.8 \
-DPython3_EXECUTABLE=/home/behnam/anaconda3/envs/NeRF-SLAM/bin/python3 -B build_gtsam


cmake -DPython3_EXECUTABLE=/usr/local/bin/python3.8 
DPython3_FIND_STRATEGY=VERSION

/home/behnam/anaconda3/envs/NeRF-SLAM/lib/libpython3.8.so
/home/behnam/anaconda3/envs/NeRF-SLAM/include/python3.8
/home/behnam/anaconda3/envs/NeRF-SLAM/bin/python3




-DPython3_FIND_STRATEGY=VERSION









git clone https://github.com/gflags/gflags
cd gflags
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=~/usr 
cmake --build build -j8
cmake --install build 



git clone https://github.com/google/glog
cd glog
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=~/usr 
cmake --build build -j8
cmake --install build 


pip list -v


python3 -m pip show pip









