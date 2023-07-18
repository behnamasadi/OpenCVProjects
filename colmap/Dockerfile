FROM nvidia/cuda:11.6.0-devel-ubuntu20.04


MAINTAINER Behnam Asadi behnam.asadi@gmail.com


# this is for timezone config
ENV DEBIAN_FRONTEND=noninteractive 
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt upgrade -y
#-y is for accepting yes when the system asked us for installing the package




# 1) Dependencies
RUN echo "************************ Dependencies ************************"
RUN apt-get install -y build-essential cmake git openssh-server gdb pkg-config valgrind systemd-coredump apt-utils libeigen3-dev libsuitesparse-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgflags-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libatlas-base-dev libsuitesparse-dev libflann-dev libvtk7-dev libboost-all-dev libqhull-dev ffmpeg x11-apps mesa-utils pciutils libsqlite3-dev libboost-test-dev libboost-system-dev libboost-graph-dev libboost-filesystem-dev libboost-program-options-dev




# 2) Ceres Solver
RUN echo "************************ Ceres Solver ************************"
RUN git clone https://ceres-solver.googlesource.com/ceres-solver
RUN cd ceres-solver
WORKDIR "/ceres-solver"
RUN git checkout $(git describe --tags) # Checkout the latest release
RUN mkdir  build && cd build
WORKDIR "/ceres-solver/build"
#RUN cmake -DCMAKE_CXX_FLAGS=-std=c++1z   -DGOOGLETEST_PATH=../../googletest   -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ../ && cmake --build .  --parallel && cmake --install .
RUN cmake ../ -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && make -j6 && make install
WORKDIR "/"





# 3)  COLMAP
RUN echo "************************ COLMAP ************************"
RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap
WORKDIR "/colmap"
RUN git checkout dev
RUN mkdir build && cd build
WORKDIR "/colmap/build"
RUN cmake ../ && make -j6 && make install
WORKDIR "/"



# 4)  PointCloud Library 
#RUN echo "************************ PointCloud Library ************************"
#RUN git clone https://github.com/PointCloudLibrary/pcl
#RUN cd pcl
#WORKDIR "/pcl"
#RUN mkdir build && cd build
#WORKDIR "/pcl/build"
#RUN cmake ../ && make -j3 && make install
#WORKDIR "/"







