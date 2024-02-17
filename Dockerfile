# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

LABEL maintainer="Behnam Asadi <behnam.asadi@gmail.com>"

# Configure timezone and install packages
ENV DEBIAN_FRONTEND=noninteractive 
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt update && apt install -y build-essential cmake git openssh-server gdb pkg-config libeigen3-dev libsuitesparse-dev libva-dev ffmpeg libcanberra-gtk-module libcanberra-gtk3-module libgtk2.0-dev locales x11-apps

# Clone OpenCV_contrib and OpenCV, then build OpenCV
RUN echo "************************ opencv_contrib ************************" \
    && git clone https://github.com/opencv/opencv_contrib.git \
    && echo "************************ opencv ************************" \
    && git clone https://github.com/opencv/opencv.git \
    && mkdir -p opencv/build \
    && cd opencv/build \
    && cmake -DCMAKE_CXX_FLAGS=-std=c++1z -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules -DOPENCV_ENABLE_NONFREE=True -DBUILD_TIFF=True .. \
    && make -j$(nproc) all install

# Set the working directory back to the root
WORKDIR "/"

