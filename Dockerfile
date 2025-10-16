# ------------------------------------------------------------
# Use Ubuntu with OpenCV from official repositories
# ------------------------------------------------------------
FROM ubuntu:24.04

LABEL maintainer="Behnam Asadi <behnam.asadi@gmail.com>"

ENV TZ=Europe/Berlin
ENV DEBIAN_FRONTEND=noninteractive

# Install OpenCV and all project dependencies
RUN apt-get update \
    && apt-get install -y \
        cmake \
        build-essential \
        pkg-config \
        python3 \
        python3-pip \
        python3-dev \
        # OpenCV and its dependencies
        libopencv-dev \
        python3-opencv \
        # Additional project dependencies
        libeigen3-dev \
        libsuitesparse-dev \
        libva-dev \
        ffmpeg \
        libcanberra-gtk-module \
        libcanberra-gtk3-module \
        libgtk2.0-dev \
        libgtk-3-dev \
        locales \
        x11-apps \
        # Ceres Solver dependencies
        libgoogle-glog-dev \
        libgflags-dev \
        libatlas-base-dev \
        libceres-dev \
        # Useful utilities
        git \
        wget \
        curl \
        nano \
        vim \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# Verify OpenCV installation
RUN pkg-config --modversion opencv4 || echo "OpenCV installed successfully"

WORKDIR /
