# ------------------------------------------------------------
# Use official OpenCV image as base
# ------------------------------------------------------------
FROM opencvcourses/opencv-docker:latest

LABEL maintainer="Behnam Asadi <behnam.asadi@gmail.com>"

ENV TZ=Europe/Berlin

# Install additional dependencies needed for the project
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        cmake \
        build-essential \
        pkg-config \
        libeigen3-dev \
        libsuitesparse-dev \
        libva-dev \
        ffmpeg \
        libcanberra-gtk-module \
        libcanberra-gtk3-module \
        libgtk2.0-dev \
        locales \
        x11-apps \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

WORKDIR /
