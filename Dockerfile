# ------------------------------------------------------------
# Stage 1: Build OpenCV from source
# ------------------------------------------------------------
FROM ubuntu:24.04 AS opencv-builder

LABEL maintainer="Behnam Asadi <behnam.asadi@gmail.com>"

ARG OPENCV_VERSION=""   # Optional pin (e.g. "4.9.0")

ENV TZ=Europe/Berlin

# Install build dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        cmake \
        git \
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
        wget \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# Determine which tag to use
# If OPENCV_VERSION is empty, fetch latest release tag
RUN if [ -z "$OPENCV_VERSION" ]; then \
      export OPENCV_VERSION=$(curl -s https://api.github.com/repos/opencv/opencv/releases/latest | grep -Po '"tag_name": "\K.*?(?=")') ; \
      echo "Using latest OpenCV release: $OPENCV_VERSION" ; \
    else \
      echo "Using pinned OpenCV version: $OPENCV_VERSION" ; \
    fi \
    && git clone --branch ${OPENCV_VERSION:-master} --depth=1 https://github.com/opencv/opencv.git /opencv \
    && git clone --branch ${OPENCV_VERSION:-master} --depth=1 https://github.com/opencv/opencv_contrib.git /opencv_contrib

# Build OpenCV
RUN mkdir -p /opencv/build \
    && cd /opencv/build \
    && cmake \
        -DCMAKE_CXX_FLAGS=-std=c++17 \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DBUILD_TIFF=ON \
        .. \
    && make -j$(nproc) \
    && make install \
    && rm -rf /opencv /opencv_contrib

# ------------------------------------------------------------
# Stage 2: Minimal runtime image
# ------------------------------------------------------------
FROM ubuntu:24.04

LABEL maintainer="Behnam Asadi <behnam.asadi@gmail.com>"

ENV TZ=Europe/Berlin

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        cmake \
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

COPY --from=opencv-builder /usr/local/ /usr/local/

WORKDIR /


# Copy OpenCV from build stage
COPY --from=opencv-builder /usr/local/ /usr/local/

WORKDIR /
