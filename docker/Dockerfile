FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

COPY deps/neuralrecon-0.1-py3-none-any.whl neuralrecon-0.1-py3-none-any.whl
COPY docker/cuda-keyring_1.0-1_all.deb cuda-keyring_1.0-1_all.deb

RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list && dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
          git \
		cmake \
		build-essential \
		curl \
		wget \
		gnupg2 \
		lsb-release \
		ca-certificates \
            python3-pip \
		  python3-dev \
		  libopenblas-dev \
		  libopenmpi-dev \
            openmpi-bin \
            openmpi-common \
		  gfortran \
		  libomp-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update
RUN apt-get install -y libsparsehash-dev git vim ffmpeg libsm6 libxext6 freeglut3-dev

RUN pip3 install --no-cache-dir setuptools Cython wheel
RUN pip3 install --no-cache-dir --verbose numpy

RUN pip3 install --upgrade pip && pip3 install Cython scipy pycuda plyfile Ninja
RUN pip3 install torch==1.6.0 torchvision==0.7.0 ipython tqdm sparsehash ray transforms3d tqdm opencv-python numba pycuda tensorboardX scikit-image trimesh~=3.9.18 yacs h5py loguru gdown pyrender~=0.1.45 pyglet open3d 
RUN pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
RUN pip3 install neuralrecon-0.1-py3-none-any.whl


RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -


#
# install bootstrap dependencies
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
          libpython3-dev \
          python3-rosdep \
          python3-rosinstall-generator \
          python3-vcstool \
          build-essential && \
    rosdep init && \
    rosdep update && \
    rm -rf /var/lib/apt/lists/*


#
# download/build the ROS source
#
RUN mkdir ros_catkin_ws && \
    cd ros_catkin_ws && \
    rosinstall_generator desktop --rosdistro noetic --deps --tar > noetic-desktop.rosinstall && \
    mkdir src && \
    vcs import --input noetic-desktop.rosinstall ./src && \
    apt-get update && \
    rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro noetic --skip-keys python3-pykdl -y && \
    python3 ./src/catkin/bin/catkin_make_isolated --install --install-space ${ROS_ROOT} -DCMAKE_BUILD_TYPE=Release && \
    rm -rf /var/lib/apt/lists/*
