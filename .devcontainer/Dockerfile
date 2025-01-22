FROM ubuntu:jammy

ARG WORKSPACE
ENV WORKSPACE=${WORKSPACE}

ENV AMENT_CPPCHECK_ALLOW_SLOW_VERSIONS=1
ENV AMENT_PREFIX_PATH=/opt/ros/humble
ENV COLCON_PREFIX_PATH=/opt/ros/humble
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PATH=/opt/ros/humble/bin:$PATH
ENV PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages
ENV QT_X11_NO_MITSHM=1
ENV ROS_DISTRO=humble
ENV ROS_PYTHON_VERSION=3
ENV ROS_VERSION=2


# install packages & setup locales
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-utils \
    bash-completion \
    build-essential \
    ca-certificates \
    curl \
    dirmngr \
    gdb \
    git \
    gnupg2 \
    keyboard-configuration \
    locales \
    lsb-release \
    python3-pip \
    software-properties-common \
    vim \
    wget \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8; dpkg-reconfigure -f noninteractive locales

ENV LANG="en_US.UTF-8" \
    LANGUAGE="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8"


# install ROS2 & init rosdep
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ros-humble-ros-base \
    ros-humble-rviz2 \
    python3-autopep8 \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && /bin/bash -c 'source /opt/ros/humble/setup.bash; rosdep init'

# customize bash
COPY bashrc-git-prompt /tmp/
RUN curl -L https://raw.github.com/git/git/master/contrib/completion/git-prompt.sh >> /etc/skel/.bashrc \
    && cat /tmp/bashrc-git-prompt >> /etc/skel/.bashrc \
    && rm -rf /tmp/bashrc-git-prompt \
    && echo "source /opt/ros/humble/setup.bash" >> /etc/skel/.bashrc


# install python packages
RUN python3 -m pip --no-cache-dir install \
    setuptools==58.2.0 \
    && pip3 install pandas pyarrow rosbags 


# install your custom packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-joint-state-publisher-gui \
    ros-humble-joint-state-publisher \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-imu-tools \
    ros-humble-plotjuggler-ros \
    ros-humble-xacro \
    ros-humble-ros2-control \
    ros-humble-rqt-* \ 
    libglib2.0-dev \
    libboost-all-dev \
    libusb-1.0-0-dev \
    unzip \
    net-tools \
    iputils-ping \
    nano

RUN curl -L -o lcm-v1.4.0.tar.gz https://github.com/lcm-proj/lcm/archive/refs/tags/v1.4.0.tar.gz \
    && tar -xzvf lcm-v1.4.0.tar.gz \
    && rm lcm-v1.4.0.tar.gz \
    && cd lcm-1.4.0 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install

RUN curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip \
    && unzip libtorch.zip \
    && rm libtorch.zip \
    && mv libtorch /usr/local/lib/

ENV Torch_DIR=/usr/local/lib/libtorch
ENV CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

WORKDIR /home
RUN git clone https://github.com/PPI-PUT/intention_ws.git \
    && cd intention_ws \
    && vcs import < deps.repos \
    && colcon build --packages-up-to --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=On -Wall -Wextra -Wpedantic
# set up entrypoint
COPY entrypoint /entrypoint
ENTRYPOINT ["/entrypoint"]

CMD ["/bin/bash"]
LABEL maintainer="Maciej Krupka (maciej.p.krupka@student.put.poznan.pl)"
