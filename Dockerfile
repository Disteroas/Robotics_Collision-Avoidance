FROM osrf/ros:humble-desktop

RUN apt-get update && apt-get install -y \
    python3-pip \
    nano \
    ros-humble-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/*

# Trucco PRO: Aumentiamo il timeout a 1000 secondi e chiediamo esplicitamente la versione CPU leggerissima
RUN pip3 install --default-timeout=1000 torch numpy --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install pytest gymnasium

WORKDIR /home/usv_ws

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
