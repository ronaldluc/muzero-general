# Use python3.7
# See https://docs.docker.com/samples/library/python/ for latest 
#FROM python:3.9
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

LABEL maintainer="Tom Titcombe <t.j.titcombe@gmail.com>"

# Working directory is / by default. We explictly state it here for posterity
WORKDIR /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
        autoconf \
        automake \
        build-essential \
        cmake \
        git-core \
        libass-dev \
        libfreetype6-dev \
        libgnutls28-dev \
        libmp3lame-dev \
        libsdl2-dev \
        libtool \
        libva-dev \
        libvdpau-dev \
        libvorbis-dev \
        libxcb1-dev \
        libxcb-shm0-dev \
        libxcb-xfixes0-dev \
        meson \
        ninja-build \
        pkg-config \
        texinfo \
        wget \
        yasm \
        zlib1g-dev \
        libunistring-dev \
        libx264-dev

# ffmpeg does not have libx264
RUN mkdir ~/ffmpeg_sources && cd ~/ffmpeg_sources && \
wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
tar xjvf ffmpeg-snapshot.tar.bz2 && \
cd ffmpeg && \
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
  --prefix="$HOME/ffmpeg_build" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/ffmpeg_build/include" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="$HOME/bin" \
  --enable-gpl \
  --enable-libx264 && \
PATH="$HOME/bin:$PATH" make && \
make install && \
hash -r

# Upgrade pip3
RUN pip3 install --upgrade pip

# Move the requirements file into the image
COPY requirements.txt /tmp/

RUN conda install -c conda-forge ffmpeg
# Install the python requirements on the image
RUN pip3 install --trusted-host pypi.python.org -r /tmp/requirements.txt

# Remove the requirements file - this is no longer needed
RUN rm /tmp/requirements.txt

# Create a directory in which we can do our work
RUN mkdir /workdir/

# Set it as the working directory
WORKDIR /home/my_rl/

# Copy over the start-up script
ADD scripts/startup_script.sh /usr/local/bin/startup_script.sh

# Give permissions to execute
RUN chmod 777 /usr/local/bin/startup_script.sh

# Set the display when we run the container. This allows us to record without the user needing to type anything explicitly
# This code snippet was taken from https://github.com/duckietown/gym-duckietown/issues/123
ENTRYPOINT ["/usr/local/bin/startup_script.sh"]
