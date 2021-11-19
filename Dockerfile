FROM jupyter/tensorflow-notebook as base

# FROM base as full
# Windows Render related
# USER root
# RUN apt-get update -y && \
#     apt-get install -y xvfb
#    apt-get install -y build-essential checkinstall libffi-dev python-dev && \
#    apt-get install -y libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev && \
#    apt-get install -y python-opengl
#
# RUN conda install swig==3.0.12
# RUN conda install -c conda-forge box2d-py
# RUN pip install box2d-py
# Optional, needed for some environments
# RUN apt-get install -y cmake && \
#     apt-get install -y zlib1g zlib1g-dev
#
# RUN pip install \
#         gym \
#         pyvirtualdisplay \
#         pygame \
#         numpy \
#         torch \
#         tensorboard \
#         ray \
#         seaborn \
#         nevergrad
#
# Needed for some environments
# RUN #apt-get update -y && apt-get install -y build-essential checkinstall libffi-dev python-dev && \
#    apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
# RUN #pip install box2d-py
# RUN pip install atari_py pystan pyglet
#
# USER ${NB_USER}
# WORKDIR work
#
# FROM base as orig
#
# Windows Render related
# USER root
# RUN apt-get update -y && \
#     apt-get install -y xvfb && \
#     apt-get install -y python-opengl
#
# Optional, needed for some environments
# RUN apt-get install -y cmake && \
#     apt-get install -y zlib1g zlib1g-dev
#
# USER ${NB_USER}
#
# RUN pip install \
#         gym \
#         pyvirtualdisplay
#
# Needed for some environments
# RUN conda install swig==3.0.12
# RUN conda install box2d-py
# RUN pip install atari_py pystan
#
# WORKDIR work

FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel as torch

COPY requirements.txt requirements.txt
RUN apt-get update -y && \
    apt-get install -y xvfb && \
    apt-get install -y python-opengl

# Optional, needed for some environments
RUN apt-get install -y cmake && \
    apt-get install -y zlib1g zlib1g-dev
RUN pip install -r requirements.txt
RUN pip install pyvirtualdisplay

RUN conda install swig==3.0.12
RUN conda install -c conda-forge box2d-py
RUN pip install atari_py pystan pyglet
RUN conda install -c conda-forge fontconfig
RUN apt-get install -y x11-apps

WORKDIR work
