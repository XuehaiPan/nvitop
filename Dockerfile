# Dockerfile for nvitop
#
# Build the image with:
#
#  docker build --tag nvitop:latest .
#
# ==============================================================================
FROM ubuntu:latest

RUN . /etc/os-release && [ "${NAME}" = "Ubuntu" ] || \
  (echo "This Dockerfile is only supported on Ubuntu" >&2 && exit 1)

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3
RUN apt-get update && \
  apt-get install --quiet --yes --no-install-recommends python3-dev python3-venv locales && \
  rm -rf /var/lib/apt/lists/*

# Setup locale
ENV LC_ALL=C.UTF-8
RUN update-locale LC_ALL="C.UTF-8"

# Setup environment
RUN python3 -m venv /venv && \
  . /venv/bin/activate && \
  python3 -m pip install --upgrade pip setuptools && \
  rm -rf /root/.cache && \
  ( echo && echo && echo "source /venv/bin/activate" ) >> /root/.bashrc
ENV SHELL=/bin/bash

# Install nvitop
COPY . /nvitop
WORKDIR /nvitop
RUN . /venv/bin/activate && \
  python3 -m pip install /nvitop && \
  rm -rf /root/.cache

# Entrypoint
ENTRYPOINT [ "/venv/bin/python3", "-m", "nvitop" ]
