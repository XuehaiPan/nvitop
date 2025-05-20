ARG basetag="450-signed-ubuntu22.04"  # Ubuntu only
FROM nvcr.io/nvidia/driver:"${basetag}"

SHELL [ "/bin/bash" ]

RUN . /etc/os-release && [ "${NAME}" = "Ubuntu" ] || \
  (echo "This Dockerfile is only supported on Ubuntu" >&2 && exit 1)

ENV NVIDIA_DISABLE_REQUIRE=true
ENV DEBIAN_FRONTEND=noninteractive

# Update APT sources
RUN . /etc/os-release && \
  echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu ${UBUNTU_CODENAME} main universe" > /etc/apt/sources.list && \
  echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu ${UBUNTU_CODENAME}-updates main universe" >> /etc/apt/sources.list && \
  echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu ${UBUNTU_CODENAME}-security main universe" >> /etc/apt/sources.list

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
  echo && echo && echo "source /venv/bin/activate" >> /root/.bashrc
ENV SHELL=/bin/bash

# Install nvitop
COPY . /nvitop
WORKDIR /nvitop
RUN . /venv/bin/activate && \
  python3 -m pip install . && \
  rm -rf /root/.cache

# Entrypoint
ENTRYPOINT [ "/venv/bin/python3", "-m", "nvitop" ]
