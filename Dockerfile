FROM nvidia/driver:418.87.01-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

# Update APT sources
RUN echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu bionic main universe" > /etc/apt/sources.list && \
  echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu bionic-updates main universe" >> /etc/apt/sources.list && \
  echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu bionic-security main universe" >> /etc/apt/sources.list

# Install Python 3
RUN apt-get update && \
  apt-get install --quiet --yes --no-install-recommends \
  python3-dev python3-pip python3-setuptools python3-wheel locales && \
  rm -rf /var/lib/{apt,dpkg,cache,log}

# Setup locale
ENV LC_ALL=C.UTF-8
RUN update-locale LC_ALL="C.UTF-8"

# Install nvitop
COPY . /nvitop
WORKDIR /nvitop
RUN pip3 install --compile . && \
  rm -rf /root/.cache

# Entrypoint
ENTRYPOINT [ "python3", "-m", "nvitop" ]
