FROM nvidia/driver:418.87.01-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ bionic main universe" > /etc/apt/sources.list && \
  echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ bionic-updates main universe" >> /etc/apt/sources.list && \
  echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ bionic-security main universe" >> /etc/apt/sources.list && \
  usermod -o -u 0 -g 0 _apt

RUN apt-get update
RUN apt-get install -yq python3 python3-pip
RUN apt-get install -yq git

RUN python3 -m pip install git+https://github.com/XuehaiPan/nvitop.git#egg=nvitop

ENV LANG C.UTF-8

ENTRYPOINT [ "nvitop" ]
