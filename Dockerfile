FROM nvidia/driver:418.87.01-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ bionic main universe" > /etc/apt/sources.list && \
  echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ bionic-updates main universe" >> /etc/apt/sources.list && \
  echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ bionic-security main universe" >> /etc/apt/sources.list && \
  usermod -o -u 0 -g 0 _apt

RUN apt-get update
RUN apt-get install -yq python3 python3-pip

RUN apt-get clean autoclean
RUN apt-get autoremove --yes
RUN rm -rf /var/lib/{apt,dpkg,cache,log}

COPY . /nvitop
WORKDIR /nvitop
RUN pip3 install .
RUN rm -rf /root/.cache

ENV LC_ALL C.UTF-8
ENTRYPOINT [ "python3", "nvitop.py" ]
