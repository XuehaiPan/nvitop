# Ubuntu only
ARG basetag="418.87.01-ubuntu18.04"  

FROM nvidia/driver:"${basetag}"

ENV DEBIAN_FRONTEND=noninteractive

# Update APT sources
RUN . /etc/os-release && [ "${NAME}" = "Ubuntu" ] && \
  echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu ${UBUNTU_CODENAME} main universe" > /etc/apt/sources.list && \
  echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu ${UBUNTU_CODENAME}-updates main universe" >> /etc/apt/sources.list && \
  echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu ${UBUNTU_CODENAME}-security main universe" >> /etc/apt/sources.list

# Install Python 3
RUN apt-get update && \
  apt-get install --quiet --yes --no-install-recommends \
  python3-dev python3-pip python3-setuptools python3-wheel locales && \
  rm -rf /var/lib/apt/lists/*

# Setup locale
ENV LC_ALL=C.UTF-8
RUN update-locale LC_ALL="C.UTF-8"

# Install dependencies
RUN python3 -m pip install --upgrade pip
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Install nvitop
COPY . /nvitop
WORKDIR /nvitop
RUN pip3 install --compile . && \
  rm -rf /root/.cache

# Entrypoint
ENTRYPOINT [ "python3", "-m", "nvitop" ]
