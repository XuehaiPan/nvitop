ARG basetag="535-5.15.0-140-generic-ubuntu22.04"  # Ubuntu only
FROM nvcr.io/nvidia/driver:"${basetag}"

SHELL [ "/bin/bash" ]

RUN . /etc/os-release && [ "${NAME}" = "Ubuntu" ] || \
  (echo "This Dockerfile is only supported on Ubuntu" >&2 && exit 1)

# Setup locale
RUN apt-get update && \
  apt-get install --quiet --yes --no-install-recommends locales && \
  rm -rf /var/lib/apt/lists/*
ENV LC_ALL=C.UTF-8
RUN update-locale LC_ALL="C.UTF-8"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install nvitop-exporter
RUN /bin/uv tool install nvitop-exporter

ENTRYPOINT [ "/bin/uvx", "--quiet", "--offline", "nvitop-exporter", "--bind-address", "0.0.0.0", "--port", "5050" ]
