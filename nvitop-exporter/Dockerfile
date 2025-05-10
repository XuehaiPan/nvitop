ARG basetag="450-signed-ubuntu22.04"  # Ubuntu only
FROM nvcr.io/nvidia/driver:"${basetag}"

RUN apt-get update && apt-get install -y pipx
RUN pipx ensurepath
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /nvitop/nvitop-exporter

RUN pipx install nvitop-exporter

ENTRYPOINT [ "nvitop-exporter", "--bind-address", "0.0.0.0", "--port", "5050" ]
