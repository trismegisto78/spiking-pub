# CPU image for Runpod pods (or any Docker host). Build for the Runpod architecture:
#   docker build --platform=linux/amd64 --build-arg GIT_COMMIT=$(git rev-parse HEAD) \
#       -t <registry>/spiking-evo:<tag> spiking-evo/
FROM python:3.12.8-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    SPIKING_EVO_OUT=/workspace/runs

# openssh-server: Runpod SSH access (start.sh installs $PUBLIC_KEY).
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd

WORKDIR /opt/spiking-evo
# Dependency layer first, so code edits do not reinstall numpy.
RUN pip install "numpy>=1.26" "pyyaml>=6.0"
COPY pyproject.toml README.md ./
COPY spiking_evo ./spiking_evo
RUN pip install --no-deps .

# No .git inside the image: the commit recorded in config.yaml comes from here.
ARG GIT_COMMIT=unknown
ENV SPIKING_EVO_COMMIT=${GIT_COMMIT}

COPY runpod/start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
