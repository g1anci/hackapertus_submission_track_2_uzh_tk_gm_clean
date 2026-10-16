# Shared GPU environment for src/synthetic_data and src/train
FROM python:3.12.14-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential cmake zlib1g-dev libbz2-dev liblzma-dev curl unzip \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && python -m ipykernel install --sys-prefix --name apertus15 --display-name "Python (Apertus 1.5)"
ENV HF_HOME=/cache/huggingface
WORKDIR /work
