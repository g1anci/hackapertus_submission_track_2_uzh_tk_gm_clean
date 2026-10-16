FROM python:3.13.5-slim-bookworm
ENV HF_HOME=/cache/huggingface
WORKDIR /work
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY . ./
CMD ["sh", "-c", "mkdir -p executed && papermill --log-output RSD_Validation_Pipeline_batched_calls.ipynb executed/RSD_Validation_Pipeline_batched_calls.ipynb"]
