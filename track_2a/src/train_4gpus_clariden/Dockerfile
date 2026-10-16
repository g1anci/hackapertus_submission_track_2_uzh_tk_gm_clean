FROM nvcr.io/nvidia/pytorch:25.06-py3

# keep NVIDIA's torch + numpy; pyarrow 26+ needs numpy 2, so stay below it
# transformers: Swiss AI fork with Apertus v1.5 support (commit from the model card)
RUN pip list --format=freeze | grep -E "^(torch|torchvision|triton|numpy)==" > /tmp/keep.txt && \
    echo "pyarrow<26" >> /tmp/keep.txt && \
    pip install --no-cache-dir -c /tmp/keep.txt \
      "transformers @ git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4" \
      trl==1.15.0 accelerate==1.15.0 datasets scipy papermill ipykernel && \
    python -c "import numpy, pyarrow, datasets, torch, transformers, trl; from transformers.models.apertus1p5 import modeling_apertus1p5; print('numpy', numpy.__version__, '| pyarrow', pyarrow.__version__, '| datasets', datasets.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__, '| trl', trl.__version__, '| apertus1p5 OK')"
