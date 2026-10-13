#!/usr/bin/env bash
# Run on Linux: bash scripts/setup_mkgl.sh [environment_name]
set -euo pipefail

ENV_NAME="${1:-mkgl}"
if ! command -v conda >/dev/null 2>&1; then
    echo "Load your cluster's Conda module before running this script." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

conda create -n "$ENV_NAME" python=3.8 -y
conda activate "$ENV_NAME"

python -m pip install 'pip<25.1' 'setuptools<75.4' 'wheel<0.46'
python -m pip install \
    torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 \
    --index-url https://download.pytorch.org/whl/cu117

# NVCC and headers are needed by TorchDrug's JIT CUDA extensions.
conda install -n "$ENV_NAME" -c nvidia -y \
    cuda-runtime=11.7 cuda-cudart=11.7 cuda-cudart-dev=11.7 cuda-nvcc=11.7 \
    libcusparse=11.7 libcublas=11.10
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -m pip install numpy==1.24.4 scipy==1.10.1
python -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    --only-binary=:all: -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

python -m pip install \
    pandas==1.5.3 swifter==1.4.0 easydict pyyaml sentencepiece==0.1.99 \
    cffi==1.15.1 lmdb==1.4.1 \
    transformers==4.36.2 peft==0.7.1 bitsandbytes==0.41.1 accelerate==0.25.0 \
    datasets==2.19.2 pyarrow==16.1.0 protobuf==3.20.3 \
    tensorboardX tensorboard ninja torchdrug==0.2.1

python -m pip check
python - <<'PY'
import torch
import transformers
import peft
import accelerate
import torch_scatter
import torch_sparse
import torchdrug
print("MKGL environment ready")
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__, "PEFT:", peft.__version__)
print("Accelerate:", accelerate.__version__)
PY
echo "Activate with: conda activate $ENV_NAME"
