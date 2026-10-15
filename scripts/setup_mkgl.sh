#!/usr/bin/env bash
# Run on Linux: bash scripts/setup_mkgl.sh [environment_name]
set -euo pipefail

ENV_NAME="${1:-mkgl}"
if ! command -v conda >/dev/null 2>&1; then
    echo "Load your cluster's Conda module before running this script." >&2
    exit 1
fi
set +u
eval "$(conda shell.bash hook)"

conda create -n "$ENV_NAME" python=3.8 -y
conda activate "$ENV_NAME"
set -u

python -m pip install 'pip<25.1' 'setuptools<75.4' 'wheel<0.46'
python -m pip install \
    torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 \
    --index-url https://download.pytorch.org/whl/cu117

# TorchDrug compiles CUDA kernels: runtime-only packages lack cuSPARSE headers.
# Installing into the active environment can also invoke Conda compiler hooks.
set +u
conda install -n "$ENV_NAME" -c nvidia/label/cuda-11.7.1 -c nvidia -y \
    'cuda-toolkit=11.7' 'cuda-cccl=11.7.91'
# CUDA 11.7 rejects GCC > 11; use an environment-local host toolchain.
conda install -n "$ENV_NAME" -c conda-forge -y \
    'gcc_linux-64=11' 'gxx_linux-64=11' libxcrypt
set -u
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
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
