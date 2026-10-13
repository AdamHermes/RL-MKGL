#!/usr/bin/env bash
#SBATCH --job-name=mkgl-baseline
#SBATCH --partition=gpu-long
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Submit from the MKGL repo: sbatch scripts/train_baseline.sh [config] [version]
# Example: sbatch scripts/train_baseline.sh config/fb15k237_ind.yaml v1
# Request the cluster's 80 GB A100 type; generic gpu:1 may allocate a 12 GB GPU.
# Multiple GPUs replicate the model; they do not pool VRAM with this launcher.
# Set MKGL_DIR if submitting from outside the workspace.
set -euo pipefail

MKGL_DIR="${MKGL_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$MKGL_DIR/main.py" && -f "$MKGL_DIR/MKGL/main.py" ]]; then
    MKGL_DIR="$MKGL_DIR/MKGL"
fi
cd "$MKGL_DIR"
if [[ ! -f main.py ]]; then
    echo "Submit from the MKGL repo or export MKGL_DIR=/path/to/MKGL." >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Load Conda before submitting; use sbatch --export=ALL." >&2
    exit 1
fi
# Conda compiler hooks may reference unset CONDA_BACKUP_* variables.
# Keep errexit/pipefail, but suspend nounset while Conda switches environments.
set +u
eval "$(conda shell.bash hook)"
conda activate "${MKGL_ENV:-mkgl}"
set -u
export CUDA_HOME="${MKGL_CUDA_HOME:-$CONDA_PREFIX}"
export CC="${MKGL_CC:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${MKGL_CXX:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++}"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/data/torch_extensions/cu117_gcc11}"
export MAX_JOBS="${MAX_JOBS:-2}"
if [[ ! -x "$CC" || ! -x "$CXX" ]]; then
    echo "Missing GCC 11 toolchain. Install gcc_linux-64=11 and gxx_linux-64=11 in the MKGL environment." >&2
    exit 1
fi
if [[ ! -f "$CUDA_HOME/include/cusparse.h" && ! -f "$CUDA_HOME/targets/x86_64-linux/include/cusparse.h" ]]; then
    echo "Missing cusparse.h. Install the full cuda-toolkit=11.7 development package." >&2
    exit 1
fi
# HF_HOME also controls the saved login token. Preserve the login location;
# relocate only model downloads so an existing `hf auth login` still works.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/data/hf_cache/hub}"
export PYTHONUNBUFFERED=1

CONFIG="${1:-config/fb15k237.yaml}"
VERSION="${2:-}"
SEED="${SEED:-42}"
CONFIG_NAME="${CONFIG##*/}"
CONFIG_NAME="${CONFIG_NAME%.*}"
if [[ "$CONFIG_NAME" == *_ind && -z "$VERSION" ]]; then
    VERSION=v1
fi
VERSION_ARGS=()
if [[ -n "$VERSION" ]]; then
    CONFIG_NAME="${CONFIG_NAME}_${VERSION}"
    VERSION_ARGS=(--version "$VERSION")
fi

mkdir -p logs "$HF_HUB_CACHE" "$TORCH_EXTENSIONS_DIR"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
exec > >(tee "logs/${CONFIG_NAME}_baseline_${RUN_ID}.log") 2>&1
echo "Config: $CONFIG; version: ${VERSION:-transductive}; seed: $SEED"
echo "Slurm job: ${SLURM_JOB_ID:-local}; environment: $CONDA_DEFAULT_ENV"

NUM_PROCESSES="${NUM_PROCESSES:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
if [[ ! "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
    echo "No visible CUDA GPU. Check the Slurm GPU allocation." >&2
    exit 1
fi
THREADS_PER_PROCESS=$(( ${SLURM_CPUS_PER_TASK:-8} / NUM_PROCESSES ))
if (( THREADS_PER_PROCESS < 1 )); then
    THREADS_PER_PROCESS=1
fi
export OMP_NUM_THREADS="$THREADS_PER_PROCESS"

# Compile and exercise TorchDrug's CUDA kernel before loading the 7B model.
python - <<'PY'
import torch
from torchdrug.layers import functional
device = torch.device("cuda:0")
properties = torch.cuda.get_device_properties(device)
print("GPU:", properties.name, "VRAM GiB:", round(properties.total_memory / 2**30, 2))
adjacency = torch.sparse_coo_tensor(
    torch.zeros(3, 1, dtype=torch.long, device=device),
    torch.ones(1, device=device), (1, 1, 1)).coalesce()
relation = torch.ones(1, 32, device=device, requires_grad=True)
features = torch.ones(1, 32, device=device, requires_grad=True)
output = functional.generalized_rspmm(adjacency, relation, features, sum="add", mul="mul")
torch.testing.assert_close(output, torch.ones_like(output))
output.sum().backward()
torch.testing.assert_close(relation.grad, torch.ones_like(relation))
torch.testing.assert_close(features.grad, torch.ones_like(features))
print("TorchDrug CUDA kernel forward/backward: OK")
PY

if [[ "${PREPROCESS:-0}" == 1 || ! -f "data/preprocessed/${CONFIG_NAME}.pkl" ]]; then
    python preprocess.py --config "$CONFIG" "${VERSION_ARGS[@]}"
fi

LAUNCH_ARGS=(--num_processes "$NUM_PROCESSES" --num_machines 1
    --mixed_precision bf16 --dynamo_backend no
    --num_cpu_threads_per_process "$THREADS_PER_PROCESS")
if (( NUM_PROCESSES > 1 )); then
    LAUNCH_ARGS+=(--multi_gpu)
fi

accelerate launch "${LAUNCH_ARGS[@]}" main.py \
    --config "$CONFIG" "${VERSION_ARGS[@]}" --seed "$SEED" \
    --eval-split valid --checkpoint-steps "${CHECKPOINT_STEPS:-500}"

echo "Baseline training finished. Check the printed checkpoint paths above."
