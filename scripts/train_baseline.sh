#!/usr/bin/env bash
#SBATCH --job-name=mkgl-baseline
#SBATCH --partition=gpu-long
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Submit from the MKGL repo: sbatch scripts/train_baseline.sh [config] [version]
# Example: sbatch scripts/train_baseline.sh config/fb15k237_ind.yaml v1
# Override GPU allocation: sbatch --gres=gpu:4 scripts/train_baseline.sh
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
eval "$(conda shell.bash hook)"
conda activate "${MKGL_ENV:-mkgl}"
export CUDA_HOME="${MKGL_CUDA_HOME:-$CONDA_PREFIX}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="${HF_HOME:-$PWD/data/hf_cache}"
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

mkdir -p logs "$HF_HOME"
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
