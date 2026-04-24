#!/usr/bin/env bash
set -euo pipefail

# LatentFormer smoke/eval launcher.
# Defaults are intentionally small so the first run checks that the model,
# config, data registration, and container environment are wired correctly.

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${SIF:-/data/andy.barcia/fcclip-torch27-cu126.sif}"
DATASETS="${DATASETS:-/data/datasets}"
WORKDIR="${WORKDIR:-/workspace/LatentFormer}"
CONFIG_FILE="${CONFIG_FILE:-configs/latentformer/latentformer_R50_bs16_50ep.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/latentformer_R50_smoke}"
RUN_MODE="${RUN_MODE:-smoke}"  # smoke, train, or eval

NUM_GPUS="${NUM_GPUS:-1}"
IMS_PER_BATCH="${IMS_PER_BATCH:-1}"
MAX_ITER="${MAX_ITER:-20}"
EVAL_PERIOD="${EVAL_PERIOD:-0}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-20}"
NUM_WORKERS="${NUM_WORKERS:-2}"
USE_TSP="${USE_TSP:-1}"

LOGDIR="${LOGDIR:-${PROJ_ROOT}/${OUTPUT_DIR}/logs}"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/latentformer_${RUN_MODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1

echo "Logging to: ${LOGFILE}"
echo "Project root: ${PROJ_ROOT}"
echo "Config: ${CONFIG_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Run mode: ${RUN_MODE}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-${DATASETS}}"

# Keep Apptainer scratch/cache writes out of shared read-only cache locations.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/tmp/${USER}/apptainer/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/${USER}/apptainer/tmp}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

APPTAINER_OPTS=(
  --nv
  --bind "${PROJ_ROOT}:${WORKDIR}"
  --bind "${DATASETS}:${DATASETS}:ro"
  --pwd "${WORKDIR}"
)

TRAIN_ARGS=(
  python train_net.py
  --num-gpus "${NUM_GPUS}"
  --config-file "${CONFIG_FILE}"
)

CFG_OPTS=(
  OUTPUT_DIR "${OUTPUT_DIR}"
  SOLVER.IMS_PER_BATCH "${IMS_PER_BATCH}"
  DATALOADER.NUM_WORKERS "${NUM_WORKERS}"
)

case "${RUN_MODE}" in
  smoke)
    CFG_OPTS+=(
      SOLVER.MAX_ITER "${MAX_ITER}"
      SOLVER.CHECKPOINT_PERIOD "${CHECKPOINT_PERIOD}"
      TEST.EVAL_PERIOD "${EVAL_PERIOD}"
    )
    ;;
  train)
    TRAIN_ARGS+=(--resume)
    ;;
  eval)
    TRAIN_ARGS+=(--eval-only)
    ;;
  *)
    echo "Unknown RUN_MODE='${RUN_MODE}'. Expected smoke, train, or eval." >&2
    exit 2
    ;;
esac

CMD=(
  apptainer exec
  "${APPTAINER_OPTS[@]}"
  "${SIF}"
  "${TRAIN_ARGS[@]}"
  "${CFG_OPTS[@]}"
)

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${USE_TSP}" == "1" ]]; then
  exec tsp gpu exec "${CMD[@]}"
else
  exec "${CMD[@]}"
fi
