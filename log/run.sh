#!/bin/bash
set -euo pipefail

export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export MAX_NUM_SEQS=96
export MAX_MODEL_LEN=26000
export MAX_NUM_BATCHED_TOKENS=2048
# export NIC_NAME=enp35s0f2
export LOCAL_IP=141.61.141.22
export GPU_MEMORY_UTILIZATION=0.9

export KV_PORT="${KV_PORT:-56010}"
export ENABLE_PREFIX_CACHE_ON_DECODE="${ENABLE_PREFIX_CACHE_ON_DECODE:-1}"
if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [visible_device_start]"
  echo "Example: $0 4"
  exit 1
fi

VISIBLE_DEVICE_START="${1:-0}"
if ! [[ "${VISIBLE_DEVICE_START}" =~ ^[0-9]+$ ]]; then
  echo "visible_device_start must be a non-negative integer, got: ${VISIBLE_DEVICE_START}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python ./launch_online_dp.py \
  --dp-size 1 \
  --tp-size 8 \
  --dp-size-local 1 \
  --dp-rank-start 0 \
  --dp-address ${LOCAL_IP} \
  --dp-rpc-port 22321 \
  --vllm-start-port 50000 \
  --visible-device-start "${VISIBLE_DEVICE_START}" \
  --template-path ./run_dp_template_decode.sh
  
