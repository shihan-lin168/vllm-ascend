#!/bin/bash
set -euo pipefail

# ============================================================
# Arguments from launch_online_dp.py
#   $1: visible_devices
#   $2: vllm_engine_port
#   $3: dp_size
#   $4: dp_rank
#   $5: dp_address
#   $6: dp_rpc_port
#   $7: tp_size
# ============================================================

if [ "$#" -ne 7 ]; then
  echo "Usage: $0 <visible_devices> <port> <dp_size> <dp_rank> <dp_address> <dp_rpc_port> <tp_size>"
  echo "Example: $0 0,1,2,3 9000 1 0 141.61.133.109 22321 4"
  exit 1
fi

VISIBLE_DEVICES="$1"
VLLM_ENGINE_PORT="$2"
DP_SIZE="$3"
DP_RANK="$4"
DP_ADDRESS="$5"
DP_RPC_PORT="$6"
TP_SIZE="$7"

# ============================================================
# Basic config
# 可通过环境变量覆盖，便于多节点复用
# ============================================================

MODEL_PATH="${MODEL_PATH:-/mnt/weight/DeepSeek-V3.1-Terminus-w4a8_w8a8_pack}"
# SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35}"

LOCAL_IP="${LOCAL_IP:-141.61.133.109}"
# NIC_NAME="${NIC_NAME:-eth2}"

# 建议监听 0.0.0.0，外部 proxy 用 LOCAL_IP:PORT 访问
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"

# Decode 侧固定为 consumer
KV_ROLE="kv_consumer"

# Decode 侧 Mooncake 端口，建议和 Prefill 侧错开
KV_PORT="${KV_PORT:-58010}"

# P/D 并行形态。当前部署中每个服务内部都是 TP4DP1。
PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-1}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
DECODE_DP_SIZE="${DECODE_DP_SIZE:-${DP_SIZE}}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-${TP_SIZE}}"

# engine_id 建议不同角色、不同节点唯一
# 如果你的版本要求纯数字，可以改成 ENGINE_ID="$((2000 + DP_RANK))"
ENGINE_ID="${ENGINE_ID:-decode_${DP_RANK}}"

# ============================================================
# Environment
# ============================================================

unset ftp_proxy FTP_PROXY
unset https_proxy HTTPS_PROXY
unset http_proxy HTTP_PROXY

export IP_ADDRESS="${LOCAL_IP}"
# export NETWORK_CARD_NAME="${NIC_NAME}"

export HCCL_IF_IP="${LOCAL_IP}"
# export GLOO_SOCKET_IFNAME="${NIC_NAME}"
# export TP_SOCKET_IFNAME="${NIC_NAME}"
# export HCCL_SOCKET_IFNAME="${NIC_NAME}"

export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}"

export VLLM_ASCEND_SKIP_LOW_BLOCKS="${VLLM_ASCEND_SKIP_LOW_BLOCKS:-0}"

export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2048}"

# vLLM V1
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# W4A8 场景下不要默认强开 FUSED_MC2；
# 你原脚本注释也写了“FUSED_MC2 暂不支持 w4a8，只在 w8a8 下打开”
# export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
#export VLLM_ASCEND_ENABLE_FUSED_MC2=1
#export DYNAMIC_EPLB="true"

# jemalloc 可选
JEMALLOC_PATH="${JEMALLOC_PATH:-/usr/lib/aarch64-linux-gnu/libjemalloc.so.2}"
if [ -f "${JEMALLOC_PATH}" ]; then
  export LD_PRELOAD="${JEMALLOC_PATH}:${LD_PRELOAD:-}"
fi

# 可选系统调优，失败不影响启动
if ls /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1; then
  echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null || true
fi

sysctl -w vm.swappiness=0 || true
sysctl -w kernel.numa_balancing=0 || true
sysctl -w kernel.sched_migration_cost_ns=50000 || true

# ============================================================
# vLLM DP args
# ============================================================

DP_ARGS=(--data-parallel-size "${DP_SIZE}")
if (( DP_SIZE > 1 )); then
  DP_ARGS+=(
    --data-parallel-rank "${DP_RANK}"
    --data-parallel-address "${DP_ADDRESS}"
    --data-parallel-rpc-port "${DP_RPC_PORT}"
  )
fi

# ============================================================
# Decode KV Transfer Config
# ============================================================

KV_TRANSFER_CONFIG=$(cat <<EOF
{
  "kv_connector": "MooncakeConnectorV1",
  "kv_buffer_device": "npu",
  "kv_role": "${KV_ROLE}",
  "kv_port": "${KV_PORT}",
  "engine_id": "${ENGINE_ID}",
  "kv_connector_extra_config": {
    "ascend_local_comm_res_path": "/etc/hixlep",
    "prefill": {
      "dp_size": ${PREFILL_DP_SIZE},
      "tp_size": ${PREFILL_TP_SIZE}
    },
    "decode": {
      "dp_size": ${DECODE_DP_SIZE},
      "tp_size": ${DECODE_TP_SIZE}
    }
  }
}
EOF
)

echo "============================================================"
echo "Starting vLLM Decode DP rank"
echo "  MODEL_PATH=${MODEL_PATH}"
# echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "  VLLM_HOST=${VLLM_HOST}"
echo "  VLLM_ENGINE_PORT=${VLLM_ENGINE_PORT}"
echo "  DP_SIZE=${DP_SIZE}"
echo "  DP_RANK=${DP_RANK}"
echo "  DP_ADDRESS=${DP_ADDRESS}"
echo "  DP_RPC_PORT=${DP_RPC_PORT}"
echo "  TP_SIZE=${TP_SIZE}"
echo "  KV_ROLE=${KV_ROLE}"
echo "  KV_PORT=${KV_PORT}"
echo "  ENGINE_ID=${ENGINE_ID}"
echo "  PREFILL_DP_SIZE=${PREFILL_DP_SIZE}"
echo "  PREFILL_TP_SIZE=${PREFILL_TP_SIZE}"
echo "  DECODE_DP_SIZE=${DECODE_DP_SIZE}"
echo "  DECODE_TP_SIZE=${DECODE_TP_SIZE}"
echo "============================================================"

# ============================================================
# Prefix cache on Decode
# 官方 DeepSeek PD Decode 示例通常关闭 prefix caching；
# Decode 侧主要消费 Prefill 传来的 KV，不建议默认开启。
# 如需开启，可设置 ENABLE_PREFIX_CACHE_ON_DECODE=1
# ============================================================

PREFIX_CACHE_ARGS=()
if [ "${ENABLE_PREFIX_CACHE_ON_DECODE:-0}" = "1" ]; then
  PREFIX_CACHE_ARGS+=(--enable-prefix-caching)
else
  PREFIX_CACHE_ARGS+=(--no-enable-prefix-caching)
fi

# ============================================================
# Start vLLM Decode
# ============================================================

  # --served-model-name "${SERVED_MODEL_NAME}" \
exec vllm serve "${MODEL_PATH}" \
  --allowed-local-media-path / \
  --trust-remote-code \
  --quantization ascend \
  "${DP_ARGS[@]}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --enable-expert-parallel \
  --host "${VLLM_HOST}" \
  --port "${VLLM_ENGINE_PORT}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --max-model-len "${MAX_MODEL_LEN:-26000}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --seed "${SEED:-1024}" \
  "${PREFIX_CACHE_ARGS[@]}" \
  --async-scheduling \
  --compilation-config '{"cudagraph_capture_sizes":[4,8,12,16,32,48,64,80,96,112,128,160,192,256,384], "cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode data \
  --distributed-executor-backend mp \
  --no-disable-hybrid-kv-cache-manager \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./profile/0811", "torch_profiler_with_stack": true, "torch_profiler_with_memory": true}' \

  # --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
  # --additional-config '{"recompute_scheduler_enable": true, "enable_cpu_binding": true, "ascend_compilation_config":{"enable_npugraph_ex":false}}' \
  # --compilation-config '{"cudagraph_capture_sizes":[128,256,384], "cudagraph_mode":"FULL_DECODE_ONLY"}' \