import importlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from multiprocessing import Process
from time import sleep

import torch

try:
    analyse = importlib.import_module("torch_npu.profiler.profiler").analyse
except ImportError:
    analyse = None

VLLM = importlib.import_module("vllm")
LLM = VLLM.LLM
SamplingParams = VLLM.SamplingParams
RequestOutput = importlib.import_module("vllm.outputs").RequestOutput
get_open_port = importlib.import_module("vllm.utils.network_utils").get_open_port
metrics_reader = importlib.import_module("vllm.v1.metrics.reader")
Counter = metrics_reader.Counter
Vector = metrics_reader.Vector


# -----------------------------
# Frequently toggled parameters
# -----------------------------

CLEAN_TARGETS = [
    "extra-info",
    "kernel_meta",
    "fusion_result.json",
    "fx-graph",
    "profile/default",
    "$HOME/ascend/log",
    "$HOME/ascend/atb/log",
]
KILL_PROCESS_PATTERNS = ["python", "VLLM"]
KILL_EXISTING_PROCESSES = True

# source ../repos/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

GLOBAL_ENV_VARS = {
    # Ascend
    # "TASK_QUEUE_ENABLE": "1",
    # "DISABLE_L2_CACHE": "1",
    # "ASCEND_LAUNCH_BLOCKING": "1",
    # "ASCEND_RT_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15",
    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
    # "ASCEND_CUSTOM_OPP_PATH": "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize:$ASCEND_CUSTOM_OPP_PATH",
    # "LD_LIBRARY_PATH": "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH",
    # Communication
    # "HCCL_OP_EXPANSION_MODE": "AIV",
    # "HCCL_BUFFSIZE": "512",
    # "HCCL_DETERMINISTIC": "true",
    # "HCCL_DETERMINISTIC": "strict",
    # "HCCL_ENTRY_LOG_ENABLE": "1",
    # Ascend logging
    # "ASCEND_GLOBAL_LOG_LEVEL": "0",
    # "ASCEND_SLOG_PRINT_TO_STDOUT": "1",
    # "ASCEND_WORK_PATH": "/home/liuyizhou/ascend-log",
    # PyTorch
    # "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "OMP_NUM_THREADS": "1",
    # "OMP_WAIT_POLICY": "PASSIVE",
    # vLLM
    # "VLLM_TORCH_PROFILER_WITH_STACK": "0",
    # "VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY": "0",
    # "VLLM_CUSTOM_SCOPES_FOR_PROFILING": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    # "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
    # "VLLM_DISABLE_COMPILE_CACHE": "1",
    # "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
    # "VLLM_VERSION": "0.0.0",
    # "VLLM_LOG_STATS_INTERVAL": "5",
    # "VLLM_LOGGING_LEVEL": "DEBUG",
    # "VLLM_USE_V2_MODEL_RUNNER": "1",
    # vLLM Ascend
    # "VLLM_ASCEND_ENABLE_MLAPO": "1",
    # "VLLM_ASCEND_ENABLE_NZ": "0",
    # "VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE": "1",
    # "VLLM_ASCEND_ENABLE_FLASHCOMM": "1",
    # "VLLM_ASCEND_ENABLE_PREFETCH_MLP": "1",
    
    # "ASCEND_LOG_DEVICE_FLUSH_TIMEOUT": "0",
    # "OMP_PROC_BIND": "false",
    # "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    # "TASK_QUEUE_ENABLE": "1",
    # "HCCL_OP_EXPANSION_MODE": "AIV",
    # "HCCL_BUFFSIZE": "2048",
    # "VLLM_ASCEND_ENABLE_MLAPO": "1",
    # # "HCCL_IF_BASE_PORT": "50000",
    # "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
}

# MODEL = "/home/weights/deepseek-ai/DeepSeek-Pruned-with-MTP"
# MODEL = "/home/weights/deepseek-ai/DeepSeek-V4-Flash-w8a8-mtp"
# MODEL = "/home/weights/deepseek-ai/DeepSeek-V3.2-Exp-W8A8"
# MODEL = "/mnt/share/weights/Qwen3-30B-A3B"
MODEL = "/mnt/share/weights/Qwen3-0.6B"
# MODEL = "/home/weights/Qwen/Qwen3-32B-Int8"
# MODEL = "/home/weights/deepseek-ai/DeepSeek-V3-Lite-W8A8"
# MODEL = "/home/weights/Llama/Meta-Llama-3.1-8B-Instruct"
# MODEL = "/mnt/weight/DeepSeek-V3.1-Terminus-w4a8_w8a8_pack"
# MODEL = "/mnt/share/weights/Qwen3.5-35B-A3B"
# MODEL = "/mnt/share/weights/Qwen3.6-35B-A3B-w8a8"

DP_SIZE = 1
# LMHEAD_TP_SIZE = 2
TP_SIZE = 4
NODE_SIZE = 1
NODE_RANK = 0
MASTER_ADDR = ""
MASTER_PORT = 0

PROMPTS = (
    [
        # "Hello, my name is",
        # "The president of the United States is",
        # "The capital of France is",
        # "The future of AI is",
        # "你好，我的名字是",
        "美国总统是",
        # "法国的首都是",
        # "人工智能的未来是",
    ]
    * DP_SIZE
    * 1
)

PROFILE_DIR = None
# PROFILE_DIR = "./profile/0813"

PROCESS_TIMEOUT_SECONDS = 1800
LOG_DIR = "logs"
INTERNAL_RUN_FLAG = "--internal-run"

# NUM_GPU_BLOCKS_OVERRIDE=10913

# from dataclasses import replace

# 必须放在导入 SpecDecodeBaseProposer 之前。
# 确保 deepseek_v2 第一次导入时拿到的是 Ascend FusedMoE。
# from vllm.plugins import load_general_plugins

# load_general_plugins()

# from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


# _original_create_draft_vllm_config = (
#     SpecDecodeBaseProposer._create_draft_vllm_config
# )


# def _create_draft_vllm_config_for_mtp(self):
#     draft_vllm_config = _original_create_draft_vllm_config(self)

#     if self.method != "mtp":
#         return draft_vllm_config

#     # ModelSlim 缺少 DeepSeek MTP 的 fused QKV 映射。
#     modelslim_config = importlib.import_module(
#         "vllm_ascend.quantization.modelslim_config"
#     )
#     modelslim_config.packed_modules_model_mapping["deepseek_mtp"][
#         "fused_qkv_a_proj"
#     ] = [
#         "q_a_proj",
#         "kv_a_proj_with_mqa",
#     ]

#     # MTP 使用原始 draft 配置，保持其真实层号 61。
#     return replace(
#         draft_vllm_config,
#         model_config=self.speculative_config.draft_model_config,
#     )


# SpecDecodeBaseProposer._create_draft_vllm_config = (
#     _create_draft_vllm_config_for_mtp
# )

def run_vllm_main(
    model: str,
    prompts: list[str],
    dp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    tp_size: int,
    profile_dir: str | None,
) -> None:
    os.environ.update(
        {
            "VLLM_DP_RANK": str(global_dp_rank),
            "VLLM_DP_RANK_LOCAL": str(local_dp_rank),
            "VLLM_DP_SIZE": str(dp_size),
            "VLLM_DP_MASTER_IP": dp_master_ip,
            "VLLM_DP_MASTER_PORT": str(dp_master_port),
        }
    )
    # print(f"{os.environ['VLLM_USE_BREAKABLE_CUDAGRAPH']=}")

    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        # enforce_eager=True,
        # enable_expert_parallel=True,
        trust_remote_code=True,
        # gpu_memory_utilization=0.7,
        # max_num_batched_tokens=4096,
        max_model_len=1024,
        # max_model_len=40960,
        # disable_log_stats=False,
        # async_scheduling=False,
        # max_num_seqs=4,
        # num_gpu_blocks_override=NUM_GPU_BLOCKS_OVERRIDE,
        # quantization="ascend",
        seed=1024,
        # max_num_batched_tokens=384,
        # max_num_seqs=192,
        gpu_memory_utilization=0.9,
        # enable_prefix_caching=False,
        # dtype="bfloat16",
        # load_format="dummy",
        # speculative_config={
        #     "method": "mtp",
        #     # "method": "eagle3",
        #     # "model": "/home/weights/RedHatAI/Qwen3-32B-speculator.eagle3",
        #     "num_speculative_tokens": 1,
        # },
        compilation_config={
            "max_cudagraph_capture_size": 24, "cudagraph_mode": "FULL"
        },
        # enforce_eager=True,
        # additional_config={
        #     "finegrained_tp_config": {
        #         "lmhead_tensor_parallel_size": LMHEAD_TP_SIZE,
        #     },
        #     "multistream_overlap_shared_expert": True,
        # },
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": profile_dir,
            "torch_profiler_with_stack": True,
            "torch_profiler_with_memory": False,
            "torch_profiler_use_gzip": False,
            "max_iterations": 6,
        } if profile_dir else None,
        # hf_overrides={
        #     "num_hidden_layers":6,
        #     "num_nextn_predict_layers": 1,
        # },
        # mm_processor_cache_gb=0,
        # mm_encoder_tp_mode='data',
        # async_scheduling=True
        # kv_cache_metrics=True,
        # cudagraph_metrics=True,
        # additional_config={
        #     "ascend_compilation_config": {
        #         # "fuse_norm_quant": False,
        #         "enable_static_kernel": True,
        #     }
        # },
    )

    rank_prompts = shard_prompts(prompts, dp_size, global_dp_rank)
    print(f"rank={global_dp_rank} prompts={len(rank_prompts)}")

    sampling_params = SamplingParams(temperature=0, max_tokens=16)
    # sampling_params = [
    #     SamplingParams(temperature=0, max_tokens=8 if i % 2 == 0 else 16) for i in range(len(rank_prompts))
    # ]
    if profile_dir:
        llm.start_profile()
    outputs = llm.generate(rank_prompts, sampling_params)
    if profile_dir:
        llm.stop_profile()

    torch.npu.synchronize()

    for index, output in enumerate(outputs[:10]):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(
            f"-" * 20 + "\n"
            f"DP rank {global_dp_rank}, Prompt {index}: {prompt!r}\n"
            f"Generated text: {generated_text!r}"
        )

    sleep(1)
    check_spec_decode_acceptance(llm, outputs)


def expand_path(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def kill_existing_processes() -> None:
    if not KILL_EXISTING_PROCESSES:
        return

    current_pid = os.getpid()
    for pattern in KILL_PROCESS_PATTERNS:
        result = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True
        )
        if not result.stdout:
            continue

        for value in result.stdout.strip().splitlines():
            pid = int(value)
            if pid == current_pid:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"killed pid={pid} pattern={pattern}")
            except OSError as exc:
                print(f"kill failed pid={pid} pattern={pattern} err={exc}")


def clean_paths() -> None:
    for raw_path in CLEAN_TARGETS:
        path = expand_path(raw_path)
        if not os.path.exists(path):
            continue
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
            print(f"removed {path}")
        except OSError as exc:
            print(f"remove failed path={path} err={exc}")

    for raw_path in ["/root/ascend/log", "/root/atb/log"]:
        path = expand_path(raw_path)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
            print(f"removed {path}")
        os.makedirs(path, exist_ok=True)


def setup_environment() -> None:
    kill_existing_processes()
    clean_paths()


def shard_prompts(prompts: list[str], dp_size: int, rank: int) -> list[str]:
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    def start(index: int) -> int:
        return index * floor + min(index, remainder)

    rank_prompts = prompts[start(rank) : start(rank + 1)]
    return rank_prompts or ["Placeholder"]


def check_spec_decode_acceptance(llm, outputs) -> None:
    if llm.llm_engine.vllm_config.speculative_config is None:
        return

    metrics = llm.get_metrics()
    total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [
        0
    ] * llm.llm_engine.vllm_config.speculative_config.num_speculative_tokens

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos, value in enumerate(metric.values):
                acceptance_counts[pos] += value

    print(f"output_tokens={total_output_tokens}")
    print(f"drafts={num_drafts}")
    print(f"draft_tokens={num_draft_tokens}")
    print(f"accepted_tokens={num_accepted_tokens}")
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts else 1
    print(f"acceptance_length={acceptance_length:.2f}")
    for pos, count in enumerate(acceptance_counts):
        acceptance_rate = count / num_drafts if num_drafts else 0
        print(f"acceptance_token_{pos}={acceptance_rate:.2f}")


def build_logged_command(log_filename: str) -> str:
    script = shlex.quote(os.path.abspath(__file__))
    args = [shlex.quote(arg) for arg in sys.argv[1:] if arg != INTERNAL_RUN_FLAG]
    command = ["python", "-u", script, INTERNAL_RUN_FLAG, *args]
    inner = " ".join(command)
    tee_log = shlex.quote(log_filename)
    return f"{inner} 2>&1 | tee {tee_log}"


def maybe_relaunch_with_logging() -> None:
    if INTERNAL_RUN_FLAG in sys.argv:
        sys.argv.remove(INTERNAL_RUN_FLAG)
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"inference_{timestamp}.log")
    print(f"log={log_filename}")
    command = build_logged_command(log_filename)
    env = os.environ.copy()
    env.update(GLOBAL_ENV_VARS)
    try:
        _ = subprocess.run(["bash", "-lc", command], check=True, env=env)
        raise SystemExit(0)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except KeyboardInterrupt:
        raise SystemExit(130) from None


def main() -> int:
    maybe_relaunch_with_logging()
    setup_environment()

    if PROFILE_DIR:
        os.makedirs(PROFILE_DIR, exist_ok=True)

    if NODE_SIZE == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = MASTER_ADDR
        dp_master_port = MASTER_PORT

    assert DP_SIZE % NODE_SIZE == 0, "DP_SIZE must be divisible by NODE_SIZE"
    dp_per_node = DP_SIZE // NODE_SIZE
    node_dp_ranks = range(NODE_RANK * dp_per_node, (NODE_RANK + 1) * dp_per_node)

    print(
        f"model={MODEL} dp={DP_SIZE} tp={TP_SIZE} node_size={NODE_SIZE} node_rank={NODE_RANK}"
    )

    processes: list[Process] = []
    for local_dp_rank, global_dp_rank in enumerate(node_dp_ranks):
        process = Process(
            target=run_vllm_main,
            args=(
                MODEL,
                PROMPTS,
                DP_SIZE,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                TP_SIZE,
                PROFILE_DIR,
            ),
        )
        process.start()
        processes.append(process)

    exit_code = 0
    for process in processes:
        process.join(timeout=PROCESS_TIMEOUT_SECONDS)
        if process.exitcode is None:
            print(f"timeout pid={process.pid}")
            process.kill()
            exit_code = 1
        elif process.exitcode != 0:
            print(f"failed pid={process.pid} code={process.exitcode}")
            exit_code = process.exitcode

    if PROFILE_DIR and analyse:
        try:
            analyse(profiler_path=PROFILE_DIR)
        except Exception as exc:
            print(f"profile_analyse_failed err={exc}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

