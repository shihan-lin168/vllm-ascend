#!/usr/bin/env python3
"""
vLLM Online DP Launch Script
"""
import argparse
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True, help="Global data parallel size.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--dp-size-local", type=int, default=-1, help="Local data parallel size.")
    parser.add_argument("--dp-rank-start", type=int, default=0, help="Starting rank for local data parallel.")
    parser.add_argument("--dp-address", type=str, required=True, help="IP address for data parallel master node.")
    parser.add_argument("--dp-rpc-port", type=str, default="12321", help="Port for data parallel master node.")
    parser.add_argument("--vllm-start-port", type=int, default=8000, help="Starting port for the engine.")
    parser.add_argument(
        "--visible-device-start",
        type=int,
        default=0,
        help="First visible device id used for local DP workers.",
    )
    parser.add_argument(
        "--template-path",
        type=str,
        default="/mnt/share/l00656382/qwen3_5/a5-profiler_estimate_cudagraph/run_dp_template_prefill.sh",
        help="Path to the vLLM run template, relative to the service directory.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.dp_size <= 0:
        raise ValueError(f"--dp-size must be > 0, got {args.dp_size}")
    if args.tp_size <= 0:
        raise ValueError(f"--tp-size must be > 0, got {args.tp_size}")
    if args.dp_size_local == 0 or args.dp_size_local < -1:
        raise ValueError(f"--dp-size-local must be -1 or > 0, got {args.dp_size_local}")
    if args.dp_rank_start < 0:
        raise ValueError(f"--dp-rank-start must be >= 0, got {args.dp_rank_start}")
    if args.visible_device_start < 0:
        raise ValueError(
            f"--visible-device-start must be >= 0, got {args.visible_device_start}"
        )
    if args.vllm_start_port <= 0:
        raise ValueError(f"--vllm-start-port must be > 0, got {args.vllm_start_port}")

    dp_size_local = args.dp_size if args.dp_size_local == -1 else args.dp_size_local
    if dp_size_local > args.dp_size:
        raise ValueError(
            f"--dp-size-local ({dp_size_local}) must be <= --dp-size ({args.dp_size})"
        )
    if args.dp_rank_start + dp_size_local > args.dp_size:
        raise ValueError(
            "--dp-rank-start + local dp size must be <= --dp-size "
            f"({args.dp_rank_start} + {dp_size_local} > {args.dp_size})"
        )
    return dp_size_local


def run_command(visible_devices, dp_rank, vllm_engine_port, args):
    command = [
        "bash",
        args.template_path,
        visible_devices,
        str(vllm_engine_port),
        str(args.dp_size),
        str(dp_rank),
        args.dp_address,
        args.dp_rpc_port,
        str(args.tp_size),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        dp_size_local = validate_args(args)
    except ValueError as exc:
        print(f"Invalid launch arguments: {exc}", file=sys.stderr)
        sys.exit(2)

    template_path = Path(args.template_path)
    if not template_path.is_absolute():
        template_path = Path.cwd() / template_path
    template_path = template_path.resolve()
    if not template_path.exists():
        print(f"Template file {template_path} does not exist.", file=sys.stderr)
        sys.exit(1)
    args.template_path = str(template_path)

    num_cards = dp_size_local * args.tp_size
    visible_device_end = args.visible_device_start + num_cards - 1
    print(
        "Launching local DP workers: "
        f"dp_size={args.dp_size}, dp_size_local={dp_size_local}, "
        f"tp_size={args.tp_size}, visible_devices="
        f"{args.visible_device_start}-{visible_device_end}, "
        f"template={args.template_path}"
    )

    processes = []
    for i in range(dp_size_local):
        dp_rank = args.dp_rank_start + i
        vllm_engine_port = args.vllm_start_port + i
        start = args.visible_device_start + i * args.tp_size
        end = start + args.tp_size
        visible_devices = ",".join(str(x) for x in range(start, end))
        print(
            f"  rank={dp_rank} port={vllm_engine_port} "
            f"visible_devices={visible_devices}"
        )
        process = multiprocessing.Process(
            target=run_command,
            args=(visible_devices, dp_rank, vllm_engine_port, args),
        )
        processes.append(process)
        process.start()

    failed = False
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed = True
            print(
                f"Local worker process {process.pid} exited with code {process.exitcode}",
                file=sys.stderr,
            )

    # if failed:
    #     sys.exit(1)
