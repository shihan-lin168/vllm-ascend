#!/usr/bin/env python3
"""Minimal LMHead-TP NPUGraph lifecycle reproducer."""

from __future__ import annotations

import argparse
import gc
import os
from datetime import datetime
from time import sleep

import torch
import torch.distributed as dist
import torch_npu


WORLD_SIZE = 2
TOKENS_PER_RANK = 16
HIDDEN_SIZE = 1024
VOCAB_SHARD_SIZE = 2048
LMHEAD_CALLS_PER_GRAPH = 2


def log(rank: int, message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}] rank={rank} {message}", flush=True)


def make_hidden_states(rank: int) -> torch.Tensor:
    return torch.full(
        (TOKENS_PER_RANK, HIDDEN_SIZE),
        float(rank + 1),
        dtype=torch.float16,
        device="npu",
    )


def lmhead_tp(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    temporary_op: str,
) -> list[torch.Tensor]:
    logits = []
    for _ in range(LMHEAD_CALLS_PER_GRAPH):
        if temporary_op == "all_gather":
            gathered_hidden_states = torch.empty(
                (WORLD_SIZE * TOKENS_PER_RANK, HIDDEN_SIZE),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            dist.all_gather_into_tensor(
                gathered_hidden_states,
                hidden_states,
            )
        else:
            gathered_hidden_states = hidden_states.repeat((WORLD_SIZE, 1))
        logits.append(torch.matmul(gathered_hidden_states, weight.t()))
    return logits


def capture_and_release_temporary_graph(
    rank: int,
    weight: torch.Tensor,
    temporary_op: str,
) -> None:
    hidden_states = make_hidden_states(rank)
    graph_pool = torch.npu.graph_pool_handle()
    graph = torch.npu.NPUGraph()

    log(rank, f"temporary capture begin: {temporary_op}")
    with torch.npu.graph(graph, pool=graph_pool):
        logits = lmhead_tp(hidden_states, weight, temporary_op)
    torch.npu.synchronize()
    log(rank, "temporary capture end")

    del logits
    del hidden_states
    del graph_pool
    del graph
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    log(rank, "temporary graph released")


def capture_and_replay_persistent_graph(
    rank: int,
    weight: torch.Tensor,
    reference_logits: list[torch.Tensor],
    replays: int,
    rank1_replay_delay_ms: int,
) -> None:
    hidden_states = make_hidden_states(rank)
    graph_pool = torch.npu.graph_pool_handle()
    graph = torch.npu.NPUGraph()

    log(rank, "persistent capture begin")
    with torch.npu.graph(graph, pool=graph_pool):
        logits = lmhead_tp(hidden_states, weight, "all_gather")
    torch.npu.synchronize()
    log(rank, "persistent capture end")

    if rank == 1 and rank1_replay_delay_ms:
        log(
            rank,
            f"delaying persistent replay by {rank1_replay_delay_ms} ms",
        )
        sleep(rank1_replay_delay_ms / 1000)

    for replay_index in range(replays):
        graph.replay()
        torch.npu.synchronize()
        for actual, expected in zip(logits, reference_logits, strict=True):
            if not torch.equal(actual, expected):
                max_diff = (actual.float() - expected.float()).abs().max().item()
                raise RuntimeError(
                    f"persistent replay {replay_index + 1} logits mismatch: "
                    f"max_abs_diff={max_diff}"
                )
        log(rank, f"persistent replay {replay_index + 1} passed")


def eager_reference(rank: int, weight: torch.Tensor) -> list[torch.Tensor]:
    hidden_states = make_hidden_states(rank)
    logits = lmhead_tp(hidden_states, weight, "all_gather")
    torch.npu.synchronize()
    return [tensor.clone() for tensor in logits]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temporary-op",
        choices=("all_gather", "local_repeat"),
        required=True,
    )
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--rank1-replay-delay-ms", type=int, default=0)
    args = parser.parse_args()
    if args.replays <= 0:
        parser.error("--replays must be greater than zero")
    if args.rank1_replay_delay_ms < 0:
        parser.error("--rank1-replay-delay-ms must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"expected WORLD_SIZE={WORLD_SIZE}, got {world_size}")

    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    try:
        log(
            rank,
            (
                f"start temporary_op={args.temporary_op} "
                f"rank1_replay_delay_ms={args.rank1_replay_delay_ms} "
                f"HCCL_OP_EXPANSION_MODE="
                f"{os.getenv('HCCL_OP_EXPANSION_MODE', '<unset>')} "
                f"torch={torch.__version__} "
                f"torch_npu={getattr(torch_npu, '__version__', '<unknown>')}"
            ),
        )

        torch.manual_seed(rank)
        weight = torch.randn(
            (VOCAB_SHARD_SIZE, HIDDEN_SIZE),
            dtype=torch.float16,
            device="npu",
        )

        dist.barrier()
        reference_logits = eager_reference(rank, weight)
        dist.barrier()
        log(rank, "eager LMHead-TP passed")

        capture_and_release_temporary_graph(rank, weight, args.temporary_op)
        capture_and_replay_persistent_graph(
            rank,
            weight,
            reference_logits,
            args.replays,
            args.rank1_replay_delay_ms,
        )

        torch.npu.synchronize()
        dist.barrier()
        log(rank, "PASS")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
