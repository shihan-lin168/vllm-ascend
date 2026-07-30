#!/usr/bin/env python3
"""Minimal LMHead-TP NPUGraph profiling lifecycle reproducer."""

from __future__ import annotations

import argparse
import gc
import os
from datetime import datetime

import torch
import torch.distributed as dist
import torch_npu


WORLD_SIZE = 2
CAPTURE_SIZES = (48, 3)
HIDDEN_SIZE = 7168
VOCAB_SHARD_SIZE = 2048
LMHEAD_CALLS_PER_GRAPH = 3


def log(rank: int, message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}] rank={rank} {message}", flush=True)


def make_hidden_states(rank: int, num_tokens: int) -> torch.Tensor:
    return torch.full(
        (num_tokens, HIDDEN_SIZE),
        float(rank + 1),
        dtype=torch.bfloat16,
        device="npu",
    )


def lmhead_tp(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    op: str,
) -> list[torch.Tensor]:
    logits = []
    for _ in range(LMHEAD_CALLS_PER_GRAPH):
        if op == "all_gather":
            gathered_hidden_states = torch.empty(
                (
                    WORLD_SIZE * hidden_states.shape[0],
                    hidden_states.shape[1],
                ),
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


def eager_references(
    rank: int,
    weight: torch.Tensor,
) -> list[list[torch.Tensor]]:
    references = []
    for capture_size in CAPTURE_SIZES:
        hidden_states = make_hidden_states(rank, capture_size)
        logits = lmhead_tp(hidden_states, weight, "all_gather")
        torch.npu.synchronize()
        references.append([tensor.clone() for tensor in logits])
    return references


def capture_graph_set(
    rank: int,
    weight: torch.Tensor,
    op: str,
) -> tuple[
    object,
    list[torch.npu.NPUGraph],
    list[torch.Tensor],
    list[list[torch.Tensor]],
]:
    graph_pool = torch.npu.graph_pool_handle()
    graphs = []
    inputs = []
    outputs = []

    for capture_size in CAPTURE_SIZES:
        hidden_states = make_hidden_states(rank, capture_size)
        graph = torch.npu.NPUGraph()
        log(rank, f"capture begin: op={op} tokens={capture_size}")
        with torch.npu.graph(graph, pool=graph_pool):
            logits = lmhead_tp(hidden_states, weight, op)
        torch.npu.synchronize()
        log(rank, f"capture end: op={op} tokens={capture_size}")

        graphs.append(graph)
        inputs.append(hidden_states)
        outputs.append(logits)

    return graph_pool, graphs, inputs, outputs


def capture_and_release_temporary_graphs(
    rank: int,
    weight: torch.Tensor,
    temporary_op: str,
) -> None:
    graph_pool, graphs, inputs, outputs = capture_graph_set(
        rank,
        weight,
        temporary_op,
    )

    del outputs
    del inputs
    del graphs
    del graph_pool
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    log(rank, "temporary graphs released")


def capture_and_replay_persistent_graphs(
    rank: int,
    weight: torch.Tensor,
    references: list[list[torch.Tensor]],
    replays: int,
) -> None:
    graph_pool, graphs, inputs, outputs = capture_graph_set(
        rank,
        weight,
        "all_gather",
    )

    for replay_index in range(replays):
        for capture_size, graph, actuals, expecteds in zip(
            CAPTURE_SIZES,
            graphs,
            outputs,
            references,
            strict=True,
        ):
            graph.replay()
            torch.npu.synchronize()
            for actual, expected in zip(actuals, expecteds, strict=True):
                if torch.equal(actual, expected):
                    continue
                max_diff = (actual.float() - expected.float()).abs().max().item()
                raise RuntimeError(
                    f"replay {replay_index + 1} tokens={capture_size} "
                    f"logits mismatch: max_abs_diff={max_diff}"
                )
            log(
                rank,
                f"replay {replay_index + 1} tokens={capture_size} passed",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temporary-op",
        choices=("all_gather", "local_repeat"),
        required=True,
    )
    parser.add_argument("--replays", type=int, default=20)
    args = parser.parse_args()
    if args.replays <= 0:
        parser.error("--replays must be greater than zero")
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
                f"capture_sizes={CAPTURE_SIZES} "
                f"lmhead_calls_per_graph={LMHEAD_CALLS_PER_GRAPH} "
                f"HCCL_OP_EXPANSION_MODE="
                f"{os.getenv('HCCL_OP_EXPANSION_MODE', '<unset>')} "
                f"TASK_QUEUE_ENABLE="
                f"{os.getenv('TASK_QUEUE_ENABLE', '<unset>')} "
                f"torch={torch.__version__} "
                f"torch_npu={getattr(torch_npu, '__version__', '<unknown>')}"
            ),
        )

        torch.manual_seed(rank)
        weight = torch.randn(
            (VOCAB_SHARD_SIZE, HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device="npu",
        )

        dist.barrier()
        references = eager_references(rank, weight)
        dist.barrier()
        log(rank, "eager LMHead-TP passed")

        capture_and_release_temporary_graphs(
            rank,
            weight,
            args.temporary_op,
        )
        capture_and_replay_persistent_graphs(
            rank,
            weight,
            references,
            args.replays,
        )

        torch.npu.synchronize()
        dist.barrier()
        log(rank, "PASS")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
