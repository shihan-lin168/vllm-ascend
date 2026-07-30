#!/usr/bin/env python3
"""Minimal LMHead-TP NPUGraph lifecycle reproducer."""

from __future__ import annotations

import argparse
import gc
import os
from datetime import datetime

import torch
import torch.distributed as dist
import torch_npu


WORLD_SIZE = 4
MODEL_TP_GROUP_RANKS = ((0, 1), (2, 3))
LMHEAD_TP_SIZE = 2
LMHEAD_TP_GROUP_RANKS = ((0, 2), (1, 3))
MODEL_TOKENS = 3
LMHEAD_TOKENS_PER_RANK = 16
HIDDEN_SIZE = 7168
VOCAB_SHARD_SIZE = 2048
LMHEAD_CALLS_PER_GRAPH = 2


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


def init_tp_group(
    rank: int,
    tp_group_ranks: tuple[tuple[int, ...], ...],
    group_name: str,
) -> tuple[dist.ProcessGroup, tuple[int, ...]]:
    rank_group = None
    rank_group_ranks = None
    for group_ranks in tp_group_ranks:
        group = dist.new_group(ranks=list(group_ranks), backend="hccl")
        if rank in group_ranks:
            rank_group = group
            rank_group_ranks = group_ranks

    if rank_group is None or rank_group_ranks is None:
        raise RuntimeError(f"rank {rank} is not in a {group_name} group")
    return rank_group, rank_group_ranks


def lmhead_tp(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    temporary_op: str,
    lmhead_tp_group: dist.ProcessGroup,
) -> list[torch.Tensor]:
    logits = []
    for _ in range(LMHEAD_CALLS_PER_GRAPH):
        if temporary_op == "all_gather":
            gathered_hidden_states = torch.empty(
                (
                    LMHEAD_TP_SIZE * hidden_states.shape[0],
                    hidden_states.shape[1],
                ),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            dist.all_gather_into_tensor(
                gathered_hidden_states,
                hidden_states,
                group=lmhead_tp_group,
            )
        else:
            gathered_hidden_states = hidden_states.repeat(
                (LMHEAD_TP_SIZE, 1)
            )
        logits.append(torch.matmul(gathered_hidden_states, weight.t()))
    return logits


def graph_workload(
    model_hidden_states: torch.Tensor,
    lmhead_hidden_states: torch.Tensor,
    weight: torch.Tensor,
    lmhead_op: str,
    model_tp_group: dist.ProcessGroup,
    lmhead_tp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    reduced_hidden_states = model_hidden_states.clone()
    dist.all_reduce(reduced_hidden_states, group=model_tp_group)
    logits = lmhead_tp(
        lmhead_hidden_states,
        weight,
        lmhead_op,
        lmhead_tp_group,
    )
    return reduced_hidden_states, logits


def capture_and_release_temporary_graph(
    rank: int,
    weight: torch.Tensor,
    temporary_op: str,
    model_tp_group: dist.ProcessGroup,
    lmhead_tp_group: dist.ProcessGroup,
) -> None:
    model_hidden_states = make_hidden_states(rank, MODEL_TOKENS)
    lmhead_hidden_states = make_hidden_states(
        rank,
        LMHEAD_TOKENS_PER_RANK,
    )
    graph_pool = torch.npu.graph_pool_handle()
    graph = torch.npu.NPUGraph()

    log(rank, f"temporary capture begin: {temporary_op}")
    with torch.npu.graph(graph, pool=graph_pool):
        reduced_hidden_states, logits = graph_workload(
            model_hidden_states,
            lmhead_hidden_states,
            weight,
            temporary_op,
            model_tp_group,
            lmhead_tp_group,
        )
    torch.npu.synchronize()
    log(rank, "temporary capture end")

    del reduced_hidden_states
    del logits
    del model_hidden_states
    del lmhead_hidden_states
    del graph_pool
    del graph
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    log(rank, "temporary graph released")


def capture_and_replay_persistent_graph(
    rank: int,
    weight: torch.Tensor,
    reference_reduced_hidden_states: torch.Tensor,
    reference_logits: list[torch.Tensor],
    replays: int,
    model_tp_group: dist.ProcessGroup,
    lmhead_tp_group: dist.ProcessGroup,
) -> None:
    model_hidden_states = make_hidden_states(rank, MODEL_TOKENS)
    lmhead_hidden_states = make_hidden_states(
        rank,
        LMHEAD_TOKENS_PER_RANK,
    )
    graph_pool = torch.npu.graph_pool_handle()
    graph = torch.npu.NPUGraph()

    log(rank, "persistent capture begin")
    with torch.npu.graph(graph, pool=graph_pool):
        reduced_hidden_states, logits = graph_workload(
            model_hidden_states,
            lmhead_hidden_states,
            weight,
            "all_gather",
            model_tp_group,
            lmhead_tp_group,
        )
    torch.npu.synchronize()
    log(rank, "persistent capture end")

    for replay_index in range(replays):
        graph.replay()
        torch.npu.synchronize()
        if not torch.equal(
            reduced_hidden_states,
            reference_reduced_hidden_states,
        ):
            max_diff = (
                reduced_hidden_states.float()
                - reference_reduced_hidden_states.float()
            ).abs().max().item()
            raise RuntimeError(
                f"persistent replay {replay_index + 1} all-reduce mismatch: "
                f"max_abs_diff={max_diff}"
            )
        for actual, expected in zip(logits, reference_logits, strict=True):
            if not torch.equal(actual, expected):
                max_diff = (actual.float() - expected.float()).abs().max().item()
                raise RuntimeError(
                    f"persistent replay {replay_index + 1} logits mismatch: "
                    f"max_abs_diff={max_diff}"
                )
        log(rank, f"persistent replay {replay_index + 1} passed")


def eager_reference(
    rank: int,
    weight: torch.Tensor,
    model_tp_group: dist.ProcessGroup,
    lmhead_tp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    model_hidden_states = make_hidden_states(rank, MODEL_TOKENS)
    lmhead_hidden_states = make_hidden_states(
        rank,
        LMHEAD_TOKENS_PER_RANK,
    )
    reduced_hidden_states, logits = graph_workload(
        model_hidden_states,
        lmhead_hidden_states,
        weight,
        "all_gather",
        model_tp_group,
        lmhead_tp_group,
    )
    torch.npu.synchronize()
    return (
        reduced_hidden_states.clone(),
        [tensor.clone() for tensor in logits],
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
        model_tp_group, model_tp_group_ranks = init_tp_group(
            rank,
            MODEL_TP_GROUP_RANKS,
            "model TP",
        )
        lmhead_tp_group, lmhead_tp_group_ranks = init_tp_group(
            rank,
            LMHEAD_TP_GROUP_RANKS,
            "LMHead TP",
        )
        log(
            rank,
            (
                f"start temporary_op={args.temporary_op} "
                f"model_tp_group={model_tp_group_ranks} "
                f"lmhead_tp_group={lmhead_tp_group_ranks} "
                f"HCCL_OP_EXPANSION_MODE="
                f"{os.getenv('HCCL_OP_EXPANSION_MODE', '<unset>')} "
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
        reference_reduced_hidden_states, reference_logits = eager_reference(
            rank,
            weight,
            model_tp_group,
            lmhead_tp_group,
        )
        dist.barrier()
        log(rank, "eager model-TP and LMHead-TP passed")

        capture_and_release_temporary_graph(
            rank,
            weight,
            args.temporary_op,
            model_tp_group,
            lmhead_tp_group,
        )
        capture_and_replay_persistent_graph(
            rank,
            weight,
            reference_reduced_hidden_states,
            reference_logits,
            args.replays,
            model_tp_group,
            lmhead_tp_group,
        )

        torch.npu.synchronize()
        dist.barrier()
        log(rank, "PASS")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
