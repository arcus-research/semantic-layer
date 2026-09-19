"""Dense ranks with best rank 0 (paper Table 8)."""

from __future__ import annotations


def ranks_descending(values: list[float], ids: list[int]) -> list[int]:
    if len(values) != len(ids):
        raise ValueError("values and ids must have the same length")
    order = sorted(range(len(values)), key=lambda index: (-values[index], ids[index]))
    ranks = [0] * len(values)
    rank = 0
    for position, index in enumerate(order):
        if position > 0 and values[index] != values[order[position - 1]]:
            rank += 1
        ranks[index] = rank
    return ranks


def combined_rank_score(bt_rank: int, accuracy_rank: int, alpha: float, beta: float) -> float:
    return alpha * bt_rank + beta * accuracy_rank


def select_top_by_combined_rank(
    bt_ranks: list[int],
    accuracy_ranks: list[int],
    ids: list[int],
    exclude_id: int,
    limit: int,
    alpha: float,
    beta: float,
) -> list[int]:
    scored = [
        (
            combined_rank_score(bt_ranks[index], accuracy_ranks[index], alpha, beta),
            ids[index],
        )
        for index, node_id in enumerate(ids)
        if node_id != exclude_id
    ]
    scored.sort()
    return [node_id for _, node_id in scored[:limit]]
