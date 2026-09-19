"""Rank-based parent sampling (paper Eq. 1)."""

from __future__ import annotations

import math
from collections.abc import Callable


def parent_sampling_weights(
    bt_ranks: list[int],
    accuracy_ranks: list[int],
    visits: list[int],
    alpha: float,
    beta: float,
    eta: float,
) -> list[float]:
    weights: list[float] = []
    for index, bt_rank in enumerate(bt_ranks):
        quality = math.exp(-alpha * bt_rank - beta * accuracy_ranks[index])
        weights.append(quality * (1 + visits[index]) ** -eta)
    return weights


def parent_sampling_probabilities(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(weights) for _ in weights]
    return [weight / total for weight in weights]


def sample_weighted_index(weights: list[float], random: Callable[[], float]) -> int:
    if not weights:
        raise ValueError("cannot sample from an empty archive")
    probe = random() * sum(weights)
    for index, weight in enumerate(weights):
        probe -= weight
        if probe <= 0:
            return index
    return len(weights) - 1


def top_m_geometric_mass(alpha: float, top_m: int, n: int) -> float:
    if n <= 0:
        return 0.0
    if alpha == 0:
        return top_m / n
    return (1.0 - math.exp(-alpha * top_m)) / (1.0 - math.exp(-alpha * n))
