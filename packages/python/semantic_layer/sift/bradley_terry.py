"""Regularized Bradley–Terry MM fit (paper Appendix B.4)."""

from __future__ import annotations

from .types import PAPER_DEFAULTS


def empty_win_matrix(n: int) -> list[list[float]]:
    return [[0.0 for _ in range(n)] for _ in range(n)]


def grow_win_matrix(matrix: list[list[float]], n: int) -> list[list[float]]:
    grown = [row[:] + [0.0] * (n - len(row)) for row in matrix]
    while len(grown) < n:
        grown.append([0.0] * n)
    return grown


def fit_regularized_bradley_terry(
    win_matrix: list[list[float]],
    lambda_: float,
    *,
    max_iterations: int | None = None,
    tolerance: float | None = None,
) -> list[float]:
    n = len(win_matrix)
    if n == 0:
        return []
    if not lambda_ > 0:
        raise ValueError("SIFT BT prior lambda must be > 0")
    if n == 1:
        return [1.0]

    if max_iterations is None:
        max_iterations = PAPER_DEFAULTS.bt_max_iterations
    tolerance = tolerance if tolerance is not None else PAPER_DEFAULTS.bt_tolerance
    gamma = [1.0] * n

    for _ in range(max_iterations):
        nxt = [0.0] * n
        for i in range(n):
            numer = 0.0
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                wins_ij = win_matrix[i][j] if j < len(win_matrix[i]) else 0.0
                wins_ji = win_matrix[j][i] if i < len(win_matrix[j]) else 0.0
                numer += wins_ij + lambda_
                denom += (wins_ij + wins_ji + 2.0 * lambda_) / (gamma[i] + gamma[j])
            nxt[i] = numer / denom
        total = sum(nxt)
        nxt = [value / total * n for value in nxt]
        max_delta = max(abs(left - right) for left, right in zip(nxt, gamma, strict=True))
        gamma = nxt
        if max_delta < tolerance:
            break
    return gamma
