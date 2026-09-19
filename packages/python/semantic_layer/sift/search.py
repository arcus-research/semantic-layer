"""Disaggregated SIFT master loop (paper Algorithm 1)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from random import random as default_random

from .bradley_terry import fit_regularized_bradley_terry, grow_win_matrix
from .ranking import combined_rank_score, ranks_descending, select_top_by_combined_rank
from .safety import admit_patch
from .sampling import parent_sampling_weights, sample_weighted_index
from .types import (
    PAPER_DEFAULTS,
    AgentFiles,
    FailedExpansion,
    NodeCosts,
    PendingJob,
    QueuedEvaluation,
    SiftCheckpoint,
    SiftHost,
    SiftNode,
    SiftResult,
)


@dataclass(frozen=True)
class _ResolvedConfig:
    max_archive_size: int
    comparison_targets: int
    alpha: float
    beta: float
    eta: float
    lambda_: float
    expansion_slots: int
    evaluator_slots: int
    max_expansion_attempts: int
    random: Callable[[], float]
    checkpoint: Callable[[SiftCheckpoint], Awaitable[None] | None] | None


def _resolve_config(
    max_archive_size: int,
    *,
    comparison_targets: int | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    eta: float | None = None,
    lambda_: float | None = None,
    expansion_slots: int | None = None,
    evaluator_slots: int | None = None,
    max_expansion_attempts: int | None = None,
    random: Callable[[], float] | None = None,
    checkpoint: Callable[[SiftCheckpoint], Awaitable[None] | None] | None = None,
) -> _ResolvedConfig:
    slots = expansion_slots if expansion_slots is not None else PAPER_DEFAULTS.expansion_slots
    eval_slots = evaluator_slots if evaluator_slots is not None else PAPER_DEFAULTS.evaluator_slots
    if slots < 1 or eval_slots < 1:
        raise ValueError("SIFT expansion_slots and evaluator_slots must be >= 1")
    return _ResolvedConfig(
        max_archive_size=max_archive_size,
        comparison_targets=(
            comparison_targets
            if comparison_targets is not None
            else PAPER_DEFAULTS.comparison_targets
        ),
        alpha=PAPER_DEFAULTS.alpha if alpha is None else alpha,
        beta=PAPER_DEFAULTS.beta if beta is None else beta,
        eta=PAPER_DEFAULTS.eta if eta is None else eta,
        lambda_=PAPER_DEFAULTS.lambda_ if lambda_ is None else lambda_,
        expansion_slots=slots,
        evaluator_slots=eval_slots,
        max_expansion_attempts=(
            max_expansion_attempts
            if max_expansion_attempts is not None
            else max_archive_size * 4
        ),
        random=default_random if random is None else random,
        checkpoint=checkpoint,
    )


def _clone_node(node: SiftNode) -> SiftNode:
    return replace(
        node,
        files=dict(node.files),
        costs=replace(node.costs),
    )


class SiftSearch:
    def __init__(
        self,
        host: SiftHost,
        max_archive_size: int,
        *,
        comparison_targets: int | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        eta: float | None = None,
        lambda_: float | None = None,
        expansion_slots: int | None = None,
        evaluator_slots: int | None = None,
        max_expansion_attempts: int | None = None,
        random: Callable[[], float] | None = None,
        checkpoint: Callable[[SiftCheckpoint], Awaitable[None] | None] | None = None,
    ) -> None:
        self._host = host
        self._config = _resolve_config(
            max_archive_size,
            comparison_targets=comparison_targets,
            alpha=alpha,
            beta=beta,
            eta=eta,
            lambda_=lambda_,
            expansion_slots=expansion_slots,
            evaluator_slots=evaluator_slots,
            max_expansion_attempts=max_expansion_attempts,
            random=random,
            checkpoint=checkpoint,
        )
        self._archive: list[SiftNode] = []
        self._win_matrix: list[list[float]] = []
        self._evaluation_queue: list[QueuedEvaluation] = []
        self._failed: list[FailedExpansion] = []
        self._expansion_attempts = 0
        self._next_id = 1
        self._inflight_expansions: dict[asyncio.Task[None], int] = {}
        self._inflight_evaluations: dict[asyncio.Task[None], int] = {}

    async def run(self, root_files: AgentFiles) -> SiftResult:
        root_eval = await self._host.full_evaluate(
            SiftNode(
                id=0,
                parent_id=None,
                files=dict(root_files),
                accuracy=0.0,
                accuracy_source="measured",
                bt_strength=1.0,
                visits=0,
            )
        )
        root = SiftNode(
            id=0,
            parent_id=None,
            files=dict(root_files),
            accuracy=root_eval.accuracy,
            accuracy_source="measured",
            bt_strength=1.0,
            visits=0,
            costs=NodeCosts(eval=root_eval.cost),
        )
        self._archive = [root]
        self._win_matrix = [[0.0]]
        self._next_id = 1

        while self._should_continue():
            self._launch_expansions()
            self._launch_evaluations()
            await self._checkpoint()
            inflight = list(self._inflight_expansions) + list(self._inflight_evaluations)
            if not inflight:
                if not self._can_spawn_expansion() and not self._evaluation_queue:
                    break
                continue
            done, _ = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                self._inflight_expansions.pop(task, None)
                self._inflight_evaluations.pop(task, None)
                await task

        remaining = list(self._inflight_expansions) + list(self._inflight_evaluations)
        if remaining:
            await asyncio.gather(*remaining)
            self._inflight_expansions.clear()
            self._inflight_evaluations.clear()
        await self._checkpoint()
        return self._result()

    def snapshot(self) -> SiftCheckpoint:
        return SiftCheckpoint(
            archive=[_clone_node(node) for node in self._archive],
            win_matrix=deepcopy(self._win_matrix),
            evaluation_queue=list(self._evaluation_queue),
            pending_jobs=self._pending_jobs(),
            failed=list(self._failed),
            expansion_attempts=self._expansion_attempts,
            next_id=self._next_id,
        )

    def _should_continue(self) -> bool:
        return (
            len(self._archive) < self._config.max_archive_size
            or bool(self._evaluation_queue)
            or bool(self._inflight_expansions)
            or bool(self._inflight_evaluations)
        )

    def _can_spawn_expansion(self) -> bool:
        return (
            len(self._archive) + len(self._inflight_expansions) < self._config.max_archive_size
            and self._expansion_attempts < self._config.max_expansion_attempts
            and bool(self._archive)
        )

    def _launch_expansions(self) -> None:
        while (
            len(self._inflight_expansions) < self._config.expansion_slots
            and self._can_spawn_expansion()
        ):
            parent = self._sample_parent()
            parent.visits += 1
            self._expansion_attempts += 1
            parent_id = parent.id
            task = asyncio.create_task(self._run_expansion(parent_id))
            self._inflight_expansions[task] = parent_id

    def _launch_evaluations(self) -> None:
        self._evaluation_queue.sort(key=lambda item: (item.priority, item.node_id))
        while (
            len(self._inflight_evaluations) < self._config.evaluator_slots
            and self._evaluation_queue
        ):
            queued = self._evaluation_queue.pop(0)
            task = asyncio.create_task(self._run_evaluation(queued.node_id))
            self._inflight_evaluations[task] = queued.node_id

    async def _run_expansion(self, parent_id: int) -> None:
        parent = self._require_node(parent_id)
        try:
            expanded = await self._host.expand(parent)
        except Exception:
            self._failed.append(FailedExpansion(parent_id=parent_id, reason="expand_error"))
            return
        if not admit_patch(parent.files, expanded.files).ok:
            self._failed.append(FailedExpansion(parent_id=parent_id, reason="unsafe_patch"))
            return
        child_id = self._next_id
        easy = await self._host.easy_evaluate(
            SiftNode(
                id=child_id,
                parent_id=parent_id,
                files=dict(expanded.files),
                accuracy=parent.accuracy,
                accuracy_source="inherited",
                bt_strength=1.0,
                visits=0,
            )
        )
        if not easy.passed:
            self._failed.append(FailedExpansion(parent_id=parent_id, reason="easy_eval"))
            return
        child = SiftNode(
            id=child_id,
            parent_id=parent_id,
            files=dict(expanded.files),
            accuracy=parent.accuracy,
            accuracy_source="inherited",
            bt_strength=1.0,
            visits=0,
            costs=NodeCosts(expansion=expanded.cost, eval=easy.cost),
        )
        self._next_id += 1
        self._archive.append(child)
        self._win_matrix = grow_win_matrix(self._win_matrix, len(self._archive))
        await self._judge_against_targets(child)
        self._refit_bradley_terry()
        self._enqueue_evaluation(child.id)

    async def _judge_against_targets(self, child: SiftNode) -> None:
        bt_ranks, accuracy_ranks, ids = self._current_ranks()
        targets = select_top_by_combined_rank(
            bt_ranks,
            accuracy_ranks,
            ids,
            child.id,
            self._config.comparison_targets,
            self._config.alpha,
            self._config.beta,
        )
        child_index = self._index_of(child.id)
        for target_id in targets:
            target = self._require_node(target_id)
            preference = await self._host.judge(child, target)
            target_index = self._index_of(target_id)
            if preference == "candidate":
                self._win_matrix[child_index][target_index] += 1
            else:
                self._win_matrix[target_index][child_index] += 1

    async def _run_evaluation(self, node_id: int) -> None:
        node = next((item for item in self._archive if item.id == node_id), None)
        if node is None:
            return
        measured = await self._host.full_evaluate(node)
        node.accuracy = measured.accuracy
        node.accuracy_source = "measured"
        node.costs.eval += measured.cost

    def _refit_bradley_terry(self) -> None:
        strengths = fit_regularized_bradley_terry(self._win_matrix, self._config.lambda_)
        for index, node in enumerate(self._archive):
            node.bt_strength = strengths[index] if index < len(strengths) else 1.0

    def _current_ranks(self) -> tuple[list[int], list[int], list[int]]:
        ids = [node.id for node in self._archive]
        bt_ranks = ranks_descending([node.bt_strength for node in self._archive], ids)
        accuracy_ranks = ranks_descending([node.accuracy for node in self._archive], ids)
        return bt_ranks, accuracy_ranks, ids

    def _enqueue_evaluation(self, node_id: int) -> None:
        bt_ranks, accuracy_ranks, _ = self._current_ranks()
        index = self._index_of(node_id)
        priority = combined_rank_score(
            bt_ranks[index],
            accuracy_ranks[index],
            self._config.alpha,
            self._config.beta,
        )
        self._evaluation_queue.append(QueuedEvaluation(node_id=node_id, priority=priority))

    def _sample_parent(self) -> SiftNode:
        bt_ranks, accuracy_ranks, _ = self._current_ranks()
        weights = parent_sampling_weights(
            bt_ranks,
            accuracy_ranks,
            [node.visits for node in self._archive],
            self._config.alpha,
            self._config.beta,
            self._config.eta,
        )
        return self._archive[sample_weighted_index(weights, self._config.random)]

    def _index_of(self, node_id: int) -> int:
        for index, node in enumerate(self._archive):
            if node.id == node_id:
                return index
        raise KeyError(f"unknown SIFT node {node_id}")

    def _require_node(self, node_id: int) -> SiftNode:
        return self._archive[self._index_of(node_id)]

    def _pending_jobs(self) -> list[PendingJob]:
        jobs = [
            PendingJob(kind="expansion", parent_id=parent_id)
            for parent_id in self._inflight_expansions.values()
        ]
        jobs.extend(
            PendingJob(kind="evaluation", node_id=node_id)
            for node_id in self._inflight_evaluations.values()
        )
        jobs.extend(
            PendingJob(kind="evaluation", node_id=item.node_id, priority=item.priority)
            for item in self._evaluation_queue
        )
        return jobs

    async def _checkpoint(self) -> None:
        if self._config.checkpoint is None:
            return
        maybe = self._config.checkpoint(self.snapshot())
        if asyncio.iscoroutine(maybe):
            await maybe

    def _result(self) -> SiftResult:
        measured = [node for node in self._archive if node.accuracy_source == "measured"]
        pool = measured or self._archive
        winner = max(pool, key=lambda node: node.accuracy)
        return SiftResult(
            winner=_clone_node(winner),
            archive=[_clone_node(node) for node in self._archive],
            win_matrix=deepcopy(self._win_matrix),
            failed=list(self._failed),
            checkpoint=self.snapshot(),
        )


async def run_sift_search(
    host: SiftHost,
    root_files: AgentFiles,
    max_archive_size: int,
    *,
    comparison_targets: int | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    eta: float | None = None,
    lambda_: float | None = None,
    expansion_slots: int | None = None,
    evaluator_slots: int | None = None,
    max_expansion_attempts: int | None = None,
    random: Callable[[], float] | None = None,
    checkpoint: Callable[[SiftCheckpoint], Awaitable[None] | None] | None = None,
) -> SiftResult:
    return await SiftSearch(
        host,
        max_archive_size,
        comparison_targets=comparison_targets,
        alpha=alpha,
        beta=beta,
        eta=eta,
        lambda_=lambda_,
        expansion_slots=expansion_slots,
        evaluator_slots=evaluator_slots,
        max_expansion_attempts=max_expansion_attempts,
        random=random,
        checkpoint=checkpoint,
    ).run(root_files)
