"""Archive node and host seams for SIFT (arXiv:2609.19526)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

AgentFiles = dict[str, str]
AccuracySource = Literal["measured", "inherited"]
JudgePreference = Literal["candidate", "target"]
FailureReason = Literal["easy_eval", "unsafe_patch", "expand_error"]


@dataclass
class NodeCosts:
    expansion: float = 0.0
    judge: float = 0.0
    eval: float = 0.0


@dataclass
class SiftNode:
    id: int
    parent_id: int | None
    files: AgentFiles
    accuracy: float
    accuracy_source: AccuracySource
    bt_strength: float
    visits: int
    costs: NodeCosts = field(default_factory=NodeCosts)


@dataclass(frozen=True)
class ExpansionResult:
    files: AgentFiles
    cost: float = 0.0


@dataclass(frozen=True)
class EasyEvalResult:
    passed: bool
    cost: float = 0.0


@dataclass(frozen=True)
class FullEvalResult:
    accuracy: float
    cost: float = 0.0


class SiftHost(Protocol):
    async def expand(self, parent: SiftNode) -> ExpansionResult:
        """Diagnose+SelfImprove from a sampled parent."""

    async def easy_evaluate(self, node: SiftNode) -> EasyEvalResult:
        """4-task easy gate used to discard broken patches."""

    async def full_evaluate(self, node: SiftNode) -> FullEvalResult:
        """Fixed-subset evaluation that produces measured accuracy."""

    async def judge(self, candidate: SiftNode, target: SiftNode) -> JudgePreference:
        """Pairwise preference. The judge must not receive task outcomes."""


@dataclass(frozen=True)
class QueuedEvaluation:
    node_id: int
    priority: float


@dataclass(frozen=True)
class PendingJob:
    kind: Literal["expansion", "evaluation"]
    parent_id: int | None = None
    node_id: int | None = None
    priority: float | None = None


@dataclass(frozen=True)
class FailedExpansion:
    parent_id: int
    reason: FailureReason


@dataclass
class SiftCheckpoint:
    archive: list[SiftNode]
    win_matrix: list[list[float]]
    evaluation_queue: list[QueuedEvaluation]
    pending_jobs: list[PendingJob]
    failed: list[FailedExpansion]
    expansion_attempts: int
    next_id: int


@dataclass
class SiftResult:
    winner: SiftNode
    archive: list[SiftNode]
    win_matrix: list[list[float]]
    failed: list[FailedExpansion]
    checkpoint: SiftCheckpoint


@dataclass(frozen=True)
class PaperDefaults:
    comparison_targets: int = 10
    alpha: float = 1.0
    beta: float = 1.0
    eta: float = 1.0
    lambda_: float = 1.0
    expansion_slots: int = 1
    evaluator_slots: int = 1
    bt_tolerance: float = 1e-6
    bt_max_iterations: int = 100
    file_truncate_chars: int = 30_000
    easy_gate_tasks: int = 4


PAPER_DEFAULTS = PaperDefaults()
