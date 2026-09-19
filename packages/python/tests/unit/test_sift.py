from __future__ import annotations

import asyncio
import math

import pytest

from semantic_layer.sift import (
    FULL_FILES_SYSTEM_PROMPT,
    PAPER_DEFAULTS,
    SiftSearch,
    admit_patch,
    combined_rank_score,
    fit_regularized_bradley_terry,
    parent_sampling_probabilities,
    parent_sampling_weights,
    parse_judge_ranking,
    preference_from_ranking,
    ranks_descending,
    render_full_files_judge_prompt,
    select_top_by_combined_rank,
    top_m_geometric_mass,
    truncate_file_contents,
)
from semantic_layer.sift.types import EasyEvalResult, ExpansionResult, FullEvalResult, SiftNode


def _files(quality: int, extra: dict[str, str] | None = None) -> dict[str, str]:
    payload = {"coding_agent.py": f"quality={quality}"}
    if extra:
        payload.update(extra)
    return payload


def _quality(files: dict[str, str]) -> int:
    text = files.get("coding_agent.py", "")
    if text.startswith("quality="):
        return int(text.split("=", 1)[1])
    return 0


def test_bt_root_strength_and_normalization() -> None:
    assert fit_regularized_bradley_terry([[0.0]], 1.0) == [1.0]
    strengths = fit_regularized_bradley_terry(
        [
            [0.0, 3.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 0.0],
        ],
        1.0,
    )
    assert len(strengths) == 3
    assert sum(strengths) == pytest.approx(3.0)
    assert strengths[0] > strengths[1] > strengths[2]


def test_bt_skips_self_comparisons_and_matches_two_node_fixed_point() -> None:
    with_diag = [[9.0, 1.0], [0.0, 4.0]]
    without = [[0.0, 1.0], [0.0, 0.0]]
    assert fit_regularized_bradley_terry(with_diag, 1.0) == pytest.approx(
        fit_regularized_bradley_terry(without, 1.0)
    )
    strengths = fit_regularized_bradley_terry(without, 1.0)
    assert strengths[0] == pytest.approx(4 / 3)
    assert strengths[1] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="lambda"):
        fit_regularized_bradley_terry([[0.0, 1.0], [0.0, 0.0]], 0.0)


def test_ranks_and_parent_sampling_match_paper() -> None:
    assert ranks_descending([1.9, 2.8, 1.9], [0, 1, 2]) == [1, 0, 1]
    selected = select_top_by_combined_rank(
        [3, 0, 1, 2], [0, 2, 1, 3], [0, 1, 2, 3], 1, 2, 1.0, 1.0
    )
    assert selected == [
        2,
        0,
    ]
    assert combined_rank_score(1, 1, 1.0, 1.0) == 2.0
    weights = parent_sampling_weights([0, 1], [0, 0], [0, 3], 1.0, 1.0, 1.0)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(math.exp(-1) * 4**-1)
    probabilities = parent_sampling_probabilities(
        parent_sampling_weights([0, 1, 2, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], 1.0, 0.0, 0.0)
    )
    assert probabilities[0] + probabilities[1] == pytest.approx(top_m_geometric_mass(1.0, 2, 5))


def test_full_file_judge_prompt() -> None:
    prompt = render_full_files_judge_prompt(
        [
            {
                "id": 2,
                "files": {
                    "coding_agent.py": "root agent",
                    "tools/test_runner.py": "runner",
                    "tools/__pycache__/skip.pyc": "cache",
                    "README.md": "not runtime",
                },
            },
            {
                "id": 7,
                "files": {
                    "coding_agent.py": "child agent",
                    "tools/test_runner.py": "runner",
                    "utils/git_utils.py": "utils",
                },
            },
        ]
    )
    assert prompt["system"] == FULL_FILES_SYSTEM_PROMPT
    assert "Below are 2 candidate modifications" in prompt["user"]
    assert "## Candidate 7" in prompt["user"]
    assert "(file absent in this candidate)" in prompt["user"]
    assert "test_runner.py" not in prompt["user"]
    assert "README.md" not in prompt["user"]
    assert len(truncate_file_contents("x" * 30_050)) == 30_000
    ranking = parse_judge_ranking('prefix {"reasoning": "better tools", "ranking": [7, 2]} suffix')
    assert ranking == [7, 2]
    assert preference_from_ranking(ranking or [], 7, 2) == "candidate"


def test_patch_admission() -> None:
    parent = {"coding_agent.py": "base", "tools/edit.py": "edit"}
    assert admit_patch(
        parent, {**parent, "coding_agent.py": "base\nrun tests", "tools/test_runner.py": "new"}
    ).ok
    forbidden = admit_patch(parent, {**parent, "harness/eval.py": "relax"})
    assert forbidden.ok is False
    assert forbidden.reason == "forbidden"
    outside = admit_patch(parent, {**parent, "docker-compose.yml": "host"})
    assert outside.reason == "not_allowlisted"


class _ScriptedHost:
    def __init__(self) -> None:
        self.expansions = 0
        self.evaluated: list[int] = []
        self.expand_parents: list[int] = []
        self.eval_events: dict[int, asyncio.Event] = {}
        self.eval_results: dict[int, float] = {}
        self.search: SiftSearch | None = None
        self.inherited: list[str] = []

    async def expand(self, parent: SiftNode) -> ExpansionResult:
        self.expansions += 1
        self.expand_parents.append(parent.id)
        if self.expansions == 1:
            return ExpansionResult(files=_files(0, {"harness/eval.py": "broken"}))
        if self.expansions == 2:
            return ExpansionResult(files={"coding_agent.py": "broken-easy"})
        return ExpansionResult(files=_files(_quality(parent.files) + 1))

    async def easy_evaluate(self, node: SiftNode) -> EasyEvalResult:
        return EasyEvalResult(passed=node.files.get("coding_agent.py") != "broken-easy")

    async def full_evaluate(self, node: SiftNode) -> FullEvalResult:
        self.evaluated.append(node.id)
        return FullEvalResult(accuracy=_quality(node.files) / 10)

    async def judge(self, candidate: SiftNode, target: SiftNode) -> str:
        return "candidate" if _quality(candidate.files) >= _quality(target.files) else "target"


@pytest.mark.asyncio
async def test_search_discards_failed_and_unsafe_patches() -> None:
    host = _ScriptedHost()
    result = await SiftSearch(host, 2, random=lambda: 0.0).run(_files(2))
    assert [node.id for node in result.archive] == [0, 1]
    assert sorted(item.reason for item in result.failed) == ["easy_eval", "unsafe_patch"]
    assert host.evaluated == [0, 1]


class _InheritHost:
    def __init__(self) -> None:
        self.search: SiftSearch | None = None
        self.release = asyncio.Event()
        self.inherited: list[str] = []

    async def expand(self, parent: SiftNode) -> ExpansionResult:
        return ExpansionResult(files=_files(9))

    async def easy_evaluate(self, node: SiftNode) -> EasyEvalResult:
        return EasyEvalResult(passed=True)

    async def full_evaluate(self, node: SiftNode) -> FullEvalResult:
        if node.id == 0:
            return FullEvalResult(accuracy=0.16)
        assert self.search is not None
        child = next(item for item in self.search.snapshot().archive if item.id == 1)
        self.inherited.append(child.accuracy_source)
        await self.release.wait()
        return FullEvalResult(accuracy=0.44)

    async def judge(self, candidate: SiftNode, target: SiftNode) -> str:
        return "candidate"


@pytest.mark.asyncio
async def test_search_inherits_parent_accuracy_until_eval() -> None:
    host = _InheritHost()
    search = SiftSearch(host, 2, random=lambda: 0.0)
    host.search = search
    task = asyncio.create_task(search.run(_files(1)))
    for _ in range(200):
        snapshot = search.snapshot()
        child = next((node for node in snapshot.archive if node.id == 1), None)
        if child is not None:
            assert child.accuracy_source == "inherited"
            assert child.accuracy == 0.16
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("child was not archived")
    host.release.set()
    result = await task
    assert result.archive[1].accuracy == 0.44
    assert result.archive[1].accuracy_source == "measured"
    assert host.inherited == ["inherited"]


class _SpeculativeHost:
    def __init__(self) -> None:
        self.expand_parents: list[int] = []
        self.first_child_eval = asyncio.Event()

    async def expand(self, parent: SiftNode) -> ExpansionResult:
        self.expand_parents.append(parent.id)
        return ExpansionResult(files=_files(parent.id + 3))

    async def easy_evaluate(self, node: SiftNode) -> EasyEvalResult:
        return EasyEvalResult(passed=True)

    async def full_evaluate(self, node: SiftNode) -> FullEvalResult:
        if node.id == 0:
            return FullEvalResult(accuracy=0.2)
        if node.id == 1:
            await self.first_child_eval.wait()
        return FullEvalResult(accuracy=0.3)

    async def judge(self, candidate: SiftNode, target: SiftNode) -> str:
        return "candidate" if _quality(candidate.files) >= _quality(target.files) else "target"


@pytest.mark.asyncio
async def test_search_speculative_expansion_and_measured_winner() -> None:
    host = _SpeculativeHost()
    search = SiftSearch(host, 3, expansion_slots=1, evaluator_slots=1, random=lambda: 0.5)
    task = asyncio.create_task(search.run(_files(1)))
    for _ in range(200):
        if 1 in host.expand_parents:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("did not expand unevaluated child")
    host.first_child_eval.set()
    result = await task
    assert [node.id for node in result.archive] == [0, 1, 2]
    assert result.archive[1].visits >= 1
    assert result.winner.accuracy_source == "measured"
    assert PAPER_DEFAULTS.easy_gate_tasks == 4
    assert PAPER_DEFAULTS.comparison_targets == 10
