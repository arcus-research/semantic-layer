"""Full-file judge prompts (paper Appendix B.2)."""

from __future__ import annotations

import json
import re
from typing import TypedDict

from .types import PAPER_DEFAULTS, AgentFiles, JudgePreference

FULL_FILES_SYSTEM_PROMPT = """You are an expert code reviewer evaluating multiple candidate modifications of an AI coding agent. For each candidate you will be shown its FULL implementation files (already modified). Compare them directly.

Your task is to RANK the candidates from most to least likely to improve the agent's performance.

## IMPORTANT: Rank the candidates on how likely they are to improve performance, but also check for correctness. You can assume that imports to files that you cannot see will work fine.

You MUST respond in the following JSON format and nothing else.
Write the `reasoning` field FIRST so your critique drives the ranking, not the
other way around — once the ranking tokens are emitted you cannot revise them:
{"reasoning": "<brief explanation of your ranking>", "ranking": [<list of candidate ids from best to worst>]}"""

FULL_FILES_USER_TEMPLATE = """Below are {num_candidates} candidate modifications of the AI coding agent. Each candidate's complete implementation files are shown. Rank them from most to least likely to improve agent performance.

{candidates_section}

Respond with JSON only, `reasoning` field first then `ranking`: {"reasoning": "<str>", "ranking": [<candidate ids best to worst>]}"""

_CACHE_PATH = re.compile(r"(?:^|/)__pycache__(?:/|$)|(?:^|/)[^/]+\.py[co]$")


class CandidateView(TypedDict):
    id: int
    files: AgentFiles


class JudgePrompt(TypedDict):
    system: str
    user: str


def is_runtime_agent_path(path: str) -> bool:
    if _CACHE_PATH.search(path):
        return False
    if path == "coding_agent.py":
        return True
    return bool(re.match(r"^(?:tools|utils)/.*\.py$", path))


def runtime_files(files: AgentFiles) -> AgentFiles:
    return {path: contents for path, contents in files.items() if is_runtime_agent_path(path)}


def truncate_file_contents(contents: str, limit: int | None = None) -> str:
    cap = PAPER_DEFAULTS.file_truncate_chars if limit is None else limit
    return contents if len(contents) <= cap else contents[:cap]


def _union_runtime_paths(candidates: list[CandidateView]) -> list[str]:
    paths: set[str] = set()
    for candidate in candidates:
        paths.update(runtime_files(candidate["files"]))
    return sorted(paths)


def _differing_paths(candidates: list[CandidateView]) -> list[str]:
    differing: list[str] = []
    for path in _union_runtime_paths(candidates):
        values = [runtime_files(candidate["files"]).get(path) for candidate in candidates]
        if any(value != values[0] for value in values):
            differing.append(path)
    return differing


def render_candidates_section(candidates: list[CandidateView]) -> str:
    paths = _differing_paths(candidates)
    lines = [
        "Files that are byte-identical across all candidates have been omitted; "
        "only files that differ between candidates are shown below."
    ]
    for candidate in candidates:
        files = runtime_files(candidate["files"])
        lines.append(f"## Candidate {candidate['id']}")
        for path in paths:
            lines.append(f"### {path}")
            if path in files:
                lines.append("```python")
                lines.append(truncate_file_contents(files[path]))
                lines.append("```")
            else:
                lines.append("(file absent in this candidate)")
    return "\n".join(lines)


def render_full_files_judge_prompt(candidates: list[CandidateView]) -> JudgePrompt:
    user = FULL_FILES_USER_TEMPLATE.replace("{num_candidates}", str(len(candidates))).replace(
        "{candidates_section}", render_candidates_section(candidates)
    )
    return {"system": FULL_FILES_SYSTEM_PROMPT, "user": user}


def parse_judge_ranking(text: str) -> list[int] | None:
    trimmed = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", trimmed)
    payload = fenced.group(1) if fenced else trimmed
    start = payload.find("{")
    end = payload.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(payload[start : end + 1])
    except json.JSONDecodeError:
        return None
    ranking = parsed.get("ranking")
    if not isinstance(ranking, list):
        return None
    values: list[int] = []
    for item in ranking:
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            return None
    return values


def preference_from_ranking(
    ranking: list[int], candidate_id: int, target_id: int
) -> JudgePreference | None:
    try:
        candidate_index = ranking.index(candidate_id)
        target_index = ranking.index(target_id)
    except ValueError:
        return None
    return "candidate" if candidate_index < target_index else "target"
