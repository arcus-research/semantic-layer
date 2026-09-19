"""Allow-list admission for expander patches (paper safety section)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .types import AgentFiles

DEFAULT_EDITABLE_PATHS = (
    re.compile(r"^coding_agent\.py$"),
    re.compile(r"^tools/.+\.py$"),
    re.compile(r"^utils/.+\.py$"),
)

DEFAULT_FORBIDDEN_PATHS = (
    re.compile(r"(?:^|/)harness(?:/|$)", re.IGNORECASE),
    re.compile(r"(?:^|/)benchmark(?:/|$)", re.IGNORECASE),
    re.compile(r"(?:^|/)polyglot(?:/|$)", re.IGNORECASE),
    re.compile(r"(?:^|/)swe-?bench(?:/|$)", re.IGNORECASE),
    re.compile(r"(?:^|/)terminal-?bench(?:/|$)", re.IGNORECASE),
)


@dataclass(frozen=True)
class PatchAdmission:
    ok: bool
    path: str | None = None
    reason: Literal["forbidden", "not_allowlisted"] | None = None


def _matches_any(path: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(path) for pattern in patterns)


def admit_patch(
    parent_files: AgentFiles,
    child_files: AgentFiles,
    allow: tuple[re.Pattern[str], ...] = DEFAULT_EDITABLE_PATHS,
    deny: tuple[re.Pattern[str], ...] = DEFAULT_FORBIDDEN_PATHS,
) -> PatchAdmission:
    paths = set(parent_files) | set(child_files)
    for path in paths:
        if parent_files.get(path) == child_files.get(path):
            continue
        if _matches_any(path, deny):
            return PatchAdmission(ok=False, path=path, reason="forbidden")
        if not _matches_any(path, allow):
            return PatchAdmission(ok=False, path=path, reason="not_allowlisted")
    return PatchAdmission(ok=True)
