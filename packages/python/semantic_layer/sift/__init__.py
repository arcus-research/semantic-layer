"""SIFT search kernel (arXiv:2609.19526). Not part of the capture public API."""

from .bradley_terry import empty_win_matrix, fit_regularized_bradley_terry, grow_win_matrix
from .judge_prompt import (
    FULL_FILES_SYSTEM_PROMPT,
    FULL_FILES_USER_TEMPLATE,
    parse_judge_ranking,
    preference_from_ranking,
    render_candidates_section,
    render_full_files_judge_prompt,
    runtime_files,
    truncate_file_contents,
)
from .ranking import combined_rank_score, ranks_descending, select_top_by_combined_rank
from .safety import DEFAULT_EDITABLE_PATHS, DEFAULT_FORBIDDEN_PATHS, admit_patch
from .sampling import (
    parent_sampling_probabilities,
    parent_sampling_weights,
    sample_weighted_index,
    top_m_geometric_mass,
)
from .search import SiftSearch, run_sift_search
from .types import PAPER_DEFAULTS

__all__ = [
    "DEFAULT_EDITABLE_PATHS",
    "DEFAULT_FORBIDDEN_PATHS",
    "FULL_FILES_SYSTEM_PROMPT",
    "FULL_FILES_USER_TEMPLATE",
    "PAPER_DEFAULTS",
    "SiftSearch",
    "admit_patch",
    "combined_rank_score",
    "empty_win_matrix",
    "fit_regularized_bradley_terry",
    "grow_win_matrix",
    "parse_judge_ranking",
    "parent_sampling_probabilities",
    "parent_sampling_weights",
    "preference_from_ranking",
    "ranks_descending",
    "render_candidates_section",
    "render_full_files_judge_prompt",
    "run_sift_search",
    "runtime_files",
    "sample_weighted_index",
    "select_top_by_combined_rank",
    "top_m_geometric_mass",
    "truncate_file_contents",
]
