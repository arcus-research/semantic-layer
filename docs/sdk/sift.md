# SIFT tree search

SIFT (Recursive Self-Improvement via Fast Tree-search) is the search kernel
from [arXiv:2609.19526](https://arxiv.org/abs/2609.19526). It is an adjacent
local module. It does not capture traces, write sidecars, or replace the
coding agent, judge model, or benchmark runner.

## What it does

The kernel keeps an archive of agent implementations and runs the paper’s
Algorithm 1 loop:

1. Sample a parent with rank-based weights
   \(P(i)\propto\exp(-\alpha r_b(i)-\beta r_a(i)-\eta\log(1+v_i))\).
2. Ask the host to Diagnose+SelfImprove.
3. Reject patches that leave the editable allow-list or touch harness or
   benchmark paths.
4. Run the host easy gate (paper: 4-task filter). Failures never enter the
   archive or the full-evaluation queue.
5. Compare the child pairwise against up to \(K=10\) incumbents chosen by
   \(\alpha r_b+\beta r_a\). The judge sees full runtime files only, never
   task outcomes.
6. Refit a regularized Bradley–Terry model (MM update, \(\sum_i\theta_i=n\),
   skip self-comparisons, stop at \(10^{-6}\) or 100 iterations).
7. Enqueue subset evaluation with priority \(\alpha r_b+\beta r_a\). Unevaluated
   nodes inherit the parent accuracy for ranking.
8. Expand and evaluate concurrently. Return the archived node with the highest
   **measured** accuracy.

Paper defaults used in reported runs: \(\alpha=\beta=\eta=1\), \(K=10\). Best
rank is 0. The paper requires \(\lambda>0\) but does not publish the numeric
prior; this implementation defaults to \(1\).

## Interfaces

TypeScript: `semantic-layer-sift` (`packages/sift`).
Python: `semantic_layer.sift` (not exported from `semantic_layer`).

The host supplies `expand`, `easyEvaluate` / `easy_evaluate`,
`fullEvaluate` / `full_evaluate`, and pairwise `judge`. Prompt helpers render
the paper’s `full_files` system and user templates.

## Out of scope

This module does not run Polyglot, SWE-bench, or TerminalBench, call model
APIs, or evolve a DGM `coding_agent.py` harness. Those remain host
responsibilities.
