# semantic-layer-sift

Local SIFT search kernel from [Self Improvement via Fast Tree-search](https://arxiv.org/abs/2609.19526).

This package implements the paper’s ranking and tree-search loop. Diagnose+SelfImprove, the judge model, and benchmark runners stay host-injected. It does not call model APIs or run Polyglot, SWE-bench, or TerminalBench.

See [SIFT documentation](../../docs/sdk/sift.md).
