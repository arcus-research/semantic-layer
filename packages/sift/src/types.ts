/** Archive node and host seams for SIFT (arXiv:2609.19526). */

export type AgentFiles = Record<string, string>;

export type AccuracySource = 'measured' | 'inherited';

export interface NodeCosts {
  expansion: number;
  judge: number;
  eval: number;
}

export interface SiftNode {
  id: number;
  parentId: number | null;
  files: AgentFiles;
  accuracy: number;
  accuracySource: AccuracySource;
  btStrength: number;
  visits: number;
  costs: NodeCosts;
}

export interface ExpansionResult {
  files: AgentFiles;
  cost?: number;
}

export interface EasyEvalResult {
  passed: boolean;
  cost?: number;
}

export interface FullEvalResult {
  accuracy: number;
  cost?: number;
}

export type JudgePreference = 'candidate' | 'target';

export interface SiftHost {
  expand(parent: SiftNode): Promise<ExpansionResult>;
  easyEvaluate(node: Pick<SiftNode, 'id' | 'files'>): Promise<EasyEvalResult>;
  fullEvaluate(node: Pick<SiftNode, 'id' | 'files'>): Promise<FullEvalResult>;
  judge(
    candidate: Pick<SiftNode, 'id' | 'files'>,
    target: Pick<SiftNode, 'id' | 'files'>,
  ): Promise<JudgePreference>;
}

export interface SiftConfig {
  /** Algorithm 1 archive cap \(N\). */
  maxArchiveSize: number;
  /** Comparison targets per new node \(K\). Paper default 10. */
  comparisonTargets?: number;
  /** Paper default 1. */
  alpha?: number;
  /** Paper default 1. */
  beta?: number;
  /** Paper default 1. */
  eta?: number;
  /**
   * BT pseudo-count \(\lambda>0\).
   * The paper requires a positive prior but does not publish the experimental value.
   */
  lambda?: number;
  expansionSlots?: number;
  evaluatorSlots?: number;
  maxExpansionAttempts?: number;
  random?: () => number;
  checkpoint?: (state: SiftCheckpoint) => void | Promise<void>;
}

export interface QueuedEvaluation {
  nodeId: number;
  priority: number;
}

export interface PendingJob {
  kind: 'expansion' | 'evaluation';
  parentId?: number;
  nodeId?: number;
  priority?: number;
}

export interface FailedExpansion {
  parentId: number;
  reason: 'easy_eval' | 'unsafe_patch' | 'expand_error';
}

export interface SiftCheckpoint {
  archive: SiftNode[];
  winMatrix: number[][];
  evaluationQueue: QueuedEvaluation[];
  pendingJobs: PendingJob[];
  failed: FailedExpansion[];
  expansionAttempts: number;
  nextId: number;
}

export interface SiftResult {
  winner: SiftNode;
  archive: SiftNode[];
  winMatrix: number[][];
  failed: FailedExpansion[];
  checkpoint: SiftCheckpoint;
}

export const PAPER_DEFAULTS = {
  comparisonTargets: 10,
  alpha: 1,
  beta: 1,
  eta: 1,
  /** Unpublished in the paper; unit pseudo-count used when the host omits \(\lambda\). */
  lambda: 1,
  expansionSlots: 1,
  evaluatorSlots: 1,
  btTolerance: 1e-6,
  btMaxIterations: 100,
  fileTruncateChars: 30_000,
  easyGateTasks: 4,
} as const;
