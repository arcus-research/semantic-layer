import { fitRegularizedBradleyTerry, growWinMatrix } from './bradley-terry.js';
import {
  combinedRankScore,
  ranksDescending,
  selectTopByCombinedRank,
} from './ranking.js';
import { parentSamplingWeights, sampleWeightedIndex } from './sampling.js';
import { admitPatch } from './safety.js';
import {
  PAPER_DEFAULTS,
  type FailedExpansion,
  type PendingJob,
  type QueuedEvaluation,
  type SiftCheckpoint,
  type SiftConfig,
  type SiftHost,
  type SiftNode,
  type SiftResult,
} from './types.js';

interface ResolvedConfig {
  maxArchiveSize: number;
  comparisonTargets: number;
  alpha: number;
  beta: number;
  eta: number;
  lambda: number;
  expansionSlots: number;
  evaluatorSlots: number;
  maxExpansionAttempts: number;
  random: () => number;
  checkpoint?: SiftConfig['checkpoint'];
}

function resolveConfig(config: SiftConfig): ResolvedConfig {
  const maxArchiveSize = config.maxArchiveSize;
  const expansionSlots = config.expansionSlots ?? PAPER_DEFAULTS.expansionSlots;
  const evaluatorSlots = config.evaluatorSlots ?? PAPER_DEFAULTS.evaluatorSlots;
  if (expansionSlots < 1 || evaluatorSlots < 1) {
    throw new Error('SIFT expansionSlots and evaluatorSlots must be >= 1');
  }
  return {
    maxArchiveSize,
    comparisonTargets: config.comparisonTargets ?? PAPER_DEFAULTS.comparisonTargets,
    alpha: config.alpha ?? PAPER_DEFAULTS.alpha,
    beta: config.beta ?? PAPER_DEFAULTS.beta,
    eta: config.eta ?? PAPER_DEFAULTS.eta,
    lambda: config.lambda ?? PAPER_DEFAULTS.lambda,
    expansionSlots,
    evaluatorSlots,
    maxExpansionAttempts: config.maxExpansionAttempts ?? maxArchiveSize * 4,
    random: config.random ?? Math.random,
    checkpoint: config.checkpoint,
  };
}

function emptyCosts(): SiftNode['costs'] {
  return { expansion: 0, judge: 0, eval: 0 };
}

function cloneNode(node: SiftNode): SiftNode {
  return {
    ...node,
    files: { ...node.files },
    costs: { ...node.costs },
  };
}

export class SiftSearch {
  private readonly host: SiftHost;
  private readonly config: ResolvedConfig;
  private archive: SiftNode[] = [];
  private winMatrix: number[][] = [];
  private evaluationQueue: QueuedEvaluation[] = [];
  private failed: FailedExpansion[] = [];
  private expansionAttempts = 0;
  private nextId = 1;
  private readonly inflightExpansions = new Map<
    Promise<void>,
    { parentId: number; done: boolean }
  >();
  private readonly inflightEvaluations = new Map<Promise<void>, { nodeId: number }>();

  constructor(host: SiftHost, config: SiftConfig) {
    this.host = host;
    this.config = resolveConfig(config);
  }

  async run(rootFiles: SiftNode['files']): Promise<SiftResult> {
    const rootEval = await this.host.fullEvaluate({ id: 0, files: rootFiles });
    const root: SiftNode = {
      id: 0,
      parentId: null,
      files: { ...rootFiles },
      accuracy: rootEval.accuracy,
      accuracySource: 'measured',
      btStrength: 1,
      visits: 0,
      costs: { ...emptyCosts(), eval: rootEval.cost ?? 0 },
    };
    this.archive = [root];
    this.winMatrix = [[0]];
    this.nextId = 1;

    while (this.shouldContinue()) {
      this.launchExpansions();
      this.launchEvaluations();
      await this.checkpoint();
      if (this.inflightExpansions.size === 0 && this.inflightEvaluations.size === 0) {
        if (!this.canSpawnExpansion() && this.evaluationQueue.length === 0) {
          break;
        }
        continue;
      }
      await Promise.race([
        ...this.inflightExpansions.keys(),
        ...this.inflightEvaluations.keys(),
      ]);
    }

    await Promise.all([
      ...this.inflightExpansions.keys(),
      ...this.inflightEvaluations.keys(),
    ]);
    await this.checkpoint();
    return this.result();
  }

  snapshot(): SiftCheckpoint {
    return {
      archive: this.archive.map(cloneNode),
      winMatrix: this.winMatrix.map((row) => row.slice()),
      evaluationQueue: this.evaluationQueue.map((item) => ({ ...item })),
      pendingJobs: this.pendingJobs(),
      failed: this.failed.map((item) => ({ ...item })),
      expansionAttempts: this.expansionAttempts,
      nextId: this.nextId,
    };
  }

  private shouldContinue(): boolean {
    return (
      this.archive.length < this.config.maxArchiveSize ||
      this.evaluationQueue.length > 0 ||
      this.inflightExpansions.size > 0 ||
      this.inflightEvaluations.size > 0
    );
  }

  private canSpawnExpansion(): boolean {
    return (
      this.archive.length + this.inflightExpansions.size < this.config.maxArchiveSize &&
      this.expansionAttempts < this.config.maxExpansionAttempts &&
      this.archive.length > 0
    );
  }

  private launchExpansions(): void {
    while (
      this.inflightExpansions.size < this.config.expansionSlots &&
      this.canSpawnExpansion()
    ) {
      const parent = this.sampleParent();
      parent.visits += 1;
      this.expansionAttempts += 1;
      const parentId = parent.id;
      const tracked = { parentId, done: false };
      const job = this.runExpansion(parentId).finally(() => {
        tracked.done = true;
        this.inflightExpansions.delete(job);
      });
      this.inflightExpansions.set(job, tracked);
    }
  }

  private launchEvaluations(): void {
    this.evaluationQueue.sort((left, right) => left.priority - right.priority || left.nodeId - right.nodeId);
    while (
      this.inflightEvaluations.size < this.config.evaluatorSlots &&
      this.evaluationQueue.length > 0
    ) {
      const next = this.evaluationQueue.shift();
      if (!next) {
        break;
      }
      const nodeId = next.nodeId;
      const job = this.runEvaluation(nodeId).finally(() => {
        this.inflightEvaluations.delete(job);
      });
      this.inflightEvaluations.set(job, { nodeId });
    }
  }

  private async runExpansion(parentId: number): Promise<void> {
    const parent = this.requireNode(parentId);
    let files: SiftNode['files'];
    let expansionCost = 0;
    try {
      const expanded = await this.host.expand(parent);
      files = expanded.files;
      expansionCost = expanded.cost ?? 0;
    } catch {
      this.failed.push({ parentId, reason: 'expand_error' });
      return;
    }

    const admission = admitPatch(parent.files, files);
    if (!admission.ok) {
      this.failed.push({ parentId, reason: 'unsafe_patch' });
      return;
    }

    const childId = this.nextId;
    const easy = await this.host.easyEvaluate({ id: childId, files });
    if (!easy.passed) {
      this.failed.push({ parentId, reason: 'easy_eval' });
      return;
    }

    const child: SiftNode = {
      id: childId,
      parentId,
      files: { ...files },
      accuracy: parent.accuracy,
      accuracySource: 'inherited',
      btStrength: 1,
      visits: 0,
      costs: { expansion: expansionCost, judge: 0, eval: easy.cost ?? 0 },
    };
    this.nextId += 1;
    this.archive.push(child);
    this.winMatrix = growWinMatrix(this.winMatrix, this.archive.length);

    await this.judgeAgainstTargets(child);
    this.refitBradleyTerry();
    this.enqueueEvaluation(child.id);
  }

  private async judgeAgainstTargets(child: SiftNode): Promise<void> {
    const { btRanks, accuracyRanks } = this.currentRanks();
    const ids = this.archive.map((node) => node.id);
    const targets = selectTopByCombinedRank(
      btRanks,
      accuracyRanks,
      ids,
      child.id,
      this.config.comparisonTargets,
      this.config.alpha,
      this.config.beta,
    );
    const childIndex = this.indexOf(child.id);
    for (const targetId of targets) {
      const target = this.requireNode(targetId);
      const preference = await this.host.judge(child, target);
      const targetIndex = this.indexOf(targetId);
      if (preference === 'candidate') {
        this.winMatrix[childIndex][targetIndex] += 1;
      } else {
        this.winMatrix[targetIndex][childIndex] += 1;
      }
    }
  }

  private async runEvaluation(nodeId: number): Promise<void> {
    const node = this.archive.find((item) => item.id === nodeId);
    if (!node) {
      return;
    }
    const measured = await this.host.fullEvaluate(node);
    node.accuracy = measured.accuracy;
    node.accuracySource = 'measured';
    node.costs.eval += measured.cost ?? 0;
  }

  private refitBradleyTerry(): void {
    const strengths = fitRegularizedBradleyTerry(this.winMatrix, this.config.lambda);
    for (let index = 0; index < this.archive.length; index += 1) {
      this.archive[index].btStrength = strengths[index] ?? 1;
    }
  }

  private currentRanks(): { btRanks: number[]; accuracyRanks: number[]; ids: number[] } {
    const ids = this.archive.map((node) => node.id);
    const btRanks = ranksDescending(
      this.archive.map((node) => node.btStrength),
      ids,
    );
    const accuracyRanks = ranksDescending(
      this.archive.map((node) => node.accuracy),
      ids,
    );
    return { btRanks, accuracyRanks, ids };
  }

  private enqueueEvaluation(nodeId: number): void {
    const { btRanks, accuracyRanks } = this.currentRanks();
    const index = this.indexOf(nodeId);
    const priority = combinedRankScore(
      btRanks[index],
      accuracyRanks[index],
      this.config.alpha,
      this.config.beta,
    );
    this.evaluationQueue.push({ nodeId, priority });
  }

  private sampleParent(): SiftNode {
    const { btRanks, accuracyRanks } = this.currentRanks();
    const weights = parentSamplingWeights(
      btRanks,
      accuracyRanks,
      this.archive.map((node) => node.visits),
      this.config.alpha,
      this.config.beta,
      this.config.eta,
    );
    return this.archive[sampleWeightedIndex(weights, this.config.random)];
  }

  private indexOf(id: number): number {
    const index = this.archive.findIndex((node) => node.id === id);
    if (index < 0) {
      throw new Error(`unknown SIFT node ${id}`);
    }
    return index;
  }

  private requireNode(id: number): SiftNode {
    return this.archive[this.indexOf(id)];
  }

  private pendingJobs(): PendingJob[] {
    return [
      ...[...this.inflightExpansions.values()].map((job) => ({
        kind: 'expansion' as const,
        parentId: job.parentId,
      })),
      ...[...this.inflightEvaluations.values()].map((job) => ({
        kind: 'evaluation' as const,
        nodeId: job.nodeId,
      })),
      ...this.evaluationQueue.map((item) => ({
        kind: 'evaluation' as const,
        nodeId: item.nodeId,
        priority: item.priority,
      })),
    ];
  }

  private async checkpoint(): Promise<void> {
    if (this.config.checkpoint) {
      await this.config.checkpoint(this.snapshot());
    }
  }

  private result(): SiftResult {
    const measured = this.archive.filter((node) => node.accuracySource === 'measured');
    const pool = measured.length > 0 ? measured : this.archive;
    const winner = pool.reduce((best, node) => (node.accuracy > best.accuracy ? node : best));
    return {
      winner: cloneNode(winner),
      archive: this.archive.map(cloneNode),
      winMatrix: this.winMatrix.map((row) => row.slice()),
      failed: this.failed.map((item) => ({ ...item })),
      checkpoint: this.snapshot(),
    };
  }
}

export async function runSiftSearch(
  host: SiftHost,
  rootFiles: SiftNode['files'],
  config: SiftConfig,
): Promise<SiftResult> {
  return new SiftSearch(host, config).run(rootFiles);
}
