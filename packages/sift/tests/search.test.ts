import { describe, expect, it, vi } from 'vitest';

import { SiftSearch } from '../src/search.js';
import type { AgentFiles, SiftCheckpoint, SiftHost } from '../src/types.js';

function files(quality: number, extra: AgentFiles = {}): AgentFiles {
  return {
    'coding_agent.py': `quality=${quality}`,
    ...extra,
  };
}

function qualityOf(nodeFiles: AgentFiles): number {
  const match = /quality=(\d+)/.exec(nodeFiles['coding_agent.py'] ?? '');
  return match ? Number(match[1]) : 0;
}

describe('SIFT Algorithm 1 loop', () => {
  it('discards easy-eval failures and unsafe patches from the archive and eval queue', async () => {
    let expansions = 0;
    const judged: Array<[number, number]> = [];
    const evaluated: number[] = [];
    const host: SiftHost = {
      async expand(parent) {
        expansions += 1;
        if (expansions === 1) {
          return { files: files(0, { 'harness/eval.py': 'broken' }) };
        }
        if (expansions === 2) {
          return { files: { 'coding_agent.py': 'broken-easy' } };
        }
        return { files: files(qualityOf(parent.files) + 1) };
      },
      async easyEvaluate(node) {
        return { passed: node.files['coding_agent.py'] !== 'broken-easy' };
      },
      async fullEvaluate(node) {
        evaluated.push(node.id);
        return { accuracy: qualityOf(node.files) / 10 };
      },
      async judge(candidate, target) {
        judged.push([candidate.id, target.id]);
        return qualityOf(candidate.files) >= qualityOf(target.files) ? 'candidate' : 'target';
      },
    };

    const result = await new SiftSearch(host, {
      maxArchiveSize: 2,
      comparisonTargets: 10,
      random: () => 0,
    }).run(files(2));

    expect(result.archive.map((node) => node.id)).toEqual([0, 1]);
    expect(result.failed.map((item) => item.reason).sort()).toEqual(['easy_eval', 'unsafe_patch']);
    expect(evaluated).toEqual([0, 1]);
    expect(judged.length).toBeGreaterThan(0);
    expect(result.archive.every((node) => !('harness/eval.py' in node.files))).toBe(true);
  });

  it('inherits parent accuracy, then replaces it after full evaluation', async () => {
    let releaseEval: ((value: { accuracy: number }) => void) | undefined;
    const inheritedDuringEval: string[] = [];
    const refs: { search?: SiftSearch } = {};

    const host: SiftHost = {
      async expand() {
        return { files: files(9) };
      },
      async easyEvaluate() {
        return { passed: true };
      },
      async fullEvaluate(node) {
        if (node.id === 0) {
          return { accuracy: 0.16 };
        }
        inheritedDuringEval.push(
          refs.search?.snapshot().archive.find((item) => item.id === 1)?.accuracySource ?? 'missing',
        );
        return new Promise((resolve) => {
          releaseEval = resolve;
        });
      },
      async judge() {
        return 'candidate';
      },
    };

    refs.search = new SiftSearch(host, { maxArchiveSize: 2, random: () => 0 });
    const done = refs.search.run(files(1));
    await vi.waitFor(() => {
      const child = refs.search?.snapshot().archive.find((node) => node.id === 1);
      expect(child?.accuracySource).toBe('inherited');
      expect(child?.accuracy).toBe(0.16);
      expect(releaseEval).toBeTypeOf('function');
    });
    releaseEval?.({ accuracy: 0.44 });
    const result = await done;
    expect(result.archive[1]?.accuracy).toBe(0.44);
    expect(result.archive[1]?.accuracySource).toBe('measured');
    expect(inheritedDuringEval).toEqual(['inherited']);
  });

  it('expands a judged but not-yet-evaluated child (speculative expansion)', async () => {
    const expandParents: number[] = [];
    let releaseFirstChildEval: ((value: { accuracy: number }) => void) | undefined;

    const host: SiftHost = {
      async expand(parent) {
        expandParents.push(parent.id);
        return { files: files(parent.id + 3) };
      },
      async easyEvaluate() {
        return { passed: true };
      },
      async fullEvaluate(node) {
        if (node.id === 0) {
          return { accuracy: 0.2 };
        }
        if (node.id === 1) {
          return new Promise((resolve) => {
            releaseFirstChildEval = resolve;
          });
        }
        return { accuracy: 0.3 };
      },
      async judge(candidate, target) {
        return qualityOf(candidate.files) >= qualityOf(target.files) ? 'candidate' : 'target';
      },
    };

    const search = new SiftSearch(host, {
      maxArchiveSize: 3,
      expansionSlots: 1,
      evaluatorSlots: 1,
      random: () => 0.5,
    });
    const done = search.run(files(1));
    await vi.waitFor(() => {
      expect(expandParents).toContain(1);
      expect(releaseFirstChildEval).toBeTypeOf('function');
    });
    releaseFirstChildEval?.({ accuracy: 0.3 });
    const result = await done;
    expect(result.archive.map((node) => node.id)).toEqual([0, 1, 2]);
    expect(result.archive[1]?.visits).toBeGreaterThanOrEqual(1);
    expect(result.winner.accuracySource).toBe('measured');
  });

  it('returns the measured-accuracy winner and checkpoints archive state', async () => {
    const checkpoints: SiftCheckpoint[] = [];
    const host: SiftHost = {
      async expand(parent) {
        return { files: files(qualityOf(parent.files) + 2) };
      },
      async easyEvaluate() {
        return { passed: true };
      },
      async fullEvaluate(node) {
        return { accuracy: node.id === 0 ? 0.14 : 0.35 };
      },
      async judge() {
        return 'candidate';
      },
    };

    const result = await new SiftSearch(host, {
      maxArchiveSize: 3,
      random: () => 0,
      checkpoint: (state) => {
        checkpoints.push(state);
      },
    }).run(files(1));

    expect(result.winner.id).toBeGreaterThan(0);
    expect(result.winner.accuracy).toBe(0.35);
    expect(result.checkpoint.winMatrix.length).toBe(result.archive.length);
    expect(checkpoints.length).toBeGreaterThan(0);
    expect(checkpoints.at(-1)?.archive.map((node) => node.id)).toEqual(
      result.archive.map((node) => node.id),
    );
  });
});
