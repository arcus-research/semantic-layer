import { describe, expect, it } from 'vitest';

import { combinedRankScore, ranksDescending, selectTopByCombinedRank } from '../src/ranking.js';
import {
  parentSamplingProbabilities,
  parentSamplingWeights,
  topMGeometricMass,
} from '../src/sampling.js';

describe('ranks and parent sampling (paper Table 8, Eq. 1, Prop. 1)', () => {
  it('assigns best rank 0 and shares ranks for equal values', () => {
    expect(ranksDescending([1.9, 2.8, 1.9], [0, 1, 2])).toEqual([1, 0, 1]);
  });

  it('selects up to K incumbents by alpha rb + beta ra', () => {
    const ids = [0, 1, 2, 3];
    const btRanks = [3, 0, 1, 2];
    const accuracyRanks = [0, 2, 1, 3];
    expect(selectTopByCombinedRank(btRanks, accuracyRanks, ids, 1, 2, 1, 1)).toEqual([2, 0]);
    expect(combinedRankScore(1, 1, 1, 1)).toBe(2);
  });

  it('uses the visit form q_i (1+v_i)^{-eta}', () => {
    const weights = parentSamplingWeights([0, 1], [0, 0], [0, 3], 1, 1, 1);
    expect(weights[0]).toBeCloseTo(1, 10);
    expect(weights[1]).toBeCloseTo(Math.exp(-1) * 4 ** -1, 10);
  });

  it('matches Proposition 1 geometric mass on the top M ranks', () => {
    const alpha = 1;
    const n = 5;
    const topM = 2;
    const ranks = [0, 1, 2, 3, 4];
    const weights = parentSamplingWeights(ranks, [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], alpha, 0, 0);
    const probabilities = parentSamplingProbabilities(weights);
    const mass = probabilities[0] + probabilities[1];
    expect(mass).toBeCloseTo(topMGeometricMass(alpha, topM, n), 10);
    expect(topMGeometricMass(alpha, topM, n)).toBeCloseTo(
      (1 - Math.exp(-alpha * topM)) / (1 - Math.exp(-alpha * n)),
      10,
    );
  });
});
