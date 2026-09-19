import { describe, expect, it } from 'vitest';

import { fitRegularizedBradleyTerry } from '../src/bradley-terry.js';

describe('regularized Bradley–Terry (paper Appendix B.4)', () => {
  it('assigns the root strength 1 when the archive has one node', () => {
    expect(fitRegularizedBradleyTerry([[0]], 1)).toEqual([1]);
  });

  it('skips self-comparisons even if the diagonal is nonzero', () => {
    const withDiagonal = [
      [9, 1],
      [0, 4],
    ];
    const withoutDiagonal = [
      [0, 1],
      [0, 0],
    ];
    expect(fitRegularizedBradleyTerry(withDiagonal, 1)).toEqual(
      fitRegularizedBradleyTerry(withoutDiagonal, 1),
    );
  });

  it('renormalizes strengths so they sum to n', () => {
    const strengths = fitRegularizedBradleyTerry(
      [
        [0, 3, 1],
        [0, 0, 2],
        [1, 0, 0],
      ],
      1,
    );
    expect(strengths).toHaveLength(3);
    expect(strengths.reduce((sum, value) => sum + value, 0)).toBeCloseTo(3, 8);
    expect(strengths[0]).toBeGreaterThan(strengths[1]);
    expect(strengths[1]).toBeGreaterThan(strengths[2]);
  });

  it('matches the two-node MM fixed point for W01=1, W10=0, lambda=1', () => {
    const strengths = fitRegularizedBradleyTerry(
      [
        [0, 1],
        [0, 0],
      ],
      1,
    );
    expect(strengths[0]).toBeCloseTo(4 / 3, 6);
    expect(strengths[1]).toBeCloseTo(2 / 3, 6);
  });

  it('rejects a non-positive BT prior', () => {
    expect(() => fitRegularizedBradleyTerry([[0, 1], [0, 0]], 0)).toThrow(/lambda/);
  });
});
