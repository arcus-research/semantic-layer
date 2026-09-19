/**
 * Rank-based parent sampling (paper Eq. 1 and Appendix B.4 visit form).
 * \(P(i)\propto\exp(-\alpha r_b(i)-\beta r_a(i)-\eta\log(1+v_i))\).
 */
export function parentSamplingWeights(
  btRanks: readonly number[],
  accuracyRanks: readonly number[],
  visits: readonly number[],
  alpha: number,
  beta: number,
  eta: number,
): number[] {
  return btRanks.map((btRank, index) => {
    const quality = Math.exp(-alpha * btRank - beta * accuracyRanks[index]);
    return quality * (1 + visits[index]) ** -eta;
  });
}

export function parentSamplingProbabilities(weights: readonly number[]): number[] {
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  if (total <= 0) {
    return weights.map(() => 1 / weights.length);
  }
  return weights.map((weight) => weight / total);
}

export function sampleWeightedIndex(weights: readonly number[], random: () => number): number {
  if (weights.length === 0) {
    throw new Error('cannot sample from an empty archive');
  }
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  let probe = random() * total;
  for (let index = 0; index < weights.length; index += 1) {
    probe -= weights[index];
    if (probe <= 0) {
      return index;
    }
  }
  return weights.length - 1;
}

/** Proposition 1: probability mass on ranks \(0..M-1\) when \(\beta=\eta=0\). */
export function topMGeometricMass(alpha: number, topM: number, n: number): number {
  if (n <= 0) {
    return 0;
  }
  if (alpha === 0) {
    return topM / n;
  }
  return (1 - Math.exp(-alpha * topM)) / (1 - Math.exp(-alpha * n));
}
