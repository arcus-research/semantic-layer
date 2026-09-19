import { PAPER_DEFAULTS } from './types.js';

/**
 * Regularized Bradley–Terry MM fit (paper Appendix B.4).
 * Skips self-comparisons, adds pseudo-count \(\lambda\) on every ordered pair,
 * and renormalizes \(\sum_i \gamma_i = n\) after each update.
 */
export function fitRegularizedBradleyTerry(
  winMatrix: number[][],
  lambda: number,
  options?: { maxIterations?: number; tolerance?: number },
): number[] {
  const n = winMatrix.length;
  if (n === 0) {
    return [];
  }
  if (!(lambda > 0)) {
    throw new Error('SIFT BT prior lambda must be > 0');
  }
  if (n === 1) {
    return [1];
  }

  const maxIterations = options?.maxIterations ?? PAPER_DEFAULTS.btMaxIterations;
  const tolerance = options?.tolerance ?? PAPER_DEFAULTS.btTolerance;
  let gamma = Array.from({ length: n }, () => 1);

  for (let iter = 0; iter < maxIterations; iter += 1) {
    const next = new Array<number>(n);
    for (let i = 0; i < n; i += 1) {
      let numer = 0;
      let denom = 0;
      for (let j = 0; j < n; j += 1) {
        if (i === j) {
          continue;
        }
        const winsIj = winMatrix[i]?.[j] ?? 0;
        const winsJi = winMatrix[j]?.[i] ?? 0;
        numer += winsIj + lambda;
        denom += (winsIj + winsJi + 2 * lambda) / (gamma[i] + gamma[j]);
      }
      next[i] = numer / denom;
    }

    const sum = next.reduce((total, value) => total + value, 0);
    for (let i = 0; i < n; i += 1) {
      next[i] = (next[i] / sum) * n;
    }

    let maxDelta = 0;
    for (let i = 0; i < n; i += 1) {
      maxDelta = Math.max(maxDelta, Math.abs(next[i] - gamma[i]));
    }
    gamma = next;
    if (maxDelta < tolerance) {
      break;
    }
  }

  return gamma;
}

export function emptyWinMatrix(n: number): number[][] {
  return Array.from({ length: n }, () => Array.from({ length: n }, () => 0));
}

export function growWinMatrix(matrix: number[][], n: number): number[][] {
  const grown = matrix.map((row) => {
    const next = row.slice();
    while (next.length < n) {
      next.push(0);
    }
    return next;
  });
  while (grown.length < n) {
    grown.push(Array.from({ length: n }, () => 0));
  }
  return grown;
}
