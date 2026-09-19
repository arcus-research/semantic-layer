/**
 * Dense ranks with best rank 0 (paper Table 8).
 * Equal values share a rank. Node id breaks sort order only, not the rank.
 */
export function ranksDescending(values: readonly number[], ids: readonly number[]): number[] {
  if (values.length !== ids.length) {
    throw new Error('values and ids must have the same length');
  }
  const order = values.map((value, index) => ({ value, index, id: ids[index] }));
  order.sort((left, right) => right.value - left.value || left.id - right.id);

  const ranks = new Array<number>(values.length);
  let rank = 0;
  for (let position = 0; position < order.length; position += 1) {
    if (position > 0 && order[position].value !== order[position - 1].value) {
      rank += 1;
    }
    ranks[order[position].index] = rank;
  }
  return ranks;
}

/** Lower combined rank is stronger (Algorithm 1 target and eval priority). */
export function combinedRankScore(
  btRank: number,
  accuracyRank: number,
  alpha: number,
  beta: number,
): number {
  return alpha * btRank + beta * accuracyRank;
}

export function selectTopByCombinedRank(
  btRanks: readonly number[],
  accuracyRanks: readonly number[],
  ids: readonly number[],
  excludeId: number,
  limit: number,
  alpha: number,
  beta: number,
): number[] {
  const scored = ids
    .map((id, index) => ({
      id,
      score: combinedRankScore(btRanks[index], accuracyRanks[index], alpha, beta),
    }))
    .filter((row) => row.id !== excludeId)
    .sort((left, right) => left.score - right.score || left.id - right.id);
  return scored.slice(0, limit).map((row) => row.id);
}
