import type { AgentFiles } from './types.js';

/**
 * Paper safety hygiene: allow-list of editable runtime files and rejection of
 * patches that touch harness or benchmark code.
 */
export const DEFAULT_EDITABLE_PATHS = [
  /^coding_agent\.py$/,
  /^tools\/.+\.py$/,
  /^utils\/.+\.py$/,
];

export const DEFAULT_FORBIDDEN_PATHS = [
  /(?:^|\/)harness(?:\/|$)/i,
  /(?:^|\/)benchmark(?:\/|$)/i,
  /(?:^|\/)polyglot(?:\/|$)/i,
  /(?:^|\/)swe-?bench(?:\/|$)/i,
  /(?:^|\/)terminal-?bench(?:\/|$)/i,
];

export interface PatchAdmission {
  ok: boolean;
  path?: string;
  reason?: 'forbidden' | 'not_allowlisted';
}

function matchesAny(path: string, patterns: readonly RegExp[]): boolean {
  return patterns.some((pattern) => pattern.test(path));
}

export function admitPatch(
  parentFiles: AgentFiles,
  childFiles: AgentFiles,
  allow = DEFAULT_EDITABLE_PATHS,
  deny = DEFAULT_FORBIDDEN_PATHS,
): PatchAdmission {
  const paths = new Set([...Object.keys(parentFiles), ...Object.keys(childFiles)]);
  for (const path of paths) {
    const before = parentFiles[path];
    const after = childFiles[path];
    if (before === after) {
      continue;
    }
    if (matchesAny(path, deny)) {
      return { ok: false, path, reason: 'forbidden' };
    }
    if (!matchesAny(path, allow)) {
      return { ok: false, path, reason: 'not_allowlisted' };
    }
  }
  return { ok: true };
}
