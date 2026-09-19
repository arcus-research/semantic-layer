import { PAPER_DEFAULTS, type AgentFiles } from './types.js';

export const FULL_FILES_SYSTEM_PROMPT = `You are an expert code reviewer evaluating multiple candidate modifications of an AI coding agent. For each candidate you will be shown its FULL implementation files (already modified). Compare them directly.

Your task is to RANK the candidates from most to least likely to improve the agent's performance.

## IMPORTANT: Rank the candidates on how likely they are to improve performance, but also check for correctness. You can assume that imports to files that you cannot see will work fine.

You MUST respond in the following JSON format and nothing else.
Write the \`reasoning\` field FIRST so your critique drives the ranking, not the
other way around — once the ranking tokens are emitted you cannot revise them:
{"reasoning": "<brief explanation of your ranking>", "ranking": [<list of candidate ids from best to worst>]}`;

export const FULL_FILES_USER_TEMPLATE = `Below are {num_candidates} candidate modifications of the AI coding agent. Each candidate's complete implementation files are shown. Rank them from most to least likely to improve agent performance.

{candidates_section}

Respond with JSON only, \`reasoning\` field first then \`ranking\`: {"reasoning": "<str>", "ranking": [<candidate ids best to worst>]}`;

const CACHE_PATH = /(?:^|\/)__pycache__(?:\/|$)|(?:^|\/)[^/]+\.py[co]$/;

export function isRuntimeAgentPath(path: string): boolean {
  if (CACHE_PATH.test(path)) {
    return false;
  }
  if (path === 'coding_agent.py') {
    return true;
  }
  return /^(?:tools|utils)\/.*\.py$/.test(path);
}

export function runtimeFiles(files: AgentFiles): AgentFiles {
  const selected: AgentFiles = {};
  for (const [path, contents] of Object.entries(files)) {
    if (isRuntimeAgentPath(path)) {
      selected[path] = contents;
    }
  }
  return selected;
}

export function truncateFileContents(
  contents: string,
  limit = PAPER_DEFAULTS.fileTruncateChars,
): string {
  return contents.length <= limit ? contents : contents.slice(0, limit);
}

function unionRuntimePaths(candidates: ReadonlyArray<{ id: number; files: AgentFiles }>): string[] {
  const paths = new Set<string>();
  for (const candidate of candidates) {
    for (const path of Object.keys(runtimeFiles(candidate.files))) {
      paths.add(path);
    }
  }
  return [...paths].sort();
}

function differingPaths(candidates: ReadonlyArray<{ id: number; files: AgentFiles }>): string[] {
  return unionRuntimePaths(candidates).filter((path) => {
    const values = candidates.map((candidate) => runtimeFiles(candidate.files)[path] ?? null);
    return values.some((value) => value !== values[0]);
  });
}

export function renderCandidatesSection(
  candidates: ReadonlyArray<{ id: number; files: AgentFiles }>,
): string {
  const paths = differingPaths(candidates);
  const lines = [
    'Files that are byte-identical across all candidates have been omitted; only files that differ between candidates are shown below.',
  ];
  for (const candidate of candidates) {
    const files = runtimeFiles(candidate.files);
    lines.push(`## Candidate ${candidate.id}`);
    for (const path of paths) {
      lines.push(`### ${path}`);
      if (path in files) {
        lines.push('```python');
        lines.push(truncateFileContents(files[path]));
        lines.push('```');
      } else {
        lines.push('(file absent in this candidate)');
      }
    }
  }
  return lines.join('\n');
}

export function renderFullFilesJudgePrompt(
  candidates: ReadonlyArray<{ id: number; files: AgentFiles }>,
): { system: string; user: string } {
  const user = FULL_FILES_USER_TEMPLATE
    .replace('{num_candidates}', String(candidates.length))
    .replace('{candidates_section}', renderCandidatesSection(candidates));
  return { system: FULL_FILES_SYSTEM_PROMPT, user };
}

export function parseJudgeRanking(text: string): number[] | null {
  const trimmed = text.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/);
  const payload = fenced?.[1] ?? trimmed;
  const start = payload.indexOf('{');
  const end = payload.lastIndexOf('}');
  if (start < 0 || end <= start) {
    return null;
  }
  try {
    const parsed = JSON.parse(payload.slice(start, end + 1)) as { ranking?: unknown };
    if (!Array.isArray(parsed.ranking)) {
      return null;
    }
    const ranking = parsed.ranking.map((value) => Number(value));
    if (ranking.some((value) => !Number.isFinite(value))) {
      return null;
    }
    return ranking;
  } catch {
    return null;
  }
}

export function preferenceFromRanking(
  ranking: readonly number[],
  candidateId: number,
  targetId: number,
): 'candidate' | 'target' | null {
  const candidateIndex = ranking.indexOf(candidateId);
  const targetIndex = ranking.indexOf(targetId);
  if (candidateIndex < 0 || targetIndex < 0) {
    return null;
  }
  return candidateIndex < targetIndex ? 'candidate' : 'target';
}
