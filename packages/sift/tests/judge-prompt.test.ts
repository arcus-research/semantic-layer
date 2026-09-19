import { describe, expect, it } from 'vitest';

import {
  FULL_FILES_SYSTEM_PROMPT,
  parseJudgeRanking,
  preferenceFromRanking,
  renderFullFilesJudgePrompt,
  truncateFileContents,
} from '../src/judge-prompt.js';

describe('full-file judge presentation (paper Appendix B.2)', () => {
  it('renders differing runtime files and omits identical ones', () => {
    const long = 'x'.repeat(30_050);
    const prompt = renderFullFilesJudgePrompt([
      {
        id: 2,
        files: {
          'coding_agent.py': 'root agent',
          'tools/test_runner.py': 'runner',
          'tools/__pycache__/skip.pyc': 'cache',
          'README.md': 'not runtime',
        },
      },
      {
        id: 7,
        files: {
          'coding_agent.py': 'child agent',
          'tools/test_runner.py': 'runner',
          'utils/git_utils.py': 'utils',
        },
      },
    ]);

    expect(prompt.system).toBe(FULL_FILES_SYSTEM_PROMPT);
    expect(prompt.user).toContain('Below are 2 candidate modifications');
    expect(prompt.user).toContain('## Candidate 2');
    expect(prompt.user).toContain('## Candidate 7');
    expect(prompt.user).toContain('### coding_agent.py');
    expect(prompt.user).toContain('### utils/git_utils.py');
    expect(prompt.user).toContain('(file absent in this candidate)');
    expect(prompt.user).not.toContain('test_runner.py');
    expect(prompt.user).not.toContain('README.md');
    expect(prompt.user).not.toContain('__pycache__');
    expect(truncateFileContents(long)).toHaveLength(30_000);
  });

  it('parses a ranking JSON object and the pairwise winner', () => {
    const ranking = parseJudgeRanking(
      'prefix {"reasoning": "better tools", "ranking": [7, 2]} suffix',
    );
    expect(ranking).toEqual([7, 2]);
    expect(preferenceFromRanking(ranking ?? [], 7, 2)).toBe('candidate');
    expect(preferenceFromRanking(ranking ?? [], 2, 7)).toBe('target');
  });
});
