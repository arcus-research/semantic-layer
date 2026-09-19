import { describe, expect, it } from 'vitest';

import { admitPatch } from '../src/safety.js';

describe('patch admission (paper safety allow-list)', () => {
  const parent = {
    'coding_agent.py': 'base',
    'tools/edit.py': 'edit',
  };

  it('admits runtime tool and agent edits', () => {
    expect(
      admitPatch(parent, {
        ...parent,
        'coding_agent.py': 'base\nrun tests',
        'tools/test_runner.py': 'new tool',
      }),
    ).toEqual({ ok: true });
  });

  it('rejects harness and benchmark edits', () => {
    expect(
      admitPatch(parent, {
        ...parent,
        'harness/eval.py': 'relax timeouts',
      }),
    ).toMatchObject({ ok: false, reason: 'forbidden', path: 'harness/eval.py' });
    expect(
      admitPatch(parent, {
        ...parent,
        'benchmark/polyglot/task.py': 'leak',
      }),
    ).toMatchObject({ ok: false, reason: 'forbidden' });
  });

  it('rejects files outside the editable allow-list', () => {
    expect(
      admitPatch(parent, {
        ...parent,
        'docker-compose.yml': 'host mount',
      }),
    ).toMatchObject({ ok: false, reason: 'not_allowlisted', path: 'docker-compose.yml' });
  });
});
