import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, expect, it } from 'vitest';

import { initialize, resetCaptureForTests, validateArtifact, type SourceSink } from '../src/index.js';

afterEach(async () => resetCaptureForTests());

it.each(['valid', 'absent', 'empty', 'malformed', 'mixed', 'unresolved', 'omitted', 'mixed_omitted'])(
  'seals source loss scope from public receipt refs: %s', async (mode) => {
    const output = await mkdtemp(join(tmpdir(), 'scoped-source-loss-'));
    const capture = initialize({ output, serviceName: 'scoped-source-loss' });
    let installed: SourceSink | undefined;
    capture.installSource({
      metadata: { name: 'fixture', seam: 'callback', identityDomain: 'fixture', coverage: [] },
      install(sink) {
        installed = sink;
        return { deactivate() {}, drain() {} };
      },
    });
    const sink = installed!;
    const root = sink.openTrace({ name: 'run', semantic: { type: 'agent.run', name: 'run' } });
    if (!root.accepted) throw new Error(root.reason);
    const requests = ['first', 'second', 'unrelated'].map((name) => {
      const receipt = sink.record({
        kind: 'model', phase: 'start', name, trace: root.identity,
        parentRecordId: root.recordId, native: null,
        semantic: { type: 'model.request' },
      });
      if (!receipt.accepted) throw new Error(receipt.reason);
      return receipt.recordId;
    });
    const omitted = sink.record({
      kind: 'log', phase: 'event', name: 'redundant', trace: root.identity,
      parentRecordId: root.recordId, native: null,
      semantic: { type: 'capture.redundant' },
    });
    if (!omitted.accepted) throw new Error(omitted.reason);
    const refs: Record<string, unknown> = {
      omitted: [omitted.recordId], mixed_omitted: [requests[0], omitted.recordId],
      valid: [requests[1], requests[0], requests[1]], empty: [],
      malformed: requests[0], mixed: [requests[0], 42],
      unresolved: [requests[0], 'future-source-record'],
    };
    const gap = sink.record({
      kind: 'unknown', phase: 'gap', name: 'gap', trace: root.identity,
      parentRecordId: root.recordId, native: null,
      semantic: {
        type: 'capture.gap', reason: 'fixture_missing_evidence',
        detail: 'The callback omitted provider evidence.', count: 3,
        ...(mode === 'absent' ? {} : { affects_refs: refs[mode] }),
      },
    });
    expect(gap.accepted).toBe(true);
    sink.record({
      kind: 'lifecycle', phase: 'end', name: 'run', trace: root.identity,
      parentRecordId: root.recordId, native: null,
      semantic: { type: 'agent.run', status: 'succeeded' },
    });
    const closed = await capture.shutdown();
    const rows = (await readFile(join(closed.artifactPath, 'trace.jsonl'), 'utf8'))
      .trim().split('\n').map((line) => JSON.parse(line));
    const losses = rows.filter((row) => row.kind === 'loss');
    const original = losses.find((row) => row.data.reason === 'fixture_missing_evidence');
    expect(original.data).toMatchObject({
      reason: 'fixture_missing_evidence', detail: 'The callback omitted provider evidence.', count: 3,
    });
    expect((original.links ?? []).filter((link: { type: string }) => link.type === 'affects'))
      .toEqual(mode === 'valid' ? [
        { type: 'affects', record: requests[1] }, { type: 'affects', record: requests[0] },
      ] : []);
    const invalid = ['malformed', 'mixed', 'unresolved', 'omitted', 'mixed_omitted'].includes(mode);
    expect(losses).toHaveLength(invalid ? 2 : 1);
    if (invalid) expect(losses[1].data.reason).toBe('unresolved_affected_ref');
    expect(closed.losses.fixture_missing_evidence).toBe(3);
    await expect(validateArtifact(closed.artifactPath)).resolves.toMatchObject({ valid: true, issues: [] });
  },
);
