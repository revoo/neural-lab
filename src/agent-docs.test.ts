// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';

/** CLAUDE.md is read by Claude Code; AGENTS.md (and AGENT.md, a link to it) by other coding agents. They must say the same thing. */
describe('agent instruction files', () => {
  it('CLAUDE.md, AGENTS.md and AGENT.md are identical', async () => {
    // node:fs through a computed specifier: the app's tsconfig has no Node types.
    const fs = (await import(/* @vite-ignore */ 'node:' + 'fs')) as { readFileSync: (p: string, enc: string) => string };
    const claude = fs.readFileSync('CLAUDE.md', 'utf8');
    expect(fs.readFileSync('AGENTS.md', 'utf8')).toBe(claude);
    expect(fs.readFileSync('AGENT.md', 'utf8')).toBe(claude);
  });
});
