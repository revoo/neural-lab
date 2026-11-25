// authored-by: Claude Opus 5.5

/** Module names for the lab's three toy layers. */
export const LAB_MODULES = ['early', 'mid', 'head'] as const;

/**
 * The `peft` LoraConfig for a per-layer rank allocation. `r` is the most common
 * non-zero rank, `rank_pattern` overrides the others, and rank-0 layers are left
 * out of `target_modules` (LoRA needs r ≥ 1, so a frozen layer is simply not adapted).
 */
export function peftLoraLines(ranks: readonly number[]): string {
  const used = LAB_MODULES.map((m, i) => ({ m, r: ranks[i] })).filter((x) => x.r > 0);
  if (used.length === 0) return '# every rank is 0: nothing is adapted, so there is no LoraConfig to write';
  const counts = new Map<number, number>();
  for (const { r } of used) counts.set(r, (counts.get(r) ?? 0) + 1);
  const base = [...counts.entries()].sort((a, b) => b[1] - a[1] || b[0] - a[0])[0][0];
  const overrides = used.filter((x) => x.r !== base);
  const lines = [
    'LoraConfig(',
    `    target_modules=[${used.map((x) => `"${x.m}"`).join(', ')}],`,
    `    r=${base},`,
  ];
  if (overrides.length) lines.push(`    rank_pattern={${overrides.map((x) => `"${x.m}": ${x.r}`).join(', ')}},`);
  lines.push(')');
  const frozen = LAB_MODULES.filter((_, i) => ranks[i] === 0);
  if (frozen.length) lines.push(`# ${frozen.join(', ')}: rank 0, left frozen`);
  return lines.join('\n');
}
