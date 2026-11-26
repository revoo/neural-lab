// authored-by: Claude Opus 5.5
/**
 * Records the CNN-vs-ViT race and the vision briefs of the lab (seeds 1–3 each).
 * Run in parallel shards:  node --experimental-strip-types --import ./scripts/ts-hooks.mjs src/lessons/multimodal-vit/vision.run.ts <shard> <shards>
 * and paste the printed lines (sorted by job) into visionResults.ts.
 */
import { PATCH_GRID, PATCH_N, RACE_SIZES, RACE_VIT, RESCUE_GRID, RESCUE_N, runOne, SEEDS, type Arch } from './visionLab';

declare const process: { argv: string[] };
const shard = Number(process.argv[2] ?? 0);
const shards = Number(process.argv[3] ?? 1);

const jobs: { group: string; arch: Arch; n: number }[] = [
  ...RACE_SIZES.flatMap((n) => [
    { group: 'race', arch: { kind: 'cnn' } as Arch, n },
    { group: 'race', arch: RACE_VIT, n },
  ]),
  ...RESCUE_GRID.map((s) => ({ group: 'rescue', arch: { kind: 'vit', ...s } as Arch, n: RESCUE_N })),
  ...PATCH_GRID.map((s) => ({ group: 'patch', arch: { kind: 'vit', ...s } as Arch, n: PATCH_N })),
];
jobs.forEach((j, i) => {
  if (i % shards !== shard) return;
  const accs = SEEDS.map((s) => +runOne(j.arch, j.n, s).toFixed(4));
  console.log(JSON.stringify({ i, group: j.group, arch: j.arch, n: j.n, accs }) + ',');
});
