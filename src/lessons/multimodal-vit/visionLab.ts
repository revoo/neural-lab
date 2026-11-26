// authored-by: Claude Opus 5.5
/**
 * The vision side of Lab: Wire the Multimodal Model, and the CNN-vs-ViT race.
 * Every run sees the same number of training examples (BUDGET presentations), whatever the data size,
 * so bigger datasets aren't also given more training. Settings are scored on seeds 1–3.
 * The recorded results come from `vision.run.ts` (→ `visionResults.ts`), re-verified in `vit.test.ts`.
 */
import { Cnn, makeSet, trainModel, accuracy, Vit, TRAIN_OPTS, type Model } from './vision';
import { trainClip, zeroShot, type ClipSetting } from './clip';

export const SEEDS = [1, 2, 3] as const;
export const BUDGET = 8192;
export const epochsFor = (n: number) => Math.max(1, Math.round(BUDGET / n));
export const trainSet = (n: number, seed: number) => makeSet(n, 100 + seed);

export type VitSetting = { P: 1 | 2 | 4; d: 4 | 8 | 16; pos: boolean; augment: boolean };
export type Arch = { kind: 'cnn' } | ({ kind: 'vit' } & VitSetting);

export function build(a: Arch, seed: number): Model {
  return a.kind === 'cnn' ? new Cnn(4, seed) : new Vit(a.P, a.d, a.pos, seed);
}

export function runOne(a: Arch, n: number, seed: number) {
  const m = build(a, seed);
  trainModel(m, trainSet(n, seed), seed, { ...TRAIN_OPTS, epochs: epochsFor(n), augment: a.kind === 'vit' && a.augment });
  return accuracy(m);
}

export const meanAcc = (a: Arch, n: number) => SEEDS.reduce((s, seed) => s + runOne(a, n, seed), 0) / SEEDS.length;

/** Brief 1: the team must ship the transformer, trained on 32 images. */
export const RESCUE_N = 32;
export const RESCUE_BAR = 0.8;
/** Brief 2: attention multiply-adds per image, N² · d, and the budget. */
export const PATCH_N = 256;
export const attnCost = (P: number, d: number) => ((8 / P) ** 2) ** 2 * d;
export const ATTN_CAP = 4096;
export const PATCH_BAR = 0.8;

export const RESCUE_GRID: VitSetting[] = [2, 4].flatMap((P) => [4, 8].flatMap((d) => [false, true].flatMap((pos) => [false, true].map((augment) => ({ P, d, pos, augment }) as VitSetting))));
export const PATCH_GRID: VitSetting[] = [2, 4].flatMap((P) => [4, 8, 16].map((d) => ({ P, d, pos: true, augment: false }) as VitSetting));
export const RACE_SIZES = [16, 64, 256, 1024];
export const RACE_VIT: Arch = { kind: 'vit', P: 2, d: 8, pos: true, augment: false };

// ------------------------------------------------------------------ the CLIP brief
export const CLIP_BAR = 0.9;
export const CLIP_GRID: ClipSetting[] = [0.05, 0.1, 0.3, 1].flatMap((tau) => [2, 4, 8].flatMap((k) => [4, 8, 16].map((batch) => ({ tau, k, batch }))));
export const clipMean = (s: ClipSetting) => SEEDS.reduce((a, seed) => a + zeroShot(trainClip(s, seed)), 0) / SEEDS.length;
