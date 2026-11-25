// authored-by: Claude Opus 5.5
/**
 * Lab: The Face Printer. The learner designs a generator — code size, regulariser (plain
 * autoencoder or VAE at some β) and hidden width — and it is trained for real and checked against a
 * brief's spec with three seeds. Every design in the space is tried by `lab.run.ts`; results are recorded in
 * `labResults.ts` and spot-checked by `lab.test.ts`.
 */
import { createRng, type Rng } from '../../lib/rng';
import { DATA_SEED, PIX, REAL_TOL, VAE, makeDataset, nearestFace, rmse, trainer, type Dataset, type Mode } from './vae';

export const LATENTS = [1, 2, 3] as const;
export const HIDDENS = [8, 16, 32] as const;
/** 0 means "plain autoencoder" (no noise, no pull towards the bell). */
export const BETAS = [0, 0.1, 0.3, 1, 3, 10] as const;

export type Design = { latent: number; hidden: number; beta: number };
export type BriefId = 'printer' | 'compressor';

export type Brief = {
  id: BriefId;
  title: string;
  story: string;
  /** held-out faces must come back within this pixel RMSE on average */
  maxRecon: number;
  /** share of printed faces that must be real-looking (0 = the printer isn't used) */
  minReal: number;
  /** how many of the 16 (smile × eyes) cells the printed faces must reach */
  minCells: number;
};

export const BRIEFS: Brief[] = [
  {
    id: 'printer',
    title: 'Brief 1 · The face printer',
    story:
      'A novelty printer will feed your decoder random numbers drawn from the standard bell (one per code number, mean 0, SD 1) and print whatever comes out. Faces it has never seen must also survive a trip through your encoder and decoder.',
    maxRecon: 0.04,
    minReal: 0.95,
    minCells: 14,
  },
  {
    id: 'compressor',
    title: 'Brief 2 · The compressor',
    story:
      'No printer this time. You are building a compressor: store each face as its code and rebuild it on demand. Faces it has never seen must come back sharp.',
    maxRecon: 0.02,
    minReal: 0,
    minCells: 0,
  },
];

/** A design only counts if it meets the spec when trained with each of these seeds. */
export const LAB_SEEDS = [1, 2, 3] as const;
export const LAB_STEPS = 3000;
export const PRINT_SEED = 2024;
export const N_PRINT = 200;
export const TEST_SEED = 12345;
export const N_TEST = 60;

export const modeOf = (d: Design): Mode => (d.beta === 0 ? 'ae' : 'vae');

/** Parameters the generator really uses (a plain autoencoder has no log-variance head). */
export function paramCount(d: Design) {
  const heads = d.beta === 0 ? d.latent : 2 * d.latent;
  return PIX * d.hidden + d.hidden + heads * d.hidden + heads + d.latent * d.hidden + d.hidden + d.hidden * PIX + PIX;
}

export const TRAIN_FACES = 400;
let dataCache: Dataset | null = null;
export const labData = (): Dataset => (dataCache ??= makeDataset(TRAIN_FACES, DATA_SEED));
let testCache: Dataset | null = null;
export const labTest = () => (testCache ??= makeDataset(N_TEST, TEST_SEED));

export function labTrainer(d: Design, seed: number) {
  return trainer({ mode: modeOf(d), beta: d.beta, latent: d.latent, H: d.hidden, steps: LAB_STEPS, batch: 32, lr: 0.003, seed }, labData());
}

export type PrintResult = {
  recon: number;
  realShare: number;
  cells: number;
  /** the printed codes and what came out, for display */
  prints: { img: Float64Array; real: boolean; dist: number }[];
};

/** Run the printer: standard-bell codes in, faces out, each judged against the atlas of real faces. */
export function printAndJudge(net: VAE, nPrint = N_PRINT, seed = PRINT_SEED): PrintResult {
  const test = labTest();
  let recon = 0;
  for (const x of test.imgs) recon += rmse(x, net.reconstruct(x)) / test.imgs.length;
  const rng: Rng = createRng(seed);
  const cells = new Set<number>();
  const prints: PrintResult['prints'] = [];
  let real = 0;
  for (let i = 0; i < nPrint; i++) {
    const z = Array.from({ length: net.L }, () => rng.normal());
    const img = net.decode(z).out;
    const nf = nearestFace(img);
    const ok = nf.dist <= REAL_TOL;
    if (ok) {
      real++;
      cells.add(Math.min(3, Math.floor(((nf.knobs.smile + 1) / 2) * 4)) * 4 + Math.min(3, Math.floor(nf.knobs.eyes * 4)));
    }
    prints.push({ img, real: ok, dist: nf.dist });
  }
  return { recon, realShare: real / nPrint, cells: cells.size, prints };
}

export type Measured = { recon: number; realShare: number; cells: number };
export type Verdict = { pass: boolean; recon: boolean; real: boolean; cells: boolean };
export function verdict(b: Brief, r: Measured): Verdict {
  const recon = r.recon <= b.maxRecon;
  const real = r.realShare >= b.minReal;
  const cells = r.cells >= b.minCells;
  return { pass: recon && real && cells, recon, real, cells };
}

/** The lab's rule: a design passes only if every seed's run passes. */
export const verdictAll = (b: Brief, runs: Measured[]) => runs.every((r) => verdict(b, r).pass);

export const allDesigns = (): Design[] =>
  LATENTS.flatMap((latent) => HIDDENS.flatMap((hidden) => BETAS.map((beta) => ({ latent, hidden, beta }))));

export const designKey = (d: Design) => `${d.latent}-${d.hidden}-${d.beta}`;

export function facePrinterHint(brief: BriefId, d: Design, runs?: Measured[] | null): string {
  if (runs && runs.length === 3) {
    const b = BRIEFS.find((x) => x.id === brief)!;
    const vs = runs.map((r) => verdict(b, r));
    if (vs.every((v) => v.pass)) {
      return `All three seeds meet the spec with ${paramCount(d).toLocaleString('en-GB')} parameters. Optional goal: see if a smaller network still clears every seed.`;
    }
    if (d.latent === 1) {
      return 'Every face varies along two independent knobs (smile and eyes). A single code number has to fold a 2D sheet onto a 1D line, so held-out reconstructions cannot get sharp.';
    }
    if (brief === 'printer' && d.beta === 0) {
      return 'A plain autoencoder never places its codes on the standard bell, so random standard-bell codes miss where the decoder learned to draw.';
    }
    if (d.beta >= 3) {
      return 'The pull towards the bell is stronger than the pixel detail: codes collapse towards the prior and faces come back blurry or identical.';
    }
    if (vs.some((v) => v.pass)) {
      return `Passing on ${vs.filter((v) => v.pass).length} of 3 seeds is a lucky run, not a reliable design. Inspect the failing seed's column to see which metric slipped.`;
    }
    if (brief === 'printer' && vs.some((v) => !v.real)) {
      return 'Some standard-bell codes still land outside the region of real faces — the codes need a slightly stronger pull towards the standard bell.';
    }
  }
  if (brief === 'printer') {
    return 'Count the knobs that make one face different from another: the code needs at least that many numbers. Then remember who supplies the codes: the printer draws from the standard bell, so training must place faces there without blurring them out.';
  }
  return 'Nobody will ever feed this decoder random codes, so filling the standard bell is no longer the goal. Watch what a heavy pull costs reconstruction error — and whether a gentle pull still stabilises training across seeds.';
}

