// authored-by: Gemini Antigravity
/**
 * 1D Generative Adversarial Network (GAN) engine on a 3-mode Gaussian mixture:
 * - Generator G and Discriminator D built from src/lib/mlp.ts + src/lib/optimizers.ts
 * - Exact backprop through D(G(z)) for both original saturating minimax loss log(1 - D(G(z)))
 *   and Goodfellow's non-saturating loss -log D(G(z)) (verified against central finite differences)
 * - Mode coverage, Wasserstein-1 (1D Earth Mover's) distance, vanishing-gradient probe,
 *   and 5-seed lab grid search for "Keep the Duel Balanced".
 */
import { MLP } from '../../lib/mlp';
import { createOptimizer, type Optimizer } from '../../lib/optimizers';
import { createRng, type Rng } from '../../lib/rng';

export const MODE_CENTRES = [-2.4, 0.0, 2.4] as const;
export const MODE_STD = 0.28;
export const X_MIN = -3.8;
export const X_MAX = 3.8;
export const HIST_BINS = 38;

export type GanObjective = 'non-saturating' | 'saturating';

export type GanConfig = {
  seed: number;
  noiseDim: 1 | 2;
  lrD: number;
  lrG: number;
  dSteps: number;
  batchSize: number;
  objective: GanObjective;
  /** Optional pre-training steps for D before the duel begins (demonstrates vanishing gradient). */
  pretrainDSteps?: number;
  /** Optional initial output bias for G (e.g. 1.2 to place the untrained forger in the valley). */
  initBias?: number;
  /** Optimizer for G (default 'adam'; 'sgd' exposes raw vanishing gradient without Adam rescaling). */
  optGName?: 'adam' | 'sgd';
};

export type GanMetrics = {
  step: number;
  totalUpdates: number;
  dLoss: number;
  gLoss: number;
  gGradNorm: number;
  meanDReal: number;
  meanDFake: number;
  /** Number of the 3 real modes covered (each mode gets >= 12% of generated mass within ±0.85 of centre). */
  modesCovered: number;
  /** Share of generated samples falling inside each mode window [c - 0.85, c + 0.85]. */
  modeShares: [number, number, number];
  /** Share of generated samples outside all 3 mode windows ("junk / between modes"). */
  offModeShare: number;
  /** 1D Wasserstein-1 distance between sorted real and generated evaluation samples. */
  w1: number;
};

export type GanSnapshot = {
  metrics: GanMetrics;
  realHist: number[];
  fakeHist: number[];
  dCurve: [number, number][];
};

/** Sample n points from the 3-mode Gaussian mixture. */
export function sampleReal(n: number, rng: Rng): number[] {
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const c = MODE_CENTRES[i % 3];
    out[i] = c + rng.normal(0, MODE_STD);
  }
  return out;
}

/** Sample n noise vectors of dimension noiseDim from N(0, I). */
export function sampleNoise(n: number, noiseDim: number, rng: Rng): number[][] {
  const out: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const v = new Array<number>(noiseDim);
    for (let k = 0; k < noiseDim; k++) v[k] = rng.normal(0, 1);
    out[i] = v;
  }
  return out;
}

/**
 * Exact generator loss and backprop through frozen D into G.gW / G.gb.
 * Returns { gLoss, gGradNorm, meanDFake }.
 */
export function backpropGenerator(
  G: MLP,
  D: MLP,
  Z: number[][],
  objective: GanObjective,
): { gLoss: number; gGradNorm: number; meanDFake: number } {
  for (const L of G.gW) for (const r of L) r.fill(0);
  for (const r of G.gb) r.fill(0);

  const n = Z.length;
  const lastG = G.W.length - 1;
  const lastD = D.W.length - 1;
  let totalLoss = 0;
  let totalP = 0;

  for (let s = 0; s < n; s++) {
    const fg = G.forward(Z[s]);
    const fakeX = fg.a[lastG + 1];
    const fd = D.forward(fakeX);
    const pRaw = fd.a[lastD + 1][0];
    const p = Math.min(Math.max(pRaw, 1e-12), 1 - 1e-12);
    totalP += pRaw;

    // Output layer of D is sigmoid: p = sigma(z_out), dp/dz_out = p(1 - p)
    let dOut: number;
    if (objective === 'non-saturating') {
      totalLoss += -Math.log(p);
      // d(-log sigma(z))/dz = -(1 - p) = p - 1
      dOut = pRaw - 1;
    } else {
      totalLoss += Math.log(1 - p);
      // d(log(1 - sigma(z)))/dz = -p
      dOut = -pRaw;
    }

    // Backprop through frozen D down to its 1D input fakeX
    let delta: number[] = [dOut];
    for (let l = lastD; l >= 0; l--) {
      const W = D.W[l];
      const prevLen = D.sizes[l];
      const nd = new Array<number>(prevLen).fill(0);
      for (let j = 0; j < W.length; j++) {
        for (let i = 0; i < prevLen; i++) nd[i] += W[j][i] * delta[j];
      }
      if (l > 0) {
        const df = D.acts[l - 1].df;
        for (let i = 0; i < prevLen; i++) nd[i] *= df(fd.z[l - 1][i]);
      }
      delta = nd;
    }

    // Now delta is dL/d(fakeX); multiply by G's output activation derivative
    const dfOutG = G.acts[lastG].df;
    for (let j = 0; j < delta.length; j++) delta[j] *= dfOutG(fg.z[lastG][j]);

    // Backprop through G and accumulate into G.gW / G.gb
    for (let l = lastG; l >= 0; l--) {
      const W = G.W[l];
      const prev = fg.a[l];
      const gW = G.gW[l];
      const gb = G.gb[l];
      for (let j = 0; j < W.length; j++) {
        gb[j] += delta[j];
        const gr = gW[j];
        for (let i = 0; i < prev.length; i++) gr[i] += delta[j] * prev[i];
      }
      if (l > 0) {
        const nd = new Array<number>(prev.length).fill(0);
        for (let j = 0; j < W.length; j++) {
          for (let i = 0; i < prev.length; i++) nd[i] += W[j][i] * delta[j];
        }
        const df = G.acts[l - 1].df;
        for (let i = 0; i < prev.length; i++) nd[i] *= df(fg.z[l - 1][i]);
        delta = nd;
      }
    }
  }

  let gSq = 0;
  for (const L of G.gW) {
    for (const r of L) {
      for (let i = 0; i < r.length; i++) {
        r[i] /= n;
        gSq += r[i] * r[i];
      }
    }
  }
  for (const r of G.gb) {
    for (let j = 0; j < r.length; j++) {
      r[j] /= n;
      gSq += r[j] * r[j];
    }
  }

  return { gLoss: totalLoss / n, gGradNorm: Math.sqrt(gSq), meanDFake: totalP / n };
}

/** Histogram densities over [X_MIN, X_MAX] into HIST_BINS bins. */
export function histogram1D(xs: number[]): number[] {
  const bins = new Array<number>(HIST_BINS).fill(0);
  const span = X_MAX - X_MIN;
  for (const v of xs) {
    const idx = Math.floor(((v - X_MIN) / span) * HIST_BINS);
    if (idx >= 0 && idx < HIST_BINS) bins[idx]++;
  }
  const n = Math.max(1, xs.length);
  return bins.map((c) => c / n);
}

/** 1D Wasserstein-1 distance between two equal-size sample arrays. */
export function wasserstein1D(a: number[], b: number[]): number {
  const sa = a.slice().sort((u, v) => u - v);
  const sb = b.slice().sort((u, v) => u - v);
  let sum = 0;
  for (let i = 0; i < sa.length; i++) sum += Math.abs(sa[i] - sb[i]);
  return sum / sa.length;
}

/** Measure how many of the 3 real modes are populated by generated samples. */
export function evaluateModes(fakes: number[]): {
  modesCovered: number;
  modeShares: [number, number, number];
  offModeShare: number;
} {
  const counts: [number, number, number] = [0, 0, 0];
  let off = 0;
  const radius = 0.85;
  for (const x of fakes) {
    let matched = false;
    for (let m = 0; m < 3; m++) {
      if (Math.abs(x - MODE_CENTRES[m]) <= radius) {
        counts[m]++;
        matched = true;
        break;
      }
    }
    if (!matched) off++;
  }
  const n = Math.max(1, fakes.length);
  const shares: [number, number, number] = [counts[0] / n, counts[1] / n, counts[2] / n];
  const modesCovered = shares.filter((s) => s >= 0.12).length;
  return { modesCovered, modeShares: shares, offModeShare: off / n };
}

/** Fixed evaluation pool so metrics are smooth and comparable across steps & configs. */
const EVAL_REAL: number[] = sampleReal(360, createRng(7777));
const EVAL_REAL_HIST: number[] = histogram1D(EVAL_REAL);

export class GanRunner {
  readonly cfg: GanConfig;
  readonly G: MLP;
  readonly D: MLP;
  private optG: Optimizer;
  private optD: Optimizer;
  private rng: Rng;
  private evalZ: number[][];
  stepCount = 0;
  totalUpdates = 0;
  lastDLoss = 0;
  lastGLoss = 0;
  lastGGradNorm = 0;
  lastMeanDReal = 0.5;
  lastMeanDFake = 0.5;

  constructor(cfg: GanConfig) {
    this.cfg = cfg;
    const rngG = createRng(cfg.seed * 1009 + 17);
    const rngD = createRng(cfg.seed * 2029 + 31);
    this.rng = createRng(cfg.seed * 3037 + 53);
    this.G = new MLP([cfg.noiseDim, 14, 14, 1], ['tanh', 'tanh', 'linear'], rngG, 'xavier');
    // Start G as a tight, unformed blob around initBias (default 0) so training must genuinely discover the 3 modes
    const lastG = this.G.W.length - 1;
    for (let i = 0; i < this.G.W[lastG][0].length; i++) this.G.W[lastG][0][i] *= 0.25;
    if (cfg.initBias) this.G.b[lastG][0] = cfg.initBias;

    this.D = new MLP([1, 14, 14, 1], ['tanh', 'tanh', 'sigmoid'], rngD, 'xavier');
    this.optG =
      cfg.optGName === 'sgd'
        ? createOptimizer({ name: 'sgd', lr: cfg.lrG })
        : createOptimizer({ name: 'adam', lr: cfg.lrG, beta1: 0.5, beta2: 0.999 });
    this.optD = createOptimizer({ name: 'adam', lr: cfg.lrD, beta1: 0.5, beta2: 0.999 });
    this.evalZ = sampleNoise(360, cfg.noiseDim, createRng(9991 + cfg.noiseDim * 13));

    if (cfg.pretrainDSteps && cfg.pretrainDSteps > 0) {
      for (let i = 0; i < cfg.pretrainDSteps; i++) this.trainDStep();
    }
    this.measureLossesWithoutStep();
  }

  private trainDStep() {
    const B = this.cfg.batchSize;
    const reals = sampleReal(B, this.rng);
    const Z = sampleNoise(B, this.cfg.noiseDim, this.rng);
    const X: number[][] = [];
    const Y: number[][] = [];
    for (let i = 0; i < B; i++) {
      X.push([reals[i]]);
      Y.push([1]);
    }
    for (let i = 0; i < B; i++) {
      X.push(this.G.predict(Z[i]));
      Y.push([0]);
    }
    this.lastDLoss = this.D.backprop(X, Y, 'bce');
    this.optD.step(this.D.params(), this.D.grads());
    this.totalUpdates++;
  }

  private measureLossesWithoutStep() {
    const B = this.cfg.batchSize;
    const probeRng = createRng(this.cfg.seed * 4099 + this.stepCount + 1);
    const reals = sampleReal(B, probeRng);
    let rSum = 0;
    for (const r of reals) rSum += this.D.predict([r])[0];
    this.lastMeanDReal = rSum / B;
    const Z = sampleNoise(B, this.cfg.noiseDim, probeRng);
    const gInfo = backpropGenerator(this.G, this.D, Z, this.cfg.objective);
    this.lastGLoss = gInfo.gLoss;
    this.lastGGradNorm = gInfo.gGradNorm;
    this.lastMeanDFake = gInfo.meanDFake;
  }

  /** Run one alternating GAN iteration (dSteps of D, then 1 step of G). */
  step(count = 1) {
    for (let c = 0; c < count; c++) {
      for (let k = 0; k < this.cfg.dSteps; k++) this.trainDStep();

      const Z = sampleNoise(this.cfg.batchSize, this.cfg.noiseDim, this.rng);
      const gInfo = backpropGenerator(this.G, this.D, Z, this.cfg.objective);
      this.optG.step(this.G.params(), this.G.grads());
      this.totalUpdates++;
      this.stepCount++;

      this.lastGLoss = gInfo.gLoss;
      this.lastGGradNorm = gInfo.gGradNorm;
      this.lastMeanDFake = gInfo.meanDFake;
    }
    let s = 0;
    for (let i = 0; i < 60; i++) s += this.D.predict([EVAL_REAL[i * 6]])[0];
    this.lastMeanDReal = s / 60;
  }

  snapshot(): GanSnapshot {
    const fakes = this.evalZ.map((z) => this.G.predict(z)[0]);
    const { modesCovered, modeShares, offModeShare } = evaluateModes(fakes);
    const w1 = wasserstein1D(EVAL_REAL, fakes);
    const dCurve: [number, number][] = [];
    for (let i = 0; i <= 76; i++) {
      const xv = X_MIN + ((X_MAX - X_MIN) * i) / 76;
      dCurve.push([xv, this.D.predict([xv])[0]]);
    }
    return {
      metrics: {
        step: this.stepCount,
        totalUpdates: this.totalUpdates,
        dLoss: this.lastDLoss,
        gLoss: this.lastGLoss,
        gGradNorm: this.lastGGradNorm,
        meanDReal: this.lastMeanDReal,
        meanDFake: this.lastMeanDFake,
        modesCovered,
        modeShares,
        offModeShare,
        w1,
      },
      realHist: EVAL_REAL_HIST,
      fakeHist: histogram1D(fakes),
      dCurve,
    };
  }
}

// ------------------------------------------------------------------ Presets & Lab grid

export type GanPresetId = 'balanced' | 'vanishing' | 'collapse';

export const GAN_PRESETS: Record<GanPresetId, { label: string; summary: string; config: GanConfig }> = {
  balanced: {
    label: 'Balanced duel',
    summary: 'Non-saturating loss, D faster than G (lrD = 0.015, lrG = 0.0075), 2D noise, 1 D-step per G-step.',
    config: { seed: 1, noiseDim: 2, lrD: 0.015, lrG: 0.0075, dSteps: 1, batchSize: 36, objective: 'non-saturating' },
  },
  vanishing: {
    label: 'Detective too strong (vanishing)',
    summary: 'Original saturating loss log(1 − D) with D pre-trained for 120 steps while G starts in the valley (x = 1.2, SGD) — D(fake) ≈ 0 kills G’s gradient.',
    config: { seed: 1, noiseDim: 2, lrD: 0.03, lrG: 0.05, dSteps: 2, batchSize: 36, objective: 'saturating', pretrainDSteps: 120, initBias: 1.2, optGName: 'sgd' },
  },
  collapse: {
    label: 'Mode collapse',
    summary: 'Over-trained discriminator (4 D-steps, slow 1D generator, seed 2) traps the forger on a single peak (1/3 modes) while the other two stay empty.',
    config: { seed: 2, noiseDim: 1, lrD: 0.015, lrG: 0.00375, dSteps: 4, batchSize: 36, objective: 'non-saturating' },
  },
};

export const LAB_SEEDS = [1, 2, 3, 4, 5] as const;
export const LAB_MAX_STEPS = 240;
export const LAB_CHECK_INTERVAL = 20;

export const LAB_LR_RATIOS = [0.25, 0.5, 1, 2, 4] as const; // lrG / lrD, with base lrD = 0.015
export const LAB_D_STEPS = [1, 2, 4] as const;
export const LAB_NOISE_DIMS = [1, 2] as const;
export const LAB_BASE_LR_D = 0.015;

export type LabSetting = {
  lrRatio: (typeof LAB_LR_RATIOS)[number];
  dSteps: (typeof LAB_D_STEPS)[number];
  noiseDim: (typeof LAB_NOISE_DIMS)[number];
};

export type LabEvalResult = {
  setting: LabSetting;
  meanModes: number;
  minModes: number;
  allThreeSeeds: number;
  meanW1: number;
  meanOffMode: number;
  /** First update count where meanModes >= 2.8 and meanW1 <= 0.45 across the 5 seeds, or null if never reached. */
  updatesToCover3: number | null;
  finalUpdates: number;
  perSeed: { seed: number; modes: number; w1: number; offMode: number }[];
};

export function evaluateLabSetting(setting: LabSetting): LabEvalResult {
  const lrD = LAB_BASE_LR_D;
  const lrG = lrD * setting.lrRatio;
  const runners = LAB_SEEDS.map(
    (seed) =>
      new GanRunner({
        seed,
        noiseDim: setting.noiseDim,
        lrD,
        lrG,
        dSteps: setting.dSteps,
        batchSize: 36,
        objective: 'non-saturating',
      }),
  );

  let updatesToCover3: number | null = null;
  for (let s = LAB_CHECK_INTERVAL; s <= LAB_MAX_STEPS; s += LAB_CHECK_INTERVAL) {
    let sumModes = 0;
    let sumW1 = 0;
    let sumOff = 0;
    for (const r of runners) {
      r.step(LAB_CHECK_INTERVAL);
      const snap = r.snapshot();
      sumModes += snap.metrics.modesCovered;
      sumW1 += snap.metrics.w1;
      sumOff += snap.metrics.offModeShare;
    }
    if (
      updatesToCover3 === null &&
      sumModes / LAB_SEEDS.length >= 2.8 &&
      sumW1 / LAB_SEEDS.length <= 0.45 &&
      sumOff / LAB_SEEDS.length <= 0.25
    ) {
      updatesToCover3 = runners[0].totalUpdates;
    }
  }

  const perSeed = runners.map((r) => {
    const m = r.snapshot().metrics;
    return { seed: r.cfg.seed, modes: m.modesCovered, w1: m.w1, offMode: m.offModeShare };
  });
  const meanModes = perSeed.reduce((a, b) => a + b.modes, 0) / perSeed.length;
  const minModes = Math.min(...perSeed.map((p) => p.modes));
  const allThreeSeeds = perSeed.filter((p) => p.modes === 3).length;
  const meanW1 = perSeed.reduce((a, b) => a + b.w1, 0) / perSeed.length;
  const meanOffMode = perSeed.reduce((a, b) => a + b.offMode, 0) / perSeed.length;

  return {
    setting,
    meanModes,
    minModes,
    allThreeSeeds,
    meanW1,
    meanOffMode,
    updatesToCover3,
    finalUpdates: runners[0].totalUpdates,
    perSeed,
  };
}

export const GAN_LAB_BRIEFS = [
  { id: 'cover2', tab: '1 · Cover 2+ modes' },
  { id: 'cover3', tab: '2 · Cover all 3' },
  { id: 'fast', tab: '3 · Fewest updates' },
] as const;

export type GanBriefId = (typeof GAN_LAB_BRIEFS)[number]['id'];

export function ganLabHint(
  brief: GanBriefId,
  state: { meanModes: number; meanW1: number; updatesToCover3: number | null; meanOffMode: number },
): string {
  if (brief === 'cover2') {
    if (state.meanModes < 2) {
      return 'The generator is collapsing onto a single peak across most seeds. Watch the discriminator curve D(x): when one network runs far ahead of the other, the forger hides in one spot rather than spreading out.';
    }
    return 'Your setting already averages at least two covered modes across the five seeds.';
  }
  if (brief === 'cover3') {
    if (state.meanModes < 2.8) {
      return `Currently covering ${state.meanModes.toFixed(1)} of 3 modes on average. Notice whether a single 1D noise knob struggles to jump cleanly across empty gaps between three separated peaks, or whether the generator and discriminator are out of balance.`;
    }
    if (state.meanW1 > 0.45) {
      return `All modes get visited, but the generated histogram still drifts (W₁ = ${state.meanW1.toFixed(2)}). Check whether too many samples land in the empty valleys between peaks.`;
    }
    return 'All three modes are cleanly covered across the five seeds.';
  }
  if (state.updatesToCover3 === null) {
    return 'First reach the Brief 2 quality bar (≥ 2.8 average modes and W₁ ≤ 0.45 across the five seeds) before trimming total updates.';
  }
  return `You reached the bar in ${state.updatesToCover3} total network updates (each outer step costs D-steps + 1 updates). Watch how extra discriminator steps per iteration multiply the update bill.`;
}
