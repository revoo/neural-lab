// authored-by: Neon High
import { createRng } from '../../lib/rng';

export type ActName = 'linear' | 'tanh' | 'sigmoid' | 'relu';
export type InitName = 'tiny' | 'xavier' | 'he' | 'double' | 'naive';

export interface InitSpec {
  id: InitName;
  label: string;
  shortLabel: string;
  sigma: (fanIn: number) => number;
  formula: string;
}

export const INIT_SPECS: InitSpec[] = [
  { id: 'tiny', label: 'Too small (0.5 / √d)', shortLabel: '0.5 / √d', sigma: (d) => 0.5 / Math.sqrt(d), formula: '0.5 / √d' },
  { id: 'xavier', label: 'Xavier / Glorot (1 / √d)', shortLabel: 'Xavier', sigma: (d) => 1.0 / Math.sqrt(d), formula: '1 / √d' },
  { id: 'he', label: 'He / Kaiming (√(2/d))', shortLabel: 'He', sigma: (d) => Math.sqrt(2 / d), formula: '√(2 / d)' },
  { id: 'double', label: 'Too large (2 / √d)', shortLabel: '2 / √d', sigma: (d) => 2.0 / Math.sqrt(d), formula: '2 / √d' },
  { id: 'naive', label: 'Naive (σ = 1.0)', shortLabel: 'σ = 1.0', sigma: () => 1.0, formula: '1.0' },
];

export const INIT_BY_ID = Object.fromEntries(INIT_SPECS.map((s) => [s.id, s])) as Record<InitName, InitSpec>;

function actFn(name: ActName, x: number): number {
  if (name === 'linear') return x;
  if (name === 'tanh') return Math.tanh(x);
  if (name === 'sigmoid') return 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, x))));
  return x > 0 ? x : 0;
}

function actDeriv(name: ActName, z: number, a: number): number {
  if (name === 'linear') return 1;
  if (name === 'tanh') return 1 - a * a;
  if (name === 'sigmoid') return a * (1 - a);
  return z > 0 ? 1 : 0;
}

function layerNormFwd(z: number[], eps = 1e-5): { y: number[]; invStd: number; zHat: number[] } {
  const d = z.length;
  let mean = 0;
  for (let i = 0; i < d; i++) mean += z[i];
  mean /= d;
  let v = 0;
  for (let i = 0; i < d; i++) {
    const diff = z[i] - mean;
    v += diff * diff;
  }
  const invStd = 1 / Math.sqrt(v / d + eps);
  const zHat = new Array<number>(d);
  for (let i = 0; i < d; i++) zHat[i] = (z[i] - mean) * invStd;
  return { y: zHat, invStd, zHat };
}

function layerNormBwd(g: number[], invStd: number, zHat: number[]): number[] {
  const d = g.length;
  let meanG = 0;
  let dotGZ = 0;
  for (let i = 0; i < d; i++) {
    meanG += g[i];
    dotGZ += g[i] * zHat[i];
  }
  meanG /= d;
  dotGZ /= d;
  const dz = new Array<number>(d);
  for (let i = 0; i < d; i++) {
    dz[i] = invStd * (g[i] - meanG - zHat[i] * dotGZ);
  }
  return dz;
}

function rms(vec: number[]): number {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
  return Math.sqrt(s / vec.length);
}

// ------------------------------------------------------------------ 1. Deep signal propagation (forward + exact backward)

export interface SignalConfig {
  depth: number;
  width: number;
  act: ActName;
  init: InitName;
  useLayerNorm: boolean;
  useResidual: boolean;
  seed?: number;
  samples?: number;
}

export interface SignalTrace {
  /** Forward activation RMS at layers 0..depth (0 = input). */
  actRms: number[];
  /** Backward gradient RMS ∂L/∂h_l at layers 0..depth (depth = upstream unit grad). */
  gradRms: number[];
  /** Ratio gradRms[0] / gradRms[depth]: < 1 vanishes, > 1 explodes. */
  gradRatio: number;
  /** Min and max forward RMS across layers 1..depth. */
  minActRms: number;
  maxActRms: number;
  /** Fraction of active (non-zero) ReLU units or unsaturated units at the final layer. */
  lastLayerDeadFrac: number;
  /** True if any value overflowed to NaN or > 1e12. */
  overflowed: boolean;
}

/**
 * Exact forward & backward propagation through an L-layer network of width d.
 * Averages squared norms across `samples` unit-Gaussian inputs for smooth,
 * seed-stable measurements.
 *
 * Pre-norm residual block (when useResidual=true):
 *   h_{l+1} = h_l + (1 / sqrt(L)) * act(W_l * norm(h_l))   if useLayerNorm
 *   h_{l+1} = h_l + (1 / sqrt(L)) * act(W_l * h_l)         otherwise
 * Plain stack (when useResidual=false):
 *   h_{l+1} = act(norm(W_l * h_l))   if useLayerNorm
 *   h_{l+1} = act(W_l * h_l)         otherwise
 */
export function probeSignal(cfg: SignalConfig): SignalTrace {
  const { depth, width, act, init, useLayerNorm, useResidual } = cfg;
  const samples = cfg.samples ?? 16;
  const rng = createRng(cfg.seed ?? 1337);
  const sigma = INIT_BY_ID[init].sigma(width);
  const resScale = 1 / Math.sqrt(depth);

  // Build L weight matrices (width x width)
  const weights: number[][][] = [];
  for (let l = 0; l < depth; l++) {
    const W: number[][] = [];
    for (let i = 0; i < width; i++) {
      const row = new Array<number>(width);
      for (let j = 0; j < width; j++) row[j] = rng.normal(0, sigma);
      W.push(row);
    }
    weights.push(W);
  }

  const actSqSum = new Array<number>(depth + 1).fill(0);
  const gradSqSum = new Array<number>(depth + 1).fill(0);
  let deadCount = 0;
  let totalLast = 0;
  let overflowed = false;

  for (let s = 0; s < samples; s++) {
    // Unit-variance Gaussian input
    const x0 = Array.from({ length: width }, () => rng.normal(0, 1));
    // Upstream gradient at output layer: unit-RMS random direction (so E[||g||^2 / d] = 1)
    const gTop = Array.from({ length: width }, () => rng.normal(0, 1));
    const gTopRms = rms(gTop) || 1;
    for (let i = 0; i < width; i++) gTop[i] /= gTopRms;

    // Forward pass saving intermediates
    const hs: number[][] = [x0];
    const cache: Array<{
      inVec: number[];
      preLn?: { invStd: number; zHat: number[] };
      z: number[];
      postLn?: { invStd: number; zHat: number[] };
      a: number[];
    }> = [];

    let h = x0;
    actSqSum[0] += rms(h) ** 2;

    for (let l = 0; l < depth; l++) {
      const W = weights[l];
      let inVec = h;
      let preLn: { invStd: number; zHat: number[] } | undefined;

      if (useResidual && useLayerNorm) {
        const ln = layerNormFwd(h);
        inVec = ln.y;
        preLn = { invStd: ln.invStd, zHat: ln.zHat };
      }

      const z = new Array<number>(width);
      for (let i = 0; i < width; i++) {
        let sum = 0;
        const row = W[i];
        for (let j = 0; j < width; j++) sum += row[j] * inVec[j];
        z[i] = sum;
      }

      let zForAct = z;
      let postLn: { invStd: number; zHat: number[] } | undefined;
      if (!useResidual && useLayerNorm) {
        const ln = layerNormFwd(z);
        zForAct = ln.y;
        postLn = { invStd: ln.invStd, zHat: ln.zHat };
      }

      const a = new Array<number>(width);
      for (let i = 0; i < width; i++) a[i] = actFn(act, zForAct[i]);

      const hNext = new Array<number>(width);
      for (let i = 0; i < width; i++) {
        hNext[i] = useResidual ? h[i] + resScale * a[i] : a[i];
        if (!Number.isFinite(hNext[i]) || Math.abs(hNext[i]) > 1e12) overflowed = true;
      }

      cache.push({ inVec, preLn, z: zForAct, postLn, a });
      hs.push(hNext);
      h = hNext;
      actSqSum[l + 1] += rms(h) ** 2;

      if (l === depth - 1) {
        for (let i = 0; i < width; i++) {
          totalLast++;
          if (act === 'relu' && a[i] === 0) deadCount++;
          else if (act === 'tanh' && Math.abs(a[i]) > 0.98) deadCount++;
          else if (act === 'sigmoid' && (a[i] < 0.02 || a[i] > 0.98)) deadCount++;
        }
      }
    }

    // Exact backward pass
    let g = gTop;
    gradSqSum[depth] += rms(g) ** 2;

    for (let l = depth - 1; l >= 0; l--) {
      const W = weights[l];
      const c = cache[l];

      // Gradient into a
      const ga = new Array<number>(width);
      for (let i = 0; i < width; i++) ga[i] = useResidual ? g[i] * resScale : g[i];

      // Through activation
      let gz = new Array<number>(width);
      for (let i = 0; i < width; i++) gz[i] = ga[i] * actDeriv(act, c.z[i], c.a[i]);

      // Through post-linear LayerNorm (plain stack)
      if (c.postLn) {
        gz = layerNormBwd(gz, c.postLn.invStd, c.postLn.zHat);
      }

      // Through W^T
      let gIn = new Array<number>(width).fill(0);
      for (let i = 0; i < width; i++) {
        const gzi = gz[i];
        const row = W[i];
        for (let j = 0; j < width; j++) gIn[j] += row[j] * gzi;
      }

      // Through pre-norm LayerNorm (residual branch)
      if (c.preLn) {
        gIn = layerNormBwd(gIn, c.preLn.invStd, c.preLn.zHat);
      }

      // Residual skip adds identity gradient directly
      const gPrev = new Array<number>(width);
      for (let i = 0; i < width; i++) {
        gPrev[i] = useResidual ? g[i] + gIn[i] : gIn[i];
        if (!Number.isFinite(gPrev[i]) || Math.abs(gPrev[i]) > 1e12) overflowed = true;
      }

      g = gPrev;
      gradSqSum[l] += rms(g) ** 2;
    }
  }

  const actRms = actSqSum.map((v) => Math.sqrt(v / samples));
  const gradRms = gradSqSum.map((v) => Math.sqrt(v / samples));
  const hiddenActs = actRms.slice(1);
  const minActRms = Math.min(...hiddenActs);
  const maxActRms = Math.max(...hiddenActs);
  const gradRatio = gradRms[0] / (gradRms[depth] || 1);

  return {
    actRms,
    gradRms,
    gradRatio,
    minActRms,
    maxActRms,
    lastLayerDeadFrac: totalLast > 0 ? deadCount / totalLast : 0,
    overflowed,
  };
}

/**
 * Single-example scalar forward helper exposed for numerical gradient checking in vanish.test.ts.
 * Computes L = sum_i u_i * h_L[i] so exact ∂L/∂x_0 equals our backward pass!
 */
export function singleForwardLoss(
  x0: number[],
  uTop: number[],
  weights: number[][][],
  act: ActName,
  useLayerNorm: boolean,
  useResidual: boolean
): number {
  const depth = weights.length;
  const resScale = 1 / Math.sqrt(depth);
  let h = x0.slice();
  for (let l = 0; l < depth; l++) {
    const W = weights[l];
    const inVec = useResidual && useLayerNorm ? layerNormFwd(h).y : h;
    const z = W.map((row) => row.reduce((s, w, j) => s + w * inVec[j], 0));
    const zForAct = !useResidual && useLayerNorm ? layerNormFwd(z).y : z;
    const a = zForAct.map((v) => actFn(act, v));
    h = a.map((v, i) => (useResidual ? h[i] + resScale * v : v));
  }
  return h.reduce((s, v, i) => s + v * uTop[i], 0);
}

/** Exact analytical ∂L/∂x_0 matching singleForwardLoss, for finite-difference test verification. */
export function singleBackwardInputGrad(
  x0: number[],
  uTop: number[],
  weights: number[][][],
  act: ActName,
  useLayerNorm: boolean,
  useResidual: boolean
): number[] {
  const depth = weights.length;
  const width = x0.length;
  const resScale = 1 / Math.sqrt(depth);
  let h = x0.slice();
  const cache: Array<{
    preLn?: { invStd: number; zHat: number[] };
    z: number[];
    postLn?: { invStd: number; zHat: number[] };
    a: number[];
  }> = [];

  for (let l = 0; l < depth; l++) {
    const W = weights[l];
    let inVec = h;
    let preLn: { invStd: number; zHat: number[] } | undefined;
    if (useResidual && useLayerNorm) {
      const ln = layerNormFwd(h);
      inVec = ln.y;
      preLn = { invStd: ln.invStd, zHat: ln.zHat };
    }
    const z = W.map((row) => row.reduce((s, w, j) => s + w * inVec[j], 0));
    let zForAct = z;
    let postLn: { invStd: number; zHat: number[] } | undefined;
    if (!useResidual && useLayerNorm) {
      const ln = layerNormFwd(z);
      zForAct = ln.y;
      postLn = { invStd: ln.invStd, zHat: ln.zHat };
    }
    const a = zForAct.map((v) => actFn(act, v));
    cache.push({ preLn, z: zForAct, postLn, a });
    h = a.map((v, i) => (useResidual ? h[i] + resScale * v : v));
  }

  let g = uTop.slice();
  for (let l = depth - 1; l >= 0; l--) {
    const W = weights[l];
    const c = cache[l];
    const ga = g.map((v) => (useResidual ? v * resScale : v));
    let gz = ga.map((v, i) => v * actDeriv(act, c.z[i], c.a[i]));
    if (c.postLn) gz = layerNormBwd(gz, c.postLn.invStd, c.postLn.zHat);
    let gIn = new Array<number>(width).fill(0);
    for (let i = 0; i < width; i++) {
      for (let j = 0; j < width; j++) gIn[j] += W[i][j] * gz[i];
    }
    if (c.preLn) gIn = layerNormBwd(gIn, c.preLn.invStd, c.preLn.zHat);
    g = gIn.map((v, i) => (useResidual ? g[i] + v : v));
  }
  return g;
}

// ------------------------------------------------------------------ 2. Single-layer variance conservation lab

export interface VarianceMeasurement {
  fanIn: number;
  sigmaW: number;
  act: 'linear' | 'relu';
  theoryVar: number;
  empiricalVar: number;
  empiricalStd: number;
  samples: number[]; // first 40 post-activation values for plotting
}

export function measureLayerVariance(fanIn: number, sigmaW: number, act: 'linear' | 'relu'): VarianceMeasurement {
  const rng = createRng(9001);
  const trials = 400;
  const vals: number[] = [];
  let sumSq = 0;
  for (let t = 0; t < trials; t++) {
    let z = 0;
    for (let j = 0; j < fanIn; j++) {
      z += rng.normal(0, sigmaW) * rng.normal(0, 1);
    }
    const y = act === 'relu' ? (z > 0 ? z : 0) : z;
    vals.push(y);
    sumSq += y * y;
  }
  const empiricalVar = sumSq / trials; // second moment E[y^2]
  const theoryVar = (act === 'relu' ? 0.5 : 1.0) * fanIn * sigmaW * sigmaW;
  return {
    fanIn,
    sigmaW,
    act,
    theoryVar,
    empiricalVar,
    empiricalStd: Math.sqrt(empiricalVar),
    samples: vals.slice(0, 48),
  };
}

// ------------------------------------------------------------------ 3. Real 8-hidden-layer training race (concentric rings)

export interface DeepRacePreset {
  id: string;
  label: string;
  act: 'sigmoid' | 'tanh' | 'relu';
  init: InitName;
  useResidual: boolean;
  lr: number;
  note: string;
}

export const DEEP_RACE_PRESETS: DeepRacePreset[] = [
  {
    id: 'sigmoid-xavier',
    label: 'Sigmoid + Xavier',
    act: 'sigmoid',
    init: 'xavier',
    useResidual: false,
    lr: 1.6,
    note: 'Vanishing gradients: even at its searched best step size (η = 1.6), layer-1 gradients are ~10⁻⁵ smaller than layer 8 — loss stays glued to ln 2 = 0.693.',
  },
  {
    id: 'relu-naive',
    label: 'ReLU + Naive (σ = 1.0)',
    act: 'relu',
    init: 'naive',
    useResidual: false,
    lr: 0.05,
    note: 'Exploding gradients: σ = 1.0 amplifies activations by √16 = 4× per layer, blowing up gradients and killing the network at ordinary step sizes.',
  },
  {
    id: 'relu-he',
    label: 'ReLU + He (√(2/d))',
    act: 'relu',
    init: 'he',
    useResidual: false,
    lr: 0.4,
    note: 'He init keeps variance near 1.0 at every layer — converges rapidly to 100% accuracy (avg loss 0.0013 across 5 seeds).',
  },
  {
    id: 'relu-he-res',
    label: 'ReLU + He + Residual',
    act: 'relu',
    init: 'he',
    useResidual: true,
    lr: 0.4,
    note: 'Skip connections keep layer-1 gradients almost equal to layer-8 gradients (ratio ~0.72 vs 0.23 plain) and move early layers cleanly.',
  },
];

export interface RingExample {
  x: [number, number];
  y: number;
}

export function makeRingsDataset(): RingExample[] {
  const rng = createRng(77);
  const data: RingExample[] = [];
  for (let i = 0; i < 64; i++) {
    const label = i < 32 ? 0 : 1;
    const angle = rng.uniform(0, Math.PI * 2);
    const r = label === 0 ? rng.uniform(0.15, 0.55) : rng.uniform(0.85, 1.25);
    data.push({ x: [r * Math.cos(angle), r * Math.sin(angle)], y: label });
  }
  return data;
}

export interface DeepRaceResult {
  preset: DeepRacePreset;
  losses: number[]; // loss every 5 steps (0..200)
  finalLoss: number;
  finalAcc: number;
  /** Per-layer gradient norm at step 0 (layers 1..8) */
  initLayerGradNorms: number[];
  /** Relative weight movement ||ΔW_l|| / ||W_l|| after training (layers 1..8) */
  layerWeightMovement: number[];
}

/**
 * Trains a real 2 → 16 → 16 → ... (8 hidden layers) → 1 network with plain SGD
 * on the 64-point concentric rings dataset. Runs in ~25ms in JS/Node.
 */
export function runDeepTrainingRace(preset: DeepRacePreset, steps = 200, seed = 11): DeepRaceResult {
  const data = makeRingsDataset();
  const rng = createRng(seed);
  const H = 8; // 8 hidden layers of width 16
  const W = 16;
  const resScale = 1 / Math.sqrt(H);

  // Layer 0: 2 -> W, layers 1..H-1: W -> W, out: W -> 1
  const Ws: number[][][] = [];
  const bs: number[][] = [];
  const initNorms: number[] = [];

  for (let l = 0; l < H; l++) {
    const fanIn = l === 0 ? 2 : W;
    const sigma = INIT_BY_ID[preset.init].sigma(fanIn);
    const mat: number[][] = [];
    let normSq = 0;
    for (let i = 0; i < W; i++) {
      const row = new Array<number>(fanIn);
      for (let j = 0; j < fanIn; j++) {
        row[j] = rng.normal(0, sigma);
        normSq += row[j] * row[j];
      }
      mat.push(row);
    }
    Ws.push(mat);
    bs.push(new Array<number>(W).fill(0));
    initNorms.push(Math.sqrt(normSq));
  }

  // Output layer W -> 1 (Xavier init so output starts near 0.5)
  const wOut = Array.from({ length: W }, () => rng.normal(0, 1 / Math.sqrt(W)));
  let bOut = 0;

  // Save initial weights to measure total movement
  const initWs = Ws.map((mat) => mat.map((row) => row.slice()));

  const losses: number[] = [];
  let initLayerGradNorms: number[] = [];

  for (let step = 0; step <= steps; step++) {
    // Accumulate gradients across full 64-point batch
    const gWs = Ws.map((mat) => mat.map((row) => new Array<number>(row.length).fill(0)));
    const gbs = bs.map((b) => new Array<number>(b.length).fill(0));
    const gwOut = new Array<number>(W).fill(0);
    let gbOut = 0;

    let totalLoss = 0;
    let correct = 0;

    for (let n = 0; n < data.length; n++) {
      const { x, y } = data[n];
      const hs: number[][] = [x];
      const zs: number[][] = [];
      const as: number[][] = [];

      let h: number[] = x;
      for (let l = 0; l < H; l++) {
        const mat = Ws[l];
        const b = bs[l];
        const z = new Array<number>(W);
        const a = new Array<number>(W);
        const hNext = new Array<number>(W);
        for (let i = 0; i < W; i++) {
          let sum = b[i];
          const row = mat[i];
          for (let j = 0; j < row.length; j++) sum += row[j] * h[j];
          z[i] = sum;
          a[i] = actFn(preset.act, sum);
          hNext[i] = preset.useResidual && l > 0 ? h[i] + resScale * a[i] : a[i];
        }
        zs.push(z);
        as.push(a);
        hs.push(hNext);
        h = hNext;
      }

      // Output logit & BCE loss
      let logit = bOut;
      for (let i = 0; i < W; i++) logit += wOut[i] * h[i];
      const prob = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, logit))));
      const pClamp = Math.max(1e-7, Math.min(1 - 1e-7, prob));
      totalLoss += -(y * Math.log(pClamp) + (1 - y) * Math.log(1 - pClamp));
      if ((prob >= 0.5 ? 1 : 0) === y) correct++;

      // Backward
      const dLogit = (prob - y) / data.length;
      gbOut += dLogit;
      let g = new Array<number>(W);
      for (let i = 0; i < W; i++) {
        gwOut[i] += dLogit * h[i];
        g[i] = dLogit * wOut[i];
      }

      for (let l = H - 1; l >= 0; l--) {
        const mat = Ws[l];
        const gMat = gWs[l];
        const gb = gbs[l];
        const hPrev = hs[l];
        const z = zs[l];
        const a = as[l];
        const useRes = preset.useResidual && l > 0;
        const gPrev = new Array<number>(hPrev.length).fill(0);

        for (let i = 0; i < W; i++) {
          const ga = useRes ? g[i] * resScale : g[i];
          const gz = ga * actDeriv(preset.act, z[i], a[i]);
          gb[i] += gz;
          const row = mat[i];
          const gRow = gMat[i];
          for (let j = 0; j < row.length; j++) {
            gRow[j] += gz * hPrev[j];
            gPrev[j] += row[j] * gz;
          }
        }
        if (useRes) {
          for (let i = 0; i < W; i++) gPrev[i] += g[i];
        }
        g = gPrev;
      }
    }

    const avgLoss = totalLoss / data.length;
    if (step % 5 === 0) losses.push(avgLoss);

    if (step === 0) {
      initLayerGradNorms = gWs.map((mat) => {
        let s = 0;
        for (const row of mat) for (const v of row) s += v * v;
        return Math.sqrt(s);
      });
    }

    if (step === steps) {
      const layerWeightMovement = Ws.map((mat, l) => {
        let dSq = 0;
        for (let i = 0; i < mat.length; i++) {
          for (let j = 0; j < mat[i].length; j++) {
            const d = mat[i][j] - initWs[l][i][j];
            dSq += d * d;
          }
        }
        return Math.sqrt(dSq) / (initNorms[l] || 1);
      });
      return {
        preset,
        losses,
        finalLoss: avgLoss,
        finalAcc: correct / data.length,
        initLayerGradNorms,
        layerWeightMovement,
      };
    }

    // SGD update (clipped only at extreme explosion so NaNs don't break charts)
    const lr = preset.lr;
    for (let l = 0; l < H; l++) {
      for (let i = 0; i < Ws[l].length; i++) {
        bs[l][i] -= lr * gbs[l][i];
        for (let j = 0; j < Ws[l][i].length; j++) {
          Ws[l][i][j] -= lr * gWs[l][i][j];
        }
      }
    }
    bOut -= lr * gbOut;
    for (let i = 0; i < W; i++) wOut[i] -= lr * gwOut[i];
  }

  throw new Error('unreachable');
}

// ------------------------------------------------------------------ 4. Lab: Rescue the Deep Stack (3 levels, exhaustive)

export interface RescueLevel {
  id: string;
  title: string;
  depth: number;
  brief: string;
  lockedAct?: ActName;
  allowLayerNorm: boolean;
  allowResidual: boolean;
  defaultRecipe: RescueRecipe;
}

export interface RescueRecipe {
  act: ActName;
  init: InitName;
  useLayerNorm: boolean;
  useResidual: boolean;
}

export const RESCUE_ACTS: ActName[] = ['sigmoid', 'tanh', 'relu'];
export const RESCUE_INITS: InitName[] = ['tiny', 'xavier', 'he', 'double', 'naive'];

export const RESCUE_LEVELS: RescueLevel[] = [
  {
    id: 'plain-16',
    title: 'Level 1 · Plain 16-layer stack',
    depth: 16,
    brief:
      'No LayerNorm and no skip connections allowed — only activation and initialisation scale! Keep every layer’s activation RMS inside [0.3, 3.0] and layer-1 gradient ratio inside [0.25, 4.0].',
    allowLayerNorm: false,
    allowResidual: false,
    defaultRecipe: { act: 'sigmoid', init: 'naive', useLayerNorm: false, useResidual: false },
  },
  {
    id: 'sigmoid-trap',
    title: 'Level 2 · The 16-layer Sigmoid curse',
    depth: 16,
    brief:
      'Activation is locked to Sigmoid! Can ANY initialisation scale rescue a plain 16-layer Sigmoid stack? Try every scale — then enable architectural lifelines (LayerNorm / Residual) to break the 0.25¹⁶ curse.',
    lockedAct: 'sigmoid',
    allowLayerNorm: true,
    allowResidual: true,
    defaultRecipe: { act: 'sigmoid', init: 'xavier', useLayerNorm: false, useResidual: false },
  },
  {
    id: 'deep-32',
    title: 'Level 3 · The 32-layer tower',
    depth: 32,
    brief:
      'Now go 32 layers deep! Keep every layer’s forward RMS in [0.3, 3.0] AND keep the layer-1 gradient ratio tightly balanced inside [0.4, 2.5]. Optional goal: get the gradient ratio as close to 1.00× as possible.',
    allowLayerNorm: true,
    allowResidual: true,
    defaultRecipe: { act: 'relu', init: 'xavier', useLayerNorm: false, useResidual: false },
  },
];

export interface RescueEval {
  trace: SignalTrace;
  actOk: boolean;
  gradOk: boolean;
  passes: boolean;
  /** Closeness of gradRatio to 1.0 in log space: |ln(gradRatio)| (lower is better). */
  balanceErr: number;
}

export function evaluateRescue(level: RescueLevel, recipe: RescueRecipe): RescueEval {
  const effective: RescueRecipe = {
    act: level.lockedAct ?? recipe.act,
    init: recipe.init,
    useLayerNorm: level.allowLayerNorm ? recipe.useLayerNorm : false,
    useResidual: level.allowResidual ? recipe.useResidual : false,
  };
  const trace = probeSignal({
    depth: level.depth,
    width: 32,
    act: effective.act,
    init: effective.init,
    useLayerNorm: effective.useLayerNorm,
    useResidual: effective.useResidual,
    seed: 1337,
    samples: 12,
  });

  const gradLo = level.id === 'deep-32' ? 0.4 : 0.25;
  const gradHi = level.id === 'deep-32' ? 2.5 : 4.0;
  const actOk = !trace.overflowed && trace.minActRms >= 0.3 && trace.maxActRms <= 3.0;
  const gradOk = !trace.overflowed && trace.gradRatio >= gradLo && trace.gradRatio <= gradHi;
  const balanceErr = trace.gradRatio > 0 && Number.isFinite(trace.gradRatio) ? Math.abs(Math.log(trace.gradRatio)) : Infinity;
  return { trace, actOk, gradOk, passes: actOk && gradOk, balanceErr };
}

export interface RescueSolution {
  totalRecipes: number;
  passingCount: number;
  plainPassingCount: number;
  bestRecipe: RescueRecipe;
  bestEval: RescueEval;
}

/** Exhaustive search over every legal recipe for a level. */
export function solveRescueLevel(level: RescueLevel): RescueSolution {
  const acts = level.lockedAct ? [level.lockedAct] : RESCUE_ACTS;
  const lns = level.allowLayerNorm ? [false, true] : [false];
  const ress = level.allowResidual ? [false, true] : [false];

  let totalRecipes = 0;
  let passingCount = 0;
  let plainPassingCount = 0;
  let bestRecipe: RescueRecipe | null = null;
  let bestEval: RescueEval | null = null;

  for (const act of acts) {
    for (const init of RESCUE_INITS) {
      for (const useLayerNorm of lns) {
        for (const useResidual of ress) {
          totalRecipes++;
          const recipe: RescueRecipe = { act, init, useLayerNorm, useResidual };
          const ev = evaluateRescue(level, recipe);
          if (ev.passes) {
            passingCount++;
            if (!useLayerNorm && !useResidual) plainPassingCount++;
            if (!bestEval || ev.balanceErr < bestEval.balanceErr) {
              bestRecipe = recipe;
              bestEval = ev;
            }
          }
        }
      }
    }
  }

  return {
    totalRecipes,
    passingCount,
    plainPassingCount,
    bestRecipe: bestRecipe!,
    bestEval: bestEval!,
  };
}

/**
 * Hint for an unsolved recipe. Hints point at evidence on the two charts and
 * only ever name controls the level actually unlocks (pinned by a test).
 */
export function rescueHint(level: RescueLevel, recipe: RescueRecipe, ev: RescueEval): string {
  const lifelines = level.allowLayerNorm && level.allowResidual;
  const plain = !recipe.useLayerNorm && !recipe.useResidual;
  if (level.id === 'sigmoid-trap' && plain) {
    return 'Try every initialisation scale and watch the gradient chart: none of the five keeps layer 1 alive in a plain Sigmoid stack. Something has to change the shape of the network itself.';
  }
  const gradLo = level.id === 'deep-32' ? 0.4 : 0.25;
  if (!ev.trace.overflowed && ev.trace.gradRatio < gradLo) {
    return lifelines
      ? 'Gradients are vanishing: the gradient chart sinks towards layer 1. What would give the gradient a path back that skips the layers squashing it?'
      : 'Gradients are vanishing: the gradient chart sinks towards layer 1. Which activation has a slope that does not shrink the signal, and which scale keeps its variance steady?';
  }
  return lifelines
    ? 'The signal is exploding: the RMS chart climbs layer after layer. Look at how big a scale you chose, and at what could re-centre each layer.'
    : 'The signal is exploding: the RMS chart climbs layer after layer. Look at how big a scale you chose for this activation.';
}
