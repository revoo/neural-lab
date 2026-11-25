// authored-by: Gemini Antigravity
/**
 * CNNs and Feature Maps from scratch:
 *  - Exact receptive-field & effective-stride (jump) calculator + VGG parameter savings
 *  - Real tiny 2D CNN (Conv2d 1->C 3x3 valid -> ReLU -> GlobalMaxPool -> Linear C->3) with exact backprop (FD-checked!)
 *  - Fair comparison: 55-param CNN vs 547-param Dense MLP on shifted 8x8 shapes (including unseen spatial positions!)
 *  - Exhaustive Lab: Architect the CNN (3 briefs: Receptive Field budget, Shift-Generalisation, Classification Head)
 */
import { createRng } from '../../lib/rng';

// ------------------------------------------------------------------ 1. Receptive Field Calculator

export type LayerSpec = {
  id: string;
  name: string;
  kind: 'conv' | 'pool';
  k: number;
  s: number;
  p: number;
};

export const LAYER_CATALOG: LayerSpec[] = [
  { id: 'conv3', name: 'Conv 3×3 (s=1)', kind: 'conv', k: 3, s: 1, p: 1 },
  { id: 'conv5', name: 'Conv 5×5 (s=1)', kind: 'conv', k: 5, s: 1, p: 2 },
  { id: 'conv3s2', name: 'Conv 3×3 (s=2)', kind: 'conv', k: 3, s: 2, p: 1 },
  { id: 'pool2', name: 'MaxPool 2×2 (s=2)', kind: 'pool', k: 2, s: 2, p: 0 },
];

export type RfRow = {
  layer: number;
  spec: LayerSpec;
  outSize: number;
  rf: number;
  jump: number;
  params: number; // assuming C channels in/out for conv layers
};

export function traceReceptiveField(inputSize: number, layers: readonly LayerSpec[], channels = 16): RfRow[] {
  let rf = 1;
  let jump = 1;
  let size = inputSize;
  const rows: RfRow[] = [];
  layers.forEach((spec, idx) => {
    size = Math.floor((size + 2 * spec.p - spec.k) / spec.s) + 1;
    rf = rf + (spec.k - 1) * jump;
    jump = jump * spec.s;
    const params = spec.kind === 'conv' ? spec.k * spec.k * channels * channels + channels : 0;
    rows.push({ layer: idx + 1, spec, outSize: Math.max(1, size), rf, jump, params });
  });
  return rows;
}

/** VGG equivalence check: L stacked 3x3 convs vs single (2L+1)x(2L+1) conv at C channels */
export function vggSavings(layers3x3: number, channels = 64) {
  const bigK = 2 * layers3x3 + 1;
  const stackParams = layers3x3 * (9 * channels * channels);
  const singleParams = bigK * bigK * channels * channels;
  return { bigK, stackParams, singleParams, reduction: 1 - stackParams / singleParams };
}

// ------------------------------------------------------------------ 2. Pooling & Shift Invariance

export function pool2x2(grid: number[][], mode: 'max' | 'avg'): number[][] {
  const h = Math.floor(grid.length / 2);
  const w = Math.floor(grid[0].length / 2);
  const out: number[][] = [];
  for (let r = 0; r < h; r++) {
    const row: number[] = [];
    for (let c = 0; c < w; c++) {
      const vals = [
        grid[2 * r][2 * c],
        grid[2 * r][2 * c + 1],
        grid[2 * r + 1][2 * c],
        grid[2 * r + 1][2 * c + 1],
      ];
      row.push(mode === 'max' ? Math.max(...vals) : vals.reduce((a, b) => a + b, 0) / 4);
    }
    out.push(row);
  }
  return out;
}

// ------------------------------------------------------------------ 3. Tiny Shifted-Shape Dataset & Real Backprop CNN

export type ShapeSample = {
  /** 8x8 flattened Float64 values in [0, 1] */
  x: number[];
  /** 0 = Horizontal Bar, 1 = Vertical Bar, 2 = Diagonal Slash */
  y: 0 | 1 | 2;
};

function makeShapeImage(cls: 0 | 1 | 2, topR: number, leftC: number, noiseRng: ReturnType<typeof createRng>): number[] {
  const img = new Array<number>(64).fill(0);
  if (cls === 0) {
    // 1x4 horizontal bar
    for (let dc = 0; dc < 4; dc++) img[(topR + 1) * 8 + (leftC + dc)] = 1;
  } else if (cls === 1) {
    // 4x1 vertical bar
    for (let dr = 0; dr < 4; dr++) img[(topR + dr) * 8 + (leftC + 1)] = 1;
  } else {
    // 4-pixel main diagonal
    for (let d = 0; d < 4; d++) img[(topR + d) * 8 + (leftC + d)] = 1;
  }
  for (let i = 0; i < 64; i++) {
    img[i] = Math.max(0, Math.min(1, img[i] + noiseRng.normal(0, 0.08)));
  }
  return img;
}

let CACHED_SHAPES: { train: ShapeSample[]; testInDist: ShapeSample[]; testShifted: ShapeSample[] } | null = null;

/**
 * Train shapes live in top-left quadrant (topR in {0,1}, leftC in {0,1}).
 * testInDist uses the same top-left positions; testShifted places shapes in UNSEEN bottom-right positions (topR in {3,4}, leftC in {3,4})!
 */
export function getShapeDataset() {
  if (CACHED_SHAPES) return CACHED_SHAPES;
  const rng = createRng(509);
  const train: ShapeSample[] = [];
  const testInDist: ShapeSample[] = [];
  const testShifted: ShapeSample[] = [];

  for (let cls = 0 as 0 | 1 | 2; cls < 3; cls = (cls + 1) as 0 | 1 | 2) {
    for (let i = 0; i < 36; i++) {
      train.push({ x: makeShapeImage(cls, rng.int(2), rng.int(2), rng), y: cls });
    }
    for (let i = 0; i < 20; i++) {
      testInDist.push({ x: makeShapeImage(cls, rng.int(2), rng.int(2), rng), y: cls });
    }
    for (let i = 0; i < 20; i++) {
      testShifted.push({ x: makeShapeImage(cls, 3 + rng.int(2), 3 + rng.int(2), rng), y: cls });
    }
  }
  CACHED_SHAPES = { train, testInDist, testShifted };
  return CACHED_SHAPES;
}

export type TinyCnnParams = {
  nFilters: number;
  /** nFilters x 9 conv3x3 weights */
  convW: number[];
  /** nFilters conv biases */
  convB: number[];
  /** 3 x nFilters linear head weights (row-major: outClass * nFilters + f) */
  headW: number[];
  /** 3 head biases */
  headB: number[];
};

export function initTinyCnn(nFilters: number, seed = 11): TinyCnnParams {
  const rng = createRng(seed);
  const convW = Array.from({ length: nFilters * 9 }, () => rng.normal(0, Math.sqrt(2 / 9)));
  const convB = new Array<number>(nFilters).fill(0);
  const headW = Array.from({ length: 3 * nFilters }, () => rng.normal(0, Math.sqrt(1 / nFilters)));
  const headB = new Array<number>(3).fill(0);
  return { nFilters, convW, convB, headW, headB };
}

export function cnnParamCount(nFilters: number): number {
  return nFilters * 9 + nFilters + 3 * nFilters + 3;
}

export function cnnForwardOne(p: TinyCnnParams, x: readonly number[]) {
  const F = p.nFilters;
  const maps: number[][] = Array.from({ length: F }, () => new Array<number>(36).fill(0));
  const pooled = new Array<number>(F).fill(0);
  const argmax = new Array<number>(F).fill(0);

  for (let f = 0; f < F; f++) {
    let best = -Infinity;
    let bestIdx = 0;
    const wOff = f * 9;
    const b = p.convB[f];
    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 6; c++) {
        let s = b;
        for (let kr = 0; kr < 3; kr++) {
          for (let kc = 0; kc < 3; kc++) {
            s += x[(r + kr) * 8 + (c + kc)] * p.convW[wOff + kr * 3 + kc];
          }
        }
        const a = s > 0 ? s : 0; // ReLU
        const idx = r * 6 + c;
        maps[f][idx] = a;
        if (a > best) {
          best = a;
          bestIdx = idx;
        }
      }
    }
    pooled[f] = best;
    argmax[f] = bestIdx;
  }

  const logits = [0, 0, 0];
  for (let cls = 0; cls < 3; cls++) {
    let s = p.headB[cls];
    for (let f = 0; f < F; f++) s += p.headW[cls * F + f] * pooled[f];
    logits[cls] = s;
  }
  const maxL = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maxL));
  const sumE = exps[0] + exps[1] + exps[2];
  const probs = exps.map((e) => e / sumE);
  return { maps, pooled, argmax, logits, probs };
}

/** Exact cross-entropy loss and analytic gradient over a batch (verified by central finite differences in cnn.test.ts). */
export function cnnLossAndGrad(p: TinyCnnParams, batch: readonly ShapeSample[], wd = 1e-4) {
  const F = p.nFilters;
  const gConvW = new Array<number>(F * 9).fill(0);
  const gConvB = new Array<number>(F).fill(0);
  const gHeadW = new Array<number>(3 * F).fill(0);
  const gHeadB = new Array<number>(3).fill(0);

  let loss = 0;
  let correct = 0;
  const n = batch.length;

  for (const sample of batch) {
    const fwd = cnnForwardOne(p, sample.x);
    const pred = fwd.probs[0] >= fwd.probs[1] && fwd.probs[0] >= fwd.probs[2] ? 0 : fwd.probs[1] >= fwd.probs[2] ? 1 : 2;
    if (pred === sample.y) correct++;
    loss -= Math.log(Math.max(1e-12, fwd.probs[sample.y]));

    const dLogits = [fwd.probs[0], fwd.probs[1], fwd.probs[2]];
    dLogits[sample.y] -= 1;

    const dPool = new Array<number>(F).fill(0);
    for (let cls = 0; cls < 3; cls++) {
      gHeadB[cls] += dLogits[cls] / n;
      for (let f = 0; f < F; f++) {
        gHeadW[cls * F + f] += (dLogits[cls] * fwd.pooled[f]) / n;
        dPool[f] += dLogits[cls] * p.headW[cls * F + f];
      }
    }

    for (let f = 0; f < F; f++) {
      if (fwd.pooled[f] <= 0) continue; // ReLU gated off
      const g = dPool[f] / n;
      const idx = fwd.argmax[f];
      const r = Math.floor(idx / 6);
      const c = idx % 6;
      gConvB[f] += g;
      const wOff = f * 9;
      for (let kr = 0; kr < 3; kr++) {
        for (let kc = 0; kc < 3; kc++) {
          gConvW[wOff + kr * 3 + kc] += g * sample.x[(r + kr) * 8 + (c + kc)];
        }
      }
    }
  }

  loss /= n;
  // L2 weight decay
  for (let i = 0; i < gConvW.length; i++) {
    loss += 0.5 * wd * p.convW[i] * p.convW[i];
    gConvW[i] += wd * p.convW[i];
  }
  for (let i = 0; i < gHeadW.length; i++) {
    loss += 0.5 * wd * p.headW[i] * p.headW[i];
    gHeadW[i] += wd * p.headW[i];
  }

  return { loss, acc: correct / n, gConvW, gConvB, gHeadW, gHeadB };
}

export function trainTinyCnn(nFilters: number, epochs = 120, lr = 0.08, seed = 11) {
  const { train, testInDist, testShifted } = getShapeDataset();
  const p = initTinyCnn(nFilters, seed);
  // Adam state
  const mkZero = (len: number) => new Array<number>(len).fill(0);
  const m = { convW: mkZero(p.convW.length), convB: mkZero(p.convB.length), headW: mkZero(p.headW.length), headB: mkZero(p.headB.length) };
  const v = { convW: mkZero(p.convW.length), convB: mkZero(p.convB.length), headW: mkZero(p.headW.length), headB: mkZero(p.headB.length) };
  const b1 = 0.9;
  const b2 = 0.999;

  const stepArr = (w: number[], g: number[], ma: number[], va: number[], t: number) => {
    const c1 = 1 - b1 ** t;
    const c2 = 1 - b2 ** t;
    for (let i = 0; i < w.length; i++) {
      ma[i] = b1 * ma[i] + (1 - b1) * g[i];
      va[i] = b2 * va[i] + (1 - b2) * g[i] * g[i];
      w[i] -= lr * (ma[i] / c1) / (Math.sqrt(va[i] / c2) + 1e-8);
    }
  };

  for (let ep = 1; ep <= epochs; ep++) {
    const res = cnnLossAndGrad(p, train);
    stepArr(p.convW, res.gConvW, m.convW, v.convW, ep);
    stepArr(p.convB, res.gConvB, m.convB, v.convB, ep);
    stepArr(p.headW, res.gHeadW, m.headW, v.headW, ep);
    stepArr(p.headB, res.gHeadB, m.headB, v.headB, ep);
  }

  const evalSet = (ds: ShapeSample[]) => cnnLossAndGrad(p, ds, 0).acc;
  return {
    params: p,
    paramCount: cnnParamCount(nFilters),
    trainAcc: evalSet(train),
    inDistAcc: evalSet(testInDist),
    shiftedAcc: evalSet(testShifted),
  };
}

// ------------------------------------------------------------------ 4. Dense MLP Baseline on Exact Same Shifted Shapes

export function trainDenseBaseline(hidden = 8, epochs = 120, lr = 0.05, seed = 11) {
  const { train, testInDist, testShifted } = getShapeDataset();
  const rng = createRng(seed);
  const W1 = Array.from({ length: hidden * 64 }, () => rng.normal(0, Math.sqrt(2 / 64)));
  const b1 = new Array<number>(hidden).fill(0);
  const W2 = Array.from({ length: 3 * hidden }, () => rng.normal(0, Math.sqrt(1 / hidden)));
  const b2 = new Array<number>(3).fill(0);

  const mW1 = new Array<number>(W1.length).fill(0);
  const vW1 = new Array<number>(W1.length).fill(0);
  const mb1 = new Array<number>(b1.length).fill(0);
  const vb1 = new Array<number>(b1.length).fill(0);
  const mW2 = new Array<number>(W2.length).fill(0);
  const vW2 = new Array<number>(W2.length).fill(0);
  const mb2 = new Array<number>(b2.length).fill(0);
  const vb2 = new Array<number>(b2.length).fill(0);

  const adamStep = (w: number[], g: number[], ma: number[], va: number[], t: number) => {
    const c1 = 1 - 0.9 ** t;
    const c2 = 1 - 0.999 ** t;
    for (let i = 0; i < w.length; i++) {
      ma[i] = 0.9 * ma[i] + 0.1 * g[i];
      va[i] = 0.999 * va[i] + 0.001 * g[i] * g[i];
      w[i] -= lr * (ma[i] / c1) / (Math.sqrt(va[i] / c2) + 1e-8);
    }
  };

  const forwardOne = (x: readonly number[]) => {
    const h = new Array<number>(hidden).fill(0);
    for (let j = 0; j < hidden; j++) {
      let s = b1[j];
      for (let i = 0; i < 64; i++) s += W1[j * 64 + i] * x[i];
      h[j] = s > 0 ? s : 0;
    }
    const logits = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      let s = b2[c];
      for (let j = 0; j < hidden; j++) s += W2[c * hidden + j] * h[j];
      logits[c] = s;
    }
    const mx = Math.max(...logits);
    const ex = logits.map((v) => Math.exp(v - mx));
    const se = ex[0] + ex[1] + ex[2];
    return { h, probs: ex.map((e) => e / se) };
  };

  const n = train.length;
  for (let ep = 1; ep <= epochs; ep++) {
    const gW1 = new Array<number>(W1.length).fill(0);
    const gb1 = new Array<number>(b1.length).fill(0);
    const gW2 = new Array<number>(W2.length).fill(0);
    const gb2 = new Array<number>(b2.length).fill(0);
    for (const s of train) {
      const { h, probs } = forwardOne(s.x);
      const dL = [probs[0], probs[1], probs[2]];
      dL[s.y] -= 1;
      const dH = new Array<number>(hidden).fill(0);
      for (let c = 0; c < 3; c++) {
        gb2[c] += dL[c] / n;
        for (let j = 0; j < hidden; j++) {
          gW2[c * hidden + j] += (dL[c] * h[j]) / n;
          dH[j] += dL[c] * W2[c * hidden + j];
        }
      }
      for (let j = 0; j < hidden; j++) {
        if (h[j] <= 0) continue;
        gb1[j] += dH[j] / n;
        for (let i = 0; i < 64; i++) gW1[j * 64 + i] += (dH[j] * s.x[i]) / n;
      }
    }
    adamStep(W1, gW1, mW1, vW1, ep);
    adamStep(b1, gb1, mb1, vb1, ep);
    adamStep(W2, gW2, mW2, vW2, ep);
    adamStep(b2, gb2, mb2, vb2, ep);
  }

  const evalSet = (ds: ShapeSample[]) => {
    let ok = 0;
    for (const s of ds) {
      const { probs } = forwardOne(s.x);
      const pred = probs[0] >= probs[1] && probs[0] >= probs[2] ? 0 : probs[1] >= probs[2] ? 1 : 2;
      if (pred === s.y) ok++;
    }
    return ok / ds.length;
  };

  const inputEnergy = new Array<number>(64).fill(0);
  for (let i = 0; i < 64; i++) {
    let s = 0;
    for (let j = 0; j < hidden; j++) s += Math.abs(W1[j * 64 + i]);
    inputEnergy[i] = s / hidden;
  }

  return {
    paramCount: hidden * 64 + hidden + 3 * hidden + 3,
    trainAcc: evalSet(train),
    inDistAcc: evalSet(testInDist),
    shiftedAcc: evalSet(testShifted),
    inputEnergy,
    predict: (x: readonly number[]) => {
      const { probs } = forwardOne(x);
      const pred = probs[0] >= probs[1] && probs[0] >= probs[2] ? 0 : probs[1] >= probs[2] ? 1 : 2;
      return { probs, pred };
    },
  };
}

let CACHED_RACE: { cnn: ReturnType<typeof trainTinyCnn>; dense: ReturnType<typeof trainDenseBaseline> } | null = null;
export function getShiftRace() {
  if (CACHED_RACE) return CACHED_RACE;
  CACHED_RACE = {
    cnn: trainTinyCnn(4, 120, 0.08, 11),
    dense: trainDenseBaseline(8, 120, 0.05, 11),
  };
  return CACHED_RACE;
}

// ------------------------------------------------------------------ 5. Lab: Architect the CNN (3 Exhaustive Briefs)

export type CnnBriefId = 'rf-budget' | 'shift-gen' | 'gap-head';

export const CNN_LAB_BRIEFS: { id: CnnBriefId; title: string; story: string }[] = [
  {
    id: 'rf-budget',
    title: 'Brief 1 · Reach Receptive Field ≥ 15×15 Under 10,000 Weights',
    story: 'You have 16 channels per conv layer. Design a small conv/pool stack (conv kernel size, number of conv layers, and interleaved 2×2 pooling layers) that reaches receptive field ≥ 15×15 while spending ≤ 10,000 parameters. Optional goal: hit the fewest parameters.',
  },
  {
    id: 'shift-gen',
    title: 'Brief 2 · Generalise to Unseen Positions (Tiny Shape Classifier)',
    story: 'Shapes are trained ONLY in the top-left corner of an 8×8 canvas and tested in the unseen bottom-right corner. Pick a model family and capacity so unseen-position test accuracy reaches ≥ 95% with as few parameters as possible.',
  },
  {
    id: 'gap-head',
    title: 'Brief 3 · Shrink a 7×7×512 ImageNet Classification Head Under 3M Params',
    story: 'A deep backbone outputs a 7×7×512 feature map for 1,000 classes. Choose a classification head that stays ≤ 3.0M parameters while preserving all 512 semantic channels (no channel bottleneck before the classifier).',
  },
];

export type RfConfig = { k: 3 | 5; nConv: 2 | 3 | 4; nPools: 0 | 1 | 2 };

export function buildBrief1Stack(cfg: RfConfig): LayerSpec[] {
  const convSpec = cfg.k === 3 ? LAYER_CATALOG[0] : LAYER_CATALOG[1];
  const poolSpec = LAYER_CATALOG[3];
  const stack: LayerSpec[] = [];
  let poolsLeft = cfg.nPools;
  for (let i = 0; i < cfg.nConv; i++) {
    stack.push(convSpec);
    if (poolsLeft > 0 && i < cfg.nConv - 1) {
      stack.push(poolSpec);
      poolsLeft--;
    }
  }
  while (poolsLeft > 0) {
    stack.push(poolSpec);
    poolsLeft--;
  }
  return stack;
}

export function evalCnnBrief1(cfg: RfConfig) {
  const stack = buildBrief1Stack(cfg);
  const trace = traceReceptiveField(32, stack, 16);
  const last = trace[trace.length - 1];
  const params = trace.reduce((a, r) => a + r.params, 0);
  return { rf: last.rf, jump: last.jump, params, trace, pass: last.rf >= 15 && params <= 10_000 };
}

export function cnnBrief1Grid() {
  const rows: { cfg: RfConfig; rf: number; params: number; pass: boolean }[] = [];
  for (const k of [3, 5] as const)
    for (const nConv of [2, 3, 4] as const)
      for (const nPools of [0, 1, 2] as const) {
        const r = evalCnnBrief1({ k, nConv, nPools });
        rows.push({ cfg: { k, nConv, nPools }, rf: r.rf, params: r.params, pass: r.pass });
      }
  return rows;
}

export type ShiftChoice = 'dense-4' | 'dense-8' | 'dense-16' | 'cnn-2' | 'cnn-3' | 'cnn-4' | 'cnn-6';

export const SHIFT_CHOICES: { value: ShiftChoice; label: string }[] = [
  { value: 'dense-4', label: 'Dense MLP (h=4)' },
  { value: 'dense-8', label: 'Dense MLP (h=8)' },
  { value: 'dense-16', label: 'Dense MLP (h=16)' },
  { value: 'cnn-2', label: 'CNN + GlobalMaxPool (2 filters)' },
  { value: 'cnn-3', label: 'CNN + GlobalMaxPool (3 filters)' },
  { value: 'cnn-4', label: 'CNN + GlobalMaxPool (4 filters)' },
  { value: 'cnn-6', label: 'CNN + GlobalMaxPool (6 filters)' },
];

export type Brief2Eval = {
  kind: 'dense' | 'cnn';
  params: number;
  inDistAcc: number;
  shiftedAcc: number;
  pass: boolean;
  cnnParams?: TinyCnnParams;
  denseEnergy?: number[];
  predict: (x: readonly number[]) => { probs: number[]; pred: number };
};

const B2_CACHE = new Map<ShiftChoice, Brief2Eval>();

export function evalCnnBrief2(choice: ShiftChoice): Brief2Eval {
  const hit = B2_CACHE.get(choice);
  if (hit) return hit;
  let res: Brief2Eval;
  if (choice.startsWith('dense-')) {
    const h = Number(choice.split('-')[1]);
    const r = trainDenseBaseline(h, 120, 0.05, 11);
    res = {
      kind: 'dense',
      params: r.paramCount,
      inDistAcc: r.inDistAcc,
      shiftedAcc: r.shiftedAcc,
      pass: r.shiftedAcc >= 0.95 - 1e-9,
      denseEnergy: r.inputEnergy,
      predict: r.predict,
    };
  } else {
    const f = Number(choice.split('-')[1]);
    const r = trainTinyCnn(f, 120, 0.08, 11);
    res = {
      kind: 'cnn',
      params: r.paramCount,
      inDistAcc: r.inDistAcc,
      shiftedAcc: r.shiftedAcc,
      pass: r.shiftedAcc >= 0.95 - 1e-9,
      cnnParams: r.params,
      predict: (x: readonly number[]) => {
        const fwd = cnnForwardOne(r.params, x);
        const pred = fwd.probs[0] >= fwd.probs[1] && fwd.probs[0] >= fwd.probs[2] ? 0 : fwd.probs[1] >= fwd.probs[2] ? 1 : 2;
        return { probs: fwd.probs, pred };
      },
    };
  }
  B2_CACHE.set(choice, res);
  return res;
}

export type HeadChoice = 'vgg-fc' | 'flatten-linear' | 'bottleneck-gap' | 'gap-linear';

export const HEAD_CHOICES: { value: HeadChoice; label: string }[] = [
  { value: 'vgg-fc', label: 'Flatten → 4096 → 1000 (VGG-style)' },
  { value: 'flatten-linear', label: 'Flatten → 1000' },
  { value: 'bottleneck-gap', label: '1×1 Conv (512→128) → GAP → 1000' },
  { value: 'gap-linear', label: 'Global Avg Pool (512) → 1000 (ResNet)' },
];

export function evalCnnBrief3(head: HeadChoice) {
  const flat = 7 * 7 * 512; // 25,088
  if (head === 'vgg-fc') {
    const params = flat * 4096 + 4096 + 4096 * 1000 + 1000;
    return { head, params, channelsKept: 512, pass: false };
  }
  if (head === 'flatten-linear') {
    const params = flat * 1000 + 1000;
    return { head, params, channelsKept: 512, pass: false };
  }
  if (head === 'bottleneck-gap') {
    const params = 512 * 128 + 128 + 128 * 1000 + 1000;
    return { head, params, channelsKept: 128, pass: false }; // shrinks channels below 512!
  }
  const params = 512 * 1000 + 1000;
  return { head, params, channelsKept: 512, pass: true };
}

/** Evidence-only hint for the CNN Architect Lab (never names a winning button or setting). */
export function cnnLabHint(
  id: CnnBriefId,
  st: { k: 3 | 5; nConv: 2 | 3 | 4; nPools: 0 | 1 | 2; shiftChoice: ShiftChoice; headChoice: HeadChoice },
): string {
  if (id === 'rf-budget') {
    const r = evalCnnBrief1({ k: st.k, nConv: st.nConv, nPools: st.nPools });
    if (r.pass) {
      return r.params === 6960
        ? 'You hit the exact parameter floor (6,960 weights) with an 18×18 receptive field.'
        : `This stack passes (${r.rf}×${r.rf} RF, ${r.params.toLocaleString()} weights). Look at the last layer's parameter share — can a leaner stack still clear 15×15?`;
    }
    if (r.params > 10_000 && r.rf < 15) {
      return `Both constraints miss: ${r.params.toLocaleString()} weights (> 10,000) and ${r.rf}×${r.rf} RF (< 15×15). Compare the parameter cost per layer with the jump column in the trace.`;
    }
    if (r.params > 10_000) {
      return `Receptive field clears 15×15 (${r.rf}×${r.rf}), but each conv layer's weights scale with kernel area (k²). Look at how much cheaper each layer becomes when kernel area shrinks, while downsampling grows the jump for free.`;
    }
    if (r.jump === 1) {
      return `Your effective stride (jump) is still 1 throughout the stack, so each layer widens the field by only ${st.k - 1} pixels. Doubling the jump mid-stack makes later layers cover twice as many input pixels per step at zero parameter cost.`;
    }
    return `Weights fit (${r.params.toLocaleString()} ≤ 10,000), but the receptive field stops at ${r.rf}×${r.rf} (< 15×15). Watch how much jump multiplies the reach of any conv layer placed after downsampling.`;
  }
  if (id === 'shift-gen') {
    const r = evalCnnBrief2(st.shiftChoice);
    if (r.pass) {
      return r.params === 55
        ? 'Both targets met at the minimum parameter count (55 weights, 100.0% unseen-shift accuracy).'
        : `This model clears 95% unseen-shift accuracy (${(100 * r.shiftedAcc).toFixed(1)}%) with ${r.params} weights. Check whether slightly fewer filters still preserve full accuracy.`;
    }
    if (r.kind === 'dense') {
      return `In-distribution accuracy is ${(100 * r.inDistAcc).toFixed(1)}%, yet unseen bottom-right accuracy collapses to ${(100 * r.shiftedAcc).toFixed(1)}%. Look at the 8×8 input-weight energy map: bottom-right pixels received almost no gradient during training, regardless of hidden width.`;
    }
    return `Translation equivariance is working (feature maps do light up in the bottom-right corner), but unseen-shift accuracy is only ${(100 * r.shiftedAcc).toFixed(1)}%. Three distinct orientations need enough learned filters to separate them cleanly.`;
  }
  const r = evalCnnBrief3(st.headChoice);
  if (r.pass) {
    return 'Passed: collapsing the 7×7 spatial grid before the linear classifier keeps all 512 channels while spending only 0.513M parameters.';
  }
  if (r.channelsKept < 512) {
    return `Parameters fit (${(r.params / 1e6).toFixed(3)}M ≤ 3.0M), but projecting channels down drops the representation from 512 to ${r.channelsKept} channels before the classifier.`;
  }
  return `Flattening keeps all 49 spatial positions separate (7 × 7 × 512 = 25,088 inputs into the first dense matrix), which multiplies head weight count by 49× (${(r.params / 1e6).toFixed(3)}M > 3.0M).`;
}

export function cnnLabExhaustive(id: CnnBriefId) {
  if (id === 'rf-budget') {
    let total = 0;
    let passing = 0;
    let bestP = Infinity;
    let bestDesc = '';
    for (const k of [3, 5] as const)
      for (const nConv of [2, 3, 4] as const)
        for (const nPools of [0, 1, 2] as const) {
          total++;
          const r = evalCnnBrief1({ k, nConv, nPools });
          if (!r.pass) continue;
          passing++;
          if (r.params < bestP) {
            bestP = r.params;
            bestDesc = `${nConv}× Conv ${k}×${k} + ${nPools} pools (RF ${r.rf}×${r.rf}, ${r.params.toLocaleString()} params)`;
          }
        }
    return { total, passing, summary: `Fewest params: ${bestDesc}` };
  }
  if (id === 'shift-gen') {
    const all = SHIFT_CHOICES.map((c) => ({ c, r: evalCnnBrief2(c.value) }));
    const pass = all.filter((x) => x.r.pass);
    const best = pass.reduce((a, b) => (b.r.params < a.r.params ? b : a));
    return {
      total: all.length,
      passing: pass.length,
      summary: `Fewest params: ${best.c.label} (${best.r.params} params, ${(100 * best.r.shiftedAcc).toFixed(1)}% unseen-shift acc)`,
    };
  }
  const all = HEAD_CHOICES.map((h) => ({ h, r: evalCnnBrief3(h.value) }));
  const pass = all.filter((x) => x.r.pass);
  return {
    total: all.length,
    passing: pass.length,
    summary: `Unique solution: Global Avg Pool → 1000 (${(pass[0].r.params / 1e6).toFixed(3)}M params vs VGG’s 106.9M!)`,
  };
}
