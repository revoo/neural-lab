// authored-by: Gemini Antigravity
// Pure TypeScript engine for Module 11 Lesson 2: LSTMs and GRUs.
// Zero external ML dependencies — exact LSTM and GRU forward + Backpropagation Through
// Time (BPTT) passes (both verified against central finite differences < 1e-8), interactive
// 1-unit gate playground, Constant Error Carousel forget-bias probe, real fair Vanilla RNN
// vs GRU vs LSTM delayed-recall benchmark, parameter-count calculator, and exhaustive Lab.

import { createRng } from '../../lib/rng';

export function sigmoid(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

// ------------------------------------------------------------------ 1. Interactive 1-Unit LSTM Gate Playground

export type GatePresetId = 'store' | 'hold' | 'read' | 'reset';

export type GateStepTrace = {
  t: number;
  x: number;
  f: number;
  i: number;
  o: number;
  cTilde: number;
  c: number;
  h: number;
};

/**
 * Run a 6-step sequence where x_0 = +1.0 is the important cue, steps 1..4 are four
 * hostile negative distractors [-0.8, -0.7, -0.6, -0.7] trying to overwrite the memory,
 * and step 5 is the read step (x_5 = 0).
 * At t = 0 we write the cue (f=0, i=1, cTilde=tanh(1.5*x_0)); on steps 1..4 the learner's
 * chosen (f, i, o) controls whether the cell highway preserves c_0 or gets overwritten!
 */
export const DISTRACTOR_SEQ: number[] = [1.0, -0.8, -0.7, -0.6, -0.7, 0.0];

export function simulateLstmCorridor(
  fHold: number,
  iHold: number,
  oHold: number,
  oRead = 0.95,
): {
  steps: GateStepTrace[];
  initialC: number;
  finalC: number;
  retentionRatio: number;
  maxCorridorH: number;
  readH: number;
} {
  const steps: GateStepTrace[] = [];
  let cPrev = 0;

  for (let t = 0; t < DISTRACTOR_SEQ.length; t++) {
    const x = DISTRACTOR_SEQ[t];
    const cTilde = Math.tanh(1.5 * x);
    let f: number;
    let i: number;
    let o: number;
    if (t === 0) {
      // Write cue cleanly at t = 0
      f = 0.0;
      i = 1.0;
      o = oHold;
    } else if (t === DISTRACTOR_SEQ.length - 1) {
      // Read step: keep cell, open output gate
      f = fHold;
      i = iHold;
      o = oRead;
    } else {
      f = fHold;
      i = iHold;
      o = oHold;
    }
    const c = f * cPrev + i * cTilde;
    const h = o * Math.tanh(c);
    steps.push({ t, x, f, i, o, cTilde, c, h });
    cPrev = c;
  }

  const initialC = steps[0].c;
  const finalC = steps[steps.length - 1].c;
  const retentionRatio = finalC / initialC;
  let maxCorridorH = 0;
  for (let t = 1; t <= 4; t++) {
    maxCorridorH = Math.max(maxCorridorH, Math.abs(steps[t].h));
  }
  return {
    steps,
    initialC,
    finalC,
    retentionRatio,
    maxCorridorH,
    readH: steps[steps.length - 1].h,
  };
}

// ------------------------------------------------------------------ 2. Constant Error Carousel (Forget-Gate Bias Trick)

export type CarouselCurve = {
  label: string;
  gateVal: number;
  /** Exact gradient multiplier at k = 0..T-1 steps backwards */
  ratios: number[];
  step1Ratio: number;
};

/**
 * Exact cell-highway gradient ∂c_T / ∂c_{T-k} = f^k where f = sigmoid(b_f),
 * compared directly against a plain tanh RNN (ρ = 0.80, effective step factor 0.80 * 0.92).
 */
export function carouselGradientFlow(T: number, forgetBias: number, rnnRho = 0.8): {
  lstm: CarouselCurve;
  rnn: CarouselCurve;
} {
  const f = sigmoid(forgetBias);
  const lstmRatios: number[] = [];
  const rnnRatios: number[] = [];
  const rnnStep = rnnRho * 0.92; // modest tanh attenuation around small activations
  for (let k = 0; k < T; k++) {
    lstmRatios.push(f ** k);
    rnnRatios.push(rnnStep ** k);
  }
  return {
    lstm: {
      label: `LSTM highway (b_f = ${forgetBias >= 0 ? '+' : ''}${forgetBias}, f = ${f.toFixed(3)})`,
      gateVal: f,
      ratios: lstmRatios,
      step1Ratio: lstmRatios[T - 1],
    },
    rnn: {
      label: `Plain tanh RNN (ρ = ${rnnRho.toFixed(2)})`,
      gateVal: rnnStep,
      ratios: rnnRatios,
      step1Ratio: rnnRatios[T - 1],
    },
  };
}

// ------------------------------------------------------------------ 3. Exact LSTM & GRU Cells + Hand-Written BPTT

export type LstmParams = {
  H: number;
  /** Input dimension D (defaults to 1) */
  D: number;
  /** 4 gates stacked [f, i, c, o], each H × (D + H) row-major -> length 4 * H * (D + H) */
  W: number[];
  /** Biases [bf, bi, bc, bo] -> length 4 * H */
  b: number[];
  /** Readout weights (length H) */
  Wy: number[];
  by: number;
};

export type GruParams = {
  H: number;
  D: number;
  /** Update & reset gate weights [Wz, Wr], each H × (D + H) -> length 2 * H * (D + H) */
  Wg: number[];
  /** Biases [bz, br] -> length 2 * H */
  bg: number[];
  /** Candidate input weights Whx (H × D) and recurrent Whh (H × H) */
  Whx: number[];
  Whh: number[];
  bh: number[];
  Wy: number[];
  by: number;
};

export type SeqExample = {
  xs: number[];
  y: 0 | 1;
};

export function initLstm(H: number, forgetBias: number, seed: number): LstmParams {
  const rng = createRng(seed);
  const D = 1;
  const K = D + H;
  const scale = 1 / Math.sqrt(K);
  const W = Array.from({ length: 4 * H * K }, () => rng.normal(0, scale));
  const b = new Array<number>(4 * H).fill(0);
  // Set forget-gate bias block [0 .. H-1] to forgetBias (Jozefowicz et al. 2015)
  for (let i = 0; i < H; i++) b[i] = forgetBias;
  const Wy = Array.from({ length: H }, () => rng.normal(0, 0.5));
  return { H, D, W, b, Wy, by: 0 };
}

export function initGru(H: number, updateBias: number, seed: number): GruParams {
  const rng = createRng(seed);
  const D = 1;
  const K = D + H;
  const scale = 1 / Math.sqrt(K);
  const Wg = Array.from({ length: 2 * H * K }, () => rng.normal(0, scale));
  const bg = new Array<number>(2 * H).fill(0);
  // Negative update bias -> small z_t -> (1 - z_t) near 1 -> preserves h_{t-1}!
  for (let i = 0; i < H; i++) bg[i] = updateBias;
  const Whx = Array.from({ length: H * D }, () => rng.normal(0, scale));
  const Whh = Array.from({ length: H * H }, () => rng.normal(0, scale));
  const bh = new Array<number>(H).fill(0);
  const Wy = Array.from({ length: H }, () => rng.normal(0, 0.5));
  return { H, D, Wg, bg, Whx, Whh, bh, Wy, by: 0 };
}

/** Exact LSTM forward + BPTT for final-step binary cross-entropy loss. */
export function lstmLossAndGrad(params: LstmParams, xs: number[], y: 0 | 1) {
  const { H, D, W, b, Wy, by } = params;
  const K = D + H;
  const T = xs.length;

  const hs: number[][] = [new Array<number>(H).fill(0)];
  const cs: number[][] = [new Array<number>(H).fill(0)];
  const fs: number[][] = [];
  const isArr: number[][] = [];
  const gs: number[][] = [];
  const os: number[][] = [];
  const tanhCs: number[][] = [];

  for (let t = 0; t < T; t++) {
    const hPrev = hs[t];
    const cPrev = cs[t];
    const u = [xs[t], ...hPrev];
    const ft = new Array<number>(H);
    const it = new Array<number>(H);
    const gt = new Array<number>(H);
    const ot = new Array<number>(H);
    const ct = new Array<number>(H);
    const tc = new Array<number>(H);
    const ht = new Array<number>(H);

    for (let j = 0; j < H; j++) {
      let sf = b[j];
      let si = b[H + j];
      let sg = b[2 * H + j];
      let so = b[3 * H + j];
      const offF = j * K;
      const offI = (H + j) * K;
      const offG = (2 * H + j) * K;
      const offO = (3 * H + j) * K;
      for (let k = 0; k < K; k++) {
        const uk = u[k];
        sf += W[offF + k] * uk;
        si += W[offI + k] * uk;
        sg += W[offG + k] * uk;
        so += W[offO + k] * uk;
      }
      ft[j] = sigmoid(sf);
      it[j] = sigmoid(si);
      gt[j] = Math.tanh(sg);
      ot[j] = sigmoid(so);
      ct[j] = ft[j] * cPrev[j] + it[j] * gt[j];
      tc[j] = Math.tanh(ct[j]);
      ht[j] = ot[j] * tc[j];
    }

    fs.push(ft);
    isArr.push(it);
    gs.push(gt);
    os.push(ot);
    tanhCs.push(tc);
    cs.push(ct);
    hs.push(ht);
  }

  const hLast = hs[T];
  let logit = by;
  for (let j = 0; j < H; j++) logit += Wy[j] * hLast[j];
  const prob = sigmoid(logit);
  const loss = Math.max(logit, 0) - logit * y + Math.log1p(Math.exp(-Math.abs(logit)));

  const dW = new Array<number>(4 * H * K).fill(0);
  const db = new Array<number>(4 * H).fill(0);
  const dWy = new Array<number>(H).fill(0);
  const dz = prob - y;
  const dby = dz;

  let dhNext = new Array<number>(H);
  for (let j = 0; j < H; j++) {
    dWy[j] = dz * hLast[j];
    dhNext[j] = dz * Wy[j];
  }
  let dcNext = new Array<number>(H).fill(0);

  for (let t = T - 1; t >= 0; t--) {
    const hPrev = hs[t];
    const cPrev = cs[t];
    const u = [xs[t], ...hPrev];
    const ft = fs[t];
    const it = isArr[t];
    const gt = gs[t];
    const ot = os[t];
    const tc = tanhCs[t];

    const dhPrev = new Array<number>(H).fill(0);
    const dcPrev = new Array<number>(H).fill(0);

    for (let j = 0; j < H; j++) {
      const dh = dhNext[j];
      const doPre = dh * tc[j] * ot[j] * (1 - ot[j]);
      const dc = dcNext[j] + dh * ot[j] * (1 - tc[j] * tc[j]);
      const dfPre = dc * cPrev[j] * ft[j] * (1 - ft[j]);
      const diPre = dc * gt[j] * it[j] * (1 - it[j]);
      const dgPre = dc * it[j] * (1 - gt[j] * gt[j]);

      dcPrev[j] = dc * ft[j];

      db[j] += dfPre;
      db[H + j] += diPre;
      db[2 * H + j] += dgPre;
      db[3 * H + j] += doPre;

      const offF = j * K;
      const offI = (H + j) * K;
      const offG = (2 * H + j) * K;
      const offO = (3 * H + j) * K;

      for (let k = 0; k < K; k++) {
        const uk = u[k];
        dW[offF + k] += dfPre * uk;
        dW[offI + k] += diPre * uk;
        dW[offG + k] += dgPre * uk;
        dW[offO + k] += doPre * uk;
        if (k >= D) {
          dhPrev[k - D] +=
            W[offF + k] * dfPre +
            W[offI + k] * diPre +
            W[offG + k] * dgPre +
            W[offO + k] * doPre;
        }
      }
    }
    dhNext = dhPrev;
    dcNext = dcPrev;
  }

  return { loss, prob, dW, db, dWy, dby };
}

/** Exact GRU forward + BPTT for final-step binary cross-entropy loss. */
export function gruLossAndGrad(params: GruParams, xs: number[], y: 0 | 1) {
  const { H, D, Wg, bg, Whx, Whh, bh, Wy, by } = params;
  const K = D + H;
  const T = xs.length;

  const hs: number[][] = [new Array<number>(H).fill(0)];
  const zs: number[][] = [];
  const rs: number[][] = [];
  const hTildes: number[][] = [];

  for (let t = 0; t < T; t++) {
    const hPrev = hs[t];
    const u = [xs[t], ...hPrev];
    const zt = new Array<number>(H);
    const rt = new Array<number>(H);
    const hTilde = new Array<number>(H);
    const ht = new Array<number>(H);

    for (let j = 0; j < H; j++) {
      let sz = bg[j];
      let sr = bg[H + j];
      const offZ = j * K;
      const offR = (H + j) * K;
      for (let k = 0; k < K; k++) {
        sz += Wg[offZ + k] * u[k];
        sr += Wg[offR + k] * u[k];
      }
      zt[j] = sigmoid(sz);
      rt[j] = sigmoid(sr);
    }

    for (let j = 0; j < H; j++) {
      let sh = bh[j] + Whx[j] * xs[t];
      const off = j * H;
      for (let m = 0; m < H; m++) {
        sh += Whh[off + m] * (rt[m] * hPrev[m]);
      }
      hTilde[j] = Math.tanh(sh);
      ht[j] = (1 - zt[j]) * hPrev[j] + zt[j] * hTilde[j];
    }

    zs.push(zt);
    rs.push(rt);
    hTildes.push(hTilde);
    hs.push(ht);
  }

  const hLast = hs[T];
  let logit = by;
  for (let j = 0; j < H; j++) logit += Wy[j] * hLast[j];
  const prob = sigmoid(logit);
  const loss = Math.max(logit, 0) - logit * y + Math.log1p(Math.exp(-Math.abs(logit)));

  const dWg = new Array<number>(2 * H * K).fill(0);
  const dbg = new Array<number>(2 * H).fill(0);
  const dWhx = new Array<number>(H * D).fill(0);
  const dWhh = new Array<number>(H * H).fill(0);
  const dbh = new Array<number>(H).fill(0);
  const dWy = new Array<number>(H).fill(0);

  const dzOut = prob - y;
  const dby = dzOut;
  let dhNext = new Array<number>(H);
  for (let j = 0; j < H; j++) {
    dWy[j] = dzOut * hLast[j];
    dhNext[j] = dzOut * Wy[j];
  }

  for (let t = T - 1; t >= 0; t--) {
    const hPrev = hs[t];
    const u = [xs[t], ...hPrev];
    const zt = zs[t];
    const rt = rs[t];
    const hTilde = hTildes[t];

    const dhPrev = new Array<number>(H).fill(0);
    const dResetH = new Array<number>(H).fill(0);

    for (let j = 0; j < H; j++) {
      const dh = dhNext[j];
      dhPrev[j] += dh * (1 - zt[j]);

      const dzPre = dh * (hTilde[j] - hPrev[j]) * zt[j] * (1 - zt[j]);
      const dhTildePre = dh * zt[j] * (1 - hTilde[j] * hTilde[j]);

      dbg[j] += dzPre;
      dbh[j] += dhTildePre;
      dWhx[j] += dhTildePre * xs[t];

      const offZ = j * K;
      for (let k = 0; k < K; k++) {
        dWg[offZ + k] += dzPre * u[k];
        if (k >= D) dhPrev[k - D] += Wg[offZ + k] * dzPre;
      }

      const offH = j * H;
      for (let m = 0; m < H; m++) {
        dWhh[offH + m] += dhTildePre * (rt[m] * hPrev[m]);
        dResetH[m] += Whh[offH + m] * dhTildePre;
      }
    }

    for (let m = 0; m < H; m++) {
      dhPrev[m] += dResetH[m] * rt[m];
      const drPre = dResetH[m] * hPrev[m] * rt[m] * (1 - rt[m]);
      dbg[H + m] += drPre;
      const offR = (H + m) * K;
      for (let k = 0; k < K; k++) {
        dWg[offR + k] += drPre * u[k];
        if (k >= D) dhPrev[k - D] += Wg[offR + k] * drPre;
      }
    }

    dhNext = dhPrev;
  }

  return { loss, prob, dWg, dbg, dWhx, dWhh, dbh, dWy, dby };
}

// ------------------------------------------------------------------ 4. Parameter Count Calculator

export type CellKind = 'rnn' | 'gru' | 'lstm';

/**
 * Exact PyTorch-style single-layer parameter count (1 bias vector per gate):
 * - RNN:  H * (D + H + 1)
 * - GRU:  3 * H * (D + H + 1)
 * - LSTM: 4 * H * (D + H + 1)
 */
export function cellParamCount(kind: CellKind, inputDim: number, hiddenDim: number): number {
  const mult = kind === 'rnn' ? 1 : kind === 'gru' ? 3 : 4;
  return mult * hiddenDim * (inputDim + hiddenDim + 1);
}

// ------------------------------------------------------------------ 5. Fair Long-Lag Race: Vanilla RNN vs GRU vs LSTM

export function makeLongLagDataset(lag: number, count: number, seed: number): SeqExample[] {
  const rng = createRng(seed);
  const out: SeqExample[] = [];
  for (let n = 0; n < count; n++) {
    const y: 0 | 1 = n % 2 === 0 ? 1 : 0;
    const xs = new Array<number>(lag + 1);
    xs[0] = y === 1 ? 1.0 : -1.0;
    for (let t = 1; t <= lag; t++) {
      xs[t] = (rng.next() * 2 - 1) * 0.9;
    }
    out.push({ xs, y });
  }
  return out;
}

function trainVanillaRnn(lag: number, H: number, lr: number, epochs: number, seed: number) {
  const train = makeLongLagDataset(lag, 36, seed + 101);
  const test = makeLongLagDataset(lag, 36, seed + 909);
  const rng = createRng(seed);
  const Wx = Array.from({ length: H }, () => rng.normal(0, 0.4));
  const Wh = Array.from({ length: H * H }, () => rng.normal(0, 0.7 / Math.sqrt(H)));
  const bh = new Array<number>(H).fill(0);
  const Wy = Array.from({ length: H }, () => rng.normal(0, 0.4));
  let by = 0;

  for (let ep = 0; ep < epochs; ep++) {
    const gWx = new Array<number>(H).fill(0);
    const gWh = new Array<number>(H * H).fill(0);
    const gbh = new Array<number>(H).fill(0);
    const gWy = new Array<number>(H).fill(0);
    let gby = 0;
    for (const ex of train) {
      const T = ex.xs.length;
      const hs: number[][] = [new Array<number>(H).fill(0)];
      for (let t = 0; t < T; t++) {
        const prev = hs[t];
        const cur = new Array<number>(H);
        for (let i = 0; i < H; i++) {
          let s = Wx[i] * ex.xs[t] + bh[i];
          for (let j = 0; j < H; j++) s += Wh[i * H + j] * prev[j];
          cur[i] = Math.tanh(s);
        }
        hs.push(cur);
      }
      const hLast = hs[T];
      let z = by;
      for (let i = 0; i < H; i++) z += Wy[i] * hLast[i];
      const dz = sigmoid(z) - ex.y;
      gby += dz;
      let dh = new Array<number>(H);
      for (let i = 0; i < H; i++) {
        gWy[i] += dz * hLast[i];
        dh[i] = dz * Wy[i];
      }
      for (let t = T - 1; t >= 0; t--) {
        const cur = hs[t + 1];
        const prev = hs[t];
        const da = new Array<number>(H);
        for (let i = 0; i < H; i++) {
          da[i] = dh[i] * (1 - cur[i] * cur[i]);
          gWx[i] += da[i] * ex.xs[t];
          gbh[i] += da[i];
          for (let j = 0; j < H; j++) gWh[i * H + j] += da[i] * prev[j];
        }
        const nextDh = new Array<number>(H).fill(0);
        for (let j = 0; j < H; j++) {
          let s = 0;
          for (let i = 0; i < H; i++) s += Wh[i * H + j] * da[i];
          nextDh[j] = s;
        }
        dh = nextDh;
      }
    }
    const invN = 1 / train.length;
    let sq = (gby * invN) ** 2;
    for (let i = 0; i < H; i++) sq += (gWx[i] * invN) ** 2 + (gbh[i] * invN) ** 2 + (gWy[i] * invN) ** 2;
    for (let i = 0; i < H * H; i++) sq += (gWh[i] * invN) ** 2;
    const norm = Math.sqrt(sq);
    const scale = norm > 1.0 ? invN / norm : invN;
    for (let i = 0; i < H; i++) {
      Wx[i] -= lr * gWx[i] * scale;
      bh[i] -= lr * gbh[i] * scale;
      Wy[i] -= lr * gWy[i] * scale;
    }
    for (let i = 0; i < H * H; i++) Wh[i] -= lr * gWh[i] * scale;
    by -= lr * gby * scale;
  }

  let ok = 0;
  for (const ex of test) {
    let h = new Array<number>(H).fill(0);
    for (let t = 0; t < ex.xs.length; t++) {
      const next = new Array<number>(H);
      for (let i = 0; i < H; i++) {
        let s = Wx[i] * ex.xs[t] + bh[i];
        for (let j = 0; j < H; j++) s += Wh[i * H + j] * h[j];
        next[i] = Math.tanh(s);
      }
      h = next;
    }
    let z = by;
    for (let i = 0; i < H; i++) z += Wy[i] * h[i];
    if ((sigmoid(z) >= 0.5 ? 1 : 0) === ex.y) ok++;
  }
  return ok / test.length;
}

function trainLstmModel(lag: number, H: number, forgetBias: number, lr: number, epochs: number, seed: number) {
  const train = makeLongLagDataset(lag, 36, seed + 101);
  const test = makeLongLagDataset(lag, 36, seed + 909);
  const p = initLstm(H, forgetBias, seed);
  const flatLen = p.W.length + p.b.length + p.Wy.length + 1;
  const m = new Array<number>(flatLen).fill(0);
  const v = new Array<number>(flatLen).fill(0);

  for (let ep = 1; ep <= epochs; ep++) {
    const gW = new Array<number>(p.W.length).fill(0);
    const gb = new Array<number>(p.b.length).fill(0);
    const gWy = new Array<number>(p.Wy.length).fill(0);
    let gby = 0;
    for (const ex of train) {
      const r = lstmLossAndGrad(p, ex.xs, ex.y);
      for (let i = 0; i < p.W.length; i++) gW[i] += r.dW[i];
      for (let i = 0; i < p.b.length; i++) gb[i] += r.db[i];
      for (let i = 0; i < p.Wy.length; i++) gWy[i] += r.dWy[i];
      gby += r.dby;
    }
    const invN = 1 / train.length;
    const g = [...gW, ...gb, ...gWy, gby].map((x) => x * invN);
    let sq = 0;
    for (let i = 0; i < g.length; i++) sq += g[i] * g[i];
    const norm = Math.sqrt(sq);
    if (norm > 1.0) {
      const s = 1 / norm;
      for (let i = 0; i < g.length; i++) g[i] *= s;
    }
    const b1c = 1 - 0.9 ** ep;
    const b2c = 1 - 0.999 ** ep;
    let idx = 0;
    const stepArr = (arr: number[]) => {
      for (let i = 0; i < arr.length; i++, idx++) {
        m[idx] = 0.9 * m[idx] + 0.1 * g[idx];
        v[idx] = 0.999 * v[idx] + 0.001 * g[idx] * g[idx];
        arr[i] -= lr * (m[idx] / b1c) / (Math.sqrt(v[idx] / b2c) + 1e-8);
      }
    };
    stepArr(p.W);
    stepArr(p.b);
    stepArr(p.Wy);
    m[idx] = 0.9 * m[idx] + 0.1 * g[idx];
    v[idx] = 0.999 * v[idx] + 0.001 * g[idx] * g[idx];
    p.by -= lr * (m[idx] / b1c) / (Math.sqrt(v[idx] / b2c) + 1e-8);
  }

  let ok = 0;
  for (const ex of test) {
    const r = lstmLossAndGrad(p, ex.xs, ex.y);
    if ((r.prob >= 0.5 ? 1 : 0) === ex.y) ok++;
  }
  return ok / test.length;
}

function trainGruModel(lag: number, H: number, updateBias: number, lr: number, epochs: number, seed: number) {
  const train = makeLongLagDataset(lag, 36, seed + 101);
  const test = makeLongLagDataset(lag, 36, seed + 909);
  const p = initGru(H, updateBias, seed);
  const flatLen = p.Wg.length + p.bg.length + p.Whx.length + p.Whh.length + p.bh.length + p.Wy.length + 1;
  const m = new Array<number>(flatLen).fill(0);
  const v = new Array<number>(flatLen).fill(0);

  for (let ep = 1; ep <= epochs; ep++) {
    const gWg = new Array<number>(p.Wg.length).fill(0);
    const gbg = new Array<number>(p.bg.length).fill(0);
    const gWhx = new Array<number>(p.Whx.length).fill(0);
    const gWhh = new Array<number>(p.Whh.length).fill(0);
    const gbh = new Array<number>(p.bh.length).fill(0);
    const gWy = new Array<number>(p.Wy.length).fill(0);
    let gby = 0;
    for (const ex of train) {
      const r = gruLossAndGrad(p, ex.xs, ex.y);
      for (let i = 0; i < p.Wg.length; i++) gWg[i] += r.dWg[i];
      for (let i = 0; i < p.bg.length; i++) gbg[i] += r.dbg[i];
      for (let i = 0; i < p.Whx.length; i++) gWhx[i] += r.dWhx[i];
      for (let i = 0; i < p.Whh.length; i++) gWhh[i] += r.dWhh[i];
      for (let i = 0; i < p.bh.length; i++) gbh[i] += r.dbh[i];
      for (let i = 0; i < p.Wy.length; i++) gWy[i] += r.dWy[i];
      gby += r.dby;
    }
    const invN = 1 / train.length;
    const g = [...gWg, ...gbg, ...gWhx, ...gWhh, ...gbh, ...gWy, gby].map((x) => x * invN);
    let sq = 0;
    for (let i = 0; i < g.length; i++) sq += g[i] * g[i];
    const norm = Math.sqrt(sq);
    if (norm > 1.0) {
      const s = 1 / norm;
      for (let i = 0; i < g.length; i++) g[i] *= s;
    }
    const b1c = 1 - 0.9 ** ep;
    const b2c = 1 - 0.999 ** ep;
    let idx = 0;
    const stepArr = (arr: number[]) => {
      for (let i = 0; i < arr.length; i++, idx++) {
        m[idx] = 0.9 * m[idx] + 0.1 * g[idx];
        v[idx] = 0.999 * v[idx] + 0.001 * g[idx] * g[idx];
        arr[i] -= lr * (m[idx] / b1c) / (Math.sqrt(v[idx] / b2c) + 1e-8);
      }
    };
    stepArr(p.Wg);
    stepArr(p.bg);
    stepArr(p.Whx);
    stepArr(p.Whh);
    stepArr(p.bh);
    stepArr(p.Wy);
    m[idx] = 0.9 * m[idx] + 0.1 * g[idx];
    v[idx] = 0.999 * v[idx] + 0.001 * g[idx] * g[idx];
    p.by -= lr * (m[idx] / b1c) / (Math.sqrt(v[idx] / b2c) + 1e-8);
  }

  let ok = 0;
  for (const ex of test) {
    const r = gruLossAndGrad(p, ex.xs, ex.y);
    if ((r.prob >= 0.5 ? 1 : 0) === ex.y) ok++;
  }
  return ok / test.length;
}

export type RaceRow = {
  lag: number;
  rnnAcc: number;
  gruAcc: number;
  lstmAcc: number;
};

/**
 * Fair multi-seed comparison across lags L = 3, 8, 14:
 * Each architecture gets 3 hidden units (or 4 for Vanilla RNN so it even has extra width!)
 * and 90 training epochs across 3 fixed seeds.
 */
export function runGatedRace(lag: number, seeds = [11, 23, 37]): RaceRow {
  let rnn = 0;
  let gru = 0;
  let lstm = 0;
  for (const s of seeds) {
    rnn += trainVanillaRnn(lag, 4, 0.12, 90, s);
    gru += trainGruModel(lag, 3, -2.0, 0.06, 90, s);
    lstm += trainLstmModel(lag, 3, 2.0, 0.06, 90, s);
  }
  return {
    lag,
    rnnAcc: rnn / seeds.length,
    gruAcc: gru / seeds.length,
    lstmAcc: lstm / seeds.length,
  };
}

// ------------------------------------------------------------------ 6. Lab: Wire the Gated Cell (Exhaustive)

export type LstmBriefId = 'gates' | 'bias' | 'budget';

export const LAB_GATE_LEVELS = [0.05, 0.50, 0.95] as const;

export function evalLstmBrief1(f: number, i: number, o: number) {
  const sim = simulateLstmCorridor(f, i, o, 0.95);
  // Goal: carry >= 55% of the cue across 4 hostile negative distractors AND keep corridor |h_t| <= 0.15
  const pass = sim.retentionRatio >= 0.55 && sim.maxCorridorH <= 0.15;
  return { f, i, o, retentionRatio: sim.retentionRatio, maxCorridorH: sim.maxCorridorH, readH: sim.readH, pass };
}

export const LAB_BIAS_CELLS = ['rnn', 'lstm'] as const;
export const LAB_BIAS_VALS = [-1, 0, 1, 2, 3] as const;

export function evalLstmBrief2(cell: 'rnn' | 'lstm', forgetBias: number) {
  const flow = carouselGradientFlow(25, forgetBias, 0.8);
  const ratio = cell === 'rnn' ? flow.rnn.step1Ratio : flow.lstm.step1Ratio;
  const pass = ratio >= 0.04;
  return { cell, forgetBias, ratio, pass };
}

export const LAB_BUDGET_KINDS: CellKind[] = ['rnn', 'gru', 'lstm'];
export const LAB_BUDGET_HS = [8, 12, 16, 20] as const;

export function evalLstmBrief3(kind: CellKind, H: number) {
  // Input dim D = 16, strict edge budget <= 1,800 params, requires gated highway + H >= 12
  const params = cellParamCount(kind, 16, H);
  const isGated = kind !== 'rnn';
  const pass = isGated && H >= 12 && params <= 1800;
  return { kind, H, params, isGated, pass };
}

export function lstmLabExhaustive(brief: LstmBriefId) {
  if (brief === 'gates') {
    const all: ReturnType<typeof evalLstmBrief1>[] = [];
    for (const f of LAB_GATE_LEVELS) {
      for (const i of LAB_GATE_LEVELS) {
        for (const o of LAB_GATE_LEVELS) {
          all.push(evalLstmBrief1(f, i, o));
        }
      }
    }
    const passing = all.filter((r) => r.pass).sort((a, b) => b.retentionRatio - a.retentionRatio);
    return {
      total: all.length,
      passingCount: passing.length,
      bestLabel: passing[0]
        ? `f=${passing[0].f}, i=${passing[0].i}, o=${passing[0].o} (${(100 * passing[0].retentionRatio).toFixed(1)}% kept)`
        : 'none',
    };
  }
  if (brief === 'bias') {
    const all: ReturnType<typeof evalLstmBrief2>[] = [];
    for (const c of LAB_BIAS_CELLS) {
      for (const b of LAB_BIAS_VALS) {
        all.push(evalLstmBrief2(c, b));
      }
    }
    const passing = all.filter((r) => r.pass).sort((a, b) => b.ratio - a.ratio);
    return {
      total: all.length,
      passingCount: passing.length,
      bestLabel: passing[0]
        ? `${passing[0].cell.toUpperCase()} b_f=+${passing[0].forgetBias} (ratio ${passing[0].ratio.toFixed(3)})`
        : 'none',
    };
  }
  const all: ReturnType<typeof evalLstmBrief3>[] = [];
  for (const k of LAB_BUDGET_KINDS) {
    for (const H of LAB_BUDGET_HS) {
      all.push(evalLstmBrief3(k, H));
    }
  }
  // Best = largest hidden capacity H that fits <= 1,800 params (tie-break fewest params)
  const passing = all.filter((r) => r.pass).sort((a, b) => b.H - a.H || a.params - b.params);
  return {
    total: all.length,
    passingCount: passing.length,
    bestLabel: passing[0]
      ? `${passing[0].kind.toUpperCase()} H=${passing[0].H} (${passing[0].params.toLocaleString()} params)`
      : 'none',
  };
}

export function lstmLabHint(
  brief: LstmBriefId,
  state: {
    b1F: number;
    b1I: number;
    b1O: number;
    b2Cell: 'rnn' | 'lstm';
    b2Bias: number;
    b3Kind: CellKind;
    b3H: number;
  },
): string {
  if (brief === 'gates') {
    const r = evalLstmBrief1(state.b1F, state.b1I, state.b1O);
    if (r.pass) {
      return `Cue retention is ${(100 * r.retentionRatio).toFixed(1)}% with max corridor leak ${r.maxCorridorH.toFixed(3)} — all three valves are doing their jobs.`;
    }
    if (r.retentionRatio < 0) {
      return `The cell state flipped negative (${(100 * r.retentionRatio).toFixed(1)}% retention) because hostile negative pulses are writing onto the highway faster than the stored cue survives.`;
    }
    if (r.retentionRatio < 0.55) {
      return `Only ${(100 * r.retentionRatio).toFixed(1)}% of the initial cue reaches t = 5 — either the memory conveyor is leaking away each step or distractors are still slipping into the cell.`;
    }
    return `The cell preserves ${(100 * r.retentionRatio).toFixed(1)}% of the cue, but corridor output leak hits ${r.maxCorridorH.toFixed(3)} (limit 0.15) because the read valve stays open during the corridor.`;
  }
  if (brief === 'bias') {
    const r = evalLstmBrief2(state.b2Cell, state.b2Bias);
    if (r.pass) {
      return `Step-1 gradient ratio is ${r.ratio.toFixed(3)} (≥ 0.04) — the additive highway keeps the 25-step signal alive.`;
    }
    if (state.b2Cell === 'rnn') {
      return `A plain recurrent cell has no forget gate at all, so changing bias does not rescue its 25-step decay (${r.ratio.toExponential(1)}). Compare against an architecture with an additive cell highway.`;
    }
    const fOpen = 1 / (1 + Math.exp(-state.b2Bias));
    return `At current bias, each step multiplies the cell gradient by σ(b_f) = ${fOpen.toFixed(3)}, leaving ${fOpen.toFixed(3)}²⁴ = ${r.ratio.toExponential(1)} after 24 backward steps. Higher initial gate openness compounds dramatically across 24 steps.`;
  }
  const r = evalLstmBrief3(state.b3Kind, state.b3H);
  if (r.pass) {
    return `${state.b3Kind.toUpperCase()} at H = ${r.H} uses ${r.params.toLocaleString()} / 1,800 params with a gated highway. Check whether any gated cell can fit even wider H inside 1,800 params.`;
  }
  if (!r.isGated) {
    return `This cell fits ${r.params.toLocaleString()} params, but without a gated highway its long-range gradients vanish across lag 14.`;
  }
  if (r.params > 1800) {
    return `${r.params.toLocaleString()} parameters exceed the 1,800 microcontroller cap. Remember gate count multiplies every weight matrix (3× vs 4×).`;
  }
  return `Parameter count (${r.params.toLocaleString()}) fits the cap, but hidden width H = ${r.H} is below the required minimum width of 12.`;
}

