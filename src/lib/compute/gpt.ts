// authored-by: Claude Opus 5.5
/**
 * A tiny GPT, CPU reference tier (V1 capstone, step A1).
 *
 * Architecture (GPT-2 style, pre-norm):
 *   x₀ = token embedding + position embedding
 *   for each of L blocks:
 *     x = x + Attn(LN₁(x))      causal multi-head self-attention, H heads of width d/H
 *     x = x + MLP(LN₂(x))       d → F (= 4d) → d, GELU (tanh form)
 *   logits = LN_f(x) · W_teᵀ    output layer tied to the token embedding, no bias
 *   loss = mean cross-entropy of the next token
 * trained with AdamW (decoupled weight decay on matrices only, as nanoGPT does) and global-norm
 * gradient clipping.
 *
 * Plain loops over typed arrays, written like `cpuBackend.ts`. The precision is a choice:
 * float32 (the default, what the browser runs) or float64 (what the finite-difference tests use).
 * Every transcendental goes through `detmath.ts`, so a float32 run is bit-identical in Node and in
 * any browser. Every backward pass is checked against central finite differences in `gpt.test.ts`.
 */
import { createRng } from '../rng';
import { dexp, dtanh, irwinHall } from './detmath';

export type GptConfig = {
  /** Vocabulary size V. */
  vocab: number;
  /** Context length: the longest sequence the position table covers. */
  ctx: number;
  /** Model width d (must be divisible by heads). */
  d: number;
  heads: number;
  layers: number;
  /** MLP hidden width as a multiple of d (GPT-2 uses 4). */
  ff?: number;
};

export type AdamWOpts = {
  beta1?: number;
  beta2?: number;
  eps?: number;
  /** Decoupled weight decay, applied to matrices (embeddings and linear weights) only. */
  weightDecay?: number;
  /** Clip the global gradient norm to this value (0 or undefined: no clipping). */
  clip?: number;
};

type Arr = Float32Array | Float64Array;

export type BlockLayout = {
  ln1g: number; ln1b: number; wqkv: number; bqkv: number; wo: number; bo: number;
  ln2g: number; ln2b: number; w1: number; b1: number; w2: number; b2: number;
};
export type GptLayout = {
  wte: number;
  wpe: number;
  blocks: BlockLayout[];
  lnfg: number;
  lnfb: number;
  total: number;
  /** [offset, length] of every matrix (the tensors weight decay applies to). */
  decay: [number, number][];
  /** Named tensors, for the lesson's parameter table: [name, offset, length]. */
  named: [string, number, number][];
};

export function gptLayout(cfg: GptConfig): GptLayout {
  const { vocab: V, ctx: T, d, layers: L } = cfg;
  const F = (cfg.ff ?? 4) * d;
  if (d % cfg.heads) throw new Error('d must be divisible by heads');
  let o = 0;
  const named: [string, number, number][] = [];
  const decay: [number, number][] = [];
  const take = (name: string, n: number, isMatrix = false) => {
    const s = o;
    o += n;
    named.push([name, s, n]);
    if (isMatrix) decay.push([s, n]);
    return s;
  };
  const wte = take('token embedding', V * d, true);
  const wpe = take('position embedding', T * d, true);
  const blocks: BlockLayout[] = [];
  for (let l = 0; l < L; l++) {
    const p = `block ${l + 1} `;
    blocks.push({
      ln1g: take(p + 'LN₁ gain', d),
      ln1b: take(p + 'LN₁ shift', d),
      wqkv: take(p + 'W_QKV', 3 * d * d, true),
      bqkv: take(p + 'b_QKV', 3 * d),
      wo: take(p + 'W_O', d * d, true),
      bo: take(p + 'b_O', d),
      ln2g: take(p + 'LN₂ gain', d),
      ln2b: take(p + 'LN₂ shift', d),
      w1: take(p + 'MLP W₁', F * d, true),
      b1: take(p + 'MLP b₁', F),
      w2: take(p + 'MLP W₂', d * F, true),
      b2: take(p + 'MLP b₂', d),
    });
  }
  const lnfg = take('final LN gain', d);
  const lnfb = take('final LN shift', d);
  return { wte, wpe, blocks, lnfg, lnfb, total: o, decay, named };
}

export const gptParamCount = (cfg: GptConfig) => gptLayout(cfg).total;

// ------------------------------------------------------------------ kernels

const SQRT_2_PI = Math.sqrt(2 / Math.PI);
const gelu = (z: number) => 0.5 * z * (1 + dtanh(SQRT_2_PI * (z + 0.044715 * z * z * z)));
const geluGrad = (z: number) => {
  const t = dtanh(SQRT_2_PI * (z + 0.044715 * z * z * z));
  return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * SQRT_2_PI * (1 + 3 * 0.044715 * z * z);
};

let rowBuf = new Float64Array(0);
let rowBuf2 = new Float64Array(0);
const rowScratch = (n: number) => {
  if (rowBuf.length < n) {
    rowBuf = new Float64Array(n);
    rowBuf2 = new Float64Array(n);
  }
};

/**
 * Y[n,o] = b[o] + Σᵢ X[n,i]·W[o,i]   (W row-major [dout × din], starting at wOff in P).
 * Four outputs per pass over the input row (register blocking, about 3× faster in V8); each sum
 * still starts from the bias and adds the products in order of i, so the result is the same as the
 * plain loop's to the last bit.
 */
function linFwd(X: Arr, N: number, din: number, P: Arr, wOff: number, bOff: number, dout: number, Y: Arr) {
  rowScratch(din);
  const xr = rowBuf;
  for (let n = 0; n < N; n++) {
    const xo = n * din;
    const yo = n * dout;
    for (let i = 0; i < din; i++) xr[i] = X[xo + i];
    let o = 0;
    for (; o + 3 < dout; o += 4) {
      let a0 = bOff >= 0 ? P[bOff + o] : 0;
      let a1 = bOff >= 0 ? P[bOff + o + 1] : 0;
      let a2 = bOff >= 0 ? P[bOff + o + 2] : 0;
      let a3 = bOff >= 0 ? P[bOff + o + 3] : 0;
      const w0 = wOff + o * din;
      const w1 = w0 + din;
      const w2 = w1 + din;
      const w3 = w2 + din;
      for (let i = 0; i < din; i++) {
        const x = xr[i];
        a0 += x * P[w0 + i];
        a1 += x * P[w1 + i];
        a2 += x * P[w2 + i];
        a3 += x * P[w3 + i];
      }
      Y[yo + o] = a0;
      Y[yo + o + 1] = a1;
      Y[yo + o + 2] = a2;
      Y[yo + o + 3] = a3;
    }
    for (; o < dout; o++) {
      let s = bOff >= 0 ? P[bOff + o] : 0;
      const wo = wOff + o * din;
      for (let i = 0; i < din; i++) s += xr[i] * P[wo + i];
      Y[yo + o] = s;
    }
  }
}

/** Backward of linFwd: dW += dYᵀ·X, db += Σₙ dY, and (if dX) dX = dY·W (overwritten). */
function linBwd(dY: Arr, X: Arr, N: number, din: number, P: Arr, G: Arr, wOff: number, bOff: number, dout: number, dX: Arr | null) {
  rowScratch(din);
  const xr = rowBuf;
  const dxr = rowBuf2;
  for (let n = 0; n < N; n++) {
    const xo = n * din;
    const yo = n * dout;
    for (let i = 0; i < din; i++) {
      xr[i] = X[xo + i];
      dxr[i] = 0;
    }
    let o = 0;
    for (; o + 3 < dout; o += 4) {
      const g0 = dY[yo + o];
      const g1 = dY[yo + o + 1];
      const g2 = dY[yo + o + 2];
      const g3 = dY[yo + o + 3];
      const w0 = wOff + o * din;
      const w1 = w0 + din;
      const w2 = w1 + din;
      const w3 = w2 + din;
      if (bOff >= 0) {
        G[bOff + o] += g0;
        G[bOff + o + 1] += g1;
        G[bOff + o + 2] += g2;
        G[bOff + o + 3] += g3;
      }
      for (let i = 0; i < din; i++) {
        const x = xr[i];
        G[w0 + i] += g0 * x;
        G[w1 + i] += g1 * x;
        G[w2 + i] += g2 * x;
        G[w3 + i] += g3 * x;
      }
      if (dX) for (let i = 0; i < din; i++) dxr[i] += g0 * P[w0 + i] + g1 * P[w1 + i] + g2 * P[w2 + i] + g3 * P[w3 + i];
    }
    for (; o < dout; o++) {
      const g = dY[yo + o];
      const wo = wOff + o * din;
      if (bOff >= 0) G[bOff + o] += g;
      for (let i = 0; i < din; i++) G[wo + i] += g * xr[i];
      if (dX) for (let i = 0; i < din; i++) dxr[i] += g * P[wo + i];
    }
    if (dX) for (let i = 0; i < din; i++) dX[xo + i] = dxr[i];
  }
}

/** Row-wise layer norm (ε = 1e-5) into Y, keeping x̂ and 1/σ for the backward pass. */
function lnFwd(X: Arr, N: number, d: number, P: Arr, gOff: number, bOff: number, Y: Arr, Xh: Arr, Rs: Arr) {
  for (let n = 0; n < N; n++) {
    const o = n * d;
    let mean = 0;
    for (let j = 0; j < d; j++) mean += X[o + j];
    mean /= d;
    let v = 0;
    for (let j = 0; j < d; j++) {
      const c = X[o + j] - mean;
      v += c * c;
    }
    v /= d;
    const r = 1 / Math.sqrt(v + 1e-5);
    Rs[n] = r;
    for (let j = 0; j < d; j++) {
      const h = (X[o + j] - mean) * r;
      Xh[o + j] = h;
      Y[o + j] = P[gOff + j] * h + P[bOff + j];
    }
  }
}

/** Backward of lnFwd: accumulates dγ, dβ into G and ADDS dX into dXacc. */
function lnBwd(dY: Arr, N: number, d: number, P: Arr, G: Arr, gOff: number, bOff: number, Xh: Arr, Rs: Arr, dXacc: Arr) {
  for (let n = 0; n < N; n++) {
    const o = n * d;
    let s1 = 0;
    let s2 = 0;
    for (let j = 0; j < d; j++) {
      const g = dY[o + j];
      const h = Xh[o + j];
      G[gOff + j] += g * h;
      G[bOff + j] += g;
      const gh = g * P[gOff + j];
      s1 += gh;
      s2 += gh * h;
    }
    const r = Rs[n] / d;
    for (let j = 0; j < d; j++) dXacc[o + j] += r * (d * dY[o + j] * P[gOff + j] - s1 - Xh[o + j] * s2);
  }
}

// ------------------------------------------------------------------ the model

type LayerCache = {
  xin: Arr; xh1: Arr; rs1: Arr; a1: Arr; qkv: Arr; att: Arr; y: Arr;
  xmid: Arr; xh2: Arr; rs2: Arr; a2: Arr; hpre: Arr; hact: Arr;
};

type Workspace = {
  B: number; T: number;
  /** Capacity: rows (B·T), attention cells (B·T·T) and the longest T it was sized for. */
  capN: number; capAtt: number; capT: number;
  x0: Arr; layers: LayerCache[]; xL: Arr; xhf: Arr; rsf: Arr; af: Arr; logits: Arr;
  // backward scratch
  dx: Arr; dtmp: Arr; dF: Arr; dqkv: Arr; dP: Float64Array;
};

export type GenerateOpts = {
  temperature?: number;
  /** Nucleus sampling: keep the smallest set of tokens whose probabilities add up to at least topP. */
  topP?: number;
  seed?: number;
};

export class Gpt {
  readonly cfg: GptConfig;
  readonly layout: GptLayout;
  readonly F: number;
  readonly hd: number;
  readonly precision: 'f32' | 'f64';
  /** Parameters, gradients and AdamW moments, all in the flat layout of `gptLayout`. */
  p: Arr;
  g: Arr;
  m: Arr;
  v: Arr;
  /** AdamW step count and the running bias corrections β₁ᵗ, β₂ᵗ (kept by multiplication, so exact). */
  t = 0;
  private b1t = 1;
  private b2t = 1;
  private ws: Workspace | null = null;
  private decayMask: Uint8Array;
  private tokens: Int32Array = new Int32Array(0);
  private targets: Int32Array = new Int32Array(0);

  constructor(cfg: GptConfig, precision: 'f32' | 'f64' = 'f32') {
    this.cfg = cfg;
    this.precision = precision;
    this.layout = gptLayout(cfg);
    this.F = (cfg.ff ?? 4) * cfg.d;
    this.hd = cfg.d / cfg.heads;
    const A = precision === 'f32' ? Float32Array : Float64Array;
    const n = this.layout.total;
    this.p = new A(n);
    this.g = new A(n);
    this.m = new A(n);
    this.v = new A(n);
    this.decayMask = new Uint8Array(n);
    for (const [o, len] of this.layout.decay) this.decayMask.fill(1, o, o + len);
  }

  private arr(n: number): Arr {
    return this.precision === 'f32' ? new Float32Array(n) : new Float64Array(n);
  }

  /**
   * GPT-2 initialisation: weights ~ N(0, 0.02²) (here the Irwin–Hall approximation of a normal),
   * the two residual projections (W_O and MLP W₂) scaled down by 1/√(2L), biases 0, LN gains 1.
   * Resets the optimiser.
   */
  init(seed: number) {
    const rng = createRng(seed);
    const { p, layout, cfg } = this;
    p.fill(0);
    const fill = (o: number, n: number, std: number) => {
      for (let i = 0; i < n; i++) p[o + i] = std * irwinHall(rng.next);
    };
    const { vocab: V, ctx: T, d } = cfg;
    const F = this.F;
    fill(layout.wte, V * d, 0.02);
    fill(layout.wpe, T * d, 0.02);
    const proj = 0.02 / Math.sqrt(2 * cfg.layers);
    for (const b of layout.blocks) {
      p.fill(1, b.ln1g, b.ln1g + d);
      p.fill(1, b.ln2g, b.ln2g + d);
      fill(b.wqkv, 3 * d * d, 0.02);
      fill(b.wo, d * d, proj);
      fill(b.w1, F * d, 0.02);
      fill(b.w2, d * F, proj);
    }
    p.fill(1, layout.lnfg, layout.lnfg + d);
    this.resetOptimiser();
  }

  resetOptimiser() {
    this.m.fill(0);
    this.v.fill(0);
    this.t = 0;
    this.b1t = 1;
    this.b2t = 1;
  }

  private workspace(B: number, T: number): Workspace {
    if (T > this.cfg.ctx) throw new Error(`sequence of ${T} exceeds the context of ${this.cfg.ctx}`);
    const w = this.ws;
    if (w && w.capN >= B * T && w.capAtt >= B * T * T && w.capT >= T) return w;
    // grow to cover both the old and the new shape, so training and generation don't thrash
    const N = Math.max(B * T, w?.capN ?? 0);
    const capAtt = Math.max(B * T * T, w?.capAtt ?? 0);
    const capT = Math.max(T, w?.capT ?? 0);
    const { d, heads: H, layers: L, vocab: V } = this.cfg;
    const F = this.F;
    const A = (n: number) => this.arr(n);
    const layers: LayerCache[] = [];
    for (let l = 0; l < L; l++)
      layers.push({
        xin: A(N * d), xh1: A(N * d), rs1: A(N), a1: A(N * d), qkv: A(N * 3 * d), att: A(H * capAtt), y: A(N * d),
        xmid: A(N * d), xh2: A(N * d), rs2: A(N), a2: A(N * d), hpre: A(N * F), hact: A(N * F),
      });
    this.ws = {
      B, T, capN: N, capAtt, capT, x0: A(N * d), layers, xL: A(N * d), xhf: A(N * d), rsf: A(N), af: A(N * d), logits: A(N * V),
      dx: A(N * d), dtmp: A(N * d), dF: A(N * F), dqkv: A(N * 3 * d), dP: new Float64Array(capT),
    };
    return this.ws;
  }

  /**
   * Forward pass over B sequences of T tokens (row-major [B × T]). With targets (−1 = ignore),
   * returns the mean next-token cross-entropy and leaves softmax probabilities in the workspace for
   * `backward`; without, returns NaN and leaves the logits. Attention weights are kept per layer.
   */
  forward(tokens: Int32Array, B: number, T: number, targets?: Int32Array): number {
    const ws = this.workspace(B, T);
    ws.B = B;
    ws.T = T;
    this.tokens = tokens;
    this.targets = targets ?? new Int32Array(0);
    const { p, layout, cfg, F, hd } = this;
    const { d, heads: H, vocab: V } = cfg;
    const N = B * T;
    const x0 = ws.x0;
    for (let b = 0; b < B; b++)
      for (let t = 0; t < T; t++) {
        const n = b * T + t;
        const tok = tokens[n];
        const eo = layout.wte + tok * d;
        const po = layout.wpe + t * d;
        for (let j = 0; j < d; j++) x0[n * d + j] = p[eo + j] + p[po + j];
      }
    let x = x0;
    const scale = 1 / Math.sqrt(hd);
    const d3 = 3 * d;
    for (let l = 0; l < cfg.layers; l++) {
      const c = ws.layers[l];
      const bl = layout.blocks[l];
      c.xin.set(x.subarray(0, N * d));
      lnFwd(c.xin, N, d, p, bl.ln1g, bl.ln1b, c.a1, c.xh1, c.rs1);
      linFwd(c.a1, N, d, p, bl.wqkv, bl.bqkv, d3, c.qkv);
      // causal attention, head by head
      const { qkv, att, y } = c;
      for (let b = 0; b < B; b++)
        for (let h = 0; h < H; h++) {
          const ab = (b * H + h) * T * T;
          for (let i = 0; i < T; i++) {
            const qo = (b * T + i) * d3 + h * hd;
            const row = ab + i * T;
            let mx = -Infinity;
            for (let j = 0; j <= i; j++) {
              const ko = (b * T + j) * d3 + d + h * hd;
              let s = 0;
              for (let e = 0; e < hd; e++) s += qkv[qo + e] * qkv[ko + e];
              s *= scale;
              att[row + j] = s;
              if (s > mx) mx = s;
            }
            let sum = 0;
            for (let j = 0; j <= i; j++) {
              const e = dexp(att[row + j] - mx);
              att[row + j] = e;
              sum += e;
            }
            for (let j = 0; j <= i; j++) att[row + j] /= sum;
            for (let j = i + 1; j < T; j++) att[row + j] = 0;
            const yo = (b * T + i) * d + h * hd;
            for (let e = 0; e < hd; e++) y[yo + e] = 0;
            for (let j = 0; j <= i; j++) {
              const a = att[row + j];
              const vo = (b * T + j) * d3 + 2 * d + h * hd;
              for (let e = 0; e < hd; e++) y[yo + e] += a * qkv[vo + e];
            }
          }
        }
      linFwd(y, N, d, p, bl.wo, bl.bo, d, c.xmid);
      for (let i = 0; i < N * d; i++) c.xmid[i] += c.xin[i];
      lnFwd(c.xmid, N, d, p, bl.ln2g, bl.ln2b, c.a2, c.xh2, c.rs2);
      linFwd(c.a2, N, d, p, bl.w1, bl.b1, F, c.hpre);
      for (let i = 0; i < N * F; i++) c.hact[i] = gelu(c.hpre[i]);
      const out = l + 1 < cfg.layers ? ws.layers[l + 1].xin : ws.xL;
      linFwd(c.hact, N, F, p, bl.w2, bl.b2, d, out);
      for (let i = 0; i < N * d; i++) out[i] += c.xmid[i];
      x = out;
    }
    if (cfg.layers === 0) ws.xL.set(x0.subarray(0, N * d));
    lnFwd(ws.xL, N, d, p, layout.lnfg, layout.lnfb, ws.af, ws.xhf, ws.rsf);
    // tied output layer: logits = a_f · W_teᵀ
    linFwd(ws.af, N, d, p, layout.wte, -1, V, ws.logits);
    if (!targets) return NaN;
    let loss = 0;
    let count = 0;
    const L = ws.logits;
    for (let n = 0; n < N; n++) {
      const o = n * V;
      let mx = -Infinity;
      for (let k = 0; k < V; k++) if (L[o + k] > mx) mx = L[o + k];
      let s = 0;
      for (let k = 0; k < V; k++) {
        const e = dexp(L[o + k] - mx);
        L[o + k] = e;
        s += e;
      }
      for (let k = 0; k < V; k++) L[o + k] /= s;
      const y = targets[n];
      if (y >= 0) {
        loss -= Math.log(Math.max(L[o + y], 1e-30));
        count++;
      }
    }
    return count ? loss / count : 0;
  }

  /** Backward pass after `forward(…, targets)`: overwrites `g` with ∂loss/∂parameters. */
  backward() {
    const ws = this.ws!;
    const { B, T } = ws;
    const { p, g, layout, cfg, F, hd } = this;
    const { d, heads: H, vocab: V } = cfg;
    const N = B * T;
    const d3 = 3 * d;
    g.fill(0);
    const targets = this.targets;
    let count = 0;
    for (let n = 0; n < N; n++) if (targets[n] >= 0) count++;
    // dlogits = (softmax − one-hot)/count, in place in the logits buffer
    const dl = ws.logits;
    for (let n = 0; n < N; n++) {
      const o = n * V;
      const y = targets[n];
      if (y < 0) {
        for (let k = 0; k < V; k++) dl[o + k] = 0;
        continue;
      }
      for (let k = 0; k < V; k++) dl[o + k] /= count;
      dl[o + y] -= 1 / count;
    }
    const daf = ws.dtmp;
    linBwd(dl, ws.af, N, d, p, g, layout.wte, -1, V, daf);
    const dx = ws.dx;
    dx.fill(0);
    lnBwd(daf, N, d, p, g, layout.lnfg, layout.lnfb, ws.xhf, ws.rsf, dx);
    const scale = 1 / Math.sqrt(hd);
    for (let l = cfg.layers - 1; l >= 0; l--) {
      const c = ws.layers[l];
      const bl = layout.blocks[l];
      // MLP: x_out = x_mid + W₂·gelu(W₁·LN₂(x_mid) + b₁) + b₂
      const dF = ws.dF;
      linBwd(dx, c.hact, N, F, p, g, bl.w2, bl.b2, d, dF);
      for (let i = 0; i < N * F; i++) dF[i] *= geluGrad(c.hpre[i]);
      const da2 = ws.dtmp;
      linBwd(dF, c.a2, N, d, p, g, bl.w1, bl.b1, F, da2);
      lnBwd(da2, N, d, p, g, bl.ln2g, bl.ln2b, c.xh2, c.rs2, dx); // dx is now ∂/∂x_mid
      // attention: x_mid = x_in + W_O·y + b_O
      const dy = ws.dtmp;
      linBwd(dx, c.y, N, d, p, g, bl.wo, bl.bo, d, dy);
      const { qkv, att } = c;
      const dqkv = ws.dqkv;
      const dP = ws.dP;
      dqkv.fill(0);
      for (let b = 0; b < B; b++)
        for (let h = 0; h < H; h++) {
          const ab = (b * H + h) * T * T;
          for (let i = 0; i < T; i++) {
            const row = ab + i * T;
            const yo = (b * T + i) * d + h * hd;
            // dP_ij = dy_i · v_j ; dv_j += P_ij dy_i
            let dot = 0;
            for (let j = 0; j <= i; j++) {
              const vo = (b * T + j) * d3 + 2 * d + h * hd;
              let s = 0;
              const a = att[row + j];
              for (let e = 0; e < hd; e++) {
                s += dy[yo + e] * qkv[vo + e];
                dqkv[vo + e] += a * dy[yo + e];
              }
              dP[j] = s;
              dot += s * a;
            }
            const qo = (b * T + i) * d3 + h * hd;
            for (let j = 0; j <= i; j++) {
              const dS = att[row + j] * (dP[j] - dot) * scale;
              if (dS === 0) continue;
              const ko = (b * T + j) * d3 + d + h * hd;
              for (let e = 0; e < hd; e++) {
                dqkv[qo + e] += dS * qkv[ko + e];
                dqkv[ko + e] += dS * qkv[qo + e];
              }
            }
          }
        }
      const da1 = ws.dtmp;
      linBwd(dqkv, c.a1, N, d, p, g, bl.wqkv, bl.bqkv, d3, da1);
      lnBwd(da1, N, d, p, g, bl.ln1g, bl.ln1b, c.xh1, c.rs1, dx); // dx is now ∂/∂x_in
    }
    const tokens = this.tokens;
    for (let b = 0; b < B; b++)
      for (let t = 0; t < T; t++) {
        const n = b * T + t;
        const eo = layout.wte + tokens[n] * d;
        const po = layout.wpe + t * d;
        for (let j = 0; j < d; j++) {
          g[eo + j] += dx[n * d + j];
          g[po + j] += dx[n * d + j];
        }
      }
  }

  /** After `forward(…, targets)`: the softmax probabilities, laid out [B·T × V]. */
  lastProbs(): Arr {
    return this.ws!.logits;
  }

  /** Global L2 norm of the current gradient. */
  gradNorm(): number {
    let s = 0;
    const g = this.g;
    for (let i = 0; i < g.length; i++) s += g[i] * g[i];
    return Math.sqrt(s);
  }

  /**
   * One AdamW update with learning rate lr, in PyTorch's form:
   *   θ ← θ·(1 − lr·λ)  (matrices only);  θ ← θ − (lr/(1−β₁ᵗ))·m / (√v/√(1−β₂ᵗ) + ε).
   * Clips the gradient to a global norm first if `clip` is set. Returns the pre-clip norm.
   */
  step(lr: number, o: AdamWOpts = {}): number {
    const b1 = o.beta1 ?? 0.9;
    const b2 = o.beta2 ?? 0.99;
    const eps = o.eps ?? 1e-8;
    const wd = o.weightDecay ?? 0.1;
    const norm = this.gradNorm();
    const k = o.clip && norm > o.clip ? o.clip / (norm + 1e-6) : 1;
    this.t++;
    this.b1t *= b1;
    this.b2t *= b2;
    const stepSize = lr / (1 - this.b1t);
    const c2 = Math.sqrt(1 - this.b2t);
    const { p, g, m, v, decayMask } = this;
    const shrink = 1 - lr * wd;
    for (let j = 0; j < p.length; j++) {
      const gj = g[j] * k;
      m[j] = b1 * m[j] + (1 - b1) * gj;
      v[j] = b2 * v[j] + (1 - b2) * gj * gj;
      if (decayMask[j]) p[j] *= shrink;
      p[j] -= (stepSize * m[j]) / (Math.sqrt(v[j]) / c2 + eps);
    }
    return norm;
  }

  /** forward + backward + step on a batch laid out [B × (T+1)] (inputs are the first T of each row). */
  trainStep(batch: Int32Array, B: number, T: number, lr: number, o?: AdamWOpts): number {
    const x = new Int32Array(B * T);
    const y = new Int32Array(B * T);
    for (let b = 0; b < B; b++)
      for (let t = 0; t < T; t++) {
        x[b * T + t] = batch[b * (T + 1) + t];
        y[b * T + t] = batch[b * (T + 1) + t + 1];
      }
    const loss = this.forward(x, B, T, y);
    this.backward();
    this.step(lr, o);
    return loss;
  }

  /** Mean loss on a batch, no update. */
  evalLoss(batch: Int32Array, B: number, T: number): number {
    const x = new Int32Array(B * T);
    const y = new Int32Array(B * T);
    for (let b = 0; b < B; b++)
      for (let t = 0; t < T; t++) {
        x[b * T + t] = batch[b * (T + 1) + t];
        y[b * T + t] = batch[b * (T + 1) + t + 1];
      }
    return this.forward(x, B, T, y);
  }

  /** Next-token logits after a context (the last `ctx` tokens are used). */
  nextLogits(context: ArrayLike<number>): Float64Array {
    const T = Math.min(context.length, this.cfg.ctx);
    const x = new Int32Array(T);
    for (let t = 0; t < T; t++) x[t] = context[context.length - T + t];
    this.forward(x, 1, T);
    const V = this.cfg.vocab;
    return Float64Array.from(this.ws!.logits.subarray((T - 1) * V, T * V));
  }

  /**
   * Attention weights of the last forward pass for sequence 0: [layer][head] → T×T row-major
   * (row i is how query position i shares its attention over positions 0 … i).
   */
  attention(): Float64Array[][] {
    const ws = this.ws!;
    const { T } = ws;
    const H = this.cfg.heads;
    return ws.layers.map((c) => Array.from({ length: H }, (_, h) => Float64Array.from(c.att.subarray(h * T * T, (h + 1) * T * T))));
  }

  /**
   * Sample n tokens after a (non-empty) prompt, decoding with a KV cache. Deterministic for a given
   * seed. When the context window fills up, the oldest half is dropped and the rest is re-read, so
   * the model always sees at least ctx/2 tokens of history.
   */
  generate(prompt: ArrayLike<number>, n: number, o: GenerateOpts = {}): number[] {
    if (!prompt.length) throw new Error('generate needs a prompt of at least one token');
    const rng = createRng(o.seed ?? 1);
    const ctxLen = this.cfg.ctx;
    const history = Array.from(prompt).slice(-ctxLen);
    const dec = new Decoder(this);
    let logits = dec.prefill(history);
    const out: number[] = [];
    for (let i = 0; i < n; i++) {
      const probs = sampleDistribution(logits, o.temperature ?? 1, o.topP ?? 1);
      const tok = drawFrom(probs, rng.next());
      out.push(tok);
      history.push(tok);
      if (dec.len === ctxLen) {
        const keep = history.slice(-Math.max(1, Math.floor(ctxLen / 2)));
        history.length = 0;
        history.push(...keep);
        logits = dec.prefill(history);
      } else logits = dec.step(tok);
    }
    return out;
  }

  /** Copy of the parameters (for checkpoints). */
  snapshot(): Float32Array {
    return Float32Array.from(this.p);
  }
  load(params: ArrayLike<number>) {
    this.p.set(params);
  }
}

/**
 * Probabilities after temperature and nucleus (top-p) filtering. Temperature 0 means greedy
 * (all mass on the arg-max, lowest index on ties).
 */
export function sampleDistribution(logits: ArrayLike<number>, temperature: number, topP: number): Float64Array {
  const V = logits.length;
  const p = new Float64Array(V);
  if (temperature <= 0) {
    let best = 0;
    for (let k = 1; k < V; k++) if (logits[k] > logits[best]) best = k;
    p[best] = 1;
    return p;
  }
  let mx = -Infinity;
  for (let k = 0; k < V; k++) mx = Math.max(mx, logits[k]);
  let s = 0;
  for (let k = 0; k < V; k++) s += p[k] = dexp((logits[k] - mx) / temperature);
  for (let k = 0; k < V; k++) p[k] /= s;
  if (topP < 1) {
    const order = Array.from({ length: V }, (_, k) => k).sort((a, b) => p[b] - p[a] || a - b);
    let cum = 0;
    let keep = 0;
    while (keep < V && cum < topP) cum += p[order[keep++]];
    for (let r = keep; r < V; r++) p[order[r]] = 0;
    for (let k = 0; k < V; k++) p[k] /= cum;
  }
  return p;
}

/** Inverse-CDF draw of an index from a probability vector, given u in [0, 1). */
export function drawFrom(p: ArrayLike<number>, u: number): number {
  let c = 0;
  let last = 0;
  for (let k = 0; k < p.length; k++) {
    if (p[k] <= 0) continue;
    last = k;
    c += p[k];
    if (u < c) return k;
  }
  return last;
}

/**
 * Incremental decoding with a KV cache: each new token costs one pass through the weights instead
 * of a pass over the whole context. Gives the same logits as `Gpt.nextLogits` (checked in tests).
 */
export class Decoder {
  readonly model: Gpt;
  len = 0;
  private K: Arr[];
  private Vc: Arr[];
  private x: Float64Array;
  private a: Float64Array;
  private qkv: Float64Array;
  private y: Float64Array;
  private h: Float64Array;
  private tmp: Float64Array;
  private sc: Float64Array;
  constructor(model: Gpt) {
    this.model = model;
    const { d, ctx, layers } = model.cfg;
    const A = () => (model.precision === 'f32' ? new Float32Array(ctx * d) : new Float64Array(ctx * d));
    this.K = Array.from({ length: layers }, A);
    this.Vc = Array.from({ length: layers }, A);
    this.x = new Float64Array(d);
    this.a = new Float64Array(d);
    this.qkv = new Float64Array(3 * d);
    this.y = new Float64Array(d);
    this.h = new Float64Array(model.F);
    this.tmp = new Float64Array(d);
    this.sc = new Float64Array(ctx);
  }
  /** An independent copy of this decoder's state (the cached keys and values so far). */
  clone(): Decoder {
    const c = new Decoder(this.model);
    c.len = this.len;
    this.K.forEach((k, l) => c.K[l].set(k));
    this.Vc.forEach((v, l) => c.Vc[l].set(v));
    return c;
  }
  /** Forget the cache and read `tokens`; returns the logits after the last one. */
  prefill(tokens: ArrayLike<number>): Float64Array {
    this.len = 0;
    let out: Float64Array = new Float64Array(0);
    for (let i = 0; i < tokens.length; i++) out = this.step(tokens[i]);
    return out;
  }
  /** Values are rounded to the model's precision at the same points as in `Gpt.forward`. */
  private r(v: number) {
    return this.model.precision === 'f32' ? Math.fround(v) : v;
  }
  private ln(x: Float64Array, P: Arr, gOff: number, bOff: number, out: Float64Array) {
    const d = x.length;
    let mean = 0;
    for (let j = 0; j < d; j++) mean += x[j];
    mean /= d;
    let v = 0;
    for (let j = 0; j < d; j++) {
      const c = x[j] - mean;
      v += c * c;
    }
    v /= d;
    const rs = 1 / Math.sqrt(v + 1e-5);
    for (let j = 0; j < d; j++) out[j] = this.r(P[gOff + j] * ((x[j] - mean) * rs) + P[bOff + j]);
  }
  private lin(x: Float64Array, din: number, P: Arr, wOff: number, bOff: number, dout: number, out: Float64Array) {
    for (let o = 0; o < dout; o++) {
      let s = bOff >= 0 ? P[bOff + o] : 0;
      const wo = wOff + o * din;
      for (let i = 0; i < din; i++) s += x[i] * P[wo + i];
      out[o] = this.r(s);
    }
  }
  /** Feed one token at position `len`; returns the next-token logits. */
  step(tok: number): Float64Array {
    const m = this.model;
    const { d, heads: H, vocab: V, ctx } = m.cfg;
    if (this.len >= ctx) throw new Error('KV cache full');
    const { p, layout, F, hd } = m;
    const t = this.len;
    const { x, a, qkv, y, h, tmp, sc } = this;
    for (let j = 0; j < d; j++) x[j] = this.r(p[layout.wte + tok * d + j] + p[layout.wpe + t * d + j]);
    const scale = 1 / Math.sqrt(hd);
    for (let l = 0; l < m.cfg.layers; l++) {
      const bl = layout.blocks[l];
      this.ln(x, p, bl.ln1g, bl.ln1b, a);
      this.lin(a, d, p, bl.wqkv, bl.bqkv, 3 * d, qkv);
      const K = this.K[l];
      const Vc = this.Vc[l];
      for (let j = 0; j < d; j++) {
        K[t * d + j] = qkv[d + j];
        Vc[t * d + j] = qkv[2 * d + j];
      }
      for (let hh = 0; hh < H; hh++) {
        let mx = -Infinity;
        for (let j = 0; j <= t; j++) {
          let s = 0;
          for (let e = 0; e < hd; e++) s += qkv[hh * hd + e] * K[j * d + hh * hd + e];
          s *= scale;
          sc[j] = this.r(s);
          if (s > mx) mx = s;
        }
        let sum = 0;
        for (let j = 0; j <= t; j++) {
          const e = dexp(sc[j] - mx);
          sc[j] = this.r(e);
          sum += e;
        }
        for (let j = 0; j <= t; j++) sc[j] = this.r(sc[j] / sum);
        for (let e = 0; e < hd; e++) {
          let acc = 0;
          for (let j = 0; j <= t; j++) acc = this.r(acc + sc[j] * Vc[j * d + hh * hd + e]);
          y[hh * hd + e] = acc;
        }
      }
      this.lin(y, d, p, bl.wo, bl.bo, d, tmp);
      for (let j = 0; j < d; j++) x[j] = this.r(tmp[j] + x[j]);
      this.ln(x, p, bl.ln2g, bl.ln2b, a);
      this.lin(a, d, p, bl.w1, bl.b1, F, h);
      for (let i = 0; i < F; i++) h[i] = this.r(gelu(h[i]));
      this.lin(h, F, p, bl.w2, bl.b2, d, tmp);
      for (let j = 0; j < d; j++) x[j] = this.r(tmp[j] + x[j]);
    }
    this.ln(x, p, layout.lnfg, layout.lnfb, a);
    const logits = new Float64Array(V);
    this.lin(a, d, p, layout.wte, -1, V, logits);
    this.len++;
    return logits;
  }
}
