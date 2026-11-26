// authored-by: Claude Opus 5.5
/**
 * A real, tiny mixture-of-experts layer with hand-written backprop (checked against central finite
 * differences in `moe.test.ts`).
 *
 * The task: 2D "tokens" from four clusters. Inside each cluster the label follows a different
 * straight-line rule, so a single linear expert can't get them all right but four specialists can.
 *
 * The layer: a linear router gives each token one score per expert; softmax turns the scores into
 * probabilities p; the top-k experts by (noisy) score run, and the output is Σ p_i · e_i(x) over those
 * experts (the Switch Transformer convention: the gates are the router's own probabilities, so the
 * router learns from how useful each chosen expert was). Each expert is a single linear rule
 * e_i(x) = a_i·x + b_i, deliberately tiny so that specialising is the only way to do well.
 *
 * Capacity: in each mini-batch, an expert accepts at most ⌈C · k · B / E⌉ token slots; later tokens
 * routed to a full expert are dropped from that expert (their output just lacks its term).
 *
 * Load balancing (Switch Transformer): L_aux = α · E · Σ_i f_i · P_i, with f_i the share of the
 * batch's token slots dispatched to expert i (before capacity) and P_i its mean router probability.
 * Only P carries a gradient.
 */
import { createRng, type Rng } from '../../lib/rng';

export const E_MAX = 8;
export const CLUSTERS = 4;

export type Tokens = { x: Float64Array; y: Uint8Array; cluster: Uint8Array; n: number };

const CENTRES = [45, 135, 225, 315].map((d) => [1.6 * Math.cos((d * Math.PI) / 180), 1.6 * Math.sin((d * Math.PI) / 180)]);
/** Each cluster's rule: label 1 on one side of a line through its centre, at these angles. */
const RULES = [0, 70, 150, 110].map((d) => [Math.cos((d * Math.PI) / 180), Math.sin((d * Math.PI) / 180)]);
export const CLUSTER_CENTRES = CENTRES;

export function makeTokens(n: number, seed: number): Tokens {
  const r = createRng(seed);
  const x = new Float64Array(2 * n);
  const y = new Uint8Array(n);
  const cluster = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const c = r.int(CLUSTERS);
    const px = CENTRES[c][0] + r.normal(0, 0.45);
    const py = CENTRES[c][1] + r.normal(0, 0.45);
    x[2 * i] = px;
    x[2 * i + 1] = py;
    cluster[i] = c;
    y[i] = (px - CENTRES[c][0]) * RULES[c][0] + (py - CENTRES[c][1]) * RULES[c][1] > 0 ? 1 : 0;
  }
  return { x, y, cluster, n };
}

export const TRAIN = makeTokens(256, 11);
export const TEST = makeTokens(512, 12);

export type MoeConfig = {
  experts: number;
  k: 1 | 2;
  /** Load-balancing weight α. */
  alpha: number;
  /** Capacity factor C (Infinity = no limit). */
  capacity: number;
  /** Standard deviation of the noise added to router scores while training (Shazeer's noisy top-k). */
  noise: number;
  lr: number;
  steps: number;
  batch: number;
};

export const DEFAULTS: MoeConfig = { experts: 4, k: 1, alpha: 0, capacity: Infinity, noise: 0, lr: 0.05, steps: 600, batch: 64 };

/** Hidden units in each expert (a 2 → H → 1 tanh network). */
export let HID = 8;
export const setHid = (h: number) => (HID = h);
/** Parameters, flat: router W (E×2), router b (E), then per expert: W1 (H×2), b1 (H), w2 (H), b2. */
export type Params = Float64Array;
const PER = () => 4 * HID + 1;
export const paramCount = (E: number) => 3 * E + E * PER();
const RW = 0;
const RB = (E: number) => 2 * E;
const EX = (E: number, e: number) => 3 * E + e * PER();

export function initParams(E: number, seed: number): Params {
  const r = createRng(seed * 7477 + 3);
  const p = new Float64Array(paramCount(E));
  for (let i = 0; i < 2 * E; i++) p[RW + i] = r.normal(0, 0.3);
  for (let e = 0; e < E; e++) {
    const o = EX(E, e);
    for (let j = 0; j < 2 * HID; j++) p[o + j] = r.normal(0, 1);
    for (let j = 0; j < HID; j++) p[o + 3 * HID + j] = r.normal(0, 1 / Math.sqrt(HID));
  }
  return p;
}

/** One expert's output and hidden activations. */
function expert(p: Params, E: number, e: number, x0: number, x1: number) {
  const o = EX(E, e);
  const h = new Array(HID);
  let out = p[o + 4 * HID];
  for (let j = 0; j < HID; j++) {
    h[j] = Math.tanh(p[o + 2 * j] * x0 + p[o + 2 * j + 1] * x1 + p[o + 2 * HID + j]);
    out += p[o + 3 * HID + j] * h[j];
  }
  return { out, h };
}

/** Add w · d(expert e output)/d(params) to g. */
function expertBack(p: Params, g: Float64Array, E: number, e: number, x0: number, x1: number, h: number[], w: number) {
  const o = EX(E, e);
  g[o + 4 * HID] += w;
  for (let j = 0; j < HID; j++) {
    g[o + 3 * HID + j] += w * h[j];
    const d = w * p[o + 3 * HID + j] * (1 - h[j] * h[j]);
    g[o + 2 * j] += d * x0;
    g[o + 2 * j + 1] += d * x1;
    g[o + 2 * HID + j] += d;
  }
}

export type TokenTrace = {
  scores: number[];
  probs: number[];
  chosen: number[];
  kept: boolean[];
  expertOut: number[];
  hidden: number[][];
  z: number;
};

export function softmax(v: number[]) {
  const m = Math.max(...v);
  const e = v.map((s) => Math.exp(s - m));
  const t = e.reduce((a, b) => a + b, 0);
  return e.map((s) => s / t);
}

export function topK(scores: number[], k: number) {
  return scores
    .map((s, i) => [s, i] as const)
    .sort((a, b) => b[0] - a[0] || a[1] - b[1])
    .slice(0, k)
    .map(([, i]) => i);
}

/** Router and experts for one token (no capacity limit, no noise unless given). */
export function traceToken(p: Params, E: number, k: number, x0: number, x1: number, noise?: number[]): TokenTrace {
  const scores = Array.from({ length: E }, (_, i) => p[RW + 2 * i] * x0 + p[RW + 2 * i + 1] * x1 + p[RB(E) + i] + (noise ? noise[i] : 0));
  const probs = softmax(scores);
  const chosen = topK(scores, k);
  const ex = Array.from({ length: E }, (_, i) => expert(p, E, i, x0, x1));
  const expertOut = ex.map((v) => v.out);
  const z = chosen.reduce((s, i) => s + probs[i] * expertOut[i], 0);
  return { scores, probs, chosen, kept: chosen.map(() => true), expertOut, hidden: ex.map((v) => v.h), z };
}

const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

export type BatchResult = {
  loss: number;
  task: number;
  aux: number;
  /** Token slots dispatched to each expert (before capacity). */
  dispatched: number[];
  dropped: number;
};

/**
 * Loss and gradient for one batch. `noise[t][i]` is the router noise for token t (omit for none).
 * Fills `g` when given.
 */
export function batchLossGrad(p: Params, cfg: MoeConfig, data: Tokens, rows: ArrayLike<number>, noise: number[][] | null, g?: Float64Array): BatchResult {
  const E = cfg.experts;
  const B = rows.length;
  const cap = Number.isFinite(cfg.capacity) ? Math.ceil((cfg.capacity * cfg.k * B) / E) : Infinity;
  const traces: TokenTrace[] = [];
  const dispatched = new Array(E).fill(0);
  const used = new Array(E).fill(0);
  let dropped = 0;
  for (let t = 0; t < B; t++) {
    const i = rows[t];
    const tr = traceToken(p, E, cfg.k, data.x[2 * i], data.x[2 * i + 1], noise ? noise[t] : undefined);
    tr.chosen.forEach((e, r) => {
      dispatched[e]++;
      if (used[e] < cap) used[e]++;
      else {
        tr.kept[r] = false;
        dropped++;
      }
    });
    tr.z = tr.chosen.reduce((s, e, r) => s + (tr.kept[r] ? tr.probs[e] * tr.expertOut[e] : 0), 0);
    traces.push(tr);
  }
  const f = dispatched.map((d) => d / (B * cfg.k));
  const P = new Array(E).fill(0);
  traces.forEach((tr) => tr.probs.forEach((q, e) => (P[e] += q / B)));
  let task = 0;
  traces.forEach((tr, t) => {
    const y = data.y[rows[t]];
    const q = Math.min(Math.max(sigmoid(tr.z), 1e-12), 1 - 1e-12);
    task -= (y ? Math.log(q) : Math.log(1 - q)) / B;
  });
  const aux = cfg.alpha * E * f.reduce((s, fi, e) => s + fi * P[e], 0);
  if (g) {
    g.fill(0);
    traces.forEach((tr, t) => {
      const i = rows[t];
      const x0 = data.x[2 * i];
      const x1 = data.x[2 * i + 1];
      const dz = (sigmoid(tr.z) - data.y[i]) / B;
      const dp = f.map((fi) => (cfg.alpha * E * fi) / B);
      tr.chosen.forEach((e, r) => {
        if (!tr.kept[r]) return;
        expertBack(p, g, E, e, x0, x1, tr.hidden[e], dz * tr.probs[e]);
        dp[e] += dz * tr.expertOut[e];
      });
      const s = tr.probs.reduce((a, q, e) => a + q * dp[e], 0);
      tr.probs.forEach((q, e) => {
        const dr = q * (dp[e] - s);
        g[RW + 2 * e] += dr * x0;
        g[RW + 2 * e + 1] += dr * x1;
        g[RB(E) + e] += dr;
      });
    });
  }
  return { loss: task + aux, task, aux, dispatched, dropped };
}

export type TrainSnapshot = { step: number; load: number[]; acc: number; dropRate: number };

export type TrainResult = {
  params: Params;
  history: TrainSnapshot[];
  /** Test accuracy at the end (no noise, no capacity limit). */
  acc: number;
  /** Each expert's share of the test tokens' expert slots (with top-2, every token fills two slots). */
  load: number[];
  maxLoad: number;
  /** Share of token slots dropped by the capacity limit over the last 100 training steps. */
  dropRate: number;
};

export function evaluate(p: Params, cfg: Pick<MoeConfig, 'experts' | 'k'>, data: Tokens = TEST) {
  const E = cfg.experts;
  const load = new Array(E).fill(0);
  let correct = 0;
  for (let i = 0; i < data.n; i++) {
    const tr = traceToken(p, E, cfg.k, data.x[2 * i], data.x[2 * i + 1]);
    for (const e of tr.chosen) load[e] += 1 / (data.n * tr.chosen.length);
    if ((tr.z > 0 ? 1 : 0) === data.y[i]) correct++;
  }
  return { acc: correct / data.n, load, maxLoad: Math.max(...load) };
}

/** Adam (β₁ 0.9, β₂ 0.999), mini-batches drawn with replacement from TRAIN. */
export function train(cfg: MoeConfig, seed: number, every = 0): TrainResult {
  const E = cfg.experts;
  const p = initParams(E, seed);
  const g = new Float64Array(p.length);
  const m = new Float64Array(p.length);
  const v = new Float64Array(p.length);
  const r: Rng = createRng(seed * 104729 + 7);
  const history: TrainSnapshot[] = [];
  let slots = 0;
  let dropped = 0;
  const rows = new Int32Array(cfg.batch);
  for (let t = 1; t <= cfg.steps; t++) {
    for (let b = 0; b < cfg.batch; b++) rows[b] = r.int(TRAIN.n);
    const noise = cfg.noise > 0 ? Array.from({ length: cfg.batch }, () => Array.from({ length: E }, () => r.normal(0, cfg.noise))) : null;
    const res = batchLossGrad(p, cfg, TRAIN, rows, noise, g);
    if (t > cfg.steps - 100) {
      slots += cfg.batch * cfg.k;
      dropped += res.dropped;
    }
    const c1 = 1 - 0.9 ** t;
    const c2 = 1 - 0.999 ** t;
    for (let j = 0; j < p.length; j++) {
      m[j] = 0.9 * m[j] + 0.1 * g[j];
      v[j] = 0.999 * v[j] + 0.001 * g[j] * g[j];
      p[j] -= (cfg.lr * (m[j] / c1)) / (Math.sqrt(v[j] / c2) + 1e-8);
    }
    if (every && t % every === 0) {
      const ev = evaluate(p, cfg);
      history.push({ step: t, load: ev.load, acc: ev.acc, dropRate: res.dropped / (cfg.batch * cfg.k) });
    }
  }
  const ev = evaluate(p, cfg);
  return { params: p, history, acc: ev.acc, load: ev.load, maxLoad: ev.maxLoad, dropRate: dropped / slots };
}
