// authored-by: Claude Opus 5.5
/**
 * Mini-CLIP: two encoders trained so that matching (image, caption) pairs land close together.
 *
 * Images: 8×8, one of four sprites (plus, cross, ring, bar) on the left or right side, at a random
 * spot, with a little pixel noise. Captions: "a <shape> on the <side>", read as a bag of words.
 * - Image encoder: 6 learned 3×3 filters slid over the image (with ReLU), averaged separately over the
 *   left and right halves (12 numbers), then a linear map to k numbers, scaled to length 1. (A linear
 *   map straight from pixels can't recognise a shape that moves: it reached only about 35%.)
 * - Text encoder: the sum of a learned k-number vector per word, then scaled to length 1.
 * Loss (CLIP's symmetric InfoNCE): for a batch of B pairs, similarities s_ij = img_i · txt_j / τ;
 * each image should pick out its own caption among the B (a softmax over the row), and each caption
 * its own image (over the column). Gradients by hand, finite-difference checked in `vit.test.ts`.
 *
 * Zero-shot: classify new images by which of the 8 captions they are most similar to. No classifier
 * was trained on labels; the classes are just sentences.
 */
import { createRng, type Rng } from '../../lib/rng';

const S = 8;
export const SHAPES = ['plus', 'cross', 'ring', 'bar'] as const;
export const SIDES = ['left', 'right'] as const;
export const WORDS = [...SHAPES, ...SIDES];
const SPR = [
  [0, 1, 0, 1, 1, 1, 0, 1, 0],
  [1, 0, 1, 0, 1, 0, 1, 0, 1],
  [1, 1, 1, 1, 0, 1, 1, 1, 1],
  [0, 1, 0, 0, 1, 0, 0, 1, 0],
];
export type Pair = { img: Float64Array; shape: number; side: number; cls: number };
export const caption = (shape: number, side: number) => `a ${SHAPES[shape]} on the ${SIDES[side]}`;
export const captionWords = (cls: number) => [cls >> 1, 4 + (cls & 1)];

export function drawPair(cls: number, r: Rng, noise = 0.03): Pair {
  const shape = cls >> 1;
  const side = cls & 1;
  const img = new Float64Array(S * S);
  const ox = side * 3 + r.int(3);
  const oy = r.int(6);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) img[(oy + i) * S + ox + j] = SPR[shape][i * 3 + j];
  for (let k = 0; k < img.length; k++) if (r.next() < noise) img[k] = 1 - img[k];
  return { img, shape, side, cls };
}

export const makePairs = (n: number, seed: number) => {
  const r = createRng(seed);
  return Array.from({ length: n }, (_, i) => drawPair(i % 8, r));
};
export const CLIP_TEST = makePairs(400, 4242);

export const FILTERS = 6;
const F = FILTERS;
const FEAT = 2 * F;
/** Wi holds the filters (F × 10: 9 weights and a bias) followed by the projection (k × 2F). */
export type ClipModel = { k: number; Wi: Float64Array; We: Float64Array };

export function initClip(k: number, seed: number): ClipModel {
  const r = createRng(seed * 313 + 9);
  const Wi = new Float64Array(F * 10 + k * FEAT);
  for (let f = 0; f < F; f++) {
    for (let i = 0; i < 9; i++) Wi[f * 10 + i] = r.normal(0, Math.sqrt(2 / 9));
    Wi[f * 10 + 9] = 0.01;
  }
  for (let i = 0; i < k * FEAT; i++) Wi[F * 10 + i] = r.normal(0, Math.sqrt(1 / FEAT));
  return { k, Wi, We: Float64Array.from({ length: WORDS.length * k }, () => r.normal(0, 0.5)) };
}

/** Conv features: pre-activations (F × 6 × 6) and the 2F pooled values (left half, right half). */
function features(m: ClipModel, img: Float64Array) {
  const pre = new Float64Array(F * 36);
  const feat = new Float64Array(FEAT);
  for (let f = 0; f < F; f++)
    for (let i = 0; i < 6; i++)
      for (let j = 0; j < 6; j++) {
        let z = m.Wi[f * 10 + 9];
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) z += m.Wi[f * 10 + a * 3 + b] * img[(i + a) * S + j + b];
        pre[f * 36 + i * 6 + j] = z;
        if (z > 0) feat[2 * f + (j < 3 ? 0 : 1)] += z / 18;
      }
  return { pre, feat };
}

const norm = (v: Float64Array) => {
  const n = Math.sqrt(v.reduce((a, b) => a + b * b, 0)) || 1;
  return { u: v.map((x) => x / n), n };
};

export function encodeImage(m: ClipModel, img: Float64Array) {
  const { pre, feat } = features(m, img);
  const v = new Float64Array(m.k);
  for (let i = 0; i < m.k; i++) {
    let s = 0;
    for (let j = 0; j < FEAT; j++) s += m.Wi[F * 10 + i * FEAT + j] * feat[j];
    v[i] = s;
  }
  return { ...norm(v), pre, feat };
}
export function encodeText(m: ClipModel, words: number[]) {
  const v = new Float64Array(m.k);
  for (const w of words) for (let i = 0; i < m.k; i++) v[i] += m.We[w * m.k + i];
  return norm(v);
}

/** Backprop through v ↦ v/|v|: d v = (d u − u (u · d u)) / |v|. */
const normBack = (u: Float64Array, n: number, du: Float64Array) => {
  const dot = u.reduce((a, x, i) => a + x * du[i], 0);
  return du.map((g, i) => (g - u[i] * dot) / n);
};

/** Symmetric InfoNCE over a batch; accumulates gradients into gi, ge when given. */
export function clipLoss(m: ClipModel, batch: Pair[], tau: number, gi?: Float64Array, ge?: Float64Array) {
  const B = batch.length;
  const I = batch.map((p) => encodeImage(m, p.img));
  const T = batch.map((p) => encodeText(m, captionWords(p.cls)));
  const sim = I.map((a) => T.map((b) => a.u.reduce((s, x, i) => s + x * b.u[i], 0) / tau));
  const soft = (row: number[]) => {
    const mx = Math.max(...row);
    const e = row.map((v) => Math.exp(v - mx));
    const z = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / z);
  };
  const Prow = sim.map(soft);
  const Pcol = sim[0].map((_, j) => soft(sim.map((r) => r[j])));
  let loss = 0;
  for (let i = 0; i < B; i++) loss -= (Math.log(Prow[i][i]) + Math.log(Pcol[i][i])) / (2 * B);
  if (gi && ge) {
    const dS = sim.map((r, i) => r.map((_, j) => ((Prow[i][j] - (i === j ? 1 : 0)) + (Pcol[j][i] - (i === j ? 1 : 0))) / (2 * B)));
    for (let i = 0; i < B; i++) {
      const dIu = new Float64Array(m.k);
      const dTu = new Float64Array(m.k);
      for (let j = 0; j < B; j++)
        for (let c = 0; c < m.k; c++) {
          dIu[c] += (dS[i][j] * T[j].u[c]) / tau;
        }
      for (let j = 0; j < B; j++)
        for (let c = 0; c < m.k; c++) {
          dTu[c] += (dS[j][i] * I[j].u[c]) / tau;
        }
      const dIv = normBack(I[i].u, I[i].n, dIu);
      const dFeat = new Float64Array(FEAT);
      for (let c = 0; c < m.k; c++)
        for (let q = 0; q < FEAT; q++) {
          gi[F * 10 + c * FEAT + q] += dIv[c] * I[i].feat[q];
          dFeat[q] += dIv[c] * m.Wi[F * 10 + c * FEAT + q];
        }
      const img = batch[i].img;
      for (let f = 0; f < F; f++)
        for (let a0 = 0; a0 < 6; a0++)
          for (let b0 = 0; b0 < 6; b0++) {
            if (I[i].pre[f * 36 + a0 * 6 + b0] <= 0) continue;
            const d = dFeat[2 * f + (b0 < 3 ? 0 : 1)] / 18;
            gi[f * 10 + 9] += d;
            for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) gi[f * 10 + a * 3 + b] += d * img[(a0 + a) * S + b0 + b];
          }
      const dTv = normBack(T[i].u, T[i].n, dTu);
      for (const w of captionWords(batch[i].cls)) for (let c = 0; c < m.k; c++) ge[w * m.k + c] += dTv[c];
    }
  }
  return { loss, sim };
}

export type ClipSetting = { tau: number; k: number; batch: number };
export const CLIP_STEPS = 400;

export function trainClip(s: ClipSetting, seed: number, steps = CLIP_STEPS, lr = 0.05) {
  const m = initClip(s.k, seed);
  const r = createRng(seed * 17 + 1);
  const pool = makePairs(512, 100 + seed);
  const params = [m.Wi, m.We];
  const mo = params.map((p) => new Float64Array(p.length));
  const ve = params.map((p) => new Float64Array(p.length));
  for (let t = 1; t <= steps; t++) {
    const batch = Array.from({ length: s.batch }, () => pool[r.int(pool.length)]);
    const gi = new Float64Array(m.Wi.length);
    const ge = new Float64Array(m.We.length);
    clipLoss(m, batch, s.tau, gi, ge);
    [gi, ge].forEach((g, pi) => {
      const p = params[pi];
      for (let j = 0; j < p.length; j++) {
        mo[pi][j] = 0.9 * mo[pi][j] + 0.1 * g[j];
        ve[pi][j] = 0.999 * ve[pi][j] + 0.001 * g[j] * g[j];
        p[j] -= (lr * (mo[pi][j] / (1 - 0.9 ** t))) / (Math.sqrt(ve[pi][j] / (1 - 0.999 ** t)) + 1e-8);
      }
    });
  }
  return m;
}

/** Zero-shot: each test image gets the caption (of all 8) it is most similar to. */
export function zeroShot(m: ClipModel, test: Pair[] = CLIP_TEST) {
  const caps = Array.from({ length: 8 }, (_, c) => encodeText(m, captionWords(c)).u);
  let ok = 0;
  for (const p of test) {
    const u = encodeImage(m, p.img).u;
    let best = 0;
    let bs = -Infinity;
    caps.forEach((t, c) => {
      const s = u.reduce((a, x, i) => a + x * t[i], 0);
      if (s > bs) {
        bs = s;
        best = c;
      }
    });
    if (best === p.cls) ok++;
  }
  return ok / test.length;
}
