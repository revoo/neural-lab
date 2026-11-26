// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { dexp, dtanh, irwinHall } from './detmath';
import { drawFrom, Gpt, gptLayout, gptParamCount, sampleDistribution, type GptConfig } from './gpt';
import { createRng } from '../rng';

describe('deterministic maths', () => {
  it('dexp matches Math.exp to 1e-15 (relative) across the useful range', () => {
    let worst = 0;
    for (let x = -700; x <= 700; x += 0.0137) worst = Math.max(worst, Math.abs(dexp(x) - Math.exp(x)) / Math.exp(x));
    for (const x of [0, 1e-12, -1e-12, 0.5, -0.5, 88.7, -87.3]) worst = Math.max(worst, Math.abs(dexp(x) - Math.exp(x)) / Math.exp(x));
    expect(worst).toBeLessThan(1e-15);
    expect(dexp(0)).toBe(1);
    expect(dexp(800)).toBe(Infinity);
    expect(dexp(-800)).toBe(0);
  });
  it('dtanh matches Math.tanh to 1e-15 (absolute)', () => {
    let worst = 0;
    for (let x = -25; x <= 25; x += 0.0031) worst = Math.max(worst, Math.abs(dtanh(x) - Math.tanh(x)));
    expect(worst).toBeLessThan(1e-15);
  });
  it('Irwin–Hall draws have mean ≈ 0 and variance ≈ 1', () => {
    const r = createRng(3);
    const xs = Array.from({ length: 40000 }, () => irwinHall(r.next));
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
    const vari = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length;
    expect(Math.abs(mean)).toBeLessThan(0.02);
    expect(Math.abs(vari - 1)).toBeLessThan(0.03);
  });
});

const TINY: GptConfig = { vocab: 11, ctx: 6, d: 8, heads: 2, layers: 2 };

describe('layout', () => {
  it('counts V·d + T·d + L·(12d² + 13d) + 2d parameters for ff = 4', () => {
    for (const c of [TINY, { vocab: 65, ctx: 64, d: 64, heads: 4, layers: 4 }, { vocab: 30, ctx: 16, d: 12, heads: 3, layers: 1 }]) {
      const { vocab: V, ctx: T, d, layers: L } = c;
      expect(gptParamCount(c)).toBe(V * d + T * d + L * (12 * d * d + 13 * d) + 2 * d);
    }
  });
  it('lays tensors out back to back and decays only matrices', () => {
    const lay = gptLayout(TINY);
    let o = 0;
    for (const [, off, len] of lay.named) {
      expect(off).toBe(o);
      o += len;
    }
    expect(o).toBe(lay.total);
    const decayed = lay.decay.reduce((s, [, n]) => s + n, 0);
    const { vocab: V, ctx: T, d, layers: L } = TINY;
    expect(decayed).toBe(V * d + T * d + L * 12 * d * d);
  });
});

/** A float64 model with every parameter moved well off its initial value, so no gradient is trivially 0. */
function busyModel(cfg: GptConfig, seed: number) {
  const m = new Gpt(cfg, 'f64');
  const r = createRng(seed);
  for (let i = 0; i < m.p.length; i++) m.p[i] = r.uniform(-0.6, 0.6);
  for (const b of m.layout.blocks) for (let j = 0; j < cfg.d; j++) {
    m.p[b.ln1g + j] = 1 + r.uniform(-0.3, 0.3);
    m.p[b.ln2g + j] = 1 + r.uniform(-0.3, 0.3);
  }
  return m;
}

function checkGradients(cfg: GptConfig, B: number, T: number, ignoreSome: boolean) {
  const m = busyModel(cfg, 11);
  const r = createRng(5);
  const x = Int32Array.from({ length: B * T }, () => r.int(cfg.vocab));
  const y = Int32Array.from({ length: B * T }, (_, i) => (ignoreSome && i % 4 === 1 ? -1 : r.int(cfg.vocab)));
  m.forward(x, B, T, y);
  m.backward();
  const analytic = Float64Array.from(m.g);
  const h = 1e-5;
  let worst = 0;
  let worstAt = '';
  for (let i = 0; i < m.p.length; i++) {
    const v = m.p[i];
    m.p[i] = v + h;
    const a = m.forward(x, B, T, y);
    m.p[i] = v - h;
    const b = m.forward(x, B, T, y);
    m.p[i] = v;
    const num = (a - b) / (2 * h);
    const err = Math.abs(num - analytic[i]) / Math.max(1e-4, Math.abs(num) + Math.abs(analytic[i]));
    if (err > worst) {
      worst = err;
      worstAt = m.layout.named.find(([, o, n]) => i >= o && i < o + n)![0];
    }
  }
  return { worst, worstAt, n: m.p.length };
}

describe('backward pass against central finite differences (float64)', () => {
  it('every parameter of V = 11, d = 8, 2 heads, 2 layers, T = 6', () => {
    const { worst, worstAt, n } = checkGradients(TINY, 2, 6, false);
    expect(n).toBe(1896);
    expect(worst, worstAt).toBeLessThan(1e-6);
  });
  it('with ignored targets, a shorter sequence than the context, and 3 heads in 1 layer', () => {
    const { worst, worstAt } = checkGradients({ vocab: 7, ctx: 9, d: 6, heads: 3, layers: 1, ff: 2 }, 3, 5, true);
    expect(worst, worstAt).toBeLessThan(1e-6);
  });
});

describe('forward pass', () => {
  it('is causal: changing a later token leaves earlier logits untouched', () => {
    const m = busyModel(TINY, 2);
    const a = [1, 4, 2, 9, 0, 3];
    const b = [1, 4, 2, 7, 5, 3];
    const la = m.nextLogits(a.slice(0, 3));
    m.forward(Int32Array.from(b), 1, 6);
    const lb = m.nextLogits(b.slice(0, 3));
    expect(Array.from(lb)).toEqual(Array.from(la));
    // and the full-sequence logits at position 2 equal the 3-token logits
    m.forward(Int32Array.from(a), 1, 6);
    const att = m.attention();
    expect(att.length).toBe(2);
    for (const layer of att)
      for (const head of layer)
        for (let i = 0; i < 6; i++) {
          let s = 0;
          for (let j = 0; j < 6; j++) {
            if (j > i) expect(head[i * 6 + j]).toBe(0);
            s += head[i * 6 + j];
          }
          expect(s).toBeCloseTo(1, 12);
        }
  });
  it('a fresh model is close to uniform: loss ≈ ln V', () => {
    const cfg: GptConfig = { vocab: 40, ctx: 16, d: 32, heads: 4, layers: 2 };
    const m = new Gpt(cfg);
    m.init(1);
    const r = createRng(9);
    const x = Int32Array.from({ length: 4 * 16 }, () => r.int(40));
    const y = Int32Array.from({ length: 4 * 16 }, () => r.int(40));
    expect(Math.abs(m.forward(x, 4, 16, y) - Math.log(40))).toBeLessThan(0.05);
  });
});

describe('AdamW', () => {
  it('matches the PyTorch update rule written out by hand, with decay on matrices only and clipping', () => {
    const m = new Gpt({ vocab: 5, ctx: 3, d: 4, heads: 1, layers: 1 }, 'f64');
    m.init(4);
    const p0 = Float64Array.from(m.p);
    const r = createRng(8);
    const grads = [0, 1, 2].map(() => Float64Array.from({ length: m.p.length }, () => r.normal()));
    const ref = Float64Array.from(p0);
    const mm = new Float64Array(p0.length);
    const vv = new Float64Array(p0.length);
    const lr = 0.01, wd = 0.1, b1 = 0.9, b2 = 0.99, clip = 1;
    const decayed = new Set<number>();
    for (const [o, n] of m.layout.decay) for (let i = o; i < o + n; i++) decayed.add(i);
    grads.forEach((g, s) => {
      m.g.set(g);
      m.step(lr, { weightDecay: wd, beta1: b1, beta2: b2, clip });
      const norm = Math.sqrt(g.reduce((a, b) => a + b * b, 0));
      const k = clip / (norm + 1e-6);
      const t = s + 1;
      for (let i = 0; i < ref.length; i++) {
        const gi = g[i] * k;
        mm[i] = b1 * mm[i] + (1 - b1) * gi;
        vv[i] = b2 * vv[i] + (1 - b2) * gi * gi;
        if (decayed.has(i)) ref[i] *= 1 - lr * wd;
        ref[i] -= (lr / (1 - b1 ** t)) * mm[i] / (Math.sqrt(vv[i]) / Math.sqrt(1 - b2 ** t) + 1e-8);
      }
    });
    let worst = 0;
    for (let i = 0; i < ref.length; i++) worst = Math.max(worst, Math.abs(ref[i] - m.p[i]));
    expect(worst).toBeLessThan(1e-12);
  });
});

describe('training', () => {
  const sentence = 'the cat sat on the mat.';
  const chars = [...new Set(sentence)].sort();
  const enc = (s: string) => [...s].map((c) => chars.indexOf(c));
  const cfg: GptConfig = { vocab: chars.length, ctx: 22, d: 24, heads: 2, layers: 2 };
  const ids = enc(sentence);

  function overfit(seed: number) {
    const m = new Gpt(cfg);
    m.init(seed);
    const batch = Int32Array.from(ids); // one row of T + 1 = 23 tokens
    const losses: number[] = [];
    for (let s = 0; s < 150; s++) losses.push(m.trainStep(batch, 1, 22, 3e-3, { weightDecay: 0, clip: 1 }));
    return { m, losses };
  }

  it('overfits a one-sentence corpus to near-zero loss and then recites it greedily', () => {
    const { m, losses } = overfit(1);
    expect(losses[0]).toBeGreaterThan(2);
    expect(losses[149]).toBeLessThan(0.02);
    const out = m.generate(ids.slice(0, 1), 22, { temperature: 0 });
    expect(out.map((k) => chars[k]).join('')).toBe(sentence.slice(1));
  });

  it('is deterministic: the same seed gives bit-identical parameters', () => {
    const a = overfit(3).m.snapshot();
    const b = overfit(3).m.snapshot();
    expect(new Uint8Array(a.buffer)).toEqual(new Uint8Array(b.buffer));
  });
});

describe('sampling helpers', () => {
  it('temperature 0 is greedy, top-p keeps the smallest nucleus, draws follow the CDF', () => {
    const logits = [0, 2, 1, -1];
    expect(Array.from(sampleDistribution(logits, 0, 1))).toEqual([0, 1, 0, 0]);
    const full = sampleDistribution(logits, 1, 1);
    expect(full.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    // probabilities ≈ [0.087, 0.644, 0.237, 0.032]: top-p 0.7 keeps the two largest
    const nucleus = sampleDistribution(logits, 1, 0.7);
    expect(nucleus[0]).toBe(0);
    expect(nucleus[3]).toBe(0);
    expect(nucleus[1] + nucleus[2]).toBeCloseTo(1, 12);
    expect(drawFrom([0.2, 0, 0.8], 0.1)).toBe(0);
    expect(drawFrom([0.2, 0, 0.8], 0.5)).toBe(2);
    expect(drawFrom([0.2, 0, 0.8], 0.9999999)).toBe(2);
  });
});

describe('cross-engine determinism reference', () => {
  it('the pinned 30-step hash is what Node computes (check-webgpu.cjs requires the same in Chromium)', async () => {
    const { cpuDeterminismHash, CPU_DETERMINISM_HASH } = await import('./gptPinned');
    expect(cpuDeterminismHash()).toBe(CPU_DETERMINISM_HASH);
    expect(CPU_DETERMINISM_HASH).toBe('fba13580');
  });
});

describe('KV-cache decoding', () => {
  it('gives bit-identical logits to a full forward pass, in float32 and float64', async () => {
    const { Decoder } = await import('./gpt');
    for (const prec of ['f32', 'f64'] as const) {
      const m = new Gpt({ vocab: 13, ctx: 10, d: 12, heads: 3, layers: 2 }, prec);
      m.init(6);
      const r = createRng(2);
      for (let i = 0; i < m.p.length; i++) m.p[i] += r.uniform(-0.2, 0.2);
      const toks = Array.from({ length: 10 }, () => r.int(13));
      const dec = new Decoder(m);
      for (let t = 0; t < 10; t++) {
        const a = dec.step(toks[t]);
        const b = m.nextLogits(toks.slice(0, t + 1));
        expect(Array.from(a)).toEqual(Array.from(b));
      }
    }
  });
  it('keeps generating past the context window (dropping the oldest half)', () => {
    const m = new Gpt({ vocab: 9, ctx: 8, d: 8, heads: 2, layers: 1 });
    m.init(1);
    const out = m.generate([1, 2, 3], 30, { seed: 4 });
    expect(out.length).toBe(30);
    expect(out).toEqual(m.generate([1, 2, 3], 30, { seed: 4 }));
  });
});
