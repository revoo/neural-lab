/**
 * A tiny, real neural language model (2-word context → embeddings → tanh hidden → softmax)
 * plus fair smoothed n-gram baselines, trained and evaluated on a clean train / val / test split.
 * Everything runs live in plain Float64Arrays, fast enough to train in the browser and exact
 * enough to check gradients numerically.
 */
import { createRng } from '../../lib/rng';

export const TRAIN_TEXT = `
the cat sat on the warm mat .
the cat slept on the soft rug .
the cat chased the quick mouse across the room .
the cat drank the cold milk in the kitchen .
the dog sat on the warm rug .
the dog slept on the soft mat .
the dog chased the quick cat across the yard .
the dog drank the cold water in the kitchen .
a small bird sang in the tall tree .
a small bird flew over the green garden .
a bright sun shone over the green yard .
a bright moon shone over the quiet garden .
the cold rain fell on the dark roof .
the cold wind blew across the quiet yard .
the warm sun rose over the tall tree .
the quick fox ran across the green garden .
the quick mouse ran into the dark kitchen .
the quiet cat watched the small bird in the tree .
the quiet dog watched the quick fox in the yard .
the warm milk sat on the kitchen table .
the cold water sat on the wooden table .
a tall tree stood beside the quiet house .
a small mouse hid under the wooden table .
the bright sun warmed the soft mat .
the dark rain cooled the warm roof .
the cat walked into the quiet room .
the dog walked into the warm house .
a quick bird flew into the tall tree .
the fox slept under the tall tree .
the mouse hid behind the warm rug .
`;

export const VAL_TEXT = `
the cat slept on the warm rug .
a bright sun shone over the tall tree .
the cold rain fell on the quiet house .
`;

export const TEST_SENTENCES = [
  'the fox watched the small mouse in the yard .',
  'the cat drank the cold water in the room .',
  'the dog hid under the wooden table .',
  'a bright moon shone over the quiet house .',
] as const;

export const TEST_TEXT = TEST_SENTENCES.join('\n');

export const tokenizeWords = (text: string): string[] =>
  text
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);

const TRAIN_TOKS = tokenizeWords(TRAIN_TEXT);
export const VOCAB: readonly string[] = ['<start>', ...Array.from(new Set(TRAIN_TOKS))];
export const V = VOCAB.length;
export const ID_OF = new Map<string, number>(VOCAB.map((w, i) => [w, i]));

export type Sample = { c0: number; c1: number; y: number };

export function makeSamples(text: string): Sample[] {
  const toks = tokenizeWords(text);
  const out: Sample[] = [];
  let c0 = 0;
  let c1 = 0;
  for (const w of toks) {
    const y = ID_OF.get(w);
    if (y === undefined) throw new Error(`Unknown word in corpus: ${w}`);
    out.push({ c0, c1, y });
    if (w === '.') {
      c0 = 0;
      c1 = 0;
    } else {
      c0 = c1;
      c1 = y;
    }
  }
  return out;
}

export const TRAIN_SAMPLES = makeSamples(TRAIN_TEXT);
export const VAL_SAMPLES = makeSamples(VAL_TEXT);
export const TEST_SAMPLES = makeSamples(TEST_TEXT);

// ------------------------------------------------------------------ n-gram baselines

const UNI_COUNTS = new Float64Array(V);
for (const d of TRAIN_SAMPLES) UNI_COUNTS[d.y]++;
const BI_COUNTS = new Float64Array(V * V);
const BI_TOTALS = new Float64Array(V);
for (const d of TRAIN_SAMPLES) {
  BI_COUNTS[d.c1 * V + d.y]++;
  BI_TOTALS[d.c1]++;
}

export function evalSamples(samples: Sample[], probOf: (c0: number, c1: number, y: number) => number) {
  let total = 0;
  for (const d of samples) {
    const p = Math.max(1e-12, probOf(d.c0, d.c1, d.y));
    total -= Math.log(p);
  }
  const loss = total / samples.length;
  return { loss, ppl: Math.exp(loss) };
}

export function uniformProb() {
  return 1 / V;
}

export function unigramProb(y: number, alpha: number) {
  return (UNI_COUNTS[y] + alpha) / (TRAIN_SAMPLES.length + alpha * V);
}

export function bigramProb(c1: number, y: number, alpha: number) {
  return (BI_COUNTS[c1 * V + y] + alpha) / (BI_TOTALS[c1] + alpha * V);
}

const ALPHA_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0];

/** Pick each n-gram baseline's best smoothing α on the validation split (never peeking at test). */
export function tuneBaselines() {
  let uniAlpha = ALPHA_GRID[0];
  let uniVal = Infinity;
  let biAlpha = ALPHA_GRID[0];
  let biVal = Infinity;
  for (const a of ALPHA_GRID) {
    const u = evalSamples(VAL_SAMPLES, (_c0, _c1, y) => unigramProb(y, a)).ppl;
    if (u < uniVal) {
      uniVal = u;
      uniAlpha = a;
    }
    const b = evalSamples(VAL_SAMPLES, (_c0, c1, y) => bigramProb(c1, y, a)).ppl;
    if (b < biVal) {
      biVal = b;
      biAlpha = a;
    }
  }
  return {
    uniAlpha,
    biAlpha,
    testUniform: evalSamples(TEST_SAMPLES, () => uniformProb()),
    testUnigram: evalSamples(TEST_SAMPLES, (_c0, _c1, y) => unigramProb(y, uniAlpha)),
    testBigram: evalSamples(TEST_SAMPLES, (_c0, c1, y) => bigramProb(c1, y, biAlpha)),
  };
}

export const BASELINES = tuneBaselines();

// ------------------------------------------------------------------ Tiny Neural LM

export const LM_D = 6;
export const LM_H = 16;

export type LMGrads = {
  E: Float64Array;
  W1: Float64Array;
  b1: Float64Array;
  W2: Float64Array;
  b2: Float64Array;
};

export class TinyNeuralLM {
  E: Float64Array;
  W1: Float64Array;
  b1: Float64Array;
  W2: Float64Array;
  b2: Float64Array;

  private vE: Float64Array;
  private vW1: Float64Array;
  private vb1: Float64Array;
  private vW2: Float64Array;
  private vb2: Float64Array;
  epoch = 0;

  constructor(seed = 7) {
    const rng = createRng(seed);
    this.E = Float64Array.from({ length: V * LM_D }, () => rng.normal(0, 0.35));
    this.W1 = Float64Array.from({ length: LM_H * 2 * LM_D }, () => rng.normal(0, 0.35));
    this.b1 = new Float64Array(LM_H);
    this.W2 = Float64Array.from({ length: V * LM_H }, () => rng.normal(0, 0.08));
    this.b2 = new Float64Array(V);

    this.vE = new Float64Array(this.E.length);
    this.vW1 = new Float64Array(this.W1.length);
    this.vb1 = new Float64Array(this.b1.length);
    this.vW2 = new Float64Array(this.W2.length);
    this.vb2 = new Float64Array(this.b2.length);
  }

  get paramCount() {
    return this.E.length + this.W1.length + this.b1.length + this.W2.length + this.b2.length;
  }

  clone(): TinyNeuralLM {
    const c = new TinyNeuralLM(1);
    c.E.set(this.E);
    c.W1.set(this.W1);
    c.b1.set(this.b1);
    c.W2.set(this.W2);
    c.b2.set(this.b2);
    c.vE.set(this.vE);
    c.vW1.set(this.vW1);
    c.vb1.set(this.vb1);
    c.vW2.set(this.vW2);
    c.vb2.set(this.vb2);
    c.epoch = this.epoch;
    return c;
  }

  forward(c0: number, c1: number) {
    const h = new Float64Array(LM_H);
    const o0 = c0 * LM_D;
    const o1 = c1 * LM_D;
    for (let j = 0; j < LM_H; j++) {
      let s = this.b1[j];
      const row = j * 2 * LM_D;
      for (let i = 0; i < LM_D; i++) {
        s += this.W1[row + i] * this.E[o0 + i] + this.W1[row + LM_D + i] * this.E[o1 + i];
      }
      h[j] = Math.tanh(s);
    }
    const logits = new Float64Array(V);
    let maxL = -Infinity;
    for (let k = 0; k < V; k++) {
      let s = this.b2[k];
      const row = k * LM_H;
      for (let j = 0; j < LM_H; j++) s += this.W2[row + j] * h[j];
      logits[k] = s;
      if (s > maxL) maxL = s;
    }
    const probs = new Float64Array(V);
    let sumE = 0;
    for (let k = 0; k < V; k++) {
      const e = Math.exp(logits[k] - maxL);
      probs[k] = e;
      sumE += e;
    }
    for (let k = 0; k < V; k++) probs[k] /= sumE;
    return { h, logits, probs };
  }

  evaluate(samples: Sample[]) {
    return evalSamples(samples, (c0, c1, y) => this.forward(c0, c1).probs[y]);
  }

  /** Pure gradient computation of average cross-entropy loss (no weight decay) for testing & stepping. */
  gradients(samples: Sample[]): LMGrads {
    const gE = new Float64Array(this.E.length);
    const gW1 = new Float64Array(this.W1.length);
    const gb1 = new Float64Array(this.b1.length);
    const gW2 = new Float64Array(this.W2.length);
    const gb2 = new Float64Array(this.b2.length);
    const dh = new Float64Array(LM_H);
    const invN = 1 / samples.length;

    for (const d of samples) {
      const { h, probs } = this.forward(d.c0, d.c1);
      dh.fill(0);
      for (let k = 0; k < V; k++) {
        const dl = (probs[k] - (k === d.y ? 1 : 0)) * invN;
        gb2[k] += dl;
        const row = k * LM_H;
        for (let j = 0; j < LM_H; j++) {
          gW2[row + j] += dl * h[j];
          dh[j] += dl * this.W2[row + j];
        }
      }
      const o0 = d.c0 * LM_D;
      const o1 = d.c1 * LM_D;
      for (let j = 0; j < LM_H; j++) {
        const ds = dh[j] * (1 - h[j] * h[j]);
        gb1[j] += ds;
        const row = j * 2 * LM_D;
        for (let i = 0; i < LM_D; i++) {
          gW1[row + i] += ds * this.E[o0 + i];
          gW1[row + LM_D + i] += ds * this.E[o1 + i];
          gE[o0 + i] += ds * this.W1[row + i];
          gE[o1 + i] += ds * this.W1[row + LM_D + i];
        }
      }
    }
    return { E: gE, W1: gW1, b1: gb1, W2: gW2, b2: gb2 };
  }

  step(epochs = 1, lr = 0.5, beta = 0.85, wd = 0.005) {
    const upd = (p: Float64Array, g: Float64Array, v: Float64Array, decay: number) => {
      for (let i = 0; i < p.length; i++) {
        v[i] = beta * v[i] + (g[i] + decay * p[i]);
        p[i] -= lr * v[i];
      }
    };
    for (let e = 0; e < epochs; e++) {
      const g = this.gradients(TRAIN_SAMPLES);
      upd(this.E, g.E, this.vE, wd);
      upd(this.W1, g.W1, this.vW1, wd);
      upd(this.b1, g.b1, this.vb1, 0);
      upd(this.W2, g.W2, this.vW2, wd);
      upd(this.b2, g.b2, this.vb2, 0);
      this.epoch++;
    }
  }

  topNext(c0: number, c1: number, limit = 8): { id: number; word: string; prob: number; logit: number }[] {
    const { probs, logits } = this.forward(c0, c1);
    return Array.from({ length: V }, (_, id) => ({ id, word: VOCAB[id], prob: probs[id], logit: logits[id] }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, limit);
  }

  /** Autoregressively generate tokens from a starting prefix until '.' or maxTokens. */
  generate(prefixWords: string[], maxTokens = 12, seed = 1, mode: 'sample' | 'greedy' = 'sample'): string[] {
    const rng = createRng(seed);
    const out = [...prefixWords];
    for (let step = 0; step < maxTokens; step++) {
      const w0 = out.length >= 2 ? out[out.length - 2] : '<start>';
      const w1 = out.length >= 1 ? out[out.length - 1] : '<start>';
      const c0 = ID_OF.get(w0) ?? 0;
      const c1 = ID_OF.get(w1) ?? 0;
      const { probs } = this.forward(c0, c1);
      let pick = 1;
      if (mode === 'greedy') {
        let bestP = -1;
        for (let k = 1; k < V; k++) {
          if (probs[k] > bestP) {
            bestP = probs[k];
            pick = k;
          }
        }
      } else {
        // Sample over real vocabulary words (exclude <start> at index 0)
        const total = 1 - probs[0];
        let r = rng.next() * total;
        pick = V - 1;
        for (let k = 1; k < V; k++) {
          r -= probs[k];
          if (r <= 0) {
            pick = k;
            break;
          }
        }
      }
      const word = VOCAB[pick];
      out.push(word);
      if (word === '.') break;
    }
    return out;
  }
}

/** Pre-train a standard 120-epoch model and record checkpoints every 10 epochs for instant UI playback. */
export function buildTrainingRun(seed = 7, maxEpoch = 120, interval = 10) {
  const lm = new TinyNeuralLM(seed);
  const history: { epoch: number; trainPpl: number; valPpl: number; testPpl: number; sample: string }[] = [];
  const snapshots = new Map<number, TinyNeuralLM>();
  const record = () => {
    const tr = lm.evaluate(TRAIN_SAMPLES);
    const va = lm.evaluate(VAL_SAMPLES);
    const te = lm.evaluate(TEST_SAMPLES);
    history.push({
      epoch: lm.epoch,
      trainPpl: tr.ppl,
      valPpl: va.ppl,
      testPpl: te.ppl,
      sample: lm.generate(['the', 'cat'], 10, 11 + lm.epoch, 'sample').join(' '),
    });
    snapshots.set(lm.epoch, lm.clone());
  };
  record();
  for (let ep = interval; ep <= maxEpoch; ep += interval) {
    lm.step(interval);
    record();
  }
  return { final: lm, history, snapshots };
}
