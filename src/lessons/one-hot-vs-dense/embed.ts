// authored-by: Gemini Antigravity
/**
 * One-Hot vs Dense Vectors from scratch:
 *  - Exact 8-word one-hot vs 3D dense similarity matrix (cosine & Euclidean)
 *  - Exact matrix-multiply vs table-lookup equivalence & wasted-FLOP calculator (GPT-2, Llama-3)
 *  - Real trained categorical classifier comparing One-Hot vs Scalar Integer ID vs Dense Embeddings on unseen synonyms
 *  - Exhaustive Lab: Choose & Size the Representation (3 briefs)
 */

export type WordEntry = {
  id: number;
  word: string;
  group: 'animal' | 'vehicle' | 'drink';
  /** Compact 3D semantic coordinates: [animal-ness, compactness, wheeled-ness] */
  dense: [number, number, number];
};

export const MINI_VOCAB: WordEntry[] = [
  { id: 0, word: 'cat', group: 'animal', dense: [0.95, 0.72, -0.80] },
  { id: 1, word: 'kitten', group: 'animal', dense: [0.94, 0.88, -0.82] },
  { id: 2, word: 'dog', group: 'animal', dense: [0.92, 0.35, -0.78] },
  { id: 3, word: 'puppy', group: 'animal', dense: [0.90, 0.60, -0.80] },
  { id: 4, word: 'car', group: 'vehicle', dense: [-0.85, -0.20, 0.96] },
  { id: 5, word: 'truck', group: 'vehicle', dense: [-0.88, -0.75, 0.94] },
  { id: 6, word: 'bicycle', group: 'vehicle', dense: [-0.60, 0.55, 0.88] },
  { id: 7, word: 'espresso', group: 'drink', dense: [-0.15, 0.85, -0.25] },
];

export function oneHot(id: number, vocabSize: number): number[] {
  const v = new Array<number>(vocabSize).fill(0);
  v[id] = 1;
  return v;
}

export function cosineSim(a: readonly number[], b: readonly number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 1e-12 ? dot / denom : 0;
}

export function euclideanDist(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
}

export function similarityMatrix(mode: 'onehot' | 'dense'): number[][] {
  const n = MINI_VOCAB.length;
  const vecs = MINI_VOCAB.map((w) => (mode === 'onehot' ? oneHot(w.id, n) : w.dense));
  return vecs.map((vi) => vecs.map((vj) => cosineSim(vi, vj)));
}

// ------------------------------------------------------------------ 1. Matmul vs Table Lookup Equivalence

/** Explicitly compute e_id^T @ E and compare against direct row lookup E[id]. */
export function verifyLookupEquivalence(id: number) {
  const V = MINI_VOCAB.length;
  const d = 3;
  const e = oneHot(id, V);
  const E = MINI_VOCAB.map((w) => w.dense);

  const matmulOut = new Array<number>(d).fill(0);
  let zeroMults = 0;
  let nonZeroMults = 0;
  for (let i = 0; i < V; i++) {
    for (let j = 0; j < d; j++) {
      if (e[i] === 0) zeroMults++;
      else nonZeroMults++;
      matmulOut[j] += e[i] * E[i][j];
    }
  }
  const lookupOut = [...E[id]];
  let maxErr = 0;
  for (let j = 0; j < d; j++) maxErr = Math.max(maxErr, Math.abs(matmulOut[j] - lookupOut[j]));
  return { id, word: MINI_VOCAB[id].word, matmulOut, lookupOut, maxErr, zeroMults, nonZeroMults };
}

export type ModelEmbedSpec = {
  id: string;
  name: string;
  vocab: number;
  dim: number;
};

export const MODEL_EMBED_SPECS: ModelEmbedSpec[] = [
  { id: 'mini', name: 'Our 8-word toy', vocab: 8, dim: 3 },
  { id: 'gpt2', name: 'GPT-2 Small', vocab: 50_257, dim: 768 },
  { id: 'llama3', name: 'Llama-3 8B', vocab: 128_256, dim: 4096 },
];

export function embeddingCosts(vocab: number, dim: number, bytesPerParam: 1 | 2 | 4) {
  const params = vocab * dim;
  const bytes = params * bytesPerParam;
  const matmulFlopsPerToken = 2 * vocab * dim; // multiply + add
  const wastedFraction = (vocab - 1) / vocab;
  return { params, bytes, mib: bytes / (1024 * 1024), matmulFlopsPerToken, wastedFraction };
}

// ------------------------------------------------------------------ 2. Synonym Transfer Experiment (One-Hot vs Integer ID vs Dense)

type CatItem = {
  id: number;
  name: string;
  /** True label: 1 = living creature, 0 = non-living object */
  y: 0 | 1;
  /** 2D pretrained semantic coordinates */
  emb2: [number, number];
  /** 4D pretrained semantic coordinates (2 signal + 2 subtle) */
  emb4: [number, number, number, number];
  split: 'train' | 'test';
};

// Deliberately interleaved IDs so a naive scalar integer ID cannot linearly separate living vs non-living!
export const CAT_ITEMS: CatItem[] = [
  { id: 0, name: 'cat', y: 1, emb2: [0.92, 0.70], emb4: [0.92, 0.70, 0.15, -0.20], split: 'train' },
  { id: 1, name: 'car', y: 0, emb2: [-0.88, -0.15], emb4: [-0.88, -0.15, 0.40, 0.10], split: 'train' },
  { id: 2, name: 'dog', y: 1, emb2: [0.89, 0.35], emb4: [0.89, 0.35, -0.10, 0.25], split: 'train' },
  { id: 3, name: 'bus', y: 0, emb2: [-0.91, -0.65], emb4: [-0.91, -0.65, 0.25, -0.15], split: 'train' },
  { id: 4, name: 'bird', y: 1, emb2: [0.78, 0.85], emb4: [0.78, 0.85, -0.35, 0.10], split: 'train' },
  { id: 5, name: 'train', y: 0, emb2: [-0.85, -0.78], emb4: [-0.85, -0.78, 0.30, 0.20], split: 'train' },
  { id: 6, name: 'horse', y: 1, emb2: [0.84, -0.25], emb4: [0.84, -0.25, 0.05, -0.30], split: 'train' },
  { id: 7, name: 'boat', y: 0, emb2: [-0.79, -0.45], emb4: [-0.79, -0.45, -0.20, 0.35], split: 'train' },
  // Held-out UNSEEN synonyms (never seen in training labels!)
  { id: 8, name: 'kitten', y: 1, emb2: [0.90, 0.86], emb4: [0.90, 0.86, 0.12, -0.18], split: 'test' },
  { id: 9, name: 'truck', y: 0, emb2: [-0.89, -0.70], emb4: [-0.89, -0.70, 0.35, 0.05], split: 'test' },
  { id: 10, name: 'puppy', y: 1, emb2: [0.87, 0.58], emb4: [0.87, 0.58, -0.08, 0.22], split: 'test' },
  { id: 11, name: 'tram', y: 0, emb2: [-0.82, -0.60], emb4: [-0.82, -0.60, 0.28, 0.15], split: 'test' },
  { id: 12, name: 'robin', y: 1, emb2: [0.76, 0.88], emb4: [0.76, 0.88, -0.30, 0.12], split: 'test' },
  { id: 13, name: 'ferry', y: 0, emb2: [-0.76, -0.50], emb4: [-0.76, -0.50, -0.18, 0.30], split: 'test' },
];

export type RepKind = 'onehot' | 'scalar-id' | 'dense-2' | 'dense-4';

function featurise(item: CatItem, rep: RepKind): number[] {
  if (rep === 'onehot') return oneHot(item.id, CAT_ITEMS.length);
  if (rep === 'scalar-id') return [(item.id - 6.5) / 4.0];
  if (rep === 'dense-2') return [...item.emb2];
  return [...item.emb4];
}

function sigmoid(z: number): number {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

export function trainRepClassifier(rep: RepKind) {
  const train = CAT_ITEMS.filter((c) => c.split === 'train');
  const test = CAT_ITEMS.filter((c) => c.split === 'test');
  const d = featurise(train[0], rep).length;
  const w = new Array<number>(d).fill(0);
  let b = 0;
  const lr = 0.8;
  const l2 = 1e-3;

  for (let step = 0; step < 250; step++) {
    const gw = new Array<number>(d).fill(0);
    let gb = 0;
    for (const item of train) {
      const x = featurise(item, rep);
      let z = b;
      for (let j = 0; j < d; j++) z += w[j] * x[j];
      const err = sigmoid(z) - item.y;
      gb += err / train.length;
      for (let j = 0; j < d; j++) gw[j] += (err * x[j]) / train.length;
    }
    b -= lr * gb;
    for (let j = 0; j < d; j++) w[j] -= lr * (gw[j] + l2 * w[j]);
  }

  const scoreSet = (items: CatItem[]) => {
    let ok = 0;
    const preds = items.map((item) => {
      const x = featurise(item, rep);
      let z = b;
      for (let j = 0; j < d; j++) z += w[j] * x[j];
      const p = sigmoid(z);
      const pred = p >= 0.5 ? 1 : 0;
      if (pred === item.y) ok++;
      return { name: item.name, y: item.y, prob: p, ok: pred === item.y };
    });
    return { acc: ok / items.length, preds };
  };

  const tr = scoreSet(train);
  const te = scoreSet(test);
  return { rep, dim: d, params: d + 1, weights: w, bias: b, trainAcc: tr.acc, testAcc: te.acc, testPreds: te.preds };
}

// ------------------------------------------------------------------ 3. Lab: Choose & Size the Representation

export type EmbedBriefId = 'transfer' | 'edge-ram' | 'sku-mlp';

export const EMBED_LAB_BRIEFS: { id: EmbedBriefId; title: string; story: string }[] = [
  {
    id: 'transfer',
    title: 'Brief 1 · Zero-Shot Synonym Transfer (Unseen Category IDs)',
    story: 'Train a linear classifier on 8 known nouns (cat, car, dog, bus, bird, train, horse, boat) and test on 6 UNSEEN synonyms (kitten, truck, puppy, tram, robin, ferry). Pick the input representation that hits ≥ 90% unseen test accuracy with the fewest parameters.',
  },
  {
    id: 'edge-ram',
    title: 'Brief 2 · Fit a 50,000-Word Embedding Table Under 8.0 MiB RAM',
    story: 'An on-device keyboard has vocabulary V = 50,000 and an 8.0 MiB RAM cap for its embedding table. Choose embedding dimension d ∈ {32, 64, 128, 256} and precision (FP32, FP16, INT8) to keep rich semantic capacity (d ≥ 64) under 8.0 MiB — maximising d.',
  },
  {
    id: 'sku-mlp',
    title: 'Brief 3 · High-Cardinality Feature Budget (100,000 SKU IDs → 128-Unit MLP)',
    story: 'An e-commerce recommender feeds 100,000 product SKU IDs into a 128-unit hidden layer. Compare feeding 100,000-dim One-Hot directly vs a learned Embedding(100,000, d) → Linear(d, 128). Stay ≤ 3.50M total parameters while keeping d ≥ 16 (optional: fewest params).',
  },
];

export const LAB_DIMS_B2 = [32, 64, 128, 256] as const;
export const LAB_DTYPES_B2: { value: 1 | 2 | 4; label: string }[] = [
  { value: 4, label: 'FP32 (4 B)' },
  { value: 2, label: 'FP16 (2 B)' },
  { value: 1, label: 'INT8 (1 B)' },
];

export function evalEmbedBrief1(rep: RepKind) {
  const r = trainRepClassifier(rep);
  return { ...r, pass: r.testAcc >= 0.9 - 1e-9 };
}

export function evalEmbedBrief2(d: number, bytesPerParam: 1 | 2 | 4) {
  const c = embeddingCosts(50_000, d, bytesPerParam);
  return { d, bytesPerParam, mib: c.mib, pass: d >= 64 && c.mib <= 8.0 + 1e-9 };
}

export const LAB_SKU_MODES = ['onehot', 'emb-8', 'emb-16', 'emb-32', 'emb-64'] as const;
export type SkuMode = (typeof LAB_SKU_MODES)[number];

export function evalEmbedBrief3(mode: SkuMode) {
  const V = 100_000;
  const H = 128;
  if (mode === 'onehot') {
    const params = V * H + H;
    return { mode, d: 0, params, pass: false };
  }
  const d = Number(mode.split('-')[1]);
  const params = V * d + d * H + H;
  return { mode, d, params, pass: d >= 16 && params <= 3_500_000 };
}

/** Evidence-only hint for the Representation Lab (never names winning buttons or exact settings). */
export function embedLabHint(
  id: EmbedBriefId,
  st: { rep: RepKind; b2Dim: number; b2Bytes: 1 | 2 | 4; sku: SkuMode },
): string {
  if (id === 'transfer') {
    const r = evalEmbedBrief1(st.rep);
    if (r.pass) {
      return r.params === 3
        ? 'Passed at the minimum parameter count (3 weights, 100.0% unseen accuracy).'
        : `Passed with 100.0% unseen accuracy using ${r.params} weights. Check whether fewer semantic coordinates still keep the two classes linearly separable.`;
    }
    if (st.rep === 'onehot') {
      return 'Look at weights w[8..13] on the bar chart: those columns belong to unseen words and received zero gradient during training, leaving every unseen prediction at P = 0.50.';
    }
    return 'Look at the 1D number line: Living and Vehicle IDs alternate along the axis, so no single scalar cut can separate them — even on the 8 training nouns (50.0% accuracy).';
  }
  if (id === 'edge-ram') {
    const r = evalEmbedBrief2(st.b2Dim, st.b2Bytes);
    if (r.pass) {
      return r.d === 128
        ? 'Passed at the maximum vector dimension (128D) that still fits inside 8.0 MiB.'
        : `Fits in ${r.mib.toFixed(3)} MiB with ${r.d}D vectors. Look at bytes per coordinate — halving storage width per float frees half the table RAM for extra dimensions.`;
    }
    if (r.d < 64) {
      return `Table fits easily (${r.mib.toFixed(3)} MiB ≤ 8.0 MiB), but ${r.d}D falls short of the 64D semantic capacity floor.`;
    }
    return `Table takes ${r.mib.toFixed(3)} MiB (> 8.0 MiB) because RAM equals V × d × bytes_per_weight. Watch how reducing bytes per coordinate shrinks the footprint while preserving dimension d.`;
  }
  const r = evalEmbedBrief3(st.sku);
  if (r.pass) {
    return r.d === 16
      ? 'Passed at the minimum total parameter count (1.602M weights, 8.0× smaller than direct One-Hot).'
      : `Passed with ${(r.params / 1e6).toFixed(3)}M weights (≤ 3.50M). Check whether a smaller embedding dimension still meets the d ≥ 16 floor.`;
  }
  if (st.sku === 'onehot') {
    return 'A direct dense matrix from 100,000 one-hot inputs to 128 hidden units needs 100,000 × 128 = 12.800M weights. Factoring through a narrow intermediate lookup dimension shrinks the dominant 100,000 × d term.';
  }
  if (r.d < 16) {
    return `Total weights fit comfortably (${(r.params / 1e6).toFixed(3)}M ≤ 3.50M), but embedding dimension ${r.d}D is below the 16D capacity floor.`;
  }
  return `Embedding dimension ${r.d}D passes the floor, but 100,000 × ${r.d} alone pushes total weights to ${(r.params / 1e6).toFixed(3)}M (> 3.50M).`;
}

export function embedLabExhaustive(id: EmbedBriefId) {
  if (id === 'transfer') {
    const reps: RepKind[] = ['onehot', 'scalar-id', 'dense-2', 'dense-4'];
    const all = reps.map((r) => evalEmbedBrief1(r));
    const pass = all.filter((x) => x.pass);
    const best = pass.reduce((a, b) => (b.params < a.params ? b : a));
    return {
      total: all.length,
      passing: pass.length,
      summary: `Fewest params: ${best.rep} (${best.params} params, ${(100 * best.testAcc).toFixed(1)}% unseen acc)`,
    };
  }
  if (id === 'edge-ram') {
    let total = 0;
    let passing = 0;
    let bestD = 0;
    let bestDesc = '';
    for (const d of LAB_DIMS_B2)
      for (const dt of LAB_DTYPES_B2) {
        total++;
        const r = evalEmbedBrief2(d, dt.value);
        if (!r.pass) continue;
        passing++;
        if (d > bestD) {
          bestD = d;
          bestDesc = `d = ${d} in ${dt.label} (${r.mib.toFixed(3)} MiB)`;
        }
      }
    return { total, passing, summary: `Largest d under 8 MiB: ${bestDesc}` };
  }
  const all = LAB_SKU_MODES.map((m) => evalEmbedBrief3(m));
  const pass = all.filter((x) => x.pass);
  const best = pass.reduce((a, b) => (b.params < a.params ? b : a));
  return {
    total: all.length,
    passing: pass.length,
    summary: `Fewest params: d = ${best.d} (${(best.params / 1e6).toFixed(3)}M vs One-Hot’s 12.800M — ${(12_800_128 / best.params).toFixed(1)}× smaller!)`,
  };
}
