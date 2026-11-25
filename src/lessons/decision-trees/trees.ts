// authored-by: Gemini Antigravity
/**
 * Decision Trees, Random Forests, and Gradient Boosted Trees (XGBoost-style Newton leaf updates),
 * implemented from scratch with zero external ML libraries.
 */
import { createRng, type Rng } from '../../lib/rng';

export type Sample = { x: number[]; y: 0 | 1 };

export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

// ------------------------------------------------------------------ Gini impurity & greedy splits

export function giniCounts(n0: number, n1: number): number {
  const n = n0 + n1;
  if (n === 0) return 0;
  const p1 = n1 / n;
  return 2 * p1 * (1 - p1);
}

export function evaluateSplit(samples: readonly Sample[], feature: number, threshold: number) {
  let l0 = 0;
  let l1 = 0;
  let r0 = 0;
  let r1 = 0;
  for (const s of samples) {
    if (s.x[feature] <= threshold) {
      if (s.y === 1) l1++;
      else l0++;
    } else {
      if (s.y === 1) r1++;
      else r0++;
    }
  }
  const nL = l0 + l1;
  const nR = r0 + r1;
  const n = nL + nR;
  const parentGini = giniCounts(l0 + r0, l1 + r1);
  const leftGini = giniCounts(l0, l1);
  const rightGini = giniCounts(r0, r1);
  const weightedGini = n > 0 ? (nL / n) * leftGini + (nR / n) * rightGini : 0;
  const gain = parentGini - weightedGini;
  return { l0, l1, r0, r1, nL, nR, parentGini, leftGini, rightGini, weightedGini, gain };
}

export function bestGiniSplit(samples: readonly Sample[], featureIndices: readonly number[], minLeaf = 1) {
  const n = samples.length;
  let tot1 = 0;
  for (const s of samples) tot1 += s.y;
  const parentGini = giniCounts(n - tot1, tot1);
  let bestGain = 0;
  let bestFeat = -1;
  let bestThresh = 0;

  for (const f of featureIndices) {
    const pairs = samples.map((s) => ({ v: s.x[f], y: s.y }));
    pairs.sort((a, b) => a.v - b.v);
    let l1 = 0;
    for (let i = 0; i < n - 1; i++) {
      l1 += pairs[i].y;
      const nL = i + 1;
      const nR = n - nL;
      if (pairs[i].v === pairs[i + 1].v) continue;
      if (nL < minLeaf || nR < minLeaf) continue;
      const r1 = tot1 - l1;
      const gL = giniCounts(nL - l1, l1);
      const gR = giniCounts(nR - r1, r1);
      const gain = parentGini - (nL / n) * gL - (nR / n) * gR;
      if (gain > bestGain + 1e-12) {
        bestGain = gain;
        bestFeat = f;
        bestThresh = (pairs[i].v + pairs[i + 1].v) / 2;
      }
    }
  }
  return bestFeat >= 0 ? { feature: bestFeat, threshold: bestThresh, gain: bestGain } : null;
}

// ------------------------------------------------------------------ CART Classification Tree

export type ClassTreeNode =
  | { leaf: true; prob1: number; n: number }
  | { leaf: false; feature: number; threshold: number; gain: number; n: number; left: ClassTreeNode; right: ClassTreeNode };

export function fitClassTree(
  samples: readonly Sample[],
  opts: { maxDepth: number; minLeaf?: number; maxFeatures?: number; rng?: Rng },
  depth = 0,
): ClassTreeNode {
  const n = samples.length;
  let ones = 0;
  for (const s of samples) ones += s.y;
  const prob1 = n > 0 ? ones / n : 0.5;
  const minLeaf = opts.minLeaf ?? 1;

  if (depth >= opts.maxDepth || ones === 0 || ones === n || n < 2 * minLeaf) {
    return { leaf: true, prob1, n };
  }

  const d = samples[0].x.length;
  let feats = Array.from({ length: d }, (_, i) => i);
  if (opts.maxFeatures !== undefined && opts.maxFeatures < d && opts.rng) {
    // Fisher-Yates partial shuffle
    for (let i = 0; i < opts.maxFeatures; i++) {
      const j = i + opts.rng.int(d - i);
      [feats[i], feats[j]] = [feats[j], feats[i]];
    }
    feats = feats.slice(0, opts.maxFeatures);
  }

  const split = bestGiniSplit(samples, feats, minLeaf);
  if (!split) return { leaf: true, prob1, n };

  const leftSamples: Sample[] = [];
  const rightSamples: Sample[] = [];
  for (const s of samples) {
    if (s.x[split.feature] <= split.threshold) leftSamples.push(s);
    else rightSamples.push(s);
  }
  return {
    leaf: false,
    feature: split.feature,
    threshold: split.threshold,
    gain: split.gain,
    n,
    left: fitClassTree(leftSamples, opts, depth + 1),
    right: fitClassTree(rightSamples, opts, depth + 1),
  };
}

export function predictClassTree(node: ClassTreeNode, x: readonly number[]): number {
  let cur = node;
  while (!cur.leaf) {
    cur = x[cur.feature] <= cur.threshold ? cur.left : cur.right;
  }
  return cur.prob1;
}

export function countLeaves(node: ClassTreeNode | RegTreeNode): number {
  if (node.leaf) return 1;
  return countLeaves(node.left) + countLeaves(node.right);
}

export function treeFeatureImportances(node: ClassTreeNode, d: number): number[] {
  const imp = new Array<number>(d).fill(0);
  const walk = (cur: ClassTreeNode) => {
    if (cur.leaf) return;
    imp[cur.feature] += cur.n * cur.gain;
    walk(cur.left);
    walk(cur.right);
  };
  walk(node);
  const tot = imp.reduce((a, b) => a + b, 0) || 1;
  return imp.map((v) => v / tot);
}

// ------------------------------------------------------------------ Random Forest (Bagging + Feature Subsampling)

export type RandomForest = {
  trees: ClassTreeNode[];
  leaves: number;
  importances: number[];
};

export function fitRandomForest(
  samples: readonly Sample[],
  opts: { nTrees: number; maxDepth: number; minLeaf?: number; maxFeatures?: number; seed?: number },
): RandomForest {
  const rng = createRng(opts.seed ?? 101);
  const n = samples.length;
  const d = samples[0].x.length;
  const maxFeatures = opts.maxFeatures ?? Math.max(1, Math.round(Math.sqrt(d)));
  const trees: ClassTreeNode[] = [];
  let leaves = 0;
  const imp = new Array<number>(d).fill(0);

  for (let t = 0; t < opts.nTrees; t++) {
    const boot: Sample[] = new Array(n);
    for (let i = 0; i < n; i++) boot[i] = samples[rng.int(n)];
    const tree = fitClassTree(boot, { maxDepth: opts.maxDepth, minLeaf: opts.minLeaf ?? 1, maxFeatures, rng });
    trees.push(tree);
    leaves += countLeaves(tree);
    const ti = treeFeatureImportances(tree, d);
    for (let j = 0; j < d; j++) imp[j] += ti[j];
  }
  const tot = imp.reduce((a, b) => a + b, 0) || 1;
  return { trees, leaves, importances: imp.map((v) => v / tot) };
}

export function predictForest(rf: RandomForest, x: readonly number[]): number {
  let s = 0;
  for (const t of rf.trees) s += predictClassTree(t, x);
  return s / rf.trees.length;
}

// ------------------------------------------------------------------ Gradient Boosted Trees (Log-Loss + Newton Leaf Updates)

export type RegTreeNode =
  | { leaf: true; val: number }
  | { leaf: false; feature: number; threshold: number; left: RegTreeNode; right: RegTreeNode };

function fitNewtonTree(
  xs: readonly number[][],
  grads: readonly number[], // r_i = y_i - p_i
  hess: readonly number[], // h_i = p_i * (1 - p_i)
  indices: readonly number[],
  maxDepth: number,
  minLeaf: number,
  regLambda: number,
  depth = 0,
): RegTreeNode {
  let G = 0;
  let H = 0;
  for (const idx of indices) {
    G += grads[idx];
    H += hess[idx];
  }
  const leafVal = G / (H + regLambda);
  if (depth >= maxDepth || indices.length < 2 * minLeaf) {
    return { leaf: true, val: leafVal };
  }

  const d = xs[0].length;
  const parentScore = (G * G) / (H + regLambda);
  let bestGain = 1e-9;
  let bestFeat = -1;
  let bestThresh = 0;

  for (let f = 0; f < d; f++) {
    const sorted = [...indices].sort((a, b) => xs[a][f] - xs[b][f]);
    let gL = 0;
    let hL = 0;
    for (let i = 0; i < sorted.length - 1; i++) {
      const idx = sorted[i];
      gL += grads[idx];
      hL += hess[idx];
      const nL = i + 1;
      const nR = sorted.length - nL;
      if (xs[sorted[i]][f] === xs[sorted[i + 1]][f]) continue;
      if (nL < minLeaf || nR < minLeaf) continue;
      const gR = G - gL;
      const hR = H - hL;
      const gain = (gL * gL) / (hL + regLambda) + (gR * gR) / (hR + regLambda) - parentScore;
      if (gain > bestGain) {
        bestGain = gain;
        bestFeat = f;
        bestThresh = (xs[sorted[i]][f] + xs[sorted[i + 1]][f]) / 2;
      }
    }
  }

  if (bestFeat < 0) return { leaf: true, val: leafVal };
  const leftIdx: number[] = [];
  const rightIdx: number[] = [];
  for (const idx of indices) {
    if (xs[idx][bestFeat] <= bestThresh) leftIdx.push(idx);
    else rightIdx.push(idx);
  }
  return {
    leaf: false,
    feature: bestFeat,
    threshold: bestThresh,
    left: fitNewtonTree(xs, grads, hess, leftIdx, maxDepth, minLeaf, regLambda, depth + 1),
    right: fitNewtonTree(xs, grads, hess, rightIdx, maxDepth, minLeaf, regLambda, depth + 1),
  };
}

function predictRegTree(node: RegTreeNode, x: readonly number[]): number {
  let cur = node;
  while (!cur.leaf) cur = x[cur.feature] <= cur.threshold ? cur.left : cur.right;
  return cur.val;
}

export type GBTModel = {
  baseLogit: number;
  lr: number;
  trees: RegTreeNode[];
  leaves: number;
};

export function fitGBT(
  samples: readonly Sample[],
  opts: { nRounds: number; maxDepth: number; lr?: number; minLeaf?: number; regLambda?: number },
): GBTModel {
  const n = samples.length;
  const lr = opts.lr ?? 0.3;
  const minLeaf = opts.minLeaf ?? 2;
  const regLambda = opts.regLambda ?? 1.0;

  let pos = 0;
  for (const s of samples) pos += s.y;
  const p0 = Math.min(0.99, Math.max(0.01, pos / n));
  const baseLogit = Math.log(p0 / (1 - p0));
  const F = new Array<number>(n).fill(baseLogit);
  const xs = samples.map((s) => s.x);
  const allIdx = Array.from({ length: n }, (_, i) => i);
  const trees: RegTreeNode[] = [];
  let leaves = 0;

  for (let m = 0; m < opts.nRounds; m++) {
    const grads = new Array<number>(n);
    const hess = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      const p = sigmoid(F[i]);
      grads[i] = samples[i].y - p;
      hess[i] = Math.max(1e-6, p * (1 - p));
    }
    const tree = fitNewtonTree(xs, grads, hess, allIdx, opts.maxDepth, minLeaf, regLambda);
    trees.push(tree);
    leaves += countLeaves(tree);
    for (let i = 0; i < n; i++) {
      F[i] += lr * predictRegTree(tree, xs[i]);
    }
  }
  return { baseLogit, lr, trees, leaves };
}

export function predictGBT(model: GBTModel, x: readonly number[], upToRounds = model.trees.length): number {
  let z = model.baseLogit;
  const limit = Math.min(upToRounds, model.trees.length);
  for (let m = 0; m < limit; m++) z += model.lr * predictRegTree(model.trees[m], x);
  return sigmoid(z);
}

export function accuracyOf(samples: readonly Sample[], predProb: (x: readonly number[]) => number): number {
  let ok = 0;
  for (const s of samples) {
    if ((predProb(s.x) >= 0.5 ? 1 : 0) === s.y) ok++;
  }
  return ok / samples.length;
}

// ------------------------------------------------------------------ 1. Gini Split Explorer Dataset

let CACHED_GINI: Sample[] | null = null;
export function getGiniData(): Sample[] {
  if (CACHED_GINI) return CACHED_GINI;
  const rng = createRng(19);
  const out: Sample[] = [];
  for (let i = 0; i < 60; i++) {
    const x1 = rng.uniform(0, 10);
    const x2 = rng.uniform(0, 10);
    // Clean L-shaped rule: positive mostly when x1 > 5.2 and x2 > 3.5
    const y: 0 | 1 = (x1 > 5.2 && x2 > 3.2) || (x1 > 7.8) ? (rng.next() < 0.92 ? 1 : 0) : rng.next() < 0.08 ? 1 : 0;
    out.push({ x: [x1, x2], y });
  }
  CACHED_GINI = out;
  return out;
}

// ------------------------------------------------------------------ 2. Single Tree vs Random Forest Dataset (Noisy 2D)

let CACHED_NOISY2D: { train: Sample[]; test: Sample[] } | null = null;
export function getNoisy2DData() {
  if (CACHED_NOISY2D) return CACHED_NOISY2D;
  const rng = createRng(53);
  const make = (n: number, flipRate: number): Sample[] => {
    const out: Sample[] = [];
    for (let i = 0; i < n; i++) {
      const x1 = rng.uniform(-2.2, 2.2);
      const x2 = rng.uniform(-2.2, 2.2);
      const trueY: 0 | 1 = x2 > 0.7 * Math.sin(1.4 * x1) + 0.15 * x1 ? 1 : 0;
      const y: 0 | 1 = rng.next() < flipRate ? (trueY === 1 ? 0 : 1) : trueY;
      out.push({ x: [x1, x2], y });
    }
    return out;
  };
  CACHED_NOISY2D = { train: make(150, 0.12), test: make(250, 0) };
  return CACHED_NOISY2D;
}

// ------------------------------------------------------------------ 3. Messy Tabular Benchmark (Scale Mismatch + 6 Noise Columns)

export const MESSY_COL_NAMES = ['Temp (°C)', 'RPM (×1000)', 'Noise 1', 'Noise 2', 'Noise 3', 'Noise 4', 'Noise 5', 'Noise 6'];

let CACHED_MESSY: { train: Sample[]; test: Sample[] } | null = null;
export function getMessyTabularData() {
  if (CACHED_MESSY) return CACHED_MESSY;
  const rng = createRng(67);
  const make = (n: number, flipRate: number): Sample[] => {
    const out: Sample[] = [];
    for (let i = 0; i < n; i++) {
      const t = rng.uniform(0, 10); // 0..10
      const rpm = rng.uniform(0, 10000); // 0..10,000 (1,000x scale mismatch!)
      const rule = (t > 5.5 && rpm > 4200) || (t < 3.0 && rpm > 7200) || (t > 8.2);
      const trueY: 0 | 1 = rule ? 1 : 0;
      const y: 0 | 1 = rng.next() < flipRate ? (trueY === 1 ? 0 : 1) : trueY;
      const x = [t, rpm];
      for (let j = 0; j < 6; j++) x.push(rng.normal(0, 1));
      out.push({ x, y });
    }
    return out;
  };
  CACHED_MESSY = { train: make(180, 0.06), test: make(260, 0) };
  return CACHED_MESSY;
}

function quickKnnAcc(train: readonly Sample[], test: readonly Sample[], k: number, scaled: boolean): number {
  const d = train[0].x.length;
  let tr = train;
  let te = test;
  if (scaled) {
    const mu = new Array<number>(d).fill(0);
    const sd = new Array<number>(d).fill(0);
    for (const s of train) for (let j = 0; j < d; j++) mu[j] += s.x[j];
    for (let j = 0; j < d; j++) mu[j] /= train.length;
    for (const s of train) for (let j = 0; j < d; j++) sd[j] += (s.x[j] - mu[j]) ** 2;
    for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j] / train.length) || 1;
    tr = train.map((s) => ({ x: s.x.map((v, j) => (v - mu[j]) / sd[j]), y: s.y }));
    te = test.map((s) => ({ x: s.x.map((v, j) => (v - mu[j]) / sd[j]), y: s.y }));
  }
  let ok = 0;
  for (const q of te) {
    const dists = tr.map((s) => {
      let sum = 0;
      for (let j = 0; j < d; j++) {
        const diff = s.x[j] - q.x[j];
        sum += diff * diff;
      }
      return { d: sum, y: s.y };
    });
    dists.sort((a, b) => a.d - b.d);
    let votes = 0;
    for (let i = 0; i < k; i++) votes += dists[i].y;
    if ((votes * 2 >= k ? 1 : 0) === q.y) ok++;
  }
  return ok / te.length;
}

export type MessyBenchSummary = {
  knnRaw: number;
  knnScaled: number;
  singleTree: number;
  randomForest: number;
  gbt: number;
  rfImportances: number[];
  gbtModel: GBTModel;
  gbtRoundAccs: { round: number; trainAcc: number; testAcc: number }[];
};

let CACHED_BENCH: MessyBenchSummary | null = null;
export function getMessyBench(): MessyBenchSummary {
  if (CACHED_BENCH) return CACHED_BENCH;
  const { train, test } = getMessyTabularData();
  const knnRaw = quickKnnAcc(train, test, 5, false);
  const knnScaled = quickKnnAcc(train, test, 5, true);
  const tree = fitClassTree(train, { maxDepth: 8, minLeaf: 1 });
  const singleTree = accuracyOf(test, (x) => predictClassTree(tree, x));
  const rf = fitRandomForest(train, { nTrees: 30, maxDepth: 6, minLeaf: 2, maxFeatures: 3, seed: 202 });
  const randomForest = accuracyOf(test, (x) => predictForest(rf, x));
  const gbtModel = fitGBT(train, { nRounds: 30, maxDepth: 2, lr: 0.3, minLeaf: 3 });
  const gbt = accuracyOf(test, (x) => predictGBT(gbtModel, x));
  const gbtRoundAccs = [1, 3, 5, 10, 15, 20, 30].map((round) => ({
    round,
    trainAcc: accuracyOf(train, (x) => predictGBT(gbtModel, x, round)),
    testAcc: accuracyOf(test, (x) => predictGBT(gbtModel, x, round)),
  }));
  CACHED_BENCH = { knnRaw, knnScaled, singleTree, randomForest, gbt, rfImportances: rf.importances, gbtModel, gbtRoundAccs };
  return CACHED_BENCH;
}

// ------------------------------------------------------------------ 4. Lab: Build the Tabular Ensemble

export type TreeMethod = 'tree' | 'rf' | 'gbt';
export const LAB_METHODS: TreeMethod[] = ['tree', 'rf', 'gbt'];
export const LAB_DEPTHS = [1, 2, 3, 5, 8] as const;
export const LAB_NTREES = [5, 15, 30] as const;

export type TreeBrief = {
  id: 'rules' | 'noisy-table' | 'diagonal';
  title: string;
  story: string;
  minAcc: number;
};

export const LAB_BRIEFS: TreeBrief[] = [
  {
    id: 'rules',
    title: 'Brief 1 · Clean Axis-Aligned Alert Rules',
    story: 'Alerts fire from a clean threshold rule on two sensors (plus 4 irrelevant columns). Reach ≥ 96% held-out accuracy using the fewest total leaves possible.',
    minAcc: 0.96,
  },
  {
    id: 'noisy-table',
    title: 'Brief 2 · Messy 8-Column Table (1,000× Scale + 6 Noise Columns)',
    story: 'An 8-column industrial table (Temp, RPM ×1000, 6 pure noise columns, and 6% label noise). Reach ≥ 98.0% held-out accuracy using the fewest total leaves possible.',
    minAcc: 0.98,
  },
  {
    id: 'diagonal',
    title: 'Brief 3 · Noisy Curved Boundary (12% Label Flips)',
    story: 'A curved sine boundary where 12% of training labels were randomly flipped. Reach ≥ 97.0% held-out accuracy using the fewest total leaves possible.',
    minAcc: 0.97,
  },
];

const LAB_DATA_CACHE = new Map<TreeBrief['id'], { train: Sample[]; test: Sample[] }>();

export function getTreeLabData(id: TreeBrief['id']) {
  const hit = LAB_DATA_CACHE.get(id);
  if (hit) return hit;
  if (id === 'noisy-table') return getMessyTabularData();
  if (id === 'diagonal') return getNoisy2DData();
  // rules: clean 3-region axis-aligned threshold rule + 4 noise columns
  const rng = createRng(83);
  const make = (n: number): Sample[] => {
    const out: Sample[] = [];
    for (let i = 0; i < n; i++) {
      const a = rng.uniform(0, 10);
      const b = rng.uniform(0, 10);
      const y: 0 | 1 = a > 5.0 && b > 4.0 ? 1 : 0;
      out.push({ x: [a, b, rng.normal(), rng.normal(), rng.normal(), rng.normal()], y });
    }
    return out;
  };
  const data = { train: make(160), test: make(240) };
  LAB_DATA_CACHE.set(id, data);
  return data;
}

export const BRIEF_COL_NAMES: Record<TreeBrief['id'], string[]> = {
  rules: ['Sensor A', 'Sensor B', 'Noise 1', 'Noise 2', 'Noise 3', 'Noise 4'],
  'noisy-table': MESSY_COL_NAMES,
  diagonal: ['x₁', 'x₂'],
};

type AnySplitNode = { leaf: true } | { leaf: false; feature: number; left: AnySplitNode; right: AnySplitNode };

function splitSharesOf(roots: readonly AnySplitNode[], d: number): number[] {
  const counts = new Array<number>(d).fill(0);
  let total = 0;
  const walk = (n: AnySplitNode) => {
    if (n.leaf) return;
    counts[n.feature]++;
    total++;
    walk(n.left);
    walk(n.right);
  };
  for (const r of roots) walk(r);
  return total > 0 ? counts.map((c) => c / total) : counts;
}

export type TreeLabConfig = {
  method: TreeMethod;
  maxDepth: number;
  nTrees: number; // ignored (always 1) when method === 'tree'
};

export function evaluateTreeLab(id: TreeBrief['id'], cfg: TreeLabConfig) {
  const { train, test } = getTreeLabData(id);
  const d = train[0].x.length;
  const padZero = (x0: number, x1: number) => {
    const v = new Array<number>(d).fill(0);
    v[0] = x0;
    v[1] = x1;
    return v;
  };

  if (cfg.method === 'tree') {
    const t = fitClassTree(train, { maxDepth: cfg.maxDepth, minLeaf: 2 });
    const pred = (x: readonly number[]) => predictClassTree(t, x);
    return {
      trainAcc: accuracyOf(train, pred),
      testAcc: accuracyOf(test, pred),
      leaves: countLeaves(t),
      splitShares: splitSharesOf([t], d),
      tree: t,
      curve: [{ step: 1, trainAcc: accuracyOf(train, pred), testAcc: accuracyOf(test, pred) }],
      testPreds: test.map((s) => (pred(s.x) >= 0.5 ? 1 : 0) as 0 | 1),
      predictXY: (x0: number, x1: number) => pred(padZero(x0, x1)),
    };
  }
  if (cfg.method === 'rf') {
    const rf = fitRandomForest(train, {
      nTrees: cfg.nTrees,
      maxDepth: cfg.maxDepth,
      minLeaf: 2,
      maxFeatures: Math.max(1, Math.round(Math.sqrt(d))),
      seed: 202,
    });
    const pred = (x: readonly number[]) => predictForest(rf, x);
    // Cumulative train/test accuracy across ensemble size 1..nTrees
    const trSums = new Array<number>(train.length).fill(0);
    const teSums = new Array<number>(test.length).fill(0);
    const curve: { step: number; trainAcc: number; testAcc: number }[] = [];
    for (let m = 0; m < rf.trees.length; m++) {
      let trOk = 0;
      let teOk = 0;
      for (let i = 0; i < train.length; i++) {
        trSums[i] += predictClassTree(rf.trees[m], train[i].x);
        if ((trSums[i] / (m + 1) >= 0.5 ? 1 : 0) === train[i].y) trOk++;
      }
      for (let i = 0; i < test.length; i++) {
        teSums[i] += predictClassTree(rf.trees[m], test[i].x);
        if ((teSums[i] / (m + 1) >= 0.5 ? 1 : 0) === test[i].y) teOk++;
      }
      curve.push({ step: m + 1, trainAcc: trOk / train.length, testAcc: teOk / test.length });
    }
    return {
      trainAcc: accuracyOf(train, pred),
      testAcc: accuracyOf(test, pred),
      leaves: rf.leaves,
      splitShares: splitSharesOf(rf.trees, d),
      tree: null as ClassTreeNode | null,
      curve,
      testPreds: test.map((s) => (pred(s.x) >= 0.5 ? 1 : 0) as 0 | 1),
      predictXY: (x0: number, x1: number) => pred(padZero(x0, x1)),
    };
  }
  const gbt = fitGBT(train, { nRounds: cfg.nTrees, maxDepth: cfg.maxDepth, lr: 0.3, minLeaf: 3 });
  const pred = (x: readonly number[]) => predictGBT(gbt, x);
  const curve: { step: number; trainAcc: number; testAcc: number }[] = [];
  for (let m = 1; m <= gbt.trees.length; m++) {
    curve.push({
      step: m,
      trainAcc: accuracyOf(train, (x) => predictGBT(gbt, x, m)),
      testAcc: accuracyOf(test, (x) => predictGBT(gbt, x, m)),
    });
  }
  return {
    trainAcc: accuracyOf(train, pred),
    testAcc: accuracyOf(test, pred),
    leaves: gbt.leaves,
    splitShares: splitSharesOf(gbt.trees, d),
    tree: null as ClassTreeNode | null,
    curve,
    testPreds: test.map((s) => (pred(s.x) >= 0.5 ? 1 : 0) as 0 | 1),
    predictXY: (x0: number, x1: number) => pred(padZero(x0, x1)),
  };
}

/**
 * Evidence-only hint for the tabular ensemble lab.
 * Points at train-vs-test gap, split shares on noise columns, or boundary geometry —
 * never names the winning tree family or a locked control.
 */
export function treeLabHint(b: TreeBrief, cfg: TreeLabConfig, r: ReturnType<typeof evaluateTreeLab>): string {
  const pass = r.testAcc >= b.minAcc - 1e-9;
  const gap = r.trainAcc - r.testAcc;
  const noiseShare = r.splitShares.slice(2).reduce((a, v) => a + v, 0);
  if (pass) {
    return cfg.method === 'tree'
      ? `It passes using ${r.leaves} leaves! Check the printed split rules to see whether a shallower depth still clears ${(b.minAcc * 100).toFixed(1)}%.`
      : `It passes using ${r.leaves} leaves! Check the boundary and learning curve to see whether a leaner family, shallower depth, or smaller ensemble still clears ${(b.minAcc * 100).toFixed(1)}%.`;
  }
  if (b.id === 'rules') {
    return 'Look at the 2D plot: the positive class is a single clean rectangular corner. Check how many axis-aligned cuts are actually needed to carve out one rectangle.';
  }
  if (b.id === 'noisy-table') {
    if (noiseShare > 0.2) {
      return `Look at the split-share bars: ${(noiseShare * 100).toFixed(0)}% of your model’s splits landed on the 6 pure noise columns! Try a family that inspects all columns at every split, and watch the round-by-round curve for where test accuracy peaks.`;
    }
    if (gap > 0.04) {
      return `Train accuracy (${(r.trainAcc * 100).toFixed(1)}%) is pulling ahead of held-out accuracy (${(r.testAcc * 100).toFixed(1)}%) — the model is chasing the 6% label noise. Check where the learning curve peaked before bending down.`;
    }
    return 'Look at the decision surface and error rings: the true rule has three rectangular regions across the two sensor axes, so depth 1–2 underfits it while too many rounds overfit the 6% label noise.';
  }
  // Brief 3 (diagonal noisy sine, 12% label flips)
  if (gap > 0.05) {
    return `Look at the gap between train (${(r.trainAcc * 100).toFixed(1)}%) and held-out accuracy (${(r.testAcc * 100).toFixed(1)}%): the model is memorising the 12% flipped training labels. Compare how averaging independent trees behaves versus chasing residuals.`;
  }
  return 'Look at the blocky decision surface on the left: the true boundary is curved with 12% flipped training labels. Check which ensemble smooths a curve by averaging while ignoring isolated flipped points.';
}

export type TreeSearchRow = { cfg: TreeLabConfig; testAcc: number; trainAcc: number; leaves: number; pass: boolean };

const LAB_EX_CACHE = new Map<TreeBrief['id'], ReturnType<typeof computeTreeLabExhaustive>>();

function computeTreeLabExhaustive(b: TreeBrief) {
  let total = 0;
  let passing = 0;
  let bestLeaves = Infinity;
  let bestAccAtMin = -1;
  let best: { cfg: TreeLabConfig; testAcc: number; leaves: number }[] = [];
  const all: TreeSearchRow[] = [];

  for (const method of LAB_METHODS) {
    const counts = method === 'tree' ? [1] : [...LAB_NTREES];
    for (const maxDepth of LAB_DEPTHS)
      for (const nTrees of counts) {
        total++;
        const cfg: TreeLabConfig = { method, maxDepth, nTrees };
        const r = evaluateTreeLab(b.id, cfg);
        const pass = r.testAcc >= b.minAcc - 1e-9;
        all.push({ cfg, testAcc: r.testAcc, trainAcc: r.trainAcc, leaves: r.leaves, pass });
        if (!pass) continue;
        passing++;
        if (r.leaves < bestLeaves) {
          bestLeaves = r.leaves;
          bestAccAtMin = r.testAcc;
          best = [{ cfg, testAcc: r.testAcc, leaves: r.leaves }];
        } else if (r.leaves === bestLeaves) {
          if (r.testAcc > bestAccAtMin + 1e-9) {
            bestAccAtMin = r.testAcc;
            best = [{ cfg, testAcc: r.testAcc, leaves: r.leaves }];
          } else if (Math.abs(r.testAcc - bestAccAtMin) <= 1e-9) {
            best.push({ cfg, testAcc: r.testAcc, leaves: r.leaves });
          }
        }
      }
  }
  return { total, passing, bestLeaves, bestAccAtMin, best, all };
}

export function labExhaustive(b: TreeBrief) {
  const hit = LAB_EX_CACHE.get(b.id);
  if (hit) return hit;
  const res = computeTreeLabExhaustive(b);
  LAB_EX_CACHE.set(b.id, res);
  return res;
}
