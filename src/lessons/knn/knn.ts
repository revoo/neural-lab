// authored-by: Gemini Antigravity
/**
 * k-Nearest Neighbours (kNN), computed from scratch:
 *  - Exact L2 (Euclidean) and L1 (Manhattan) distances, uniform vs 1/d distance weighting
 *  - Leakage-free z-score standardisation (fit on training set only)
 *  - Dataset 1: Noisy two-moons for interactive draggable query & k bias-variance curve
 *  - Dataset 2: Age (22–68) vs Annual Income (€25k–€115k) scaling trap
 *  - Dataset 3: Curse of dimensionality — adding 0..80 pure random noise columns
 *  - Lab: Rescue the kNN Classifier (3 briefs, exhaustively searched)
 */
import { createRng } from '../../lib/rng';

export type Metric = 'l2' | 'l1';
export type Weighting = 'uniform' | 'distance';
export type Sample = { x: number[]; y: 0 | 1 };

export function dist(a: readonly number[], b: readonly number[], metric: Metric): number {
  let s = 0;
  if (metric === 'l2') {
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - b[i];
      s += d * d;
    }
    return Math.sqrt(s);
  }
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s;
}

export type Scaler = { mean: number[]; std: number[] };

/** Fit z-score mean and SD on training data ONLY (never touch test data). */
export function fitScaler(train: readonly Sample[]): Scaler {
  const d = train[0].x.length;
  const mean = new Array<number>(d).fill(0);
  const std = new Array<number>(d).fill(0);
  for (const s of train) for (let j = 0; j < d; j++) mean[j] += s.x[j];
  for (let j = 0; j < d; j++) mean[j] /= train.length;
  for (const s of train) for (let j = 0; j < d; j++) std[j] += (s.x[j] - mean[j]) ** 2;
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / train.length) || 1;
  return { mean, std };
}

export function applyScaler(samples: readonly Sample[], sc: Scaler): Sample[] {
  return samples.map((s) => ({
    x: s.x.map((v, j) => (v - sc.mean[j]) / sc.std[j]),
    y: s.y,
  }));
}

export function knnQuery(
  train: readonly Sample[],
  q: readonly number[],
  k: number,
  metric: Metric = 'l2',
  weight: Weighting = 'uniform',
) {
  const all = train.map((s, idx) => ({ idx, dist: dist(s.x, q, metric), y: s.y }));
  all.sort((a, b) => a.dist - b.dist || a.idx - b.idx);
  const neighbors = all.slice(0, Math.min(k, all.length));
  let w0 = 0;
  let w1 = 0;
  for (const nb of neighbors) {
    const w = weight === 'uniform' ? 1 : 1 / (nb.dist + 1e-6);
    if (nb.y === 1) w1 += w;
    else w0 += w;
  }
  const prob1 = w1 / (w0 + w1);
  return {
    prob1,
    pred: (prob1 >= 0.5 ? 1 : 0) as 0 | 1,
    neighbors,
    kthDist: neighbors[neighbors.length - 1]?.dist ?? 0,
  };
}

export function knnMetrics(
  train: readonly Sample[],
  evalSet: readonly Sample[],
  k: number,
  metric: Metric = 'l2',
  weight: Weighting = 'uniform',
) {
  let correct = 0;
  let tp = 0;
  let pos = 0;
  for (const s of evalSet) {
    const { pred } = knnQuery(train, s.x, k, metric, weight);
    if (pred === s.y) correct++;
    if (s.y === 1) {
      pos++;
      if (pred === 1) tp++;
    }
  }
  return {
    accuracy: correct / evalSet.length,
    posRecall: pos > 0 ? tp / pos : 0,
  };
}

// ------------------------------------------------------------------ 1. Interactive Moons Dataset

let CACHED_MOONS: { train: Sample[]; test: Sample[] } | null = null;

export function getMoonsData() {
  if (CACHED_MOONS) return CACHED_MOONS;
  const rng = createRng(23);
  const make = (n: number, flipCount: number): Sample[] => {
    const out: Sample[] = [];
    for (let i = 0; i < n; i++) {
      const cls: 0 | 1 = i < n / 2 ? 0 : 1;
      const t = rng.uniform(0, Math.PI);
      const noiseX = rng.normal(0, 0.24);
      const noiseY = rng.normal(0, 0.24);
      const x1 = cls === 0 ? Math.cos(t) + noiseX : 1 - Math.cos(t) + noiseX;
      const x2 = cls === 0 ? Math.sin(t) + noiseY : 0.45 - Math.sin(t) + noiseY;
      out.push({ x: [x1, x2], y: cls });
    }
    // Flip a handful of clear training labels so k=1 visibly draws outlier islands
    for (let f = 0; f < flipCount; f++) {
      const idx = Math.floor(((f + 0.5) / flipCount) * n);
      out[idx] = { ...out[idx], y: out[idx].y === 1 ? 0 : 1 };
    }
    return out;
  };
  CACHED_MOONS = { train: make(80, 6), test: make(180, 0) };
  return CACHED_MOONS;
}

export const MOONS_KS = [1, 3, 5, 9, 15, 25, 45] as const;

// ------------------------------------------------------------------ 2. Feature Scaling Trap (Age vs Income)

let CACHED_LOANS: { train: Sample[]; test: Sample[] } | null = null;

export function getLoansData() {
  if (CACHED_LOANS) return CACHED_LOANS;
  const rng = createRng(41);
  const make = (n: number): Sample[] => {
    const out: Sample[] = [];
    for (let i = 0; i < n; i++) {
      const zAge = rng.normal(0, 1);
      const zInc = rng.normal(0, 1);
      const age = Math.max(20, Math.min(72, 44 + 12 * zAge));
      const inc = Math.max(18000, Math.min(130000, 65000 + 24000 * zInc));
      // True approval depends strongly on BOTH age and income (even slightly more on age!)
      const score = 1.6 * zAge + 1.0 * zInc + rng.normal(0, 0.45);
      out.push({ x: [age, inc], y: score > 0.1 ? 1 : 0 });
    }
    return out;
  };
  CACHED_LOANS = { train: make(120), test: make(200) };
  return CACHED_LOANS;
}

export function evaluateScalingTrap(k: number, scaled: boolean) {
  const raw = getLoansData();
  const sc = fitScaler(raw.train);
  const tr = scaled ? applyScaler(raw.train, sc) : raw.train;
  const te = scaled ? applyScaler(raw.test, sc) : raw.test;
  const m = knnMetrics(tr, te, k, 'l2', 'uniform');
  return { accuracy: m.accuracy, scaler: sc, train: tr, test: te };
}

// ------------------------------------------------------------------ 3. Curse of Dimensionality (Noise Columns)

export const NOISE_DIMS = [0, 2, 5, 10, 20, 40, 80] as const;

export type NoiseDimRow = {
  dNoise: number;
  totalDims: number;
  testAcc: number;
  farOverNear: number;
};

let CACHED_NOISE_CURVE: NoiseDimRow[] | null = null;

export function getNoiseDimCurve(): NoiseDimRow[] {
  if (CACHED_NOISE_CURVE) return CACHED_NOISE_CURVE;
  const rng = createRng(89);
  const maxNoise = NOISE_DIMS[NOISE_DIMS.length - 1];
  const make = (n: number): Sample[] => {
    const out: Sample[] = [];
    for (let i = 0; i < n; i++) {
      const x1 = rng.normal(0, 1);
      const x2 = rng.normal(0, 1);
      const y: 0 | 1 = x1 * x1 + x2 * x2 > 1.45 ? 1 : 0;
      const x = [x1, x2];
      for (let j = 0; j < maxNoise; j++) x.push(rng.normal(0, 1));
      out.push({ x, y });
    }
    return out;
  };
  const fullTr = make(150);
  const fullTe = make(200);

  CACHED_NOISE_CURVE = NOISE_DIMS.map((dNoise) => {
    const d = 2 + dNoise;
    const tr = fullTr.map((s) => ({ x: s.x.slice(0, d), y: s.y }));
    const te = fullTe.map((s) => ({ x: s.x.slice(0, d), y: s.y }));
    const { accuracy } = knnMetrics(tr, te, 5, 'l2', 'uniform');
    // Compute median farthest / nearest ratio across test queries
    const ratios: number[] = [];
    for (const q of te) {
      let mn = Infinity;
      let mx = 0;
      for (const s of tr) {
        const dd = dist(s.x, q.x, 'l2');
        if (dd < mn) mn = dd;
        if (dd > mx) mx = dd;
      }
      ratios.push(mx / Math.max(1e-9, mn));
    }
    ratios.sort((a, b) => a - b);
    const farOverNear = ratios[Math.floor(ratios.length / 2)];
    return { dNoise, totalDims: d, testAcc: accuracy, farOverNear };
  });
  return CACHED_NOISE_CURVE;
}

// ------------------------------------------------------------------ 4. Lab: Rescue the kNN Classifier

export const LAB_KS = [1, 3, 5, 9, 15, 25] as const;
export const LAB_METRICS: Metric[] = ['l2', 'l1'];
export const LAB_WEIGHTS: Weighting[] = ['uniform', 'distance'];

export type KnnBrief = {
  id: 'unscaled' | 'noisy-cols' | 'pockets';
  title: string;
  story: string;
  minAcc: number;
  minPosRecall?: number;
  featureLabels?: string[];
  allowScalingToggle: boolean;
};

export const LAB_BRIEFS: KnnBrief[] = [
  {
    id: 'unscaled',
    title: 'Brief 1 · Unscaled Telemetry (1,000× Range Mismatch)',
    story: 'Sensor A reads 0.0–5.0 V while Sensor B reads 0–5,000 RPM. Both carry signal, plus a little label noise. Reach ≥ 90% held-out accuracy — optionally, hit the highest possible test accuracy.',
    minAcc: 0.9,
    allowScalingToggle: true,
  },
  {
    id: 'noisy-cols',
    title: 'Brief 2 · Drowning in Irrelevant Columns',
    story: 'A 6-column table has 2 real signal features (A, B) buried alongside 4 pure random noise columns (C–F). Drop the dead weight and tune kNN to reach ≥ 91% held-out accuracy using the fewest columns.',
    minAcc: 0.91,
    featureLabels: ['Col A', 'Col B', 'Col C', 'Col D', 'Col E', 'Col F'],
    allowScalingToggle: false,
  },
  {
    id: 'pockets',
    title: 'Brief 3 · Small Fault Pockets',
    story: 'Two small clusters of rare faults sit inside a normal operating field (~18% faults). Reach ≥ 94% overall held-out accuracy AND ≥ 85% fault recall — notice how large uniform k erases small pockets!',
    minAcc: 0.94,
    minPosRecall: 0.85,
    allowScalingToggle: false,
  },
];

const LAB_DATA_CACHE = new Map<KnnBrief['id'], { train: Sample[]; test: Sample[] }>();

export function getLabData(id: KnnBrief['id']) {
  const hit = LAB_DATA_CACHE.get(id);
  if (hit) return hit;
  if (id === 'unscaled') {
    const rng = createRng(111);
    const make = (n: number, flip: number): Sample[] => {
      const out: Sample[] = [];
      for (let i = 0; i < n; i++) {
        const z1 = rng.normal(0, 1);
        const z2 = rng.normal(0, 1);
        const y: 0 | 1 = 1.5 * z1 + 0.9 * Math.sin(1.4 * z2) > 0.1 ? 1 : 0;
        out.push({ x: [2.5 + 0.8 * z1, 2500 + 800 * z2], y });
      }
      for (let f = 0; f < flip; f++) out[f * 15] = { ...out[f * 15], y: out[f * 15].y === 1 ? 0 : 1 };
      return out;
    };
    const data = { train: make(120, 6), test: make(180, 0) };
    LAB_DATA_CACHE.set(id, data);
    return data;
  }
  if (id === 'noisy-cols') {
    const rng = createRng(127);
    const make = (n: number): Sample[] => {
      const out: Sample[] = [];
      for (let i = 0; i < n; i++) {
        const a = rng.normal(0, 1);
        const b = rng.normal(0, 1);
        const y: 0 | 1 = a * a + b * b < 1.55 ? 1 : 0;
        out.push({ x: [a, b, rng.normal(0, 1), rng.normal(0, 1), rng.normal(0, 1), rng.normal(0, 1)], y });
      }
      return out;
    };
    const data = { train: make(130), test: make(180) };
    LAB_DATA_CACHE.set(id, data);
    return data;
  }
  // pockets: two small tight minority pockets inside normal background
  const rng = createRng(139);
  const make = (n: number): Sample[] => {
    const out: Sample[] = [];
    for (let i = 0; i < n; i++) {
      const u = rng.next();
      if (u < 0.09) {
        out.push({ x: [-1.1 + rng.normal(0, 0.24), 0.9 + rng.normal(0, 0.24)], y: 1 });
      } else if (u < 0.18) {
        out.push({ x: [1.1 + rng.normal(0, 0.24), -0.8 + rng.normal(0, 0.24)], y: 1 });
      } else {
        // Normal field avoiding the exact pocket centres
        let x1 = rng.uniform(-2.2, 2.2);
        let x2 = rng.uniform(-2.2, 2.2);
        while (Math.hypot(x1 + 1.1, x2 - 0.9) < 0.5 || Math.hypot(x1 - 1.1, x2 + 0.8) < 0.5) {
          x1 = rng.uniform(-2.2, 2.2);
          x2 = rng.uniform(-2.2, 2.2);
        }
        out.push({ x: [x1, x2], y: 0 });
      }
    }
    return out;
  };
  const data = { train: make(140), test: make(200) };
  LAB_DATA_CACHE.set(id, data);
  return data;
}

export type KnnLabConfig = {
  scaled: boolean;
  metric: Metric;
  weight: Weighting;
  k: number;
  cols: number[]; // active column indices (only used in noisy-cols; [0,1] otherwise)
};

export function evaluateKnnLab(b: KnnBrief, cfg: KnnLabConfig) {
  const raw = getLabData(b.id);
  const proj = (arr: Sample[]) => arr.map((s) => ({ x: cfg.cols.map((c) => s.x[c]), y: s.y }));
  let tr = proj(raw.train);
  let te = proj(raw.test);
  if (cfg.scaled) {
    const sc = fitScaler(tr);
    tr = applyScaler(tr, sc);
    te = applyScaler(te, sc);
  }
  const m = knnMetrics(tr, te, cfg.k, cfg.metric, cfg.weight);
  return { ...m, colsUsed: cfg.cols.length, train: tr, test: te };
}

export function briefPasses(b: KnnBrief, r: ReturnType<typeof evaluateKnnLab>): boolean {
  if (r.accuracy < b.minAcc - 1e-9) return false;
  if (b.minPosRecall !== undefined && r.posRecall < b.minPosRecall - 1e-9) return false;
  return true;
}

const LAB_EX_CACHE = new Map<KnnBrief['id'], ReturnType<typeof computeKnnLabExhaustive>>();

function computeKnnLabExhaustive(b: KnnBrief) {
  const raw = getLabData(b.id);
  const scaleOpts = b.allowScalingToggle ? [false, true] : [false];
  const colMasks: number[][] = [];
  if (b.featureLabels) {
    const d = b.featureLabels.length;
    for (let mask = 1; mask < 1 << d; mask++) {
      const cols: number[] = [];
      for (let j = 0; j < d; j++) if (mask & (1 << j)) cols.push(j);
      colMasks.push(cols);
    }
  } else {
    colMasks.push([0, 1]);
  }

  let total = 0;
  let passing = 0;
  let bestScore = -Infinity;
  let bestAcc = 0;
  let bestRecall = 0;
  let bestCols = Infinity;
  let best: { cfg: KnnLabConfig; acc: number; posRecall: number }[] = [];

  const maxK = LAB_KS[LAB_KS.length - 1];
  const posTotal = raw.test.reduce((s, x) => s + x.y, 0);

  for (const scaled of scaleOpts)
    for (const cols of colMasks) {
      let tr = raw.train.map((s) => ({ x: cols.map((c) => s.x[c]), y: s.y }));
      let te = raw.test.map((s) => ({ x: cols.map((c) => s.x[c]), y: s.y }));
      if (scaled) {
        const sc = fitScaler(tr);
        tr = applyScaler(tr, sc);
        te = applyScaler(te, sc);
      }

      for (const metric of LAB_METRICS) {
        // Pre-sort top-maxK neighbours for every test point ONCE
        const sortedPerTest = te.map((q) => {
          const arr = tr.map((s, idx) => ({ idx, dist: dist(s.x, q.x, metric), y: s.y }));
          arr.sort((a, c) => a.dist - c.dist || a.idx - c.idx);
          return arr.slice(0, maxK);
        });

        for (const weight of LAB_WEIGHTS)
          for (const k of LAB_KS) {
            total++;
            let correct = 0;
            let tp = 0;
            for (let ti = 0; ti < te.length; ti++) {
              const nbs = sortedPerTest[ti];
              let w0 = 0;
              let w1 = 0;
              for (let j = 0; j < k; j++) {
                const nb = nbs[j];
                const w = weight === 'uniform' ? 1 : 1 / (nb.dist + 1e-6);
                if (nb.y === 1) w1 += w;
                else w0 += w;
              }
              const pred = w1 >= w0 ? 1 : 0;
              if (pred === te[ti].y) correct++;
              if (te[ti].y === 1 && pred === 1) tp++;
            }
            const accuracy = correct / te.length;
            const posRecall = posTotal > 0 ? tp / posTotal : 0;
            if (accuracy < b.minAcc - 1e-9) continue;
            if (b.minPosRecall !== undefined && posRecall < b.minPosRecall - 1e-9) continue;
            passing++;
            const cfg: KnnLabConfig = { scaled, metric, weight, k, cols };
            const score = b.id === 'noisy-cols' ? -cols.length * 10 + accuracy : b.id === 'pockets' ? posRecall * 10 + accuracy : accuracy;
            if (score > bestScore + 1e-9) {
              bestScore = score;
              bestAcc = accuracy;
              bestRecall = posRecall;
              bestCols = cols.length;
              best = [{ cfg, acc: accuracy, posRecall }];
            } else if (Math.abs(score - bestScore) <= 1e-9) {
              best.push({ cfg, acc: accuracy, posRecall });
            }
          }
      }
    }

  return { total, passing, bestAcc, bestRecall, bestCols, best };
}

export function labExhaustive(b: KnnBrief) {
  const hit = LAB_EX_CACHE.get(b.id);
  if (hit) return hit;
  const res = computeKnnLabExhaustive(b);
  LAB_EX_CACHE.set(b.id, res);
  return res;
}
