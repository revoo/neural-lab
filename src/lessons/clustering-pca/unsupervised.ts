// authored-by: Gemini Antigravity
/**
 * Unsupervised Learning from scratch:
 *  - Lloyd's k-Means with full step-by-step trace, Random vs k-Means++ seeding, and exact Silhouette score
 *  - Exact symmetric Jacobi eigensolver for PCA (verified C v = lambda v and V^T V = I in tests)
 *  - 2D interactive projection angle proving Projected Variance + Reconstruction MSE = Total Variance
 *  - Lab: Compress & Segment (3 exhaustive briefs: PCA scaling trap, k-Means init/scaling, PCA->k-Means denoising)
 */
import { createRng, type Rng } from '../../lib/rng';

export type Vec = number[];

export function sqDist(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

export function standardize(pts: readonly Vec[]): { data: Vec[]; mean: Vec; std: Vec } {
  const n = pts.length;
  const d = pts[0].length;
  const mean = new Array<number>(d).fill(0);
  const std = new Array<number>(d).fill(0);
  for (const p of pts) for (let j = 0; j < d; j++) mean[j] += p[j];
  for (let j = 0; j < d; j++) mean[j] /= n;
  for (const p of pts) for (let j = 0; j < d; j++) std[j] += (p[j] - mean[j]) ** 2;
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / n) || 1;
  const data = pts.map((p) => p.map((v, j) => (v - mean[j]) / std[j]));
  return { data, mean, std };
}

export function center(pts: readonly Vec[]): { data: Vec[]; mean: Vec } {
  const n = pts.length;
  const d = pts[0].length;
  const mean = new Array<number>(d).fill(0);
  for (const p of pts) for (let j = 0; j < d; j++) mean[j] += p[j];
  for (let j = 0; j < d; j++) mean[j] /= n;
  return { data: pts.map((p) => p.map((v, j) => v - mean[j])), mean };
}

// ------------------------------------------------------------------ k-Means & k-Means++

export type InitMode = 'random' | 'kmeans++';

export function initCentroids(pts: readonly Vec[], k: number, mode: InitMode, rng: Rng): Vec[] {
  const n = pts.length;
  if (mode === 'random') {
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < k; i++) {
      const j = i + rng.int(n - i);
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return idx.slice(0, k).map((i) => [...pts[i]]);
  }
  // Exact k-Means++ (D^2 weighted sampling)
  const centers: Vec[] = [[...pts[rng.int(n)]]];
  for (let c = 1; c < k; c++) {
    const d2 = pts.map((p) => {
      let best = Infinity;
      for (const mu of centers) {
        const dd = sqDist(p, mu);
        if (dd < best) best = dd;
      }
      return best;
    });
    const tot = d2.reduce((a, b) => a + b, 0);
    if (tot <= 1e-12) {
      centers.push([...pts[rng.int(n)]]);
      continue;
    }
    let r = rng.next() * tot;
    let pick = n - 1;
    for (let i = 0; i < n; i++) {
      r -= d2[i];
      if (r <= 0) {
        pick = i;
        break;
      }
    }
    centers.push([...pts[pick]]);
  }
  return centers;
}

export type KMeansSnapshot = {
  iter: number;
  centroids: Vec[];
  assign: number[];
  inertia: number;
};

export function runKMeansFrom(pts: readonly Vec[], init: readonly Vec[], maxIter = 40): KMeansSnapshot[] {
  const k = init.length;
  const d = pts[0].length;
  let centroids = init.map((c) => [...c]);
  const snaps: KMeansSnapshot[] = [];

  const assignPoints = (cs: Vec[]) => {
    let inertia = 0;
    const assign = pts.map((p) => {
      let best = 0;
      let bestD = sqDist(p, cs[0]);
      for (let c = 1; c < k; c++) {
        const dd = sqDist(p, cs[c]);
        if (dd < bestD) {
          bestD = dd;
          best = c;
        }
      }
      inertia += bestD;
      return best;
    });
    return { assign, inertia };
  };

  let cur = assignPoints(centroids);
  snaps.push({ iter: 0, centroids: centroids.map((c) => [...c]), assign: [...cur.assign], inertia: cur.inertia });

  for (let iter = 1; iter <= maxIter; iter++) {
    const sums = Array.from({ length: k }, () => new Array<number>(d).fill(0));
    const counts = new Array<number>(k).fill(0);
    for (let i = 0; i < pts.length; i++) {
      const c = cur.assign[i];
      counts[c]++;
      for (let j = 0; j < d; j++) sums[c][j] += pts[i][j];
    }
    const nextCentroids = centroids.map((old, c) =>
      counts[c] > 0 ? sums[c].map((v) => v / counts[c]) : [...old],
    );
    const next = assignPoints(nextCentroids);
    const same = next.assign.every((a, i) => a === cur.assign[i]);
    centroids = nextCentroids;
    cur = next;
    snaps.push({ iter, centroids: centroids.map((c) => [...c]), assign: [...cur.assign], inertia: cur.inertia });
    if (same) break;
  }
  return snaps;
}

export function runKMeans(pts: readonly Vec[], k: number, mode: InitMode, seed: number) {
  const init = initCentroids(pts, k, mode, createRng(seed));
  const trace = runKMeansFrom(pts, init);
  return { trace, final: trace[trace.length - 1] };
}

/** Mean Silhouette score in [-1, 1] (higher = tighter, better separated clusters). */
export function silhouetteScore(pts: readonly Vec[], assign: readonly number[], k: number): number {
  const n = pts.length;
  if (k <= 1 || n <= k) return 0;
  // Precompute pairwise Euclidean distances
  const D = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dd = Math.sqrt(sqDist(pts[i], pts[j]));
      D[i][j] = dd;
      D[j][i] = dd;
    }
  }
  const members: number[][] = Array.from({ length: k }, () => []);
  for (let i = 0; i < n; i++) members[assign[i]].push(i);

  let total = 0;
  for (let i = 0; i < n; i++) {
    const ci = assign[i];
    const own = members[ci];
    if (own.length <= 1) continue;
    let a = 0;
    for (const j of own) if (j !== i) a += D[i][j];
    a /= own.length - 1;

    let b = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === ci || members[c].length === 0) continue;
      let meanDist = 0;
      for (const j of members[c]) meanDist += D[i][j];
      meanDist /= members[c].length;
      if (meanDist < b) b = meanDist;
    }
    const denom = Math.max(a, b);
    if (denom > 0) total += (b - a) / denom;
  }
  return total / n;
}

/** Cluster purity: share of points matching their cluster's majority true label. */
export function clusterPurity(assign: readonly number[], truth: readonly number[], k: number): number {
  let correct = 0;
  for (let c = 0; c < k; c++) {
    const counts = new Map<number, number>();
    for (let i = 0; i < assign.length; i++) {
      if (assign[i] === c) counts.set(truth[i], (counts.get(truth[i]) ?? 0) + 1);
    }
    let best = 0;
    counts.forEach((v) => {
      if (v > best) best = v;
    });
    correct += best;
  }
  return correct / assign.length;
}

// ------------------------------------------------------------------ Exact Symmetric Jacobi Eigensolver & PCA

export type PcaResult = {
  mean: Vec;
  eigenvalues: number[]; // descending
  components: Vec[]; // unit eigenvectors (rows), matching descending eigenvalues
  explainedRatio: number[];
  totalVar: number;
};

/** Exact cyclic Jacobi eigensolver for a d x d symmetric matrix. */
export function jacobiEigen(mat: number[][]): { values: number[]; vectors: Vec[] } {
  const d = mat.length;
  const A = mat.map((r) => [...r]);
  const V = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 1 : 0)));

  for (let sweep = 0; sweep < 50; sweep++) {
    let maxOff = 0;
    for (let p = 0; p < d; p++) {
      for (let q = p + 1; q < d; q++) {
        const apq = A[p][q];
        if (Math.abs(apq) > maxOff) maxOff = Math.abs(apq);
        if (Math.abs(apq) < 1e-14) continue;
        const app = A[p][p];
        const aqq = A[q][q];
        const tau = (aqq - app) / (2 * apq);
        const t = tau >= 0 ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;

        A[p][p] = app - t * apq;
        A[q][q] = aqq + t * apq;
        A[p][q] = 0;
        A[q][p] = 0;
        for (let r = 0; r < d; r++) {
          if (r === p || r === q) continue;
          const arp = A[r][p];
          const arq = A[r][q];
          A[r][p] = c * arp - s * arq;
          A[p][r] = A[r][p];
          A[r][q] = s * arp + c * arq;
          A[q][r] = A[r][q];
        }
        for (let r = 0; r < d; r++) {
          const vrp = V[r][p];
          const vrq = V[r][q];
          V[r][p] = c * vrp - s * vrq;
          V[r][q] = s * vrp + c * vrq;
        }
      }
    }
    if (maxOff < 1e-13) break;
  }

  const order = Array.from({ length: d }, (_, i) => i).sort((i, j) => A[j][j] - A[i][i]);
  const values = order.map((i) => Math.max(0, A[i][i]));
  const vectors = order.map((col) => {
    const v = Array.from({ length: d }, (_, row) => V[row][col]);
    // Canonical sign: make largest-magnitude entry positive for determinism
    let maxIdx = 0;
    for (let i = 1; i < d; i++) if (Math.abs(v[i]) > Math.abs(v[maxIdx])) maxIdx = i;
    return v[maxIdx] < 0 ? v.map((x) => -x) : v;
  });
  return { values, vectors };
}

export function fitPCA(pts: readonly Vec[]): PcaResult {
  const { data, mean } = center(pts);
  const n = data.length;
  const d = data[0].length;
  const cov = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  for (const row of data) {
    for (let i = 0; i < d; i++) {
      for (let j = i; j < d; j++) cov[i][j] += row[i] * row[j];
    }
  }
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      cov[i][j] /= n;
      cov[j][i] = cov[i][j];
    }
  }
  const { values, vectors } = jacobiEigen(cov);
  const totalVar = values.reduce((a, b) => a + b, 0) || 1;
  return {
    mean,
    eigenvalues: values,
    components: vectors,
    explainedRatio: values.map((v) => v / totalVar),
    totalVar,
  };
}

export function projectPCA(pts: readonly Vec[], pca: PcaResult, m: number): Vec[] {
  const comps = pca.components.slice(0, m);
  return pts.map((p) => {
    const centered = p.map((v, j) => v - pca.mean[j]);
    return comps.map((u) => {
      let s = 0;
      for (let j = 0; j < u.length; j++) s += centered[j] * u[j];
      return s;
    });
  });
}

// ------------------------------------------------------------------ 1. Four-Blobs & Init-Trap Dataset

let CACHED_BLOBS4: { pts: Vec[]; truth: number[] } | null = null;
export function getBlobs4() {
  if (CACHED_BLOBS4) return CACHED_BLOBS4;
  const rng = createRng(37);
  const centers: [number, number][] = [
    [-2.4, 2.2],
    [2.4, 2.2],
    [-2.4, -2.2],
    [2.4, -2.2],
  ];
  const pts: Vec[] = [];
  const truth: number[] = [];
  for (let c = 0; c < 4; c++) {
    for (let i = 0; i < 30; i++) {
      pts.push([centers[c][0] + rng.normal(0, 0.55), centers[c][1] + rng.normal(0, 0.55)]);
      truth.push(c);
    }
  }
  CACHED_BLOBS4 = { pts, truth };
  return CACHED_BLOBS4;
}

/** Deliberate bad random seed: 2 centroids start inside top-left blob, 1 in top-right, 1 between bottom blobs. */
export const BAD_INIT_BLOBS4: Vec[] = [
  [-2.8, 2.5],
  [-2.0, 1.9],
  [2.5, 2.3],
  [0.1, -2.2],
];

let CACHED_INIT_RACE: { randomOptimal: number; ppOptimal: number; randomMeanInertia: number; ppMeanInertia: number; globalInertia: number; seeds: number } | null = null;

export function getInitRace() {
  if (CACHED_INIT_RACE) return CACHED_INIT_RACE;
  const { pts } = getBlobs4();
  const seeds = 40;
  const randInertias: number[] = [];
  const ppInertias: number[] = [];
  for (let s = 1; s <= seeds; s++) {
    randInertias.push(runKMeans(pts, 4, 'random', s * 17 + 3).final.inertia);
    ppInertias.push(runKMeans(pts, 4, 'kmeans++', s * 17 + 3).final.inertia);
  }
  const globalInertia = Math.min(...randInertias, ...ppInertias);
  const tol = globalInertia * 1.05;
  CACHED_INIT_RACE = {
    randomOptimal: randInertias.filter((v) => v <= tol).length,
    ppOptimal: ppInertias.filter((v) => v <= tol).length,
    randomMeanInertia: randInertias.reduce((a, b) => a + b, 0) / seeds,
    ppMeanInertia: ppInertias.reduce((a, b) => a + b, 0) / seeds,
    globalInertia,
    seeds,
  };
  return CACHED_INIT_RACE;
}

// ------------------------------------------------------------------ 2. Elbow & Concentric Rings Dataset

let CACHED_RINGS: { pts: Vec[]; truth: number[] } | null = null;
export function getRingsData() {
  if (CACHED_RINGS) return CACHED_RINGS;
  const rng = createRng(59);
  const pts: Vec[] = [];
  const truth: number[] = [];
  for (let i = 0; i < 60; i++) {
    const ang = (i / 60) * 2 * Math.PI + rng.normal(0, 0.05);
    const r = 1.0 + rng.normal(0, 0.12);
    pts.push([r * Math.cos(ang), r * Math.sin(ang)]);
    truth.push(0);
  }
  for (let i = 0; i < 80; i++) {
    const ang = (i / 80) * 2 * Math.PI + rng.normal(0, 0.05);
    const r = 2.6 + rng.normal(0, 0.16);
    pts.push([r * Math.cos(ang), r * Math.sin(ang)]);
    truth.push(1);
  }
  CACHED_RINGS = { pts, truth };
  return CACHED_RINGS;
}

export function elbowSweep(pts: readonly Vec[]) {
  const rows: { k: number; inertia: number; silhouette: number; assign: number[]; centroids: Vec[] }[] = [];
  for (let k = 2; k <= 7; k++) {
    // Best of 8 k-Means++ starts so elbow curve is clean and honest
    let best = runKMeans(pts, k, 'kmeans++', 11).final;
    for (let s = 2; s <= 8; s++) {
      const cand = runKMeans(pts, k, 'kmeans++', s * 13 + k).final;
      if (cand.inertia < best.inertia) best = cand;
    }
    rows.push({
      k,
      inertia: best.inertia,
      silhouette: silhouetteScore(pts, best.assign, k),
      assign: best.assign,
      centroids: best.centroids,
    });
  }
  return rows;
}

// ------------------------------------------------------------------ 3. 2D PCA Rotation Explorer

let CACHED_PCA2D: Vec[] | null = null;
export function getPca2DData(): Vec[] {
  if (CACHED_PCA2D) return CACHED_PCA2D;
  const rng = createRng(71);
  const theta = (34 * Math.PI) / 180;
  const ct = Math.cos(theta);
  const st = Math.sin(theta);
  const pts: Vec[] = [];
  for (let i = 0; i < 80; i++) {
    const u = rng.normal(0, 2.0);
    const v = rng.normal(0, 0.55);
    pts.push([ct * u - st * v, st * u + ct * v]);
  }
  CACHED_PCA2D = center(pts).data;
  return CACHED_PCA2D;
}

export function pca2DAtAngle(deg: number) {
  const pts = getPca2DData();
  const rad = (deg * Math.PI) / 180;
  const ux = Math.cos(rad);
  const uy = Math.sin(rad);
  let projVar = 0;
  let reconMse = 0;
  const projections = pts.map(([x, y]) => {
    const t = x * ux + y * uy;
    const px = t * ux;
    const py = t * uy;
    const err2 = (x - px) ** 2 + (y - py) ** 2;
    projVar += t * t;
    reconMse += err2;
    return { x, y, px, py };
  });
  projVar /= pts.length;
  reconMse /= pts.length;
  const totalVar = projVar + reconMse;
  const pca = fitPCA(pts);
  let optDeg = (Math.atan2(pca.components[0][1], pca.components[0][0]) * 180) / Math.PI;
  if (optDeg < 0) optDeg += 180;
  return {
    deg,
    ux,
    uy,
    projVar,
    reconMse,
    totalVar,
    share: projVar / totalVar,
    optDeg,
    optShare: pca.explainedRatio[0],
    projections,
  };
}

// ------------------------------------------------------------------ 4. Lab: Compress & Segment

export type UnsupBrief = {
  id: 'pca-scale' | 'segment' | 'denoise';
  title: string;
  story: string;
};

export const LAB_BRIEFS: UnsupBrief[] = [
  {
    id: 'pca-scale',
    title: 'Brief 1 · Compress 6 Unscaled Sensors (PCA)',
    story: 'Six telemetry sensors capture 2 underlying physical modes (plus mild noise), except Sensor 1 is logged in millimetres (100× larger numeric scale than Sensors 2–6). Choose whether to standardise and how many components m to keep to preserve ≥ 90% of true standardised variance using as few components as possible.',
  },
  {
    id: 'segment',
    title: 'Brief 2 · Segment Unscaled Customers (k-Means)',
    story: 'Customer Spend (€500–€4,500) and Visits (2–18) form natural segments. Evaluated across 40 fixed seeds, choose scaling, k (2–6) and seeding (Random vs k-Means++) to achieve mean Silhouette ≥ 0.72 AND at least 90% of seeds landing on the global optimum.',
  },
  {
    id: 'denoise',
    title: 'Brief 3 · Denoise 8D Logs Before Clustering (PCA → k-Means)',
    story: 'Four clean 2D clusters (in a square) are buried inside 8 columns (6 pure random noise columns destroy cluster separation!). Project onto m PCA components before running k-Means++ to achieve Silhouette ≥ 0.65 AND hidden-segment purity ≥ 95% with the fewest PCA dimensions.',
  },
];

// Brief 1 data: 6 sensors where sensor 0 is 100x scaled, driven by 2 latent factors
let CACHED_B1: Vec[] | null = null;
function getB1Data(): Vec[] {
  if (CACHED_B1) return CACHED_B1;
  const rng = createRng(97);
  const pts: Vec[] = [];
  for (let i = 0; i < 150; i++) {
    const f1 = rng.normal(0, 1);
    const f2 = rng.normal(0, 1);
    pts.push([
      100 * (0.9 * f1 + 0.15 * rng.normal()), // 100x larger scale!
      0.85 * f1 + 0.2 * f2 + 0.15 * rng.normal(),
      -0.8 * f1 + 0.25 * f2 + 0.15 * rng.normal(),
      0.15 * f1 + 0.9 * f2 + 0.15 * rng.normal(),
      -0.1 * f1 + 0.85 * f2 + 0.15 * rng.normal(),
      0.2 * f1 - 0.8 * f2 + 0.15 * rng.normal(),
    ]);
  }
  CACHED_B1 = pts;
  return pts;
}

/**
 * Evaluate how much TRUE standardised variance is preserved by an m-component PCA basis
 * (whether that basis was fitted on raw unscaled data or on standardised data!).
 */
export function evalBrief1(scaled: boolean, m: number) {
  const raw = getB1Data();
  const std = standardize(raw).data;
  const pcaStd = fitPCA(std);
  const pcaFit = scaled ? pcaStd : fitPCA(raw);
  const basis = pcaFit.components.slice(0, m);

  const apparentVar = pcaFit.explainedRatio.slice(0, m).reduce((a, b) => a + b, 0);

  let retained = 0;
  let total = 0;
  for (const x of std) {
    let x2 = 0;
    for (let j = 0; j < x.length; j++) x2 += x[j] * x[j];
    total += x2;
    for (const u of basis) {
      let dot = 0;
      for (let j = 0; j < x.length; j++) dot += u[j] * x[j];
      retained += dot * dot;
    }
  }
  const trueVar = retained / total;
  return { apparentVar, trueVar, m, pass: trueVar >= 0.9 - 1e-9 };
}

// Brief 2 data: 4 unscaled customer clusters (Spend €1000/€4000, Visits 4/15)
let CACHED_B2: Vec[] | null = null;
function getB2Data(): Vec[] {
  if (CACHED_B2) return CACHED_B2;
  const rng = createRng(109);
  const centers: [number, number][] = [
    [1000, 4],
    [1000, 15],
    [4000, 4],
    [4000, 15],
  ];
  const pts: Vec[] = [];
  for (const [sp, vis] of centers) {
    for (let i = 0; i < 30; i++) {
      pts.push([sp + rng.normal(0, 260), vis + rng.normal(0, 1.3)]);
    }
  }
  CACHED_B2 = pts;
  return pts;
}

const B2_CACHE = new Map<string, { meanSil: number; stableSeeds: number; seeds: number; pass: boolean }>();

export function evalBrief2(scaled: boolean, k: number, init: InitMode) {
  const key = `${scaled}:${k}:${init}`;
  const hit = B2_CACHE.get(key);
  if (hit) return hit;
  const raw = getB2Data();
  const std = standardize(raw).data;
  const work = scaled ? std : raw;
  const seeds = 40;
  const runs = Array.from({ length: seeds }, (_, i) => runKMeans(work, k, init, 100 + i * 19).final);
  const sils = runs.map((r) => silhouetteScore(std, r.assign, k));
  const meanSil = sils.reduce((a, b) => a + b, 0) / seeds;
  let bestIn = Infinity;
  for (let i = 0; i < 20; i++) {
    const inVal = runKMeans(work, k, 'kmeans++', 500 + i * 7).final.inertia;
    if (inVal < bestIn) bestIn = inVal;
  }
  const stableSeeds = runs.filter((r) => r.inertia <= bestIn * 1.05).length;
  const res = { meanSil, stableSeeds, seeds, pass: meanSil >= 0.72 - 1e-9 && stableSeeds / seeds >= 0.9 - 1e-9 };
  B2_CACHE.set(key, res);
  return res;
}

// Brief 3 data: 4 tight 2D square clusters buried inside 8D (6 noisy dimensions)
let CACHED_B3: { pts: Vec[]; truth: number[] } | null = null;
function getB3Data() {
  if (CACHED_B3) return CACHED_B3;
  const rng = createRng(131);
  const centers: [number, number][] = [
    [-2.8, -2.8],
    [-2.8, 2.8],
    [2.8, -2.8],
    [2.8, 2.8],
  ];
  const pts: Vec[] = [];
  const truth: number[] = [];
  for (let c = 0; c < 4; c++) {
    for (let i = 0; i < 30; i++) {
      const row = [centers[c][0] + rng.normal(0, 0.55), centers[c][1] + rng.normal(0, 0.55)];
      for (let j = 0; j < 6; j++) row.push(rng.normal(0, 1.0));
      pts.push(row);
      truth.push(c);
    }
  }
  CACHED_B3 = { pts, truth };
  return CACHED_B3;
}

export const B3_MS = [1, 2, 3, 4, 6, 8] as const;

export function evalBrief3(m: number, k: number) {
  const { pts, truth } = getB3Data();
  const pca = fitPCA(pts);
  const proj = m === 8 ? pts : projectPCA(pts, pca, m);
  let best = runKMeans(proj, k, 'kmeans++', 21).final;
  for (let s = 2; s <= 5; s++) {
    const cand = runKMeans(proj, k, 'kmeans++', s * 29).final;
    if (cand.inertia < best.inertia) best = cand;
  }
  const sil = silhouetteScore(proj, best.assign, k);
  const purity = clusterPurity(best.assign, truth, k);
  return { sil, purity, m, k, pass: sil >= 0.65 - 1e-9 && purity >= 0.95 - 1e-9 };
}

export function labExhaustive(id: UnsupBrief['id']) {
  if (id === 'pca-scale') {
    let total = 0;
    let passing = 0;
    let bestM = Infinity;
    const best: { scaled: boolean; m: number; trueVar: number }[] = [];
    for (const scaled of [false, true])
      for (const m of [1, 2, 3, 4, 5, 6]) {
        total++;
        const r = evalBrief1(scaled, m);
        if (!r.pass) continue;
        passing++;
        if (m < bestM) {
          bestM = m;
          best.length = 0;
          best.push({ scaled, m, trueVar: r.trueVar });
        } else if (m === bestM) best.push({ scaled, m, trueVar: r.trueVar });
      }
    return { total, passing, summary: `Fewest components: m = ${bestM} (${best.map((b) => `${b.scaled ? 'standardised' : 'raw'}, ${(100 * b.trueVar).toFixed(1)}% true variance`).join('; ')})` };
  }
  if (id === 'segment') {
    let total = 0;
    let passing = 0;
    let bestSil = -1;
    let bestDesc = '';
    for (const scaled of [false, true])
      for (const k of [2, 3, 4, 5, 6])
        for (const init of ['random', 'kmeans++'] as InitMode[]) {
          total++;
          const r = evalBrief2(scaled, k, init);
          if (!r.pass) continue;
          passing++;
          if (r.meanSil > bestSil) {
            bestSil = r.meanSil;
            bestDesc = `${scaled ? 'standardised' : 'raw'}, k = ${k}, ${init} (Silhouette ${r.meanSil.toFixed(3)}, ${r.stableSeeds}/${r.seeds} seeds optimal)`;
          }
        }
    return { total, passing, summary: `Best: ${bestDesc}` };
  }
  let total = 0;
  let passing = 0;
  let bestM = Infinity;
  let bestDesc = '';
  for (const m of B3_MS)
    for (const k of [2, 3, 4, 5]) {
      total++;
      const r = evalBrief3(m, k);
      if (!r.pass) continue;
      passing++;
      if (m < bestM) {
        bestM = m;
        bestDesc = `m = ${m} PCA components, k = ${k} (Silhouette ${r.sil.toFixed(3)}, purity ${(100 * r.purity).toFixed(1)}%)`;
      }
    }
  return { total, passing, summary: `Fewest PCA dims: ${bestDesc}` };
}
