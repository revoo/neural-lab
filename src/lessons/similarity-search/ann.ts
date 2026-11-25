// authored-by: Gemini Antigravity
/**
 * Similarity Search (ANN: IVF, Random-Hyperplane LSH, & Vector Compression) from scratch:
 *  - Exact Cosine vs Dot Product vs Euclidean ranking comparator (and unit-norm equivalence proof)
 *  - Real 2D & 16D IVF (k-Means Voronoi partitioning + multi-probe nprobe sweep) measuring real Recall@5 vs scan %
 *  - Real Multi-Table Random-Hyperplane LSH (SimHash) measuring exact collision probability 1 - theta/pi & Recall@5
 *  - Exact 10M x 1536D vector-database RAM calculator (FP32 / FP16 / INT8 / PQ-64 / Binary) + 3-brief exhaustive Lab
 */
import { createRng } from '../../lib/rng';

export type Vec = number[];

export function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a: readonly number[]): number {
  return Math.sqrt(dot(a, a));
}

export function sqDist(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

export function cosine(a: readonly number[], b: readonly number[]): number {
  const d = norm(a) * norm(b);
  return d > 1e-12 ? dot(a, b) / d : 0;
}

export function normalize(a: readonly number[]): Vec {
  const n = norm(a) || 1;
  return a.map((v) => v / n);
}

// ------------------------------------------------------------------ 1. Metric Comparison (Cosine vs Dot vs L2)

export type MetricCand = { id: string; label: string; vec: [number, number] };

export function compareMetrics(query: [number, number], cands: readonly MetricCand[], unitNorm: boolean) {
  const q = unitNorm ? (normalize(query) as [number, number]) : query;
  const rows = cands.map((c) => {
    const v = unitNorm ? (normalize(c.vec) as [number, number]) : c.vec;
    return {
      id: c.id,
      label: c.label,
      vec: v,
      cos: cosine(q, v),
      dotVal: dot(q, v),
      l2: Math.sqrt(sqDist(q, v)),
    };
  });
  const bestCos = rows.reduce((a, b) => (b.cos > a.cos ? b : a)).id;
  const bestDot = rows.reduce((a, b) => (b.dotVal > a.dotVal ? b : a)).id;
  const bestL2 = rows.reduce((a, b) => (b.l2 < a.l2 ? b : a)).id;
  return { q, rows, bestCos, bestDot, bestL2, allAgree: bestCos === bestDot && bestDot === bestL2 };
}

// ------------------------------------------------------------------ 2. IVF (Inverted File Index via k-Means Voronoi Cells)

export type IvfIndex = {
  nlist: number;
  centroids: Vec[];
  buckets: number[][]; // bucket c -> indices into data
  assign: number[];
};

export function buildIvf(data: readonly Vec[], nlist: number, seed = 31): IvfIndex {
  const rng = createRng(seed);
  const n = data.length;
  const d = data[0].length;
  // k-Means++ init
  const centroids: Vec[] = [[...data[rng.int(n)]]];
  for (let c = 1; c < nlist; c++) {
    const d2 = data.map((p) => {
      let best = Infinity;
      for (const mu of centroids) {
        const dd = sqDist(p, mu);
        if (dd < best) best = dd;
      }
      return best;
    });
    const tot = d2.reduce((a, b) => a + b, 0) || 1;
    let r = rng.next() * tot;
    let pick = n - 1;
    for (let i = 0; i < n; i++) {
      r -= d2[i];
      if (r <= 0) { pick = i; break; }
    }
    centroids.push([...data[pick]]);
  }

  let assign = new Array<number>(n).fill(0);
  for (let iter = 0; iter < 15; iter++) {
    let changed = false;
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestD = sqDist(data[i], centroids[0]);
      for (let c = 1; c < nlist; c++) {
        const dd = sqDist(data[i], centroids[c]);
        if (dd < bestD) { bestD = dd; best = c; }
      }
      if (best !== assign[i]) { assign[i] = best; changed = true; }
    }
    if (!changed && iter > 0) break;
    const sums = Array.from({ length: nlist }, () => new Array<number>(d).fill(0));
    const cnts = new Array<number>(nlist).fill(0);
    for (let i = 0; i < n; i++) {
      cnts[assign[i]]++;
      for (let j = 0; j < d; j++) sums[assign[i]][j] += data[i][j];
    }
    for (let c = 0; c < nlist; c++) {
      if (cnts[c] > 0) centroids[c] = sums[c].map((v) => v / cnts[c]);
    }
  }

  const buckets: number[][] = Array.from({ length: nlist }, () => []);
  for (let i = 0; i < n; i++) buckets[assign[i]].push(i);
  return { nlist, centroids, buckets, assign };
}

export function exactTopK(data: readonly Vec[], query: readonly number[], k: number): number[] {
  return data
    .map((p, i) => ({ i, d: sqDist(p, query) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, k)
    .map((x) => x.i);
}

export function queryIvf(data: readonly Vec[], ivf: IvfIndex, query: readonly number[], nprobe: number, k: number) {
  const probedBuckets = ivf.centroids
    .map((c, idx) => ({ idx, d: sqDist(c, query) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, nprobe)
    .map((x) => x.idx);

  const cands: { i: number; d: number }[] = [];
  for (const b of probedBuckets) {
    for (const idx of ivf.buckets[b]) {
      cands.push({ i: idx, d: sqDist(data[idx], query) });
    }
  }
  cands.sort((a, b) => a.d - b.d);
  const topK = cands.slice(0, k).map((x) => x.i);
  // Total distance evals = nlist centroid checks + scanned candidates
  const distEvals = ivf.nlist + cands.length;
  return { topK, probedBuckets, scanned: cands.length, distEvals };
}

// 2D Interactive IVF visual dataset (240 points in 12 clusters)
let CACHED_2D: { data: Vec[]; ivf: IvfIndex; boundaryQuery: Vec } | null = null;
export function getIvf2DScene() {
  if (CACHED_2D) return CACHED_2D;
  const rng = createRng(607);
  const data: Vec[] = [];
  const centers: [number, number][] = [
    [-3.2, 2.8], [-1.0, 3.0], [1.3, 2.9], [3.3, 2.4],
    [-3.1, 0.2], [-0.9, 0.4], [1.2, 0.1], [3.2, -0.3],
    [-2.9, -2.5], [-0.8, -2.6], [1.4, -2.4], [3.1, -2.8],
  ];
  for (const [cx, cy] of centers) {
    for (let i = 0; i < 20; i++) {
      data.push([cx + rng.normal(0, 0.55), cy + rng.normal(0, 0.55)]);
    }
  }
  const ivf = buildIvf(data, 12, 31);
  // Query intentionally perched right on a Voronoi fence between central clusters
  const boundaryQuery: Vec = [0.15, 0.25];
  CACHED_2D = { data, ivf, boundaryQuery };
  return CACHED_2D;
}

// ------------------------------------------------------------------ 3. 16D Benchmark Dataset for IVF & LSH

let CACHED_16D: { data: Vec[]; queries: Vec[]; exactTop5: Set<number>[] } | null = null;
export function getAnn16DData() {
  if (CACHED_16D) return CACHED_16D;
  const rng = createRng(733);
  const d = 16;
  const nClusters = 24;
  const clusterCenters = Array.from({ length: nClusters }, () => normalize(Array.from({ length: d }, () => rng.normal(0, 1))));
  const data: Vec[] = [];
  for (let i = 0; i < 600; i++) {
    const c = clusterCenters[i % nClusters];
    data.push(normalize(c.map((v) => v + rng.normal(0, 0.28))));
  }
  const queries: Vec[] = [];
  for (let q = 0; q < 60; q++) {
    const c = clusterCenters[q % nClusters];
    queries.push(normalize(c.map((v) => v + rng.normal(0, 0.34))));
  }
  const exactTop5 = queries.map((q) => new Set(exactTopK(data, q, 5)));
  CACHED_16D = { data, queries, exactTop5 };
  return CACHED_16D;
}

const IVF_CACHE = new Map<string, { recall5: number; scanFrac: number; speedup: number }>();

export function benchmarkIvf16D(nlist: number, nprobe: number) {
  const key = `${nlist}:${nprobe}`;
  const hit = IVF_CACHE.get(key);
  if (hit) return hit;
  const { data, queries, exactTop5 } = getAnn16DData();
  const ivf = buildIvf(data, nlist, 41);
  let hits = 0;
  let totalEvals = 0;
  for (let q = 0; q < queries.length; q++) {
    const res = queryIvf(data, ivf, queries[q], nprobe, 5);
    totalEvals += res.distEvals;
    for (const id of res.topK) if (exactTop5[q].has(id)) hits++;
  }
  const recall5 = hits / (queries.length * 5);
  const meanEvals = totalEvals / queries.length;
  const scanFrac = meanEvals / data.length;
  const res = { recall5, scanFrac, speedup: data.length / meanEvals };
  IVF_CACHE.set(key, res);
  return res;
}

// ------------------------------------------------------------------ 4. Multi-Table Random-Hyperplane LSH (SimHash)

export function lshCollisionProb(angleDeg: number, bits: number, tables: number): number {
  const pOneBit = 1 - angleDeg / 180;
  const pOneTable = pOneBit ** bits;
  return 1 - (1 - pOneTable) ** tables;
}

const LSH_CACHE = new Map<string, { recall5: number; scanFrac: number; speedup: number }>();

export function benchmarkLsh16D(bits: number, tables: number) {
  const key = `${bits}:${tables}`;
  const hit = LSH_CACHE.get(key);
  if (hit) return hit;
  const { data, queries, exactTop5 } = getAnn16DData();
  const d = data[0].length;
  const rng = createRng(821);

  // Build L hash tables of b random unit hyperplanes each
  const planes: Vec[][] = [];
  const buckets: Map<number, number[]>[] = [];
  for (let t = 0; t < tables; t++) {
    const tPlanes = Array.from({ length: bits }, () => Array.from({ length: d }, () => rng.normal(0, 1)));
    planes.push(tPlanes);
    const map = new Map<number, number[]>();
    for (let i = 0; i < data.length; i++) {
      let code = 0;
      for (let b = 0; b < bits; b++) if (dot(data[i], tPlanes[b]) >= 0) code |= 1 << b;
      const arr = map.get(code);
      if (arr) arr.push(i);
      else map.set(code, [i]);
    }
    buckets.push(map);
  }

  let hits = 0;
  let totalScanned = 0;
  for (let q = 0; q < queries.length; q++) {
    const candSet = new Set<number>();
    for (let t = 0; t < tables; t++) {
      let code = 0;
      for (let b = 0; b < bits; b++) if (dot(queries[q], planes[t][b]) >= 0) code |= 1 << b;
      const arr = buckets[t].get(code);
      if (arr) for (const idx of arr) candSet.add(idx);
    }
    const cands = Array.from(candSet)
      .map((i) => ({ i, d: sqDist(data[i], queries[q]) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, 5);
    totalScanned += candSet.size;
    for (const c of cands) if (exactTop5[q].has(c.i)) hits++;
  }
  const recall5 = hits / (queries.length * 5);
  const meanScanned = Math.max(1, totalScanned / queries.length);
  const res = { recall5, scanFrac: meanScanned / data.length, speedup: data.length / meanScanned };
  LSH_CACHE.set(key, res);
  return res;
}

// ------------------------------------------------------------------ 5. Vector Compression RAM Calculator & 3-Brief Lab

export type CompFormat = 'fp32' | 'fp16' | 'int8' | 'pq64' | 'binary';

export type CompSpec = {
  id: CompFormat;
  label: string;
  bytesPerVec: number; // for d = 1536
  typicalRecall: number; // realistic reference recall retention for 1536D retrieval
};

export const COMP_SPECS: CompSpec[] = [
  { id: 'fp32', label: 'FP32 (4 B/dim)', bytesPerVec: 1536 * 4, typicalRecall: 1.0 },
  { id: 'fp16', label: 'FP16 (2 B/dim)', bytesPerVec: 1536 * 2, typicalRecall: 0.999 },
  { id: 'int8', label: 'Scalar INT8 (1 B/dim)', bytesPerVec: 1536 * 1, typicalRecall: 0.985 },
  { id: 'pq64', label: 'PQ-64 (64 bytes/vec)', bytesPerVec: 64, typicalRecall: 0.89 },
  { id: 'binary', label: 'Binary 1-bit (192 B/vec)', bytesPerVec: 1536 / 8, typicalRecall: 0.82 },
];

export function vectorDbGiB(nVectors: number, bytesPerVec: number): number {
  return (nVectors * bytesPerVec) / (1024 ** 3);
}

export type AnnBriefId = 'ivf' | 'lsh' | 'ram';

export const ANN_LAB_BRIEFS: { id: AnnBriefId; title: string; story: string }[] = [
  {
    id: 'ivf',
    title: 'Brief 1 · Tune IVF Index Under a 25% Compute Budget (600 × 16D)',
    story: 'Search 600 16D embeddings across 60 queries. Choose number of Voronoi buckets (nlist ∈ {8, 16, 32}) and probed buckets (nprobe ∈ {1, 2, 3, 4, 6}) to achieve Recall@5 ≥ 90% while doing ≤ 25% of brute-force distance evaluations. Optional: minimise distance work!',
  },
  {
    id: 'lsh',
    title: 'Brief 2 · Multi-Table LSH Recall vs Candidate Scan (600 × 16D)',
    story: 'Pick hash width (bits b ∈ {4, 6, 8}) and independent hash tables (L ∈ {1, 4, 8, 16}) to achieve Recall@5 ≥ 85% while scanning ≤ 35% of the database. Watch how data-oblivious LSH compares honestly against data-aware IVF!',
  },
  {
    id: 'ram',
    title: 'Brief 3 · Fit 10 Million 1,536D Embeddings Under 16.0 GiB RAM at ≥ 95% Recall',
    story: 'You need to keep 10,000,000 1,536-dim embeddings memory-resident on a 16 GiB RAM node without losing retrieval quality (typical recall retention ≥ 95%). Choose the storage format.',
  },
];

export const LAB_NLISTS = [8, 16, 32] as const;
export const LAB_NPROBES = [1, 2, 3, 4, 6] as const;
export const LAB_LSH_BITS = [4, 6, 8] as const;
export const LAB_LSH_TABLES = [1, 4, 8, 16] as const;

export function evalAnnBrief1(nlist: number, nprobe: number) {
  const r = benchmarkIvf16D(nlist, nprobe);
  return { nlist, nprobe, ...r, pass: r.recall5 >= 0.9 - 1e-9 && r.scanFrac <= 0.25 + 1e-9 };
}

export function evalAnnBrief2(bits: number, tables: number) {
  const r = benchmarkLsh16D(bits, tables);
  return { bits, tables, ...r, pass: r.recall5 >= 0.85 - 1e-9 && r.scanFrac <= 0.35 + 1e-9 };
}

export function evalAnnBrief3(fmt: CompFormat) {
  const spec = COMP_SPECS.find((s) => s.id === fmt)!;
  const gib = vectorDbGiB(10_000_000, spec.bytesPerVec);
  const ratio = (1536 * 4) / spec.bytesPerVec;
  return { fmt, gib, ratio, typicalRecall: spec.typicalRecall, pass: gib <= 16.0 + 1e-9 && spec.typicalRecall >= 0.95 - 1e-9 };
}

export function annLabExhaustive(id: AnnBriefId) {
  if (id === 'ivf') {
    let total = 0;
    let passing = 0;
    let bestScan = Infinity;
    let bestDesc = '';
    for (const nl of LAB_NLISTS)
      for (const np of LAB_NPROBES) {
        total++;
        const r = evalAnnBrief1(nl, np);
        if (!r.pass) continue;
        passing++;
        if (r.scanFrac < bestScan) {
          bestScan = r.scanFrac;
          bestDesc = `nlist=${nl}, nprobe=${np} (Recall@5 ${(100 * r.recall5).toFixed(1)}%, scan ${(100 * r.scanFrac).toFixed(1)}% = ${r.speedup.toFixed(1)}× faster)`;
        }
      }
    return { total, passing, summary: `Fastest passing: ${bestDesc}` };
  }
  if (id === 'lsh') {
    let total = 0;
    let passing = 0;
    let bestScan = Infinity;
    let bestDesc = '';
    for (const b of LAB_LSH_BITS)
      for (const t of LAB_LSH_TABLES) {
        total++;
        const r = evalAnnBrief2(b, t);
        if (!r.pass) continue;
        passing++;
        if (r.scanFrac < bestScan) {
          bestScan = r.scanFrac;
          bestDesc = `b=${b} bits, L=${t} tables (Recall@5 ${(100 * r.recall5).toFixed(1)}%, scan ${(100 * r.scanFrac).toFixed(1)}%)`;
        }
      }
    return { total, passing, summary: `Fastest passing: ${bestDesc}` };
  }
  const all = COMP_SPECS.map((s) => evalAnnBrief3(s.id));
  const pass = all.filter((x) => x.pass);
  return {
    total: all.length,
    passing: pass.length,
    summary: `Unique solution: Scalar INT8 (${pass[0].gib.toFixed(2)} GiB, 4× smaller than FP32’s 57.22 GiB, 98.5% recall retention)`,
  };
}
