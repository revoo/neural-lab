// authored-by: Gemini Antigravity
/**
 * Rubber-sheet space folding: a real 15-parameter 2 -> 2 -> 2 -> 1 GELU network
 * trained deterministically on Concentric Rings and Two Moons.
 *
 * Because both hidden layers have width 2, every intermediate activation lives in exact 2D
 * with zero dimensionality reduction — we can warp a regular 13 x 13 coordinate mesh and the
 * p = 0.5 decision contour directly through Layer 1 and Layer 2 and watch a curved boundary
 * unbend into a single straight knife cut.
 */

export type FoldDatasetId = 'rings' | 'moons';

export type FoldPoint = { x: number; y: number; label: 0 | 1 };

export type Vec2 = [number, number];

export type FoldNet = {
  W1: [[number, number], [number, number]];
  b1: [number, number];
  W2: [[number, number], [number, number]];
  b2: [number, number];
  W3: [number, number];
  b3: number;
};

export type StageTrace = {
  h0: Vec2;
  h1: Vec2;
  h2: Vec2;
  logit: number;
  prob: number;
};

export type ViewBoxBounds = {
  cx: number;
  cy: number;
  half: number;
};

export type BoundarySegment = {
  a: [Vec2, Vec2, Vec2]; // stages 0, 1, 2
  b: [Vec2, Vec2, Vec2];
};

export type SpaceFoldResult = {
  id: FoldDatasetId;
  title: string;
  net: FoldNet;
  points: FoldPoint[];
  pointStages: StageTrace[];
  acc0: number; // best straight-line accuracy on raw input
  acc1: number; // best straight-line accuracy after Layer 1
  acc2: number; // trained output straight-line accuracy after Layer 2
  denseAcc: number; // accuracy on 400 noiseless continuous test points
  stageBounds: [ViewBoxBounds, ViewBoxBounds, ViewBoxBounds];
  boundarySegments: BoundarySegment[];
  paramCount: number;
};

const GELU_K = 1.702;

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function sigmoid(z: number): number {
  const c = Math.max(-35, Math.min(35, z));
  return 1 / (1 + Math.exp(-c));
}

/** Smooth sigmoid-approximated GELU: z * sigmoid(1.702 * z), differentiable everywhere. */
export function gelu(z: number): number {
  return z * sigmoid(GELU_K * z);
}

export function geluDeriv(z: number): number {
  const s = sigmoid(GELU_K * z);
  return s + z * GELU_K * s * (1 - s);
}

export function makeDataset(id: FoldDatasetId, nPerClass = 24, jitter = 1): FoldPoint[] {
  const rng = mulberry32(id === 'rings' ? 7 : 11);
  const pts: FoldPoint[] = [];
  if (id === 'rings') {
    for (let i = 0; i < nPerClass; i++) {
      const a = (2 * Math.PI * i) / nPerClass + jitter * (rng() - 0.5) * 0.14;
      const r = 0.42 + jitter * (rng() - 0.5) * 0.12;
      pts.push({ x: r * Math.cos(a), y: r * Math.sin(a), label: 1 });
    }
    for (let i = 0; i < nPerClass; i++) {
      const a = (2 * Math.PI * i) / nPerClass + jitter * (rng() - 0.5) * 0.14;
      const r = 1.15 + jitter * (rng() - 0.5) * 0.14;
      pts.push({ x: r * Math.cos(a), y: r * Math.sin(a), label: 0 });
    }
  } else {
    for (let i = 0; i < nPerClass; i++) {
      const t = (Math.PI * (i + 0.5)) / nPerClass + jitter * (rng() - 0.5) * 0.08;
      const r = 0.95 + jitter * (rng() - 0.5) * 0.10;
      pts.push({ x: r * Math.cos(t) - 0.42, y: r * Math.sin(t) - 0.28, label: 1 });
    }
    for (let i = 0; i < nPerClass; i++) {
      const t = (Math.PI * (i + 0.5)) / nPerClass + jitter * (rng() - 0.5) * 0.08;
      const r = 0.95 + jitter * (rng() - 0.5) * 0.10;
      pts.push({ x: 0.42 - r * Math.cos(t), y: 0.28 - r * Math.sin(t), label: 0 });
    }
  }
  return pts;
}

export function forwardStages(net: FoldNet, x: number, y: number): StageTrace {
  const z10 = net.W1[0][0] * x + net.W1[0][1] * y + net.b1[0];
  const z11 = net.W1[1][0] * x + net.W1[1][1] * y + net.b1[1];
  const h1: Vec2 = [gelu(z10), gelu(z11)];
  const z20 = net.W2[0][0] * h1[0] + net.W2[0][1] * h1[1] + net.b2[0];
  const z21 = net.W2[1][0] * h1[0] + net.W2[1][1] * h1[1] + net.b2[1];
  const h2: Vec2 = [gelu(z20), gelu(z21)];
  const logit = net.W3[0] * h2[0] + net.W3[1] * h2[1] + net.b3;
  return { h0: [x, y], h1, h2, logit, prob: sigmoid(logit) };
}

/** Exact binary cross-entropy loss (plus L2 decay on weights) and analytic gradients. */
export function lossAndGrad(net: FoldNet, pts: readonly FoldPoint[], wd = 0.0005): { loss: number; grad: FoldNet } {
  const g: FoldNet = {
    W1: [[0, 0], [0, 0]],
    b1: [0, 0],
    W2: [[0, 0], [0, 0]],
    b2: [0, 0],
    W3: [0, 0],
    b3: 0,
  };
  let loss = 0;
  const invN = 1 / pts.length;
  for (const p of pts) {
    const z10 = net.W1[0][0] * p.x + net.W1[0][1] * p.y + net.b1[0];
    const z11 = net.W1[1][0] * p.x + net.W1[1][1] * p.y + net.b1[1];
    const h10 = gelu(z10);
    const h11 = gelu(z11);
    const z20 = net.W2[0][0] * h10 + net.W2[0][1] * h11 + net.b2[0];
    const z21 = net.W2[1][0] * h10 + net.W2[1][1] * h11 + net.b2[1];
    const h20 = gelu(z20);
    const h21 = gelu(z21);
    const z3 = net.W3[0] * h20 + net.W3[1] * h21 + net.b3;
    const prob = sigmoid(z3);
    // Numerically stable BCE: max(z, 0) - z*y + log(1 + exp(-|z|))
    loss += (Math.max(z3, 0) - z3 * p.label + Math.log1p(Math.exp(-Math.abs(z3)))) * invN;

    const dz3 = (prob - p.label) * invN;
    g.b3 += dz3;
    g.W3[0] += dz3 * h20;
    g.W3[1] += dz3 * h21;

    const dz20 = net.W3[0] * dz3 * geluDeriv(z20);
    const dz21 = net.W3[1] * dz3 * geluDeriv(z21);
    g.b2[0] += dz20;
    g.b2[1] += dz21;
    g.W2[0][0] += dz20 * h10;
    g.W2[0][1] += dz20 * h11;
    g.W2[1][0] += dz21 * h10;
    g.W2[1][1] += dz21 * h11;

    const dz10 = (net.W2[0][0] * dz20 + net.W2[1][0] * dz21) * geluDeriv(z10);
    const dz11 = (net.W2[0][1] * dz20 + net.W2[1][1] * dz21) * geluDeriv(z11);
    g.b1[0] += dz10;
    g.b1[1] += dz11;
    g.W1[0][0] += dz10 * p.x;
    g.W1[0][1] += dz10 * p.y;
    g.W1[1][0] += dz11 * p.x;
    g.W1[1][1] += dz11 * p.y;
  }

  // L2 weight decay on weights (not biases)
  const addWd = (w: number) => {
    loss += 0.5 * wd * w * w;
    return wd * w;
  };
  g.W1[0][0] += addWd(net.W1[0][0]);
  g.W1[0][1] += addWd(net.W1[0][1]);
  g.W1[1][0] += addWd(net.W1[1][0]);
  g.W1[1][1] += addWd(net.W1[1][1]);
  g.W2[0][0] += addWd(net.W2[0][0]);
  g.W2[0][1] += addWd(net.W2[0][1]);
  g.W2[1][0] += addWd(net.W2[1][0]);
  g.W2[1][1] += addWd(net.W2[1][1]);
  g.W3[0] += addWd(net.W3[0]);
  g.W3[1] += addWd(net.W3[1]);

  return { loss, grad: g };
}

/** Best straight-line accuracy in 2D via 1-degree angle sweep + exact threshold scan. */
export function bestLinearAccuracy(coords: readonly Vec2[], labels: readonly (0 | 1)[]): number {
  let best = 0;
  const totalPos = labels.reduce<number>((s, l) => s + l, 0);
  for (let deg = 0; deg < 180; deg++) {
    const th = (Math.PI * deg) / 180;
    const c = Math.cos(th);
    const s = Math.sin(th);
    const proj = coords
      .map(([x, y], idx) => ({ v: c * x + s * y, y: labels[idx] }))
      .sort((a, b) => a.v - b.v);
    let posRight = totalPos;
    let negLeft = 0;
    for (let i = 0; i < proj.length; i++) {
      const acc = (negLeft + posRight) / proj.length;
      if (acc > best) best = acc;
      if (1 - acc > best) best = 1 - acc;
      if (proj[i].y === 1) posRight--;
      else negLeft++;
    }
  }
  return best;
}

/** Interpolate a 2D point smoothly across layers: t=0 input, t=1 Layer 1, t=2 Layer 2. */
export function interpolateStage(tr: Pick<StageTrace, 'h0' | 'h1' | 'h2'>, t: number): Vec2 {
  const tc = Math.max(0, Math.min(2, t));
  if (tc <= 1) {
    return [tr.h0[0] + (tr.h1[0] - tr.h0[0]) * tc, tr.h0[1] + (tr.h1[1] - tr.h0[1]) * tc];
  }
  const u = tc - 1;
  return [tr.h1[0] + (tr.h2[0] - tr.h1[0]) * u, tr.h1[1] + (tr.h2[1] - tr.h1[1]) * u];
}

/** Deform a regular n x n coordinate mesh over [-bound, bound]^2 at layer time t in [0, 2]. */
export function deformGrid(net: FoldNet, t: number, n = 13, bound = 1.45): Vec2[][] {
  const rows: Vec2[][] = [];
  for (let r = 0; r < n; r++) {
    const y = -bound + (2 * bound * r) / (n - 1);
    const row: Vec2[] = [];
    for (let c = 0; c < n; c++) {
      const x = -bound + (2 * bound * c) / (n - 1);
      row.push(interpolateStage(forwardStages(net, x, y), t));
    }
    rows.push(row);
  }
  return rows;
}

/** Smoothly interpolate camera bounds between stages so the warped rubber sheet stays framed. */
export function interpolateBounds(bounds: readonly [ViewBoxBounds, ViewBoxBounds, ViewBoxBounds], t: number): ViewBoxBounds {
  const tc = Math.max(0, Math.min(2, t));
  const [a, b, u] = tc <= 1 ? [bounds[0], bounds[1], tc] : [bounds[1], bounds[2], tc - 1];
  return {
    cx: a.cx + (b.cx - a.cx) * u,
    cy: a.cy + (b.cy - a.cy) * u,
    half: a.half + (b.half - a.half) * u,
  };
}

/** Clip the exact final-layer straight knife line W3[0]*u + W3[1]*v + b3 = 0 to the current box. */
export function outputBoundaryLine(net: FoldNet, box: ViewBoxBounds): [Vec2, Vec2] | null {
  const [w0, w1] = net.W3;
  const c = net.b3;
  const xMin = box.cx - box.half;
  const xMax = box.cx + box.half;
  const yMin = box.cy - box.half;
  const yMax = box.cy + box.half;
  const hits: Vec2[] = [];
  if (Math.abs(w1) > 1e-9) {
    for (const x of [xMin, xMax]) {
      const y = -(w0 * x + c) / w1;
      if (y >= yMin - 1e-6 && y <= yMax + 1e-6) hits.push([x, Math.max(yMin, Math.min(yMax, y))]);
    }
  }
  if (Math.abs(w0) > 1e-9) {
    for (const y of [yMin, yMax]) {
      const x = -(w1 * y + c) / w0;
      if (x >= xMin - 1e-6 && x <= xMax + 1e-6) hits.push([Math.max(xMin, Math.min(xMax, x)), y]);
    }
  }
  // Deduplicate corners
  const uniq: Vec2[] = [];
  for (const h of hits) {
    if (!uniq.some((q) => Math.hypot(q[0] - h[0], q[1] - h[1]) < 1e-4)) uniq.push(h);
  }
  if (uniq.length < 2) return null;
  return [uniq[0], uniq[1]];
}

/** Marching-squares extraction of the p(x, y) = 0.5 contour in input space, traced through every layer. */
function extractBoundarySegments(net: FoldNet, bound = 1.45, gridN = 44): BoundarySegment[] {
  const logits: number[][] = [];
  for (let r = 0; r <= gridN; r++) {
    const y = -bound + (2 * bound * r) / gridN;
    const row: number[] = [];
    for (let c = 0; c <= gridN; c++) {
      const x = -bound + (2 * bound * c) / gridN;
      row.push(forwardStages(net, x, y).logit);
    }
    logits.push(row);
  }
  const segs: BoundarySegment[] = [];
  const step = (2 * bound) / gridN;
  const interp = (x0: number, y0: number, v0: number, x1: number, y1: number, v1: number): Vec2 => {
    const denom = v1 - v0;
    const f = Math.abs(denom) < 1e-12 ? 0.5 : Math.max(0, Math.min(1, -v0 / denom));
    return [x0 + (x1 - x0) * f, y0 + (y1 - y0) * f];
  };
  for (let r = 0; r < gridN; r++) {
    const y0 = -bound + r * step;
    const y1 = y0 + step;
    for (let c = 0; c < gridN; c++) {
      const x0 = -bound + c * step;
      const x1 = x0 + step;
      const v00 = logits[r][c];
      const v10 = logits[r][c + 1];
      const v11 = logits[r + 1][c + 1];
      const v01 = logits[r + 1][c];
      const cross: Vec2[] = [];
      if ((v00 < 0) !== (v10 < 0)) cross.push(interp(x0, y0, v00, x1, y0, v10));
      if ((v10 < 0) !== (v11 < 0)) cross.push(interp(x1, y0, v10, x1, y1, v11));
      if ((v11 < 0) !== (v01 < 0)) cross.push(interp(x1, y1, v11, x0, y1, v01));
      if ((v01 < 0) !== (v00 < 0)) cross.push(interp(x0, y1, v01, x0, y0, v00));
      if (cross.length >= 2) {
        const ta = forwardStages(net, cross[0][0], cross[0][1]);
        const tb = forwardStages(net, cross[1][0], cross[1][1]);
        segs.push({ a: [ta.h0, ta.h1, ta.h2], b: [tb.h0, tb.h1, tb.h2] });
      }
    }
  }
  return segs;
}

function fitBounds(coords: readonly Vec2[], padRatio = 0.22): ViewBoxBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of coords) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const cx = 0.5 * (minX + maxX);
  const cy = 0.5 * (minY + maxY);
  const half = Math.max(0.5 * Math.max(maxX - minX, maxY - minY) * (1 + padRatio), 0.35);
  return { cx, cy, half };
}

const cache = new Map<FoldDatasetId, SpaceFoldResult>();

/** Deterministically train the 15-parameter 2->2->2->1 GELU network on rings or moons. */
export function trainSpaceFold(id: FoldDatasetId): SpaceFoldResult {
  const cached = cache.get(id);
  if (cached) return cached;

  const points = makeDataset(id, 24, 1);
  const seed = id === 'rings' ? 6 : 4;
  const rng = mulberry32(seed);
  const randW = () => (rng() * 2 - 1) * 1.1;
  const randB = () => (rng() - 0.5) * 0.3;

  const net: FoldNet = {
    W1: [[randW(), randW()], [randW(), randW()]],
    b1: [randB(), randB()],
    W2: [[randW(), randW()], [randW(), randW()]],
    b2: [randB(), randB()],
    W3: [randW(), randW()],
    b3: randB(),
  };

  // Flat view helpers for deterministic Adam
  const getParams = (n: FoldNet): number[] => [
    n.W1[0][0], n.W1[0][1], n.W1[1][0], n.W1[1][1], n.b1[0], n.b1[1],
    n.W2[0][0], n.W2[0][1], n.W2[1][0], n.W2[1][1], n.b2[0], n.b2[1],
    n.W3[0], n.W3[1], n.b3,
  ];
  const setParams = (n: FoldNet, v: number[]) => {
    n.W1[0][0] = v[0]; n.W1[0][1] = v[1]; n.W1[1][0] = v[2]; n.W1[1][1] = v[3];
    n.b1[0] = v[4]; n.b1[1] = v[5];
    n.W2[0][0] = v[6]; n.W2[0][1] = v[7]; n.W2[1][0] = v[8]; n.W2[1][1] = v[9];
    n.b2[0] = v[10]; n.b2[1] = v[11];
    n.W3[0] = v[12]; n.W3[1] = v[13]; n.b3 = v[14];
  };

  const params = getParams(net);
  const m = new Array(15).fill(0);
  const v = new Array(15).fill(0);
  const steps = 1000;
  const lr = 0.04;
  const wd = 0.0005;
  const b1 = 0.9;
  const b2 = 0.999;
  const eps = 1e-8;

  for (let step = 1; step <= steps; step++) {
    setParams(net, params);
    const { grad } = lossAndGrad(net, points, wd);
    const gv = getParams(grad);
    const bc1 = 1 - b1 ** step;
    const bc2 = 1 - b2 ** step;
    for (let i = 0; i < 15; i++) {
      m[i] = b1 * m[i] + (1 - b1) * gv[i];
      v[i] = b2 * v[i] + (1 - b2) * gv[i] * gv[i];
      params[i] -= (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + eps);
    }
  }
  setParams(net, params);

  const pointStages = points.map((p) => forwardStages(net, p.x, p.y));
  const labels = points.map((p) => p.label);
  const acc0 = bestLinearAccuracy(pointStages.map((s) => s.h0), labels);
  const acc1 = bestLinearAccuracy(pointStages.map((s) => s.h1), labels);
  const acc2 = pointStages.filter((s, i) => (s.logit >= 0 ? 1 : 0) === points[i].label).length / points.length;

  const densePts = makeDataset(id, 200, 0);
  const denseAcc =
    densePts.filter((p) => (forwardStages(net, p.x, p.y).logit >= 0 ? 1 : 0) === p.label).length / densePts.length;

  const stageBounds: [ViewBoxBounds, ViewBoxBounds, ViewBoxBounds] = [
    { cx: 0, cy: 0, half: 1.55 },
    fitBounds(pointStages.map((s) => s.h1), 0.28),
    fitBounds(pointStages.map((s) => s.h2), 0.28),
  ];

  const boundarySegments = extractBoundarySegments(net, 1.45, 44);

  const result: SpaceFoldResult = {
    id,
    title: id === 'rings' ? 'Concentric rings' : 'Two moons',
    net,
    points,
    pointStages,
    acc0,
    acc1,
    acc2,
    denseAcc,
    stageBounds,
    boundarySegments,
    paramCount: 15,
  };
  cache.set(id, result);
  return result;
}
