// authored-by: Gemini Antigravity
/**
 * Multi-layer scalar credit-assignment engine (`2 → 2 → 2`) and 9-parameter
 * geometry microscope engine (`2 → 2 → 1`) for Module 5 Backpropagation.
 *
 * Every analytic backward pass is cross-checked in `tinyMlp.test.ts` against
 * both `src/lib/autodiff.ts` (`Value.backward()`) and central finite
 * differences (< 1e-8).
 */
import { Value } from '../../lib/autodiff';

export type TinyAct = 'tanh' | 'relu';

export type Mat2x2 = [[number, number], [number, number]];
export type Vec2 = [number, number];

/**
 * Parameters for the 2 → 2 → 2 scalar credit-assignment network.
 *
 * Matrix layout matches `network-from-scratch` row-vector convention `(in × out)`:
 *   z1 = x @ W1 + b1   where W1[i][j] connects input x_i to hidden h_j
 *   y  = h @ W2 + b2   where W2[j][k] connects hidden h_j to output y_k
 *
 * Therefore, during the backward pass:
 *   dh = dy @ W2.T     so h_j gathers along row j of W2 (= column j of W2.T),
 *                      i.e. the two outgoing edges from h_j to y_1 and y_2!
 */
export interface Tiny222Params {
  W1: Mat2x2;
  b1: Vec2;
  W2: Mat2x2;
  b2: Vec2;
}

export interface Tiny222Trace {
  x: Vec2;
  t: Vec2;
  act: TinyAct;
  z1: Vec2;
  h: Vec2;
  actDeriv: Vec2;
  y: Vec2;
  loss: number;
  dy: Vec2;
  /** Explicit per-path blame arriving at each hidden unit h_j from y_1 and y_2. */
  dhPaths: [[number, number], [number, number]];
  dh: Vec2;
  dz1: Vec2;
  dW2: Mat2x2;
  db2: Vec2;
  dW1: Mat2x2;
  db1: Vec2;
  dx: Vec2;
}

export interface Tiny222Preset {
  id: 'balanced' | 'cancelling' | 'twins' | 'saturated';
  label: string;
  tagline: string;
  x: Vec2;
  t: Vec2;
  defaultAct: TinyAct;
  params: Tiny222Params;
}

export const TINY222_PRESETS: Record<Tiny222Preset['id'], Tiny222Preset> = {
  balanced: {
    id: 'balanced',
    label: 'Balanced',
    tagline: 'Both outputs send clear, reinforcing credit back into the hidden layer.',
    x: [1.0, -0.6],
    t: [0.9, -0.7],
    defaultAct: 'tanh',
    params: {
      W1: [
        [0.7, -0.5],
        [-0.4, 0.8],
      ],
      b1: [0.1, -0.2],
      W2: [
        [0.9, -0.6],
        [-0.5, 0.7],
      ],
      b2: [0.05, -0.05],
    },
  },
  cancelling: {
    id: 'cancelling',
    label: 'Tug-of-war',
    tagline: 'Output y₁ pulls h₁ up while y₂ pulls h₁ down — the two paths nearly cancel.',
    x: [1.0, 0.5],
    t: [1.45, -0.15],
    defaultAct: 'tanh',
    params: {
      W1: [
        [0.6, -0.3],
        [0.2, 0.7],
      ],
      b1: [0.0, 0.1],
      W2: [
        [1.0, 1.0],
        [-0.4, 0.9],
      ],
      b2: [0.1, -0.1],
    },
  },
  twins: {
    id: 'twins',
    label: 'Identical twins',
    tagline: 'Both hidden neurons start with identical weights. Step as often as you like — they stay fused.',
    x: [1.2, -0.7],
    t: [0.8, -0.5],
    defaultAct: 'tanh',
    params: {
      W1: [
        [0.5, 0.5],
        [-0.4, -0.4],
      ],
      b1: [0.15, 0.15],
      W2: [
        [0.8, -0.6],
        [0.8, -0.6],
      ],
      b2: [0.0, 0.1],
    },
  },
  saturated: {
    id: 'saturated',
    label: 'Saturated tanh',
    tagline: 'Hidden neuron h₁ sits deep on tanh’s flat shelf (z₁ ≈ 4), starving its incoming weights by >500×.',
    x: [1.5, 1.0],
    t: [0.9, -0.8],
    defaultAct: 'tanh',
    params: {
      W1: [
        [2.0, 0.2],
        [1.0, -0.1],
      ],
      b1: [0.0, 0.1],
      W2: [
        [0.8, -0.7],
        [0.8, -0.7],
      ],
      b2: [0.0, 0.0],
    },
  },
};

export function cloneTiny222Params(p: Tiny222Params): Tiny222Params {
  return {
    W1: [
      [p.W1[0][0], p.W1[0][1]],
      [p.W1[1][0], p.W1[1][1]],
    ],
    b1: [p.b1[0], p.b1[1]],
    W2: [
      [p.W2[0][0], p.W2[0][1]],
      [p.W2[1][0], p.W2[1][1]],
    ],
    b2: [p.b2[0], p.b2[1]],
  };
}

export function traceTiny222(params: Tiny222Params, x: Vec2, t: Vec2, act: TinyAct): Tiny222Trace {
  const { W1, b1, W2, b2 } = params;

  // Forward pass
  const z1: Vec2 = [
    x[0] * W1[0][0] + x[1] * W1[1][0] + b1[0],
    x[0] * W1[0][1] + x[1] * W1[1][1] + b1[1],
  ];
  const h: Vec2 = [
    act === 'tanh' ? Math.tanh(z1[0]) : Math.max(0, z1[0]),
    act === 'tanh' ? Math.tanh(z1[1]) : Math.max(0, z1[1]),
  ];
  const actDeriv: Vec2 = [
    act === 'tanh' ? 1 - h[0] * h[0] : z1[0] > 0 ? 1 : 0,
    act === 'tanh' ? 1 - h[1] * h[1] : z1[1] > 0 ? 1 : 0,
  ];

  const y: Vec2 = [
    h[0] * W2[0][0] + h[1] * W2[1][0] + b2[0],
    h[0] * W2[0][1] + h[1] * W2[1][1] + b2[1],
  ];
  const d0 = y[0] - t[0];
  const d1 = y[1] - t[1];
  const loss = 0.5 * (d0 * d0 + d1 * d1);

  // Backward pass
  const dy: Vec2 = [d0, d1];
  const dW2: Mat2x2 = [
    [h[0] * dy[0], h[0] * dy[1]],
    [h[1] * dy[0], h[1] * dy[1]],
  ];
  const db2: Vec2 = [dy[0], dy[1]];

  // Fan-in into each hidden neuron: dh = dy @ W2.T
  const dhPaths: [[number, number], [number, number]] = [
    [dy[0] * W2[0][0], dy[1] * W2[0][1]],
    [dy[0] * W2[1][0], dy[1] * W2[1][1]],
  ];
  const dh: Vec2 = [dhPaths[0][0] + dhPaths[0][1], dhPaths[1][0] + dhPaths[1][1]];
  const dz1: Vec2 = [dh[0] * actDeriv[0], dh[1] * actDeriv[1]];

  const dW1: Mat2x2 = [
    [x[0] * dz1[0], x[0] * dz1[1]],
    [x[1] * dz1[0], x[1] * dz1[1]],
  ];
  const db1: Vec2 = [dz1[0], dz1[1]];
  const dx: Vec2 = [
    dz1[0] * W1[0][0] + dz1[1] * W1[0][1],
    dz1[0] * W1[1][0] + dz1[1] * W1[1][1],
  ];

  return { x, t, act, z1, h, actDeriv, y, loss, dy, dhPaths, dh, dz1, dW2, db2, dW1, db1, dx };
}

/** Independent cross-check via scalar Value autodiff (`src/lib/autodiff.ts`). */
export function tiny222ViaAutodiff(params: Tiny222Params, x: Vec2, t: Vec2, act: TinyAct) {
  const vx = [new Value(x[0], 'x1'), new Value(x[1], 'x2')];
  const vW1 = [
    [new Value(params.W1[0][0], 'w11'), new Value(params.W1[0][1], 'w12')],
    [new Value(params.W1[1][0], 'w21'), new Value(params.W1[1][1], 'w22')],
  ];
  const vb1 = [new Value(params.b1[0], 'b1_0'), new Value(params.b1[1], 'b1_1')];
  const vW2 = [
    [new Value(params.W2[0][0], 'v11'), new Value(params.W2[0][1], 'v12')],
    [new Value(params.W2[1][0], 'v21'), new Value(params.W2[1][1], 'v22')],
  ];
  const vb2 = [new Value(params.b2[0], 'b2_0'), new Value(params.b2[1], 'b2_1')];

  const vz1 = [0, 1].map((j) => vx[0].mul(vW1[0][j]).add(vx[1].mul(vW1[1][j])).add(vb1[j]));
  const vh = vz1.map((z) => (act === 'tanh' ? z.tanh() : z.relu()));
  const vy = [0, 1].map((k) => vh[0].mul(vW2[0][k]).add(vh[1].mul(vW2[1][k])).add(vb2[k]));
  const L = vy[0].sub(t[0]).pow(2).add(vy[1].sub(t[1]).pow(2)).mul(0.5);
  L.backward();

  return {
    loss: L.data,
    dW1: [
      [vW1[0][0].grad, vW1[0][1].grad],
      [vW1[1][0].grad, vW1[1][1].grad],
    ] as Mat2x2,
    db1: [vb1[0].grad, vb1[1].grad] as Vec2,
    dW2: [
      [vW2[0][0].grad, vW2[0][1].grad],
      [vW2[1][0].grad, vW2[1][1].grad],
    ] as Mat2x2,
    db2: [vb2[0].grad, vb2[1].grad] as Vec2,
    dh: [vh[0].grad, vh[1].grad] as Vec2,
    dz1: [vz1[0].grad, vz1[1].grad] as Vec2,
    dy: [vy[0].grad, vy[1].grad] as Vec2,
    dx: [vx[0].grad, vx[1].grad] as Vec2,
  };
}

export function stepTiny222(params: Tiny222Params, x: Vec2, t: Vec2, act: TinyAct, lr: number): Tiny222Params {
  const tr = traceTiny222(params, x, t, act);
  return {
    W1: [
      [params.W1[0][0] - lr * tr.dW1[0][0], params.W1[0][1] - lr * tr.dW1[0][1]],
      [params.W1[1][0] - lr * tr.dW1[1][0], params.W1[1][1] - lr * tr.dW1[1][1]],
    ],
    b1: [params.b1[0] - lr * tr.db1[0], params.b1[1] - lr * tr.db1[1]],
    W2: [
      [params.W2[0][0] - lr * tr.dW2[0][0], params.W2[0][1] - lr * tr.dW2[0][1]],
      [params.W2[1][0] - lr * tr.dW2[1][0], params.W2[1][1] - lr * tr.dW2[1][1]],
    ],
    b2: [params.b2[0] - lr * tr.db2[0], params.b2[1] - lr * tr.db2[1]],
  };
}

/** Max absolute difference between hidden neuron 1 and hidden neuron 2 weights. */
export function twinGap(params: Tiny222Params): number {
  return Math.max(
    Math.abs(params.W1[0][0] - params.W1[0][1]),
    Math.abs(params.W1[1][0] - params.W1[1][1]),
    Math.abs(params.b1[0] - params.b1[1]),
    Math.abs(params.W2[0][0] - params.W2[1][0]),
    Math.abs(params.W2[0][1] - params.W2[1][1]),
  );
}

// ---------------------------------------------------------------------------
// 2 → 2 → 1 Geometry Microscope Network (9 parameters, tanh hidden + sigmoid)
// ---------------------------------------------------------------------------

export interface MicroscopePoint {
  id: string;
  x1: number;
  x2: number;
  y: 0 | 1;
}

export const MICROSCOPE_XOR: readonly MicroscopePoint[] = [
  { id: 'P0', x1: -1, x2: -1, y: 0 },
  { id: 'P1', x1: -1, x2: 1, y: 1 },
  { id: 'P2', x1: 1, x2: -1, y: 1 },
  { id: 'P3', x1: 1, x2: 1, y: 0 },
];

export const MICROSCOPE_BOUNDS: [number, number, number, number] = [-1.6, 1.6, -1.6, 1.6];

export const MICROSCOPE_KEYS = ['w11', 'w12', 'b1', 'w21', 'w22', 'b2', 'v1', 'v2', 'c'] as const;
export type MicroscopeKey = (typeof MICROSCOPE_KEYS)[number];
export type MicroscopeParams = Record<MicroscopeKey, number>;

/**
 * Deterministic starting weights for StepMicroscope:
 * - Both hidden zero-contour lines visibly cross [-1.6, 1.6]^2
 * - Starts with 50% accuracy (P0 and P1 misclassified or similar)
 * - Converges smoothly to 100% accuracy under full-batch SGD (lr = 0.9)
 */
export const MICROSCOPE_INIT: MicroscopeParams = {
  w11: 0.9,
  w12: 0.35,
  b1: 0.35,
  w21: 0.4,
  w22: 1.0,
  b2: -0.45,
  v1: 1.1,
  v2: -1.0,
  c: -0.1,
};

export interface MicroscopePointEval {
  id: string;
  x1: number;
  x2: number;
  y: 0 | 1;
  z1: number;
  z2: number;
  h1: number;
  h2: number;
  zOut: number;
  p: number;
  loss: number;
  correct: boolean;
}

export interface MicroscopeStepResult {
  points: MicroscopePointEval[];
  meanLoss: number;
  selectedLoss: number;
  accuracy: number;
  grads: MicroscopeParams;
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

export function microscopePredict(p: MicroscopeParams, x1: number, x2: number): number {
  const h1 = Math.tanh(p.w11 * x1 + p.w12 * x2 + p.b1);
  const h2 = Math.tanh(p.w21 * x1 + p.w22 * x2 + p.b2);
  return sigmoid(p.v1 * h1 + p.v2 * h2 + p.c);
}

export function microscopeEvalPoint(p: MicroscopeParams, pt: MicroscopePoint): MicroscopePointEval {
  const z1 = p.w11 * pt.x1 + p.w12 * pt.x2 + p.b1;
  const z2 = p.w21 * pt.x1 + p.w22 * pt.x2 + p.b2;
  const h1 = Math.tanh(z1);
  const h2 = Math.tanh(z2);
  const zOut = p.v1 * h1 + p.v2 * h2 + p.c;
  const prob = sigmoid(zOut);
  const pc = Math.min(1 - 1e-12, Math.max(1e-12, prob));
  const loss = -(pt.y * Math.log(pc) + (1 - pt.y) * Math.log(1 - pc));
  const correct = (prob >= 0.5 ? 1 : 0) === pt.y;
  return { id: pt.id, x1: pt.x1, x2: pt.x2, y: pt.y, z1, z2, h1, h2, zOut, p: prob, loss, correct };
}

/**
 * Evaluate the 2 → 2 → 1 network on all 4 XOR points and compute exact gradients
 * for either a single point (`selectedIdx = 0..3`) or the full batch (`selectedIdx = null`).
 */
export function microscopeStepInfo(
  p: MicroscopeParams,
  selectedIdx: number | null,
  data: readonly MicroscopePoint[] = MICROSCOPE_XOR,
): MicroscopeStepResult {
  const points = data.map((pt) => microscopeEvalPoint(p, pt));
  const meanLoss = points.reduce((s, e) => s + e.loss, 0) / points.length;
  const accuracy = points.filter((e) => e.correct).length / points.length;

  const active = selectedIdx === null ? points : [points[selectedIdx]];
  const n = active.length;
  const selectedLoss = active.reduce((s, e) => s + e.loss, 0) / n;

  const grads: MicroscopeParams = {
    w11: 0,
    w12: 0,
    b1: 0,
    w21: 0,
    w22: 0,
    b2: 0,
    v1: 0,
    v2: 0,
    c: 0,
  };

  for (const e of active) {
    const dzOut = (e.p - e.y) / n;
    grads.v1 += dzOut * e.h1;
    grads.v2 += dzOut * e.h2;
    grads.c += dzOut;

    const dh1 = dzOut * p.v1;
    const dh2 = dzOut * p.v2;
    const dz1 = dh1 * (1 - e.h1 * e.h1);
    const dz2 = dh2 * (1 - e.h2 * e.h2);

    grads.w11 += dz1 * e.x1;
    grads.w12 += dz1 * e.x2;
    grads.b1 += dz1;
    grads.w21 += dz2 * e.x1;
    grads.w22 += dz2 * e.x2;
    grads.b2 += dz2;
  }

  return { points, meanLoss, selectedLoss, accuracy, grads };
}

export function applyMicroscopeStep(
  p: MicroscopeParams,
  grads: MicroscopeParams,
  lr: number,
): MicroscopeParams {
  const out = { ...p };
  for (const k of MICROSCOPE_KEYS) out[k] = p[k] - lr * grads[k];
  return out;
}

export function runMicroscopeSteps(
  init: MicroscopeParams,
  steps: number,
  lr: number,
  selectedIdx: number | null = null,
): MicroscopeParams[] {
  const history: MicroscopeParams[] = [{ ...init }];
  let cur = { ...init };
  for (let i = 0; i < steps; i++) {
    const info = microscopeStepInfo(cur, selectedIdx);
    cur = applyMicroscopeStep(cur, info.grads, lr);
    history.push(cur);
  }
  return history;
}

/**
 * Compare one coordinated 9-weight backprop step against nudging only the single
 * steepest weight with the same learning rate `lr`.
 */
export function compareCoordinatedVsSingle(
  p: MicroscopeParams,
  selectedIdx: number | null,
  lr: number,
) {
  const info = microscopeStepInfo(p, selectedIdx);
  const fullNext = applyMicroscopeStep(p, info.grads, lr);
  const fullInfo = microscopeStepInfo(fullNext, selectedIdx);

  let steepestKey: MicroscopeKey = 'w11';
  let steepestAbs = -1;
  for (const k of MICROSCOPE_KEYS) {
    if (Math.abs(info.grads[k]) > steepestAbs) {
      steepestAbs = Math.abs(info.grads[k]);
      steepestKey = k;
    }
  }
  const singleNext: MicroscopeParams = { ...p, [steepestKey]: p[steepestKey] - lr * info.grads[steepestKey] };
  const singleInfo = microscopeStepInfo(singleNext, selectedIdx);

  return {
    beforeLoss: info.selectedLoss,
    fullAfterLoss: fullInfo.selectedLoss,
    fullDrop: info.selectedLoss - fullInfo.selectedLoss,
    steepestKey,
    singleAfterLoss: singleInfo.selectedLoss,
    singleDrop: info.selectedLoss - singleInfo.selectedLoss,
  };
}

export interface LineSegment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  mx: number;
  my: number;
  /** Unit normal pointing toward the positive half-plane (w1*x + w2*y + b > 0). */
  nx: number;
  ny: number;
}

/**
 * Clip the exact zero-contour line `w1 * x + w2 * y + b = 0` against a 2D
 * bounding box `[xmin, xmax, ymin, ymax]`. Returns null if the line lies
 * outside the box or has zero norm.
 */
export function hiddenLineSegment(
  w1: number,
  w2: number,
  b: number,
  bounds: readonly [number, number, number, number] = MICROSCOPE_BOUNDS,
): LineSegment | null {
  const norm = Math.hypot(w1, w2);
  if (norm < 1e-10) return null;
  const [xmin, xmax, ymin, ymax] = bounds;
  const eps = 1e-9;
  const pts: Array<[number, number]> = [];

  const pushUnique = (x: number, y: number) => {
    if (x < xmin - eps || x > xmax + eps || y < ymin - eps || y > ymax + eps) return;
    const cx = Math.min(xmax, Math.max(xmin, x));
    const cy = Math.min(ymax, Math.max(ymin, y));
    if (!pts.some(([px, py]) => Math.hypot(px - cx, py - cy) < 1e-6)) {
      pts.push([cx, cy]);
    }
  };

  if (Math.abs(w2) > 1e-10) {
    pushUnique(xmin, -(w1 * xmin + b) / w2);
    pushUnique(xmax, -(w1 * xmax + b) / w2);
  }
  if (Math.abs(w1) > 1e-10) {
    pushUnique(-(w2 * ymin + b) / w1, ymin);
    pushUnique(-(w2 * ymax + b) / w1, ymax);
  }
  if (pts.length < 2) return null;

  // Pick the two farthest intersections in case the line passes through a corner.
  let bestA = pts[0];
  let bestB = pts[1];
  let bestD = -1;
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      const d = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
      if (d > bestD) {
        bestD = d;
        bestA = pts[i];
        bestB = pts[j];
      }
    }
  }
  if (bestD < 1e-6) return null;

  return {
    x1: bestA[0],
    y1: bestA[1],
    x2: bestB[0],
    y2: bestB[1],
    mx: (bestA[0] + bestB[0]) / 2,
    my: (bestA[1] + bestB[1]) / 2,
    nx: w1 / norm,
    ny: w2 / norm,
  };
}
