// authored-by: Gemini Antigravity
/**
 * Engine for Lab: The Credit-Assignment Workbench (`src/lessons/backpropagation/`).
 *
 * Three briefs:
 *  - Brief 1: Trace the exact update signs (and largest |grad|) on a frozen
 *    2 -> 2 -> 1 ReLU network without a calculator.
 *  - Brief 2: Unstick a broken 2 -> 4 -> 1 XOR network across 5 fixed seeds
 *    (48-setting exhaustive grid: initMode x activation x lr).
 *  - Brief 3: Freeze/unfreeze individual hidden neurons and the output layer
 *    on a 2 -> 2 -> 1 tanh XOR network (32-setting exhaustive grid) and watch
 *    XOR untangle in the exact 2D hidden plane (h1, h2).
 */

export type CreditAct = 'tanh' | 'sigmoid' | 'relu';
export type InitMode = 'broken' | 'zeros' | 'asym-large' | 'asym-small';
export type SignChoice = '+' | '-' | '0';

export const CREDIT_XOR: readonly [number, number, 0 | 1][] = [
  [-1, -1, 0],
  [-1, 1, 1],
  [1, -1, 1],
  [1, 1, 0],
];

export const BRIEF2_SEEDS = [1, 2, 3, 4, 5] as const;
export const BRIEF2_STEPS = 200;
export const BRIEF2_TARGET_LOSS = 0.08;
export const BRIEF2_INIT_MODES: readonly InitMode[] = ['broken', 'zeros', 'asym-large', 'asym-small'];
export const BRIEF2_ACTS: readonly CreditAct[] = ['tanh', 'sigmoid', 'relu'];
export const BRIEF2_LRS = [0.2, 0.5, 1.0, 2.0] as const;

export const BRIEF3_STEPS = 200;
export const BRIEF3_TARGET_LOSS = 0.08;
export const BRIEF3_LRS = [0.4, 0.8, 1.5, 2.5] as const;

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng: () => number): number {
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

function actFn(z: number, act: CreditAct): number {
  if (act === 'tanh') return Math.tanh(z);
  if (act === 'sigmoid') return sigmoid(z);
  return z > 0 ? z : 0;
}

function actDeriv(z: number, h: number, act: CreditAct): number {
  if (act === 'tanh') return 1 - h * h;
  if (act === 'sigmoid') return h * (1 - h);
  return z > 0 ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Brief 1: Sign-tracing on a concrete 2 -> 2 -> 1 ReLU network
// ---------------------------------------------------------------------------

export const BRIEF1_PROBES = ['v1', 'w11', 'w21', 'w12'] as const;
export type Brief1WeightKey = (typeof BRIEF1_PROBES)[number];

export interface Brief1Network {
  x1: number;
  x2: number;
  y: number;
  w11: number;
  w21: number;
  b1: number;
  w12: number;
  w22: number;
  b2: number;
  v1: number;
  v2: number;
  c: number;
}

export const BRIEF1_NET: Brief1Network = {
  x1: 2.0,
  x2: -1.0,
  y: 1.0,
  w11: 0.5,
  w21: -0.5,
  b1: 0.0,
  w12: -0.6,
  w22: 0.4,
  b2: -0.2,
  v1: -0.8,
  v2: 0.7,
  c: 0.2,
};

export interface Brief1Trace {
  z1: number;
  z2: number;
  h1: number;
  h2: number;
  yHat: number;
  error: number;
  loss: number;
  dh1: number;
  dh2: number;
  dz1: number;
  dz2: number;
  grads: Record<Brief1WeightKey, number>;
  /** Update direction for gradient descent w <- w - lr * grad. */
  updateSigns: Record<Brief1WeightKey, SignChoice>;
  largestKey: Brief1WeightKey;
}

export function evaluateBrief1(net: Brief1Network = BRIEF1_NET): Brief1Trace {
  const z1 = net.w11 * net.x1 + net.w21 * net.x2 + net.b1;
  const z2 = net.w12 * net.x1 + net.w22 * net.x2 + net.b2;
  const h1 = z1 > 0 ? z1 : 0;
  const h2 = z2 > 0 ? z2 : 0;
  const yHat = net.v1 * h1 + net.v2 * h2 + net.c;
  const error = yHat - net.y;
  const loss = 0.5 * error * error;

  const dh1 = error * net.v1;
  const dh2 = error * net.v2;
  const dz1 = z1 > 0 ? dh1 : 0;
  const dz2 = z2 > 0 ? dh2 : 0;

  const grads: Record<Brief1WeightKey, number> = {
    v1: error * h1,
    w11: dz1 * net.x1,
    w21: dz1 * net.x2,
    w12: dz2 * net.x1,
  };

  const toSign = (g: number): SignChoice => {
    if (Math.abs(g) < 1e-12) return '0';
    // Update is w - lr * grad, so negative grad => increase (+), positive grad => decrease (-).
    return g < 0 ? '+' : '-';
  };

  const updateSigns: Record<Brief1WeightKey, SignChoice> = {
    v1: toSign(grads.v1),
    w11: toSign(grads.w11),
    w21: toSign(grads.w21),
    w12: toSign(grads.w12),
  };

  let largestKey: Brief1WeightKey = 'v1';
  let largestAbs = -1;
  for (const k of BRIEF1_PROBES) {
    if (Math.abs(grads[k]) > largestAbs) {
      largestAbs = Math.abs(grads[k]);
      largestKey = k;
    }
  }

  return { z1, z2, h1, h2, yHat, error, loss, dh1, dh2, dz1, dz2, grads, updateSigns, largestKey };
}

export interface Brief1Check {
  correctCount: number;
  allSignsCorrect: boolean;
  largestCorrect: boolean;
  perWeight: Record<Brief1WeightKey, boolean>;
}

export function checkBrief1(
  guesses: Record<Brief1WeightKey, SignChoice | null>,
  largestGuess: Brief1WeightKey | null,
): Brief1Check {
  const truth = evaluateBrief1();
  const perWeight: Record<Brief1WeightKey, boolean> = {
    v1: guesses.v1 === truth.updateSigns.v1,
    w11: guesses.w11 === truth.updateSigns.w11,
    w21: guesses.w21 === truth.updateSigns.w21,
    w12: guesses.w12 === truth.updateSigns.w12,
  };
  const correctCount = BRIEF1_PROBES.filter((k) => perWeight[k]).length;
  return {
    correctCount,
    allSignsCorrect: correctCount === 4,
    largestCorrect: largestGuess === truth.largestKey,
    perWeight,
  };
}

// ---------------------------------------------------------------------------
// Shared MLP trainer for Briefs 2 and 3
// ---------------------------------------------------------------------------

export interface LabMlp {
  H: number;
  W1: number[][]; // [H][2]
  b1: number[]; // [H]
  W2: number[]; // [H]
  b2: number;
}

export interface LabMlpGrads {
  dW1: number[][];
  db1: number[];
  dW2: number[];
  db2: number;
  loss: number;
  acc: number;
  probs: number[];
  hiddenCoords: Array<[number, number]>;
}

export function cloneLabMlp(m: LabMlp): LabMlp {
  return {
    H: m.H,
    W1: m.W1.map((r) => [r[0], r[1]]),
    b1: [...m.b1],
    W2: [...m.W2],
    b2: m.b2,
  };
}

export function evalAndGradMlp(m: LabMlp, act: CreditAct): LabMlpGrads {
  let loss = 0;
  let right = 0;
  const dW1 = Array.from({ length: m.H }, () => [0, 0]);
  const db1 = Array(m.H).fill(0);
  const dW2 = Array(m.H).fill(0);
  let db2 = 0;
  const probs: number[] = [];
  const hiddenCoords: Array<[number, number]> = [];

  for (const [x1, x2, y] of CREDIT_XOR) {
    const z1 = new Array<number>(m.H);
    const h = new Array<number>(m.H);
    let z2 = m.b2;
    for (let j = 0; j < m.H; j++) {
      z1[j] = m.W1[j][0] * x1 + m.W1[j][1] * x2 + m.b1[j];
      h[j] = actFn(z1[j], act);
      z2 += m.W2[j] * h[j];
    }
    const p = sigmoid(z2);
    probs.push(p);
    if (m.H >= 2) hiddenCoords.push([h[0], h[1]]);
    const pc = Math.min(1 - 1e-12, Math.max(1e-12, p));
    loss += -(y * Math.log(pc) + (1 - y) * Math.log(1 - pc)) / 4;
    if ((p >= 0.5 ? 1 : 0) === y) right++;

    const dz2 = (p - y) / 4;
    db2 += dz2;
    for (let j = 0; j < m.H; j++) {
      dW2[j] += dz2 * h[j];
      const dh = dz2 * m.W2[j];
      const dz = dh * actDeriv(z1[j], h[j], act);
      dW1[j][0] += dz * x1;
      dW1[j][1] += dz * x2;
      db1[j] += dz;
    }
  }

  return { dW1, db1, dW2, db2, loss, acc: right / 4, probs, hiddenCoords };
}

export function mlpPredict(m: LabMlp, act: CreditAct, x1: number, x2: number): number {
  let z2 = m.b2;
  for (let j = 0; j < m.H; j++) {
    z2 += m.W2[j] * actFn(m.W1[j][0] * x1 + m.W1[j][1] * x2 + m.b1[j], act);
  }
  return sigmoid(z2);
}

// ---------------------------------------------------------------------------
// Brief 2: Unstick the broken hidden layer (2 -> 4 -> 1 across 5 fixed seeds)
// ---------------------------------------------------------------------------

export interface Brief2Config {
  initMode: InitMode;
  act: CreditAct;
  lr: number;
}

export function makeBrief2Init(mode: InitMode, seed: number): LabMlp {
  const H = 4;
  if (mode === 'zeros') {
    return {
      H,
      W1: Array.from({ length: H }, () => [0, 0]),
      b1: Array(H).fill(0),
      W2: Array(H).fill(0),
      b2: 0,
    };
  }
  if (mode === 'broken') {
    // Identical twin hidden neurons at saturated scale
    return {
      H,
      W1: Array.from({ length: H }, () => [3.5, 3.5]),
      b1: Array(H).fill(0.2),
      W2: Array(H).fill(1.0),
      b2: 0.0,
    };
  }
  const rng = mulberry32(seed);
  const scale = mode === 'asym-large' ? 6.0 : 0.8;
  const W1 = Array.from({ length: H }, () => [randn(rng) * scale, randn(rng) * scale]);
  const b1 = Array.from({ length: H }, () => randn(rng) * (mode === 'asym-large' ? scale * 0.4 : 0.4));
  const W2 = Array.from({ length: H }, () => randn(rng) * (mode === 'asym-large' ? 2.5 : 0.8));
  const b2 = randn(rng) * 0.2;
  return { H, W1, b1, W2, b2 };
}

export interface Brief2SeedResult {
  seed: number;
  finalLoss: number;
  finalAcc: number;
  solvedStep: number | null;
  lossCurve: number[];
  /** Gradient norm of each of the 4 hidden neurons at step 0 (telemetry!). */
  initHiddenGradNorms: number[];
  /** Max weight difference between hidden neuron 0 and hidden neuron 1 at the end. */
  finalTwinGap: number;
  finalModel: LabMlp;
}

export interface Brief2Result {
  cfg: Brief2Config;
  seeds: Brief2SeedResult[];
  solvedSeeds: number;
  meanLoss: number;
  meanAcc: number;
  meanSolveStep: number | null;
  pass: boolean;
}

export function runBrief2Seed(cfg: Brief2Config, seed: number): Brief2SeedResult {
  const m = makeBrief2Init(cfg.initMode, seed);
  const lossCurve: number[] = [];
  let solvedStep: number | null = null;
  let initHiddenGradNorms: number[] = [];
  let finalLoss = 0;
  let finalAcc = 0;

  for (let s = 0; s <= BRIEF2_STEPS; s++) {
    const g = evalAndGradMlp(m, cfg.act);
    if (s === 0) {
      initHiddenGradNorms = g.dW1.map((row, j) => Math.hypot(row[0], row[1], g.db1[j]));
    }
    if (s % 4 === 0 || s === BRIEF2_STEPS) lossCurve.push(g.loss);
    finalLoss = g.loss;
    finalAcc = g.acc;
    if (solvedStep === null && g.acc === 1 && g.loss <= BRIEF2_TARGET_LOSS) solvedStep = s;
    if (s === BRIEF2_STEPS) break;

    for (let j = 0; j < m.H; j++) {
      m.W2[j] -= cfg.lr * g.dW2[j];
      m.W1[j][0] -= cfg.lr * g.dW1[j][0];
      m.W1[j][1] -= cfg.lr * g.dW1[j][1];
      m.b1[j] -= cfg.lr * g.db1[j];
    }
    m.b2 -= cfg.lr * g.db2;
  }

  const finalTwinGap = Math.max(
    Math.abs(m.W1[0][0] - m.W1[1][0]),
    Math.abs(m.W1[0][1] - m.W1[1][1]),
    Math.abs(m.b1[0] - m.b1[1]),
    Math.abs(m.W2[0] - m.W2[1]),
  );

  return {
    seed,
    finalLoss,
    finalAcc,
    solvedStep,
    lossCurve,
    initHiddenGradNorms,
    finalTwinGap,
    finalModel: m,
  };
}

export function evaluateBrief2(cfg: Brief2Config): Brief2Result {
  const seeds = BRIEF2_SEEDS.map((sd) => runBrief2Seed(cfg, sd));
  const solvedSeeds = seeds.filter((r) => r.solvedStep !== null).length;
  const meanLoss = seeds.reduce((s, r) => s + r.finalLoss, 0) / seeds.length;
  const meanAcc = seeds.reduce((s, r) => s + r.finalAcc, 0) / seeds.length;
  const pass = solvedSeeds === BRIEF2_SEEDS.length;
  const meanSolveStep = pass ? seeds.reduce((s, r) => s + r.solvedStep!, 0) / seeds.length : null;
  return { cfg, seeds, solvedSeeds, meanLoss, meanAcc, meanSolveStep, pass };
}

// ---------------------------------------------------------------------------
// Brief 3: Freeze/unfreeze hidden layers (2 -> 2 -> 1 tanh, exact 2D (h1,h2))
// ---------------------------------------------------------------------------

export const BRIEF3_INIT: LabMlp = {
  H: 2,
  W1: [
    [0.85, 0.55],
    [0.45, 0.9],
  ],
  b1: [0.15, -0.12],
  W2: [1.1, -1.1],
  b2: 0.05,
};

export interface Brief3Config {
  trainH1: boolean;
  trainH2: boolean;
  trainOut: boolean;
  lr: number;
}

/**
 * Exact 2D linear-separability test for the 4 XOR points in (h1, h2) space.
 * Two class-0 points (A, B) and two class-1 points (C, D) in R^2 are NOT
 * linearly separable whenever segments AB and CD intersect (Radon's theorem).
 */
export function isXorHiddenSeparable(coords: readonly [number, number][]): boolean {
  const a = coords[0]; // (-1,-1) -> class 0
  const b = coords[3]; // (+1,+1) -> class 0
  const c = coords[1]; // (-1,+1) -> class 1
  const d = coords[2]; // (+1,-1) -> class 1
  const cross = (o: readonly [number, number], p: readonly [number, number], q: readonly [number, number]) =>
    (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0]);
  const d1 = cross(c, d, a);
  const d2 = cross(c, d, b);
  const d3 = cross(a, b, c);
  const d4 = cross(a, b, d);
  const properCross =
    ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
  return !properCross;
}

export interface Brief3Snapshot {
  step: number;
  loss: number;
  acc: number;
  hiddenCoords: Array<[number, number]>;
  separable: boolean;
  model: LabMlp;
}

export interface Brief3Result {
  cfg: Brief3Config;
  unfrozenHiddenCount: number;
  finalLoss: number;
  finalAcc: number;
  solvedStep: number | null;
  initSeparable: boolean;
  finalSeparable: boolean;
  pass: boolean;
  lossCurve: number[];
  snapshots: Brief3Snapshot[];
  finalModel: LabMlp;
}

export function evaluateBrief3(cfg: Brief3Config, steps = BRIEF3_STEPS): Brief3Result {
  const m = cloneLabMlp(BRIEF3_INIT);
  const lossCurve: number[] = [];
  const snapshots: Brief3Snapshot[] = [];
  let solvedStep: number | null = null;
  let initSeparable = false;
  let finalSeparable = false;
  let finalLoss = 0;
  let finalAcc = 0;

  for (let s = 0; s <= steps; s++) {
    const g = evalAndGradMlp(m, 'tanh');
    const sep = isXorHiddenSeparable(g.hiddenCoords);
    if (s === 0) initSeparable = sep;
    finalSeparable = sep;
    finalLoss = g.loss;
    finalAcc = g.acc;
    if (s % 4 === 0 || s === steps) lossCurve.push(g.loss);
    if (s % 10 === 0 || s === steps) {
      snapshots.push({
        step: s,
        loss: g.loss,
        acc: g.acc,
        hiddenCoords: g.hiddenCoords.map((pt) => [pt[0], pt[1]]),
        separable: sep,
        model: cloneLabMlp(m),
      });
    }
    if (solvedStep === null && g.acc === 1 && g.loss <= BRIEF3_TARGET_LOSS) solvedStep = s;
    if (s === steps) break;

    if (cfg.trainOut) {
      m.W2[0] -= cfg.lr * g.dW2[0];
      m.W2[1] -= cfg.lr * g.dW2[1];
      m.b2 -= cfg.lr * g.db2;
    }
    if (cfg.trainH1) {
      m.W1[0][0] -= cfg.lr * g.dW1[0][0];
      m.W1[0][1] -= cfg.lr * g.dW1[0][1];
      m.b1[0] -= cfg.lr * g.db1[0];
    }
    if (cfg.trainH2) {
      m.W1[1][0] -= cfg.lr * g.dW1[1][0];
      m.W1[1][1] -= cfg.lr * g.dW1[1][1];
      m.b1[1] -= cfg.lr * g.db1[1];
    }
  }

  return {
    cfg,
    unfrozenHiddenCount: (cfg.trainH1 ? 1 : 0) + (cfg.trainH2 ? 1 : 0),
    finalLoss,
    finalAcc,
    solvedStep,
    initSeparable,
    finalSeparable,
    pass: solvedStep !== null,
    lossCurve,
    snapshots,
    finalModel: m,
  };
}

// ---------------------------------------------------------------------------
// Evidence-only hints (never name a button, toggle, mode, activation, or LR!)
// ---------------------------------------------------------------------------

export function brief1Hint(check: Brief1Check | null): string {
  if (!check) {
    return 'Read the forward numbers from right to left: first compare the prediction with the target, then notice the sign on each outgoing edge from h1 and the pre-activation sign inside h2.';
  }
  if (!check.perWeight.w12) {
    return 'Look closely at h2’s pre-activation value in the forward table: when a ReLU tally sits below zero, check what its local slope passes through to incoming weights.';
  }
  if (!check.perWeight.w11 || !check.perWeight.w21) {
    return 'Trace the sign across every link into h1: an edge with a negative weight reverses the direction that upstream blame asks h1 to move, and a negative input reverses it once more.';
  }
  if (!check.perWeight.v1) {
    return 'Compare the prediction with the target: the output is below the target, and h1 is positive, so check which direction on that outgoing weight raises the output.';
  }
  if (!check.largestCorrect) {
    return 'Multiply each path’s magnitudes step by step: upstream error × downstream weight × input magnitude can make an earlier weight’s slope bigger than the output weight’s.';
  }
  return 'Every sign and magnitude matches the exact chain-rule products shown in the reveal table.';
}

export function brief2Hint(res: Brief2Result): string {
  const s0 = res.seeds[0];
  const maxGrad = Math.max(...s0.initHiddenGradNorms);
  if (maxGrad < 1e-12) {
    return 'Inspect the hidden-layer gradient telemetry at step 0: every hidden unit receives exactly 0.0000 gradient, so the first layer never moves off its starting point.';
  }
  if (s0.finalTwinGap < 1e-9) {
    return 'Compare hidden unit 1 and hidden unit 2 in the probe: their weights and step-0 gradients are identical to 4 decimal places, so all four units behave like a single straight line capped at 75% accuracy.';
  }
  if (!res.pass && res.solvedSeeds > 0) {
    return `Only ${res.solvedSeeds} of 5 seeds reached the target within ${BRIEF2_STEPS} steps: inspect the failing seed’s step-0 hidden gradient norms to see which units started flat or stalled too slowly.`;
  }
  if (!res.pass) {
    return `Mean loss stopped at ${res.meanLoss.toFixed(3)} with ${(res.meanAcc * 100).toFixed(0)}% accuracy after ${BRIEF2_STEPS} steps — compare the initial hidden gradient norms and the slope of the loss curve.`;
  }
  return `All 5 seeds solved the task in an average of ${res.meanSolveStep?.toFixed(1)} steps; check whether a different combination of scale, non-linearity, and step size reaches the target sooner.`;
}

export function brief3Hint(res: Brief3Result): string {
  if (!res.finalSeparable) {
    return 'Look at the hidden-space plot (h1, h2): the two class-0 dots and two class-1 dots still cross each other like an X, so no straight boundary in that plane can separate all four.';
  }
  if (!res.pass) {
    return `The hidden coordinates untangled, yet loss only reached ${res.finalLoss.toFixed(3)} (${(res.finalAcc * 100).toFixed(0)}% accuracy) in ${BRIEF3_STEPS} steps — check whether the separating line itself is moving fast enough or moving at all.`;
  }
  if (res.unfrozenHiddenCount > 1) {
    return 'Both hidden units moved to untangle the four dots; watch how far each individual dot actually needed to slide in (h1, h2) to break the crossing.';
  }
  return 'Only one hidden unit moved, yet the four points uncrossed in (h1, h2) so the straight output boundary could slice cleanly between the classes.';
}
