// authored-by: Gemini Antigravity
/**
 * 2D Convolution & Kernels from scratch:
 *  - Exact 2D cross-correlation with configurable zero-padding and stride
 *  - Classic image-processing kernels (Identity, Box Blur, Sharpen, Sobel Gx/Gy, Laplacian, Emboss)
 *  - Translation-equivariance verifier and Dense-vs-Conv parameter calculator
 *  - Exhaustive 3x3 ternary kernel search (all 3^9 = 19,683 kernels) for Lab: Design the Kernel
 */

export type Grid = number[][];
export type Kernel3 = [
  [number, number, number],
  [number, number, number],
  [number, number, number],
];

export function outSize(h: number, k: number, pad: number, stride: number): number {
  return Math.floor((h + 2 * pad - k) / stride) + 1;
}

export function droppedBorderPixels(h: number, k: number, pad: number, stride: number): number {
  const rem = (h + 2 * pad - k) % stride;
  return rem < 0 ? 0 : rem;
}

/** Exact 2D cross-correlation (what PyTorch Conv2d computes) with zero padding and stride. */
export function conv2d(img: Grid, kernel: Grid, pad = 0, stride = 1): Grid {
  const h = img.length;
  const w = img[0].length;
  const kh = kernel.length;
  const kw = kernel[0].length;
  const oh = outSize(h, kh, pad, stride);
  const ow = outSize(w, kw, pad, stride);
  const out: Grid = [];
  for (let r = 0; r < oh; r++) {
    const row: number[] = [];
    for (let c = 0; c < ow; c++) {
      let sum = 0;
      for (let kr = 0; kr < kh; kr++) {
        for (let kc = 0; kc < kw; kc++) {
          const ir = r * stride + kr - pad;
          const ic = c * stride + kc - pad;
          const pix = ir >= 0 && ir < h && ic >= 0 && ic < w ? img[ir][ic] : 0;
          sum += pix * kernel[kr][kc];
        }
      }
      row.push(sum);
    }
    out.push(row);
  }
  return out;
}

/** Extract the exact 3x3 input patch (with 0-padding) aligned with output cell (or, oc). */
export function patchAt(img: Grid, or: number, oc: number, pad = 0, stride = 1, k = 3): Grid {
  const h = img.length;
  const w = img[0].length;
  const patch: Grid = [];
  for (let kr = 0; kr < k; kr++) {
    const row: number[] = [];
    for (let kc = 0; kc < k; kc++) {
      const ir = or * stride + kr - pad;
      const ic = oc * stride + kc - pad;
      row.push(ir >= 0 && ir < h && ic >= 0 && ic < w ? img[ir][ic] : 0);
    }
    patch.push(row);
  }
  return patch;
}

// ------------------------------------------------------------------ 1. Classic Kernels & Test Images

export type KernelPreset = {
  id: string;
  name: string;
  desc: string;
  kernel: Kernel3;
};

export const KERNEL_PRESETS: KernelPreset[] = [
  {
    id: 'sobel-x',
    name: 'Sobel Vertical Edge (Gx)',
    desc: 'Positive on dark→bright vertical edges, negative on bright→dark, zero on flat areas.',
    kernel: [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1],
    ],
  },
  {
    id: 'sobel-y',
    name: 'Sobel Horizontal Edge (Gy)',
    desc: 'Fires on horizontal edges (top→bottom brightness jumps) and ignores vertical bars.',
    kernel: [
      [-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1],
    ],
  },
  {
    id: 'laplacian',
    name: 'Laplacian (All Edges)',
    desc: 'Centre-surround filter: fires strongly on corners, thin lines, and isolated bright spots.',
    kernel: [
      [0, -1, 0],
      [-1, 4, -1],
      [0, -1, 0],
    ],
  },
  {
    id: 'blur',
    name: 'Box Blur (1/9)',
    desc: 'Averages every 3×3 neighbourhood equally, smoothing away sharp edges and noise.',
    kernel: [
      [1 / 9, 1 / 9, 1 / 9],
      [1 / 9, 1 / 9, 1 / 9],
      [1 / 9, 1 / 9, 1 / 9],
    ],
  },
  {
    id: 'sharpen',
    name: 'Sharpen',
    desc: 'Identity plus Laplacian: boosts local contrast while preserving average brightness (sum = 1).',
    kernel: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
  },
  {
    id: 'identity',
    name: 'Identity',
    desc: 'Copies the centre pixel verbatim.',
    kernel: [
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
    ],
  },
];

export type TestImage = {
  id: string;
  name: string;
  grid: Grid; // values in [0, 1]
};

export const TEST_IMAGES: TestImage[] = [
  {
    id: 'house',
    name: 'House & Roof',
    grid: [
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
      [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
      [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
      [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
  },
  {
    id: 'bars',
    name: 'Vertical + Horizontal Cross',
    grid: [
      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    ],
  },
  {
    id: 'step-diag',
    name: 'Diagonal Step',
    grid: Array.from({ length: 10 }, (_, r) => Array.from({ length: 10 }, (_, c) => (c >= r ? 1 : 0))),
  },
];

// ------------------------------------------------------------------ 2. Translation Equivariance & Parameter Sharing

export const CROSS_KERNEL: Kernel3 = [
  [-1, 1, -1],
  [1, 1, 1],
  [-1, 1, -1],
];

export function makeShiftedCross(cx: number, cy: number, size = 9): Grid {
  const g: Grid = Array.from({ length: size }, () => new Array<number>(size).fill(0));
  const offsets: [number, number][] = [
    [0, 0],
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
  ];
  for (const [dr, dc] of offsets) {
    const r = cy + dr;
    const c = cx + dc;
    if (r >= 0 && r < size && c >= 0 && c < size) g[r][c] = 1;
  }
  return g;
}

export function verifyEquivariance(cx: number, cy: number) {
  const img = makeShiftedCross(cx, cy, 9);
  const feat = conv2d(img, CROSS_KERNEL, 1, 1);
  let bestVal = -Infinity;
  let bestR = 0;
  let bestC = 0;
  for (let r = 0; r < feat.length; r++) {
    for (let c = 0; c < feat[0].length; c++) {
      if (feat[r][c] > bestVal) {
        bestVal = feat[r][c];
        bestR = r;
        bestC = c;
      }
    }
  }
  return { img, feat, peakVal: bestVal, peakR: bestR, peakC: bestC, exactMatch: bestR === cy && bestC === cx };
}

/** Exact parameter counts: Dense MLP layer vs Conv2d layer mapping (H,W,Cin) -> (H,W,Cout) */
export function compareDenseVsConvParams(h: number, w: number, cin: number, cout: number, k = 3) {
  const inElems = h * w * cin;
  const outElems = h * w * cout;
  const denseParams = inElems * outElems + outElems;
  const convParams = k * k * cin * cout + cout;
  return { denseParams, convParams, ratio: denseParams / convParams };
}

/** Border shrinkage after stacking L valid (pad=0) vs same (pad=(k-1)/2) 3x3 convolutions */
export function validStackShrinkage(startH: number, layers: number, k = 3) {
  const validH = Math.max(0, startH - layers * (k - 1));
  const sameH = startH;
  const pixelLoss = 1 - (validH * validH) / (startH * startH);
  return { startH, layers, validH, sameH, pixelLoss };
}

// ------------------------------------------------------------------ 3. Lab: Design the Kernel (Exhaustive over 3^9 = 19,683 ternary kernels)

export type ConvBriefId = 'vertical' | 'diagonal' | 'stack-shape';

export type ConvBrief = {
  id: ConvBriefId;
  title: string;
  story: string;
};

export const CONV_LAB_BRIEFS: ConvBrief[] = [
  {
    id: 'vertical',
    title: 'Brief 1 · Pure Vertical-Edge Detector (3×3 Ternary Kernel)',
    story: 'Design a 3×3 kernel using entries in {−1, 0, +1} that: (1) outputs exactly 0 on any flat region (weights sum to 0), (2) outputs 0 on a horizontal edge, and (3) outputs ≥ +2 on a dark→bright vertical edge. Optional goal: use as few non-zero weights as possible!',
  },
  {
    id: 'diagonal',
    title: 'Brief 2 · 45° Diagonal-Edge Detector (3×3 Ternary Kernel)',
    story: 'Design a 3×3 kernel using entries in {−1, 0, +1} that: (1) outputs 0 on any flat region, (2) stays muted (|resp| ≤ 1) on both vertical and horizontal step edges, and (3) fires ≥ +2 on a diagonal (↘) bright-upper-right edge. Optional goal: hit the maximum possible diagonal response (+3).',
  },
  {
    id: 'stack-shape',
    title: 'Brief 3 · Shape Engineer: Shrink 17×17 → Exactly 5×5 in Two Layers',
    story: 'Configure a 2-layer convolutional stack (choosing kernel size K ∈ {3, 5}, padding P ∈ {0, 1, 2}, and stride S ∈ {1, 2} shared by both layers) that shrinks a 17×17 feature map down to an exact 5×5 output with ZERO dropped border pixels on either layer — using as few kernel weights as possible.',
  },
];

// Canonical 3x3 test patches (row-major 9 values)
const PATCH_FLAT = [1, 1, 1, 1, 1, 1, 1, 1, 1];
const PATCH_VERT = [0, 0, 1, 0, 0, 1, 0, 0, 1];       // dark left -> bright right
const PATCH_VERT_MID = [0, 1, 1, 0, 1, 1, 0, 1, 1];   // shifted vertical step
const PATCH_HORIZ = [0, 0, 0, 0, 0, 0, 1, 1, 1];      // dark top -> bright bottom
const PATCH_HORIZ_MID = [0, 0, 0, 1, 1, 1, 1, 1, 1];  // shifted horizontal step
const PATCH_DIAG = [0, 1, 1, 0, 0, 1, 0, 0, 0];       // bright upper-right triangle

function dot9(w: readonly number[], p: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < 9; i++) s += w[i] * p[i];
  return s;
}

export function evalVerticalKernel(w: readonly number[]) {
  const flat = dot9(w, PATCH_FLAT);
  const horiz = Math.max(Math.abs(dot9(w, PATCH_HORIZ)), Math.abs(dot9(w, PATCH_HORIZ_MID)));
  const vert = Math.min(dot9(w, PATCH_VERT), dot9(w, PATCH_VERT_MID));
  const vertPeak = Math.max(dot9(w, PATCH_VERT), dot9(w, PATCH_VERT_MID));
  const nnz = w.filter((v) => v !== 0).length;
  const pass = flat === 0 && horiz === 0 && vertPeak >= 2 && vert >= 0;
  return { flat, horiz, vertPeak, nnz, pass };
}

export function evalDiagonalKernel(w: readonly number[]) {
  const flat = dot9(w, PATCH_FLAT);
  const vert = Math.max(Math.abs(dot9(w, PATCH_VERT)), Math.abs(dot9(w, PATCH_VERT_MID)));
  const horiz = Math.max(Math.abs(dot9(w, PATCH_HORIZ)), Math.abs(dot9(w, PATCH_HORIZ_MID)));
  const diag = dot9(w, PATCH_DIAG);
  const nnz = w.filter((v) => v !== 0).length;
  const pass = flat === 0 && vert <= 1 && horiz <= 1 && diag >= 2;
  return { flat, vert, horiz, diag, nnz, pass };
}

export function evalStackShape(k: 3 | 5, pad: 0 | 1 | 2, stride: 1 | 2) {
  const h1 = outSize(17, k, pad, stride);
  const drop1 = droppedBorderPixels(17, k, pad, stride);
  const h2 = h1 >= k - 2 * pad ? outSize(h1, k, pad, stride) : 0;
  const drop2 = h1 >= k - 2 * pad ? droppedBorderPixels(h1, k, pad, stride) : 1;
  const weightsPerFilter = 2 * k * k;
  const pass = h2 === 5 && drop1 === 0 && drop2 === 0;
  return { k, pad, stride, h1, h2, drop1, drop2, weightsPerFilter, pass };
}

type ExhaustSummary = { total: number; passing: number; summary: string };
const EXHAUST_CACHE = new Map<ConvBriefId, ExhaustSummary>();

export function convLabExhaustive(id: ConvBriefId): ExhaustSummary {
  const hit = EXHAUST_CACHE.get(id);
  if (hit) return hit;

  if (id === 'stack-shape') {
    let total = 0;
    let passing = 0;
    let bestW = Infinity;
    let bestDesc = '';
    for (const k of [3, 5] as const)
      for (const pad of [0, 1, 2] as const)
        for (const stride of [1, 2] as const) {
          total++;
          const r = evalStackShape(k, pad, stride);
          if (!r.pass) continue;
          passing++;
          if (r.weightsPerFilter < bestW) {
            bestW = r.weightsPerFilter;
            bestDesc = `K=${k}, P=${pad}, S=${stride} (17→${r.h1}→${r.h2}, ${r.weightsPerFilter} weights/filter)`;
          }
        }
    const res = { total, passing, summary: `Fewest weights: ${bestDesc}` };
    EXHAUST_CACHE.set(id, res);
    return res;
  }

  // Exact exhaustive scan of all 3^9 = 19,683 ternary 3x3 kernels!
  const w = new Array<number>(9).fill(-1);
  let total = 0;
  let passing = 0;
  let bestMetric = id === 'vertical' ? Infinity : -Infinity;
  let bestCount = 0;

  for (let mask = 0; mask < 19683; mask++) {
    let x = mask;
    for (let i = 0; i < 9; i++) {
      w[i] = (x % 3) - 1;
      x = Math.floor(x / 3);
    }
    total++;
    if (id === 'vertical') {
      const r = evalVerticalKernel(w);
      if (!r.pass) continue;
      passing++;
      if (r.nnz < bestMetric) {
        bestMetric = r.nnz;
        bestCount = 1;
      } else if (r.nnz === bestMetric) {
        bestCount++;
      }
    } else {
      const r = evalDiagonalKernel(w);
      if (!r.pass) continue;
      passing++;
      if (r.diag > bestMetric) {
        bestMetric = r.diag;
        bestCount = 1;
      } else if (r.diag === bestMetric) {
        bestCount++;
      }
    }
  }

  const summary =
    id === 'vertical'
      ? `Fewest non-zero weights: ${bestMetric} (${bestCount} optimal kernels out of all 19,683 ternary 3×3s)`
      : `Maximum diagonal response: +${bestMetric} (${bestCount} optimal kernels out of all 19,683 ternary 3×3s)`;

  const res = { total, passing, summary };
  EXHAUST_CACHE.set(id, res);
  return res;
}
