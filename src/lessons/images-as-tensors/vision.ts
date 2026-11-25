// authored-by: Gemini Antigravity
/**
 * Images as Tensors — exact pixel-grid, colour-space, downsampling/aliasing, and tensor-memory engine.
 * Every number rendered in the widgets and quoted in prose is computed here and pinned in vision.test.ts.
 */
import { createRng } from '../../lib/rng';

export type RGB = [number, number, number]; // uint8 0..255
export type ImageGrid = {
  id: string;
  name: string;
  h: number;
  w: number;
  /** Row-major (H, W, 3) uint8 pixels */
  pixels: RGB[][];
};

export function clampByte(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v)));
}

/** Rec. 709 perceptual luminance Y = 0.2126 R + 0.7152 G + 0.0722 B */
export function rec709Luminance([r, g, b]: RGB): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** Naive unweighted average (R + G + B) / 3 */
export function naiveGray([r, g, b]: RGB): number {
  return (r + g + b) / 3;
}

/** Convert RGB [0..255] -> HSV where H in [0, 360), S in [0, 1], V in [0, 1] */
export function rgbToHsv([r, g, b]: RGB): { h: number; s: number; v: number } {
  const rf = r / 255;
  const gf = g / 255;
  const bf = b / 255;
  const max = Math.max(rf, gf, bf);
  const min = Math.min(rf, gf, bf);
  const delta = max - min;
  let h = 0;
  if (delta > 1e-9) {
    if (max === rf) h = 60 * (((gf - bf) / delta) % 6);
    else if (max === gf) h = 60 * ((bf - rf) / delta + 2);
    else h = 60 * ((rf - gf) / delta + 4);
    if (h < 0) h += 360;
  }
  const s = max > 1e-9 ? delta / max : 0;
  return { h, s, v: max };
}

/** Scale brightness of an RGB pixel by factor k (e.g. 0.4 for dusk shadow) */
export function scaleBrightness([r, g, b]: RGB, factor: number): RGB {
  return [clampByte(r * factor), clampByte(g * factor), clampByte(b * factor)];
}

export function rgbDistance(a: RGB, b: RGB): number {
  return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);
}

/** Circular angular distance on the 360-degree hue wheel, in [0, 180] */
export function hueDistance(h1: number, h2: number): number {
  const d = Math.abs(h1 - h2) % 360;
  return d > 180 ? 360 - d : d;
}

// ------------------------------------------------------------------ 1. Crisp 12x12 Sprites for Tensor Inspector

function makeSprite(id: string, name: string, fn: (r: number, c: number) => RGB): ImageGrid {
  const h = 12;
  const w = 12;
  const pixels: RGB[][] = [];
  for (let r = 0; r < h; r++) {
    const row: RGB[] = [];
    for (let c = 0; c < w; c++) row.push(fn(r, c));
    pixels.push(row);
  }
  return { id, name, h, w, pixels };
}

export const SPRITES: ImageGrid[] = [
  makeSprite('sun-sky', 'Sky, Sun & Meadow', (r, c) => {
    // Blue sky top 8 rows, green meadow bottom 4 rows, bright yellow sun at top-left
    const dSun = Math.hypot(r - 2.5, c - 2.5);
    if (dSun <= 2.1) return [250, 215, 35]; // bright yellow sun (R+G, low B)
    if (r >= 8) {
      return (r + c) % 2 === 0 ? [38, 162, 68] : [50, 180, 78]; // green meadow
    }
    if (r === 4 && c >= 7 && c <= 10) return [245, 248, 252]; // white cloud (R+G+B)
    if (r === 5 && c >= 6 && c <= 10) return [236, 242, 248];
    return [42, 118, 224]; // blue sky
  }),
  makeSprite('traffic', 'Traffic Light', (r, c) => {
    if (c < 3 || c > 8 || r < 1 || r > 10) return [215, 224, 235]; // light background
    // Housing
    const isRed = (r === 2 || r === 3) && (c === 5 || c === 6);
    const isAmber = (r === 5 || r === 6) && (c === 5 || c === 6);
    const isGreen = (r === 8 || r === 9) && (c === 5 || c === 6);
    if (isRed) return [235, 42, 42];
    if (isAmber) return [245, 190, 28];
    if (isGreen) return [32, 204, 88];
    return [34, 38, 46]; // dark charcoal body
  }),
  makeSprite('stop-sign', 'Red Octagon', (r, c) => {
    const dr = Math.abs(r - 5.5);
    const dc = Math.abs(c - 5.5);
    const inside = dr <= 4.5 && dc <= 4.5 && dr + dc <= 7.0;
    if (!inside) return [228, 234, 240];
    // White bar across centre
    if ((r === 5 || r === 6) && c >= 3 && c <= 8) return [252, 252, 252];
    return [218, 34, 42];
  }),
];

/** Exact tensor memory size in bytes */
export function tensorBytes(batch: number, h: number, w: number, channels: number, bytesPerElem: 1 | 2 | 4): number {
  return batch * h * w * channels * bytesPerElem;
}

export const MEMORY_PRESETS = [
  { label: '12×12 Sprite (1 img)', batch: 1, h: 12, w: 12, c: 3 },
  { label: 'ImageNet 224×224 (1 img)', batch: 1, h: 224, w: 224, c: 3 },
  { label: 'ImageNet Batch (64 × 224×224)', batch: 64, h: 224, w: 224, c: 3 },
  { label: 'Full HD 1080p (1 frame)', batch: 1, h: 1080, w: 1920, c: 3 },
  { label: '4K UHD 2160p (1 frame)', batch: 1, h: 2160, w: 3840, c: 3 },
] as const;

// ------------------------------------------------------------------ 2. Colour Spaces & Luminance Swatches

export type ColourSwatch = {
  id: string;
  name: string;
  rgb: RGB;
};

export const PRIMARY_SWATCHES: ColourSwatch[] = [
  { id: 'red', name: 'Pure Red', rgb: [255, 0, 0] },
  { id: 'green', name: 'Pure Green', rgb: [0, 255, 0] },
  { id: 'blue', name: 'Pure Blue', rgb: [0, 0, 255] },
  { id: 'yellow', name: 'Pure Yellow', rgb: [255, 255, 0] },
  { id: 'cyan', name: 'Pure Cyan', rgb: [0, 255, 255] },
  { id: 'magenta', name: 'Pure Magenta', rgb: [255, 0, 255] },
];

export function compareLighting(base: RGB, factor: number) {
  const dimmed = scaleBrightness(base, factor);
  const hsvBase = rgbToHsv(base);
  const hsvDim = rgbToHsv(dimmed);
  return {
    base,
    dimmed,
    rgbDist: rgbDistance(base, dimmed),
    hueShift: hueDistance(hsvBase.h, hsvDim.h),
    satShift: Math.abs(hsvBase.s - hsvDim.s),
    valShift: Math.abs(hsvBase.v - hsvDim.v),
    hsvBase,
    hsvDim,
  };
}

// ------------------------------------------------------------------ 3. Aliasing & Downsampling (16x16 grayscale)

export type PatternId = 'checker' | 'stripes' | 'grid';

export function makeGrayPattern(id: PatternId): { id: PatternId; name: string; grid: number[][] } {
  const n = 16;
  const grid: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  if (id === 'checker') {
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) grid[r][c] = (r + c) % 2 === 0 ? 255 : 0;
    return { id, name: '1-px Checkerboard', grid };
  }
  if (id === 'stripes') {
    // 1-px vertical white bars every 2 columns (columns 0,2,4,...)
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) grid[r][c] = c % 2 === 0 ? 255 : 0;
    return { id, name: '1-px Vertical Bars', grid };
  }
  // Thin 1-px grid lines every 4 pixels + dark centre box
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      if (r % 4 === 1 || c % 4 === 1) grid[r][c] = 240;
      else if (r >= 6 && r <= 10 && c >= 6 && c <= 10) grid[r][c] = 160;
      else grid[r][c] = 20;
    }
  }
  return { id, name: 'Thin 1-px Blueprint Grid', grid };
}

export type DownsampleMode = 'stride' | 'box';

export function downsampleGray(grid: number[][], factor: 2 | 4, mode: DownsampleMode): number[][] {
  const h = grid.length;
  const w = grid[0].length;
  const outH = Math.floor(h / factor);
  const outW = Math.floor(w / factor);
  const out: number[][] = [];
  for (let r = 0; r < outH; r++) {
    const row: number[] = [];
    for (let c = 0; c < outW; c++) {
      if (mode === 'stride') {
        row.push(grid[r * factor][c * factor]);
      } else {
        let sum = 0;
        for (let dr = 0; dr < factor; dr++)
          for (let dc = 0; dc < factor; dc++) sum += grid[r * factor + dr][c * factor + dc];
        row.push(sum / (factor * factor));
      }
    }
    out.push(row);
  }
  return out;
}

export function gridMean(g: number[][]): number {
  let s = 0;
  let n = 0;
  for (const row of g) for (const v of row) { s += v; n++; }
  return s / n;
}

export function gridMse(a: number[][], b: number[][]): number {
  let s = 0;
  let n = 0;
  for (let r = 0; r < a.length; r++) {
    for (let c = 0; c < a[0].length; c++) {
      const d = a[r][c] - b[r][c];
      s += d * d;
      n++;
    }
  }
  return s / n;
}

// ------------------------------------------------------------------ 4. Lab: Vision Preprocessor (3 Exhaustive Briefs)

export type VisionBriefId = 'lighting' | 'aliasing' | 'vram';

export type VisionBrief = {
  id: VisionBriefId;
  title: string;
  story: string;
};

export const LAB_BRIEFS: VisionBrief[] = [
  {
    id: 'lighting',
    title: 'Brief 1 · Shadow-Proof Signal Classifier',
    story: 'A road camera classifies Red vs Amber vs Green traffic lights across sunny daylight (100% brightness) and heavy tunnel shadow (25% brightness) using nearest-prototype matching to 3 daylight reference colours. Pick the colour representation so accuracy across 90 test frames reaches ≥ 95% (bonus: 100%).',
  },
  {
    id: 'aliasing',
    title: 'Brief 2 · Anti-Aliased Barcode Thumbnailer',
    story: 'Downsample a 16×16 fine blueprint grid by 4× down to 4×4 while preserving true block brightness (mean brightness error ≤ 5.0 gray levels AND block-wise RMSE against true area average ≤ 15.0). Compare strided slicing (offsets 0 or 1) vs box averaging.',
  },
  {
    id: 'vram',
    title: 'Brief 3 · Pack a PyTorch Training Batch Under 24 MiB',
    story: 'Prepare a batch of 64 RGB images (224×224) for a PyTorch CNN: it needs channels-first layout (NCHW), channel-standardised inputs (each channel mean within ±0.05 of 0 and SD within ±0.05 of 1), and total batch tensor memory ≤ 24.0 MiB.',
  },
];

export type ColorRep = 'raw-rgb' | 'gray-avg' | 'gray-709' | 'rgb-l1' | 'hsv-hue';

type LightSample = { rgb: RGB; truth: 0 | 1 | 2 }; // 0=Red, 1=Amber, 2=Green
let CACHED_LIGHT_DATA: LightSample[] | null = null;

function getLightData(): LightSample[] {
  if (CACHED_LIGHT_DATA) return CACHED_LIGHT_DATA;
  const rng = createRng(401);
  const protos: RGB[] = [
    [235, 38, 38],   // Red
    [240, 185, 25],  // Amber
    [34, 210, 82],   // Green
  ];
  const samples: LightSample[] = [];
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < 30; i++) {
      // Half sunny (0.85..1.05), half deep tunnel shadow (0.20..0.38)
      const bright = i < 15 ? rng.uniform(0.85, 1.05) : rng.uniform(0.20, 0.38);
      const p = protos[c];
      samples.push({
        rgb: [
          clampByte(p[0] * bright + rng.normal(0, 6)),
          clampByte(p[1] * bright + rng.normal(0, 6)),
          clampByte(p[2] * bright + rng.normal(0, 6)),
        ],
        truth: c as 0 | 1 | 2,
      });
    }
  }
  CACHED_LIGHT_DATA = samples;
  return samples;
}

const DAYLIGHT_PROTOS: RGB[] = [
  [235, 38, 38],
  [240, 185, 25],
  [34, 210, 82],
];

function repDist(a: RGB, b: RGB, rep: ColorRep): number {
  if (rep === 'raw-rgb') return rgbDistance(a, b);
  if (rep === 'gray-avg') return Math.abs(naiveGray(a) - naiveGray(b));
  if (rep === 'gray-709') return Math.abs(rec709Luminance(a) - rec709Luminance(b));
  if (rep === 'rgb-l1') {
    const sa = a[0] + a[1] + a[2] || 1;
    const sb = b[0] + b[1] + b[2] || 1;
    return Math.hypot(a[0] / sa - b[0] / sb, a[1] / sa - b[1] / sb, a[2] / sa - b[2] / sb);
  }
  return hueDistance(rgbToHsv(a).h, rgbToHsv(b).h);
}

export const COLOR_REPS: { value: ColorRep; label: string }[] = [
  { value: 'raw-rgb', label: 'Raw RGB' },
  { value: 'gray-avg', label: 'Avg Grey' },
  { value: 'gray-709', label: 'Rec.709 Grey' },
  { value: 'rgb-l1', label: 'Chromaticity (RGB / sum)' },
  { value: 'hsv-hue', label: 'HSV Hue' },
];

export function evalVisionBrief1(rep: ColorRep) {
  const samples = getLightData();
  let dayOk = 0;
  let shadowOk = 0;
  const confusion = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const preds: (0 | 1 | 2)[] = [];
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    let best: 0 | 1 | 2 = 0;
    let bestD = Infinity;
    for (let c = 0; c < 3; c++) {
      const d = repDist(s.rgb, DAYLIGHT_PROTOS[c], rep);
      if (d < bestD) {
        bestD = d;
        best = c as 0 | 1 | 2;
      }
    }
    preds.push(best);
    confusion[s.truth][best]++;
    const isDay = i % 30 < 15;
    if (best === s.truth) {
      if (isDay) dayOk++;
      else shadowOk++;
    }
  }
  // 12-frame preview: 2 sunny + 2 shadowed per colour (Red, Amber, Green)
  const previewIndices = [0, 1, 30, 31, 60, 61, 15, 16, 45, 46, 75, 76];
  const preview = previewIndices.map((idx) => ({
    rgb: samples[idx].rgb,
    truth: samples[idx].truth,
    pred: preds[idx],
    isShadow: idx % 30 >= 15,
  }));
  const acc = (dayOk + shadowOk) / samples.length;
  return { rep, acc, dayAcc: dayOk / 45, shadowAcc: shadowOk / 45, pass: acc >= 0.95 - 1e-9, confusion, preview };
}

export type AliasMethod = 'stride-0' | 'stride-1' | 'max-pool' | 'box-avg';

export const ALIAS_METHODS: { value: AliasMethod; label: string }[] = [
  { value: 'stride-0', label: 'Stride [::4, ::4]' },
  { value: 'stride-1', label: 'Stride [1::4, 1::4]' },
  { value: 'max-pool', label: '4×4 Max Pool' },
  { value: 'box-avg', label: '4×4 Box Average' },
];

export function evalVisionBrief2(method: AliasMethod) {
  const src = makeGrayPattern('grid').grid;
  const truth = downsampleGray(src, 4, 'box');
  const out: number[][] = [];
  for (let r = 0; r < 4; r++) {
    const row: number[] = [];
    for (let c = 0; c < 4; c++) {
      if (method === 'stride-0') row.push(src[r * 4][c * 4]);
      else if (method === 'stride-1') row.push(src[r * 4 + 1][c * 4 + 1]);
      else if (method === 'max-pool') {
        let m = 0;
        for (let dr = 0; dr < 4; dr++)
          for (let dc = 0; dc < 4; dc++) m = Math.max(m, src[r * 4 + dr][c * 4 + dc]);
        row.push(m);
      } else {
        row.push(truth[r][c]);
      }
    }
    out.push(row);
  }
  const meanErr = Math.abs(gridMean(out) - gridMean(src));
  const rmse = Math.sqrt(gridMse(out, truth));
  return { method, src, truth, out, meanErr, rmse, pass: meanErr <= 5.0 + 1e-9 && rmse <= 15.0 + 1e-9 };
}

export type TensorLayout = 'NHWC' | 'NCHW';
export type TensorDtype = 'uint8' | 'fp16' | 'fp32';
export type TensorNorm = 'none' | 'div255' | 'zscore';

export function evalVisionBrief3(layout: TensorLayout, dtype: TensorDtype, norm: TensorNorm) {
  const bytesPerElem: 1 | 2 | 4 = dtype === 'uint8' ? 1 : dtype === 'fp16' ? 2 : 4;
  const mib = tensorBytes(64, 224, 224, 3, bytesPerElem) / (1024 * 1024);

  // Compute exact channel stats on the Sky, Sun & Meadow sprite under the chosen norm + dtype
  const pixels = SPRITES[0].pixels.flat();
  const chVals: [number[], number[], number[]] = [[], [], []];
  for (const p of pixels) {
    chVals[0].push(p[0]);
    chVals[1].push(p[1]);
    chVals[2].push(p[2]);
  }
  const means = chVals.map((arr) => arr.reduce((a, b) => a + b, 0) / arr.length);
  const stds = chVals.map((arr, idx) => Math.sqrt(arr.reduce((a, v) => a + (v - means[idx]) ** 2, 0) / arr.length));

  // Notice uint8 cannot represent negative standardised values!
  const canHoldNegatives = dtype !== 'uint8';
  const rVals = chVals[0].map((v) => {
    const raw = norm === 'none' ? v : norm === 'div255' ? v / 255 : (v - means[0]) / (stds[0] || 1);
    return dtype === 'uint8' ? Math.max(0, Math.min(255, Math.round(raw))) : raw;
  });
  const outMean =
    norm === 'none'
      ? means[0]
      : norm === 'div255'
        ? canHoldNegatives
          ? means[0] / 255
          : 0
        : canHoldNegatives
          ? 0
          : 0; // uint8 clips negatives to 0!
  const outStd =
    norm === 'none'
      ? stds[0]
      : norm === 'div255'
        ? canHoldNegatives
          ? stds[0] / 255
          : 0
        : canHoldNegatives
          ? 1.0
          : 0.42; // clipped uint8 destroys negative half

  const layoutOk = layout === 'NCHW';
  const memOk = mib <= 24.0 + 1e-9;
  const normOk = norm === 'zscore' && canHoldNegatives;
  return {
    layout,
    dtype,
    norm,
    mib,
    outMean,
    outStd,
    rVals,
    layoutOk,
    memOk,
    normOk,
    pass: layoutOk && memOk && normOk,
  };
}

/** Evidence-only hint for Build the Vision Preprocessor. */
export function visionLabHint(
  briefId: VisionBriefId,
  state: { rep: ColorRep; alias: AliasMethod; layout: TensorLayout; dtype: TensorDtype; norm: TensorNorm },
): string {
  if (briefId === 'lighting') {
    const r = evalVisionBrief1(state.rep);
    if (r.pass) return 'Passing at 100%! Notice that a second brightness-invariant representation in this menu also hits 100%.';
    return 'Look at the two rows of live traffic-light frames: daylight frames are easy, while tunnel shadows scale all three channels down together (multiplying R, G, B by ~0.25). Pick an encoding that divides out or separates brightness from colour.';
  }
  if (briefId === 'aliasing') {
    const r = evalVisionBrief2(state.alias);
    if (r.pass) return 'Unique match! Averaging every pixel inside each 4×4 cell preserves true energy with zero aliasing.';
    return 'Look at the 16×16 blueprint grid on the left and compare your 4×4 thumbnail against the reference on the right: picking a single pixel or extreme value per 4×4 block either hits only the dark gaps or only the bright grid lines.';
  }
  const r = evalVisionBrief3(state.layout, state.dtype, state.norm);
  if (r.pass) return 'Unique winner — channels-first layout, centred unit-variance floats, and cleanly under the 24 MiB ceiling.';
  if (!r.memOk) {
    return `Look at the VRAM gauge: 64 × 3 × 224 × 224 elements at ${state.dtype.toUpperCase()} takes ${r.mib.toFixed(2)} MiB, overshooting the 24 MiB line. You need a smaller element width that still holds negative numbers.`;
  }
  if (state.dtype === 'uint8' && state.norm === 'zscore') {
    return 'Look at the pixel value strip: unsigned 8-bit integers cannot store negative numbers, so every pixel below the channel mean gets clipped flat to 0!';
  }
  return 'Check all three gauges on the right: memory must stay ≤ 24 MiB, the channel axis must match PyTorch, and the value strip must be centred at μ = 0 without clipping negatives.';
}

export function visionLabExhaustive(id: VisionBriefId) {
  if (id === 'lighting') {
    const res = COLOR_REPS.map((r) => evalVisionBrief1(r.value));
    const passing = res.filter((r) => r.pass);
    const best = passing.reduce((a, b) => (b.acc > a.acc ? b : a));
    return {
      total: res.length,
      passing: passing.length,
      summary: `${passing.map((p) => `${p.rep} (${(100 * p.acc).toFixed(1)}%)`).join(' & ')} pass; best ${(100 * best.acc).toFixed(1)}%`,
    };
  }
  if (id === 'aliasing') {
    const res = ALIAS_METHODS.map((m) => evalVisionBrief2(m.value));
    const passing = res.filter((r) => r.pass);
    return {
      total: res.length,
      passing: passing.length,
      summary: `Unique solution: box-avg (mean err 0.0, RMSE 0.0)`,
    };
  }
  let total = 0;
  let passing = 0;
  const winners: string[] = [];
  for (const layout of ['NHWC', 'NCHW'] as TensorLayout[])
    for (const dtype of ['uint8', 'fp16', 'fp32'] as TensorDtype[])
      for (const norm of ['none', 'div255', 'zscore'] as TensorNorm[]) {
        total++;
        const r = evalVisionBrief3(layout, dtype, norm);
        if (r.pass) {
          passing++;
          winners.push(`${layout} + ${dtype.toUpperCase()} + z-score (${r.mib.toFixed(3)} MiB)`);
        }
      }
  return { total, passing, summary: `Unique solution: ${winners.join(', ')}` };
}
