// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  MEMORY_PRESETS,
  SPRITES,
  compareLighting,
  downsampleGray,
  evalVisionBrief1,
  evalVisionBrief2,
  evalVisionBrief3,
  gridMean,
  makeGrayPattern,
  naiveGray,
  rec709Luminance,
  rgbToHsv,
  tensorBytes,
  visionLabExhaustive,
} from './vision';

describe('images-as-tensors: colour spaces and luminance', () => {
  it('pins Rec. 709 luminance vs naive average on RGB primaries', () => {
    expect(naiveGray([255, 0, 0])).toBeCloseTo(85.0, 2);
    expect(naiveGray([0, 255, 0])).toBeCloseTo(85.0, 2);
    expect(naiveGray([0, 0, 255])).toBeCloseTo(85.0, 2);

    const yR = rec709Luminance([255, 0, 0]);
    const yG = rec709Luminance([0, 255, 0]);
    const yB = rec709Luminance([0, 0, 255]);
    expect(yR).toBeCloseTo(54.213, 2);
    expect(yG).toBeCloseTo(182.376, 2);
    expect(yB).toBeCloseTo(18.411, 2);
    // Green is 9.9x brighter to human eyes than Blue!
    expect(yG / yB).toBeCloseTo(9.906, 2);
    expect(yR + yG + yB).toBeCloseTo(255.0, 5);
  });

  it('proves dimming shifts RGB heavily while keeping HSV Hue invariant', () => {
    const cmp = compareLighting([235, 38, 38], 0.5);
    expect(cmp.rgbDist).toBeCloseTo(120.05, 1);
    expect(cmp.hueShift).toBeCloseTo(0.0, 4);
    expect(rgbToHsv([0, 255, 0]).h).toBeCloseTo(120, 5);
    expect(rgbToHsv([0, 0, 255]).h).toBeCloseTo(240, 5);
  });

  it('pins exact tensor memory sizes across shapes and dtypes', () => {
    expect(SPRITES[0].h).toBe(12);
    expect(SPRITES[0].w).toBe(12);
    expect(tensorBytes(1, 224, 224, 3, 1)).toBe(150_528);
    expect(tensorBytes(64, 224, 224, 3, 4) / (1024 * 1024)).toBeCloseTo(36.75, 4);
    expect(tensorBytes(64, 224, 224, 3, 2) / (1024 * 1024)).toBeCloseTo(18.375, 4);
    expect(tensorBytes(1, 2160, 3840, 3, 1)).toBe(24_883_200);
    expect(MEMORY_PRESETS.length).toBe(5);
  });
});

describe('images-as-tensors: aliasing and downsampling', () => {
  it('shows 2x strided slicing aliases a 1-px checkerboard to solid white (255) while box avg preserves 127.5', () => {
    const chk = makeGrayPattern('checker').grid;
    expect(gridMean(chk)).toBeCloseTo(127.5, 4);
    const strided = downsampleGray(chk, 2, 'stride');
    const boxed = downsampleGray(chk, 2, 'box');
    expect(gridMean(strided)).toBeCloseTo(255.0, 4);
    expect(gridMean(boxed)).toBeCloseTo(127.5, 4);
  });
});

describe('images-as-tensors: Lab (Vision Preprocessor) exhaustive checks', () => {
  it('Brief 1 (shadow-proof signals): raw RGB drops to 66.7% in shadow; chromaticity & Hue hit 100%', () => {
    const raw = evalVisionBrief1('raw-rgb');
    expect(raw.dayAcc).toBeCloseTo(1.0, 4);
    expect(raw.shadowAcc).toBeCloseTo(0.6667, 3);
    expect(raw.acc).toBeCloseTo(0.8333, 3);
    expect(raw.pass).toBe(false);

    expect(evalVisionBrief1('gray-avg').acc).toBeCloseTo(0.5667, 3);
    expect(evalVisionBrief1('gray-709').acc).toBeCloseTo(0.6222, 3);
    expect(evalVisionBrief1('rgb-l1').acc).toBeCloseTo(1.0, 4);
    expect(evalVisionBrief1('hsv-hue').acc).toBeCloseTo(1.0, 4);

    const ex = visionLabExhaustive('lighting');
    expect(ex.total).toBe(5);
    expect(ex.passing).toBe(2);
  });

  it('Brief 2 (anti-aliased thumbnailer): only 4x4 box average passes', () => {
    const s0 = evalVisionBrief2('stride-0');
    expect(s0.meanErr).toBeCloseTo(96.3, 1);
    expect(s0.rmse).toBeCloseTo(100.9, 1);
    expect(s0.pass).toBe(false);

    const s1 = evalVisionBrief2('stride-1');
    expect(s1.meanErr).toBeCloseTo(115.0, 1);
    expect(s1.pass).toBe(false);

    const box = evalVisionBrief2('box-avg');
    expect(box.meanErr).toBeCloseTo(0.0, 5);
    expect(box.rmse).toBeCloseTo(0.0, 5);
    expect(box.pass).toBe(true);

    const ex = visionLabExhaustive('aliasing');
    expect(ex.total).toBe(4);
    expect(ex.passing).toBe(1);
  });

  it('Brief 3 (PyTorch batch under 24 MiB): unique solution NCHW + FP16 + z-score (18.375 MiB)', () => {
    const fp32 = evalVisionBrief3('NCHW', 'fp32', 'zscore');
    expect(fp32.mib).toBeCloseTo(36.75, 4);
    expect(fp32.pass).toBe(false);

    const u8 = evalVisionBrief3('NCHW', 'uint8', 'zscore');
    expect(u8.mib).toBeCloseTo(9.1875, 4);
    expect(u8.normOk).toBe(false);
    expect(u8.pass).toBe(false);

    const opt = evalVisionBrief3('NCHW', 'fp16', 'zscore');
    expect(opt.mib).toBeCloseTo(18.375, 4);
    expect(opt.pass).toBe(true);

    const ex = visionLabExhaustive('vram');
    expect(ex.total).toBe(18);
    expect(ex.passing).toBe(1);
  });
});
