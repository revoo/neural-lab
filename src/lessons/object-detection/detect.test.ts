// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import {
  STREET_SCENE,
  computeCocoMap,
  detLabExhaustive,
  evalDetLab,
  evaluateDetections,
  iou,
  runHardNms,
  runSoftNms,
  squareShiftIou,
} from './detect';

describe('object-detection: IoU geometry and small-object penalty', () => {
  it('pins 10-pixel horizontal shift IoU on 100x100 vs 20x20 boxes', () => {
    expect(squareShiftIou(100, 10)).toBeCloseTo(9000 / 11000, 5); // 0.8182
    expect(squareShiftIou(20, 10)).toBeCloseTo(200 / 600, 5); // 0.3333
    expect(iou({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 20, y1: 20, x2: 30, y2: 30 })).toBe(0);
    expect(iou({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 0, y1: 0, x2: 10, y2: 10 })).toBe(1);
  });
});

describe('object-detection: NMS, Soft-NMS and AP / mAP', () => {
  it('pins tight, sweet-spot, and loose Hard NMS on the street scene', () => {
    expect(iou(STREET_SCENE.dets[0], STREET_SCENE.dets[2])).toBeCloseTo(0.3389, 3);

    const tight = evaluateDetections(runHardNms(STREET_SCENE.dets, 0.25, 0.4).kept, STREET_SCENE.gts, 0.5);
    expect(tight.tp).toBe(2);
    expect(tight.fn).toBe(1);
    expect(tight.f1).toBeCloseTo(0.8, 3);
    expect(tight.ap).toBeCloseTo(2 / 3, 3);

    const sweet = runHardNms(STREET_SCENE.dets, 0.5, 0.4).kept;
    const sweetEv = evaluateDetections(sweet, STREET_SCENE.gts, 0.5);
    expect(sweetEv.tp).toBe(3);
    expect(sweetEv.fp).toBe(0);
    expect(sweetEv.f1).toBeCloseTo(1.0, 4);
    expect(sweetEv.ap).toBeCloseTo(1.0, 4);
    expect(computeCocoMap(sweet, STREET_SCENE.gts)).toBeCloseTo(0.9, 3);

    const loose = evaluateDetections(runHardNms(STREET_SCENE.dets, 0.85, 0.4).kept, STREET_SCENE.gts, 0.5);
    expect(loose.tp).toBe(3);
    expect(loose.fp).toBe(4);
    expect(loose.f1).toBeCloseTo(0.6, 3);
  });

  it('proves Soft-NMS decays duplicate scores smoothly without deleting overlapping people', () => {
    const soft = runSoftNms(STREET_SCENE.dets, 0.45, 0.35).kept;
    const ev = evaluateDetections(soft, STREET_SCENE.gts, 0.5);
    expect(ev.tp).toBe(3);
    expect(ev.fp).toBe(0);
  });
});

describe('object-detection: Lab (Tune the Object Detector) exhaustive checks', () => {
  it('Brief 1 (clean street): 15/42 pass', () => {
    const ex = detLabExhaustive('clean');
    expect(ex.total).toBe(42);
    expect(ex.passing).toBe(15);
  });

  it('Brief 2 (crowded sidewalk): tight Hard NMS 0.40 deletes G2; Hard 0.65 & Soft-NMS pass (11/42)', () => {
    expect(evalDetLab('crowd', 0.45, 0.4, 'hard').pass).toBe(false);
    expect(evalDetLab('crowd', 0.45, 0.4, 'hard').fn).toBe(1);
    expect(evalDetLab('crowd', 0.45, 0.65, 'hard').pass).toBe(true);
    expect(evalDetLab('crowd', 0.35, 0.5, 'soft').pass).toBe(true);

    const ex = detLabExhaustive('crowd');
    expect(ex.total).toBe(42);
    expect(ex.passing).toBe(11);
  });

  it('Brief 3 (safety gate): conf 0.35 misses distant cyclist; conf 0.30 catches all 4 with 0 FP (10/42)', () => {
    expect(evalDetLab('safety', 0.35, 0.5, 'hard').pass).toBe(false);
    expect(evalDetLab('safety', 0.3, 0.5, 'hard').pass).toBe(true);
    expect(evalDetLab('safety', 0.3, 0.5, 'hard').tp).toBe(4);
    expect(evalDetLab('safety', 0.3, 0.5, 'hard').fp).toBe(0);

    const ex = detLabExhaustive('safety');
    expect(ex.total).toBe(42);
    expect(ex.passing).toBe(10);
  });
});
