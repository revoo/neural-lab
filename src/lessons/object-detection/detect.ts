// authored-by: Gemini Antigravity
/**
 * Object Detection from scratch:
 *  - Exact axis-aligned bounding-box IoU (Intersection over Union) & small-vs-large shift sensitivity
 *  - Step-by-step Greedy NMS and Gaussian Soft-NMS over crowded multi-object street scenes
 *  - Exact ranked TP/FP matching (1-to-1 ground truth assignment) and interpolated AP / mAP@[.50:.95]
 *  - Exhaustive Lab: Tune the Object Detector (3 briefs: Clean Street, Crowded Sidewalk, Safety Gate)
 */

export type Box = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type Detection = Box & {
  id: string;
  score: number;
  cls: string;
};

export type GroundTruth = Box & {
  id: string;
  cls: string;
};

export function boxArea(b: Box): number {
  return Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
}

export function intersectionBox(a: Box, b: Box): Box | null {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  if (x2 <= x1 || y2 <= y1) return null;
  return { x1, y1, x2, y2 };
}

export function iou(a: Box, b: Box): number {
  const inter = intersectionBox(a, b);
  const interArea = inter ? boxArea(inter) : 0;
  const unionArea = boxArea(a) + boxArea(b) - interArea;
  return unionArea > 0 ? interArea / unionArea : 0;
}

export function iouBreakdown(a: Box, b: Box) {
  const inter = intersectionBox(a, b);
  const interArea = inter ? boxArea(inter) : 0;
  const areaA = boxArea(a);
  const areaB = boxArea(b);
  const unionArea = areaA + areaB - interArea;
  return { inter, interArea, areaA, areaB, unionArea, iou: unionArea > 0 ? interArea / unionArea : 0 };
}

/** Exact IoU of a square box of side S shifted horizontally by dx pixels. */
export function squareShiftIou(side: number, dx: number): number {
  const a: Box = { x1: 0, y1: 0, x2: side, y2: side };
  const b: Box = { x1: dx, y1: 0, x2: side + dx, y2: side };
  return iou(a, b);
}

// ------------------------------------------------------------------ 1. NMS & Soft-NMS Engine

export type NmsStep = {
  pickedId: string;
  pickedScore: number;
  suppressed: { id: string; iouWithPicked: number }[];
};

export function runHardNms(dets: readonly Detection[], iouThresh: number, confThresh = 0) {
  const sorted = dets
    .filter((d) => d.score >= confThresh)
    .slice()
    .sort((a, b) => b.score - a.score);
  const kept: Detection[] = [];
  const suppressedIds = new Set<string>();
  const steps: NmsStep[] = [];

  for (const cand of sorted) {
    if (suppressedIds.has(cand.id)) continue;
    kept.push(cand);
    const supThis: { id: string; iouWithPicked: number }[] = [];
    for (const other of sorted) {
      if (other.id === cand.id || suppressedIds.has(other.id) || other.cls !== cand.cls) continue;
      const ov = iou(cand, other);
      if (ov >= iouThresh) {
        suppressedIds.add(other.id);
        supThis.push({ id: other.id, iouWithPicked: ov });
      }
    }
    steps.push({ pickedId: cand.id, pickedScore: cand.score, suppressed: supThis });
  }
  return { kept, suppressedIds, steps };
}

/** Gaussian Soft-NMS: instead of zeroing overlapping boxes, decay score by exp(-iou^2 / sigma). */
export function runSoftNms(dets: readonly Detection[], sigma: number, confThresh = 0) {
  const pool = dets.map((d) => ({ ...d }));
  const kept: Detection[] = [];
  while (pool.length > 0) {
    let bestIdx = 0;
    for (let i = 1; i < pool.length; i++) {
      if (pool[i].score > pool[bestIdx].score) bestIdx = i;
    }
    const picked = pool.splice(bestIdx, 1)[0];
    if (picked.score < confThresh) break;
    kept.push(picked);
    for (const other of pool) {
      if (other.cls !== picked.cls) continue;
      const ov = iou(picked, other);
      if (ov > 0) other.score *= Math.exp(-(ov * ov) / sigma);
    }
  }
  return { kept };
}

// ------------------------------------------------------------------ 2. Exact Precision/Recall & Interpolated AP / mAP

export type MatchRow = {
  det: Detection;
  isTp: boolean;
  matchedGtId: string | null;
  bestIou: number;
  cumTp: number;
  cumFp: number;
  precision: number;
  recall: number;
};

export function evaluateDetections(dets: readonly Detection[], gts: readonly GroundTruth[], iouThresh = 0.5) {
  const sorted = dets.slice().sort((a, b) => b.score - a.score);
  const matchedGt = new Set<string>();
  const rows: MatchRow[] = [];
  let tp = 0;
  let fp = 0;

  for (const d of sorted) {
    let bestGt: GroundTruth | null = null;
    let bestIou = 0;
    for (const g of gts) {
      if (g.cls !== d.cls) continue;
      const ov = iou(d, g);
      if (ov > bestIou) {
        bestIou = ov;
        bestGt = g;
      }
    }
    const isTp = bestGt !== null && bestIou >= iouThresh && !matchedGt.has(bestGt.id);
    if (isTp && bestGt) {
      matchedGt.add(bestGt.id);
      tp++;
    } else {
      fp++;
    }
    rows.push({
      det: d,
      isTp,
      matchedGtId: isTp && bestGt ? bestGt.id : null,
      bestIou,
      cumTp: tp,
      cumFp: fp,
      precision: tp / (tp + fp),
      recall: gts.length > 0 ? tp / gts.length : 0,
    });
  }

  // All-point interpolated AP (standard Pascal VOC 2010+ / COCO envelope)
  const mrec = [0, ...rows.map((r) => r.recall), 1];
  const mpre = [0, ...rows.map((r) => r.precision), 0];
  for (let i = mpre.length - 2; i >= 0; i--) {
    if (mpre[i] < mpre[i + 1]) mpre[i] = mpre[i + 1];
  }
  let ap = 0;
  for (let i = 1; i < mrec.length; i++) {
    if (mrec[i] !== mrec[i - 1]) ap += (mrec[i] - mrec[i - 1]) * mpre[i];
  }

  const finalPrec = tp + fp > 0 ? tp / (tp + fp) : 0;
  const finalRec = gts.length > 0 ? tp / gts.length : 0;
  const f1 = finalPrec + finalRec > 0 ? (2 * finalPrec * finalRec) / (finalPrec + finalRec) : 0;
  return { tp, fp, fn: gts.length - tp, precision: finalPrec, recall: finalRec, f1, ap, rows };
}

export function computeCocoMap(dets: readonly Detection[], gts: readonly GroundTruth[]): number {
  let sum = 0;
  for (let i = 0; i < 10; i++) {
    const t = 0.5 + 0.05 * i;
    sum += evaluateDetections(dets, gts, t).ap;
  }
  return sum / 10;
}

// ------------------------------------------------------------------ 3. Street Scenes for Interactive NMS, PR & Lab

export const STREET_SCENE: { gts: GroundTruth[]; dets: Detection[] } = {
  gts: [
    { id: 'P1', cls: 'person', x1: 36, y1: 54, x2: 96, y2: 174 },
    { id: 'P2', cls: 'person', x1: 68, y1: 60, x2: 128, y2: 180 }, // overlapping neighbour! IoU(P1,P2) ~ 0.30
    { id: 'C1', cls: 'car', x1: 168, y1: 92, x2: 274, y2: 166 },
  ],
  dets: [
    { id: 'd1', cls: 'person', score: 0.94, x1: 38, y1: 56, x2: 98, y2: 174 },  // P1 primary
    { id: 'd2', cls: 'person', score: 0.81, x1: 42, y1: 60, x2: 100, y2: 178 }, // P1 dup (IoU ~ 0.81 with d1)
    { id: 'd3', cls: 'person', score: 0.76, x1: 66, y1: 62, x2: 126, y2: 180 }, // P2 primary (IoU ~ 0.34 with d1!)
    { id: 'd4', cls: 'person', score: 0.52, x1: 72, y1: 66, x2: 130, y2: 182 }, // P2 dup (IoU ~ 0.82 with d3)
    { id: 'd5', cls: 'car', score: 0.91, x1: 170, y1: 94, x2: 272, y2: 164 },   // C1 primary
    { id: 'd6', cls: 'car', score: 0.72, x1: 176, y1: 98, x2: 278, y2: 168 },   // C1 dup (IoU ~ 0.79 with d5)
    { id: 'd7', cls: 'car', score: 0.48, x1: 162, y1: 88, x2: 266, y2: 160 },   // C1 loose dup (IoU ~ 0.73 with d5)
    { id: 'd8', cls: 'person', score: 0.31, x1: 210, y1: 42, x2: 245, y2: 105 }, // background phantom
    { id: 'd9', cls: 'car', score: 0.22, x1: 14, y1: 120, x2: 74, y2: 170 },     // background phantom
  ],
};

// ------------------------------------------------------------------ 4. Lab: Tune the Object Detector (3 Exhaustive Briefs)

export type DetBriefId = 'clean' | 'crowd' | 'safety';

export const DET_LAB_BRIEFS: { id: DetBriefId; title: string; story: string }[] = [
  {
    id: 'clean',
    title: 'Brief 1 · Clean Street Scene (Balanced Precision & Recall)',
    story: 'Three objects (two nearby pedestrians + one car) with 9 raw anchor boxes. Tune Confidence threshold and Hard-NMS IoU threshold to reach F1 = 1.00 (0 false positives, 0 false negatives) at IoU 0.50.',
  },
  {
    id: 'crowd',
    title: 'Brief 2 · Crowded Sidewalk (Heavy Overlap: IoU = 0.53 Between Real People!)',
    story: 'Two pedestrians walk shoulder-to-shoulder (ground-truth overlap IoU = 0.53) while duplicate boxes sit at IoU ≥ 0.74. Pick NMS mode (Hard NMS vs Soft-NMS) and thresholds to keep 100% Recall AND Precision ≥ 0.80.',
  },
  {
    id: 'safety',
    title: 'Brief 3 · Autonomous-Braking Gate (Catch the Low-Confidence Distant Cyclist)',
    story: 'Four road users including a distant foggy cyclist predicted at confidence 0.32 (plus duplicate boxes and low-score clutter ≤ 0.22). Hit Recall = 100% (4/4) with at most 1 False Positive.',
  },
];

// Brief 2 crowded scene: 2 heavily overlapping people (IoU 0.53) + 1 solo person
const CROWD_SCENE: { gts: GroundTruth[]; dets: Detection[] } = {
  gts: [
    { id: 'G1', cls: 'person', x1: 40, y1: 50, x2: 100, y2: 170 },
    { id: 'G2', cls: 'person', x1: 58, y1: 52, x2: 118, y2: 172 }, // IoU with G1 = 0.533!
    { id: 'G3', cls: 'person', x1: 200, y1: 60, x2: 256, y2: 174 },
  ],
  dets: [
    { id: 'c1', cls: 'person', score: 0.93, x1: 41, y1: 51, x2: 101, y2: 171 }, // G1
    { id: 'c2', cls: 'person', score: 0.80, x1: 44, y1: 54, x2: 103, y2: 173 }, // G1 dup (IoU 0.87 with c1)
    { id: 'c3', cls: 'person', score: 0.75, x1: 58, y1: 52, x2: 118, y2: 172 }, // G2 (IoU 0.55 with c1!)
    { id: 'c4', cls: 'person', score: 0.61, x1: 61, y1: 55, x2: 120, y2: 174 }, // G2 dup (IoU 0.88 with c3)
    { id: 'c5', cls: 'person', score: 0.89, x1: 201, y1: 61, x2: 257, y2: 173 }, // G3
    { id: 'c6', cls: 'person', score: 0.68, x1: 205, y1: 64, x2: 260, y2: 176 }, // G3 dup (IoU 0.84 with c5)
    { id: 'c7', cls: 'person', score: 0.28, x1: 135, y1: 40, x2: 175, y2: 110 }, // phantom
  ],
};

// Brief 3 safety scene: 4 road users including distant cyclist at 0.32
const SAFETY_SCENE: { gts: GroundTruth[]; dets: Detection[] } = {
  gts: [
    { id: 'S1', cls: 'car', x1: 25, y1: 95, x2: 115, y2: 165 },
    { id: 'S2', cls: 'person', x1: 130, y1: 65, x2: 172, y2: 168 },
    { id: 'S3', cls: 'car', x1: 195, y1: 100, x2: 275, y2: 166 },
    { id: 'S4', cls: 'cyclist', x1: 160, y1: 105, x2: 186, y2: 155 }, // small distant cyclist!
  ],
  dets: [
    { id: 's1', cls: 'car', score: 0.95, x1: 27, y1: 96, x2: 116, y2: 164 },
    { id: 's2', cls: 'car', score: 0.74, x1: 31, y1: 100, x2: 120, y2: 168 }, // S1 dup (IoU 0.81)
    { id: 's3', cls: 'person', score: 0.88, x1: 131, y1: 66, x2: 171, y2: 167 },
    { id: 's4', cls: 'person', score: 0.66, x1: 134, y1: 69, x2: 174, y2: 170 }, // S2 dup (IoU 0.81)
    { id: 's5', cls: 'car', score: 0.91, x1: 196, y1: 101, x2: 274, y2: 165 },
    { id: 's6', cls: 'car', score: 0.69, x1: 200, y1: 104, x2: 278, y2: 168 }, // S3 dup (IoU 0.82)
    { id: 's7', cls: 'cyclist', score: 0.32, x1: 161, y1: 106, x2: 185, y2: 154 }, // S4 distant cyclist!
    { id: 's8', cls: 'person', score: 0.26, x1: 255, y1: 45, x2: 282, y2: 95 }, // 1 mild phantom
    { id: 's9', cls: 'car', score: 0.21, x1: 80, y1: 40, x2: 125, y2: 80 },     // clutter
    { id: 's10', cls: 'person', score: 0.18, x1: 10, y1: 50, x2: 40, y2: 100 }, // clutter
  ],
};

export const LAB_CONFS = [0.15, 0.25, 0.30, 0.35, 0.45, 0.55, 0.65] as const;
export const LAB_NMS_IOUS = [0.25, 0.40, 0.50, 0.65, 0.85] as const;
export type NmsMode = 'hard' | 'soft';

export function evalDetLab(brief: DetBriefId, conf: number, nmsIou: number, mode: NmsMode) {
  const scene = brief === 'clean' ? STREET_SCENE : brief === 'crowd' ? CROWD_SCENE : SAFETY_SCENE;
  const kept =
    mode === 'hard'
      ? runHardNms(scene.dets, nmsIou, conf).kept
      : runSoftNms(scene.dets, 0.45, conf).kept;
  const ev = evaluateDetections(kept, scene.gts, 0.5);
  const pass =
    brief === 'clean'
      ? ev.f1 >= 0.999
      : brief === 'crowd'
        ? ev.recall >= 0.999 && ev.precision >= 0.8 - 1e-9
        : ev.recall >= 0.999 && ev.fp <= 1;
  return { ...ev, kept, scene, pass };
}

export type BoxFate = 'kept-tp' | 'kept-fp' | 'suppressed' | 'below-conf';

export type DetTraceRow = {
  det: Detection;
  finalScore: number;
  fate: BoxFate;
  matchedGtId: string | null;
  suppressedBy: string | null;
  overlapWithActor: number;
};

/** Full per-box NMS / Soft-NMS trace for live inspection in the Object Detector lab. */
export function traceDetLab(brief: DetBriefId, conf: number, nmsIou: number, mode: NmsMode) {
  const res = evalDetLab(brief, conf, nmsIou, mode);
  const { scene } = res;
  const matchById = new Map(res.rows.map((r) => [r.det.id, r]));
  const matchedGtIds = new Set(res.rows.filter((r) => r.isTp && r.matchedGtId).map((r) => r.matchedGtId!));
  const missedGtIds = new Set(scene.gts.filter((g) => !matchedGtIds.has(g.id)).map((g) => g.id));

  const trace: DetTraceRow[] = [];

  if (mode === 'hard') {
    const hard = runHardNms(scene.dets, nmsIou, conf);
    const suppressedByMap = new Map<string, { by: string; ov: number }>();
    for (const step of hard.steps) {
      for (const s of step.suppressed) {
        suppressedByMap.set(s.id, { by: step.pickedId, ov: s.iouWithPicked });
      }
    }
    const sorted = scene.dets.slice().sort((a, b) => b.score - a.score);
    for (const det of sorted) {
      const m = matchById.get(det.id);
      if (m) {
        trace.push({
          det,
          finalScore: det.score,
          fate: m.isTp ? 'kept-tp' : 'kept-fp',
          matchedGtId: m.matchedGtId,
          suppressedBy: null,
          overlapWithActor: m.bestIou,
        });
      } else if (det.score < conf) {
        trace.push({
          det,
          finalScore: det.score,
          fate: 'below-conf',
          matchedGtId: null,
          suppressedBy: null,
          overlapWithActor: 0,
        });
      } else {
        const sup = suppressedByMap.get(det.id);
        trace.push({
          det,
          finalScore: 0,
          fate: 'suppressed',
          matchedGtId: null,
          suppressedBy: sup?.by ?? null,
          overlapWithActor: sup?.ov ?? 0,
        });
      }
    }
  } else {
    // Re-run Soft-NMS step by step to capture final decayed score & main suppressor
    const sigma = 0.45;
    const pool = scene.dets.map((d) => ({ det: d, score: d.score, topActor: null as string | null, topOv: 0 }));
    const finalScores = new Map<string, { score: number; actor: string | null; ov: number }>();
    while (pool.length > 0) {
      let bestIdx = 0;
      for (let i = 1; i < pool.length; i++) {
        if (pool[i].score > pool[bestIdx].score) bestIdx = i;
      }
      const picked = pool.splice(bestIdx, 1)[0];
      finalScores.set(picked.det.id, { score: picked.score, actor: picked.topActor, ov: picked.topOv });
      if (picked.score < conf) {
        for (const rest of pool) {
          finalScores.set(rest.det.id, { score: rest.score, actor: rest.topActor, ov: rest.topOv });
        }
        break;
      }
      for (const other of pool) {
        if (other.det.cls !== picked.det.cls) continue;
        const ov = iou(picked.det, other.det);
        if (ov > 0) {
          other.score *= Math.exp(-(ov * ov) / sigma);
          if (ov > other.topOv) {
            other.topOv = ov;
            other.topActor = picked.det.id;
          }
        }
      }
    }
    const sorted = scene.dets.slice().sort((a, b) => b.score - a.score);
    for (const det of sorted) {
      const m = matchById.get(det.id);
      const info = finalScores.get(det.id) ?? { score: det.score, actor: null, ov: 0 };
      if (m) {
        trace.push({
          det,
          finalScore: info.score,
          fate: m.isTp ? 'kept-tp' : 'kept-fp',
          matchedGtId: m.matchedGtId,
          suppressedBy: info.actor,
          overlapWithActor: info.ov,
        });
      } else {
        const decayedOut = det.score >= conf && info.score < conf;
        trace.push({
          det,
          finalScore: info.score,
          fate: decayedOut ? 'suppressed' : 'below-conf',
          matchedGtId: null,
          suppressedBy: decayedOut ? info.actor : null,
          overlapWithActor: decayedOut ? info.ov : 0,
        });
      }
    }
  }

  return { ...res, trace, missedGtIds };
}

export function detLabGrid(brief: DetBriefId) {
  const cells: { mode: NmsMode; conf: number; nmsIou: number; f1: number; tp: number; fp: number; fn: number; pass: boolean }[] = [];
  for (const mode of ['hard', 'soft'] as NmsMode[]) {
    const ious = mode === 'hard' ? LAB_NMS_IOUS : ([0.5] as const);
    for (const conf of LAB_CONFS) {
      for (const nmsIou of ious) {
        const r = evalDetLab(brief, conf, nmsIou, mode);
        cells.push({ mode, conf, nmsIou, f1: r.f1, tp: r.tp, fp: r.fp, fn: r.fn, pass: r.pass });
      }
    }
  }
  return cells;
}

/** Evidence-only hint for the Object Detector Lab (never names winning buttons or exact thresholds). */
export function detLabHint(brief: DetBriefId, conf: number, nmsIou: number, mode: NmsMode): string {
  const tr = traceDetLab(brief, conf, nmsIou, mode);
  if (tr.pass) {
    return tr.f1 >= 0.999
      ? `All targets cleared with F1 = 1.00 (${tr.tp} TP, 0 FP, 0 FN). Inspect the post-solve map to see the whole passing region.`
      : `Brief cleared (Recall ${(100 * tr.recall).toFixed(1)}%, ${tr.fp} FP). Check the trace to see whether the remaining false positive can be trimmed too.`;
  }
  if (tr.fn > 0) {
    const missed = [...tr.missedGtIds].join(', ');
    const supRows = tr.trace.filter((r) => r.fate === 'suppressed');
    const belowRows = tr.trace.filter((r) => r.fate === 'below-conf');
    // Did an object get wiped out by overlap suppression or by confidence filtering?
    if (brief === 'crowd' && mode === 'hard' && supRows.some((r) => r.overlapWithActor < 0.7 && r.overlapWithActor > 0.4)) {
      return `Ground truth ${missed} is missing because its box was suppressed by a neighbouring box at moderate IoU (~0.55), well below the true-duplicate IoUs (≥ 0.84). Look at the trace's IoU column.`;
    }
    if (belowRows.some((r) => tr.scene.gts.some((g) => g.cls === r.det.cls && iou(r.det, g) >= 0.5 && tr.missedGtIds.has(g.id)))) {
      return `Ground truth ${missed} is missing because its highest-scoring detection fell below your current confidence cut-off. Check the trace rows marked below-conf.`;
    }
    return `Ground truth ${missed} is missing (Recall ${(100 * tr.recall).toFixed(1)}%). Inspect the trace table to see whether its matching box was dropped by confidence or overlap suppression.`;
  }
  // Recall is 100%, so FPs are the blocker
  const fps = tr.trace.filter((r) => r.fate === 'kept-fp');
  const minFpScore = Math.min(...fps.map((r) => r.det.score));
  const hasLowPhantom = fps.some((r) => tr.scene.gts.every((g) => iou(r.det, g) < 0.3));
  const hasDup = fps.some((r) => tr.scene.gts.some((g) => g.cls === r.det.cls && iou(r.det, g) >= 0.5));
  if (hasLowPhantom && hasDup) {
    return `You have ${tr.fp} false positives: both low-scoring background phantoms (down to ${minFpScore.toFixed(2)}) and surviving duplicate boxes on already-matched objects. Check both suppression and confidence.`;
  }
  if (hasDup) {
    return `Recall is 100%, but ${tr.fp} duplicate box${tr.fp > 1 ? 'es' : ''} survived on already-matched objects. Look at their high overlap with the primary detections in the trace.`;
  }
  return `Recall is 100%, but ${tr.fp} background phantom${tr.fp > 1 ? 's' : ''} (score as low as ${minFpScore.toFixed(2)}) still clear your confidence filter.`;
}

export function detLabExhaustive(brief: DetBriefId) {
  let total = 0;
  let passing = 0;
  let bestF1 = -1;
  let bestDesc = '';
  for (const mode of ['hard', 'soft'] as NmsMode[]) {
    const ious = mode === 'hard' ? LAB_NMS_IOUS : ([0.5] as const);
    for (const conf of LAB_CONFS) {
      for (const nmsIou of ious) {
        total++;
        const r = evalDetLab(brief, conf, nmsIou, mode);
        if (!r.pass) continue;
        passing++;
        if (r.f1 > bestF1) {
          bestF1 = r.f1;
          bestDesc =
            mode === 'hard'
              ? `Hard NMS (conf ${conf.toFixed(2)}, IoU ${nmsIou.toFixed(2)}) -> F1 ${r.f1.toFixed(2)}, TP ${r.tp}, FP ${r.fp}`
              : `Soft-NMS (conf ${conf.toFixed(2)}) -> F1 ${r.f1.toFixed(2)}, TP ${r.tp}, FP ${r.fp}`;
        }
      }
    }
  }
  return { total, passing, summary: `Best: ${bestDesc}` };
}
