// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'det.iou-calc',
    kind: 'numeric',
    concepts: ['iou'],
    tolerance: 0.01,
    make: (rng) => {
      const side = [40, 60, 80, 100][rng.int(4)];
      const dx = Math.round(side * [0.1, 0.2, 0.25, 0.5][rng.int(4)]);
      const inter = (side - dx) * side;
      const union = 2 * side * side - inter;
      const val = Number((inter / union).toFixed(3));
      return {
        prompt: `A ground-truth square box of size ${side}×${side} pixels is predicted with the exact same size, shifted horizontally by ${dx} pixels (0 vertical shift). What is its IoU (to 3 decimal places)?`,
        answer: val,
        explain: `Intersection = (${side} − ${dx}) × ${side} = ${inter} px². Union = 2×${side * side} − ${inter} = ${union} px². IoU = ${inter}/${union} = ${val.toFixed(3)}.`,
      };
    },
  },
  {
    id: 'det.small-objects',
    kind: 'choice',
    concepts: ['iou'],
    prompt: 'Shifting a 100×100 car box by 10 pixels gives IoU = 0.818 (passes AP75), while shifting a 20×20 distant pedestrian by the exact same 10 pixels drops IoU to 0.333 (fails AP50). Why?',
    options: [
      {
        text: 'IoU is scale-normalised: a fixed pixel error is a much larger fraction of a small object’s width, slashing intersection and ballooning union relative to box area.',
        correct: true,
        why: 'That is why object detectors use multi-scale Feature Pyramid Networks (FPNs) and higher-resolution grids to localise small distant objects accurately.',
      },
      {
        text: 'Because IoU adds a flat penalty to boxes smaller than 32×32 pixels.',
        why: 'The formula Intersection / Union is identical for every box size; the sensitivity is pure geometry.',
      },
      {
        text: 'Because small objects are always suppressed by NMS.',
        why: 'Even an isolated 20×20 box with zero neighbours suffers the exact same 0.333 IoU from a 10-pixel shift.',
      },
    ],
  },
  {
    id: 'det.crowd-nms',
    kind: 'choice',
    concepts: ['nms'],
    prompt: 'Two real pedestrians walk shoulder-to-shoulder with ground-truth overlap IoU = 0.53. What goes wrong if you deploy standard Hard NMS with a tight threshold of 0.40?',
    options: [
      {
        text: 'After keeping the higher-confidence pedestrian, Hard NMS sees IoU 0.55 ≥ 0.40 and permanently deletes the second pedestrian, causing a missed detection (False Negative).',
        correct: true,
        why: 'In crowded scenes you either raise the NMS IoU threshold (e.g., 0.65), switch to Soft-NMS (decaying scores instead of hard-zeroing), or use set-prediction detectors like DETR that need no NMS at all.',
      },
      {
        text: 'Hard NMS averages the two boxes into one giant pedestrian.',
        why: 'Hard NMS never averages boxes; it keeps the top box unchanged and deletes overlapping ones.',
      },
      {
        text: 'Hard NMS ignores overlaps between detections of the same class.',
        why: 'The opposite: standard NMS specifically suppresses overlapping boxes of the *same* class.',
      },
    ],
  },
  {
    id: 'det.duplicate-fp',
    kind: 'choice',
    concepts: ['map'],
    prompt: 'Suppose a detector outputs TWO boxes with IoU = 0.90 over the exact same car (one at confidence 0.91, one at 0.72). When computing AP50, how is the second box scored?',
    options: [
      {
        text: 'As a False Positive (FP), because each ground-truth object can only be matched once to the highest-confidence prediction.',
        correct: true,
        why: 'Otherwise a model could cheat AP by outputting 100 copies of every easy car! That 1-to-1 matching rule is precisely why NMS is indispensable.',
      },
      {
        text: 'As a second True Positive, boosting recall above 100%.',
        why: 'Recall can never exceed 100%; every ground-truth box can only be claimed once.',
      },
      {
        text: 'It is ignored and excluded from precision and recall.',
        why: 'Extra duplicate boxes actively penalise precision as False Positives.',
      },
    ],
  },
];
