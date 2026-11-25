// authored-by: Gemini Antigravity
/**
 * Recorded 30-configuration × 5-seed grid search for "Lab: Keep the Duel Balanced",
 * produced by src/lessons/gans/search.run.ts (seeds 1..5, 240 outer steps, base lrD = 0.015).
 * Re-verified against live evaluateLabSetting runs in src/lessons/gans/gan.test.ts.
 */
import type { LabSetting } from './gan';

export type GridSummaryRow = LabSetting & {
  meanModes: number;
  allThreeSeeds: number;
  meanW1: number;
  meanOffMode: number;
  updatesToCover3: number | null;
};

export const LAB_GRID_RESULTS: GridSummaryRow[] = [
  { noiseDim: 1, dSteps: 1, lrRatio: 0.25, meanModes: 3.0, allThreeSeeds: 5, meanW1: 0.407, meanOffMode: 0.217, updatesToCover3: 80 },
  { noiseDim: 1, dSteps: 1, lrRatio: 0.5, meanModes: 2.6, allThreeSeeds: 3, meanW1: 0.433, meanOffMode: 0.171, updatesToCover3: 40 },
  { noiseDim: 1, dSteps: 1, lrRatio: 1, meanModes: 2.8, allThreeSeeds: 4, meanW1: 0.714, meanOffMode: 0.196, updatesToCover3: null },
  { noiseDim: 1, dSteps: 1, lrRatio: 2, meanModes: 2.0, allThreeSeeds: 1, meanW1: 0.98, meanOffMode: 0.313, updatesToCover3: null },
  { noiseDim: 1, dSteps: 1, lrRatio: 4, meanModes: 0.4, allThreeSeeds: 0, meanW1: 2.686, meanOffMode: 0.681, updatesToCover3: null },
  { noiseDim: 1, dSteps: 2, lrRatio: 0.25, meanModes: 2.8, allThreeSeeds: 4, meanW1: 0.363, meanOffMode: 0.183, updatesToCover3: 120 },
  { noiseDim: 1, dSteps: 2, lrRatio: 0.5, meanModes: 2.4, allThreeSeeds: 2, meanW1: 0.61, meanOffMode: 0.118, updatesToCover3: 60 },
  { noiseDim: 1, dSteps: 2, lrRatio: 1, meanModes: 2.4, allThreeSeeds: 2, meanW1: 0.583, meanOffMode: 0.094, updatesToCover3: 60 },
  { noiseDim: 1, dSteps: 2, lrRatio: 2, meanModes: 2.0, allThreeSeeds: 1, meanW1: 1.105, meanOffMode: 0.277, updatesToCover3: null },
  { noiseDim: 1, dSteps: 2, lrRatio: 4, meanModes: 1.8, allThreeSeeds: 1, meanW1: 1.226, meanOffMode: 0.429, updatesToCover3: null },
  { noiseDim: 1, dSteps: 4, lrRatio: 0.25, meanModes: 1.4, allThreeSeeds: 0, meanW1: 1.095, meanOffMode: 0.038, updatesToCover3: null },
  { noiseDim: 1, dSteps: 4, lrRatio: 0.5, meanModes: 2.6, allThreeSeeds: 3, meanW1: 0.417, meanOffMode: 0.146, updatesToCover3: 100 },
  { noiseDim: 1, dSteps: 4, lrRatio: 1, meanModes: 2.6, allThreeSeeds: 3, meanW1: 0.433, meanOffMode: 0.145, updatesToCover3: 100 },
  { noiseDim: 1, dSteps: 4, lrRatio: 2, meanModes: 2.2, allThreeSeeds: 1, meanW1: 0.736, meanOffMode: 0.216, updatesToCover3: null },
  { noiseDim: 1, dSteps: 4, lrRatio: 4, meanModes: 1.6, allThreeSeeds: 1, meanW1: 1.042, meanOffMode: 0.357, updatesToCover3: null },
  { noiseDim: 2, dSteps: 1, lrRatio: 0.25, meanModes: 3.0, allThreeSeeds: 5, meanW1: 0.382, meanOffMode: 0.233, updatesToCover3: 80 },
  { noiseDim: 2, dSteps: 1, lrRatio: 0.5, meanModes: 3.0, allThreeSeeds: 5, meanW1: 0.396, meanOffMode: 0.173, updatesToCover3: 40 },
  { noiseDim: 2, dSteps: 1, lrRatio: 1, meanModes: 2.4, allThreeSeeds: 3, meanW1: 0.54, meanOffMode: 0.187, updatesToCover3: 400 },
  { noiseDim: 2, dSteps: 1, lrRatio: 2, meanModes: 2.0, allThreeSeeds: 2, meanW1: 1.121, meanOffMode: 0.417, updatesToCover3: null },
  { noiseDim: 2, dSteps: 1, lrRatio: 4, meanModes: 1.6, allThreeSeeds: 1, meanW1: 1.155, meanOffMode: 0.353, updatesToCover3: null },
  { noiseDim: 2, dSteps: 2, lrRatio: 0.25, meanModes: 3.0, allThreeSeeds: 5, meanW1: 0.272, meanOffMode: 0.166, updatesToCover3: 120 },
  { noiseDim: 2, dSteps: 2, lrRatio: 0.5, meanModes: 2.4, allThreeSeeds: 2, meanW1: 0.494, meanOffMode: 0.138, updatesToCover3: 60 },
  { noiseDim: 2, dSteps: 2, lrRatio: 1, meanModes: 2.4, allThreeSeeds: 2, meanW1: 0.591, meanOffMode: 0.107, updatesToCover3: 60 },
  { noiseDim: 2, dSteps: 2, lrRatio: 2, meanModes: 2.4, allThreeSeeds: 3, meanW1: 0.499, meanOffMode: 0.124, updatesToCover3: null },
  { noiseDim: 2, dSteps: 2, lrRatio: 4, meanModes: 1.8, allThreeSeeds: 0, meanW1: 1.839, meanOffMode: 0.319, updatesToCover3: null },
  { noiseDim: 2, dSteps: 4, lrRatio: 0.25, meanModes: 1.6, allThreeSeeds: 0, meanW1: 0.924, meanOffMode: 0.049, updatesToCover3: null },
  { noiseDim: 2, dSteps: 4, lrRatio: 0.5, meanModes: 2.8, allThreeSeeds: 4, meanW1: 0.35, meanOffMode: 0.147, updatesToCover3: 100 },
  { noiseDim: 2, dSteps: 4, lrRatio: 1, meanModes: 2.6, allThreeSeeds: 3, meanW1: 0.443, meanOffMode: 0.141, updatesToCover3: 100 },
  { noiseDim: 2, dSteps: 4, lrRatio: 2, meanModes: 2.4, allThreeSeeds: 3, meanW1: 0.636, meanOffMode: 0.199, updatesToCover3: null },
  { noiseDim: 2, dSteps: 4, lrRatio: 4, meanModes: 2.0, allThreeSeeds: 1, meanW1: 1.668, meanOffMode: 0.317, updatesToCover3: null },
];

export const passesBrief1 = (r: { meanModes: number; meanW1: number }) => r.meanModes >= 2.0 && r.meanW1 <= 0.8;
export const passesBrief2 = (r: { meanModes: number; meanW1: number }) => r.meanModes >= 2.8 && r.meanW1 <= 0.45;
export const passesBrief3 = (r: { meanModes: number; meanW1: number; updatesToCover3: number | null }) =>
  passesBrief2(r) && r.updatesToCover3 !== null && r.updatesToCover3 <= 40;
