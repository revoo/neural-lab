// authored-by: Claude Opus 5.5
/** Recorded by lab.run.ts (exhaustive over every plan; noise seed 5, with seeds 6 and 7 as a robustness check). Verified by lab.test.ts. */
import type { LabBriefId } from './diffusion';

export type StepRow = { steps: number; total: number; passing: number; ddim: number; ddpm: number; best: number };
export const LAB_RESULTS: Record<LabBriefId, { fewest: number; perSteps: StepRow[]; fewestWithOtherSeeds: number[] }> = {
 "ring": {
  "fewest": 4,
  "perSteps": [
   {
    "steps": 1,
    "total": 2,
    "passing": 0,
    "ddim": 0,
    "ddpm": 0,
    "best": 1
   },
   {
    "steps": 2,
    "total": 28,
    "passing": 0,
    "ddim": 0,
    "ddpm": 0,
    "best": 0.7505
   },
   {
    "steps": 3,
    "total": 182,
    "passing": 0,
    "ddim": 0,
    "ddpm": 0,
    "best": 0.1595
   },
   {
    "steps": 4,
    "total": 728,
    "passing": 55,
    "ddim": 20,
    "ddpm": 35,
    "best": 0.047
   }
  ],
  "fewestWithOtherSeeds": [
   4,
   4
  ]
 },
 "smiley": {
  "fewest": 5,
  "perSteps": [
   {
    "steps": 1,
    "total": 2,
    "passing": 0,
    "ddim": 0,
    "ddpm": 0,
    "best": 1
   },
   {
    "steps": 2,
    "total": 28,
    "passing": 0,
    "ddim": 0,
    "ddpm": 0,
    "best": 0.4622
   },
   {
    "steps": 3,
    "total": 182,
    "passing": 0,
    "ddim": 0,
    "ddpm": 0,
    "best": 0.2166
   },
   {
    "steps": 4,
    "total": 728,
    "passing": 0,
    "ddim": 0,
    "ddpm": 0,
    "best": 0.1401
   },
   {
    "steps": 5,
    "total": 2002,
    "passing": 63,
    "ddim": 63,
    "ddpm": 0,
    "best": 0.1026
   }
  ],
  "fewestWithOtherSeeds": [
   5,
   5
  ]
 },
 "lopsided": {
  "fewest": 4,
  "perSteps": [
   {
    "steps": 1,
    "total": 2,
    "passing": 0,
    "ddim": 0,
    "ddpm": 0,
    "best": 0.1
   },
   {
    "steps": 2,
    "total": 28,
    "passing": 0,
    "ddim": 0,
    "ddpm": 0,
    "best": 0.0975
   },
   {
    "steps": 3,
    "total": 182,
    "passing": 0,
    "ddim": 0,
    "ddpm": 0,
    "best": 0.076
   },
   {
    "steps": 4,
    "total": 728,
    "passing": 6,
    "ddim": 5,
    "ddpm": 1,
    "best": 0.0585
   }
  ],
  "fewestWithOtherSeeds": [
   4,
   4
  ]
 }
};
