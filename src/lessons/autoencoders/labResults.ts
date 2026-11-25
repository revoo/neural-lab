// authored-by: Claude Opus 5.5
/**
 * Recorded by `lab.run.ts` (every design × seeds 1, 2, 3, in Node). Do not edit by hand: re-run
 * the script and regenerate. `lab.test.ts` re-trains a sample of designs and checks them.
 */
import { BRIEFS, LAB_SEEDS, allDesigns, designKey, paramCount, verdict, verdictAll, type BriefId, type Design, type Measured } from './lab';

export const LAB_RESULTS: Record<string, { seeds: Measured[] }> = {
 "1-8-0": {
  "seeds": [
   {
    "recon": 0.12063,
    "realShare": 0.155,
    "cells": 1
   },
   {
    "recon": 0.12168,
    "realShare": 0.095,
    "cells": 1
   },
   {
    "recon": 0.12096,
    "realShare": 0.12,
    "cells": 2
   }
  ]
 },
 "1-8-0.1": {
  "seeds": [
   {
    "recon": 0.08745,
    "realShare": 0.92,
    "cells": 7
   },
   {
    "recon": 0.08165,
    "realShare": 0.965,
    "cells": 5
   },
   {
    "recon": 0.08049,
    "realShare": 0.92,
    "cells": 5
   }
  ]
 },
 "1-8-0.3": {
  "seeds": [
   {
    "recon": 0.08273,
    "realShare": 0.99,
    "cells": 5
   },
   {
    "recon": 0.08223,
    "realShare": 1,
    "cells": 5
   },
   {
    "recon": 0.0797,
    "realShare": 1,
    "cells": 5
   }
  ]
 },
 "1-8-1": {
  "seeds": [
   {
    "recon": 0.0979,
    "realShare": 1,
    "cells": 4
   },
   {
    "recon": 0.12136,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12024,
    "realShare": 1,
    "cells": 2
   }
  ]
 },
 "1-8-3": {
  "seeds": [
   {
    "recon": 0.12066,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12162,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12095,
    "realShare": 1,
    "cells": 1
   }
  ]
 },
 "1-8-10": {
  "seeds": [
   {
    "recon": 0.12069,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12155,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12096,
    "realShare": 1,
    "cells": 1
   }
  ]
 },
 "1-16-0": {
  "seeds": [
   {
    "recon": 0.12053,
    "realShare": 0.055,
    "cells": 1
   },
   {
    "recon": 0.12264,
    "realShare": 0.045,
    "cells": 1
   },
   {
    "recon": 0.12096,
    "realShare": 0.06,
    "cells": 1
   }
  ]
 },
 "1-16-0.1": {
  "seeds": [
   {
    "recon": 0.06932,
    "realShare": 0.975,
    "cells": 5
   },
   {
    "recon": 0.0513,
    "realShare": 0.955,
    "cells": 8
   },
   {
    "recon": 0.06174,
    "realShare": 0.99,
    "cells": 6
   }
  ]
 },
 "1-16-0.3": {
  "seeds": [
   {
    "recon": 0.07747,
    "realShare": 1,
    "cells": 6
   },
   {
    "recon": 0.07353,
    "realShare": 1,
    "cells": 6
   },
   {
    "recon": 0.07366,
    "realShare": 1,
    "cells": 6
   }
  ]
 },
 "1-16-1": {
  "seeds": [
   {
    "recon": 0.08,
    "realShare": 1,
    "cells": 4
   },
   {
    "recon": 0.08308,
    "realShare": 1,
    "cells": 5
   },
   {
    "recon": 0.08067,
    "realShare": 1,
    "cells": 5
   }
  ]
 },
 "1-16-3": {
  "seeds": [
   {
    "recon": 0.1205,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12234,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12096,
    "realShare": 1,
    "cells": 1
   }
  ]
 },
 "1-16-10": {
  "seeds": [
   {
    "recon": 0.12049,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12217,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12098,
    "realShare": 1,
    "cells": 1
   }
  ]
 },
 "1-32-0": {
  "seeds": [
   {
    "recon": 0.06258,
    "realShare": 0.445,
    "cells": 3
   },
   {
    "recon": 0.06597,
    "realShare": 0.465,
    "cells": 3
   },
   {
    "recon": 0.07766,
    "realShare": 0.23,
    "cells": 1
   }
  ]
 },
 "1-32-0.1": {
  "seeds": [
   {
    "recon": 0.0453,
    "realShare": 1,
    "cells": 9
   },
   {
    "recon": 0.06592,
    "realShare": 1,
    "cells": 6
   },
   {
    "recon": 0.06062,
    "realShare": 1,
    "cells": 10
   }
  ]
 },
 "1-32-0.3": {
  "seeds": [
   {
    "recon": 0.04569,
    "realShare": 1,
    "cells": 9
   },
   {
    "recon": 0.04748,
    "realShare": 1,
    "cells": 9
   },
   {
    "recon": 0.05609,
    "realShare": 1,
    "cells": 11
   }
  ]
 },
 "1-32-1": {
  "seeds": [
   {
    "recon": 0.05913,
    "realShare": 1,
    "cells": 7
   },
   {
    "recon": 0.07966,
    "realShare": 1,
    "cells": 5
   },
   {
    "recon": 0.08084,
    "realShare": 1,
    "cells": 5
   }
  ]
 },
 "1-32-3": {
  "seeds": [
   {
    "recon": 0.10417,
    "realShare": 1,
    "cells": 5
   },
   {
    "recon": 0.09592,
    "realShare": 1,
    "cells": 4
   },
   {
    "recon": 0.09282,
    "realShare": 1,
    "cells": 4
   }
  ]
 },
 "1-32-10": {
  "seeds": [
   {
    "recon": 0.12058,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12289,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12094,
    "realShare": 1,
    "cells": 1
   }
  ]
 },
 "2-8-0": {
  "seeds": [
   {
    "recon": 0.12055,
    "realShare": 0.15,
    "cells": 2
   },
   {
    "recon": 0.12029,
    "realShare": 0.205,
    "cells": 2
   },
   {
    "recon": 0.12118,
    "realShare": 0.145,
    "cells": 2
   }
  ]
 },
 "2-8-0.1": {
  "seeds": [
   {
    "recon": 0.03041,
    "realShare": 0.925,
    "cells": 16
   },
   {
    "recon": 0.02995,
    "realShare": 0.745,
    "cells": 16
   },
   {
    "recon": 0.03002,
    "realShare": 0.83,
    "cells": 15
   }
  ]
 },
 "2-8-0.3": {
  "seeds": [
   {
    "recon": 0.03593,
    "realShare": 1,
    "cells": 13
   },
   {
    "recon": 0.0813,
    "realShare": 0.99,
    "cells": 5
   },
   {
    "recon": 0.03056,
    "realShare": 0.99,
    "cells": 16
   }
  ]
 },
 "2-8-1": {
  "seeds": [
   {
    "recon": 0.12055,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12024,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.08651,
    "realShare": 1,
    "cells": 5
   }
  ]
 },
 "2-8-3": {
  "seeds": [
   {
    "recon": 0.12056,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12032,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12114,
    "realShare": 1,
    "cells": 1
   }
  ]
 },
 "2-8-10": {
  "seeds": [
   {
    "recon": 0.12058,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12035,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12111,
    "realShare": 1,
    "cells": 1
   }
  ]
 },
 "2-16-0": {
  "seeds": [
   {
    "recon": 0.02304,
    "realShare": 0.425,
    "cells": 5
   },
   {
    "recon": 0.12004,
    "realShare": 0.075,
    "cells": 1
   },
   {
    "recon": 0.04272,
    "realShare": 0.28,
    "cells": 7
   }
  ]
 },
 "2-16-0.1": {
  "seeds": [
   {
    "recon": 0.01901,
    "realShare": 0.885,
    "cells": 16
   },
   {
    "recon": 0.01599,
    "realShare": 0.765,
    "cells": 15
   },
   {
    "recon": 0.01455,
    "realShare": 0.93,
    "cells": 16
   }
  ]
 },
 "2-16-0.3": {
  "seeds": [
   {
    "recon": 0.01926,
    "realShare": 1,
    "cells": 16
   },
   {
    "recon": 0.01677,
    "realShare": 1,
    "cells": 16
   },
   {
    "recon": 0.02479,
    "realShare": 1,
    "cells": 16
   }
  ]
 },
 "2-16-1": {
  "seeds": [
   {
    "recon": 0.02867,
    "realShare": 1,
    "cells": 16
   },
   {
    "recon": 0.0787,
    "realShare": 1,
    "cells": 5
   },
   {
    "recon": 0.03116,
    "realShare": 1,
    "cells": 16
   }
  ]
 },
 "2-16-3": {
  "seeds": [
   {
    "recon": 0.12026,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12002,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12167,
    "realShare": 1,
    "cells": 1
   }
  ]
 },
 "2-16-10": {
  "seeds": [
   {
    "recon": 0.12033,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12005,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12155,
    "realShare": 1,
    "cells": 1
   }
  ]
 },
 "2-32-0": {
  "seeds": [
   {
    "recon": 0.0142,
    "realShare": 0.19,
    "cells": 3
   },
   {
    "recon": 0.04965,
    "realShare": 0.15,
    "cells": 7
   },
   {
    "recon": 0.01455,
    "realShare": 0.195,
    "cells": 4
   }
  ]
 },
 "2-32-0.1": {
  "seeds": [
   {
    "recon": 0.01372,
    "realShare": 0.885,
    "cells": 16
   },
   {
    "recon": 0.01132,
    "realShare": 0.97,
    "cells": 16
   },
   {
    "recon": 0.01372,
    "realShare": 0.94,
    "cells": 16
   }
  ]
 },
 "2-32-0.3": {
  "seeds": [
   {
    "recon": 0.01493,
    "realShare": 1,
    "cells": 16
   },
   {
    "recon": 0.01396,
    "realShare": 1,
    "cells": 16
   },
   {
    "recon": 0.0146,
    "realShare": 1,
    "cells": 16
   }
  ]
 },
 "2-32-1": {
  "seeds": [
   {
    "recon": 0.0266,
    "realShare": 1,
    "cells": 16
   },
   {
    "recon": 0.0245,
    "realShare": 1,
    "cells": 16
   },
   {
    "recon": 0.02357,
    "realShare": 1,
    "cells": 16
   }
  ]
 },
 "2-32-3": {
  "seeds": [
   {
    "recon": 0.09891,
    "realShare": 1,
    "cells": 4
   },
   {
    "recon": 0.06474,
    "realShare": 1,
    "cells": 14
   },
   {
    "recon": 0.09187,
    "realShare": 1,
    "cells": 5
   }
  ]
 },
 "2-32-10": {
  "seeds": [
   {
    "recon": 0.12017,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12007,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12208,
    "realShare": 1,
    "cells": 1
   }
  ]
 },
 "3-8-0": {
  "seeds": [
   {
    "recon": 0.12044,
    "realShare": 0.125,
    "cells": 1
   },
   {
    "recon": 0.12138,
    "realShare": 0.12,
    "cells": 2
   },
   {
    "recon": 0.12122,
    "realShare": 0.16,
    "cells": 2
   }
  ]
 },
 "3-8-0.1": {
  "seeds": [
   {
    "recon": 0.03589,
    "realShare": 0.95,
    "cells": 14
   },
   {
    "recon": 0.08212,
    "realShare": 0.845,
    "cells": 6
   },
   {
    "recon": 0.02971,
    "realShare": 0.93,
    "cells": 12
   }
  ]
 },
 "3-8-0.3": {
  "seeds": [
   {
    "recon": 0.03564,
    "realShare": 1,
    "cells": 16
   },
   {
    "recon": 0.08587,
    "realShare": 1,
    "cells": 3
   },
   {
    "recon": 0.08864,
    "realShare": 1,
    "cells": 4
   }
  ]
 },
 "3-8-1": {
  "seeds": [
   {
    "recon": 0.12016,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12134,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12119,
    "realShare": 1,
    "cells": 1
   }
  ]
 },
 "3-8-3": {
  "seeds": [
   {
    "recon": 0.12044,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12138,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12122,
    "realShare": 1,
    "cells": 1
   }
  ]
 },
 "3-8-10": {
  "seeds": [
   {
    "recon": 0.12044,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12138,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12121,
    "realShare": 1,
    "cells": 1
   }
  ]
 },
 "3-16-0": {
  "seeds": [
   {
    "recon": 0.00907,
    "realShare": 0.155,
    "cells": 12
   },
   {
    "recon": 0.01269,
    "realShare": 0.17,
    "cells": 12
   },
   {
    "recon": 0.12116,
    "realShare": 0.045,
    "cells": 2
   }
  ]
 },
 "3-16-0.1": {
  "seeds": [
   {
    "recon": 0.01422,
    "realShare": 0.69,
    "cells": 16
   },
   {
    "recon": 0.02621,
    "realShare": 0.66,
    "cells": 16
   },
   {
    "recon": 0.01322,
    "realShare": 0.805,
    "cells": 16
   }
  ]
 },
 "3-16-0.3": {
  "seeds": [
   {
    "recon": 0.01601,
    "realShare": 0.915,
    "cells": 16
   },
   {
    "recon": 0.01809,
    "realShare": 1,
    "cells": 16
   },
   {
    "recon": 0.01853,
    "realShare": 1,
    "cells": 16
   }
  ]
 },
 "3-16-1": {
  "seeds": [
   {
    "recon": 0.03006,
    "realShare": 1,
    "cells": 16
   },
   {
    "recon": 0.03125,
    "realShare": 1,
    "cells": 16
   },
   {
    "recon": 0.03625,
    "realShare": 1,
    "cells": 16
   }
  ]
 },
 "3-16-3": {
  "seeds": [
   {
    "recon": 0.12043,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.1212,
    "realShare": 1,
    "cells": 2
   },
   {
    "recon": 0.12118,
    "realShare": 1,
    "cells": 1
   }
  ]
 },
 "3-16-10": {
  "seeds": [
   {
    "recon": 0.12044,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12128,
    "realShare": 1,
    "cells": 2
   },
   {
    "recon": 0.12121,
    "realShare": 1,
    "cells": 1
   }
  ]
 },
 "3-32-0": {
  "seeds": [
   {
    "recon": 0.00894,
    "realShare": 0.045,
    "cells": 6
   },
   {
    "recon": 0.01189,
    "realShare": 0.185,
    "cells": 11
   },
   {
    "recon": 0.00471,
    "realShare": 0.075,
    "cells": 9
   }
  ]
 },
 "3-32-0.1": {
  "seeds": [
   {
    "recon": 0.00928,
    "realShare": 0.795,
    "cells": 16
   },
   {
    "recon": 0.01383,
    "realShare": 0.84,
    "cells": 16
   },
   {
    "recon": 0.01246,
    "realShare": 0.735,
    "cells": 16
   }
  ]
 },
 "3-32-0.3": {
  "seeds": [
   {
    "recon": 0.01523,
    "realShare": 1,
    "cells": 16
   },
   {
    "recon": 0.01596,
    "realShare": 1,
    "cells": 16
   },
   {
    "recon": 0.01344,
    "realShare": 1,
    "cells": 16
   }
  ]
 },
 "3-32-1": {
  "seeds": [
   {
    "recon": 0.02335,
    "realShare": 1,
    "cells": 16
   },
   {
    "recon": 0.0248,
    "realShare": 1,
    "cells": 16
   },
   {
    "recon": 0.02183,
    "realShare": 1,
    "cells": 16
   }
  ]
 },
 "3-32-3": {
  "seeds": [
   {
    "recon": 0.06828,
    "realShare": 1,
    "cells": 12
   },
   {
    "recon": 0.06756,
    "realShare": 1,
    "cells": 13
   },
   {
    "recon": 0.0633,
    "realShare": 1,
    "cells": 12
   }
  ]
 },
 "3-32-10": {
  "seeds": [
   {
    "recon": 0.12043,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12106,
    "realShare": 1,
    "cells": 1
   },
   {
    "recon": 0.12109,
    "realShare": 1,
    "cells": 1
   }
  ]
 }
};

const brief = (id: BriefId) => BRIEFS.find((b) => b.id === id)!;

/** Designs that meet the brief's spec with every one of the lab's seeds. */
export function passingDesigns(id: BriefId): Design[] {
  return allDesigns().filter((d) => verdictAll(brief(id), LAB_RESULTS[designKey(d)].seeds));
}

/** Designs that would have passed if only seed 1 had been tried (lucky ones included). */
export function passingSeed1(id: BriefId): Design[] {
  return allDesigns().filter((d) => verdict(brief(id), LAB_RESULTS[designKey(d)].seeds[0]).pass);
}

/** The passing design with the fewest parameters. */
export function labOptimum(id: BriefId) {
  const pass = passingDesigns(id);
  const min = Math.min(...pass.map(paramCount));
  const best = pass.filter((d) => paramCount(d) === min);
  const lucky = passingSeed1(id);
  const luckyMin = Math.min(...lucky.map(paramCount));
  return { design: best[0], ties: best.length, params: min, passing: pass.length, seeds: LAB_SEEDS.length, seed1Passing: lucky.length, seed1Smallest: lucky.find((d) => paramCount(d) === luckyMin)! };
}
