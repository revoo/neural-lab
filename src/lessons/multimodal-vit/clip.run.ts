// authored-by: Claude Opus 5.5
/**
 * Records the mini-CLIP brief's grid (τ × projection size × batch, seeds 1–3).
 * Run: node --experimental-strip-types --import ./scripts/ts-hooks.mjs src/lessons/multimodal-vit/clip.run.ts
 * and paste the output into CLIP_RESULTS in visionResults.ts.
 */
import { CLIP_GRID, clipMean } from './visionLab';

for (const s of CLIP_GRID) console.log(JSON.stringify({ ...s, acc: +clipMean(s).toFixed(4) }) + ',');
