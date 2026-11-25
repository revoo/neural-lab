// authored-by: Gemini Antigravity
/** Committed search script for the 30-config x 5-seed GAN lab grid. Run via:
 *  node --experimental-strip-types --import ./scripts/ts-hooks.mjs src/lessons/gans/search.run.ts
 */
import * as g from './gan';
import { createRng } from '../../lib/rng';

console.log('=== PRESETS ===');
for (const [id, p] of Object.entries(g.GAN_PRESETS)) {
  const r = new g.GanRunner(p.config);
  const s0 = r.snapshot().metrics;
  // Also measure non-saturating grad at step 0 for vanishing preset
  const probeZ = g.sampleNoise(36, p.config.noiseDim, createRng(p.config.seed * 4099 + 1));
  const ns0 = g.backpropGenerator(r.G, r.D, probeZ, 'non-saturating');
  r.step(240);
  const sEnd = r.snapshot().metrics;
  console.log(
    id,
    'step0:', { modes: s0.modesCovered, dFake: +s0.meanDFake.toFixed(5), gGrad: +s0.gGradNorm.toFixed(5), nsGrad: +ns0.gGradNorm.toFixed(5) },
    'step240:', { modes: sEnd.modesCovered, shares: sEnd.modeShares.map((x) => +x.toFixed(2)), w1: +sEnd.w1.toFixed(3), gGrad: +sEnd.gGradNorm.toFixed(5) },
  );
}

console.log('\n=== LAB GRID (30 configs x 5 seeds) ===');
const rows = [];
for (const noiseDim of g.LAB_NOISE_DIMS) {
  for (const dSteps of g.LAB_D_STEPS) {
    for (const lrRatio of g.LAB_LR_RATIOS) {
      const res = g.evaluateLabSetting({ lrRatio, dSteps, noiseDim });
      rows.push({
        noiseDim,
        dSteps,
        lrRatio,
        meanModes: +res.meanModes.toFixed(2),
        allThreeSeeds: res.allThreeSeeds,
        meanW1: +res.meanW1.toFixed(3),
        meanOffMode: +res.meanOffMode.toFixed(3),
        updatesToCover3: res.updatesToCover3,
      });
    }
  }
}
for (const r of rows) {
  console.log(JSON.stringify(r));
}
