// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { RESCUE_ACTS, RESCUE_INITS, RESCUE_LEVELS, evaluateRescue, rescueHint } from './vanish';

describe('rescue-lab hints never name a locked control', () => {
  it('levels without lifelines never mention LayerNorm, residuals or skips', () => {
    for (const level of RESCUE_LEVELS) {
      const acts = level.lockedAct ? [level.lockedAct] : RESCUE_ACTS;
      for (const act of acts)
        for (const init of RESCUE_INITS)
          for (const useLayerNorm of level.allowLayerNorm ? [false, true] : [false])
            for (const useResidual of level.allowResidual ? [false, true] : [false]) {
              const recipe = { act, init, useLayerNorm, useResidual };
              const ev = evaluateRescue(level, recipe);
              if (ev.passes) continue;
              const hint = rescueHint(level, recipe, ev);
              if (!level.allowResidual) expect(hint).not.toMatch(/residual|skip/i);
              if (!level.allowLayerNorm) expect(hint).not.toMatch(/layernorm|layer norm|re-centre/i);
              if (level.lockedAct) expect(hint).not.toMatch(/which activation/i);
            }
    }
  });
});
