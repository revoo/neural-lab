// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { stepBudgetRealCode } from './realCode';

describe('diffusion lab live diffusers code', () => {
  it('emits DDIMScheduler / DDPMScheduler with custom descending timesteps matching stops', () => {
    const ddim = stepBudgetRealCode([0.3, 0.8, 0.98], 0);
    expect(ddim).toContain('DDIMScheduler');
    expect(ddim).toContain('4 denoiser calls');
    expect(ddim).toContain('eta=0.0');
    expect(ddim).toContain('timesteps=[1000,');

    const ddpm = stepBudgetRealCode([0.3, 0.6], 1);
    expect(ddpm).toContain('DDPMScheduler');
    expect(ddpm).toContain('3 denoiser calls');
    expect(ddpm).not.toContain('eta=0.0');
  });
});
