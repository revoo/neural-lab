// authored-by: Gemini Antigravity
import { LINEAR_ABAR } from './diffusion';

/** Nearest 1000-step DDPM training timestep t whose signal share ᾱ_t is closest to `a`. */
function nearestTimestep(a: number): number {
  let bestT = 0;
  let bestErr = Infinity;
  for (let t = 0; t < LINEAR_ABAR.length; t++) {
    const err = Math.abs(LINEAR_ABAR[t] - a);
    if (err < bestErr) {
      bestErr = err;
      bestT = t;
    }
  }
  return bestT;
}

export function stepBudgetRealCode(levels: number[], eta: number): string {
  const sorted = [...levels].sort((a, b) => a - b);
  const steps = sorted.length + 1;
  const ts = [1000, ...sorted.map(nearestTimestep)];
  const cls = eta === 0 ? 'DDIMScheduler' : 'DDPMScheduler';
  return [
    `# ${steps} denoiser calls (${eta === 0 ? 'DDIM eta=0.0' : 'DDPM eta=1.0'}): alpha_bar = [0, ${sorted.join(', ')}, 1]`,
    `scheduler = diffusers.${cls}.from_pretrained("google/ddpm-cifar10-32")`,
    `scheduler.set_timesteps(timesteps=[${ts.join(', ')}])  # custom reverse stops`,
    `for t in scheduler.timesteps:`,
    `    x = scheduler.step(unet(x, t).sample, t, x${eta === 0 ? ', eta=0.0' : ''}).prev_sample`,
  ].join('\n');
}
