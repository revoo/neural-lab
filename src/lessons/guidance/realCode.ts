// authored-by: Gemini Antigravity

export function guidanceRealCode(steps: number, w: number): string {
  const passes = w === 0 || w === 1 ? steps : 2 * steps;
  const note =
    w === 0
      ? 'unconditional only (ignores prompt)'
      : w === 1
        ? 'plain conditioning (1 denoiser pass/step)'
        : `CFG w=${w.toFixed(2)} (2 denoiser passes/step = ${passes} calls)`;
  return [
    `# ${note}`,
    `image = pipe(prompt="blue object", num_inference_steps=${steps}, guidance_scale=${w.toFixed(2)}).images[0]`,
    `# Inside each DDIM step: eps = eps_uncond + ${w.toFixed(2)} * (eps_cond - eps_uncond)`,
  ].join('\n');
}
