// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { guidanceRealCode } from './realCode';

describe('guidance lab live diffusers code', () => {
  it('emits matching diffusers pipe call and CFG formula with exact pass counts', () => {
    const plain = guidanceRealCode(5, 1.0);
    expect(plain).toContain('num_inference_steps=5');
    expect(plain).toContain('guidance_scale=1.00');
    expect(plain).toContain('1 denoiser pass/step');

    const cfg = guidanceRealCode(2, 3.0);
    expect(cfg).toContain('num_inference_steps=2');
    expect(cfg).toContain('guidance_scale=3.00');
    expect(cfg).toContain('4 calls');
  });
});
