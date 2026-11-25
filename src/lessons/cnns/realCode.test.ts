// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { cnnRealCode } from './realCode';

describe('cnns: real PyTorch code generator', () => {
  it('emits matching PyTorch nn.Sequential definitions for all three briefs', () => {
    const base = { k: 3 as const, nConv: 3 as const, nPools: 2 as const, shiftChoice: 'cnn-4' as const, headChoice: 'gap-linear' as const };

    const c1 = cnnRealCode('rf-budget', base);
    expect(c1).toContain('nn.Conv2d(16, 16, kernel_size=3, padding=1)');
    expect((c1.match(/nn\.MaxPool2d\(kernel_size=2, stride=2\)/g) ?? []).length).toBe(2);

    const c2Dense = cnnRealCode('shift-gen', { ...base, shiftChoice: 'dense-8' });
    expect(c2Dense).toContain('nn.Linear(64, 8)');
    const c2Cnn = cnnRealCode('shift-gen', base);
    expect(c2Cnn).toContain('nn.Conv2d(1, 4, kernel_size=3)');
    expect(c2Cnn).toContain('nn.AdaptiveMaxPool2d(1)');

    const c3Vgg = cnnRealCode('gap-head', { ...base, headChoice: 'vgg-fc' });
    expect(c3Vgg).toContain('nn.Linear(7 * 7 * 512, 4096)');
    const c3Gap = cnnRealCode('gap-head', base);
    expect(c3Gap).toContain('nn.AdaptiveAvgPool2d(1)');
    expect(c3Gap).toContain('nn.Linear(512, 1000)');
  });
});
