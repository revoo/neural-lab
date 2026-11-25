// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { createOptimizer } from '../../lib/optimizers';
import { torchOptimLine } from './realCode';

describe('torch.optim code line', () => {
  it('formats each optimizer', () => {
    expect(torchOptimLine({ name: 'sgd', lr: 0.00316 })).toBe('torch.optim.SGD(params, lr=0.00316)');
    expect(torchOptimLine({ name: 'momentum', lr: 0.01, beta1: 0.9 })).toBe('torch.optim.SGD(params, lr=0.01, momentum=0.9)');
    expect(torchOptimLine({ name: 'rmsprop', lr: 0.1, beta2: 0.99 })).toBe('torch.optim.RMSprop(params, lr=0.1, alpha=0.99, eps=1e-8)');
    expect(torchOptimLine({ name: 'adam', lr: 0.5, beta1: 0.95, beta2: 0.999 })).toBe('torch.optim.Adam(params, lr=0.5, betas=(0.95, 0.999))');
  });
  it('momentum matches torch: v = βv + g, p -= lr·v (two hand-computed steps)', () => {
    const opt = createOptimizer({ name: 'momentum', lr: 0.1, beta1: 0.9 });
    const p = [[1]];
    opt.step(p, [[2]]); // v = 2, p = 1 - 0.2 = 0.8
    opt.step(p, [[1]]); // v = 0.9·2 + 1 = 2.8, p = 0.8 - 0.28 = 0.52
    expect(p[0][0]).toBeCloseTo(0.52, 12);
  });
});
