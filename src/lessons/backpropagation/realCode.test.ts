// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { brief1PyTorchCode, brief2PyTorchCode, brief3PyTorchCode } from './realCode';

describe('Credit-Assignment Workbench realCode: PyTorch LiveCode snippets', () => {
  it('Brief 1 snippet uses standard torch autograd and matches exact leaf gradients', () => {
    const code = brief1PyTorchCode();
    expect(code).toContain('requires_grad=True');
    expect(code).toContain('F.relu(');
    expect(code).toContain('loss.backward()');
    expect(code).toContain('-3.0');
    expect(code).toContain('+3.2');
    expect(code).toContain('-1.6');
  });

  it('Brief 2 snippet maps initMode, activation, and lr to exact nn.init & SGD calls', () => {
    const broken = brief2PyTorchCode({ initMode: 'broken', act: 'tanh', lr: 1.0 });
    expect(broken).toContain('nn.Tanh()');
    expect(broken).toContain('nn.init.constant_(fc1.weight, 3.5)');
    expect(broken).toContain('lr=1');

    const zeros = brief2PyTorchCode({ initMode: 'zeros', act: 'sigmoid', lr: 0.5 });
    expect(zeros).toContain('nn.Sigmoid()');
    expect(zeros).toContain('nn.init.zeros_(fc1.weight)');

    const good = brief2PyTorchCode({ initMode: 'asym-small', act: 'relu', lr: 2.0 });
    expect(good).toContain('nn.ReLU()');
    expect(good).toContain('nn.init.normal_(fc1.weight, std=0.8)');
    expect(good).toContain('lr=2');
  });

  it('Brief 3 snippet maps frozen/unfrozen masks to requires_grad or register_hook', () => {
    const frozen = brief3PyTorchCode({ trainH1: false, trainH2: false, trainOut: true, lr: 1.5 });
    expect(frozen).toContain('fc1.parameters(): p.requires_grad = False');
    expect(frozen).toContain('fc2.parameters(): p.requires_grad = True');

    const partial = brief3PyTorchCode({ trainH1: false, trainH2: true, trainOut: true, lr: 2.5 });
    expect(partial).toContain('mask = torch.tensor([0.0, 1.0])[:, None]');
    expect(partial).toContain('fc1.weight.register_hook');
    expect(partial).toContain('lr=2.5');
  });
});
