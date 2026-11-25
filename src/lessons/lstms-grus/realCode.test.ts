// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { lstmLabRealCode } from './realCode';

describe('lstms-grus lab live PyTorch code', () => {
  it('emits accurate PyTorch gate equation, bias_ih_l0 slice, and cell constructor', () => {
    const c1 = lstmLabRealCode('gates', { b1F: 0.95, b1I: 0.05, b1O: 0.05, b2Cell: 'lstm', b2Bias: 2, b3Kind: 'gru', b3H: 16 });
    expect(c1).toContain('0.95 * c + 0.05 * torch.tanh(cand)');
    expect(c1).toContain('0.05 * torch.tanh(c)');

    const c2 = lstmLabRealCode('bias', { b1F: 0.5, b1I: 0.5, b1O: 0.5, b2Cell: 'lstm', b2Bias: 2, b3Kind: 'gru', b3H: 16 });
    expect(c2).toContain('bias_ih_l0[H:2*H].fill_(2.0)');

    const c3 = lstmLabRealCode('budget', { b1F: 0.5, b1I: 0.5, b1O: 0.5, b2Cell: 'lstm', b2Bias: 2, b3Kind: 'gru', b3H: 16 });
    expect(c3).toContain('nn.GRU(input_size=16, hidden_size=16');
    expect(c3).toContain('1,584 params');
  });
});
