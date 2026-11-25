// authored-by: Gemini Antigravity
import type { RnnBriefId } from './rnn';

export function rnnLabRealCode(
  brief: RnnBriefId,
  state: {
    b1Act: 'tanh' | 'linear';
    b1Rho: number;
    b2Lr: number;
    b2Clip: number | null;
    b3K: number;
    b3H: number;
  },
): string {
  if (brief === 'flow') {
    const nonlin = state.b1Act === 'tanh' ? 'tanh' : 'linear (custom cell)';
    return [
      `# Orthogonal recurrent init with exact spectral radius rho = ${state.b1Rho} (${nonlin})`,
      `rnn = nn.RNN(input_size=1, hidden_size=2, nonlinearity="${state.b1Act === 'tanh' ? 'tanh' : 'relu'}")`,
      `nn.init.orthogonal_(rnn.weight_hh_l0, gain=${state.b1Rho})`,
    ].join('\n');
  }
  if (brief === 'clip') {
    if (state.b2Clip === null) {
      return [
        `# No gradient clipping — raw cliff slope hits 93.9`,
        `optimizer = torch.optim.SGD(model.parameters(), lr=${state.b2Lr})`,
        `loss.backward(); optimizer.step()`,
      ].join('\n');
    }
    return [
      `optimizer = torch.optim.SGD(model.parameters(), lr=${state.b2Lr})`,
      `loss.backward()`,
      `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=${state.b2Clip})`,
      `optimizer.step()`,
    ].join('\n');
  }
  return [
    `# Truncated BPTT: chunk length K = ${state.b3K}, hidden width H = ${state.b3H} (${state.b3K * state.b3H} slots)`,
    `rnn = nn.RNN(input_size=1, hidden_size=${state.b3H}, batch_first=True)`,
    `for chunk in x.split(${state.b3K}, dim=1):`,
    `    h = h.detach()  # cuts backward graph after ${state.b3K} steps`,
    `    out, h = rnn(chunk, h)`,
  ].join('\n');
}
