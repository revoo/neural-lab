// authored-by: Gemini Antigravity
import { cellParamCount, type CellKind, type LstmBriefId } from './gated';

export function lstmLabRealCode(
  brief: LstmBriefId,
  state: {
    b1F: number;
    b1I: number;
    b1O: number;
    b2Cell: 'rnn' | 'lstm';
    b2Bias: number;
    b3Kind: CellKind;
    b3H: number;
  },
): string {
  if (brief === 'gates') {
    return [
      `# One corridor step of an LSTM cell (f=${state.b1F.toFixed(2)}, i=${state.b1I.toFixed(2)}, o=${state.b1O.toFixed(2)})`,
      `c = ${state.b1F.toFixed(2)} * c + ${state.b1I.toFixed(2)} * torch.tanh(cand)   # additive cell highway`,
      `h = ${state.b1O.toFixed(2)} * torch.tanh(c)                  # shielded working output`,
    ].join('\n');
  }
  if (brief === 'bias') {
    if (state.b2Cell === 'rnn') {
      return [
        `# Vanilla nn.RNN has no forget gate (bias b_f=${state.b2Bias} does not open a cell highway)`,
        `rnn = nn.RNN(input_size=1, hidden_size=32, nonlinearity="tanh")`,
      ].join('\n');
    }
    return [
      `# PyTorch LSTM gate order is (i, f, g, o); slice [H:2*H] sets forget-gate bias`,
      `lstm = nn.LSTM(input_size=1, hidden_size=H)`,
      `with torch.no_grad():`,
      `    lstm.bias_ih_l0[H:2*H].fill_(${state.b2Bias}.0)  # sigma(${state.b2Bias}) = ${(1 / (1 + Math.exp(-state.b2Bias))).toFixed(3)}`,
    ].join('\n');
  }
  const cls = state.b3Kind === 'rnn' ? 'nn.RNN' : state.b3Kind === 'gru' ? 'nn.GRU' : 'nn.LSTM';
  const mult = state.b3Kind === 'rnn' ? 1 : state.b3Kind === 'gru' ? 3 : 4;
  const params = cellParamCount(state.b3Kind, 16, state.b3H);
  return [
    `cell = ${cls}(input_size=16, hidden_size=${state.b3H}, batch_first=True)`,
    `# ${mult} gate(s) * ${state.b3H} * (16 + ${state.b3H} + 1) = ${params.toLocaleString()} params (cap: 1,800)`,
  ].join('\n');
}
