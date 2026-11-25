// authored-by: Gemini Antigravity
import type { ScoreKind, SeqArchId, SeqBriefId } from './seq2seq';

export function seqLabRealCode(
  brief: SeqBriefId,
  state: {
    b1Arch: 'fixed' | 'cross-attn';
    b1D: number;
    b2Kind: ScoreKind;
    b2Tau: number;
    b3Arch: SeqArchId;
  },
): string {
  if (brief === 'bottleneck') {
    if (state.b1Arch === 'fixed') {
      return [
        `# Fixed-vector bottleneck: drop h_1..h_{L-1}, pass only final h_L (shape B, ${state.b1D})`,
        `_, (h_n, c_n) = encoder_lstm(src)   # state width d = ${state.b1D}`,
        `out, _ = decoder_lstm(tgt, (h_n, c_n))`,
      ].join('\n');
    }
    return [
      `# Cross-attention bridge: keep all L=20 encoder states H (shape B, 20, ${state.b1D})`,
      `H, (h_n, c_n) = encoder_lstm(src)   # 20 * ${state.b1D} = ${20 * state.b1D} floats`,
      `ctx, attn = nn.MultiheadAttention(${state.b1D}, num_heads=1, batch_first=True)(dec_q, H, H)`,
    ].join('\n');
  }
  if (brief === 'align') {
    if (state.b2Kind === 'luong') {
      return [
        `# Luong multiplicative dot-product attention (tau = ${state.b2Tau})`,
        `logits = (dec_states @ enc_states.transpose(-1, -2)) / ${state.b2Tau}`,
        `weights = F.softmax(logits, dim=-1)   # ctx = weights @ enc_states`,
      ].join('\n');
    }
    return [
      `# Bahdanau additive attention (tau = ${state.b2Tau})`,
      `energy = v(torch.tanh(W_q(dec_q).unsqueeze(2) + W_k(enc_states).unsqueeze(1))).squeeze(-1)`,
      `weights = F.softmax(energy / ${state.b2Tau}, dim=-1)`,
    ].join('\n');
  }
  if (state.b3Arch === 'fixed-lstm') {
    return [
      `# Fixed-Vector LSTM (L=512 -> 1,024 sequential steps, path=512, ~81.9 ms)`,
      `enc = nn.LSTM(d_model, d_model, batch_first=True)`,
      `dec = nn.LSTM(d_model, d_model, batch_first=True)`,
    ].join('\n');
  }
  if (state.b3Arch === 'lstm-attn') {
    return [
      `# LSTM + Cross-Attention (O(1) path, still 1,024 serial LSTM steps -> ~85.1 ms)`,
      `H, state = encoder_lstm(src)   # sequential over L=512`,
      `ctx = F.scaled_dot_product_attention(dec_h, H, H)`,
    ].join('\n');
  }
  return [
    `# Transformer (all L=512 tokens processed in parallel: depth=1, path=1, ~7.75 ms)`,
    `layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)`,
    `out = nn.TransformerEncoder(layer, num_layers=6)(src_emb)`,
  ].join('\n');
}
