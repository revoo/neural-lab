// authored-by: Gemini Antigravity
import type { EmbedBriefId, RepKind, SkuMode } from './embed';

export function embedRealCode(
  briefId: EmbedBriefId,
  st: { rep: RepKind; b2Dim: number; b2Bytes: 1 | 2 | 4; sku: SkuMode },
): string {
  if (briefId === 'transfer') {
    if (st.rep === 'onehot') {
      return `x = F.one_hot(token_ids, num_classes=14).float()\nclf = nn.Linear(14, 1)`;
    }
    if (st.rep === 'scalar-id') {
      return `x = ((token_ids.float() - 6.5) / 4.0).unsqueeze(-1)\nclf = nn.Linear(1, 1)`;
    }
    const d = st.rep === 'dense-2' ? 2 : 4;
    return `x = pretrained_embeddings[token_ids]  # shape (N, ${d})\nclf = nn.Linear(${d}, 1)`;
  }

  if (briefId === 'edge-ram') {
    const dtype = st.b2Bytes === 4 ? 'torch.float32' : st.b2Bytes === 2 ? 'torch.float16' : 'torch.int8';
    return [
      `emb = nn.Embedding(num_embeddings=50_000, embedding_dim=${st.b2Dim})`,
      st.b2Bytes === 1
        ? `q_emb = torch.ao.quantization.quantize_dynamic(emb, {nn.Embedding}, dtype=torch.qint8)  # ${dtype}`
        : `emb = emb.to(dtype=${dtype})`,
    ].join('\n');
  }

  if (st.sku === 'onehot') {
    return [
      `x = F.one_hot(sku_ids, num_classes=100_000).float()`,
      `layer = nn.Linear(100_000, 128)  # 12,800,128 params`,
    ].join('\n');
  }
  const d = Number(st.sku.split('-')[1]);
  return [
    `layer = nn.Sequential(`,
    `    nn.Embedding(num_embeddings=100_000, embedding_dim=${d}),`,
    `    nn.Linear(${d}, 128),`,
    `)`,
  ].join('\n');
}
