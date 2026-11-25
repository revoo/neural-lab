// authored-by: Gemini Antigravity
import { BATCH_OPTIONS, type BatchOption, type ContrastBriefId } from './contrastive';

export function contrastiveRealCode(
  briefId: ContrastBriefId,
  st: { b1Negs: boolean; b1Tau: number; b2Tau: number; b3Opt: BatchOption },
): string {
  if (briefId === 'circle') {
    if (!st.b1Negs) {
      return [
        `# Alignment-only (NO negatives): collapses every pair onto one point`,
        `loss = ((F.normalize(img, dim=-1) - F.normalize(txt, dim=-1)) ** 2).sum(-1).mean()`,
      ].join('\n');
    }
    return [
      `logits = (F.normalize(img, dim=-1) @ F.normalize(txt, dim=-1).T) / ${st.b1Tau}`,
      `labels = torch.arange(len(img), device=logits.device)`,
      `loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))`,
    ].join('\n');
  }

  if (briefId === 'temp') {
    return [
      `logit_scale = ${(1 / st.b2Tau).toFixed(1)}  # 1 / tau  (CLIP clamps exp(logit_scale) <= 100)`,
      `logits = (img_emb @ txt_emb.T) * logit_scale`,
      `loss = F.cross_entropy(logits, labels)`,
    ].join('\n');
  }

  const opt = BATCH_OPTIONS.find((o) => o.value === st.b3Opt)!;
  if (opt.value.startsWith('simclr-')) {
    return [
      `# SimCLR in-batch InfoNCE: B=${opt.activeB} active encoder graphs (${opt.vramGiB.toFixed(2)} GiB VRAM)`,
      `logits = (z_a @ z_b.T) / tau   # ${opt.totalNegs} in-batch negatives per sample`,
      `loss = F.cross_entropy(logits, torch.arange(${opt.activeB}))`,
    ].join('\n');
  }
  return [
    `# MoCo FIFO queue: B=${opt.activeB} active + Q=${opt.totalNegs} detached momentum keys (${opt.vramGiB.toFixed(2)} GiB)`,
    `l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)`,
    `l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])  # shape (${opt.activeB}, ${opt.totalNegs})`,
    `loss = F.cross_entropy(torch.cat([l_pos, l_neg], dim=1) / tau, torch.zeros(${opt.activeB}, dtype=torch.long))`,
  ].join('\n');
}
