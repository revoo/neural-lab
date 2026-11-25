// authored-by: Gemini Antigravity
/** Live PyTorch-style training-loop code builder for the GAN lab. */
import { LAB_BASE_LR_D, type LabSetting } from './gan';

export function buildGanCode(s: LabSetting): string {
  const lrG = +(LAB_BASE_LR_D * s.lrRatio).toFixed(5);
  return [
    `opt_D = torch.optim.Adam(D.parameters(), lr=${LAB_BASE_LR_D}, betas=(0.5, 0.999))`,
    `opt_G = torch.optim.Adam(G.parameters(), lr=${lrG}, betas=(0.5, 0.999))  # ratio ${s.lrRatio}x`,
    `for _ in range(${s.dSteps}):  # D steps per G step`,
    `    z = torch.randn(batch, ${s.noiseDim}); d_loss = bce(D(real), 1) + bce(D(G(z).detach()), 0)`,
    `z = torch.randn(batch, ${s.noiseDim}); g_loss = bce(D(G(z)), 1)  # non-saturating -log D(G(z))`,
  ].join('\n');
}
