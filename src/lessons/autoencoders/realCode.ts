// authored-by: Gemini Antigravity
import { paramCount, type Design } from './lab';

export function facePrinterRealCode(d: Design): string {
  const params = paramCount(d).toLocaleString('en-GB');
  if (d.beta === 0) {
    return [
      `# Plain Autoencoder: latent=${d.latent}, hidden=${d.hidden} (${params} params, no KL pull)`,
      `enc = nn.Sequential(nn.Linear(144, ${d.hidden}), nn.Tanh(), nn.Linear(${d.hidden}, ${d.latent}))`,
      `dec = nn.Sequential(nn.Linear(${d.latent}, ${d.hidden}), nn.Tanh(), nn.Linear(${d.hidden}, 144))`,
      `loss = F.binary_cross_entropy_with_logits(dec(enc(x)), x, reduction="sum") / len(x)`,
    ].join('\n');
  }
  return [
    `# beta-VAE: latent=${d.latent}, hidden=${d.hidden}, beta=${d.beta} (${params} params)`,
    `enc = nn.Sequential(nn.Linear(144, ${d.hidden}), nn.Tanh(), nn.Linear(${d.hidden}, ${2 * d.latent}))  # mu, logvar`,
    `dec = nn.Sequential(nn.Linear(${d.latent}, ${d.hidden}), nn.Tanh(), nn.Linear(${d.hidden}, 144))`,
    `loss = recon + ${d.beta} * 0.5 * (mu**2 + logvar.exp() - 1 - logvar).sum() / len(x)`,
  ].join('\n');
}
