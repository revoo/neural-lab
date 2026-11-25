// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { facePrinterRealCode } from './realCode';

describe('autoencoders lab live PyTorch code', () => {
  it('emits plain AE when beta=0 and beta-VAE with 2*latent head otherwise', () => {
    const ae = facePrinterRealCode({ latent: 2, hidden: 16, beta: 0 });
    expect(ae).toContain('Plain Autoencoder');
    expect(ae).toContain('nn.Linear(16, 2)');
    expect(ae).not.toContain('logvar');

    const vae = facePrinterRealCode({ latent: 2, hidden: 16, beta: 0.3 });
    expect(vae).toContain('nn.Linear(16, 4)');
    expect(vae).toContain('loss = recon + 0.3 *');
    expect(vae).toContain('4,884 params');
  });
});
