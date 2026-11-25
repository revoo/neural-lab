// authored-by: Claude Opus 5.5
import type { Config } from './optim';

const num = (x: number) => String(Number(x.toPrecision(4)));

/**
 * The PyTorch line for an optimizer-golf configuration. The lesson's update
 * rules match torch.optim's: SGD momentum accumulates v = βv + g, RMSprop's
 * `alpha` is the lesson's β₂, and both RMSprop and Adam use eps = 1e-8.
 */
export function torchOptimLine(cfg: Config): string {
  const lr = num(cfg.lr);
  switch (cfg.name) {
    case 'sgd':
      return `torch.optim.SGD(params, lr=${lr})`;
    case 'momentum':
      return `torch.optim.SGD(params, lr=${lr}, momentum=${num(cfg.beta1 ?? 0.9)})`;
    case 'rmsprop':
      return `torch.optim.RMSprop(params, lr=${lr}, alpha=${num(cfg.beta2 ?? 0.9)}, eps=1e-8)`;
    case 'adam':
      return `torch.optim.Adam(params, lr=${lr}, betas=(${num(cfg.beta1 ?? 0.9)}, ${num(cfg.beta2 ?? 0.999)}))`;
    default:
      return `torch.optim.AdamW(params, lr=${lr})`;
  }
}
