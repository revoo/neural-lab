/**
 * Optimizers operating on lists of parameter arrays (updated in place).
 * The same code drives the 2D loss-landscape demos and real network training.
 */
export type OptimizerName = 'sgd' | 'momentum' | 'rmsprop' | 'adam' | 'adamw';

export type OptimizerConfig = {
  name: OptimizerName;
  lr: number;
  beta1?: number; // momentum / Adam first moment
  beta2?: number; // RMSProp / Adam second moment
  eps?: number;
  weightDecay?: number; // AdamW decoupled decay
};

export type Optimizer = {
  config: OptimizerConfig;
  step(params: number[][], grads: number[][]): void;
  reset(): void;
};

export function createOptimizer(config: OptimizerConfig): Optimizer {
  const { name, lr } = config;
  const b1 = config.beta1 ?? (name === 'momentum' ? 0.9 : 0.9);
  const b2 = config.beta2 ?? (name === 'rmsprop' ? 0.9 : 0.999);
  const eps = config.eps ?? 1e-8;
  const wd = config.weightDecay ?? 0;
  let m: number[][] = [];
  let v: number[][] = [];
  let t = 0;
  const ensure = (params: number[][]) => {
    if (m.length !== params.length) {
      m = params.map((p) => p.map(() => 0));
      v = params.map((p) => p.map(() => 0));
      t = 0;
    }
  };
  return {
    config,
    reset() {
      m = [];
      v = [];
      t = 0;
    },
    step(params, grads) {
      ensure(params);
      t++;
      for (let i = 0; i < params.length; i++) {
        const p = params[i];
        const g = grads[i];
        const mi = m[i];
        const vi = v[i];
        for (let j = 0; j < p.length; j++) {
          const gj = g[j];
          switch (name) {
            case 'sgd':
              p[j] -= lr * gj;
              break;
            case 'momentum':
              // velocity accumulates past gradients; the step follows the velocity
              mi[j] = b1 * mi[j] + gj;
              p[j] -= lr * mi[j];
              break;
            case 'rmsprop':
              vi[j] = b2 * vi[j] + (1 - b2) * gj * gj;
              p[j] -= (lr * gj) / (Math.sqrt(vi[j]) + eps);
              break;
            case 'adam':
            case 'adamw': {
              mi[j] = b1 * mi[j] + (1 - b1) * gj;
              vi[j] = b2 * vi[j] + (1 - b2) * gj * gj;
              const mh = mi[j] / (1 - b1 ** t);
              const vh = vi[j] / (1 - b2 ** t);
              if (name === 'adamw' && wd) p[j] -= lr * wd * p[j];
              p[j] -= (lr * mh) / (Math.sqrt(vh) + eps);
              break;
            }
          }
        }
      }
    },
  };
}
