/**
 * A small neural network fitted to the same noisy points as the polynomials, trained with full-batch
 * Adam so the learner can watch it pass through underfitting → a good fit → overfitting.
 */
import { MLP } from '../../lib/mlp';
import { createOptimizer, type Optimizer } from '../../lib/optimizers';
import { createRng } from '../../lib/rng';
import { TEST_SET, type Sample } from './fit';

export const NET_SIZES = [1, 32, 32, 1];
export const NET_LR = 0.01;

export class CurveNet {
  readonly net: MLP;
  private opt: Optimizer;
  private X: number[][];
  private Y: number[][];
  steps = 0;

  constructor(s: Sample, seed = 1, sizes = NET_SIZES, lr = NET_LR) {
    this.net = new MLP(sizes, [...sizes.slice(1, -1).map(() => 'tanh' as const), 'linear'], createRng(seed), 'xavier');
    this.opt = createOptimizer({ name: 'adam', lr });
    this.X = s.x.map((x) => [x]);
    this.Y = s.y.map((y) => [y]);
  }

  step(k = 1) {
    for (let i = 0; i < k; i++) {
      this.net.backprop(this.X, this.Y, 'mse');
      this.opt.step(this.net.params(), this.net.grads());
      this.steps++;
    }
  }

  predict(x: number) {
    return this.net.predict([x])[0];
  }

  private mseOn(xs: number[], ys: number[]) {
    let t = 0;
    for (let i = 0; i < xs.length; i++) t += (this.predict(xs[i]) - ys[i]) ** 2;
    return t / xs.length;
  }

  trainError() {
    return this.mseOn(
      this.X.map((r) => r[0]),
      this.Y.map((r) => r[0]),
    );
  }

  testError() {
    return this.mseOn(TEST_SET.x, TEST_SET.y);
  }
}

/** Train for `steps`, recording train and test error every `every` steps (step 0 included). */
export function history(s: Sample, steps: number, every: number, seed = 1) {
  const c = new CurveNet(s, seed);
  const h = [{ step: 0, train: c.trainError(), test: c.testError() }];
  for (let t = every; t <= steps; t += every) {
    c.step(every);
    h.push({ step: t, train: c.trainError(), test: c.testError() });
  }
  return h;
}
