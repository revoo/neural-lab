// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { Cnn, cnnLayout, cnnShapes, type CnnConfig } from './cnn';
import { createRng } from '../../lib/rng';

const TINY: CnnConfig = { size: 12, c1: 3, k1: 3, c2: 2, k2: 3, hidden: 5, classes: 4 };

describe('shapes and layout', () => {
  it('the lesson network: 28×28 → 8@24×24 → 12×12 → 16@8×8 → 4×4 → 256 → 32 → 10', () => {
    const c: CnnConfig = { size: 28, c1: 8, k1: 5, c2: 16, k2: 5, hidden: 32, classes: 10 };
    expect(cnnShapes(c)).toEqual({ s1: 24, p1: 12, s2: 8, p2: 4, flat: 256 });
    expect(cnnLayout(c).total).toBe(8 * 25 + 8 + 16 * 8 * 25 + 16 + 32 * 256 + 32 + 10 * 32 + 10);
  });
});

describe('backward pass against central finite differences (float64)', () => {
  const net = new Cnn(TINY, 'f64');
  net.init(3);
  const r = createRng(9);
  for (let i = 0; i < net.p.length; i++) net.p[i] += r.uniform(-0.2, 0.2); // biases off the ReLU kinks
  const imgs = [0, 1, 2].map(() => Float64Array.from({ length: 144 }, () => r.next()));
  const labels = [0, 2, 3];
  const tr = net.newTrace();
  const lossOf = () => {
    let l = 0;
    for (let i = 0; i < 3; i++) l -= Math.log(net.forward(imgs[i], tr)[labels[i]]);
    return l / 3;
  };
  it('every parameter', () => {
    net.batchGrad(imgs, labels, tr);
    const g = Float64Array.from(net.g);
    const h = 1e-6;
    let worst = 0;
    for (let i = 0; i < net.p.length; i++) {
      const v = net.p[i];
      net.p[i] = v + h;
      const a = lossOf();
      net.p[i] = v - h;
      const b = lossOf();
      net.p[i] = v;
      const num = (a - b) / (2 * h);
      worst = Math.max(worst, Math.abs(num - g[i]) / Math.max(1e-4, Math.abs(num) + Math.abs(g[i])));
    }
    expect(worst).toBeLessThan(1e-5);
  });
  it('pixel gradients (the saliency map)', () => {
    const img = imgs[1];
    const pr = net.forward(img, tr);
    const k = 2;
    // d logit_k / d pixel: backward with a one-hot on the logit
    const dl = new Float64Array(4);
    dl[k] = 1;
    const dx = new Float64Array(144);
    net.g.fill(0);
    net.backward(tr, dl, dx);
    expect(pr.length).toBe(4);
    const h = 1e-6;
    let worst = 0;
    for (let i = 0; i < 144; i++) {
      const v = img[i];
      img[i] = v + h;
      net.forward(img, tr);
      const a = tr.logits[k];
      img[i] = v - h;
      net.forward(img, tr);
      const b = tr.logits[k];
      img[i] = v;
      const num = (a - b) / (2 * h);
      worst = Math.max(worst, Math.abs(num - dx[i]) / Math.max(1e-4, Math.abs(num) + Math.abs(dx[i])));
    }
    expect(worst).toBeLessThan(1e-5);
  });
});
