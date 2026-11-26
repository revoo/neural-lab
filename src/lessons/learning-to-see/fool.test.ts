// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { nodeFs } from '../../lib/nodeIo';
import { Cnn } from './cnn';
import { CLASSES, decodeFashion } from './fashion';
import { apply, briefImages, lead, searchFool, topClass, type Change } from './fool';
import { FOOL_RESULTS } from './foolResults';
import { SEE_RUN } from './see';

async function setup() {
  const fs = await nodeFs();
  const d = decodeFashion(new Uint8Array(fs.readFileSync(new URL('./fashion.bin', import.meta.url))));
  const b = fs.readFileSync(new URL('./seeNet.bin', import.meta.url));
  const n = new Cnn(SEE_RUN.cfg);
  n.p.set(new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)));
  return { d, n };
}

describe('Fool the classifier: the recorded search', () => {
  it('brief photos follow the fixed rule (first correctly classified Trouser, Sneaker, Shirt)', async () => {
    const { d, n } = await setup();
    expect(briefImages(n, d)).toEqual(FOOL_RESULTS.map((r) => r.image));
    expect(FOOL_RESULTS.map((r) => [r.image, CLASSES[r.label], r.method, r.best.length])).toEqual([
      [2, 'Trouser', 'greedy', 21],
      [9, 'Sneaker', 'greedy', 7],
      [4, 'Shirt', 'pair', 2],
    ]);
    expect(FOOL_RESULTS.map((r) => [r.singles, r.pairsChecked])).toEqual([
      [1004, 503286],
      [1027, 526608],
      [1322, 39702],
    ]);
  });
  it('every recorded attack works and lands where the debrief says; confidence and lead as quoted', async () => {
    const { d, n } = await setup();
    const where = FOOL_RESULTS.map((r) => CLASSES[topClass(n, apply(d.test[r.image], r.best as unknown as Change[]))]);
    expect(where).toEqual(['Dress', 'Ankle boot', 'Pullover']);
    const conf = FOOL_RESULTS.map((r) => (100 * n.forward(d.test[r.image], n.newTrace())[r.label]).toFixed(1));
    expect(conf).toEqual(['100.0', '99.7', '63.2']);
    expect(FOOL_RESULTS.map((r) => lead(n, d.test[r.image], r.label).toFixed(2))).toEqual(['18.70', '6.46', '0.72']);
  });
  it('no single change fools it on any of the three photos (exhaustive), and the Shirt search reproduces', async () => {
    const { d, n } = await setup();
    const tr = n.newTrace();
    for (const r of FOOL_RESULTS) {
      const img = d.test[r.image];
      for (let px = 0; px < 784; px++)
        for (const to of [0, 1] as const) if (img[px] !== to) expect(lead(n, apply(img, [{ px, to }]), r.label, tr)).toBeGreaterThan(0);
    }
    const shirt = FOOL_RESULTS[2];
    const again = searchFool(n, d.test[shirt.image], shirt.label);
    expect(again.method).toBe('pair');
    expect(again.best).toEqual(shirt.best);
  }, 120_000);
});
