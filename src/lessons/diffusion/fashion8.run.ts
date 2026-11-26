// authored-by: Claude Opus 5.5
/**
 * Builds `fashion8.bin` for "learning to draw": Fashion-MNIST (Zalando Research, MIT licence,
 * https://github.com/zalandoresearch/fashion-mnist) trousers, sneakers and bags, shrunk to 8 × 8 by
 * averaging 3 × 3 blocks of the central 24 × 24 pixels. The first 500 of each class from the training
 * file (1,500 images to learn from), then the first 50 of each from the test file (150 held out).
 *   node --experimental-strip-types --import ./scripts/ts-hooks.mjs src/lessons/diffusion/fashion8.run.ts <dir with the unzipped idx files>
 * Layout: 1,650 images × 64 bytes.
 */
import { argv, nodeFs } from '../../lib/nodeIo';

const fs = await nodeFs();
const dir = argv()[0];
const KEEP = [1, 7, 8];
function pick(images: Uint8Array, labels: Uint8Array, per: number) {
  const out: Uint8Array[] = [];
  const cnt = [0, 0, 0];
  for (let i = 0; i < labels.length - 8 && out.length < per * 3; i++) {
    const k = KEEP.indexOf(labels[8 + i]);
    if (k < 0 || cnt[k] >= per) continue;
    cnt[k]++;
    const im = new Uint8Array(64);
    for (let r = 0; r < 8; r++)
      for (let c = 0; c < 8; c++) {
        let s = 0;
        for (let dy = 0; dy < 3; dy++) for (let dx = 0; dx < 3; dx++) s += images[16 + i * 784 + (2 + r * 3 + dy) * 28 + 2 + c * 3 + dx];
        im[r * 8 + c] = Math.round(s / 9);
      }
    out.push(im);
  }
  return out;
}
const tr = pick(fs.readFileSync(`${dir}/train-images-idx3-ubyte`), fs.readFileSync(`${dir}/train-labels-idx1-ubyte`), 500);
const te = pick(fs.readFileSync(`${dir}/t10k-images-idx3-ubyte`), fs.readFileSync(`${dir}/t10k-labels-idx1-ubyte`), 50);
const out = new Uint8Array((tr.length + te.length) * 64);
[...tr, ...te].forEach((im, i) => out.set(im, i * 64));
fs.writeFileSync(new URL('./fashion8.bin', import.meta.url), out);
console.log(tr.length, te.length, out.length);
