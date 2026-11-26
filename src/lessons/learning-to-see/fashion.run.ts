// authored-by: Claude Opus 5.5
/**
 * Builds `fashion.bin`, the small image set for "learning to see": a class-balanced subset of
 * Fashion-MNIST (Zalando Research, MIT licence: https://github.com/zalandoresearch/fashion-mnist).
 * The first TRAIN_PER_CLASS images of each class from the official training file and the first
 * TEST_PER_CLASS of each class from the official test file, in file order.
 *
 *   curl -LO https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{train,t10k}-{images-idx3,labels-idx1}-ubyte.gz
 *   gunzip *.gz
 *   node --experimental-strip-types --import ./scripts/ts-hooks.mjs src/lessons/learning-to-see/fashion.run.ts <dir>
 *
 * Layout: [train images][test images] as 28×28 bytes each, then [train labels][test labels] as bytes.
 */
import { argv, nodeFs } from '../../lib/nodeIo';
import { TEST_PER_CLASS, TRAIN_PER_CLASS } from './fashion';

const fs = await nodeFs();
const dir = argv()[0];
function pick(images: Uint8Array, labels: Uint8Array, perClass: number) {
  const n = (labels.length - 8);
  const count = new Array(10).fill(0);
  const imgs: Uint8Array[] = [];
  const labs: number[] = [];
  for (let i = 0; i < n && imgs.length < perClass * 10; i++) {
    const y = labels[8 + i];
    if (count[y] >= perClass) continue;
    count[y]++;
    imgs.push(images.subarray(16 + i * 784, 16 + (i + 1) * 784));
    labs.push(y);
  }
  return { imgs, labs };
}
const tr = pick(fs.readFileSync(`${dir}/train-images-idx3-ubyte`), fs.readFileSync(`${dir}/train-labels-idx1-ubyte`), TRAIN_PER_CLASS);
const te = pick(fs.readFileSync(`${dir}/t10k-images-idx3-ubyte`), fs.readFileSync(`${dir}/t10k-labels-idx1-ubyte`), TEST_PER_CLASS);
const all = [...tr.imgs, ...te.imgs];
const out = new Uint8Array(all.length * 785);
all.forEach((im, i) => out.set(im, i * 784));
out.set([...tr.labs, ...te.labs], all.length * 784);
fs.writeFileSync(new URL('./fashion.bin', import.meta.url), out);
console.log(`${tr.imgs.length} train + ${te.imgs.length} test images, ${out.length} bytes`);
