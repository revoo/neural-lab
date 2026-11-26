// authored-by: Claude Opus 5.5
/**
 * The image set for "learning to see": a class-balanced subset of Fashion-MNIST (Zalando Research,
 * MIT licence), built by `fashion.run.ts` into `fashion.bin`. 28 × 28 grey images, 10 classes.
 */
export const TRAIN_PER_CLASS = 300;
export const TEST_PER_CLASS = 50;
export const CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];
export const SIZE = 28;

export type ImageSet = { train: Float32Array[]; trainY: Uint8Array; test: Float32Array[]; testY: Uint8Array };

/** Decode the bytes of fashion.bin: pixels scaled to [0, 1]. */
export function decodeFashion(bytes: Uint8Array): ImageSet {
  const nTr = TRAIN_PER_CLASS * 10;
  const nTe = TEST_PER_CLASS * 10;
  const px = SIZE * SIZE;
  const img = (i: number) => Float32Array.from(bytes.subarray(i * px, (i + 1) * px), (v) => v / 255);
  const lab = (nTr + nTe) * px;
  return {
    train: Array.from({ length: nTr }, (_, i) => img(i)),
    trainY: bytes.slice(lab, lab + nTr),
    test: Array.from({ length: nTe }, (_, i) => img(nTr + i)),
    testY: bytes.slice(lab + nTr, lab + nTr + nTe),
  };
}
