// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'images.rec709-green',
    kind: 'choice',
    concepts: ['image-tensor', 'luminance'],
    prompt: 'Why does the Rec. 709 grayscale formula Y = 0.2126 R + 0.7152 G + 0.0722 B weight Green nearly 10× heavier than Blue?',
    options: [
      {
        text: 'Human cone cells are far more sensitive to green wavelengths than blue wavelengths, so pure green looks ~9.9× brighter than pure blue at the same numeric intensity (255).',
        correct: true,
        why: 'Naive (R+G+B)/3 maps pure red, green, and blue all to 85.0, flattening bright green foliage and dark blue shadows to the same grey.',
      },
      {
        text: 'Because green pixels take up 71.5% of the bytes in a uint8 tensor.',
        why: 'Every RGB channel takes the exact same 1 byte per pixel in uint8.',
      },
      {
        text: 'Because camera sensors only record green light.',
        why: 'Cameras do often use twice as many green photosites in a Bayer filter *because* of human vision, but they still record red and blue.',
      },
    ],
  },
  {
    id: 'images.shadow-hsv',
    kind: 'choice',
    concepts: ['colour-space'],
    prompt: 'A sunny red traffic light [235, 38, 38] enters deep shadow and dims by 50% to [118, 19, 19]. What happens to its Euclidean distance in RGB versus its Hue in HSV?',
    options: [
      {
        text: 'RGB Euclidean distance jumps by ~120 units (halfway across the cube), whereas HSV Hue shifts by 0° because all three channels scaled by the same factor.',
        correct: true,
        why: 'Hue and chromaticity (RGB / sum) depend only on the *ratios* between R, G, and B, making them invariant to uniform brightness scaling.',
      },
      {
        text: 'Both RGB distance and Hue change by 50%.',
        why: 'Only the V (value/brightness) coordinate in HSV drops by 50%; H (hue) and S (saturation) stay unchanged.',
      },
      {
        text: 'RGB distance stays 0 because the pixel is still red.',
        why: 'Euclidean distance in RGB treats [235, 38, 38] and [118, 19, 19] as 120 units apart — closer to dark grey than to daylight red!',
      },
    ],
  },
  {
    id: 'images.aliasing',
    kind: 'choice',
    concepts: ['aliasing'],
    prompt: 'You shrink a 16×16 1-pixel black-and-white checkerboard to 8×8 using NumPy slicing img[::2, ::2]. What comes out?',
    options: [
      {
        text: 'A completely solid white (255) square — 100% of the black pixels were skipped because the stride matched the pattern period.',
        correct: true,
        why: 'That is spatial aliasing: subsampling faster than the Nyquist limit folds high-frequency detail into fake low-frequency blocks. Always box-average or low-pass filter before downsampling.',
      },
      {
        text: 'An 8×8 checkerboard with half as many squares.',
        why: 'Since every even row + even column has (r + c) even, img[::2, ::2] hits exclusively white cells!',
      },
      {
        text: 'Uniform 50% grey (127.5).',
        why: 'Uniform 127.5 is what 2×2 *box averaging* gives; strided slicing never averages neighbours.',
      },
    ],
  },
  {
    id: 'images.batch-vram',
    kind: 'numeric',
    concepts: ['image-tensor'],
    tolerance: 0.05,
    make: (rng) => {
      const batch = [16, 32, 64, 128][rng.int(4)];
      const bytesPerElem = [2, 4][rng.int(2)];
      const dtype = bytesPerElem === 2 ? 'FP16 (2 bytes)' : 'FP32 (4 bytes)';
      const mib = Number(((batch * 3 * 224 * 224 * bytesPerElem) / (1024 * 1024)).toFixed(3));
      return {
        prompt: `How many MiB (1 MiB = 1,048,576 bytes) does a PyTorch image batch of shape (${batch}, 3, 224, 224) occupy in ${dtype}?`,
        answer: mib,
        explain: `${batch} × 3 × 224 × 224 × ${bytesPerElem} bytes = ${(batch * 3 * 224 * 224 * bytesPerElem).toLocaleString()} bytes ÷ 1,048,576 = ${mib.toFixed(3)} MiB.`,
      };
    },
  },
];
