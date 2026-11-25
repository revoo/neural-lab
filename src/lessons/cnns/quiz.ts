// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'cnns.vgg-stack',
    kind: 'choice',
    concepts: ['receptive-field', 'cnn'],
    prompt: 'Why did VGGNet and ResNet replace 5×5 and 7×7 convolutional layers with stacks of two or three 3×3 layers?',
    options: [
      {
        text: 'Two 3×3 layers reach the exact same 5×5 receptive field with 28% fewer parameters (and three 3×3s reach 7×7 with 44.9% fewer), while adding extra ReLU nonlinearities in between.',
        correct: true,
        why: 'Stacking small kernels factors a big spatial patch into cheaper, more expressive multi-step transformations.',
      },
      {
        text: 'Because 5×5 kernels cannot compute edge detection.',
        why: '5×5 kernels are strictly more general linearly, just far heavier in parameters and with only one ReLU.',
      },
      {
        text: 'Because two 3×3 layers double the spatial resolution of the feature map.',
        why: 'Same-padded 3×3 stride-1 convolutions keep spatial resolution unchanged.',
      },
    ],
  },
  {
    id: 'cnns.rf-calc',
    kind: 'numeric',
    concepts: ['receptive-field'],
    tolerance: 0.01,
    make: (rng) => {
      const k1 = 3;
      const k2 = [3, 5][rng.int(2)];
      // Conv(k1, s=1) -> MaxPool(2, s=2) -> Conv(k2, s=1)
      const r1 = 1 + (k1 - 1) * 1;
      const r2 = r1 + (2 - 1) * 1;
      const r3 = r2 + (k2 - 1) * 2;
      return {
        prompt: `Start at receptive field r = 1, jump j = 1. Apply Conv ${k1}×${k1} (stride 1), then MaxPool 2×2 (stride 2), then Conv ${k2}×${k2} (stride 1). What is the final receptive field width?`,
        answer: r3,
        explain: `After Conv ${k1}×${k1}: r = ${r1}, j = 1. After MaxPool 2×2: r = ${r1} + 1×1 = ${r2}, j = 2. After Conv ${k2}×${k2}: r = ${r2} + (${k2}−1)×2 = ${r3}.`,
      };
    },
  },
  {
    id: 'cnns.shift-generalisation',
    kind: 'choice',
    concepts: ['cnn', 'pooling'],
    prompt: 'Trained ONLY on shapes in the top-left corner of an 8×8 grid, our 547-parameter Dense MLP collapsed to 28.3% on shapes moved to the bottom-right, whereas our 55-parameter CNN scored 100.0%. Why?',
    options: [
      {
        text: 'The Dense MLP had zero gradient signal on bottom-right input weights during training, whereas the CNN applies the exact same 3×3 filters everywhere and aggregates them with GlobalMaxPool.',
        correct: true,
        why: 'Weight sharing makes feature extraction equivariant; global pooling makes the final class vector completely invariant to where the shape sits.',
      },
      {
        text: 'Because 547 parameters was too small to fit the training data.',
        why: 'The Dense MLP hit 100.0% on both training and top-left test images — it memorised positions rather than shapes.',
      },
      {
        text: 'Because GlobalMaxPool randomly shifts training images.',
        why: 'GlobalMaxPool doesn’t augment data; it takes the maximum response over the whole spatial map regardless of (r, c).',
      },
    ],
  },
  {
    id: 'cnns.gap-head',
    kind: 'choice',
    concepts: ['global-average-pooling'],
    prompt: 'Replacing VGG’s Flatten → 4096 → 1000 head with Global Average Pooling → 1000 on a 7×7×512 feature map cuts head parameters from 106.9 million down to 0.513 million (208× smaller). What does GAP compute?',
    options: [
      {
        text: 'The spatial mean across all 7×7 = 49 positions for each of the 512 channels independently, turning (7, 7, 512) into one 512-element vector.',
        correct: true,
        why: 'That removes every spatial index from the classifier head, slashes parameters by >200×, and even lets the CNN accept arbitrary input resolutions!',
      },
      {
        text: 'The average across the 512 channels at each pixel, turning (7, 7, 512) into a 7×7 grayscale image.',
        why: 'Averaging across channels would throw away all 512 learned semantic detectors (eyes, wheels, fur); GAP averages across *space* (H, W) per channel.',
      },
      {
        text: 'It drops 49 of every 50 filters before classification.',
        why: 'All 512 channels are preserved intact.',
      },
    ],
  },
];
