// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'conv.zero-sum-flat',
    kind: 'choice',
    concepts: ['convolution', 'kernel'],
    prompt: 'Why do edge-detection kernels like Sobel Gx and the Laplacian always have weights that sum to 0?',
    options: [
      {
        text: 'So that multiplying a constant flat patch of any brightness c produces c × (sum of weights) = 0 — only actual brightness changes fire.',
        correct: true,
        why: 'If the weights summed to 1 (like Box Blur or Sharpen), a bright flat wall would produce a strong positive response rather than zero.',
      },
      {
        text: 'Because PyTorch Conv2d rejects kernels whose weights do not sum to 0.',
        why: 'PyTorch allows any real-valued weights; summing to 0 is a geometric property of derivative / edge filters.',
      },
      {
        text: 'To keep the output shape identical to the input shape.',
        why: 'Output shape depends only on H, K, padding P, and stride S — never on the numeric values of the weights.',
      },
    ],
  },
  {
    id: 'conv.output-size',
    kind: 'numeric',
    concepts: ['padding-stride'],
    tolerance: 0.01,
    make: (rng) => {
      const h = [12, 16, 28, 32][rng.int(4)];
      const k = [3, 5][rng.int(2)];
      const pad = [0, 1, 2][rng.int(3)];
      const stride = [1, 2][rng.int(2)];
      const out = Math.floor((h + 2 * pad - k) / stride) + 1;
      return {
        prompt: `What is the output height of a 2D convolution on an input of height ${h} with kernel size K = ${k}, padding P = ${pad}, and stride S = ${stride}?`,
        answer: out,
        explain: `O = floor((H + 2P − K) / S) + 1 = floor((${h} + ${2 * pad} − ${k}) / ${stride}) + 1 = ${out}.`,
      };
    },
  },
  {
    id: 'conv.parameter-sharing',
    kind: 'choice',
    concepts: ['convolution', 'translation-equivariance'],
    prompt: 'A single 3×3 convolution mapping a 224×224×3 image to a 224×224×32 feature map uses 896 parameters instead of a dense layer’s 241.7 billion. What assumption makes that 270-million-fold saving work?',
    options: [
      {
        text: 'Locality and translation equivariance: an edge or corner detector useful in the top-left corner is equally useful in the bottom-right corner, so one shared 3×3 filter can scan the entire image.',
        correct: true,
        why: 'Weight sharing ties every pixel position to the exact same small kernel, slashing parameters and baking shift equivariance right into the architecture.',
      },
      {
        text: 'Convolution deletes 99.99% of the input pixels before computing anything.',
        why: 'With stride 1 and pad 1, every single input pixel contributes to the output feature map.',
      },
      {
        text: 'Convolution freezes all weights to Sobel filters so nothing needs to be learned.',
        why: 'In a CNN, every single weight inside the 3×3×C_in×C_out kernel tensor is learned by backpropagation!',
      },
    ],
  },
  {
    id: 'conv.valid-shrinkage',
    kind: 'choice',
    concepts: ['padding-stride'],
    prompt: 'If you stack five 3×3 valid convolutions (padding P = 0, stride S = 1) on a 12×12 image, what happens to the spatial size?',
    options: [
      {
        text: 'Each layer shaves 2 pixels off both height and width, shrinking 12×12 down to 2×2 — losing 97.2% of spatial positions.',
        correct: true,
        why: 'That is why deep CNNs (VGG, ResNet) use same-padding P = (K−1)/2 = 1 for 3×3 layers so depth does not collapse the map.',
      },
      {
        text: 'It stays 12×12 because stride is 1.',
        why: 'Without padding, a 3×3 kernel cannot centre on the outermost 1-pixel rim, losing 2 pixels per dimension on every layer.',
      },
      {
        text: 'It halves at each layer (12 → 6 → 3 → 1).',
        why: 'Halving comes from stride 2 or 2×2 pooling, not from valid padding.',
      },
    ],
  },
];
