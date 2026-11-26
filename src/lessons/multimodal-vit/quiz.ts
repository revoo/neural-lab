// authored-by: Claude Opus 5.5
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'vit.tokens',
    kind: 'choice',
    concepts: ['patch-embedding'],
    prompt: 'A 224 × 224 image is cut into 16 × 16 patches. How many tokens does the transformer see, and what happens to attention’s cost if you use 8 × 8 patches instead?',
    options: [
      { text: '196 tokens; with 8 × 8 patches, 784 tokens and about 16 times the attention cost.', correct: true, why: '(224/16)² = 14² = 196. Halving the patch side quadruples the tokens (28² = 784), and attention compares every pair, so its cost grows 4² = 16 times.' },
      { text: '196 tokens; with 8 × 8 patches, 392 tokens and twice the cost.', why: 'Halving the patch side halves it in both directions: 4 times the tokens, not 2.' },
      { text: '50,176 tokens: one per pixel.', why: 'That would be 1 × 1 patches. Patches exist precisely to avoid sequences that long.' },
    ],
  },
  {
    id: 'vit.shuffle',
    kind: 'choice',
    concepts: ['vision-transformer'],
    prompt: 'A vision transformer without position embeddings reads an image and, separately, the same image with its patches shuffled. What does it output?',
    options: [
      { text: 'Exactly the same thing: self-attention followed by averaging doesn’t depend on the order of the tokens.', correct: true, why: 'Attention treats its input as a set. Without position embeddings, a picture is a bag of patches, which is why ViTs add them (the original ViT used learned ones).' },
      { text: 'Something different, because attention compares neighbouring patches.', why: 'Attention has no notion of neighbours: every token is compared with every other in the same way.' },
      { text: 'An error, because the patches are no longer in order.', why: 'Nothing in the computation checks the order; that’s the problem.' },
    ],
  },
  {
    id: 'vit.data',
    kind: 'choice',
    concepts: ['vision-transformer'],
    prompt: 'With 16 training images, our small CNN averaged 81% and our small ViT 56%. With 1,024 images, 93% and 88%. What explains the pattern?',
    options: [
      { text: 'The CNN has translation built in (the same filter everywhere, then pooling); the ViT has to learn it from examples, which takes data.', correct: true, why: 'The ViT paper put it the same way: transformers lack the inductive biases of CNNs, such as translation equivariance and locality, and so need more data; at large scale, training trumps inductive bias.' },
      { text: 'The ViT is simply a worse architecture for images.', why: 'With enough data the gap closes, and at very large scale ViTs matched or beat the best CNNs.' },
      { text: 'The CNN has more weights.', why: 'It has far fewer: 45 against 369.' },
    ],
  },
  {
    id: 'vit.clip',
    kind: 'choice',
    concepts: ['clip'],
    prompt: 'After CLIP-style training, how does zero-shot classification work?',
    options: [
      { text: 'Write a caption for each class, encode them, and give each image the class whose caption is most similar to it.', correct: true, why: 'No classifier is trained on labels: the classes are sentences. That’s how CLIP matched a fully trained ResNet-50 on ImageNet without using its training images.' },
      { text: 'Train a new output layer for each new set of classes.', why: 'That’s ordinary fine-tuning. Zero-shot means no training on the new classes at all.' },
      { text: 'Generate a caption for the image and compare the words.', why: 'CLIP doesn’t generate text; it compares an image vector with caption vectors.' },
    ],
  },
];
