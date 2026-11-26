// authored-by: Claude Opus 5.5
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'tiny-gpt.first-thing',
    kind: 'choice',
    concepts: ['gpt', 'language-model'],
    prompt: 'A character-level GPT starts from random weights. Which does it learn first, and why?',
    options: [
      {
        text: 'How often each character appears, because getting the letter mix right removes a large share of the loss for very little knowledge.',
        correct: true,
        why: 'Letter frequencies alone bring the loss from ln 68 ≈ 4.22 to about 2.99 nats per character. The gradient points at the cheapest improvement first.',
      },
      {
        text: 'Whole common words like “the”, because they are the most frequent units of meaning.',
        why: 'Words only emerge after the model knows which letters are common and where words end; in the recorded run half the words were real only by step 1,668.',
      },
      {
        text: 'Grammar, because the transformer’s attention is designed for sentence structure.',
        why: 'Attention is a mechanism for gathering information from earlier positions, not a built-in grammar. Our small model never reached even six real words in a row.',
      },
    ],
  },
  {
    id: 'tiny-gpt.tied',
    kind: 'choice',
    concepts: ['weight-tying', 'embedding'],
    prompt: 'With weight tying, how does the tiny GPT score each of the 68 possible next characters?',
    options: [
      {
        text: 'By the dot product of the final hidden vector with that character’s row of the token-embedding table.',
        correct: true,
        why: 'The same table turns characters into vectors on the way in and vectors into scores on the way out, so the output layer adds no parameters.',
      },
      {
        text: 'With a separate output matrix of 68 rows, initialised as a copy of the embedding and then trained independently.',
        why: 'That would be an untied output layer: 68 × 32 = 2,176 extra parameters here. Tying means literally sharing the same numbers.',
      },
      {
        text: 'By looking up which character’s embedding is nearest to the hidden vector and giving it probability 1.',
        why: 'The scores go through a softmax, giving every character a probability; nearest-neighbour lookup would make the model unable to express uncertainty.',
      },
    ],
  },
  {
    id: 'tiny-gpt.square',
    kind: 'choice',
    concepts: ['gpt'],
    prompt: 'You make the tiny GPT three times wider (same number of blocks). Roughly what happens to the parameters inside each block?',
    options: [
      { text: 'About 9 times as many.', correct: true, why: 'Attention holds about 4d² and the MLP about 8d²; tripling d multiplies d² by 9. (Here 4,224 + 8,352 + 128 = 12,704 per block becomes 111,840 at width 96.)' },
      { text: 'About 3 times as many.', why: 'That is how the embedding tables grow (V × d and T × d are linear in d), but the blocks are made of d × d matrices.' },
      { text: 'About 27 times as many.', why: 'A matrix has two dimensions, not three: the count grows with d², not d³.' },
    ],
  },
  {
    id: 'tiny-gpt.budget',
    kind: 'choice',
    concepts: ['scaling-law', 'chinchilla-optimal'],
    prompt: 'Every design in the lab gets the same compute budget (about 6 × parameters × characters seen). Why does the biggest design not win?',
    options: [
      {
        text: 'Because it can afford very few training steps: with ten times the parameters it sees a tenth as much text, and stops while still improving.',
        correct: true,
        why: 'Size and training length trade against each other at fixed compute; the best loss comes from a balance, which is what compute-optimal scaling laws describe.',
      },
      {
        text: 'Because large models always overfit small datasets.',
        why: 'Overfitting needs many passes over the data. In the lab the big designs see far less than one pass over the tales; they are under-trained, not over-fitted.',
      },
      {
        text: 'Because the budget counts parameters only, so bigger models are penalised directly.',
        why: 'The budget counts operations: parameters × characters. A bigger model is allowed; it just has to train for fewer steps.',
      },
    ],
  },
  {
    id: 'tiny-gpt.honest',
    kind: 'choice',
    concepts: ['perplexity', 'language-model'],
    prompt: 'The tiny GPT scores 1.712 nats per character on held-out tales, against 2.286 for letter-pair counts. What can you honestly conclude?',
    options: [
      {
        text: 'It predicts the next character of unseen fairy-tale text much better than pair counts; that says nothing yet about whether its writing makes sense.',
        correct: true,
        why: 'Loss measures prediction. Sense has to be judged separately; in its samples only about half the words are real and none of it is coherent.',
      },
      {
        text: 'It understands the tales about 25% better than a pair-count table.',
        why: 'A loss ratio is not a measure of understanding, and losses are logarithmic, so “25% better” has no meaning here.',
      },
      {
        text: 'It has memorised the training tales.',
        why: 'The score is measured on four tales it never trained on. Memorising would help on the training text, not on unseen text.',
      },
    ],
  },
  {
    id: 'tiny-gpt.order',
    kind: 'order',
    concepts: ['gpt', 'residual-connection', 'layer-norm'],
    prompt: 'Put one transformer block of the tiny GPT in order, as a character’s vector passes through it.',
    items: [
      'Layer-normalise the residual stream',
      'Causal multi-head attention gathers from earlier positions',
      'Add the result back into the residual stream',
      'Layer-normalise again',
      'MLP: widen to 4d, GELU, narrow back to d',
      'Add the result back into the residual stream',
    ],
    explain: 'Pre-norm, as in GPT-2: each sub-layer reads a normalised copy of the stream and adds its output back, so the stream itself is never overwritten.',
  },
];
