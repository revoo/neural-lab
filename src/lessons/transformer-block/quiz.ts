import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'transformer-block.roles',
    kind: 'choice',
    concepts: ['transformer'],
    prompt: 'In a transformer block, what is the division of labour between attention and the feed-forward network?',
    options: [
      { text: 'Attention moves information between tokens; the feed-forward network processes each token’s vector on its own.', correct: true, why: 'Communicate, then compute. The feed-forward network is the same small MLP applied to every token separately; only attention lets tokens see each other.' },
      { text: 'Attention processes each token; the feed-forward network moves information between tokens.', why: 'The other way round: the feed-forward network never looks at other tokens.' },
      { text: 'They do the same thing; the second one just refines the first.', why: 'They do different kinds of work, which is why a block needs both.' },
    ],
  },
  {
    id: 'transformer-block.residual',
    kind: 'choice',
    concepts: ['residual-connection'],
    prompt: 'Why do residual connections (x + sublayer(x)) make deep stacks trainable?',
    options: [
      { text: 'Each layer adds to its input instead of replacing it, so there’s always a direct path back through the additions and the gradient can’t fade away with depth.', correct: true, why: 'In the lesson’s experiment, plain stacks kept about 6% of the signal at depth 48, while residual stacks never dropped below where they started. (They can grow, which layer norm helps control.)' },
      { text: 'They reduce the number of parameters.', why: 'An addition has no parameters at all; the count is unchanged.' },
      { text: 'They make each layer’s output smaller.', why: 'If anything the stream tends to grow as notes are added; layer norm keeps each sublayer’s input at a fixed scale.' },
    ],
  },
  {
    id: 'transformer-block.mask',
    kind: 'choice',
    concepts: ['causal-mask'],
    prompt: 'A language model is trained without the causal mask. Its training loss falls almost to zero, but it writes nonsense. Why?',
    options: [
      { text: 'Each position could see the very token it was supposed to predict, so it learned to copy rather than to predict.', correct: true, why: 'Next-token training predicts every position at once; without the mask the answer is in plain view. When generating, the future isn’t there to copy.' },
      { text: 'The model overfitted because it was too large.', why: 'Overfitting would show a gap between training and validation loss; here the model learned a shortcut that only exists during training.' },
      { text: 'Without the mask, attention weights don’t add up to 1.', why: 'Softmax always normalises; the mask decides which tokens may be attended to.' },
    ],
  },
  {
    id: 'transformer-block.params',
    kind: 'numeric',
    concepts: ['transformer'],
    tolerance: 0.5,
    make: (r) => {
      const d = [256, 512, 1024][r.int(3)];
      return {
        prompt: `Ignoring biases and layer norms, how many parameters are in one standard transformer block of width d = ${d} (attention 4d², feed-forward d → 4d → d)?`,
        answer: 12 * d * d,
        explain: `Attention: 4 × ${d}² = ${(4 * d * d).toLocaleString('en-GB')}. Feed-forward: ${d} × ${4 * d} + ${4 * d} × ${d} = 8 × ${d}² = ${(8 * d * d).toLocaleString('en-GB')}. Total 12 × ${d}² = ${(12 * d * d).toLocaleString('en-GB')} — two-thirds of it in the feed-forward network.`,
      };
    },
  },
  {
    id: 'transformer-block.layernorm',
    kind: 'choice',
    concepts: ['layer-norm'],
    prompt: 'What does layer norm do to a token’s vector?',
    options: [
      { text: 'Shifts its numbers to average 0 and scales them to spread 1 (then applies a learned scale and shift).', correct: true, why: 'It works on each token’s vector separately, so a sublayer always sees inputs of a predictable size, however large the residual stream has grown.' },
      { text: 'Makes the vector have length exactly 1.', why: 'Close in spirit, but layer norm subtracts the mean and divides by the standard deviation, so the length becomes about √d, not 1.' },
      { text: 'Averages the vector with its neighbouring tokens’ vectors.', why: 'That would be a kind of attention. Layer norm never looks at other tokens.' },
    ],
  },
  {
    id: 'transformer-block.ablation',
    kind: 'choice',
    concepts: ['transformer', 'residual-connection'],
    prompt: 'In the 4-token sentence microscope on “The cat sat on → mat”, why did switching off the residual skips wreck the prediction even though attention still focused on “cat” and “sat”?',
    options: [
      { text: 'Without the residual addition, attention’s output overwrote “on”’s own embedding — erasing the token’s identity so the MLP could no longer combine “cat”, “sat”, and “on” together.', correct: true, why: 'The residual stream is a shared whiteboard: attention writes notes fetched from earlier tokens while the skip connection keeps the current token’s own word and position intact for the MLP to combine.' },
      { text: 'Because disabling residual skips turns off the causal mask.', why: 'The causal mask lives inside attention’s softmax scores; disabling residual skips changes x + attn(LN(x)) into attn(LN(x)).' },
      { text: 'Because residual skips are where the unembedding matrix is stored.', why: 'Residual skips have zero parameters — they are plain vector additions.' },
    ],
  },
];
