// authored-by: Neon High
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'ntp.autoregressive',
    kind: 'choice',
    concepts: ['autoregressive', 'language-model'],
    prompt: 'How does an autoregressive language model generate a 100-token answer?',
    options: [
      {
        text: 'It predicts a probability distribution for just one next token, picks one, appends it to the prompt, and repeats the whole loop 100 times.',
        correct: true,
        why: 'There is no separate "writing" engine. Every generated token becomes part of the input for predicting the next one — which is also why generation runs sequentially, one token at a time.',
      },
      {
        text: 'It plans all 100 tokens in parallel in a single forward pass.',
        why: 'Training scores all positions in parallel (using the causal mask), because the true tokens are already known. Generation cannot — token 50 depends on what was actually chosen for token 49.',
      },
      {
        text: 'It picks the whole sentence with the highest score out of all possible sentences.',
        why: 'With a 50,000-token vocabulary, there are 50,000¹⁰⁰ possible 100-token sequences — far more than atoms in the observable universe.',
      },
    ],
  },
  {
    id: 'ntp.perplexity-meaning',
    kind: 'choice',
    concepts: ['perplexity'],
    prompt: 'A language model achieves a perplexity of 15 on a held-out English news article (vocabulary size 50,000). What does “15” intuitively mean?',
    options: [
      {
        text: 'On average, the model is about as uncertain at each token as someone rolling a fair 15-sided die — down from a 50,000-sided die.',
        correct: true,
        why: 'Perplexity is e^(average cross-entropy). A uniform distribution over k equally likely choices has perplexity exactly k, so perplexity measures the model’s effective branching factor.',
      },
      {
        text: 'The model predicts the exact right next token 15% of the time.',
        why: 'Perplexity is not an accuracy percentage; lower perplexity is better, and its minimum possible value is 1 (100% probability on every true token).',
      },
      {
        text: 'The model makes 15 mistakes per paragraph.',
        why: 'Perplexity is averaged per token and stays the same whether you score 50 tokens or 50,000 tokens from the same distribution.',
      },
    ],
  },
  {
    id: 'ntp.blunder',
    kind: 'choice',
    concepts: ['perplexity', 'cross-entropy'],
    prompt: 'Model A assigns 50% to the true word on every step. Model B assigns 90% on four steps and 0.1% on the fifth. Which has the worse (higher) perplexity, and why?',
    options: [
      {
        text: 'Model B (perplexity ≈ 4.3 vs 2.0): cross-entropy takes −ln(p), so one overconfident 0.1% blunder adds 6.9 nats of surprise and swamps four easy wins.',
        correct: true,
        why: 'Four words at 90% cost 4 × 0.105 = 0.42 nats; one word at 0.1% costs 6.91 nats. Average = 1.466 nats → exp(1.466) ≈ 4.33. Cross-entropy brutally punishes arrogance.',
      },
      {
        text: 'Model A, because its average probability (50%) is lower than Model B’s average probability (72%).',
        why: 'Perplexity uses the geometric mean (average of logs), not the arithmetic mean. A single probability near zero pulls the geometric mean down dramatically.',
      },
      {
        text: 'They tie, because both assign high probability most of the time.',
        why: 'The logarithmic penalty is asymmetric: being confident and wrong hurts far more than being confident and right helps.',
      },
    ],
  },
  {
    id: 'ntp.numeric-ppl',
    kind: 'numeric',
    concepts: ['perplexity'],
    tolerance: 0.05,
    make: (r) => {
      const k = [4, 8, 10, 16][r.int(4)];
      return {
        prompt: `Suppose a model assigns probability 1/${k} (${(100 / k).toFixed(1)}%) to the true next token at every single position. What is its perplexity?`,
        answer: k,
        explain: `Cross-entropy at every step is −ln(1/${k}) = ln(${k}), so perplexity = exp(ln(${k})) = ${k}. That’s why perplexity equals the effective number of equally likely choices.`,
      };
    },
  },
  {
    id: 'ntp.neural-vs-ngram',
    kind: 'choice',
    concepts: ['language-model'],
    prompt: 'Why did our tiny 2-word neural language model beat a bigram lookup table on “the fox watched the small mouse in the yard .”, even though “fox watched” never appeared in training?',
    options: [
      {
        text: 'Learned word embeddings place “fox” near “cat” and “dog”, so the network transfers animal verbs it learned from cats and dogs to the unseen pair “fox watched”.',
        correct: true,
        why: 'A count table treats every word ID as an unrelated slot, so an unseen pair falls back to a tiny smoothing floor (0.28% here). Neural embeddings share statistical strength across similar words.',
      },
      {
        text: 'Because neural networks always beat count tables on every single sentence.',
        why: 'Not always! On Sentence 4 (“a bright moon shone over the quiet house .”), where every adjacent pair had appeared in training, the simple bigram table actually beat our tiny neural net (2.84 vs 4.48).',
      },
      {
        text: 'Because the neural model memorised the held-out test sentence.',
        why: 'The test sentences were never shown during training; the neural net generalised through shared embedding geometry.',
      },
    ],
  },
];
