// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'w2v.distributional',
    kind: 'choice',
    concepts: ['distributional-hypothesis', 'word2vec'],
    prompt: 'Word2Vec never sees a dictionary definition or a single human label. How does it discover that “king” and “queen” belong near each other?',
    options: [
      {
        text: 'Both words appear in nearly identical surrounding contexts (“rules royal palace”, “wears royal crown”), so training each word vector to predict its neighbours pulls them to the same neighbourhood.',
        correct: true,
        why: 'That is J. R. Firth’s distributional hypothesis: you shall know a word by the company it keeps.',
      },
      {
        text: 'Because “king” and “queen” are spelled with similar letters.',
        why: 'Standard Word2Vec treats every word as an opaque integer ID with zero access to spelling!',
      },
      {
        text: 'Because “king” and “queen” always appear immediately adjacent to each other.',
        why: 'Words do not even need to co-occur directly — sharing the *same* context words (“royal”, “crown”, “palace”) pulls them together via second-order co-occurrence.',
      },
    ],
  },
  {
    id: 'w2v.collapse',
    kind: 'choice',
    concepts: ['negative-sampling'],
    prompt: 'What happens when you train Skip-Gram using ONLY positive `(center, context)` pairs (setting negative samples K = 0)?',
    options: [
      {
        text: 'Representation collapse: every word vector aligns in the exact same direction so every σ(u·v) approaches 1, pushing mean pairwise cosine near +1 and destroying analogies.',
        correct: true,
        why: 'In our corpus, K = 0 collapsed mean pairwise cosine to +0.84 and broke 3 of 4 analogies. Negative samples supply the repulsive force that keeps distinct concepts spread apart.',
      },
      {
        text: 'Training becomes 100% equivalent to Full Softmax.',
        why: 'Full Softmax normalises across all V words (implicitly pushing all V − 1 non-context words down); dropping negatives removes that repulsive force completely.',
      },
      {
        text: 'All vectors shrink to the zero vector [0, 0, …, 0].',
        why: 'The zero vector gives σ(0) = 0.5 (loss = log 2 = 0.693), whereas pointing all vectors in the same large direction drives σ(u·v) → 1 (loss → 0)!',
      },
    ],
  },
  {
    id: 'w2v.speedup',
    kind: 'numeric',
    concepts: ['negative-sampling'],
    tolerance: 5,
    make: (rng) => {
      const vocab = [50_000, 100_000, 200_000][rng.int(3)];
      const k = [4, 9, 19][rng.int(3)];
      const speedup = Math.round(vocab / (1 + k));
      return {
        prompt: `On a vocabulary of V = ${vocab.toLocaleString()} words, Full Softmax computes ${vocab.toLocaleString()} dot products per training pair. How many times faster (in dot products per pair) is Negative Sampling with K = ${k} negative samples (so 1 + ${k} = ${1 + k} dot products)?`,
        answer: speedup,
        explain: `${vocab.toLocaleString()} ÷ ${1 + k} = ${speedup.toLocaleString()}× fewer dot products per pair.`,
      };
    },
  },
  {
    id: 'w2v.static-limit',
    kind: 'choice',
    concepts: ['word2vec'],
    prompt: 'What is the fundamental limitation of static word embeddings like Word2Vec and GloVe that later motivated Transformers?',
    options: [
      {
        text: 'Every word gets exactly ONE fixed vector regardless of sentence context, so “river bank” and “savings bank” are mashed into the same averaged vector.',
        correct: true,
        why: 'Static tables cannot disambiguate polysemous words; Transformers start with a static embedding table and then use self-attention to build *contextual* embeddings that change with every sentence.',
      },
      {
        text: 'Static embeddings cannot perform cosine similarity.',
        why: 'Cosine similarity works great on static embeddings — the issue is polysemy (one word, multiple meanings).',
      },
      {
        text: 'Static embeddings require labelled sentiment data to train.',
        why: 'Word2Vec is completely self-supervised from raw unlabelled text.',
      },
    ],
  },
];
