// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'seq2seq-bottleneck',
    kind: 'choice',
    prompt: 'Why does a plain fixed-vector LSTM Encoder–Decoder degrade sharply as sentence length L grows from 4 words to 24 words, even when the LSTM has no vanishing-gradient problem?',
    options: [
      {
        text: 'Because the entire source sentence must be squeezed into ONE fixed-size context vector c ∈ ℝᵈ, so cramming L ≫ d tokens into a single vector creates unavoidable superposition cross-talk.',
        correct: true,
        why: 'Memory capacity of a single vector is O(d), independent of sentence length L. Cross-attention fixes this by letting the decoder query ALL L encoder states — scaling memory linearly as O(L·d).',
      },
      {
        text: 'Because LSTMs cannot process sequences longer than 16 tokens.',
        why: 'An LSTM can step through any number of tokens; the failure is an information-compression bottleneck at the hand-off between encoder and decoder.',
      },
      {
        text: 'Because softmax outputs cannot sum to 1 when L > 16.',
        why: 'Softmax always sums to 1 for any length L.',
      },
    ],
  },
  {
    id: 'seq2seq-cross-attention',
    kind: 'choice',
    prompt: 'In Bahdanau/Luong cross-attention during English → French translation (“the black cat” → “le chat noir”), what does the decoder compute at every output step t?',
    options: [
      {
        text: 'A single static average of all source hidden states, reused unchanged for every French word.',
        why: 'A static average cannot focus on “cat” when generating “chat” and then jump back to “black” when generating “noir”.',
      },
      {
        text: 'A fresh softmax distribution α_{t,1…S} over all source hidden states h_1…h_S scored against the current decoder state s_t, producing a custom context vector c_t = Σ_s α_{t,s} h_s.',
        correct: true,
        why: 'Computing a fresh soft lookup at every output step lets the decoder reorder words effortlessly (off-diagonal attention!) and gives every source word an O(1) direct path.',
      },
      {
        text: 'It runs the encoder again from scratch for every output token.',
        why: 'The encoder runs only once to produce h_1…h_S; each decoder step simply queries that cached sequence of encoder vectors.',
      },
    ],
  },
  {
    id: 'seq2seq-parallelism',
    kind: 'numeric',
    make: (rng) => {
      const lengths = [128, 256, 512, 1024];
      const L = lengths[Math.floor(rng.next() * lengths.length)];
      const depth = 2 * L;
      return {
        prompt: `An LSTM Encoder–Decoder with cross-attention processes a source sentence of length L = ${L} and generates a target sentence of length L = ${L}. Even though cross-attention cuts the maximum dependency path to O(1), how many sequential recurrent steps (encoder + decoder) must still execute one after another?`,
        answer: depth,
        tolerance: 0,
        explain: `The encoder must step L = ${L} times sequentially and the decoder L = ${L} times sequentially, giving 2L = ${depth} serial steps — which is why Transformers removed the RNN altogether!`,
      };
    },
  },
  {
    id: 'seq2seq-bridge-to-transformer',
    kind: 'choice',
    prompt: 'Bahdanau’s RNN + Cross-Attention already broke the fixed-vector bottleneck and achieved O(1) dependency path length. What single remaining limitation motivated Vaswani et al. (2017) to invent the Transformer (“Attention Is All You Need”)?',
    options: [
      {
        text: 'Recurrent hidden states h_t still depend on h_{t−1}, forcing O(L) sequential steps that starve modern GPUs of parallelism during training.',
        correct: true,
        why: 'Replacing recurrence with parallel self-attention drops training sequential depth from O(L) to O(1), letting GPUs process all 512+ tokens in one matrix multiplication.',
      },
      {
        text: 'Cross-attention could only translate between English and French.',
        why: 'Cross-attention works on any sequence pair; the bottleneck was hardware sequentiality, not language.',
      },
      {
        text: 'Transformers have O(L) attention memory instead of O(L²).',
        why: 'Standard self-attention actually pays O(L²) memory/compute — trading quadratic attention cost for O(1) GPU parallelism!',
      },
    ],
  },
];
