// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'lstm-highway',
    kind: 'choice',
    prompt: 'What architectural change allows an LSTM to carry gradients cleanly across 25+ steps when a plain tanh RNN vanishes?',
    options: [
      {
        text: 'An additive cell-state highway c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t whose backward path multiplies element-wise by f_t rather than repeatedly multiplying by W_hᵀ and tanh′.',
        correct: true,
        why: 'When f_t ≈ 1, ∂c_t / ∂c_{t−1} ≈ 1 — the “Constant Error Carousel” — just like a residual skip connection stretched across time!',
      },
      {
        text: 'Using ReLU instead of tanh inside every gate.',
        why: 'LSTM gates use sigmoid (to output numbers in [0, 1]) and tanh for the candidate cell; the breakthrough is the additive cell-state update, not ReLU.',
      },
      {
        text: 'Having 4× as many parameters makes every gradient 4× larger.',
        why: 'Extra weights alone do not prevent exponential decay; in fact an LSTM initialised with b_f = 0 (f_t = 0.5) STILL vanishes to 0.5²⁴ ≈ 6×10⁻⁸!',
      },
    ],
  },
  {
    id: 'lstm-forget-bias',
    kind: 'choice',
    prompt: 'Why did Jozefowicz et al. (2015) recommend initialising an LSTM’s forget-gate bias b_f to +1 or +2 instead of 0?',
    options: [
      {
        text: 'At b_f = 0, σ(0) = 0.5 so the cell throws away half its memory and half its gradient at every single step (0.5²⁴ ≈ 6×10⁻⁸); b_f = +2 starts f_t ≈ 0.88 so gradients survive long lags right from step 0.',
        correct: true,
        why: 'Initialising b_f > 0 defaults the cell highway to “remember by default” so BPTT can actually hear long-range dependencies before learning begins.',
      },
      {
        text: 'Because sigmoid(0) is undefined and produces NaNs.',
        why: 'sigmoid(0) = 0.5 is completely well-defined; the issue is exponential decay 0.5^T across T steps.',
      },
      {
        text: 'It cuts the LSTM parameter count by 25%.',
        why: 'Changing the initial value of b_f does not change the number of parameters at all.',
      },
    ],
  },
  {
    id: 'gru-vs-lstm-params',
    kind: 'numeric',
    make: (rng) => {
      const pairs = [
        { D: 8, H: 8 },
        { D: 16, H: 16 },
        { D: 10, H: 20 },
        { D: 32, H: 16 },
      ];
      const { D, H } = pairs[Math.floor(rng.next() * pairs.length)];
      const gru = 3 * H * (D + H + 1);
      return {
        prompt: `A single-layer GRU has 3 gates (update, reset, candidate), each with weight matrices of size H × (D + H) plus H biases. How many parameters does a GRU with input size D = ${D} and hidden size H = ${H} have?`,
        answer: gru,
        tolerance: 0,
        explain: `GRU params = 3 × H × (D + H + 1) = 3 × ${H} × (${D} + ${H} + 1) = ${gru}. (A same-width LSTM would need 4 × ${H} × (${D + H + 1}) = ${(4 * H * (D + H + 1)).toLocaleString()}.)`,
      };
    },
  },
  {
    id: 'lstm-output-vs-input-gate',
    kind: 'choice',
    prompt: 'During a noisy corridor of irrelevant tokens, an LSTM wants to keep its stored memory c_t untouched while hiding that memory from downstream layers until the end of the sentence. Which gate settings achieve this?',
    options: [
      {
        text: 'Forget gate f_t ≈ 1, Input gate i_t ≈ 0, Output gate o_t ≈ 0 (opening o_t ≈ 1 only at the read step).',
        correct: true,
        why: 'f_t ≈ 1 preserves c_{t−1}; i_t ≈ 0 blocks noisy distractors from writing into c_t; o_t ≈ 0 shields h_t until downstream layers actually need the memory.',
      },
      {
        text: 'Forget gate f_t ≈ 0, Input gate i_t ≈ 1, Output gate o_t ≈ 1.',
        why: 'f_t ≈ 0 immediately wipes out c_{t−1} and i_t ≈ 1 overwrites the cell with the distractors!',
      },
      {
        text: 'Forget gate f_t ≈ 1, Input gate i_t ≈ 1, Output gate o_t ≈ 0.',
        why: 'Leaving i_t ≈ 1 lets every hostile distractor add into c_t and corrupt the stored cue.',
      },
    ],
  },
];
