// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'rnn-weight-sharing',
    kind: 'choice',
    prompt: 'When we “unroll” a single-layer RNN across T = 50 time steps, how does it compare to a 50-layer feed-forward network?',
    options: [
      {
        text: 'It has 50 separate weight matrices W₁ … W₅₀, one per time step.',
        why: 'If every step had its own weights, the model could not generalise to unseen sequence lengths. An RNN reuses the exact same weights at every step.',
      },
      {
        text: 'It is a 50-layer deep network whose weights (W_x, W_h, b_h) are tied to the exact same values at every step, so BPTT sums their gradients across all 50 steps.',
        correct: true,
        why: 'Weight sharing across time lets one small cell process any sequence length — and means backprop multiplies by the same W_hᵀ at every step.',
      },
      {
        text: 'Hidden states do not depend on earlier steps during backpropagation.',
        why: 'Because h_t depends on h_{t−1}, the chain rule must walk all the way backwards through time.',
      },
    ],
  },
  {
    id: 'rnn-spectral-radius',
    kind: 'numeric',
    make: (rng) => {
      const choices = [
        { rho: 0.8, steps: 10 },
        { rho: 0.7, steps: 10 },
        { rho: 0.5, steps: 8 },
        { rho: 1.2, steps: 10 },
      ];
      const c = choices[Math.floor(rng.next() * choices.length)];
      const ans = Number((c.rho ** c.steps).toFixed(4));
      return {
        prompt: `In a linear 1D recurrent chain h_t = w·h_{t−1} with w = ${c.rho}, what is the exact sensitivity |∂h_T / ∂h_{T−${c.steps}}| across ${c.steps} steps? (Round to 4 decimal places.)`,
        answer: ans,
        tolerance: Math.max(0.002, ans * 0.02),
        explain: `Each step backwards multiplies the gradient by w = ${c.rho}, so across ${c.steps} steps the multiplier is ${c.rho}^${c.steps} = ${ans}.`,
      };
    },
  },
  {
    id: 'rnn-clipping-what-it-fixes',
    kind: 'choice',
    prompt: 'Gradient-norm clipping caps ‖g‖₂ at a threshold c (scaling g ← g · c / ‖g‖₂ whenever ‖g‖₂ > c). Which problem does it fix — and which does it NOT fix?',
    options: [
      {
        text: 'It cures exploding-gradient cliffs while preserving gradient direction, but it cannot rescue vanishing gradients that have already decayed to ~10⁻⁸.',
        correct: true,
        why: 'Clipping only shrinks oversized steps (‖g‖ > c). When gradients vanish below the noise floor, there is no signal left to scale up — that requires architectural changes like LSTM/GRU gates or attention.',
      },
      {
        text: 'It cures both vanishing and exploding gradients by normalising every gradient to norm 1.',
        why: 'Standard gradient-norm clipping never amplifies tiny gradients; even if you did amplify a 10⁻⁸ gradient drowned in late-step noise, you would only amplify the noise.',
      },
      {
        text: 'It clips each weight inside W_h to [−c, +c] after every update.',
        why: 'Weight clipping changes the parameters directly; gradient-norm clipping rescales the gradient vector while keeping its exact direction.',
      },
    ],
  },
  {
    id: 'rnn-tbptt-blindspot',
    kind: 'choice',
    prompt: 'You train an RNN with Truncated BPTT horizon K = 5 on a task where the answer at step t = 12 depends purely on the token at step t = 0. What happens to the gradient reaching step t = 0?',
    options: [
      {
        text: 'It is reduced by 5×, so training just takes 5× longer.',
        why: 'Truncation does not merely attenuate earlier gradients — it cuts the computational graph cleanly at step t = 12 − 5 = 7.',
      },
      {
        text: 'It is identically zero: Truncated BPTT cuts the backward graph after K = 5 steps, so dependencies longer than K cannot send direct learning signal back to t = 0.',
        correct: true,
        why: 'Truncated BPTT trades activation memory O(K·H) for a hard horizon: any dependency spanning more than K steps is invisible to the backward pass.',
      },
      {
        text: 'The forward pass also resets h_t = 0 every 5 steps, so information cannot travel past 5 steps.',
        why: 'In standard TBPTT, the forward hidden state h_t carries across chunks; only the backward gradient graph is detached after K steps.',
      },
    ],
  },
];
