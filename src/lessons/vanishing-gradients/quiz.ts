// authored-by: Neon High
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'vanish.sigmoid-ceiling',
    kind: 'choice',
    concepts: ['vanishing-gradients'],
    prompt: 'Why can NO weight initialisation scale rescue a plain 16-layer Sigmoid network from vanishing gradients?',
    options: [
      {
        text: 'The sigmoid derivative σ′(z) = σ(z)(1 − σ(z)) has a hard maximum of 0.25 (at z = 0). Multiplying by ≤ 0.25 at every layer shrinks gradients by at least 0.25¹⁶ ≈ 2 × 10⁻¹⁰ unless weights are made so large they immediately saturate the sigmoid to near-zero slope.',
        correct: true,
        why: 'In Level 2 of our lab, 0 of 5 initialisation scales worked for a plain 16-layer Sigmoid stack — you need ReLU or architectural lifelines like residual connections.',
      },
      {
        text: 'Because sigmoid outputs are always negative.',
        why: 'Sigmoid outputs lie in (0, 1), never negative. The vanishing culprit is its derivative ceiling of 1/4.',
      },
      {
        text: 'Because sigmoid has too many parameters to train.',
        why: 'Sigmoid has zero learnable parameters of its own — it is a fixed element-wise function.',
      },
    ],
  },
  {
    id: 'vanish.he-vs-xavier',
    kind: 'choice',
    concepts: ['he-init', 'xavier-init'],
    prompt: 'Why does He (Kaiming) initialisation multiply Xavier’s 1/√d weight standard deviation by an extra factor of √2 for ReLU networks?',
    options: [
      {
        text: 'Because ReLU zeroes out the entire negative half of a symmetric pre-activation distribution, cutting the output variance exactly in half at every layer — so Var(w) must be doubled (σ multiplied by √2) to keep signal variance equal to 1.',
        correct: true,
        why: 'Without that √2 factor, a 16-layer ReLU network shrinks signal RMS by (1/√2)¹⁶ = 1/256 ≈ 0.0039 (measured 0.0031 in our 16-layer probe).',
      },
      {
        text: 'Because ReLU’s slope is 2 for positive inputs.',
        why: 'ReLU’s slope is 1 for positive inputs and 0 for negative inputs — the √2 compensates for half the inputs being zeroed out.',
      },
      {
        text: 'Because GPUs run faster when weights are multiplied by √2.',
        why: 'Hardware speed does not depend on weight magnitude; the √2 is purely variance conservation.',
      },
    ],
  },
  {
    id: 'vanish.init-calc',
    kind: 'numeric',
    concepts: ['he-init', 'xavier-init'],
    tolerance: 0.005,
    make: (r) => {
      const d = [32, 64, 128, 256, 512][r.int(5)];
      const isHe = r.int(2) === 1;
      const ans = isHe ? Math.sqrt(2 / d) : 1 / Math.sqrt(d);
      return {
        prompt: `A dense layer receives ${d} inputs. What initial weight standard deviation σ does ${isHe ? 'He (ReLU)' : 'Xavier (linear)'} initialisation pick? (Give 3 decimal places.)`,
        answer: Number(ans.toFixed(3)),
        explain: isHe
          ? `He initialisation sets σ = √(2 / d) = √(2 / ${d}) = ${ans.toFixed(4)} ≈ ${ans.toFixed(3)}.`
          : `Xavier initialisation sets σ = 1 / √d = 1 / √${d} = ${ans.toFixed(4)} ≈ ${ans.toFixed(3)}.`,
      };
    },
  },
  {
    id: 'vanish.residual-highway',
    kind: 'choice',
    concepts: ['vanishing-gradients', 'residual-connection'],
    prompt: 'How does a residual connection h_{l+1} = h_l + f(h_l) cure vanishing gradients in a 32-layer network?',
    options: [
      {
        text: 'Its derivative is I + ∂f/∂h_l: the identity term “1” lets upstream gradients flow straight back to layer 1 untouched, even if the branch f(h_l) has tiny derivatives.',
        correct: true,
        why: 'That is why adding residual connections rescued even a 16-layer Sigmoid network (grad ratio jumped from 6 × 10⁻¹¹ straight to 1.02!).',
      },
      {
        text: 'It removes all nonlinear activations from the network.',
        why: 'The nonlinear activation still lives inside f(h_l); the skip simply adds an unobstructed linear bypass alongside it.',
      },
      {
        text: 'It clips every weight gradient to 1.0 automatically.',
        why: 'Gradient clipping is a separate optimisation safeguard (mostly against exploding gradients); residual connections change the computation graph itself.',
      },
    ],
  },
];
