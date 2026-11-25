// authored-by: Neon High
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'dpo.kl-leash',
    kind: 'choice',
    concepts: ['rlhf', 'reward-hacking'],
    prompt: 'In KL-regularised RLHF (max E[r̂(y)] − β · D_KL(π ‖ π_ref)), what happens if you shrink the KL penalty β too close to zero?',
    options: [
      {
        text: 'The policy drifts far from the SFT reference to exploit loopholes in the proxy reward model (reward hacking), driving proxy reward up while true human helpfulness crashes.',
        correct: true,
        why: 'In our Goodhart curve, shrinking β from 1.1 down to 0.1 pushed proxy reward to its maximum 5.0 while crashing true helpfulness from +2.36 all the way down to −2.80.',
      },
      {
        text: 'The policy stays frozen at the SFT reference model.',
        why: 'That happens when β is very LARGE (strong leash), not when β approaches zero.',
      },
      {
        text: 'Both proxy reward and true human helpfulness monotonically improve together.',
        why: 'Goodhart’s Law: any imperfect learned proxy reward eventually diverges from true human quality under extreme optimisation.',
      },
    ],
  },
  {
    id: 'dpo.why-no-rm',
    kind: 'choice',
    concepts: ['dpo'],
    prompt: 'How does Direct Preference Optimisation (DPO) avoid training a separate reward model and running RL (like PPO)?',
    options: [
      {
        text: 'By inverting the closed-form KL-regularised optimum to write the reward directly in terms of β·ln(π_θ / π_ref), then plugging that difference into the Bradley–Terry preference loss so the intractable Z(x) cancels out.',
        correct: true,
        why: 'Because Z(x) depends only on the prompt x and not on the reply y, it cancels identically inside r(x, y_w) − r(x, y_l), turning RLHF into a single supervised classification loss on pairs.',
      },
      {
        text: 'By simply running standard SFT cross-entropy on the winning replies and discarding the losing replies.',
        why: 'Discarding losing replies misses DPO’s crucial negative push away from rejected outputs — that is just SFT on chosen replies.',
      },
      {
        text: 'By removing the reference policy π_ref altogether.',
        why: 'DPO still keeps the frozen reference policy π_ref inside the log-ratio ln(π_θ / π_ref) to enforce the implicit KL leash.',
      },
    ],
  },
  {
    id: 'dpo.implicit-margin',
    kind: 'numeric',
    concepts: ['dpo'],
    tolerance: 0.02,
    make: (r) => {
      const beta = [0.2, 0.5, 1.0][r.int(3)];
      const lrW = [0.5, 1.0, 2.0][r.int(3)];
      const lrL = [-2.0, -1.0, 0.0][r.int(3)];
      const margin = beta * (lrW - lrL);
      return {
        prompt: `With β = ${beta}, suppose ln(π_θ(y_w)/π_ref(y_w)) = ${lrW} and ln(π_θ(y_l)/π_ref(y_l)) = ${lrL}. What is the DPO implicit reward margin β·[ln(π_θ(y_w)/π_ref(y_w)) − ln(π_θ(y_l)/π_ref(y_l))]?`,
        answer: Number(margin.toFixed(2)),
        explain: `Implicit chosen reward = ${beta} × ${lrW} = ${(beta * lrW).toFixed(2)}; rejected = ${beta} × (${lrL}) = ${(beta * lrL).toFixed(2)}; margin = ${margin.toFixed(2)}.`,
      };
    },
  },
  {
    id: 'dpo.likelihood-displacement',
    kind: 'choice',
    concepts: ['dpo'],
    prompt: 'What is “likelihood displacement” — the subtle quirk you can observe directly in the DPO objective?',
    options: [
      {
        text: 'Because DPO only maximises the DIFFERENCE ln(π(y_w)/π_ref(y_w)) − ln(π(y_l)/π_ref(y_l)), the loss can drop toward zero even if the chosen reply’s probability π(y_w) decreases — as long as the rejected reply’s probability drops even faster.',
        correct: true,
        why: 'When both chosen and rejected share similar phrasing, pushing the rejected reply down can drag the chosen reply down with it while still widening the margin.',
      },
      {
        text: 'DPO always forces the chosen reply’s probability to exactly 1.0.',
        why: 'The KL term β keeps π_θ anchored to π_ref, and the objective depends only on the relative margin between winner and loser.',
      },
      {
        text: 'DPO ignores the β parameter whenever the margin is negative.',
        why: 'β scales both the margin and the gradient magnitude everywhere.',
      },
    ],
  },
];
