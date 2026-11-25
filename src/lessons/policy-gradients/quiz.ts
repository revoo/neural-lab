// authored-by: Claude Opus 5.5
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'pg.why-stochastic',
    kind: 'choice',
    concepts: ['policy-gradient', 'policy'],
    prompt: 'In the aliased corridor, why does the best policy have to be random?',
    options: [
      {
        text: 'The agent can’t tell the three squares apart, so it must act the same way in each; any fixed choice gets stuck somewhere, and only a mix of Left and Right gets through every square.',
        correct: true,
        why: 'Always-Left sticks at the wall; always-Right bounces between squares 1 and 2. A coin weighted about 59 : 41 towards Right is the best compromise one rule can make.',
      },
      {
        text: 'Randomness is needed for exploration while learning; once learning is finished, the best policy is deterministic.',
        why: 'That is true when the agent can see the full state, but here the best *final* policy is random: P(Right) = 0.586 beats every deterministic policy, forever.',
      },
      {
        text: 'Because the rewards are noisy.',
        why: 'The corridor’s rewards are exactly −1 per step, with no noise at all. The randomness is needed because of what the agent can’t see, not because of noise.',
      },
    ],
  },
  {
    id: 'pg.log-derivative',
    kind: 'choice',
    concepts: ['policy-gradient'],
    prompt: 'A softmax policy over three actions has π = (0.2, 0.5, 0.3). It samples the third action. What does ∇θ log π(a) = e_a − π say to do to the three logits (before scaling by the reward)?',
    options: [
      {
        text: 'Push them by (−0.2, −0.5, +0.7): raise the chosen action, lower the others in proportion to how likely they were.',
        correct: true,
        why: 'e₃ − π = (0, 0, 1) − (0.2, 0.5, 0.3). The bigger an action’s current share, the more it gives up to make room.',
      },
      {
        text: 'Push them by (0, 0, +1): only the chosen action’s logit changes.',
        why: 'Raising one logit alone would also raise its probability, but the true derivative of log softmax subtracts π from every entry, because all probabilities share one 100% budget.',
      },
      {
        text: 'Push them by (0, 0, +0.3): proportional to the chosen action’s probability.',
        why: 'The push on the chosen action is 1 − π(a), which is largest for actions the policy thought unlikely. Surprising successes get the biggest boost.',
      },
    ],
  },
  {
    id: 'pg.push-numeric',
    kind: 'numeric',
    concepts: ['policy-gradient', 'baseline'],
    tolerance: 0.01,
    make: (r) => {
      const p = [0.25, 0.4, 0.5][r.int(3)];
      const reward = [3, 5, 8][r.int(3)];
      const b = [4, 6][r.int(2)];
      const alpha = [0.1, 0.2][r.int(2)];
      const push = alpha * (reward - b) * (1 - p);
      return {
        prompt: `A policy picks an action with probability ${p}. It gets reward ${reward}; the baseline is ${b}; the step size is ${alpha}. By how much does that action’s own logit change? (Use Δθ_a = α·(R − b)·(1 − π(a)).)`,
        answer: Number(push.toFixed(3)),
        explain: `(R − b) = ${reward} − ${b} = ${reward - b}; 1 − π(a) = ${(1 - p).toFixed(2)}; Δθ_a = ${alpha} × ${reward - b} × ${(1 - p).toFixed(2)} = ${push.toFixed(3)}. ${reward < b ? 'The reward was below the baseline, so the action is made less likely even though the reward was positive.' : 'Better than the baseline, so the action becomes more likely.'}`,
      };
    },
  },
  {
    id: 'pg.baseline-bias',
    kind: 'choice',
    concepts: ['baseline'],
    prompt: 'Why is subtracting a baseline b(s) from the return allowed — why doesn’t it bias the gradient?',
    options: [
      {
        text: 'Because the pushes e_a − π, weighted by the policy’s own probabilities, add up to zero: Σₐ π(a)(e_a − π) = 0. A constant multiplied by something whose average is zero has average zero.',
        correct: true,
        why: 'The baseline adds b·(e_a − π) to each sample, and on average that is b × 0. It changes each individual push, not their average.',
      },
      {
        text: 'It does bias the gradient, but only a little, and the lower variance is worth it.',
        why: 'That trade-off is real for bootstrapped critics (actor-critic), but a baseline that depends only on the state, not on the action taken, adds exactly zero bias.',
      },
      {
        text: 'Because the baseline is usually close to zero.',
        why: 'In the push explorer the ideal baseline was 52, nowhere near zero, and the expected gradient still didn’t change at all.',
      },
    ],
  },
  {
    id: 'pg.actor-critic',
    kind: 'choice',
    concepts: ['actor-critic', 'td-learning'],
    prompt: 'In the noisy hallway, a per-state baseline V(s) barely helped, but actor-critic learned much faster. Why?',
    options: [
      {
        text: 'The Monte-Carlo return G_t adds up the reward noise of every remaining step; subtracting V(s) removes its average but not that noise. The TD error r + γV(s′) − V(s) contains only one step’s noise.',
        correct: true,
        why: 'The advantage signal’s spread fell from about 6 (no baseline) to 5.4 (baseline) to 2.3 (actor-critic): the critic replaces the noisy future with its estimate.',
      },
      {
        text: 'Actor-critic uses a larger step size, so it simply moves faster.',
        why: 'Each method was given its own best step size from the same grid. Actor-critic’s best α is larger *because* its signal is less noisy and can tolerate it.',
      },
      {
        text: 'Actor-critic is unbiased while REINFORCE is biased.',
        why: 'It’s the other way round: REINFORCE (with or without a baseline) is unbiased; actor-critic’s bootstrapped target is biased while the critic is wrong. It wins on variance.',
      },
    ],
  },
  {
    id: 'pg.ppo-clip',
    kind: 'choice',
    concepts: ['ppo'],
    prompt: 'What does PPO’s clip actually do?',
    options: [
      {
        text: 'When a sample’s probability ratio has already moved more than ε in the direction its advantage wants, that sample’s gradient is switched off, so reusing a batch can’t push the policy far from the one that collected it.',
        correct: true,
        why: 'Inside [1 − ε, 1 + ε] PPO is the ordinary policy gradient. Outside it, in the direction of improvement, the objective goes flat.',
      },
      {
        text: 'It clips the gradient’s length, like gradient clipping in supervised learning.',
        why: 'PPO clips the probability ratio inside the objective, per sample, and only on one side. Gradient-norm clipping is a different (and complementary) trick.',
      },
      {
        text: 'It clips the rewards to the range [−1, 1].',
        why: 'Reward clipping was used in DQN on Atari; PPO’s clip is on π_new(a)/π_old(a), not on rewards.',
      },
    ],
  },
  {
    id: 'pg.order',
    kind: 'order',
    concepts: ['actor-critic', 'policy-gradient'],
    prompt: 'Put one actor-critic step in order.',
    items: [
      'Sample an action from the policy π(a | s)',
      'Observe the reward r and the next state s′',
      'Compute the TD error δ = r + γV(s′) − V(s)',
      'Nudge the critic: V(s) += β·δ',
      'Nudge the actor: θ += α·δ·∇log π(a | s)',
    ],
    explain:
      'Act, observe, then judge the step by how much better it went than the critic expected (δ). The same δ trains both: the critic moves its estimate towards what happened; the actor makes the action more likely if δ > 0 and less likely if δ < 0. (Updating the actor before the critic is also common; what matters is that both use the same δ.)',
  },
];
