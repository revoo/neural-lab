// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'ql.why-q-not-v',
    kind: 'choice',
    concepts: ['q-learning', 'value-function'],
    prompt: 'Why does model-free RL learn action-values Q(s, a) instead of just state-values V(s)?',
    options: [
      {
        text: 'Because choosing an action from V(s) requires knowing the transition probabilities P(s′|s,a) to look one step ahead, whereas Q(s, a) lets you act greedily with argmaxₐ Q(s, a) without a model.',
        correct: true,
        why: 'Caching one value per (state, action) pair bakes the one-step lookahead directly into the table so the agent never needs the simulator’s transition rules.',
      },
      {
        text: 'Because V(s) cannot handle discount factors γ < 1.',
        why: 'Both V(s) and Q(s, a) use the exact same discount factor γ.',
      },
      {
        text: 'Because Q(s, a) requires fewer entries in memory than V(s).',
        why: 'Q(s, a) actually stores |A| times MORE numbers than V(s) — paying extra memory to eliminate the need for a transition model.',
      },
    ],
  },
  {
    id: 'ql.td-update-numeric',
    kind: 'numeric',
    concepts: ['td-learning', 'q-learning'],
    tolerance: 0.02,
    make: (r) => {
      const oldQ = [1.0, 2.0, 3.0][r.int(3)];
      const reward = [-0.1, 0.0, 1.0][r.int(3)];
      const maxNextQ = [4.0, 6.0, 8.0][r.int(3)];
      const gamma = 0.9;
      const alpha = [0.2, 0.5][r.int(2)];
      const target = reward + gamma * maxNextQ;
      const newQ = oldQ + alpha * (target - oldQ);
      return {
        prompt: `Suppose Q(s, a) = ${oldQ.toFixed(1)}, α = ${alpha}, and γ = ${gamma}. The agent takes action a, observes reward r = ${reward}, and lands in a non-terminal state s′ where max Q(s′, a′) = ${maxNextQ.toFixed(1)}. What is the updated Q(s, a)?`,
        answer: Number(newQ.toFixed(2)),
        explain: `TD target = ${reward} + ${gamma} × ${maxNextQ.toFixed(1)} = ${target.toFixed(2)}. TD error δ = ${target.toFixed(2)} − ${oldQ.toFixed(1)} = ${(target - oldQ).toFixed(2)}. New Q = ${oldQ.toFixed(1)} + ${alpha} × ${(target - oldQ).toFixed(2)} = ${newQ.toFixed(2)}.`,
      };
    },
  },
  {
    id: 'ql.sarsa-vs-qlearning',
    kind: 'choice',
    concepts: ['q-learning', 'sarsa'],
    prompt: 'On Cliff Walking with ε = 0.10 exploration, Q-learning learns the 9-step cliff edge while SARSA learns a longer inland path. Which statement is true?',
    options: [
      {
        text: 'Q-learning achieves a better GREEDY evaluation path (−8.0 vs −10.4), while SARSA achieves a better ONLINE training return (−14.5 vs −19.9) because it accounts for its own ε-exploration slips.',
        correct: true,
        why: 'Q-learning is off-policy (bootstraps with maxₐ′ Q, assuming future perfection), whereas SARSA is on-policy (bootstraps with the actual ε-greedy next action, pricing in cliff falls).',
      },
      {
        text: 'SARSA always finds a strictly better greedy policy than Q-learning once exploration is turned off.',
        why: 'Once ε is turned off at test time, Q-learning’s shorter cliff-edge path (−8.0) beats SARSA’s cautious detour (−10.4).',
      },
      {
        text: 'Q-learning and SARSA converge to the exact same policy for any fixed ε > 0.',
        why: 'Only when ε decays to 0 do their policies coincide; for any fixed ε > 0 on a risky map, on-policy SARSA stays cautious while off-policy Q-learning targets the greedy optimum.',
      },
    ],
  },
  {
    id: 'ql.ucb-vs-greedy',
    kind: 'choice',
    concepts: ['exploration-exploitation'],
    prompt: 'Why did UCB1 cut cumulative regret nearly 5× compared to Pure Greedy (20.2 vs 94.9) in our 5-arm bandit race?',
    options: [
      {
        text: 'Pure greedy locks onto whichever arm got lucky first, whereas UCB1 adds an uncertainty bonus c·√(ln t / N(a)) that systematically explores under-tested arms and fades as evidence accumulates.',
        correct: true,
        why: 'Optimism in the face of uncertainty directs exploration where it actually resolves doubt, unlike ε-greedy which keeps pulling obviously bad arms randomly forever.',
      },
      {
        text: 'UCB1 peeks at the true hidden arm means on step 1.',
        why: 'UCB1 only uses the observed sample mean Q(a) and pull count N(a) — no hidden information.',
      },
      {
        text: 'UCB1 pulls every arm an equal number of times regardless of their rewards.',
        why: 'UCB1 quickly concentrates pulls on the best arm (82.5% optimal pulls across 200 steps in our race).',
      },
    ],
  },
];
