// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'mdp.discount-myopia',
    kind: 'choice',
    concepts: ['mdp', 'discount-factor'],
    prompt: 'On the Two-Treasure Cliff (no slip), shrinking the discount factor γ from 0.90 down to 0.50 flips the optimal policy from the +10 Vault to the nearby +2 Gem. Why?',
    options: [
      {
        text: 'Because rewards k steps away are multiplied by γᵏ: at γ = 0.50, a +10 reward 5 steps away shrinks by 0.5⁴ = 0.0625 to just 0.625 before step costs, which is less than the nearby +2 Gem.',
        correct: true,
        why: 'Discounting controls the agent’s planning horizon (~1/(1−γ) steps). Small γ makes the agent myopic, favouring small immediate payouts over big distant ones.',
      },
      {
        text: 'Because γ < 0.6 prevents Value Iteration from ever reaching the far side of the grid.',
        why: 'Value Iteration still propagates values across the entire grid — the distant +10 simply arrives discounted so heavily that it loses to the nearby +2.',
      },
      {
        text: 'Because a smaller γ increases the probability of slipping into a pit.',
        why: 'Slip probability belongs to the transition model P(s′ | s, a), completely separate from the discount factor γ.',
      },
    ],
  },
  {
    id: 'mdp.discounted-return',
    kind: 'numeric',
    concepts: ['discount-factor', 'mdp'],
    tolerance: 0.02,
    make: (r) => {
      const gamma = [0.5, 0.8, 0.9][r.int(3)];
      const reward = [4, 6, 10][r.int(3)];
      const stepCost = -0.1;
      // 3-step trajectory: R1 = -0.1, R2 = -0.1, R3 = reward
      const g = stepCost + gamma * stepCost + gamma * gamma * reward;
      return {
        prompt: `An agent takes 3 steps with rewards R₁ = −0.1, R₂ = −0.1, and R₃ = +${reward} (terminal). With discount factor γ = ${gamma}, what is the total discounted return G₀ = R₁ + γR₂ + γ²R₃?`,
        answer: Number(g.toFixed(2)),
        explain: `G₀ = −0.1 + ${gamma}×(−0.1) + ${gamma}²×${reward} = −0.1 − ${(gamma * 0.1).toFixed(2)} + ${(gamma * gamma * reward).toFixed(2)} = ${g.toFixed(2)}.`,
      };
    },
  },
  {
    id: 'mdp.bellman-backup',
    kind: 'choice',
    concepts: ['bellman-equation', 'value-function'],
    prompt: 'What does the Bellman optimality equation say about the best value V*(s) of being in a state s?',
    options: [
      {
        text: 'It equals the best action’s expected immediate reward plus γ times the optimal value of wherever you land next.',
        correct: true,
        why: 'Optimal behaviour is self-consistent across one step: look one step ahead, average over transition probabilities P(s′ | s, a), add γV*(s′), and pick the best action.',
      },
      {
        text: 'It equals the single highest reward anywhere on the map, divided by the number of steps to get there.',
        why: 'Bellman values discount exponentially by γ per step and average over stochastic outcomes (including hazards), rather than dividing by distance.',
      },
      {
        text: 'It requires summing over every possible trajectory of length ∞ by brute force before taking a single action.',
        why: 'The magic of Bellman’s equation is dynamic programming: V*(s′) summarises all future trajectories from s′ in a single cached number.',
      },
    ],
  },
  {
    id: 'mdp.stochastic-detour',
    kind: 'choice',
    concepts: ['mdp', 'policy'],
    prompt: 'In Level 3 (The Ice Corridor, 30% slip), why did the exact Bellman-optimal policy deliberately aim ↑ into the north wall at cell (0,2) instead of → across the gap?',
    options: [
      {
        text: 'Because moving → risks a 15% perpendicular slip south straight into the −12 pit, whereas aiming ↑ into the wall bounces safely in place (70%) and still slips east (15%) with 0% chance of falling south.',
        correct: true,
        why: 'In stochastic MDPs, perpendicular slip turns walls into guardrails. Paying a small step cost to wait for a safe eastward slip beats risking a catastrophic −12 fall.',
      },
      {
        text: 'Because Value Iteration has a bug when slip exceeds 25%.',
        why: 'Value Iteration is exact for any slip in [0, 1]; aiming into the wall genuinely raises expected return from 2.279 to 2.352 and cuts pit falls from 18.5% to 1.9%.',
      },
      {
        text: 'Because bouncing off a wall gives a positive bonus reward.',
        why: 'Bouncing off a wall still costs −0.08 per turn — it wins purely because it eliminates the 15% downward slip into the −12 pit.',
      },
    ],
  },
];
