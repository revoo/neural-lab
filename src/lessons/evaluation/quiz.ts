// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'eval.paired',
    kind: 'choice',
    concepts: ['paired-eval'],
    prompt: 'Model A scores 84% and Model B scores 78% on 100 questions. Why is evaluating both on the SAME 100 questions much sharper than evaluating them on two separate 100-question samples?',
    options: [
      {
        text: 'Questions that both models get right or both get wrong cancel out, leaving only the disagreements to judge — which shrinks the noise dramatically when both models agree on easy and hard items.',
        correct: true,
        why: 'On two separate 100-question sets, 84% vs 78% has p = 0.32 (indistinguishable from luck). On the same 100 questions with 6 pure wins and 0 regressions, McNemar’s exact p drops to 0.031.',
      },
      {
        text: 'Because running both on the same questions doubles the sample size to 200.',
        why: 'The sample size is still 100 questions; the gain comes from cancelling question-difficulty variance, not from having more items.',
      },
      {
        text: 'Because McNemar’s test ignores false positives.',
        why: 'McNemar’s exact test controls the false-positive rate rigorously under the fair-coin null on disagreements.',
      },
    ],
  },
  {
    id: 'eval.winners-curse',
    kind: 'choice',
    concepts: ['eval-overfitting'],
    prompt: 'You test 20 prompt variations on a 50-question eval set and keep the highest scorer, which jumps from 80% to 90%. None of the prompts is actually better than 80%. What will happen on a fresh held-out test set?',
    options: [
      {
        text: 'Its score will regress back toward 80%, because picking the maximum of 20 noisy 50-question runs selects whichever prompt got luckiest on those 50 questions (a ~10-point mirage on average).',
        correct: true,
        why: 'On n = 50 at true accuracy 80%, the best of 20 identical prompts averages 89.9% and beats 85% 98.5% of the time by pure selection luck.',
      },
      {
        text: 'It will stay around 90%, because 20 prompts gave the optimiser more chances to learn the task.',
        why: 'If every prompt has true accuracy 80%, none of them is better on unseen data — you only fitted the noise of those 50 items.',
      },
      {
        text: 'It will drop to 50%, because overfitting reverses every answer.',
        why: 'Selection bias reverts the score toward its true accuracy (80%), not to chance.',
      },
    ],
  },
  {
    id: 'eval.attenuation-num',
    kind: 'numeric',
    concepts: ['llm-as-a-judge'],
    tolerance: 0.05,
    make: (r) => {
      const delta = [4, 5, 6, 8][r.int(4)];
      const qPct = [80, 85, 90][r.int(3)];
      const q = qPct / 100;
      const obs = delta * (2 * q - 1);
      return {
        prompt: `A new model improves true accuracy by +${delta} percentage points, but you grade it with a symmetric noisy LLM judge that is right ${qPct}% of the time. What observed gain (in percentage points) will the judge measure on average?`,
        answer: Number(obs.toFixed(2)),
        explain: `A noisy grader shrinks any true gap by (2q − 1): ${delta} × (2 × ${q.toFixed(2)} − 1) = ${delta} × ${(2 * q - 1).toFixed(2)} = ${obs.toFixed(2)} percentage points.`,
      };
    },
  },
  {
    id: 'eval.position-bias',
    kind: 'choice',
    concepts: ['llm-as-a-judge'],
    prompt: 'An LLM judge prefers whichever response appears in Slot 1 whenever a pair is close. How does position swapping (judging both (A, B) and (B, A)) fix this?',
    options: [
      {
        text: 'Counting a win only when both orderings agree makes the bias symmetric so it cancels out cleanly — and the fraction of pairs that flip verdicts measures how biased the judge is.',
        correct: true,
        why: 'When A and B are equally good, consistent A wins and consistent B wins are mathematically identical for any position bias b, while the flip rate exposes judge unreliability.',
      },
      {
        text: 'It forces the judge to become 100% accurate on every example.',
        why: 'Swapping cancels systematic position bias, not genuine judge errors — a judge that misunderstands the question can still be consistently wrong in both orders.',
      },
      {
        text: 'It halves the number of API calls needed.',
        why: 'Swapping doubles the judge calls per pair; you pay 2× judge compute to remove directional bias.',
      },
    ],
  },
  {
    id: 'eval.noisy-vs-human',
    kind: 'choice',
    concepts: ['llm-as-a-judge', 'paired-eval'],
    prompt: 'In the Eval Sizing Lab, why did a cheap 85%-accurate LLM judge win Brief 1 (unlimited items, +5% gain), yet fail completely on Brief 3 (subtle +3% gain, strict 2% FPR)?',
    options: [
      {
        text: 'With abundant items, cheap volume (n = 2,000) overcomes judge attenuation; for a subtle +3% gain under a strict gate, judge noise shrinks the signal and adds fake disagreements faster than 2,000 items can overcome.',
        correct: true,
        why: 'Noisy judges trade sample size for per-item cost. When the true effect is small or the benchmark size is capped, exact grading becomes mandatory.',
      },
      {
        text: 'Because LLM judges cannot be used in paired evaluations.',
        why: 'You can pair items with an LLM judge, and Brief 1’s optimum actually used Paired + Fast LLM judge — noisy grading just weakens the pairing benefit by adding fake disagreements.',
      },
      {
        text: 'Because α = 0.02 is impossible to satisfy with an LLM judge.',
        why: 'You can always enforce FPR ≤ 2% by raising the decision threshold — the problem is that power collapses below 85% when you do.',
      },
    ],
  },
];
