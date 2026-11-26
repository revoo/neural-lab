import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'sched.why',
    kind: 'choice',
    concepts: ['lr-schedule', 'sgd'],
    prompt: 'Why does training with mini-batches benefit from lowering the step size over time?',
    options: [
      { text: 'Near the minimum, each step is mostly noise, and the size of the leftover jitter is proportional to the step size.', correct: true, why: 'Far away, big steps make fast progress despite the noise. Close to the minimum, the true gradient is tiny but the noise isn’t, so only smaller steps (or bigger batches) let the model settle.' },
      { text: 'Because the gradient gets bigger as training goes on.', why: 'It generally gets smaller. The issue is the noise in the gradient estimates, not their size.' },
      { text: 'To save computation at the end of training.', why: 'Every step costs the same whatever its size. The point is precision, not cost.' },
    ],
  },
  {
    id: 'sched.cosine',
    kind: 'numeric',
    concepts: ['lr-schedule'],
    tolerance: 0.0001,
    make: (r) => {
      const peak = [0.1, 0.01, 0.003][r.int(3)];
      const frac = [0.25, 0.5, 0.75][r.int(3)];
      const ans = Math.round(peak * 0.5 * (1 + Math.cos(Math.PI * frac)) * 1e6) / 1e6;
      return {
        prompt: `A cosine schedule has a peak step size of $${peak}$ and lasts $T$ steps. What is the step size after ${frac * 100}% of training? (Round to 6 decimal places if needed.)`,
        answer: ans,
        explain: `$\\eta = ${peak} \\times \\tfrac{1}{2}(1 + \\cos(${frac}\\pi)) = ${peak} \\times \\tfrac{1}{2}(1 + ${Math.cos(Math.PI * frac).toFixed(4)}) = ${ans}$.`,
      };
    },
  },
  {
    id: 'sched.jitter',
    kind: 'numeric',
    concepts: ['lr-schedule'],
    tolerance: 0.01,
    make: (r) => {
      const base = [0.05, 0.2, 0.08][r.int(3)];
      const f = [2, 4, 10][r.int(3)];
      return {
        prompt: `With a constant step size, a model ends up jittering about ${base} above the best possible loss. Roughly how far above would it sit with a step size ${f} times **smaller** (same batch size)?`,
        answer: Math.round((base / f) * 10000) / 10000,
        explain: `The leftover jitter is roughly proportional to the step size, so dividing the step size by ${f} divides it by ${f}: $${base} / ${f} \\approx ${(base / f).toFixed(4)}$ — at the cost of getting there more slowly.`,
      };
    },
  },
  {
    id: 'sched.peak',
    kind: 'choice',
    concepts: ['lr-schedule'],
    prompt: 'In the schedule race, the decaying schedules’ best peak step sizes were higher than the best constant step size. Why?',
    options: [
      { text: 'A constant step size must also suit the end of training, which forces it down; a decaying schedule only uses its peak early, when big steps help.', correct: true, why: 'The constant runner compromises between “fast” and “precise”. A schedule doesn’t have to — so it can afford to be bolder at the start.' },
      { text: 'Decaying schedules are more numerically stable.', why: 'At the same step size they’re exactly as stable. They simply don’t stay at the peak.' },
      { text: 'It was a coincidence of the random seeds.', why: 'The results are averaged over 20 runs with different random batches; the effect is systematic.' },
    ],
  },
  {
    id: 'sched.warmup',
    kind: 'choice',
    concepts: ['lr-schedule'],
    prompt: 'What is learning-rate warmup for?',
    options: [
      { text: 'Protecting the fragile start of training in large models, when gradients are erratic and Adam’s running averages are based on very few steps.', correct: true, why: 'Big steps from random weights with unreliable statistics can knock a large model into a region it never recovers from. In this lesson’s deep spiral network it calmed the start (about half the early loss spike) but didn’t change how many runs succeeded.' },
      { text: 'It makes every model train faster.', why: 'In our tests it changed nothing, and on the delivery problem it was very slightly worse. Its benefit is stability at scale.' },
      { text: 'It’s how momentum builds up velocity.', why: 'Momentum builds velocity on its own; warmup is a schedule for the step size.' },
    ],
  },
  {
    id: 'sched.end',
    kind: 'choice',
    concepts: ['lr-schedule'],
    prompt: 'In the schedule designer, which single change does the most damage to a good schedule?',
    options: [
      { text: 'Raising the last knot, so training ends with a big step size.', correct: true, why: 'The final position is whatever the last few hundred noisy steps left behind. End with big steps and the model is left rattling around the minimum, wasting everything the earlier phases achieved.' },
      { text: 'Raising the first knot moderately.', why: 'A bigger start is usually fine — early on, big steps make fast progress, and the later decay cleans up.' },
      { text: 'Making the middle slightly flatter.', why: 'The middle shape matters much less than the start and especially the end.' },
    ],
  },
];
