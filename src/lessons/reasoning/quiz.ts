// authored-by: Claude Opus 5.5
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'reasoning.thinking',
    kind: 'choice',
    concepts: ['test-time-compute'],
    prompt: 'Why can writing out intermediate steps help a model at all, when every token comes from the same fixed-size forward pass?',
    options: [
      {
        text: 'Each written step is extra computation the model gets to build on: later tokens can read earlier ones, so a hard problem is spread over many forward passes instead of one.',
        correct: true,
        why: 'A single forward pass spends the same arithmetic on “2 + 2” and on a proof. Thinking tokens let the model spend more where it needs more.',
      },
      { text: 'Longer answers are graded more generously.', why: 'In this lesson every answer is graded only on whether its final result is right.' },
      { text: 'The model gets bigger while it writes.', why: 'The weights are fixed at answer time. What grows is the number of forward passes spent on the problem.' },
    ],
  },
  {
    id: 'reasoning.majority',
    kind: 'choice',
    concepts: ['best-of-n'],
    prompt: 'Majority voting over many samples helps on some problems and hurts on others. When does adding votes make things worse?',
    options: [
      { text: 'When one wrong answer is more common than the right one: more votes make that wrong answer win more reliably.', correct: true, why: 'Voting amplifies whatever is most likely. With a 30% chance of being right and a trap that catches 60% of the misses (42% overall), 31 votes are right only 21.4% of the time.' },
      { text: 'Never: more samples always give more information.', why: 'More samples give a better estimate of the most common answer. That’s only useful if the most common answer is the right one.' },
      { text: 'Whenever the model is wrong more than half the time.', why: 'Not quite: if the wrong answers are scattered, the right one can still be the most common, and voting helps. What matters is whether one wrong answer beats it.' },
    ],
  },
  {
    id: 'reasoning.goodhart',
    kind: 'choice',
    concepts: ['best-of-n'],
    prompt: 'With a reward model that is right 85% of the time but loves a rare kind of polished wrong answer, best-of-N accuracy rises with N, peaks, then falls. Why does it fall?',
    options: [
      { text: 'The more samples, the more likely at least one is a wrong answer the judge scores above everything, and best-of-N picks exactly that one.', correct: true, why: 'Optimising hard against an imperfect proxy finds its flaws: Goodhart’s law, at answer time. On a 30% problem the stand-in peaks at 62.1% with 8 samples and drops to 11.7% at 128.' },
      { text: 'The model gets tired after many samples.', why: 'Each sample is independent; the 100th is as good as the first.' },
      { text: 'The reward model runs out of compute.', why: 'The judge scores every sample the same way. The problem is which sample its scores single out.' },
    ],
  },
  {
    id: 'reasoning.grpo',
    kind: 'choice',
    concepts: ['grpo'],
    prompt: 'In GRPO, a group of 8 answers to one problem all get reward 0. What does the update do for that problem?',
    options: [
      { text: 'Nothing: every advantage is zero, because each reward equals the group mean, so there is no better answer to push towards.', correct: true, why: 'That’s why the hardest problems learn last in the live demo: until some attempt succeeds, the group carries no signal.' },
      { text: 'It pushes every answer’s probability down equally.', why: 'Advantages are rewards relative to the group. With no differences, there is nothing relative to push on.' },
      { text: 'It asks a value network how good the problem is.', why: 'GRPO has no value network: the group’s own mean is the baseline.' },
    ],
  },
];
