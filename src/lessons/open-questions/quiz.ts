// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'open-questions.double-descent',
    kind: 'choice',
    concepts: ['overfitting'],
    prompt: 'In our minimum-norm random-features experiment with 24 noisy training points, where does test error peak — and why does it fall again past that point?',
    options: [
      {
        text: 'Right at p = 24, where only one exact fit exists and huge weights fight the noise; beyond 24, infinitely many exact fits exist and the smallest-norm preference picks calmer curves.',
        correct: true,
        why: 'Exactly. At p = n = 24 the model has no slack at all, so fitting noisy labels forces huge cancelling weights. Past p = 24, the extra freedom lets the minimum-norm rule pick a smoother interpolator.',
      },
      {
        text: 'At p = 160, because more parameters always mean worse generalisation.',
        why: 'That is the classical textbook curve, which stops before the interpolation threshold. Past p = 24, median test error falls more than a hundredfold — from 9.35 down to 0.070 at p = 160.',
      },
      {
        text: 'At p = 24 even when label noise is zero, because exact interpolation always fails.',
        why: 'Toggle label noise to σ = 0 in the widget: the spike vanishes completely. The explosion at p = n is driven by forcing the unique interpolator to absorb noise.',
      },
      {
        text: 'At p = 16, where training error first drops below 0.05.',
        why: 'p = 16 is the classical sweet spot (median test MSE 0.056), not the peak. The peak hits right at the interpolation threshold p = n = 24.',
      },
    ],
  },
  {
    id: 'open-questions.grokking',
    kind: 'choice',
    concepts: ['overfitting', 'regularization'],
    prompt: 'What is the surprise reported in Power et al.’s 2022 paper on “grokking”?',
    options: [
      {
        text: 'On small algorithmic datasets, a network can jump from random-chance to perfect generalisation well after it has already overfit the training set.',
        correct: true,
        why: 'Right. Classical early stopping assumes that once a model has memorised the training set with terrible validation accuracy, further training can only hurt — grokking shows a delayed transition to generalisation.',
      },
      {
        text: 'Smaller training sets make networks generalise in fewer optimisation steps.',
        why: 'Power et al. observed the opposite: smaller datasets required increasing amounts of optimisation before generalisation appeared.',
      },
      {
        text: 'Networks never memorise algorithmic datasets; they always learn the rule on epoch 1.',
        why: 'In grokking, the network memorises the training set first and sits at near-chance validation accuracy for a long time before suddenly generalising.',
      },
      {
        text: 'Grokking happens only in trillion-parameter language models, never in small networks.',
        why: 'Power et al. specifically studied small algorithmically generated datasets and small networks precisely so memorisation and generalisation could be inspected in detail.',
      },
    ],
  },
  {
    id: 'open-questions.emergence',
    kind: 'choice',
    concepts: ['scaling-laws'],
    prompt: 'In the debate over “emergent abilities” of large language models, what alternative explanation did Schaeffer, Miranda and Koyejo (2023) propose?',
    options: [
      {
        text: 'Sharp jumps can be produced by nonlinear or discontinuous metrics (like exact-match accuracy), whereas smooth, continuous metrics on the same model outputs change gradually with scale.',
        correct: true,
        why: 'Exactly. Their analysis showed that changing how the same fixed model outputs are scored can turn a cliff into a smooth curve — and even create apparent emergence on vision tasks by choosing a discontinuous metric.',
      },
      {
        text: 'Larger models never outperform smaller models on reasoning benchmarks.',
        why: 'Both sides agree that larger models get much better on hard tasks; the disagreement is whether the underlying capability jumps abruptly or improves smoothly beneath a sharp threshold metric.',
      },
      {
        text: 'Emergent abilities are caused solely by data contamination in the training set.',
        why: 'Contamination is a separate evaluation hazard (covered in the evaluation lesson). Schaeffer et al.’s central argument is about how discontinuous evaluation metrics create sharp apparent cliffs.',
      },
      {
        text: 'Scaling laws prove mathematically that every benchmark must have a sudden jump at 100 billion parameters.',
        why: 'Scaling laws describe smooth power-law drops in loss — the exact opposite of a sudden discontinuity, which is why emergent jumps looked puzzling in the first place.',
      },
    ],
  },
];
