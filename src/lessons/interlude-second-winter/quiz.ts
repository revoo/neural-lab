// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'interlude-second-winter.rivals',
    kind: 'choice',
    concepts: ['history'],
    prompt: 'Even after backpropagation became well known in 1986, why were neural networks unfashionable through much of the 1990s and 2000s?',
    options: [
      {
        text: 'On the datasets and CPUs of the era, methods such as support vector machines and random forests matched or beat neural networks while offering cleaner guarantees and easier training.',
        correct: true,
        why: 'Between 1990 and 2010, SVMs, kernel methods, boosting and random forests routinely outperformed neural networks without the training headaches of deep stacks.',
      },
      {
        text: 'Backpropagation was later proved to be mathematically incorrect.',
        why: 'Backpropagation is exact reverse-mode differentiation; it still trains modern deep networks. What was missing in the 1990s was large data, fast parallel hardware and architectural fixes for deep stacks.',
      },
      {
        text: 'Nobody published any neural-network papers between 1987 and 2012.',
        why: 'Crucial work — including LeNet (1989), LSTMs (1997), the 2004 CIFAR programme and 2006 deep belief nets — happened during those years.',
      },
    ],
  },
  {
    id: 'interlude-second-winter.tabular',
    kind: 'choice',
    concepts: ['classical-ml'],
    prompt: 'Why does a modern AI course still dedicate a whole module to classical methods like trees, SVMs and linear models?',
    options: [
      {
        text: 'On structured tabular datasets they remain fast, strong and often the right first tool — not just historical curiosities.',
        correct: true,
        why: 'Deep networks dominate raw pixels, audio and text, but on tables of numbers with moderate sample sizes, classical methods and tree ensembles are frequently simpler, cheaper and just as accurate or better.',
      },
      {
        text: 'Only to show what failed before deep learning replaced every other algorithm.',
        why: 'Classical methods did not become obsolete; linear models, trees and forests are still workhorses across industry.',
      },
      {
        text: 'Because neural networks cannot run on CPUs.',
        why: 'Small neural networks run easily on CPUs; classical models earn their place through performance, interpretability and low data requirements on tabular problems.',
      },
    ],
  },
];
