// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'interlude-first-winter.perceptrons',
    kind: 'choice',
    concepts: ['history'],
    prompt: 'What did Minsky and Papert’s 1969 book Perceptrons actually prove — and what made it hard to move past?',
    options: [
      {
        text: 'It proved the limits of single-layer perceptrons (such as XOR); multilayer networks escape that limit, but nobody in the 1960s knew how to train them.',
        correct: true,
        why: 'Multilayer perceptrons were already known to bypass the single-layer wall. The bottleneck was training hidden units — backpropagation was still years away — alongside broader funding pressures.',
      },
      {
        text: 'It proved that no neural network of any depth could ever compute XOR.',
        why: 'Even in the logic lab you built XOR with one hidden layer. The book’s famous theorem was about single-layer perceptrons, not deep networks.',
      },
      {
        text: 'It was the sole cause of the first AI winter on its own.',
        why: 'Historians note that story is simplified: early neural-network projects like MINOS had already lost funding in 1966, and the 1973 Lighthill report and 1973–74 DARPA cutbacks hit the wider AI field too.',
      },
    ],
  },
  {
    id: 'interlude-first-winter.hardware',
    kind: 'choice',
    concepts: ['history'],
    prompt: 'Why did symbolic, rule-based AI look like the practical route during the 1960s and 1970s?',
    options: [
      {
        text: 'Digital computers had tiny memory and limited speed, while early neural networks mostly required bespoke analogue hardware and could not yet be trained in multiple layers.',
        correct: true,
        why: 'Ross Quillian’s language demo fit only 20 words in memory, a 1976 desktop ran at roughly 1 MIPS, and machines like SNARC, ADALINE and MINOS were built from motors, potentiometers or ferrite cores.',
      },
      {
        text: 'Symbolic AI never suffered from combinatorial explosion on real-world problems.',
        why: 'The 1973 Lighthill report specifically criticised symbolic AI for running into combinatorial explosion as soon as methods left toy domains.',
      },
      {
        text: 'Backpropagation was already standard in 1970, but researchers chose not to use it.',
        why: 'Reverse-mode differentiation appeared in Linnainmaa’s 1970 thesis, but backpropagation only became widely known in AI in the mid-1980s.',
      },
    ],
  },
];
