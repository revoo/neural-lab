import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'logic.bias-role',
    kind: 'choice',
    concepts: ['bias', 'neuron'],
    prompt: 'In the workbench, what is a neuron’s bias, in plain terms?',
    options: [
      { text: 'Its tally when every input is off — its default mood before any evidence arrives.', correct: true, why: 'With all inputs at 0, every weight × input term is 0, so the tally is just the bias. A positive bias means “fire unless something stops me”; a negative bias means “stay quiet unless convinced”.' },
      { text: 'An extra input that is always 0.', why: 'If it were always multiplied by 0 it would never matter. It’s more like an input that is always **1**, multiplied by its own weight.' },
      { text: 'A measure of how unfair the neuron is.', why: 'The name is unfortunate. In this context “bias” just means an offset added to the tally — nothing to do with fairness.' },
    ],
  },
  {
    id: 'logic.and-check',
    kind: 'choice',
    concepts: ['neuron'],
    prompt: 'A neuron fires when its tally is above zero. Which settings make it an **AND** gate?',
    options: [
      { text: 'Weights $+1, +1$, bias $-1.5$', correct: true, why: 'Tallies: nothing on → −1.5; one on → −0.5; both on → +0.5. Only “both on” crosses zero.' },
      { text: 'Weights $+1, +1$, bias $-0.5$', why: 'Here one input alone gives $1 - 0.5 = +0.5$, which already fires. That’s an OR gate.' },
      { text: 'Weights $+1, +1$, bias $+1$', why: 'With a positive bias, the neuron fires even with no inputs on (tally = 1). It fires for every row.' },
      { text: 'Weights $-1, -1$, bias $1.5$', why: 'That fires unless both are on — it’s NAND, the opposite of AND.' },
    ],
  },
  {
    id: 'logic.hidden',
    kind: 'choice',
    concepts: ['linearly-separable'],
    prompt: 'Why does XOR need a hidden layer when AND and OR don’t?',
    options: [
      { text: 'XOR’s “on” cases sit in opposite corners, so no single line separates them; hidden neurons build simpler features that the output can then combine.', correct: true, why: 'AND and OR can each be split by one line through the four corners. XOR can’t. Hidden neurons each draw a line — e.g. “A but not B” and “B but not A” — and the output neuron ORs them together.' },
      { text: 'XOR involves more arithmetic.', why: 'The arithmetic per neuron is identical. The difference is geometric: whether one straight line can separate the cases.' },
      { text: 'Because XOR needs negative weights.', why: 'NAND also needs negative weights, and a single neuron handles it fine.' },
    ],
  },
  {
    id: 'logic.multiple',
    kind: 'choice',
    concepts: ['neuron'],
    prompt: 'Two people solve XOR in the workbench with completely different weights. Which is true?',
    options: [
      { text: 'Both can be correct — many internal arrangements produce the same behaviour.', correct: true, why: 'One design uses “A-not-B” and “B-not-A” detectors; another uses OR and NAND. Trained networks show the same freedom, which is part of why interpreting their insides is hard.' },
      { text: 'Only one can be correct; there is a unique set of weights for XOR.', why: 'Even scaling every weight in a working neuron by 2 keeps it working. There are infinitely many solutions.' },
    ],
  },
];
