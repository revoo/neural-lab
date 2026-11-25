// authored-by: Claude Opus 5.5
import type { Question } from '../../components/quizTypes';

const p = (x: number) => (x < 0 ? `(${x})` : `${x}`);

export const questions: Question[] = [
  {
    id: 'perceptron.update',
    kind: 'numeric',
    concepts: ['perceptron'],
    make: (r) => {
      const w = [r.int(5) - 2, r.int(5) - 2, r.int(5) - 2];
      // pick an input the unit gets wrong
      let x = [1, 1];
      let y = 1;
      for (let tries = 0; tries < 50; tries++) {
        x = [r.int(3), r.int(3)];
        y = r.int(2);
        const said = w[0] * x[0] + w[1] * x[1] + w[2] >= 0 ? 1 : 0;
        if (said !== y) break;
      }
      const said = w[0] * x[0] + w[1] * x[1] + w[2] >= 0 ? 1 : 0;
      const d = y - said;
      const nw = [w[0] + d * x[0], w[1] + d * x[1], w[2] + d];
      const z = w[0] * x[0] + w[1] * x[1] + w[2];
      return {
        prompt: `A perceptron has $w_1 = ${w[0]}$, $w_2 = ${w[1]}$, $b = ${w[2]}$ and step size 1. It fires (says 1) when $w_1 x_1 + w_2 x_2 + b \\ge 0$. It is shown the input $(x_1, x_2) = (${x[0]}, ${x[1]})$, whose right answer is ${y}. After the update, what is $w_1$?`,
        answer: nw[0],
        explain:
          said === y
            ? `The score is $${p(w[0])}\\times${p(x[0])} + ${p(w[1])}\\times${p(x[1])} + ${p(w[2])} = ${z}$, so it says ${said}, which is right. Nothing changes: $w_1$ stays ${w[0]}.`
            : `The score is $${p(w[0])}\\times${p(x[0])} + ${p(w[1])}\\times${p(x[1])} + ${p(w[2])} = ${z}$, so it says ${said} but should say ${y}: a mistake. The rule ${d > 0 ? 'adds' : 'subtracts'} the input: $w_1 = ${w[0]} ${d > 0 ? '+' : '-'} ${x[0]} = ${nw[0]}$ (and $w_2 = ${nw[1]}$, $b = ${nw[2]}$).`,
      };
    },
  },
  {
    id: 'perceptron.right',
    kind: 'choice',
    concepts: ['perceptron'],
    prompt: 'The perceptron is shown an example and gets it right. What does the rule do to the weights?',
    options: [
      { text: 'Nothing: the bracket (right answer − what it said) is 0.', correct: true, why: 'The perceptron only learns from mistakes. When the output is right, $y - \\hat{y} = 0$ and the whole update is zero. That is also why it stops for good once a full pass has no mistakes.' },
      { text: 'It nudges the weights a little towards the input, to make it more confident.', why: 'That is what a sigmoid neuron trained on cross-entropy does: it keeps pushing even on correct examples. The perceptron’s step output gives it no notion of confidence, so a correct answer teaches it nothing.' },
      { text: 'It shrinks all the weights slightly.', why: 'That would be weight decay, a different idea you’ll meet later. The perceptron rule changes nothing when the output is right.' },
    ],
  },
  {
    id: 'perceptron.xor',
    kind: 'choice',
    concepts: ['perceptron', 'linearly-separable'],
    prompt: 'You run the perceptron rule on XOR for a million steps with a tiny step size. What happens?',
    options: [
      { text: 'It never stops making mistakes: no line separates XOR, so some example is always wrong, and the weights keep cycling.', correct: true, why: 'Any line gets at least one of the four corners wrong, so the rule always has a fix to make, and each fix breaks another corner. A smaller step size only makes the cycle smaller; it doesn’t remove it.' },
      { text: 'It converges, just very slowly.', why: 'Novikoff’s theorem guarantees convergence only when a separating line exists. For XOR there is none, so there is nothing to converge to.' },
      { text: 'It converges if the examples are shown in the right order.', why: 'In the lab, every ordering, start and step size fails on XOR. Order changes how many mistakes it makes on separable data, not whether an impossible problem becomes possible.' },
    ],
  },
  {
    id: 'perceptron.hidden',
    kind: 'choice',
    concepts: ['credit-assignment', 'perceptron'],
    prompt: 'A two-layer network of threshold units can compute XOR (you wired one in the logic lab). Why can’t the perceptron rule train its hidden units?',
    options: [
      { text: 'The rule needs each unit’s right answer, and only the output’s right answer is known. Nothing says what a hidden unit should have said.', correct: true, why: 'This is the credit assignment problem. The data says the network’s final answer was wrong, but not which hidden unit is to blame, or in which direction it should change. Backpropagation answers exactly that question, using derivatives, which is why it needs smooth activations instead of steps.' },
      { text: 'Hidden units have too many weights for the rule to handle.', why: 'The rule works for any number of weights. The problem is the missing target: the update uses (right answer − what it said), and a hidden unit has no right answer.' },
      { text: 'Threshold units can’t be stacked.', why: 'They can: the logic lab’s XOR network is a stack of them. They can be stacked, just not trained by this rule.' },
    ],
  },
];
