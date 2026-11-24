import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'playground.contribution',
    kind: 'numeric',
    concepts: ['weight', 'neuron'],
    tolerance: 0.001,
    make: (r) => {
      const w = [2.5, -1.5, 4, -3][r.int(4)];
      const a = [0.2, 0.8, -0.5, 0][r.int(4)];
      const ans = Math.round(w * a * 1000) / 1000;
      return {
        prompt: `A connection has weight $${w}$. For the current input, the neuron it comes from outputs $${a}$. How much does this connection add to the receiving neuron’s tally?`,
        answer: ans,
        explain: `Contribution = weight × what’s flowing in $= ${w} \\times ${a < 0 ? `(${a})` : a} = ${ans}$.${a === 0 ? ' A big weight with nothing flowing through it contributes nothing.' : ''}`,
      };
    },
  },
  {
    id: 'playground.contribution-varies',
    kind: 'choice',
    concepts: ['weight'],
    prompt: 'In the playground, why do the contribution bars change when you drag the probe point, even though no weights change?',
    options: [
      { text: 'Each contribution is weight × incoming value, and the incoming values depend on the input.', correct: true, why: 'The weights are fixed, but what flows along each connection is the previous neuron’s output for *this* input. So the same connection can matter a lot for one input and not at all for another.' },
      { text: 'The weights are being retrained on the probe point.', why: 'Nothing is trained when you move the probe. It’s a pure forward pass, run again for the new input.' },
      { text: 'The bars are random noise.', why: 'Everything in the inspector is computed exactly from the weights and the probe’s position.' },
    ],
  },
  {
    id: 'playground.redundancy',
    kind: 'choice',
    concepts: ['ablation'],
    prompt: 'Switching off any single neuron in a layer costs 0% accuracy. Switching off the whole layer drops accuracy to chance. What’s the best explanation?',
    options: [
      { text: 'The layer’s neurons carry overlapping information, so any one can be lost while the others cover for it.', correct: true, why: 'This is redundancy. Ablating one part at a time can’t see it — you have to remove groups. The same pattern appears in replicated systems: no single replica is critical, all of them together are.' },
      { text: 'The layer doesn’t do anything; the drop is a bug.', why: 'If the layer did nothing, removing all of it wouldn’t hurt. The output neuron reads only from that layer, so it clearly matters.' },
      { text: 'Importance can’t be measured by removing things.', why: 'Ablation is a perfectly good measurement — it just answers “what does removing *this* cost?”, which isn’t the same as “does this part do anything?”.' },
    ],
  },
  {
    id: 'playground.dead',
    kind: 'choice',
    concepts: ['relu', 'dead-neuron'],
    prompt: 'A ReLU neuron outputs 0 for every input in the training set. Why can’t training fix it?',
    options: [
      { text: 'Its tally is negative everywhere, where ReLU is flat, so the slope that training follows is zero and its weights never change.', correct: true, why: 'Learning needs a slope. ReLU’s slope is 0 for negative tallies, so no learning signal reaches this neuron’s weights: it stays dead. Leaky ReLU keeps a small slope there to avoid exactly this.' },
      { text: 'Training deliberately freezes unimportant neurons.', why: 'Plain training has no notion of importance. The neuron is stuck because of the shape of ReLU, not because anything decided to freeze it.' },
      { text: 'It can — given enough steps it will come back.', why: 'More steps don’t help: with zero slope, every step changes its weights by zero.' },
    ],
  },
  {
    id: 'playground.params',
    kind: 'numeric',
    concepts: ['parameter'],
    tolerance: 0,
    make: (r) => {
      const h1 = [3, 4, 6, 8][r.int(4)];
      const h2 = [2, 4, 5][r.int(3)];
      const total = 2 * h1 + h1 + h1 * h2 + h2 + h2 * 1 + 1;
      return {
        prompt: `A network takes 2 inputs, has hidden layers of ${h1} and ${h2} neurons, and 1 output neuron. How many parameters (weights and biases) does it have?`,
        answer: total,
        explain: `Layer by layer, inputs × neurons + neurons: $(2 \\times ${h1} + ${h1}) + (${h1} \\times ${h2} + ${h2}) + (${h2} \\times 1 + 1) = ${2 * h1 + h1} + ${h1 * h2 + h2} + ${h2 + 1} = ${total}$.`,
      };
    },
  },
  {
    id: 'playground.big-weight',
    kind: 'choice',
    concepts: ['weight', 'ablation'],
    prompt: 'You find a connection with the largest weight in the whole network. What can you conclude?',
    options: [
      { text: 'Not much on its own: its effect depends on how active its source neuron is, and on whether later layers use its target.', correct: true, why: 'Contribution is weight × incoming value. A huge weight out of a neuron that’s almost always zero barely matters. To know whether it matters, measure: look at its contributions across inputs, or set it to zero and see what breaks.' },
      { text: 'It’s the most important connection in the network.', why: 'That’s the tempting conclusion, but weight size ignores what flows through the connection and what happens downstream.' },
      { text: 'The network is broken.', why: 'Large weights are normal, especially for neurons that need a sharp boundary.' },
    ],
  },
];
