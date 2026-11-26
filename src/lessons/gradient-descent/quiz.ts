import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'gd.update',
    kind: 'numeric',
    concepts: ['gradient-descent', 'learning-rate'],
    tolerance: 0.001,
    make: (r) => {
      const w = [2, -1, 0.5, 3][r.int(4)];
      const g = [4, -2, 1.5, -6][r.int(4)];
      const lr = [0.1, 0.05, 0.2][r.int(3)];
      const ans = Math.round((w - lr * g) * 1000) / 1000;
      return {
        prompt: `A weight is currently $${w}$. The slope of the loss with respect to it is $${g}$, and the step size is $${lr}$. What is the weight after one step of gradient descent?`,
        answer: ans,
        explain: `New weight = old weight − step size × slope $= ${w} - ${lr} \\times ${g < 0 ? `(${g})` : g} = ${ans}$. ${g < 0 ? 'The slope is negative (loss falls as the weight rises), so the weight goes **up**.' : 'The slope is positive (loss rises with the weight), so the weight goes **down**.'}`,
      };
    },
  },
  {
    id: 'gd.too-big',
    kind: 'choice',
    concepts: ['learning-rate'],
    prompt: 'Training loss starts jumping up and down and then shoots off to infinity (or `NaN`). What’s the most likely culprit?',
    options: [
      { text: 'The learning rate is too large: each step overshoots, and the overshoots grow.', correct: true, why: 'Gradient descent still knows the right *direction*, but the steps are so big that it lands further up the opposite slope each time. Lowering the learning rate is the first thing to try.' },
      { text: 'The learning rate is too small.', why: 'A too-small learning rate makes training slow and smooth, not explosive.' },
      { text: 'The model has found the minimum.', why: 'At a minimum the loss stops changing. Growing, oscillating loss means the steps are overshooting.' },
      { text: 'The gradient is pointing the wrong way.', why: 'The gradient (from backpropagation) is exact. The issue is the size of the step taken along it.' },
    ],
  },
  {
    id: 'gd.steps-shrink',
    kind: 'choice',
    concepts: ['gradient-descent'],
    prompt: 'With a fixed learning rate, why do gradient descent’s steps naturally get smaller as it approaches a smooth minimum?',
    options: [
      { text: 'Because the step is proportional to the slope, and the ground flattens out near the bottom.', correct: true, why: 'Step = −learning rate × slope. As the slope approaches zero, so does the step — no schedule needed for that part.' },
      { text: 'Because the learning rate automatically decreases.', why: 'In plain gradient descent the learning rate is constant. (Schedules that decrease it exist, but they’re an extra.)' },
      { text: 'Because the loss is smaller.', why: 'The step depends on the *slope* of the loss, not its value. A loss could be large but locally flat (tiny steps), or small but steep (big steps).' },
    ],
  },
  {
    id: 'gd.local',
    kind: 'choice',
    concepts: ['gradient-descent'],
    prompt: 'On the bumpy landscape, gradient descent settles in a dip that isn’t the lowest point. Why can’t it tell?',
    options: [
      { text: 'It only ever measures the slope where it’s standing; at the bottom of any dip, that slope is zero.', correct: true, why: 'Gradient descent is a purely local method — walking downhill in fog. Every dip looks like a destination from the inside.' },
      { text: 'The learning rate was too large.', why: 'A larger step can sometimes hop *out* of a small dip, but that’s luck, not knowledge. The fundamental issue is that it only sees locally.' },
      { text: 'Gradient descent always finds the lowest point if you run it long enough.', why: 'Once the slope is zero the steps are zero; it will sit in that dip forever.' },
    ],
  },
  {
    id: 'gd.valley',
    kind: 'choice',
    concepts: ['learning-rate', 'gradient'],
    prompt: 'In a long, narrow valley, why does plain gradient descent struggle?',
    options: [
      { text: 'The steep sides cap the step size (or it bounces out), and with that small step, progress along the gentle floor is slow.', correct: true, why: 'One learning rate has to serve every direction. It must be small enough not to explode across the steep direction, which makes it far too small for the shallow one. Optimizers like momentum and Adam exist largely to fix this.' },
      { text: 'The gradient doesn’t exist in a valley.', why: 'The gradient exists everywhere here; the problem is that it points mostly across the valley rather than along it.' },
      { text: 'It doesn’t struggle — valleys are the easy case.', why: 'Try descent golf with plain gradient descent: the best possible run still takes more than 20 steps, while momentum can do it in a handful.' },
    ],
  },
  {
    id: 'gd.saddle',
    kind: 'choice',
    concepts: ['gradient-descent', 'gradient'],
    prompt: 'Why are flat spots (where every slope is zero) far more likely to be saddle points than true local minima in a network with many weights?',
    options: [
      { text: 'A true local minimum requires every single direction to curve upward at once — a (1/2)^d coincidence — whereas even one downward-curving direction creates an escape ramp.', correct: true, why: 'At d = 10 independent directions, only 1 in 1,024 flat spots curves up in every direction; 99.8% are saddles. That is why high-dimensional loss plateaus usually end in an escape rather than a permanent trap.' },
      { text: 'Because neural-network loss functions are always convex bowls with a single minimum.', why: 'Neural-network losses are strongly non-convex and have huge numbers of critical points — it is just that almost all of them are saddles rather than high-loss local minima.' },
      { text: 'Because the learning rate automatically flips the sign of upward-curving directions.', why: 'The learning rate scales the step; it cannot change whether a direction curves up or down.' },
    ],
  },
];
