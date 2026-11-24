import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'diagnosis-lab.ln2',
    kind: 'choice',
    concepts: ['loss'],
    prompt: 'A binary classifier’s loss (cross-entropy) sits at 0.693 from the first step to the last. What does that number tell you?',
    options: [
      { text: 'The model outputs 0.5 for everything: ln 2 ≈ 0.693 is the loss of a 50–50 guess, so nothing useful is getting through.', correct: true, why: 'Cross-entropy for a prediction of 0.5 is −ln 0.5 = ln 2. A loss stuck there means the output never moves: dead neurons, labels with no pattern, or a model that can’t see the signal.' },
      { text: 'The model is about 69% accurate.', why: 'Loss and accuracy are different scales. 0.693 is specifically ln 2, the loss of predicting 0.5 every time — which gives about 50% accuracy on balanced data.' },
      { text: 'Training has converged to a good solution.', why: 'A good classifier on these problems reaches losses well below 0.1. Staying at exactly ln 2 means it never learned anything.' },
    ],
  },
  {
    id: 'diagnosis-lab.step-zero',
    kind: 'choice',
    concepts: ['dead-neuron'],
    prompt: 'Most of a network’s ReLUs are dead at the end of training. What single piece of evidence best tells you whether to blame the initialisation or the step size?',
    options: [
      { text: 'Whether the neurons were already dead at step 0, before any training.', correct: true, why: 'Dead at step 0: the starting weights did it, and no step size is involved. Alive at step 0 and dead later: training killed them, typically with steps that were too large.' },
      { text: 'The final test accuracy.', why: 'Both causes give similarly bad accuracy. You need evidence that separates them.' },
      { text: 'How many parameters the network has.', why: 'Size doesn’t distinguish the two. Timing does: were they dead before the first update?' },
    ],
  },
  {
    id: 'diagnosis-lab.leak',
    kind: 'choice',
    concepts: ['test-set'],
    prompt: 'A model scores 93% on its test set but 75% once deployed on genuinely new data. Which check should you run first?',
    options: [
      { text: 'Look for test examples that also appear in the training data.', correct: true, why: 'A test score far above real-world performance is the classic sign of leakage. If test rows are copies of training rows, the test score measures memory. (Distribution shift — new data that differs from the old — is the other suspect.)' },
      { text: 'Train a bigger model.', why: 'The model may be overfitting, but the urgent problem is that your measurement is lying to you. Fix the scoreboard first.' },
      { text: 'Lower the step size.', why: 'Nothing about the step size explains why the test score is so much better than real-world performance.' },
    ],
  },
  {
    id: 'diagnosis-lab.labels',
    kind: 'choice',
    concepts: ['overfitting'],
    prompt: 'Only about half of the training points share a label with the majority of their five nearest neighbours, on data that should form two clean shapes. What is the most likely problem, and the fix?',
    options: [
      { text: 'The labels are largely wrong; fix the labelling. No model setting can learn a pattern that isn’t there.', correct: true, why: 'On clean data, nearby points almost always share a label. Agreement near 50% is what coin-flip labels look like. In the clinic, fixing the labels was the only single change that worked.' },
      { text: 'The model needs more capacity.', why: 'More capacity can memorise random labels on the training set, but it can’t generalise them: there’s nothing to generalise.' },
      { text: 'The data needs shuffling.', why: 'Shuffling changes the order the network sees examples in, not whether their labels make sense.' },
    ],
  },
  {
    id: 'diagnosis-lab.seed',
    kind: 'choice',
    concepts: ['overfitting'],
    prompt: 'You try a change and fresh-data accuracy jumps from 75% to 83%. Before declaring victory, what should you do?',
    options: [
      { text: 'Re-run with a few other random seeds (data split and starting weights) and see if the improvement holds.', correct: true, why: 'With small data, one run can be lucky. In the clinic, switching the memoriser to ReLU “fixed” it once but did nothing on other random draws.' },
      { text: 'Nothing: an 8-point improvement is too big to be chance.', why: 'With 30 training points, the spread between random draws of the same configuration was over 10 points. Large improvements can still be luck.' },
      { text: 'Train for longer to make it even better.', why: 'That changes something else as well. First find out whether the improvement you have is real.' },
    ],
  },
  {
    id: 'diagnosis-lab.order',
    kind: 'order',
    prompt: 'Put these debugging steps in a sensible order for a new training pipeline.',
    items: [
      'Look at the data: ranges, duplicates, label sanity',
      'Check the loss at initialisation (≈ ln 2 for a balanced binary problem)',
      'Overfit a tiny batch of ten examples',
      'Train on the full data and compare training and validation curves',
      'Tune regularisation and settings on the validation set',
    ],
    explain: 'Cheap checks first: the data, then whether the untrained model behaves as expected, then whether it can learn anything at all (a tiny batch). Only then train for real, and tune last — tuning can’t fix a broken pipeline.',
  },
];
