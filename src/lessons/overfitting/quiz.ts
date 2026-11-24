import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'overfitting.diagnose',
    kind: 'choice',
    concepts: ['overfitting', 'underfitting'],
    prompt: 'A model has training error 0.02 and test error 0.85. Which diagnosis fits best?',
    options: [
      { text: 'Overfitting: it has learned quirks of the training examples that don’t carry over to new data.', correct: true, why: 'A large gap between a tiny training error and a big test error is the signature of fitting noise: high variance.' },
      { text: 'Underfitting: it needs more capacity.', why: 'An underfitting model is wrong on the training data too. Here training error is tiny, so capacity isn’t the problem; if anything there’s too much of it.' },
      { text: 'The test set must be broken, because the model clearly works.', why: 'Training error alone says nothing about new data. When the two disagree, believe the test error (unless you have evidence the test set itself is flawed).' },
      { text: 'Nothing is wrong: test error is always higher.', why: 'It is usually somewhat higher, but a gap this large means most of what the model learned doesn’t generalise.' },
    ],
  },
  {
    id: 'overfitting.underfit-fix',
    kind: 'choice',
    concepts: ['underfitting', 'bias-variance'],
    prompt: 'A straight line is fitted to clearly curved data. Training error 0.40, test error 0.42. Will collecting ten times more data help much?',
    options: [
      { text: 'No: the problem is bias. A straight line can’t follow the curve however many points it gets.', correct: true, why: 'More data shrinks variance, but this model’s error is almost all bias: training and test errors are both high and nearly equal. It needs more capacity (or better features).' },
      { text: 'Yes: more data always reduces test error a lot.', why: 'More data mainly reduces variance. When a model underfits, its error comes from bias, which more data doesn’t touch.' },
      { text: 'Yes, because the gap of 0.02 will close.', why: 'Even if the gap closes completely, test error only drops to about 0.40. The big problem is the shared error, not the gap.' },
    ],
  },
  {
    id: 'overfitting.more-data',
    kind: 'choice',
    concepts: ['overfitting', 'capacity'],
    prompt: 'A degree-11 polynomial overfits badly on 12 points. What happens if you fit the same degree to 100 points from the same source?',
    options: [
      { text: 'It generalises about as well as the best degree, because the extra data pins the curve down.', correct: true, why: 'Overfitting is a mismatch between capacity and data. With 100 points there is nowhere for the curve to wiggle without missing many of them; in the lesson its test error fell from about 1,500 to 0.099.' },
      { text: 'It still overfits just as badly: high degree is always bad.', why: 'Capacity is only too much relative to the data. The same model can be reckless on a small dataset and sensible on a large one.' },
      { text: 'It underfits, because 100 points are too many for it.', why: 'More data can’t make a model too stiff; underfitting is about the model being unable to represent the pattern.' },
    ],
  },
  {
    id: 'overfitting.bv-sum',
    kind: 'numeric',
    concepts: ['bias-variance'],
    tolerance: 0.001,
    make: (r) => {
      const b = [0.01, 0.04, 0.09][r.int(3)];
      const v = [0.02, 0.05, 0.12][r.int(3)];
      const s = [0.25, 0.3, 0.5][r.int(3)];
      const ans = b + v + s * s;
      return {
        prompt: `A regression model has bias² = ${b} and variance = ${v}, and the measurements contain noise with standard deviation ${s}. What is its expected squared error on new data?`,
        answer: Number(ans.toFixed(4)),
        explain: `Expected error = bias² + variance + noise² = ${b} + ${v} + ${s}² = ${b} + ${v} + ${(s * s).toFixed(4)} = ${ans.toFixed(4)}. The noise term is a floor no model can get below.`,
      };
    },
  },
  {
    id: 'overfitting.variance-meaning',
    kind: 'choice',
    concepts: ['bias-variance'],
    prompt: 'In the bias–variance picture, what does a model’s **variance** measure?',
    options: [
      { text: 'How much its predictions would change if it were trained on a different random sample of data.', correct: true, why: 'Variance is sensitivity to the accidents of one particular training set: the spread of the fits from the “parallel universes”.' },
      { text: 'How spread out its training data is.', why: 'That’s the variance of the data. The model’s variance is about how much the *fitted model* changes from one training set to another.' },
      { text: 'How much noise there is in the labels.', why: 'That’s the noise term, σ², which no model can remove. Variance is about the model’s reaction to that noise.' },
      { text: 'How far the average prediction is from the truth.', why: 'That’s bias. A model can be right on average (low bias) and still have high variance.' },
    ],
  },
  {
    id: 'overfitting.network-time',
    kind: 'choice',
    concepts: ['overfitting'],
    prompt: 'While training a large network, the training error keeps falling but the validation error, after falling for a while, starts to rise. What is happening?',
    options: [
      { text: 'The network has learned the broad pattern and is now fitting the noise in the training examples.', correct: true, why: 'Networks pass through underfitting, a good fit and then overfitting as training goes on. The number of training steps acts as a capacity knob.' },
      { text: 'The step size has become too large.', why: 'A step size problem usually shows up in the training error too: it would stall, oscillate or blow up. Here training error is still falling smoothly.' },
      { text: 'The validation set has changed.', why: 'The validation set is fixed; what changes is the model.' },
    ],
  },
  {
    id: 'overfitting.three-sets',
    kind: 'order',
    prompt: 'Put the uses of the three data splits in the order they happen in a well-run project.',
    items: [
      'Fit the parameters on the training set',
      'Compare settings (size, step size, when to stop) on the validation set',
      'Measure the chosen model once on the test set',
    ],
    explain: 'The training set fits parameters, the validation set chooses between settings (as often as you like), and the test set gives one honest final measurement. Choosing settings with the test set turns it into a second validation set and its score becomes optimistic.',
  },
];
