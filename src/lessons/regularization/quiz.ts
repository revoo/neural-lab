import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'regularization.choose-strength',
    kind: 'choice',
    concepts: ['regularization', 'validation-set'],
    prompt: 'You’re choosing the weight-decay strength λ. Which data should decide it?',
    options: [
      { text: 'The validation set: data the model didn’t train on, kept separate from the final test set.', correct: true, why: 'The validation set is for choices like this. The test set must stay untouched so its score is an honest estimate of performance on new data.' },
      { text: 'The training set: pick the λ with the lowest training error.', why: 'That always picks λ = 0. Any penalty makes the training error worse; that’s the point of it. You need data the model didn’t fit.' },
      { text: 'The test set: it’s the most reliable measure of new-data performance.', why: 'It is — until you use it to make choices. Then it’s been fitted to, a little, and its score becomes optimistic.' },
    ],
  },
  {
    id: 'regularization.decay-factor',
    kind: 'numeric',
    concepts: ['weight-decay'],
    tolerance: 0.0001,
    make: (r) => {
      const eta = [0.1, 0.05, 0.01][r.int(3)];
      const lam = [0.1, 0.2, 0.5][r.int(3)];
      const f = 1 - eta * lam;
      return {
        prompt: `With plain gradient descent, step size η = ${eta} and weight decay λ = ${lam}, every step multiplies each weight by what factor before the usual gradient update?`,
        answer: Number(f.toFixed(4)),
        explain: `The penalty λ/2·w² has slope λw, so the step is w − η(λw + ∂L/∂w) = (1 − ηλ)w − η·∂L/∂w. Here 1 − ${eta} × ${lam} = ${f.toFixed(4)}: every weight shrinks a little each step unless the data pushes it back.`,
      };
    },
  },
  {
    id: 'regularization.dropout-scale',
    kind: 'numeric',
    concepts: ['dropout'],
    tolerance: 0.01,
    make: (r) => {
      const p = [0.2, 0.5, 0.25][r.int(3)];
      const h = [0.6, 0.8, 1.2][r.int(3)];
      const out = h / (1 - p);
      return {
        prompt: `With dropout p = ${p}, a hidden neuron whose usual output is ${h} survives this training step. What value does it pass on to the next layer?`,
        answer: Number(out.toFixed(4)),
        explain: `Survivors are scaled by 1/(1 − p) = 1/${1 - p}, so it passes ${h} / ${1 - p} = ${out.toFixed(3)}. On average (counting the times it’s switched off and passes 0) the next layer still receives ${h}, so nothing changes at test time when every neuron is present.`,
      };
    },
  },
  {
    id: 'regularization.dropout-test',
    kind: 'choice',
    concepts: ['dropout'],
    prompt: 'A model trained with dropout is deployed, and its predictions for the same input change every time it’s called. What’s the most likely cause?',
    options: [
      { text: 'Dropout is still switched on at prediction time (in PyTorch: `model.eval()` was never called).', correct: true, why: 'Dropout is only for training. At test time every neuron is used and there is no randomness; leaving it on makes predictions noisy and worse.' },
      { text: 'Dropout always makes predictions random; that’s the price of using it.', why: 'In training, yes. At test time the full network is used, deterministically.' },
      { text: 'The weights are still being updated during prediction.', why: 'Prediction doesn’t change the weights. The randomness comes from neurons being switched off at random.' },
    ],
  },
  {
    id: 'regularization.early-patience',
    kind: 'choice',
    concepts: ['early-stopping'],
    prompt: 'Why does early stopping wait for a “patience” of several checks, instead of stopping the first time the validation error goes up?',
    options: [
      { text: 'The validation error jitters from step to step, so a single rise is often just noise rather than the start of overfitting.', correct: true, why: 'With a small validation set especially, the error wobbles. Stopping at the first rise can end training before the model has learned the pattern — in the lesson, patience 50 stopped at step 110 with a much worse model.' },
      { text: 'To make training take longer, which always helps.', why: 'Training longer is exactly what leads to overfitting. Patience is about not being fooled by noise, and the best weights are restored at the end anyway.' },
      { text: 'Because the validation error always rises before it falls.', why: 'It usually falls first, while the model learns the pattern. Patience guards against random wobbles.' },
    ],
  },
  {
    id: 'regularization.why-small-weights',
    kind: 'choice',
    concepts: ['weight-decay'],
    prompt: 'Why do smaller weights tend to give a model that generalises better?',
    options: [
      { text: 'Small weights make gentle functions; the sharp wiggles needed to thread noisy points require large, finely balanced weights.', correct: true, why: 'The degree-11 polynomial needed coefficients whose squares summed to 814 to pass through every noisy point; with a suitable penalty they summed to about 3 and the curve calmed down. Neurons with small weights switch gently rather than abruptly.' },
      { text: 'Small weights make the network compute faster.', why: 'The computation is the same size whatever the values of the weights.' },
      { text: 'Small weights mean fewer parameters.', why: 'Weight decay shrinks weights but keeps all of them. The number of parameters doesn’t change.' },
    ],
  },
  {
    id: 'regularization.training-error',
    kind: 'choice',
    concepts: ['regularization'],
    prompt: 'You add weight decay to a model that was overfitting. What should happen to the training error?',
    options: [
      { text: 'It goes up a little, while the test error (ideally) goes down.', correct: true, why: 'Regularisation trades fit on the training data for better generalisation. If the training error improves, something else changed.' },
      { text: 'It goes down, because the model is better.', why: 'The model now pays for large weights, so it can’t fit the training points as closely. It gets better on new data, not on the training data.' },
      { text: 'It stays exactly the same; only the test error changes.', why: 'The training objective has changed, so the fitted model, and its training error, change too.' },
    ],
  },
];
