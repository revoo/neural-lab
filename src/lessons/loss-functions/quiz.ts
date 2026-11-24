import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'loss.mean-median',
    kind: 'choice',
    concepts: ['loss', 'mse', 'mae'],
    prompt: 'A model must predict the same number for every example. Which number minimises the **mean absolute error**?',
    options: [
      { text: 'The median of the targets.', correct: true, why: 'Each point pulls with the same strength (the slope of |error| is ±1), so the balance point has as many targets above as below: the median.' },
      { text: 'The mean of the targets.', why: 'That’s the answer for *squared* error, whose pulls grow with distance. With absolute error, distance doesn’t matter — only which side each point is on.' },
      { text: 'The most common target.', why: 'That would minimise the *number* of wrong predictions, a different loss again.' },
      { text: 'Halfway between the smallest and largest target.', why: 'That minimises the *worst* error, not the average one.' },
    ],
  },
  {
    id: 'loss.outlier',
    kind: 'choice',
    concepts: ['mse'],
    prompt: 'Your regression data contains a few wildly wrong readings from a faulty sensor. Why might squared error be a poor choice?',
    options: [
      { text: 'Squaring makes those huge errors dominate the loss, so the fitted model bends towards the faulty readings.', correct: true, why: 'A reading that’s 10× further off costs 100× as much. The model reduces the loss most by accommodating the outliers, at the expense of every good point. Absolute error or Huber loss resist this.' },
      { text: 'Squared error ignores large errors.', why: 'It does the opposite: it amplifies them.' },
      { text: 'Squared error can’t be minimised by gradient descent.', why: 'It’s smooth and easy to minimise. The issue is *what* it makes the model prefer.' },
    ],
  },
  {
    id: 'loss.cost-ratio',
    kind: 'numeric',
    concepts: ['mse'],
    tolerance: 0.001,
    make: (r) => {
      const a = [2, 5, 10][r.int(3)];
      const k = [3, 4, 6][r.int(3)];
      return {
        prompt: `Under squared error, how many times more does an error of ${a * k} cost than an error of ${a}?`,
        answer: k * k,
        explain: `$${a * k}^2 / ${a}^2 = ${(a * k) ** 2} / ${a * a} = ${k * k}$. Under absolute error it would cost only ${k}× as much. Squaring is what makes big errors dominate.`,
      };
    },
  },
  {
    id: 'loss.stuck',
    kind: 'choice',
    concepts: ['cross-entropy', 'sigmoid'],
    prompt: 'A sigmoid classifier is confidently wrong (it gives the true class 0.2%). Why does training with squared error on the probability barely fix it?',
    options: [
      { text: 'The sigmoid is almost flat there, and by the chain rule that flatness multiplies squared error’s slope, so the push is tiny.', correct: true, why: 'The push on the score is (loss slope w.r.t. p) × (sigmoid slope). Far from zero, the sigmoid slope is nearly 0. Cross-entropy’s slope w.r.t. p is −1/p, which cancels that flatness exactly, leaving p − y.' },
      { text: 'Squared error doesn’t notice the mistake: its value is small.', why: 'Its value is close to 1, its maximum — it certainly “notices”. The problem is its *slope*, which is what gradient descent follows.' },
      { text: 'The learning rate is too small.', why: 'A bigger step helps both losses equally in proportion; cross-entropy’s push is still hundreds of times larger in this region.' },
    ],
  },
  {
    id: 'loss.bce',
    kind: 'numeric',
    concepts: ['cross-entropy'],
    tolerance: 0.01,
    make: (r) => {
      const p = [0.9, 0.8, 0.6, 0.3][r.int(4)];
      const y = r.int(2);
      const ans = -(y ? Math.log(p) : Math.log(1 - p));
      return {
        prompt: `A spam filter says $p = ${p}$ that an email is spam. The email is ${y ? '**spam**' : '**not spam**'} ($y = ${y}$). What is the binary cross-entropy loss? (Use the natural log; 2 decimal places.)`,
        answer: Math.round(ans * 100) / 100,
        explain: `With $y = ${y}$, only the ${y ? 'first' : 'second'} term survives: $L = -\\ln ${y ? p : `(1 - ${p})`} = -\\ln ${y ? p : (1 - p).toFixed(1)} \\approx ${ans.toFixed(2)}$.`,
      };
    },
  },
  {
    id: 'loss.logits',
    kind: 'choice',
    concepts: ['cross-entropy', 'logits'],
    prompt: 'Why does PyTorch’s `CrossEntropyLoss` want raw scores (logits) instead of probabilities?',
    options: [
      { text: 'Combining softmax and the logarithm in one step can be computed stably; taking the log of an already-rounded probability can give −∞.', correct: true, why: 'For a very wrong model, the probability of the truth can round to exactly 0 in floating point, and ln 0 is −∞. Working from the scores avoids ever forming that tiny number.' },
      { text: 'Because probabilities aren’t needed for cross-entropy.', why: 'Cross-entropy is all about the probability of the truth — the library just computes it internally, from the scores.' },
      { text: 'It’s faster, but the results are identical.', why: 'Mathematically identical, yes — numerically, no. The fused version avoids overflow and underflow.' },
    ],
  },
  {
    id: 'loss.accuracy',
    kind: 'choice',
    concepts: ['loss', 'gradient-descent'],
    prompt: 'Why don’t we train classifiers by directly maximising accuracy?',
    options: [
      { text: 'Accuracy only changes when a prediction crosses the threshold, so small weight changes almost never change it: its slope is zero almost everywhere.', correct: true, why: 'Gradient descent needs a slope to follow. Accuracy is a staircase; cross-entropy is a smooth ramp that keeps improving as the model becomes more confidently right.' },
      { text: 'Accuracy is too expensive to compute.', why: 'It’s cheap — it’s counting. The problem is that it gives no gradient.' },
      { text: 'High accuracy causes overfitting.', why: 'That’s a separate issue. The reason here is that accuracy has no useful slope.' },
    ],
  },
];
