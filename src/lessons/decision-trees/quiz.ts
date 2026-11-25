// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'trees.scale-invariance',
    kind: 'choice',
    concepts: ['decision-tree'],
    prompt: 'You multiply one column of a tabular dataset (e.g. RPM) by 1,000 before training a CART decision tree. What happens to the tree’s predictions?',
    options: [
      {
        text: 'Nothing at all: every split threshold on that column simply multiplies by 1,000, leaving the ordering of points and every prediction identical.',
        correct: true,
        why: 'Decision trees only compare whether x_j <= threshold, so any strictly increasing rescaling of a feature leaves splits completely unchanged.',
      },
      {
        text: 'The tree puts 1,000× more weight on RPM, just like unscaled kNN.',
        why: 'Unlike Euclidean distance, Gini impurity only cares about how a split partitions class counts, never the numeric magnitude of the feature.',
      },
      {
        text: 'Gini impurity doubles.',
        why: 'Gini depends solely on class proportions p and (1 − p), never on feature units.',
      },
    ],
  },
  {
    id: 'trees.gini-num',
    kind: 'numeric',
    concepts: ['gini-impurity'],
    tolerance: 0.005,
    make: (r) => {
      const [n0, n1] = [
        [8, 2],
        [6, 4],
        [9, 1],
        [5, 5],
        [7, 3],
      ][r.int(5)];
      const p = n1 / (n0 + n1);
      const g = 2 * p * (1 - p);
      return {
        prompt: `A tree node contains ${n0} negative examples and ${n1} positive examples. What is its Gini impurity G = 2p(1 − p)? (Two decimal places.)`,
        answer: Number(g.toFixed(2)),
        explain: `p = ${n1}/${n0 + n1} = ${p.toFixed(2)}, so G = 2 × ${p.toFixed(2)} × ${(1 - p).toFixed(2)} = ${g.toFixed(3)}.`,
      };
    },
  },
  {
    id: 'trees.bagging-vs-boosting',
    kind: 'choice',
    concepts: ['random-forest', 'gradient-boosting'],
    prompt: 'How do Random Forests (bagging) and Gradient Boosted Trees (boosting) differ in how they combine multiple trees?',
    options: [
      {
        text: 'Random Forests train many decorrelated trees in parallel on bootstrap samples and average them to cut variance; Gradient Boosting trains shallow trees sequentially, each fitting the remaining residuals of the previous ones to cut bias.',
        correct: true,
        why: 'Because bagging averages independent noisy trees, adding more trees rarely overfits; because boosting chases residuals sequentially, too many rounds on noisy labels can overfit.',
      },
      {
        text: 'Random Forests use gradient descent on every leaf while Gradient Boosting picks random splits.',
        why: 'It is Gradient Boosting that uses functional gradient descent (residuals); Random Forests use bootstrap sampling + random feature subsets.',
      },
      {
        text: 'Both train identical trees and only differ in whether they use Gini or cross-entropy.',
        why: 'Parallel variance-averaging vs sequential residual-fitting are completely different ensemble mechanisms.',
      },
    ],
  },
  {
    id: 'trees.gbt-overfit',
    kind: 'choice',
    concepts: ['gradient-boosting', 'overfitting'],
    prompt: 'On our messy industrial table (with 6% flipped labels), Gradient Boosted Trees peaked at 93.1% test accuracy at round 5 and fell to 89.2% by round 30. Why?',
    options: [
      {
        text: 'Once the real rule is learned, the only large residuals left are the 6% mislabelled training examples — so later boosting rounds dedicate new trees to memorising label noise.',
        correct: true,
        why: 'That is why production GBT libraries (XGBoost, LightGBM) combine shallow trees, small learning rates, regularisation and early stopping on a validation set.',
      },
      {
        text: 'Because trees cannot handle 6 noise columns.',
        why: 'GBT handled the 6 noise columns brilliantly at round 5 (reaching 93.1% — and 98.8% with depth 3); the later drop came from fitting the 6% label flips.',
      },
      {
        text: 'Because averaging more trees always increases variance.',
        why: 'Averaging independent trees (bagging) reduces variance; GBT doesn’t average independent trees, it *adds* residual corrections sequentially.',
      },
    ],
  },
  {
    id: 'trees.rf-noise',
    kind: 'choice',
    concepts: ['random-forest'],
    prompt: 'On the 8-column table (2 signal + 6 pure noise columns), why did Random Forest with max_features = 3 score lower (91.9%) than Gradient Boosted Trees (98.8%)?',
    options: [
      {
        text: 'With 6 noise columns out of 8, drawing only 3 random candidate features per split frequently offers a node ONLY noise columns to split on, diluting good splits.',
        correct: true,
        why: 'Feature subsampling decorrelates trees when many features carry signal, but hurts when most columns are pure noise and max_features is tiny.',
      },
      {
        text: 'Because Random Forest requires z-score feature scaling.',
        why: 'Like single trees, Random Forests are 100% invariant to feature scaling.',
      },
      {
        text: 'Because Random Forest can only use depth 1.',
        why: 'Random Forests usually use deep trees; the issue here was random feature subsampling hitting mostly noise columns.',
      },
    ],
  },
];
