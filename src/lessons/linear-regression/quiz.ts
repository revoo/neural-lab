// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'logreg.normal-eq',
    kind: 'choice',
    concepts: ['normal-equations'],
    prompt: 'Why does ordinary least squares (OLS) explode into huge cancelling weights when two input features are 99.9% correlated?',
    options: [
      {
        text: 'Because XᵀX becomes nearly singular (huge condition number): adding +100 to one weight and −100 to the other barely changes training predictions, so the solver fits training noise with wild weights that break on new data.',
        correct: true,
        why: 'Adding a small Ridge penalty λI lifts the smallest eigenvalue away from zero and splits the weight cleanly between the two features.',
      },
      {
        text: 'Because gradient descent uses the wrong sign on correlated features.',
        why: 'The explosion happens in the exact algebraic solution (XᵀX)⁻¹Xᵀy itself, even without gradient descent.',
      },
      {
        text: 'Because linear regression can only handle one input feature.',
        why: 'Linear regression handles any number of features cleanly as long as they aren’t nearly redundant (or Ridge regularisation is used).',
      },
    ],
  },
  {
    id: 'logreg.outlier',
    kind: 'choice',
    concepts: ['logistic-regression'],
    prompt: 'You add one “super-obvious” positive example far to the right (x = 25, y = 1) to a 1D classification dataset. Why does OLS linear regression misclassify real points near the boundary while logistic regression barely moves?',
    options: [
      {
        text: 'OLS penalises (w·x + b − 1)² so predicting 3.5 on an easy positive looks like a huge error and drags the whole slope down; logistic regression squashes x = 25 to σ(z) ≈ 1 with near-zero cross-entropy loss.',
        correct: true,
        why: 'In our 1D experiment, moving one positive point to x = 25 dragged OLS’s boundary from 4.14 to 5.64 (3 errors), while logistic regression stayed locked at 4.11 (0 errors).',
      },
      {
        text: 'Logistic regression ignores all points with x > 10.',
        why: 'It includes every point in its loss, but well-classified far-out points have sigmoid output ≈ 1 and gradient (σ(z) − 1) ≈ 0.',
      },
      {
        text: 'OLS cannot fit a positive slope.',
        why: 'OLS fits a positive slope — just one that is far too shallow because it tries to keep the line near y = 1 all the way out at x = 25.',
      },
    ],
  },
  {
    id: 'logreg.logodds-num',
    kind: 'numeric',
    concepts: ['logistic-regression'],
    tolerance: 0.01,
    make: (r) => {
      const z = [-2, -1, 0, 1, 2][r.int(5)];
      const p = 1 / (1 + Math.exp(-z));
      return {
        prompt: `A logistic regression model computes a linear score (log-odds) of z = ${z}. What predicted probability p = σ(z) does it output? (Round to two decimal places.)`,
        answer: Number(p.toFixed(2)),
        explain: `σ(${z}) = 1 / (1 + e^(${-z})) = ${p.toFixed(4)} ≈ ${p.toFixed(2)}.`,
      };
    },
  },
  {
    id: 'logreg.threshold',
    kind: 'choice',
    concepts: ['precision-recall', 'roc-auc'],
    prompt: 'On a fraud dataset where only 5% of transactions are fraud, a logistic model at threshold t = 0.50 achieves 97.0% accuracy and 50.0% recall. Lowering t to 0.10 drops accuracy to 91.5% but cuts expected business cost nearly in half. Why?',
    options: [
      {
        text: 'When positives are rare and missed frauds cost far more than false-alarm reviews, accuracy rewards saying “not fraud”; lowering the threshold trades cheap false alarms for catching much more fraud (higher recall).',
        correct: true,
        why: 'The model’s probabilities and ROC AUC (0.939) didn’t change at all — choosing the threshold is a business decision based on error costs.',
      },
      {
        text: 'Lowering the threshold retrains the weights to become more accurate.',
        why: 'Moving the threshold never changes the weights or probabilities; it only changes where you draw the cut-off.',
      },
      {
        text: 'Precision and recall always move in the same direction.',
        why: 'Lowering the threshold generally raises recall while lowering precision, which is why you balance them against real error costs.',
      },
    ],
  },
  {
    id: 'logreg.features',
    kind: 'choice',
    concepts: ['logistic-regression', 'feature-engineering'],
    prompt: 'How can a linear classifier like logistic regression achieve 94.8% accuracy on a circular ring dataset using only 3 parameters?',
    options: [
      {
        text: 'By computing engineered features x₁² and x₂² first: a circle in (x₁, x₂) space becomes a flat straight line w₁x₁² + w₂x₂² + b = 0 in the new feature space.',
        correct: true,
        why: 'A model that is linear in its weights can represent curved boundaries in raw coordinates as soon as you supply non-linear features.',
      },
      {
        text: 'By running Newton–Raphson for more iterations on raw (x₁, x₂).',
        why: 'Raw (x₁, x₂) is strictly limited to straight lines and scores 54.8% no matter how well you optimise it.',
      },
      {
        text: 'By increasing L2 regularisation λ until the boundary bends into a circle.',
        why: 'L2 shrinks weights toward zero; it never makes a linear function curved.',
      },
    ],
  },
];
