// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'svm.support-vectors',
    kind: 'choice',
    concepts: ['svm', 'support-vector'],
    prompt: 'You have trained a hard-margin SVM on 500 points, and 4 of them are support vectors. If you move one of the other 496 points slightly further away from the boundary, what happens to the decision line?',
    options: [
      {
        text: 'Nothing at all: the street is pinned exclusively by the support vectors sitting on the gutters.',
        correct: true,
        why: 'Only points with margin yᵢ(w·xᵢ + b) = 1 touch the gutters and determine the maximum-margin boundary. Non-support points carry zero weight (αᵢ = 0) in the solution.',
      },
      {
        text: 'The line shifts slightly toward the moved point because the class centroid moved.',
        why: 'Methods like linear discriminant analysis or least squares depend on every point’s average position, but a hard-margin SVM ignores every non-support point.',
      },
      {
        text: 'The margin doubles because the class is now wider.',
        why: 'Moving an already-safe interior point changes neither gutter, so the street width 2/‖w‖ stays identical.',
      },
    ],
  },
  {
    id: 'svm.margin-norm',
    kind: 'choice',
    concepts: ['svm', 'margin'],
    prompt: 'With the two gutters scaled to w·x + b = +1 and w·x + b = −1, the street width is 2/‖w‖. To make the street as wide as possible, what must the optimiser do to ‖w‖?',
    options: [
      {
        text: 'Minimise ‖w‖ (equivalently ½‖w‖²) while keeping every point at yᵢ(w·xᵢ + b) ≥ 1.',
        correct: true,
        why: 'Because the width is 2/‖w‖, shrinking the length of the weight vector w widens the physical gap between the +1 and −1 gutters.',
      },
      {
        text: 'Maximise ‖w‖ so the scores w·x + b are as large as possible.',
        why: 'Scaling w and b up makes raw scores bigger without moving the line, and shrinking 2/‖w‖ actually narrows the street between +1 and −1.',
      },
      {
        text: 'Set w = 0 so 2/‖w‖ is infinite.',
        why: 'If w = 0 then w·x + b is constant and cannot satisfy yᵢ(w·xᵢ + b) ≥ 1 for both +1 and −1 points at once.',
      },
    ],
  },
  {
    id: 'svm.soft-c',
    kind: 'choice',
    concepts: ['svm', 'regularization'],
    prompt: 'On a dataset with two mislabelled outliers near the boundary, what usually happens if you crank the soft-margin penalty C very high?',
    options: [
      {
        text: 'The SVM narrows and tilts the street to classify those outliers on the training set, often hurting validation accuracy.',
        correct: true,
        why: 'A huge C makes slack violations expensive, forcing the boundary to bend around noisy training points at the cost of margin width and generalisation.',
      },
      {
        text: 'The SVM ignores the outliers completely and widens the street.',
        why: 'That is what a *small* C does — relaxing the violation penalty so the street stays wide and ignores isolated outliers.',
      },
      {
        text: 'C only controls the learning rate, not the final solution.',
        why: 'The soft-margin SVM objective is convex with a unique optimum that directly depends on C: it trades margin width (½‖w‖²) against slack penalty (C Σ ξᵢ).',
      },
    ],
  },
  {
    id: 'svm.kernel-trick',
    kind: 'choice',
    concepts: ['svm', 'kernel-trick'],
    prompt: 'How does the kernel trick let a linear SVM draw curved boundaries like a circle around concentric rings?',
    options: [
      {
        text: 'It replaces pairwise dot products xᵢ·xⱼ with a kernel k(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ), finding a flat hyperplane in a higher-dimensional feature space whose slice back in 2D is curved.',
        correct: true,
        why: 'Because the dual SVM only ever needs dot products between examples, a kernel function computes inner products in a lifted space (even an infinite-dimensional one for RBF) without explicitly storing the lifted vectors.',
      },
      {
        text: 'It trains a deep neural network first and then fits a line to its output.',
        why: 'Kernel SVMs use fixed similarity functions (polynomial, RBF, etc.) and solve a single convex optimisation problem with no neural-network layers.',
      },
      {
        text: 'It averages many random straight lines together like a random forest.',
        why: 'A kernel SVM fits one maximum-margin hyperplane in the lifted feature space, not an ensemble of lines.',
      },
    ],
  },
];
