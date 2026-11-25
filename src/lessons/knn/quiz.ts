// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'knn.k1-train',
    kind: 'choice',
    concepts: ['knn', 'overfitting'],
    prompt: 'With k = 1, why does k-nearest neighbours almost always score 100% on its own training set — even on noisy data?',
    options: [
      {
        text: 'Because every training point’s single closest neighbour in the training set is itself (distance 0), so k = 1 memorises every label including mislabelled outliers.',
        correct: true,
        why: 'On our noisy moons, k = 1 scored 100.0% train but only 84.4% test; raising k to 9 smoothed out the noisy islands and lifted test accuracy to 93.9%.',
      },
      {
        text: 'Because k = 1 has the highest bias and lowest variance.',
        why: 'It’s the exact opposite: k = 1 has lowest bias and highest variance (most jagged boundary). Large k has high bias and low variance.',
      },
      {
        text: 'Because Euclidean distance is always zero on training data.',
        why: 'Only the distance from a point to *itself* is zero; distances to other training points are positive.',
      },
    ],
  },
  {
    id: 'knn.scaling',
    kind: 'choice',
    concepts: ['knn', 'feature-scaling'],
    prompt: 'A loan dataset has Age (20–70) and Annual Income (€20,000–€120,000). Without standardising features, kNN accuracy drops from 93.0% to 62.0%. Why?',
    options: [
      {
        text: 'Because squared differences in euros (~10⁹) completely swamp squared differences in years (~10²), so Euclidean distance effectively ignores Age altogether.',
        correct: true,
        why: 'Standardising each feature to mean 0 and SD 1 (fit on training data only) puts Age and Income on equal footing in the distance sum.',
      },
      {
        text: 'Because kNN cannot handle numbers larger than 1,000.',
        why: 'Floating-point arithmetic handles large numbers easily; the problem is the *mismatch* in scale between features.',
      },
      {
        text: 'Because income is correlated with age.',
        why: 'Even if Age and Income are completely independent, a 2,000× scale difference lets the larger-unit feature monopolise distance.',
      },
    ],
  },
  {
    id: 'knn.dist-num',
    kind: 'numeric',
    concepts: ['knn'],
    tolerance: 0.05,
    make: (r) => {
      const a = [
        [1, 2],
        [0, 3],
        [2, 1],
      ][r.int(3)];
      const b = [
        [4, 6],
        [5, 3],
        [6, 4],
      ][r.int(3)];
      const metric = r.int(2) === 0 ? 'Euclidean (L2)' : 'Manhattan (L1)';
      const dx = Math.abs(a[0] - b[0]);
      const dy = Math.abs(a[1] - b[1]);
      const ans = metric.startsWith('E') ? Math.hypot(dx, dy) : dx + dy;
      return {
        prompt: `Compute the ${metric} distance between point (${a[0]}, ${a[1]}) and point (${b[0]}, ${b[1]}). (Round to two decimal places.)`,
        answer: Number(ans.toFixed(2)),
        explain: metric.startsWith('E')
          ? `√((${a[0]} − ${b[0]})² + (${a[1]} − ${b[1]})²) = √(${dx * dx} + ${dy * dy}) = ${ans.toFixed(2)}.`
          : `|${a[0]} − ${b[0]}| + |${a[1]} − ${b[1]}| = ${dx} + ${dy} = ${ans}.`,
      };
    },
  },
  {
    id: 'knn.curse',
    kind: 'choice',
    concepts: ['knn', 'curse-of-dimensionality'],
    prompt: 'Adding 80 pure random noise columns to a 2D problem dropped kNN test accuracy from 96.0% to 46.0%. What happened to the distances?',
    options: [
      {
        text: 'All 80 noise columns contribute equally to Euclidean distance, burying the 2 signal columns so the nearest and farthest points end up almost the same distance away (ratio 30.1× → 1.43×).',
        correct: true,
        why: 'kNN has no learned feature weights — every column counts equally in distance, so irrelevant columns destroy neighbourhoods.',
      },
      {
        text: 'kNN overfitted the noise columns by giving them huge weights.',
        why: 'kNN doesn’t learn feature weights at all! Its problem is the opposite: it treats signal and noise columns with identical weight 1.',
      },
      {
        text: 'The dataset became linearly inseparable.',
        why: 'Adding independent noise columns never changes the true relationship between the 2 signal columns and the label.',
      },
    ],
  },
  {
    id: 'knn.weighting',
    kind: 'choice',
    concepts: ['knn'],
    prompt: 'Small minority fault pockets sit inside a large normal region. Uniform k = 25 misses over half the faults (42.9% recall), while distance-weighted (1/d) k = 15 catches 100%. Why?',
    options: [
      {
        text: 'With uniform voting, once k exceeds a pocket’s size, surrounding majority points outvote the pocket; weighting each vote by 1/d lets the few very close pocket neighbours outweigh distant background points.',
        correct: true,
        why: 'Distance weighting preserves sharp local pockets even at moderate k while still smoothing away isolated noise.',
      },
      {
        text: 'Distance weighting ignores the majority class.',
        why: 'It counts both classes, but scales each neighbour’s vote inversely by its distance to the query.',
      },
      {
        text: 'Distance weighting standardises the features automatically.',
        why: 'Distance weighting only re-weights votes after distances are computed; it never fixes unscaled feature units.',
      },
    ],
  },
];
