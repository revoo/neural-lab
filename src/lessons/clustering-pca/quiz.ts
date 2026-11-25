// authored-by: Gemini Antigravity
import type { Question } from '../../components/quizTypes';

export const questions: Question[] = [
  {
    id: 'clustering-pca.kmeans-init-trap',
    kind: 'choice',
    concepts: ['kmeans', 'kmeans-pp'],
    prompt: 'You run Lloyd’s k-Means once with random starting centroids on four well-separated blobs and get inertia 415.6 instead of 73.2. Why did it stop so far from the best clustering?',
    options: [
      {
        text: 'Lloyd’s algorithm is greedy: once no single point is closer to another centroid, updates stop even if two centroids split one blob and another centroid bridges two blobs.',
        correct: true,
        why: 'k-Means minimises a non-convex objective by coordinate descent, so it converges to a local minimum. k-Means++ seeding and multiple restarts fix this.',
      },
      {
        text: 'k-Means only converges to the global minimum when k = 2.',
        why: 'Even for k = 2, non-convexity can create local minima on certain shapes; on our 4 blobs, k-Means++ finds the global minimum on 38/40 seeds.',
      },
      {
        text: 'Because inertia can increase on some steps, pushing the centroids away from the blobs.',
        why: 'Both steps of Lloyd’s algorithm strictly decrease (or preserve) inertia — it never climbs uphill, which is precisely why it cannot escape a bad local valley.',
      },
      {
        text: 'Because Euclidean distance is undefined without labels.',
        why: 'Euclidean distance uses only feature coordinates; k-Means is completely unsupervised.',
      },
    ],
  },
  {
    id: 'clustering-pca.kmeans-rings',
    kind: 'choice',
    concepts: ['kmeans', 'silhouette-score'],
    prompt: 'On two clean concentric rings (inner radius 1.0, outer radius 2.6), k-Means with k = 2 scores a Silhouette of 0.326 and slices both rings in half. Why?',
    options: [
      {
        text: 'Assigning every point to its nearest centroid partitions space into convex Voronoi cells separated by straight lines, so a cluster cannot surround another cluster.',
        correct: true,
        why: 'Both rings share the exact same geometric centre at (0, 0), so two centroids must sit off-centre and cut straight across both rings.',
      },
      {
        text: 'Because k-Means++ picked bad starting points on the rings.',
        why: 'Even the global minimum of k-Means inertia on concentric rings splits them like pie slices — the issue is the convex Voronoi assumption of the objective itself.',
      },
      {
        text: 'Because the rings have too few points for Euclidean distance to work.',
        why: 'Adding a million points to each ring produces the exact same straight-line pie cut.',
      },
      {
        text: 'Because concentric rings require standardising x and y first.',
        why: 'The rings already have identical variance along x and y; standardising leaves the geometry unchanged.',
      },
    ],
  },
  {
    id: 'clustering-pca.pca-pythagoras',
    kind: 'numeric',
    concepts: ['pca'],
    tolerance: 0.02,
    make: (rng) => {
      const totalVar = Math.round((3 + rng.next() * 5) * 10) / 10;
      const projVar = Math.round(totalVar * (0.55 + rng.next() * 0.35) * 10) / 10;
      const mse = Number((totalVar - projVar).toFixed(2));
      return {
        prompt: `A centred 2D dataset has total variance ${totalVar.toFixed(2)}. You project every point onto a line and measure a projected variance of ${projVar.toFixed(2)}. What is the mean squared reconstruction error?`,
        answer: mse,
        explain: `By Pythagoras on every centred point, ‖x‖² = (x·u)² + ‖x − (x·u)u‖². Averaging across the dataset gives Projected Variance + Reconstruction MSE = Total Variance, so MSE = ${totalVar.toFixed(2)} − ${projVar.toFixed(2)} = ${mse.toFixed(2)}.`,
      };
    },
  },
  {
    id: 'clustering-pca.pca-scaling-trap',
    kind: 'choice',
    concepts: ['pca', 'feature-scaling'],
    prompt: 'You run PCA directly on 6 raw sensors where Sensor 1 is measured in millimetres (100× larger numbers than Sensors 2–6). PC1 reports 99.97% explained variance. What actually happened?',
    options: [
      {
        text: 'Because variance scales as the square of units (100² = 10,000×), PC1 simply copied Sensor 1 and discarded the other five sensors.',
        correct: true,
        why: 'In our lab, that raw 1-component basis kept only 17.3% of the true standardised variance across the 6 sensors. Standardise first whenever columns have different units.',
      },
      {
        text: 'Sensors 2–6 are mathematically redundant and can safely be deleted.',
        why: 'Their numbers look tiny only because of unit choice, not because they carry no signal.',
      },
      {
        text: 'PCA automatically ignores columns with large variance.',
        why: 'The exact opposite: PCA chases maximum variance, so a high-scale column hijacks the top component.',
      },
      {
        text: 'The covariance matrix became non-symmetric.',
        why: 'A sample covariance matrix XᵀX/n is always symmetric regardless of feature scales.',
      },
    ],
  },
];
