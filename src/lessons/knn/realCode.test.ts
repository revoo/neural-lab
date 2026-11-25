// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { LAB_BRIEFS } from './knn';
import { sklearnKnnLines } from './realCode';

describe('knn realCode', () => {
  it('wraps KNeighborsClassifier in StandardScaler pipeline when scaled', () => {
    const code = sklearnKnnLines(LAB_BRIEFS[0], { scaled: true, metric: 'l2', weight: 'uniform', k: 5, cols: [0, 1] });
    expect(code).toContain('make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="euclidean"))');
  });

  it('slices selected columns and maps L1 to manhattan', () => {
    const code = sklearnKnnLines(LAB_BRIEFS[1], { scaled: false, metric: 'l1', weight: 'distance', k: 9, cols: [0, 1] });
    expect(code).toContain('X_train[:, [0, 1]]');
    expect(code).toContain('weights="distance", metric="manhattan"');
  });
});
