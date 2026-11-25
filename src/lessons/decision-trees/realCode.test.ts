// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { sklearnTreeLines } from './realCode';

describe('decision-trees realCode', () => {
  it('emits DecisionTreeClassifier, RandomForestClassifier and GradientBoostingClassifier', () => {
    expect(sklearnTreeLines({ method: 'tree', maxDepth: 2, nTrees: 1 })).toContain('DecisionTreeClassifier(max_depth=2, min_samples_leaf=2)');
    expect(sklearnTreeLines({ method: 'rf', maxDepth: 1, nTrees: 15 })).toContain('RandomForestClassifier(n_estimators=15, max_depth=1, max_features="sqrt"');
    expect(sklearnTreeLines({ method: 'gbt', maxDepth: 3, nTrees: 5 })).toContain('GradientBoostingClassifier(n_estimators=5, max_depth=3, learning_rate=0.3');
  });
});
