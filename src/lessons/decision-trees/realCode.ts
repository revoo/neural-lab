// authored-by: Gemini Antigravity
import type { TreeLabConfig } from './trees';

/**
 * Equivalent scikit-learn tree/ensemble classifier for the current lab config.
 */
export function sklearnTreeLines(cfg: TreeLabConfig): string {
  if (cfg.method === 'tree') {
    return `model = DecisionTreeClassifier(max_depth=${cfg.maxDepth}, min_samples_leaf=2)\nmodel.fit(X_train, y_train)`;
  }
  if (cfg.method === 'rf') {
    return `model = RandomForestClassifier(n_estimators=${cfg.nTrees}, max_depth=${cfg.maxDepth}, max_features="sqrt", min_samples_leaf=2)\nmodel.fit(X_train, y_train)`;
  }
  return `model = GradientBoostingClassifier(n_estimators=${cfg.nTrees}, max_depth=${cfg.maxDepth}, learning_rate=0.3, min_samples_leaf=3)\nmodel.fit(X_train, y_train)`;
}
