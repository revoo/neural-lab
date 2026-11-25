// authored-by: Gemini Antigravity
import type { KnnBrief, KnnLabConfig } from './knn';

/**
 * Equivalent scikit-learn KNeighborsClassifier pipeline for the current lab config.
 * Checked against sklearn/neighbors/_classification.py (`n_neighbors`, `weights`, `metric`).
 */
export function sklearnKnnLines(brief: KnnBrief, cfg: KnnLabConfig): string {
  const metric = cfg.metric === 'l2' ? 'euclidean' : 'manhattan';
  const clf = `KNeighborsClassifier(n_neighbors=${cfg.k}, weights="${cfg.weight}", metric="${metric}")`;
  const lines: string[] = [];
  if (brief.featureLabels && cfg.cols.length < brief.featureLabels.length) {
    lines.push(`X_tr, X_te = X_train[:, [${cfg.cols.join(', ')}]], X_test[:, [${cfg.cols.join(', ')}]]`);
  }
  const xTr = brief.featureLabels && cfg.cols.length < brief.featureLabels.length ? 'X_tr' : 'X_train';
  if (cfg.scaled) {
    lines.push(`model = make_pipeline(StandardScaler(), ${clf})`);
  } else {
    lines.push(`model = ${clf}`);
  }
  lines.push(`model.fit(${xTr}, y_train)`);
  return lines.join('\n');
}
