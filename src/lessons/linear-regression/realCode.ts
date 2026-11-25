// authored-by: Gemini Antigravity
import type { FeatureKey } from './regression';

const COL_EXPR: Record<FeatureKey, string> = {
  x1: 'x1',
  x2: 'x2',
  x1sq: 'x1**2',
  x2sq: 'x2**2',
  x1x2: 'x1 * x2',
  abs: 'np.abs(x1) + np.abs(x2)',
};

/**
 * Equivalent scikit-learn LogisticRegression call for the active lab settings.
 * In sklearn, the L2 objective is (1/N) Σ logloss + (1 / (2·C·N)) ‖w‖², which
 * matches our lab's λ exactly with C = 1 / λ. Verified against
 * sklearn/linear_model/_logistic.py (`solver="newton-cholesky"`).
 */
export function sklearnLogRegLines(keys: readonly FeatureKey[], lambda: number, thresh: number): string {
  const cVal = Number((1 / lambda).toPrecision(4));
  const cols = keys.map((k) => COL_EXPR[k]).join(', ');
  const lines = [
    `X = np.column_stack([${cols}])`,
    `clf = LogisticRegression(C=${cVal}, solver="newton-cholesky").fit(X, y)`,
  ];
  if (Math.abs(thresh - 0.5) < 1e-9) {
    lines.push('preds = clf.predict(X_test)  # default threshold 0.50');
  } else {
    lines.push(`preds = (clf.predict_proba(X_test)[:, 1] >= ${thresh.toFixed(2)}).astype(int)`);
  }
  return lines.join('\n');
}
