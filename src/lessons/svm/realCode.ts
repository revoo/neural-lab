// authored-by: Gemini Antigravity
/** Live scikit-learn code builder for the SVM lab. */
import type { FeatureId, SvmBriefId } from './svm';

export function buildSvmCode(
  brief: SvmBriefId,
  opts: { features?: FeatureId[]; C?: number },
): string {
  if (brief === 'separate' || brief === 'widen') {
    return `from sklearn.svm import SVC\nclf = SVC(kernel="linear", C=1e6).fit(X, y)  # hard-margin limit\nmargin = 2.0 / np.linalg.norm(clf.coef_)`;
  }
  if (brief === 'lift') {
    const f = opts.features && opts.features.length > 0 ? opts.features.join(', ') : 'none';
    const usesR2 = opts.features?.includes('r2');
    const kernelHint = usesR2 ? '  # implicit equivalent: SVC(kernel="rbf")' : '';
    return `from sklearn.svm import LinearSVC\n# engineered features: [${f}]${kernelHint}\nclf = LinearSVC(C=50.0, dual="auto").fit(Phi_X, y)`;
  }
  const C = opts.C ?? 1;
  return `from sklearn.svm import SVC\nclf = SVC(kernel="linear", C=${C}).fit(X_train, y_train)\nval_acc = clf.score(X_val, y_val)`;
}
