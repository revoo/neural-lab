// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { sklearnLogRegLines } from './realCode';

describe('linear-regression realCode', () => {
  it('maps lambda to C = 1/lambda and uses default predict at t = 0.50', () => {
    const code = sklearnLogRegLines(['x1sq', 'x2sq'], 1, 0.5);
    expect(code).toContain('np.column_stack([x1**2, x2**2])');
    expect(code).toContain('LogisticRegression(C=1, solver="newton-cholesky")');
    expect(code).toContain('clf.predict(X_test)');
  });

  it('emits explicit predict_proba threshold when t != 0.50', () => {
    const code = sklearnLogRegLines(['x1', 'x2', 'abs'], 0.001, 0.35);
    expect(code).toContain('C=1000');
    expect(code).toContain('np.abs(x1) + np.abs(x2)');
    expect(code).toContain('>= 0.35');
  });
});
