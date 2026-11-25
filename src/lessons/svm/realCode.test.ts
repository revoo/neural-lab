// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { buildSvmCode } from './realCode';

describe('SVM live code builder', () => {
  it('emits valid scikit-learn SVC / LinearSVC snippets matching each brief', () => {
    expect(buildSvmCode('widen', {})).toContain('SVC(kernel="linear"');
    expect(buildSvmCode('lift', { features: ['r2'] })).toContain('LinearSVC');
    expect(buildSvmCode('lift', { features: ['r2'] })).toContain('r2');
    expect(buildSvmCode('soft', { C: 0.3 })).toContain('C=0.3');
  });
});
