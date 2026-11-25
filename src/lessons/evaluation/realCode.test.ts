// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { LAB_BRIEFS, evaluateEvalDesign } from './eval';
import { evalSuiteRealCode } from './realCode';

describe('evalSuiteRealCode', () => {
  it('renders paired binomtest and unpaired fisher_exact snippets reflecting live settings', () => {
    const b1 = LAB_BRIEFS[0];
    const pairedDesign = { mode: 'paired' as const, grader: 'judge1' as const, n: 2000, alpha: 0.05 };
    const pairedCode = evalSuiteRealCode(pairedDesign, b1, evaluateEvalDesign(pairedDesign, b1));
    expect(pairedCode).toContain('binomtest(n10, n10 + n01, p=0.5, alternative="greater")');
    expect(pairedCode).toContain('dataset[:2000]');
    expect(pairedCode).toContain('fast_llm_judge');
    expect(pairedCode).toContain('p_val <= 0.05');

    const unpairedDesign = { mode: 'unpaired' as const, grader: 'human' as const, n: 500, alpha: 0.02 };
    const unpairedCode = evalSuiteRealCode(unpairedDesign, LAB_BRIEFS[2], evaluateEvalDesign(unpairedDesign, LAB_BRIEFS[2]));
    expect(unpairedCode).toContain('fisher_exact');
    expect(unpairedCode).toContain('split_a[:500]');
    expect(unpairedCode).toContain('exact_unit_test');
    expect(unpairedCode).toContain('p_val <= 0.02');
  });
});
