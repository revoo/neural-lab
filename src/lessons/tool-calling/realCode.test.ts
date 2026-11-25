// authored-by: Gemini Antigravity
import { describe, expect, it } from 'vitest';
import { pipelineLabRealCode } from './realCode';

describe('tool-calling lab live pipeline code', () => {
  it('emits 4 Stage entries matching model, verifier, and retries', () => {
    const code = pipelineLabRealCode([
      { model: 'fast', check: true, retries: 1 },
      { model: 'strong', check: true, retries: 0 },
      { model: 'fast', check: true, retries: 2 },
      { model: 'fast', check: false, retries: 0 },
    ]);
    expect(code).toContain('Stage("lookup", model="claude-haiku", verify=validate_order_schema, max_retries=1)');
    expect(code).toContain('Stage("policy", model="claude-sonnet", verify=llm_policy_judge, max_retries=0)');
    expect(code).toContain('Stage("math", model="claude-haiku", verify=assert_refund_formula, max_retries=2)');
    expect(code).toContain('Stage("action", model="claude-haiku", verify=None, max_retries=0)');
  });
});
