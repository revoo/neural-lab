// authored-by: Gemini Antigravity
import { LAB_STAGES, evaluatePipeline, type PipelineConfig } from './reliability';

const VERIFIER_FN: Record<string, string> = {
  lookup: 'validate_order_schema',
  policy: 'llm_policy_judge',
  math: 'assert_refund_formula',
  action: 'dry_run_refund_guard',
};

export function pipelineLabRealCode(cfg: PipelineConfig): string {
  const ev = evaluatePipeline(cfg);
  const rows = cfg.map((c, i) => {
    const id = LAB_STAGES[i].id;
    const model = c.model === 'strong' ? '"claude-sonnet"' : '"claude-haiku"';
    const check = c.check ? VERIFIER_FN[id] : 'None';
    const retries = c.check ? c.retries : 0;
    return `    Stage("${id}", model=${model}, verify=${check}, max_retries=${retries}),`;
  });
  return [
    `# Expected cost: ${ev.expectedCost.toFixed(2)} cr/ticket · success: ${(100 * ev.pTask).toFixed(1)}% · silent: ${(100 * ev.pSilentAny).toFixed(1)}%`,
    `pipeline = [`,
    ...rows,
    `]`,
  ].join('\n');
}
