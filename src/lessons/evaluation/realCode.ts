// authored-by: Gemini Antigravity
import { GRADER_BY_ID, type EvalBrief, type EvalDesign, type evaluateEvalDesign } from './eval';

const GRADER_FN: Record<EvalDesign['grader'], string> = {
  human: 'exact_unit_test',
  judge2: 'dual_pass_llm_judge',
  judge1: 'fast_llm_judge',
};

export function evalSuiteRealCode(
  design: EvalDesign,
  brief: EvalBrief,
  ev: ReturnType<typeof evaluateEvalDesign>,
): string {
  const g = GRADER_BY_ID[design.grader];
  const fn = GRADER_FN[design.grader];
  const header = `# ${design.mode === 'paired' ? 'Paired McNemar' : 'Unpaired two-sample'} · ${g.shortLabel} · n=${design.n} · alpha=${design.alpha} -> power ${(100 * ev.power).toFixed(1)}%, cost ${ev.cost.toLocaleString()} cr`;
  if (design.mode === 'paired') {
    return [
      header,
      'from scipy.stats import binomtest',
      '',
      `scores_a, scores_b = grade_paired(dataset[:${design.n}], grader=${fn})  # q=${g.accuracy.toFixed(2)}`,
      'n10 = sum(a == 1 and b == 0 for a, b in zip(scores_a, scores_b))  # A wins',
      'n01 = sum(a == 0 and b == 1 for a, b in zip(scores_a, scores_b))  # regressions',
      'p_val = binomtest(n10, n10 + n01, p=0.5, alternative="greater").pvalue',
      `ship = p_val <= ${design.alpha}  # target: detect +${(brief.delta * 100).toFixed(1)} pp lift`,
    ].join('\n');
  }
  return [
    header,
    'from scipy.stats import fisher_exact',
    '',
    `k_a = sum(${fn}(cand_out) for cand_out in split_a[:${design.n}])  # q=${g.accuracy.toFixed(2)}`,
    `k_b = sum(${fn}(base_out) for base_out in split_b[:${design.n}])`,
    `_, p_val = fisher_exact([[k_a, ${design.n} - k_a], [k_b, ${design.n} - k_b]], alternative="greater")`,
    `ship = p_val <= ${design.alpha}  # target: detect +${(brief.delta * 100).toFixed(1)} pp lift`,
  ].join('\n');
}
