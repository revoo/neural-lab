// authored-by: Claude Opus 5.5
/**
 * Reference runs of the grokking demo (CPU Float32 tier, seeds 1–4, 6,000 full-batch steps, checked
 * every 100 steps), produced by `grok.run.ts`. `grokking.test.ts` re-runs seed 1 to verify.
 * memorised: first checkpoint with 100% training accuracy; grokked: first with ≥ 99% validation accuracy.
 */
export const GROK_RESULTS = {
  wd2: { memorised: [100, 100, 100, 100], grokked: [1400, 3000, 3200, 2700], finalVal: [1, 1, 1, 1], bestVal: [1, 1, 1, 1] },
  wd0: { memorised: [100, 100, 100, 100], grokked: [null, null, null, null], finalVal: [0.035, 0.042, 0.014, 0.035], bestVal: [0.042, 0.042, 0.014, 0.042] },
} as const;
