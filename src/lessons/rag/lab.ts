// authored-by: Claude Opus 5.5
/** Lab: Tune the Retriever. Every setting is evaluated exactly on all labelled questions. */
import { evaluate } from './rag';

export const LAB_SIZES = [1, 2, 3, 4, 5] as const;
export const LAB_LAMBDAS = [0, 0.25, 0.5, 0.75, 1] as const;
export const LAB_KS = [1, 2, 3, 4, 5] as const;

export type LabBrief = { id: 'everything' | 'tight' | 'nolex'; title: string; story: string; minRecall: number; fixedLambda?: number };

export const LAB_BRIEFS: LabBrief[] = [
  { id: 'everything', title: 'Brief 1 · Miss nothing', story: 'A support bot that must find the answer to every one of the 34 questions in its sources.', minRecall: 1 },
  { id: 'tight', title: 'Brief 2 · A small context window', story: 'A cheap model with little room in its prompt. Finding 90% of the answers is acceptable.', minRecall: 0.9 },
  {
    id: 'nolex',
    title: 'Brief 3 · Vectors only',
    story: 'Your vector database only does semantic search (λ = 1). Find at least 90% of the answers.',
    minRecall: 0.9,
    fixedLambda: 1,
  },
];

export function labExhaustive(b: LabBrief) {
  const rows: { size: number; lambda: number; k: number; recall: number; words: number }[] = [];
  for (const size of LAB_SIZES)
    for (const lambda of b.fixedLambda !== undefined ? [b.fixedLambda] : LAB_LAMBDAS)
      for (const k of LAB_KS) {
        const r = evaluate(size, lambda, k);
        rows.push({ size, lambda, k, recall: r.recall, words: Math.round(r.promptWords * 10) / 10 });
      }
  const pass = rows.filter((r) => r.recall >= b.minRecall - 1e-9);
  const bestWords = Math.min(...pass.map((r) => r.words));
  return { total: rows.length, passing: pass.length, bestWords, best: pass.filter((r) => r.words === bestWords) };
}
