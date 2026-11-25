// authored-by: Claude Opus 5.5
/**
 * "Why this, and not something else?" — a card per rival, each judged on the
 * three tests every AI building block must pass. The rows are data computed
 * in each lesson's `rivals.ts` (and pinned by its test); this only displays them.
 */

export type Verdict = 'pass' | 'fail' | 'partly';

export type Rival = {
  /** Short name of the rival, e.g. "Divide by the sum". */
  name: string;
  /** What happens on the lesson's running example, with the computed numbers. */
  result: string;
  /** Verdicts on the three tests: right meaning, nudgeable by training, cheap at scale. */
  verdicts: [Verdict, Verdict, Verdict];
  /** The building block the lesson teaches. */
  winner?: boolean;
};

export const TESTS = ['Means the right thing', 'Training can nudge it', 'Cheap at scale'] as const;

const MARK: Record<Verdict, string> = { pass: '✓', fail: '✗', partly: '~' };
const WORD: Record<Verdict, string> = { pass: 'passes', fail: 'fails', partly: 'partly passes' };

export function RivalTable({ rivals, caption }: { rivals: Rival[]; caption?: string }) {
  return (
    <figure className="rival-table">
      {caption ? <figcaption>{caption}</figcaption> : null}
      <ol>
        {rivals.map((r) => (
          <li key={r.name} className={r.winner ? 'rival winner' : 'rival'}>
            <div className="rival-name">{r.name}</div>
            <p className="rival-result">{r.result}</p>
            <ul className="rival-verdicts" aria-label={`Verdicts for ${r.name}`}>
              {r.verdicts.map((v, i) => (
                <li key={TESTS[i]} className={`verdict ${v}`}>
                  <span aria-hidden="true">{MARK[v]}</span> {TESTS[i]}
                  <span className="sr-only"> — {WORD[v]}</span>
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ol>
    </figure>
  );
}
