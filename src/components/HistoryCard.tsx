// authored-by: Claude Opus 5.5
import { claimsOf, historyById, type Claim } from '../history';

/**
 * The history of a lab's idea, collapsed by default and placed after the lab:
 * the learner feels the problem first, then learns who else did and when.
 * Every sentence carries a numbered link to its source (data in src/history.ts).
 */
export function HistoryCard({ id }: { id: string }) {
  const h = historyById(id);
  if (!h) return null;
  const sources: string[] = [];
  for (const c of claimsOf(h)) if (!sources.includes(c.source)) sources.push(c.source);
  const Cite = ({ c }: { c: Claim }) => {
    const n = sources.indexOf(c.source) + 1;
    return (
      <>
        {c.text}
        <sup>
          <a href={c.source} target="_blank" rel="noreferrer" aria-label={`source ${n}`}>
            [{n}]
          </a>
        </sup>
      </>
    );
  };
  return (
    <details className="deeper history-card">
      <summary>
        <span style={{ flex: 1 }}>
          History: {h.idea} <span className="history-when">{h.when}</span>
        </span>
        <span className="tag">History</span>
      </summary>
      <div className="deeper-body">
        <p className="history-who">
          {h.who.map((c, i) => (
            <span key={i}>
              {i > 0 && ' '}
              <Cite c={c} />
            </span>
          ))}
        </p>
        {h.mood && (
          <p className="history-mood">
            <strong>The mood of the moment.</strong> <Cite c={h.mood} />
          </p>
        )}
        <ol className="history-story">
          <li>
            <span className="history-step">Before</span> <Cite c={h.story.before} />
          </li>
          <li>
            <span className="history-step">This</span> <Cite c={h.story.this} />
          </li>
          <li>
            <span className="history-step">After</span> <Cite c={h.story.after} />
          </li>
        </ol>
        {h.dispute && (
          <p className="history-dispute">
            <strong>Disputed.</strong> <Cite c={h.dispute} />
          </p>
        )}
      </div>
    </details>
  );
}
