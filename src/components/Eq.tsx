import { useEffect, useState, type ReactNode } from 'react';
import { Tex } from './Tex';
import { useMathMode } from '../mathMode';

/**
 * A key equation presented accessibly: an optional plain-English "word equation" on top,
 * the symbolic version below (with colour-matched symbols), and a legend explaining each symbol.
 * Children (usually a markdown list) form the legend.
 *
 * In `words-first` mode (when `words` is provided), the symbolic TeX starts collapsed behind a
 * compact 1-click "Show symbols" peek button while keeping the word equation and legend visible.
 */
export function Eq({ tex, words, children, caption }: { tex: string; words?: string; children?: ReactNode; caption?: ReactNode }) {
  const { mathMode } = useMathMode();
  const canCollapse = Boolean(words) && mathMode === 'words-first';
  const [peeked, setPeeked] = useState(false);

  useEffect(() => {
    if (!canCollapse) setPeeked(false);
  }, [canCollapse]);

  const showTex = !canCollapse || peeked;

  return (
    <figure className={`eq${canCollapse ? ' eq-words-first' : ''}`}>
      {words && (
        <div className="eq-words">
          <Tex block>{words}</Tex>
        </div>
      )}
      {canCollapse && (
        <div className="eq-peek-row">
          <button
            type="button"
            className="eq-peek-btn"
            aria-expanded={peeked}
            onClick={() => setPeeked((v) => !v)}
          >
            {peeked ? 'Hide symbols' : 'Show symbols'}
          </button>
        </div>
      )}
      {showTex && (
        <div className="eq-tex">
          <Tex block>{tex}</Tex>
        </div>
      )}
      {children && <div className="eq-legend">{children}</div>}
      {caption && <figcaption className="eq-caption">{caption}</figcaption>}
    </figure>
  );
}
