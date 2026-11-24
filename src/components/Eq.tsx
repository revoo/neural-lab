import type { ReactNode } from 'react';
import { Tex } from './Tex';

/**
 * A key equation presented accessibly: an optional plain-English "word equation" on top,
 * the symbolic version below (with colour-matched symbols), and a legend explaining each symbol.
 * Children (usually a markdown list) form the legend.
 */
export function Eq({ tex, words, children, caption }: { tex: string; words?: string; children?: ReactNode; caption?: ReactNode }) {
  return (
    <figure className="eq">
      {words && (
        <div className="eq-words">
          <Tex block>{words}</Tex>
        </div>
      )}
      <div className="eq-tex">
        <Tex block>{tex}</Tex>
      </div>
      {children && <div className="eq-legend">{children}</div>}
      {caption && <figcaption className="eq-caption">{caption}</figcaption>}
    </figure>
  );
}
