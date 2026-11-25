// authored-by: Claude Opus 5.5
import type { ReactNode } from 'react';
import { findLesson } from '../curriculum';
import { lessonHref } from '../router';

/** End-of-lesson card: where this maths is used next, with a link. */
export function Bridge({ to, children }: { to: string; children: ReactNode }) {
  const l = findLesson(to);
  return (
    <aside className="callout bridge">
      <div className="callout-title">
        Where you’ll use this next
        {l ? (
          <>
            {' · '}
            <a href={lessonHref(to)}>{l.title} →</a>
          </>
        ) : null}
      </div>
      {children}
    </aside>
  );
}
