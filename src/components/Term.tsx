import { useState, type ReactNode } from 'react';
import { glossaryById } from '../glossary';
import { glossaryHref, lessonHref } from '../router';
import { findLesson, MODULES } from '../curriculum';

/** Inline glossary term: hover or focus for a plain-language definition. */
export function Term({ id, children }: { id: string; children?: ReactNode }) {
  const g = glossaryById[id];
  const [open, setOpen] = useState(false);
  if (!g) {
    if (import.meta.env.DEV) console.warn(`Unknown glossary term: ${id}`);
    return <>{children}</>;
  }
  const lesson = g.lesson ? findLesson(g.lesson) : undefined;
  return (
    <span
      className="term"
      tabIndex={0}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {children ?? g.term.toLowerCase()}
      {open && (
        <span className="term-pop" role="tooltip">
          <span className="t">{g.term}</span>
          {g.short}
          <br />
          {lesson && !location.hash.includes(`/learn/${lesson.id}`) ? (
            <a href={lessonHref(lesson.id)}>Learn it properly: {lesson.title} →</a>
          ) : (
            <a href={glossaryHref(g.id)}>Open in glossary →</a>
          )}
        </span>
      )}
    </span>
  );
}

/** Link to another lesson by id, using its title if no children are given. */
export function LessonLink({ to, children }: { to: string; children?: ReactNode }) {
  const l = findLesson(to);
  return <a href={lessonHref(to)}>{children ?? l?.title ?? to}</a>;
}

/** "Module 14 · Reinforcement Learning", linked to the module's first lesson. Renders the live number, so reordering modules never breaks prose. */
export function ModuleRef({ id, short = false, children }: { id: string; short?: boolean; children?: ReactNode }) {
  const m = MODULES.find((x) => x.id === id);
  if (!m) {
    if (import.meta.env.DEV) console.warn(`Unknown module: ${id}`);
    return <>{children ?? id}</>;
  }
  const label = children ?? (short ? `Module ${m.number}` : `Module ${m.number} · ${m.title}`);
  const first = m.lessons[0];
  return first ? <a href={lessonHref(first.id)}>{label}</a> : <>{label}</>;
}
