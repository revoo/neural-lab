import { useEffect, useMemo, useState } from 'react';
import { GLOSSARY } from '../glossary';
import { findLesson } from '../curriculum';
import { glossaryHref, lessonHref } from '../router';

export function GlossaryPage({ term }: { term?: string }) {
  const [q, setQ] = useState('');
  const items = useMemo(() => {
    const s = q.trim().toLowerCase();
    const sorted = [...GLOSSARY].sort((a, b) => a.term.localeCompare(b.term));
    if (!s) return sorted;
    return sorted.filter((g) => g.term.toLowerCase().includes(s) || g.short.toLowerCase().includes(s) || g.aka?.some((a) => a.includes(s)));
  }, [q]);
  useEffect(() => {
    if (term) document.getElementById(`g-${term}`)?.scrollIntoView({ block: 'center' });
  }, [term]);
  return (
    <div className="page">
      <h1 style={{ marginTop: 0 }}>Glossary</h1>
      <p className="muted">Every term in plain language, with a link to the lesson where it’s properly built up.</p>
      <input className="search" placeholder={`Search ${GLOSSARY.length} concepts…`} value={q} onChange={(e) => setQ(e.target.value)} autoFocus />
      <div className="gloss-list">
        {items.map((g) => {
          const lesson = g.lesson ? findLesson(g.lesson) : undefined;
          return (
            <div key={g.id} id={`g-${g.id}`} className={`gloss-item${g.id === term ? ' hl' : ''}`}>
              <h3>{g.term}</h3>
              <p>{g.short}</p>
              <div className="links">
                {lesson && <a href={lessonHref(lesson.id)}>Lesson: {lesson.title}</a>}
                {g.related && (
                  <span className="faint">
                    {lesson ? ' · ' : ''}Related:{' '}
                    {g.related.map((r, i) => (
                      <span key={r}>
                        {i > 0 && ', '}
                        <a href={glossaryHref(r)}>{GLOSSARY.find((x) => x.id === r)?.term ?? r}</a>
                      </span>
                    ))}
                  </span>
                )}
              </div>
            </div>
          );
        })}
        {items.length === 0 && <p className="muted">No matches.</p>}
      </div>
    </div>
  );
}
