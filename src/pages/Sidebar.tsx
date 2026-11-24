import { useState } from 'react';
import { MODULES } from '../curriculum';
import { useProgress } from '../progress';
import { lessonHref, type Route } from '../router';

export function Sidebar({ route, open }: { route: Route; open: boolean }) {
  const progress = useProgress();
  const activeLesson = route.page === 'lesson' ? route.id : '';
  const [expanded, setExpanded] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(MODULES.map((m) => [m.id, m.lessons.length > 0])),
  );
  return (
    <nav className={`sidebar${open ? ' open' : ''}`} aria-label="Curriculum">
      <a className="brand" href="#/">
        <BrandMark />
        Neural Lab
      </a>
      <div className="nav-top">
        <a className={`nav-link${route.page === 'home' ? ' active' : ''}`} href="#/">⌂ Overview</a>
        <a className={`nav-link${route.page === 'review' ? ' active' : ''}`} href="#/review">↻ Review & practice</a>
        <a className={`nav-link${route.page === 'glossary' ? ' active' : ''}`} href="#/glossary">≡ Glossary</a>
      </div>
      {MODULES.map((m) => {
        const done = m.lessons.filter((l) => progress.completed[l.id]).length;
        const planned = m.lessons.length === 0;
        return (
          <div key={m.id} className={`nav-module${planned ? ' planned' : ''}`}>
            <button className="nav-module-head" onClick={() => setExpanded({ ...expanded, [m.id]: !expanded[m.id] })} aria-expanded={!!expanded[m.id]}>
              <span className="num">{m.number}</span>
              <span className="title">{m.title}</span>
              <span className="count">{planned ? 'soon' : `${done}/${m.lessons.length}`}</span>
            </button>
            {expanded[m.id] && (
              <ul className="nav-lessons">
                {m.lessons.map((l) => (
                  <li key={l.id}>
                    <a href={lessonHref(l.id)} className={activeLesson === l.id ? 'active' : ''}>
                      <span className={`dot${progress.completed[l.id] ? ' done' : ''}`} />
                      {l.title}
                    </a>
                  </li>
                ))}
                {(m.planned ?? []).map((t) => (
                  <li key={t}>
                    <span title="Planned">
                      <span className="dot" style={{ borderStyle: 'dashed' }} />
                      {t}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        );
      })}
    </nav>
  );
}

export function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden>
      <path d="M7 16L25 7M7 16L25 25M7 16L25 16" stroke="var(--border-strong)" strokeWidth="1.8" />
      <circle cx="7" cy="16" r="4.2" fill="var(--c-a)" />
      <circle cx="25" cy="7" r="3.4" fill="var(--c-b)" />
      <circle cx="25" cy="16" r="3.4" fill="var(--accent)" />
      <circle cx="25" cy="25" r="3.4" fill="var(--c-c)" />
    </svg>
  );
}
