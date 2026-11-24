import { Component, lazy, Suspense, useMemo, type ComponentType, type ReactNode } from 'react';
import { neighbours, type LessonMeta, type ModuleMeta } from '../curriculum';
import { progressStore, useProgress } from '../progress';
import { lessonHref } from '../router';

const cache = new Map<string, ComponentType>();
function lessonComponent(l: LessonMeta): ComponentType {
  if (!cache.has(l.id)) cache.set(l.id, lazy(l.load!));
  return cache.get(l.id)!;
}

export function LessonPage({ lesson }: { lesson: LessonMeta & { module: ModuleMeta } }) {
  const progress = useProgress();
  const Content = useMemo(() => lessonComponent(lesson), [lesson]);
  const { prev, next } = neighbours(lesson.id);
  const done = !!progress.completed[lesson.id];
  const idx = lesson.module.lessons.findIndex((l) => l.id === lesson.id) + 1;
  return (
    <div className="page">
      <header className="lesson-header">
        <div className="lesson-kicker">
          Module {lesson.module.number} · {lesson.module.title} · Lesson {idx}
        </div>
        <h1>{lesson.title}</h1>
        <p className="summary">{lesson.summary}</p>
        <div className="lesson-meta">
          <span>≈ {lesson.minutes} min</span>
          {done && <span className="c-good">✓ Completed</span>}
        </div>
      </header>
      <ErrorBoundary>
        <Suspense fallback={<p className="muted">Loading lesson…</p>}>
          <article className="prose">
            <Content />
          </article>
        </Suspense>
      </ErrorBoundary>
      <footer className="lesson-footer">
        <div className="complete-box">
          <div>
            <strong>{done ? 'Lesson completed.' : 'Finished this lesson?'}</strong>
            <div className="muted" style={{ fontSize: 14 }}>
              {done ? 'Questions you answered will come back in Review when they’re due.' : 'Mark it complete to track your progress.'}
            </div>
          </div>
          <button className={`btn${done ? '' : ' primary'}`} onClick={() => progressStore.setCompleted(lesson.id, !done)}>
            {done ? 'Mark as not complete' : '✓ Mark complete'}
          </button>
        </div>
        <nav className="pager">
          {prev && (
            <a href={lessonHref(prev.id)}>
              <div className="dir">← Previous</div>
              {prev.title}
            </a>
          )}
          {next && (
            <a className="next" href={lessonHref(next.id)}>
              <div className="dir">Next →</div>
              {next.title}
            </a>
          )}
        </nav>
      </footer>
    </div>
  );
}

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error)
      return (
        <div className="callout misconception">
          <div className="callout-title">This lesson failed to load</div>
          <p className="mono" style={{ fontSize: 13 }}>{this.state.error.message}</p>
        </div>
      );
    return this.props.children;
  }
}
