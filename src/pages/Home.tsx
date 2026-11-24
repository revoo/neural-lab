import { ALL_LESSONS, MODULES } from '../curriculum';
import { useProgress } from '../progress';
import { lessonHref } from '../router';

export function Home() {
  const p = useProgress();
  const done = ALL_LESSONS.filter((l) => p.completed[l.id]).length;
  const qs = Object.values(p.questions);
  const answered = qs.length;
  const accuracy = qs.reduce((s, q) => s + q.correct, 0) / Math.max(1, qs.reduce((s, q) => s + q.attempts, 0));
  const due = qs.filter((q) => q.due <= Date.now()).length;
  const nextLesson = ALL_LESSONS.find((l) => !p.completed[l.id]);
  return (
    <div className="page">
      <section className="hero">
        <h1>Understand AI from the inside out.</h1>
        <p>
          Neural Lab teaches machine learning the way you’d want a brilliant colleague to explain it: start with the problem, build a mental model, then
          get your hands on the real mechanism. Every diagram here computes real numbers. Drag things. Break things. Predict what happens before you find out.
        </p>
        <p style={{ fontSize: 17 }}>
          Written for software engineers. No prior maths beyond algebra assumed — every equation is introduced as a description of something you’ve
          already played with.
        </p>
        {nextLesson && (
          <div className="btn-row" style={{ marginTop: 18 }}>
            <a className="btn primary" href={lessonHref(nextLesson.id)} style={{ textDecoration: 'none' }}>
              {done === 0 ? 'Start with lesson 1' : 'Continue'}: {nextLesson.title} →
            </a>
          </div>
        )}
      </section>
      <div className="stats">
        <div className="stat"><div className="v">{done}<span className="faint" style={{ fontSize: 16 }}> / {ALL_LESSONS.length}</span></div><div className="l">lessons completed</div></div>
        <div className="stat"><div className="v">{answered}</div><div className="l">questions attempted</div></div>
        <div className="stat"><div className="v">{answered ? `${Math.round(accuracy * 100)}%` : '—'}</div><div className="l">answer accuracy</div></div>
        <div className="stat"><div className="v">{due}</div><div className="l"><a href="#/review">reviews due →</a></div></div>
      </div>
      <h2 style={{ fontSize: 22, marginTop: 36 }}>The curriculum</h2>
      <div className="module-grid">
        {MODULES.map((m) => {
          const mdone = m.lessons.filter((l) => p.completed[l.id]).length;
          const planned = m.lessons.length === 0;
          return (
            <div key={m.id} className={`module-card${planned ? ' planned' : ''}`}>
              <div className="n">Module {m.number} {planned && <span className="pill" style={{ marginLeft: 6 }}>in development</span>}</div>
              <h3>{m.title}</h3>
              <p>{m.blurb}</p>
              {!planned && (
                <>
                  <div className="progress-bar"><div style={{ width: `${(100 * mdone) / m.lessons.length}%` }} /></div>
                  <ul>
                    {m.lessons.map((l) => (
                      <li key={l.id}>
                        <a href={lessonHref(l.id)}>
                          <span className={`dot${p.completed[l.id] ? ' done' : ''}`} />
                          {l.title}
                        </a>
                      </li>
                    ))}
                  </ul>
                </>
              )}
              {planned && m.planned && <p className="faint" style={{ fontSize: 13 }}>Planned: {m.planned.join(' · ')}</p>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
