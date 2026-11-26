import { ALL_LESSONS, inTrack, MODULES, trackProgress } from '../curriculum';
import { useProgress } from '../progress';
import { lessonHref } from '../router';
import { useTrack } from '../track';
import { OptionalPill, TrackToggle } from '../components/Track';

export function Home() {
  const p = useProgress();
  const track = useTrack();
  const overall = trackProgress(ALL_LESSONS, p.completed, track);
  const done = overall.done;
  const qs = Object.values(p.questions);
  const answered = qs.length;
  const accuracy = qs.reduce((s, q) => s + q.correct, 0) / Math.max(1, qs.reduce((s, q) => s + q.attempts, 0));
  const due = qs.filter((q) => q.due <= Date.now()).length;
  const nextLesson = ALL_LESSONS.find((l) => inTrack(l, track) && !p.completed[l.id]);
  return (
    <div className="page">
      <section className="hero">
        <h1>Understand AI from the inside out.</h1>
        <p>
          Neural Lab teaches machine learning the way you’d want a brilliant colleague to explain it: start with the problem, build a mental model, then
          get your hands on the real mechanism. Every diagram here computes real numbers. Drag things. Break things. Predict what happens before you find out.
        </p>
        <p style={{ fontSize: 17 }}>
          Written for software engineers and lifelong learners. No maths beyond school algebra assumed — every equation is introduced as a description of something you’ve
          already played with.
        </p>
        {nextLesson && (
          <div className="btn-row" style={{ marginTop: 18 }}>
            <a className="btn primary" href={lessonHref(nextLesson.id)} style={{ textDecoration: 'none' }}>
              {done === 0 ? 'Start with the prologue' : 'Continue'}: {nextLesson.title} →
            </a>
          </div>
        )}
      </section>
      <div className="stats">
        <div className="stat"><div className="v">{done}<span className="faint" style={{ fontSize: 16 }}> / {overall.total}</span></div><div className="l">lessons completed · ≈ {Math.round(overall.minutesLeft / 60)} h left</div></div>
        <div className="stat"><div className="v">{answered}</div><div className="l">questions attempted</div></div>
        <div className="stat"><div className="v">{answered ? `${Math.round(accuracy * 100)}%` : '—'}</div><div className="l">answer accuracy</div></div>
        <div className="stat"><div className="v">{due}</div><div className="l"><a href="#/review">reviews due →</a></div></div>
      </div>
      <div className="curriculum-head">
        <h2 style={{ fontSize: 22, margin: 0 }}>The curriculum</h2>
        <TrackToggle />
      </div>
      <p className="muted track-note">
        {track === 'full'
          ? `Full Course: all ${ALL_LESSONS.length} lessons, about ${Math.round(overall.minutes / 60)} hours. Lessons marked Optional are historical bridges and interludes; the Core Track skips them.`
          : `Core Track: ${overall.total} lessons, about ${Math.round(overall.minutes / 60)} hours. Optional lessons are greyed out and left out of progress and Next/Previous, but you can open any of them at any time.`}
      </p>
      <div className="module-grid">
        {MODULES.map((m) => {
          const mp = trackProgress(m.lessons, p.completed, track);
          const planned = m.lessons.length === 0;
          return (
            <div key={m.id} className={`module-card${planned ? ' planned' : ''}`}>
              <div className="n">Module {m.number}{m.era && <> · <span style={{ textTransform: 'none' }}>{m.era}</span></>} {planned && <span className="pill" style={{ marginLeft: 6 }}>in development</span>}</div>
              <h3>{m.title}</h3>
              <p>{m.blurb}</p>
              {!planned && (
                <>
                  <div className="progress-bar"><div style={{ width: `${(100 * mp.done) / Math.max(1, mp.total)}%` }} /></div>
                  <ul>
                    {m.lessons.map((l) => (
                      <li key={l.id} className={inTrack(l, track) ? '' : 'off-track'}>
                        <a href={lessonHref(l.id)}>
                          <span className={`dot${p.completed[l.id] ? ' done' : ''}`} />
                          <span>{l.title}</span>
                          <OptionalPill lesson={l} />
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
