import { useEffect, useMemo, useState } from 'react';
import { QuestionCard } from '../components/Quiz';
import type { Question } from '../components/quizTypes';
import { ALL_LESSONS, findLesson } from '../curriculum';
import { useProgress } from '../progress';
import { lessonHref } from '../router';

const quizModules = import.meta.glob<{ questions: Question[] }>('../lessons/*/quiz.ts');

type Bank = { lessonId: string; q: Question }[];

function useQuestionBank(): Bank | null {
  const [bank, setBank] = useState<Bank | null>(null);
  useEffect(() => {
    Promise.all(
      Object.entries(quizModules).map(async ([path, load]) => {
        const lessonId = path.split('/').at(-2)!;
        const mod = await load();
        return mod.questions.map((q) => ({ lessonId, q }));
      }),
    ).then((all) => setBank(all.flat()));
  }, []);
  return bank;
}

export function ReviewPage() {
  const bank = useQuestionBank();
  const progress = useProgress();
  const [mode, setMode] = useState<'due' | 'mixed' | null>(null);
  const [session, setSession] = useState<Bank>([]);
  const [i, setI] = useState(0);
  const [answered, setAnswered] = useState(false);

  const now = Date.now();
  const stats = useMemo(() => {
    if (!bank) return null;
    const seen = bank.filter((b) => progress.questions[b.q.id]);
    const due = seen.filter((b) => progress.questions[b.q.id].due <= now);
    const missed = seen.filter((b) => !progress.questions[b.q.id].lastCorrect);
    const completedLessons = new Set(ALL_LESSONS.filter((l) => progress.completed[l.id]).map((l) => l.id));
    const fromCompleted = bank.filter((b) => completedLessons.has(b.lessonId) || progress.questions[b.q.id]);
    return { seen, due, missed, fromCompleted, total: bank.length };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bank, progress, mode]);

  const start = (m: 'due' | 'mixed') => {
    if (!stats) return;
    const pool = m === 'due' ? stats.due : stats.fromCompleted;
    const shuffled = [...pool].sort(() => Math.random() - 0.5).slice(0, 10);
    setSession(shuffled);
    setI(0);
    setAnswered(false);
    setMode(m);
  };

  if (!bank || !stats) return <div className="page"><p className="muted">Loading question bank…</p></div>;

  if (mode && session.length) {
    const cur = session[i];
    const lesson = findLesson(cur.lessonId);
    return (
      <div className="page">
        <div className="btn-row" style={{ justifyContent: 'space-between' }}>
          <span className="muted" style={{ fontSize: 14 }}>
            {mode === 'due' ? 'Due review' : 'Mixed practice'} · {i + 1} of {session.length} · from <a href={lessonHref(cur.lessonId)}>{lesson?.title}</a>
          </span>
          <button className="btn small" onClick={() => setMode(null)}>End session</button>
        </div>
        <QuestionCard key={cur.q.id + i} q={cur.q} onAnswered={() => setAnswered(true)} />
        <div className="btn-row">
          {i < session.length - 1 ? (
            <button className="btn primary" disabled={!answered} onClick={() => { setI(i + 1); setAnswered(false); }}>Next question →</button>
          ) : (
            <button className="btn primary" disabled={!answered} onClick={() => setMode(null)}>Finish session</button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <h1 style={{ marginTop: 0 }}>Review & practice</h1>
      <p style={{ fontFamily: 'var(--font-prose)', fontSize: 18 }}>
        Understanding fades unless you use it. Every question you answer in a lesson is scheduled for review: get it right and it comes back
        after a longer gap (1, 3, 8, then 21 days); get it wrong and it returns at your next session. Numerical questions generate fresh numbers
        each time, so you can’t just remember the answer.
      </p>
      <div className="stats">
        <div className="stat"><div className="v">{stats.due.length}</div><div className="l">due now</div></div>
        <div className="stat"><div className="v">{stats.seen.length}<span className="faint" style={{ fontSize: 16 }}> / {stats.total}</span></div><div className="l">questions seen</div></div>
        <div className="stat"><div className="v">{stats.missed.length}</div><div className="l">last answered wrong</div></div>
      </div>
      <div className="btn-row">
        <button className="btn primary" disabled={!stats.due.length} onClick={() => start('due')}>
          Review {stats.due.length} due question{stats.due.length === 1 ? '' : 's'}
        </button>
        <button className="btn" disabled={!stats.fromCompleted.length} onClick={() => start('mixed')}>
          Mixed practice across lessons you’ve studied
        </button>
      </div>
      {!stats.seen.length && (
        <p className="muted" style={{ marginTop: 24 }}>
          Nothing to review yet. Answer the questions at the end of a lesson and they’ll show up here.
        </p>
      )}
    </div>
  );
}
