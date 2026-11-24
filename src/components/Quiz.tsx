import { useMemo, useState } from 'react';
import { createRng } from '../lib/rng';
import { progressStore } from '../progress';
import { RichText } from './RichText';
import type { ChoiceQuestion, NumericQuestion, OrderQuestion, Question } from './quizTypes';

export function Quiz({ questions, title = 'Check your understanding' }: { questions: Question[]; title?: string }) {
  return (
    <section className="quiz">
      <div className="quiz-head">
        <h3 style={{ fontFamily: 'var(--font-ui)' }}>{title}</h3>
      </div>
      {questions.map((q, i) => (
        <QuestionCard key={q.id} q={q} index={i + 1} total={questions.length} />
      ))}
    </section>
  );
}

export function QuestionCard({
  q,
  index,
  total,
  onAnswered,
}: {
  q: Question;
  index?: number;
  total?: number;
  onAnswered?: (correct: boolean) => void;
}) {
  const [recorded, setRecorded] = useState(false);
  const record = (correct: boolean) => {
    if (!recorded) {
      progressStore.recordAnswer(q.id, correct);
      setRecorded(true);
    }
    onAnswered?.(correct);
  };
  return (
    <div className="quiz-q">
      {index && <div className="quiz-num">Question {index}{total ? ` of ${total}` : ''}</div>}
      {q.kind === 'choice' && <Choice q={q} onAnswer={record} />}
      {q.kind === 'numeric' && <Numeric q={q} onAnswer={record} />}
      {q.kind === 'order' && <Order q={q} onAnswer={record} />}
    </div>
  );
}

function shuffled<T>(xs: T[], seed: number): T[] {
  const r = createRng(seed);
  const a = xs.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = r.int(i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function Choice({ q, onAnswer }: { q: ChoiceQuestion; onAnswer: (c: boolean) => void }) {
  const [seed] = useState(() => Math.floor(Math.random() * 1e9));
  const opts = useMemo(() => (q.shuffle === false ? q.options : shuffled(q.options, seed)), [q, seed]);
  const [picked, setPicked] = useState<number | null>(null);
  const done = picked !== null;
  const pickedOpt = done ? opts[picked] : null;
  return (
    <>
      <div className="quiz-prompt"><RichText text={q.prompt} /></div>
      <div className="options">
        {opts.map((o, i) => {
          const cls = !done ? '' : o.correct ? 'correct' : i === picked ? 'wrong' : '';
          return (
            <button
              key={i}
              type="button"
              className={`option ${cls}`}
              disabled={done}
              onClick={() => {
                setPicked(i);
                onAnswer(!!o.correct);
              }}
            >
              <span className="marker">{done && o.correct ? '✓' : done && i === picked ? '✕' : String.fromCharCode(65 + i)}</span>
              <span><RichText text={o.text} /></span>
            </button>
          );
        })}
      </div>
      {pickedOpt && (
        <div className={`feedback ${pickedOpt.correct ? 'good' : 'bad'}`}>
          <div className="feedback-title">{pickedOpt.correct ? 'Exactly right.' : 'Not quite.'}</div>
          {pickedOpt.why && <RichText block text={pickedOpt.why} />}
          {!pickedOpt.correct && opts.find((o) => o.correct)?.why && (
            <div style={{ marginTop: 8 }}>
              <RichText block text={opts.find((o) => o.correct)!.why!} />
            </div>
          )}
          {q.explain && <div style={{ marginTop: 8 }}><RichText block text={q.explain} /></div>}
          {!pickedOpt.correct && (
            <button className="btn small" style={{ marginTop: 10 }} onClick={() => setPicked(null)}>
              Try again
            </button>
          )}
        </div>
      )}
    </>
  );
}

function Numeric({ q, onAnswer }: { q: NumericQuestion; onAnswer: (c: boolean) => void }) {
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 1e9));
  const inst = useMemo(() => q.make(createRng(seed)), [q, seed]);
  const [input, setInput] = useState('');
  const [result, setResult] = useState<null | boolean>(null);
  const tol = q.tolerance ?? 0.01;
  const submit = () => {
    const v = Number(input.replace(',', '.').replace('−', '-'));
    if (input.trim() === '' || Number.isNaN(v)) return;
    const ok = Math.abs(v - inst.answer) <= tol + 1e-12;
    setResult(ok);
    onAnswer(ok);
  };
  return (
    <>
      <div className="quiz-prompt"><RichText text={inst.prompt} /></div>
      <form
        className="numeric-input"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <input value={input} onChange={(e) => setInput(e.target.value)} inputMode="decimal" placeholder="Your answer" disabled={result !== null} aria-label="Your answer" />
        {result === null && <button className="btn primary small" type="submit">Check</button>}
      </form>
      {result !== null && (
        <div className={`feedback ${result ? 'good' : 'bad'}`}>
          <div className="feedback-title">{result ? 'Correct.' : `Not quite — the answer is ${round(inst.answer)}.`}</div>
          <RichText block text={inst.explain} />
          <div className="btn-row" style={{ marginTop: 10 }}>
            {!result && (
              <button className="btn small" onClick={() => { setResult(null); setInput(''); }}>Try again</button>
            )}
            <button className="btn small" onClick={() => { setSeed(seed + 1); setResult(null); setInput(''); }}>New numbers</button>
          </div>
        </div>
      )}
    </>
  );
}

const round = (x: number) => String(Math.round(x * 1e4) / 1e4);

function Order({ q, onAnswer }: { q: OrderQuestion; onAnswer: (c: boolean) => void }) {
  const [seed] = useState(() => Math.floor(Math.random() * 1e9));
  const [items, setItems] = useState(() => {
    let s = shuffled(q.items, seed);
    if (s.every((x, i) => x === q.items[i])) s = [...s.slice(1), s[0]];
    return s;
  });
  const [checked, setChecked] = useState<null | boolean>(null);
  const move = (i: number, d: number) => {
    const j = i + d;
    if (j < 0 || j >= items.length) return;
    const a = items.slice();
    [a[i], a[j]] = [a[j], a[i]];
    setItems(a);
    setChecked(null);
  };
  return (
    <>
      <div className="quiz-prompt"><RichText text={q.prompt} /></div>
      <ol className="order-list">
        {items.map((it, i) => (
          <li key={it} className={`order-item ${checked === null ? '' : it === q.items[i] ? 'correct' : 'wrong'}`}>
            <span className="arrows">
              <button type="button" aria-label="Move up" onClick={() => move(i, -1)}>▲</button>
              <button type="button" aria-label="Move down" onClick={() => move(i, 1)}>▼</button>
            </span>
            <span className="mono faint" style={{ width: 18 }}>{i + 1}.</span>
            <span><RichText text={it} /></span>
          </li>
        ))}
      </ol>
      <div className="btn-row" style={{ marginTop: 10 }}>
        <button
          className="btn primary small"
          onClick={() => {
            const ok = items.every((x, i) => x === q.items[i]);
            setChecked(ok);
            onAnswer(ok);
          }}
        >
          Check order
        </button>
      </div>
      {checked !== null && (
        <div className={`feedback ${checked ? 'good' : 'bad'}`}>
          <div className="feedback-title">{checked ? 'Correct order.' : 'Some steps are out of place (marked in red). Rearrange and check again.'}</div>
          {checked && <RichText block text={q.explain} />}
        </div>
      )}
    </>
  );
}
