// authored-by: Claude Opus 5.5
/**
 * Lab: Steer the sentence. The trained tiny GPT continues "There was once" greedily; force characters
 * (each costs one) until the target word appears in the first WINDOW characters. Optimum from
 * steer.run.ts (every plan with up to two forcings, then "spell it out").
 */
import { useEffect, useMemo, useState } from 'react';
import { LabShell, useLabProgress, type LabBrief } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { Decoder, Gpt, sampleDistribution } from '../../lib/compute/gpt';
import { CPU_RUN } from '../../lib/compute/gptReference';
import { CHARS, encode } from '../../lib/corpora/grimm';
import { steer, STEER_PROMPT, WINDOW, type Forcing } from './steer';
import { STEER_RESULTS } from './steerResults';
import { loadFinalParams } from './Widgets';

type Rec = { target: string; cost: number; plan: Forcing[]; exact: boolean; plansTried: number };
const RESULTS = STEER_RESULTS as unknown as Rec[];
const KEY = 'learning-to-speak-steer';
const show = (c: string) => (c === ' ' ? '␣' : c === '\n' ? '↵' : c);
const BRIEFS: LabBrief[] = RESULTS.map((r, i) => ({
  id: `t${i}`,
  tab: `${i + 1} · “${r.target.trim()}”`,
  title: `Make it write “${r.target.trim()}”`,
  text: (
    <>
      Left alone, the model continues “{STEER_PROMPT}” with its most likely character every time. Force characters (click a position, pick a character) until “{r.target}” (with
      its leading space) appears within the first {WINDOW} characters. Every forced character costs one. Optional goal: as few as the best plan the search found.
    </>
  ),
}));

/** The model's top choices at position `pos`, given the forcings before it. */
function choicesAt(model: Gpt, forced: Forcing[], pos: number) {
  const text = steer(model, STEER_PROMPT, forced, pos);
  const dec = new Decoder(model);
  const logits = dec.prefill(encode(STEER_PROMPT + text));
  const p = sampleDistribution(logits, 1, 1);
  return Array.from(p, (v, k) => ({ ch: CHARS[k], p: v }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 8);
}

export function SteerLab() {
  const [model, setModel] = useState<Gpt | null>(null);
  useEffect(() => {
    let ok = true;
    loadFinalParams().then((p) => {
      if (!ok) return;
      const m = new Gpt(CPU_RUN.cfg);
      m.load(p);
      setModel(m);
    });
    return () => {
      ok = false;
    };
  }, []);
  const progress = useLabProgress(KEY);
  const [active, setActive] = useState(0);
  const [forced, setForced] = useState<Forcing[][]>(() => {
    try {
      const s = JSON.parse(localStorage.getItem(KEY + '-plans') ?? '');
      if (Array.isArray(s) && s.length === BRIEFS.length) return s;
    } catch {
      /* ignore */
    }
    return BRIEFS.map(() => []);
  });
  useEffect(() => {
    try {
      localStorage.setItem(KEY + '-plans', JSON.stringify(forced));
    } catch {
      /* ignore */
    }
  }, [forced]);
  const [pos, setPos] = useState<number | null>(null);
  const [typed, setTyped] = useState('');
  const log = useMachineLog();
  const rec = RESULTS[active];
  const mine = forced[active];
  const out = useMemo(() => (model ? steer(model, STEER_PROMPT, mine) : ''), [model, mine]);
  const hit = out.indexOf(rec.target);
  const passing = hit >= 0;
  const optimal = passing && mine.length <= rec.cost;
  const choices = useMemo(() => (model && pos !== null ? choicesAt(model, mine, pos) : []), [model, mine, pos]);

  const setPlan = (next: Forcing[]) => {
    next = [...next].sort((a, b) => a.pos - b.pos);
    setForced(forced.map((f, i) => (i === active ? next : f)));
    if (model) {
      const o = steer(model, STEER_PROMPT, next);
      log.push(`${next.length} forced: ${next.map((f) => `${show(f.ch)}@${f.pos + 1}`).join(', ') || 'none'} → “${o}”${o.includes(rec.target) ? ' (target written)' : ''}`, o.includes(rec.target) ? 'good' : 'info');
    }
  };
  const force = (ch: string) => {
    if (pos === null || !model) return;
    const natural = steer(model, STEER_PROMPT, mine.filter((f) => f.pos < pos), pos + 1)[pos];
    const rest = mine.filter((f) => f.pos !== pos);
    setPlan(ch === natural ? rest : [...rest, { pos, ch }]);
    setPos(null);
    setTyped('');
  };

  return (
    <LabShell
      title="Lab: Steer the sentence"
      caption={
        <>
          The capstone's trained tiny GPT (the recorded run's final weights), writing greedily: always its single most likely next character, so the same forcings
          always give the same text. Click any character of the continuation to see what the model was choosing between there, and force one. A forced character
          only changes what comes after it.
        </>
      }
      briefs={BRIEFS}
      active={active}
      onSelect={(i) => (setActive(i), setPos(null))}
      progress={progress}
      passing={passing}
      optimal={optimal}
      status={
        !model
          ? 'Loading the trained model…'
          : passing
            ? `It writes “${rec.target}” with ${mine.length} forced character${mine.length === 1 ? '' : 's'}.${optimal ? (rec.exact ? ' That is the fewest possible.' : ' That matches the best the search found.') : ` The search found a plan with ${rec.cost}.`}${mine.length < rec.cost ? ' You beat the search: it only forced lower-case letters and spaces.' : ''}`
            : `${mine.length} forced character${mine.length === 1 ? '' : 's'}; “${rec.target}” is not in the first ${WINDOW} characters yet.`
      }
      hint={
        model && !passing ? (
          <>Click positions near the start and look at the model's second and third choices there: a forcing early on changes everything after it, and sometimes the model finishes a word by itself once it has the first letter.</>
        ) : undefined
      }
      reveal={{
        label: 'the search’s best plan',
        content: (
          <div>
            <p style={{ marginTop: 0 }}>
              {rec.cost <= 2
                ? `The search tried every plan with no forcing, then one, then two (${rec.plansTried.toLocaleString('en-GB')} plans in all) and stopped at the first that works: ${rec.cost} is the fewest possible with lower-case letters and spaces.`
                : rec.exact
                  ? `No plan with one or two forcings works (${rec.plansTried.toLocaleString('en-GB')} plans tried), and this one uses three: so three is the fewest possible with lower-case letters and spaces.`
                  : `No plan with one or two forcings works (${rec.plansTried.toLocaleString('en-GB')} plans tried). Spelling the word out, forcing only the letters the model wouldn't write itself, takes ${rec.cost}. So the minimum is 3 or ${rec.cost}; the search did not try every plan with three.`}
            </p>
            {model && (
              <div className="mono gpt-sample">
                {STEER_PROMPT}
                <b>{steer(model, STEER_PROMPT, rec.plan)}</b>
                <div className="plot-tick">forced: {rec.plan.map((f) => `“${show(f.ch)}” at position ${f.pos + 1}`).join(', ')}</div>
              </div>
            )}
          </div>
        ),
      }}
      log={{ lines: log.lines, onClear: log.clear, title: 'steer.log', empty: 'Each change to your plan is logged here.' }}
      actions={
        <button className="btn" onClick={() => setPlan([])}>
          Clear forcings
        </button>
      }
    >
      {model && (
        <div className="gpt-panel">
          <div className="mono" style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', fontSize: 16, lineHeight: 2 }}>
            <span style={{ color: 'var(--text-muted)', marginRight: 4 }}>{STEER_PROMPT}</span>
            {[...out].map((c, i) => {
              const f = mine.some((x) => x.pos === i);
              const inTarget = hit >= 0 && i >= hit && i < hit + rec.target.length;
              return (
                <button
                  key={i}
                  onClick={() => (setPos(i), setTyped(''))}
                  title={`position ${i + 1}${f ? ' (forced)' : ''}`}
                  style={{
                    border: 'none',
                    font: 'inherit',
                    padding: '0 1px',
                    minWidth: 11,
                    cursor: 'pointer',
                    color: f ? 'var(--c-b)' : inTarget ? 'var(--c-c)' : 'var(--text)',
                    fontWeight: f || inTarget ? 700 : 400,
                    background: i === pos ? 'var(--accent-soft)' : 'transparent',
                    textDecoration: f ? 'underline' : undefined,
                  }}
                >
                  {show(c)}
                </button>
              );
            })}
          </div>
          <div className="plot-tick">forced characters are underlined; the target, once written, is highlighted</div>
          {pos !== null && (
            <div style={{ marginTop: 8 }}>
              <div className="field-label" style={{ textTransform: 'none' }}>Position {pos + 1}: what the model was choosing between</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 4 }}>
                {choices.map((c) => (
                  <button key={c.ch} className="btn" style={{ fontSize: 13 }} onClick={() => force(c.ch)}>
                    <span className="mono">{show(c.ch)}</span> {(100 * c.p).toFixed(1)}%
                  </button>
                ))}
                <input
                  className="text-input"
                  value={typed}
                  maxLength={1}
                  placeholder="any"
                  aria-label="Force any character"
                  style={{ width: 54 }}
                  onChange={(e) => {
                    const ch = e.target.value;
                    setTyped(ch);
                    if (ch && CHARS.includes(ch)) force(ch);
                  }}
                />
                <button className="btn" onClick={() => setPos(null)}>
                  cancel
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </LabShell>
  );
}
