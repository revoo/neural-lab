// authored-by: Claude Opus 5.5
import { useMemo, useState } from 'react';
import { Segmented, Slider } from '../../components/controls';
import { LabShell, useLabProgress, type LabBrief } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { useReplay } from '../../components/motion';
import { ReplayControls } from '../../components/ReplayControls';
import {
  AND,
  AND_START,
  ETAS,
  OR,
  OR_START,
  RUN_CAP,
  THIN,
  THIN_TARGET,
  XOR,
  XOR_CAP,
  andSearch,
  bestMargin,
  novikoffBound,
  radius,
  run,
  thinSearch,
  type Pt,
  type RunResult,
  type W,
} from './perceptron';
import { PerceptronPlot, fmtN, fmtW } from './PerceptronPlot';

type BriefId = 'or' | 'and' | 'thin' | 'xor';
const BRIEFS: (LabBrief & { id: BriefId })[] = [
  {
    id: 'or',
    tab: '1 · OR',
    title: 'Teach it OR',
    text: <>The Mark I’s knobs are where the last experiment left them: computing AND. Retrain it to compute OR. Choose the order the four examples are shown in, the starting knob positions and the step size, then press Run.</>,
  },
  {
    id: 'and',
    tab: '2 · AND',
    title: 'AND, with as few mistakes as possible',
    text: <>The knobs start at a fixed factory setting, w = {fmtW(AND_START)}, which you can’t change. Teach the machine AND. Then, optionally, find an order and step size that gets there with the fewest mistakes.</>,
  },
  {
    id: 'thin',
    tab: '3 · Thin margin',
    title: 'One point too close',
    text: <>Eleven points, all weights starting at 0, step 1, shown in the order listed. A line exists that separates them, but the machine makes many mistakes finding it. Remove exactly one point so that it learns in {THIN_TARGET} mistakes or fewer. Optional: the fewest possible.</>,
  },
  {
    id: 'xor',
    tab: '4 · XOR',
    title: 'XOR',
    text: <>Teach it XOR. Any order, start and step size. The machine gives up after {XOR_CAP} mistakes; run it at least once to the end.</>,
  },
];

type Saved = { order: Record<string, number[]>; w0: Record<string, W>; eta: Record<string, number>; removed: number | null };
const KEY = 'neural-lab:perceptron-lab';
function load(): Saved {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    /* ignore */
  }
  return { order: {}, w0: {}, eta: {}, removed: null };
}

const dataOf = (b: BriefId): Pt[] => (b === 'or' ? OR : b === 'and' ? AND : b === 'xor' ? XOR : THIN);
const defaultW0 = (b: BriefId): W => (b === 'or' ? OR_START : b === 'and' ? AND_START : [0, 0, 0]);

function OrderEditor({ pts, order, onChange }: { pts: Pt[]; order: number[]; onChange: (o: number[]) => void }) {
  const move = (k: number, d: number) => {
    const o = order.slice();
    [o[k], o[k + d]] = [o[k + d], o[k]];
    onChange(o);
  };
  return (
    <div>
      <div className="field-label">Order of examples (then repeat)</div>
      <ol style={{ margin: 0, paddingLeft: 22, display: 'grid', gap: 4 }}>
        {order.map((i, k) => (
          <li key={i} style={{ fontSize: 13.5 }}>
            <span className="mono">{pts[i].name}</span> <span className="muted">→ {pts[i].y}</span>{' '}
            <button className="btn small" onClick={() => move(k, -1)} disabled={k === 0} aria-label={`Move ${pts[i].name} earlier`}>
              ↑
            </button>{' '}
            <button className="btn small" onClick={() => move(k, 1)} disabled={k === order.length - 1} aria-label={`Move ${pts[i].name} later`}>
              ↓
            </button>
          </li>
        ))}
      </ol>
    </div>
  );
}

export function MarkILab() {
  const [saved, setSaved] = useState<Saved>(load);
  const [bi, setBi] = useState(0);
  const progress = useLabProgress('perceptron-lab');
  const log = useMachineLog();
  const brief = BRIEFS[bi];
  const id = brief.id;
  const pts = dataOf(id);
  const order = id === 'thin' ? [] : (saved.order[id] ?? pts.map((_, i) => i));
  const w0 = saved.w0[id] ?? defaultW0(id);
  const eta = id === 'thin' ? 1 : (saved.eta[id] ?? 1);
  const removed = id === 'thin' ? saved.removed : null;
  const settingsKey = JSON.stringify([id, order, w0, eta, removed]);
  const [result, setResult] = useState<{ key: string; r: RunResult; pts: Pt[] } | null>(null);
  const current = result && result.key === settingsKey ? result : null;

  const persist = (next: Saved) => {
    setSaved(next);
    try {
      localStorage.setItem(KEY, JSON.stringify(next));
    } catch {
      /* ignore */
    }
  };

  const doRun = () => {
    let r: RunResult;
    let used = pts;
    if (id === 'thin') {
      used = removed === null ? THIN : THIN.filter((_, i) => i !== removed);
      r = run(used, used.map((_, i) => i), [0, 0, 0], 1, 1000, true);
    } else {
      r = run(pts, order, w0, eta, id === 'xor' ? XOR_CAP : RUN_CAP, true);
    }
    setResult({ key: settingsKey, r, pts: used });
    const what = id === 'thin' ? (removed === null ? 'all 11 points' : `${THIN[removed].name} removed`) : `order ${order.map((i) => pts[i].name).join(' ')}, start ${fmtW(w0)}, step ${eta}`;
    log.push(
      r.converged
        ? `Run (${what}): learned after ${r.mistakes} mistake${r.mistakes === 1 ? '' : 's'} in ${r.looks} looks. Final w = ${fmtW(r.w)}.`
        : `Run (${what}): gave up after ${r.mistakes} mistakes, still wrong somewhere. w = ${fmtW(r.w)}.`,
      r.converged ? 'good' : 'bad',
    );
  };

  const mistakes = useMemo(() => (current ? current.r.events.filter((e) => e.mistake) : []), [current]);
  const replay = useReplay(mistakes.length, Math.max(60, Math.min(300, 6000 / Math.max(1, mistakes.length))), current?.key + String(mistakes.length));
  const shownW: W = current ? (replay.i === 0 ? (id === 'thin' ? [0, 0, 0] : w0) : mistakes[replay.i - 1].after) : w0;
  const shownFocus = current && replay.i > 0 ? (id === 'thin' ? THIN.indexOf(current.pts[mistakes[replay.i - 1].i]) : mistakes[replay.i - 1].i) : undefined;

  // Live margin readout for the thin brief.
  const thinPts = removed === null ? THIN : THIN.filter((_, i) => i !== removed);
  const margin = useMemo(() => (id === 'thin' ? { g: bestMargin(thinPts).gamma, R: radius(thinPts), bound: novikoffBound(thinPts) } : null), [id, removed]); // eslint-disable-line react-hooks/exhaustive-deps

  const passing = !!current && (id === 'xor' ? current.r.mistakes >= XOR_CAP : id === 'thin' ? removed !== null && current.r.converged && current.r.mistakes <= THIN_TARGET : current.r.converged);
  const andS = useMemo(() => andSearch(), []);
  const thinS = useMemo(() => thinSearch(), []);
  const optimal = passing && ((id === 'and' && current!.r.mistakes === andS.min) || (id === 'thin' && current!.r.mistakes === thinS.min));

  const status = !current
    ? 'Press Run to train with these settings.'
    : id === 'xor'
      ? `Gave up after ${current.r.mistakes} mistakes, with w = ${fmtW(current.r.w)}. Scrub the replay: the line keeps swinging between the same few positions.`
      : current.r.converged
        ? id === 'thin' && removed === null
          ? `Learned, but only after ${current.r.mistakes} mistakes. Remove one point to make it faster.`
          : id === 'thin' && current.r.mistakes > THIN_TARGET
            ? `Learned after ${current.r.mistakes} mistakes: still more than ${THIN_TARGET}. Removing this point didn’t widen the gap enough.`
            : `Learned after ${current.r.mistakes} mistake${current.r.mistakes === 1 ? '' : 's'}.${id === 'and' && !optimal ? ' Fewer is possible.' : ''}${optimal ? ' That is the fewest possible.' : ''}`
        : `Still making mistakes after ${current.r.mistakes}.`;

  const hint =
    id === 'or' ? (
      <>Watch the log after a run: which examples caused the mistakes? The starting knobs already compute AND, so only examples where AND and OR disagree can be wrong at first.</>
    ) : id === 'and' ? (
      <>Every mistake moves the knobs by step × input. Compare the factory setting with the knobs at the end of a run that worked: how far did each knob have to travel, and how far does one mistake move it? Scrub the replay to see which example made each move, and which moves were later undone.</>
    ) : id === 'thin' ? (
      <>Watch the margin readout as you remove different points. The number of mistakes the perceptron can make is limited by the narrowest gap between the two groups.</>
    ) : (
      <>Read the log after a run, and look at the weights after each mistake. Do they ever come back to a value they had before?</>
    );

  const reveal =
    id === 'and'
      ? {
          label: 'the full search',
          content: (
            <>
              All {andS.total} combinations (24 orders × 3 step sizes) were run from the factory setting. Every one learns AND, using between {andS.min} and {andS.worst} mistakes. The fewest, {andS.min}, is reached by {andS.best.length}:{' '}
              {andS.best.map((b, k) => (
                <span key={k}>
                  {k ? ' and ' : ''}step {b.eta} with order <span className="mono">{b.order.map((i) => AND[i].name).join(' → ')}</span>
                </span>
              ))}
              . Look closely: they are the same cycle, started at different places. Because the examples repeat, what matters is the cycle, not where it starts.
            </>
          ),
        }
      : id === 'thin'
        ? {
            label: 'every single removal',
            content: (
              <>
                With all 11 points: {thinS.full} mistakes. With one removed:{' '}
                {thinS.each.map((e) => `${e.name} ${e.mistakes}`).join(', ')}. Only {thinS.passing.map((e) => e.name).join(', ')} get to {THIN_TARGET} or fewer, and only removing K reaches {thinS.min}, the same as the cloud without K. K sits just across the line from H: it forces the separating line into a narrow gap.
              </>
            ),
          }
        : id === 'xor'
          ? {
              label: 'why XOR is impossible',
              content: (
                <>
                  No straight line puts (0, 1) and (1, 0) on one side and (0, 0) and (1, 1) on the other: the two pairs cross in the middle. So every setting of the knobs gets at least one example wrong, and the rule always has something to fix. Each fix to one corner breaks another, and the weights go round in a cycle for ever. The hand-wired XOR network from the logic lab works, but it needs hidden neurons, and this rule can’t train them: it only knows what the <em>output</em> should have been, and nothing tells a hidden neuron what <em>it</em> should have said.
                </>
              ),
            }
          : undefined;

  const setOrder = (o: number[]) => persist({ ...saved, order: { ...saved.order, [id]: o } });
  const setW0 = (k: number, v: number) => {
    const n: W = [...w0];
    n[k] = v;
    persist({ ...saved, w0: { ...saved.w0, [id]: n } });
  };

  return (
    <LabShell
      title="Lab: Train the Mark I"
      caption={
        <>
          The plot shows the examples and the machine’s line; the shaded side is where it says 1, and ringed points are wrong. After a run, the replay steps through every mistake, and the machine log lists each run. Your settings are saved in this browser.
        </>
      }
      briefs={BRIEFS}
      active={bi}
      onSelect={(i) => setBi(i)}
      progress={progress}
      passing={passing}
      optimal={optimal}
      status={status}
      statusTone={current && !passing ? 'bad' : undefined}
      hint={hint}
      reveal={reveal}
      log={{ lines: log.lines, onClear: log.clear, empty: 'Each run is logged here.' }}
      actions={
        <button className="btn primary" onClick={doRun}>
          ▶ Run
        </button>
      }
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(290px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <div>
          <PerceptronPlot pts={id === 'thin' ? THIN : pts} w={shownW} focus={shownFocus} faded={removed !== null ? new Set([removed]) : undefined} label="The examples and the Mark I's current line" />
          {current && mistakes.length > 0 && <ReplayControls r={replay} label="mistake" />}
          <div className="readout" style={{ marginTop: 6 }}>
            <span className="k">w₁, w₂, b </span>
            {fmtW(shownW)}
            {current && (
              <>
                {'\n'}
                <span className="k">mistakes  </span>
                {current.r.mistakes} <span className="k">in</span> {current.r.looks} <span className="k">looks</span>
              </>
            )}
            {margin && (
              <>
                {'\n'}
                <span className="k">margin γ  </span>
                {margin.g.toFixed(4)} <span className="k">(widest gap)</span>
                {'\n'}
                <span className="k">(R/γ)²    </span>
                {Math.round(margin.bound)} <span className="k">(most mistakes possible)</span>
              </>
            )}
          </div>
        </div>
        <div className="controls">
          {id === 'thin' ? (
            <div>
              <div className="field-label">Remove one point</div>
              <div className="btn-row">
                {THIN.map((p, i) => (
                  <button key={i} className={`btn small${removed === i ? ' primary' : ''}`} onClick={() => persist({ ...saved, removed: removed === i ? null : i })} aria-pressed={removed === i}>
                    {p.name}
                  </button>
                ))}
              </div>
              <div className="muted" style={{ fontSize: 12.5, marginTop: 4 }}>
                {removed === null ? 'Nothing removed.' : `${THIN[removed].name} removed. Click it again to put it back.`}
              </div>
            </div>
          ) : (
            <>
              <OrderEditor pts={pts} order={order} onChange={setOrder} />
              {id === 'and' ? (
                <div className="readout" style={{ fontSize: 12.5 }}>
                  <span className="k">start (fixed) </span>
                  {fmtW(AND_START)}
                </div>
              ) : (
                <>
                  <Slider label="start w₁" value={w0[0]} min={-2} max={2} step={0.5} format={fmtN} onChange={(v) => setW0(0, v)} />
                  <Slider label="start w₂" value={w0[1]} min={-2} max={2} step={0.5} format={fmtN} onChange={(v) => setW0(1, v)} />
                  <Slider label="start b" value={w0[2]} min={-2} max={2} step={0.5} format={fmtN} onChange={(v) => setW0(2, v)} />
                </>
              )}
              <Segmented label="step size" options={ETAS.map((e) => ({ value: e, label: String(e) }))} value={eta as (typeof ETAS)[number]} onChange={(v) => persist({ ...saved, eta: { ...saved.eta, [id]: v } })} />
            </>
          )}
        </div>
      </div>
    </LabShell>
  );
}
