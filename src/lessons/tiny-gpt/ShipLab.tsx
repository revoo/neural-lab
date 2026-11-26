// authored-by: Claude Opus 5.5
/**
 * Lab: Ship the tiny GPT. One compute budget; the learner chooses depth, width, heads, context and
 * learning rate, predicts, then sees the design's two recorded runs (from `search.run.ts`) drawn
 * step by step, and can re-run seed 1 live to check the record.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { LabShell, useLabProgress, type LabBrief } from '../../components/LabShell';
import { Segmented } from '../../components/controls';
import { useMachineLog } from '../../components/MachineLog';
import { usePrefersReducedMotion } from '../../components/motion';
import { runTraining } from '../../lib/compute/gptRun';
import { createGptTrainer, type GptTrainer } from '../../lib/compute/gptTrainer';
import {
  BATCH,
  BIGRAM,
  BUDGET,
  CONTEXTS,
  designKey,
  designParams,
  designSpec,
  designSteps,
  gridSummary,
  HEADS,
  LAYERS,
  LRS,
  meanVal,
  passes,
  shipHint,
  TARGET,
  WIDTHS,
  type BriefId,
  type Design,
} from './shipLab';
import { SHIP_RESULTS } from './results';
import { useCorpus } from './TrainWidget';

const KEY = 'tiny-gpt-ship-lab';
const BRIEFS: (LabBrief & { id: BriefId })[] = [
  {
    id: 'beat-bigram',
    tab: '1 · Beat pair counts',
    title: 'Ship something better than counting',
    text: (
      <>
        A table of which letter follows which (pair counts) scores {BIGRAM.toFixed(3)} nats per character on the held-out tales. Design a GPT that does better within the
        budget, averaged over its two seeds.
      </>
    ),
  },
  {
    id: 'target',
    tab: '2 · Hit the target',
    title: `Reach ${TARGET.toFixed(2)} or better`,
    text: <>Product wants a validation loss of {TARGET.toFixed(2)} or lower, averaged over both seeds, with the same budget. Spend it wisely.</>,
  },
  {
    id: 'lean',
    tab: '3 · Ship it small',
    title: 'Hit the target with as few parameters as you can',
    text: <>The model must fit on a tiny device. Reach {TARGET.toFixed(2)} again; the optional goal is the fewest parameters of any design in the grid that gets there.</>,
  },
];

const fmt = (n: number) => n.toLocaleString('en-GB');
const bn = (n: number) => `${(n / 1e9).toLocaleString('en-GB', { maximumFractionDigits: 1 })} billion`;
const W = 460;
const H = 200;
const PAD = { l: 40, r: 12, t: 10, b: 30 };

export function ShipLab() {
  const progress = useLabProgress(KEY);
  const [active, setActive] = useState(0);
  const [x, setX] = useState<Design>(() => {
    try {
      const s = JSON.parse(localStorage.getItem(KEY + '-design') ?? '');
      if (s && LAYERS.includes(s.layers) && WIDTHS.includes(s.d)) return s;
    } catch {
      /* ignore */
    }
    return { layers: 3, d: 48, heads: 4, T: 64, lr: 1e-3 };
  });
  useEffect(() => {
    try {
      localStorage.setItem(KEY + '-design', JSON.stringify(x));
    } catch {
      /* ignore */
    }
  }, [x]);
  const [guess, setGuess] = useState<'yes' | 'no' | null>(null);
  const [shown, setShown] = useState<{ x: Design; guess: 'yes' | 'no' | null } | null>(null);
  const [frame, setFrame] = useState(1);
  const reduced = usePrefersReducedMotion();
  const log = useMachineLog();
  const tried = useRef(new Set<string>());
  const brief = BRIEFS[active].id;

  const params = designParams(x);
  const steps = designSteps(x);
  const result = shown ? SHIP_RESULTS[designKey(shown.x)] : null;
  const score = result ? meanVal(result) : null;
  const passing = !!(result && shown && passes(brief, result, TARGET));
  const summary = useMemo(() => gridSummary(SHIP_RESULTS, TARGET), []);
  const optimal = brief === 'lean' && passing && result!.params === summary.leanest?.r.params;

  // draw the recorded curves step by step
  const nFrames = result ? result.curve[0].length : 1;
  useEffect(() => {
    if (!result) return;
    if (reduced) {
      setFrame(nFrames);
      return;
    }
    setFrame(1);
    const t = setInterval(() => setFrame((f) => (f >= nFrames ? (clearInterval(t), f) : f + 1)), 90);
    return () => clearInterval(t);
  }, [result, nFrames, reduced]);

  const run = () => {
    const r = SHIP_RESULTS[designKey(x)];
    setShown({ x, guess });
    if (!r) return;
    tried.current.add(designKey(x));
    const v = meanVal(r);
    const ok = passes(brief, r, TARGET);
    log.push(
      `${x.layers} blocks × width ${x.d}, ${x.heads} head${x.heads > 1 ? 's' : ''}, context ${x.T}, lr ${x.lr}: ${fmt(r.params)} params × ${fmt(r.steps)} steps → ${r.val.map((s) => s.toFixed(3)).join(' / ')} (mean ${v.toFixed(3)})${guess ? `; you predicted ${guess === 'yes' ? 'pass' : 'fail'}` : ''}`,
      ok ? 'good' : 'bad',
    );
    setGuess(null);
  };

  // live re-run of seed 1
  const lang = useCorpus();
  const [live, setLive] = useState<{ key: string; pts: [number, number][]; done: boolean } | null>(null);
  const trainer = useRef<GptTrainer | null>(null);
  const stopLive = useRef(false);
  useEffect(() => () => (stopLive.current = true, trainer.current?.destroy()), []);
  const runLive = async () => {
    if (!lang || !shown) return;
    stopLive.current = false;
    const spec = designSpec(shown.x, 1);
    const key = designKey(shown.x);
    trainer.current?.destroy();
    const tr = await createGptTrainer(spec.cfg);
    trainer.current = tr;
    const pts: [number, number][] = [];
    setLive({ key, pts: [], done: false });
    try {
      await runTraining(spec, lang.corpus, tr, (c) => {
        pts.push([c.step, c.valLoss]);
        setLive({ key, pts: [...pts], done: c.step === spec.steps });
        return !stopLive.current;
      });
    } catch {
      /* stopped */
    }
    const rec = SHIP_RESULTS[key].val[0];
    const got = pts.at(-1)?.[1];
    if (got !== undefined && pts.at(-1)![0] === spec.steps)
      log.push(`live re-run of seed 1 on the ${tr.kind === 'webgpu' ? 'GPU' : 'CPU'}: ${got.toFixed(4)} (recorded ${rec.toFixed(4)}${tr.kind === 'cpu' ? (got === rec ? ', identical' : ', differs') : ''})`, 'info');
  };

  // chart
  const maxStep = result ? result.steps : 1;
  const xs = (s: number) => PAD.l + ((W - PAD.l - PAD.r) * Math.log10(s + 1)) / Math.log10(maxStep + 1);
  const lo = 1.5;
  const hi = 4.3;
  const ys = (v: number) => PAD.t + ((H - PAD.t - PAD.b) * (hi - Math.min(hi, Math.max(lo, v)))) / (hi - lo);
  const path = (pts: readonly (readonly [number, number])[]) => pts.map(([s, v], i) => `${i ? 'L' : 'M'}${xs(s).toFixed(1)},${ys(v).toFixed(1)}`).join('');

  const status = !shown
    ? `Your design: ${fmt(params)} parameters. The budget allows ${fmt(steps)} steps of ${BATCH} windows × ${x.T} characters (${fmt(steps * BATCH * x.T)} characters seen). Predict, then press Train.`
    : !result
      ? 'That design is not in the recorded grid.'
      : `${frame < nFrames ? 'Training…' : `Mean validation loss ${score!.toFixed(3)} over seeds 1 and 2.`} ${
          frame < nFrames ? '' : passing ? (optimal ? 'Passes, with the fewest parameters of any design in the grid that reaches the target.' : 'Passes.') : `Not yet: ${brief === 'beat-bigram' ? `pair counts score ${BIGRAM.toFixed(3)}` : `the target is ${TARGET.toFixed(2)}`}.`
        }${shown.guess && frame >= nFrames ? ` You predicted it would ${shown.guess === 'yes' ? 'pass' : 'fail'}.` : ''}`;

  const opt = <T extends number>(vals: readonly T[], label: (v: T) => string) => vals.map((v) => ({ value: v, label: label(v) }));

  return (
    <LabShell
      title="Lab: Ship the tiny GPT"
      caption={
        <>
          Every design gets the same training budget: {bn(BUDGET)} floating-point operations by the usual estimate, 6 × parameters × characters seen. A bigger model therefore trains for fewer steps. Each design in the grid was trained on two fixed seeds by a committed search script; Train draws those two
          recorded runs, and “Re-run live” trains seed 1 again in your browser.
        </>
      }
      briefs={BRIEFS}
      active={active}
      onSelect={(i) => (setActive(i), setShown(null))}
      progress={progress}
      passing={passing && frame >= nFrames}
      optimal={optimal && frame >= nFrames}
      status={status}
      statusTone={shown && result && frame >= nFrames ? (passing ? 'good' : 'bad') : 'neutral'}
      hint={shown && result && frame >= nFrames && !passing ? shipHint(shown.x, result) : undefined}
      reveal={{
        label: 'Show the whole grid',
        content: <GridMap tried={tried.current} />,
      }}
      log={{ lines: log.lines, onClear: log.clear, title: 'ship.log', empty: 'Runs you try are listed here.' }}
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 16px' }}>
        <Segmented label="blocks" options={opt(LAYERS, (v) => String(v))} value={x.layers} onChange={(v) => (setX({ ...x, layers: v }), setShown(null))} />
        <Segmented label="width" options={opt(WIDTHS, (v) => String(v))} value={x.d} onChange={(v) => (setX({ ...x, d: v }), setShown(null))} />
        <Segmented label="heads" options={opt(HEADS, (v) => String(v))} value={x.heads} onChange={(v) => (setX({ ...x, heads: v }), setShown(null))} />
        <Segmented label="context (characters)" options={opt(CONTEXTS, (v) => String(v))} value={x.T} onChange={(v) => (setX({ ...x, T: v }), setShown(null))} />
        <Segmented label="learning rate" options={opt(LRS, (v) => String(v))} value={x.lr} onChange={(v) => (setX({ ...x, lr: v }), setShown(null))} />
      </div>
      <div className="mono" style={{ fontSize: 12.5, margin: '8px 0', color: 'var(--text-muted)' }}>
        {fmt(params)} parameters × {fmt(steps)} steps × {fmt(BATCH * x.T)} characters per step × 6 ≈ {bn(6 * params * steps * BATCH * x.T)} of the {bn(BUDGET)} budget
      </div>
      <div className="btn-row" style={{ flexWrap: 'wrap', gap: 6 }}>
        <span style={{ fontSize: 13 }}>Will it pass?</span>
        <button className={`btn${guess === 'yes' ? ' primary' : ''}`} onClick={() => setGuess('yes')}>
          yes
        </button>
        <button className={`btn${guess === 'no' ? ' primary' : ''}`} onClick={() => setGuess('no')}>
          no
        </button>
        <button className="btn primary" onClick={run}>
          ▶ Train (recorded runs)
        </button>
        {shown && result && frame >= nFrames && (
          <button className="btn" onClick={runLive} disabled={!lang}>
            ↻ Re-run seed 1 live
          </button>
        )}
      </div>
      {result && (
        <div style={{ overflowX: 'auto', marginTop: 8 }}>
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, minWidth: 300, display: 'block' }} role="img" aria-label="Validation loss of the two seeds">
            {[2, 2.5, 3, 3.5, 4].map((v) => (
              <g key={v}>
                <line className="plot-grid" x1={PAD.l} x2={W - PAD.r} y1={ys(v)} y2={ys(v)} />
                <text className="plot-tick" x={PAD.l - 5} y={ys(v) + 3.5} textAnchor="end">{v.toFixed(1)}</text>
              </g>
            ))}
            {[0, 10, 100, 1000, 10000].filter((t) => t <= maxStep).map((t) => (
              <text key={t} className="plot-tick" x={xs(t)} y={H - PAD.b + 14} textAnchor={t === 0 ? 'start' : 'middle'}>{fmt(t)}</text>
            ))}
            <text className="plot-tick" x={(W + PAD.l) / 2} y={H - 2} textAnchor="middle">steps (log scale)</text>
            {[
              ['pair counts', BIGRAM],
              ['target', TARGET],
            ].map(([n, v]) => (
              <g key={n as string}>
                <line x1={PAD.l} x2={W - PAD.r} y1={ys(v as number)} y2={ys(v as number)} stroke="var(--text-faint)" strokeDasharray="2 3" />
                <text className="plot-tick" x={W - PAD.r} y={ys(v as number) - 4} textAnchor="end">{n as string} {(v as number).toFixed(2)}</text>
              </g>
            ))}
            {result.curve.map((c, k) => (
              <path key={k} d={path(c.slice(0, frame))} fill="none" stroke={k ? 'var(--c-b)' : 'var(--c-a)'} strokeWidth={2} />
            ))}
            {live && live.key === designKey(shown!.x) && <path d={path(live.pts)} fill="none" stroke="var(--accent)" strokeWidth={1.2} strokeDasharray="4 3" />}
          </svg>
          <div className="plot-tick">
            <span className="c-a">seed 1</span> and <span className="c-b">seed 2</span>, recorded{live && live.key === designKey(shown!.x) ? '; dashed: your live re-run of seed 1' : ''}.
          </div>
        </div>
      )}
    </LabShell>
  );
}

function GridMap({ tried }: { tried: Set<string> }) {
  const s = gridSummary(SHIP_RESULTS, TARGET);
  const rows = Object.entries(SHIP_RESULTS);
  const minP = Math.min(...rows.map(([, r]) => r.params));
  const maxP = Math.max(...rows.map(([, r]) => r.params));
  const w = 440;
  const h = 210;
  const px = (p: number) => 40 + ((w - 52) * Math.log(p / minP)) / Math.log(maxP / minP);
  const lo = 1.6;
  const hi = 2.6;
  const py = (v: number) => 10 + ((h - 40) * (Math.min(hi, Math.max(lo, v)) - lo)) / (hi - lo);
  return (
    <div>
      <p style={{ marginTop: 0 }}>
        All {s.total} designs of the grid (three depths × three widths × two head counts × two contexts × three learning rates), each the mean of two seeds. {s.beatBigram} beat
        pair counts; {s.hitTarget} reach {TARGET.toFixed(2)}. Best in the grid: {s.best.x.layers} blocks × width {s.best.x.d}, {s.best.x.heads} head{s.best.x.heads > 1 ? 's' : ''},
        context {s.best.x.T}, learning rate {s.best.x.lr}, at {meanVal(s.best.r).toFixed(3)}.
        {s.leanest && ` Fewest parameters reaching the target: ${fmt(s.leanest.r.params)} (${s.leanestCount} design${s.leanestCount > 1 ? 's' : ''} of that size).`}
      </p>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', maxWidth: w, display: 'block' }} role="img" aria-label="Every design: parameters against validation loss">
        {[1.7, 1.9, 2.1, 2.3, 2.5].map((v) => (
          <g key={v}>
            <line className="plot-grid" x1={40} x2={w - 12} y1={py(v)} y2={py(v)} />
            <text className="plot-tick" x={35} y={py(v) + 3.5} textAnchor="end">{v.toFixed(1)}</text>
          </g>
        ))}
        <line x1={40} x2={w - 12} y1={py(TARGET)} y2={py(TARGET)} stroke="var(--c-c)" strokeDasharray="3 3" />
        {rows.map(([k, r]) => (
          <circle key={k} cx={px(r.params)} cy={py(meanVal(r))} r={tried.has(k) ? 5 : 3} fill={k.includes('-T16-') ? 'var(--c-a)' : 'var(--c-b)'} opacity={tried.has(k) ? 1 : 0.55} stroke={tried.has(k) ? 'var(--text)' : 'none'}>
            <title>{`${k}: ${fmt(r.params)} params, ${meanVal(r).toFixed(3)}`}</title>
          </circle>
        ))}
        <text className="plot-tick" x={(w + 40) / 2} y={h - 4} textAnchor="middle">parameters (log scale), {fmt(minP)} to {fmt(maxP)}; lower is better</text>
      </svg>
      <div className="plot-tick">
        <span className="c-a">context 16</span> · <span className="c-b">context 64</span> · outlined: designs you tried.
      </div>
    </div>
  );
}
