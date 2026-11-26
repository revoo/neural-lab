// authored-by: Claude Opus 5.5
/**
 * "Learning to speak" widgets. The timeline, heads and embedding panels read the recorded CPU
 * reference run (gptReferenceResults.ts: samples, statistics and probes at every checkpoint, made by
 * gptReference.run.ts). "Check it yourself" re-trains the same run live on the CPU and compares every
 * checkpoint with the record.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { usePrefersReducedMotion } from '../../components/motion';
import { Gpt } from '../../lib/compute/gpt';
import { CPU_RUN, PROMPT, samplesAt } from '../../lib/compute/gptReference';
import { CPU_REFERENCE } from '../../lib/compute/gptReferenceResults';
import { initialParams, runTraining } from '../../lib/compute/gptRun';
import { createGptTrainer, type GptTrainer } from '../../lib/compute/gptTrainer';
import { CHAR_GROUP_IDS, PROBE_L, repeatBatch } from '../../lib/compute/gptProbes';
import { CHARS, decode, loadGrimm } from '../../lib/corpora/grimm';
import binUrl from '../../lib/compute/tinyGpt.cpu.bin?url';
import { GROUP_LABELS, IND_BAR, PREV_BAR, speakMoments, type Moment, type SpeakCheckpoint } from './speak';

export const REF = CPU_REFERENCE as unknown as { calibration: typeof CPU_REFERENCE.calibration; checkpoints: SpeakCheckpoint[] };
const CAL = CPU_REFERENCE.calibration;
const STEPS = REF.checkpoints.map((c) => c.step);
const LAST = STEPS[STEPS.length - 1];
const fmt = (n: number) => n.toLocaleString('en-GB');
const pct = (v: number) => `${(100 * v).toFixed(0)}%`;

let refParams: Promise<Float32Array> | null = null;
export function loadFinalParams() {
  refParams ??= fetch(binUrl)
    .then((r) => r.arrayBuffer())
    .then((b) => new Float32Array(b));
  return refParams;
}

// ------------------------------------------------------------------ a small line chart over log steps

type Line = { pts: { step: number; v: number }[]; color: string; width?: number; dash?: string; opacity?: number };
function StepChart({ lines, y, bar, marker, label, height = 120, yTicks, fmtY = (v: number) => String(v) }: { lines: Line[]; y: [number, number]; bar?: { v: number; text: string }; marker?: number; label: string; height?: number; yTicks: number[]; fmtY?: (v: number) => string }) {
  const W = 300;
  const P = { l: 34, r: 8, t: 8, b: 20 };
  const x = (s: number) => P.l + ((W - P.l - P.r) * Math.log10(s + 1)) / Math.log10(LAST + 1);
  const yy = (v: number) => P.t + ((height - P.t - P.b) * (y[1] - Math.min(y[1], Math.max(y[0], v)))) / (y[1] - y[0]);
  return (
    <svg viewBox={`0 0 ${W} ${height}`} style={{ width: '100%', maxWidth: W, display: 'block' }} role="img" aria-label={label}>
      {yTicks.map((t) => (
        <g key={t}>
          <line className="plot-grid" x1={P.l} x2={W - P.r} y1={yy(t)} y2={yy(t)} />
          <text className="plot-tick" x={P.l - 4} y={yy(t) + 3.5} textAnchor="end">{fmtY(t)}</text>
        </g>
      ))}
      {[0, 10, 100, 1000].map((t) => (
        <text key={t} className="plot-tick" x={x(t)} y={height - 5} textAnchor={t ? 'middle' : 'start'}>{fmt(t)}</text>
      ))}
      {bar && (
        <g>
          <line x1={P.l} x2={W - P.r} y1={yy(bar.v)} y2={yy(bar.v)} stroke="var(--c-c)" strokeDasharray="3 3" />
          <text className="plot-tick" x={W - P.r} y={yy(bar.v) - 3} textAnchor="end">{bar.text}</text>
        </g>
      )}
      {lines.map((l, i) => (
        <path key={i} d={l.pts.map((p, k) => `${k ? 'L' : 'M'}${x(p.step).toFixed(1)},${yy(p.v).toFixed(1)}`).join('')} fill="none" stroke={l.color} strokeWidth={l.width ?? 1.8} strokeDasharray={l.dash} opacity={l.opacity ?? 1} />
      ))}
      {marker !== undefined && <line x1={x(marker)} x2={x(marker)} y1={P.t} y2={height - P.b} stroke="var(--accent)" strokeOpacity={0.5} />}
    </svg>
  );
}

function useTimeline(n: number) {
  const reduced = usePrefersReducedMotion();
  const [sel, setSel] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing) return;
    if (reduced) {
      setSel(n - 1);
      setPlaying(false);
      return;
    }
    const t = setInterval(() => setSel((s) => (s >= n - 1 ? (setPlaying(false), s) : s + 1)), 500);
    return () => clearInterval(t);
  }, [playing, reduced, n]);
  return { sel, setSel, playing, play: () => (setSel(0), setPlaying(true)), reduced };
}

function Scrubber({ sel, setSel, play }: { sel: number; setSel: (i: number) => void; play: () => void }) {
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', margin: '6px 0' }}>
      <button className="btn" onClick={play}>
        ▶ Replay its life
      </button>
      <label className="slider" style={{ flex: '1 1 220px' }}>
        <span className="slider-top">
          <span className="slider-label">timeline</span>
          <span className="slider-value">step {fmt(STEPS[sel])}</span>
        </span>
        <input type="range" min={0} max={STEPS.length - 1} value={sel} onChange={(e) => setSel(+e.target.value)} aria-label="Training timeline" />
      </label>
    </div>
  );
}

// ------------------------------------------------------------------ 1. from noise to language

export function NoiseToLanguage() {
  const { sel, setSel, play, playing, reduced } = useTimeline(STEPS.length);
  const moments = useMemo(() => speakMoments(REF.checkpoints, CAL), []);
  const cur = REF.checkpoints[sel];
  const [flash, setFlash] = useState<Moment | null>(null);
  useEffect(() => {
    if (!playing) return;
    const m = moments.find((mm) => mm.step === cur.step);
    if (m) setFlash(m);
  }, [sel, playing, moments, cur.step]);
  useEffect(() => {
    if (!flash) return;
    const t = setTimeout(() => setFlash(null), reduced ? 6000 : 2500);
    return () => clearTimeout(t);
  }, [flash, reduced]);
  const series = (f: (c: SpeakCheckpoint) => number) => REF.checkpoints.map((c) => ({ step: c.step, v: f(c) }));
  const reached = moments.filter((m) => m.step <= cur.step);
  return (
    <Widget
      title="From noise to language: one model's life, measured"
      wide
      caption={
        <>
          The recorded training run of the capstone's tiny GPT: at each of its {STEPS.length} checkpoints, four samples of 200 characters from the prompt “{PROMPT}” (seeds 1 to 4,
          temperature 1) and four statistics of what they contain. The dashed lines are what a real passage of the tales scores; a milestone counts from the checkpoint where
          it is reached and stays reached. Drag the timeline or press Replay; the button below re-trains the whole run in your browser to check every number.
        </>
      }
    >
      <Scrubber sel={sel} setSel={setSel} play={play} />
      {flash && (
        <div className="birth-flash" role="status">
          Step {fmt(flash.step)}: {flash.label}.
        </div>
      )}
      <div className="btn-row" style={{ flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
        {moments.map((m, k) => (
          <button key={m.id} className="btn" style={{ fontSize: 12.5, outline: m.step <= cur.step ? '1px solid var(--c-c)' : undefined }} onClick={() => setSel(STEPS.indexOf(m.step))}>
            {k + 1}. step {fmt(m.step)}: {m.label}
          </button>
        ))}
      </div>
      <div className="gpt-panel">
        <div className="field-label" style={{ textTransform: 'none' }}>
          Step {fmt(cur.step)} · validation loss {cur.valLoss.toFixed(3)} · four samples
        </div>
        {cur.samples.map((s, i) => (
          <div key={i} className="mono gpt-sample" style={{ opacity: i ? 0.8 : 1 }}>
            {PROMPT}
            <b>{s.slice(0, 120)}</b>
          </div>
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(230px, 100%), 1fr))', gap: 10, marginTop: 10 }}>
        <div>
          <div className="plot-tick">letter-mix gap (KL, lower is closer): {cur.stats.kl.toFixed(3)}</div>
          <StepChart label="Letter-mix gap over training" lines={[{ pts: series((c) => c.stats.kl), color: 'var(--accent)' }]} y={[0, 1.3]} yTicks={[0, 0.5, 1]} bar={{ v: CAL.klP95, text: 'real passages' }} marker={cur.step} />
        </div>
        <div>
          <div className="plot-tick">average word length: {cur.stats.meanWordLen.toFixed(2)} (the tales: {CAL.wordLen.toFixed(2)})</div>
          <StepChart label="Average word length over training" lines={[{ pts: series((c) => c.stats.meanWordLen), color: 'var(--accent)' }]} y={[2, 14]} yTicks={[4, 8, 12]} bar={{ v: CAL.wordLen, text: 'the tales' }} marker={cur.step} />
        </div>
        <div>
          <div className="plot-tick">real words: {pct(cur.stats.realShare)}</div>
          <StepChart label="Share of real words over training" lines={[{ pts: series((c) => c.stats.realShare), color: 'var(--accent)' }]} y={[0, 1]} yTicks={[0, 0.5, 1]} fmtY={(v) => pct(v)} bar={{ v: 0.5, text: 'half' }} marker={cur.step} />
        </div>
        <div>
          <div className="plot-tick">longest run of real words: {cur.stats.longestRun} (real passages: {CAL.runP05}+)</div>
          <StepChart label="Longest run of real words over training" lines={[{ pts: series((c) => c.stats.longestRun), color: 'var(--accent)' }]} y={[0, 8]} yTicks={[0, 2, 4, 6]} bar={{ v: 6, text: 'six in a row' }} marker={cur.step} />
        </div>
      </div>
      <p className="live-note">
        {cur.step === 0
          ? 'Step 0: random weights. Capitals, digits and brackets turn up as often as e or t; 4% of the "words" are real, by luck.'
          : `Step ${fmt(cur.step)}: ${pct(cur.stats.realShare)} of its words are real. ${reached.length ? `Reached so far: ${reached.map((m) => m.label).join('; ')}.` : 'No milestone yet.'}${cur.step === LAST ? ' It never reaches four real words in five, or six real words in a row.' : ''}`}
      </p>
      <CheckRun />
    </Widget>
  );
}

/** Re-train the recorded run live on the CPU tier and compare each checkpoint's samples with the record. */
function CheckRun() {
  const [state, setState] = useState<{ step: number; same: number; diff: number; done: boolean; where: string } | null>(null);
  const tr = useRef<GptTrainer | null>(null);
  const stop = useRef(false);
  useEffect(() => () => void ((stop.current = true), tr.current?.destroy()), []);
  const run = async () => {
    stop.current = false;
    const corpus = await loadGrimm();
    tr.current?.destroy();
    const t = await createGptTrainer(CPU_RUN.cfg, 'cpu');
    tr.current = t;
    let same = 0;
    let diff = 0;
    setState({ step: 0, same, diff, done: false, where: t.where });
    await runTraining(CPU_RUN, corpus, t, (c) => {
      const rec = REF.checkpoints.find((r) => r.step === c.step)!;
      const ok = c.valLoss === rec.valLoss && samplesAt(CPU_RUN, c.params).every((s, i) => s === rec.samples[i]);
      if (ok) same++;
      else diff++;
      setState({ step: c.step, same, diff, done: c.step === CPU_RUN.steps, where: t.where });
      return !stop.current;
    }).catch(() => undefined);
  };
  return (
    <div className="gpt-panel" style={{ marginTop: 8, display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
      <button className="btn" onClick={() => (state && !state.done ? (stop.current = true) : run())}>
        {state && !state.done ? '❚❚ Stop the check' : '✓ Check it yourself: re-train this run (CPU, a few minutes)'}
      </button>
      {state && (
        <span className="mono" style={{ fontSize: 12.5 }}>
          {state.where === 'worker' ? 'in a Web Worker' : state.where}: step {fmt(state.step)} of {fmt(CPU_RUN.steps)}; {state.same} checkpoint{state.same === 1 ? '' : 's'} identical to the record
          {state.diff ? `, ${state.diff} different` : ''}
          {state.done ? (state.diff ? '.' : '. Every number above is what your browser computes.') : '…'}
        </span>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ 2. heads

function HeadMap({ att, T, label }: { att: Float64Array; T: number; label: string }) {
  const c = 3.5;
  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${c * T} ${c * T}`} style={{ width: '100%', maxWidth: 150, display: 'block', background: 'var(--bg)', border: '1px solid var(--border)' }} role="img" aria-label={label}>
        {Array.from({ length: T }, (_, i) =>
          Array.from({ length: i + 1 }, (_, j) => {
            const w = att[i * T + j];
            return w > 0.03 ? <rect key={`${i}-${j}`} x={j * c} y={i * c} width={c} height={c} fill="var(--accent)" opacity={Math.min(1, 0.1 + w)} /> : null;
          }),
        )}
        <line x1={0} y1={0} x2={c * T} y2={c * T} stroke="var(--text-faint)" strokeWidth={0.3} />
      </svg>
      <figcaption className="plot-tick" style={{ textAlign: 'center' }}>{label}</figcaption>
    </figure>
  );
}

export function HeadsWidget() {
  const { sel, setSel, play } = useTimeline(STEPS.length);
  const cur = REF.checkpoints[sel];
  const [final, setFinal] = useState<Float32Array | null>(null);
  useEffect(() => {
    let ok = true;
    loadFinalParams().then((p) => ok && setFinal(p));
    return () => {
      ok = false;
    };
  }, []);
  const seq = useMemo(() => repeatBatch(1, PROBE_L)[0], []);
  const maps = useMemo(() => {
    const at = (p: Float32Array) => {
      const m = new Gpt(CPU_RUN.cfg);
      m.load(p);
      m.forward(seq, 1, seq.length);
      return m.attention();
    };
    return { first: at(initialParams(CPU_RUN)), last: final ? at(final) : null };
  }, [final, seq]);
  const H = CPU_RUN.cfg.heads;
  const lines = (key: 'prev' | 'induction') =>
    Array.from({ length: CPU_RUN.cfg.layers * H }, (_, k) => ({
      pts: REF.checkpoints.map((c) => ({ step: c.step, v: c.probe[key][Math.floor(k / H)][k % H] })),
      color: k < H ? 'var(--c-a)' : 'var(--c-b)',
      width: 1.4,
    }));
  const nPrev = cur.probe.prev.flat().filter((v) => v > PREV_BAR).length;
  const bestInd = Math.max(...cur.probe.induction.flat());
  return (
    <Widget
      title="Attention heads forming (and one that never does)"
      wide
      caption={
        <>
          Each line is one head of the recorded run: <span className="c-a">block 1</span> and <span className="c-b">block 2</span>. The test text is {PROBE_L} random
          letters typed twice. Left: how much attention each head puts on the previous character. Middle: in the second copy, how much it puts on the character that
          followed the same letter the first time (what an induction head does). Dashed: twice what perfectly even attention would score. Right: the model's loss on the
          first and second copies.
        </>
      }
    >
      <Scrubber sel={sel} setSel={setSel} play={play} />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(230px, 100%), 1fr))', gap: 10 }}>
        <div>
          <div className="plot-tick">previous-character score</div>
          <StepChart label="Previous-character score per head" lines={lines('prev')} y={[0, 0.4]} yTicks={[0, 0.1, 0.2, 0.3]} bar={{ v: PREV_BAR, text: 'head bar' }} marker={cur.step} />
        </div>
        <div>
          <div className="plot-tick">induction score</div>
          <StepChart label="Induction score per head" lines={lines('induction')} y={[0, 0.4]} yTicks={[0, 0.1, 0.2, 0.3]} bar={{ v: IND_BAR, text: 'head bar' }} marker={cur.step} />
        </div>
        <div>
          <div className="plot-tick">loss on the first and second copy</div>
          <StepChart
            label="Loss on the first and second copies"
            lines={[
              { pts: REF.checkpoints.map((c) => ({ step: c.step, v: c.probe.lossFirst })), color: 'var(--accent)' },
              { pts: REF.checkpoints.map((c) => ({ step: c.step, v: c.probe.lossSecond })), color: 'var(--c-c)', dash: '4 3' },
            ]}
            y={[3, 7]}
            yTicks={[3, 4, 5, 6, 7]}
            marker={cur.step}
          />
          <div className="plot-tick">
            solid: first copy; dashed: second copy
          </div>
        </div>
      </div>
      <p className="live-note">
        Step {fmt(cur.step)}: {nPrev} head{nPrev === 1 ? '' : 's'} above the previous-character bar ({PREV_BAR.toFixed(3)}); the best induction score is {bestInd.toFixed(3)}, against a bar of{' '}
        {IND_BAR.toFixed(3)}. Loss on the second copy {cur.probe.lossSecond.toFixed(2)}, on the first {cur.probe.lossFirst.toFixed(2)}: {cur.probe.lossSecond < cur.probe.lossFirst - 0.1 ? 'the repeat helps.' : 'seeing the letters once does not help it predict them the second time.'}
      </p>
      <div className="gpt-panel">
        <div className="field-label" style={{ textTransform: 'none' }}>Attention on “{decode(seq)}”: step 0 (top row) and step {fmt(LAST)} (bottom row)</div>
        {[maps.first, maps.last].map((att, r) =>
          att ? (
            <div key={r} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(90px, 22%), 1fr))', gap: 8, marginTop: 8 }}>
              {att.flatMap((layer, l) => layer.map((a, h) => <HeadMap key={`${l}-${h}`} att={a} T={seq.length} label={`block ${l + 1} · head ${h + 1}`} />))}
            </div>
          ) : (
            <p key={r} className="plot-tick">Loading the trained weights…</p>
          ),
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. embeddings

export function EmbeddingWidget() {
  const { sel, setSel, play } = useTimeline(STEPS.length);
  const cur = REF.checkpoints[sel];
  const pts = cur.probe.embPca;
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const W = 360;
  const sx = (v: number) => 14 + ((W - 28) * (v - x0)) / (x1 - x0 || 1);
  const sy = (v: number) => 14 + ((W - 28) * (y1 - v)) / (y1 - y0 || 1);
  const shown = (c: string) => (c === ' ' ? '␣' : c === '\n' ? '↵' : c);
  return (
    <Widget
      title="The character map organising itself"
      wide
      caption={
        <>
          Each of the 68 characters' embedding vectors (32 numbers) in the recorded run, squashed to two dimensions with PCA and coloured by kind. The number that counts
          is computed in all 32 dimensions: hide one character, and guess its kind from the nearest kind-average of the others. The dashed line is how well that guess
          does when the kinds are shuffled at random (99th percentile of 200 shuffles).
        </>
      }
    >
      <Scrubber sel={sel} setSel={setSel} play={play} />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 12 }}>
        <svg viewBox={`0 0 ${W} ${W}`} style={{ width: '100%', maxWidth: W, display: 'block', background: 'var(--bg-elev)', border: '1px solid var(--border)', borderRadius: 4 }} role="img" aria-label={`Character embeddings at step ${cur.step}`}>
          {pts.map((p, k) => (
            <text key={k} x={sx(p[0])} y={sy(p[1])} fontSize={13} fontFamily="var(--font-mono)" fontWeight={600} textAnchor="middle" dominantBaseline="middle" fill={`var(--cls-${CHAR_GROUP_IDS[k]})`}>
              {shown(CHARS[k])}
            </text>
          ))}
        </svg>
        <div>
          <div className="plot-tick">kind guessed from the nearest average: {pct(cur.probe.emb)} (shuffled: {pct(cur.probe.embBar)})</div>
          <StepChart
            label="Embedding kind accuracy over training"
            lines={[
              { pts: REF.checkpoints.map((c) => ({ step: c.step, v: c.probe.emb })), color: 'var(--accent)' },
              { pts: REF.checkpoints.map((c) => ({ step: c.step, v: c.probe.embBar })), color: 'var(--text-faint)', dash: '3 3' },
            ]}
            y={[0, 1]}
            yTicks={[0, 0.5, 1]}
            fmtY={(v) => pct(v)}
            marker={cur.step}
            height={150}
          />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px 10px', marginTop: 6, fontSize: 12 }}>
            {GROUP_LABELS.map((g, k) => (
              <span key={g}>
                <span style={{ display: 'inline-block', width: 9, height: 9, borderRadius: 5, background: `var(--cls-${k})`, marginRight: 4 }} />
                {g}
              </span>
            ))}
          </div>
        </div>
      </div>
      <p className="live-note">
        Step {fmt(cur.step)}: hiding one character and guessing its kind from where it sits gets {pct(cur.probe.emb)} right; with the kinds shuffled at random, 99% of 200
        shuffles score at most {pct(cur.probe.embBar)}.{cur.probe.emb > cur.probe.embBar ? ' The map has real structure.' : ' No more structure than chance yet.'}
      </p>
    </Widget>
  );
}
