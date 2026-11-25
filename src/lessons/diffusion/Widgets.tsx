// authored-by: Claude Opus 5.5
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Arrow, Handle, Plot, usePlot } from '../../components/plot';
import { LogChart } from '../../components/LogChart';
import { createRng } from '../../lib/rng';
import {
  LAB_LEVELS,
  LAB_MIX,
  LAB_SAMPLES,
  LOPSIDED,
  ON_SHAPE,
  RING,
  SMILEY,
  T,
  abarAt,
  bestGuess,
  dataMean,
  density,
  evenLevels,
  exactDenoiser,
  fixedCloud,
  labRun,
  noiseStart,
  outOfPlace,
  sample,
  sampleData,
  shapeLeft,
  type LabBriefId,
  type Mixture,
  type Pt,
} from './diffusion';
import { LEARN, lossFloor, noiseTrainer, type NoiseNet } from './learned';
import { BRIEFS } from './lab';
import { LAB_RESULTS } from './labResults';

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const DATASETS: Record<string, Mixture> = { ring: RING, smiley: SMILEY, lopsided: LOPSIDED };
type DataId = 'ring' | 'smiley' | 'lopsided';
const DATA_OPTIONS = [
  { value: 'ring' as DataId, label: 'Ring' },
  { value: 'smiley' as DataId, label: 'Smiley' },
  { value: 'lopsided' as DataId, label: 'Rare blob' },
];
const VIEW: [number, number] = [-2.6, 2.6];

// ------------------------------------------------------------------ shared drawing

function Blobs({ mix, scale = 1, opacity = 0.7 }: { mix: Mixture; scale?: number; opacity?: number }) {
  const { x, y } = usePlot();
  return (
    <g fill="none" stroke="var(--c-good)" strokeWidth={1.4} strokeDasharray="3 3" opacity={opacity}>
      {mix.m.map((m, k) => (
        <circle key={k} cx={x(m[0] * scale)} cy={y(m[1] * scale)} r={(x(ON_SHAPE * mix.s[k]) - x(0)) * scale} />
      ))}
    </g>
  );
}

function Dots({ pts, color = 'var(--c-a)', r = 1.7, opacity = 0.6 }: { pts: Pt[]; color?: string; r?: number; opacity?: number }) {
  const { x, y } = usePlot();
  const [lo, hi] = x.domain;
  return (
    <g fill={color} opacity={opacity}>
      {pts.map((p, i) =>
        p[0] < lo || p[0] > hi || p[1] < lo || p[1] > hi ? null : <circle key={i} cx={x(p[0])} cy={y(p[1])} r={r} />,
      )}
    </g>
  );
}

/** Faint shading of the exact noisy density at signal share a. */
function DensityShade({ mix, a, n = 36 }: { mix: Mixture; a: number; n?: number }) {
  const { x, y } = usePlot();
  const cells = useMemo(() => {
    const [lo, hi] = VIEW;
    const h = (hi - lo) / n;
    const vals: { cx: number; cy: number; v: number }[] = [];
    let mx = 0;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        const cx = lo + (i + 0.5) * h;
        const cy = lo + (j + 0.5) * h;
        const v = density(mix, a, [cx, cy]);
        mx = Math.max(mx, v);
        vals.push({ cx, cy, v });
      }
    return { vals, mx, h };
  }, [mix, a, n]);
  const w = x(cells.h) - x(0) + 0.6;
  return (
    <g>
      {cells.vals.map((c, i) =>
        c.v < cells.mx * 0.01 ? null : (
          <rect key={i} x={x(c.cx - cells.h / 2)} y={y(c.cy + cells.h / 2)} width={w} height={w} fill="var(--accent)" opacity={0.42 * Math.sqrt(c.v / cells.mx)} />
        ),
      )}
    </g>
  );
}

type PlotChildren = Parameters<typeof Plot>[0]['children'];
type PlotChild = Exclude<PlotChildren, unknown[] | undefined>;
function Scatter({ children, size = 330, label }: { children: PlotChildren; size?: number; label: string }) {
  const bg: PlotChild = ({ x, y }) => <rect x={x(VIEW[0])} y={y(VIEW[1])} width={x(VIEW[1]) - x(VIEW[0])} height={y(VIEW[0]) - y(VIEW[1])} fill="var(--bg-sunken)" rx={6} />;
  const kids = [bg, ...((Array.isArray(children) ? children : [children]) as PlotChild[])];
  return <Plot width={size} height={size} x={VIEW} y={VIEW} grid={false} axes={false} ariaLabel={label} style={{ maxWidth: '100%', height: 'auto', touchAction: 'none' }} children={kids} />;
}

// ------------------------------------------------------------------ 1. Forward: dissolve the shape

export function ForwardNoise() {
  const [data, setData] = useState<DataId>('ring');
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const mix = DATASETS[data];
  const a = abarAt(t);
  const cloud = useMemo(() => fixedCloud(mix, 700, 3), [mix]);
  const pts = useMemo(() => cloud.at(a), [cloud, a]);
  const left = useMemo(() => shapeLeft(mix, a), [mix, a]);
  const raf = useRef(0);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const loop = (now: number) => {
      const dt = now - last;
      last = now;
      setT((cur) => {
        const next = Math.min(T, cur + dt * 0.25);
        if (next >= T) setPlaying(false);
        return Math.round(next);
      });
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf.current);
  }, [playing]);

  return (
    <Widget
      title="Dissolve the shape"
      wide
      caption={
        <>
          Each dot is a data point. Every forward step shrinks every point slightly towards the middle and adds a little fresh random noise, using the 1,000-step schedule from the original diffusion paper (DDPM). The slider jumps straight to any step: each dot keeps its own fixed noise, so you can watch individual points drift. The shading is the <strong>exact</strong> distribution of noisy points at that step. Press <strong>Play</strong>, or drag the slider, and watch when the shape stops being recognisable.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10, gap: 14 }}>
        <Segmented value={data} onChange={setData} options={DATA_OPTIONS} />
        <button
          type="button"
          className="btn"
          onClick={() => {
            if (t >= T) setT(0);
            setPlaying((p) => !p);
          }}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
      </div>
      <Slider label="forward step t" value={t} min={0} max={T} step={10} onChange={(v) => { setPlaying(false); setT(v); }} format={(v) => `${Math.round(v)} of ${T}`} color="c-p" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start', marginTop: 8 }}>
        <Scatter label="Data points being noised">
          <DensityShade mix={mix} a={a} />
          <Blobs mix={mix} opacity={0.35} />
          <Dots pts={pts} />
        </Scatter>
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">signal share ᾱ </span>
            {a.toFixed(4)}
            {'\n'}
            <span className="k">clean point scaled by √ᾱ </span>
            {Math.sqrt(a).toFixed(3)}
            {'\n'}
            <span className="k">noise added, SD √(1 − ᾱ) </span>
            {Math.sqrt(1 - a).toFixed(3)}
            {'\n'}
            <span className="k">shape left (distance from pure noise) </span>
            {pct(left)}
          </div>
          <div className="live-note">
            {t === 0 ? (
              <>The clean data: every point sits on one of the blobs (dashed circles). Nothing has happened yet.</>
            ) : left > 0.3 ? (
              <>
                The blobs are blurring and shrinking towards the middle, but the shape is still obvious: {pct(left)} of the probability would have to move to make this pure noise.
              </>
            ) : left > 0.03 ? (
              <>
                Most of the structure has gone: only {pct(left)} of the probability is still out of place compared with pure noise. A faint hint of the shape remains in the shading.
              </>
            ) : t < T ? (
              <>
                By now the cloud is practically pure noise ({pct(left)} left). The remaining steps are spent turning “almost noise” into “exactly noise”. Remember this when we choose where to spend the backward steps.
              </>
            ) : (
              <>
                Step 1,000: the signal share is {a.toExponential(1)}. Whatever the data looked like, the points are now a standard bell, the same shape a computer can sample from in one line. That is the whole trick: the end point of the forward process is something we already know how to make.
              </>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. The perfect denoiser's best guess

export function BestGuess() {
  const [data, setData] = useState<DataId>('ring');
  const [t, setT] = useState(300);
  const [probe, setProbe] = useState<Pt>([0.9, 0.5]);
  const mix = DATASETS[data];
  const a = abarAt(t);
  const g = useMemo(() => bestGuess(mix, a, probe), [mix, a, probe]);
  const field = useMemo(() => {
    const out: { from: Pt; to: Pt }[] = [];
    for (let i = 0; i <= 10; i++)
      for (let j = 0; j <= 10; j++) {
        const p: Pt = [-2.2 + (4.4 * i) / 10, -2.2 + (4.4 * j) / 10];
        out.push({ from: p, to: bestGuess(mix, a, p).x0 });
      }
    return out;
  }, [mix, a]);
  const order = g.resp.map((r, k) => ({ r, k })).sort((p, q) => q.r - p.r);
  const top = order[0];
  const mean = dataMean(mix);

  return (
    <Widget
      title="The perfect denoiser’s best guess"
      wide
      caption={
        <>
          Pick a noise level and drag the ringed noisy point. The arrow goes to the <strong>best possible guess</strong> of the clean point it came from: the average of every clean point that could have produced it, weighted by how likely each is. Because our data is a mixture of round blobs, this is computed exactly. The faint lines show the best guess from a grid of other noisy points. Try a low noise level, then step 1,000.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10 }}>
        <Segmented value={data} onChange={setData} options={DATA_OPTIONS} />
      </div>
      <Slider label="noise level: forward step t" value={t} min={0} max={T} step={10} onChange={setT} format={(v) => `${Math.round(v)} (ᾱ = ${abarAt(v).toFixed(3)})`} color="c-p" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start', marginTop: 8 }}>
        <Plot width={330} height={330} x={VIEW} y={VIEW} grid={false} axes={false} ariaLabel="Best-guess denoising arrows" onPointerDown={(p) => setProbe([p.x, p.y])} style={{ maxWidth: '100%', height: 'auto', touchAction: 'none' }}>
          {({ x, y }) => <rect x={x(VIEW[0])} y={y(VIEW[1])} width={x(VIEW[1]) - x(VIEW[0])} height={y(VIEW[0]) - y(VIEW[1])} fill="var(--bg-sunken)" rx={6} />}
          <DensityShade mix={mix} a={a} n={30} />
          <Blobs mix={mix} />
          {({ x, y }) => (
            <g stroke="var(--text-faint)" strokeWidth={1} opacity={0.55}>
              {field.map((f, i) => (
                <g key={i}>
                  <line x1={x(f.from[0])} y1={y(f.from[1])} x2={x(f.to[0])} y2={y(f.to[1])} />
                  <circle cx={x(f.from[0])} cy={y(f.from[1])} r={1.3} fill="var(--text-faint)" stroke="none" />
                </g>
              ))}
            </g>
          )}
          <Arrow from={probe} to={g.x0} color="var(--c-good)" width={2.6} />
          <Handle x={probe[0]} y={probe[1]} onDrag={(px, py) => setProbe([px, py])} color="var(--accent)" label="noisy point" />
        </Plot>
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">noisy point </span>({probe[0].toFixed(2)}, {probe[1].toFixed(2)}){'\n'}
            <span className="k">best guess of the clean point </span>({g.x0[0].toFixed(2)}, {g.x0[1].toFixed(2)}){'\n'}
            <span className="k">data’s overall average </span>({mean[0].toFixed(2)}, {mean[1].toFixed(2)})
          </div>
          <div>
            <div className="field-label">Which blob did it come from?</div>
            <div style={{ display: 'grid', gap: 4 }}>
              {order.slice(0, 4).map(({ r, k }) => (
                <div key={k} style={{ display: 'grid', gridTemplateColumns: '74px 1fr 52px', gap: 8, alignItems: 'center', fontSize: 12.5 }}>
                  <span style={{ color: 'var(--text-muted)' }}>blob {k + 1}</span>
                  <div className="progress-bar" style={{ height: 8 }}>
                    <div style={{ width: `${100 * r}%`, background: 'var(--c-good)' }} />
                  </div>
                  <span style={{ fontFamily: 'var(--font-mono)', textAlign: 'right' }}>{pct(r)}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="live-note">
            {top.r > 0.95 ? (
              <>
                At this noise level the point is clearly from blob {top.k + 1} ({pct(top.r)} sure), so the best guess lands on that blob. Denoising is easy when there is little noise or the point is close to one blob.
              </>
            ) : top.r < 0.4 ? (
              <>
                The point could have come from many blobs, so the best guess is a <strong>weighted average</strong> of them, which sits in empty space between them, close to the data’s overall average. Nothing real lives there. At step 1,000, every arrow points to the same place: from pure noise, the best single guess is the average of all the data. That is the blur of the last lesson, and it is why diffusion never jumps straight from noise to data.
              </>
            ) : (
              <>
                Torn between blobs: the best guess is pulled part of the way towards each likely blob, landing between them. A guess between two blobs isn’t a real data point, but it is the best <em>single</em> answer when you can’t tell which blob is right.
              </>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Run it backwards

const STEP_CHOICES = [1, 2, 3, 5, 10, 20, 50, 100, 1000];
const N_SAMPLES = 2000;

function runEven(mix: Mixture, steps: number, eta: number, keep = false) {
  return sample(exactDenoiser(mix), evenLevels(steps), noiseStart(N_SAMPLES, 5), eta, createRng(9), keep);
}

const curveCache = new Map<string, { steps: number; ddim: number; ddpm: number }[]>();
function curves(id: DataId) {
  if (!curveCache.has(id)) {
    const mix = DATASETS[id];
    curveCache.set(
      id,
      STEP_CHOICES.map((s) => ({ steps: s, ddim: outOfPlace(mix, runEven(mix, s, 0).xs).tv, ddpm: outOfPlace(mix, runEven(mix, s, 1).xs).tv })),
    );
  }
  return curveCache.get(id)!;
}

export function RunBackwards() {
  const [data, setData] = useState<DataId>('ring');
  const [steps, setSteps] = useState(10);
  const [eta, setEta] = useState(0);
  const [showPaths, setShowPaths] = useState(false);
  const [showCurve, setShowCurve] = useState(false);
  const mix = DATASETS[data];
  const run = useMemo(() => runEven(mix, steps, eta, true), [mix, steps, eta]);
  const judge = useMemo(() => outOfPlace(mix, run.xs), [mix, run]);
  const perfect = useMemo(() => outOfPlace(mix, sampleData(mix, N_SAMPLES, createRng(3))).tv, [mix]);
  const curve = useMemo(() => (showCurve ? curves(data) : null), [showCurve, data]);
  const paths = useMemo(() => (showPaths ? Array.from({ length: 40 }, (_, i) => run.path.map((xs) => xs[i * 7])) : []), [showPaths, run]);
  const spread = useMemo(() => {
    let s = 0;
    const m = dataMean(mix);
    for (const p of run.xs) s += Math.hypot(p[0] - m[0], p[1] - m[1]) / run.xs.length;
    return s;
  }, [mix, run]);

  return (
    <Widget
      title="Run it backwards"
      wide
      caption={
        <>
          Start from 2,000 points of pure noise and walk them back to clean data using the <strong>exact</strong> denoiser, stopping at evenly spaced steps of DDPM’s 1,000-step schedule. Each stop: guess the clean point, work out the noise that guess implies, then rebuild the point at the next, less noisy level. <strong>DDIM</strong> reuses the implied noise; <strong>DDPM</strong> swaps part of it for fresh random noise. The score is the share of probability out of place: how many samples would have to move to match the true blobs (dashed circles) in the right proportions.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10, gap: 14 }}>
        <Segmented value={data} onChange={setData} options={DATA_OPTIONS} />
        <Segmented
          value={eta}
          onChange={setEta}
          options={[
            { value: 0, label: 'DDIM' },
            { value: 1, label: 'DDPM' },
          ]}
        />
      </div>
      <Segmented label="Denoising steps" value={steps} onChange={setSteps} options={STEP_CHOICES.map((s) => ({ value: s, label: String(s) }))} />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start', marginTop: 10 }}>
        <Scatter label="Samples after running the reverse process">
          <Blobs mix={mix} />
          <Dots pts={run.xs} r={1.5} opacity={0.45} />
          {({ x, y }) => (
            <g fill="none" stroke="var(--c-b)" strokeWidth={1} opacity={0.8}>
              {paths.map((p, i) => (
                <polyline key={i} points={p.map((q) => `${x(Math.max(VIEW[0], Math.min(VIEW[1], q[0])))},${y(Math.max(VIEW[0], Math.min(VIEW[1], q[1])))}`).join(' ')} />
              ))}
            </g>
          )}
        </Scatter>
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">denoiser calls </span>
            {steps}
            {'\n'}
            <span className="k">probability out of place </span>
            <span style={{ color: judge.tv <= perfect + 0.02 ? 'var(--c-good)' : judge.tv > 0.3 ? 'var(--c-bad)' : 'var(--text)' }}>{pct(judge.tv)}</span>
            {'\n'}
            <span className="k">samples off every blob </span>
            {pct(judge.offShape)}
            {'\n'}
            <span className="k">a perfect sampler’s score (2,000 draws) </span>
            {pct(perfect)}
          </div>
          <div className="btn-row">
            <button type="button" className="btn small" onClick={() => setShowPaths((s) => !s)}>{showPaths ? 'Hide paths' : 'Show 40 paths'}</button>
            <button type="button" className="btn small" onClick={() => setShowCurve((s) => !s)}>{showCurve ? 'Hide the chart' : 'Chart every step count'}</button>
          </div>
          <div className="live-note">
            {steps === 1 ? (
              <>
                One step means one guess from pure noise, and the best guess from pure noise is the data’s average. All 2,000 samples land on the same point (average distance from the middle: {spread.toFixed(3)}). One-shot denoising produces the blur, exactly.
              </>
            ) : judge.tv > 0.5 ? (
              <>
                Still terrible: {pct(judge.tv)} out of place. Evenly spaced steps waste most of their budget on near-pure noise (remember the forward widget: by step 400 the shape had essentially gone), and the last jump to clean data is still a long, averaging guess. Show the paths to see where the samples end up.
              </>
            ) : judge.tv > perfect + 0.03 ? (
              <>
                Getting there: the samples have found the blobs, but {pct(judge.offShape)} still sit off them, in the gaps where averaged guesses land. More, smaller steps let each guess commit to one blob before the next.
              </>
            ) : (
              <>
                As good as a perfect sampler with this many samples ({pct(judge.tv)} against {pct(perfect)}). Each small step only has to remove a little noise, so the best guess is never far wrong.
              </>
            )}
          </div>
          {curve && (
            <LogChart
              series={[
                { points: curve.map((c) => [c.steps, Math.max(c.ddim, 0.005)] as [number, number]), color: 'var(--accent)', dots: true, label: 'DDIM' },
                { points: curve.map((c) => [c.steps, Math.max(c.ddpm, 0.005)] as [number, number]), color: 'var(--c-b)', dots: true, dashed: true, label: 'DDPM' },
              ]}
              hlines={[{ y: perfect, label: 'perfect sampler', color: 'var(--c-good)' }]}
              xDomain={[1, 1000]}
              yDomain={[0.005, 1]}
              logX
              xTicks={[1, 10, 100, 1000]}
              xLabel="steps"
              marker={steps}
              ariaLabel="Probability out of place against the number of steps, DDIM and DDPM"
            />
          )}
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Learning the denoiser

type LearnRun = { done: number; net: NoiseNet; history: { step: number; loss: number }[]; finished: boolean };
let learnRun: LearnRun | null = null;
const learnListeners = new Set<() => void>();
let learnVersion = 0;
const bump = () => {
  learnVersion++;
  learnListeners.forEach((l) => l());
};
function startLearning() {
  if (learnRun) return;
  const tr = noiseTrainer(RING, LEARN);
  learnRun = { done: 0, net: tr.net, history: tr.history, finished: false };
  const tick = () => {
    const t0 = performance.now();
    let fin = false;
    while (performance.now() - t0 < 30 && !(fin = tr.advance(10)));
    learnRun!.done = tr.done;
    learnRun!.finished = fin;
    bump();
    if (!fin) setTimeout(tick, 0);
  };
  setTimeout(tick, 0);
}

const FLOOR = () => lossFloor(RING, null).exact;

function medianDistance(mix: Mixture, xs: Pt[]) {
  const d = xs.map((p) => Math.min(...mix.m.map((m) => Math.hypot(p[0] - m[0], p[1] - m[1])))).sort((a, b) => a - b);
  return d[Math.floor(d.length / 2)];
}

export function LearnedDenoiser() {
  useSyncExternalStore(
    (l) => {
      learnListeners.add(l);
      return () => learnListeners.delete(l);
    },
    () => learnVersion,
  );
  const floor = useMemo(FLOOR, []);
  const run = learnRun;
  const levels = useMemo(() => evenLevels(100), []);
  const exact = useMemo(() => sample(exactDenoiser(RING), levels, noiseStart(N_SAMPLES, 5), 0, createRng(9)).xs, [levels]);
  const learned = useMemo(() => (run?.finished ? sample(run.net.denoiser(), levels, noiseStart(N_SAMPLES, 5), 0, createRng(9)).xs : null), [run?.finished, run, levels]);
  const finalLoss = useMemo(() => (run?.finished ? lossFloor(RING, run.net).learned : null), [run?.finished, run]);
  const dataMed = useMemo(() => medianDistance(RING, sampleData(RING, N_SAMPLES, createRng(3))), []);

  return (
    <Widget
      title="Learning the denoiser"
      wide
      caption={
        <>
          A small network (two hidden layers of 64, {LEARN.steps.toLocaleString('en-GB')} training steps of 64 examples) learns the ring’s denoiser from examples alone: take a clean point, pick a random step, add noise, and ask the network to predict the noise. Press <strong>Train</strong> and watch its loss approach the best any denoiser could do (the dashed line, computed with the exact one). Then compare samples from the two denoisers, both with 100 DDIM steps from the same starting noise.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10 }}>
        {!run && (
          <button type="button" className="btn primary" onClick={startLearning}>
            Train the network
          </button>
        )}
        {run && !run.finished && <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>training: step {run.done} of {LEARN.steps}</span>}
      </div>
      {run && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
          <div>
            <div className="field-label">Training loss (average over 100 steps)</div>
            <Plot width={330} height={200} x={[0, LEARN.steps]} y={[0, 1.2]} ariaLabel="Training loss against step, with the best possible loss" style={{ maxWidth: '100%', height: 'auto' }} xLabel="step" pad={30}>
              {({ x, y }) => (
                <g>
                  <line x1={x(0)} x2={x(LEARN.steps)} y1={y(floor)} y2={y(floor)} stroke="var(--c-good)" strokeDasharray="5 4" strokeWidth={1.5} />
                  <text x={x(LEARN.steps) - 4} y={y(floor) + 15} textAnchor="end" fontSize={11} fill="var(--c-good)">best possible {floor.toFixed(3)}</text>
                  <polyline fill="none" stroke="var(--accent)" strokeWidth={2} points={run.history.map((h) => `${x(h.step)},${y(Math.min(1.2, h.loss))}`).join(' ')} />
                </g>
              )}
            </Plot>
          </div>
          {learned && (
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <div>
                <div className="field-label">Exact denoiser</div>
                <Scatter size={170} label="Samples from the exact denoiser">
                  <Blobs mix={RING} />
                  <Dots pts={exact} r={1.1} opacity={0.4} />
                </Scatter>
              </div>
              <div>
                <div className="field-label">Learned denoiser</div>
                <Scatter size={170} label="Samples from the learned denoiser">
                  <Blobs mix={RING} />
                  <Dots pts={learned} r={1.1} opacity={0.4} color="var(--c-b)" />
                </Scatter>
              </div>
            </div>
          )}
        </div>
      )}
      {learned && finalLoss !== null && (
        <div style={{ display: 'grid', gap: 10, marginTop: 12 }}>
          <div className="readout">
            <span className="k">loss: learned / best possible </span>
            {finalLoss.toFixed(3)} / {floor.toFixed(3)}
            {'\n'}
            <span className="k">typical distance to the nearest blob centre (median) </span>
            {'\n'}
            <span className="k">  real data </span>
            {dataMed.toFixed(3)}
            <span className="k">   exact denoiser </span>
            {medianDistance(RING, exact).toFixed(3)}
            <span className="k">   learned </span>
            {medianDistance(RING, learned).toFixed(3)}
          </div>
          <div className="live-note">
            The network has learned the ring but not the blobs: its samples are smeared around the circle, about {(medianDistance(RING, learned) / dataMed).toFixed(1)} times further from the nearest blob centre than real data. Making tiny blobs needs a very precise denoiser at low noise, and its loss ({finalLoss.toFixed(3)}) is still above the best possible ({floor.toFixed(3)}). Two more things to notice. Even the perfect denoiser’s loss is far from 0: from a noisy point you can only ever guess the noise on average. And even the exact denoiser’s samples are a little too tight ({medianDistance(RING, exact).toFixed(3)} against {dataMed.toFixed(3)}), because the very last step returns a best guess, an average, which pulls each sample towards its blob’s centre.
          </div>
        </div>
      )}
    </Widget>
  );
}

// ------------------------------------------------------------------ 5. Lab: Spend the step budget

const LAB_KEY = 'nl-step-budget-lab-v1';
type LabEntry = { levels: number[]; eta: number; solved: boolean; fewest: number | null };
type LabState = Record<LabBriefId, LabEntry>;
const STARTER: LabEntry = { levels: [0.3, 0.6], eta: 0, solved: false, fewest: null };

function loadLab(): LabState {
  const d: LabState = { ring: { ...STARTER }, smiley: { ...STARTER }, lopsided: { ...STARTER } };
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (!raw) return d;
    const parsed = JSON.parse(raw) as Partial<LabState>;
    for (const b of BRIEFS) {
      const v = parsed[b.id];
      if (!v || !Array.isArray(v.levels)) continue;
      const levels = v.levels.filter((l) => (LAB_LEVELS as readonly number[]).includes(l));
      d[b.id] = { levels, eta: v.eta === 1 ? 1 : 0, solved: !!v.solved, fewest: typeof v.fewest === 'number' ? v.fewest : null };
    }
  } catch {
    // ignore storage errors
  }
  return d;
}

const LAB_HINTS: Record<LabBriefId, string> = {
  ring: 'Where does the work happen? Go back to the forward widget: below some signal share the shape has essentially gone, so stops there are wasted. And the final jump to clean data is a single best guess: how noisy can the last stop be before that guess lands between blobs?',
  smiley: 'The eyes and the blobs of the smile are close together, so the sampler must decide between neighbours late, at fairly low noise. Spend stops where those decisions happen. Compare the two samplers too.',
  lopsided: 'Watch the rare blob’s count. Samples have to commit to the rare blob early, while the noise is still large enough for some of them to wander that far.',
};

export function StepBudgetLab() {
  const [lab, setLab] = useState<LabState>(() => loadLab());
  const [briefIdx, setBriefIdx] = useState(0);
  const [hint, setHint] = useState(false);
  const [showOpt, setShowOpt] = useState(false);
  const brief = BRIEFS[briefIdx];
  const entry = lab[brief.id];
  const mix = LAB_MIX[brief.id];
  const run = useMemo(() => labRun(brief.id, entry.levels, entry.eta), [brief.id, entry.levels, entry.eta]);
  const pass = run.judge.tv <= brief.maxTv;
  const rec = LAB_RESULTS[brief.id];
  const savedRef = useRef(lab);
  savedRef.current = lab;

  const persist = (next: LabState) => {
    setLab(next);
    try {
      localStorage.setItem(LAB_KEY, JSON.stringify(next));
    } catch {
      // ignore storage errors
    }
  };
  const update = (patch: Partial<LabEntry>) => persist({ ...lab, [brief.id]: { ...entry, ...patch } });

  useEffect(() => {
    if (!pass) return;
    const cur = savedRef.current[brief.id];
    if (cur.solved && cur.fewest !== null && cur.fewest <= run.steps) return;
    persist({ ...savedRef.current, [brief.id]: { ...cur, solved: true, fewest: cur.fewest === null ? run.steps : Math.min(cur.fewest, run.steps) } });
  }, [pass, brief.id, run.steps]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = (l: number) => update({ levels: entry.levels.includes(l) ? entry.levels.filter((v) => v !== l) : [...entry.levels, l].sort((p, q) => p - q) });
  const rare = brief.id === 'lopsided' ? run.judge.counts[1] / LAB_SAMPLES : null;
  const row = rec.perSteps.find((p) => p.steps === rec.fewest)!;
  const below = rec.perSteps.find((p) => p.steps === rec.fewest - 1)!;

  return (
    <Widget
      title="Lab: Spend the Step Budget"
      wide
      caption={
        <>
          The denoiser is perfect; your job is the <strong>step plan</strong>. The sampler always starts from pure noise (ᾱ = 0) and always finishes on clean data; you choose the signal shares where it stops in between, and the sampler. Every stop costs one call to the denoiser. Results update instantly (2,000 samples, fixed starting noise). Objective: get the probability out of place under the brief’s limit. Optional: use as few steps as possible. Your plans are saved in this browser.
        </>
      }
    >
      <Segmented
        value={briefIdx}
        onChange={(i) => {
          setBriefIdx(i);
          setHint(false);
          setShowOpt(false);
        }}
        options={BRIEFS.map((b, i) => ({ value: i, label: `${lab[b.id].solved ? '✓ ' : ''}${b.title.split(' · ')[0]}` }))}
      />
      <div style={{ margin: '12px 0' }}>
        <div style={{ fontWeight: 650, marginBottom: 4 }}>{brief.title}</div>
        <div style={{ fontSize: 14.5, lineHeight: 1.55 }}>{brief.story}</div>
      </div>
      <div className="field-label">Stops: click a signal share to add or remove it</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
        <span className="btn small" style={{ opacity: 0.6, cursor: 'default' }}>start: 0</span>
        {LAB_LEVELS.map((l) => (
          <button key={l} type="button" className={`btn small${entry.levels.includes(l) ? ' primary' : ''}`} aria-pressed={entry.levels.includes(l)} onClick={() => toggle(l)}>
            {l}
          </button>
        ))}
        <span className="btn small" style={{ opacity: 0.6, cursor: 'default' }}>finish: 1</span>
      </div>
      <div className="btn-row" style={{ marginBottom: 12, gap: 14 }}>
        <Segmented
          value={entry.eta}
          onChange={(eta) => update({ eta })}
          options={[
            { value: 0, label: 'DDIM' },
            { value: 1, label: 'DDPM' },
          ]}
        />
        <button type="button" className="btn" onClick={() => setHint((h) => !h)}>{hint ? 'Hide hint' : 'Hint'}</button>
        {entry.solved && <span className="status good">Solved{entry.fewest !== null ? ` · fewest so far: ${entry.fewest} steps` : ''}</span>}
      </div>
      {hint && <div className="live-note" style={{ marginBottom: 12 }}>{LAB_HINTS[brief.id]}</div>}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, alignItems: 'start' }}>
        <Scatter label="Samples produced by your step plan">
          <Blobs mix={mix} />
          <Dots pts={run.xs} r={1.4} opacity={0.45} />
        </Scatter>
        <div style={{ display: 'grid', gap: 10 }}>
          <span className={`status ${pass ? 'good' : 'bad'}`} style={{ justifySelf: 'start' }}>
            {pass ? 'Meets the brief' : 'Not yet'}
          </span>
          <div className="readout">
            <span className="k">steps (denoiser calls) </span>
            {run.steps}
            {'\n'}
            <span className="k">plan </span>0 → {[...entry.levels, 1].join(' → ')}
            {'\n'}
            <span className="k">probability out of place </span>
            <span style={{ color: pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct(run.judge.tv)}</span> (limit {pct(brief.maxTv)})
            {'\n'}
            <span className="k">samples off every blob </span>
            {pct(run.judge.offShape)}
            {rare !== null && (
              <>
                {'\n'}
                <span className="k">share on the rare blob </span>
                {pct(rare)} (fair share 10.0%)
              </>
            )}
          </div>
          <div className="live-note">
            {pass
              ? `It works with ${run.steps} steps. ${entry.solved && entry.fewest !== null && entry.fewest > rec.fewest ? 'Optional: can you do it with fewer?' : ''}`
              : run.judge.offShape > 0.5
                ? 'Most samples end up between the blobs: the last jump to clean data starts from too much noise, so its best guess is an average. Add a stop closer to clean data.'
                : rare !== null && rare < 0.07
                  ? 'The rare blob is starved: samples commit to the big blob before any can reach the small one. Where do they decide?'
                  : 'Close. Look at where the misplaced samples are: in the gaps (a late decision, too few low-noise stops) or on the wrong blobs (an early decision)?'}
          </div>
        </div>
      </div>
      {entry.solved && (
        <div style={{ marginTop: 14 }}>
          <button type="button" className="btn" onClick={() => setShowOpt((s) => !s)}>{showOpt ? 'Hide' : 'Show'} the exhaustive search</button>
          {showOpt && (
            <div className="live-note" style={{ marginTop: 10 }}>
              Every plan with up to {rec.fewest} steps was tried, with both samplers. No plan with {rec.fewest - 1} steps works (the best reaches {pct(below.best)}). With <strong>{rec.fewest} steps</strong>, {row.passing} of {row.total} plans work ({row.ddim} with DDIM, {row.ddpm} with DDPM); the best reaches {pct(row.best)}. Starting from two other sets of random noise, the fewest steps needed is {rec.fewestWithOtherSeeds.join(' and ')}, so the answer isn’t luck.
            </div>
          )}
        </div>
      )}
    </Widget>
  );
}
