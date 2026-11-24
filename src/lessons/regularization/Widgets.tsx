import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider, Toggle } from '../../components/controls';
import { Clip, FnPath, Plot } from '../../components/plot';
import { LogChart } from '../../components/LogChart';
import { NOISE, MAX_STEPS, Run, TEST_SET, coefMse, evalCoef, ridgeFit, trainSet, truth, valSet, type Cure } from './reg';
import { RESULTS } from './results';
import { DATASETS, GRID, runSetting, type Setting } from './search';

const clampY = (v: number) => Math.max(-40, Math.min(40, v));
const f3 = (v: number) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(3));

const TRAIN = trainSet(1);
const VAL = valSet(1);

/** Training points (filled blue), validation points (hollow green squares), the hidden curve, plus children. */
function DataPlot({ showTruth, children, ariaLabel, showVal = true }: { showTruth: boolean; children?: ReactNode; ariaLabel: string; showVal?: boolean }) {
  return (
    <Plot x={[-1.05, 1.05]} y={[-2.2, 2.2]} width={440} height={290} ariaLabel={ariaLabel} tickLabels={false}>
      <Clip>
        {showTruth && <FnPath f={truth} color="var(--text-faint)" width={2} dashed />}
        {children}
      </Clip>
      {({ x, y }) => (
        <g>
          {showVal &&
            VAL.x.map((xv, i) => (
              <rect key={'v' + i} x={x(xv) - 4} y={y(VAL.y[i]) - 4} width={8} height={8} fill="none" stroke="var(--c-c)" strokeWidth={1.8} />
            ))}
          {TRAIN.x.map((xv, i) => (
            <circle key={'p' + i} cx={x(xv)} cy={y(TRAIN.y[i])} r={4.5} fill="var(--c-a)" stroke="var(--bg-elev)" strokeWidth={1.2} />
          ))}
        </g>
      )}
    </Plot>
  );
}

type Point = { step: number; train: number; val: number; test: number };

/** Shared live-training loop: ~300 steps a second, recording errors every 10 steps. */
function useTraining(make: () => Run, deps: unknown[], onRecord?: (run: Run, p: Point) => boolean) {
  const run = useRef<Run | null>(null);
  const [hist, setHist] = useState<Point[]>([]);
  const [running, setRunning] = useState(false);
  const [, force] = useState(0);
  const measure = (r: Run): Point => ({ step: r.steps, train: r.errorOn(r.train), val: r.errorOn(VAL), test: r.errorOn(TEST_SET) });
  const reset = () => {
    const r = make();
    run.current = r;
    setHist([measure(r)]);
    setRunning(false);
  };
  useEffect(reset, deps); // eslint-disable-line react-hooks/exhaustive-deps
  const advance = (k: number) => {
    const r = run.current;
    if (!r) return;
    const add: Point[] = [];
    let stop = false;
    for (let i = 0; i < k && r.steps < MAX_STEPS && !stop; i += 10) {
      r.step(10);
      const p = measure(r);
      add.push(p);
      if (onRecord && onRecord(r, p)) stop = true;
    }
    setHist((h) => [...h, ...add]);
    if (stop || r.steps >= MAX_STEPS) setRunning(false);
    force((v) => v + 1);
  };
  useEffect(() => {
    if (!running) return;
    let raf = 0;
    let last = 0;
    const loop = (t: number) => {
      if (t - last >= 33) {
        last = t;
        advance(10);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [running]); // eslint-disable-line react-hooks/exhaustive-deps
  return { run, hist, running, setRunning, reset, advance };
}

function TrainButtons({ t, done, label = 'Train' }: { t: ReturnType<typeof useTraining>; done: boolean; label?: string }) {
  const steps = t.run.current?.steps ?? 0;
  return (
    <div className="btn-row">
      <button className="btn primary" onClick={() => t.setRunning(!t.running)} disabled={done}>
        {t.running ? '❚❚ Pause' : steps ? '▶ Continue' : `▶ ${label}`}
      </button>
      <button className="btn" onClick={() => t.advance(100)} disabled={t.running || done}>
        +100 steps
      </button>
      <button className="btn" onClick={t.reset}>
        Reset
      </button>
    </div>
  );
}

const STEP_TICKS = [0, 1000, 2000, 3000, 4000];
const errorSeries = (hist: Point[]) => [
  { points: hist.map((r) => [r.step, Math.max(r.train, 1e-4)] as [number, number]), color: 'var(--c-a)' },
  { points: hist.map((r) => [r.step, r.val] as [number, number]), color: 'var(--c-c)' },
  { points: hist.map((r) => [r.step, r.test] as [number, number]), color: 'var(--c-b)', dashed: true },
];
const ErrorLegend = () => (
  <div className="muted" style={{ fontSize: 12.5, marginBottom: 8 }}>
    <span className="c-a">training</span>, <span className="c-c">validation</span> and <span className="c-b">test</span> error (log scale). Training only ever sees the
    blue points; you may look at validation; the test error is shown here for teaching, but in real life you wouldn't see it until the very end.
  </div>
);

// ------------------------------------------------------------------ 1. early stopping

const PATIENCES = [50, 100, 200, 400, 800] as const;

export function EarlyStopping() {
  const [patience, setPatience] = useState<number>(200);
  const best = useRef<{ val: number; step: number; W: number[][][]; b: number[][] } | null>(null);
  const [stoppedAt, setStoppedAt] = useState<number | null>(null);
  const t = useTraining(
    () => {
      best.current = null;
      setStoppedAt(null);
      return new Run(TRAIN, { l2: 0, dropout: 0 }, 1);
    },
    [patience],
    (r, p) => {
      if (!best.current || p.val < best.current.val) {
        const s = r.snapshot();
        best.current = { val: p.val, step: p.step, ...s };
      } else if (p.step - best.current.step >= patience) {
        setStoppedAt(p.step);
        return true;
      }
      return false;
    },
  );
  const r = t.run.current;
  const steps = r?.steps ?? 0;
  const cur = t.hist[t.hist.length - 1];
  const b = best.current;
  // the kept model: the snapshot with the best validation error so far
  const kept = useMemo(() => {
    if (!b || !r) return null;
    const m = new Run(TRAIN, { l2: 0, dropout: 0 }, 1);
    m.net.W = b.W;
    m.net.b = b.b;
    return { f: (x: number) => clampY(m.predict(x)), test: m.errorOn(TEST_SET) };
  }, [b?.step]); // eslint-disable-line react-hooks/exhaustive-deps
  const curF = (x: number) => clampY(r ? r.predict(x) : 0);

  let note: ReactNode;
  if (!steps) note = <>Press <strong>Train</strong>. Training stops by itself once the validation error hasn't improved for {patience} steps.</>;
  else if (stoppedAt !== null && b && kept)
    note = (
      <>
        <strong>Stopped at step {stoppedAt}.</strong> The validation error was last at its best ({f3(b.val)}) at step {b.step}, and {patience} steps without
        improvement ran out the patience. We throw away everything since and keep the step-{b.step} weights (solid line): test error <strong>{f3(kept.test)}</strong>.
        {b.step < 200
          ? ' That was early: with so little patience, a small wobble in the validation error can end training before the network has learned the shape. Try more patience.'
          : ' Press Continue to see what training on would have done to the test error.'}
      </>
    );
  else if (b && kept && steps > b.step + 20)
    note = (
      <>
        Validation error was best at step {b.step} ({f3(b.val)}) and hasn't improved for {steps - b.step} steps. The solid line is the model we'd keep (test error{' '}
        {f3(kept.test)}); the faint one is where training has wandered since (test error {f3(cur.test)}).
      </>
    );
  else note = <>Validation error is still improving (best so far {b ? f3(b.val) : '—'} at step {b?.step ?? 0}): keep going.</>;

  return (
    <Widget
      title="Early stopping"
      wide
      caption={
        <>
          The same 12 training points (blue) and the same 1,153-parameter network as in the last lesson, plus 12 <strong>validation</strong> points (green squares) it
          never trains on. Every 10 steps we check the validation error and remember the best weights so far. <strong>Patience</strong> is how many steps we're willing
          to wait for a new best before giving up. Train with a few different patience values.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <TrainButtons t={t} done={steps >= MAX_STEPS} />
        <Segmented label="patience (steps)" value={patience} onChange={setPatience} options={PATIENCES.map((p) => ({ value: p, label: String(p) }))} />
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <DataPlot showTruth={false} ariaLabel="Training and validation points, the kept model and the current model">
            {steps > 0 && <FnPath f={curF} color="var(--text-faint)" width={1.6} samples={300} />}
            {kept && <FnPath f={kept.f} color="var(--accent)" width={2.8} samples={300} />}
          </DataPlot>
        </div>
        <div className="widget-col" style={{ flex: '1 1 380px' }}>
          <LogChart
            series={errorSeries(t.hist)}
            xDomain={[0, MAX_STEPS]}
            yDomain={[0.0001, 10]}
            marker={b?.step}
            xLabel="training steps"
            xTicks={STEP_TICKS}
            hlines={[{ y: NOISE * NOISE, label: 'noise floor', color: 'var(--text-faint)' }]}
            ariaLabel="Training, validation and test error during training"
          />
          <ErrorLegend />
          <div className="readout">
            <span className="k">step                </span> {steps}
            {stoppedAt !== null ? ' (stopped)' : ''}
            {'\n'}
            <span className="k">best validation     </span> <span className="c-c">{b ? f3(b.val) : '—'}</span> <span className="k">at step</span> {b?.step ?? '—'}
            {'\n'}
            <span className="k">test, kept weights  </span> <span className="c-b">{kept ? f3(kept.test) : '—'}</span>
            {'\n'}
            <span className="k">test, latest weights</span> <span className="c-b">{cur ? f3(cur.test) : '—'}</span>
          </div>
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 12 }}>
        {note}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. weight decay on the polynomial (exact)

/** Slider position −9 means λ = 0; otherwise λ = 10^position. */
const lambdaAt = (v: number) => (v <= -9 ? 0 : 10 ** v);
const fmtLambda = (l: number) => (l === 0 ? '0' : l >= 0.01 ? String(Number(l.toPrecision(2))) : l.toExponential(0).replace('e-', '×10⁻').replace(/⁻(\d)/, (_, d) => '⁻' + '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]));
const SWEEP = Array.from({ length: 37 }, (_, i) => -8 + i * 0.25); // 1e-8 … 10

export function WeightDecayDial() {
  const [pos, setPos] = useState(-9);
  const [showTruth, setShowTruth] = useState(true);
  const lambda = lambdaAt(pos);
  const coef = useMemo(() => ridgeFit(TRAIN, 11, lambda), [lambda]);
  const f = useMemo(() => (x: number) => clampY(evalCoef(coef, x)), [coef]);
  const sweep = useMemo(
    () =>
      SWEEP.map((p) => {
        const c = ridgeFit(TRAIN, 11, 10 ** p);
        return { l: 10 ** p, train: coefMse(c, TRAIN), val: coefMse(c, VAL), test: coefMse(c, TEST_SET) };
      }),
    [],
  );
  const bestVal = sweep.reduce((b, r) => (r.val < b.val ? r : b));
  const train = coefMse(coef, TRAIN);
  const val = coefMse(coef, VAL);
  const test = coefMse(coef, TEST_SET);
  const size = coef.slice(1).reduce((t, c) => t + c * c, 0);
  const maxBar = 16;

  let note: ReactNode;
  if (lambda < 1e-5)
    note = (
      <>
        <strong>{lambda === 0 ? 'No penalty.' : 'Almost no penalty.'}</strong> The fit threads every point, and to do it the coefficients have grown huge (their
        squares add up to {f3(size)}). Big coefficients that nearly cancel each other are exactly what produce the wild swings between points. Slide λ up.
      </>
    );
  else if (lambda > 0.15)
    note = (
      <>
        <strong>Too much penalty.</strong> Every coefficient is being squeezed towards zero so hard that the curve can no longer follow the pattern: training error{' '}
        {f3(train)}, test {f3(test)}. You've traded overfitting for underfitting.
      </>
    );
  else if (val <= 1.25 * bestVal.val)
    note = (
      <>
        <strong>About right.</strong> Same 12 coefficients, same degree 11 — but now they're small (squares adding up to {f3(size)}), and the curve is calm. Validation
        error {f3(val)} is near its best ({f3(bestVal.val)} at λ ≈ {fmtLambda(bestVal.l)}), and so is the test error: {f3(test)}, down from 70.7 with no penalty.
      </>
    );
  else
    note = (
      <>
        The penalty is starting to bite: the coefficients' squares now add up to {f3(size)}, and the curve's wildest swings have gone. Validation error {f3(val)}.
      </>
    );

  return (
    <Widget
      title="Weight decay on the wildest polynomial"
      wide
      caption={
        <>
          The degree-11 polynomial from the last lesson: 12 coefficients, 12 training points, test error 70.7. Now we add a penalty for large coefficients, of strength{' '}
          <strong>λ</strong>, and find the exact best fit again. Slide λ up from 0 and watch the curve, the size of each coefficient (bars), and the three errors.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <div style={{ flex: '1 1 280px', maxWidth: 420 }}>
          <Slider label="penalty strength λ" value={pos} min={-9} max={1} step={0.25} onChange={setPos} format={(v) => fmtLambda(lambdaAt(v))} color="accent" />
        </div>
        <Toggle label="show the hidden curve" checked={showTruth} onChange={setShowTruth} />
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <DataPlot showTruth={showTruth} ariaLabel="The degree-11 fit with the chosen penalty">
            <FnPath f={f} color="var(--accent)" width={2.6} samples={500} />
          </DataPlot>
          <div className="field-label" style={{ marginTop: 6 }}>
            coefficient sizes (c₀ … c₁₁)
          </div>
          <svg viewBox="0 0 440 70" width={440} height={70} role="img" aria-label="Size of each coefficient">
            <line x1={0} x2={440} y1={35} y2={35} className="plot-axis" />
            {coef.map((c, i) => {
              const h = (Math.max(-maxBar, Math.min(maxBar, c)) / maxBar) * 33;
              return <rect key={i} x={8 + i * 36} width={24} y={h > 0 ? 35 - h : 35} height={Math.max(1, Math.abs(h))} fill={i === 0 ? 'var(--text-faint)' : 'var(--accent)'} rx={2} />;
            })}
          </svg>
        </div>
        <div className="widget-col" style={{ flex: '1 1 380px' }}>
          <LogChart
            series={[
              { points: sweep.map((r) => [r.l, Math.max(r.train, 1e-4)] as [number, number]), color: 'var(--c-a)' },
              { points: sweep.map((r) => [r.l, r.val] as [number, number]), color: 'var(--c-c)' },
              { points: sweep.map((r) => [r.l, r.test] as [number, number]), color: 'var(--c-b)', dashed: true },
            ]}
            xDomain={[1e-8, 10]}
            logX
            yDomain={[0.0001, 100]}
            marker={Math.max(lambda, 1e-8)}
            xLabel="λ"
            xTicks={[1e-8, 1e-6, 1e-4, 1e-2, 1]}
            xTickFormat={(t) => fmtLambda(t)}
            hlines={[{ y: NOISE * NOISE, label: 'noise floor', color: 'var(--text-faint)' }]}
            ariaLabel="Training, validation and test error for every penalty strength"
          />
          <div className="muted" style={{ fontSize: 12.5, marginBottom: 8 }}>
            <span className="c-a">training</span>, <span className="c-c">validation</span> and <span className="c-b">test</span> error for every λ (both axes
            logarithmic).
          </div>
          <div className="readout">
            <span className="k">λ                 </span> {fmtLambda(lambda)}
            {'\n'}
            <span className="k">Σ c² (c₁…c₁₁)     </span> {f3(size)}
            {'\n'}
            <span className="k">training error    </span> <span className="c-a">{f3(train)}</span>
            {'\n'}
            <span className="k">validation error  </span> <span className="c-c">{f3(val)}</span>
            {'\n'}
            <span className="k">test error        </span> <span className="c-b">{f3(test)}</span>
          </div>
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 12 }}>
        {note}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. the cure bench (network)

const L2S = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1];
const DROPS = [0.1, 0.2, 0.3, 0.5];

function MaskGrid({ masks }: { masks: number[][][] }) {
  // masks[example][layer][neuron]; draw one row per example, both hidden layers side by side
  if (!masks.length) return null;
  const cell = 6;
  const gap = 10;
  const w = masks[0].reduce((t, L) => t + L.length * cell, 0) + gap * (masks[0].length - 1);
  return (
    <svg viewBox={`0 0 ${w} ${masks.length * cell}`} width={w} height={masks.length * cell} role="img" aria-label="Which neurons each training example saw switched off this step">
      {masks.map((ex, r) => {
        let x0 = 0;
        return ex.map((L, l) => {
          const g = (
            <g key={`${r}-${l}`}>
              {L.map((v, j) => (
                <rect key={j} x={x0 + j * cell} y={r * cell} width={cell - 1} height={cell - 1} fill={v === 0 ? 'var(--bg-sunken)' : 'var(--accent)'} opacity={v === 0 ? 1 : 0.85} />
              ))}
            </g>
          );
          x0 += L.length * cell + gap;
          return g;
        });
      })}
    </svg>
  );
}

export function CureBench() {
  const [kind, setKind] = useState<'none' | 'l2' | 'dropout'>('l2');
  const [l2, setL2] = useState(1e-2);
  const [drop, setDrop] = useState(0.3);
  const [showTruth, setShowTruth] = useState(false);
  const cure: Cure = kind === 'l2' ? { l2, dropout: 0 } : kind === 'dropout' ? { l2: 0, dropout: drop } : { l2: 0, dropout: 0 };
  const t = useTraining(() => new Run(TRAIN, cure, 1), [kind, l2, drop]);
  const r = t.run.current;
  const steps = r?.steps ?? 0;
  const cur = t.hist[t.hist.length - 1];
  const f = (x: number) => clampY(r ? r.predict(x) : 0);
  const best = t.hist.reduce((b, p) => (p.val < b.val ? p : b), t.hist[0] ?? cur);

  let note: ReactNode;
  if (!steps) note = <>Press <strong>Train</strong>. With no cure, this exact run ends at test error 0.213 (the last lesson's overfitting run, step 4,000).</>;
  else if (kind === 'none')
    note = (
      <>
        No cure: the network is free to chase the noise. Step {steps}: training error {f3(cur.train)}, test error {f3(cur.test)}.
      </>
    );
  else if (kind === 'l2')
    note = (
      <>
        <strong>Weight decay, λ = {l2}.</strong> Every step, each weight is pulled a little towards zero, so the network can only keep large weights that earn their
        keep on the training data. Sum of squared weights: {f3(r!.weightNorm2())}. Step {steps}: training error {f3(cur.train)} (it can no longer reach zero),
        test error {f3(cur.test)}. {l2 >= 0.1 ? 'This much decay is starting to flatten the curve: underfitting.' : ''}
      </>
    );
  else
    note = (
      <>
        <strong>Dropout, p = {drop}.</strong> On every step, each training example is processed by a different random thinned network, with about {Math.round(drop * 100)}%
        of the hidden neurons switched off (dark cells below; one row per example). No neuron can rely on a particular partner being there. Training and test errors are
        measured with the full network. Step {steps}: training {f3(cur.train)}, test {f3(cur.test)}.
      </>
    );

  return (
    <Widget
      title="The cure bench"
      wide
      caption={
        <>
          The same overfitting network and data. Pick a cure and its strength, then train for up to 4,000 steps, as in the last lesson. Compare the final test error with
          the unregularised run's 0.213, and look at how each cure changes the <em>shape</em> of the curve.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <TrainButtons t={t} done={steps >= MAX_STEPS} />
        <Segmented
          label="cure"
          value={kind}
          onChange={setKind}
          options={[
            { value: 'none', label: 'none' },
            { value: 'l2', label: 'weight decay' },
            { value: 'dropout', label: 'dropout' },
          ]}
        />
        {kind === 'l2' && <Segmented label="λ" value={l2} onChange={setL2} options={L2S.map((v) => ({ value: v, label: String(v) }))} />}
        {kind === 'dropout' && <Segmented label="drop probability p" value={drop} onChange={setDrop} options={DROPS.map((v) => ({ value: v, label: String(v) }))} />}
        <Toggle label="show the hidden curve" checked={showTruth} onChange={setShowTruth} />
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <DataPlot showTruth={showTruth} ariaLabel="Training and validation points and the network's current curve">
            <FnPath f={f} color="var(--accent)" width={2.6} samples={300} />
          </DataPlot>
          {kind === 'dropout' && r && r.net.lastMasks.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <div className="field-label">this step's thinned networks: hidden layer 1 | hidden layer 2</div>
              <MaskGrid masks={r.net.lastMasks} />
            </div>
          )}
        </div>
        <div className="widget-col" style={{ flex: '1 1 380px' }}>
          <LogChart
            series={errorSeries(t.hist)}
            xDomain={[0, MAX_STEPS]}
            yDomain={[0.0001, 10]}
            marker={steps}
            xLabel="training steps"
            xTicks={STEP_TICKS}
            hlines={[{ y: NOISE * NOISE, label: 'noise floor', color: 'var(--text-faint)' }]}
            ariaLabel="Training, validation and test error during training"
          />
          <ErrorLegend />
          <div className="readout">
            <span className="k">step              </span> {steps}
            {'\n'}
            <span className="k">training error    </span> <span className="c-a">{cur ? f3(cur.train) : '—'}</span>
            {'\n'}
            <span className="k">validation error  </span> <span className="c-c">{cur ? f3(cur.val) : '—'}</span>
            <span className="k"> (best {best ? f3(best.val) : '—'})</span>
            {'\n'}
            <span className="k">test error        </span> <span className="c-b">{cur ? f3(cur.test) : '—'}</span>
            {'\n'}
            <span className="k">Σ w²              </span> {r ? f3(r.weightNorm2()) : '—'}
          </div>
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 12 }}>
        {note}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. the fair comparison

const METHODS = [
  { id: 'none', name: 'no cure' },
  { id: 'early', name: 'early stopping' },
  { id: 'l2', name: 'weight decay' },
  { id: 'dropout', name: 'dropout' },
] as const;

/** Each method's setting with the lowest mean validation error, from the recorded search. */
export function winners() {
  return METHODS.map((m) => {
    const rows = RESULTS.filter((r) => r.method === m.id);
    return { ...m, rows, best: rows.reduce((b, r) => (r.val < b.val ? r : b)) };
  });
}

/**
 * Browsers and Node can differ in the last binary digit of Math.exp and friends, and over 4,000
 * training steps such differences grow. Measured in Chromium against the Node-recorded search: the
 * 10-dataset means differ by up to 0.0014 (no cure; single runs by up to 0.009), early stopping and
 * dropout by under 1e-6.
 */
const RERUN_TOLERANCE = 0.003;

export function FairComparison() {
  const ws = useMemo(winners, []);
  const [method, setMethod] = useState<(typeof METHODS)[number]['id']>('l2');
  const [live, setLive] = useState<Record<string, number[]>>({});
  const [busy, setBusy] = useState(false);
  const maxTest = 0.3;
  const shown = ws.find((w) => w.id === method)!;

  const rerun = () => {
    setBusy(true);
    setLive({});
    const jobs: { w: (typeof ws)[number]; k: number }[] = [];
    for (const w of ws) for (let k = 1; k <= DATASETS; k++) jobs.push({ w, k });
    let i = 0;
    const next = () => {
      if (i >= jobs.length) {
        setBusy(false);
        return;
      }
      const { w, k } = jobs[i++];
      const s = GRID.find((g) => g.method === w.best.method && g.strength === w.best.strength) as Setting;
      const o = runSetting(s, k);
      setLive((L) => ({ ...L, [w.id]: [...(L[w.id] ?? []), o.test] }));
      setTimeout(next, 0);
    };
    setTimeout(next, 30);
  };

  return (
    <Widget
      title="A fair race between the cures"
      wide
      caption={
        <>
          Each cure has a strength to choose, and the choice changes everything. So each one gets its <strong>own best setting</strong>, picked from a grid by its average{' '}
          <strong>validation</strong> error over {DATASETS} different datasets of 12 points (the test set plays no part in the choice). The bars show the test error
          of each winner, averaged over the same {DATASETS} datasets. Pick a cure below the bars to see its whole search.
        </>
      }
    >
      <div role="img" aria-label="Average test error of each cure at its best setting" style={{ display: 'grid', gap: 8 }}>
        {ws.map((w) => {
          const L = live[w.id];
          const liveMean = L && L.length === DATASETS ? L.reduce((a, b) => a + b, 0) / DATASETS : null;
          const pct = (v: number) => `${(Math.min(v, maxTest) / maxTest) * 100}%`;
          return (
            <div key={w.id} style={{ display: 'grid', gridTemplateColumns: 'minmax(96px, 130px) 1fr', gap: 10, alignItems: 'center' }}>
              <div style={{ textAlign: 'right', lineHeight: 1.2 }}>
                <div style={{ fontWeight: w.id === method ? 700 : 500 }}>{w.name}</div>
                {w.id !== 'none' && <div className="faint" style={{ fontSize: 11.5 }}>{w.best.label}</div>}
              </div>
              <div style={{ position: 'relative', height: 28, background: 'var(--bg-sunken)', borderRadius: 4 }}>
                <div style={{ width: pct(w.best.test), height: '100%', borderRadius: 4, background: w.id === 'none' ? 'var(--text-faint)' : 'var(--accent)', opacity: 0.85 }} />
                <div style={{ position: 'absolute', left: pct(NOISE * NOISE), top: 0, bottom: 0, borderLeft: '1.5px dashed var(--text-faint)' }} />
                <span className="mono" style={{ position: 'absolute', left: 6, top: 5, fontSize: 12.5, color: 'var(--bg-elev)', fontWeight: 700 }}>
                  {w.best.test.toFixed(3)}
                </span>
                <span className="mono" style={{ position: 'absolute', right: 6, top: 6, fontSize: 11.5, color: 'var(--text-muted)' }}>
                  {liveMean !== null
                    ? Math.abs(liveMean - w.best.test) < RERUN_TOLERANCE
                      ? `✓ re-run ${liveMean.toFixed(4)}`
                      : `✗ re-run ${liveMean.toFixed(4)}`
                    : L
                      ? `${L.length}/${DATASETS}`
                      : ''}
                </span>
              </div>
            </div>
          );
        })}
        <div className="faint" style={{ fontSize: 11.5, paddingLeft: 140 }}>
          bar length: average test error (scale 0 to {maxTest}); dashed line: the noise floor, {NOISE * NOISE}
        </div>
      </div>
      <div className="controls inline" style={{ margin: '10px 0' }}>
        <Segmented label="show the search for" value={method} onChange={setMethod} options={METHODS.filter((m) => m.id !== 'none').map((m) => ({ value: m.id, label: m.name }))} />
        <button className="btn" onClick={rerun} disabled={busy}>
          {busy ? 'Training…' : `Re-run the ${4 * DATASETS} winning runs here`}
        </button>
      </div>
      {Object.keys(live).length > 0 && (
        <p className="muted" style={{ fontSize: 13, margin: '0 0 10px' }}>
          These runs are training in your browser right now, from scratch. Expect agreement with the recorded results to about the third decimal place, not exactly:
          browsers' maths libraries can differ in the last binary digit of functions like <code>exp</code>, and over 4,000 steps those tiny differences grow.
        </p>
      )}
      <table className="num-table" style={{ width: '100%', maxWidth: 560 }}>
        <thead>
          <tr>
            <th style={{ textAlign: 'left' }}>setting</th>
            <th>validation error</th>
            <th>test error</th>
            {method === 'early' && <th>stopped at (avg step)</th>}
          </tr>
        </thead>
        <tbody>
          {shown.rows.map((r) => (
            <tr key={r.label} style={r === shown.best ? { fontWeight: 700, color: 'var(--accent-text)' } : undefined}>
              <td style={{ textAlign: 'left', fontFamily: 'var(--font-ui)' }}>
                {r.label}
                {r === shown.best ? ' ← chosen' : ''}
              </td>
              <td>{r.val.toFixed(4)}</td>
              <td>{r.test.toFixed(4)}</td>
              {method === 'early' && <td>{Math.round(r.stoppedAt)}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </Widget>
  );
}
