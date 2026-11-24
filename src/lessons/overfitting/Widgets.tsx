import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider, Toggle } from '../../components/controls';
import { Clip, FnPath, Plot } from '../../components/plot';
import { LogChart } from '../../components/LogChart';
import {
  NOISE,
  SHOWN_TEST,
  avgExpectedError,
  biasVariance,
  errorCurve,
  evalPoly,
  fitPoly,
  sample,
  truth,
  type Sample,
} from './fit';
import { CurveNet } from './net';

const clampY = (v: number) => Math.max(-40, Math.min(40, v));
const f3 = (v: number) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(3));

// ------------------------------------------------------------------ shared drawing pieces

/** The data plot: truth (dashed), training points (filled), optional test points (hollow), plus children. */
function DataPlot({
  s,
  showTruth,
  showTest,
  children,
  width = 440,
  height = 300,
  ariaLabel,
}: {
  s: Sample;
  showTruth: boolean;
  showTest: boolean;
  children?: ReactNode;
  width?: number;
  height?: number;
  ariaLabel: string;
}) {
  return (
    <Plot x={[-1.05, 1.05]} y={[-2.2, 2.2]} width={width} height={height} ariaLabel={ariaLabel} tickLabels={false}>
      <Clip>
        {showTruth && <FnPath f={truth} color="var(--text-faint)" width={2} dashed />}
        {children}
      </Clip>
      {({ x, y }) => (
        <g>
          {showTest &&
            SHOWN_TEST.x.map((xv, i) => (
              <circle key={'h' + i} cx={x(xv)} cy={y(SHOWN_TEST.y[i])} r={4} fill="none" stroke="var(--c-b)" strokeWidth={1.6} opacity={0.85} />
            ))}
          {s.x.map((xv, i) => (
            <circle key={'p' + i} cx={x(xv)} cy={y(s.y[i])} r={4.5} fill="var(--c-a)" stroke="var(--bg-elev)" strokeWidth={1.2} />
          ))}
        </g>
      )}
    </Plot>
  );
}

// ------------------------------------------------------------------ 1. the capacity dial

const SIZES = [12, 25, 50, 100] as const;

export function CapacityDial() {
  const [degree, setDegree] = useState(1);
  const [n, setN] = useState<(typeof SIZES)[number]>(12);
  const [seed, setSeed] = useState(1);
  const [showTruth, setShowTruth] = useState(false);
  const [showTest, setShowTest] = useState(true);
  const s = useMemo(() => sample(n, seed), [n, seed]);
  const curve = useMemo(() => errorCurve(s, 11), [s]);
  const fit = useMemo(() => fitPoly(s, degree), [s, degree]);
  const f = useMemo(() => (x: number) => clampY(evalPoly(fit, x)), [fit]);
  const cur = curve[degree];
  const bestTest = curve.reduce((b, r) => (r.test < b.test ? r : b), curve[0]);

  let note: ReactNode;
  if (degree <= 2 && cur.test > 1.3 * bestTest.test)
    note = (
      <>
        <strong>Underfitting.</strong> A {degree === 0 ? 'flat line' : degree === 1 ? 'straight line' : 'parabola'} can't bend enough to follow the pattern, so it
        misses it on the training points (error {f3(cur.train)}) as well as on new ones ({f3(cur.test)}, against {f3(bestTest.test)} for the best degree). More data
        wouldn't fix that: the model needs more flexibility. Turn the degree up.
      </>
    );
  else if (cur.test <= 1.15 * bestTest.test)
    note = (
      <>
        <strong>About right.</strong> The curve follows the broad pattern and ignores the scatter. Training error ({f3(cur.train)}) is a little <em>below</em> the
        test error ({f3(cur.test)}): it always is, because these coefficients were chosen to suit the training points. But the gap is small.
      </>
    );
  else if (cur.train < 0.5 * cur.test)
    note = (
      <>
        <strong>Overfitting.</strong> Training error keeps falling ({f3(cur.train)}) but test error has {cur.test > 10 * bestTest.test ? 'exploded' : 'risen'} to{' '}
        {f3(cur.test)}. The curve is bending to pass near each training point, so it's fitting the <em>noise</em> in them — wiggles that won't be there in new data.
        {n < 50 && ' Now try more training points.'}
      </>
    );
  else
    note = (
      <>
        Training error {f3(cur.train)}, test error {f3(cur.test)}. The best degree for this sample is {bestTest.degree} (test error {f3(bestTest.test)}).
      </>
    );

  return (
    <Widget
      title="The capacity dial"
      wide
      caption={
        <>
          Blue dots are <strong>training</strong> points: noisy measurements of a hidden curve. The purple line is the best polynomial of the chosen{' '}
          <strong>degree</strong> through them, found exactly by least squares. Hollow orange circles are fresh <strong>test</strong> points it never saw (the test error
          is measured on 2,000 of them). Turn the degree up from 0 to 11 and watch both errors on the chart. Then switch to 100 training points and do it again.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <div style={{ flex: '1 1 220px' }}>
          <Slider label="degree (flexibility)" value={degree} min={0} max={11} step={1} onChange={setDegree} format={(v) => String(v)} color="accent" />
        </div>
        <Segmented label="training points" value={n} onChange={setN} options={SIZES.map((v) => ({ value: v, label: String(v) }))} />
        <button className="btn small" onClick={() => setSeed(seed + 1)}>
          New sample of data
        </button>
        <Toggle label="show the hidden curve" checked={showTruth} onChange={setShowTruth} />
        <Toggle label="show test points" checked={showTest} onChange={setShowTest} />
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <DataPlot s={s} showTruth={showTruth} showTest={showTest} ariaLabel="Training points, test points and the fitted polynomial">
            <FnPath f={f} color="var(--accent)" width={2.6} samples={400} />
          </DataPlot>
        </div>
        <div className="widget-col" style={{ flex: '1 1 380px' }}>
          <LogChart
            series={[
              { points: curve.map((r) => [r.degree, Math.max(r.train, 1e-4)]), color: 'var(--c-a)', dots: true },
              { points: curve.map((r) => [r.degree, r.test]), color: 'var(--c-b)', dots: true },
            ]}
            xDomain={[-0.3, 11.3]}
            yDomain={[0.001, 1000]}
            marker={degree}
            xLabel="degree"
            xTicks={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
            hlines={[{ y: NOISE * NOISE, label: 'noise floor', color: 'var(--text-faint)' }]}
            ariaLabel="Training and test error for every degree"
          />
          <div className="muted" style={{ fontSize: 12.5, marginBottom: 8 }}>
            <span className="c-a">training error</span> and <span className="c-b">test error</span> (mean squared error, log scale) for every degree, on this sample.
            The noise floor is the error even the hidden curve itself would make.
          </div>
          <div className="readout">
            <span className="k">degree {String(degree).padEnd(3)} </span> {degree + 1} coefficients, {n} training points{'\n'}
            <span className="k">training error </span> <span className="c-a">{f3(cur.train)}</span>
            {'\n'}
            <span className="k">test error     </span> <span className="c-b">{f3(cur.test)}</span>
            {'\n'}
            <span className="k">gap            </span> {f3(cur.test - cur.train)}
          </div>
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 12 }}>
        {note}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. parallel universes (bias–variance)

const UNIVERSES_SHOWN = 20;
const UNIVERSES_MEASURED = 200;

export function ParallelUniverses() {
  const [degree, setDegree] = useState(1);
  const all = useMemo(() => Array.from({ length: 12 }, (_, d) => biasVariance(12, d, UNIVERSES_MEASURED)), []);
  const bv = all[degree];
  const fits = useMemo(() => Array.from({ length: UNIVERSES_SHOWN }, (_, u) => fitPoly(sample(12, u + 1), degree)), [degree]);
  const meanFit = useMemo(() => {
    const m = bv.mean;
    return (x: number) => {
      const t = ((x + 1) / 2) * (m.length - 1);
      const i = Math.max(0, Math.min(m.length - 2, Math.floor(t)));
      return clampY(m[i] + (m[i + 1] - m[i]) * (t - i));
    };
  }, [bv]);
  const fitFns = useMemo(() => fits.map((p) => (x: number) => clampY(evalPoly(p, x))), [fits]);
  const dominant = bv.bias2 > bv.variance ? 'bias' : 'variance';

  return (
    <Widget
      title="Twenty parallel universes"
      wide
      caption={
        <>
          Imagine running the whole experiment again in 20 parallel universes: same hidden curve (dashed), same amount of noise, but a different random set of 12
          training points each time. Each thin line is the fit from one universe; the thick magenta line is their <strong>average</strong>. Slide the degree and watch
          two different kinds of failure. The numbers are measured over {UNIVERSES_MEASURED} universes.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <div style={{ flex: '1 1 260px', maxWidth: 360 }}>
          <Slider label="degree" value={degree} min={0} max={11} step={1} onChange={setDegree} format={(v) => String(v)} color="accent" />
        </div>
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <Plot x={[-1.05, 1.05]} y={[-2.2, 2.2]} width={440} height={300} tickLabels={false} ariaLabel="Fits from twenty different training sets">
            <Clip>
              {fitFns.map((f, i) => (
                <FnPath key={i} f={f} color="var(--accent)" width={1.1} samples={200} />
              ))}
              <g opacity={1}>
                <FnPath f={truth} color="var(--text)" width={2} dashed />
                <FnPath f={meanFit} color="var(--c-grad)" width={3.2} samples={200} />
              </g>
            </Clip>
          </Plot>
        </div>
        <div className="widget-col" style={{ flex: '1 1 380px' }}>
          <LogChart
            series={[
              { points: all.map((r, d) => [d, Math.max(r.bias2, 1e-4)]), color: 'var(--c-grad)', dots: true },
              { points: all.map((r, d) => [d, Math.max(r.variance, 1e-4)]), color: 'var(--accent)', dots: true },
            ]}
            xDomain={[-0.3, 11.3]}
            yDomain={[0.0001, 1000]}
            marker={degree}
            xLabel="degree"
            xTicks={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
            ariaLabel="Bias squared and variance for every degree"
          />
          <div className="muted" style={{ fontSize: 12.5, marginBottom: 8 }}>
            <span className="c-grad">bias²</span> and <span style={{ color: 'var(--accent)' }}>variance</span> for every degree (log scale).
          </div>
          <div className="readout">
            <span className="k">bias²     </span> <span className="c-grad">{f3(bv.bias2)}</span>
            <span className="k">  average fit vs truth</span>
            {'\n'}
            <span className="k">variance  </span> <span style={{ color: 'var(--accent)' }}>{f3(bv.variance)}</span>
            <span className="k">  fits disagreeing</span>
            {'\n'}
            <span className="k">noise     </span> {f3(bv.noise)}
            <span className="k">  unavoidable (0.3²)</span>
            {'\n'}
            <span className="k">sum       </span> <strong>{f3(bv.total)}</strong>
            {'\n'}
            <span className="k">measured  </span> {f3(bv.direct)}
            <span className="k">  error on fresh points</span>
          </div>
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 12 }}>
        {dominant === 'bias' && bv.bias2 > 0.02 ? (
          <>
            <strong>Bias dominates.</strong> The universes agree with each other — the thin lines are bunched together — but they're all wrong in the{' '}
            <em>same</em> way: the average misses the hidden curve. That's underfitting: the model is too stiff to follow the pattern, whatever data it gets.
          </>
        ) : bv.variance > 0.1 ? (
          <>
            <strong>Variance dominates.</strong> On average the fits are close to the truth (bias² is only {f3(bv.bias2)}), but each universe's fit is pulled around by
            its own particular noise, so they disagree wildly. Any single one of them — and in real life you only get one — is likely to be badly off. That's
            overfitting.
          </>
        ) : (
          <>
            <strong>Balanced.</strong> Bias² ({f3(bv.bias2)}) and variance ({f3(bv.variance)}) are both small, and their sum plus the noise ({f3(bv.total)}) is close to
            the lowest error any degree achieves. Notice that the two parts always add up to the measured error.
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. a neural network, over time

const NET_MAX = 4000;
const RECORD_EVERY = 10;

export function NetOverTime() {
  const s = useMemo(() => sample(12, 1), []);
  const [seed, setSeed] = useState(1);
  const net = useRef<CurveNet | null>(null);
  const [hist, setHist] = useState<{ step: number; train: number; test: number }[]>([]);
  const [running, setRunning] = useState(false);
  const [, force] = useState(0);
  const [showTruth, setShowTruth] = useState(true);

  const reset = (sd = seed) => {
    const c = new CurveNet(s, sd);
    net.current = c;
    setHist([{ step: 0, train: c.trainError(), test: c.testError() }]);
    setRunning(false);
  };
  useEffect(() => reset(seed), [seed]); // eslint-disable-line react-hooks/exhaustive-deps

  const advance = (k: number) => {
    const c = net.current;
    if (!c) return;
    const add: { step: number; train: number; test: number }[] = [];
    for (let i = 0; i < k && c.steps < NET_MAX; i += RECORD_EVERY) {
      c.step(RECORD_EVERY);
      add.push({ step: c.steps, train: c.trainError(), test: c.testError() });
    }
    setHist((h) => [...h, ...add]);
    if (c.steps >= NET_MAX) setRunning(false);
    force((v) => v + 1);
  };

  useEffect(() => {
    if (!running) return;
    // ~300 steps a second: slow enough to see the sweet spot go by
    let raf = 0;
    let last = 0;
    const loop = (t: number) => {
      if (t - last >= 33) {
        last = t;
        advance(RECORD_EVERY);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [running]); // eslint-disable-line react-hooks/exhaustive-deps

  const c = net.current;
  const steps = c?.steps ?? 0;
  const cur = hist[hist.length - 1] ?? { step: 0, train: NaN, test: NaN };
  const best = hist.reduce((b, r) => (r.test < b.test ? r : b), hist[0] ?? cur);
  // the curve changes every frame, so a fresh function each render is intended
  const f = (x: number) => clampY(c ? c.predict(x) : 0);

  let note: ReactNode;
  if (steps === 0) note = <>A freshly initialised network: its curve is a gentle, random slope that ignores the data. Press <strong>Train</strong>.</>;
  else if (cur.test > 1.25 * best.test && best.step < steps)
    note = (
      <>
        <strong>Overfitting, in slow motion.</strong> Training error is {cur.train < 1e-3 ? 'essentially zero' : f3(cur.train)}, but test error has risen from its
        best, {f3(best.test)} at step {best.step}, to {f3(cur.test)}. Look at the curve: it's now bending to pass through individual training points. Had you stopped
        at step {best.step}, you'd have kept the better model.
      </>
    );
  else if (cur.test <= 1.05 * best.test && steps > 150)
    note = (
      <>
        <strong>The sweet spot.</strong> The network has the broad shape and hasn't started chasing individual points. Test error {f3(cur.test)} — about what the
        best polynomial managed. Keep training and watch what happens to the two curves.
      </>
    );
  else
    note = (
      <>
        <strong>Still underfitting.</strong> The network learns the broad shape first: training and test error fall together. Step {steps}.
      </>
    );

  return (
    <Widget
      title="A neural network, over time"
      wide
      caption={
        <>
          The same 12 training points, now fitted by a neural network with two hidden layers of 32 tanh neurons — 1,153 parameters, nearly a hundred per data point —
          trained by full-batch Adam with step size 0.01. Press <strong>Train</strong> and watch the curve and the two errors as training goes on. Try other random
          starts.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <div className="btn-row">
          <button className="btn primary" onClick={() => setRunning(!running)} disabled={steps >= NET_MAX}>
            {running ? '❚❚ Pause' : steps ? '▶ Continue' : '▶ Train'}
          </button>
          <button className="btn" onClick={() => advance(100)} disabled={running || steps >= NET_MAX}>
            +100 steps
          </button>
          <button className="btn" onClick={() => reset()}>
            Reset
          </button>
        </div>
        <Segmented label="random start" value={seed} onChange={setSeed} options={[1, 2, 3].map((v) => ({ value: v, label: `#${v}` }))} />
        <Toggle label="show the hidden curve" checked={showTruth} onChange={setShowTruth} />
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <DataPlot s={s} showTruth={showTruth} showTest={false} ariaLabel="Training points and the network's current curve">
            <FnPath f={f} color="var(--accent)" width={2.6} samples={300} />
          </DataPlot>
        </div>
        <div className="widget-col" style={{ flex: '1 1 380px' }}>
          <LogChart
            series={[
              { points: hist.map((r) => [r.step, Math.max(r.train, 1e-4)]), color: 'var(--c-a)' },
              { points: hist.map((r) => [r.step, r.test]), color: 'var(--c-b)' },
            ]}
            xDomain={[0, NET_MAX]}
            yDomain={[0.0001, 10]}
            marker={steps}
            xLabel="training steps"
            xTicks={[0, 1000, 2000, 3000, 4000]}
            hlines={[{ y: NOISE * NOISE, label: 'noise floor', color: 'var(--text-faint)' }]}
            ariaLabel="Training and test error during training"
          />
          <div className="readout">
            <span className="k">step           </span> {steps}
            {'\n'}
            <span className="k">training error </span> <span className="c-a">{f3(cur.train)}</span>
            {'\n'}
            <span className="k">test error     </span> <span className="c-b">{f3(cur.test)}</span>
            {'\n'}
            <span className="k">best test so far</span> {f3(best.test)} <span className="k">at step</span> {best.step}
          </div>
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 12 }}>
        {note}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. beyond the interpolation threshold

const DD_DEGREES = Array.from({ length: 61 }, (_, d) => d);

export function BeyondInterpolation() {
  const [degree, setDegree] = useState(11);
  const avg = useMemo(() => DD_DEGREES.map((d) => [d, avgExpectedError(12, d, 60)] as [number, number]), []);
  const s = useMemo(() => sample(12, 1), []);
  const fit = useMemo(() => fitPoly(s, degree), [s, degree]);
  const f = useMemo(() => (x: number) => clampY(evalPoly(fit, x)), [fit]);
  const here = avg[degree][1];
  return (
    <Widget
      title="Past the point of perfect fit"
      wide
      caption={
        <>
          Degrees above 11 have more coefficients than there are training points, so infinitely many polynomials pass exactly through all 12. We pick the one with the{' '}
          <strong>smallest coefficients</strong>. Slide the degree past 11 and watch both the fit and the chart of average error (over 60 training sets, computed
          exactly from the known curve).
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <div style={{ flex: '1 1 300px', maxWidth: 420 }}>
          <Slider label="degree" value={degree} min={0} max={60} step={1} onChange={setDegree} format={(v) => String(v)} color="accent" />
        </div>
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <DataPlot s={s} showTruth showTest={false} ariaLabel="A fit of the chosen degree">
            <FnPath f={f} color="var(--accent)" width={2.6} samples={600} />
          </DataPlot>
        </div>
        <div className="widget-col" style={{ flex: '1 1 380px' }}>
          <LogChart
            series={[{ points: avg, color: 'var(--c-b)', dots: false }]}
            xDomain={[0, 60]}
            yDomain={[0.05, 10000]}
            marker={degree}
            xLabel="degree"
            xTicks={[0, 11, 20, 30, 40, 50, 60]}
            hlines={[{ y: NOISE * NOISE, label: 'noise floor', color: 'var(--text-faint)' }]}
            ariaLabel="Average test error for degrees 0 to 60"
          />
          <div className="readout">
            <span className="k">degree {degree}: </span> average error <span className="c-b">{f3(here)}</span>
            {degree >= 12 ? '  (perfect fit to all 12 points)' : degree === 11 ? '  (exactly one perfect fit)' : ''}
          </div>
        </div>
      </div>
    </Widget>
  );
}
