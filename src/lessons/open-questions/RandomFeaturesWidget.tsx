// authored-by: Gemini Antigravity
import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Plot } from '../../components/plot';
import { LogChart } from '../../components/LogChart';
import { Segmented, Slider } from '../../components/controls';
import {
  CURVE_CLEAN,
  CURVE_NOISY,
  DEFAULT_NOISE,
  EVAL_GRID,
  EVAL_TRUE,
  N_TRAIN,
  P_VALUES,
  SHOWCASE_CLEAN_DATA,
  SHOWCASE_FEATURE_SEED,
  SHOWCASE_NOISY_DATA,
  fitRandomFeatures,
  predictSingle,
} from './doubleDescent';

type NoiseMode = 'noisy' | 'clean';

const PRESETS: { label: string; p: number }[] = [
  { label: 'Underfit (p=4)', p: 4 },
  { label: 'Sweet spot (p=16)', p: 16 },
  { label: 'Peak (p=24)', p: 24 },
  { label: 'Over-param (p=160)', p: 160 },
];

export function RandomFeaturesWidget() {
  const [pIdx, setPIdx] = useState<number>(P_VALUES.indexOf(24));
  const [noiseMode, setNoiseMode] = useState<NoiseMode>('noisy');

  const p = P_VALUES[pIdx];
  const noisy = noiseMode === 'noisy';
  const data = noisy ? SHOWCASE_NOISY_DATA : SHOWCASE_CLEAN_DATA;
  const curve = noisy ? CURVE_NOISY : CURVE_CLEAN;

  const fit = useMemo(
    () => fitRandomFeatures(data, p, SHOWCASE_FEATURE_SEED),
    [data, p],
  );

  const curvePoint = curve[pIdx];

  const fitCurvePts = useMemo<[number, number][]>(
    () => EVAL_GRID.map((x) => [x, predictSingle(fit.features, fit.coef, x)]),
    [fit],
  );

  const trueCurvePts = useMemo<[number, number][]>(
    () => EVAL_GRID.map((x, i) => [x, EVAL_TRUE[i]]),
    [],
  );

  const testSeries = useMemo<[number, number][]>(
    () => curve.map((c) => [c.p, Math.max(1e-3, c.testMse)]),
    [curve],
  );

  const trainSeries = useMemo<[number, number][]>(
    () => curve.map((c) => [c.p, Math.max(1e-3, c.trainMse)]),
    [curve],
  );

  const regimeNote = (() => {
    if (!noisy) {
      if (p < 16) return `Without label noise, adding random features steadily improves the fit (median test MSE ${curvePoint.testMse.toFixed(4)}).`;
      if (p === N_TRAIN) return 'Zero label noise means no spike at p = n: fitting clean points exactly does not require huge cancelling weights.';
      return `With zero noise, the fit stays tight across the whole over-parameterised regime (median test MSE ${curvePoint.testMse.toFixed(4)}).`;
    }
    if (p < 12) {
      return `Underfitting (p = ${p} < n = ${N_TRAIN}): too few random waves to bend with the true curve.`;
    }
    if (p <= 18) {
      return `Classical sweet spot (p = ${p}): enough features to follow the signal without chasing every noisy point (median test MSE ${curvePoint.testMse.toFixed(3)}).`;
    }
    if (p === N_TRAIN) {
      return `Interpolation threshold (p = n = ${N_TRAIN}): training error hits zero, but there is only one exact fit — forcing it through noisy dots blows weights up to ‖c‖ ≈ ${Math.round(curvePoint.weightNorm).toLocaleString()} and spikes median test MSE to ${curvePoint.testMse.toFixed(2)}.`;
    }
    if (p < N_TRAIN) {
      return `Approaching the threshold (p = ${p}): the model has almost enough features to hit every noisy dot, so weights and test error are already climbing.`;
    }
    return `Second descent (p = ${p} > n = ${N_TRAIN}): training error stays zero, yet choosing the smallest-norm interpolator lets excess weights relax (‖c‖ ≈ ${curvePoint.weightNorm.toFixed(1)}) and pulls median test MSE back down to ${curvePoint.testMse.toFixed(3)}.`;
  })();

  const fmtSci = (v: number) => (v < 1e-4 ? '< 1e-4' : v < 0.01 ? v.toFixed(4) : v.toFixed(3));
  const fmtNorm = (v: number) => (v >= 1000 ? Math.round(v).toLocaleString() : v.toFixed(1));

  return (
    <Widget
      title="Random-features regression: test error across the interpolation wall"
      caption="Slide the number of random features p past n = 24 training points. Left: one fitted curve (dashed = true function, solid = minimum-norm fit). Right: 41-trial median train and test MSE on a log axis. Switch label noise off to see what actually drives the spike at p = 24."
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', justifyContent: 'space-between' }}>
          <Segmented
            label="Label noise"
            value={noiseMode}
            onChange={setNoiseMode}
            options={[
              { value: 'noisy', label: `Noisy labels (σ = ${DEFAULT_NOISE})` },
              { value: 'clean', label: 'Noiseless (σ = 0)' },
            ]}
          />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {PRESETS.map((pr) => (
              <button
                key={pr.p}
                type="button"
                className={p === pr.p ? 'btn primary' : 'btn'}
                onClick={() => setPIdx(P_VALUES.indexOf(pr.p as (typeof P_VALUES)[number]))}
              >
                {pr.label}
              </button>
            ))}
          </div>
        </div>

        <Slider
          label={`Random features p (training points n = ${N_TRAIN})`}
          value={pIdx}
          min={0}
          max={P_VALUES.length - 1}
          step={1}
          onChange={setPIdx}
          format={(idx) => `p = ${P_VALUES[idx]}`}
        />

        <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
          <div style={{ flex: '1 1 300px', minWidth: 260 }}>
            <Plot
              x={[-1.02, 1.02]}
              y={[-2.2, 2.2]}
              width={380}
              height={240}
              pad={34}
              ariaLabel="Random-features fitted curve against 24 training points"
            >
              {({ x, y }) => {
                const clampY = (v: number) => Math.max(-2.2, Math.min(2.2, v));
                const truePath = trueCurvePts
                  .map(([px, py], i) => `${i === 0 ? 'M' : 'L'}${x(px).toFixed(1)},${y(clampY(py)).toFixed(1)}`)
                  .join(' ');
                const fitPath = fitCurvePts
                  .map(([px, py], i) => `${i === 0 ? 'M' : 'L'}${x(px).toFixed(1)},${y(clampY(py)).toFixed(1)}`)
                  .join(' ');
                return (
                  <g>
                    <path
                      d={truePath}
                      fill="none"
                      stroke="var(--text-muted)"
                      strokeWidth={1.8}
                      strokeDasharray="5 4"
                    />
                    <path
                      d={fitPath}
                      fill="none"
                      stroke="var(--c-b)"
                      strokeWidth={2.3}
                      strokeLinejoin="round"
                    />
                    {data.x.map((xi, i) => (
                      <circle
                        key={i}
                        cx={x(xi)}
                        cy={y(clampY(data.y[i]))}
                        r={3.2}
                        fill="var(--c-a)"
                        stroke="var(--bg)"
                        strokeWidth={1}
                      />
                    ))}
                  </g>
                );
              }}
            </Plot>
            <div style={{ display: 'flex', gap: 14, justifyContent: 'center', fontSize: 12, color: 'var(--text-muted)', marginTop: 4, flexWrap: 'wrap' }}>
              <span>
                <span style={{ display: 'inline-block', width: 14, borderTop: '2px dashed var(--text-muted)', verticalAlign: 'middle', marginRight: 5 }} />
                True curve
              </span>
              <span>
                <span style={{ display: 'inline-block', width: 14, height: 2.5, background: 'var(--c-b)', verticalAlign: 'middle', marginRight: 5 }} />
                Fit (p = {p})
              </span>
            </div>
          </div>

          <div style={{ flex: '1 1 300px', minWidth: 260 }}>
            <LogChart
              series={[
                { points: testSeries, color: 'var(--c-b)', dots: true, label: 'Test MSE' },
                { points: trainSeries, color: 'var(--c-a)', dashed: true, label: 'Train MSE' },
              ]}
              xDomain={[2, 165]}
              yDomain={[1e-3, 30]}
              marker={p}
              xLabel="random features p"
              width={380}
              height={240}
              xTicks={[2, 24, 60, 120, 160]}
              ariaLabel="Median train and test MSE across feature counts p from 2 to 160"
            />
            <div style={{ display: 'flex', gap: 14, justifyContent: 'center', fontSize: 12, color: 'var(--text-muted)', marginTop: 4, flexWrap: 'wrap' }}>
              <span>
                <span style={{ display: 'inline-block', width: 14, height: 2.5, background: 'var(--c-b)', verticalAlign: 'middle', marginRight: 5 }} />
                Median test MSE (41 trials)
              </span>
              <span>
                <span style={{ display: 'inline-block', width: 14, borderTop: '2px dashed var(--c-a)', verticalAlign: 'middle', marginRight: 5 }} />
                Median train MSE
              </span>
            </div>
          </div>
        </div>

        <div className="readout" style={{ display: 'flex', flexWrap: 'wrap', gap: '8px 20px' }}>
          <div><strong>Features p:</strong> {p} (n = {N_TRAIN})</div>
          <div><strong>Median train MSE:</strong> {fmtSci(curvePoint.trainMse)}</div>
          <div><strong>Median test MSE:</strong> {fmtSci(curvePoint.testMse)}</div>
          <div><strong>Median weight norm ‖c‖:</strong> {fmtNorm(curvePoint.weightNorm)}</div>
        </div>

        <div className="live-note">{regimeNote}</div>
      </div>
    </Widget>
  );
}
