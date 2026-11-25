// authored-by: Gemini Antigravity
import { useEffect, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Plot } from '../../components/plot';
import { Segmented } from '../../components/controls';
import { usePrefersReducedMotion } from '../../components/motion';
import {
  GAN_PRESETS,
  GanRunner,
  HIST_BINS,
  MODE_CENTRES,
  X_MAX,
  X_MIN,
  type GanPresetId,
  type GanSnapshot,
} from './gan';

/** Shared 1D histogram + D(x) probability curve SVG plot. */
export function GanDensityPlot({
  snap,
  width = 440,
  height = 260,
}: {
  snap: GanSnapshot;
  width?: number;
  height?: number;
}) {
  const maxBar = Math.max(0.2, ...snap.realHist, ...snap.fakeHist);
  const binW = (X_MAX - X_MIN) / HIST_BINS;

  return (
    <Plot
      width={width}
      height={height}
      x={[X_MIN, X_MAX]}
      y={[0, 1.05]}
      grid={false}
      axes={false}
      pad={34}
      ariaLabel="1D GAN real versus generated histograms and discriminator probability curve D(x)"
      style={{ width: '100%', height: 'auto', display: 'block' }}
    >
      {({ x, y }) => {
        const dPath = snap.dCurve
          .map(([xv, pv], i) => `${i === 0 ? 'M' : 'L'}${x(xv).toFixed(1)},${y(pv).toFixed(1)}`)
          .join(' ');
        return (
          <g>
            {/* Horizontal reference lines for D(x) = 0, 0.5, 1 */}
            {[0, 0.5, 1].map((v) => (
              <g key={v}>
                <line
                  x1={x(X_MIN)}
                  x2={x(X_MAX)}
                  y1={y(v)}
                  y2={y(v)}
                  stroke={v === 0.5 ? 'var(--border-strong)' : 'var(--c-grid)'}
                  strokeDasharray={v === 0.5 ? '4 4' : undefined}
                  strokeWidth={1}
                />
                <text className="plot-tick" x={x(X_MIN) - 5} y={y(v) + 3.5} textAnchor="end">
                  {v.toFixed(1)}
                </text>
              </g>
            ))}

            {/* X-axis ticks at mode centres */}
            {[-2.4, 0, 2.4].map((c) => (
              <text key={c} className="plot-tick" x={x(c)} y={y(0) + 15} textAnchor="middle">
                {c === 0 ? '0' : c > 0 ? '+2.4' : '−2.4'}
              </text>
            ))}

            {/* Real distribution histogram (background fill) */}
            {snap.realHist.map((d, i) => {
              const x0 = X_MIN + i * binW;
              const hNorm = (d / maxBar) * 0.9;
              if (hNorm <= 0.001) return null;
              return (
                <rect
                  key={`r-${i}`}
                  x={x(x0)}
                  y={y(hNorm)}
                  width={Math.max(1, x(x0 + binW) - x(x0) - 0.8)}
                  height={Math.max(0, y(0) - y(hNorm))}
                  fill="var(--c-a)"
                  opacity={0.32}
                />
              );
            })}

            {/* Generated (fake) distribution histogram (foreground outline + tint) */}
            {snap.fakeHist.map((d, i) => {
              const x0 = X_MIN + i * binW;
              const hNorm = (d / maxBar) * 0.9;
              if (hNorm <= 0.001) return null;
              return (
                <rect
                  key={`f-${i}`}
                  x={x(x0) + 1}
                  y={y(hNorm)}
                  width={Math.max(1, x(x0 + binW) - x(x0) - 2.2)}
                  height={Math.max(0, y(0) - y(hNorm))}
                  fill="var(--c-b)"
                  opacity={0.62}
                  stroke="var(--c-b)"
                  strokeWidth={0.8}
                />
              );
            })}

            {/* Mode centre markers along axis */}
            {MODE_CENTRES.map((c, idx) => (
              <circle
                key={`m-${idx}`}
                cx={x(c)}
                cy={y(0)}
                r={3.2}
                fill={snap.metrics.modeShares[idx] >= 0.12 ? 'var(--c-c)' : 'var(--c-bad)'}
              />
            ))}

            {/* Discriminator probability curve D(x) in [0, 1] */}
            <path d={dPath} fill="none" stroke="var(--c-grad)" strokeWidth={2.4} strokeLinejoin="round" />

            {/* Compact legend inside top-left */}
            <g transform={`translate(${x(X_MIN) + 6}, 14)`} fontSize={11} fontFamily="var(--font-ui)">
              <rect x={0} y={-8} width={10} height={8} fill="var(--c-a)" opacity={0.45} />
              <text x={14} y={0} fill="var(--text-muted)">Real</text>
              <rect x={54} y={-8} width={10} height={8} fill="var(--c-b)" opacity={0.75} />
              <text x={68} y={0} fill="var(--text-muted)">Forged G(z)</text>
              <line x1={156} y1={-4} x2={170} y2={-4} stroke="var(--c-grad)" strokeWidth={2.4} />
              <text x={174} y={0} fill="var(--text-muted)">Detective D(x)</text>
            </g>
          </g>
        );
      }}
    </Plot>
  );
}

export function GanDuelWidget() {
  const reduced = usePrefersReducedMotion();
  const [preset, setPreset] = useState<GanPresetId>('balanced');
  const runnerRef = useRef<GanRunner>(new GanRunner(GAN_PRESETS.balanced.config));
  const [snap, setSnap] = useState<GanSnapshot>(() => runnerRef.current.snapshot());
  const [playing, setPlaying] = useState(false);

  const loadPreset = (id: GanPresetId) => {
    setPlaying(false);
    setPreset(id);
    runnerRef.current = new GanRunner(GAN_PRESETS[id].config);
    setSnap(runnerRef.current.snapshot());
  };

  const advance = (steps: number) => {
    runnerRef.current.step(steps);
    setSnap(runnerRef.current.snapshot());
  };

  useEffect(() => {
    if (!playing) return;
    if (reduced) {
      const rem = Math.max(0, 240 - runnerRef.current.stepCount);
      if (rem > 0) advance(rem);
      setPlaying(false);
      return;
    }
    const id = window.setInterval(() => {
      if (runnerRef.current.stepCount >= 240) {
        setPlaying(false);
        return;
      }
      advance(8);
    }, 90);
    return () => window.clearInterval(id);
  }, [playing, reduced]);

  const m = snap.metrics;
  const pInfo = GAN_PRESETS[preset];

  return (
    <Widget
      title="Live 1D GAN: a forger and a detective on three peaks"
      wide
      caption="Press Train to watch the generator (foreground bars) learn to match the three real peaks (tinted background) while the detective curve D(x) rises on real regions and falls on fakes. Try all three presets: in Balanced duel, D(x) flattens near 0.5 as all three peaks fill in; in Detective too strong, D(fake) ≈ 0 kills the saturating gradient; in Mode collapse, all fakes pile onto one peak."
    >
      <div style={{ marginBottom: 12 }}>
        <Segmented
          label="Preset"
          value={preset}
          onChange={loadPreset}
          options={[
            { value: 'balanced', label: 'Balanced duel' },
            { value: 'vanishing', label: 'Detective too strong' },
            { value: 'collapse', label: 'Mode collapse' },
          ]}
        />
      </div>

      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 360px', maxWidth: 480 }}>
          <GanDensityPlot snap={snap} width={450} height={260} />
          <div className="btn-row" style={{ marginTop: 10 }}>
            <button
              type="button"
              className="btn primary"
              onClick={() => {
                if (m.step >= 240) loadPreset(preset);
                setPlaying((p) => !p);
              }}
            >
              {playing ? 'Pause' : m.step >= 240 ? 'Restart & Train' : 'Train live'}
            </button>
            <button type="button" className="btn" onClick={() => advance(20)} disabled={playing}>
              +20 steps
            </button>
            <button type="button" className="btn" onClick={() => advance(Math.max(1, 240 - m.step))} disabled={playing || m.step >= 240}>
              Jump to step 240
            </button>
            <button type="button" className="btn" onClick={() => loadPreset(preset)}>
              Reset
            </button>
          </div>
        </div>

        <div style={{ flex: '1 1 240px', display: 'grid', gap: 10 }}>
          <div className="readout">
            <div>
              <strong>Step:</strong> <span className="mono">{m.step} / 240</span> ({m.totalUpdates} updates)
            </div>
            <div>
              <strong>Modes covered:</strong>{' '}
              <span className="mono">{m.modesCovered} / 3</span>{' '}
              <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                ({m.modeShares.map((s) => `${Math.round(s * 100)}%`).join(' · ')})
              </span>
            </div>
            <div>
              <strong>Distribution error (W₁):</strong> <span className="mono">{m.w1.toFixed(3)}</span>
            </div>
            <div>
              <strong>Detective D(real) / D(fake):</strong>{' '}
              <span className="mono">{m.meanDReal.toFixed(2)} / {m.meanDFake.toFixed(3)}</span>
            </div>
            <div>
              <strong>Forger gradient ‖∇G‖:</strong> <span className="mono">{m.gGradNorm.toFixed(4)}</span>
            </div>
          </div>

          <div className="live-note">
            <strong>{pInfo.label}.</strong> {pInfo.summary}
          </div>
        </div>
      </div>
    </Widget>
  );
}
