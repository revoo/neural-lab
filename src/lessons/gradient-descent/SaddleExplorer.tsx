// authored-by: Gemini Antigravity
import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot, usePlot } from '../../components/plot';
import { Heatmap } from '../../components/heatmap';
import { useReplay } from '../../components/motion';
import { ReplayControls } from '../../components/ReplayControls';
import {
  minProbability,
  runSaddleDescent,
  SADDLE_LR,
  SADDLE_PRESETS,
  SADDLE_STEPS,
  saddleLoss,
  saddleProbability,
  sampleCurvatures,
  type SaddlePresetId,
  type SaddleStep,
} from './saddle';

function formatOdds(d: number): string {
  const denom = Math.pow(2, d);
  if (denom < 1_000_000) return `1 in ${denom.toLocaleString('en-GB')}`;
  if (denom < 1_000_000_000) return `1 in ${(denom / 1_000_000).toFixed(2)}M`;
  return `1 in ${(denom / 1_000_000_000).toFixed(2)}B`;
}

function formatPct(p: number): string {
  const pct = p * 100;
  if (pct === 0) return '0%';
  if (pct >= 99.995) return '99.999%+';
  if (pct >= 10) return `${pct.toFixed(1)}%`;
  if (pct >= 0.01) return `${pct.toFixed(2)}%`;
  return `${pct.toExponential(1)}%`;
}

function SaddlePathLayer({
  path,
  stepIdx,
}: {
  path: readonly SaddleStep[];
  stepIdx: number;
}) {
  const { x, y } = usePlot();
  const visible = path.slice(0, stepIdx + 1);
  const pts = visible.map((p) => `${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`).join(' ');
  const cur = visible[visible.length - 1];
  const start = path[0];
  const sx = x(0);
  const sy = y(0);

  return (
    <g>
      {/* Ridge line (y = 0) and escape valley (x = 0) guides */}
      <line x1={x(-1.55)} y1={sy} x2={x(1.55)} y2={sy} stroke="var(--text-faint)" strokeWidth={1} strokeDasharray="4 4" opacity={0.6} />
      <line x1={sx} y1={y(-1.55)} x2={sx} y2={y(1.55)} stroke="var(--c-b)" strokeWidth={1.2} strokeDasharray="3 3" opacity={0.65} />

      {/* Saddle point marker */}
      <circle cx={sx} cy={sy} r={5} fill="var(--bg-elev)" stroke="var(--c-grad)" strokeWidth={2} />
      <text x={sx + 8} y={sy - 7} fontSize={10.5} fontFamily="var(--font-mono)" fill="var(--text)" fontWeight={700}>
        saddle (0, 0)
      </text>

      {/* Escape direction labels */}
      <text x={sx + 6} y={y(1.38)} fontSize={10} fontFamily="var(--font-mono)" fill="var(--bg-elev)" fontWeight={700}>
        ↓ escape (−y² cliff)
      </text>

      {/* Trajectory polyline */}
      {visible.length > 1 && (
        <polyline points={pts} fill="none" stroke="var(--c-grad)" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" />
      )}
      {visible.map((p, idx) =>
        idx % 3 === 0 || idx === stepIdx ? (
          <circle
            key={p.step}
            cx={x(p.x)}
            cy={y(p.y)}
            r={idx === stepIdx ? 5.5 : 2.3}
            fill={idx === stepIdx ? 'var(--c-grad)' : 'var(--bg-elev)'}
            stroke={idx === stepIdx ? 'var(--bg-elev)' : 'var(--c-grad)'}
            strokeWidth={1.5}
          />
        ) : null,
      )}
      <circle cx={x(start.x)} cy={y(start.y)} r={4} fill="var(--c-c)" stroke="var(--bg-elev)" strokeWidth={1.4} />
      <text x={x(cur.x) + 8} y={y(cur.y) + 4} fontSize={10.5} fontFamily="var(--font-mono)" fill="var(--bg-elev)" fontWeight={700}>
        t={cur.step}
      </text>
    </g>
  );
}

function LossMiniChart({ path, stepIdx }: { path: readonly SaddleStep[]; stepIdx: number }) {
  const W = 280;
  const H = 88;
  const padL = 34;
  const padR = 10;
  const padT = 10;
  const padB = 20;
  const maxStep = path.length - 1;
  const minL = -0.8;
  const maxL = 0.85;
  const sx = (s: number) => padL + (s / maxStep) * (W - padL - padR);
  const sy = (l: number) => {
    const c = Math.max(minL, Math.min(maxL, l));
    return padT + (1 - (c - minL) / (maxL - minL)) * (H - padT - padB);
  };
  const allPts = path.map((p) => `${sx(p.step).toFixed(1)},${sy(p.loss).toFixed(1)}`).join(' ');
  const curPts = path.slice(0, stepIdx + 1).map((p) => `${sx(p.step).toFixed(1)},${sy(p.loss).toFixed(1)}`).join(' ');
  const cur = path[stepIdx];

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: '100%', height: 'auto', display: 'block', border: '1px solid var(--border)', borderRadius: 8, background: 'var(--bg-sunken)' }}
      role="img"
      aria-label="Loss versus gradient descent step showing plateau near the saddle before escape"
    >
      {/* Zero line */}
      <line x1={padL} y1={sy(0)} x2={W - padR} y2={sy(0)} stroke="var(--border-strong)" strokeDasharray="3 3" />
      <text x={padL - 4} y={sy(0) + 3} textAnchor="end" fontSize={9.5} fontFamily="var(--font-mono)" fill="var(--text-muted)">0.0</text>
      <text x={padL - 4} y={sy(0.72) + 3} textAnchor="end" fontSize={9.5} fontFamily="var(--font-mono)" fill="var(--text-muted)">0.7</text>
      <text x={padL - 4} y={sy(-0.7) + 3} textAnchor="end" fontSize={9.5} fontFamily="var(--font-mono)" fill="var(--text-muted)">−0.7</text>

      {/* Plateau zone highlight (steps 12..25) */}
      <rect x={sx(12)} y={padT} width={sx(25) - sx(12)} height={H - padT - padB} fill="var(--accent-soft)" opacity={0.45} />
      <text x={(sx(12) + sx(25)) / 2} y={padT + 10} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill="var(--accent-text)">
        plateau
      </text>

      {/* Full faint curve + active curve */}
      <polyline points={allPts} fill="none" stroke="var(--text-faint)" strokeWidth={1.3} opacity={0.45} />
      <polyline points={curPts} fill="none" stroke="var(--c-grad)" strokeWidth={2.2} />
      <circle cx={sx(cur.step)} cy={sy(cur.loss)} r={4} fill="var(--c-grad)" stroke="var(--bg-elev)" strokeWidth={1.3} />

      <text x={padL} y={H - 5} fontSize={9.5} fontFamily="var(--font-mono)" fill="var(--text-muted)">step 0</text>
      <text x={W - padR} y={H - 5} textAnchor="end" fontSize={9.5} fontFamily="var(--font-mono)" fill="var(--text-muted)">step {maxStep}</text>
    </svg>
  );
}

export function SaddleExplorer() {
  const [preset, setPreset] = useState<SaddlePresetId>('near');
  const [customStart, setCustomStart] = useState<[number, number] | null>(null);
  const [dims, setDims] = useState(10);
  const [sampleSeed, setSampleSeed] = useState(1);

  const [x0, y0] = customStart ?? [SADDLE_PRESETS[preset].x0, SADDLE_PRESETS[preset].y0];
  const path = useMemo(() => runSaddleDescent(x0, y0, SADDLE_LR, SADDLE_STEPS), [x0, y0]);
  const pathKey = `${x0.toFixed(3)},${y0.toFixed(3)}`;
  const replay = useReplay(SADDLE_STEPS, 65, pathKey, false);

  const cur = path[replay.i];
  const pMin = minProbability(dims);
  const pSaddle = saddleProbability(dims);
  const sample = useMemo(() => sampleCurvatures(dims, sampleSeed * 97 + dims * 13), [dims, sampleSeed]);

  const selectPreset = (p: SaddlePresetId) => {
    setPreset(p);
    setCustomStart(null);
  };

  return (
    <Widget
      title="Saddle points: why high-dimensional flat spots are almost never traps"
      caption="Left: click anywhere on the 2D saddle f(x, y) = ½(x² − 0.6y²) or pick a preset, then press Play. Right: raise the dimension d to see how quickly true local minima vanish and downhill escape ramps take over."
    >
      <div className="controls">
        <Segmented
          label="Start point"
          value={preset}
          onChange={selectPreset}
          options={[
            { value: 'near', label: 'Near ridge (1.2, 0.02)' },
            { value: 'dead', label: 'Dead on ridge (1.2, 0.0)' },
            { value: 'off', label: 'Off ridge (1.2, 0.30)' },
          ]}
        />
      </div>

      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        {/* Left: 2D saddle heatmap + replay + loss curve */}
        <div style={{ flex: '1 1 280px', minWidth: 260, maxWidth: 370, display: 'grid', gap: 8 }}>
          <Plot
            x={[-1.55, 1.55]}
            y={[-1.55, 1.55]}
            width={340}
            height={265}
            xLabel="x (curves up +)"
            yLabel="y (curves down −)"
            ariaLabel="2D saddle contour map with gradient descent path"
            style={{ width: '100%', height: 'auto', cursor: 'crosshair' }}
            onPointerDown={(p) => {
              const cx = Math.max(-1.45, Math.min(1.45, p.x));
              const cy = Math.max(-1.45, Math.min(1.45, p.y));
              setCustomStart([cx, cy]);
            }}
          >
            <Heatmap f={saddleLoss} range={[-0.75, 0.9]} contours={14} opacity={0.82} />
            <SaddlePathLayer path={path} stepIdx={replay.i} />
          </Plot>

          <ReplayControls r={replay} label="step" />
          <LossMiniChart path={path} stepIdx={replay.i} />

          <div className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 6 }}>
            <span>pos: ({cur.x.toFixed(2)}, {cur.y.toFixed(2)})</span>
            <span>loss: {cur.loss >= 0 ? '+' : '−'}{Math.abs(cur.loss).toFixed(3)}</span>
            <span>|∇f|: {cur.gradNorm.toFixed(3)}</span>
          </div>
        </div>

        {/* Right: dimension d slider + sampled curvature bars */}
        <div style={{ flex: '1 1 240px', minWidth: 230, display: 'grid', gap: 10 }}>
          <div style={{ padding: '12px 14px', border: '1px solid var(--border)', borderRadius: 10, background: 'var(--bg-sunken)' }}>
            <Slider
              label="Dimensions d (weights)"
              value={dims}
              min={1}
              max={30}
              step={1}
              onChange={setDims}
              format={(v) => `${v}`}
            />

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 10 }}>
              <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
                <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>TRUE LOCAL MIN</div>
                <div className="mono" style={{ fontSize: 17, fontWeight: 700, color: 'var(--c-a)' }}>
                  {formatPct(pMin)}
                </div>
                <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>{formatOdds(dims)}</div>
              </div>
              <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
                <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>SADDLE POINT</div>
                <div className="mono" style={{ fontSize: 17, fontWeight: 700, color: 'var(--c-b)' }}>
                  {formatPct(pSaddle)}
                </div>
                <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  {dims === 1 ? 'needs d ≥ 2' : 'has downhill exit'}
                </div>
              </div>
            </div>
          </div>

          <div style={{ padding: '12px 14px', border: '1px solid var(--border)', borderRadius: 10, background: 'var(--bg-sunken)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, gap: 8 }}>
              <span className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                SAMPLED CRITICAL POINT ({dims} AXES)
              </span>
              <button
                type="button"
                className="btn small"
                onClick={() => setSampleSeed((s) => s + 1)}
              >
                Resample
              </button>
            </div>

            {/* Signed curvature bars */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${dims}, minmax(0, 1fr))`,
                gap: dims > 18 ? 2 : 4,
                height: 76,
                alignItems: 'center',
                padding: '4px 0',
              }}
              aria-label={`${dims} sampled curvature directions: ${sample.upCount} curving up and ${sample.downCount} curving down`}
            >
              {sample.curvatures.map((c, idx) => {
                const hPct = Math.round((Math.abs(c) / 1.25) * 44);
                const up = c > 0;
                return (
                  <div
                    key={idx}
                    style={{
                      position: 'relative',
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'center',
                    }}
                  >
                    <div
                      style={{
                        position: 'absolute',
                        left: 0,
                        right: 0,
                        top: '50%',
                        height: 1,
                        background: 'var(--border-strong)',
                      }}
                    />
                    <div
                      style={{
                        position: 'absolute',
                        left: 0,
                        right: 0,
                        height: `${hPct}%`,
                        ...(up ? { bottom: '50%' } : { top: '50%' }),
                        borderRadius: 2,
                        background: up ? 'var(--c-a)' : 'var(--c-b)',
                      }}
                    />
                  </div>
                );
              })}
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginTop: 6, flexWrap: 'wrap', gap: 6 }}>
              <span style={{ color: 'var(--c-a)', fontWeight: 600 }}>↑ {sample.upCount} bowl (+ curvature)</span>
              <span style={{ color: 'var(--c-b)', fontWeight: 600 }}>↓ {sample.downCount} cliff (− curvature)</span>
            </div>
            <div className="mono" style={{ fontSize: 12, marginTop: 6, fontWeight: 700, color: sample.kind === 'saddle' ? 'var(--c-c)' : 'var(--c-grad)' }}>
              {sample.kind === 'local-min' && 'Verdict: TRUE LOCAL MINIMUM (every direction curves up!)'}
              {sample.kind === 'local-max' && 'Verdict: local maximum (every direction curves down)'}
              {sample.kind === 'saddle' && `Verdict: SADDLE — ${sample.downCount} downhill escape ${sample.downCount === 1 ? 'direction' : 'directions'}!`}
            </div>
          </div>
        </div>
      </div>

      <p className="live-note" style={{ marginTop: 12 }}>
        {Math.abs(y0) < 1e-9
          ? `Dead on the ridge (y₀ = 0): the y-slope is identically zero at every step, so gradient descent marches to (0, 0) and freezes forever — even though an escape cliff sits right beside it. In practice, mini-batch SGD noise knocks the walker off y = 0 immediately.`
          : `Starting at (${x0.toFixed(2)}, ${y0.toFixed(2)}): steps 0→12 plunge onto the saddle, steps 12→25 crawl across a deceptive flat plateau near loss 0.00, then the downhill −y² direction takes over and flings loss down to ${path[path.length - 1].loss.toFixed(2)}. Meanwhile in d = ${dims} dimensions, only ${formatPct(pMin)} (${formatOdds(dims)}) of flat points are local minima — ${formatPct(pSaddle)} are saddles.`}
      </p>
    </Widget>
  );
}
