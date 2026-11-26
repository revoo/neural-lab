// authored-by: Gemini Antigravity
import { useCallback, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import { Heatmap, diverging } from '../../components/heatmap';
import { useFramePlayer } from '../../components/motion';
import {
  MICROSCOPE_BOUNDS,
  MICROSCOPE_INIT,
  MICROSCOPE_KEYS,
  MICROSCOPE_XOR,
  applyMicroscopeStep,
  compareCoordinatedVsSingle,
  hiddenLineSegment,
  microscopePredict,
  microscopeStepInfo,
  type MicroscopeKey,
  type MicroscopeParams,
} from './tinyMlp';

const KEY_LABELS: Record<MicroscopeKey, string> = {
  w11: 'w₁₁ (x₁→h₁)',
  w12: 'w₁₂ (x₂→h₁)',
  b1: 'b₁ (h₁ bias)',
  w21: 'w₂₁ (x₁→h₂)',
  w22: 'w₂₂ (x₂→h₂)',
  b2: 'b₂ (h₂ bias)',
  v1: 'v₁ (h₁→out)',
  v2: 'v₂ (h₂→out)',
  c: 'c (out bias)',
};

const fmtS = (v: number, d = 3) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`;

export function StepMicroscope() {
  const [params, setParams] = useState<MicroscopeParams>({ ...MICROSCOPE_INIT });
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [lr, setLr] = useState(0.9);
  const [showGhost, setShowGhost] = useState(true);
  const [stepCount, setStepCount] = useState(0);

  // Ref for animated 25-step player so each frame advances cleanly from latest weights
  const paramsRef = useRef(params);
  paramsRef.current = params;
  const selRef = useRef(selectedIdx);
  selRef.current = selectedIdx;
  const lrRef = useRef(lr);
  lrRef.current = lr;

  const info = useMemo(() => microscopeStepInfo(params, selectedIdx), [params, selectedIdx]);
  const nextParams = useMemo(() => applyMicroscopeStep(params, info.grads, lr), [params, info.grads, lr]);
  const nextInfo = useMemo(() => microscopeStepInfo(nextParams, selectedIdx), [nextParams, selectedIdx]);
  const cmp = useMemo(() => compareCoordinatedVsSingle(params, selectedIdx, lr), [params, selectedIdx, lr]);

  const h1Line = useMemo(() => hiddenLineSegment(params.w11, params.w12, params.b1, MICROSCOPE_BOUNDS), [params]);
  const h2Line = useMemo(() => hiddenLineSegment(params.w21, params.w22, params.b2, MICROSCOPE_BOUNDS), [params]);
  const h1Ghost = useMemo(
    () => hiddenLineSegment(nextParams.w11, nextParams.w12, nextParams.b1, MICROSCOPE_BOUNDS),
    [nextParams],
  );
  const h2Ghost = useMemo(
    () => hiddenLineSegment(nextParams.w21, nextParams.w22, nextParams.b2, MICROSCOPE_BOUNDS),
    [nextParams],
  );

  const predictField = useCallback((x1: number, x2: number) => microscopePredict(params, x1, x2), [params]);

  const player = useFramePlayer(25, 95, (frame) => {
    if (player.reduced) {
      let cur = { ...paramsRef.current };
      for (let i = 0; i < 25; i++) {
        const g = microscopeStepInfo(cur, selRef.current).grads;
        cur = applyMicroscopeStep(cur, g, lrRef.current);
      }
      paramsRef.current = cur;
      setParams(cur);
      setStepCount((s) => s + 25);
      return;
    }
    const g = microscopeStepInfo(paramsRef.current, selRef.current).grads;
    const nxt = applyMicroscopeStep(paramsRef.current, g, lrRef.current);
    paramsRef.current = nxt;
    setParams(nxt);
    setStepCount((s) => s + 1);
    void frame;
  });

  const takeOne = () => {
    setParams(nextParams);
    setStepCount((s) => s + 1);
  };

  const reset = () => {
    const fresh = { ...MICROSCOPE_INIT };
    paramsRef.current = fresh;
    setParams(fresh);
    setStepCount(0);
  };

  // Live note
  let liveNote: React.ReactNode;
  if (selectedIdx !== null) {
    const pt = info.points[selectedIdx];
    const nxtPt = nextInfo.points[selectedIdx];
    const selfDelta = nxtPt.loss - pt.loss;
    const worseOthers = nextInfo.points
      .map((p, idx) => ({ id: p.id, d: p.loss - info.points[idx].loss }))
      .filter((o, idx) => idx !== selectedIdx && o.d > 0.01);
    liveNote = (
      <>
        <strong>Single-point microscope ({pt.id}, target {pt.y}):</strong> stepping on{' '}
        <span className="mono">{pt.id}</span> alone would change its loss by{' '}
        <strong className="mono" style={{ color: selfDelta <= 0 ? 'var(--c-good)' : 'var(--c-bad)' }}>
          {fmtS(selfDelta, 3)}
        </strong>
        {worseOthers.length > 0 ? (
          <>
            , while tugging{' '}
            {worseOthers.map((o, idx) => (
              <span key={o.id}>
                {idx > 0 ? ', ' : ''}
                <span className="mono">{o.id}</span> ({fmtS(o.d, 3)})
              </span>
            ))}{' '}
            the wrong way. That tug-of-war is why full-batch or mini-batch steps average the pulls.
          </>
        ) : (
          <> without hurting any neighbour noticeably.</>
        )}
      </>
    );
  } else {
    const ratio = cmp.singleDrop > 1e-9 ? cmp.fullDrop / cmp.singleDrop : 1;
    liveNote = (
      <>
        <strong>Full-batch step {stepCount}:</strong> mean loss{' '}
        <strong className="mono">{info.meanLoss.toFixed(4)}</strong>, accuracy{' '}
        <strong className="mono">{(info.accuracy * 100).toFixed(0)}%</strong>. The dashed ghost lines preview where{' '}
        <span className="mono">h₁=0</span> and <span className="mono">h₂=0</span> will move next: one coordinated
        9-weight step drops loss by <strong className="mono">{cmp.fullDrop.toFixed(4)}</strong> vs{' '}
        <strong className="mono">{cmp.singleDrop.toFixed(4)}</strong> for nudging only the steepest weight (
        <span className="mono">{cmp.steepestKey}</span>, {ratio.toFixed(1)}× less).
      </>
    );
  }

  return (
    <Widget
      title="One backprop step under a microscope (2 → 2 → 1 on XOR)"
      wide
      caption={
        <>
          A 9-parameter <span className="mono">2 → 2 → 1</span> tanh network on 4-point XOR. Solid lines show where{' '}
          <span className="mono">h₁ = 0</span> and <span className="mono">h₂ = 0</span> sit right now; dashed ghost lines
          preview where the next step moves them. <strong>Click any misclassified point</strong> (try{' '}
          <span className="mono">P1</span>) and press <strong>Take 1 step</strong> to see single-point tug-of-war, then
          switch back to <strong>Full batch</strong> and press <strong>Run 25 steps</strong>.
        </>
      }
    >
      {/* Mode + controls row */}
      <div className="controls inline" style={{ gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <div className="btn-row" role="group" aria-label="Target point selection">
          <button
            type="button"
            className={`btn small ${selectedIdx === null ? 'primary' : ''}`}
            onClick={() => setSelectedIdx(null)}
          >
            Full batch
          </button>
          {MICROSCOPE_XOR.map((pt, idx) => (
            <button
              key={pt.id}
              type="button"
              className={`btn small ${selectedIdx === idx ? 'primary' : ''}`}
              onClick={() => setSelectedIdx(selectedIdx === idx ? null : idx)}
              data-point={pt.id}
            >
              {pt.id} ({pt.x1 > 0 ? '+' : '−'}1,{pt.x2 > 0 ? '+' : '−'}1)
            </button>
          ))}
        </div>

        <Slider
          label="step size η"
          value={lr}
          min={0.2}
          max={1.5}
          step={0.1}
          onChange={setLr}
          format={(v) => v.toFixed(1)}
        />

        <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12.5, cursor: 'pointer' }}>
          <input type="checkbox" checked={showGhost} onChange={(e) => setShowGhost(e.target.checked)} />
          Ghost preview
        </label>

        <div className="btn-row" style={{ marginLeft: 'auto' }}>
          <button type="button" className="btn primary small" onClick={takeOne} disabled={player.running}>
            Take 1 step
          </button>
          <button type="button" className="btn small" onClick={player.start} disabled={player.running}>
            {player.running ? 'Stepping…' : 'Run 25 steps'}
          </button>
          <button type="button" className="btn small" onClick={reset} disabled={stepCount === 0 && !player.running}>
            Reset
          </button>
        </div>
      </div>

      {/* Main split */}
      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap', marginTop: 12 }}>
        {/* Left: 2D field + hidden lines + ghost lines + clickable points */}
        <div style={{ flex: '0 1 340px', maxWidth: '100%' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>
            <span className="mono">
              step {stepCount} · acc <strong style={{ color: 'var(--text)' }}>{(info.accuracy * 100).toFixed(0)}%</strong>
            </span>
            <span className="mono">
              loss <strong style={{ color: 'var(--text)' }}>{info.meanLoss.toFixed(4)}</strong>
            </span>
          </div>
          <Plot
            width={340}
            height={340}
            x={[-1.6, 1.6]}
            y={[-1.6, 1.6]}
            xLabel="x₁"
            yLabel="x₂"
            ariaLabel="2D decision field and hidden neuron crease lines on XOR"
            style={{ width: '100%', height: 'auto', display: 'block' }}
          >
            <Heatmap f={predictField} range={[0, 1]} colormap={diverging} resolution={88} opacity={0.44} />
            {({ x, y }) => (
              <g>
                {/* Ghost preview lines */}
                {showGhost && h1Ghost && (
                  <line
                    x1={x(h1Ghost.x1)}
                    y1={y(h1Ghost.y1)}
                    x2={x(h1Ghost.x2)}
                    y2={y(h1Ghost.y2)}
                    stroke="var(--c-a)"
                    strokeWidth={2}
                    strokeDasharray="5 4"
                    strokeOpacity={0.8}
                  />
                )}
                {showGhost && h2Ghost && (
                  <line
                    x1={x(h2Ghost.x1)}
                    y1={y(h2Ghost.y1)}
                    x2={x(h2Ghost.x2)}
                    y2={y(h2Ghost.y2)}
                    stroke="var(--c-c)"
                    strokeWidth={2}
                    strokeDasharray="5 4"
                    strokeOpacity={0.8}
                  />
                )}

                {/* Current hidden crease lines + positive normal tick */}
                {h1Line && (
                  <g>
                    <line
                      x1={x(h1Line.x1)}
                      y1={y(h1Line.y1)}
                      x2={x(h1Line.x2)}
                      y2={y(h1Line.y2)}
                      stroke="var(--c-a)"
                      strokeWidth={2.6}
                    />
                    <line
                      x1={x(h1Line.mx)}
                      y1={y(h1Line.my)}
                      x2={x(h1Line.mx + h1Line.nx * 0.22)}
                      y2={y(h1Line.my + h1Line.ny * 0.22)}
                      stroke="var(--c-a)"
                      strokeWidth={2}
                    />
                    <text
                      x={x(h1Line.mx + h1Line.nx * 0.34)}
                      y={y(h1Line.my + h1Line.ny * 0.34)}
                      textAnchor="middle"
                      fontSize={10.5}
                      fontWeight={700}
                      fill="var(--c-a)"
                      style={{ fontFamily: 'var(--font-mono)', paintOrder: 'stroke', stroke: 'var(--bg-elev)', strokeWidth: 4 }}
                    >
                      h₁=0
                    </text>
                  </g>
                )}

                {h2Line && (
                  <g>
                    <line
                      x1={x(h2Line.x1)}
                      y1={y(h2Line.y1)}
                      x2={x(h2Line.x2)}
                      y2={y(h2Line.y2)}
                      stroke="var(--c-c)"
                      strokeWidth={2.6}
                    />
                    <line
                      x1={x(h2Line.mx)}
                      y1={y(h2Line.my)}
                      x2={x(h2Line.mx + h2Line.nx * 0.22)}
                      y2={y(h2Line.my + h2Line.ny * 0.22)}
                      stroke="var(--c-c)"
                      strokeWidth={2}
                    />
                    <text
                      x={x(h2Line.mx + h2Line.nx * 0.34)}
                      y={y(h2Line.my + h2Line.ny * 0.34)}
                      textAnchor="middle"
                      fontSize={10.5}
                      fontWeight={700}
                      fill="var(--c-c)"
                      style={{ fontFamily: 'var(--font-mono)', paintOrder: 'stroke', stroke: 'var(--bg-elev)', strokeWidth: 4 }}
                    >
                      h₂=0
                    </text>
                  </g>
                )}

                {/* Clickable XOR data points */}
                {info.points.map((pt, idx) => {
                  const cx = x(pt.x1);
                  const cy = y(pt.x2);
                  const isSel = selectedIdx === idx;
                  const fill = pt.y === 1 ? 'var(--c-a)' : 'var(--c-b)';
                  return (
                    <g
                      key={pt.id}
                      style={{ cursor: 'pointer' }}
                      onClick={() => setSelectedIdx(selectedIdx === idx ? null : idx)}
                    >
                      {isSel && (
                        <circle cx={cx} cy={cy} r={14} fill="none" stroke="var(--c-grad)" strokeWidth={2.6} />
                      )}
                      {!pt.correct && (
                        <circle
                          cx={cx}
                          cy={cy}
                          r={11}
                          fill="none"
                          stroke="var(--c-bad)"
                          strokeWidth={1.6}
                          strokeDasharray="3 2"
                        />
                      )}
                      <circle cx={cx} cy={cy} r={7.5} fill={fill} stroke="var(--bg-elev)" strokeWidth={2} />
                      <text
                        x={cx}
                        y={cy - 13}
                        textAnchor="middle"
                        fontSize={10.5}
                        fontWeight={700}
                        fill="var(--text)"
                        style={{ fontFamily: 'var(--font-mono)', paintOrder: 'stroke', stroke: 'var(--bg-elev)', strokeWidth: 3.5 }}
                      >
                        {pt.id} · p={pt.p.toFixed(2)}
                      </text>
                    </g>
                  );
                })}
              </g>
            )}
          </Plot>
          <div style={{ display: 'flex', gap: 12, fontSize: 11.5, color: 'var(--text-muted)', marginTop: 6, flexWrap: 'wrap' }}>
            <span>
              <strong style={{ color: 'var(--c-a)' }}>—</strong> h₁=0 (dashed = next)
            </span>
            <span>
              <strong style={{ color: 'var(--c-c)' }}>—</strong> h₂=0 (dashed = next)
            </span>
            <span>
              <strong style={{ color: 'var(--c-bad)' }}>◌</strong> wrong class
            </span>
          </div>
        </div>

        {/* Right: per-point loss preview + 9-parameter gradient table */}
        <div style={{ flex: '1 1 300px', minWidth: 260, display: 'flex', flexDirection: 'column', gap: 10 }}>
          {/* Per-point loss before -> after */}
          <div
            style={{
              padding: '10px 12px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-sunken)',
              border: '1px solid var(--border)',
            }}
          >
            <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>
              Per-point loss preview (now → next step)
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              {info.points.map((pt, idx) => {
                const nxt = nextInfo.points[idx];
                const delta = nxt.loss - pt.loss;
                const isSel = selectedIdx === idx;
                return (
                  <div
                    key={pt.id}
                    onClick={() => setSelectedIdx(isSel ? null : idx)}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '64px 1fr 122px',
                      gap: 8,
                      alignItems: 'center',
                      fontSize: 12,
                      padding: '3px 6px',
                      borderRadius: 4,
                      cursor: 'pointer',
                      background: isSel ? 'color-mix(in srgb, var(--c-grad) 14%, transparent)' : 'transparent',
                    }}
                  >
                    <span className="mono" style={{ fontWeight: isSel ? 700 : 500, whiteSpace: 'nowrap' }}>
                      {pt.id} (y={pt.y})
                    </span>
                    <div style={{ height: 10, background: 'var(--bg-elev)', borderRadius: 3, overflow: 'hidden', position: 'relative' }}>
                      <div
                        style={{
                          width: `${Math.min(100, (pt.loss / 1.4) * 100)}%`,
                          height: '100%',
                          background: pt.correct ? 'var(--c-good)' : 'var(--c-bad)',
                          opacity: 0.75,
                        }}
                      />
                    </div>
                    <span className="mono" style={{ textAlign: 'right', whiteSpace: 'nowrap', fontSize: 11.5 }}>
                      {pt.loss.toFixed(3)}{' '}
                      <strong style={{ color: delta <= 0 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                        ({fmtS(delta, 3)})
                      </strong>
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 9-parameter gradient table */}
          <div
            style={{
              padding: '10px 12px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-sunken)',
              border: '1px solid var(--border)',
              overflowX: 'auto',
            }}
          >
            <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>
              All 9 weights: value, gradient &amp; next nudge (−η·grad)
            </div>
            <table className="num-table" style={{ width: '100%', fontSize: 12 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left' }}>param</th>
                  <th style={{ textAlign: 'right' }}>now</th>
                  <th style={{ textAlign: 'right', color: 'var(--c-grad)' }}>grad</th>
                  <th style={{ textAlign: 'right' }}>Δw</th>
                </tr>
              </thead>
              <tbody>
                {MICROSCOPE_KEYS.map((k) => {
                  const g = info.grads[k];
                  const dw = -lr * g;
                  const isSteep = k === cmp.steepestKey;
                  return (
                    <tr
                      key={k}
                      style={isSteep ? { background: 'color-mix(in srgb, var(--c-grad) 10%, transparent)' } : undefined}
                    >
                      <td className="mono" style={{ textAlign: 'left', whiteSpace: 'nowrap' }}>
                        {KEY_LABELS[k]}
                      </td>
                      <td className="mono" style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                        {fmtS(params[k], 3)}
                      </td>
                      <td className="mono" style={{ textAlign: 'right', color: 'var(--c-grad)', fontWeight: isSteep ? 700 : 500, whiteSpace: 'nowrap' }}>
                        {fmtS(g, 3)}
                      </td>
                      <td className="mono" style={{ textAlign: 'right', color: dw >= 0 ? 'var(--c-a)' : 'var(--c-b)', whiteSpace: 'nowrap' }}>
                        {fmtS(dw, 3)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 12 }}>
        {liveNote}
      </div>
    </Widget>
  );
}
