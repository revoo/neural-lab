// authored-by: Gemini Antigravity
import { useMemo, useState } from 'react';
import { LabShell, useLabProgress, type LabBrief } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { LiveCode } from '../../components/LabJob';
import { Clip, Handle, Plot } from '../../components/plot';
import { Segmented, Toggle } from '../../components/controls';
import { ClassMarker, StreetOverlay } from './Widgets';
import { buildSvmCode } from './realCode';
import {
  BRIEF4_DATA,
  C_GRID,
  FEATURE_BIN,
  LAB_BRIEFS,
  RING_POINTS,
  SEPARABLE_POINTS,
  evalFeatureSubset,
  evaluateLine2D,
  predictScore,
  searchCGrid,
  searchFeatureSubsets,
  solveHardMargin2D,
  svmLabHint,
  type FeatureId,
} from './svm';

const INIT_A: [number, number] = [-2.5, -1.5];
const INIT_B: [number, number] = [2.4, 1.4];

export function WidenTheStreetLab() {
  const progress = useLabProgress('svm-widen-lab-v1');
  const log = useMachineLog();
  const [active, setActive] = useState(0);

  // Briefs 1 & 2 state: draggable line endpoints
  const [handleA, setHandleA] = useState<[number, number]>(INIT_A);
  const [handleB, setHandleB] = useState<[number, number]>(INIT_B);

  // Brief 3 state: chosen feature subset
  const [selectedFeatures, setSelectedFeatures] = useState<FeatureId[]>(['x1', 'x2']);

  // Brief 4 state: chosen C index in C_GRID
  const [cIndex, setCIndex] = useState(8); // start at C=50 (overfit) so learner sees the trap
  const [showValPoints, setShowValPoints] = useState(true);

  const exactHard = useMemo(() => solveHardMargin2D(SEPARABLE_POINTS), []);
  const maxStreet = exactHard.separable ? exactHard.streetWidth : 1;
  const lineEval = useMemo(() => evaluateLine2D(SEPARABLE_POINTS, handleA, handleB), [handleA, handleB]);
  const marginPct = lineEval.separates ? (100 * lineEval.streetWidth) / maxStreet : 0;

  const ringEval = useMemo(() => evalFeatureSubset(RING_POINTS, selectedFeatures), [selectedFeatures]);
  const ringSearch = useMemo(() => searchFeatureSubsets(RING_POINTS), []);

  const cSearch = useMemo(() => searchCGrid(BRIEF4_DATA.train, BRIEF4_DATA.val), []);
  const activeC = cSearch.rows[cIndex];

  const briefId = LAB_BRIEFS[active].id;

  const briefs: LabBrief[] = [
    {
      id: 'separate',
      tab: '1 · Separate',
      title: 'Brief 1 — Cut the two classes cleanly',
      text: 'Drag the two line handles so that every circle (class +1) sits on one side of your line and every diamond (class −1) sits on the other.',
    },
    {
      id: 'widen',
      tab: '2 · Widen',
      title: 'Brief 2 — Widen the street to ≥ 95% of maximum',
      text: (
        <>
          Not every separating line is equal. Shift and tilt the line until your street width reaches at least{' '}
          <strong>95% of the maximum possible margin</strong>. <em>Optional goal (★):</em> reach ≥ 99%.
        </>
      ),
    },
    {
      id: 'lift',
      tab: '3 · Lift rings',
      title: 'Brief 3 — Separate the concentric rings',
      text: (
        <>
          A straight cut in raw <code>(x₁, x₂)</code> always fails on concentric rings. Pick feature(s) from the parts bin so a linear SVM separates all 24 points cleanly.{' '}
          <em>Optional goal (★):</em> do it with the fewest features possible.
        </>
      ),
    },
    {
      id: 'soft',
      tab: '4 · Soft C',
      title: 'Brief 4 — Choose C on noisy training data',
      text: 'Two training outliers pull a strict SVM off course. Test values of C from the 9-point grid to maximise validation accuracy on the 60 held-out points.',
    },
  ];

  // Pass & optimal flags
  let passing = false;
  let optimal = false;
  if (briefId === 'separate') {
    passing = lineEval.separates;
    optimal = lineEval.separates && marginPct >= 95;
  } else if (briefId === 'widen') {
    passing = lineEval.separates && marginPct >= 95;
    optimal = lineEval.separates && marginPct >= 99;
  } else if (briefId === 'lift') {
    passing = ringEval.separates;
    optimal = ringEval.separates && selectedFeatures.length === ringSearch.minCount;
  } else {
    passing = Math.abs(activeC.valAcc - cSearch.bestValAcc) < 1e-9;
    optimal = passing;
  }

  const toggleFeature = (id: FeatureId) => {
    setSelectedFeatures((prev) => {
      const next = prev.includes(id) ? prev.filter((f) => f !== id) : [...prev, id];
      const res = evalFeatureSubset(RING_POINTS, next);
      log.push(
        `Brief 3 features=[${next.join(', ') || 'none'}] → ${(res.acc * 100).toFixed(1)}% (${res.separates ? 'separable' : 'not separable'})`,
        res.separates ? 'good' : undefined,
      );
      return next;
    });
  };

  const selectC = (idx: number) => {
    setCIndex(idx);
    const r = cSearch.rows[idx];
    log.push(
      `Brief 4 C=${r.C} → train ${(r.trainAcc * 100).toFixed(1)}%, val ${(r.valAcc * 100).toFixed(1)}%, street ${r.streetWidth.toFixed(2)}`,
      Math.abs(r.valAcc - cSearch.bestValAcc) < 1e-9 ? 'good' : undefined,
    );
  };

  const statusText = (() => {
    if (briefId === 'separate') {
      return lineEval.separates
        ? `Separated all 10/10 points! Current street width: ${lineEval.streetWidth.toFixed(3)} (${marginPct.toFixed(1)}% of max).`
        : `${lineEval.correct}/10 points on the right side. Keep dragging the two handles until all 10 separate cleanly.`;
    }
    if (briefId === 'widen') {
      if (!lineEval.separates) return `Not separating yet (${lineEval.correct}/10). Separate all points first to open a street.`;
      return `Street width ${lineEval.streetWidth.toFixed(3)} / ${maxStreet.toFixed(3)} (${marginPct.toFixed(1)}% of max). ${
        optimal ? '★ Optimal! Right on the exact max-margin street.' : passing ? '✓ Passed (≥ 95%). Try nudging to ≥ 99% for ★.' : 'Need ≥ 95%.'
      }`;
    }
    if (briefId === 'lift') {
      const n = selectedFeatures.length;
      if (n === 0) return 'Select at least one feature from the parts bin.';
      return `${Math.round(ringEval.acc * 24)}/24 points separated using ${n} feature${n === 1 ? '' : 's'}. ${
        optimal
          ? '★ Minimal-feature optimum!'
          : passing
            ? '✓ Separated! Can you do it with fewer features?'
            : 'Not yet separable — try another feature from the bin.'
      }`;
    }
    return `C = ${activeC.C}: train ${(activeC.trainAcc * 100).toFixed(1)}%, val ${(activeC.valAcc * 100).toFixed(1)}% (${Math.round(activeC.valAcc * 60)}/60), street ${activeC.streetWidth.toFixed(2)}. ${
      passing ? '✓ Best validation accuracy on the 9-point grid!' : 'Try other C values and compare train vs validation accuracy.'
    }`;
  })();

  const hintText = svmLabHint(briefId, {
    separates: lineEval.separates,
    marginPct,
    featureCount: selectedFeatures.length,
    ringAcc: ringEval.acc,
    valAcc: activeC.valAcc,
    trainAcc: activeC.trainAcc,
    streetWidth: activeC.streetWidth,
  });

  // Precompute 2D grid shading for Brief 3 induced boundary
  const ringGrid = useMemo(() => {
    if (briefId !== 'lift' || !ringEval.res || selectedFeatures.length === 0) return [];
    const defs = selectedFeatures.map((id) => FEATURE_BIN.find((f) => f.id === id)!);
    const cells: { x: number; y: number; pos: boolean }[] = [];
    const step = 0.22;
    for (let gx = -2.5; gx <= 2.5; gx += step) {
      for (let gy = -2.5; gy <= 2.5; gy += step) {
        const feat = defs.map((d) => d.fn({ id: '', x: gx, y: gy, label: 1 }));
        const s = predictScore(ringEval.res, feat);
        cells.push({ x: gx, y: gy, pos: s >= 0 });
      }
    }
    return cells;
  }, [briefId, ringEval.res, selectedFeatures]);

  const revealContent = (() => {
    if (briefId === 'separate' || briefId === 'widen') {
      return (
        <div>
          <strong>Exact exhaustive hard-margin solution</strong> (checked over all pairs and triples of points):{' '}
          <span className="mono">w = [0.50, 1.00], b = 0.40</span>, street width{' '}
          <span className="mono">2/‖w‖ = {maxStreet.toFixed(3)}</span>, pinned by support vectors{' '}
          <span className="mono">P1 (−1.0, 1.1), P2 (1.0, 0.1), N1 (−0.6, −1.1)</span>. Any line with slope −0.5 passing near{' '}
          <span className="mono">(0, −0.4)</span> achieves the widest street.
        </div>
      );
    }
    if (briefId === 'lift') {
      return (
        <div>
          <strong>Exhaustive search over all {ringSearch.total} non-empty feature subsets:</strong>{' '}
          {ringSearch.passing.length} separate the rings cleanly, and the fewest needed is{' '}
          <strong>{ringSearch.minCount} feature</strong> — achieved by two different single features:{' '}
          <span className="mono">x₁² + x₂²</span> (squared Euclidean radius) and{' '}
          <span className="mono">|x₁| + |x₂|</span> (Manhattan distance from the origin). Pairs such as{' '}
          <span className="mono">x₁², x₂²</span> also separate cleanly with 2 features.
        </div>
      );
    }
    return (
      <div>
        <strong>Exhaustive search over the 9-point C grid:</strong> validation accuracy peaks at{' '}
        <strong>{(cSearch.bestValAcc * 100).toFixed(1)}%</strong> for{' '}
        <span className="mono">C ∈ &#123;{cSearch.bestCs.join(', ')}&#125;</span>.
        <table className="num-table" style={{ marginTop: 8 }}>
          <thead>
            <tr>
              <th>C</th>
              <th>Train</th>
              <th>Val (60)</th>
              <th>Street</th>
              <th>SVs</th>
            </tr>
          </thead>
          <tbody>
            {cSearch.rows.map((r) => (
              <tr
                key={r.C}
                style={
                  Math.abs(r.valAcc - cSearch.bestValAcc) < 1e-9
                    ? { fontWeight: 700, background: 'var(--accent-soft)' }
                    : undefined
                }
              >
                <td className="mono">{r.C}</td>
                <td className="mono">{(r.trainAcc * 100).toFixed(1)}%</td>
                <td className="mono">{(r.valAcc * 100).toFixed(1)}%</td>
                <td className="mono">{r.streetWidth.toFixed(2)}</td>
                <td className="mono">{r.supportCount}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  })();

  const resetBrief = () => {
    if (briefId === 'separate' || briefId === 'widen') {
      setHandleA(INIT_A);
      setHandleB(INIT_B);
      log.push(`Reset handles for ${briefId}`);
    } else if (briefId === 'lift') {
      setSelectedFeatures(['x1', 'x2']);
      log.push('Reset Brief 3 features to [x1, x2]');
    } else {
      setCIndex(8);
      log.push('Reset Brief 4 to C = 50');
    }
  };

  return (
    <LabShell
      title="Lab: Widen the Street"
      caption="Solve all four briefs: place a separating line, widen its street to ≥ 95% of the exact maximum, lift concentric rings with engineered features, and tune soft-margin C on noisy data."
      briefs={briefs}
      active={active}
      onSelect={setActive}
      progress={progress}
      passing={passing}
      optimal={optimal}
      status={statusText}
      hint={hintText}
      reveal={{
        label: briefId === 'soft' ? 'the exhaustive 9-point C search' : briefId === 'lift' ? 'the exhaustive 127-subset search' : 'the exact max-margin street',
        content: revealContent,
      }}
      log={{ lines: log.lines, onClear: log.clear, empty: 'Adjust controls or handles to log measurements.' }}
      actions={
        <button type="button" className="btn" onClick={resetBrief}>
          Reset brief
        </button>
      }
    >
      {(briefId === 'separate' || briefId === 'widen') && (
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap' }}>
          <div style={{ flex: '1 1 320px', maxWidth: 400 }}>
            <Plot
              width={380}
              height={380}
              x={[-3, 3]}
              y={[-3, 3]}
              xLabel="x₁"
              yLabel="x₂"
              ariaLabel="Draggable line for Widen the Street lab"
              style={{ width: '100%', height: 'auto' }}
            >
              {({ x, y }) => {
                // Draw perpendicular segment from nearest point to drawn line
                const drawPerp = (ptIdx: number, col: string) => {
                  if (ptIdx < 0 || !lineEval.separates) return null;
                  const p = SEPARABLE_POINTS[ptIdx];
                  const signed = lineEval.normal[0] * p.x + lineEval.normal[1] * p.y + lineEval.offset;
                  const fx = p.x - signed * lineEval.normal[0];
                  const fy = p.y - signed * lineEval.normal[1];
                  return (
                    <line
                      x1={x(p.x)}
                      y1={y(p.y)}
                      x2={x(fx)}
                      y2={y(fy)}
                      stroke={col}
                      strokeWidth={2}
                      strokeDasharray="3 2"
                    />
                  );
                };
                return (
                  <g>
                    {/* User's street */}
                    {lineEval.separates ? (
                      <StreetOverlay
                        wx={lineEval.normal[0] / lineEval.minSignedDist}
                        wy={lineEval.normal[1] / lineEval.minSignedDist}
                        b={lineEval.offset / lineEval.minSignedDist}
                        color={passing ? 'var(--c-c)' : 'var(--accent)'}
                      />
                    ) : (
                      <Clip>
                        <line
                          x1={x(handleA[0] - (handleB[0] - handleA[0]) * 5)}
                          y1={y(handleA[1] - (handleB[1] - handleA[1]) * 5)}
                          x2={x(handleB[0] + (handleB[0] - handleA[0]) * 5)}
                          y2={y(handleB[1] + (handleB[1] - handleA[1]) * 5)}
                          stroke="var(--c-bad)"
                          strokeWidth={2.2}
                        />
                      </Clip>
                    )}

                    {/* Closest perpendicular distance indicators in Brief 2 */}
                    {briefId === 'widen' && drawPerp(lineEval.closestPosIdx, 'var(--c-a)')}
                    {briefId === 'widen' && drawPerp(lineEval.closestNegIdx, 'var(--c-b)')}

                    {SEPARABLE_POINTS.map((pt) => (
                      <g key={pt.id}>
                        <ClassMarker pt={pt} r={5.5} />
                        <text
                          x={x(pt.x) + 7}
                          y={y(pt.y) - 5}
                          fontSize={10}
                          fill="var(--text-muted)"
                          fontFamily="var(--font-mono)"
                          pointerEvents="none"
                        >
                          {pt.id}
                        </text>
                      </g>
                    ))}

                    <Handle
                      x={handleA[0]}
                      y={handleA[1]}
                      color="var(--accent)"
                      label="Line handle A"
                      onDrag={(nx, ny) => setHandleA([+nx.toFixed(2), +ny.toFixed(2)])}
                    />
                    <Handle
                      x={handleB[0]}
                      y={handleB[1]}
                      color="var(--accent)"
                      label="Line handle B"
                      onDrag={(nx, ny) => setHandleB([+nx.toFixed(2), +ny.toFixed(2)])}
                    />
                  </g>
                );
              }}
            </Plot>
          </div>

          <div style={{ flex: '1 1 240px', display: 'grid', gap: 10 }}>
            <div className="readout">
              <div>
                <strong>Accuracy:</strong>{' '}
                <span className="mono">{lineEval.correct} / 10</span>
              </div>
              <div>
                <strong>Your street width:</strong>{' '}
                <span className="mono">{lineEval.streetWidth.toFixed(3)}</span>
              </div>
              {briefId === 'widen' && (
                <div>
                  <strong>Share of maximum:</strong>{' '}
                  <span className="mono">{marginPct.toFixed(1)}%</span>
                </div>
              )}
            </div>
            <div className="btn-row">
              <button
                type="button"
                className="btn"
                onClick={() =>
                  log.push(
                    `Logged line: ${lineEval.correct}/10, street ${lineEval.streetWidth.toFixed(3)} (${marginPct.toFixed(1)}% of max)`,
                    passing ? 'good' : undefined,
                  )
                }
              >
                Log current line
              </button>
            </div>
            <LiveCode note="Equivalent scikit-learn" code={buildSvmCode(briefId, {})} />
          </div>
        </div>
      )}

      {briefId === 'lift' && (
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap' }}>
          <div style={{ flex: '1 1 300px', maxWidth: 360 }}>
            <Plot
              width={340}
              height={340}
              x={[-2.6, 2.6]}
              y={[-2.6, 2.6]}
              xLabel="x₁"
              yLabel="x₂"
              ariaLabel="Concentric rings with induced linear SVM decision region"
              style={{ width: '100%', height: 'auto' }}
            >
              {({ x, y }) => (
                <g>
                  <Clip>
                    {ringGrid.map((c, idx) => (
                      <rect
                        key={idx}
                        x={x(c.x - 0.11)}
                        y={y(c.y + 0.11)}
                        width={Math.abs(x(0.22) - x(0)) + 0.6}
                        height={Math.abs(y(0) - y(0.22)) + 0.6}
                        fill={c.pos ? 'var(--c-a)' : 'var(--c-b)'}
                        opacity={0.14}
                      />
                    ))}
                  </Clip>
                  {RING_POINTS.map((pt) => (
                    <ClassMarker key={pt.id} pt={pt} r={5.5} />
                  ))}
                </g>
              )}
            </Plot>
          </div>

          <div style={{ flex: '1 1 250px', display: 'grid', gap: 12 }}>
            <div>
              <div className="field-label">Parts bin — click features to toggle</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {FEATURE_BIN.map((f) => {
                  const on = selectedFeatures.includes(f.id);
                  return (
                    <button
                      key={f.id}
                      type="button"
                      className={`btn${on ? ' primary' : ''}`}
                      onClick={() => toggleFeature(f.id)}
                      aria-pressed={on}
                      style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}
                    >
                      {on ? '✓ ' : '+ '}
                      {f.label}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="readout">
              <div>
                <strong>Active features ({selectedFeatures.length}):</strong>{' '}
                <span className="mono">{selectedFeatures.length ? selectedFeatures.map((id) => FEATURE_BIN.find((f) => f.id === id)!.label).join(', ') : 'none'}</span>
              </div>
              <div>
                <strong>Linear SVM accuracy:</strong>{' '}
                <span className="mono">{(ringEval.acc * 100).toFixed(1)}% ({Math.round(ringEval.acc * 24)}/24)</span>
              </div>
            </div>

            <LiveCode note="Equivalent scikit-learn" code={buildSvmCode('lift', { features: selectedFeatures })} />
          </div>
        </div>
      )}

      {briefId === 'soft' && (
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap' }}>
          <div style={{ flex: '1 1 320px', maxWidth: 400 }}>
            <Plot
              width={380}
              height={340}
              x={[-2.7, 2.7]}
              y={[-2.7, 2.7]}
              xLabel="x₁"
              yLabel="x₂"
              ariaLabel="Soft-margin SVM boundary on noisy training and validation sets"
              style={{ width: '100%', height: 'auto' }}
            >
              <StreetOverlay wx={activeC.res.w[0]} wy={activeC.res.w[1]} b={activeC.res.b} color="var(--accent)" />
              {showValPoints && BRIEF4_DATA.val.map((pt) => <ClassMarker key={pt.id} pt={pt} r={3.2} opacity={0.35} />)}
              {BRIEF4_DATA.train.map((pt, i) => (
                <ClassMarker key={pt.id} pt={pt} isSV={activeC.res.supportIndices.includes(i)} r={5.2} />
              ))}
            </Plot>
          </div>

          <div style={{ flex: '1 1 250px', display: 'grid', gap: 12 }}>
            <Segmented
              label="Penalty C (9-point grid)"
              value={cIndex}
              onChange={selectC}
              options={C_GRID.map((c, i) => ({ value: i, label: String(c) }))}
            />
            <Toggle label="Show 60 held-out validation points (faint)" checked={showValPoints} onChange={setShowValPoints} />

            <div className="readout">
              <div>
                <strong>Training accuracy (32 pts):</strong>{' '}
                <span className="mono">{(activeC.trainAcc * 100).toFixed(1)}%</span>
              </div>
              <div>
                <strong>Validation accuracy (60 pts):</strong>{' '}
                <span className="mono">{(activeC.valAcc * 100).toFixed(1)}%</span>
              </div>
              <div>
                <strong>Street width (2/‖w‖):</strong>{' '}
                <span className="mono">{activeC.streetWidth.toFixed(3)}</span>
              </div>
              <div>
                <strong>Support vectors:</strong>{' '}
                <span className="mono">{activeC.supportCount} / 32</span>
              </div>
            </div>

            <LiveCode note="Equivalent scikit-learn" code={buildSvmCode('soft', { C: activeC.C })} />
          </div>
        </div>
      )}
    </LabShell>
  );
}
