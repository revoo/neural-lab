// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { LiveCode } from '../../components/LabJob';
import { useMachineLog } from '../../components/MachineLog';
import { useTweenedArray } from '../../components/motion';
import {
  DET_LAB_BRIEFS,
  LAB_CONFS,
  LAB_NMS_IOUS,
  STREET_SCENE,
  computeCocoMap,
  detLabExhaustive,
  detLabGrid,
  detLabHint,
  evaluateDetections,
  iouBreakdown,
  runHardNms,
  runSoftNms,
  squareShiftIou,
  traceDetLab,
  type Box,
  type DetTraceRow,
  type Detection,
  type GroundTruth,
  type NmsMode,
} from './detect';
import { detectorRealCode } from './realCode';

const pct1 = (v: number) => `${(100 * v).toFixed(1)}%`;

// ------------------------------------------------------------------ 1. Interactive IoU Explorer

export function IouExplorerWidget() {
  const [size, setSize] = useState<'large' | 'small'>('large');
  const [dx, setDx] = useState<number>(10);
  const [dy, setDy] = useState<number>(0);

  const side = size === 'large' ? 100 : 20;
  const gt: Box = { x1: 60, y1: 50, x2: 60 + side, y2: 50 + side };
  const pred: Box = { x1: 60 + dx, y1: 50 + dy, x2: 60 + side + dx, y2: 50 + side + dy };
  const b = useMemo(() => iouBreakdown(gt, pred), [gt.x1, gt.y1, gt.x2, gt.y2, pred.x1, pred.y1, pred.x2, pred.y2]);

  const large10 = squareShiftIou(100, 10);
  const small10 = squareShiftIou(20, 10);

  return (
    <Widget
      title="Intersection over Union (IoU) — and why small objects are brutally unforgiving"
      wide
      caption={
        <>
          Slide the predicted box (orange) relative to ground truth (blue). Keep the shift at <code>dx = 10 px</code> and toggle between a <strong>Large 100×100 car</strong> (IoU <strong>{large10.toFixed(3)}</strong> — passes even strict AP75!) and a <strong>Small 20×20 distant pedestrian</strong> (IoU <strong>{small10.toFixed(3)}</strong> — fails AP50!).
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end', marginBottom: 12 }}>
        <Segmented
          label="Object size"
          value={size}
          onChange={setSize}
          options={[
            { value: 'large', label: 'Large (100×100 px)' },
            { value: 'small', label: 'Small (20×20 px)' },
          ]}
        />
        <div style={{ flex: '1 1 160px' }}>
          <Slider label="Horizontal error dx" value={dx} min={-40} max={60} step={2} onChange={setDx} format={(v) => `${v} px`} />
        </div>
        <div style={{ flex: '1 1 160px' }}>
          <Slider label="Vertical error dy" value={dy} min={-40} max={40} step={2} onChange={setDy} format={(v) => `${v} px`} />
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center' }}>
        <svg viewBox="0 0 260 190" style={{ width: '100%', maxWidth: 320, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
          {/* Ground truth */}
          <rect x={gt.x1} y={gt.y1} width={gt.x2 - gt.x1} height={gt.y2 - gt.y1} fill="var(--c-a)" fillOpacity={0.22} stroke="var(--c-a)" strokeWidth={2} />
          {/* Predicted */}
          <rect x={pred.x1} y={pred.y1} width={pred.x2 - pred.x1} height={pred.y2 - pred.y1} fill="var(--c-b)" fillOpacity={0.22} stroke="var(--c-b)" strokeWidth={2} strokeDasharray="5 3" />
          {/* Intersection */}
          {b.inter && (
            <rect
              x={b.inter.x1}
              y={b.inter.y1}
              width={b.inter.x2 - b.inter.x1}
              height={b.inter.y2 - b.inter.y1}
              fill="var(--c-good)"
              fillOpacity={0.45}
              stroke="var(--c-good)"
              strokeWidth={1.5}
            />
          )}
          <text x={10} y={18} fill="var(--c-a)" fontSize={11} fontWeight={600}>■ Ground truth</text>
          <text x={100} y={18} fill="var(--c-b)" fontSize={11} fontWeight={600}>□ Prediction</text>
          <text x={182} y={18} fill="var(--c-good)" fontSize={11} fontWeight={600}>■ Intersection</text>
        </svg>

        <div className="readouts" style={{ flex: '1 1 240px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(115px, 1fr))', gap: 8, margin: 0 }}>
          <div className="readout">
            <div className="readout-label">Intersection</div>
            <div className="readout-value">{b.interArea.toLocaleString()} px²</div>
          </div>
          <div className="readout">
            <div className="readout-label">Union</div>
            <div className="readout-value">{b.unionArea.toLocaleString()} px²</div>
          </div>
          <div className="readout">
            <div className="readout-label">IoU</div>
            <div className="readout-value" style={{ color: b.iou >= 0.75 ? 'var(--c-good)' : b.iou >= 0.5 ? 'var(--c-a)' : 'var(--c-bad)' }}>
              {b.iou.toFixed(3)}
            </div>
          </div>
          <div className="readout">
            <div className="readout-label">COCO verdict</div>
            <div className="readout-value" style={{ fontSize: 13.5, color: b.iou >= 0.5 ? 'var(--c-good)' : 'var(--c-bad)' }}>
              {b.iou >= 0.75 ? 'TP @ AP75 ✓' : b.iou >= 0.5 ? 'TP @ AP50 ✓' : 'Miss (FP) ✕'}
            </div>
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Scene SVG helper & NMS Explorer

function SceneSvg({
  gts,
  allDets,
  keptIds,
  showGt,
}: {
  gts: readonly GroundTruth[];
  allDets: readonly Detection[];
  keptIds: Set<string>;
  showGt: boolean;
}) {
  return (
    <svg viewBox="0 0 300 195" style={{ width: '100%', maxWidth: 420, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
      {/* Horizon line */}
      <line x1={0} y1={145} x2={300} y2={145} stroke="var(--border)" strokeWidth={1} strokeDasharray="4 4" />
      {showGt &&
        gts.map((g) => (
          <g key={g.id}>
            <rect
              x={g.x1}
              y={g.y1}
              width={g.x2 - g.x1}
              height={g.y2 - g.y1}
              fill="var(--c-a)"
              fillOpacity={0.14}
              stroke="var(--c-a)"
              strokeWidth={1.6}
              strokeDasharray="4 2"
            />
            <text x={g.x1 + 3} y={g.y2 - 4} fill="var(--c-a)" fontSize={9.5} fontWeight={600}>
              GT {g.id}
            </text>
          </g>
        ))}
      {/* Suppressed detections (faint) */}
      {allDets
        .filter((d) => !keptIds.has(d.id))
        .map((d) => (
          <rect
            key={d.id}
            x={d.x1}
            y={d.y1}
            width={d.x2 - d.x1}
            height={d.y2 - d.y1}
            fill="none"
            stroke="var(--text-muted)"
            strokeOpacity={0.3}
            strokeWidth={1}
            strokeDasharray="2 2"
          />
        ))}
      {/* Kept detections */}
      {allDets
        .filter((d) => keptIds.has(d.id))
        .map((d) => (
          <g key={d.id}>
            <rect
              x={d.x1}
              y={d.y1}
              width={d.x2 - d.x1}
              height={d.y2 - d.y1}
              fill="none"
              stroke="var(--c-good)"
              strokeWidth={2.2}
            />
            <text x={d.x1 + 2} y={Math.max(11, d.y1 - 3)} fill="var(--c-good)" fontSize={9.5} fontWeight={700}>
              {d.cls} {d.score.toFixed(2)}
            </text>
          </g>
        ))}
    </svg>
  );
}

export function NmsExplorerWidget() {
  const [nmsIou, setNmsIou] = useState<number>(0.5);
  const [conf, setConf] = useState<number>(0.4);
  const [mode, setMode] = useState<NmsMode>('hard');

  const kept = useMemo(
    () => (mode === 'hard' ? runHardNms(STREET_SCENE.dets, nmsIou, conf).kept : runSoftNms(STREET_SCENE.dets, 0.45, conf).kept),
    [nmsIou, conf, mode],
  );
  const keptIds = useMemo(() => new Set(kept.map((d) => d.id)), [kept]);
  const ev = useMemo(() => evaluateDetections(kept, STREET_SCENE.gts, 0.5), [kept]);

  return (
    <Widget
      title="Non-Maximum Suppression (NMS): deleting duplicates without deleting neighbours"
      wide
      caption={
        <>
          Nine raw anchor boxes predict 3 real street objects (dashed blue = ground truth; solid green = kept; faint grey = suppressed). Drag <strong>NMS IoU threshold</strong> down to <code>0.25</code>: Pedestrian 2 (overlapping P1 at IoU 0.34) gets wiped out (<code>FN = 1</code>)! Drag it up to <code>0.85</code>: duplicate boxes flood back (<code>FP = 4</code>).
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end', marginBottom: 12 }}>
        <Segmented
          label="Suppression mode"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'hard', label: 'Hard NMS' },
            { value: 'soft', label: 'Soft-NMS (Gaussian)' },
          ]}
        />
        <div style={{ flex: '1 1 160px' }}>
          <Slider label="Confidence cut-off" value={conf} min={0.15} max={0.85} step={0.05} onChange={setConf} format={(v) => v.toFixed(2)} />
        </div>
        {mode === 'hard' && (
          <div style={{ flex: '1 1 160px' }}>
            <Slider label="NMS IoU threshold" value={nmsIou} min={0.15} max={0.90} step={0.05} onChange={setNmsIou} format={(v) => v.toFixed(2)} />
          </div>
        )}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center' }}>
        <SceneSvg gts={STREET_SCENE.gts} allDets={STREET_SCENE.dets} keptIds={keptIds} showGt />
        <div className="readouts" style={{ flex: '1 1 220px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(105px, 1fr))', gap: 8, margin: 0 }}>
          <div className="readout">
            <div className="readout-label">TP / FP / FN</div>
            <div className="readout-value" style={{ fontSize: 15 }}>
              <span style={{ color: 'var(--c-good)' }}>{ev.tp}</span> / <span style={{ color: ev.fp > 0 ? 'var(--c-bad)' : 'var(--text)' }}>{ev.fp}</span> / <span style={{ color: ev.fn > 0 ? 'var(--c-bad)' : 'var(--text)' }}>{ev.fn}</span>
            </div>
          </div>
          <div className="readout">
            <div className="readout-label">F1 score</div>
            <div className="readout-value" style={{ color: ev.f1 >= 0.99 ? 'var(--c-good)' : 'var(--c-b)' }}>{ev.f1.toFixed(3)}</div>
          </div>
          <div className="readout">
            <div className="readout-label">AP50</div>
            <div className="readout-value">{ev.ap.toFixed(3)}</div>
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Precision–Recall Curve & AP50 vs AP75 vs mAP

export function ApCurveWidget() {
  const [filtered, setFiltered] = useState<'raw' | 'nms'>('raw');
  const [evalIou, setEvalIou] = useState<number>(0.5);

  const dets = useMemo(
    () => (filtered === 'raw' ? STREET_SCENE.dets : runHardNms(STREET_SCENE.dets, 0.5, 0.4).kept),
    [filtered],
  );
  const ev = useMemo(() => evaluateDetections(dets, STREET_SCENE.gts, evalIou), [dets, evalIou]);
  const cocoMap = useMemo(() => computeCocoMap(dets, STREET_SCENE.gts), [dets]);

  // Build PR path
  const prPoints = useMemo(() => {
    const pts: [number, number][] = [[0, ev.rows[0]?.precision ?? 0]];
    for (const r of ev.rows) pts.push([r.recall, r.precision]);
    return pts;
  }, [ev]);

  return (
    <Widget
      title="Average Precision (AP) & COCO mAP@[.50:.95]"
      wide
      caption={
        <>
          Detections are ranked by confidence and matched 1-to-1 against ground truth (a duplicate box on an already-matched car counts as a <strong>False Positive</strong>!). Raise the evaluation IoU threshold from <code>0.50</code> to <code>0.90</code> to see why COCO’s 10-threshold <strong>mAP@[.50:.95]</strong> rewards crisp box alignment, not just rough hits.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end', marginBottom: 12 }}>
        <Segmented
          label="Detections"
          value={filtered}
          onChange={setFiltered}
          options={[
            { value: 'raw', label: 'All 9 raw boxes (no NMS)' },
            { value: 'nms', label: 'After NMS (3 clean boxes)' },
          ]}
        />
        <Segmented
          label="Match IoU threshold"
          value={evalIou}
          onChange={setEvalIou}
          options={[
            { value: 0.5, label: 'AP50 (0.50)' },
            { value: 0.75, label: 'AP75 (0.75)' },
            { value: 0.9, label: 'AP90 (0.90)' },
          ]}
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 14, alignItems: 'start' }}>
        <Plot x={[0, 1.05]} y={[0, 1.08]} height={220} xLabel="Recall" yLabel="Precision">
          {({ x: sx, y: sy }) => {
            const path = prPoints.map(([r, p], i) => `${i === 0 ? 'M' : 'L'} ${sx(r).toFixed(1)} ${sy(p).toFixed(1)}`).join(' ');
            return (
              <>
                <path d={path} fill="none" stroke="var(--c-a)" strokeWidth={2.4} />
                {ev.rows.map((r) => (
                  <circle
                    key={r.det.id}
                    cx={sx(r.recall)}
                    cy={sy(r.precision)}
                    r={4.2}
                    fill={r.isTp ? 'var(--c-good)' : 'var(--c-bad)'}
                  />
                ))}
              </>
            );
          }}
        </Plot>

        <div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(115px, 1fr))', gap: 8, marginBottom: 10 }}>
            <div className="readout">
              <div className="readout-label">AP @ {evalIou.toFixed(2)}</div>
              <div className="readout-value" style={{ color: 'var(--c-a)' }}>{ev.ap.toFixed(3)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">COCO mAP[.5:.95]</div>
              <div className="readout-value" style={{ color: 'var(--accent)' }}>{cocoMap.toFixed(3)}</div>
            </div>
          </div>
          <div style={{ maxHeight: 165, overflowY: 'auto', border: '1px solid var(--border)', borderRadius: 6 }}>
            <table className="num-table" style={{ width: '100%', fontSize: 12 }}>
              <thead>
                <tr>
                  <th>Box</th>
                  <th>Conf</th>
                  <th>IoU</th>
                  <th>Match</th>
                </tr>
              </thead>
              <tbody>
                {ev.rows.map((r) => (
                  <tr key={r.det.id}>
                    <td>{r.det.id} ({r.det.cls})</td>
                    <td>{r.det.score.toFixed(2)}</td>
                    <td>{r.bestIou.toFixed(2)}</td>
                    <td style={{ color: r.isTp ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 600 }}>
                      {r.isTp ? `TP (${r.matchedGtId})` : 'FP'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Tune the Object Detector

type DetLabState = {
  mode: NmsMode;
  conf: number;
  nmsIou: number;
};

const DET_LAB_KEY = 'nl-det-lab-v1';
const DET_LAB_PROG_KEY = 'nl-det-lab-progress-v2';
const DET_LAB_DEFAULT: DetLabState = {
  mode: 'hard',
  conf: 0.15,
  nmsIou: 0.85,
};

function loadLegacyDetSolved(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(DET_LAB_KEY);
    if (!raw) return {};
    const p = JSON.parse(raw);
    return p?.solved ?? {};
  } catch {
    return {};
  }
}

function loadDetLab(): DetLabState {
  try {
    const raw = localStorage.getItem(DET_LAB_KEY);
    if (!raw) return DET_LAB_DEFAULT;
    const p = JSON.parse(raw);
    return {
      mode: p.mode === 'hard' || p.mode === 'soft' ? p.mode : DET_LAB_DEFAULT.mode,
      conf: (LAB_CONFS as readonly number[]).includes(p.conf) ? p.conf : DET_LAB_DEFAULT.conf,
      nmsIou: (LAB_NMS_IOUS as readonly number[]).includes(p.nmsIou) ? p.nmsIou : DET_LAB_DEFAULT.nmsIou,
    };
  } catch {
    return DET_LAB_DEFAULT;
  }
}

function LabSceneSvg({
  gts,
  trace,
  missedGtIds,
  focusId,
  onFocus,
}: {
  gts: readonly GroundTruth[];
  trace: readonly DetTraceRow[];
  missedGtIds: Set<string>;
  focusId: string | null;
  onFocus: (id: string | null) => void;
}) {
  const focusedRow = focusId ? trace.find((r) => r.det.id === focusId) : null;
  const actorId = focusedRow?.suppressedBy ?? null;

  return (
    <svg viewBox="0 0 300 198" style={{ width: '100%', maxWidth: 420, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', display: 'block' }}>
      <line x1={0} y1={145} x2={300} y2={145} stroke="var(--border)" strokeWidth={1} strokeDasharray="4 4" />
      {/* Ground truth boxes (missed GTs highlighted in bad colour) */}
      {gts.map((g) => {
        const missed = missedGtIds.has(g.id);
        return (
          <g key={g.id}>
            <rect
              x={g.x1}
              y={g.y1}
              width={g.x2 - g.x1}
              height={g.y2 - g.y1}
              fill={missed ? 'var(--c-bad)' : 'var(--c-a)'}
              fillOpacity={missed ? 0.18 : 0.12}
              stroke={missed ? 'var(--c-bad)' : 'var(--c-a)'}
              strokeWidth={missed ? 2.2 : 1.5}
              strokeDasharray="4 2"
            />
            <text x={g.x1 + 3} y={g.y2 - 4} fill={missed ? 'var(--c-bad)' : 'var(--c-a)'} fontSize={9} fontWeight={700}>
              {missed ? `MISSED ${g.id}` : `GT ${g.id}`}
            </text>
          </g>
        );
      })}

      {/* Dropped / suppressed boxes */}
      {trace
        .filter((r) => r.fate === 'suppressed' || r.fate === 'below-conf')
        .map((r) => {
          const isFocus = r.det.id === focusId;
          return (
            <rect
              key={r.det.id}
              x={r.det.x1}
              y={r.det.y1}
              width={r.det.x2 - r.det.x1}
              height={r.det.y2 - r.det.y1}
              fill={isFocus ? 'var(--accent)' : 'none'}
              fillOpacity={isFocus ? 0.14 : 0}
              stroke={isFocus ? 'var(--accent)' : 'var(--text-muted)'}
              strokeOpacity={isFocus ? 0.95 : 0.32}
              strokeWidth={isFocus ? 2.2 : 1}
              strokeDasharray="2 2"
              style={{ cursor: 'pointer' }}
              onClick={() => onFocus(isFocus ? null : r.det.id)}
            />
          );
        })}

      {/* Kept boxes: TP in good colour, FP in bad colour */}
      {trace
        .filter((r) => r.fate === 'kept-tp' || r.fate === 'kept-fp')
        .map((r) => {
          const isTp = r.fate === 'kept-tp';
          const isFocus = r.det.id === focusId;
          const isActor = r.det.id === actorId;
          const stroke = isActor ? 'var(--accent)' : isTp ? 'var(--c-good)' : 'var(--c-bad)';
          return (
            <g key={r.det.id} style={{ cursor: 'pointer' }} onClick={() => onFocus(isFocus ? null : r.det.id)}>
              <rect
                x={r.det.x1}
                y={r.det.y1}
                width={r.det.x2 - r.det.x1}
                height={r.det.y2 - r.det.y1}
                fill={isFocus ? stroke : 'none'}
                fillOpacity={isFocus ? 0.15 : 0}
                stroke={stroke}
                strokeWidth={isFocus || isActor ? 2.8 : 2}
              />
              <text x={r.det.x1 + 2} y={Math.max(11, r.det.y1 - 3)} fill={stroke} fontSize={9} fontWeight={700}>
                {r.det.id} {r.finalScore.toFixed(2)} {isTp ? 'TP' : 'FP'}
              </text>
            </g>
          );
        })}
    </svg>
  );
}

export function DetectorTuningLab() {
  const [activeIdx, setActiveIdx] = useState<number>(0);
  const briefId = DET_LAB_BRIEFS[activeIdx].id;
  const [st, setSt] = useState<DetLabState>(loadDetLab);
  const [focusId, setFocusId] = useState<string | null>(null);
  const progress = useLabProgress(DET_LAB_PROG_KEY, () => ({ solved: loadLegacyDetSolved() }));
  const log = useMachineLog(18);

  useEffect(() => {
    try {
      localStorage.setItem(DET_LAB_KEY, JSON.stringify({ ...st, solved: progress.solved }));
    } catch {
      /* ignore */
    }
  }, [st, progress.solved]);

  const tr = useMemo(() => traceDetLab(briefId, st.conf, st.nmsIou, st.mode), [briefId, st.conf, st.nmsIou, st.mode]);
  const exhaust = useMemo(() => detLabExhaustive(briefId), [briefId]);
  const grid = useMemo(() => detLabGrid(briefId), [briefId]);
  const hint = useMemo(() => detLabHint(briefId, st.conf, st.nmsIou, st.mode), [briefId, st.conf, st.nmsIou, st.mode]);
  const realCode = useMemo(() => detectorRealCode(st.mode, st.conf, st.nmsIou), [st.mode, st.conf, st.nmsIou]);
  const tweenedMetrics = useTweenedArray([tr.recall, tr.precision, tr.f1], 240);

  const curOptimal = tr.pass && tr.f1 >= 0.999;

  const updateState = (patch: Partial<DetLabState>) => {
    const next = { ...st, ...patch };
    const ev = traceDetLab(briefId, next.conf, next.nmsIou, next.mode);
    setSt(next);
    const modeStr = next.mode === 'hard' ? `Hard NMS IoU=${next.nmsIou.toFixed(2)}` : 'Soft-NMS';
    log.push(
      `B${activeIdx + 1}: ${modeStr}, conf=${next.conf.toFixed(2)} → TP ${ev.tp}, FP ${ev.fp}, FN ${ev.fn}, F1 ${ev.f1.toFixed(2)} (${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : ev.fn > 0 ? 'bad' : undefined,
    );
  };

  const resetBrief = () => {
    setSt(DET_LAB_DEFAULT);
    setFocusId(null);
    log.push(`B${activeIdx + 1}: reset to Hard NMS, conf=0.15, IoU=0.85`);
  };

  const statusText = useMemo(() => {
    const metrics = `TP ${tr.tp} · FP ${tr.fp} · FN ${tr.fn} · Recall ${pct1(tr.recall)} · Precision ${pct1(tr.precision)} · F1 ${tr.f1.toFixed(2)}`;
    if (tr.pass) {
      return curOptimal
        ? `${metrics} — perfect F1 = 1.00!`
        : `${metrics} — brief passed. Optional goal: can you trim the remaining false positive for F1 = 1.00?`;
    }
    if (tr.fn > 0) {
      return `${metrics} — missed ground truth: ${[...tr.missedGtIds].join(', ')}.`;
    }
    return `${metrics}.`;
  }, [tr, curOptimal]);

  // Columns for 42-config post-solve grid: 5 Hard IoU columns + 1 Soft-NMS column
  const gridCols: { key: string; label: string; mode: NmsMode; nmsIou: number }[] = [
    ...LAB_NMS_IOUS.map((t) => ({ key: `hard-${t}`, label: `H ${t.toFixed(2)}`, mode: 'hard' as const, nmsIou: t })),
    { key: 'soft', label: 'Soft', mode: 'soft' as const, nmsIou: 0.5 },
  ];

  return (
    <LabShell
      title="Lab: Tune the Object Detector"
      caption={
        <>
          Tune Hard NMS vs Soft-NMS and confidence cut-offs across three traffic scenes — inspect every raw box&apos;s suppression or Gaussian decay trace live.
        </>
      }
      briefs={DET_LAB_BRIEFS.map((b, i) => ({
        id: b.id,
        tab: `Brief ${i + 1}`,
        title: b.title,
        text: b.story,
      }))}
      active={activeIdx}
      onSelect={(i) => {
        setActiveIdx(i);
        setFocusId(null);
      }}
      progress={progress}
      passing={tr.pass}
      optimal={curOptimal}
      status={statusText}
      hint={hint}
      actions={
        <button type="button" className="btn" onClick={resetBrief}>
          Reset brief
        </button>
      }
      log={{
        title: 'Detector bench log',
        lines: log.lines,
        onClear: log.clear,
        empty: 'Adjust NMS mode, confidence, or IoU threshold to record post-processing runs.',
      }}
      reveal={{
        label: `exhaustive search (${exhaust.total} configs)`,
        content: (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ fontSize: 13 }}>
              <strong>
                {exhaust.passing}/{exhaust.total}
              </strong>{' '}
              pass · {exhaust.summary}
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
              All 42 settings: confidence rows × NMS columns (highlighted cells pass; outlined cell = your current setting)
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="num-table" style={{ width: '100%', fontSize: 11.5, textAlign: 'center' }}>
                <thead>
                  <tr>
                    <th>Conf</th>
                    {gridCols.map((col) => (
                      <th key={col.key}>{col.label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {LAB_CONFS.map((c) => (
                    <tr key={c}>
                      <td style={{ fontWeight: 600 }}>{c.toFixed(2)}</td>
                      {gridCols.map((col) => {
                        const cell = grid.find(
                          (g) => g.mode === col.mode && g.conf === c && (col.mode === 'soft' || g.nmsIou === col.nmsIou),
                        )!;
                        const isCur = st.mode === col.mode && st.conf === c && (col.mode === 'soft' || st.nmsIou === col.nmsIou);
                        return (
                          <td
                            key={col.key}
                            style={{
                              background: cell.pass
                                ? 'color-mix(in srgb, var(--c-good) 22%, var(--bg))'
                                : 'color-mix(in srgb, var(--c-bad) 10%, var(--bg))',
                              color: cell.pass ? 'var(--c-good)' : 'var(--text-muted)',
                              fontWeight: cell.pass || isCur ? 700 : 400,
                              outline: isCur ? '2px solid var(--accent)' : undefined,
                              outlineOffset: -2,
                            }}
                          >
                            {cell.f1.toFixed(2)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ),
      }}
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 12 }}>
        <Segmented
          label="NMS mode"
          value={st.mode}
          onChange={(v) => updateState({ mode: v })}
          options={[
            { value: 'hard', label: 'Hard NMS' },
            { value: 'soft', label: 'Soft-NMS' },
          ]}
        />
        <Segmented
          label="Confidence"
          value={st.conf}
          onChange={(v) => updateState({ conf: v })}
          options={LAB_CONFS.map((c) => ({ value: c, label: c.toFixed(2) }))}
        />
        {st.mode === 'hard' && (
          <Segmented
            label="NMS IoU"
            value={st.nmsIou}
            onChange={(v) => updateState({ nmsIou: v })}
            options={LAB_NMS_IOUS.map((t) => ({ value: t, label: t.toFixed(2) }))}
          />
        )}
      </div>

      {/* Animated metric bars */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 8, marginBottom: 12 }}>
        {[
          { label: 'Recall', val: tweenedMetrics[0], raw: tr.recall, ok: tr.recall >= 0.999 },
          { label: 'Precision', val: tweenedMetrics[1], raw: tr.precision, ok: tr.precision >= 0.8 },
          { label: 'F1 score', val: tweenedMetrics[2], raw: tr.f1, ok: tr.pass },
        ].map((m) => (
          <div key={m.label} style={{ padding: 8, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
              <span>{m.label}</span>
              <strong style={{ fontFamily: 'var(--mono)', color: m.ok ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {m.label === 'F1 score' ? m.raw.toFixed(3) : pct1(m.raw)}
              </strong>
            </div>
            <div style={{ height: 8, borderRadius: 4, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${Math.max(0, Math.min(100, m.val * 100))}%`, background: m.ok ? 'var(--c-good)' : 'var(--c-bad)' }} />
            </div>
          </div>
        ))}
      </div>

      {/* Live scene + clickable NMS suppression trace */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 14, alignItems: 'start', marginBottom: 12 }}>
        <div>
          <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 4 }}>
            Road frame (solid = kept TP/FP · dashed = GT or MISSED · click any box or row to inspect)
          </div>
          <LabSceneSvg gts={tr.scene.gts} trace={tr.trace} missedGtIds={tr.missedGtIds} focusId={focusId} onFocus={setFocusId} />
        </div>

        <div style={{ maxHeight: 235, overflowY: 'auto', border: '1px solid var(--border)', borderRadius: 8 }}>
          <table className="num-table" style={{ width: '100%', fontSize: 11.5 }}>
            <thead>
              <tr>
                <th>Box</th>
                <th>{st.mode === 'soft' ? 'Raw → Soft' : 'Score'}</th>
                <th>Fate</th>
              </tr>
            </thead>
            <tbody>
              {tr.trace.map((r) => {
                const isFocus = r.det.id === focusId;
                const fateColor =
                  r.fate === 'kept-tp' ? 'var(--c-good)' : r.fate === 'kept-fp' ? 'var(--c-bad)' : 'var(--text-muted)';
                const fateText =
                  r.fate === 'kept-tp'
                    ? `TP → ${r.matchedGtId}`
                    : r.fate === 'kept-fp'
                      ? 'FP (extra)'
                      : r.fate === 'below-conf'
                        ? `below ${st.conf.toFixed(2)}`
                        : r.suppressedBy
                          ? st.mode === 'soft'
                            ? `decayed by ${r.suppressedBy} (IoU ${r.overlapWithActor.toFixed(2)})`
                            : `supp by ${r.suppressedBy} (IoU ${r.overlapWithActor.toFixed(2)})`
                          : 'suppressed';
                return (
                  <tr
                    key={r.det.id}
                    onClick={() => setFocusId(isFocus ? null : r.det.id)}
                    style={{
                      cursor: 'pointer',
                      background: isFocus ? 'color-mix(in srgb, var(--accent) 14%, transparent)' : undefined,
                    }}
                  >
                    <td style={{ fontWeight: 600 }}>
                      {r.det.id} <span style={{ fontWeight: 400, color: 'var(--text-muted)' }}>({r.det.cls})</span>
                    </td>
                    <td style={{ fontFamily: 'var(--mono)' }}>
                      {st.mode === 'soft' && Math.abs(r.finalScore - r.det.score) > 0.005
                        ? `${r.det.score.toFixed(2)} → ${r.finalScore.toFixed(2)}`
                        : r.det.score.toFixed(2)}
                    </td>
                    <td style={{ color: fateColor, fontWeight: r.fate.startsWith('kept') ? 600 : 400 }}>{fateText}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <LiveCode note={st.mode === 'hard' ? 'torchvision.ops.batched_nms' : 'Gaussian Soft-NMS'} code={realCode} />
    </LabShell>
  );
}
