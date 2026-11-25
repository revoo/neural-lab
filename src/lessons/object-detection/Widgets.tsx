// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Plot } from '../../components/plot';
import {
  DET_LAB_BRIEFS,
  LAB_CONFS,
  LAB_NMS_IOUS,
  STREET_SCENE,
  computeCocoMap,
  detLabExhaustive,
  evalDetLab,
  evaluateDetections,
  iouBreakdown,
  runHardNms,
  runSoftNms,
  squareShiftIou,
  type Box,
  type DetBriefId,
  type Detection,
  type GroundTruth,
  type NmsMode,
} from './detect';

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
  solved: Record<DetBriefId, boolean>;
};

const DET_LAB_KEY = 'nl-det-lab-v1';
const DET_LAB_DEFAULT: DetLabState = {
  mode: 'hard',
  conf: 0.15,
  nmsIou: 0.85,
  solved: { clean: false, crowd: false, safety: false },
};

function loadDetLab(): DetLabState {
  try {
    const raw = localStorage.getItem(DET_LAB_KEY);
    if (!raw) return DET_LAB_DEFAULT;
    const p = JSON.parse(raw);
    return { ...DET_LAB_DEFAULT, ...p, solved: { ...DET_LAB_DEFAULT.solved, ...(p.solved ?? {}) } };
  } catch {
    return DET_LAB_DEFAULT;
  }
}

export function DetectorTuningLab() {
  const [briefId, setBriefId] = useState<DetBriefId>('clean');
  const [st, setSt] = useState<DetLabState>(loadDetLab);
  const [showOpt, setShowOpt] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(DET_LAB_KEY, JSON.stringify(st));
    } catch {
      /* ignore */
    }
  }, [st]);

  const res = useMemo(() => evalDetLab(briefId, st.conf, st.nmsIou, st.mode), [briefId, st.conf, st.nmsIou, st.mode]);
  const keptIds = useMemo(() => new Set(res.kept.map((d) => d.id)), [res.kept]);
  const exhaust = useMemo(() => detLabExhaustive(briefId), [briefId]);

  useEffect(() => {
    if (res.pass && !st.solved[briefId]) {
      setSt((p) => ({ ...p, solved: { ...p.solved, [briefId]: true } }));
    }
  }, [res.pass, briefId, st.solved]);

  const brief = DET_LAB_BRIEFS.find((b) => b.id === briefId)!;
  const solvedCount = Object.values(st.solved).filter(Boolean).length;

  return (
    <Widget
      title="Lab: Tune the Object Detector"
      wide
      caption={
        <>
          Configure post-processing (Hard NMS vs Soft-NMS, confidence cut-off, IoU threshold) across three real traffic scenes, exhaustively verified over all 42 settings per brief. Solved: <strong>{solvedCount}/3</strong>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
        {DET_LAB_BRIEFS.map((b, i) => (
          <button
            key={b.id}
            type="button"
            className="btn"
            style={{
              borderColor: briefId === b.id ? 'var(--accent)' : 'var(--border)',
              background: briefId === b.id ? 'var(--bg-sunken)' : 'transparent',
              fontWeight: briefId === b.id ? 600 : 400,
            }}
            onClick={() => {
              setBriefId(b.id);
              setShowOpt(false);
            }}
          >
            {st.solved[b.id] ? '✓ ' : ''}Brief {i + 1}
          </button>
        ))}
      </div>

      <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 4 }}>{brief.title}</div>
      <p style={{ fontSize: 13.5, color: 'var(--text-muted)', marginTop: 0, marginBottom: 12, lineHeight: 1.5 }}>{brief.story}</p>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 12 }}>
        <Segmented
          label="NMS mode"
          value={st.mode}
          onChange={(v) => setSt((p) => ({ ...p, mode: v }))}
          options={[
            { value: 'hard', label: 'Hard NMS' },
            { value: 'soft', label: 'Soft-NMS' },
          ]}
        />
        <Segmented
          label="Confidence"
          value={st.conf}
          onChange={(v) => setSt((p) => ({ ...p, conf: v }))}
          options={LAB_CONFS.map((c) => ({ value: c, label: c.toFixed(2) }))}
        />
        {st.mode === 'hard' && (
          <Segmented
            label="NMS IoU"
            value={st.nmsIou}
            onChange={(v) => setSt((p) => ({ ...p, nmsIou: v }))}
            options={LAB_NMS_IOUS.map((t) => ({ value: t, label: t.toFixed(2) }))}
          />
        )}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center' }}>
        <SceneSvg gts={res.scene.gts} allDets={res.scene.dets} keptIds={keptIds} showGt />
        <div className="readouts" style={{ flex: '1 1 220px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: 8, margin: 0 }}>
          <div className="readout">
            <div className="readout-label">Recall</div>
            <div className="readout-value" style={{ color: res.recall >= 0.999 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(res.recall)}</div>
          </div>
          <div className="readout">
            <div className="readout-label">Precision</div>
            <div className="readout-value" style={{ color: res.precision >= 0.8 ? 'var(--c-good)' : 'var(--c-bad)' }}>{pct1(res.precision)}</div>
          </div>
          <div className="readout">
            <div className="readout-label">TP / FP / FN</div>
            <div className="readout-value" style={{ fontSize: 15 }}>{res.tp} / {res.fp} / {res.fn}</div>
          </div>
          <div className="readout">
            <div className="readout-label">Status</div>
            <div className="readout-value" style={{ color: res.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{res.pass ? 'PASS' : 'FAIL'}</div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid var(--border)', display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between' }}>
        <button type="button" className="btn" onClick={() => setShowOpt((v) => !v)}>
          {showOpt ? 'Hide exhaustive check' : `Reveal exhaustive search (${exhaust.total} configs)`}
        </button>
        {showOpt && (
          <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
            <strong>{exhaust.passing}/{exhaust.total}</strong> pass · {exhaust.summary}
          </div>
        )}
      </div>
    </Widget>
  );
}
