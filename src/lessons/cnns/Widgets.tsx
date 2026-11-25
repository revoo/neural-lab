// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import {
  CNN_LAB_BRIEFS,
  HEAD_CHOICES,
  LAYER_CATALOG,
  SHIFT_CHOICES,
  cnnForwardOne,
  cnnLabExhaustive,
  evalCnnBrief1,
  evalCnnBrief2,
  evalCnnBrief3,
  getShapeDataset,
  getShiftRace,
  pool2x2,
  traceReceptiveField,
  vggSavings,
  type CnnBriefId,
  type HeadChoice,
  type LayerSpec,
  type ShiftChoice,
} from './cnn';

const pct1 = (v: number) => `${(100 * v).toFixed(1)}%`;

function actBg(v: number, maxAbs: number): string {
  if (maxAbs < 1e-9 || Math.abs(v) < 1e-6) return 'var(--bg-sunken)';
  const t = Math.min(1, Math.abs(v) / maxAbs);
  const pct = Math.round(14 + 68 * t);
  const base = v > 0 ? 'var(--c-a)' : 'var(--c-b)';
  return `color-mix(in srgb, ${base} ${pct}%, var(--bg))`;
}

// ------------------------------------------------------------------ 1. Receptive Field Growth Explorer

const RF_PRESETS: { id: string; label: string; stack: string[] }[] = [
  { id: 'vgg', label: 'Conv3 + Conv3 + Pool + Conv3 + Pool + Conv3', stack: ['conv3', 'conv3', 'pool2', 'conv3', 'pool2', 'conv3'] },
  { id: 'plain', label: '6× Plain Conv 3×3 (no pooling)', stack: ['conv3', 'conv3', 'conv3', 'conv3', 'conv3', 'conv3'] },
  { id: 'big', label: '2× Conv 5×5 (compare to 4× Conv 3×3)', stack: ['conv5', 'conv5'] },
];

export function ReceptiveFieldWidget() {
  const [stackIds, setStackIds] = useState<string[]>(RF_PRESETS[0].stack);
  const layers = useMemo<LayerSpec[]>(
    () => stackIds.map((id) => LAYER_CATALOG.find((l) => l.id === id) ?? LAYER_CATALOG[0]),
    [stackIds],
  );
  const rows = useMemo(() => traceReceptiveField(24, layers, 16), [layers]);
  const last = rows[rows.length - 1];
  const totalParams = rows.reduce((a, r) => a + r.params, 0);
  const vgg2 = useMemo(() => vggSavings(2, 64), []);
  const vgg3 = useMemo(() => vggSavings(3, 64), []);

  const rfClamped = Math.min(24, last ? last.rf : 1);
  const gridN = 21;
  const center = Math.floor(gridN / 2);
  const half = Math.floor(rfClamped / 2);

  return (
    <Widget
      title="How a deep stack of tiny 3×3 kernels sees the whole image"
      wide
      caption={
        <>
          Append layers or pick a preset and watch the highlighted <strong>receptive field</strong> (blue square) expand on a 21×21 canvas. Without pooling, six <code>3×3</code> convs reach only <code>13×13</code>; inserting two <code>2×2</code> MaxPools doubles the effective stride (jump) twice so four <code>3×3</code> convs reach <strong>20×20</strong>!
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
        {RF_PRESETS.map((p) => (
          <button key={p.id} type="button" className="btn" style={{ fontSize: 12, padding: '4px 9px' }} onClick={() => setStackIds(p.stack)}>
            {p.label}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center', marginBottom: 12 }}>
        <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600 }}>Add layer:</span>
        {LAYER_CATALOG.map((l) => (
          <button
            key={l.id}
            type="button"
            className="btn"
            style={{ fontSize: 12, padding: '4px 8px' }}
            disabled={stackIds.length >= 7}
            onClick={() => setStackIds((s) => [...s, l.id])}
          >
            + {l.name}
          </button>
        ))}
        <button
          type="button"
          className="btn"
          style={{ fontSize: 12, padding: '4px 8px' }}
          disabled={stackIds.length <= 1}
          onClick={() => setStackIds((s) => s.slice(0, -1))}
        >
          Remove last
        </button>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>
            Input pixels seen by 1 output neuron ({last?.rf ?? 1}×{last?.rf ?? 1})
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${gridN}, 10px)`, gap: 1, background: 'var(--border)', padding: 1, borderRadius: 6, width: 'fit-content' }}>
            {Array.from({ length: gridN }, (_, r) =>
              Array.from({ length: gridN }, (_, c) => {
                const isCentre = r === center && c === center;
                const inRf = Math.abs(r - center) <= half && Math.abs(c - center) <= half;
                return (
                  <div
                    key={`${r}-${c}`}
                    style={{
                      width: 10,
                      height: 10,
                      background: isCentre
                        ? 'var(--accent)'
                        : inRf
                          ? 'color-mix(in srgb, var(--c-a) 55%, var(--bg))'
                          : 'var(--bg-sunken)',
                    }}
                  />
                );
              }),
            )}
          </div>
        </div>

        <div style={{ flex: '1 1 280px', overflowX: 'auto' }}>
          <table className="num-table" style={{ width: '100%', fontSize: 12.5 }}>
            <thead>
              <tr>
                <th>Layer</th>
                <th>Op</th>
                <th>RF</th>
                <th>Jump</th>
                <th>Params (16ch)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.layer}>
                  <td>L{r.layer}</td>
                  <td>{r.spec.name}</td>
                  <td style={{ fontWeight: 600, color: 'var(--c-a)' }}>{r.rf}×{r.rf}</td>
                  <td>{r.jump}</td>
                  <td>{r.params.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ fontSize: 12.5, color: 'var(--text-muted)', marginTop: 8, lineHeight: 1.5 }}>
            Total params (16ch): <strong>{totalParams.toLocaleString()}</strong> · VGG insight: two <code>3×3</code>s match a <code>5×5</code> RF with <strong>{(100 * vgg2.reduction).toFixed(0)}%</strong> fewer weights; three <code>3×3</code>s match a <code>7×7</code> with <strong>{(100 * vgg3.reduction).toFixed(1)}%</strong> fewer weights!
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Pooling & Local Shift Invariance

export function PoolingWidget() {
  const [shift, setShift] = useState<number>(0);
  const [mode, setMode] = useState<'max' | 'avg'>('max');

  const grid = useMemo(() => {
    const g = Array.from({ length: 6 }, () => new Array<number>(6).fill(0));
    // Vertical edge highlight at column (2 + shift)
    for (let r = 1; r <= 4; r++) g[r][2 + shift] = 1.0;
    return g;
  }, [shift]);

  const pooled = useMemo(() => pool2x2(grid, mode), [grid, mode]);

  return (
    <Widget
      title="Pooling: trading exact pixel coordinates for local shift invariance"
      wide
      caption={
        <>
          Toggle <strong>1-pixel jitter</strong> between <code>+0 px</code> and <code>+1 px</code>. With <strong>2×2 Max Pool</strong>, shifting the bright vertical bar inside its 2-pixel cell leaves the pooled <code>3×3</code> output <strong>100% identical</strong> (peak stays <code>1.00</code>). Switch to <strong>2×2 Avg Pool</strong> and a sharp 1-pixel edge gets diluted in half (<code>0.50</code>).
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 12 }}>
        <Segmented
          label="Edge jitter"
          value={shift}
          onChange={setShift}
          options={[
            { value: 0, label: 'Original (+0 px)' },
            { value: 1, label: 'Shifted right (+1 px)' },
          ]}
        />
        <Segmented
          label="Pool type"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'max', label: '2×2 Max Pool' },
            { value: 'avg', label: '2×2 Avg Pool' },
          ]}
        />
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 20, alignItems: 'center' }}>
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>6×6 Feature map (2×2 blocks outlined)</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 28px)', gap: 2, background: 'var(--border)', padding: 2, borderRadius: 6, width: 'fit-content' }}>
            {grid.map((row, r) =>
              row.map((v, c) => (
                <div
                  key={`${r}-${c}`}
                  style={{
                    width: 28,
                    height: 28,
                    background: actBg(v, 1),
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontFamily: 'var(--mono)',
                    fontSize: 11,
                    borderRight: c === 1 || c === 3 ? '2px solid var(--accent)' : undefined,
                    borderBottom: r === 1 || r === 3 ? '2px solid var(--accent)' : undefined,
                    boxSizing: 'border-box',
                  }}
                >
                  {v.toFixed(0)}
                </div>
              )),
            )}
          </div>
        </div>
        <div style={{ fontSize: 20, color: 'var(--text-muted)' }}>→</div>
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>3×3 Pooled map</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 44px)', gap: 3, background: 'var(--border)', padding: 3, borderRadius: 6 }}>
            {pooled.map((row, r) =>
              row.map((v, c) => (
                <div
                  key={`${r}-${c}`}
                  style={{
                    width: 44,
                    height: 44,
                    background: actBg(v, 1),
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontFamily: 'var(--mono)',
                    fontSize: 12,
                    fontWeight: 600,
                  }}
                >
                  {v.toFixed(2)}
                </div>
              )),
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Live Trained Tiny CNN vs Dense MLP on Unseen Shifts

const CLASS_NAMES = ['Horizontal Bar', 'Vertical Bar', 'Diagonal Slash'] as const;

export function TrainedCnnWidget() {
  const { cnn, dense } = useMemo(() => getShiftRace(), []);
  const { testInDist, testShifted } = useMemo(() => getShapeDataset(), []);
  const [split, setSplit] = useState<'indist' | 'shifted'>('shifted');
  const [sampleIdx, setSampleIdx] = useState<number>(0);

  const ds = split === 'indist' ? testInDist : testShifted;
  // Pick 3 representative examples (0=horiz, 20=vert, 40=diag) + offset
  const sample = ds[(sampleIdx * 20) % ds.length];
  const fwd = useMemo(() => cnnForwardOne(cnn.params, sample.x), [cnn, sample]);
  const pred = fwd.probs[0] >= fwd.probs[1] && fwd.probs[0] >= fwd.probs[2] ? 0 : fwd.probs[1] >= fwd.probs[2] ? 1 : 2;

  return (
    <Widget
      title="Real trained 55-parameter CNN vs 547-parameter Dense MLP on unseen positions"
      wide
      caption={
        <>
          Both models were trained <strong>only</strong> on shapes sitting in the <strong>top-left corner</strong> of an 8×8 canvas. On top-left test images, both score <strong>100.0%</strong>. Now switch to <strong>Unseen bottom-right shifts</strong>: the 547-param Dense MLP collapses to <strong>28.3%</strong> (worse than guessing!), while our 55-param CNN (with 4 learned 3×3 kernels + GlobalMaxPool) scores <strong>100.0%</strong>! Inspect its 4 learned kernels and feature maps below.
        </>
      }
    >
      <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 8, marginBottom: 12 }}>
        <div className="readout">
          <div className="readout-label">Tiny CNN (55 params)</div>
          <div className="readout-value" style={{ color: 'var(--c-good)', fontSize: 15 }}>
            In-dist {pct1(cnn.inDistAcc)} · Shifted {pct1(cnn.shiftedAcc)}
          </div>
        </div>
        <div className="readout">
          <div className="readout-label">Dense MLP (547 params)</div>
          <div className="readout-value" style={{ color: 'var(--c-bad)', fontSize: 15 }}>
            In-dist {pct1(dense.inDistAcc)} · Shifted {pct1(dense.shiftedAcc)}
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end', marginBottom: 12 }}>
        <Segmented
          label="Test split"
          value={split}
          onChange={setSplit}
          options={[
            { value: 'indist', label: 'Top-left (seen region)' },
            { value: 'shifted', label: 'Bottom-right (UNSEEN!)' },
          ]}
        />
        <Segmented
          label="Inspect shape"
          value={sampleIdx}
          onChange={setSampleIdx}
          options={[
            { value: 0, label: 'Horizontal' },
            { value: 1, label: 'Vertical' },
            { value: 2, label: 'Diagonal' },
          ]}
        />
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>
            Input 8×8 ({CLASS_NAMES[sample.y]})
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 16px)', gap: 1, background: 'var(--border)', padding: 1, borderRadius: 6, width: 'fit-content' }}>
            {sample.x.map((v, i) => (
              <div key={i} style={{ width: 16, height: 16, background: actBg(v, 1) }} />
            ))}
          </div>
          <div style={{ fontSize: 12, marginTop: 6 }}>
            CNN pred: <strong style={{ color: pred === sample.y ? 'var(--c-good)' : 'var(--c-bad)' }}>{CLASS_NAMES[pred]}</strong> ({pct1(fwd.probs[pred])})
          </div>
        </div>

        <div style={{ flex: '1 1 300px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(115px, 1fr))', gap: 10 }}>
          {[0, 1, 2, 3].map((f) => {
            const w = cnn.params.convW.slice(f * 9, f * 9 + 9);
            const maxW = Math.max(...w.map(Math.abs), 0.1);
            const map = fwd.maps[f];
            const maxM = Math.max(...map, 0.1);
            return (
              <div key={f} style={{ padding: 8, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
                <div style={{ fontSize: 11.5, fontWeight: 600, marginBottom: 4 }}>
                  Filter {f + 1} · max {fwd.pooled[f].toFixed(2)}
                </div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 12px)', gap: 1, background: 'var(--border)', padding: 1, borderRadius: 3 }}>
                    {w.map((val, i) => (
                      <div key={i} style={{ width: 12, height: 12, background: actBg(val, maxW) }} />
                    ))}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 10px)', gap: 1, background: 'var(--border)', padding: 1, borderRadius: 3 }}>
                    {map.map((val, i) => (
                      <div key={i} style={{ width: 10, height: 10, background: actBg(val, maxM) }} />
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Architect the CNN

type CnnLabState = {
  k: 3 | 5;
  nConv: 2 | 3 | 4;
  nPools: 0 | 1 | 2;
  shiftChoice: ShiftChoice;
  headChoice: HeadChoice;
  solved: Record<CnnBriefId, boolean>;
};

const CNN_LAB_KEY = 'nl-cnn-lab-v1';
const CNN_LAB_DEFAULT: CnnLabState = {
  k: 5,
  nConv: 4,
  nPools: 0,
  shiftChoice: 'dense-8',
  headChoice: 'vgg-fc',
  solved: { 'rf-budget': false, 'shift-gen': false, 'gap-head': false },
};

function loadCnnLab(): CnnLabState {
  try {
    const raw = localStorage.getItem(CNN_LAB_KEY);
    if (!raw) return CNN_LAB_DEFAULT;
    const p = JSON.parse(raw);
    return { ...CNN_LAB_DEFAULT, ...p, solved: { ...CNN_LAB_DEFAULT.solved, ...(p.solved ?? {}) } };
  } catch {
    return CNN_LAB_DEFAULT;
  }
}

export function CnnArchitectLab() {
  const [briefId, setBriefId] = useState<CnnBriefId>('rf-budget');
  const [st, setSt] = useState<CnnLabState>(loadCnnLab);
  const [showOpt, setShowOpt] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(CNN_LAB_KEY, JSON.stringify(st));
    } catch {
      /* ignore */
    }
  }, [st]);

  const r1 = useMemo(() => evalCnnBrief1({ k: st.k, nConv: st.nConv, nPools: st.nPools }), [st.k, st.nConv, st.nPools]);
  const r2 = useMemo(() => evalCnnBrief2(st.shiftChoice), [st.shiftChoice]);
  const r3 = useMemo(() => evalCnnBrief3(st.headChoice), [st.headChoice]);
  const exhaust = useMemo(() => cnnLabExhaustive(briefId), [briefId]);

  const curPass = briefId === 'rf-budget' ? r1.pass : briefId === 'shift-gen' ? r2.pass : r3.pass;
  useEffect(() => {
    if (curPass && !st.solved[briefId]) {
      setSt((p) => ({ ...p, solved: { ...p.solved, [briefId]: true } }));
    }
  }, [curPass, briefId, st.solved]);

  const brief = CNN_LAB_BRIEFS.find((b) => b.id === briefId)!;
  const solvedCount = Object.values(st.solved).filter(Boolean).length;

  return (
    <Widget
      title="Lab: Architect the CNN"
      wide
      caption={
        <>
          Three real CNN architecture trade-offs checked exhaustively over every setting. Solved: <strong>{solvedCount}/3</strong>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
        {CNN_LAB_BRIEFS.map((b, i) => (
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

      {briefId === 'rf-budget' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Conv kernel"
              value={st.k}
              onChange={(v) => setSt((p) => ({ ...p, k: v }))}
              options={[
                { value: 3, label: '3×3' },
                { value: 5, label: '5×5' },
              ]}
            />
            <Segmented
              label="Conv layers"
              value={st.nConv}
              onChange={(v) => setSt((p) => ({ ...p, nConv: v }))}
              options={[
                { value: 2, label: '2 convs' },
                { value: 3, label: '3 convs' },
                { value: 4, label: '4 convs' },
              ]}
            />
            <Segmented
              label="MaxPool 2×2 layers"
              value={st.nPools}
              onChange={(v) => setSt((p) => ({ ...p, nPools: v }))}
              options={[
                { value: 0, label: '0 pools' },
                { value: 1, label: '1 pool' },
                { value: 2, label: '2 pools' },
              ]}
            />
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Receptive field (≥ 15)</div>
              <div className="readout-value" style={{ color: r1.rf >= 15 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r1.rf}×{r1.rf}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Params (≤ 10,000)</div>
              <div className="readout-value" style={{ color: r1.params <= 10_000 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r1.params.toLocaleString()}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r1.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r1.pass ? (r1.params === 6960 ? 'PASS (Opt 6,960!)' : 'PASS') : 'FAIL'}
              </div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'shift-gen' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {SHIFT_CHOICES.map((c) => (
              <button
                key={c.value}
                type="button"
                className="btn"
                style={{
                  fontSize: 12.5,
                  padding: '5px 10px',
                  borderColor: st.shiftChoice === c.value ? 'var(--accent)' : 'var(--border)',
                  background: st.shiftChoice === c.value ? 'var(--bg-sunken)' : 'transparent',
                  fontWeight: st.shiftChoice === c.value ? 600 : 400,
                }}
                onClick={() => setSt((p) => ({ ...p, shiftChoice: c.value }))}
              >
                {c.label}
              </button>
            ))}
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Parameters</div>
              <div className="readout-value">{r2.params}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Top-left (seen)</div>
              <div className="readout-value">{pct1(r2.inDistAcc)}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Bottom-right (≥ 95%)</div>
              <div className="readout-value" style={{ color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {pct1(r2.shiftedAcc)}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r2.pass ? (r2.params === 55 ? 'PASS (Opt 55!)' : 'PASS') : 'FAIL'}
              </div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'gap-head' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {HEAD_CHOICES.map((h) => (
              <button
                key={h.value}
                type="button"
                className="btn"
                style={{
                  fontSize: 12.5,
                  padding: '5px 10px',
                  borderColor: st.headChoice === h.value ? 'var(--accent)' : 'var(--border)',
                  background: st.headChoice === h.value ? 'var(--bg-sunken)' : 'transparent',
                  fontWeight: st.headChoice === h.value ? 600 : 400,
                }}
                onClick={() => setSt((p) => ({ ...p, headChoice: h.value }))}
              >
                {h.label}
              </button>
            ))}
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Head params (≤ 3.0M)</div>
              <div className="readout-value" style={{ color: r3.params <= 3_000_000 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {(r3.params / 1e6).toFixed(3)}M
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Channels kept (= 512)</div>
              <div className="readout-value" style={{ color: r3.channelsKept === 512 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r3.channelsKept}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: r3.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {r3.pass ? 'PASS (Unique!)' : 'FAIL'}
              </div>
            </div>
          </div>
        </div>
      )}

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
