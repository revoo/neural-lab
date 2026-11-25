// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { LiveCode } from '../../components/LabJob';
import { useMachineLog } from '../../components/MachineLog';
import { useTweened, useTweenedArray } from '../../components/motion';
import {
  CNN_LAB_BRIEFS,
  HEAD_CHOICES,
  LAYER_CATALOG,
  SHIFT_CHOICES,
  cnnBrief1Grid,
  cnnForwardOne,
  cnnLabExhaustive,
  cnnLabHint,
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
import { cnnRealCode } from './realCode';

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
};

const CNN_LAB_KEY = 'nl-cnn-lab-v1';
const CNN_LAB_PROG_KEY = 'nl-cnn-lab-progress-v2';
const CNN_LAB_DEFAULT: CnnLabState = {
  k: 5,
  nConv: 4,
  nPools: 0,
  shiftChoice: 'dense-8',
  headChoice: 'vgg-fc',
};

function loadLegacySolved(): Partial<Record<CnnBriefId, boolean>> {
  try {
    const raw = localStorage.getItem(CNN_LAB_KEY);
    if (!raw) return {};
    const p = JSON.parse(raw);
    return p?.solved ?? {};
  } catch {
    return {};
  }
}

function loadCnnLab(): CnnLabState {
  try {
    const raw = localStorage.getItem(CNN_LAB_KEY);
    if (!raw) return CNN_LAB_DEFAULT;
    const p = JSON.parse(raw);
    return {
      k: p.k === 3 || p.k === 5 ? p.k : CNN_LAB_DEFAULT.k,
      nConv: p.nConv === 2 || p.nConv === 3 || p.nConv === 4 ? p.nConv : CNN_LAB_DEFAULT.nConv,
      nPools: p.nPools === 0 || p.nPools === 1 || p.nPools === 2 ? p.nPools : CNN_LAB_DEFAULT.nPools,
      shiftChoice: SHIFT_CHOICES.some((c) => c.value === p.shiftChoice) ? p.shiftChoice : CNN_LAB_DEFAULT.shiftChoice,
      headChoice: HEAD_CHOICES.some((h) => h.value === p.headChoice) ? p.headChoice : CNN_LAB_DEFAULT.headChoice,
    };
  } catch {
    return CNN_LAB_DEFAULT;
  }
}

export function CnnArchitectLab() {
  const [activeIdx, setActiveIdx] = useState<number>(0);
  const briefId = CNN_LAB_BRIEFS[activeIdx].id;
  const [st, setSt] = useState<CnnLabState>(loadCnnLab);
  const [inspectShape, setInspectShape] = useState<0 | 1 | 2>(0);
  const progress = useLabProgress(CNN_LAB_PROG_KEY, () => ({ solved: loadLegacySolved() as Record<string, boolean> }));
  const log = useMachineLog(18);

  useEffect(() => {
    try {
      localStorage.setItem(CNN_LAB_KEY, JSON.stringify({ ...st, solved: progress.solved }));
    } catch {
      /* ignore */
    }
  }, [st, progress.solved]);

  const r1 = useMemo(() => evalCnnBrief1({ k: st.k, nConv: st.nConv, nPools: st.nPools }), [st.k, st.nConv, st.nPools]);
  const r2 = useMemo(() => evalCnnBrief2(st.shiftChoice), [st.shiftChoice]);
  const r3 = useMemo(() => evalCnnBrief3(st.headChoice), [st.headChoice]);
  const exhaust = useMemo(() => cnnLabExhaustive(briefId), [briefId]);
  const b1Grid = useMemo(() => cnnBrief1Grid(), []);
  const b2All = useMemo(() => SHIFT_CHOICES.map((c) => ({ c, r: evalCnnBrief2(c.value) })), []);
  const b3All = useMemo(() => HEAD_CHOICES.map((h) => ({ h, r: evalCnnBrief3(h.value) })), []);
  const { testShifted } = useMemo(() => getShapeDataset(), []);

  const curPass = briefId === 'rf-budget' ? r1.pass : briefId === 'shift-gen' ? r2.pass : r3.pass;
  const curOptimal =
    briefId === 'rf-budget'
      ? r1.pass && r1.params === 6960
      : briefId === 'shift-gen'
        ? r2.pass && r2.params === 55
        : r3.pass;

  const hint = useMemo(() => cnnLabHint(briefId, st), [briefId, st]);
  const realCode = useMemo(() => cnnRealCode(briefId, st), [briefId, st]);

  const tweenedParams1 = useTweened(r1.params, 260);
  const tweenedAccs2 = useTweenedArray([r2.inDistAcc, r2.shiftedAcc], 260);
  const tweenedHeadLog = useTweenedArray(
    b3All.map((x) => Math.log10(x.r.params)),
    260,
  );

  // Brief 2 live unseen bottom-right sample + prediction
  const b2Sample = testShifted[inspectShape * 20];
  const b2Pred = useMemo(() => r2.predict(b2Sample.x), [r2, b2Sample]);
  const b2CnnFwd = useMemo(
    () => (r2.cnnParams ? cnnForwardOne(r2.cnnParams, b2Sample.x) : null),
    [r2.cnnParams, b2Sample],
  );

  const updateB1 = (patch: Partial<Pick<CnnLabState, 'k' | 'nConv' | 'nPools'>>) => {
    const next = { k: patch.k ?? st.k, nConv: patch.nConv ?? st.nConv, nPools: patch.nPools ?? st.nPools };
    const ev = evalCnnBrief1(next);
    setSt((p) => ({ ...p, ...next }));
    log.push(
      `B1: ${next.nConv}× Conv ${next.k}×${next.k} + ${next.nPools} pool → RF ${ev.rf}×${ev.rf}, jump ${ev.jump}, ${ev.params.toLocaleString()} params (${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : undefined,
    );
  };

  const updateB2 = (choice: ShiftChoice) => {
    const ev = evalCnnBrief2(choice);
    const label = SHIFT_CHOICES.find((c) => c.value === choice)?.label ?? choice;
    setSt((p) => ({ ...p, shiftChoice: choice }));
    log.push(
      `B2: ${label} (${ev.params} params) → seen ${pct1(ev.inDistAcc)}, unseen ${pct1(ev.shiftedAcc)} (${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : 'bad',
    );
  };

  const updateB3 = (choice: HeadChoice) => {
    const ev = evalCnnBrief3(choice);
    const label = HEAD_CHOICES.find((h) => h.value === choice)?.label ?? choice;
    setSt((p) => ({ ...p, headChoice: choice }));
    log.push(
      `B3: ${label} → ${(ev.params / 1e6).toFixed(3)}M params, ${ev.channelsKept}/512 ch (${ev.pass ? 'PASS' : 'FAIL'})`,
      ev.pass ? 'good' : 'bad',
    );
  };

  const resetBrief = () => {
    if (briefId === 'rf-budget') {
      setSt((p) => ({ ...p, k: CNN_LAB_DEFAULT.k, nConv: CNN_LAB_DEFAULT.nConv, nPools: CNN_LAB_DEFAULT.nPools }));
      log.push('B1: reset to 4× Conv 5×5 + 0 pools');
    } else if (briefId === 'shift-gen') {
      setSt((p) => ({ ...p, shiftChoice: CNN_LAB_DEFAULT.shiftChoice }));
      log.push('B2: reset to Dense MLP (h=8)');
    } else {
      setSt((p) => ({ ...p, headChoice: CNN_LAB_DEFAULT.headChoice }));
      log.push('B3: reset to Flatten → 4096 → 1000');
    }
  };

  // Brief 1 receptive field canvas constants (21x21 grid, target 15x15 outline)
  const rfGridN = 21;
  const rfCenter = 10;
  const rfHalf = Math.floor(Math.min(21, r1.rf) / 2);
  const targetHalf = 7; // 15x15 = 2*7 + 1

  const statusText = useMemo(() => {
    if (briefId === 'rf-budget') {
      if (r1.pass) {
        return r1.params === 6960
          ? `RF ${r1.rf}×${r1.rf} (≥ 15×15) in ${r1.params.toLocaleString()} weights — exact parameter floor for this search space.`
          : `RF ${r1.rf}×${r1.rf} (≥ 15×15) in ${r1.params.toLocaleString()} weights (≤ 10,000). Optional challenge: can you go even leaner?`;
      }
      return `RF ${r1.rf}×${r1.rf} (need ≥ 15×15) · ${r1.params.toLocaleString()} weights (need ≤ 10,000) · final jump ${r1.jump}.`;
    }
    if (briefId === 'shift-gen') {
      if (r2.pass) {
        return r2.params === 55
          ? `Unseen bottom-right accuracy ${pct1(r2.shiftedAcc)} with ${r2.params} weights — smallest passing model in this suite.`
          : `Unseen bottom-right accuracy ${pct1(r2.shiftedAcc)} with ${r2.params} weights. Optional challenge: can fewer filters still clear 95%?`;
      }
      return `Seen top-left ${pct1(r2.inDistAcc)} · unseen bottom-right ${pct1(r2.shiftedAcc)} (need ≥ 95.0%) · ${r2.params} weights.`;
    }
    if (r3.pass) {
      return `Head uses ${(r3.params / 1e6).toFixed(3)}M weights (≤ 3.0M) and keeps all ${r3.channelsKept}/512 semantic channels.`;
    }
    return `Head uses ${(r3.params / 1e6).toFixed(3)}M weights (need ≤ 3.0M) and keeps ${r3.channelsKept}/512 channels.`;
  }, [briefId, r1, r2, r3]);

  return (
    <LabShell
      title="Lab: Architect the CNN"
      caption={
        <>
          Inspect the receptive field, learned spatial weights, and classification-head tensor flow live across three exhaustive CNN architecture briefs.
        </>
      }
      briefs={CNN_LAB_BRIEFS.map((b, i) => ({
        id: b.id,
        tab: `Brief ${i + 1}`,
        title: b.title,
        text: b.story,
      }))}
      active={activeIdx}
      onSelect={setActiveIdx}
      progress={progress}
      passing={curPass}
      optimal={curOptimal}
      status={statusText}
      hint={hint}
      actions={
        <button type="button" className="btn" onClick={resetBrief}>
          Reset brief
        </button>
      }
      log={{
        title: 'Architecture bench log',
        lines: log.lines,
        onClear: log.clear,
        empty: 'Adjust the stack, model family, or head to log architecture runs.',
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

              {briefId === 'rf-budget' && (
                <div>
                  <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 4 }}>
                    All 18 stacks: parameter count vs receptive field (shaded zone = RF ≥ 15 &amp; params ≤ 10,000)
                  </div>
                  <svg viewBox="0 0 360 148" style={{ width: '100%', maxWidth: 420, height: 'auto', display: 'block' }}>
                    {/* Axes */}
                    <rect x={44} y={10} width={302} height={106} fill="var(--bg)" stroke="var(--border)" rx={4} />
                    {/* Passing quadrant: x <= 10k, y >= 15 */}
                    {(() => {
                      const x10k = 44 + (10_000 / 28_000) * 302;
                      const y15 = 116 - ((15 - 1) / 35) * 106;
                      return (
                        <>
                          <rect
                            x={44}
                            y={10}
                            width={x10k - 44}
                            height={y15 - 10}
                            fill="color-mix(in srgb, var(--c-good) 14%, transparent)"
                          />
                          <line x1={x10k} y1={10} x2={x10k} y2={116} stroke="var(--c-good)" strokeDasharray="3 3" strokeWidth={1} />
                          <line x1={44} y1={y15} x2={346} y2={y15} stroke="var(--c-good)" strokeDasharray="3 3" strokeWidth={1} />
                        </>
                      );
                    })()}
                    {b1Grid.map((pt, idx) => {
                      const cx = 44 + (pt.params / 28_000) * 302;
                      const cy = 116 - ((Math.min(36, pt.rf) - 1) / 35) * 106;
                      const isCur = pt.cfg.k === st.k && pt.cfg.nConv === st.nConv && pt.cfg.nPools === st.nPools;
                      const isOpt = pt.pass && pt.params === 6960;
                      return (
                        <g key={idx}>
                          <circle
                            cx={cx}
                            cy={cy}
                            r={isCur ? 5.5 : isOpt ? 4.8 : 3.5}
                            fill={pt.pass ? 'var(--c-good)' : pt.cfg.k === 3 ? 'var(--c-a)' : 'var(--c-b)'}
                            fillOpacity={pt.pass || isCur ? 0.92 : 0.55}
                            stroke={isCur ? 'var(--text)' : isOpt ? 'var(--accent)' : 'none'}
                            strokeWidth={isCur ? 2 : 1.5}
                          />
                        </g>
                      );
                    })}
                    <text x={44} y={132} fontSize={9.5} fill="var(--text-muted)">0</text>
                    <text x={144} y={132} fontSize={9.5} fill="var(--text-muted)">10k ceiling</text>
                    <text x={312} y={132} fontSize={9.5} fill="var(--text-muted)">28k params</text>
                    <text x={6} y={116} fontSize={9.5} fill="var(--text-muted)">RF 1</text>
                    <text x={6} y={76} fontSize={9.5} fill="var(--c-good)">RF 15</text>
                    <text x={6} y={19} fontSize={9.5} fill="var(--text-muted)">RF 36</text>
                  </svg>
                </div>
              )}

              {briefId === 'shift-gen' && (
                <div>
                  <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 4 }}>
                    All 7 trained models: log10(parameters) vs unseen bottom-right accuracy
                  </div>
                  <svg viewBox="0 0 360 142" style={{ width: '100%', maxWidth: 420, height: 'auto', display: 'block' }}>
                    <rect x={44} y={10} width={302} height={100} fill="var(--bg)" stroke="var(--border)" rx={4} />
                    <line x1={44} y1={15} x2={346} y2={15} stroke="var(--c-good)" strokeDasharray="3 3" strokeWidth={1} />
                    {b2All.map(({ c, r }) => {
                      const lp = Math.log10(r.params); // ~1.46 to ~3.04
                      const cx = 56 + ((lp - 1.3) / 1.9) * 276;
                      const cy = 110 - r.shiftedAcc * 100;
                      const isCur = c.value === st.shiftChoice;
                      return (
                        <g key={c.value}>
                          <circle
                            cx={cx}
                            cy={cy}
                            r={isCur ? 5.5 : 4}
                            fill={r.pass ? 'var(--c-good)' : r.kind === 'dense' ? 'var(--c-bad)' : 'var(--c-a)'}
                            stroke={isCur ? 'var(--text)' : 'none'}
                            strokeWidth={2}
                          />
                          <text x={cx} y={cy + (r.shiftedAcc > 0.85 ? 13 : -7)} textAnchor="middle" fontSize={8.5} fill="var(--text-muted)">
                            {c.value}
                          </text>
                        </g>
                      );
                    })}
                    <text x={8} y={18} fontSize={9} fill="var(--c-good)">95%</text>
                    <text x={8} y={110} fontSize={9} fill="var(--text-muted)">0%</text>
                    <text x={50} y={128} fontSize={9} fill="var(--text-muted)">~30 params</text>
                    <text x={182} y={128} fontSize={9} fill="var(--text-muted)">~250</text>
                    <text x={294} y={128} fontSize={9} fill="var(--text-muted)">~1,100 params</text>
                  </svg>
                </div>
              )}

              {briefId === 'gap-head' && (
                <table className="num-table" style={{ width: '100%', fontSize: 12 }}>
                  <thead>
                    <tr>
                      <th>Head</th>
                      <th>Params</th>
                      <th>Channels</th>
                      <th>Pass</th>
                    </tr>
                  </thead>
                  <tbody>
                    {b3All.map(({ h, r }) => (
                      <tr key={h.value} style={{ fontWeight: h.value === st.headChoice ? 600 : 400 }}>
                        <td>{h.label}</td>
                        <td>{(r.params / 1e6).toFixed(3)}M</td>
                        <td>{r.channelsKept}</td>
                        <td style={{ color: r.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{r.pass ? 'PASS' : 'FAIL'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          ),
        }}
      >
        {briefId === 'rf-budget' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
              <Segmented
                label="Conv kernel"
                value={st.k}
                onChange={(v) => updateB1({ k: v })}
                options={[
                  { value: 3, label: '3×3' },
                  { value: 5, label: '5×5' },
                ]}
              />
              <Segmented
                label="Conv layers"
                value={st.nConv}
                onChange={(v) => updateB1({ nConv: v })}
                options={[
                  { value: 2, label: '2 convs' },
                  { value: 3, label: '3 convs' },
                  { value: 4, label: '4 convs' },
                ]}
              />
              <Segmented
                label="MaxPool 2×2 layers"
                value={st.nPools}
                onChange={(v) => updateB1({ nPools: v })}
                options={[
                  { value: 0, label: '0 pools' },
                  { value: 1, label: '1 pool' },
                  { value: 2, label: '2 pools' },
                ]}
              />
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'flex-start' }}>
              {/* Live 21x21 receptive field canvas with 15x15 target outline */}
              <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
                <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 6 }}>
                  Receptive field: <strong style={{ color: r1.rf >= 15 ? 'var(--c-good)' : 'var(--c-bad)' }}>{r1.rf}×{r1.rf}</strong> (outlined frame = 15×15 target)
                </div>
                <div style={{ position: 'relative', width: 'fit-content' }}>
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: `repeat(${rfGridN}, 8px)`,
                      gap: 1,
                      background: 'var(--border)',
                      padding: 1,
                      borderRadius: 4,
                    }}
                  >
                    {Array.from({ length: rfGridN }, (_, row) =>
                      Array.from({ length: rfGridN }, (_, col) => {
                        const isCentre = row === rfCenter && col === rfCenter;
                        const inRf = Math.abs(row - rfCenter) <= rfHalf && Math.abs(col - rfCenter) <= rfHalf;
                        const onTargetBorder =
                          (Math.abs(row - rfCenter) === targetHalf && Math.abs(col - rfCenter) <= targetHalf) ||
                          (Math.abs(col - rfCenter) === targetHalf && Math.abs(row - rfCenter) <= targetHalf);
                        return (
                          <div
                            key={`${row}-${col}`}
                            style={{
                              width: 8,
                              height: 8,
                              background: isCentre
                                ? 'var(--accent)'
                                : inRf
                                  ? r1.rf >= 15
                                    ? 'color-mix(in srgb, var(--c-good) 52%, var(--bg))'
                                    : 'color-mix(in srgb, var(--c-a) 55%, var(--bg))'
                                  : 'var(--bg)',
                              outline: onTargetBorder ? '1px solid var(--accent)' : undefined,
                              outlineOffset: -1,
                            }}
                          />
                        );
                      }),
                    )}
                  </div>
                </div>
              </div>

              {/* Interleaved stack trace + animated parameter meter */}
              <div style={{ flex: '1 1 260px', display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                    <span>Parameter budget (≤ 10,000)</span>
                    <strong style={{ fontFamily: 'var(--mono)', color: r1.params <= 10_000 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                      {Math.round(tweenedParams1).toLocaleString()} / 10,000
                    </strong>
                  </div>
                  <div style={{ position: 'relative', height: 12, borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                    <div
                      style={{
                        height: '100%',
                        width: `${Math.min(100, (tweenedParams1 / 26_000) * 100)}%`,
                        background: r1.params <= 10_000 ? 'var(--c-good)' : 'var(--c-bad)',
                      }}
                    />
                    <div
                      title="10,000 parameter ceiling"
                      style={{
                        position: 'absolute',
                        top: 0,
                        bottom: 0,
                        left: `${(10_000 / 26_000) * 100}%`,
                        width: 2,
                        background: 'var(--text)',
                      }}
                    />
                  </div>
                </div>

                <div style={{ overflowX: 'auto' }}>
                  <table className="num-table" style={{ width: '100%', fontSize: 12 }}>
                    <thead>
                      <tr>
                        <th>Stage</th>
                        <th>Layer</th>
                        <th>RF</th>
                        <th>Jump</th>
                        <th>Params</th>
                      </tr>
                    </thead>
                    <tbody>
                      {r1.trace.map((row) => (
                        <tr key={row.layer}>
                          <td>L{row.layer}</td>
                          <td>{row.spec.name}</td>
                          <td style={{ fontWeight: 600, color: row.rf >= 15 ? 'var(--c-good)' : 'var(--c-a)' }}>
                            {row.rf}×{row.rf}
                          </td>
                          <td>{row.jump}×</td>
                          <td>{row.params.toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
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
                  onClick={() => updateB2(c.value)}
                >
                  {c.label}
                </button>
              ))}
            </div>

            {/* Accuracy bars: seen top-left vs unseen bottom-right */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 10 }}>
              {[
                { label: 'Seen top-left accuracy', val: tweenedAccs2[0], raw: r2.inDistAcc, color: 'var(--c-a)' },
                { label: 'Unseen bottom-right accuracy (target ≥ 95%)', val: tweenedAccs2[1], raw: r2.shiftedAcc, color: r2.pass ? 'var(--c-good)' : 'var(--c-bad)' },
              ].map((bar) => (
                <div key={bar.label} style={{ padding: 9, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                    <span>{bar.label}</span>
                    <strong style={{ fontFamily: 'var(--mono)', color: bar.color }}>{pct1(bar.raw)}</strong>
                  </div>
                  <div style={{ height: 10, borderRadius: 5, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${Math.max(0, Math.min(100, bar.val * 100))}%`, background: bar.color }} />
                  </div>
                </div>
              ))}
            </div>

            {/* Live machinery X-ray on an unseen bottom-right shape */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-start', padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
              <div>
                <div style={{ display: 'flex', gap: 4, marginBottom: 6 }}>
                  {(['Horizontal', 'Vertical', 'Diagonal'] as const).map((name, idx) => (
                    <button
                      key={name}
                      type="button"
                      className="btn"
                      style={{
                        fontSize: 11.5,
                        padding: '2px 7px',
                        borderColor: inspectShape === idx ? 'var(--accent)' : 'var(--border)',
                        fontWeight: inspectShape === idx ? 600 : 400,
                      }}
                      onClick={() => setInspectShape(idx as 0 | 1 | 2)}
                    >
                      {name}
                    </button>
                  ))}
                </div>
                <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 4 }}>
                  Unseen bottom-right test input (8×8)
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 14px)', gap: 1, background: 'var(--border)', padding: 1, borderRadius: 4, width: 'fit-content' }}>
                  {b2Sample.x.map((v, i) => (
                    <div key={i} style={{ width: 14, height: 14, background: actBg(v, 1) }} />
                  ))}
                </div>
                <div style={{ fontSize: 12, marginTop: 6 }}>
                  Pred:{' '}
                  <strong style={{ color: b2Pred.pred === b2Sample.y ? 'var(--c-good)' : 'var(--c-bad)' }}>
                    {CLASS_NAMES[b2Pred.pred]}
                  </strong>{' '}
                  ({pct1(b2Pred.probs[b2Pred.pred])}) · {r2.params} params
                </div>
              </div>

              {r2.kind === 'dense' && r2.denseEnergy && (
                <div style={{ flex: '1 1 240px' }}>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
                    Inside Dense MLP: learned input-weight energy (mean |W₁| per pixel)
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
                    {(() => {
                      const maxE = Math.max(...r2.denseEnergy, 0.01);
                      return (
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 14px)', gap: 1, background: 'var(--border)', padding: 1, borderRadius: 4, width: 'fit-content' }}>
                          {r2.denseEnergy.map((val, i) => {
                            const r = Math.floor(i / 8);
                            const c = i % 8;
                            const inTrainZone = r < 5 && c < 5;
                            return (
                              <div
                                key={i}
                                style={{
                                  width: 14,
                                  height: 14,
                                  background: actBg(val, maxE),
                                  borderRight: c === 4 && inTrainZone ? '1px dashed var(--accent)' : undefined,
                                  borderBottom: r === 4 && inTrainZone ? '1px dashed var(--accent)' : undefined,
                                  boxSizing: 'border-box',
                                }}
                              />
                            );
                          })}
                        </div>
                      );
                    })()}
                    <div style={{ flex: '1 1 160px', fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.45 }}>
                      Bright top-left cells grew large weights during training; the bottom-right corner stayed at random initialisation because no training shape ever activated those inputs.
                    </div>
                  </div>
                </div>
              )}

              {r2.kind === 'cnn' && r2.cnnParams && b2CnnFwd && (
                <div style={{ flex: '1 1 260px' }}>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
                    Inside Tiny CNN: {r2.cnnParams.nFilters} shared 3×3 filters scanning the unseen bottom-right corner
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(105px, 1fr))', gap: 8 }}>
                    {Array.from({ length: r2.cnnParams.nFilters }, (_, f) => {
                      const w = r2.cnnParams!.convW.slice(f * 9, f * 9 + 9);
                      const maxW = Math.max(...w.map(Math.abs), 0.1);
                      const map = b2CnnFwd.maps[f];
                      const maxM = Math.max(...map, 0.1);
                      return (
                        <div key={f} style={{ padding: 6, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg)' }}>
                          <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 4 }}>
                            F{f + 1} · max {b2CnnFwd.pooled[f].toFixed(2)}
                          </div>
                          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 10px)', gap: 1, background: 'var(--border)', padding: 1 }}>
                              {w.map((val, i) => (
                                <div key={i} style={{ width: 10, height: 10, background: actBg(val, maxW) }} />
                              ))}
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 8px)', gap: 1, background: 'var(--border)', padding: 1 }}>
                              {map.map((val, i) => (
                                <div key={i} style={{ width: 8, height: 8, background: actBg(val, maxM) }} />
                              ))}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
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
                  onClick={() => updateB3(h.value)}
                >
                  {h.label}
                </button>
              ))}
            </div>

            {/* Live tensor flow diagram */}
            <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', fontFamily: 'var(--mono)', fontSize: 12 }}>
                <span style={{ padding: '4px 8px', borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)' }}>
                  7×7×512 (25,088)
                </span>
                <span>→</span>
                <span style={{ padding: '4px 8px', borderRadius: 6, background: 'var(--bg)', border: `1px solid ${r3.pass ? 'var(--c-good)' : 'var(--c-bad)'}`, fontWeight: 600 }}>
                  {st.headChoice === 'vgg-fc'
                    ? 'Flatten (25,088) → Dense (4,096)'
                    : st.headChoice === 'flatten-linear'
                      ? 'Flatten (25,088)'
                      : st.headChoice === 'bottleneck-gap'
                        ? '1×1 Conv (7×7×128) → GAP (128)'
                        : 'Global Avg Pool (512)'}
                </span>
                <span>→</span>
                <span style={{ padding: '4px 8px', borderRadius: 6, background: 'var(--bg)', border: '1px solid var(--border)' }}>
                  1,000 logits
                </span>
              </div>
              <div style={{ fontSize: 12, color: r3.channelsKept === 512 ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 600 }}>
                Channels kept: {r3.channelsKept} / 512
              </div>
            </div>

            {/* Selected head meters (full 4-head comparison unlocks after solve) */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
              {(() => {
                const curIdx = b3All.findIndex((x) => x.h.value === st.headChoice);
                const logVal = tweenedHeadLog[Math.max(0, curIdx)];
                const pct = Math.max(4, Math.min(100, ((logVal - 5.0) / 3.2) * 100));
                const ceilPct = ((Math.log10(3_000_000) - 5.0) / 3.2) * 100;
                return (
                  <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 5 }}>
                      <span>Head params (log₁₀ scale · marker = 3.0M)</span>
                      <strong style={{ fontFamily: 'var(--mono)', color: r3.params <= 3_000_000 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                        {(r3.params / 1e6).toFixed(3)}M
                      </strong>
                    </div>
                    <div style={{ position: 'relative', height: 12, borderRadius: 5, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${pct}%`,
                          background: r3.params <= 3_000_000 ? 'var(--c-good)' : 'var(--c-bad)',
                        }}
                      />
                      <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${ceilPct}%`, width: 2, background: 'var(--text)' }} />
                    </div>
                  </div>
                );
              })()}
              <div style={{ padding: 10, borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 5 }}>
                  <span>Semantic channels preserved (need 512)</span>
                  <strong style={{ fontFamily: 'var(--mono)', color: r3.channelsKept === 512 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                    {r3.channelsKept} / 512
                  </strong>
                </div>
                <div style={{ height: 12, borderRadius: 5, background: 'var(--bg)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${(r3.channelsKept / 512) * 100}%`,
                      background: r3.channelsKept === 512 ? 'var(--c-good)' : 'var(--c-bad)',
                    }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
        <LiveCode note="PyTorch nn.Sequential" code={realCode} />
      </LabShell>
  );
}
