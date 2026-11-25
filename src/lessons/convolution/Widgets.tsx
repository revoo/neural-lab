// authored-by: Gemini Antigravity
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import {
  CONV_LAB_BRIEFS,
  KERNEL_PRESETS,
  TEST_IMAGES,
  compareDenseVsConvParams,
  conv2d,
  convLabExhaustive,
  droppedBorderPixels,
  evalDiagonalKernel,
  evalStackShape,
  evalVerticalKernel,
  outSize,
  patchAt,
  validStackShrinkage,
  verifyEquivariance,
  type ConvBriefId,
  type Kernel3,
} from './conv';

function fmtK(v: number): string {
  if (Math.abs(v - 1 / 9) < 1e-4) return '1/9';
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}

/** Map signed activation to theme-safe CSS colour + opacity */
function actStyle(v: number, maxAbs: number): { background: string; color: string } {
  if (maxAbs < 1e-9 || Math.abs(v) < 1e-6) {
    return { background: 'var(--bg-sunken)', color: 'var(--text-muted)' };
  }
  const t = Math.min(1, Math.abs(v) / maxAbs);
  const pct = Math.round(14 + 66 * t);
  const base = v > 0 ? 'var(--c-a)' : 'var(--c-b)';
  return {
    background: `color-mix(in srgb, ${base} ${pct}%, var(--bg))`,
    color: t > 0.55 ? 'var(--bg)' : 'var(--text)',
  };
}

// ------------------------------------------------------------------ 1. Sliding-Window Kernel Explorer

export function SlidingKernelWidget() {
  const [imgId, setImgId] = useState<string>(TEST_IMAGES[0].id);
  const [presetId, setPresetId] = useState<string>('sobel-x');
  const [kernel, setKernel] = useState<Kernel3>(KERNEL_PRESETS[0].kernel);
  const [sel, setSel] = useState<[number, number]>([4, 1]);

  const img = useMemo(() => TEST_IMAGES.find((i) => i.id === imgId) ?? TEST_IMAGES[0], [imgId]);
  const out = useMemo(() => conv2d(img.grid, kernel, 1, 1), [img, kernel]);
  const patch = useMemo(() => patchAt(img.grid, sel[0], sel[1], 1, 1, 3), [img, sel]);

  const maxAbs = useMemo(() => {
    let m = 0;
    for (const row of out) for (const v of row) if (Math.abs(v) > m) m = Math.abs(v);
    return m || 1;
  }, [out]);

  const selectPreset = (id: string) => {
    setPresetId(id);
    const p = KERNEL_PRESETS.find((k) => k.id === id);
    if (p) setKernel(p.kernel.map((r) => [...r]) as Kernel3);
  };

  const cycleCell = (kr: number, kc: number) => {
    const vals = [-2, -1, 0, 1, 2];
    setPresetId('custom');
    setKernel((prev) => {
      const next = prev.map((r) => [...r]) as Kernel3;
      const cur = Math.round(next[kr][kc]);
      const idx = vals.indexOf(cur);
      next[kr][kc] = vals[(idx + 1) % vals.length];
      return next;
    });
  };

  const presetDesc = KERNEL_PRESETS.find((k) => k.id === presetId)?.desc ?? 'Custom kernel — click any of the 9 kernel weights to cycle through −2, −1, 0, +1, +2.';

  return (
    <Widget
      title="Slide a 3×3 kernel across an image — and watch every dot product"
      wide
      caption={
        <>
          Click any cell in the <strong>Feature map</strong> (right) to move the highlighted <code>3×3</code> window across the input image (left). Blue = positive response, orange = negative response. Try switching between <strong>Sobel Gx</strong> (vertical edges), <strong>Sobel Gy</strong> (horizontal edges), and <strong>Laplacian</strong> — or click the 9 kernel cells to invent your own filter!
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'flex-end', marginBottom: 10 }}>
        <Segmented
          label="Input image"
          value={imgId}
          onChange={setImgId}
          options={TEST_IMAGES.map((i) => ({ value: i.id, label: i.name }))}
        />
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
        {KERNEL_PRESETS.map((p) => (
          <button
            key={p.id}
            type="button"
            className="btn"
            style={{
              fontSize: 12.5,
              padding: '4px 9px',
              borderColor: presetId === p.id ? 'var(--accent)' : 'var(--border)',
              background: presetId === p.id ? 'var(--bg-sunken)' : 'transparent',
              fontWeight: presetId === p.id ? 600 : 400,
            }}
            onClick={() => selectPreset(p.id)}
          >
            {p.name}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'flex-start' }}>
        {/* Input 10x10 grid */}
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Input (10×10, pad=1)</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(10, 22px)', gap: 1, background: 'var(--border)', padding: 1, borderRadius: 6 }}>
            {img.grid.map((row, r) =>
              row.map((v, c) => {
                const inWin = Math.abs(r - sel[0]) <= 1 && Math.abs(c - sel[1]) <= 1;
                const isCentre = r === sel[0] && c === sel[1];
                return (
                  <button
                    key={`${r}-${c}`}
                    type="button"
                    onClick={() => setSel([r, c])}
                    aria-label={`Input row ${r} col ${c}`}
                    style={{
                      width: 22,
                      height: 22,
                      padding: 0,
                      border: isCentre ? '2px solid var(--accent)' : inWin ? '1.5px solid var(--c-a)' : 'none',
                      background: v > 0.5 ? 'var(--text)' : 'var(--bg)',
                      color: v > 0.5 ? 'var(--bg)' : 'var(--text-muted)',
                      fontSize: 10,
                      fontFamily: 'var(--mono)',
                      cursor: 'pointer',
                      boxSizing: 'border-box',
                    }}
                  >
                    {v}
                  </button>
                );
              }),
            )}
          </div>
        </div>

        {/* Editable 3x3 kernel + dot product breakdown */}
        <div style={{ minWidth: 165 }}>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>3×3 Kernel (click to edit)</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 46px)', gap: 4, marginBottom: 8 }}>
            {kernel.map((row, kr) =>
              row.map((w, kc) => (
                <button
                  key={`${kr}-${kc}`}
                  type="button"
                  onClick={() => cycleCell(kr, kc)}
                  style={{
                    height: 40,
                    borderRadius: 6,
                    border: '1px solid var(--border)',
                    ...actStyle(w, 2),
                    fontFamily: 'var(--mono)',
                    fontSize: 13,
                    fontWeight: 600,
                    cursor: 'pointer',
                  }}
                >
                  {fmtK(w)}
                </button>
              )),
            )}
          </div>
          <div style={{ padding: '8px 10px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', fontSize: 12, fontFamily: 'var(--mono)', lineHeight: 1.55 }}>
            <div style={{ color: 'var(--text-muted)', fontFamily: 'var(--sans)', marginBottom: 3 }}>
              Output at <code>({sel[0]}, {sel[1]})</code>:
            </div>
            <div>
              Sum of 9 products = <strong style={{ color: out[sel[0]][sel[1]] > 0 ? 'var(--c-a)' : out[sel[0]][sel[1]] < 0 ? 'var(--c-b)' : 'var(--text)' }}>{out[sel[0]][sel[1]].toFixed(2)}</strong>
            </div>
            <div style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 2 }}>
              Patch active: {patch.flat().filter((x) => x > 0).length}/9 · Kernel sum: {kernel.flat().reduce((a, b) => a + b, 0).toFixed(2)}
            </div>
          </div>
        </div>

        {/* Output 10x10 feature map */}
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Feature map (10×10)</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(10, 22px)', gap: 1, background: 'var(--border)', padding: 1, borderRadius: 6 }}>
            {out.map((row, r) =>
              row.map((v, c) => {
                const isSel = r === sel[0] && c === sel[1];
                return (
                  <button
                    key={`${r}-${c}`}
                    type="button"
                    onClick={() => setSel([r, c])}
                    aria-label={`Output row ${r} col ${c}`}
                    style={{
                      width: 22,
                      height: 22,
                      padding: 0,
                      border: isSel ? '2px solid var(--accent)' : 'none',
                      ...actStyle(v, maxAbs),
                      fontSize: 9.5,
                      fontFamily: 'var(--mono)',
                      cursor: 'pointer',
                      boxSizing: 'border-box',
                    }}
                  >
                    {Math.abs(v) < 0.05 ? '0' : Number.isInteger(v) ? v : v.toFixed(1)}
                  </button>
                );
              }),
            )}
          </div>
        </div>
      </div>

      <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 10 }}>{presetDesc}</div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Padding, Stride & Output Geometry

export function PadStrideWidget() {
  const [h, setH] = useState<number>(8);
  const [k, setK] = useState<number>(3);
  const [pad, setPad] = useState<number>(1);
  const [stride, setStride] = useState<number>(2);

  const outH = outSize(h, k, pad, stride);
  const dropped = droppedBorderPixels(h, k, pad, stride);
  const paddedSize = h + 2 * pad;
  const shrink = useMemo(() => validStackShrinkage(12, 5, 3), []);

  return (
    <Widget
      title="Padding, stride, and why even sizes drop a border pixel"
      wide
      caption={
        <>
          Adjust input size <code>H</code>, kernel <code>K</code>, zero-padding <code>P</code>, and stride <code>S</code>. The dashed outer ring is zero-padding; blue = first window; orange = second window (shifted by <code>S</code>). Notice that with <code>H = 8, K = 3, P = 1, S = 2</code>, <code>(8 + 2 − 3) mod 2 = 1</code>, so the last padded column is silently dropped! Switch <code>H</code> to <code>9</code> (odd) and dropped pixels hit <strong>0</strong>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end', marginBottom: 12 }}>
        <div style={{ flex: '1 1 150px' }}>
          <Slider label="Input size" value={h} min={5} max={12} step={1} onChange={setH} format={(v) => `${v}×${v}`} />
        </div>
        <Segmented
          label="Kernel"
          value={k}
          onChange={setK}
          options={[
            { value: 1, label: '1×1' },
            { value: 3, label: '3×3' },
            { value: 5, label: '5×5' },
          ]}
        />
        <Segmented
          label="Padding"
          value={pad}
          onChange={setPad}
          options={[
            { value: 0, label: '0 (valid)' },
            { value: 1, label: '1' },
            { value: 2, label: '2' },
          ]}
        />
        <Segmented
          label="Stride"
          value={stride}
          onChange={setStride}
          options={[
            { value: 1, label: '1' },
            { value: 2, label: '2' },
            { value: 3, label: '3' },
          ]}
        />
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center' }}>
        {/* 1D-extended 2D top row visual showing padded strip + window 0 + window 1 + dropped tail */}
        <div style={{ flex: '1 1 260px' }}>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>
            Padded row ({paddedSize} cells: {pad} pad + {h} input + {pad} pad)
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2, marginBottom: 8 }}>
            {Array.from({ length: paddedSize }, (_, idx) => {
              const isPad = idx < pad || idx >= pad + h;
              const inWin0 = idx < k;
              const inWin1 = outH >= 2 && idx >= stride && idx < stride + k;
              const isDropped = idx >= (outH - 1) * stride + k;
              return (
                <div
                  key={idx}
                  style={{
                    width: 24,
                    height: 24,
                    borderRadius: 4,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 11,
                    fontFamily: 'var(--mono)',
                    border: isPad ? '1px dashed var(--text-muted)' : '1px solid var(--border)',
                    background: isDropped
                      ? 'color-mix(in srgb, var(--c-bad) 25%, var(--bg))'
                      : inWin0 && inWin1
                        ? 'color-mix(in srgb, var(--accent) 30%, var(--bg))'
                        : inWin0
                          ? 'color-mix(in srgb, var(--c-a) 28%, var(--bg))'
                          : inWin1
                            ? 'color-mix(in srgb, var(--c-b) 28%, var(--bg))'
                            : 'var(--bg-sunken)',
                    color: isDropped ? 'var(--c-bad)' : 'var(--text)',
                  }}
                >
                  {isPad ? '0' : idx - pad}
                </div>
              );
            })}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <span style={{ color: 'var(--c-a)' }}>■ Window 0</span>
            <span style={{ color: 'var(--c-b)' }}>■ Window 1 (shift +{stride})</span>
            {dropped > 0 && <span style={{ color: 'var(--c-bad)' }}>■ Dropped border ({dropped} px)</span>}
          </div>
        </div>

        <div className="readouts" style={{ flex: '1 1 250px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(115px, 1fr))', gap: 8, margin: 0 }}>
          <div className="readout">
            <div className="readout-label">Output shape</div>
            <div className="readout-value" style={{ color: 'var(--accent)' }}>{outH}×{outH}</div>
          </div>
          <div className="readout">
            <div className="readout-label">Dropped px/side</div>
            <div className="readout-value" style={{ color: dropped === 0 ? 'var(--c-good)' : 'var(--c-bad)' }}>{dropped}</div>
          </div>
          <div className="readout">
            <div className="readout-label">5 valid 3×3s on 12×12</div>
            <div className="readout-value" style={{ fontSize: 14 }}>→ {shrink.validH}×{shrink.validH} ({(100 * shrink.pixelLoss).toFixed(1)}% lost)</div>
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Translation Equivariance & Parameter Sharing

export function EquivarianceWidget() {
  const [cx, setCx] = useState<number>(3);
  const [cy, setCy] = useState<number>(3);

  const res = useMemo(() => verifyEquivariance(cx, cy), [cx, cy]);
  const params = useMemo(() => compareDenseVsConvParams(224, 224, 3, 32, 3), []);

  return (
    <Widget
      title="Translation equivariance & the 270-million-times parameter discount"
      wide
      caption={
        <>
          Move the cross anywhere on the <code>9×9</code> canvas. Because the exact same 9 kernel weights scan every location, shifting the input by <code>(Δr, Δc)</code> shifts the activation peak (<strong>+5</strong>) to the exact same coordinates <code>({res.peakR}, {res.peakC})</code>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 12 }}>
        <div style={{ flex: '1 1 180px' }}>
          <Slider label="Cross column" value={cx} min={1} max={7} step={1} onChange={setCx} format={(v) => `c = ${v}`} />
        </div>
        <div style={{ flex: '1 1 180px' }}>
          <Slider label="Cross row" value={cy} min={1} max={7} step={1} onChange={setCy} format={(v) => `r = ${v}`} />
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 18, alignItems: 'center' }}>
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Input cross at ({cy}, {cx})</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(9, 20px)', gap: 1, background: 'var(--border)', padding: 1, borderRadius: 6 }}>
            {res.img.map((row, r) =>
              row.map((v, c) => (
                <div key={`${r}-${c}`} style={{ width: 20, height: 20, background: v ? 'var(--c-a)' : 'var(--bg-sunken)' }} />
              )),
            )}
          </div>
        </div>
        <div style={{ fontSize: 20, color: 'var(--text-muted)' }}>→</div>
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Cross-detector feature map</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(9, 20px)', gap: 1, background: 'var(--border)', padding: 1, borderRadius: 6 }}>
            {res.feat.map((row, r) =>
              row.map((v, c) => {
                const isPeak = r === res.peakR && c === res.peakC;
                return (
                  <div
                    key={`${r}-${c}`}
                    style={{
                      width: 20,
                      height: 20,
                      ...actStyle(v, 5),
                      border: isPeak ? '2px solid var(--accent)' : 'none',
                      boxSizing: 'border-box',
                      fontSize: 9.5,
                      fontFamily: 'var(--mono)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    {v > 0 ? v : ''}
                  </div>
                );
              }),
            )}
          </div>
        </div>

        <div className="readouts" style={{ flex: '1 1 220px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8, margin: 0 }}>
          <div className="readout">
            <div className="readout-label">Peak location</div>
            <div className="readout-value" style={{ color: 'var(--c-good)' }}>({res.peakR}, {res.peakC}) = +{res.peakVal}</div>
          </div>
          <div className="readout">
            <div className="readout-label">Dense 224²×3 → 32ch</div>
            <div className="readout-value" style={{ color: 'var(--c-bad)', fontSize: 15 }}>241.7 B params</div>
          </div>
          <div className="readout">
            <div className="readout-label">3×3 Conv 3 → 32ch</div>
            <div className="readout-value" style={{ color: 'var(--c-good)', fontSize: 15 }}>{params.convParams} params</div>
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Design the Kernel

type ConvLabState = {
  vertW: number[];
  diagW: number[];
  k: 3 | 5;
  pad: 0 | 1 | 2;
  stride: 1 | 2;
  solved: Record<ConvBriefId, boolean>;
};

const CLAB_KEY = 'nl-conv-lab-v1';
const CLAB_DEFAULT: ConvLabState = {
  vertW: [0, 0, 0, -1, 0, 1, 0, 0, 0],
  diagW: [0, 0, 0, 0, 0, 0, 0, 0, 0],
  k: 3,
  pad: 0,
  stride: 2,
  solved: { vertical: false, diagonal: false, 'stack-shape': false },
};

function loadConvLab(): ConvLabState {
  try {
    const raw = localStorage.getItem(CLAB_KEY);
    if (!raw) return CLAB_DEFAULT;
    const p = JSON.parse(raw);
    return { ...CLAB_DEFAULT, ...p, solved: { ...CLAB_DEFAULT.solved, ...(p.solved ?? {}) } };
  } catch {
    return CLAB_DEFAULT;
  }
}

export function KernelDesignLab() {
  const [briefId, setBriefId] = useState<ConvBriefId>('vertical');
  const [st, setSt] = useState<ConvLabState>(loadConvLab);
  const [showOpt, setShowOpt] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(CLAB_KEY, JSON.stringify(st));
    } catch {
      /* ignore */
    }
  }, [st]);

  const vRes = useMemo(() => evalVerticalKernel(st.vertW), [st.vertW]);
  const dRes = useMemo(() => evalDiagonalKernel(st.diagW), [st.diagW]);
  const sRes = useMemo(() => evalStackShape(st.k, st.pad, st.stride), [st.k, st.pad, st.stride]);
  const exhaust = useMemo(() => convLabExhaustive(briefId), [briefId]);

  const curPass = briefId === 'vertical' ? vRes.pass : briefId === 'diagonal' ? dRes.pass : sRes.pass;
  useEffect(() => {
    if (curPass && !st.solved[briefId]) {
      setSt((p) => ({ ...p, solved: { ...p.solved, [briefId]: true } }));
    }
  }, [curPass, briefId, st.solved]);

  const cycleLabWeight = (which: 'vertW' | 'diagW', idx: number) => {
    setSt((p) => {
      const arr = [...p[which]];
      arr[idx] = arr[idx] === -1 ? 0 : arr[idx] === 0 ? 1 : -1;
      return { ...p, [which]: arr };
    });
  };

  const brief = CONV_LAB_BRIEFS.find((b) => b.id === briefId)!;
  const solvedCount = Object.values(st.solved).filter(Boolean).length;

  const renderTernaryEditor = (which: 'vertW' | 'diagW', arr: number[]) => (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4 }}>Click cells to cycle −1 → 0 → +1</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 52px)', gap: 5 }}>
        {arr.map((w, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => cycleLabWeight(which, idx)}
            style={{
              height: 44,
              borderRadius: 6,
              border: '1px solid var(--border)',
              ...actStyle(w, 1),
              fontFamily: 'var(--mono)',
              fontSize: 15,
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            {w > 0 ? '+1' : String(w)}
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <Widget
      title="Lab: Design the Kernel"
      wide
      caption={
        <>
          Craft real <code>3×3</code> ternary kernels (exhaustively verified against all <strong>19,683</strong> possible ternary kernels!) and engineer a zero-drop convolutional stack. Solved: <strong>{solvedCount}/3</strong>.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
        {CONV_LAB_BRIEFS.map((b, i) => (
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

      {briefId === 'vertical' && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center' }}>
          {renderTernaryEditor('vertW', st.vertW)}
          <div className="readouts" style={{ flex: '1 1 260px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Flat sum (= 0)</div>
              <div className="readout-value" style={{ color: vRes.flat === 0 ? 'var(--c-good)' : 'var(--c-bad)' }}>{vRes.flat}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Horiz leak (= 0)</div>
              <div className="readout-value" style={{ color: vRes.horiz === 0 ? 'var(--c-good)' : 'var(--c-bad)' }}>{vRes.horiz}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Vert peak (≥ +2)</div>
              <div className="readout-value" style={{ color: vRes.vertPeak >= 2 ? 'var(--c-good)' : 'var(--c-bad)' }}>+{vRes.vertPeak}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Non-zero weights</div>
              <div className="readout-value" style={{ color: vRes.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {vRes.nnz} {vRes.pass ? (vRes.nnz === 4 ? '(Opt!)' : 'PASS') : 'FAIL'}
              </div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'diagonal' && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center' }}>
          {renderTernaryEditor('diagW', st.diagW)}
          <div className="readouts" style={{ flex: '1 1 260px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Flat sum (= 0)</div>
              <div className="readout-value" style={{ color: dRes.flat === 0 ? 'var(--c-good)' : 'var(--c-bad)' }}>{dRes.flat}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Vert / Horiz (≤ 1)</div>
              <div className="readout-value" style={{ color: dRes.vert <= 1 && dRes.horiz <= 1 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {dRes.vert} / {dRes.horiz}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Diagonal (≥ +2)</div>
              <div className="readout-value" style={{ color: dRes.diag >= 2 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {dRes.diag >= 0 ? `+${dRes.diag}` : dRes.diag}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: dRes.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {dRes.pass ? (dRes.diag === 3 ? 'PASS (Opt +3!)' : 'PASS') : 'FAIL'}
              </div>
            </div>
          </div>
        </div>
      )}

      {briefId === 'stack-shape' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            <Segmented
              label="Kernel"
              value={st.k}
              onChange={(v) => setSt((p) => ({ ...p, k: v }))}
              options={[
                { value: 3, label: '3×3' },
                { value: 5, label: '5×5' },
              ]}
            />
            <Segmented
              label="Padding"
              value={st.pad}
              onChange={(v) => setSt((p) => ({ ...p, pad: v }))}
              options={[
                { value: 0, label: '0' },
                { value: 1, label: '1' },
                { value: 2, label: '2' },
              ]}
            />
            <Segmented
              label="Stride"
              value={st.stride}
              onChange={(v) => setSt((p) => ({ ...p, stride: v }))}
              options={[
                { value: 1, label: '1' },
                { value: 2, label: '2' },
              ]}
            />
          </div>
          <div className="readouts" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(125px, 1fr))', gap: 8, margin: 0 }}>
            <div className="readout">
              <div className="readout-label">Shape flow (→ 5)</div>
              <div className="readout-value" style={{ color: sRes.h2 === 5 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                17 → {sRes.h1} → {sRes.h2}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Dropped border</div>
              <div className="readout-value" style={{ color: sRes.drop1 === 0 && sRes.drop2 === 0 ? 'var(--c-good)' : 'var(--c-bad)' }}>
                L1: {sRes.drop1} · L2: {sRes.drop2}
              </div>
            </div>
            <div className="readout">
              <div className="readout-label">Weights / filter</div>
              <div className="readout-value">{sRes.weightsPerFilter}</div>
            </div>
            <div className="readout">
              <div className="readout-label">Status</div>
              <div className="readout-value" style={{ color: sRes.pass ? 'var(--c-good)' : 'var(--c-bad)' }}>
                {sRes.pass ? (sRes.weightsPerFilter === 18 ? 'PASS (Opt 18!)' : 'PASS') : 'FAIL'}
              </div>
            </div>
          </div>
        </div>
      )}

      <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid var(--border)', display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between' }}>
        <button type="button" className="btn" onClick={() => setShowOpt((v) => !v)}>
          {showOpt ? 'Hide exhaustive check' : `Reveal exhaustive search (${exhaust.total.toLocaleString()} configs)`}
        </button>
        {showOpt && (
          <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
            <strong>{exhaust.passing}/{exhaust.total.toLocaleString()}</strong> pass · {exhaust.summary}
          </div>
        )}
      </div>
    </Widget>
  );
}
