// authored-by: Gemini Antigravity
import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import {
  computeRosetta,
  INPUT_PRESETS,
  maxMatrixDiff,
  type ArchId,
  type InputPresetId,
} from './rosetta';

const ARCH_OPTIONS: { value: ArchId; label: string }[] = [
  { value: 'dense', label: 'Dense MLP' },
  { value: 'conv', label: '1D Conv' },
  { value: 'rnn', label: 'RNN' },
  { value: 'attention', label: 'Self-Attention' },
  { value: 'lora', label: 'LoRA' },
];

function cellBg(v: number, isZero: boolean): string {
  if (isZero) return 'var(--bg-sunken)';
  const mag = Math.min(1, Math.abs(v) / 1.15);
  const pct = Math.round(10 + mag * 42);
  return v >= 0
    ? `color-mix(in srgb, var(--c-a) ${pct}%, var(--bg-elev))`
    : `color-mix(in srgb, var(--c-b) ${pct}%, var(--bg-elev))`;
}

function archBadge(arch: ArchId, free: number, dense: number): string {
  if (arch === 'dense') return `${free} independent weights`;
  if (arch === 'conv') return `${free} shared weights (${dense / free}× fewer than dense)`;
  if (arch === 'rnn') return `${free} shared weights (${(dense / free).toFixed(1)}× fewer · causal)`;
  if (arch === 'attention') return `${free} Q/K weights · dynamic W(x)`;
  return `${free} trainable weights (${dense / free}× fewer than ${dense})`;
}

export function RosettaMatrix() {
  const [arch, setArch] = useState<ArchId>('conv');
  const [preset, setPreset] = useState<InputPresetId>('edge');
  const [x, setX] = useState<number[]>(() => [...INPUT_PRESETS.edge.x]);
  const [loraDeltaOnly, setLoraDeltaOnly] = useState(false);

  const selectPreset = (p: InputPresetId) => {
    setPreset(p);
    setX([...INPUT_PRESETS[p].x]);
  };

  const view = useMemo(() => computeRosetta(arch, x), [arch, x]);
  const baseView = useMemo(() => computeRosetta(arch, INPUT_PRESETS[preset].x), [arch, preset]);
  const wShift = useMemo(() => maxMatrixDiff(view.W, baseView.W), [view.W, baseView.W]);

  const shownMatrix = arch === 'lora' && loraDeltaOnly && view.lora ? view.lora.dW : view.W;
  const maxY = Math.max(1.2, ...view.y.map((v) => Math.abs(v)));

  return (
    <Widget
      title="One 6 × 6 matrix, five famous architectures"
      caption="Switch architectures to see the exact structural constraint each one places on W in y = W·x. Then drag any x slider below: watch which architecture changes the numbers inside W itself."
    >
      <div className="controls">
        <Segmented label="Architecture" value={arch} onChange={setArch} options={ARCH_OPTIONS} />
        <Segmented
          label="Input x"
          value={preset}
          onChange={selectPreset}
          options={[
            { value: 'edge', label: 'Step edge' },
            { value: 'spike', label: 'Single spike' },
            { value: 'wave', label: 'Alternating' },
          ]}
        />
        {arch === 'lora' && (
          <label className="ctrl-check">
            <input
              type="checkbox"
              checked={loraDeltaOnly}
              onChange={(e) => setLoraDeltaOnly(e.target.checked)}
            />
            <span>Show rank-1 ΔW = a·bᵀ only</span>
          </label>
        )}
      </div>

      {/* Main two-column layout: 6x6 matrix on left, x->y bars + stats on right */}
      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 290px', minWidth: 260 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, gap: 8, flexWrap: 'wrap' }}>
            <span className="mono" style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)' }}>
              {arch === 'lora' && loraDeltaOnly
                ? 'ΔW = a · bᵀ (6 × 6, rank 1)'
                : arch === 'attention'
                  ? 'W(x) = softmax(QKᵀ / √2) (6 × 6)'
                  : `Effective matrix W_eff (6 × 6)`}
            </span>
            <span
              className="mono"
              style={{
                fontSize: 11.5,
                padding: '2px 8px',
                borderRadius: 999,
                border: '1px solid var(--border)',
                background: wShift > 1e-4 ? 'color-mix(in srgb, var(--c-grad) 18%, var(--bg-elev))' : 'var(--bg-sunken)',
                color: wShift > 1e-4 ? 'var(--c-grad)' : 'var(--text-muted)',
                fontWeight: wShift > 1e-4 ? 700 : 500,
              }}
            >
              {view.inputDependent
                ? wShift > 1e-4
                  ? `W shifted by ${wShift.toFixed(2)}!`
                  : 'dynamic W(x) — drag x!'
                : 'W fixed (0.00 shift)'}
            </span>
          </div>

          {/* 6 x 6 Matrix grid */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(6, minmax(0, 1fr))',
              gap: 3,
              padding: 6,
              border: '1px solid var(--border)',
              borderRadius: 10,
              background: 'var(--bg-sunken)',
            }}
            role="grid"
            aria-label={`${view.title} 6 by 6 effective weight matrix`}
          >
            {shownMatrix.map((row, i) =>
              row.map((val, j) => {
                const isZero = Math.abs(val) < 1e-9;
                const tag = view.tags[i][j];
                return (
                  <div
                    key={`${i}-${j}`}
                    role="gridcell"
                    style={{
                      position: 'relative',
                      padding: '12px 2px 5px',
                      borderRadius: 6,
                      textAlign: 'center',
                      border: isZero ? '1px dashed var(--border)' : '1px solid var(--border-strong)',
                      background: cellBg(val, isZero),
                      color: isZero ? 'var(--text-faint)' : 'var(--text)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: 11.5,
                      fontVariantNumeric: 'tabular-nums',
                      lineHeight: 1.1,
                    }}
                  >
                    {tag && (
                      <span
                        style={{
                          position: 'absolute',
                          top: 1,
                          right: 3,
                          fontSize: 8.5,
                          color: 'var(--text-muted)',
                          fontWeight: 700,
                        }}
                      >
                        {tag}
                      </span>
                    )}
                    <span style={{ fontWeight: isZero ? 400 : 600 }}>
                      {isZero ? '0' : (val >= 0 ? '+' : '−') + Math.abs(val).toFixed(2)}
                    </span>
                  </div>
                );
              }),
            )}
          </div>

          {arch === 'lora' && view.lora && (
            <div
              className="mono"
              style={{
                marginTop: 8,
                padding: '6px 10px',
                borderRadius: 8,
                border: '1px solid var(--border)',
                background: 'var(--bg-sunken)',
                fontSize: 11.5,
                color: 'var(--text-muted)',
                display: 'grid',
                gap: 2,
              }}
            >
              <div>a (6×1): [{view.lora.a.map((v) => v.toFixed(1)).join(', ')}]</div>
              <div>bᵀ (1×6): [{view.lora.b.map((v) => v.toFixed(1)).join(', ')}]</div>
            </div>
          )}
        </div>

        {/* Right panel: interactive input x sliders and output y = W*x bars */}
        <div style={{ flex: '1 1 240px', minWidth: 230, display: 'grid', gap: 10 }}>
          <div style={{ padding: '10px 12px', border: '1px solid var(--border)', borderRadius: 10, background: 'var(--bg-sunken)' }}>
            <div className="mono" style={{ fontSize: 11.5, color: 'var(--accent-text)', fontWeight: 700, marginBottom: 4 }}>
              {archBadge(arch, view.freeParams, view.denseParams)}
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.45 }}>
              {view.shortRule}
            </div>
          </div>

          <div style={{ padding: '10px 12px', border: '1px solid var(--border)', borderRadius: 10, background: 'var(--bg-sunken)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, color: 'var(--text-muted)', marginBottom: 6 }} className="mono">
              <span>Nudge input x</span>
              <span>Output y = W·x</span>
            </div>
            <div style={{ display: 'grid', gap: 5 }}>
              {x.map((xi, idx) => {
                const yi = view.y[idx];
                const barPct = Math.min(100, (Math.abs(yi) / maxY) * 100);
                return (
                  <div
                    key={idx}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '22px minmax(70px, 1fr) 38px 16px minmax(64px, 1fr) 44px',
                      alignItems: 'center',
                      gap: 5,
                      fontSize: 11.5,
                    }}
                  >
                    <span className="mono" style={{ color: 'var(--text-muted)' }}>x{idx}</span>
                    <input
                      type="range"
                      min={-1}
                      max={1}
                      step={0.1}
                      value={xi}
                      aria-label={`Input x${idx}`}
                      onChange={(e) => {
                        const next = [...x];
                        next[idx] = Number(e.target.value);
                        setX(next);
                      }}
                      style={{ width: '100%' }}
                    />
                    <span className="mono" style={{ textAlign: 'right', color: 'var(--text-muted)' }}>
                      {xi >= 0 ? '+' : '−'}{Math.abs(xi).toFixed(1)}
                    </span>
                    <span className="mono" style={{ textAlign: 'center', color: 'var(--text-faint)' }}>→</span>
                    <div style={{ height: 8, background: 'var(--border)', borderRadius: 4, overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${barPct.toFixed(1)}%`,
                          height: '100%',
                          background: yi >= 0 ? 'var(--c-a)' : 'var(--c-b)',
                        }}
                      />
                    </div>
                    <span className="mono" style={{ textAlign: 'right', fontWeight: 600, color: 'var(--text)' }}>
                      {yi >= 0 ? '+' : '−'}{Math.abs(yi).toFixed(2)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      <p className="live-note" style={{ marginTop: 12 }}>
        {arch === 'dense' &&
          'Dense MLP: all 36 cells are free. No structure is assumed — every output can listen equally to every input, at the full 6 × 6 parameter cost.'}
        {arch === 'conv' &&
          '1D Convolution: the exact same 3-weight kernel [−1, 0, +1] repeats along three diagonals (12× fewer weights than dense). Notice how on “Step edge” it lights up at the jump and drops near zero on flat stretches.'}
        {arch === 'rnn' &&
          'Unrolled RNN: three time steps of width 2 stacked into one 6 × 6 map. Everything above the block diagonal is locked to 0 (future inputs cannot leak backwards), while the same 2 × 2 input (Wₓ) and memory (Wₕ) weights repeat down the column.'}
        {arch === 'attention' &&
          'Self-Attention: every row sums to 1.00, and the weights themselves are computed from the input x. Drag any x slider — unlike the other four architectures, the matrix cells change live because each token decides whom to attend to.'}
        {arch === 'lora' &&
          'LoRA (rank 1): instead of updating all 36 numbers of a pretrained base matrix W₀, train only two 6-number vectors a and bᵀ (12 parameters — 3× fewer) and add their outer product ΔW = a·bᵀ.'}
      </p>
    </Widget>
  );
}
