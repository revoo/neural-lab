// authored-by: Neon High
import { useEffect, useMemo, useState } from 'react';
import { LiveCode } from '../../components/LabJob';
import { peftLoraLines } from './realCode';
import { Widget } from '../../components/Widget';
import { AnswerReveal } from '../../components/AnswerReveal';
import { MachineLog, useMachineLog } from '../../components/MachineLog';
import { Segmented, Slider } from '../../components/controls';
import {
  LAB_ERR_GOAL,
  LAB_LAYERS,
  LAB_PARAM_BUDGET,
  LAB_RANKS,
  LORA_MODELS,
  SVD_DIM,
  computeSvd,
  estimateFineTuneMemory,
  evaluateLoraLab,
  makeWorkshopMatrix,
  reconstructRankR,
  runFullFtReference,
  runLoraExperiment,
  solveLoraLab,
  type LoraInitMode,
  type TargetScope,
  type TuneMethod,
} from './lora';

// Shared 12x12 signed matrix heatmap SVG (blue = positive, orange = negative)
function MatrixGrid({ M, maxAbs, title }: { M: number[][]; maxAbs: number; title: string }) {
  const d = M.length;
  const cell = 15;
  const gap = 1.5;
  const size = d * (cell + gap);

  return (
    <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 10, textAlign: 'center' }}>
      <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>{title}</div>
      <svg viewBox={`0 0 ${size} ${size}`} style={{ width: '100%', maxWidth: 196, height: 'auto', display: 'block', margin: '0 auto' }}>
        {M.map((row, i) =>
          row.map((v, j) => {
            const mag = Math.min(1, Math.abs(v) / Math.max(1e-6, maxAbs));
            const color = v >= 0 ? 'var(--c-a)' : 'var(--c-b)';
            return (
              <rect
                key={`${i}-${j}`}
                x={j * (cell + gap)}
                y={i * (cell + gap)}
                width={cell}
                height={cell}
                rx={2}
                fill={color}
                opacity={0.08 + 0.88 * mag}
              />
            );
          })
        )}
      </svg>
    </div>
  );
}

// ------------------------------------------------------------------ 1. Low-Rank SVD Workshop

export function LowRankWorkshop() {
  const [kind, setKind] = useState<'task-shift' | 'random-noise'>('task-shift');
  const [rank, setRank] = useState(2);

  const svd = useMemo(() => computeSvd(makeWorkshopMatrix(kind)), [kind]);
  const rec = useMemo(() => reconstructRankR(svd, rank), [svd, rank]);

  const maxAbs = useMemo(() => {
    let m = 0.01;
    for (const row of svd.matrix) for (const v of row) m = Math.max(m, Math.abs(v));
    return m;
  }, [svd]);

  const maxSv = svd.singularValues[0] || 1;

  return (
    <Widget
      title="Low-rank approximation: why structured weight updates compress"
      wide
      caption={
        <>
          Compare a structured 12×12 fine-tuning update <strong>ΔW</strong> against pure random noise. Drag rank <strong>r</strong> from 1 to 12: on the structured task shift, rank <strong>2</strong> (48 params vs 144) already captures <strong>95.2%</strong> of the matrix’s energy, whereas random noise refuses to compress.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <Segmented
          label="target matrix ΔW (12 × 12)"
          options={[
            { value: 'task-shift', label: 'Structured task shift' },
            { value: 'random-noise', label: 'Random noise' },
          ]}
          value={kind}
          onChange={(v) => setKind(v as 'task-shift' | 'random-noise')}
        />
        <div style={{ flex: '1 1 200px', maxWidth: 320 }}>
          <Slider
            label={`LoRA rank r = ${rank}`}
            min={1}
            max={SVD_DIM}
            step={1}
            value={rank}
            onChange={setRank}
            format={(r) => `r = ${r}`}
          />
        </div>
      </div>

      {/* Heatmaps + singular value spectrum */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(175px, 1fr))', gap: 10, marginBottom: 12 }}>
        <MatrixGrid M={svd.matrix} maxAbs={maxAbs} title="Target update ΔW (144 params)" />
        <MatrixGrid M={rec.approx} maxAbs={maxAbs} title={`Rank-${rank} B·A (${rec.loraParams} params)`} />

        {/* Singular value bars */}
        <div style={{ background: 'var(--bg-sunken)', border: '1px solid var(--border)', borderRadius: 8, padding: 10, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>Singular values σ₁ … σ₁₂</div>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 4, height: 120, paddingTop: 8 }}>
            {svd.singularValues.map((sv, idx) => {
              const kept = idx < rank;
              const hPct = Math.max(4, (sv / maxSv) * 100);
              return (
                <div
                  key={idx}
                  title={`σ${idx + 1} = ${sv.toFixed(2)} (${kept ? 'kept' : 'dropped'})`}
                  style={{ flex: 1, height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', alignItems: 'center' }}
                >
                  <div
                    style={{
                      width: '100%',
                      height: `${hPct}%`,
                      borderRadius: '3px 3px 0 0',
                      background: kept ? 'var(--accent)' : 'var(--border-strong)',
                      opacity: kept ? 0.92 : 0.45,
                      transition: 'background 140ms ease',
                    }}
                  />
                  <span className="mono" style={{ fontSize: 9, color: kept ? 'var(--accent)' : 'var(--text-faint)', marginTop: 3 }}>
                    {idx + 1}
                  </span>
                </div>
              );
            })}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 6 }}>
            Accent bars = kept by rank {rank}
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(145px, 1fr))', gap: 8, marginBottom: 10 }}>
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>ENERGY CAPTURED</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: rec.energyPct >= 95 ? 'var(--c-good)' : 'var(--accent)' }}>
            {rec.energyPct.toFixed(1)}%
          </div>
        </div>
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>RELATIVE ERROR</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: rec.relError <= 0.25 ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {(rec.relError * 100).toFixed(1)}%
          </div>
        </div>
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>PARAMETERS (B + A)</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: rec.loraParams > rec.fullParams ? 'var(--c-bad)' : 'var(--c-a)' }}>
            {rec.loraParams} / {rec.fullParams}
          </div>
        </div>
      </div>

      <div className="live-note">
        {rec.loraParams > rec.fullParams ? (
          <>
            <strong>Careful!</strong> At rank {rank} on a 12×12 matrix, <code>B</code> and <code>A</code> actually hold{' '}
            <strong>{rec.loraParams}</strong> numbers — MORE than the 144 entries of the full matrix! LoRA saves parameters only when{' '}
            <code>r &lt; d / 2</code> (e.g. <code>r = 16</code> vs <code>d = 4,096</code>).
          </>
        ) : kind === 'task-shift' ? (
          <>
            <strong>Low intrinsic rank:</strong> just {rank} rank-1 sheets ({rec.loraParams} params, {((rec.loraParams / rec.fullParams) * 100).toFixed(0)}% of full) capture{' '}
            <strong>{rec.energyPct.toFixed(1)}%</strong> of the update’s energy.
          </>
        ) : (
          <>
            <strong>Random noise has flat singular values:</strong> rank {rank} only recovers <strong>{rec.energyPct.toFixed(1)}%</strong> of the energy. LoRA works on real LLMs specifically because fine-tuning updates live in a low-dimensional subspace.
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Exact LoRA / QLoRA Fine-Tuning Memory Calculator

const RANK_OPTS = [4, 8, 16, 32, 64];

export function LoraMemoryCalc() {
  const [modelIdx, setModelIdx] = useState(0); // Llama-3 8B
  const [method, setMethod] = useState<TuneMethod>('lora');
  const [rankIdx, setRankIdx] = useState(2); // r = 16
  const [scope, setScope] = useState<TargetScope>('all');

  const model = LORA_MODELS[modelIdx];
  const rank = RANK_OPTS[rankIdx];
  const mem = useMemo(() => estimateFineTuneMemory(model, method, rank, scope), [model, method, rank, scope]);
  const fullMem = useMemo(() => estimateFineTuneMemory(model, 'full', rank, scope), [model, rank, scope]);

  const maxBar = Math.max(80, mem.totalGiB * 1.05);
  const wPct = (mem.baseWeightsGiB / maxBar) * 100;
  const aPct = (mem.adapterWeightsGiB / maxBar) * 100;
  const gPct = (mem.gradsGiB / maxBar) * 100;
  const adamPct = (mem.adamStatesGiB / maxBar) * 100;
  const h100Pct = (80 / maxBar) * 100;

  const fitsLabel =
    mem.totalGiB <= 12
      ? 'Fits on a 12 GiB laptop / desktop GPU'
      : mem.totalGiB <= 24
        ? 'Fits on a single 24 GiB RTX 4090'
        : mem.totalGiB <= 80
          ? 'Fits on a single 80 GiB A100 / H100'
          : `Needs at least ${Math.ceil(mem.totalGiB / 80)} × 80 GiB H100s (weights + Adam alone!)`;

  return (
    <Widget
      title="Fine-tuning VRAM calculator: Full FT vs LoRA vs QLoRA"
      wide
      caption={
        <>
          Exact mixed-precision AdamW memory (16 bytes per trainable parameter: BF16 weights + FP32 master copy + grads + Adam <code>m</code> and <code>v</code>; excludes batch activations). Toggle between <strong>Full FT</strong>, <strong>LoRA</strong> and <strong>QLoRA</strong> to watch Adam’s giant red block vanish.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <Segmented
          label="model"
          options={LORA_MODELS.map((m, i) => ({ value: i, label: m.shortName }))}
          value={modelIdx}
          onChange={setModelIdx}
        />
        <Segmented
          label="fine-tuning method"
          options={[
            { value: 'full', label: 'Full FT' },
            { value: 'lora', label: 'LoRA (BF16)' },
            { value: 'qlora', label: 'QLoRA (4-bit)' },
          ]}
          value={method}
          onChange={(v) => setMethod(v as TuneMethod)}
        />
      </div>

      {method !== 'full' && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <Segmented
            label="adapter rank r"
            options={RANK_OPTS.map((r, i) => ({ value: i, label: `r = ${r}` }))}
            value={rankIdx}
            onChange={setRankIdx}
          />
          <Segmented
            label="target layers"
            options={[
              { value: 'qv', label: 'Q, V only' },
              { value: 'attn', label: 'All Attn' },
              { value: 'all', label: 'All Linear' },
            ]}
            value={scope}
            onChange={(v) => setScope(v as TargetScope)}
          />
        </div>
      )}

      {/* Stat cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(135px, 1fr))', gap: 8, marginBottom: 12 }}>
        <div style={{ padding: '9px 11px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>TRAINABLE PARAMS</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--accent)' }}>
            {mem.trainableParams >= 1e9 ? `${(mem.trainableParams / 1e9).toFixed(2)}B` : `${(mem.trainableParams / 1e6).toFixed(1)}M`}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>{mem.trainablePct.toFixed(2)}% of model</div>
        </div>
        <div style={{ padding: '9px 11px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>TOTAL VRAM</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: mem.totalGiB > 80 ? 'var(--c-bad)' : 'var(--c-good)' }}>
            {mem.totalGiB.toFixed(2)} GiB
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>
            {(fullMem.totalGiB / mem.totalGiB).toFixed(1)}× less than Full FT
          </div>
        </div>
        <div style={{ padding: '9px 11px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>ADAM STATES (m, v)</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: mem.adamStatesGiB > 10 ? 'var(--c-bad)' : 'var(--c-a)' }}>
            {mem.adamStatesGiB.toFixed(2)} GiB
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>8 B / trainable param</div>
        </div>
        <div style={{ padding: '9px 11px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>ADAPTER FILE SIZE</div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--text)' }}>
            {mem.checkpointMB >= 1024 ? `${(mem.checkpointMB / 1024).toFixed(1)} GiB` : `${mem.checkpointMB.toFixed(0)} MiB`}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>per fine-tuned task</div>
        </div>
      </div>

      {/* Stacked bar */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', fontSize: 11, marginBottom: 4, gap: 6 }}>
          <span className="mono" style={{ color: 'var(--text-muted)' }}>
            <span style={{ color: 'var(--c-a)' }}>■</span> Base {mem.baseWeightsGiB.toFixed(1)}G ·{' '}
            <span style={{ color: 'var(--accent)' }}>■</span> Master/Adpt {mem.adapterWeightsGiB.toFixed(1)}G ·{' '}
            <span style={{ color: 'var(--c-b)' }}>■</span> Grads {mem.gradsGiB.toFixed(1)}G ·{' '}
            <span style={{ color: 'var(--c-bad)' }}>■</span> Adam {mem.adamStatesGiB.toFixed(1)}G
          </span>
          <span className="mono" style={{ color: 'var(--text-faint)' }}>80 GiB H100</span>
        </div>
        <div style={{ position: 'relative', height: 28, borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)', overflow: 'hidden', display: 'flex' }}>
          <div style={{ width: `${wPct}%`, background: 'var(--c-a)', opacity: 0.85 }} />
          <div style={{ width: `${aPct}%`, background: 'var(--accent)', opacity: 0.9 }} />
          <div style={{ width: `${gPct}%`, background: 'var(--c-b)', opacity: 0.9 }} />
          <div style={{ width: `${adamPct}%`, background: 'var(--c-bad)', opacity: 0.9 }} />
          <div style={{ position: 'absolute', left: `${Math.min(99.5, h100Pct)}%`, top: 0, bottom: 0, width: 2, background: 'var(--text)', opacity: 0.75 }} />
        </div>
      </div>

      <div className="live-note">
        <strong>{fitsLabel}.</strong>{' '}
        {method === 'full'
          ? `Notice that in Full FT, the base BF16 weights are only ${mem.baseWeightsGiB.toFixed(1)} GiB — the other ${(mem.totalGiB - mem.baseWeightsGiB).toFixed(1)} GiB (88% of memory!) is FP32 master weights, gradients and Adam states.`
          : method === 'lora'
            ? `Freezing the base model shrank Adam states from ${fullMem.adamStatesGiB.toFixed(1)} GiB down to ${mem.adamStatesGiB.toFixed(2)} GiB — now ${((mem.baseWeightsGiB / mem.totalGiB) * 100).toFixed(0)}% of remaining memory is just holding the frozen BF16 weights. Switch to QLoRA to compress those too!`
            : `QLoRA stores the frozen base model in 4-bit (${mem.baseWeightsGiB.toFixed(1)} GiB) and keeps tiny BF16/FP32 adapters (${(mem.totalGiB - mem.baseWeightsGiB).toFixed(2)} GiB), cutting total fine-tuning memory ${(fullMem.totalGiB / mem.totalGiB).toFixed(0)}× vs Full FT!`}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Live LoRA Zero-Init Symmetry Experiment

export function LiveLoraTrainer() {
  const [initMode, setInitMode] = useState<LoraInitMode>('lora-zero-b');
  const [rank, setRank] = useState(3);

  const ref = useMemo(() => runFullFtReference(80), []);
  const res = useMemo(() => runLoraExperiment(rank, initMode, 80), [rank, initMode]);

  const W = 520;
  const H = 180;
  const pad = { l: 44, r: 16, t: 16, b: 28 };
  const nPts = res.losses.length;
  const yMax = Math.max(0.95, res.step0Loss * 1.08);
  const sx = (i: number) => pad.l + (i / Math.max(1, nPts - 1)) * (W - pad.l - pad.r);
  const sy = (v: number) => H - pad.b - (Math.min(yMax, Math.max(0, v)) / yMax) * (H - pad.t - pad.b);

  const loraPts = res.losses.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(' ');
  const fullPts = ref.losses.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(' ');
  const preY = sy(res.pretrainedLoss);

  const statusColor =
    initMode === 'both-zero' ? 'var(--c-bad)' : initMode === 'both-random' ? 'var(--c-b)' : 'var(--c-good)';

  return (
    <Widget
      title="Why B = 0 and A ~ N? Live LoRA adaptation & symmetry breaking"
      wide
      caption={
        <>
          A frozen 8×8 pretrained layer <code>W₀</code> adapted to an exact rank-3 task shift over 80 SGD steps. Switch between <strong>B = 0, A ~ N</strong> (LoRA’s trick), <strong>Both zero</strong> (permanently dead!), and <strong>Both random</strong> (corrupts the pretrained model at step 0!).
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <Segmented
          label="initialisation rule"
          options={[
            { value: 'lora-zero-b', label: 'B = 0, A ~ N (LoRA)' },
            { value: 'both-zero', label: 'Both zero (A=0, B=0)' },
            { value: 'both-random', label: 'Both random' },
          ]}
          value={initMode}
          onChange={(v) => setInitMode(v as LoraInitMode)}
        />
        <Segmented
          label="rank r"
          options={[1, 2, 3, 4].map((r) => ({ value: r, label: `r = ${r}` }))}
          value={rank}
          onChange={setRank}
        />
      </div>

      {/* Loss chart */}
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block', background: 'var(--bg-sunken)', borderRadius: 8, border: '1px solid var(--border)', marginBottom: 12 }}>
        {[0, 0.25, 0.5, 0.75].map((t) => (
          <g key={t}>
            <line x1={pad.l} y1={sy(t)} x2={W - pad.r} y2={sy(t)} stroke="var(--border)" strokeDasharray="3 3" />
            <text x={pad.l - 6} y={sy(t) + 3} textAnchor="end" fill="var(--text-faint)" fontSize={10} fontFamily="var(--font-mono)">
              {t.toFixed(2)}
            </text>
          </g>
        ))}

        {/* Pretrained baseline line */}
        <line x1={pad.l} y1={preY} x2={W - pad.r} y2={preY} stroke="var(--text-faint)" strokeDasharray="5 4" />
        <text x={W - pad.r - 4} y={preY - 5} textAnchor="end" fill="var(--text-faint)" fontSize={10} fontFamily="var(--font-mono)">
          pretrained W₀ ({res.pretrainedLoss.toFixed(3)})
        </text>

        {[0, 40, 80].map((st, idx) => (
          <text
            key={st}
            x={sx(idx * ((nPts - 1) / 2))}
            y={H - 8}
            textAnchor={idx === 0 ? 'start' : idx === 2 ? 'end' : 'middle'}
            fill="var(--text-faint)"
            fontSize={10}
            fontFamily="var(--font-mono)"
          >
            step {st}
          </text>
        ))}

        {/* Full FT reference */}
        <polyline fill="none" stroke="var(--c-a)" strokeWidth={1.8} strokeDasharray="4 3" points={fullPts} opacity={0.8} />
        {/* LoRA curve */}
        <polyline fill="none" stroke={statusColor} strokeWidth={2.6} points={loraPts} />
        <circle cx={sx(0)} cy={sy(res.step0Loss)} r={4.5} fill={statusColor} />

        {/* Legend */}
        <text x={pad.l + 8} y={pad.t + 8} fill="var(--c-a)" fontSize={10} fontFamily="var(--font-mono)">
          - - Full FT (64 params)
        </text>
        <text x={pad.l + 168} y={pad.t + 8} fill={statusColor} fontSize={10} fontWeight={700} fontFamily="var(--font-mono)">
          ── LoRA r={rank} ({res.trainableParams} params)
        </text>
      </svg>

      {/* Step-0 diagnostics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 8, marginBottom: 10 }}>
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>STEP-0 LOSS</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: res.step0Loss > res.pretrainedLoss + 0.01 ? 'var(--c-bad)' : 'var(--c-good)' }}>
            {res.step0Loss.toFixed(4)}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>W₀ baseline: {res.pretrainedLoss.toFixed(4)}</div>
        </div>
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>STEP-0 ‖∇A‖</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: 'var(--text)' }}>
            {res.step0GradANorm.toFixed(4)}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>∝ Bᵀ (e xᵀ)</div>
        </div>
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>STEP-0 ‖∇B‖</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: res.step0GradBNorm === 0 ? 'var(--c-bad)' : 'var(--c-good)' }}>
            {res.step0GradBNorm.toFixed(4)}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>∝ e (A x)ᵀ</div>
        </div>
        <div style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>FINAL LOSS (80 STEPS)</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: res.finalLoss < 0.03 ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {res.finalLoss.toFixed(4)}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>Full FT: {ref.finalLoss.toFixed(4)}</div>
        </div>
      </div>

      <div className="live-note">
        {initMode === 'lora-zero-b' ? (
          <>
            <strong>Exact zero-init symmetry breaking:</strong> because <code>B = 0</code>, the product <code>B·A = 0</code> leaves the pretrained model untouched at step 0 ({res.step0Loss.toFixed(4)}). Meanwhile, because <code>A ≠ 0</code>, <code>∇B</code> is immediately non-zero ({res.step0GradBNorm.toFixed(4)}) — moving <code>B</code> on step 1, which wakes up <code>∇A</code> on step 2!
          </>
        ) : initMode === 'both-zero' ? (
          <>
            <strong>Permanently frozen!</strong> With both <code>A = 0</code> and <code>B = 0</code>, both <code>∇A = 0</code> and <code>∇B = 0</code> — every step adds zero, so loss stays stuck at {res.finalLoss.toFixed(4)} forever.
          </>
        ) : (
          <>
            <strong>Pretrained model corrupted at step 0!</strong> Random <code>B·A</code> injects noise into every layer before training begins, spiking step-0 loss from {res.pretrainedLoss.toFixed(3)} to <strong>{res.step0Loss.toFixed(3)}</strong> (on a real 70B LLM, this scrambles carefully learned representations).
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Adapter Allocation Budget

const LAB_KEY = 'nl-lora-lab-v1';


/** One colour scale for all three layers, so a small shift looks small. */
const LAB_SHIFT_MAX = Math.max(...LAB_LAYERS.flatMap((l) => l.svd.matrix.flat().map(Math.abs)));

/** An 8×8 matrix as coloured cells: blue for positive, red for negative, strength by size (shared scale). */
function MiniMatrix({ M, max, label }: { M: number[][]; max: number; label: string }) {
  const n = M.length;
  const c = 7;
  return (
    <div style={{ textAlign: 'center' }}>
      <svg viewBox={`0 0 ${n * c} ${n * c}`} width={n * c} height={n * c} role="img" aria-label={label} style={{ display: 'block', border: '1px solid var(--border)' }}>
        {M.map((row, i) =>
          row.map((v, j) => (
            <rect key={`${i}-${j}`} x={j * c} y={i * c} width={c} height={c} fill={v >= 0 ? 'var(--c-pos)' : 'var(--c-neg)'} opacity={Math.min(1, Math.abs(v) / max)} />
          )),
        )}
      </svg>
      <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{label}</div>
    </div>
  );
}

export function LoraBudgetLab() {
  const [ranks, setRanks] = useState<[number, number, number]>(() => {
    try {
      const raw = localStorage.getItem(LAB_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed) && parsed.length === 3) return parsed as [number, number, number];
      }
    } catch {
      /* ignore */
    }
    return [1, 1, 1];
  });
  const [solvedOnce, setSolvedOnce] = useState(() => {
    try {
      return localStorage.getItem(LAB_KEY + ':solved') === '1';
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(LAB_KEY, JSON.stringify(ranks));
    } catch {
      /* ignore */
    }
  }, [ranks]);

  const ev = useMemo(() => evaluateLoraLab(ranks), [ranks]);
  const sol = useMemo(() => solveLoraLab(), []);
  const hitOpt = ev.passes && ev.avgMse <= sol.best.avgMse + 1e-6;

  useEffect(() => {
    if (ev.passes && !solvedOnce) {
      setSolvedOnce(true);
      try {
        localStorage.setItem(LAB_KEY + ':solved', '1');
      } catch {
        /* ignore */
      }
    }
  }, [ev.passes, solvedOnce]);

  const log = useMachineLog();
  const setLayerRank = (idx: number, r: number) => {
    const next: [number, number, number] = [...ranks] as [number, number, number];
    next[idx] = r;
    setRanks(next);
    const after = evaluateLoraLab(next);
    log.push(
      `${LAB_LAYERS[idx].name.split(' · ')[0]}: r ${ranks[idx]} → ${r} · layer MSE ${ev.layerMses[idx].toFixed(5)} → ${after.layerMses[idx].toFixed(5)} · ${after.totalParams} of ${LAB_PARAM_BUDGET} params · avg MSE ${after.avgMse.toFixed(5)}${after.passes ? ' · passes' : ''}`,
      after.passes ? 'good' : after.fitsBudget ? 'info' : 'bad',
    );
  };

  return (
    <Widget
      title="Lab: Spend your LoRA rank budget where the shift actually lives"
      wide
      caption={
        <>
          Three 8×8 layers of a pretrained model face a new domain shift. Assign a rank <code>r ∈ 0..4</code> to each layer to stay within <strong>≤ {LAB_PARAM_BUDGET} trainable parameters</strong> (vs 192 full FT) and drive average residual MSE to <strong>≤ {LAB_ERR_GOAL.toFixed(3)}</strong>. Inspect each layer’s singular-value bars to see where to spend your ranks!
        </>
      }
    >
      {/* Top status bar */}
      <div
        className="mono"
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 12,
          padding: '10px 12px',
          borderRadius: 8,
          background: 'var(--bg-sunken)',
          border: '1px solid var(--border)',
          marginBottom: 12,
          fontSize: 13,
        }}
      >
        <span style={{ color: ev.fitsBudget ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 700 }}>
          {ev.fitsBudget ? '✓' : '✗'} Trainable params: {ev.totalParams} / {LAB_PARAM_BUDGET}
        </span>
        <span style={{ color: ev.meetsErr ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 700 }}>
          {ev.meetsErr ? '✓' : '✗'} Avg MSE: {ev.avgMse.toFixed(5)} (goal ≤ {LAB_ERR_GOAL.toFixed(3)})
        </span>
      </div>

      {/* Layer cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 12 }}>
        {LAB_LAYERS.map((layer, idx) => {
          const r = ranks[idx];
          const maxSv = layer.svd.singularValues[0] || 1;
          return (
            <div key={layer.id} style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'baseline', gap: 6, marginBottom: 6 }}>
                <div>
                  <strong style={{ fontSize: 13 }}>{layer.name}</strong>{' '}
                  <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>— {layer.role}</span>
                </div>
                <span className="mono" style={{ fontSize: 12, color: ev.layerMses[idx] > 0.006 ? 'var(--c-bad)' : 'var(--c-good)' }}>
                  {r * 16} params · MSE {ev.layerMses[idx].toFixed(5)}
                </span>
              </div>

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between' }}>
                <div className="btn-row">
                  {LAB_RANKS.map((cand) => (
                    <button
                      key={cand}
                      className={r === cand ? 'btn primary' : 'btn'}
                      onClick={() => setLayerRank(idx, cand)}
                    >
                      r = {cand} ({cand * 16}p)
                    </button>
                  ))}
                </div>

                {(() => {
                  const rec = reconstructRankR(layer.svd, r);
                  const missed = layer.svd.matrix.map((row, i) => row.map((v, j) => v - rec.approx[i][j]));
                  const max = LAB_SHIFT_MAX;
                  return (
                    <div style={{ display: 'flex', gap: 6 }} title="The layer's shift ΔW split into the part rank r captures and the part it misses (one colour scale for all layers)">
                      <MiniMatrix M={rec.approx} max={max} label="captured" />
                      <MiniMatrix M={missed} max={max} label="missed" />
                    </div>
                  );
                })()}
                {/* Mini singular-value spectrum (first 5 values) */}
                <div style={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: 26 }} title="Singular values σ₁..σ₅ of this layer’s shift">
                  {layer.svd.singularValues.slice(0, 5).map((sv, k) => (
                    <div
                      key={k}
                      style={{
                        width: 10,
                        height: `${Math.max(10, (sv / Math.max(2.2, maxSv)) * 100)}%`,
                        borderRadius: '2px 2px 0 0',
                        background: k < r ? 'var(--c-good)' : 'var(--c-b)',
                        opacity: k < r ? 0.9 : 0.65,
                      }}
                    />
                  ))}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Quick presets */}
      <div className="btn-row" style={{ marginBottom: 10 }}>
        <button className="btn" onClick={() => setRanks([1, 1, 1])}>
          Uniform r=1 (48p)
        </button>
        <button className="btn" onClick={() => setRanks([2, 2, 2])}>
          Uniform r=2 (96p)
        </button>
      </div>

      <LiveCode code={peftLoraLines(ranks)} note="Hugging Face peft" />

      <div
        className="live-note"
        style={{
          borderColor: ev.passes ? 'var(--c-good)' : undefined,
          background: ev.passes ? 'color-mix(in srgb, var(--c-good) 10%, var(--bg-sunken))' : undefined,
        }}
      >
        {ev.passes ? (
          <>
            <strong style={{ color: 'var(--c-good)' }}>
              {hitOpt ? 'Solved — and you found the exact exhaustive optimum!' : 'Budget solved!'}
            </strong>{' '}
            ({sol.passingCount} of {sol.totalConfigs} allocations pass.)
            {!hitOpt && (
              <>
                {' '}Optional goal: drive Avg MSE all the way down to <strong>{sol.best.avgMse.toFixed(5)}</strong>!
              </>
            )}
          </>
        ) : !ev.fitsBudget ? (
          <>
            <strong>Over parameter budget ({ev.totalParams} &gt; {LAB_PARAM_BUDGET}).</strong> Check the mini singular-value bars on the right — one layer barely shifts at all and can drop to <code>r = 0</code>!
          </>
        ) : (
          <>
            <strong>Within budget, but Avg MSE ({ev.avgMse.toFixed(5)}) is too high.</strong> Look at which layer still has tall orange (uncaptured) singular values and give it more rank!
          </>
        )}
      </div>

      <AnswerReveal solved={solvedOnce || ev.passes} stuckKey="lora-lab" label="the exhaustive optimum">
        <div className="mono" style={{ marginTop: 8, padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)', fontSize: 12 }}>
          Exhaustive check of all {sol.totalConfigs} allocations ({sol.passingCount} pass): optimum is{' '}
          <strong>r = [{sol.best.ranks.join(', ')}]</strong> ({sol.best.totalParams} params) → Avg MSE{' '}
          <strong>{sol.best.avgMse.toFixed(5)}</strong> (16× lower than uniform [1,1,1]’s {sol.uniformR1.avgMse.toFixed(5)}!).
        </div>
      </AnswerReveal>
      <MachineLog lines={log.lines} onClear={log.clear} empty="Change a layer's rank: the layer's error and the budget are logged here." />
    </Widget>
  );
}
