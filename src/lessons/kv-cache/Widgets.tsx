// authored-by: Neon High
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import {
  DEMO_TOKENS,
  GPU_BY_ID,
  KIB,
  KV_FORMATS,
  MODEL_BY_ID,
  MODEL_SPECS,
  TETRIS_BATCHES,
  TETRIS_BRIEFS,
  TETRIS_CTXS,
  WEIGHT_FORMATS,
  evaluateRoofline,
  evaluateTetris,
  generationWorkTotals,
  kvBytesPerToken,
  kvCacheGiB,
  ridgePoint,
  runKvEquivalenceDemo,
  solveTetrisBrief,
  weightsGiB,
  type TetrisConfig,
} from './kv';

const fmtCtx = (n: number) => (n >= 1024 ? `${Math.round(n / 1024)}k` : `${n}`);
const fmtNum = (n: number) => Math.round(n).toLocaleString('en-GB');

// ------------------------------------------------------------------ 1. KV Cache Stepper

export function KvCacheStepper() {
  const demo = useMemo(() => runKvEquivalenceDemo(), []);
  const [stepIdx, setStepIdx] = useState(4); // default step 5 ("jumps")
  const [scalePreset, setScalePreset] = useState<'toy' | 'chat' | 'doc'>('chat');

  const cur = demo.steps[stepIdx];
  const n = cur.step;

  const scaleInfo = useMemo(() => {
    if (scalePreset === 'toy') return { label: '6-token toy', ...generationWorkTotals(1, 6) };
    if (scalePreset === 'chat') return { label: '512 prompt + 256 gen', ...generationWorkTotals(512, 256) };
    return { label: '2,048 prompt + 512 gen', ...generationWorkTotals(2048, 512) };
  }, [scalePreset]);

  const cellSize = 26;

  return (
    <Widget
      title="Recomputing everything vs caching K and V"
      wide
      caption={
        <>
          Step forward token by token. Without a KV cache, every new token forces the model to re-run earlier rows (red) only to throw them away. With a KV cache, earlier keys and values stay frozen in memory (blue), and only the single new row (accent) is computed — with <strong>zero</strong> mathematical difference.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div className="btn-row">
          <button className="btn" disabled={stepIdx <= 0} onClick={() => setStepIdx((i) => Math.max(0, i - 1))}>
            ← Prev token
          </button>
          <button className="btn primary" disabled={stepIdx >= DEMO_TOKENS.length - 1} onClick={() => setStepIdx((i) => Math.min(DEMO_TOKENS.length - 1, i + 1))}>
            Next token →
          </button>
          <button className="btn" onClick={() => setStepIdx(0)}>
            Reset
          </button>
        </div>
        <div className="mono" style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          Step <strong style={{ color: 'var(--text)' }}>{n} / {DEMO_TOKENS.length}</strong> · new token:{' '}
          <strong style={{ color: 'var(--accent)' }}>“{cur.token}”</strong>
        </div>
      </div>

      {/* Token strip */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 14 }}>
        {DEMO_TOKENS.map((tok, i) => {
          const active = i === stepIdx;
          const past = i < stepIdx;
          return (
            <button
              key={tok}
              onClick={() => setStepIdx(i)}
              className="mono"
              style={{
                padding: '4px 9px',
                borderRadius: 6,
                fontSize: 12,
                cursor: 'pointer',
                border: `1px solid ${active ? 'var(--accent)' : past ? 'var(--c-a)' : 'var(--border)'}`,
                background: active
                  ? 'color-mix(in srgb, var(--accent) 20%, transparent)'
                  : past
                    ? 'color-mix(in srgb, var(--c-a) 12%, transparent)'
                    : 'var(--bg-sunken)',
                color: active ? 'var(--accent)' : past ? 'var(--text)' : 'var(--text-faint)',
                fontWeight: active ? 700 : 500,
              }}
            >
              {i + 1}. {tok}
            </button>
          );
        })}
      </div>

      {/* Side-by-side causal attention work grids */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: 12, marginBottom: 12 }}>
        {/* Uncached */}
        <div style={{ padding: 12, borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
            <strong style={{ fontSize: 13, color: 'var(--c-bad)' }}>Without KV cache</strong>
            <span className="mono" style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              {cur.uncachedDotProducts} pairs this step
            </span>
          </div>
          <div style={{ display: 'inline-grid', gridTemplateColumns: `36px repeat(${DEMO_TOKENS.length}, ${cellSize}px)`, gap: 3, alignItems: 'center' }}>
            <div />
            {DEMO_TOKENS.map((t, j) => (
              <div key={t} className="mono" style={{ fontSize: 10, textAlign: 'center', color: j < n ? 'var(--text-muted)' : 'var(--text-faint)' }}>
                K{j + 1}
              </div>
            ))}
            {DEMO_TOKENS.map((t, r) => (
              <div key={t} style={{ display: 'contents' }}>
                <div className="mono" style={{ fontSize: 10, color: r < n ? 'var(--text-muted)' : 'var(--text-faint)' }}>
                  Q{r + 1}
                </div>
                {DEMO_TOKENS.map((_, c) => {
                  const causal = c <= r;
                  const computed = r < n && causal;
                  const wasted = computed && r < n - 1;
                  const needed = computed && r === n - 1;
                  return (
                    <div
                      key={c}
                      title={wasted ? 'Recomputed and discarded!' : needed ? 'Needed for next token' : 'Masked / future'}
                      style={{
                        width: cellSize,
                        height: cellSize,
                        borderRadius: 4,
                        border: `1px solid ${needed ? 'var(--accent)' : wasted ? 'var(--c-bad)' : 'var(--border)'}`,
                        background: needed
                          ? 'color-mix(in srgb, var(--accent) 28%, transparent)'
                          : wasted
                            ? 'color-mix(in srgb, var(--c-bad) 22%, transparent)'
                            : 'transparent',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 11,
                        color: needed ? 'var(--accent)' : wasted ? 'var(--c-bad)' : 'var(--text-faint)',
                      }}
                    >
                      {wasted ? '×' : needed ? '●' : ''}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
          <div className="mono" style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
            Cumulative: <strong style={{ color: 'var(--c-bad)' }}>{cur.cumUncachedProjections}</strong> token passes ·{' '}
            <strong style={{ color: 'var(--c-bad)' }}>{cur.cumUncachedDots}</strong> dot products
          </div>
        </div>

        {/* Cached */}
        <div style={{ padding: 12, borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
            <strong style={{ fontSize: 13, color: 'var(--c-good)' }}>With KV cache</strong>
            <span className="mono" style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              {cur.cachedDotProducts} pairs this step
            </span>
          </div>
          <div style={{ display: 'inline-grid', gridTemplateColumns: `36px repeat(${DEMO_TOKENS.length}, ${cellSize}px)`, gap: 3, alignItems: 'center' }}>
            <div />
            {DEMO_TOKENS.map((t, j) => (
              <div
                key={t}
                className="mono"
                style={{
                  fontSize: 10,
                  textAlign: 'center',
                  color: j === n - 1 ? 'var(--accent)' : j < n ? 'var(--c-a)' : 'var(--text-faint)',
                  fontWeight: j < n ? 700 : 400,
                }}
              >
                K{j + 1}
              </div>
            ))}
            {DEMO_TOKENS.map((t, r) => (
              <div key={t} style={{ display: 'contents' }}>
                <div className="mono" style={{ fontSize: 10, color: r === n - 1 ? 'var(--accent)' : 'var(--text-faint)', fontWeight: r === n - 1 ? 700 : 400 }}>
                  Q{r + 1}
                </div>
                {DEMO_TOKENS.map((_, c) => {
                  const causal = c <= r;
                  const pastCached = r < n - 1 && causal;
                  const needed = r === n - 1 && causal;
                  return (
                    <div
                      key={c}
                      title={needed ? 'Computed fresh against cached K/V' : pastCached ? 'Skipped — K and V already cached' : 'Masked / future'}
                      style={{
                        width: cellSize,
                        height: cellSize,
                        borderRadius: 4,
                        border: `1px solid ${needed ? 'var(--accent)' : pastCached ? 'color-mix(in srgb, var(--c-a) 35%, transparent)' : 'var(--border)'}`,
                        background: needed
                          ? 'color-mix(in srgb, var(--accent) 28%, transparent)'
                          : pastCached
                            ? 'color-mix(in srgb, var(--c-a) 10%, transparent)'
                            : 'transparent',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 11,
                        color: needed ? 'var(--accent)' : pastCached ? 'var(--c-a)' : 'var(--text-faint)',
                      }}
                    >
                      {needed ? '●' : pastCached ? '·' : ''}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
          <div className="mono" style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8 }}>
            Cumulative: <strong style={{ color: 'var(--c-good)' }}>{cur.cumCachedProjections}</strong> token passes ·{' '}
            <strong style={{ color: 'var(--c-good)' }}>{cur.cumCachedDots}</strong> dot products
          </div>
        </div>
      </div>

      {/* Exactness proof + real-scale comparison */}
      <div
        style={{
          padding: '10px 12px',
          borderRadius: 8,
          background: 'var(--bg-sunken)',
          border: '1px solid var(--border)',
          display: 'flex',
          flexWrap: 'wrap',
          gap: 12,
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div className="mono" style={{ fontSize: 12 }}>
          Exactness check (4-dim causal attention):{' '}
          <strong style={{ color: 'var(--c-good)' }}>
            max |cached − uncached| = {demo.maxDiffOverall.toExponential(2)}
          </strong>
        </div>
        <Segmented
          label="real-world scale"
          options={[
            { value: 'toy', label: '6 tokens' },
            { value: 'chat', label: '512 + 256' },
            { value: 'doc', label: '2k + 512' },
          ]}
          value={scalePreset}
          onChange={(v) => setScalePreset(v as 'toy' | 'chat' | 'doc')}
        />
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        <strong>{scaleInfo.label}:</strong> without a KV cache, generating the reply performs{' '}
        <strong>{fmtNum(scaleInfo.uncachedTokens)}</strong> token forward passes vs{' '}
        <strong>{fmtNum(scaleInfo.cachedTokens)}</strong> with caching — a{' '}
        <strong>{scaleInfo.tokenSpeedup.toFixed(1)}×</strong> reduction in work, with zero change to the output numbers.
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. KV Cache Size & GQA Calculator

const CTX_STOPS = [2048, 4096, 8192, 16384, 32768, 65536, 131072];
const BATCH_STOPS = [1, 2, 4, 8, 16, 32, 64];

export function KvCalculator() {
  const [modelId, setModelId] = useState('llama3-8b');
  const [kvFmtIdx, setKvFmtIdx] = useState(0); // FP16
  const [ctxIdx, setCtxIdx] = useState(4); // 32k
  const [batchIdx, setBatchIdx] = useState(0); // B=1

  const model = MODEL_BY_ID[modelId];
  const kvFmt = KV_FORMATS[kvFmtIdx];
  const ctxLen = CTX_STOPS[ctxIdx];
  const batchSize = BATCH_STOPS[batchIdx];

  const perTokKiB = kvBytesPerToken(model, kvFmt.bytesPerElem) / KIB;
  const kvGiB = kvCacheGiB(model, ctxLen, batchSize, kvFmt.bytesPerElem);
  const wFp16GiB = weightsGiB(model, 2.0);
  const totalGiB = wFp16GiB + kvGiB;
  const gqaRatio = model.qHeads / model.kvHeads;

  // Bar scale: at least 80 GiB (H100), grows smoothly if total exceeds 80
  const maxBar = Math.max(80, totalGiB * 1.05);
  const wPct = (wFp16GiB / maxBar) * 100;
  const kvPct = (kvGiB / maxBar) * 100;
  const h100Pct = (80 / maxBar) * 100;

  return (
    <Widget
      title="KV cache footprint calculator — and why GQA exists"
      wide
      caption={
        <>
          Compare <strong>Llama-3 8B</strong> and <strong>70B</strong> with their real Grouped-Query Attention (GQA) against hypothetical standard Multi-Head Attention (MHA). Drag context length to <strong>128k</strong> or raise batch size to watch the KV cache overtake the model’s own weights.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <Segmented
          label="model architecture"
          options={MODEL_SPECS.map((m) => ({ value: m.id, label: m.shortName }))}
          value={modelId}
          onChange={setModelId}
        />
        <Segmented
          label="KV cache precision"
          options={KV_FORMATS.map((f, i) => ({ value: i, label: f.id.toUpperCase() }))}
          value={kvFmtIdx}
          onChange={setKvFmtIdx}
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12, marginBottom: 14 }}>
        <Slider
          label={`Context length: ${fmtCtx(ctxLen)} tokens`}
          min={0}
          max={CTX_STOPS.length - 1}
          step={1}
          value={ctxIdx}
          onChange={setCtxIdx}
          format={(i) => fmtCtx(CTX_STOPS[i])}
        />
        <Slider
          label={`Batch size (concurrent users): ${batchSize}`}
          min={0}
          max={BATCH_STOPS.length - 1}
          step={1}
          value={batchIdx}
          onChange={setBatchIdx}
          format={(i) => `B = ${BATCH_STOPS[i]}`}
        />
      </div>

      {/* Stat cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(135px, 1fr))', gap: 8, marginBottom: 14 }}>
        <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>PER TOKEN (1 USER)</div>
          <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: 'var(--c-a)' }}>
            {perTokKiB >= 1024 ? `${(perTokKiB / 1024).toFixed(2)} MiB` : `${perTokKiB.toFixed(0)} KiB`}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>
            {model.layers}L × {model.kvHeads} KV × {model.dHead}
          </div>
        </div>
        <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>KV CACHE TOTAL</div>
          <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: kvGiB > wFp16GiB ? 'var(--c-bad)' : 'var(--accent)' }}>
            {kvGiB.toFixed(2)} GiB
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>
            {(kvGiB / wFp16GiB).toFixed(2)}× FP16 weights
          </div>
        </div>
        <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>WEIGHTS (FP16)</div>
          <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: 'var(--text)' }}>
            {wFp16GiB.toFixed(2)} GiB
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>
            {model.paramsB}B params
          </div>
        </div>
        <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>GQA COMPRESSION</div>
          <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: gqaRatio > 1 ? 'var(--c-good)' : 'var(--c-b)' }}>
            {gqaRatio}:1
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>
            {model.qHeads} Q / {model.kvHeads} KV heads
          </div>
        </div>
      </div>

      {/* Stacked memory bar against 80 GiB H100 reference */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
          <span className="mono" style={{ color: 'var(--text-muted)' }}>
            Weights ({wFp16GiB.toFixed(1)} GiB) + KV cache ({kvGiB.toFixed(1)} GiB) ={' '}
            <strong style={{ color: totalGiB > 80 ? 'var(--c-bad)' : 'var(--text)' }}>{totalGiB.toFixed(1)} GiB</strong>
          </span>
          <span className="mono" style={{ color: 'var(--text-faint)' }}>
            80 GiB H100 line
          </span>
        </div>
        <div
          style={{
            position: 'relative',
            height: 28,
            borderRadius: 6,
            background: 'var(--bg-sunken)',
            border: '1px solid var(--border)',
            overflow: 'hidden',
            display: 'flex',
          }}
        >
          <div
            style={{
              width: `${wPct}%`,
              background: 'var(--c-a)',
              opacity: 0.85,
              transition: 'width 180ms ease',
            }}
            title={`Weights: ${wFp16GiB.toFixed(2)} GiB`}
          />
          <div
            style={{
              width: `${kvPct}%`,
              background: totalGiB > 80 ? 'var(--c-bad)' : 'var(--accent)',
              opacity: 0.9,
              transition: 'width 180ms ease',
            }}
            title={`KV cache: ${kvGiB.toFixed(2)} GiB`}
          />
          {/* 80 GiB H100 vertical marker */}
          <div
            style={{
              position: 'absolute',
              left: `${Math.min(99.5, h100Pct)}%`,
              top: 0,
              bottom: 0,
              width: 2,
              background: 'var(--text)',
              opacity: 0.75,
            }}
          />
        </div>
      </div>

      <div className="live-note">
        <strong>{model.name}:</strong> {model.note}{' '}
        {kvGiB >= wFp16GiB
          ? `At ${fmtCtx(ctxLen)} context and batch ${batchSize}, the KV cache (${kvGiB.toFixed(1)} GiB) is already larger than the model’s own FP16 weights (${wFp16GiB.toFixed(1)} GiB)!`
          : `At ${fmtCtx(ctxLen)} context and batch ${batchSize}, the KV cache takes ${kvGiB.toFixed(2)} GiB (${((kvGiB / totalGiB) * 100).toFixed(0)}% of weights + KV).`}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Prefill vs Decode Roofline Explorer

const ROOF_GPUS = ['m3-max', 'rtx-4090', 'a100-80', 'h100-80'];
const ROOF_MODELS = ['llama3-8b', 'llama3-70b'];
const ROOF_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256];

export function RooflineExplorer() {
  const [gpuId, setGpuId] = useState('h100-80');
  const [modelId, setModelId] = useState('llama3-8b');
  const [phase, setPhase] = useState<'decode' | 'prefill'>('decode');
  const [wIdx, setWIdx] = useState(0); // FP16
  const [kvIdx, setKvIdx] = useState(0); // FP16
  const [batchIdx, setBatchIdx] = useState(0); // B=1
  const [ctxIdx, setCtxIdx] = useState(0); // 2k

  const gpu = GPU_BY_ID[gpuId];
  const model = MODEL_BY_ID[modelId];
  const wFmt = WEIGHT_FORMATS[wIdx];
  const kvFmt = KV_FORMATS[kvIdx];
  const batchSize = ROOF_BATCHES[batchIdx];
  const ctxLen = CTX_STOPS[ctxIdx];

  const res = useMemo(
    () =>
      evaluateRoofline({
        gpu,
        model,
        phase,
        batchSize,
        ctxLen,
        weightBytesPerElem: wFmt.bytesPerElem,
        kvBytesPerElem: kvFmt.bytesPerElem,
      }),
    [gpu, model, phase, batchSize, ctxLen, wFmt, kvFmt]
  );

  // Reference ghost point: the opposite phase at the exact same settings
  const ghostPhase = phase === 'decode' ? 'prefill' : 'decode';
  const ghost = useMemo(
    () =>
      evaluateRoofline({
        gpu,
        model,
        phase: ghostPhase,
        batchSize,
        ctxLen,
        weightBytesPerElem: wFmt.bytesPerElem,
        kvBytesPerElem: kvFmt.bytesPerElem,
      }),
    [gpu, model, ghostPhase, batchSize, ctxLen, wFmt, kvFmt]
  );

  // Log-log SVG coordinates
  const W = 560;
  const H = 235;
  const pad = { l: 52, r: 22, t: 18, b: 36 };
  const xMin = 0.5;
  const xMax = 10000;
  const yMin = 0.5;
  const yMax = 2000;

  const sx = (v: number) => {
    const c = Math.max(xMin, Math.min(xMax, v));
    return pad.l + ((Math.log10(c) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))) * (W - pad.l - pad.r);
  };
  const sy = (v: number) => {
    const c = Math.max(yMin, Math.min(yMax, v));
    return H - pad.b - ((Math.log10(c) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  };

  const ridge = ridgePoint(gpu);
  const roofLeftY = Math.max(yMin, xMin * gpu.bwTBps);
  const roofLeftX = roofLeftY / gpu.bwTBps;
  const roofPath = `M ${sx(roofLeftX).toFixed(1)} ${sy(roofLeftY).toFixed(1)} L ${sx(ridge).toFixed(1)} ${sy(gpu.peakTflops).toFixed(1)} L ${sx(xMax).toFixed(1)} ${sy(gpu.peakTflops).toFixed(1)}`;

  const xTicks = [1, 10, 100, 1000, 10000];
  const yTicks = [1, 10, 100, 1000];

  const wReadShare = (res.weightBytes / res.bytes) * 100;

  return (
    <Widget
      title="The GPU roofline: why single-user decoding starves the Tensor Cores"
      wide
      caption={
        <>
          Every GPU has two ceilings: a diagonal <strong>memory-bandwidth roof</strong> (left of the ridge point) and a flat <strong>compute roof</strong> (right). Switch between <strong>Decode</strong> and <strong>Prefill</strong>, shrink weights to <strong>INT4</strong>, or crank up <strong>Batch size</strong> to watch the workload climb the slope.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <Segmented
          label="GPU hardware"
          options={ROOF_GPUS.map((id) => ({
            value: id,
            label: id === 'm3-max' ? 'M3 Max' : id === 'rtx-4090' ? 'RTX 4090' : id === 'a100-80' ? 'A100' : 'H100',
          }))}
          value={gpuId}
          onChange={setGpuId}
        />
        <Segmented
          label="phase"
          options={[
            { value: 'decode', label: 'Decode (1 tok/step)' },
            { value: 'prefill', label: 'Prefill (prompt)' },
          ]}
          value={phase}
          onChange={(v) => setPhase(v as 'decode' | 'prefill')}
        />
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <Segmented
          label="model"
          options={ROOF_MODELS.map((id) => ({ value: id, label: MODEL_BY_ID[id].shortName }))}
          value={modelId}
          onChange={setModelId}
        />
        <Segmented
          label="weights"
          options={WEIGHT_FORMATS.map((f, i) => ({ value: i, label: f.id.toUpperCase() }))}
          value={wIdx}
          onChange={setWIdx}
        />
        <Segmented
          label="KV cache"
          options={KV_FORMATS.map((f, i) => ({ value: i, label: f.id.toUpperCase() }))}
          value={kvIdx}
          onChange={setKvIdx}
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12, marginBottom: 12 }}>
        <Slider
          label={`Batch size: ${batchSize}`}
          min={0}
          max={ROOF_BATCHES.length - 1}
          step={1}
          value={batchIdx}
          onChange={setBatchIdx}
          format={(i) => `B = ${ROOF_BATCHES[i]}`}
        />
        <Slider
          label={`Context / prompt length: ${fmtCtx(ctxLen)}`}
          min={0}
          max={CTX_STOPS.length - 1}
          step={1}
          value={ctxIdx}
          onChange={setCtxIdx}
          format={(i) => fmtCtx(CTX_STOPS[i])}
        />
      </div>

      {/* Log-log Roofline SVG */}
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block', background: 'var(--bg-sunken)', borderRadius: 8, border: '1px solid var(--border)', marginBottom: 12 }}>
        {/* Grid */}
        {xTicks.map((t) => (
          <g key={`x-${t}`}>
            <line x1={sx(t)} y1={pad.t} x2={sx(t)} y2={H - pad.b} stroke="var(--border)" strokeDasharray="3 3" />
            <text x={sx(t)} y={H - pad.b + 14} textAnchor="middle" fill="var(--text-faint)" fontSize={10} fontFamily="var(--font-mono)">
              {t >= 1000 ? `${t / 1000}k` : t}
            </text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`y-${t}`}>
            <line x1={pad.l} y1={sy(t)} x2={W - pad.r} y2={sy(t)} stroke="var(--border)" strokeDasharray="3 3" />
            <text x={pad.l - 7} y={sy(t) + 3} textAnchor="end" fill="var(--text-faint)" fontSize={10} fontFamily="var(--font-mono)">
              {t}
            </text>
          </g>
        ))}

        {/* Axis labels */}
        <text x={(pad.l + W - pad.r) / 2} y={H - 5} textAnchor="middle" fill="var(--text-muted)" fontSize={11}>
          Arithmetic intensity (FLOPs per byte read from VRAM, log scale)
        </text>
        <text x={13} y={(pad.t + H - pad.b) / 2} textAnchor="middle" fill="var(--text-muted)" fontSize={11} transform={`rotate(-90 13 ${(pad.t + H - pad.b) / 2})`}>
          TFLOP/s (log)
        </text>

        {/* Ridge vertical guide */}
        <line x1={sx(ridge)} y1={sy(gpu.peakTflops)} x2={sx(ridge)} y2={H - pad.b} stroke="var(--c-b)" strokeDasharray="4 4" opacity={0.65} />
        <text x={sx(ridge) - 6} y={sy(gpu.peakTflops) - 6} textAnchor="end" fill="var(--c-b)" fontSize={10} fontFamily="var(--font-mono)">
          ridge {ridge.toFixed(0)} FLOP/B
        </text>

        {/* Region labels */}
        <text x={pad.l + 8} y={pad.t + 14} fill="var(--c-bad)" fontSize={10} fontWeight={600}>
          MEMORY-BOUND
        </text>
        <text x={W - pad.r - 4} y={sy(gpu.peakTflops) - 6} textAnchor="end" fill="var(--c-good)" fontSize={10} fontWeight={600}>
          COMPUTE-BOUND ({gpu.peakTflops} TF)
        </text>

        {/* Roofline curve */}
        <path d={roofPath} fill="none" stroke="var(--text)" strokeWidth={2.2} />

        {/* Ghost opposite-phase dot */}
        <circle cx={sx(ghost.arithmeticIntensity)} cy={sy(ghost.achievedTflops)} r={5} fill="none" stroke="var(--text-faint)" strokeWidth={1.6} strokeDasharray="2 2" />
        <text
          x={sx(ghost.arithmeticIntensity) + (ghost.arithmeticIntensity > 400 ? -8 : 8)}
          y={sy(ghost.achievedTflops) + 16}
          textAnchor={ghost.arithmeticIntensity > 400 ? 'end' : 'start'}
          fill="var(--text-faint)"
          fontSize={10}
          fontFamily="var(--font-mono)"
        >
          {ghostPhase} ({ghost.arithmeticIntensity.toFixed(1)} F/B)
        </text>

        {/* Active workload dot */}
        <circle
          cx={sx(res.arithmeticIntensity)}
          cy={sy(res.achievedTflops)}
          r={7}
          fill={res.bottleneck === 'memory' ? 'var(--c-bad)' : 'var(--c-good)'}
          stroke="var(--bg)"
          strokeWidth={2}
        />
        <text
          x={sx(res.arithmeticIntensity) + (res.arithmeticIntensity > 400 ? -10 : 10)}
          y={sy(res.achievedTflops) + (res.bottleneck === 'compute' ? 18 : -8)}
          textAnchor={res.arithmeticIntensity > 400 ? 'end' : 'start'}
          fill={res.bottleneck === 'memory' ? 'var(--c-bad)' : 'var(--c-good)'}
          fontSize={11}
          fontWeight={700}
          fontFamily="var(--font-mono)"
        >
          {phase} ({res.arithmeticIntensity.toFixed(1)} F/B)
        </text>
      </svg>

      {/* Metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(135px, 1fr))', gap: 8, marginBottom: 10 }}>
        <div style={{ padding: '9px 11px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>BOTTLENECK</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: res.bottleneck === 'memory' ? 'var(--c-bad)' : 'var(--c-good)' }}>
            {res.bottleneck === 'memory' ? 'Memory BW' : 'Compute'}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>
            {res.arithmeticIntensity.toFixed(2)} / {ridge.toFixed(0)} FLOP/B
          </div>
        </div>
        <div style={{ padding: '9px 11px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>TENSOR CORE UTIL</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: res.computeUtilPct < 10 ? 'var(--c-bad)' : 'var(--accent)' }}>
            {res.computeUtilPct.toFixed(2)}%
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>
            {res.achievedTflops.toFixed(1)} / {gpu.peakTflops} TF
          </div>
        </div>
        <div style={{ padding: '9px 11px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>PER-USER SPEED</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: 'var(--c-a)' }}>
            {res.tokensPerSecPerUser >= 1000 ? `${(res.tokensPerSecPerUser / 1000).toFixed(1)}k` : res.tokensPerSecPerUser.toFixed(1)} tok/s
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>
            {res.stepTimeMs.toFixed(2)} ms / step
          </div>
        </div>
        <div style={{ padding: '9px 11px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)' }}>
          <div className="field-label" style={{ marginBottom: 2 }}>TOTAL THROUGHPUT</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: 'var(--c-good)' }}>
            {res.totalTokensPerSec >= 10000 ? `${(res.totalTokensPerSec / 1000).toFixed(1)}k` : res.totalTokensPerSec.toFixed(0)} tok/s
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>
            roofline ceiling
          </div>
        </div>
      </div>

      <div className="live-note">
        {res.bottleneck === 'memory' ? (
          <>
            <strong>Memory-bandwidth bound ({res.computeUtilPct.toFixed(2)}% compute used):</strong> every step streams{' '}
            <strong>{(res.bytes / 1e9).toFixed(2)} GB</strong> from VRAM ({wReadShare.toFixed(0)}% weights, {(100 - wReadShare).toFixed(0)}% KV cache).{' '}
            {wReadShare > 65
              ? 'Weight reads dominate — quantising weights to INT4 or increasing batch size immediately boosts speed.'
              : 'KV cache reads dominate — quantising the KV cache (FP8/INT4) now speeds up decoding more than shrinking weights!'}
          </>
        ) : (
          <>
            <strong>Compute-bound (100% Tensor Core ceiling):</strong> intensity ({res.arithmeticIntensity.toFixed(0)} FLOPs/byte) clears the{' '}
            {gpu.shortName} ridge point ({ridge.toFixed(0)} FLOPs/byte). The step completes in{' '}
            <strong>{res.stepTimeMs.toFixed(1)} ms</strong> ({fmtNum(res.totalTokensPerSec)} tok/s ceiling).
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: VRAM Tetris

const STORAGE_KEY = 'nl-kv-lab-v1';

interface SavedLabState {
  configs: Record<string, TetrisConfig>;
  solved: Record<string, boolean>;
}

function loadLabState(): SavedLabState {
  const defaults: Record<string, TetrisConfig> = Object.fromEntries(
    TETRIS_BRIEFS.map((b) => [b.id, { ...b.defaultConfig }])
  );
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { configs: defaults, solved: {} };
    const parsed = JSON.parse(raw);
    return {
      configs: { ...defaults, ...(parsed.configs ?? {}) },
      solved: parsed.solved ?? {},
    };
  } catch {
    return { configs: defaults, solved: {} };
  }
}

export function VramTetrisLab() {
  const [briefIdx, setBriefIdx] = useState(0);
  const [state, setState] = useState<SavedLabState>(() => loadLabState());
  const [showOpt, setShowOpt] = useState(false);

  const brief = TETRIS_BRIEFS[briefIdx];
  const gpu = GPU_BY_ID[brief.gpuId];
  const model = MODEL_BY_ID[brief.modelId];
  const cfg = state.configs[brief.id] ?? brief.defaultConfig;

  const ev = useMemo(() => evaluateTetris(brief, cfg), [brief, cfg]);
  const sol = useMemo(() => solveTetrisBrief(brief), [brief]);

  useEffect(() => {
    if (ev.passes && !state.solved[brief.id]) {
      setState((prev) => ({ ...prev, solved: { ...prev.solved, [brief.id]: true } }));
    }
  }, [ev.passes, brief.id, state.solved]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      /* ignore */
    }
  }, [state]);

  const updateCfg = (patch: Partial<TetrisConfig>) => {
    setState((prev) => ({
      ...prev,
      configs: { ...prev.configs, [brief.id]: { ...cfg, ...patch } },
    }));
  };

  const metricVal = (e: typeof ev) =>
    brief.optMetric === 'tokPerUser' ? e.roofline.tokensPerSecPerUser : e.roofline.totalTokensPerSec;
  const hitOpt = ev.passes && metricVal(ev) >= metricVal(sol.bestEval) - 0.1;

  // VRAM Tetris bar scaling
  const barMax = Math.max(gpu.vramGiB, ev.totalGiB);
  const wPct = (ev.wGiB / barMax) * 100;
  const kvPct = (ev.kvGiB / barMax) * 100;
  const ohPct = (ev.overheadGiB / barMax) * 100;
  const limitPct = (gpu.vramGiB / barMax) * 100;

  const solvedCount = TETRIS_BRIEFS.filter((b) => state.solved[b.id]).length;

  return (
    <Widget
      title="Lab: VRAM Tetris — fit the model, the KV cache, and the SLA"
      wide
      caption={
        <>
          Three real serving briefs. Pick weight precision, KV cache precision, batch size and context length so everything fits inside the GPU’s VRAM wall while hitting the speed and context targets. All {sol.totalConfigs} discrete configurations are checked exhaustively.
        </>
      }
    >
      {/* Brief selector */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div className="btn-row">
          {TETRIS_BRIEFS.map((b, i) => (
            <button
              key={b.id}
              className={briefIdx === i ? 'btn primary' : 'btn'}
              onClick={() => {
                setBriefIdx(i);
                setShowOpt(false);
              }}
            >
              {state.solved[b.id] ? '✓ ' : ''}
              {i === 0 ? 'Brief 1 · 4070' : i === 1 ? 'Brief 2 · 70B' : 'Brief 3 · API'}
            </button>
          ))}
        </div>
        <span className="mono" style={{ fontSize: 12, color: 'var(--text-faint)' }}>
          {solvedCount} / {TETRIS_BRIEFS.length} briefs solved
        </span>
      </div>

      {/* Scenario card */}
      <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-sunken)', border: '1px solid var(--border)', marginBottom: 12, fontSize: 13 }}>
        <div style={{ fontWeight: 700, marginBottom: 4 }}>
          {brief.title} — {model.shortName} on {gpu.name}
        </div>
        <div style={{ color: 'var(--text-muted)' }}>{brief.scenario}</div>
      </div>

      {/* VRAM Tetris Bar */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', fontSize: 12, marginBottom: 5, gap: 6 }}>
          <span className="mono">
            <span style={{ color: 'var(--c-a)' }}>■</span> Weights {ev.wGiB.toFixed(2)} GiB ·{' '}
            <span style={{ color: 'var(--accent)' }}>■</span> KV {ev.kvGiB.toFixed(2)} GiB ·{' '}
            <span style={{ color: 'var(--text-faint)' }}>■</span> Runtime {ev.overheadGiB.toFixed(1)} GiB
          </span>
          <strong className="mono" style={{ color: ev.fitsVram ? 'var(--c-good)' : 'var(--c-bad)' }}>
            {ev.totalGiB.toFixed(2)} / {gpu.vramGiB} GiB {ev.fitsVram ? '(fits)' : `(OOM +${(ev.totalGiB - gpu.vramGiB).toFixed(2)} GiB)`}
          </strong>
        </div>

        <div
          style={{
            position: 'relative',
            height: 32,
            borderRadius: 6,
            background: 'var(--bg-sunken)',
            border: `1.5px solid ${ev.fitsVram ? 'var(--border)' : 'var(--c-bad)'}`,
            overflow: 'hidden',
            display: 'flex',
          }}
        >
          <div style={{ width: `${wPct}%`, background: 'var(--c-a)', opacity: 0.85, transition: 'width 160ms ease' }} />
          <div style={{ width: `${kvPct}%`, background: ev.fitsVram ? 'var(--accent)' : 'var(--c-bad)', opacity: 0.9, transition: 'width 160ms ease' }} />
          <div style={{ width: `${ohPct}%`, background: 'var(--text-faint)', opacity: 0.45, transition: 'width 160ms ease' }} />
          {/* VRAM wall */}
          <div
            style={{
              position: 'absolute',
              left: `${Math.min(99.5, limitPct)}%`,
              top: 0,
              bottom: 0,
              width: 2.5,
              background: ev.fitsVram ? 'var(--c-good)' : 'var(--c-bad)',
            }}
          />
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 10, marginBottom: 12 }}>
        <div>
          <Segmented
            label="weight precision"
            options={WEIGHT_FORMATS.map((f) => ({ value: f.id, label: `${f.id.toUpperCase()} (${f.bitsLabel})` }))}
            value={cfg.weightFmt}
            onChange={(v) => updateCfg({ weightFmt: v as TetrisConfig['weightFmt'] })}
          />
        </div>
        <div>
          <Segmented
            label="KV cache precision"
            options={KV_FORMATS.map((f) => ({ value: f.id, label: `${f.id.toUpperCase()} (${f.bitsLabel})` }))}
            value={cfg.kvFmt}
            onChange={(v) => updateCfg({ kvFmt: v as TetrisConfig['kvFmt'] })}
          />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 12, marginBottom: 14 }}>
        <Slider
          label={`Batch size: ${cfg.batchSize} concurrent user${cfg.batchSize === 1 ? '' : 's'}`}
          min={0}
          max={TETRIS_BATCHES.length - 1}
          step={1}
          value={TETRIS_BATCHES.indexOf(cfg.batchSize as (typeof TETRIS_BATCHES)[number])}
          onChange={(i) => updateCfg({ batchSize: TETRIS_BATCHES[i] })}
          format={(i) => `B = ${TETRIS_BATCHES[i]}`}
        />
        <Slider
          label={`Context length: ${fmtCtx(cfg.ctxLen)} tokens`}
          min={0}
          max={TETRIS_CTXS.length - 1}
          step={1}
          value={TETRIS_CTXS.indexOf(cfg.ctxLen as (typeof TETRIS_CTXS)[number])}
          onChange={(i) => updateCfg({ ctxLen: TETRIS_CTXS[i] })}
          format={(i) => fmtCtx(TETRIS_CTXS[i])}
        />
      </div>

      {/* Constraint checklist */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(175px, 1fr))', gap: 6, marginBottom: 12 }}>
        {[
          { ok: ev.checks.fitsVram, label: `VRAM ≤ ${gpu.vramGiB} GiB`, val: `${ev.totalGiB.toFixed(2)} GiB` },
          { ok: ev.checks.meetsCtx, label: `Context ≥ ${fmtCtx(brief.minCtx)}`, val: fmtCtx(cfg.ctxLen) },
          { ok: ev.checks.meetsBatch, label: `Batch ≥ ${brief.minBatch}`, val: `B = ${cfg.batchSize}` },
          { ok: ev.checks.meetsUserSpeed, label: `Per-user ≥ ${brief.minTokPerUser} tok/s`, val: `${ev.roofline.tokensPerSecPerUser.toFixed(1)} tok/s` },
          { ok: ev.checks.meetsTotalSpeed, label: `Total ≥ ${fmtNum(brief.minTotalTokPerSec)} tok/s`, val: `${ev.roofline.totalTokensPerSec.toFixed(0)} tok/s` },
          ...(brief.forbidDoubleInt4
            ? [{ ok: ev.checks.meetsQuality, label: 'No double-INT4', val: ev.checks.meetsQuality ? 'OK' : 'INT4 + INT4' }]
            : []),
        ].map((c) => (
          <div
            key={c.label}
            className="mono"
            style={{
              padding: '7px 9px',
              borderRadius: 6,
              fontSize: 12,
              border: `1px solid ${c.ok ? 'color-mix(in srgb, var(--c-good) 45%, transparent)' : 'color-mix(in srgb, var(--c-bad) 45%, transparent)'}`,
              background: c.ok
                ? 'color-mix(in srgb, var(--c-good) 10%, transparent)'
                : 'color-mix(in srgb, var(--c-bad) 10%, transparent)',
              display: 'flex',
              justifyContent: 'space-between',
              gap: 6,
            }}
          >
            <span style={{ color: c.ok ? 'var(--c-good)' : 'var(--c-bad)', fontWeight: 700 }}>
              {c.ok ? '✓' : '✗'} {c.label}
            </span>
            <span style={{ color: 'var(--text)' }}>{c.val}</span>
          </div>
        ))}
      </div>

      {/* Status + optional exhaustive optimum */}
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
              {hitOpt ? 'Brief solved — and you hit the exhaustive optimum!' : 'Brief solved!'}
            </strong>{' '}
            ({sol.passingCount} of {sol.totalConfigs} grid configs pass.)
            {!hitOpt && (
              <>
                {' '}Optional goal: push <strong>{brief.optLabel}</strong> up to{' '}
                <strong>{metricVal(sol.bestEval).toFixed(1)} tok/s</strong>.
              </>
            )}
          </>
        ) : (
          <>
            <strong>Not yet passing.</strong>{' '}
            {!ev.checks.fitsVram
              ? `Out of memory by ${(ev.totalGiB - gpu.vramGiB).toFixed(2)} GiB — quantise weights or the KV cache (or reduce batch/context).`
              : !ev.checks.meetsQuality
                ? 'Avoid quantising BOTH weights and KV cache to INT4 at once — keep at least one at FP8/INT8 or higher.'
                : !ev.checks.meetsTotalSpeed
                  ? 'Fit more concurrent users in the batch (quantising the KV cache frees the most VRAM per user!).'
                  : 'Adjust precision, batch size or context to satisfy every red check above.'}
          </>
        )}
      </div>

      <div className="btn-row" style={{ marginTop: 10 }}>
        <button className="btn" onClick={() => updateCfg(brief.defaultConfig)}>
          Reset brief
        </button>
        <button className="btn" onClick={() => setShowOpt((s) => !s)}>
          {showOpt ? 'Hide exhaustive optimum' : 'Show exhaustive optimum'}
        </button>
      </div>

      {showOpt && (
        <div className="mono" style={{ marginTop: 8, padding: '8px 10px', borderRadius: 6, background: 'var(--bg-sunken)', border: '1px solid var(--border)', fontSize: 12 }}>
          Exhaustive search over all {sol.totalConfigs} grid configs ({sol.passingCount} pass): best is{' '}
          <strong>
            {sol.bestConfig.weightFmt.toUpperCase()} weights + {sol.bestConfig.kvFmt.toUpperCase()} KV, B={sol.bestConfig.batchSize}, ctx={fmtCtx(sol.bestConfig.ctxLen)}
          </strong>{' '}
          → {sol.bestEval.totalGiB.toFixed(2)} GiB VRAM, {sol.bestEval.roofline.tokensPerSecPerUser.toFixed(1)} tok/s/user,{' '}
          {sol.bestEval.roofline.totalTokensPerSec.toFixed(1)} tok/s total.
        </div>
      )}
    </Widget>
  );
}
