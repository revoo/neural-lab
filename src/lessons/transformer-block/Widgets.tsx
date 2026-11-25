import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { ModuleRef } from '../../components/Term';
import { Segmented, Slider, Toggle } from '../../components/controls';
import { LogChart } from '../../components/LogChart';
import { GPT2, TOKENS, depthStudy, gptParams, runBlock, type Vec } from './block';

const fmt = (n: number) => n.toLocaleString('en-GB');
const f2 = (v: number) => v.toFixed(2);

function VecRow({ v, max = 5 }: { v: Vec; max?: number }) {
  return (
    <svg viewBox={`0 0 ${v.length * 22} 56`} width={v.length * 22} height={56} role="img" aria-label={v.map(f2).join(', ')}>
      <line x1={0} x2={v.length * 22} y1={28} y2={28} className="plot-axis" />
      {v.map((x, i) => {
        const h = (Math.max(-max, Math.min(max, x)) / max) * 26;
        return <rect key={i} x={i * 22 + 3} width={16} y={h > 0 ? 28 - h : 28} height={Math.max(0.8, Math.abs(h))} rx={2} fill={x >= 0 ? 'var(--c-a)' : 'var(--c-b)'} />;
      })}
    </svg>
  );
}

// ------------------------------------------------------------------ 1. one block, stage by stage

export function BlockStepper() {
  const [stage, setStage] = useState(0);
  const [causal, setCausal] = useState(true);
  const stages = useMemo(() => runBlock(undefined, causal), [causal]);
  const s = stages[stage];
  const attn = stages[2].weights!;
  const residualStage = stage === 3 || stage === 6;
  return (
    <Widget
      title="Through one transformer block"
      wide
      caption={
        <>
          Three tokens, each a vector of 4 numbers (bars: blue positive, orange negative), passing through one block with random, untrained weights — this shows the
          machinery, not meaning. Step through the stages. Watch what the two <strong>residual</strong> additions do: the block never replaces a token's vector, it adds
          to it.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10 }}>
        <button className="btn" onClick={() => setStage(Math.max(0, stage - 1))} disabled={stage === 0}>
          ◂ Back
        </button>
        <button className="btn primary" onClick={() => setStage(Math.min(stages.length - 1, stage + 1))} disabled={stage === stages.length - 1}>
          Next stage ▸
        </button>
        <Toggle label="causal mask (no peeking ahead)" checked={causal} onChange={setCausal} />
      </div>
      <div className="nfs-code" style={{ marginBottom: 10 }}>
        {stages.map((st, i) => (
          <div key={st.name} className={i === stage ? 'nfs-line on' : 'nfs-line'} onClick={() => setStage(i)} style={{ cursor: 'pointer' }}>
            {`${i}. ${st.name}`}
          </div>
        ))}
      </div>
      <div className="widget-row">
        <div className="widget-col">
          <div className="field-label">{s.name}</div>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            {s.vecs.map((v, i) => (
              <div key={i} style={{ textAlign: 'center' }}>
                <div className="mono" style={{ fontWeight: 700, fontSize: 13 }}>
                  {TOKENS[i]}
                </div>
                <VecRow v={v} />
                <div className="mono faint" style={{ fontSize: 11 }}>
                  [{v.map(f2).join(', ')}]
                </div>
              </div>
            ))}
          </div>
          {residualStage && (
            <>
              <div className="field-label" style={{ marginTop: 8 }}>
                what it was before this addition
              </div>
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', opacity: 0.6 }}>
                {stages[stage === 3 ? 0 : 3].vecs.map((v, i) => (
                  <VecRow key={i} v={v} />
                ))}
              </div>
            </>
          )}
        </div>
        {stage === 2 && (
          <div className="widget-col" style={{ flex: '0 1 220px' }}>
            <div className="field-label">attention weights (row attends to column)</div>
            <table className="num-table">
              <thead>
                <tr>
                  <th />
                  {TOKENS.map((t) => (
                    <th key={t}>{t}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {attn.map((row, i) => (
                  <tr key={i}>
                    <th>{TOKENS[i]}</th>
                    {row.map((w, j) => (
                      <td key={j} style={{ background: `color-mix(in srgb, var(--accent) ${Math.round(w * 45)}%, transparent)`, opacity: causal && j > i ? 0.35 : 1 }}>
                        {causal && j > i ? '—' : w.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      <div className="live-note" style={{ marginTop: 10 }}>
        <strong>{s.name}:</strong> {s.note}.{' '}
        {stage === 1 && 'Look at the numbers: each vector now averages 0 and has spread 1, whatever size it arrived at.'}
        {stage === 2 && causal && ' With the causal mask, "the" can only see itself, "cat" sees "the" and itself, "sat" sees all three — never a later word.'}
        {stage === 2 && !causal && ' Without the mask, every token sees every other, including later ones.'}
        {stage === 5 && <> This is the same small MLP you met in <ModuleRef id="architecture" short />, applied to each token separately: attention moves information between tokens, the feed-forward network processes it within each one.</>}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. why residuals

const DEPTHS = [1, 2, 4, 8, 12, 16, 24, 32, 48];

export function DepthExperiment() {
  const rows = useMemo(
    () => DEPTHS.map((L) => ({ L, plain: depthStudy(L, false, false), res: depthStudy(L, true, false), resLN: depthStudy(L, true, true) })),
    [],
  );
  const [L, setL] = useState(48);
  const row = rows.find((r) => r.L === L)!;
  return (
    <Widget
      title="How much signal survives a deep stack?"
      wide
      caption={
        <>
          Stacks of random layers (each tanh of a random 32 × 32 matrix), averaged over 8 random draws. For each depth we nudge the input slightly and measure how much the
          output moves: that's the size of the gradient that would flow back to the first layer in training. Three ways of stacking: plain (each layer replaces the
          vector), residual (each layer adds to it), and residual with layer norm before each layer, as in a transformer.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 8 }}>
        <Segmented label="depth" value={L} onChange={setL} options={DEPTHS} />
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <LogChart
            series={[
              { points: rows.map((r) => [r.L, r.plain.grad] as [number, number]), color: 'var(--c-b)', dots: true },
              { points: rows.map((r) => [r.L, r.res.grad] as [number, number]), color: 'var(--c-a)', dots: true },
              { points: rows.map((r) => [r.L, r.resLN.grad] as [number, number]), color: 'var(--c-c)', dots: true },
            ]}
            xDomain={[0, 50]}
            yDomain={[0.01, 100]}
            marker={L}
            xLabel="depth (layers)"
            xTicks={[1, 12, 24, 48]}
            hlines={[{ y: 1, label: 'unchanged', color: 'var(--text-faint)' }]}
            ariaLabel="Gradient size at the input against depth"
          />
          <div className="muted" style={{ fontSize: 12.5 }}>
            Gradient reaching the input (log scale): <span className="c-b">plain</span>, <span className="c-a">residual</span>, <span className="c-c">residual + layer norm</span>.
          </div>
        </div>
        <div className="widget-col">
          <div className="readout">
            <span className="k">depth {L}</span>
            {'\n'}
            <span className="k">plain                  </span> {row.plain.grad.toFixed(3)}
            {'\n'}
            <span className="k">residual               </span> {row.res.grad.toFixed(3)}
            {'\n'}
            <span className="k">residual + layer norm  </span> {row.resLN.grad.toFixed(3)}
          </div>
          <div className="live-note" style={{ marginTop: 10 }}>
            {L === 1 ? (
              'With one layer, all three are similar. Increase the depth.'
            ) : (
              <>
                Plain stacking has lost {((1 - row.plain.grad / rows[0].plain.grad) * 100).toFixed(0)}% of the signal it had at depth 1: each layer multiplies it by a
                slope below 1, the fading signal from the backpropagation lesson. With residuals it never falls below 1, because there's always a direct path through the
                additions; but it <em>grows</em> ({row.res.grad.toFixed(1)}). Layer norm keeps that growth in check ({row.resLN.grad.toFixed(1)}).
              </>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. counting a GPT

export function GptCounter() {
  const [preset, setPreset] = useState<keyof typeof GPT2 | 'custom'>('small');
  const [custom, setCustom] = useState({ d: 512, layers: 8, vocab: 50257, context: 1024 });
  const shape = preset === 'custom' ? custom : GPT2[preset];
  const p = gptParams(shape);
  const parts = [
    { name: 'token embeddings', v: p.embed, c: 'var(--c-b)' },
    { name: 'position embeddings', v: p.pos, c: 'var(--c-grad)' },
    { name: 'attention (all blocks)', v: (p.attn) * shape.layers, c: 'var(--c-a)' },
    { name: 'feed-forward (all blocks)', v: p.ffn * shape.layers, c: 'var(--c-c)' },
    { name: 'layer norms', v: p.norms * shape.layers + p.finalNorm, c: 'var(--text-faint)' },
  ];
  return (
    <Widget
      title="Where a GPT's parameters live"
      wide
      caption={
        <>
          A GPT is an embedding table, a stack of identical blocks, and a final layer norm (the output layer reuses the embedding table). Pick a GPT-2 size, or build your
          own. These counts match the published GPT-2 sizes exactly.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <Segmented
          label="model"
          value={preset}
          onChange={setPreset}
          options={[
            { value: 'small', label: 'GPT-2 small' },
            { value: 'medium', label: 'medium' },
            { value: 'large', label: 'large' },
            { value: 'xl', label: 'XL' },
            { value: 'custom', label: 'custom' },
          ]}
        />
      </div>
      {preset === 'custom' && (
        <div className="controls inline" style={{ marginBottom: 10 }}>
          <div style={{ flex: '1 1 180px' }}>
            <Slider label="width d" value={custom.d} min={64} max={2048} step={64} onChange={(v) => setCustom({ ...custom, d: v })} format={String} />
          </div>
          <div style={{ flex: '1 1 180px' }}>
            <Slider label="blocks" value={custom.layers} min={1} max={64} step={1} onChange={(v) => setCustom({ ...custom, layers: v })} format={String} />
          </div>
          <div style={{ flex: '1 1 180px' }}>
            <Slider label="vocabulary" value={custom.vocab} min={1000} max={256000} step={1000} onChange={(v) => setCustom({ ...custom, vocab: v })} format={fmt} />
          </div>
        </div>
      )}
      <div className="readout">
        <span className="k">width {shape.d}, {shape.layers} blocks, vocabulary {fmt(shape.vocab)}, context {fmt(shape.context)}</span>
        {'\n'}
        <span className="k">one block                </span> {fmt(p.block)} <span className="k">(attention {fmt(p.attn)}, feed-forward {fmt(p.ffn)}, norms {fmt(p.norms)})</span>
        {'\n'}
        <span className="k">total                    </span> <strong>{fmt(p.total)}</strong>
      </div>
      <div style={{ display: 'flex', height: 22, borderRadius: 4, overflow: 'hidden', margin: '10px 0 6px' }}>
        {parts.map((x) => (
          <div key={x.name} style={{ width: `${(x.v / p.total) * 100}%`, background: x.c }} title={x.name} />
        ))}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 14px', fontSize: 12.5 }}>
        {parts.map((x) => (
          <span key={x.name}>
            <span style={{ display: 'inline-block', width: 10, height: 10, background: x.c, borderRadius: 2, marginRight: 5 }} />
            {x.name}: {((x.v / p.total) * 100).toFixed(1)}%
          </span>
        ))}
      </div>
      <div className="live-note" style={{ marginTop: 10 }}>
        Inside every block, the feed-forward network holds twice as many parameters as attention (8d² against 4d²). In GPT-2 small, the token embedding table alone is{' '}
        {((p.embed / gptParams(GPT2.small).total) * 100).toFixed(0)}% of the model{preset === 'small' ? '' : ' (here: ' + ((p.embed / p.total) * 100).toFixed(0) + '%)'}; in bigger models, the blocks dominate.
      </div>
    </Widget>
  );
}
