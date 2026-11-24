// authored-by: Neon High
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { createRng } from '../../lib/rng';
import {
  BEAM_TREE,
  CONTEXTS,
  LAB_LEVELS,
  applySampling,
  evaluateSampler,
  runBeamSearch,
  solveLabLevel,
} from './sampling';

const pct = (p: number) => `${(p * 100).toFixed(p > 0 && p < 0.01 ? 2 : 1)}%`;
const f2 = (n: number) => n.toFixed(2);

// ------------------------------------------------------------------ 1. Pipeline Inspector

export function PipelineInspector() {
  const [ctxIdx, setCtxIdx] = useState(0);
  const [T, setT] = useState(1.0);
  const [topK, setTopK] = useState(16);
  const [topP, setTopP] = useState(1.0);
  const [drawSeed, setDrawSeed] = useState(1);

  const ctx = CONTEXTS[ctxIdx];
  const res = useMemo(() => applySampling(ctx.tokens, T, topK, topP), [ctx, T, topK, topP]);
  const maxBar = Math.max(0.05, ...res.probs, ...res.raw);

  const draws = useMemo(() => {
    const rng = createRng(drawSeed * 131 + ctxIdx * 17);
    const out: { word: string; valid: boolean }[] = [];
    for (let d = 0; d < 10; d++) {
      let r = rng.next();
      let pick = ctx.tokens.length - 1;
      for (let i = 0; i < res.probs.length; i++) {
        r -= res.probs[i];
        if (r <= 0) {
          pick = i;
          break;
        }
      }
      out.push({ word: ctx.tokens[pick].word, valid: ctx.tokens[pick].valid });
    }
    return out;
  }, [ctx, res.probs, drawSeed, ctxIdx]);

  return (
    <Widget
      title="Inside the sampling pipeline: logits → T → top-k → top-p"
      wide
      caption={
        <>
          Switch between a sharp factual prompt, a medium prompt and a flat open-ended prompt. Adjust temperature, top-k and top-p to watch which tokens survive, how probabilities re-normalise, and how often tail-junk tokens sneak into 10 live draws.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <Segmented
          options={CONTEXTS.map((c, i) => ({ value: i, label: c.label }))}
          value={ctxIdx}
          onChange={setCtxIdx}
        />
        <div className="btn-row">
          <button
            className="btn"
            onClick={() => {
              setT(1.0);
              setTopK(16);
              setTopP(1.0);
            }}
          >
            Raw (T=1)
          </button>
          <button
            className="btn"
            onClick={() => {
              setT(0.1);
              setTopK(1);
              setTopP(1.0);
            }}
          >
            Greedy
          </button>
          <button
            className="btn"
            onClick={() => {
              setT(1.0);
              setTopK(5);
              setTopP(1.0);
            }}
          >
            Top-k = 5
          </button>
          <button
            className="btn primary"
            onClick={() => {
              setT(1.0);
              setTopK(16);
              setTopP(0.9);
            }}
          >
            Top-p = 0.90
          </button>
        </div>
      </div>

      <div
        className="mono"
        style={{
          padding: '8px 12px',
          borderRadius: 'var(--radius-sm)',
          background: 'var(--bg-sunken)',
          border: '1px solid var(--border)',
          fontSize: 14,
          marginBottom: 10,
        }}
      >
        “{ctx.prompt} <strong style={{ color: 'var(--accent-text)' }}>___</strong>”
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 12, marginBottom: 12 }}>
        <Slider label="temperature T" value={T} min={0.1} max={2.0} step={0.1} onChange={setT} format={(v) => v.toFixed(1)} color="accent" />
        <Slider label="top-k cutoff" value={topK} min={1} max={16} step={1} onChange={setTopK} format={(v) => `${v}`} color="c-a" />
        <Slider label="top-p (nucleus)" value={topP} min={0.1} max={1.0} step={0.05} onChange={setTopP} format={(v) => v.toFixed(2)} color="c-c" />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, marginBottom: 12 }}>
        {[
          { label: 'surviving pool', val: `${res.poolSize} / 16`, color: 'var(--accent-text)' },
          { label: 'effective choices', val: f2(res.effectiveChoices), color: 'var(--c-a)' },
          { label: 'valid mass', val: pct(res.validMass), color: 'var(--c-good)' },
          { label: 'tail-junk risk', val: pct(res.junkMass), color: res.junkMass > 0.02 ? 'var(--c-bad)' : 'var(--text-muted)' },
        ].map((m) => (
          <div
            key={m.label}
            style={{
              background: 'var(--bg-sunken)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              padding: '7px 10px',
            }}
          >
            <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{m.label}</div>
            <div className="mono" style={{ fontSize: 17, fontWeight: 700, color: m.color }}>
              {m.val}
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gap: 4, marginBottom: 10 }}>
        {ctx.tokens.map((tok, i) => {
          const kept = res.kept[i];
          const p = res.probs[i];
          const rawP = res.raw[i];
          return (
            <div
              key={tok.word}
              style={{
                display: 'grid',
                gridTemplateColumns: '86px 1fr 56px',
                gap: 8,
                alignItems: 'center',
                opacity: kept ? 1 : 0.38,
                fontSize: 12.5,
              }}
            >
              <span
                className="mono"
                style={{
                  textDecoration: kept ? 'none' : 'line-through',
                  color: !tok.valid ? 'var(--c-bad)' : i === 0 ? 'var(--accent-text)' : 'var(--text)',
                  fontWeight: kept && i === 0 ? 700 : 500,
                }}
              >
                {tok.word}
              </span>
              <div style={{ position: 'relative', height: 10, borderRadius: 4, background: 'var(--bg-sunken)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    width: `${(rawP / maxBar) * 100}%`,
                    background: 'var(--text-faint)',
                    opacity: 0.25,
                  }}
                />
                {kept && (
                  <div
                    style={{
                      position: 'absolute',
                      inset: 0,
                      width: `${(p / maxBar) * 100}%`,
                      background: !tok.valid ? 'var(--c-bad)' : i === 0 ? 'var(--accent)' : 'var(--c-a)',
                    }}
                  />
                )}
              </div>
              <span className="mono" style={{ textAlign: 'right', color: kept ? 'var(--text)' : 'var(--text-faint)', fontSize: 11.5 }}>
                {kept ? pct(p) : 'cut'}
              </span>
            </div>
          );
        })}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', marginBottom: 10 }}>
        <button className="btn" onClick={() => setDrawSeed((s) => s + 1)}>
          Draw 10 samples
        </button>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
          {draws.map((d, i) => (
            <span
              key={i}
              className="mono"
              style={{
                fontSize: 12,
                padding: '2px 7px',
                borderRadius: 5,
                background: d.valid ? 'var(--bg-sunken)' : 'color-mix(in srgb, var(--c-bad) 18%, transparent)',
                border: `1px solid ${d.valid ? 'var(--border)' : 'var(--c-bad)'}`,
                color: d.valid ? 'var(--text)' : 'var(--c-bad)',
              }}
            >
              {d.word}
            </span>
          ))}
        </div>
      </div>

      <div className="live-note">
        {res.junkMass > 0.05 ? (
          <>
            <strong>Tail risk is high ({pct(res.junkMass)}):</strong> red bars are off-topic tail tokens. Over a 20-token reply at this per-step error rate, you are likely to sample at least one nonsensical token — and because generation is autoregressive, that derailment sticks.
          </>
        ) : res.poolSize === 1 ? (
          <>
            <strong>Deterministic ({ctx.tokens[0].word} = 100%):</strong> only one token survives. Great for factual prompts like “The capital of France is”, but on open-ended prose it produces identical, repetitive completions every run.
          </>
        ) : (
          <>
            <strong>Clean pool ({res.poolSize} tokens, {f2(res.effectiveChoices)} effective choices):</strong> {pct(res.validMass)} of the probability sits on sensible continuations, with {pct(res.junkMass)} tail-junk risk.
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Sharp vs Flat: Top-k vs Top-p

export function SharpVsFlat() {
  const [method, setMethod] = useState<'topk' | 'topp'>('topk');
  const [k, setK] = useState(5);
  const [p, setP] = useState(0.9);

  const sharpRes = useMemo(() => applySampling(CONTEXTS[0].tokens, 1.0, method === 'topk' ? k : 16, method === 'topp' ? p : 1.0), [method, k, p]);
  const flatRes = useMemo(() => applySampling(CONTEXTS[2].tokens, 1.0, method === 'topk' ? k : 16, method === 'topp' ? p : 1.0), [method, k, p]);

  const Panel = ({ ctx, res }: { ctx: (typeof CONTEXTS)[number]; res: ReturnType<typeof applySampling> }) => (
    <div
      className="widget-col"
      style={{
        flex: '1 1 260px',
        minWidth: 0,
        background: 'var(--bg-sunken)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        padding: '10px 12px',
      }}
    >
      <div className="field-label">{ctx.label}</div>
      <div className="mono" style={{ fontSize: 13, marginBottom: 8 }}>
        “{ctx.prompt} ___”
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: 8 }}>
        {ctx.tokens.map((tok, i) => {
          const kept = res.kept[i];
          const color = !tok.valid ? 'var(--c-bad)' : 'var(--c-a)';
          return (
            <span
              key={tok.word}
              className="mono"
              style={{
                fontSize: 11.5,
                padding: '2px 6px',
                borderRadius: 4,
                opacity: kept ? 1 : 0.28,
                textDecoration: kept ? 'none' : 'line-through',
                background: kept ? `color-mix(in srgb, ${color} 15%, transparent)` : 'transparent',
                border: `1px solid ${kept ? color : 'var(--border)'}`,
                color: kept && !tok.valid ? 'var(--c-bad)' : 'var(--text)',
              }}
            >
              {tok.word} {kept ? `${(res.probs[i] * 100).toFixed(0)}%` : ''}
            </span>
          );
        })}
      </div>
      <div className="mono" style={{ fontSize: 12, color: 'var(--text-muted)' }}>
        pool: <strong>{res.poolSize}</strong> · eff. choices: <strong>{f2(res.effectiveChoices)}</strong> · junk:{' '}
        <strong style={{ color: res.junkMass > 0.01 ? 'var(--c-bad)' : 'var(--c-good)' }}>{pct(res.junkMass)}</strong>
      </div>
    </div>
  );

  return (
    <Widget
      title="One setting, two prompts: why fixed top-k fails and top-p adapts"
      wide
      caption={
        <>
          A real LLM uses one sampler setting across thousands of positions — some sharply factual, some wide open. Compare <strong>Fixed top-k</strong> with <strong>Nucleus top-p</strong> at T = 1.0. Try top-k = 5, then switch to top-p = 0.90.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', marginBottom: 12 }}>
        <Segmented
          options={[
            { value: 'topk', label: 'Fixed top-k' },
            { value: 'topp', label: 'Nucleus top-p' },
          ]}
          value={method}
          onChange={setMethod}
        />
        <div style={{ flex: '1 1 220px' }}>
          {method === 'topk' ? (
            <Slider label="top-k (fixed pool size)" value={k} min={1} max={16} step={1} onChange={setK} format={(v) => `${v} tokens`} color="c-b" />
          ) : (
            <Slider label="top-p (cumulative mass)" value={p} min={0.1} max={1.0} step={0.05} onChange={setP} format={(v) => `${Math.round(v * 100)}%`} color="c-c" />
          )}
        </div>
      </div>

      <div className="widget-row">
        <Panel ctx={CONTEXTS[0]} res={sharpRes} />
        <Panel ctx={CONTEXTS[2]} res={flatRes} />
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {method === 'topk' ? (
          <>
            <strong>Fixed top-k = {k}:</strong> both prompts are forced to keep exactly {k} {k === 1 ? 'token' : 'tokens'}.{' '}
            {k === 1
              ? 'The sharp prompt is 100% accurate, but the open-ended prompt is crushed to 1 single word.'
              : k <= 6
                ? `The sharp prompt already lets ${k - 1} wrong words in (${pct(sharpRes.junkMass)} error), while the open-ended prompt chops off ${12 - k} perfectly valid nouns!`
                : `The open-ended prompt gets plenty of variety (${f2(flatRes.effectiveChoices)} choices), but the sharp prompt now lets ${k - 1} wrong tokens into the pool (${pct(sharpRes.junkMass)} error).`}
          </>
        ) : (
          <>
            <strong>Nucleus top-p = {p.toFixed(2)}:</strong> the pool size adapts automatically to confidence —{' '}
            <strong>{sharpRes.poolSize} {sharpRes.poolSize === 1 ? 'token' : 'tokens'}</strong> on the sharp factual prompt ({pct(sharpRes.validMass)} valid) and{' '}
            <strong>{flatRes.poolSize} tokens</strong> on the open-ended prompt ({f2(flatRes.effectiveChoices)} effective choices, {pct(flatRes.junkMass)} junk).
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Beam Search vs Greedy

export function BeamVsGreedy() {
  const [beamWidth, setBeamWidth] = useState<1 | 2 | 4>(1);
  const [step, setStep] = useState(3);

  const activeAtStep = useMemo(() => {
    const sets: Set<string>[] = [];
    for (let s = 1; s <= 3; s++) {
      const paths = runBeamSearch(beamWidth, s);
      sets.push(new Set(paths.map((p) => p.words.join(' '))));
    }
    return sets;
  }, [beamWidth]);

  const finalPaths = useMemo(() => runBeamSearch(beamWidth, step), [beamWidth, step]);
  const winner = finalPaths[0];

  // Fixed clean tree coordinates inside 560 x 260 SVG
  const root = { x: 44, y: 130 };
  const s1Nodes = BEAM_TREE.map((n, i) => ({
    key: n.word,
    word: n.word,
    prob: n.prob,
    cum: n.prob,
    x: 165,
    y: i === 0 ? 68 : 192,
    children: n.children!,
  }));
  const s2Nodes = s1Nodes.flatMap((p, pi) =>
    p.children.map((c, ci) => ({
      parent: p,
      key: `${p.key} ${c.word}`,
      word: c.word,
      prob: c.prob,
      cum: p.cum * c.prob,
      x: 315,
      y: pi * 124 + ci * 62 + 37,
      children: c.children!,
    })),
  );
  const s3Nodes = s2Nodes.flatMap((p, pi) =>
    p.children.map((c, ci) => ({
      parent: p,
      key: `${p.key} ${c.word}`,
      word: c.word,
      prob: c.prob,
      cum: p.cum * c.prob,
      x: 472,
      y: pi * 62 + ci * 30 + 22,
    })),
  );

  const isBest = (key: string) => winner.words.join(' ') === key;

  return (
    <Widget
      title="Why greedy misses the highest-probability sentence"
      wide
      caption={
        <>
          Each branch shows a word and its conditional probability; leaf percentages show the <em>whole-sentence</em> probability (product along the path). Compare <strong>Beam = 1 (greedy)</strong> with <strong>Beam = 2</strong> and step through the tree.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <Segmented
          label="beam width"
          options={[
            { value: 1 as const, label: '1 (greedy)' },
            { value: 2 as const, label: '2 beams' },
            { value: 4 as const, label: '4 beams' },
          ]}
          value={beamWidth}
          onChange={setBeamWidth}
        />
        <Segmented
          label="step"
          options={[
            { value: 1, label: 'Step 1' },
            { value: 2, label: 'Step 2' },
            { value: 3, label: 'Step 3' },
          ]}
          value={step}
          onChange={setStep}
        />
      </div>

      <svg
        viewBox="0 0 560 264"
        width={560}
        height={264}
        role="img"
        aria-label="Decision tree comparing greedy decoding with beam search"
        style={{ width: '100%', height: 'auto', maxWidth: 620, display: 'block', margin: '0 auto' }}
      >
        {/* Root to step 1 */}
        {s1Nodes.map((n) => {
          const alive = activeAtStep[0].has(n.key);
          return (
            <g key={n.key} opacity={alive ? 1 : 0.28}>
              <line
                x1={root.x}
                y1={root.y}
                x2={n.x}
                y2={n.y}
                stroke={alive ? 'var(--accent)' : 'var(--text-faint)'}
                strokeWidth={alive ? 2.4 : 1.3}
                strokeDasharray={alive ? undefined : '4 3'}
              />
              <text x={(root.x + n.x) / 2 - 4} y={(root.y + n.y) / 2 - 6} fontSize={11} fontFamily="var(--font-mono)" fill="var(--text-muted)">
                {n.prob.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Step 1 to step 2 */}
        {step >= 2 &&
          s2Nodes.map((n) => {
            const pAlive = activeAtStep[0].has(n.parent.key);
            const alive = pAlive && activeAtStep[1].has(n.key);
            return (
              <g key={n.key} opacity={alive ? 1 : pAlive ? 0.35 : 0.16}>
                <line
                  x1={n.parent.x}
                  y1={n.parent.y}
                  x2={n.x}
                  y2={n.y}
                  stroke={alive ? 'var(--accent)' : 'var(--text-faint)'}
                  strokeWidth={alive ? 2.2 : 1.2}
                  strokeDasharray={alive ? undefined : '4 3'}
                />
                <text x={(n.parent.x + n.x) / 2 - 6} y={(n.parent.y + n.y) / 2 - 4} fontSize={10.5} fontFamily="var(--font-mono)" fill="var(--text-muted)">
                  {n.prob.toFixed(2)}
                </text>
              </g>
            );
          })}

        {/* Step 2 to step 3 */}
        {step >= 3 &&
          s3Nodes.map((n) => {
            const pAlive = activeAtStep[0].has(n.parent.parent.key) && activeAtStep[1].has(n.parent.key);
            const alive = pAlive && activeAtStep[2].has(n.key);
            const best = isBest(n.key);
            return (
              <g key={n.key} opacity={alive ? 1 : pAlive ? 0.35 : 0.14}>
                <line
                  x1={n.parent.x}
                  y1={n.parent.y}
                  x2={n.x}
                  y2={n.y}
                  stroke={best ? 'var(--c-good)' : alive ? 'var(--accent)' : 'var(--text-faint)'}
                  strokeWidth={best ? 3 : alive ? 2 : 1.1}
                  strokeDasharray={alive ? undefined : '3 3'}
                />
                <text x={(n.parent.x + n.x) / 2 - 6} y={(n.parent.y + n.y) / 2 - 3} fontSize={10} fontFamily="var(--font-mono)" fill="var(--text-muted)">
                  {n.prob.toFixed(2)}
                </text>
              </g>
            );
          })}

        {/* Nodes */}
        <circle cx={root.x} cy={root.y} r={16} fill="var(--bg-sunken)" stroke="var(--border-strong)" strokeWidth={1.5} />
        <text x={root.x} y={root.y + 4} textAnchor="middle" fontSize={10.5} fontFamily="var(--font-mono)" fill="var(--text-muted)">
          start
        </text>

        {s1Nodes.map((n) => {
          const alive = activeAtStep[0].has(n.key);
          return (
            <g key={n.key} opacity={alive ? 1 : 0.35}>
              <rect x={n.x - 24} y={n.y - 13} width={48} height={26} rx={6} fill="var(--bg-elev)" stroke={alive ? 'var(--accent)' : 'var(--border)'} strokeWidth={alive ? 1.8 : 1} />
              <text x={n.x} y={n.y + 4} textAnchor="middle" fontSize={12} fontWeight={alive ? 700 : 500} fontFamily="var(--font-mono)" fill="var(--text)">
                {n.word}
              </text>
            </g>
          );
        })}

        {step >= 2 &&
          s2Nodes.map((n) => {
            const pAlive = activeAtStep[0].has(n.parent.key);
            const alive = pAlive && activeAtStep[1].has(n.key);
            return (
              <g key={n.key} opacity={alive ? 1 : pAlive ? 0.38 : 0.18}>
                <rect x={n.x - 30} y={n.y - 12} width={60} height={24} rx={6} fill="var(--bg-elev)" stroke={alive ? 'var(--accent)' : 'var(--border)'} strokeWidth={alive ? 1.8 : 1} />
                <text x={n.x} y={n.y + 4} textAnchor="middle" fontSize={11.5} fontWeight={alive ? 700 : 500} fontFamily="var(--font-mono)" fill="var(--text)">
                  {n.word}
                </text>
              </g>
            );
          })}

        {step >= 3 &&
          s3Nodes.map((n) => {
            const pAlive = activeAtStep[0].has(n.parent.parent.key) && activeAtStep[1].has(n.parent.key);
            const alive = pAlive && activeAtStep[2].has(n.key);
            const best = isBest(n.key);
            return (
              <g key={n.key} opacity={alive ? 1 : pAlive ? 0.38 : 0.16}>
                <rect
                  x={n.x - 30}
                  y={n.y - 11}
                  width={60}
                  height={22}
                  rx={5}
                  fill={best ? 'color-mix(in srgb, var(--c-good) 20%, var(--bg-elev))' : 'var(--bg-elev)'}
                  stroke={best ? 'var(--c-good)' : alive ? 'var(--accent)' : 'var(--border)'}
                  strokeWidth={best ? 2 : 1.2}
                />
                <text x={n.x} y={n.y + 4} textAnchor="middle" fontSize={11} fontWeight={best ? 700 : 500} fontFamily="var(--font-mono)" fill="var(--text)">
                  {n.word}
                </text>
                <text x={n.x + 36} y={n.y + 4} fontSize={10.5} fontWeight={best ? 700 : 400} fontFamily="var(--font-mono)" fill={best ? 'var(--c-good)' : 'var(--text-muted)'}>
                  {(n.cum * 100).toFixed(1)}%
                </text>
              </g>
            );
          })}
      </svg>

      <div className="live-note" style={{ marginTop: 10 }}>
        {beamWidth === 1 ? (
          <>
            <strong>Greedy (beam = 1) picks “{winner.words.join(' ')}” ({(winner.prob * 100).toFixed(1)}%).</strong> Because “The” edged out “A” at step 1 (0.55 vs 0.45), greedy pruned the entire lower half of the tree forever — never seeing “A bright sun” at <strong>34.4%</strong> (nearly 4× higher probability!).
          </>
        ) : (
          <>
            <strong>Beam = {beamWidth} finds “{winner.words.join(' ')}” ({(winner.prob * 100).toFixed(1)}%).</strong> By keeping the top {beamWidth} partial sentences alive at each step instead of committing immediately, beam search recovers the 0.45 × 0.85 × 0.90 = <strong>34.4%</strong> path — 3.9× more likely than greedy's 8.8%.
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Tune the Sampler

const SAMPLER_LAB_KEY = 'nl-sampling-lab-v1';

function loadSolved(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(SAMPLER_LAB_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    // ignore
  }
  return {};
}

export function SamplerLab() {
  const [levelIdx, setLevelIdx] = useState(0);
  const [T, setT] = useState(1.0);
  const [topK, setTopK] = useState(16);
  const [topP, setTopP] = useState(1.0);
  const [solved, setSolved] = useState<Record<string, boolean>>(loadSolved);
  const [declaredImpossible, setDeclaredImpossible] = useState(false);

  const level = LAB_LEVELS[levelIdx];
  const stats = useMemo(() => solveLabLevel(level), [level]);

  const effT = level.lockTemp ?? T;
  const effK = level.lockK ?? topK;
  const effP = level.lockP ?? topP;
  const m = useMemo(() => evaluateSampler(effT, effK, effP), [effT, effK, effP]);
  const passesNow = level.check(m);

  const markSolved = (id: string) => {
    setSolved((prev) => {
      if (prev[id]) return prev;
      const next = { ...prev, [id]: true };
      try {
        localStorage.setItem(SAMPLER_LAB_KEY, JSON.stringify(next));
      } catch {
        // ignore
      }
      return next;
    });
  };

  useEffect(() => {
    if (passesNow) markSolved(level.id);
  }, [passesNow, level.id]);

  const switchLevel = (i: number) => {
    setLevelIdx(i);
    setDeclaredImpossible(false);
    setT(1.0);
    setTopK(16);
    setTopP(1.0);
  };

  const optScore = stats.best ? stats.best.metrics.flatEff : null;
  const nearOpt = passesNow && optScore !== null && m.flatEff >= optScore - 0.05;

  return (
    <Widget
      title="Lab: Tune the sampler"
      wide
      caption={
        <>
          Three engineering challenges evaluated simultaneously across our sharp, medium and flat prompts. Every level’s passing count and optimal score are computed exhaustively over the slider grid (6,080 settings when all three knobs are unlocked).
        </>
      }
    >
      <div style={{ marginBottom: 12 }}>
        <Segmented
          label="challenge"
          options={LAB_LEVELS.map((lv, i) => ({
            value: i,
            label: `${solved[lv.id] ? '✓ ' : ''}${lv.name}`,
          }))}
          value={levelIdx}
          onChange={switchLevel}
        />
      </div>

      <div
        style={{
          padding: '10px 12px',
          borderRadius: 'var(--radius-sm)',
          background: 'var(--bg-sunken)',
          border: '1px solid var(--border)',
          marginBottom: 12,
          fontSize: 13.5,
        }}
      >
        <strong>Brief:</strong> {level.brief}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 12, marginBottom: 12 }}>
        <div style={{ opacity: level.lockTemp !== undefined ? 0.5 : 1, pointerEvents: level.lockTemp !== undefined ? 'none' : 'auto' }}>
          <Slider
            label={level.lockTemp !== undefined ? 'temperature T (locked)' : 'temperature T'}
            value={effT}
            min={0.1}
            max={2.0}
            step={0.1}
            onChange={setT}
            format={(v) => v.toFixed(1)}
            color="accent"
          />
        </div>
        <div style={{ opacity: level.lockK !== undefined ? 0.5 : 1, pointerEvents: level.lockK !== undefined ? 'none' : 'auto' }}>
          <Slider
            label={level.lockK !== undefined ? 'top-k (locked)' : 'top-k'}
            value={effK}
            min={1}
            max={16}
            step={1}
            onChange={setTopK}
            format={(v) => `${v}`}
            color="c-a"
          />
        </div>
        <div style={{ opacity: level.lockP !== undefined ? 0.5 : 1, pointerEvents: level.lockP !== undefined ? 'none' : 'auto' }}>
          <Slider
            label={level.lockP !== undefined ? 'top-p (locked)' : 'top-p'}
            value={effP}
            min={0.1}
            max={1.0}
            step={0.05}
            onChange={setTopP}
            format={(v) => v.toFixed(2)}
            color="c-c"
          />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 8, marginBottom: 12 }}>
        {[
          {
            label: 'sharp accuracy',
            val: pct(m.sharpAcc),
            target: level.id === 'factual' ? '≥ 99%' : '≥ 98%',
            ok: level.id === 'factual' ? m.sharpAcc >= 0.99 : m.sharpAcc >= 0.98,
          },
          {
            label: 'flat eff. choices',
            val: f2(m.flatEff),
            target: level.id === 'factual' ? 'bonus: maximise' : level.id === 'topk-trap' ? '≥ 9.0' : '≥ 9.5',
            ok: level.id === 'factual' ? true : level.id === 'topk-trap' ? m.flatEff >= 9.0 : m.flatEff >= 9.5,
          },
          {
            label: 'max tail junk',
            val: pct(m.maxTailJunk),
            target: level.id === 'topk-trap' ? 'unconstrained' : '0.0%',
            ok: level.id === 'topk-trap' ? true : m.maxTailJunk === 0,
          },
        ].map((card) => (
          <div
            key={card.label}
            style={{
              background: 'var(--bg-sunken)',
              border: `1px solid ${card.ok ? 'var(--c-good)' : 'var(--border)'}`,
              borderRadius: 'var(--radius-sm)',
              padding: '8px 10px',
            }}
          >
            <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{card.label}</div>
            <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: card.ok ? 'var(--c-good)' : 'var(--c-bad)' }}>
              {card.val}
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>
              {card.ok ? '✓ ' : '○ '}target: {card.target}
            </div>
          </div>
        ))}
      </div>

      {level.impossible && !declaredImpossible && (
        <div className="btn-row" style={{ marginBottom: 10 }}>
          <button
            className="btn primary"
            onClick={() => {
              setDeclaredImpossible(true);
              markSolved(level.id);
            }}
          >
            Declare this impossible
          </button>
        </div>
      )}

      <div className="live-note">
        {level.impossible && declaredImpossible ? (
          <>
            <strong>Correct — 0 of 16 values of top-k work!</strong> At T = 1.0, reaching ≥ 98% sharp accuracy requires <code>k ≤ 2</code> (which limits flat diversity to at most 2.00 choices). Reaching ≥ 9.0 flat choices requires <code>k ≥ 10</code> (where sharp accuracy has already fallen to 91.9%). That exact tension is why fixed top-k cannot handle both sharp and flat contexts at once — try Level 3 with top-p unlocked!
          </>
        ) : passesNow ? (
          <>
            <strong>
              {nearOpt ? 'Exhaustive optimum matched!' : 'Challenge solved!'} ({stats.passing.toLocaleString('en-GB')} of {stats.gridSize.toLocaleString('en-GB')} grid settings pass.)
            </strong>{' '}
            Your setting gives <strong>{f2(m.flatEff)}</strong> effective flat choices at {pct(m.sharpAcc)} sharp accuracy and 0.0% tail junk.
            {stats.best && (
              <>
                {' '}
                Exhaustive grid optimum for maximum open-ended diversity under these constraints: <strong>{f2(stats.best.metrics.flatEff)}</strong> choices (e.g. T={stats.best.T.toFixed(1)}, k={stats.best.k}, p={stats.best.p.toFixed(2)}).
              </>
            )}
            {level.id === 'balanced' && (
              <> Notice: 0 temperature-only settings and 0 top-k-only settings pass Level 3 — nucleus top-p is what unlocks it.</>
            )}
          </>
        ) : (
          <>
            <strong>Keep tuning</strong> (sharp pool: {m.sharpPool} {m.sharpPool === 1 ? 'token' : 'tokens'}, flat pool: {m.flatPool} tokens).{' '}
            {m.maxTailJunk > 0 && level.id !== 'topk-trap'
              ? 'Remember: softmax is strictly positive on every token, so temperature alone can never reach exact 0.0% tail junk — you need top-p or top-k to cut the tail.'
              : level.impossible
                ? 'Sweep top-k from 1 to 16 and watch sharp accuracy and flat choices pull in opposite directions. If no setting works, declare it impossible.'
                : 'Try adjusting top-p (around 0.85–0.95) together with temperature to keep sharp focused and flat wide.'}
          </>
        )}
      </div>
    </Widget>
  );
}
