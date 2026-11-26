// authored-by: Gemini Antigravity
import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { STREAM_LABELS, traceSentence, VOCAB } from './sentenceMicro';

const SHORT_CHANNELS = ['subj', 'verb', 'prep', 'pos'] as const;

function StreamRow({
  title,
  badge,
  values,
  highlight = false,
  dim = false,
}: {
  title: string;
  badge?: string;
  values: readonly number[];
  highlight?: boolean;
  dim?: boolean;
}) {
  const maxAbs = 3.2;
  return (
    <div
      style={{
        padding: '7px 10px',
        borderRadius: 8,
        border: highlight ? '1.5px solid var(--accent)' : '1px solid var(--border)',
        background: highlight ? 'var(--bg-elev)' : 'var(--bg-sunken)',
        opacity: dim ? 0.55 : 1,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, marginBottom: 5, flexWrap: 'wrap' }}>
        <span className="mono" style={{ fontSize: 11.5, fontWeight: highlight ? 700 : 600, color: highlight ? 'var(--accent-text)' : 'var(--text)' }}>
          {title}
        </span>
        {badge && (
          <span className="mono" style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>
            {badge}
          </span>
        )}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 6 }}>
        {values.map((v, i) => {
          const pct = Math.min(100, (Math.abs(v) / maxAbs) * 100);
          return (
            <div key={i} style={{ display: 'grid', gap: 2 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)' }} className="mono">
                <span>{SHORT_CHANNELS[i]}</span>
                <span style={{ color: Math.abs(v) < 0.05 ? 'var(--text-faint)' : 'var(--text)', fontWeight: Math.abs(v) >= 0.4 ? 700 : 400 }}>
                  {v >= 0 ? '+' : '−'}{Math.abs(v).toFixed(2)}
                </span>
              </div>
              <div style={{ height: 7, borderRadius: 4, background: 'var(--border)', overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${pct.toFixed(1)}%`,
                    height: '100%',
                    background: v >= 0 ? 'var(--c-a)' : 'var(--c-b)',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function SentenceMicroscope() {
  const [selectedPos, setSelectedPos] = useState(3); // default: "on" -> predicts "mat"
  const [useAttention, setUseAttention] = useState(true);
  const [useResidual, setUseResidual] = useState(true);
  const [usePositions, setUsePositions] = useState(true);

  const trace = useMemo(
    () => traceSentence({ useAttention, useResidual, usePositions }),
    [useAttention, useResidual, usePositions],
  );

  const tok = trace.tokens[selectedPos];
  const topIdx = tok.probs.indexOf(Math.max(...tok.probs));
  const topWord = VOCAB[topIdx];
  const topMatches = topIdx === tok.targetId;

  const h1Row = trace.attnWeights[0][selectedPos];
  const h2Row = trace.attnWeights[1][selectedPos];
  const h1Top = h1Row.indexOf(Math.max(...h1Row));
  const h2Top = h2Row.indexOf(Math.max(...h2Row));

  const anyAblated = !useAttention || !useResidual || !usePositions;

  let liveNote: string;
  if (selectedPos === 3) {
    if (useAttention && useResidual && usePositions) {
      liveNote = `Full block at “on”: Head 1 fetches “cat” (${Math.round(h1Row[1] * 100)}%) into the subject channel, Head 2 fetches “sat” (${Math.round(h2Row[2] * 100)}%) into the verb channel, the residual skip preserves “on” in the prep channel, and position marks the phrase end — so the MLP’s conjunction fires and puts “mat” at ${(tok.probs[4] * 100).toFixed(1)}%.`;
    } else if (!useAttention) {
      liveNote = `Attention OFF at “on”: the token is sealed in a box — it can only see its own word and position, never “cat” or “sat”. Subject and verb channels stay 0, so p(“mat”) plunges to ${(tok.probs[4] * 100).toFixed(1)}% and the top guess flips to “${topWord}”.`;
    } else if (!useResidual) {
      liveNote = `Residual skips OFF at “on”: attention’s output overwrites “on”’s own embedding instead of adding to it, wiping out the preposition cue (and the MLP overwrites x₁). Without all four clues preserved together, p(“mat”) collapses to ${(tok.probs[4] * 100).toFixed(1)}% (top guess: “${topWord}”).`;
    } else {
      liveNote = `Positions OFF at “on”: without positional encodings, the sequence becomes an order-blind bag of earlier words. Attention focus softens (Head 1 → “cat” ${Math.round(h1Row[1] * 100)}%, Head 2 → “sat” ${Math.round(h2Row[2] * 100)}%) and p(“mat”) drops to ${(tok.probs[4] * 100).toFixed(1)}% (top guess: “${topWord}”).`;
    }
  } else {
    liveNote = `Inspecting token ${selectedPos} (“${tok.word}”): because of the causal mask, it can only attend to positions 0..${selectedPos}. Its stream currently gives top prediction “${topWord}” (${(tok.probs[topIdx] * 100).toFixed(1)}%), while the expected next word “${tok.targetWord}” has ${(tok.probs[tok.targetId] * 100).toFixed(1)}%.`;
  }

  return (
    <Widget
      title="Sentence microscope: follow one 4-number residual stream from word to prediction"
      caption="Click any word in “The cat sat on” to inspect its 4-number residual stream step-by-step (embedding → +position → +attention → +MLP → logits & p − y). Then switch off Attention, Residual skips, or Positions to see why every piece matters."
    >
      {/* Clickable 4-token sentence strip */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          flexWrap: 'wrap',
          marginBottom: 12,
          padding: '8px 10px',
          borderRadius: 10,
          border: '1px solid var(--border)',
          background: 'var(--bg-sunken)',
        }}
      >
        <span className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)', marginRight: 2 }}>
          Prompt:
        </span>
        {trace.tokens.map((t) => {
          const active = t.pos === selectedPos;
          return (
            <button
              key={t.pos}
              type="button"
              onClick={() => setSelectedPos(t.pos)}
              aria-pressed={active}
              className={`btn small${active ? ' primary' : ''}`}
              style={{ fontFamily: 'var(--font-mono)', minWidth: 52 }}
            >
              <span style={{ opacity: 0.65, fontSize: 10.5, marginRight: 4 }}>{t.pos}</span>
              {t.word}
            </button>
          );
        })}
        <span className="mono" style={{ color: 'var(--text-muted)', fontSize: 13 }}>→</span>
        <span
          className="mono"
          style={{
            fontSize: 12.5,
            fontWeight: 700,
            padding: '4px 10px',
            borderRadius: 999,
            border: `1.5px solid ${topMatches ? 'var(--c-c)' : 'var(--c-bad)'}`,
            color: topMatches ? 'var(--c-c)' : 'var(--c-bad)',
            background: 'var(--bg-elev)',
          }}
        >
          “{topWord}” ({(tok.probs[topIdx] * 100).toFixed(1)}%)
        </span>
      </div>

      {/* Ablation switches */}
      <div className="btn-row" style={{ marginBottom: 14 }}>
        <button
          type="button"
          className={`btn small${useAttention ? ' primary' : ''}`}
          aria-pressed={useAttention}
          onClick={() => setUseAttention((v) => !v)}
        >
          Attention: {useAttention ? 'ON' : 'off'}
        </button>
        <button
          type="button"
          className={`btn small${useResidual ? ' primary' : ''}`}
          aria-pressed={useResidual}
          onClick={() => setUseResidual((v) => !v)}
        >
          Residual skips: {useResidual ? 'ON' : 'off'}
        </button>
        <button
          type="button"
          className={`btn small${usePositions ? ' primary' : ''}`}
          aria-pressed={usePositions}
          onClick={() => setUsePositions((v) => !v)}
        >
          Positions: {usePositions ? 'ON' : 'off'}
        </button>
        {anyAblated && (
          <button
            type="button"
            className="btn small"
            onClick={() => {
              setUseAttention(true);
              setUseResidual(true);
              setUsePositions(true);
            }}
          >
            Reset all ON
          </button>
        )}
      </div>

      {/* Two responsive columns: Left = residual-stream waterfall, Right = next-token probs & p - y */}
      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        {/* Left column: 5-stage residual-stream waterfall */}
        <div style={{ flex: '1 1 280px', minWidth: 260, display: 'grid', gap: 7 }}>
          <div className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 6 }}>
            <span>RESIDUAL STREAM FOR “{tok.word}” (pos {tok.pos})</span>
            <span>d = 4 channels</span>
          </div>

          <StreamRow title="1. Token embedding eₜ" badge={`lookup("${tok.word}")`} values={tok.emb} />
          <StreamRow
            title="2. + Position pₜ"
            badge={usePositions ? `slot ${tok.pos}` : 'OFF (zeros)'}
            values={tok.posVec}
            dim={!usePositions}
          />
          <StreamRow
            title={useResidual ? '3. + Attention Δ (2 heads)' : '3. Attention replaces x₀!'}
            badge={
              useAttention
                ? `H1: ${Math.round(h1Row[h1Top] * 100)}% “${trace.tokens[h1Top].word}” · H2: ${Math.round(h2Row[h2Top] * 100)}% “${trace.tokens[h2Top].word}”`
                : 'OFF (no cross-token read)'
            }
            values={tok.attnOut}
            dim={!useAttention}
          />
          <StreamRow
            title={useResidual ? '4. + MLP Δ (4→8→4 ReLU)' : '4. MLP replaces x₁!'}
            badge={`n₀(conj) = ${tok.mlpHidden[0].toFixed(2)}`}
            values={tok.mlpOut}
          />
          <StreamRow
            title="5. = Final stream x₂"
            badge={useResidual ? 'x₀ + attn + MLP' : 'no skips (MLP only)'}
            values={tok.x2}
            highlight
          />

          <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-muted)', display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 2 }}>
            {STREAM_LABELS.map((lbl) => (
              <span key={lbl}>{lbl}</span>
            ))}
          </div>
        </div>

        {/* Right column: Unembedding logits, softmax probabilities, and exact p - y gradient */}
        <div style={{ flex: '1 1 240px', minWidth: 240, display: 'grid', gap: 10 }}>
          <div style={{ padding: '10px 12px', border: '1px solid var(--border)', borderRadius: 10, background: 'var(--bg-sunken)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <span className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                NEXT-TOKEN PROBS &amp; ∂L/∂z = p − y
              </span>
              <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                loss {tok.loss.toFixed(2)}
              </span>
            </div>

            <div style={{ display: 'grid', gap: 7 }}>
              {VOCAB.map((word, idx) => {
                const p = tok.probs[idx];
                const g = tok.dLogits[idx];
                const isTarget = idx === tok.targetId;
                const isTop = idx === topIdx;
                return (
                  <div key={word} style={{ display: 'grid', gap: 3 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', fontSize: 12.5, gap: 6 }}>
                      <span className="mono" style={{ fontWeight: isTarget || isTop ? 700 : 400, color: isTarget ? 'var(--c-c)' : 'var(--text)' }}>
                        {word}
                        {isTarget && <span style={{ fontSize: 10, marginLeft: 5, color: 'var(--c-c)' }}>target</span>}
                      </span>
                      <span className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                        <strong style={{ color: 'var(--text)' }}>{(p * 100).toFixed(1)}%</strong>
                        {' · '}
                        <span style={{ color: g < 0 ? 'var(--c-c)' : 'var(--c-grad)' }}>
                          p−y {g >= 0 ? '+' : '−'}{Math.abs(g).toFixed(2)}
                        </span>
                      </span>
                    </div>
                    <div style={{ height: 8, borderRadius: 4, background: 'var(--border)', overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${Math.max(1.5, p * 100).toFixed(1)}%`,
                          height: '100%',
                          background: isTarget ? 'var(--c-c)' : isTop ? 'var(--c-b)' : 'var(--c-a)',
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Compact causal attention breakdown for selected token */}
          <div style={{ padding: '9px 12px', border: '1px solid var(--border)', borderRadius: 10, background: 'var(--bg-sunken)' }}>
            <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
              WHERE “{tok.word}” LOOKS (CAUSAL MASK: 0..{tok.pos})
            </div>
            {[
              { label: 'Head 1 (subject)', row: h1Row },
              { label: 'Head 2 (verb)', row: h2Row },
            ].map((hd) => (
              <div key={hd.label} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4, flexWrap: 'wrap' }}>
                <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', minWidth: 110 }}>
                  {hd.label}
                </span>
                {trace.tokens.map((src) => {
                  const masked = src.pos > selectedPos;
                  const w = useAttention ? hd.row[src.pos] : 0;
                  return (
                    <span
                      key={src.pos}
                      className="mono"
                      style={{
                        fontSize: 11,
                        padding: '1px 6px',
                        borderRadius: 5,
                        border: '1px solid var(--border)',
                        background: masked
                          ? 'transparent'
                          : `color-mix(in srgb, var(--accent) ${Math.round(w * 45)}%, var(--bg-elev))`,
                        color: masked ? 'var(--text-faint)' : 'var(--text)',
                        fontWeight: w >= 0.45 ? 700 : 400,
                      }}
                    >
                      {src.word} {masked ? '×' : `${Math.round(w * 100)}%`}
                    </span>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>

      <p className="live-note" style={{ marginTop: 12 }}>
        {liveNote}
      </p>
    </Widget>
  );
}
