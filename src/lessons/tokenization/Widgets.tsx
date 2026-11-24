// authored-by: Neon High
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider, Toggle } from '../../components/controls';
import { Plot } from '../../components/plot';
import {
  CORPUS,
  LANGUAGE_SAMPLES,
  applyMergeToWords,
  bestMdl,
  encodeWord,
  encodeWordTrace,
  mdlStep,
  topPairs,
  train,
  utf8Stats,
  vocabIds,
  wordsOf,
  type Step,
} from './bpe';

const TRAINED = train(CORPUS);
const OPT = bestMdl(TRAINED.steps);
const CORPUS_LINES = CORPUS.split('\n').map((line) => line.split(/\s+/).filter(Boolean));
const INIT_WORDS = wordsOf(CORPUS);
const CHIP_VARS = ['var(--c-a)', 'var(--c-b)', 'var(--c-c)', 'var(--accent)', 'var(--c-grad)'];
const LAB_STORAGE = 'nl-tok-lab-v1';

const fmt = (n: number) => Math.round(n).toLocaleString('en-GB');
const f1 = (n: number) => n.toFixed(1);

function TokenChip({ text, highlight, tintIndex = 0, sub }: { text: string; highlight?: boolean; tintIndex?: number; sub?: string }) {
  const color = highlight ? 'var(--accent)' : CHIP_VARS[tintIndex % CHIP_VARS.length];
  return (
    <span
      className="mono"
      style={{
        display: 'inline-flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: sub ? '3px 7px 2px' : '2px 5px',
        borderRadius: 5,
        fontSize: sub ? 13.5 : 12.5,
        lineHeight: 1.25,
        background: `color-mix(in srgb, ${color} ${highlight ? 26 : 12}%, transparent)`,
        border: `1px solid color-mix(in srgb, ${color} ${highlight ? 70 : 32}%, transparent)`,
        color: 'var(--text)',
        fontWeight: highlight ? 700 : 500,
      }}
    >
      <span>
        {text.endsWith('_') ? (
          <>
            {text.slice(0, -1)}
            <span style={{ opacity: 0.45 }}>_</span>
          </>
        ) : (
          text
        )}
      </span>
      {sub && (
        <span style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 400, marginTop: 1 }}>
          {sub}
        </span>
      )}
    </span>
  );
}

// ------------------------------------------------------------------ 1. BPE step-through

export function BpeStepper() {
  const [k, setK] = useState(0);
  const step = TRAINED.steps[k];
  const latest = k > 0 ? TRAINED.merges[k - 1] : null;
  const nextPairs = useMemo(() => topPairs(TRAINED.snapshots[k], 6), [k]);
  const attnPct = ((step.tokens / TRAINED.steps[0].tokens) ** 2) * 100;

  const note = (() => {
    if (k === 0) {
      return (
        <>
          <strong>Character level (0 merges).</strong> Every word is split into single letters plus the end-of-word marker <code>_</code>: 531 tokens from a tiny vocabulary of 22 symbols. Notice the runner-up table: <code>a + t</code> appears 27 times, more than any other pair.
        </>
      );
    }
    if (k === 4) {
      return (
        <>
          <strong>Merge #4: <code>th + e_ → the_</code> (23 times).</strong> In four merges, the most common English word has become a single token. The corpus has already shrunk from 531 to {step.tokens} tokens.
        </>
      );
    }
    if (k === 14) {
      return (
        <>
          <strong>Merge #14: <code>e + st_ → est_</code> (7 times).</strong> Nobody taught the algorithm grammar, yet it just invented a reusable superlative suffix shared by <em>fastest</em>, <em>slowest</em>, <em>oldest</em>, <em>youngest</em> and <em>warmest</em>.
        </>
      );
    }
    if (k === OPT.merges) {
      return (
        <>
          <strong>Merge #{OPT.merges}: the description-length sweet spot.</strong> Tokens have been cut in half (531 → {step.tokens}), which quarters n² attention work to {attnPct.toFixed(0)}%. Beyond this point, every remaining pair appears only 2 or 3 times.
        </>
      );
    }
    if (k === TRAINED.merges.length) {
      return (
        <>
          <strong>Merge #{k}: BPE stops.</strong> No adjacent pair appears more than once anywhere in the corpus. Tokens are down to {step.tokens} (a third of character level, {attnPct.toFixed(0)}% of the n² attention pairs), at the cost of a vocabulary of {step.vocab} symbols.
        </>
      );
    }
    return (
      <>
        <strong>
          Merge #{k}: <code>{latest!.pair[0]} + {latest!.pair[1]} → {latest!.merged}</code>
        </strong>{' '}
        (seen {latest!.count} times). Highlighted chips in the corpus just fused into one token, cutting the corpus to {step.tokens} tokens while raising the vocabulary to {step.vocab}.
      </>
    );
  })();

  return (
    <Widget
      title="Watching BPE learn its vocabulary"
      wide
      caption={
        <>
          Our 10-line corpus (107 words, 50 distinct). Step through byte-pair encoding one merge at a time. At every step, BPE finds the most frequent adjacent pair and fuses it into a new token. Jump to <strong>merge 4</strong> (<code>the_</code>), <strong>merge 14</strong> (<code>est_</code>), and <strong>merge 28</strong> (half the tokens).
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10 }}>
        <button className="btn" onClick={() => setK(Math.max(0, k - 1))} disabled={k === 0}>
          ◂ Back
        </button>
        <button className="btn primary" onClick={() => setK(Math.min(TRAINED.merges.length, k + 1))} disabled={k === TRAINED.merges.length}>
          Next merge ▸
        </button>
        <button className="btn" onClick={() => setK(OPT.merges)}>
          Jump to 28
        </button>
        <button className="btn" onClick={() => setK(TRAINED.merges.length)}>
          All {TRAINED.merges.length}
        </button>
        <button className="btn" onClick={() => setK(0)} disabled={k === 0}>
          Reset
        </button>
      </div>

      <Slider
        label="merges learned"
        value={k}
        min={0}
        max={TRAINED.merges.length}
        step={1}
        onChange={setK}
        format={(v) => `${v} / ${TRAINED.merges.length}`}
        color="accent"
      />

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
          gap: 8,
          margin: '12px 0',
        }}
      >
        {[
          { label: 'tokens in corpus', val: `${step.tokens}`, sub: `was 531`, color: 'var(--c-a)' },
          { label: 'vocabulary size', val: `${step.vocab}`, sub: `started at ${TRAINED.initialVocabSize}`, color: 'var(--c-b)' },
          { label: 'n² attention work', val: `${attnPct.toFixed(0)}%`, sub: `vs char level`, color: 'var(--c-c)' },
          {
            label: 'latest merge',
            val: latest ? `${latest.pair[0]}+${latest.pair[1]}` : 'none yet',
            sub: latest ? `→ ${latest.merged} (×${latest.count})` : '22 base symbols',
            color: 'var(--accent)',
          },
        ].map((m) => (
          <div
            key={m.label}
            style={{
              background: 'var(--bg-sunken)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              padding: '8px 10px',
            }}
          >
            <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{m.label}</div>
            <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: m.color, marginTop: 2 }}>
              {m.val}
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>{m.sub}</div>
          </div>
        ))}
      </div>

      <div className="widget-row">
        <div className="widget-col" style={{ flex: '2 1 360px', minWidth: 0 }}>
          <div className="field-label">corpus after {k} {k === 1 ? 'merge' : 'merges'}</div>
          <div
            style={{
              background: 'var(--bg-sunken)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              padding: '10px 12px',
              display: 'flex',
              flexDirection: 'column',
              gap: 6,
              maxHeight: 260,
              overflowY: 'auto',
            }}
          >
            {CORPUS_LINES.map((words, li) => (
              <div key={li} style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 8px', alignItems: 'center' }}>
                {words.map((w, wi) => {
                  const toks = encodeWord(w, TRAINED.merges, k);
                  return (
                    <span key={wi} style={{ display: 'inline-flex', gap: 2, flexWrap: 'wrap' }}>
                      {toks.map((t, ti) => (
                        <TokenChip key={ti} text={t} highlight={latest !== null && t === latest.merged} tintIndex={ti} />
                      ))}
                    </span>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        <div className="widget-col" style={{ flex: '1 1 210px', minWidth: 0 }}>
          <div className="field-label">most frequent pairs right now</div>
          <div
            style={{
              background: 'var(--bg-sunken)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              padding: '8px 10px',
              display: 'grid',
              gap: 6,
            }}
          >
            {nextPairs.length === 0 || nextPairs[0].count < 2 ? (
              <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>No pair appears twice anymore — BPE stops here.</div>
            ) : (
              nextPairs.map((p, i) => {
                const maxC = nextPairs[0].count;
                const isNext = i === 0;
                return (
                  <div key={`${p.pair[0]}-${p.pair[1]}`} style={{ display: 'grid', gap: 3 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5 }}>
                      <span className="mono" style={{ fontWeight: isNext ? 700 : 500, color: isNext ? 'var(--accent-text)' : 'var(--text)' }}>
                        {p.pair[0]} + {p.pair[1]} {isNext ? '← next' : ''}
                      </span>
                      <span className="mono" style={{ color: 'var(--text-muted)' }}>
                        ×{p.count}
                      </span>
                    </div>
                    <div style={{ height: 5, borderRadius: 3, background: 'var(--border)', overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${(p.count / maxC) * 100}%`,
                          height: '100%',
                          background: isNext ? 'var(--accent)' : 'var(--c-a)',
                          opacity: isNext ? 1 : 0.55,
                        }}
                      />
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {note}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Tokenise any word

const PRESETS = [
  { label: 'strawberry', value: 'strawberry' },
  { label: 'unwatched', value: 'unwatched' },
  { label: 'fastest slowest warmest', value: 'fastest slowest warmest' },
  { label: 'catfish', value: 'catfish' },
  { label: '123456', value: '123456' },
];

export function WordTokenizer() {
  const [text, setText] = useState('strawberry');
  const [k, setK] = useState(TRAINED.merges.length);

  const ids = useMemo(() => vocabIds(TRAINED.baseVocab, TRAINED.merges, k), [k]);
  const words = useMemo(() => text.trim().toLowerCase().split(/\s+/).filter(Boolean), [text]);
  const traces = useMemo(() => words.map((w) => ({ word: w, ...encodeWordTrace(w, TRAINED.merges, k) })), [words, k]);
  const allTokens = traces.flatMap((t) => t.tokens);
  const rawChars = words.reduce((s, w) => s + [...w].length, 0);
  const unknownCount = allTokens.filter((t) => !ids.has(t)).length;

  const liveNote = (() => {
    const clean = text.trim().toLowerCase();
    if (!words.length) return <>Type a word or click a preset above to watch the learned merges replay on it.</>;
    if (clean === 'strawberry' && k >= 63) {
      return (
        <>
          <strong>Why models miscount the "r"s in strawberry:</strong> the transformer never sees ten letters. It receives the seven integer IDs above — with the first <code>r</code> alone and the other two welded inside <code>berr</code>. Counting letters inside opaque token IDs is like counting letters inside someone else's phone numbers.
        </>
      );
    }
    if (clean === 'unwatched' && k >= 50) {
      return (
        <>
          <strong>Never seen in training, zero unknown tokens.</strong> <code>unwatched</code> never appeared in our corpus, yet BPE splits it into <code>{traces[0].tokens.join(' | ')}</code> — recognising <code>watch</code> and <code>ed_</code> from related words and falling back to letters for <code>u</code> and <code>n</code>.
        </>
      );
    }
    if (unknownCount > 0) {
      return (
        <>
          <strong>{unknownCount} {unknownCount === 1 ? 'symbol' : 'symbols'} outside our toy 22-character alphabet</strong> (shown as <code>byte</code>). Our mini-corpus contains only lowercase letters — no digits or punctuation. Real tokenisers avoid even this fallback by starting from all <strong>256 raw UTF-8 bytes</strong>, so any text on earth can be encoded.
        </>
      );
    }
    return (
      <>
        <strong>{rawChars} characters → {allTokens.length} {allTokens.length === 1 ? 'token' : 'tokens'}</strong> using the first {k} learned merges. The transformer sees only the integer IDs in the readout; the human-readable pieces are just our lookup table back into text.
      </>
    );
  })();

  return (
    <Widget
      title="Tokenise any word — and see what the model actually receives"
      wide
      caption={
        <>
          Type any word or phrase, or try the presets. BPE starts from characters plus <code>_</code> and replays the first <em>k</em> learned merge rules in order. Notice the integer ID underneath each chip: that list of integers is the only thing the neural network ever sees.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', marginBottom: 10 }}>
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          aria-label="Word or phrase to tokenise"
          placeholder="Type any word..."
          className="mono"
          style={{
            flex: '1 1 200px',
            minWidth: 0,
            padding: '7px 10px',
            fontSize: 14,
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-strong)',
            background: 'var(--bg-elev)',
            color: 'var(--text)',
          }}
        />
        <div className="btn-row">
          {PRESETS.map((p) => (
            <button key={p.label} className={text.trim().toLowerCase() === p.value ? 'btn primary' : 'btn'} onClick={() => setText(p.value)}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', marginBottom: 12 }}>
        <div style={{ flex: '1 1 220px' }}>
          <Slider
            label="learned merges replayed"
            value={k}
            min={0}
            max={TRAINED.merges.length}
            step={1}
            onChange={setK}
            format={(v) => `${v}`}
            color="accent"
          />
        </div>
        <div className="btn-row">
          <button className="btn" onClick={() => setK(0)}>
            0 (chars)
          </button>
          <button className="btn" onClick={() => setK(OPT.merges)}>
            28 (MDL best)
          </button>
          <button className="btn" onClick={() => setK(TRAINED.merges.length)}>
            66 (all)
          </button>
        </div>
      </div>

      <div className="field-label">tokens and their integer IDs</div>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '8px 14px',
          padding: '12px',
          background: 'var(--bg-sunken)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)',
          minHeight: 54,
          alignItems: 'center',
        }}
      >
        {traces.map((tr, wi) => (
          <span key={wi} style={{ display: 'inline-flex', gap: 4, flexWrap: 'wrap' }}>
            {tr.tokens.map((tok, ti) => {
              const id = ids.get(tok);
              return <TokenChip key={ti} text={tok} tintIndex={ti} sub={id !== undefined ? `#${id}` : 'byte'} />;
            })}
          </span>
        ))}
      </div>

      {allTokens.length > 0 && (
        <div className="readout" style={{ marginTop: 8, fontSize: 12.5, wordBreak: 'break-word' }}>
          model input IDs: [{allTokens.map((t) => (ids.has(t) ? ids.get(t) : `byte(${t.codePointAt(0)})`)).join(', ')}]
        </div>
      )}

      {traces.length === 1 && traces[0].applied.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <div className="field-label">merge rules that fired on "{traces[0].word}"</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {traces[0].applied.map((a) => (
              <span
                key={a.rule}
                className="mono"
                style={{
                  fontSize: 12,
                  padding: '3px 7px',
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--bg-sunken)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-muted)',
                }}
              >
                #{a.rule}: {a.pair[0]}+{a.pair[1]}→<strong style={{ color: 'var(--text)' }}>{a.merged}</strong>
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="live-note" style={{ marginTop: 10 }}>
        {liveNote}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Byte-level UTF-8 & the language tax

export function Utf8LanguageTax() {
  const [selected, setSelected] = useState(0);
  const [custom, setCustom] = useState('');

  const rows = useMemo(
    () =>
      LANGUAGE_SAMPLES.map((s) => ({
        ...s,
        stats: utf8Stats(s.text),
      })),
    [],
  );
  const maxBytes = Math.max(...rows.map((r) => r.stats.byteCount));
  const activeText = custom.trim() ? custom : rows[selected].text;
  const activeLabel = custom.trim() ? 'Your text' : rows[selected].lang;
  const activeStats = useMemo(() => utf8Stats(activeText), [activeText]);

  return (
    <Widget
      title="One sentence, six scripts: the UTF-8 byte tax"
      wide
      caption={
        <>
          Byte-level BPE starts from raw UTF-8 bytes rather than characters, so no symbol is ever unknown. Click any language — or type your own text (try accents, Japanese or emoji) — to compare characters with starting UTF-8 bytes before a single merge happens.
        </>
      }
    >
      <div style={{ display: 'grid', gap: 8 }}>
        {rows.map((r, i) => {
          const on = !custom.trim() && selected === i;
          return (
            <button
              key={r.lang}
              type="button"
              onClick={() => {
                setCustom('');
                setSelected(i);
              }}
              style={{
                display: 'grid',
                gridTemplateColumns: 'minmax(74px, 90px) 1fr',
                gap: 10,
                alignItems: 'center',
                textAlign: 'left',
                padding: '7px 10px',
                borderRadius: 'var(--radius-sm)',
                border: `1px solid ${on ? 'var(--accent)' : 'var(--border)'}`,
                background: on ? 'var(--accent-soft)' : 'var(--bg-sunken)',
                cursor: 'pointer',
              }}
            >
              <div>
                <div style={{ fontSize: 13, fontWeight: 650, color: on ? 'var(--accent-text)' : 'var(--text)' }}>{r.lang}</div>
                <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  {r.stats.bytesPerChar.toFixed(2)} B/ch
                </div>
              </div>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 13, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginBottom: 4 }}>
                  {r.text}
                </div>
                <div style={{ display: 'grid', gap: 3 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ width: `${(r.stats.chars / maxBytes) * 100}%`, minWidth: 4, height: 6, borderRadius: 3, background: 'var(--c-a)' }} />
                    <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                      {r.stats.chars} chars
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ width: `${(r.stats.byteCount / maxBytes) * 100}%`, minWidth: 4, height: 6, borderRadius: 3, background: 'var(--c-b)' }} />
                    <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                      {r.stats.byteCount} UTF-8 bytes
                    </span>
                  </div>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
        <input
          type="text"
          value={custom}
          onChange={(e) => setCustom(e.target.value)}
          aria-label="Try your own text for UTF-8 byte count"
          placeholder="Or type anything here (café, 東京, 🚀)..."
          style={{
            flex: '1 1 220px',
            minWidth: 0,
            padding: '7px 10px',
            fontSize: 14,
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-strong)',
            background: 'var(--bg-elev)',
            color: 'var(--text)',
          }}
        />
        {custom && (
          <button className="btn" onClick={() => setCustom('')}>
            Clear
          </button>
        )}
      </div>

      <div className="readout" style={{ marginTop: 8, fontSize: 12, wordBreak: 'break-word' }}>
        {activeLabel} raw UTF-8 bytes ({activeStats.byteCount}):{' '}
        {activeStats.bytes
          .slice(0, 32)
          .map((b) => b.toString(16).padStart(2, '0'))
          .join(' ')}
        {activeStats.bytes.length > 32 ? ` … (+${activeStats.bytes.length - 32} more)` : ''}
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        <strong>{activeLabel}:</strong> {activeStats.chars} {activeStats.chars === 1 ? 'character' : 'characters'} ={' '}
        <strong>{activeStats.byteCount} starting UTF-8 bytes</strong> ({activeStats.bytesPerChar.toFixed(2)} bytes per character). Before BPE learns a single merge, non-Latin scripts already start 2–3× longer than English — and because most real training corpora are heavily English, English gets far more merges too.
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Compress the corpus (MDL)

type HandMerge = { a: string; b: string; count: number };

function loadLabMerges(): HandMerge[] {
  try {
    const raw = localStorage.getItem(LAB_STORAGE);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed;
  } catch {
    // ignore storage errors
  }
  return [];
}

function MdlPlot({
  showBpe,
  cursorMerges,
  handTrail,
}: {
  showBpe: boolean;
  cursorMerges?: number;
  handTrail?: Step[];
}) {
  const maxM = TRAINED.merges.length;
  const xDom: [number, number] = [0, maxM];
  const yDom: [number, number] = [0, 2500];
  const poly = (pts: [number, number][], x: (v: number) => number, y: (v: number) => number) =>
    pts.map(([a, b]) => `${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(' ');

  return (
    <Plot
      width={460}
      height={235}
      x={xDom}
      y={yDom}
      pad={46}
      grid
      axes
      xLabel="merges"
      yLabel="bits"
      ariaLabel="Minimum description length in bits against number of merges"
      style={{ width: '100%', height: 'auto', maxWidth: 480, display: 'block' }}
    >
      {({ x, y }) => (
        <g>
          {/* Goal line at 1900 bits */}
          <line x1={x(0)} x2={x(maxM)} y1={y(1900)} y2={y(1900)} stroke="var(--c-good)" strokeDasharray="4 4" strokeWidth={1.2} opacity={0.75} />
          <text x={x(maxM) - 4} y={y(1900) - 5} textAnchor="end" fill="var(--c-good)" fontSize={10.5} fontFamily="var(--font-ui)">
            goal: 1,900 bits
          </text>

          {showBpe && (
            <>
              {/* Corpus bits */}
              <polyline
                points={poly(TRAINED.steps.map((s) => [s.merges, s.corpusBits]), x, y)}
                fill="none"
                stroke="var(--c-a)"
                strokeWidth={1.8}
                strokeDasharray="5 3"
              />
              {/* Merge table bits */}
              <polyline
                points={poly(TRAINED.steps.map((s) => [s.merges, s.tableBits]), x, y)}
                fill="none"
                stroke="var(--c-b)"
                strokeWidth={1.8}
                strokeDasharray="5 3"
              />
              {/* Total MDL bits */}
              <polyline
                points={poly(TRAINED.steps.map((s) => [s.merges, s.totalBits]), x, y)}
                fill="none"
                stroke="var(--c-c)"
                strokeWidth={2.4}
              />
              {/* Exhaustive optimum marker */}
              <line x1={x(OPT.merges)} x2={x(OPT.merges)} y1={y(0)} y2={y(OPT.totalBits)} stroke="var(--c-c)" strokeDasharray="2 3" strokeWidth={1.2} opacity={0.7} />
              <circle cx={x(OPT.merges)} cy={y(OPT.totalBits)} r={4.5} fill="var(--c-c)" />
              <text x={x(OPT.merges) + 6} y={y(OPT.totalBits) + 16} fill="var(--c-c)" fontSize={10.5} fontFamily="var(--font-mono)">
                opt: {OPT.merges} ({fmt(OPT.totalBits)} b)
              </text>
            </>
          )}

          {cursorMerges !== undefined && (
            <>
              <line x1={x(cursorMerges)} x2={x(cursorMerges)} y1={y(0)} y2={y(2500)} stroke="var(--accent)" strokeWidth={1.5} opacity={0.6} />
              <circle cx={x(cursorMerges)} cy={y(TRAINED.steps[cursorMerges].totalBits)} r={5} fill="var(--accent)" />
            </>
          )}

          {handTrail && handTrail.length > 0 && (
            <>
              <polyline
                points={poly(handTrail.map((s) => [s.merges, s.totalBits]), x, y)}
                fill="none"
                stroke="var(--accent)"
                strokeWidth={2.6}
              />
              {(() => {
                const last = handTrail[handTrail.length - 1];
                return <circle cx={x(last.merges)} cy={y(last.totalBits)} r={5.5} fill="var(--accent)" stroke="var(--bg-elev)" strokeWidth={1.5} />;
              })()}
            </>
          )}
        </g>
      )}
    </Plot>
  );
}

export function CompressionLab() {
  const [mode, setMode] = useState<'hand' | 'sweep'>('hand');
  const [handMerges, setHandMerges] = useState<HandMerge[]>(loadLabMerges);
  const [showRef, setShowRef] = useState(false);
  const [sweepK, setSweepK] = useState(0);

  useEffect(() => {
    try {
      localStorage.setItem(LAB_STORAGE, JSON.stringify(handMerges));
    } catch {
      // ignore
    }
  }, [handMerges]);

  const handState = useMemo(() => {
    let words = INIT_WORDS;
    const trail: Step[] = [mdlStep(words, 0, TRAINED.initialVocabSize)];
    for (let i = 0; i < handMerges.length; i++) {
      const m = handMerges[i];
      words = applyMergeToWords(words, m.a, m.b);
      trail.push(mdlStep(words, i + 1, TRAINED.initialVocabSize));
    }
    return { words, trail, current: trail[trail.length - 1], prev: trail.length > 1 ? trail[trail.length - 2] : null };
  }, [handMerges]);

  const pairs = useMemo(() => topPairs(handState.words, 12).filter((p) => p.count >= 2), [handState.words]);

  const applyPair = (a: string, b: string, count: number) => {
    setHandMerges((prev) => [...prev, { a, b, count }]);
  };

  const goal1 = handState.current.totalBits < 1900;
  const goal2 = handState.current.totalBits <= OPT.totalBits + 0.5;
  const sweepStep = TRAINED.steps[sweepK];
  const nextBpeStep = sweepK < TRAINED.merges.length ? TRAINED.steps[sweepK + 1] : null;
  const nextBpeMerge = sweepK < TRAINED.merges.length ? TRAINED.merges[sweepK] : null;

  return (
    <Widget
      title="Lab: Compress the corpus"
      wide
      caption={
        <>
          Pack the corpus <strong>plus</strong> its merge table into as few bits as possible: <code>tokens × log₂(V) + 2 × merges × log₂(V)</code>. In <strong>Build by hand</strong>, click any adjacent pair to merge it (many valid sets beat 1,900 bits). Switch to <strong>Sweep BPE</strong> to inspect the exact U-shaped trade-off across all 0–66 greedy merges.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <Segmented
          options={[
            { value: 'hand', label: 'Build by hand' },
            { value: 'sweep', label: 'Sweep BPE (0–66)' },
          ]}
          value={mode}
          onChange={setMode}
        />
        {mode === 'hand' && <Toggle label="show BPE reference curve" checked={showRef || goal1} onChange={setShowRef} />}
      </div>

      {mode === 'hand' ? (
        <>
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 10,
              padding: '8px 12px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-sunken)',
              border: '1px solid var(--border)',
              marginBottom: 12,
              fontSize: 13.5,
            }}
          >
            <span style={{ color: goal1 ? 'var(--c-good)' : 'var(--text-muted)', fontWeight: goal1 ? 700 : 500 }}>
              {goal1 ? '✓' : '○'} Goal: total &lt; 1,900 bits
            </span>
            <span style={{ color: goal2 ? 'var(--c-good)' : 'var(--text-faint)', fontWeight: goal2 ? 700 : 500 }}>
              {goal2 ? '✓' : '○'} Optional: match BPE optimum ({fmt(OPT.totalBits)} bits)
            </span>
          </div>

          <div className="widget-row">
            <div className="widget-col" style={{ flex: '1 1 260px', minWidth: 0 }}>
              <div className="field-label">click any pair to merge it</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
                {pairs.length === 0 ? (
                  <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>No pairs appear twice anymore.</span>
                ) : (
                  pairs.map((p) => (
                    <button
                      key={`${p.pair[0]}-${p.pair[1]}`}
                      className="btn mono"
                      style={{ fontSize: 12.5, padding: '5px 8px' }}
                      onClick={() => applyPair(p.pair[0], p.pair[1], p.count)}
                    >
                      {p.pair[0]}+{p.pair[1]} <span style={{ opacity: 0.65 }}>×{p.count}</span>
                    </button>
                  ))
                )}
              </div>

              <div className="btn-row" style={{ marginBottom: 10 }}>
                <button
                  className="btn primary"
                  disabled={pairs.length === 0}
                  onClick={() => pairs[0] && applyPair(pairs[0].pair[0], pairs[0].pair[1], pairs[0].count)}
                >
                  Greedy step
                </button>
                <button className="btn" disabled={handMerges.length === 0} onClick={() => setHandMerges((prev) => prev.slice(0, -1))}>
                  Undo
                </button>
                <button className="btn" disabled={handMerges.length === 0} onClick={() => setHandMerges([])}>
                  Reset
                </button>
              </div>

              <div className="readout" style={{ fontSize: 12.5, display: 'grid', gap: 3 }}>
                <div>
                  merges: <strong>{handState.current.merges}</strong> · tokens: <strong>{handState.current.tokens}</strong> · vocab:{' '}
                  <strong>{handState.current.vocab}</strong>
                </div>
                <div>
                  corpus: <span style={{ color: 'var(--c-a)' }}>{f1(handState.current.corpusBits)} b</span> + table:{' '}
                  <span style={{ color: 'var(--c-b)' }}>{f1(handState.current.tableBits)} b</span> ={' '}
                  <strong style={{ color: goal1 ? 'var(--c-good)' : 'var(--accent-text)' }}>{f1(handState.current.totalBits)} bits</strong>
                  {handState.prev && (
                    <span style={{ color: handState.current.totalBits < handState.prev.totalBits ? 'var(--c-good)' : 'var(--c-bad)', marginLeft: 6 }}>
                      ({handState.current.totalBits - handState.prev.totalBits >= 0 ? '+' : ''}
                      {f1(handState.current.totalBits - handState.prev.totalBits)} b)
                    </span>
                  )}
                </div>
              </div>

              {handMerges.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div className="field-label">your merge table ({handMerges.length})</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, maxHeight: 84, overflowY: 'auto' }}>
                    {handMerges.map((m, i) => (
                      <span
                        key={i}
                        className="mono"
                        style={{
                          fontSize: 11,
                          padding: '2px 5px',
                          borderRadius: 4,
                          background: 'var(--bg-sunken)',
                          border: '1px solid var(--border)',
                          color: 'var(--text-muted)',
                        }}
                      >
                        {i + 1}.{m.a}+{m.b}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="widget-col" style={{ flex: '1.2 1 320px', minWidth: 0 }}>
              <MdlPlot showBpe={showRef || goal1} handTrail={handState.trail} />
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 14px', fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                <span>
                  <strong style={{ color: 'var(--accent)' }}>●</strong> your path
                </span>
                {(showRef || goal1) && (
                  <>
                    <span>
                      <strong style={{ color: 'var(--c-c)' }}>—</strong> BPE total
                    </span>
                    <span>
                      <strong style={{ color: 'var(--c-a)' }}>--</strong> corpus bits
                    </span>
                    <span>
                      <strong style={{ color: 'var(--c-b)' }}>--</strong> merge table
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>

          <div className="live-note" style={{ marginTop: 10 }}>
            {goal2 ? (
              <>
                <strong>Optimum matched ({f1(handState.current.totalBits)} bits in {handState.current.merges} merges)!</strong> Across all 67 greedy BPE checkpoints (0–66 merges), the exhaustive minimum is <strong>{OPT.merges} merges ({fmt(OPT.totalBits)} bits)</strong>. Notice why: merge 28 appears 4 times and saves more bits in the corpus than it adds to the table; every pair of count 3 or 2 costs more to record than it saves.
              </>
            ) : goal1 ? (
              <>
                <strong>Goal cleared ({f1(handState.current.totalBits)} bits)!</strong> You beat 1,900 bits with your own merge choices. The BPE reference curve is now shown: its exhaustive minimum is <strong>{OPT.merges} merges ({fmt(OPT.totalBits)} bits)</strong>. Keep going — or watch what happens when you merge pairs with count 3 or 2!
              </>
            ) : handState.prev && handState.current.totalBits > handState.prev.totalBits ? (
              <>
                <strong>That last merge made the total bigger</strong> (+{f1(handState.current.totalBits - handState.prev.totalBits)} bits). A rare pair removes only a few tokens, while adding 1 symbol to the vocabulary widens every token ID slightly <em>and</em> adds two IDs to the merge table. Try undoing it or picking a higher-frequency pair.
              </>
            ) : (
              <>
                <strong>Start merging frequent pairs.</strong> Each merge shrinks the corpus by the pair's count, but adds <code>2 · log₂(V)</code> bits to the merge table and nudges <code>log₂(V)</code> up for every remaining token. Watch when the net change turns positive!
              </>
            )}
          </div>
        </>
      ) : (
        <>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', marginBottom: 10 }}>
            <div style={{ flex: '1 1 240px' }}>
              <Slider
                label="BPE merges"
                value={sweepK}
                min={0}
                max={TRAINED.merges.length}
                step={1}
                onChange={setSweepK}
                format={(v) => `${v} / ${TRAINED.merges.length}`}
                color="accent"
              />
            </div>
            <div className="btn-row">
              <button className="btn" onClick={() => setSweepK(0)}>
                0 (chars)
              </button>
              <button className="btn primary" onClick={() => setSweepK(OPT.merges)}>
                Optimum ({OPT.merges})
              </button>
              <button className="btn" onClick={() => setSweepK(TRAINED.merges.length)}>
                {TRAINED.merges.length} (max)
              </button>
            </div>
          </div>

          <div className="widget-row">
            <div className="widget-col" style={{ flex: '1.2 1 320px', minWidth: 0 }}>
              <MdlPlot showBpe cursorMerges={sweepK} />
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 14px', fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                <span>
                  <strong style={{ color: 'var(--c-c)' }}>—</strong> total MDL
                </span>
                <span>
                  <strong style={{ color: 'var(--c-a)' }}>--</strong> corpus IDs
                </span>
                <span>
                  <strong style={{ color: 'var(--c-b)' }}>--</strong> merge table
                </span>
              </div>
            </div>
            <div className="widget-col" style={{ flex: '1 1 220px', minWidth: 0 }}>
              <div className="readout" style={{ fontSize: 13, display: 'grid', gap: 6 }}>
                <div>
                  merges: <strong>{sweepStep.merges}</strong> · vocab: <strong>{sweepStep.vocab}</strong>
                </div>
                <div>
                  tokens: <strong>{sweepStep.tokens}</strong> ({((sweepStep.tokens / TRAINED.steps[0].tokens) * 100).toFixed(0)}% of chars)
                </div>
                <div>
                  corpus IDs: <strong style={{ color: 'var(--c-a)' }}>{f1(sweepStep.corpusBits)} bits</strong>
                </div>
                <div>
                  merge table: <strong style={{ color: 'var(--c-b)' }}>{f1(sweepStep.tableBits)} bits</strong>
                </div>
                <div style={{ borderTop: '1px solid var(--border)', paddingTop: 6 }}>
                  total MDL: <strong style={{ color: 'var(--c-c)' }}>{f1(sweepStep.totalBits)} bits</strong>
                </div>
              </div>
              {nextBpeStep && nextBpeMerge && (
                <div style={{ fontSize: 12.5, color: 'var(--text-muted)', marginTop: 8 }}>
                  Next merge (#{sweepK + 1}, <code>{nextBpeMerge.pair[0]}+{nextBpeMerge.pair[1]}</code>, ×{nextBpeMerge.count}): corpus{' '}
                  <span className="mono">{f1(nextBpeStep.corpusBits - sweepStep.corpusBits)}</span> b, table{' '}
                  <span className="mono">+{f1(nextBpeStep.tableBits - sweepStep.tableBits)}</span> b → net{' '}
                  <strong className="mono" style={{ color: nextBpeStep.totalBits < sweepStep.totalBits ? 'var(--c-good)' : 'var(--c-bad)' }}>
                    {nextBpeStep.totalBits - sweepStep.totalBits >= 0 ? '+' : ''}
                    {f1(nextBpeStep.totalBits - sweepStep.totalBits)} b
                  </strong>
                </div>
              )}
            </div>
          </div>

          <div className="live-note" style={{ marginTop: 10 }}>
            {sweepK === OPT.merges ? (
              <>
                <strong>Exhaustive minimum across all 0–66 merges:</strong> 28 merges ({fmt(OPT.totalBits)} bits, down from 2,368 at 0 merges and better than 1,996 at 66). Merge 28 (<code>n+g</code>, ×4) is the very last merge that saves more bits in the corpus than it adds to the table; merge 29 (<code>s+at_</code>, ×3) tips the balance the other way.
              </>
            ) : sweepK < OPT.merges ? (
              <>
                <strong>Under-merged ({sweepK} merges):</strong> corpus bits ({fmt(sweepStep.corpusBits)}) dominate, and frequent pairs still sit unmerged. More merges shrink the total.
              </>
            ) : (
              <>
                <strong>Over-merged ({sweepK} merges):</strong> corpus bits keep falling, but the merge table ({fmt(sweepStep.tableBits)} bits) is now growing faster than the corpus shrinks. Total cost has climbed back to {fmt(sweepStep.totalBits)} bits.
              </>
            )}
          </div>
        </>
      )}
    </Widget>
  );
}
