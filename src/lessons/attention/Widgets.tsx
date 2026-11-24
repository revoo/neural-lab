import { useMemo, useState, type ReactNode } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { Arrow, Handle, Label, Plot } from '../../components/plot';
import { FEATURES, GRID, PUZZLES, WORDS, attend, maxWeightStudy, sentenceAttention, solve, weightsFor, type Vec } from './attention';

const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
const f2 = (v: number) => v.toFixed(2);

function WeightBars({ names, weights, highlight, colors }: { names: string[]; weights: number[]; highlight?: number[]; colors?: string[] }) {
  return (
    <div style={{ display: 'grid', gap: 4 }}>
      {names.map((n, i) => (
        <div key={i} style={{ display: 'grid', gridTemplateColumns: '70px 1fr 44px', gap: 8, alignItems: 'center', fontSize: 13 }}>
          <span className="mono" style={{ textAlign: 'right', fontWeight: highlight?.includes(i) ? 700 : 500 }}>
            {n}
          </span>
          <div style={{ height: 14, background: 'var(--bg-sunken)', borderRadius: 3 }}>
            <div style={{ width: `${weights[i] * 100}%`, height: '100%', borderRadius: 3, background: colors?.[i] ?? (highlight?.includes(i) ? 'var(--accent)' : 'var(--text-faint)') }} />
          </div>
          <span className="mono" style={{ fontSize: 12.5 }}>
            {pct(weights[i])}
          </span>
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ 1. the soft lookup

/** Four tokens. Keys are 2D (drawn as arrows); values are colours, so the output is a visible blend. */
const LOOKUP = [
  { name: 'cat', k: [-1, 2.5], v: [230, 80, 60] },
  { name: 'sat', k: [2.5, 1], v: [60, 170, 90] },
  { name: 'mat', k: [-2, -2], v: [70, 110, 230] },
  { name: 'on', k: [1.5, -2.5], v: [230, 180, 40] },
];
const rgb = (v: number[]) => `rgb(${v.map((c) => Math.round(c)).join(',')})`;

export function SoftLookup() {
  const [q, setQ] = useState<Vec>([0.6, 1.0]);
  const r = attend(q, LOOKUP.map((t) => t.k), LOOKUP.map((t) => t.v));
  const top = r.weights.indexOf(Math.max(...r.weights));
  const len = Math.hypot(q[0], q[1]);
  let note: ReactNode;
  if (r.weights[top] > 0.9)
    note = (
      <>
        <strong>Almost a hard lookup.</strong> The query lines up with <strong>{LOOKUP[top].name}</strong>'s key far better than with any other, so it gets{' '}
        {pct(r.weights[top])} of the attention and the output is nearly its value. Now shorten the query (drag it towards the centre) without changing its direction.
      </>
    );
  else if (len < 0.6)
    note = (
      <>
        <strong>A short query barely prefers anything.</strong> Every score is close to zero, so softmax shares the attention almost equally and the output is a muddy
        average of all four values. Length works like the inverse of temperature from the softmax lesson.
      </>
    );
  else
    note = (
      <>
        <strong>A soft lookup.</strong> The query agrees most with <strong>{LOOKUP[top].name}</strong> (score {f2(r.scores[top])}), so it gets the biggest share,{' '}
        {pct(r.weights[top])}, but the others still contribute. The output colour is the attention-weighted blend of all four values.
      </>
    );
  return (
    <Widget
      title="A soft dictionary lookup"
      wide
      caption={
        <>
          Four tokens, each with a <strong>key</strong> (the grey arrows) and a <strong>value</strong> (its colour). Drag the purple <strong>query</strong>. Each token's
          score is how much its key agrees with the query (a dot product, divided by √2); softmax turns the scores into shares that add up to 100%; the output is the
          values blended in those shares.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '0 1 360px' }}>
          <Plot x={[-3.2, 3.2]} y={[-3.2, 3.2]} width={340} height={340} ariaLabel="Keys and a draggable query">
            {LOOKUP.map((t, i) => (
              <g key={t.name}>
                <Arrow from={[0, 0]} to={t.k as [number, number]} color={i === top ? 'var(--text)' : 'var(--text-faint)'} width={i === top ? 3 : 2} />
                <Label at={t.k as [number, number]} dx={6} dy={-6} color={rgb(t.v)}>
                  {t.name}
                </Label>
              </g>
            ))}
            <Arrow from={[0, 0]} to={q as [number, number]} color="var(--accent)" width={3} />
            <Handle x={q[0]} y={q[1]} onDrag={(x, y) => setQ([x, y])} label="query" />
          </Plot>
        </div>
        <div className="widget-col" style={{ flex: '1 1 300px' }}>
          <div className="readout" style={{ marginBottom: 10 }}>
            <span className="k">query      </span> [{f2(q[0])}, {f2(q[1])}]
            {LOOKUP.map((t, i) => (
              <span key={t.name}>
                {'\n'}
                <span className="k">{`score ${t.name}`.padEnd(11)}</span>
                {f2(r.scores[i])}
              </span>
            ))}
          </div>
          <div className="field-label">attention weights (softmax of the scores)</div>
          <WeightBars names={LOOKUP.map((t) => t.name)} weights={r.weights} colors={LOOKUP.map((t) => rgb(t.v))} />
          <div className="field-label" style={{ marginTop: 12 }}>
            output = weighted blend of the values
          </div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <div style={{ width: 64, height: 40, borderRadius: 8, background: rgb(r.out), border: '1px solid var(--border-strong)' }} />
            <span className="mono muted" style={{ fontSize: 12.5 }}>
              {r.weights.map((w, i) => `${f2(w)}×${LOOKUP[i].name}`).join(' + ')}
            </span>
          </div>
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 12 }}>
        {note}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. a sentence attending to itself

export function SentenceAttention() {
  const [sel, setSel] = useState(7);
  const [variant, setVariant] = useState<'tired' | 'wide'>('tired');
  const words = useMemo(() => WORDS.map((t, i) => (i === 9 ? { ...t, w: variant } : t)), [variant]);
  const A = useMemo(() => sentenceAttention(words), [words]);
  const row = A[sel];
  const n = words.length;
  const cell = 30;
  return (
    <Widget
      title="A sentence attending to itself"
      wide
      caption={
        <>
          Every word issues a query and offers a key, built from five hand-chosen features (noun, animate, pronoun, verb/adjective, function word) by small matrices we
          set by hand so you can read them. Row <em>i</em> of the grid is word <em>i</em>'s attention over the whole sentence (stronger colour = more attention). Click a word or a row.
          Then switch the last word from "tired" to "wide".
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <Segmented label="last word" value={variant} onChange={setVariant} options={['tired', 'wide'] as const} />
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '0 1 450px', overflowX: 'auto' }}>
          <svg viewBox={`0 0 ${100 + n * cell} ${70 + n * cell}`} width={100 + n * cell} height={70 + n * cell} role="img" aria-label="Attention weights between every pair of words">
            {words.map((t, j) => (
              <text key={'c' + j} x={70 + j * cell + cell / 2} y={62} fontSize={11} fill="var(--text-muted)" textAnchor="start" transform={`rotate(-50 ${70 + j * cell + cell / 2} 62)`} fontFamily="var(--font-ui)">
                {t.w}
              </text>
            ))}
            {A.map((r, i) => (
              <g key={i} onClick={() => setSel(i)} style={{ cursor: 'pointer' }}>
                <text x={64} y={70 + i * cell + cell / 2 + 4} fontSize={12} textAnchor="end" fill={i === sel ? 'var(--accent-text)' : 'var(--text)'} fontWeight={i === sel ? 700 : 400} fontFamily="var(--font-ui)">
                  {words[i].w}
                </text>
                {r.weights.map((w, j) => (
                  <rect key={j} x={70 + j * cell} y={70 + i * cell} width={cell - 2} height={cell - 2} rx={3} fill="var(--accent)" fillOpacity={0.06 + 0.94 * w} stroke={i === sel ? 'var(--accent)' : 'none'} />
                ))}
              </g>
            ))}
          </svg>
        </div>
        <div className="widget-col" style={{ flex: '1 1 300px' }}>
          <div className="field-label">
            "{words[sel].w}" attends to…
          </div>
          <WeightBars names={words.map((t) => t.w)} weights={row.weights} highlight={[row.weights.indexOf(Math.max(...row.weights))]} />
          <div className="readout" style={{ marginTop: 10, fontSize: 12.5 }}>
            <span className="k">features of "{words[sel].w}" </span>
            {FEATURES.filter((_, i) => words[sel].x[i]).join(', ') || '—'}
            {'\n'}
            <span className="k">its query        </span> [{row.q.join(', ')}] <span className="k">(want a noun, want animate)</span>
          </div>
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 12 }}>
        {sel === 7 && variant === 'tired' && (
          <>
            <strong>"it" finds "animal"</strong> ({pct(row.weights[1])}). Its query asks for a noun <em>and</em> for something animate; "animal" is both, "street" only one.
            The output for "it" is now mostly "animal"'s value: the pronoun has picked up what it refers to.
          </>
        )}
        {sel === 7 && variant === 'wide' && (
          <>
            <strong>Nothing changed.</strong> Still {pct(row.weights[1])} on "animal" — but "because it was too wide" means the <em>street</em>. A word's query here depends only on
            that word's own features, so "it" can't take "wide" into account. Real transformers handle this by stacking layers: an earlier layer lets "it" gather
            information from "wide", and a later layer's query, built from that enriched vector, can then look for the right noun.
          </>
        )}
        {sel !== 7 && row.q.every((v) => v === 0) && (
          <>"{words[sel].w}" is a function word: its query is all zeros, so every score is 0 and it spreads its attention evenly. In a trained model even small words learn useful things to look for.</>
        )}
        {sel !== 7 && !row.q.every((v) => v === 0) && (
          <>"{words[sel].w}" asks for a noun of any kind, so it splits its attention between "animal" and "street" ({pct(row.weights[1])} each).</>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. why divide by √d

const DIMS = [2, 8, 32, 128, 512];

export function ScaleStudy() {
  const rows = useMemo(() => DIMS.map((d) => maxWeightStudy(d)), []);
  return (
    <Widget
      title="What happens to softmax as vectors get longer"
      wide
      caption={
        <>
          Random queries and keys with <em>d</em> components each (every component a standard normal number), 10 keys per query, 400 trials for each <em>d</em>. The
          bars show the <strong>largest attention weight</strong>, averaged over the trials: with raw dot products, and with dot products divided by √d.
        </>
      }
    >
      <table className="num-table" style={{ width: '100%', maxWidth: 640 }}>
        <thead>
          <tr>
            <th style={{ textAlign: 'left' }}>d</th>
            <th>typical size of a score</th>
            <th style={{ textAlign: 'left' }}>largest weight, raw scores</th>
            <th style={{ textAlign: 'left' }}>largest weight, scores ÷ √d</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.d}>
              <td style={{ textAlign: 'left' }}>{r.d}</td>
              <td>{r.spread.toFixed(1)}</td>
              {[r.raw, r.scaled].map((v, i) => (
                <td key={i} style={{ textAlign: 'left' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ width: 160, height: 12, background: 'var(--bg-sunken)', borderRadius: 3 }}>
                      <div style={{ width: `${v * 100}%`, height: '100%', background: i ? 'var(--c-c)' : 'var(--c-b)', borderRadius: 3 }} />
                    </div>
                    {pct(v)}
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. the attention puzzles

const STORE = 'nl-attention-puzzles-v1';
type Saved = Record<string, { solved?: boolean; best?: number; impossible?: boolean }>;
const load = (): Saved => {
  try {
    return JSON.parse(localStorage.getItem(STORE) ?? '{}');
  } catch {
    return {};
  }
};
const KEY_COLORS = ['var(--accent)', 'var(--c-b)', 'var(--c-a)', 'var(--c-c)'];

export function AttentionPuzzles() {
  const [idx, setIdx] = useState(0);
  const [saved, setSaved] = useState<Saved>(load);
  const [qs, setQs] = useState<Record<string, Vec>>({});
  const p = PUZZLES[idx];
  const q = qs[p.id] ?? [0, 0];
  const w = weightsFor(p, q);
  const ok = p.goal(w);
  const units = Math.abs(q[0]) + Math.abs(q[1]);
  const sol = useMemo(() => solve(p), [p]);
  const rec = saved[p.id] ?? {};
  const store = (next: Saved) => {
    setSaved(next);
    try {
      localStorage.setItem(STORE, JSON.stringify(next));
    } catch {
      /* ignore */
    }
  };
  const setQ = (v: Vec) => {
    setQs({ ...qs, [p.id]: v });
    if (p.goal(weightsFor(p, v))) {
      const u = Math.abs(v[0]) + Math.abs(v[1]);
      if (!rec.solved || u < (rec.best ?? Infinity)) store({ ...saved, [p.id]: { ...rec, solved: true, best: Math.min(u, rec.best ?? Infinity) } });
    }
  };
  const impossible = p.id === 'hidden';

  return (
    <Widget
      title="Attention puzzles"
      wide
      caption={
        <>
          Each puzzle gives you some tokens' keys (arrows). You control the <strong>query</strong>: drag it to any point with whole-number coordinates from −{GRID} to{' '}
          {GRID}. Weights are softmax(q·k / √2), exactly as in a transformer. First solve each puzzle; then, optionally, solve it with the <strong>smallest query</strong>{' '}
          (fewest units, |q₁| + |q₂|). The smallest possible was found by trying every query on the grid.
        </>
      }
    >
      <div className="clinic-tabs" role="tablist" aria-label="Puzzles">
        {PUZZLES.map((z, i) => (
          <button key={z.id} role="tab" aria-selected={i === idx} className={i === idx ? 'on' : ''} onClick={() => setIdx(i)}>
            {i + 1}. {z.title}
            {saved[z.id]?.solved || saved[z.id]?.impossible ? ' ✓' : ''}
          </button>
        ))}
      </div>
      <p style={{ margin: '4px 0 10px' }}>{p.brief}</p>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '0 1 360px' }}>
          <Plot x={[-GRID - 0.5, GRID + 0.5]} y={[-GRID - 0.5, GRID + 0.5]} width={340} height={340} ariaLabel="Keys and the query on a whole-number grid">
            {({ x, y }) => (
              <g>
                {Array.from({ length: 2 * GRID + 1 }, (_, a) =>
                  Array.from({ length: 2 * GRID + 1 }, (_, b) => <circle key={`${a}-${b}`} cx={x(a - GRID)} cy={y(b - GRID)} r={1.4} fill="var(--text-faint)" opacity={0.5} />),
                )}
              </g>
            )}
            {p.keys.map((k, i) => (
              <g key={k.name}>
                <Arrow from={[0, 0]} to={k.k as [number, number]} color={KEY_COLORS[i]} width={2.4} />
                <Label at={k.k as [number, number]} dx={6} dy={-6} color={KEY_COLORS[i]}>
                  {k.name}
                </Label>
              </g>
            ))}
            {(q[0] !== 0 || q[1] !== 0) && <Arrow from={[0, 0]} to={q as [number, number]} color="var(--text)" width={2.6} dashed />}
            <Handle x={q[0]} y={q[1]} onDrag={(a, b) => setQ([a, b])} snap={1} color="var(--text)" label="query" />
          </Plot>
        </div>
        <div className="widget-col" style={{ flex: '1 1 300px' }}>
          <div className="readout" style={{ marginBottom: 8 }}>
            <span className="k">query </span> [{q[0]}, {q[1]}] <span className="k"> size</span> {units} units
            {'\n'}
            <span className="k">goal  </span> {p.goalText}
          </div>
          <WeightBars names={p.keys.map((k) => k.name)} weights={w} colors={KEY_COLORS} />
          <div className={`feedback ${ok ? 'good' : ''}`} style={{ marginTop: 10 }}>
            {ok ? (
              <>
                <div className="feedback-title">✓ Solved with a query of {units} units.</div>
                {sol.smallest !== null && units > sol.smallest ? (
                  <p>Optional: the smallest query that solves it has {sol.smallest} units.</p>
                ) : (
                  <p>
                    That's the smallest possible: no query of fewer units on the grid solves it ({sol.solutions.length} of the {sol.total} grid queries solve it at all).
                  </p>
                )}
              </>
            ) : impossible && rec.impossible ? (
              <>
                <div className="feedback-title">Right: it can't be done.</div>
                <p>
                  "middle" = [1, 1] is exactly halfway between "left" [−1, 3] and "right" [3, −1]. A dot product is linear, so for <em>any</em> query, middle's score is
                  the average of left's and right's scores — it can never beat both of them. The best it can ever do is tie, and a three-way tie gives it a third. We
                  checked every query on the grid: the most "middle" ever gets is {pct(sol.bestFirst)}.
                </p>
                <p>
                  In general, a key that sits inside the region spanned by other keys can never be the clear winner of a dot-product lookup. Real models escape this
                  because keys live in hundreds of dimensions, where there's almost always room for every key to stick out in some direction.
                </p>
              </>
            ) : (
              <>Not yet. {sol.solutions.length > 0 ? `${sol.solutions.length} of the ${sol.total} grid queries solve this one.` : ''}</>
            )}
          </div>
          {impossible && !rec.impossible && (
            <button className="btn" style={{ marginTop: 8 }} onClick={() => store({ ...saved, [p.id]: { ...rec, impossible: true } })}>
              I think this is impossible
            </button>
          )}
        </div>
      </div>
    </Widget>
  );
}
