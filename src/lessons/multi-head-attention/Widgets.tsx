import { useMemo, useState, type ReactNode } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { FEATURES, IT, PREDICATE, REFERENT, WORDS, attentionFlops, attentionParams, combinedHead, runHead } from './mha';

const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
const fmt = (n: number) => n.toLocaleString('en-GB');

function Bars({ names, values, color, max = 1, fmtv = pct }: { names: string[]; values: number[]; color: string; max?: number; fmtv?: (v: number) => string }) {
  return (
    <div style={{ display: 'grid', gap: 3 }}>
      {names.map((n, i) => (
        <div key={i} style={{ display: 'grid', gridTemplateColumns: '64px 1fr 40px', gap: 6, alignItems: 'center', fontSize: 12.5 }}>
          <span className="mono" style={{ textAlign: 'right' }}>
            {n}
          </span>
          <div style={{ height: 11, background: 'var(--bg-sunken)', borderRadius: 3 }}>
            <div style={{ width: `${Math.min(1, values[i] / max) * 100}%`, height: '100%', background: color, borderRadius: 3 }} />
          </div>
          <span className="mono">{fmtv(values[i])}</span>
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ 1. one head vs two

export function OneVsTwoHeads() {
  const [mode, setMode] = useState<'one' | 'two'>('one');
  const [balance, setBalance] = useState(0.5);
  const words = WORDS.map((t) => t.w);
  const one = useMemo(() => runHead(combinedHead(balance))[IT], [balance]);
  const ref = useMemo(() => runHead(REFERENT)[IT], []);
  const pred = useMemo(() => runHead(PREDICATE)[IT], []);
  const refShare = mode === 'one' ? one.weights[1] : ref.weights[1];
  const predShare = mode === 'one' ? one.weights[3] + one.weights[9] : pred.weights[3] + pred.weights[9];

  let note: ReactNode;
  if (mode === 'one')
    note =
      refShare > 0.8 ? (
        <>
          <strong>The referent wins, the predicate starves.</strong> "it" gives {pct(refShare)} of its attention to "animal" and only {pct(predShare)} to "cross" and
          "tired". Softmax has one budget of 100%, and the stronger match takes it. Slide the balance towards the predicate.
        </>
      ) : predShare > 0.8 ? (
        <>
          <strong>Now the opposite.</strong> {pct(predShare)} goes to the predicates and only {pct(refShare)} to "animal": the output has nearly forgotten what "it"
          refers to.
        </>
      ) : (
        <>
          <strong>A compromise, and a blur.</strong> {pct(refShare)} to "animal", {pct(predShare)} to the predicates. Both pieces of information arrive, but each at
          half strength, mixed together in one vector — and they can never both get more than half. The best balance gives each about 46%. Switch to two heads.
        </>
      );
  else
    note = (
      <>
        <strong>Two heads, two budgets.</strong> The referent head gives "animal" {pct(refShare)}; the predicate head gives "cross" and "tired" {pct(predShare)}{' '}
        together. Each head has its own softmax, so neither has to compromise, and their outputs sit side by side in separate halves of the result instead of being
        averaged together.
      </>
    );

  return (
    <Widget
      title="One head or two?"
      wide
      caption={
        <>
          The word "it" needs two things from the sentence: <strong>what it refers to</strong> ("animal") and <strong>what is said about it</strong> (the content verbs and
          adjectives). With one head, a single query has to ask for both; the balance dial sets how much it cares about each. With two heads, each asks for one. The
          bars show "it"'s attention, then what arrives in its output (each word's value is just its five features).
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <Segmented label="heads" value={mode} onChange={setMode} options={[{ value: 'one', label: 'one head' }, { value: 'two', label: 'two heads' }]} />
        {mode === 'one' && (
          <div style={{ flex: '1 1 240px', maxWidth: 340 }}>
            <Slider label="balance: referent ← → predicate" value={balance} min={0} max={1} step={0.01} onChange={setBalance} format={(v) => v.toFixed(2)} color="accent" />
          </div>
        )}
      </div>
      <div className="widget-row">
        {mode === 'one' ? (
          <div className="widget-col">
            <div className="field-label">"it" attends to (one softmax)</div>
            <Bars names={words} values={one.weights} color="var(--accent)" />
          </div>
        ) : (
          <>
            <div className="widget-col">
              <div className="field-label">head 1: referent</div>
              <Bars names={words} values={ref.weights} color="var(--c-a)" />
            </div>
            <div className="widget-col">
              <div className="field-label">head 2: predicate</div>
              <Bars names={words} values={pred.weights} color="var(--c-b)" />
            </div>
          </>
        )}
        <div className="widget-col">
          <div className="field-label">what arrives in "it"'s output</div>
          {mode === 'one' ? (
            <Bars names={FEATURES} values={one.out} color="var(--accent)" fmtv={(v) => v.toFixed(2)} />
          ) : (
            <>
              <div className="muted" style={{ fontSize: 12 }}>
                head 1's block
              </div>
              <Bars names={FEATURES} values={ref.out} color="var(--c-a)" fmtv={(v) => v.toFixed(2)} />
              <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                head 2's block
              </div>
              <Bars names={FEATURES} values={pred.out} color="var(--c-b)" fmtv={(v) => v.toFixed(2)} />
            </>
          )}
        </div>
      </div>
      <div className="readout" style={{ marginTop: 10 }}>
        <span className="k">share to the referent  </span> {pct(refShare)}
        {'\n'}
        <span className="k">share to the predicates</span> {pct(predShare)}
        {'\n'}
        <span className="k">total                  </span> {pct(refShare + predShare)}
        {mode === 'two' ? '  (two softmaxes: up to 200%)' : '  (one softmax: never more than 100%)'}
      </div>
      <div className="live-note" style={{ marginTop: 10 }}>
        {note}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. what heads cost

const WIDTHS = [256, 512, 768, 1024, 4096];
const HEADS = [1, 2, 4, 8, 12, 16, 32, 64];

export function HeadCost() {
  const [d, setD] = useState(768);
  const [h, setH] = useState(12);
  const [n, setN] = useState(1024);
  const valid = HEADS.filter((k) => d % k === 0);
  const heads = valid.includes(h) ? h : valid[valid.length - 1];
  const p = attentionParams(d, heads);
  const one = attentionParams(d, 1);
  return (
    <Widget
      title="What do heads cost?"
      wide
      caption={
        <>
          A model of width <em>d</em> splits its attention into <em>h</em> heads, each working with vectors of length <em>d</em>/<em>h</em>. Change the number of heads
          and watch what happens to the parameter count and the arithmetic. (Width 768 with 12 heads is GPT-2 small.)
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <Segmented label="model width d" value={d} onChange={setD} options={WIDTHS} />
        <Segmented label="heads h" value={heads} onChange={setH} options={valid} />
        <Segmented label="tokens n" value={n} onChange={setN} options={[128, 1024, 8192]} />
      </div>
      <div className="readout">
        <span className="k">size of each head (d / h)        </span> {p.headSize}
        {'\n'}
        <span className="k">W_Q, W_K, W_V, W_O (4 × d × d)    </span> {fmt(p.weights)} weights + {fmt(p.biases)} biases = <strong>{fmt(p.total)}</strong>
        {'\n'}
        <span className="k">same layer with a single head     </span> {fmt(one.total)}
        {'\n'}
        <span className="k">multiply–adds for scores and blend </span> {fmt(attentionFlops(n, d, heads))}
        {'\n'}
        <span className="k">same with a single head           </span> {fmt(attentionFlops(n, d, 1))}
        {'\n'}
        <span className="k">separate attention patterns       </span> {heads} per token, each {n.toLocaleString('en-GB')} long
      </div>
      <div className="live-note" style={{ marginTop: 10 }}>
        <strong>The number of heads doesn't change the cost.</strong> {heads} heads of size {p.headSize} use exactly the parameters and arithmetic of one head of size {d}.
        What changes is the <em>shape</em> of the computation: {heads} separate softmaxes, {heads} separate attention patterns, each comparing shorter vectors.
        {n >= 8192 ? ' Notice how the arithmetic grows with the square of the number of tokens: that, not the heads, is what makes long contexts expensive.' : ''}
      </div>
    </Widget>
  );
}
