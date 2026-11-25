// authored-by: Neon High
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { LogChart } from '../../components/LogChart';
import { createRng } from '../../lib/rng';
import {
  BASELINES,
  ID_OF,
  TEST_SENTENCES,
  V,
  VOCAB,
  bigramProb,
  buildTrainingRun,
  evalSamples,
  makeSamples,
  unigramProb,
  type TinyNeuralLM,
} from './lm';

// Lazy singleton so other lessons pay zero cost until this lesson is opened.
let cachedRun: ReturnType<typeof buildTrainingRun> | null = null;
function getRun() {
  if (!cachedRun) cachedRun = buildTrainingRun(7, 120, 10);
  return cachedRun;
}

const pct = (p: number) => `${(p * 100).toFixed(p < 0.01 ? 2 : 1)}%`;
const f2 = (n: number) => n.toFixed(2);

// ------------------------------------------------------------------ 1. Autoregressive loop

const PROMPT_PRESETS = [
  ['the', 'cat'],
  ['a', 'bright'],
  ['the', 'cold'],
  ['the', 'fox'],
  ['a', 'small'],
];

export function AutoregressiveLoop() {
  const run = useMemo(() => getRun(), []);
  const lm = run.final;
  const [tokens, setTokens] = useState<string[]>(['the', 'cat']);
  const [seed, setSeed] = useState(1);

  const ended = tokens[tokens.length - 1] === '.';
  const w0 = tokens.length >= 2 ? tokens[tokens.length - 2] : '<start>';
  const w1 = tokens.length >= 1 ? tokens[tokens.length - 1] : '<start>';
  const c0 = ID_OF.get(w0) ?? 0;
  const c1 = ID_OF.get(w1) ?? 0;

  const top = useMemo(() => lm.topNext(c0, c1, 8).filter((x) => x.id !== 0), [lm, c0, c1]);
  const topProb = top[0]?.prob ?? 0.01;

  const appendWord = (word: string) => {
    if (ended || tokens.length >= 16) return;
    setTokens((prev) => [...prev, word]);
  };

  const sampleNext = () => {
    if (ended || tokens.length >= 16) return;
    const { probs } = lm.forward(c0, c1);
    const rng = createRng(seed * 97 + tokens.length * 13);
    const total = 1 - probs[0];
    let r = rng.next() * total;
    let pick = 1;
    for (let k = 1; k < V; k++) {
      r -= probs[k];
      if (r <= 0) {
        pick = k;
        break;
      }
    }
    setSeed((s) => s + 1);
    appendWord(VOCAB[pick]);
  };

  const finishSentence = () => {
    if (ended) return;
    const full = lm.generate(tokens, 14 - tokens.length, seed * 31 + 5, 'sample');
    setSeed((s) => s + 1);
    setTokens(full);
  };

  return (
    <Widget
      title="One token at a time: the autoregressive loop"
      wide
      caption={
        <>
          Our trained 1,611-parameter neural language model (61-word vocabulary, 2-word context window). Click <strong>Sample next</strong>, <strong>Pick top</strong>, or click <strong>any candidate bar</strong> to append that word. Watch the highlighted context window slide forward and immediately produce a fresh distribution.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10 }}>
        {PROMPT_PRESETS.map((p) => {
          const label = p.join(' ');
          const active = tokens.length === 2 && tokens[0] === p[0] && tokens[1] === p[1];
          return (
            <button key={label} className={active ? 'btn primary' : 'btn'} onClick={() => setTokens(p)}>
              “{label}”
            </button>
          );
        })}
      </div>

      <div className="field-label">sequence so far (highlighted = active 2-word context)</div>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 6,
          padding: '10px 12px',
          background: 'var(--bg-sunken)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)',
          marginBottom: 10,
          alignItems: 'center',
        }}
      >
        {tokens.map((t, i) => {
          const inCtx = !ended && i >= tokens.length - 2;
          return (
            <span
              key={`${i}-${t}`}
              className="mono"
              style={{
                padding: '4px 8px',
                borderRadius: 5,
                fontSize: 14,
                fontWeight: inCtx ? 700 : 500,
                background: inCtx ? 'var(--accent-soft)' : 'var(--bg-elev)',
                border: `1px solid ${inCtx ? 'var(--accent)' : 'var(--border)'}`,
                color: inCtx ? 'var(--accent-text)' : 'var(--text)',
              }}
            >
              {t}
            </span>
          );
        })}
        {!ended && (
          <span
            className="mono"
            style={{
              padding: '4px 8px',
              borderRadius: 5,
              fontSize: 14,
              border: '1px dashed var(--accent)',
              color: 'var(--accent-text)',
            }}
          >
            ___
          </span>
        )}
      </div>

      <div className="btn-row" style={{ marginBottom: 12 }}>
        <button className="btn primary" onClick={sampleNext} disabled={ended || tokens.length >= 16}>
          Sample next ▸
        </button>
        <button className="btn" onClick={() => top[0] && appendWord(top[0].word)} disabled={ended || tokens.length >= 16}>
          Pick top ({top[0]?.word})
        </button>
        <button className="btn" onClick={finishSentence} disabled={ended}>
          Finish sentence
        </button>
        <button className="btn" onClick={() => setTokens((prev) => prev.slice(0, -1))} disabled={tokens.length <= 2}>
          Undo
        </button>
        <button className="btn" onClick={() => setTokens(tokens.slice(0, 2))} disabled={tokens.length <= 2}>
          Reset
        </button>
      </div>

      {!ended && (
        <>
          <div className="field-label">
            predicted distribution after “{w0} {w1}” — click any bar to choose it
          </div>
          <div style={{ display: 'grid', gap: 5 }}>
            {top.map((item, idx) => (
              <button
                key={item.id}
                type="button"
                onClick={() => appendWord(item.word)}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '76px 1fr 54px',
                  gap: 8,
                  alignItems: 'center',
                  padding: '5px 8px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border)',
                  background: 'var(--bg-sunken)',
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
              >
                <span className="mono" style={{ fontSize: 13, fontWeight: idx === 0 ? 700 : 500, color: idx === 0 ? 'var(--accent-text)' : 'var(--text)' }}>
                  {item.word}
                </span>
                <div style={{ height: 10, borderRadius: 4, background: 'var(--border)', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${(item.prob / topProb) * 100}%`,
                      height: '100%',
                      background: idx === 0 ? 'var(--accent)' : 'var(--c-a)',
                      opacity: idx === 0 ? 1 : 0.72,
                    }}
                  />
                </div>
                <span className="mono" style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'right' }}>
                  {pct(item.prob)}
                </span>
              </button>
            ))}
          </div>
        </>
      )}

      <div className="live-note" style={{ marginTop: 10 }}>
        {ended ? (
          <>
            <strong>End of sentence (<code>.</code>).</strong> In a real LLM, a special end-of-turn token plays the role of this full stop: the loop keeps running until the model predicts it. Reset or try another prompt to see a different continuation.
          </>
        ) : (
          <>
            <strong>Step {tokens.length - 1}:</strong> feeding <code>[{w0}, {w1}]</code> through the network gives 61 logits → softmax. Top choice is{' '}
            <code>{top[0].word}</code> ({pct(top[0].prob)}). Notice there is no separate "writing" engine — generating a paragraph is just calling this exact single-word predictor in a loop.
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Live training & perplexity curve

export function LiveTrainingPpl() {
  const run = useMemo(() => getRun(), []);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    if (idx >= run.history.length - 1) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setIdx((i) => Math.min(run.history.length - 1, i + 1)), 260);
    return () => window.clearTimeout(id);
  }, [playing, idx, run.history.length]);

  const h = run.history[idx];
  const lm = run.snapshots.get(h.epoch)!;
  const catTop = useMemo(() => lm.topNext(ID_OF.get('the')!, ID_OF.get('cat')!, 5), [lm]);
  const foxTop = useMemo(() => lm.topNext(ID_OF.get('the')!, ID_OF.get('fox')!, 5), [lm]);

  const series = useMemo(
    () => [
      { points: run.history.map((r): [number, number] => [r.epoch, r.trainPpl]), color: 'var(--c-a)', label: 'train PPL' },
      { points: run.history.map((r): [number, number] => [r.epoch, r.testPpl]), color: 'var(--c-b)', label: 'test PPL' },
    ],
    [run.history],
  );

  return (
    <Widget
      title="Watch perplexity fall as the model trains"
      wide
      caption={
        <>
          Real training run of our 1,611-parameter neural LM on 30 sentences (268 training tokens, 37 held-out test tokens, 61-word vocabulary). Press <strong>Play</strong> or drag the epoch slider. Watch test perplexity drop past uniform (61), unigram (32.8) and validation-tuned bigram (5.6) — and watch the generated sentence turn from word salad into English.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10 }}>
        <button
          className="btn primary"
          onClick={() => {
            if (idx >= run.history.length - 1) setIdx(0);
            setPlaying((p) => !p);
          }}
        >
          {playing ? 'Pause' : idx >= run.history.length - 1 ? 'Replay' : 'Play training ▸'}
        </button>
        <button
          className="btn"
          onClick={() => {
            setPlaying(false);
            setIdx((i) => Math.min(run.history.length - 1, i + 1));
          }}
          disabled={idx >= run.history.length - 1}
        >
          +10 epochs
        </button>
        <button
          className="btn"
          onClick={() => {
            setPlaying(false);
            setIdx(run.history.length - 1);
          }}
        >
          Epoch 120
        </button>
        <button
          className="btn"
          onClick={() => {
            setPlaying(false);
            setIdx(0);
          }}
          disabled={idx === 0}
        >
          Reset
        </button>
      </div>

      <Slider
        label="training epoch"
        value={idx}
        min={0}
        max={run.history.length - 1}
        step={1}
        onChange={(v) => {
          setPlaying(false);
          setIdx(v);
        }}
        format={(v) => `${run.history[v].epoch} / 120`}
        color="accent"
      />

      <div className="widget-row" style={{ marginTop: 10 }}>
        <div className="widget-col" style={{ flex: '1.2 1 320px', minWidth: 0 }}>
          <LogChart
            series={series}
            xDomain={[0, 120]}
            yDomain={[2, 100]}
            xTicks={[0, 30, 60, 90, 120]}
            marker={h.epoch}
            xLabel="epochs"
            ariaLabel="Train and held-out test perplexity over training epochs"
            hlines={[
              { y: V, label: `uniform (${V})`, color: 'var(--text-faint)' },
              { y: BASELINES.testUnigram.ppl, label: `unigram (${BASELINES.testUnigram.ppl.toFixed(1)})`, color: 'var(--text-muted)' },
              { y: BASELINES.testBigram.ppl, label: `bigram (${BASELINES.testBigram.ppl.toFixed(2)})`, color: 'var(--c-c)' },
            ]}
          />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 14px', fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
            <span>
              <strong style={{ color: 'var(--c-a)' }}>—</strong> train PPL ({f2(h.trainPpl)})
            </span>
            <span>
              <strong style={{ color: 'var(--c-b)' }}>—</strong> held-out test PPL ({f2(h.testPpl)})
            </span>
          </div>
        </div>

        <div className="widget-col" style={{ flex: '1 1 240px', minWidth: 0 }}>
          <div className="field-label">sample generated at epoch {h.epoch}</div>
          <div className="readout" style={{ fontSize: 13.5, marginBottom: 10, lineHeight: 1.5 }}>
            {h.sample}
          </div>

          <div className="field-label">top guesses after “the cat ___”</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: 10 }}>
            {catTop.map((x) => (
              <span
                key={x.id}
                className="mono"
                style={{
                  fontSize: 12,
                  padding: '3px 7px',
                  borderRadius: 5,
                  background: 'var(--bg-sunken)',
                  border: '1px solid var(--border)',
                }}
              >
                {x.word} <strong style={{ color: 'var(--c-a)' }}>{pct(x.prob)}</strong>
              </span>
            ))}
          </div>

          <div className="field-label">unseen pairs: “the fox ___”</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
            {foxTop.map((x) => (
              <span
                key={x.id}
                className="mono"
                style={{
                  fontSize: 12,
                  padding: '3px 7px',
                  borderRadius: 5,
                  background: 'var(--bg-sunken)',
                  border: '1px solid var(--border)',
                }}
              >
                {x.word} <strong style={{ color: 'var(--c-b)' }}>{pct(x.prob)}</strong>
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {h.epoch === 0 ? (
          <>
            <strong>Epoch 0 (untrained):</strong> test perplexity is <strong>{f2(h.testPpl)}</strong> — right near the 61-word uniform floor. The model spreads roughly 1.6% across every word and generates pure word salad.
          </>
        ) : h.epoch < 60 ? (
          <>
            <strong>Epoch {h.epoch}:</strong> test perplexity has dropped to <strong>{f2(h.testPpl)}</strong> (well past unigram's {BASELINES.testUnigram.ppl.toFixed(1)}). The model already knows common words and sentence-ending full stops, though its phrasing is still bumpy.
          </>
        ) : h.epoch < 120 ? (
          <>
            <strong>Epoch {h.epoch}:</strong> test perplexity is <strong>{f2(h.testPpl)}</strong>, now matching or beating the validation-tuned bigram ({BASELINES.testBigram.ppl.toFixed(2)}). Look at <code>the fox ___</code>: even though <code>fox drank</code> and <code>fox chased</code> never appeared in training, the model is starting to transfer animal verbs from <code>cat</code> and <code>dog</code>.
          </>
        ) : (
          <>
            <strong>Epoch 120 (trained):</strong> held-out test perplexity reaches <strong>{f2(h.testPpl)}</strong> — out of 61 words, the model is effectively choosing among 4–5 plausible candidates. Notice <code>the fox ___</code> now predicts <code>slept</code>, <code>drank</code> and <code>chased</code> even though only <code>fox ran</code> and <code>fox slept</code> were ever in the training set.
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. Perplexity intuition & blunder penalty

const BASE_PROBS = [0.8, 0.9, 0.85, 0.9];
const TOY_WORDS = ['the', 'cat', 'sat', 'on', 'roof'];

export function PerplexityIntuition() {
  const [p5, setP5] = useState(0.95);
  const [uniformMode, setUniformMode] = useState<number | null>(null);

  const probs = uniformMode ? new Array(5).fill(1 / uniformMode) : [...BASE_PROBS, p5];
  const surprises = probs.map((p) => -Math.log(p));
  const avgLoss = surprises.reduce((a, b) => a + b, 0) / probs.length;
  const ppl = Math.exp(avgLoss);
  const bits = avgLoss / Math.LN2;
  const maxS = Math.max(5, ...surprises);

  return (
    <Widget
      title="Why one overconfident blunder wrecks perplexity"
      wide
      caption={
        <>
          Five predicted words, each scored by the probability the model gave the word that actually happened. Drag the probability on the fifth word from 95% down to 1% — or click a preset — and watch how one confident wrong guess dominates the average surprise.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10 }}>
        <button
          className={uniformMode === 2 ? 'btn primary' : 'btn'}
          onClick={() => setUniformMode(2)}
        >
          Fair coin (50%)
        </button>
        <button
          className={uniformMode === 6 ? 'btn primary' : 'btn'}
          onClick={() => setUniformMode(6)}
        >
          Fair 6-sided die
        </button>
        <button
          className={!uniformMode && p5 === 0.95 ? 'btn primary' : 'btn'}
          onClick={() => {
            setUniformMode(null);
            setP5(0.95);
          }}
        >
          Confident & right (95%)
        </button>
        <button
          className={!uniformMode && p5 === 0.01 ? 'btn primary' : 'btn'}
          onClick={() => {
            setUniformMode(null);
            setP5(0.01);
          }}
        >
          One 1% blunder
        </button>
      </div>

      <Slider
        label="probability assigned to word 5 (“roof”)"
        value={uniformMode ? 1 / uniformMode : p5}
        min={0.01}
        max={0.99}
        step={0.01}
        onChange={(v) => {
          setUniformMode(null);
          setP5(v);
        }}
        format={(v) => `${Math.round(v * 100)}%`}
        color="c-b"
      />

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, minmax(0, 1fr))', gap: 8, marginTop: 12 }}>
        {probs.map((p, i) => {
          const s = surprises[i];
          const isBlunder = !uniformMode && i === 4 && p < 0.1;
          return (
            <div
              key={i}
              style={{
                background: 'var(--bg-sunken)',
                border: `1px solid ${isBlunder ? 'var(--c-bad)' : 'var(--border)'}`,
                borderRadius: 'var(--radius-sm)',
                padding: '8px 6px',
                textAlign: 'center',
              }}
            >
              <div className="mono" style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--text)' }}>
                {uniformMode ? `t=${i + 1}` : TOY_WORDS[i]}
              </div>
              <div className="mono" style={{ fontSize: 11.5, color: isBlunder ? 'var(--c-bad)' : 'var(--c-a)', marginTop: 2 }}>
                p = {(p * 100).toFixed(p < 0.1 ? 1 : 0)}%
              </div>
              <div style={{ height: 64, display: 'flex', alignItems: 'flex-end', justifyContent: 'center', margin: '6px 0' }}>
                <div
                  style={{
                    width: 24,
                    height: `${Math.max(4, (s / maxS) * 60)}px`,
                    borderRadius: 4,
                    background: isBlunder ? 'var(--c-bad)' : 'var(--c-b)',
                  }}
                />
              </div>
              <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                {s.toFixed(2)} nats
              </div>
            </div>
          );
        })}
      </div>

      <div className="readout" style={{ marginTop: 10, fontSize: 13 }}>
        avg cross-entropy H = <strong>{avgLoss.toFixed(3)} nats</strong> ({bits.toFixed(2)} bits/token) ·{' '}
        perplexity e^H = <strong style={{ color: 'var(--accent-text)' }}>{ppl.toFixed(2)}</strong> (equivalent to a{' '}
        <strong>{ppl.toFixed(1)}-sided</strong> fair die)
      </div>

      <div className="live-note" style={{ marginTop: 10 }}>
        {uniformMode === 2 ? (
          <>
            <strong>Fair coin:</strong> 50% at every step → surprise = ln 2 ≈ 0.693 nats (1.00 bit) → perplexity = <strong>2.00</strong>. Two equally likely outcomes.
          </>
        ) : uniformMode === 6 ? (
          <>
            <strong>Fair 6-sided die:</strong> 16.7% at every step → perplexity = <strong>6.00</strong>. That is the exact meaning of perplexity: the size of the fair die that would leave you just as surprised.
          </>
        ) : p5 <= 0.05 ? (
          <>
            <strong>One overconfident miss ({Math.round(p5 * 100)}%):</strong> even though the first four words were nailed at 80–90%, that single surprise of <strong>{surprises[4].toFixed(2)} nats</strong> more than doubles the whole sentence's perplexity to <strong>{ppl.toFixed(2)}</strong>. Cross-entropy punishes arrogance brutally — which is why well-trained models learn to stay calibrated.
          </>
        ) : (
          <>
            <strong>Calibrated predictions:</strong> average loss is {avgLoss.toFixed(3)} nats, giving perplexity <strong>{ppl.toFixed(2)}</strong>. Try sliding word 5 down to 1% to feel the logarithmic penalty.
          </>
        )}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Shannon's betting game

const LAB_KEY = 'nl-ntp-lab-v1';

function loadBestScores(): Record<number, number> {
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    // ignore
  }
  return {};
}

/** Build 5 candidate words for position (c0, c1) from the neural LM's top predictions (never peeking at trueY!), sorted alphabetically. */
function candidatesFor(lm: TinyNeuralLM, c0: number, c1: number): { id: number; word: string }[] {
  const top5 = lm
    .topNext(c0, c1, 6)
    .filter((x) => x.id !== 0)
    .slice(0, 5)
    .map(({ id, word }) => ({ id, word }));
  top5.sort((a, b) => a.word.localeCompare(b.word));
  return top5;
}

type LogEntry = {
  word: string;
  youP: number;
  uniP: number;
  biP: number;
  neuP: number;
  inTop5: boolean;
};


/** Bits lost on each word so far (−log₂ p): you, the neural LM and Unigram side by side, so you can see exactly where you lost. */
function BitsStrip({ log }: { log: { word: string; youP: number; neuP: number; uniP: number }[] }) {
  const bits = (p: number) => -Math.log2(Math.max(1e-12, p));
  const cap = 14;
  const col = 34;
  const H = 120;
  const W = Math.max(log.length * col + 30, 200);
  const series = [
    { key: 'youP' as const, color: 'var(--accent)', label: 'you' },
    { key: 'neuP' as const, color: 'var(--c-c)', label: 'neural LM' },
    { key: 'uniP' as const, color: 'var(--text-faint)', label: 'Unigram' },
  ];
  const y = (b: number) => 8 + ((H - 34) * (cap - Math.min(cap, b))) / cap;
  return (
    <div style={{ marginTop: 10 }}>
      <div className="field-label">bits lost per word (taller = more surprised)</div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Bits lost on each word by you, the neural LM and Unigram">
          {[0, 5, 10].map((b) => (
            <g key={b}>
              <line x1={24} x2={W} y1={y(b)} y2={y(b)} stroke="var(--border)" />
              <text x={20} y={y(b) + 3} fontSize={9} textAnchor="end" className="plot-tick">{b}</text>
            </g>
          ))}
          {log.map((e, i) => (
            <g key={i}>
              {series.map((sr, k) => {
                const b = bits(e[sr.key]);
                return <rect key={sr.key} x={28 + i * col + k * 9} y={y(b)} width={8} height={y(0) - y(b)} fill={sr.color} opacity={0.9}><title>{`${e.word}: ${sr.label} ${b.toFixed(2)} bits`}</title></rect>;
              })}
              <text x={28 + i * col + 13} y={H - 12} fontSize={9.5} textAnchor="middle" className="plot-tick">{e.word.length > 6 ? `${e.word.slice(0, 5)}…` : e.word}</text>
            </g>
          ))}
        </svg>
      </div>
      <div style={{ display: 'flex', gap: 12, fontSize: 11.5 }}>
        {series.map((sr) => (
          <span key={sr.key} style={{ color: sr.color === 'var(--text-faint)' ? 'var(--text-muted)' : sr.color }}>■ {sr.label}</span>
        ))}
        <span className="muted">bars are capped at {cap} bits</span>
      </div>
    </div>
  );
}

export function ShannonLab() {
  const run = useMemo(() => getRun(), []);
  const lm = run.final;

  const [sentIdx, setSentIdx] = useState(0);
  const samples = useMemo(() => makeSamples(TEST_SENTENCES[sentIdx]), [sentIdx]);
  const [pos, setPos] = useState(0);
  const [weights, setWeights] = useState<number[]>([16, 16, 16, 16, 16, 20]); // 5 candidates + "other 56"
  const [log, setLog] = useState<LogEntry[]>([]);
  const [bestScores, setBestScores] = useState<Record<number, number>>(loadBestScores);

  const done = pos >= samples.length;
  const cur = !done ? samples[pos] : null;
  const cands = useMemo(() => (cur ? candidatesFor(lm, cur.c0, cur.c1) : []), [lm, cur]);

  const totalW = weights.reduce((a, b) => a + b, 0);
  const shares = weights.map((w) => w / totalW);

  const baselines = useMemo(() => {
    const uni = evalSamples(samples, (_a, _b, y) => unigramProb(y, BASELINES.uniAlpha)).ppl;
    const bi = evalSamples(samples, (_a, c1, y) => bigramProb(c1, y, BASELINES.biAlpha)).ppl;
    const neu = lm.evaluate(samples).ppl;
    return { uni, bi, neu };
  }, [samples, lm]);

  const switchSentence = (i: number) => {
    setSentIdx(i);
    setPos(0);
    setLog([]);
    setWeights([16, 16, 16, 16, 16, 20]);
  };

  const lockBet = () => {
    if (!cur) return;
    const matchIdx = cands.findIndex((c) => c.id === cur.y);
    const otherBucketCount = V - 1 - cands.length; // exclude <start> and the 5 shown candidates
    // Keep a tiny floor of 1e-4 on any word so a zero bet doesn't give NaN/Infinity; still brutally penalised (~9.2 nats).
    const youP = Math.max(1e-4, matchIdx >= 0 ? shares[matchIdx] : shares[5] / otherBucketCount);
    const uniP = unigramProb(cur.y, BASELINES.uniAlpha);
    const biP = bigramProb(cur.c1, cur.y, BASELINES.biAlpha);
    const neuP = lm.forward(cur.c0, cur.c1).probs[cur.y];

    const nextLog = [...log, { word: VOCAB[cur.y], youP, uniP, biP, neuP, inTop5: matchIdx >= 0 }];
    setLog(nextLog);
    const nextPos = pos + 1;
    setPos(nextPos);
    setWeights([16, 16, 16, 16, 16, 20]);

    if (nextPos >= samples.length) {
      const finalPpl = Math.exp(-nextLog.reduce((s, e) => s + Math.log(e.youP), 0) / nextLog.length);
      setBestScores((prev) => {
        const best = prev[sentIdx] ? Math.min(prev[sentIdx], finalPpl) : finalPpl;
        const updated = { ...prev, [sentIdx]: Number(best.toFixed(2)) };
        try {
          localStorage.setItem(LAB_KEY, JSON.stringify(updated));
        } catch {
          // ignore
        }
        return updated;
      });
    }
  };

  const runningPpl = (key: 'youP' | 'uniP' | 'biP' | 'neuP') => {
    if (!log.length) return null;
    return Math.exp(-log.reduce((s, e) => s + Math.log(Math.max(1e-12, e[key])), 0) / log.length);
  };

  const youPpl = runningPpl('youP');
  const biPpl = runningPpl('biP');
  const neuPpl = runningPpl('neuP');
  const goal1 = done && youPpl !== null && youPpl < baselines.uni;
  const goal2 = done && youPpl !== null && youPpl < baselines.neu;

  const adjust = (idx: number, delta: number) => {
    setWeights((prev) => {
      const next = prev.slice();
      next[idx] = Math.max(idx === 5 ? 1 : 0, Math.min(200, next[idx] + delta));
      if (next.reduce((a, b) => a + b, 0) === 0) next[5] = 1;
      return next;
    });
  };

  return (
    <Widget
      title="Lab: Shannon’s betting game — beat the model’s perplexity"
      wide
      caption={
        <>
          Claude Shannon measured the entropy of English in 1951 by asking people to guess the next symbol one step at a time. Choose a held-out sentence, distribute your 100% probability budget across the candidates <em>before</em> revealing each word, and try to finish with lower perplexity than the Unigram model — and, optionally, lower than our trained neural LM.
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <Segmented
          label="held-out test sentence"
          options={TEST_SENTENCES.map((_, i) => ({
            value: i,
            label: bestScores[i] ? `#${i + 1} (${bestScores[i]})` : `#${i + 1}`,
          }))}
          value={sentIdx}
          onChange={switchSentence}
        />
        <button className="btn" onClick={() => switchSentence(sentIdx)} disabled={pos === 0}>
          Restart sentence
        </button>
      </div>

      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 12,
          padding: '8px 12px',
          borderRadius: 'var(--radius-sm)',
          background: 'var(--bg-sunken)',
          border: '1px solid var(--border)',
          marginBottom: 10,
          fontSize: 13,
        }}
      >
        <span style={{ color: goal1 ? 'var(--c-good)' : 'var(--text-muted)', fontWeight: goal1 ? 700 : 500 }}>
          {goal1 ? '✓' : '○'} Goal: beat Unigram ({f2(baselines.uni)})
        </span>
        <span style={{ color: goal2 ? 'var(--c-good)' : 'var(--text-faint)', fontWeight: goal2 ? 700 : 500 }}>
          {goal2 ? '✓' : '○'} Optional: beat Neural LM ({f2(baselines.neu)})
        </span>
        <span style={{ color: 'var(--text-faint)' }}>
          Bigram ref: {f2(baselines.bi)} · word {Math.min(pos + 1, samples.length)} / {samples.length}
        </span>
      </div>

      <div className="field-label">revealed prefix</div>
      <div
        className="mono"
        style={{
          padding: '10px 12px',
          background: 'var(--bg-sunken)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)',
          fontSize: 15,
          marginBottom: 12,
          minHeight: 42,
        }}
      >
        {log.length === 0 ? <span style={{ color: 'var(--text-faint)' }}>&lt;start&gt; </span> : log.map((e) => e.word).join(' ') + ' '}
        {!done && (
          <span style={{ color: 'var(--accent-text)', fontWeight: 700, borderBottom: '2px solid var(--accent)' }}>???</span>
        )}
      </div>

      {!done && (
        <>
          <div className="field-label">allocate your probability budget (auto-normalised to 100%; candidates are sorted alphabetically)</div>
          <div style={{ display: 'grid', gap: 6, marginBottom: 10 }}>
            {[...cands.map((c) => c.word), `other ${V - 1 - cands.length} words`].map((label, i) => {
              const isOther = i === 5;
              const p = shares[i];
              return (
                <div
                  key={label}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'minmax(92px, 130px) 1fr auto',
                    gap: 8,
                    alignItems: 'center',
                    padding: '5px 8px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-sunken)',
                    border: '1px solid var(--border)',
                  }}
                >
                  <span className="mono" style={{ fontSize: 13, fontWeight: 600, color: isOther ? 'var(--text-muted)' : 'var(--text)' }}>
                    {label}
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                    <div style={{ flex: 1, height: 8, borderRadius: 4, background: 'var(--border)', overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${p * 100}%`,
                          height: '100%',
                          background: isOther ? 'var(--c-b)' : 'var(--accent)',
                        }}
                      />
                    </div>
                    <span className="mono" style={{ fontSize: 12, width: 44, textAlign: 'right', color: 'var(--text-muted)' }}>
                      {(p * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div style={{ display: 'inline-flex', gap: 4 }}>
                    <button type="button" className="btn" style={{ padding: '2px 7px', fontSize: 12 }} onClick={() => adjust(i, -10)}>
                      −
                    </button>
                    <button type="button" className="btn" style={{ padding: '2px 7px', fontSize: 12 }} onClick={() => adjust(i, 15)}>
                      +
                    </button>
                    <button type="button" className="btn" style={{ padding: '2px 7px', fontSize: 12 }} onClick={() => adjust(i, 45)}>
                      ++
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="btn-row" style={{ marginBottom: 10 }}>
            <button className="btn primary" onClick={lockBet}>
              Lock bet &amp; reveal word {pos + 1} ▸
            </button>
            <button className="btn" onClick={() => setWeights([16, 16, 16, 16, 16, 20])}>
              Spread evenly
            </button>
          </div>
        </>
      )}

      {log.length > 0 && (
        <div className="readout" style={{ fontSize: 12.5, display: 'grid', gap: 4 }}>
          <div>
            last word: <strong>“{log[log.length - 1].word}”</strong> — you gave it{' '}
            <strong style={{ color: log[log.length - 1].youP < 0.05 ? 'var(--c-bad)' : 'var(--c-good)' }}>
              {pct(log[log.length - 1].youP)}
            </strong>{' '}
            ({(-Math.log(log[log.length - 1].youP)).toFixed(2)} nats) · Neural LM gave{' '}
            <strong>{pct(log[log.length - 1].neuP)}</strong>
            {!log[log.length - 1].inTop5 && ' (was in the “other” bucket!)'}
          </div>
          <div>
            running perplexity ({log.length}/{samples.length} words) —{' '}
            <strong style={{ color: 'var(--accent-text)' }}>You: {f2(youPpl!)}</strong> · Neural LM:{' '}
            <strong style={{ color: 'var(--c-c)' }}>{f2(neuPpl!)}</strong> · Bigram: <strong>{f2(biPpl!)}</strong>
          </div>
        </div>
      )}

      {log.length > 0 && <BitsStrip log={log} />}

      <div className="live-note" style={{ marginTop: 10 }}>
        {done ? (
          goal2 ? (
            <>
              <strong>You beat the neural LM ({f2(youPpl!)} vs {f2(baselines.neu)})!</strong> Even on a tiny 61-word world, your grasp of full-sentence grammar and meaning outperforms a 2-word context window. Notice how leaving a sensible share on <em>other words</em> protects you whenever an unexpected word shows up.
            </>
          ) : goal1 ? (
            <>
              <strong>Goal cleared — you beat Unigram ({f2(youPpl!)} vs {f2(baselines.uni)})!</strong> Our trained neural LM scored <strong>{f2(baselines.neu)}</strong> on this sentence. Check if one tiny bet on an unexpected word inflated your score — that's the exact calibration pressure cross-entropy puts on every LLM during training.
            </>
          ) : (
            <>
              <strong>Finished at perplexity {f2(youPpl!)}</strong> (Unigram {f2(baselines.uni)}, Neural LM {f2(baselines.neu)}). One overconfident 0% guess on the true word adds over 9 nats of surprise! Try restarting and keeping a realistic safety margin on plausible words and the <em>other</em> bucket.
            </>
          )
        ) : (
          <>
            <strong>Tip:</strong> the five candidate buttons come from the model's top 5 guesses (sorted alphabetically, without peeking at the answer), so the true word is <em>usually</em> among them — but not always! Never starve the <em>other</em> bucket completely unless you're certain.
          </>
        )}
      </div>
    </Widget>
  );
}
