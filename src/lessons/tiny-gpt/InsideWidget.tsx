// authored-by: Claude Opus 5.5
/**
 * "Open it up": attention per head for a prompt, the next-character distribution at any position,
 * and temperature / top-p on the model's real logits. Uses the recorded run's final weights
 * (`tinyGpt.cpu.bin`, checked against the record by a test) or the learner's own latest checkpoint.
 */
import { useEffect, useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Gpt, sampleDistribution } from '../../lib/compute/gpt';
import { CPU_RUN } from '../../lib/compute/gptReference';
import { CHARS, encode } from '../../lib/corpora/grimm';
import binUrl from '../../lib/compute/tinyGpt.cpu.bin?url';
import { attentionFor, continueText, showChar } from './tinyGpt';
import { useLiveRun } from './runStore';
import { cleanPrompt } from './TrainWidget';

let refParams: Promise<Float32Array> | null = null;
export function loadReferenceParams(): Promise<Float32Array> {
  refParams ??= fetch(binUrl)
    .then((r) => r.arrayBuffer())
    .then((b) => new Float32Array(b));
  return refParams;
}

/** The model the "open it up" widgets look inside: the recorded one, or the learner's latest checkpoint. */
export function useInspectModel() {
  const live = useLiveRun();
  const [src, setSrc] = useState<'recorded' | 'yours'>('recorded');
  const [ref, setRef] = useState<Float32Array | null>(null);
  useEffect(() => {
    let ok = true;
    loadReferenceParams().then((p) => ok && setRef(p));
    return () => {
      ok = false;
    };
  }, []);
  const yours = live && live.checkpoints.length > 1 ? live.checkpoints.at(-1)! : null;
  const useYours = src === 'yours' && yours && live;
  const model = useMemo(() => {
    const spec = useYours ? live!.spec : CPU_RUN;
    const p = useYours ? yours!.params : ref;
    if (!p) return null;
    const m = new Gpt(spec.cfg);
    m.load(p);
    return m;
  }, [useYours, yours, ref, live]);
  const picker = yours ? (
    <Segmented
      options={[
        { value: 'recorded', label: 'recorded model (step 2,000)' },
        { value: 'yours', label: `your ${live!.label} (step ${yours.step.toLocaleString('en-GB')})` },
      ]}
      value={src}
      onChange={(v) => setSrc(v)}
    />
  ) : null;
  return { model, picker };
}

const MAXLEN = CPU_RUN.cfg.ctx;

function HeadMap({ att, T, pos, label, onPick }: { att: Float64Array; T: number; pos: number; label: string; onPick: (i: number) => void }) {
  const c = 4.2;
  const size = c * T;
  return (
    <figure style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${size} ${size}`} style={{ width: '100%', maxWidth: 170, display: 'block', background: 'var(--bg)', border: '1px solid var(--border)' }} role="img" aria-label={label}>
        {Array.from({ length: T }, (_, i) =>
          Array.from({ length: i + 1 }, (_, j) => {
            const w = att[i * T + j];
            return w > 0.02 ? <rect key={`${i}-${j}`} x={j * c} y={i * c} width={c} height={c} fill="var(--accent)" opacity={Math.min(1, 0.08 + w)} onClick={() => onPick(i)} /> : null;
          }),
        )}
        <rect x={0} y={pos * c} width={size} height={c} fill="none" stroke="var(--c-b)" strokeWidth={0.8} />
      </svg>
      <figcaption className="plot-tick" style={{ textAlign: 'center' }}>{label}</figcaption>
    </figure>
  );
}

export function InsideWidget() {
  const { model, picker } = useInspectModel();
  const [text, setText] = useState("The king said to his daughter, 'You shall");
  const [pos, setPos] = useState(-1);
  const [head, setHead] = useState<[number, number]>([0, 0]);
  const [temp, setTemp] = useState(1);
  const [topP, setTopP] = useState(1);
  const [seed, setSeed] = useState(1);
  const clean = (cleanPrompt(text) || 'The').slice(-MAXLEN);
  const view = useMemo(() => (model ? attentionFor(model, clean) : null), [model, clean]);
  const T = view?.tokens.length ?? 0;
  const p = pos < 0 || pos >= T ? T - 1 : pos;
  // next-character distribution after the prefix ending at position p (temperature 1, no top-p)
  const next = useMemo(() => {
    if (!model) return [];
    const probs = sampleDistribution(model.nextLogits(encode(clean.slice(0, p + 1))), 1, 1);
    return Array.from(probs, (v, i) => ({ ch: CHARS[i], p: v })).sort((a, b) => b.p - a.p).slice(0, 8);
  }, [model, clean, p]);
  const shaped = useMemo(() => {
    if (!model) return [];
    const logits = model.nextLogits(encode(clean));
    const probs = sampleDistribution(logits, temp, topP);
    return Array.from(probs, (v, i) => ({ ch: CHARS[i], p: v })).sort((a, b) => b.p - a.p).slice(0, 10);
  }, [model, clean, temp, topP]);
  const kept = model ? Array.from(sampleDistribution(model.nextLogits(encode(clean)), temp, topP)).filter((v) => v > 0).length : 0;
  const cont = useMemo(() => (model ? continueText(model, clean, 120, seed, temp, topP) : ''), [model, clean, seed, temp, topP]);
  const att = view?.att[head[0]]?.[head[1]];
  const row = att ? Array.from({ length: T }, (_, j) => (j <= p ? att[p * T + j] : 0)) : [];
  const fmtP = (v: number) => `${(100 * v).toFixed(1)}%`;

  return (
    <Widget
      title="Open it up: attention, predictions and sampling"
      wide
      caption={
        <>
          Every number here is the trained model’s own computation on your text. The small squares are the attention pattern of each head (row = the character doing the
          looking, column = the character looked at; darker = more weight). Click a character to make it the current position: the bars show what the model expects next
          there, and the coloured strip shows where the selected head looks from it. The sliders reshape the final prediction exactly as in the sampling lesson.
        </>
      }
    >
      {picker && <div style={{ marginBottom: 8 }}>{picker}</div>}
      <input className="text-input" value={text} onChange={(e) => (setText(e.target.value), setPos(-1))} maxLength={120} style={{ width: '100%' }} aria-label="Text to inspect" />
      {!view && <p className="live-note">Loading the trained weights…</p>}
      {view && (
        <>
          <div className="mono" style={{ display: 'flex', flexWrap: 'wrap', margin: '10px 0', fontSize: 15 }} aria-label="Characters; click one">
            {view.tokens.map((ch, j) => (
              <button
                key={j}
                onClick={() => setPos(j)}
                title={`position ${j + 1}`}
                style={{
                  border: 'none',
                  padding: '3px 1px',
                  minWidth: 12,
                  cursor: 'pointer',
                  font: 'inherit',
                  color: 'var(--text)',
                  background: j === p ? 'var(--accent-soft)' : `color-mix(in srgb, var(--c-b) ${Math.round(100 * (row[j] ?? 0))}%, transparent)`,
                  outline: j === p ? '1px solid var(--accent)' : 'none',
                }}
              >
                {showChar(ch)}
              </button>
            ))}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(120px, 45%), 1fr))', gap: 10 }}>
            {view.att.map((layer, l) =>
              layer.map((a, h) => (
                <div key={`${l}-${h}`} onClick={() => setHead([l, h])} style={{ cursor: 'pointer', outline: head[0] === l && head[1] === h ? '2px solid var(--c-b)' : 'none', borderRadius: 4 }}>
                  <HeadMap att={a} T={T} pos={p} label={`block ${l + 1}, head ${h + 1}`} onPick={setPos} />
                </div>
              )),
            )}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(260px, 100%), 1fr))', gap: 14, marginTop: 12 }}>
            <div>
              <div className="field-label" style={{ textTransform: 'none' }}>
                After “…{view.tokens.slice(Math.max(0, p - 11), p + 1).join('')}”, it expects:
              </div>
              {next.map((n) => (
                <div key={n.ch} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13 }}>
                  <span className="mono" style={{ width: 18, textAlign: 'center' }}>{showChar(n.ch)}</span>
                  <span style={{ height: 10, width: `${70 * n.p}%`, background: 'var(--accent)', borderRadius: 3 }} />
                  <span className="mono">{fmtP(n.p)}</span>
                </div>
              ))}
              {p + 1 < T && (
                <div className="plot-tick" style={{ marginTop: 4 }}>
                  The real next character is “{showChar(view.tokens[p + 1])}”: the model gave it {fmtP(Array.from(sampleDistribution(model!.nextLogits(encode(clean.slice(0, p + 1))), 1, 1))[CHARS.indexOf(view.tokens[p + 1])])}.
                </div>
              )}
            </div>
            <div>
              <Slider label="temperature" value={temp} min={0} max={2} step={0.05} onChange={setTemp} format={(v) => (v === 0 ? '0 (greedy)' : v.toFixed(2))} />
              <Slider label="top-p" value={topP} min={0.05} max={1} step={0.05} onChange={setTopP} format={(v) => v.toFixed(2)} />
              <div className="plot-tick">After the whole text: {kept} of 68 characters can still be drawn.</div>
              {shaped.filter((s) => s.p > 0).map((n) => (
                <div key={n.ch} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13 }}>
                  <span className="mono" style={{ width: 18, textAlign: 'center' }}>{showChar(n.ch)}</span>
                  <span style={{ height: 10, width: `${70 * n.p}%`, background: 'var(--c-c)', borderRadius: 3 }} />
                  <span className="mono">{fmtP(n.p)}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="gpt-panel" style={{ marginTop: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
              <span className="field-label" style={{ textTransform: 'none' }}>A continuation with these settings (seed {seed})</span>
              <button className="btn" onClick={() => setSeed((s) => s + 1)}>
                ↻ another draw
              </button>
            </div>
            <div className="mono gpt-sample">
              {clean}
              <b>{cont}</b>
            </div>
          </div>
        </>
      )}
    </Widget>
  );
}
