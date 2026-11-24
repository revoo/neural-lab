import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { elementwiseShape, matmulShape, parseShape } from './shapes';

const PRESETS = [
  { name: 'Linear layer on a batch', a: '32, 784', b: '784, 128', op: '@', story: '32 images, each 784 pixels, through a layer with 784 inputs and 128 outputs.' },
  { name: 'Add a bias', a: '32, 128', b: '128', op: '+', story: 'One bias vector of 128 numbers, added to every one of the 32 rows.' },
  { name: 'Transformer projection', a: '8, 512, 768', b: '768, 3072', op: '@', story: '8 sequences of 512 tokens, each a 768-dim vector, projected up to 3,072 dims.' },
  { name: 'Attention scores', a: '8, 12, 512, 64', b: '8, 12, 64, 512', op: '@', story: '8 sequences × 12 attention heads: every token’s 64-dim query dotted with every token’s key.' },
  { name: 'A classic bug', a: '32, 128', b: '32', op: '+', story: 'Trying to add a per-example value to a batch — the dims line up from the wrong end.' },
  { name: 'Wrong orientation', a: '128, 784', b: '32, 784', op: '@', story: 'The weight matrix and the batch, multiplied in the wrong arrangement.' },
] as const;

export function ShapeLab() {
  const [a, setA] = useState('32, 784');
  const [b, setB] = useState('784, 128');
  const [op, setOp] = useState<'@' | '+'>('@');
  const [story, setStory] = useState<string>(PRESETS[0].story);
  const sa = parseShape(a);
  const sb = parseShape(b);
  const res = sa && sb ? (op === '@' ? matmulShape(sa, sb) : elementwiseShape(sa, sb)) : null;
  const inputStyle = { fontFamily: 'var(--font-mono)', fontSize: 15, padding: '7px 10px', borderRadius: 8, border: '1px solid var(--border-strong)', background: 'var(--bg)', color: 'var(--text)', width: '100%' };
  return (
    <Widget
      title="Shape lab"
      wide
      caption={
        <>
          Type two tensor shapes and choose an operation. <strong>@</strong> is matrix multiplication (with any extra leading dimensions treated as a
          batch); <strong>+</strong> is element-wise addition with broadcasting. Start with the presets — they’re shapes that genuinely appear in real
          models — then try to break things.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 14 }}>
        {PRESETS.map((p) => (
          <button
            key={p.name}
            className="btn small"
            onClick={() => {
              setA(p.a);
              setB(p.b);
              setOp(p.op);
              setStory(p.story);
            }}
          >
            {p.name}
          </button>
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) auto minmax(0,1fr)', gap: 12, alignItems: 'end' }}>
        <label>
          <div className="field-label">Left shape</div>
          <input style={inputStyle} value={a} onChange={(e) => { setA(e.target.value); setStory(''); }} aria-label="left tensor shape" />
        </label>
        <Segmented value={op} onChange={(v) => { setOp(v); setStory(''); }} options={[{ value: '@', label: '@' }, { value: '+', label: '+' }]} />
        <label>
          <div className="field-label">Right shape</div>
          <input style={inputStyle} value={b} onChange={(e) => { setB(e.target.value); setStory(''); }} aria-label="right tensor shape" />
        </label>
      </div>
      {story && <p className="muted" style={{ fontSize: 13.5, margin: '10px 0 0' }}>{story}</p>}
      <div style={{ marginTop: 14 }}>
        {!res ? (
          <div className="feedback bad">Shapes should be comma-separated positive integers, like <code>32, 128</code>.</div>
        ) : 'error' in res ? (
          <div className="feedback bad">
            <div className="feedback-title">✕ Shape error</div>
            {res.error}
          </div>
        ) : (
          <div className="feedback good">
            <div className="feedback-title">
              ✓ Result shape: <span className="mono">({res.shape.join(', ')}{res.shape.length === 1 ? ',' : ''})</span>
              {res.shape.length === 0 && ' — a single number (scalar)'}
            </div>
            <span className="muted">{res.note}</span>
          </div>
        )}
      </div>
    </Widget>
  );
}
