import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Slider } from '../../components/controls';
import { Tex, col, fmt, pn } from '../../components/Tex';
import { buildGraph, lossAfterUpdate, type NodeKey, type Params } from './graph';

const DEFAULT: Params = { x1: 2, w1: -0.5, x2: 1, w2: 0.8, b: 0.4, y: 0.6 };
const QUIZ_NODE: NodeKey = 'w1';

const CW = 166;
const RH = 66;
const BW = 116;
const BH = 54;
const px = (c: number) => 14 + c * CW;
const py = (r: number) => 16 + r * RH;

export function BackpropStepper() {
  const [p, setP] = useState<Params>(DEFAULT);
  const [t, setT] = useState(0);
  const [lr, setLr] = useState(0.5);
  const [guess, setGuess] = useState('');
  const [quizState, setQuizState] = useState<'none' | 'asking' | 'right' | 'wrong' | 'revealed'>('none');
  const g = useMemo(() => buildGraph(p), [p]);
  const F = g.forwardOrder.length;
  const B = g.backwardOrder.length;
  const total = F + B + 1;

  const fwdDone = new Set<NodeKey>(g.forwardOrder.slice(0, Math.min(t, F)));
  const bwdDone = new Set<NodeKey>(t > F ? g.backwardOrder.slice(0, Math.min(t - F, B)) : []);
  const phase = t === 0 ? 'start' : t <= F ? 'forward' : t <= F + B ? 'backward' : 'update';
  const current: NodeKey | null = phase === 'forward' ? g.forwardOrder[t - 1] : phase === 'backward' ? g.backwardOrder[t - F - 1] : null;
  const nextIsQuiz = t >= F && t < F + B && g.backwardOrder[t - F] === QUIZ_NODE;

  const reset = (np?: Params) => {
    if (np) setP(np);
    setT(0);
    setGuess('');
    setQuizState('none');
  };

  const next = () => {
    if (nextIsQuiz && quizState === 'none') {
      setQuizState('asking');
      return;
    }
    setT(Math.min(total, t + 1));
  };

  const valShown = (k: NodeKey) => g.nodes[k].leaf || fwdDone.has(k);
  const gradShown = (k: NodeKey) => bwdDone.has(k);

  const edgeEls = g.edges.map((e, i) => {
    const a = g.nodes[e.from];
    const b = g.nodes[e.to];
    const x1 = px(a.col) + BW;
    const y1 = py(a.row) + BH / 2;
    const x2 = px(b.col);
    const y2 = py(b.row) + BH / 2;
    const dup = g.edges.filter((o) => o.from === e.from && o.to === e.to);
    const k = dup.indexOf(e);
    const bend = dup.length > 1 ? (k === 0 ? -18 : 18) : 0;
    const active = phase === 'backward' && current === e.from;
    const fwdActive = phase === 'forward' && current === e.to;
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2 + bend;
    return (
      <g key={i}>
        <path
          d={`M${x1},${y1} Q${mx},${my} ${x2},${y2}`}
          fill="none"
          stroke={active ? 'var(--c-grad)' : fwdActive ? 'var(--accent)' : 'var(--border-strong)'}
          strokeWidth={active || fwdActive ? 3 : 1.5}
        />
        {active && (
          <text x={mx} y={my - 5} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--c-grad)" style={{ fontFamily: 'var(--font-mono)', paintOrder: 'stroke', stroke: 'var(--bg-elev)', strokeWidth: 4 }}>
            ×{fmt(e.local, 2)}
          </text>
        )}
      </g>
    );
  });

  const nodeEls = Object.values(g.nodes).map((n) => {
    const isCur = current === n.key;
    const showV = valShown(n.key);
    const showG = gradShown(n.key);
    const stroke = isCur ? (phase === 'backward' ? 'var(--c-grad)' : 'var(--accent)') : n.param ? 'var(--c-a)' : 'var(--border-strong)';
    return (
      <g key={n.key}>
        <rect x={px(n.col)} y={py(n.row)} width={BW} height={BH} rx={9} fill="var(--bg)" stroke={stroke} strokeWidth={isCur ? 3 : n.param ? 2 : 1.3} />
        <text x={px(n.col) + 8} y={py(n.row) + 15} fontSize={11} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-ui)' }}>
          {n.v.label} · {n.name}
        </text>
        <text x={px(n.col) + 8} y={py(n.row) + 32} fontSize={13} fontWeight={700} fill="var(--text)" style={{ fontFamily: 'var(--font-mono)' }}>
          {showV ? fmt(n.v.data, 3) : '?'}
        </text>
        {n.op && (
          <text x={px(n.col) + BW - 8} y={py(n.row) + 32} fontSize={10.5} textAnchor="end" fill="var(--text-faint)" style={{ fontFamily: 'var(--font-mono)' }}>
            {n.op}
          </text>
        )}
        {n.key !== 'y' && (
          <text x={px(n.col) + 8} y={py(n.row) + 47} fontSize={11.5} fontWeight={700} fill="var(--c-grad)" style={{ fontFamily: 'var(--font-mono)' }}>
            {showG ? `grad ${fmt(n.v.grad, 3)}` : phase === 'backward' || phase === 'update' ? 'grad ?' : ''}
          </text>
        )}
      </g>
    );
  });

  // Explanation panel for the current step
  let panel: React.ReactNode = null;
  if (phase === 'start')
    panel = (
      <div className="live-note">
        <strong>Ready.</strong> The inputs, the parameters (blue outlines: the numbers we’re allowed to learn) and the target are known. Everything else is a question mark.
        Press <strong>Next</strong> to run the <strong>forward pass</strong>: compute each value from the ones before it, and <em>remember</em> it.
      </div>
    );
  else if (phase === 'forward' && current) {
    const n = g.nodes[current];
    const ins = g.edges.filter((e) => e.to === current).map((e) => g.nodes[e.from]);
    const sub = n.v.op === 'tanh' ? `\\tanh(${fmt(ins[0].v.data, 3)})` : ins.map((i) => pn(i.v.data, 3)).join(` ${n.v.op === '×' ? '\\times' : n.v.op} `);
    panel = (
      <div>
        <div className="field-label">Forward pass · step {t} of {F}</div>
        <Tex block>{`\\text{${n.name}}\\;\\; ${n.v.label} = ${n.op!.replace('×', '\\times').replace('−', '-')} = ${sub} = ${col('c-a', fmt(n.v.data, 4))}`}</Tex>
        {current === 'L' && (
          <div className="live-note">
            Forward pass complete: the loss is <strong>{fmt(n.v.data, 4)}</strong>. Now the real question: <em>how should each blue parameter change to make this smaller?</em> Press
            Next to start the <strong>backward pass</strong>.
          </div>
        )}
      </div>
    );
  } else if (phase === 'backward' && current) {
    const n = g.nodes[current];
    const outs = g.edges.filter((e) => e.from === current);
    panel =
      current === 'L' ? (
        <div>
          <div className="field-label">Backward pass · start at the loss</div>
          <Tex block>{`\\frac{\\partial L}{\\partial L} = ${col('c-grad', '1')}`}</Tex>
          <div className="live-note">
            The loss responds to itself one-for-one. That <strong>1</strong> is the seed: from here, every gradient will be built by multiplying backwards.
          </div>
        </div>
      ) : (
        <div>
          <div className="field-label">Backward pass · step {t - F} of {B}</div>
          <Tex block>{`\\underbrace{\\frac{\\partial L}{\\partial ${n.v.label}}}_{\\text{how the loss responds to ${n.name}}} = ${outs
            .map((e) => `\\underbrace{${col('c-grad', fmt(g.nodes[e.to].v.grad, 3))}}_{\\partial L/\\partial ${g.nodes[e.to].v.label}} \\times \\underbrace{${pn(e.local, 3)}}_{\\text{local}}`)
            .join(' + ')} = ${col('c-grad', fmt(n.v.grad, 4))}`}</Tex>
          <div className="live-note">
            {outs.length > 1 ? (
              <>
                <strong>{n.v.label} feeds into the loss along {outs.length} paths</strong> (L = d × d uses it twice), so its two contributions are <strong>added</strong>. This is
                gradient accumulation — the “add across paths” rule from the chain-rule lesson.
              </>
            ) : (
              <>
                Take the gradient already known one step downstream ({g.nodes[outs[0].to].v.label}), and multiply by the local ratio of that step: {outs[0].why}.
                {n.param && (
                  <>
                    {' '}
                    <strong>This is a parameter</strong>, so this number is what gradient descent needs:{' '}
                    {n.v.grad > 0 ? 'positive, so decreasing it will lower the loss.' : n.v.grad < 0 ? 'negative, so increasing it will lower the loss.' : 'zero, so it doesn’t matter right now.'}
                  </>
                )}
                {(current === 'x1' || current === 'x2') && <> Inputs get gradients too — we just can’t change them. (Later, this is how attacks and explanations probe what a model is sensitive to.)</>}
              </>
            )}
          </div>
        </div>
      );
  } else if (phase === 'update') {
    const u = lossAfterUpdate(p, lr);
    panel = (
      <div>
        <div className="field-label">Update · one step of gradient descent</div>
        <Tex block>{`${col('c-a', 'w_1')}: ${fmt(p.w1, 3)} - ${fmt(lr)}\\times${pn(g.nodes.w1.v.grad, 3)} = ${fmt(u.next.w1, 3)} \\quad ${col('c-a', 'w_2')}: ${fmt(p.w2, 3)} \\to ${fmt(u.next.w2, 3)} \\quad ${col('c-a', 'b')}: ${fmt(p.b, 3)} \\to ${fmt(u.next.b, 3)}`}</Tex>
        <div className="live-note">
          Loss before: <strong>{fmt(u.before, 4)}</strong> → after: <strong>{fmt(u.after, 4)}</strong>.{' '}
          {u.after < u.before ? 'One forward pass, one backward pass, one small step — and the network is a little less wrong. Training is this loop, repeated.' : 'The step was too big and overshot. Try a smaller learning rate.'}
          <div className="btn-row" style={{ marginTop: 8 }}>
            <button className="btn small primary" onClick={() => reset(u.next)}>Use the new weights and go again</button>
          </div>
        </div>
      </div>
    );
  }

  const correctW1 = g.nodes.w1.v.grad;
  const checkGuess = () => {
    const v = Number(guess.replace('−', '-'));
    if (guess.trim() === '' || Number.isNaN(v)) return;
    if (Math.abs(v - correctW1) < 0.01) {
      setQuizState('right');
      setT(t + 1);
    } else setQuizState('wrong');
  };
  const m1grad = g.nodes.m1.v.grad;

  return (
    <Widget
      title="Backpropagation, one step at a time"
      wide
      caption={
        <>
          A single neuron with a loss, drawn as a computational graph. Each box holds a <strong>value</strong> (computed in the forward pass) and a{' '}
          <span className="c-grad">gradient</span> (computed in the backward pass): how much the loss would change per unit nudge of that box. Blue-outlined boxes are the
          learnable parameters. Step forwards and backwards with the buttons.
        </>
      }
    >
      <div>
        <svg viewBox="0 0 1130 350" width={1130} height={350} style={{ width: '100%', height: 'auto' }} role="img" aria-label="Computational graph of a neuron with loss">
          {edgeEls}
          {nodeEls}
        </svg>
      </div>
      <div className="btn-row" style={{ margin: '6px 0 12px' }}>
        <button className="btn small" onClick={() => setT(Math.max(0, t - 1))} disabled={t === 0}>← Back</button>
        <button className="btn small primary" onClick={next} disabled={t >= total || quizState === 'asking' || quizState === 'wrong'}>
          {t === 0 ? 'Start forward pass →' : t === F ? 'Start backward pass →' : t === F + B ? 'Update the weights →' : 'Next →'}
        </button>
        <button className="btn small" onClick={() => { setQuizState('revealed'); setT(total); }}>Skip to end</button>
        <button className="btn small" onClick={() => reset()}>Reset</button>
        <span className="muted" style={{ fontSize: 13 }}>
          {phase === 'start' ? 'not started' : phase === 'forward' ? `forward ${t}/${F}` : phase === 'backward' ? `backward ${t - F}/${B}` : 'update'}
        </span>
      </div>
      {quizState === 'asking' || quizState === 'wrong' ? (
        <div className="predict" style={{ margin: 0 }}>
          <div className="predict-label">◆ Your turn</div>
          <div className="predict-q" style={{ fontWeight: 500 }}>
            The next box is <strong>w₁</strong>. It feeds only into m₁ = x₁ × w₁, and you already know ∂L/∂m₁ = <span className="mono c-grad">{fmt(m1grad, 4)}</span>. What is
            ∂L/∂w₁? (Hint: the local ratio of a multiplication is <em>the other input</em>, here x₁ = {fmt(p.x1, 2)}.)
          </div>
          <form className="numeric-input" onSubmit={(e) => { e.preventDefault(); checkGuess(); }}>
            <input value={guess} onChange={(e) => setGuess(e.target.value)} placeholder="∂L/∂w₁" inputMode="decimal" aria-label="your gradient for w1" />
            <button className="btn primary small" type="submit">Check</button>
            <button className="btn small" type="button" onClick={() => { setQuizState('revealed'); setT(t + 1); }}>Just show me</button>
          </form>
          {quizState === 'wrong' && (
            <div className="feedback bad" style={{ marginTop: 8 }}>
              Not quite. Multiply the gradient arriving from m₁ by the local ratio x₁: {fmt(m1grad, 4)} × {fmt(p.x1, 2)}.
            </div>
          )}
        </div>
      ) : (
        <>
          {quizState === 'right' && current === 'w1' && (
            <div className="feedback good" style={{ marginBottom: 8 }}>
              <div className="feedback-title">✓ You computed it yourself: {fmt(m1grad, 4)} × {fmt(p.x1, 2)} = {fmt(correctW1, 4)}.</div>
            </div>
          )}
          {panel}
        </>
      )}
      <details style={{ marginTop: 12 }}>
        <summary className="muted" style={{ cursor: 'pointer', fontSize: 13.5 }}>Change the numbers</summary>
        <div className="controls inline" style={{ marginTop: 10 }}>
          <Slider label="w₁" value={p.w1} min={-2} max={2} step={0.1} onChange={(v) => reset({ ...p, w1: v })} color="c-a" />
          <Slider label="w₂" value={p.w2} min={-2} max={2} step={0.1} onChange={(v) => reset({ ...p, w2: v })} color="c-a" />
          <Slider label="b" value={p.b} min={-2} max={2} step={0.1} onChange={(v) => reset({ ...p, b: v })} color="c-a" />
          <Slider label="target y" value={p.y} min={-1} max={1} step={0.1} onChange={(v) => reset({ ...p, y: v })} />
          <Slider label="learning rate" value={lr} min={0.05} max={3} step={0.05} onChange={setLr} color="c-grad" />
        </div>
      </details>
    </Widget>
  );
}
