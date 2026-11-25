import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Slider } from '../../components/controls';
import { Tex, col, fmt, pn } from '../../components/Tex';
import { useFramePlayer } from '../../components/motion';

/** Running products of the local derivatives: how big a nudge of 1 at w is on arrival at w, z, a and L. */
export function nudgeTrail(w: number): number[] {
  const s = forward(w);
  return [1, s.dz, s.dz * s.da, s.dz * s.da * s.dL];
}

// A real neuron with a loss: w → z = w·x + b → a = tanh(z) → L = (a − t)²
const X = 1.5;
const B = 0.2;
const T = 0.8;

export function forward(w: number) {
  const z = w * X + B;
  const a = Math.tanh(z);
  const L = (a - T) ** 2;
  return { z, a, L, dz: X, da: 1 - a * a, dL: 2 * (a - T) };
}

function Stage({ name, value, color, sub, pulse }: { name: string; value: number; color: string; sub: string; pulse?: number | null }) {
  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 10, padding: '8px 10px', minWidth: 92, textAlign: 'center', background: 'var(--bg)' }}>
      <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{sub}</div>
      <div className="mono" style={{ fontWeight: 700, fontSize: 16, color }}>
        {name} = {value.toFixed(3)}
      </div>
      <div style={{ height: 18, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }} aria-hidden={pulse == null}>
        {pulse != null && (
          <>
            <span
              className="nudge-pulse"
              style={{ width: Math.max(3, Math.min(60, 30 * Math.sqrt(Math.abs(pulse)))), height: 8, borderRadius: 4, background: 'var(--c-grad)', display: 'inline-block' }}
            />
            <span className="mono" style={{ fontSize: 11, color: 'var(--c-grad)' }}>
              {pulse.toFixed(3)}ε
            </span>
          </>
        )}
      </div>
    </div>
  );
}

function Link({ ratio, label }: { ratio: number; label: string }) {
  const weak = Math.abs(ratio) < 0.1;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 74 }}>
      <div className="mono" style={{ fontSize: 13, fontWeight: 700, color: weak ? 'var(--c-bad)' : 'var(--c-grad)' }}>×{ratio.toFixed(3)}</div>
      <div style={{ fontSize: 22, lineHeight: 1, color: 'var(--text-faint)' }}>→</div>
      <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>{label}</div>
    </div>
  );
}

export function ChainExplorer() {
  const [w, setW] = useState(0.4);
  const [eps, setEps] = useState(0.001);
  const s = forward(w);
  const s2 = forward(w + eps);
  const chain = s.dL * s.da * s.dz;
  const measured = (s2.L - s.L) / eps;
  const saturated = Math.abs(s.da) < 0.05;
  const trail = nudgeTrail(w);
  const [at, setAt] = useState<number | null>(null);
  const player = useFramePlayer(3, 800, (i) => setAt(i));
  const sendNudge = () => {
    setAt(0);
    player.start();
  };
  const factors = [s.dz, s.da, s.dL].slice(0, at ?? 0).map((f) => f.toFixed(3));
  const pulseNote =
    at === null ? null : at === 0 ? (
      <>A nudge ε arrives at w. Watch it travel: each stage multiplies it by its local derivative.</>
    ) : (
      <>
        <strong>After stage {at}:</strong> nudge × {factors.join(' × ')} = {trail[at].toFixed(3)} × ε
        {at === 3 ? <>. That final size is the gradient ∂L/∂w: the chain rule is just this journey, written down.</> : '.'}
      </>
    );

  return (
    <Widget
      title="Sensitivities multiply along a chain"
      wide
      caption={
        <>
          This is a genuine artificial neuron followed by a loss. A weight <span className="c-a">w</span> multiplies a fixed input x = {X}, a bias b = {B} is
          added, the result is squashed by tanh, and the loss measures the squared distance from a target t = {T}. Each arrow shows a{' '}
          <span className="c-grad">local derivative</span>: how much that stage amplifies a small change passing through it. Move <span className="c-a">w</span>{' '}
          and compare the chain-rule product with a real measured nudge. Then drag w all the way to −3.
        </>
      }
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', justifyContent: 'center', marginBottom: 14 }}>
        <Stage name="w" value={w} color="var(--c-a)" sub="weight" pulse={at !== null && at >= 0 ? trail[0] : null} />
        <Link ratio={s.dz} label="∂z/∂w = x" />
        <Stage name="z" value={s.z} color="var(--text-muted)" sub="z = w·x + b" pulse={at !== null && at >= 1 ? trail[1] : null} />
        <Link ratio={s.da} label="∂a/∂z = 1 − a²" />
        <Stage name="a" value={s.a} color="var(--text-muted)" sub="a = tanh(z)" pulse={at !== null && at >= 2 ? trail[2] : null} />
        <Link ratio={s.dL} label="∂L/∂a = 2(a − t)" />
        <Stage name="L" value={s.L} color="var(--c-c)" sub="L = (a − t)²" pulse={at !== null && at >= 3 ? trail[3] : null} />
      </div>
      <div className="widget-row">
        <div className="widget-col controls">
          <Slider label="weight w" value={w} min={-3} max={3} step={0.01} onChange={(v) => { setAt(null); setW(v); }} color="c-a" />
          <button type="button" className="btn primary" onClick={sendNudge} disabled={player.running}>
            {player.running ? 'Sending…' : '▶ Play: send a nudge'}
          </button>
          <Slider label="test nudge ε" value={eps} min={0.001} max={0.5} step={0.001} onChange={setEps} format={(v) => v.toFixed(3)} />
          <div className="live-note">
            {pulseNote ? pulseNote : saturated ? (
              <>
                <strong>The tanh has saturated.</strong> With z = {s.z.toFixed(2)}, tanh is almost perfectly flat, so its local derivative is only {s.da.toFixed(4)}.
                Multiplying by that nearly-zero link crushes the whole chain: the loss is {s.L.toFixed(3)}, far from zero, yet the gradient says “changing w hardly
                matters”. This is the seed of the <strong>vanishing gradient</strong> problem.
              </>
            ) : Math.abs(chain) < 0.01 ? (
              <>
                <strong>The gradient is near zero</strong> because the output is close to the target (∂L/∂a ≈ 0). The loss is almost minimised, so nudging w changes
                little. This time a small gradient is good news.
              </>
            ) : (
              <>
                Increasing w by a small amount ε should change the loss by about <strong>{chain.toFixed(3)} × ε</strong>. The {chain < 0 ? 'negative' : 'positive'} sign
                says increasing w would {chain < 0 ? 'decrease' : 'increase'} the loss — so to learn, w should move {chain < 0 ? 'up' : 'down'}.
              </>
            )}
          </div>
        </div>
        <div className="widget-col">
          <div style={{ fontSize: 15 }}>
            <Tex block>{`\\frac{\\partial L}{\\partial w} = ${col('c-grad', '\\frac{\\partial L}{\\partial a}')}\\cdot${col('c-grad', '\\frac{\\partial a}{\\partial z}')}\\cdot${col('c-grad', '\\frac{\\partial z}{\\partial w}')}`}</Tex>
            <Tex block>{`= ${pn(s.dL, 3)} \\times ${pn(s.da, 3)} \\times ${pn(s.dz, 3)} = ${col('c-grad', fmt(chain, 4))}`}</Tex>
          </div>
          <div className="readout" style={{ marginTop: 8 }}>
            <span className="k">measured:</span> (L(w+ε) − L(w)) / ε
            <br />= ({s2.L.toFixed(5)} − {s.L.toFixed(5)}) / {eps.toFixed(3)} = <strong>{measured.toFixed(4)}</strong>
            <br />
            <span className="faint">
              {Math.abs(measured - chain) < 0.01 * Math.max(1, Math.abs(chain))
                ? '✓ Matches the chain-rule product.'
                : 'Not quite equal — the nudge is big enough that the curves bend. Shrink ε and they converge.'}
            </span>
          </div>
        </div>
      </div>
    </Widget>
  );
}
