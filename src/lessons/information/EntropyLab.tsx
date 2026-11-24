import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Slider } from '../../components/controls';
import { Tex, fmt } from '../../components/Tex';
import { entropy, surprise } from '../../lib/prob';

const NAMES = ['sunny', 'cloudy', 'rain', 'snow'];
const COLORS = ['var(--c-b)', 'var(--text-muted)', 'var(--c-a)', 'var(--c-grad)'];
const TARGETS = [
  { label: '0 bits', h: 0, hint: 'A forecast with no uncertainty at all.' },
  { label: '1 bit', h: 1, hint: 'As uncertain as a single fair coin flip.' },
  { label: '2 bits', h: 2, hint: 'The most uncertain a 4-outcome forecast can be.' },
  { label: '1.5 bits', h: 1.5, hint: 'Somewhere in between — there are many solutions.' },
];

export function EntropyLab() {
  const [w, setW] = useState([6, 2, 1.5, 0.5]);
  const [target, setTarget] = useState<number | null>(null);
  const total = w.reduce((a, b) => a + b, 0);
  const p = w.map((v) => (total > 0 ? v / total : 0.25));
  const H = entropy(p);
  const hit = target !== null && Math.abs(H - TARGETS[target].h) < 0.03;

  return (
    <Widget
      title="Surprise and entropy of a weather forecast"
      wide
      caption={
        <>
          Set how likely each kind of weather is (the sliders are relative weights; they’re normalised to probabilities). Each bar shows a probability, and the
          number above it is how <strong>surprised</strong> you’d be, in bits, if that weather actually happened. <strong>Entropy</strong> is the average surprise,
          weighting each outcome by how often it happens. Try the targets.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col controls" style={{ flex: '1 1 240px' }}>
          {NAMES.map((n, i) => (
            <Slider key={n} label={n} value={w[i]} min={0} max={10} step={0.1} onChange={(v) => { const c = w.slice(); c[i] = v; setW(c); }} format={() => `${(p[i] * 100).toFixed(1)}%`} />
          ))}
          <div>
            <div className="field-label">Hit a target entropy</div>
            <div className="btn-row">
              {TARGETS.map((t, k) => (
                <button key={t.label} className={`btn small${target === k ? ' primary' : ''}`} onClick={() => setTarget(target === k ? null : k)}>
                  {t.label}
                </button>
              ))}
            </div>
            {target !== null && (
              <div className={`status ${hit ? 'good' : 'neutral'}`} style={{ marginTop: 8 }}>
                {hit ? `✓ Entropy is ${H.toFixed(2)} bits. ${TARGETS[target].hint}` : `Target: ${TARGETS[target].h} bits — now ${H.toFixed(2)}`}
              </div>
            )}
          </div>
        </div>
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <svg viewBox="0 0 400 210" width={400} height={210} role="img" aria-label="Probabilities and surprise for each outcome">
            <line x1={0} x2={400} y1={175} y2={175} stroke="var(--c-axis)" />
            {p.map((v, i) => {
              const x = 18 + i * 96;
              const h = v * 140;
              const s = surprise(v);
              return (
                <g key={i}>
                  <rect x={x} y={175 - h} width={62} height={h} rx={3} fill={COLORS[i]} opacity={0.8} />
                  <text x={x + 31} y={190} textAnchor="middle" fontSize={12} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-ui)' }}>{NAMES[i]}</text>
                  <text x={x + 31} y={168 - h} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--text)" style={{ fontFamily: 'var(--font-mono)' }}>
                    {v > 0 ? `${s.toFixed(2)} bits` : '∞'}
                  </text>
                </g>
              );
            })}
          </svg>
          <div style={{ fontSize: 13.5, overflowX: 'auto' }}>
            <Tex block>{`H = \\sum_i p_i \\cdot \\underbrace{\\log_2\\tfrac{1}{p_i}}_{\\text{surprise}} = ${p
              .filter((v) => v > 0)
              .map((v) => `${fmt(v, 2)}\\cdot${fmt(surprise(v), 2)}`)
              .join(' + ')} = \\mathbf{${fmt(H, 2)}}\\text{ bits}`}</Tex>
          </div>
          <div className="live-note">
            {H < 0.2 ? (
              <>Almost no uncertainty: you can predict the weather with near certainty, so on average you learn almost nothing when you find out.</>
            ) : H > 1.95 ? (
              <>Maximum uncertainty for four outcomes: all equally likely. Finding out the weather tells you exactly 2 bits — the same as two fair coin flips.</>
            ) : (
              <>Rare outcomes are very surprising, but they’re rare, so they contribute little to the average. Common outcomes are unsurprising. Entropy balances the two.</>
            )}
          </div>
        </div>
      </div>
    </Widget>
  );
}
