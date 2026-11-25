// authored-by: Claude Opus 5.5
import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Slider } from '../../components/controls';
import { crossEntropy, entropy, klDivergence } from '../../lib/prob';
import { KL_NAMES, klNote } from './kl';

const NAMES = KL_NAMES;
const normalise = (w: number[]) => {
  const t = w.reduce((a, b) => a + b, 0);
  return t > 0 ? w.map((x) => x / t) : w.map(() => 1 / w.length);
};
const bits = (x: number) => (Number.isFinite(x) ? `${x.toFixed(3)} bits` : '∞');

export function KLExplorer() {
  const [wp, setWp] = useState([7, 2, 1]);
  const [wq, setWq] = useState([1, 1, 1]);
  const p = normalise(wp);
  const q = normalise(wq);
  const Hp = entropy(p, 2);
  const dpq = klDivergence(p, q);
  const Hpq = Number.isFinite(dpq) ? crossEntropy(p, q, 2) : Infinity;
  const dqp = klDivergence(q, p);
  const W = 360, H = 170, base = 140, barW = 38;

  const sliders = (w: number[], set: (w: number[]) => void, probs: number[], who: string) =>
    NAMES.map((n, i) => (
      <Slider
        key={who + n}
        label={`${who}: ${n}`}
        value={w[i]}
        min={0}
        max={10}
        step={0.1}
        onChange={(v) => {
          const c = w.slice();
          c[i] = v;
          set(c);
        }}
        format={() => `${(probs[i] * 100).toFixed(1)}%`}
      />
    ));

  return (
    <Widget
      title="KL divergence: the price of believing the wrong thing"
      wide
      caption={
        <>
          In each pair, the left bar is the truth <strong>p</strong> (how often that weather really happens) and the right bar is a model’s belief <strong>q</strong>. The sliders are relative weights, normalised to probabilities. Make q equal to p and watch both KL readouts go to zero. Then make p confident and q flat, and swap them.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col controls" style={{ flex: '1 1 240px' }}>
          {sliders(wp, setWp, p, 'truth')}
          {sliders(wq, setWq, q, 'belief')}
          <div className="btn-row">
            <button type="button" className="btn small" onClick={() => setWq(wp.slice())}>
              Set q = p
            </button>
            <button
              type="button"
              className="btn small"
              onClick={() => {
                setWp(wq);
                setWq(wp);
              }}
            >
              Swap p and q
            </button>
          </div>
        </div>
        <div className="widget-col" style={{ flex: '1 1 340px' }}>
          <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Truth and belief as side-by-side bars">
            <line x1={0} x2={W} y1={base} y2={base} stroke="var(--c-axis)" />
            {NAMES.map((n, i) => {
              const x = 30 + i * 112;
              return (
                <g key={n}>
                  <rect x={x} y={base - p[i] * 120} width={barW} height={p[i] * 120} rx={3} fill="var(--c-a)" />
                  <rect x={x + barW + 4} y={base - q[i] * 120} width={barW} height={q[i] * 120} rx={3} fill="var(--c-b)" />
                  <text x={x + barW + 2} y={base + 16} textAnchor="middle" fontSize={12} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-ui)' }}>
                    {n}
                  </text>
                </g>
              );
            })}
          </svg>
          <div className="readout">
            <span className="k">entropy of the truth H(p) </span>
            {bits(Hp)}
            {'\n'}
            <span className="k">cross-entropy H(p, q)     </span>
            {bits(Hpq)}
            {'\n'}
            <span className="k">extra surprise D(p‖q)     </span>
            <strong>{bits(dpq)}</strong>
            {'\n'}
            <span className="k">the other way D(q‖p)      </span>
            <strong>{bits(dqp)}</strong>
          </div>
          <div className="live-note">{klNote(p, q)}</div>
        </div>
      </div>
    </Widget>
  );
}
