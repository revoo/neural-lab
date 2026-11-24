import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider, Toggle } from '../../components/controls';
import { Dot, FnPath, Plot } from '../../components/plot';

const TARGETS: Record<string, { label: string; f: (x: number) => number }> = {
  wave: { label: 'Wave', f: (x) => Math.sin(1.6 * x) },
  bump: { label: 'Bump', f: (x) => 1.6 * Math.exp(-x * x) - 0.6 },
  wiggle: { label: 'Wiggle', f: (x) => 0.5 * Math.sin(3 * x) + 0.25 * x },
};
type Key = keyof typeof TARGETS;

/**
 * Piecewise-linear interpolation of f at n+1 evenly spaced knots, written as
 * a straight line plus a sum of ReLU hinges — i.e. exactly what one hidden layer of ReLU neurons computes.
 */
export function hingeFit(f: (x: number) => number, n: number, lo = -3, hi = 3) {
  const knots = Array.from({ length: n + 1 }, (_, i) => lo + ((hi - lo) * i) / n);
  const vals = knots.map(f);
  const slopes = knots.slice(0, -1).map((k, i) => (vals[i + 1] - vals[i]) / (knots[i + 1] - k));
  // g(x) = vals[0] + s0 (x - k0) + Σ_{i≥1} (s_i - s_{i-1}) ReLU(x - k_i)
  const hinges = slopes.slice(1).map((s, i) => ({ at: knots[i + 1], amount: s - slopes[i] }));
  const g = (x: number) => vals[0] + slopes[0] * (x - knots[0]) + hinges.reduce((t, h) => t + h.amount * Math.max(0, x - h.at), 0);
  return { knots, vals, slopes, hinges, g };
}

export function HingeBuilder() {
  const [key, setKey] = useState<Key>('wave');
  const [n, setN] = useState(3);
  const [parts, setParts] = useState(true);
  const f = TARGETS[key].f;
  const fit = useMemo(() => hingeFit(f, n), [f, n]);
  let err = 0;
  for (let x = -3; x <= 3; x += 0.01) err = Math.max(err, Math.abs(fit.g(x) - f(x)));
  return (
    <Widget
      title="Building a curve out of hinges"
      wide
      caption={
        <>
          The faint grey curve is a target we’d like to approximate. The thick <span style={{ color: 'var(--accent)' }}>purple</span> line is a sum of ReLU hinges:
          one straight line plus {n - 1} “bends”, each of the form <em>a · ReLU(x − k)</em>. Each hinge is flat until its kink at k, then slopes off. Toggle the pieces to
          see them individually. Add more hinges and watch the fit improve.
        </>
      }
    >
      <div className="controls inline" style={{ marginBottom: 10 }}>
        <Segmented label="Target curve" value={key} onChange={setKey} options={(Object.keys(TARGETS) as Key[]).map((k) => ({ value: k, label: TARGETS[k].label }))} />
        <div style={{ minWidth: 220 }}>
          <Slider label="number of ReLU neurons (hinges)" value={n - 1} min={0} max={20} step={1} onChange={(v) => setN(v + 1)} format={(v) => String(v)} />
        </div>
        <Toggle label="Show individual hinges" checked={parts} onChange={setParts} />
      </div>
      <Plot x={[-3, 3]} y={[-2.2, 2.2]} width={720} height={320} ariaLabel="Target curve approximated by ReLU hinges">
        {parts &&
          fit.hinges.map((h, i) => (
            <FnPath key={i} f={(x) => h.amount * Math.max(0, x - h.at)} color={h.amount >= 0 ? 'var(--c-a)' : 'var(--c-b)'} width={1.3} />
          ))}
        <FnPath f={f} color="var(--text-faint)" width={6} />
        <FnPath f={fit.g} color="var(--accent)" width={3} samples={600} />
        {fit.knots.map((k, i) => (
          <Dot key={i} at={[k, fit.vals[i]]} r={3.5} color="var(--accent)" />
        ))}
      </Plot>
      <div className="live-note" style={{ marginTop: 8 }}>
        {n === 1 ? (
          <>
            <strong>Zero hinges: just a straight line.</strong> This is all a network without activation functions can ever produce. However many linear layers you stack,
            you get a line (or a flat plane, in more dimensions).
          </>
        ) : (
          <>
            <strong>{n - 1} hinge{n > 2 ? 's' : ''}</strong>, worst-case error {err.toFixed(3)}. Each <span className="c-a">blue</span> hinge bends the line upward, each{' '}
            <span className="c-b">orange</span> one bends it down. {n > 12 ? 'With enough hinges the approximation becomes almost indistinguishable from the target — that’s the intuition behind the “universal approximation” results for neural networks.' : 'Keep adding hinges.'}
          </>
        )}
      </div>
    </Widget>
  );
}
