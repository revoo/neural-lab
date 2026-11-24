import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Tex, col, fmt } from '../../components/Tex';
import { softmax, entropy } from '../../lib/prob';
import { createRng } from '../../lib/rng';

const CLASSES = ['cat', 'dog', 'fox', 'owl'];
const COLORS = ['var(--c-a)', 'var(--c-b)', 'var(--c-c)', 'var(--c-grad)'];

type Method = 'sum' | 'shift' | 'softmax';

/** Three ways to turn scores into "probabilities" — only one works for every input. */
export function normalise(z: number[], method: Method, temperature = 1): { p: number[] | null; problem?: string } {
  if (method === 'softmax') return { p: softmax(z, temperature) };
  if (method === 'sum') {
    const s = z.reduce((a, b) => a + b, 0);
    if (Math.abs(s) < 1e-9) return { p: null, problem: 'The scores sum to zero, so dividing by the total means dividing by zero.' };
    const p = z.map((v) => v / s);
    if (p.some((v) => v < 0)) return { p, problem: 'Some “probabilities” are negative. A −30% chance of “owl” is meaningless.' };
    return { p };
  }
  const m = Math.min(...z);
  const shifted = z.map((v) => v - m);
  const s = shifted.reduce((a, b) => a + b, 0);
  if (s < 1e-9) return { p: null, problem: 'All scores are equal, so after shifting everything is zero: 0 ÷ 0.' };
  const p = shifted.map((v) => v / s);
  return { p, problem: p.some((v) => v === 0) ? 'No negatives now — but the lowest class is always forced to exactly 0%, no matter how close its score was to the others. The model can never say “probably cat, but maybe owl”.' : undefined };
}

export function SoftmaxLab() {
  const [z, setZ] = useState([2.0, 1.0, 0.1, -1.0]);
  const [method, setMethod] = useState<Method>('sum');
  const [T, setT] = useState(1);
  const [samples, setSamples] = useState<number[] | null>(null);
  const [seed, setSeed] = useState(1);

  const { p, problem } = normalise(z, method, T);
  const exps = z.map((v) => Math.exp(v / T));
  const sumExp = exps.reduce((a, b) => a + b, 0);

  const counts = useMemo(() => {
    if (!samples) return null;
    const c = [0, 0, 0, 0];
    samples.forEach((k) => c[k]++);
    return c;
  }, [samples]);

  const draw = () => {
    if (!p || p.some((v) => v < 0)) return;
    const rng = createRng(seed);
    setSeed(seed + 1);
    const out: number[] = [];
    for (let s = 0; s < 200; s++) {
      let u = rng.next();
      let k = 0;
      while (k < 3 && u >= p[k]) {
        u -= p[k];
        k++;
      }
      out.push(k);
    }
    setSamples(out);
  };

  const maxBar = 1;
  return (
    <Widget
      title="From scores to probabilities"
      wide
      caption={
        method === 'softmax' ? (
          <>
            <strong>Softmax</strong> exponentiates every score (making it positive) and divides by the total (making them sum to 1). Try: add the same amount to
            every score with the <strong>shift</strong> buttons — nothing changes, because only differences matter. Then slide the temperature to both extremes.
          </>
        ) : (
          <>
            A model produces four raw scores for an image. Move the sliders and try to turn them into probabilities using the selected method. Look for inputs
            where it breaks — then switch methods.
          </>
        )
      }
    >
      <div className="controls inline" style={{ marginBottom: 12 }}>
        <Segmented
          label="Normalisation method"
          value={method}
          onChange={(m) => { setMethod(m); setSamples(null); }}
          options={[
            { value: 'sum', label: 'Divide by the sum' },
            { value: 'shift', label: 'Shift up, then divide' },
            { value: 'softmax', label: 'Softmax' },
          ]}
        />
      </div>
      <div className="widget-row">
        <div className="widget-col controls" style={{ flex: '1 1 260px' }}>
          <div className="field-label">Raw scores (logits)</div>
          {CLASSES.map((c, i) => (
            <Slider
              key={c}
              label={c}
              value={z[i]}
              min={-4}
              max={4}
              step={0.1}
              onChange={(v) => { const n = z.slice(); n[i] = v; setZ(n); setSamples(null); }}
              format={(v) => v.toFixed(1)}
              color={['c-a', 'c-b', 'c-c', 'c-grad'][i]}
            />
          ))}
          {method === 'softmax' && (
            <>
              <div className="btn-row">
                <button className="btn small" onClick={() => setZ(z.map((v) => Math.min(4, v + 1)))}>shift all +1</button>
                <button className="btn small" onClick={() => setZ(z.map((v) => Math.max(-4, v - 1)))}>shift all −1</button>
              </div>
              <Slider label="temperature T" value={T} min={0.05} max={5} step={0.05} onChange={(v) => { setT(v); setSamples(null); }} format={(v) => v.toFixed(2)} />
            </>
          )}
        </div>
        <div className="widget-col" style={{ flex: '1 1 380px' }}>
          <div className="field-label">Resulting distribution</div>
          <svg viewBox="0 0 380 190" width={380} height={190} role="img" aria-label="Probability bars">
            <line x1={0} x2={380} y1={160} y2={160} stroke="var(--c-axis)" />
            {CLASSES.map((c, i) => {
              const v = p ? p[i] : 0;
              const hgt = Math.max(-40, Math.min(150, (v / maxBar) * 150));
              const x = 20 + i * 92;
              const bad = v < 0 || !p;
              return (
                <g key={c}>
                  <rect x={x} width={56} y={hgt >= 0 ? 160 - hgt : 160} height={Math.abs(hgt)} fill={bad ? 'var(--c-bad)' : COLORS[i]} opacity={0.85} rx={3} />
                  {counts && <rect x={x + 14} width={28} y={160 - (counts[i] / 200) * 150} height={(counts[i] / 200) * 150} fill="none" stroke="var(--text)" strokeDasharray="3 2" />}
                  <text x={x + 28} y={hgt >= 0 ? 152 - hgt : 172 - hgt} textAnchor="middle" fontSize={12.5} fontWeight={700} fill={bad ? 'var(--c-bad)' : 'var(--text)'} style={{ fontFamily: 'var(--font-mono)' }}>
                    {p ? `${(v * 100).toFixed(1)}%` : '÷0'}
                  </text>
                  <text x={x + 28} y={180} textAnchor="middle" fontSize={12} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-ui)' }}>
                    {c}
                  </text>
                </g>
              );
            })}
          </svg>
          {problem ? (
            <div className="feedback bad" style={{ marginTop: 6 }}>
              <div className="feedback-title">✕ This method breaks</div>
              {problem}
            </div>
          ) : method !== 'softmax' ? (
            <div className="feedback" style={{ marginTop: 6 }}>
              This works for these particular scores… <strong>but does it always?</strong> Try dragging some scores below zero.
            </div>
          ) : (
            <>
              <div style={{ fontSize: 13.5, overflowX: 'auto' }}>
                <Tex block>{`p_{\\text{cat}} = \\frac{e^{${fmt(z[0], 1)}/${fmt(T, 2)}}}{${exps.map((e, i) => col(['c-a', 'c-b', 'c-c', 'c-grad'][i], fmt(e, 2))).join('+')}} = \\frac{${fmt(exps[0], 2)}}{${fmt(sumExp, 2)}} = ${fmt(p![0], 3)}`}</Tex>
              </div>
              <div className="btn-row">
                <button className="btn small" onClick={draw}>Sample 200 predictions</button>
                <span className="muted" style={{ fontSize: 13 }}>
                  entropy: {entropy(p!).toFixed(2)} bits {counts && ' · dashed bars: observed frequencies'}
                </span>
              </div>
              <div className="live-note" style={{ marginTop: 8 }}>
                {T < 0.3 ? (
                  <><strong>Very low temperature:</strong> differences between scores are magnified, so the top class takes almost all the probability. In the limit, softmax becomes “pick the maximum” (argmax).</>
                ) : T > 2.5 ? (
                  <><strong>High temperature:</strong> differences are shrunk, so the distribution flattens towards uniform (25% each). The model sounds less sure of everything.</>
                ) : (
                  <>Ratios depend only on score <em>differences</em>: cat’s score is {fmt(z[0] - z[1], 1)} higher than dog’s, so cat is e^({fmt(z[0] - z[1], 1)}/{fmt(T, 2)}) ≈ {(Math.exp((z[0] - z[1]) / T)).toFixed(2)}× as likely.</>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </Widget>
  );
}
