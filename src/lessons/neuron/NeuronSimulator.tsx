import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { FnPath, Dot, Plot, Seg } from '../../components/plot';
import { Tex, col, fmt, pn } from '../../components/Tex';
import { ACTIVATIONS, type ActivationName } from '../../lib/activations';

const ACTS: ActivationName[] = ['step', 'sigmoid', 'tanh', 'relu', 'linear'];

export function NeuronSimulator() {
  const [x, setX] = useState([1.0, 0.5]);
  const [w, setW] = useState([0.8, -1.2]);
  const [b, setB] = useState(0.3);
  const [act, setAct] = useState<ActivationName>('sigmoid');
  const A = ACTIVATIONS[act];
  const z = w[0] * x[0] + w[1] * x[1] + b;
  const a = A.f(z);

  const W = 520;
  const H = 260;
  const inY = [60, 150, 230];
  const nx = 330;
  const ny = 140;
  const edgeColor = (v: number) => (v >= 0 ? 'var(--c-pos)' : 'var(--c-neg)');
  const contrib = [w[0] * x[0], w[1] * x[1], b];
  const intensity = Math.max(0, Math.min(1, act === 'tanh' ? (a + 1) / 2 : act === 'relu' || act === 'linear' ? a / 3 : a));

  return (
    <Widget
      title="Neuron simulator"
      wide
      caption={
        <>
          Edge thickness shows each weight’s size; <span className="c-a">blue</span> edges are positive (they push the output up), <span className="c-b">orange</span>{' '}
          edges negative. The number on each edge is that input’s <em>contribution</em>: weight × input. Change the inputs, weights and bias and watch the sum{' '}
          <em>z</em> and the output <em>a</em> respond. The small graph on the right shows where z lands on the activation function.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 440px' }}>
          <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Diagram of a single neuron">
            {[0, 1, 2].map((i) => {
              const wv = i < 2 ? w[i] : b;
              const y = inY[i];
              return (
                <g key={i}>
                  <line x1={70} y1={y} x2={nx - 46} y2={ny} stroke={edgeColor(wv)} strokeWidth={1 + Math.abs(wv) * 3.2} opacity={0.75} />
                  <text x={(70 + nx - 46) / 2} y={(y + ny) / 2 - 9} textAnchor="middle" fontSize={12} fontWeight={700} fill={edgeColor(wv)} style={{ fontFamily: 'var(--font-mono)', paintOrder: 'stroke', stroke: 'var(--bg-elev)', strokeWidth: 4 }}>
                    {i < 2 ? `${fmt(wv)}×${fmt(x[i])} = ${fmt(contrib[i])}` : `bias ${fmt(b)}`}
                  </text>
                  <circle cx={50} cy={y} r={24} fill="var(--bg)" stroke="var(--border-strong)" strokeWidth={1.5} />
                  <text x={50} y={y - 2} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--text)" style={{ fontFamily: 'var(--font-ui)' }}>
                    {i < 2 ? `x${i === 0 ? '₁' : '₂'}` : '1'}
                  </text>
                  <text x={50} y={y + 13} textAnchor="middle" fontSize={10.5} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>
                    {i < 2 ? fmt(x[i]) : 'bias'}
                  </text>
                </g>
              );
            })}
            <circle cx={nx} cy={ny} r={46} fill={`color-mix(in srgb, var(--accent) ${Math.round(intensity * 70)}%, var(--bg))`} stroke="var(--accent)" strokeWidth={2.5} />
            <line x1={nx} y1={ny - 46} x2={nx} y2={ny + 46} stroke="var(--accent)" strokeWidth={1} opacity={0.5} />
            <text x={nx - 22} y={ny + 6} textAnchor="middle" fontSize={20} fill="var(--text)">Σ</text>
            <text x={nx + 22} y={ny + 5} textAnchor="middle" fontSize={14} fill="var(--text)" fontStyle="italic">f</text>
            <text x={nx - 22} y={ny + 26} textAnchor="middle" fontSize={10.5} fill="var(--text-muted)" style={{ fontFamily: 'var(--font-mono)' }}>z={fmt(z)}</text>
            <line x1={nx + 46} y1={ny} x2={470} y2={ny} stroke="var(--accent)" strokeWidth={2.5} />
            <polygon points={`478,${ny} 466,${ny - 6} 466,${ny + 6}`} fill="var(--accent)" />
            <text x={430} y={ny - 12} textAnchor="middle" fontSize={14} fontWeight={700} fill="var(--accent-text)" style={{ fontFamily: 'var(--font-mono)' }}>
              a = {fmt(a, 3)}
            </text>
          </svg>
          <div style={{ fontSize: 15 }}>
            <Tex block>{`z = ${col('c-a', pn(w[0]))}\\cdot${pn(x[0])} + ${col(w[1] >= 0 ? 'c-a' : 'c-b', pn(w[1]))}\\cdot${pn(x[1])} + ${pn(b)} = ${fmt(z, 3)} \\qquad a = f(z) = ${fmt(a, 3)}`}</Tex>
          </div>
        </div>
        <div className="widget-col controls" style={{ flex: '1 1 280px' }}>
          <div className="controls inline">
            <Slider label="input x₁" value={x[0]} min={-2} max={2} step={0.05} onChange={(v) => setX([v, x[1]])} />
            <Slider label="input x₂" value={x[1]} min={-2} max={2} step={0.05} onChange={(v) => setX([x[0], v])} />
            <Slider label="weight w₁" value={w[0]} min={-2} max={2} step={0.05} onChange={(v) => setW([v, w[1]])} color={w[0] >= 0 ? 'c-a' : 'c-b'} />
            <Slider label="weight w₂" value={w[1]} min={-2} max={2} step={0.05} onChange={(v) => setW([w[0], v])} color={w[1] >= 0 ? 'c-a' : 'c-b'} />
            <Slider label="bias b" value={b} min={-2} max={2} step={0.05} onChange={setB} />
          </div>
          <Segmented label="Activation f" value={act} onChange={setAct} options={ACTS.map((k) => ({ value: k, label: ACTIVATIONS[k].label.replace(' (none)', '') }))} />
          <Plot x={[-4, 4]} y={act === 'relu' || act === 'linear' ? [-2, 4] : [-1.2, 1.2]} width={280} height={140} tickLabels={false} ariaLabel="Activation function with current input marked">
            <FnPath f={A.f} color="var(--accent)" width={2.5} />
            <Seg from={[Math.max(-4, Math.min(4, z)), 0]} to={[Math.max(-4, Math.min(4, z)), a]} dash="3 3" />
            <Dot at={[Math.max(-4, Math.min(4, z)), a]} r={5} color="var(--accent)" />
          </Plot>
        </div>
      </div>
    </Widget>
  );
}
