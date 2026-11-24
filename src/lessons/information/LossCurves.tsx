import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Toggle } from '../../components/controls';
import { Dot, FnPath, Handle, Label, Plot, Seg } from '../../components/plot';
import { Tex, col, fmt } from '../../components/Tex';

export function LossCurves() {
  const [p, setP] = useState(0.6);
  const [mse, setMse] = useState(true);
  const ce = -Math.log(p);
  const sq = (1 - p) ** 2;
  return (
    <Widget
      title="How much should a wrong answer cost?"
      wide
      caption={
        <>
          The true answer is “cat”. The horizontal axis is the probability the model gave to cat. Drag the point. The{' '}
          <span className="c-a">cross-entropy</span> loss is −ln p: the model’s surprise at the truth. For comparison, the{' '}
          <span className="c-b">squared error</span> (1 − p)² is shown too. Slide p towards zero and watch what each one does to a confidently wrong model.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 420px' }}>
          <Plot x={[0, 1]} y={[0, 5]} width={460} height={300} xLabel="p(correct class)" yLabel="loss" ariaLabel="Cross-entropy and squared error curves">
            <FnPath f={(t) => -Math.log(t)} color="var(--c-a)" width={3} domain={[0.004, 1]} samples={400} />
            {mse && <FnPath f={(t) => (1 - t) ** 2} color="var(--c-b)" width={2.5} />}
            <Seg from={[p, 0]} to={[p, Math.min(5, ce)]} dash="3 3" />
            {mse && <Dot at={[p, sq]} r={5} color="var(--c-b)" />}
            <Label at={[p, Math.min(4.7, ce)]} dx={10} dy={-6} color="var(--c-a)">{`−ln p = ${ce.toFixed(2)}`}</Label>
            <Handle x={p} y={Math.min(5, ce)} onDrag={(x) => setP(Math.min(0.999, Math.max(0.005, Math.round(x * 1000) / 1000)))} color="var(--c-a)" label="probability given to the correct class" />
          </Plot>
        </div>
        <div className="widget-col controls">
          <div style={{ fontSize: 15 }}>
            <Tex block>{`${col('c-a', '\\text{cross-entropy}')} = -\\ln(${fmt(p, 3)}) = ${col('c-a', fmt(ce, 3))}`}</Tex>
            {mse && <Tex block>{`${col('c-b', '\\text{squared error}')} = (1 - ${fmt(p, 3)})^2 = ${col('c-b', fmt(sq, 3))}`}</Tex>}
          </div>
          <table className="num-table" style={{ width: '100%' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>model says p(cat) =</th>
                <th className="c-a">cross-entropy</th>
                {mse && <th className="c-b">squared error</th>}
              </tr>
            </thead>
            <tbody>
              {[0.99, 0.9, 0.5, 0.1, 0.01, 0.0001].map((q) => (
                <tr key={q} style={{ cursor: 'pointer' }} onClick={() => setP(Math.max(0.005, q))}>
                  <td style={{ textAlign: 'left' }}>{q}</td>
                  <td>{(-Math.log(q)).toFixed(2)}</td>
                  {mse && <td>{((1 - q) ** 2).toFixed(2)}</td>}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="live-note">
            {p < 0.05 ? (
              <>
                <strong>Confidently wrong.</strong> The model gave the truth only {(p * 100).toFixed(1)}%. Cross-entropy charges {ce.toFixed(1)} and keeps climbing towards
                infinity; squared error can never exceed 1, so it barely distinguishes “badly wrong” from “catastrophically wrong”.
              </>
            ) : p > 0.9 ? (
              <>
                <strong>Confident and right.</strong> Both losses are small. Cross-entropy is never exactly zero unless p = 1, so there’s always a little pressure to be
                even more confident.
              </>
            ) : (
              <>
                Each halving of p adds the same amount (ln 2 ≈ 0.69) to the cross-entropy. Going from 50% to 25% costs as much as going from 2% to 1%.
              </>
            )}
          </div>
          <Toggle label="Compare with squared error" checked={mse} onChange={setMse} />
        </div>
      </div>
    </Widget>
  );
}
