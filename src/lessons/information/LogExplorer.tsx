import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Dot, FnPath, Handle, Label, Plot, Seg } from '../../components/plot';
import { Tex, col, fmt } from '../../components/Tex';

export function LogExplorer() {
  const [x, setX] = useState(8);
  const y = Math.log2(x);
  return (
    <Widget
      title="Logarithms count doublings"
      caption={
        <>
          Drag the point. log₂(x) answers “how many times must I double 1 to reach x?”. Use the buttons: <strong>doubling x always adds exactly 1</strong> to the
          logarithm, wherever you start. Multiplication on the input side becomes addition on the output side. Values below 1 need a <em>negative</em> number of
          doublings — that is, halvings.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 340px' }}>
          <Plot x={[0, 17]} y={[-4, 5]} width={400} height={280} xLabel="x" yLabel="log₂ x" ariaLabel="Graph of log base 2">
            <FnPath f={(t) => Math.log2(t)} color="var(--c-a)" width={3} domain={[0.05, 17]} samples={400} />
            {[1, 2, 4, 8, 16].map((p) => (
              <Dot key={p} at={[p, Math.log2(p)]} r={3} color="var(--text-faint)" />
            ))}
            <Seg from={[x, 0]} to={[x, y]} dash="3 3" />
            <Seg from={[0, y]} to={[x, y]} dash="3 3" />
            <Label at={[x, y]} dx={10} dy={-8} color="var(--c-a)">{`log₂(${fmt(x, 2)}) = ${fmt(y, 2)}`}</Label>
            <Handle x={x} y={y} onDrag={(nx) => setX(Math.max(0.0625, Math.round(nx * 100) / 100))} color="var(--c-a)" label="point on the log curve" />
          </Plot>
        </div>
        <div className="widget-col controls">
          <div style={{ fontSize: 16 }}>
            <Tex block>{`2^{${col('c-a', fmt(y, 2))}} = ${fmt(x, 2)} \\quad\\Longleftrightarrow\\quad \\log_2(${fmt(x, 2)}) = ${col('c-a', fmt(y, 2))}`}</Tex>
          </div>
          <div className="btn-row">
            <button className="btn small" onClick={() => setX(Math.max(0.0625, x / 2))}>÷ 2</button>
            <button className="btn small primary" onClick={() => setX(Math.min(16, x * 2))}>× 2</button>
            <button className="btn small" onClick={() => setX(1)}>x = 1</button>
          </div>
          <div className="readout">
            <span className="k">log₂(1) = 0</span>    no doublings needed
            <br />
            <span className="k">log₂(1024) = 10</span> ten doublings — or ten yes/no questions to find one item in 1,024
            <br />
            <span className="k">log₂(1/8) = −3</span>  three halvings
          </div>
        </div>
      </div>
    </Widget>
  );
}
