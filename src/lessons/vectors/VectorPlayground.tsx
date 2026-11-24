import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { Arrow, Handle, Label, Plot } from '../../components/plot';

type V = [number, number];
const f = (x: number) => (Math.round(x * 100) / 100).toFixed(1);
const len = (v: V) => Math.hypot(v[0], v[1]);

export function VectorPlayground() {
  const [a, setA] = useState<V>([3, 1]);
  const [b, setB] = useState<V>([1, 2.5]);
  const [k, setK] = useState(2);
  const [mode, setMode] = useState<'add' | 'scale' | 'subtract'>('add');

  const sum: V = [a[0] + b[0], a[1] + b[1]];
  const diff: V = [a[0] - b[0], a[1] - b[1]];
  const scaled: V = [k * a[0], k * a[1]];
  const cancelled = mode === 'add' && len(sum) < 0.01;

  return (
    <Widget
      title="Vector playground"
      wide
      caption={
        mode === 'add' ? (
          <>
            <strong>Adding vectors</strong> means following one arrow, then the other. Notice the dashed copy of{' '}
            <span className="c-b">b</span> starting at the tip of <span className="c-a">a</span>: that’s “tip-to-tail” addition, and it
            lands in exactly the same place as adding the lists number by number. <strong>Challenge:</strong> drag the handles so that a + b lands
            exactly on the origin.
          </>
        ) : mode === 'scale' ? (
          <>
            <strong>Scaling</strong> multiplies every number in the list by the same amount. The arrow stretches or shrinks but stays on the same
            line through the origin. Try a negative scale: the arrow flips to point the opposite way.
          </>
        ) : (
          <>
            <strong>Subtracting</strong> gives the arrow that goes <em>from</em> b <em>to</em> a. Its length is the distance between the two
            points — which is how “how different are these two things?” becomes a geometric question.
          </>
        )
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <Plot x={[-6, 6]} y={[-6, 6]} width={440} height={440} ariaLabel="Coordinate plane with draggable vectors">
            {mode === 'add' && (
              <>
                <Arrow from={[0, 0]} to={b} color="var(--c-b)" opacity={0.35} />
                <Arrow from={a} to={sum} color="var(--c-b)" dashed />
                <Arrow from={[0, 0]} to={sum} color="var(--c-c)" width={3} />
                <Label at={sum} dx={8} dy={-6} color="var(--c-c)">a + b</Label>
              </>
            )}
            {mode === 'scale' && (
              <>
                <Arrow from={[0, 0]} to={scaled} color="var(--c-c)" width={5} opacity={0.75} />
                <Label at={scaled} dx={8} dy={14} color="var(--c-c)">{`${k.toFixed(1)} · a`}</Label>
              </>
            )}
            {mode === 'subtract' && (
              <>
                <Arrow from={b} to={a} color="var(--c-c)" dashed />
                <Arrow from={[0, 0]} to={diff} color="var(--c-c)" width={3} />
                <Label at={diff} dx={8} dy={-6} color="var(--c-c)">a − b</Label>
              </>
            )}
            <Arrow from={[0, 0]} to={a} color="var(--c-a)" width={3} />
            {mode !== 'scale' && <Arrow from={[0, 0]} to={b} color="var(--c-b)" width={3} />}
            <Label at={a} dx={10} dy={4} color="var(--c-a)">a</Label>
            {mode !== 'scale' && <Label at={b} dx={10} dy={4} color="var(--c-b)">b</Label>}
            <Handle x={a[0]} y={a[1]} onDrag={(x, y) => setA([x, y])} color="var(--c-a)" snap={0.5} label="tip of vector a" />
            {mode !== 'scale' && <Handle x={b[0]} y={b[1]} onDrag={(x, y) => setB([x, y])} color="var(--c-b)" snap={0.5} label="tip of vector b" />}
          </Plot>
        </div>
        <div className="widget-col controls">
          <Segmented
            label="Operation"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'add', label: 'a + b' },
              { value: 'scale', label: 'k · a' },
              { value: 'subtract', label: 'a − b' },
            ]}
          />
          {mode === 'scale' && <Slider label="Scale factor k" value={k} min={-3} max={3} step={0.1} onChange={setK} format={(v) => v.toFixed(1)} color="c-c" />}
          <div className="readout">
            <div><span className="c-a">a</span> = [{f(a[0])}, {f(a[1])}]</div>
            {mode !== 'scale' && <div><span className="c-b">b</span> = [{f(b[0])}, {f(b[1])}]</div>}
            <div style={{ height: 8 }} />
            {mode === 'add' && (
              <div>
                <span className="c-c">a + b</span> = [{f(a[0])} + {f(b[0])}, {f(a[1])} + {f(b[1])}]<br />
                <span style={{ paddingLeft: 52 }} />= <strong>[{f(sum[0])}, {f(sum[1])}]</strong>
              </div>
            )}
            {mode === 'scale' && (
              <div>
                <span className="c-c">k · a</span> = [{k.toFixed(1)} × {f(a[0])}, {k.toFixed(1)} × {f(a[1])}]<br />
                <span style={{ paddingLeft: 52 }} />= <strong>[{f(scaled[0])}, {f(scaled[1])}]</strong>
              </div>
            )}
            {mode === 'subtract' && (
              <div>
                <span className="c-c">a − b</span> = [{f(a[0])} − {f(b[0])}, {f(a[1])} − {f(b[1])}]<br />
                <span style={{ paddingLeft: 52 }} />= <strong>[{f(diff[0])}, {f(diff[1])}]</strong>
              </div>
            )}
            <div style={{ height: 8 }} />
            <div className="faint">length of a = √({f(a[0])}² + {f(a[1])}²) = {len(a).toFixed(2)}</div>
            {mode === 'scale' && <div className="faint">length of k·a = {len(scaled).toFixed(2)} = |k| × {len(a).toFixed(2)}</div>}
            {mode === 'subtract' && <div className="faint">distance a↔b = {len(diff).toFixed(2)}</div>}
          </div>
          {mode === 'add' && (
            <div>
              {cancelled ? (
                <div className="live-note"><strong>Solved.</strong> b is now exactly −a: the same arrow, flipped. Every number in b is the negative of the matching number in a, so they cancel out entry by entry.</div>
              ) : (
                <span className="status neutral">Challenge: make a + b = [0, 0]</span>
              )}
            </div>
          )}
        </div>
      </div>
    </Widget>
  );
}
