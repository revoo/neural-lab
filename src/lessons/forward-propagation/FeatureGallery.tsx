import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { Plot } from '../../components/plot';
import { Heatmap, fieldImage, sequential } from '../../components/heatmap';
import { NetDiagram, type NetSelection } from '../../components/NetDiagram';
import { LAYERS, sigmoid, type Layer } from './islands';
import { IslandOutlines, nf } from './ForwardStepper';

type Override = 'normal' | 'off' | 'on';

/** Forward pass with one neuron's output forced to 0 or 1 (a "stuck" neuron). Returns every activation. */
function traceWith(x: number[], stuck: { col: number; j: number; mode: Override } | null, layers: Layer[] = LAYERS): number[][] {
  const a: number[][] = [x];
  layers.forEach((L, l) => {
    const prev = a[l];
    const out = L.W.map((row, j) => sigmoid(row.reduce((s, w, i) => s + w * prev[i], L.b[j])));
    if (stuck && stuck.col === l + 1 && stuck.mode !== 'normal') out[stuck.j] = stuck.mode === 'on' ? 1 : 0;
    a.push(out);
  });
  return a;
}

const DOMAIN: [number, number, number, number] = [-2.2, 2.2, -2.2, 2.2];

export function FeatureGallery() {
  const [sel, setSel] = useState<NetSelection>({ kind: 'neuron', col: 2, j: 0 });
  const [mode, setMode] = useState<Override>('normal');
  const neuron = sel?.kind === 'neuron' && sel.col > 0 ? sel : null;
  const stuck = neuron && mode !== 'normal' ? { col: neuron.col, j: neuron.j, mode } : null;

  // Thumbnails for every neuron, recomputed when a neuron is forced on/off.
  const thumbs = useMemo(() => {
    const res = 44;
    const cols = [2, 6, 2, 1];
    const vals = cols.map((n) => Array.from({ length: n }, () => new Float64Array(res * res)));
    for (let r = 0; r < res; r++)
      for (let c = 0; c < res; c++) {
        const x = DOMAIN[0] + ((c + 0.5) / res) * 4.4;
        const y = DOMAIN[3] - ((r + 0.5) / res) * 4.4;
        const a = traceWith([x, y], stuck);
        a.forEach((layer, col) => layer.forEach((v, j) => (vals[col][j][r * res + c] = col === 0 ? (v + 2.2) / 4.4 : v)));
      }
    return vals.map((col) =>
      col.map((arr) => {
        let k = 0;
        return fieldImage(() => arr[k++], DOMAIN, res, (v) => sequential(v));
      }),
    );
  }, [stuck?.col, stuck?.j, stuck?.mode]); // eslint-disable-line react-hooks/exhaustive-deps

  const big = useMemo(() => {
    if (!neuron) return (x: number, y: number) => traceWith([x, y], stuck)[3][0];
    return (x: number, y: number) => traceWith([x, y], stuck)[neuron.col][neuron.j];
  }, [neuron?.col, neuron?.j, stuck?.mode]); // eslint-disable-line react-hooks/exhaustive-deps
  const finalF = useMemo(() => (x: number, y: number) => traceWith([x, y], stuck)[3][0], [stuck?.col, stuck?.j, stuck?.mode]); // eslint-disable-line react-hooks/exhaustive-deps

  const L = neuron ? LAYERS[neuron.col - 1] : null;
  const srcNames = neuron ? (neuron.col === 1 ? ['east', 'north'] : LAYERS[neuron.col - 2].names) : [];
  const name = neuron && L ? L.names[neuron.j] : 'the final output';

  let note: React.ReactNode;
  if (!neuron) note = <>Click any neuron to see, as a picture, everything it responds to across the whole map.</>;
  else if (mode === 'on')
    note =
      neuron.col === 1 ? (
        <>
          This coastline detector now says “land side” everywhere. The island it belonged to has <strong>lost a coast</strong>, and leaks off the edge of the map in that
          direction. The other island is untouched: no neuron downstream of it listens to this one.
        </>
      ) : neuron.col === 2 ? (
        <>The island detector now insists “I’m inside my island” everywhere, so the land neuron (an OR) says land everywhere.</>
      ) : (
        <>Forcing the output on makes the whole map land. Nothing upstream changed — only the final answer.</>
      );
  else if (mode === 'off')
    note =
      neuron.col === 1 ? (
        <>
          With one coastline detector silenced, its island’s detector can never collect all three “yes” votes. <strong>The whole island vanishes</strong> — one dead
          neuron deleted a feature two layers downstream.
        </>
      ) : neuron.col === 2 ? (
        <>Silencing an island detector deletes that island from the final map. The other island survives, because the OR only needs one vote.</>
      ) : (
        <>With the output forced off, everything is sea.</>
      );
  else if (neuron.col === 1)
    note = (
      <>
        A first-layer neuron can only ever draw <strong>one straight line</strong>: bright on one side, dark on the other. This one is the <strong>{name}</strong>. Every
        neuron in this layer is the single neuron from earlier lessons.
      </>
    );
  else if (neuron.col === 2)
    note = (
      <>
        A second-layer neuron combines lines into a <strong>shape</strong>. It doesn’t see the map at all, only the six coastline answers — but because those answers
        are themselves pictures, its response is a picture too: the <strong>{name}</strong>, a triangle no single neuron could draw.
      </>
    );
  else
    note = (
      <>
        The last layer combines shapes into a <strong>union of shapes</strong>: land is “west island OR east island”. Lines, then shapes, then combinations of shapes —
        each layer builds on the vocabulary of the one before.
      </>
    );

  return (
    <Widget
      title="What every neuron sees"
      wide
      caption={
        <>
          Each neuron is drawn as a tiny map of its own output for every point on the island map (bright = close to 1). Click a neuron to enlarge its map and see what it
          listens to. Then try <strong>breaking it</strong>: force it permanently off or on, and watch which pictures downstream change.
        </>
      }
    >
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 440px' }}>
          <NetDiagram
            sizes={[2, 6, 2, 1]}
            W={LAYERS.map((l) => l.W)}
            thumbs={thumbs}
            showValues={false}
            selected={sel}
            onSelect={(s) => {
              setSel(s && s.kind === 'neuron' ? s : null);
              setMode('normal');
            }}
            inputLabels={['east', 'north']}
            width={470}
            height={330}
            node={40}
            ariaLabel="Network where each neuron shows a thumbnail of its response"
          />
        </div>
        <div className="widget-col" style={{ flex: '0 1 280px' }}>
          <div className="field-label">{neuron ? `Layer ${neuron.col}: ${name}` : 'Final output'}</div>
          <Plot x={[-2.2, 2.2]} y={[-2.2, 2.2]} width={260} height={260} grid={false} tickLabels={false} axes={false} pad={8} ariaLabel="Selected neuron's response map">
            <Heatmap f={big} range={[0, 1]} resolution={90} opacity={0.9} />
            <IslandOutlines />
          </Plot>
          {neuron && L && (
            <div className="readout" style={{ marginTop: 8, fontSize: 12.5 }}>
              {L.W[neuron.j].map((w, i) =>
                w === 0 ? null : (
                  <div key={i}>
                    <span className="k">{srcNames[i].padEnd(neuron.col === 1 ? 6 : 26)}</span> <span className={w > 0 ? 'c-a' : 'c-b'}>×{nf(w)}</span>
                  </div>
                ),
              )}
              <div>
                <span className="k">{'bias'.padEnd(neuron.col === 1 ? 6 : 26)}</span> {nf(L.b[neuron.j])}
              </div>
            </div>
          )}
          {neuron && (
            <div style={{ marginTop: 8 }}>
              <Segmented
                label="Break this neuron"
                value={mode}
                onChange={setMode}
                options={[
                  { value: 'normal', label: 'working' },
                  { value: 'off', label: 'stuck at 0' },
                  { value: 'on', label: 'stuck at 1' },
                ]}
              />
            </div>
          )}
        </div>
      </div>
      <div className="widget-row" style={{ marginTop: 10, alignItems: 'center' }}>
        {stuck && (
          <div style={{ flex: '0 0 auto' }}>
            <div className="field-label">Final answer now</div>
            <Plot x={[-2.2, 2.2]} y={[-2.2, 2.2]} width={130} height={130} grid={false} tickLabels={false} axes={false} pad={4} ariaLabel="Final output with the broken neuron">
              <Heatmap f={finalF} range={[0, 1]} resolution={60} opacity={0.9} />
              <IslandOutlines />
            </Plot>
          </div>
        )}
        <div className="live-note" style={{ flex: '1 1 300px' }}>
          {note}
        </div>
      </div>
    </Widget>
  );
}
