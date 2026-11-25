// (decision-plane panel and machine log added by Claude Opus 5.5)
import { useEffect, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Handle, Plot } from '../../components/plot';
import { MachineLog, useMachineLog } from '../../components/MachineLog';
import { cost, run, score, snapLine, tallyLine, ROWS, type Circuit, type Neuron } from './workbench';

const LINE_COLORS = ['var(--accent)', 'var(--c-b)', 'var(--c-a)'];
const BOX: [number, number] = [-0.6, 1.6];

/** Clip the line w·p + b = 0 to the plotting box: its two end points, or null if it misses. */
function clipLine(n: Neuron): [[number, number], [number, number]] | null {
  const [w1, w2] = n.w;
  if (w1 === 0 && w2 === 0) return null;
  const pts: [number, number][] = [];
  const [lo, hi] = BOX;
  for (const x of [lo, hi]) if (w2 !== 0) { const y = -(w1 * x + n.b) / w2; if (y >= lo - 1e-9 && y <= hi + 1e-9) pts.push([x, y]); }
  for (const y of [lo, hi]) if (w1 !== 0) { const x = -(w2 * y + n.b) / w1; if (x >= lo - 1e-9 && x <= hi + 1e-9) pts.push([x, y]); }
  const uniq = pts.filter((p, i) => pts.findIndex((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-6) === i);
  return uniq.length >= 2 ? [uniq[0], uniq[1]] : null;
}

/**
 * The four input corners with each input neuron's decision line (it fires on
 * the side the little arrow points to). Corners are filled where the puzzle
 * wants a 1, and ringed red where the circuit gets them wrong. Drag the round
 * handle to move a line, the square one to turn it; it snaps to the nearest
 * weights the buttons can make.
 */
function DecisionPlane({ c, P, results, onChange }: { c: Circuit; P: Puzzle; results: boolean[]; onChange: (c: Circuit) => void }) {
  const neurons = c.hidden.length ? c.hidden : [c.out];
  const names = c.hidden.length ? c.hidden.map((_, i) => `h${i + 1}`) : ['output'];
  const setN = (j: number, n: Neuron) => onChange(c.hidden.length ? { ...c, hidden: c.hidden.map((x, i) => (i === j ? n : x)) } : { ...c, out: n });
  return (
    <div>
      <Plot x={BOX} y={BOX} width={260} height={260} grid={false} tickLabels={false} xLabel="A" yLabel="B" pad={22} ariaLabel="The four input corners and each neuron's decision line" style={{ width: '100%', maxWidth: 280, height: 'auto', display: 'block' }}>
        {({ x, y }) => (
          <g>
            {neurons.map((n, j) => {
              const seg = clipLine(n);
              if (!seg) return null;
              const [[x1, y1], [x2, y2]] = seg;
              const mx = (x1 + x2) / 2;
              const my = (y1 + y2) / 2;
              const len = Math.hypot(n.w[0], n.w[1]);
              const ax = mx + (0.22 * n.w[0]) / len;
              const ay = my + (0.22 * n.w[1]) / len;
              return (
                <g key={j}>
                  <line x1={x(x1)} y1={y(y1)} x2={x(x2)} y2={y(y2)} stroke={LINE_COLORS[j % 3]} strokeWidth={2.5} />
                  <line x1={x(mx)} y1={y(my)} x2={x(ax)} y2={y(ay)} stroke={LINE_COLORS[j % 3]} strokeWidth={2} markerEnd="" />
                  <circle cx={x(ax)} cy={y(ay)} r={2.5} fill={LINE_COLORS[j % 3]} />
                  <text x={x(ax) + 4} y={y(ay) - 4} fontSize={10.5} fill={LINE_COLORS[j % 3]}>{names[j]} fires</text>
                </g>
              );
            })}
            {ROWS.map(([a, b], i) => (
              <g key={i}>
                <circle cx={x(a)} cy={y(b)} r={9} fill={P.target(a, b) ? 'var(--c-c)' : 'var(--bg)'} stroke={results[i] ? 'var(--text-muted)' : 'var(--c-bad)'} strokeWidth={results[i] ? 1.5 : 3} />
                <text x={x(a)} y={y(b) + 4} textAnchor="middle" fontSize={10} fontWeight={700} fill={P.target(a, b) ? '#fff' : 'var(--text)'}>{P.target(a, b)}</text>
              </g>
            ))}
          </g>
        )}
        {neurons.map((n, j) => {
          const seg = clipLine(n);
          if (!seg) return null;
          const len = Math.hypot(n.w[0], n.w[1]);
          // the point of the line nearest the centre of the square, and one 0.6 further along it
          const d = (n.w[0] * 0.5 + n.w[1] * 0.5 + n.b) / len;
          const m: [number, number] = [0.5 - (d * n.w[0]) / len, 0.5 - (d * n.w[1]) / len];
          const dir: [number, number] = [-n.w[1] / len, n.w[0] / len];
          const t: [number, number] = [m[0] + 0.6 * dir[0], m[1] + 0.6 * dir[1]];
          return [
            <Handle key={`m${j}`} x={m[0]} y={m[1]} r={6} color={LINE_COLORS[j % 3]} onDrag={(a, b) => setN(j, snapLine([a, b], [a + dir[0], b + dir[1]], n))} />,
            <Handle key={`t${j}`} x={t[0]} y={t[1]} r={4.5} color={LINE_COLORS[j % 3]} onDrag={(a, b) => setN(j, snapLine(m, [a, b], n))} />,
          ];
        })}
      </Plot>
      <div className="muted" style={{ fontSize: 12, maxWidth: 300 }}>
        {neurons.every((n) => n.w[0] === 0 && n.w[1] === 0)
          ? 'Give a neuron some weight and its decision line appears here.'
          : 'Filled corners should fire. Each line splits the square into “fires” and “silent”; drag the big dot to move it, the small one to turn it.'}
        {P.hidden && c.hidden.length === 0 && ' Can any single line put both filled corners on one side and both empty ones on the other?'}
      </div>
    </div>
  );
}

type Puzzle = {
  id: string;
  name: string;
  brief: string;
  target: (a: number, b: number) => number;
  hidden: boolean;
  best: number;
  hint: string;
};

const PUZZLES: Puzzle[] = [
  { id: 'and', name: 'AND', brief: 'Fire only when A and B are both on.', target: (a, b) => a & b, hidden: false, best: 1.5, hint: 'Each input alone shouldn’t be enough to push the tally above zero — but the two together should. The bias sets how much evidence is “enough”.' },
  { id: 'or', name: 'OR', brief: 'Fire when at least one input is on.', target: (a, b) => a | b, hidden: false, best: 1, hint: 'Now a single input should be enough. When nothing is on, the tally is just the bias.' },
  { id: 'nand', name: 'NAND', brief: 'Fire unless both inputs are on.', target: (a, b) => 1 - (a & b), hidden: false, best: 2, hint: 'This is AND turned upside down. The neuron should fire by default (bias) and be shut off by evidence (negative weights).' },
  { id: 'anotb', name: 'A, NOT B', brief: 'Fire when A is on and B is off.', target: (a, b) => a & (1 - b), hidden: false, best: 1, hint: 'One input argues for firing, the other argues against.' },
  { id: 'xor', name: 'XOR', brief: 'Fire when exactly one input is on.', target: (a, b) => a ^ b, hidden: true, best: 3, hint: 'One neuron can’t do it — so build two simpler detectors in the hidden layer and let the output neuron combine them. You’ve already built every part you need in the earlier puzzles.' },
  { id: 'xnor', name: 'SAME', brief: 'Fire when both inputs are equal (both on, or both off).', target: (a, b) => 1 - (a ^ b), hidden: true, best: 3.5, hint: 'Two situations should make it fire: “both on” and “both off”. Detect each separately.' },
];

const KEY = 'nl-logic-lab-v1';
type Saved = Record<string, { circuit: Circuit; solved: boolean; bestCost?: number }>;
const load = (): Saved => {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? '{}');
  } catch {
    return {};
  }
};

const blank = (hidden: number): Circuit => ({
  hidden: Array.from({ length: hidden }, () => ({ w: [0, 0], b: 0 })),
  out: { w: Array(hidden || 2).fill(0), b: 0 },
});

function Stepper({ value, onChange, label }: { value: number; onChange: (v: number) => void; label: string }) {
  const set = (v: number) => onChange(Math.max(-2, Math.min(2, Math.round(v * 2) / 2)));
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 2 }} aria-label={label}>
      <button className="btn small" style={{ padding: '1px 7px' }} onClick={() => set(value - 0.5)} aria-label={`decrease ${label}`}>−</button>
      <span className="mono" style={{ width: 38, textAlign: 'center', fontWeight: 700, color: value > 0 ? 'var(--c-pos)' : value < 0 ? 'var(--c-neg)' : 'var(--text-faint)' }}>
        {value > 0 ? '+' : ''}
        {value.toFixed(1)}
      </span>
      <button className="btn small" style={{ padding: '1px 7px' }} onClick={() => set(value + 0.5)} aria-label={`increase ${label}`}>+</button>
    </span>
  );
}

function NeuronCard({ name, n, inputs, onChange, value }: { name: string; n: Neuron; inputs: string[]; onChange: (n: Neuron) => void; value: number }) {
  return (
    <div style={{ border: `2px solid ${value ? 'var(--c-c)' : 'var(--border)'}`, borderRadius: 10, padding: '8px 10px', background: 'var(--bg)', minWidth: 190 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 700, fontSize: 13, marginBottom: 6 }}>
        <span>{name}</span>
        <span className={value ? 'c-c' : 'faint'}>{value ? '● firing' : '○ silent'}</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'auto auto', gap: '4px 10px', alignItems: 'center', fontSize: 13 }}>
        {inputs.map((inp, i) => (
          <FragmentRow key={inp} label={`weight from ${inp}`}>
            <Stepper value={n.w[i]} onChange={(v) => { const w = n.w.slice(); w[i] = v; onChange({ ...n, w }); }} label={`${name} weight from ${inp}`} />
          </FragmentRow>
        ))}
        <FragmentRow label="bias">
          <Stepper value={n.b} onChange={(v) => onChange({ ...n, b: v })} label={`${name} bias`} />
        </FragmentRow>
      </div>
    </div>
  );
}

function FragmentRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <span className="muted">{label}</span>
      {children}
    </>
  );
}

function Diagram({ c, row }: { c: Circuit; row: [number, number] }) {
  const r = run(c, row[0], row[1]);
  const W = 420;
  const H = 190;
  const inPos = [[50, 55], [50, 135]];
  const hPos = c.hidden.map((_, i) => [210, c.hidden.length === 1 ? 95 : 45 + (i * 100) / Math.max(1, c.hidden.length - 1)]);
  const outPos = [370, 95];
  const src = c.hidden.length ? hPos : inPos;
  const srcVals = c.hidden.length ? r.hidden : row;
  const edge = (from: number[], to: number[], w: number, active: boolean, key: string) =>
    w === 0 ? (
      <line key={key} x1={from[0]} y1={from[1]} x2={to[0]} y2={to[1]} stroke="var(--border-strong)" strokeWidth={1} strokeDasharray="3 4" />
    ) : (
      <line key={key} x1={from[0]} y1={from[1]} x2={to[0]} y2={to[1]} stroke={w > 0 ? 'var(--c-pos)' : 'var(--c-neg)'} strokeWidth={1 + Math.abs(w) * 2.2} opacity={active ? 0.95 : 0.25} />
    );
  const node = (p: number[], label: string, v: number, key: string) => (
    <g key={key}>
      <circle cx={p[0]} cy={p[1]} r={20} fill={v ? 'var(--c-c)' : 'var(--bg)'} stroke={v ? 'var(--c-c)' : 'var(--border-strong)'} strokeWidth={2} />
      <text x={p[0]} y={p[1] + 4.5} textAnchor="middle" fontSize={13} fontWeight={700} fill={v ? '#fff' : 'var(--text)'} style={{ fontFamily: 'var(--font-ui)' }}>
        {label}
      </text>
    </g>
  );
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Circuit diagram">
      {c.hidden.map((h, j) => h.w.map((w, i) => edge(inPos[i], hPos[j], w, row[i] === 1, `ih${i}${j}`)))}
      {c.out.w.map((w, j) => edge(src[j], outPos, w, srcVals[j] === 1, `o${j}`))}
      {inPos.map((p, i) => node(p, i ? 'B' : 'A', row[i], `in${i}`))}
      {hPos.map((p, j) => node(p, `h${j + 1}`, r.hidden[j], `h${j}`))}
      {node(outPos, 'out', r.out, 'out')}
    </svg>
  );
}

export function LogicWorkbench() {
  const [saved, setSaved] = useState<Saved>(load);
  const [pi, setPi] = useState(0);
  const P = PUZZLES[pi];
  const [c, setC] = useState<Circuit>(() => saved[P.id]?.circuit ?? blank(P.hidden ? 2 : 0));
  const [row, setRow] = useState<[number, number]>([1, 1]);
  const [showHint, setShowHint] = useState(false);
  const log = useMachineLog();
  const pickRow = (r: [number, number]) => {
    setRow(r);
    const res = run(c, r[0], r[1]);
    c.hidden.forEach((h, j) => log.push(tallyLine(`h${j + 1}`, h, r, ['A', 'B'])));
    const outIn = c.hidden.length ? res.hidden : r;
    const outLabels = c.hidden.length ? c.hidden.map((_, j) => `h${j + 1}`) : ['A', 'B'];
    const ok = res.out === P.target(r[0], r[1]);
    log.push(tallyLine('output', c.out, outIn, outLabels) + ` · wanted ${P.target(r[0], r[1])} ${ok ? '✓' : '✕'}`, ok ? 'good' : 'bad');
  };

  useEffect(() => {
    setC(saved[P.id]?.circuit ?? blank(P.hidden ? 2 : 0));
    setShowHint(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pi]);

  const { results, solved } = score(c, P.target);
  const k = cost(c);

  useEffect(() => {
    const prev = saved[P.id];
    const next: Saved = { ...saved, [P.id]: { circuit: c, solved: solved || !!prev?.solved, bestCost: solved ? Math.min(k, prev?.bestCost ?? Infinity) : prev?.bestCost } };
    setSaved(next);
    try {
      localStorage.setItem(KEY, JSON.stringify(next));
    } catch {
      /* ignore */
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [c]);

  const setHiddenCount = (n: number) => {
    const hidden = Array.from({ length: n }, (_, i) => c.hidden[i] ?? { w: [0, 0], b: 0 });
    setC({ hidden, out: { w: Array.from({ length: n || 2 }, (_, i) => c.out.w[i] ?? 0), b: c.out.b } });
  };
  const outInputs = c.hidden.length ? c.hidden.map((_, i) => `h${i + 1}`) : ['A', 'B'];
  const best = saved[P.id]?.bestCost;

  return (
    <Widget
      title="Logic workbench"
      wide
      caption={
        <>
          Every neuron here uses the simplest response there is: it <strong>fires (outputs 1) if its tally is above zero</strong>, and stays silent otherwise. Set the
          weights and biases with the − / + buttons, or drag a neuron's decision line on the square of input corners. The truth table checks all four input combinations at once; click a row to watch that case flow through the circuit.
          Your work is saved in this browser.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 12 }}>
        {PUZZLES.map((p, i) => (
          <button key={p.id} className={`btn small${i === pi ? ' primary' : ''}`} onClick={() => setPi(i)}>
            {saved[p.id]?.solved ? '✓ ' : ''}
            {p.name}
          </button>
        ))}
      </div>
      <div className="live-note" style={{ marginBottom: 12 }}>
        <strong>{P.name}:</strong> {P.brief}{' '}
        {!P.hidden && <span className="muted">One neuron, no hidden layer.</span>}
        {P.hidden && <span className="muted">You may use a hidden layer.</span>}
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-start' }}>
            <Diagram c={c} row={row} />
            <DecisionPlane c={c} P={P} results={results} onChange={setC} />
          </div>
          <table className="num-table" style={{ width: '100%', marginTop: 8 }}>
            <thead>
              <tr>
                <th>A</th>
                <th>B</th>
                {c.hidden.map((_, i) => (
                  <th key={i}>h{i + 1}</th>
                ))}
                <th>out</th>
                <th>wanted</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {ROWS.map(([a, b], i) => {
                const r = run(c, a, b);
                const sel = row[0] === a && row[1] === b;
                return (
                  <tr key={i} onClick={() => pickRow([a, b])} style={{ cursor: 'pointer', background: sel ? 'var(--accent-soft)' : undefined }}>
                    <td>{a}</td>
                    <td>{b}</td>
                    {r.hidden.map((h, j) => (
                      <td key={j}>{h}</td>
                    ))}
                    <td style={{ fontWeight: 700 }}>{r.out}</td>
                    <td>{P.target(a, b)}</td>
                    <td className={results[i] ? 'c-good' : 'c-bad'} style={{ fontWeight: 700 }}>
                      {results[i] ? '✓' : '✕'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="muted" style={{ fontSize: 12.5, marginTop: 6 }}>
            Output tally for A={row[0]}, B={row[1]}: <span className="mono">{run(c, row[0], row[1]).outTally.toFixed(1)}</span> → {run(c, row[0], row[1]).outTally > 0 ? 'fires' : 'silent'}
          </div>
        </div>
        <div className="widget-col controls">
          {P.hidden && (
            <div className="btn-row">
              <span className="field-label" style={{ margin: 0 }}>Hidden neurons:</span>
              {[0, 1, 2, 3].map((n) => (
                <button key={n} className={`btn small${c.hidden.length === n ? ' primary' : ''}`} onClick={() => setHiddenCount(n)}>
                  {n}
                </button>
              ))}
            </div>
          )}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {c.hidden.map((h, j) => (
              <NeuronCard
                key={j}
                name={`h${j + 1}`}
                n={h}
                inputs={['A', 'B']}
                value={run(c, row[0], row[1]).hidden[j]}
                onChange={(n) => setC({ ...c, hidden: c.hidden.map((x, i) => (i === j ? n : x)) })}
              />
            ))}
            <NeuronCard name="output" n={c.out} inputs={outInputs} value={run(c, row[0], row[1]).out} onChange={(n) => setC({ ...c, out: n })} />
          </div>
          {solved ? (
            <div className="feedback good">
              <div className="feedback-title">✓ Working circuit.</div>
              <div>
                Parts cost (total size of all weights and biases): <strong className="mono">{k.toFixed(1)}</strong>
                {best !== undefined && best < k && <> · your best: <span className="mono">{best.toFixed(1)}</span></>}
              </div>
              <div className="muted" style={{ marginTop: 4 }}>
                {k <= P.best
                  ? 'That’s as cheap as this circuit can be built with these parts. Nicely done.'
                  : `Optional: it can be built for a cost of ${P.best.toFixed(1)}. Can you find a leaner design?`}
                {P.hidden && c.hidden.length > 2 && ' (And it only needs two hidden neurons.)'}
              </div>
            </div>
          ) : (
            <div className="btn-row">
              <span className="status neutral">{results.filter(Boolean).length} / 4 rows correct</span>
              <button className="btn small" onClick={() => setShowHint(!showHint)}>{showHint ? 'Hide hint' : 'Hint'}</button>
              <button className="btn small" onClick={() => setC(blank(P.hidden ? 2 : 0))}>Clear</button>
            </div>
          )}
          {showHint && !solved && <div className="feedback">{P.hint}</div>}
        </div>
      </div>
      <MachineLog lines={log.lines} onClear={log.clear} empty="Click a truth-table row: each neuron's tally for that case is logged here." />
    </Widget>
  );
}
