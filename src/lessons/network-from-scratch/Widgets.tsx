import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { LogChart } from '../../components/LogChart';
import {
  OPTIONS,
  PARTNER,
  SLOTS,
  X_XOR,
  Y_XOR,
  accuracy,
  add,
  backward,
  fmtShape,
  forward,
  gradCheck,
  init,
  map,
  matmul,
  mul,
  scale,
  sigmoid,
  stepsToSolve,
  sub,
  sum0,
  T,
  train,
  type Choice,
  type Env,
  type Mat,
  type Params,
  type Slot,
} from './net';

// ------------------------------------------------------------------ matrix display

function MatView({ name, M, note, hl }: { name: string; M: Mat; note?: string; hl?: boolean }) {
  const max = Math.max(1e-9, ...M.flat().map(Math.abs));
  return (
    <div className="nfs-mat" style={hl ? { outline: '2px solid var(--accent)', outlineOffset: 2 } : undefined}>
      <div className="nfs-mat-head">
        <span className="mono" style={{ fontWeight: 700 }}>{name}</span> <span className="faint mono">{fmtShape(M)}</span>
        {note && <span className="faint"> {note}</span>}
      </div>
      <table className="num-table">
        <tbody>
          {M.map((r, i) => (
            <tr key={i}>
              {r.map((v, j) => (
                <td key={j} style={{ background: `color-mix(in srgb, ${v >= 0 ? 'var(--c-a)' : 'var(--c-b)'} ${Math.round((Math.abs(v) / max) * 28)}%, transparent)` }}>
                  {Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ------------------------------------------------------------------ 1. the code stepper

type Line = { code: string; shape?: string; section?: string; run: (e: Env, lr: number) => void; shows: string[] };
const LR = 1;
const LINES: Line[] = [
  { section: '# forward pass', code: 'z1 = X @ W1 + b1', shape: '(4,2) @ (2,3) + (1,3) → (4,3)', run: (e) => (e.z1 = add(matmul(e.X, e.W1), e.b1)), shows: ['X', 'W1', 'b1', 'z1'] },
  { code: 'h = tanh(z1)', shape: '(4,3)', run: (e) => (e.h = map(e.z1!, Math.tanh)), shows: ['z1', 'h'] },
  { code: 'z2 = h @ W2 + b2', shape: '(4,3) @ (3,1) + (1,1) → (4,1)', run: (e) => (e.z2 = add(matmul(e.h!, e.W2), e.b2)), shows: ['h', 'W2', 'b2', 'z2'] },
  { code: 'p = sigmoid(z2)', shape: '(4,1)', run: (e) => (e.p = map(e.z2!, sigmoid)), shows: ['z2', 'p', 'y'] },
  {
    code: 'loss = -mean(y*log(p) + (1-y)*log(1-p))',
    shape: 'one number',
    run: (e) => (e.loss = -e.p!.reduce((s, r, i) => s + (e.y[i][0] ? Math.log(r[0]) : Math.log(1 - r[0])), 0) / e.n),
    shows: ['p', 'y'],
  },
  { section: '# backward pass', code: 'dz2 = (p - y) / n', shape: '(4,1)', run: (e) => (e.dz2 = scale(sub(e.p!, e.y), 1 / e.n)), shows: ['p', 'y', 'dz2'] },
  { code: 'dW2 = h.T @ dz2', shape: '(3,4) @ (4,1) → (3,1)', run: (e) => (e.dW2 = matmul(T(e.h!), e.dz2!)), shows: ['h', 'dz2', 'dW2'] },
  { code: 'db2 = dz2.sum(axis=0)', shape: '(1,1)', run: (e) => (e.db2 = sum0(e.dz2!)), shows: ['dz2', 'db2'] },
  { code: 'dh = dz2 @ W2.T', shape: '(4,1) @ (1,3) → (4,3)', run: (e) => (e.dh = matmul(e.dz2!, T(e.W2))), shows: ['dz2', 'W2', 'dh'] },
  { code: 'dz1 = dh * (1 - h**2)', shape: '(4,3)', run: (e) => (e.dz1 = mul(e.dh!, map(e.h!, (v) => 1 - v * v))), shows: ['dh', 'h', 'dz1'] },
  { code: 'dW1 = X.T @ dz1', shape: '(2,4) @ (4,3) → (2,3)', run: (e) => (e.dW1 = matmul(T(e.X), e.dz1!)), shows: ['X', 'dz1', 'dW1'] },
  { code: 'db1 = dz1.sum(axis=0)', shape: '(1,3)', run: (e) => (e.db1 = sum0(e.dz1!)), shows: ['dz1', 'db1'] },
  {
    section: '# update (plain gradient descent)',
    code: 'W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2',
    shape: 'every parameter keeps its shape',
    run: (e, lr) => {
      e.W1 = sub(e.W1, scale(e.dW1!, lr));
      e.b1 = sub(e.b1, scale(e.db1!, lr));
      e.W2 = sub(e.W2, scale(e.dW2!, lr));
      e.b2 = sub(e.b2, scale(e.db2!, lr));
    },
    shows: ['W1', 'b1', 'W2', 'b2'],
  },
];

const NOTES: Record<string, string> = {
  X: 'four examples, one per row',
  y: 'the right answers',
  W1: 'input → hidden weights',
  b1: 'hidden biases',
  z1: 'hidden tallies',
  h: 'hidden outputs',
  W2: 'hidden → output weights',
  b2: 'output bias',
  z2: 'output tally',
  p: 'predicted probability of 1',
  dz2: 'how the loss responds to each output tally',
  dW2: 'slope for each W2',
  db2: 'slope for b2',
  dh: 'blame passed back to each hidden output',
  dz1: 'blame on each hidden tally',
  dW1: 'slope for each W1',
  db1: 'slope for each b1',
};

function fullStep(P: Params): Params {
  const e: Env = { ...P, X: X_XOR, y: Y_XOR, n: 4 };
  for (const l of LINES) l.run(e, LR);
  return { W1: e.W1, b1: e.b1, W2: e.W2, b2: e.b2 };
}

export function CodeStepper() {
  const [P, setP] = useState<Params>(() => init(3, 1));
  const [line, setLine] = useState(-1);
  const [steps, setSteps] = useState(0);
  const [losses, setLosses] = useState<number[]>([]);

  // replay lines 0..line from the current parameters (cheap, and always consistent)
  const env = useMemo(() => {
    const e: Env = { ...P, X: X_XOR, y: Y_XOR, n: 4 };
    for (let i = 0; i <= line; i++) LINES[i].run(e, LR);
    return e;
  }, [P, line]);

  const next = () => {
    if (line < LINES.length - 1) setLine(line + 1);
    else {
      // the update line has run: these are the new parameters; start the next step
      setLosses([...losses, forward(P, X_XOR, Y_XOR).loss!]);
      setP({ W1: env.W1, b1: env.b1, W2: env.W2, b2: env.b2 });
      setSteps(steps + 1);
      setLine(-1);
    }
  };
  const runMany = (k: number) => {
    let Q = P; // always from the start of the current step (a half-finished step is discarded)
    const L = [...losses];
    for (let i = 0; i < k; i++) {
      L.push(forward(Q, X_XOR, Y_XOR).loss!);
      Q = fullStep(Q);
    }
    setLosses(L);
    setP(Q);
    setSteps(steps + k);
    setLine(-1);
  };
  const reset = () => {
    setP(init(3, 1));
    setLine(-1);
    setSteps(0);
    setLosses([]);
  };

  const cur = line >= 0 ? LINES[line] : null;
  const shown = cur ? cur.shows : ['X', 'y', 'W1', 'b1', 'W2', 'b2'];
  const val = (k: string): Mat | null => {
    if (k === 'y') return env.y;
    const v = (env as unknown as Record<string, Mat | number | undefined>)[k];
    return Array.isArray(v) ? v : null;
  };
  const f = forward(P, X_XOR, Y_XOR);
  const produced = cur ? cur.shows[cur.shows.length - 1] : null;

  let note: ReactNode;
  if (!cur)
    note = (
      <>
        Training step {steps + 1} is about to start. Current loss {f.loss!.toFixed(4)}; the network gets {accuracy(P) * 4} of the 4 XOR cases right. Press{' '}
        <strong>Next line</strong>.
      </>
    );
  else if (line === 4) note = <>The loss: {env.loss!.toFixed(4)}. A network that knew nothing would output 0.5 everywhere and score ln 2 ≈ 0.693.</>;
  else if (line === LINES.length - 1) note = <>Every parameter has taken one step downhill. Press Next line to start the next training step with them.</>;
  else note = <>{cur.code.split('=')[0].trim()} = {NOTES[produced!] ?? ''}. Its shape: {cur.shape}.</>;

  return (
    <Widget
      title="The whole network, one line at a time"
      wide
      caption={
        <>
          A 2 → 3 → 1 network learning XOR, all four examples at once. Every line of the code on the left really runs, and the matrices it reads and writes appear on
          the right with their shapes (blue positive, orange negative). Step through one full training step line by line; then use <strong>Run 100 steps</strong> and
          watch the loss fall.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10 }}>
        <button className="btn primary" onClick={next}>
          Next line ▸
        </button>
        <button className="btn" onClick={() => setLine(Math.max(-1, line - 1))} disabled={line < 0}>
          ◂ Back
        </button>
        <button className="btn" onClick={() => runMany(1)}>
          Run 1 step
        </button>
        <button className="btn" onClick={() => runMany(100)}>
          Run 100 steps
        </button>
        <button className="btn" onClick={reset}>
          Reset
        </button>
        <span className="muted mono" style={{ fontSize: 13 }}>
          step {steps} · loss {f.loss!.toFixed(4)} · {accuracy(P) * 4}/4 right
        </span>
      </div>
      <div className="widget-row">
        <div className="widget-col" style={{ flex: '1 1 400px' }}>
          <div className="nfs-code">
            {LINES.map((l, i) => (
              <div key={i}>
                {l.section && <div className="hljs-comment">{l.section}</div>}
                <div className={i === line ? 'nfs-line on' : 'nfs-line'}>{l.code}</div>
              </div>
            ))}
          </div>
          {losses.length > 1 && (
            <LogChart
              series={[{ points: losses.map((v, i) => [i + 1, v] as [number, number]), color: 'var(--accent)' }]}
              xDomain={[1, Math.max(2, losses.length)]}
              yDomain={[0.001, 1]}
              xLabel="training steps"
              xTicks={[1, losses.length]}
              hlines={[{ y: Math.LN2, label: 'ln 2', color: 'var(--text-faint)' }]}
              width={400}
              height={150}
              ariaLabel="Loss over training steps"
            />
          )}
        </div>
        <div className="widget-col" style={{ flex: '1 1 380px' }}>
          <div className="nfs-mats">
            {shown.map((k) => {
              const M = val(k);
              return M ? <MatView key={k} name={k} M={M} note={NOTES[k]} hl={k === produced && line >= 0 && line !== LINES.length - 1} /> : null;
            })}
          </div>
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 12 }}>
        {note}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. the backward-pass lab

const STORE = 'nl-backward-lab-v1';
const loadFound = (): string[] => {
  try {
    return JSON.parse(localStorage.getItem(STORE) ?? '[]');
  } catch {
    return [];
  }
};
const HIDDEN = [1, 2, 3, 4, 5, 6, 7, 8];
const LRS = [0.1, 0.3, 1, 3, 10, 30];

/**
 * The fastest any (hidden, step size) in the menus solves XOR (loss < 0.05, random start 1).
 * Recorded here so the page needn't recompute it; net.test.ts re-derives it by trying all 48.
 */
export const GOLF_PAR = { hidden: 3, lr: 3, steps: 32 };
export { HIDDEN, LRS };

/** Try every (hidden, step size) in the menus and return the fastest. */
export function golfPar() {
  let best: { hidden: number; lr: number; steps: number } | null = null;
  for (const hidden of HIDDEN)
    for (const lr of LRS) {
      const s = stepsToSolve(hidden, lr);
      if (s !== null && (!best || s < best.steps)) best = { hidden, lr, steps: s };
    }
  return best!;
}

const FIXED_LINES: Partial<Record<Slot, string>> = { dz2: 'db2 = dz2.sum(axis=0)', dz1: 'db1 = dz1.sum(axis=0)' };

export function BackwardLab() {
  const [choice, setChoice] = useState<Choice>({ dz2: 3, dW2: 2, dh: 2, dz1: 3, dW1: 2 });
  const [found, setFound] = useState<string[]>(loadFound);
  const P = useMemo(() => init(3, 1), []);
  const check = useMemo(() => gradCheck(P, choice), [P, choice]);
  const run = useMemo(() => backward(P, choice), [P, choice]);
  const [trained, setTrained] = useState<{ losses: number[]; acc: number; error: string | null } | null>(null);
  const [golf, setGolf] = useState<{ hidden: number; lr: number }>({ hidden: 3, lr: 1 });
  const [golfResult, setGolfResult] = useState<number | null | undefined>(undefined);
  const par = GOLF_PAR;

  const set = (s: Slot, i: number) => {
    setChoice({ ...choice, [s]: i });
    setTrained(null);
  };
  const key = SLOTS.map((s) => choice[s]).join('');
  const pass = check.ok && check.pass;
  useEffect(() => {
    if (!pass || found.includes(key)) return;
    const next = [...found, key];
    setFound(next);
    try {
      localStorage.setItem(STORE, JSON.stringify(next));
    } catch {
      /* ignore */
    }
  }, [pass, key]); // eslint-disable-line react-hooks/exhaustive-deps
  const failedAt = !run.ok ? run.slot : null;

  return (
    <Widget
      title="Write the backward pass"
      wide
      caption={
        <>
          The forward pass is written; five lines of the backward pass are missing. Pick an expression for each blank. The code runs as soon as you choose: a line whose
          shapes don't fit stops with an error, as NumPy would, and a <strong>gradient check</strong> compares your gradients with ones measured by nudging each weight.
          Several different choices are correct.
        </>
      }
    >
      <div className="nfs-code">
        <div className="hljs-comment"># forward (given)</div>
        <div>z1 = X @ W1 + b1;  h = tanh(z1)</div>
        <div>z2 = h @ W2 + b2;  p = sigmoid(z2)</div>
        <div className="hljs-comment"># backward (yours)</div>
        {SLOTS.map((s) => (
          <div key={s}>
            <div className={failedAt === s ? 'nfs-line bad' : 'nfs-line'}>
              {s.padEnd(4)} ={' '}
              <select className="nfs-pick" value={choice[s]} onChange={(e) => set(s, Number(e.target.value))} aria-label={`expression for ${s}`}>
                {OPTIONS[s].map((o, i) => (
                  <option key={i} value={i}>
                    {o.code}
                  </option>
                ))}
              </select>
              <span className="faint">{`   # ${PARTNER[s]}'s shape`}</span>
            </div>
            {FIXED_LINES[s] && <div className="faint">{FIXED_LINES[s]}</div>}
          </div>
        ))}
      </div>
      <div className="widget-row" style={{ marginTop: 10 }}>
        <div className="widget-col">
          {!check.ok ? (
            <div className="feedback bad">
              <div className="feedback-title">Error on the line for {check.slot}</div>
              <p className="mono" style={{ fontSize: 13 }}>
                {check.error}
              </p>
              <p>Shapes are the first test: work out what shape each side has, and which arrangement could possibly produce the shape you need.</p>
            </div>
          ) : (
            <div className={`feedback ${check.pass ? 'good' : 'bad'}`}>
              <div className="feedback-title">{check.pass ? '✓ Gradient check passed' : 'Shapes fit, but the gradient check fails'}</div>
              <table className="num-table" style={{ margin: '6px 0' }}>
                <thead>
                  <tr>
                    <th>parameter</th>
                    <th>relative error</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {(['W2', 'b2', 'W1', 'b1'] as const).map((k) => (
                    <tr key={k}>
                      <td style={{ textAlign: 'left' }}>{k}</td>
                      <td>{check.errs[k].toExponential(1)}</td>
                      <td>{check.errs[k] < 1e-5 ? '✓' : '✗'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p>
                {check.pass
                  ? 'Your backward pass agrees with the measured slopes to within a hundred-thousandth. It is correct.'
                  : check.errs.W2 < 1e-5
                    ? 'The output layer is right (W2 and b2 pass), so the mistake is further back: in dh, dz1 or dW1.'
                    : 'Even the output layer’s gradients are off, so look at the first lines: dz2 and dW2. Everything after them inherits the mistake.'}
              </p>
            </div>
          )}
        </div>
        <div className="widget-col">
          <div className="btn-row">
            <button
              className="btn primary"
              disabled={!check.ok}
              onClick={() => {
                const r = train(choice, 3, 1, 300);
                setTrained({ losses: r.losses, acc: r.error ? 0 : accuracy(r.P), error: r.error });
              }}
            >
              ▶ Train 300 steps with this code
            </button>
          </div>
          {trained && (
            <>
              <LogChart
                series={[{ points: trained.losses.map((v, i) => [i + 1, Math.max(v, 1e-4)] as [number, number]), color: 'var(--accent)' }]}
                xDomain={[1, 300]}
                yDomain={[0.001, 10]}
                xLabel="steps"
                xTicks={[1, 300]}
                hlines={[{ y: Math.LN2, label: 'ln 2', color: 'var(--text-faint)' }]}
                width={360}
                height={140}
                ariaLabel="Loss while training with your backward pass"
              />
              <div className="muted" style={{ fontSize: 13 }}>
                Final loss {trained.losses[trained.losses.length - 1]?.toFixed(4)}; {trained.acc * 4} of 4 XOR cases right.{' '}
                {!pass && trained.acc === 1
                  ? 'It learned XOR anyway! A wrong gradient can still point roughly downhill — which is exactly why you check gradients numerically instead of trusting that training “worked”.'
                  : ''}
              </div>
            </>
          )}
          <div style={{ marginTop: 10, fontSize: 13.5 }}>
            <strong>Correct versions found: {found.length}</strong>
            <span className="muted"> — different ways of writing a backward pass that passes the check.</span>
          </div>
        </div>
      </div>
      {found.length > 0 && (
        <div className="nfs-golf">
          <h4 style={{ margin: '0 0 6px' }}>Optional: train it fast</h4>
          <p className="muted" style={{ margin: '0 0 8px', fontSize: 13.5 }}>
            With a correct backward pass, plain gradient descent and the same random start: how few steps can you take to get the XOR loss below 0.05? Choose the hidden
            layer's size and the step size.
          </p>
          <div className="controls inline">
            <Segmented label="hidden neurons" value={golf.hidden} onChange={(v) => { setGolf({ ...golf, hidden: v }); setGolfResult(undefined); }} options={HIDDEN} />
            <Segmented label="step size" value={golf.lr} onChange={(v) => { setGolf({ ...golf, lr: v }); setGolfResult(undefined); }} options={LRS} />
            <button className="btn" onClick={() => setGolfResult(stepsToSolve(golf.hidden, golf.lr))}>
              Run
            </button>
          </div>
          {golfResult !== undefined && (
            <div className="live-note" style={{ marginTop: 8 }}>
              {golfResult === null ? (
                <>Never got below 0.05 within 5,000 steps (it stalled or blew up). Too big a step, or too few neurons?</>
              ) : (
                <>
                  Solved in <strong>{golfResult} steps</strong>.{' '}
                  {golfResult <= par.steps
                    ? `That's the fastest possible with these menus: we tried all ${HIDDEN.length * LRS.length} combinations, and ${par.steps} steps is the best any of them achieves.`
                    : `The fastest combination in these menus takes ${par.steps} steps (found by trying all ${HIDDEN.length * LRS.length}).`}
                </>
              )}
            </div>
          )}
        </div>
      )}
    </Widget>
  );
}
