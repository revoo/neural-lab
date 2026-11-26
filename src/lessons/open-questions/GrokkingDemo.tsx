// authored-by: Claude Opus 5.5
/**
 * V2-WP3: live grokking. Four seeds of the minimal modular-addition network train side by side
 * (four replicas of one engine: on the GPU when available, else the CPU tier), and the chart draws
 * training and validation accuracy as they go.
 */
import { useEffect, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { BackendChip, useComputeTier } from '../../components/BackendChip';
import { createMlpEngine } from '../../lib/compute/selectBackend';
import type { MlpEngine } from '../../lib/compute/types';
import { firstReach, GROK, grokSpec, runGrokking, type GrokCheckpoint } from './grokking';

const W = 460;
const H = 230;
const PAD = { l: 40, r: 12, t: 12, b: 30 };

export function GrokkingDemo() {
  const tier = useComputeTier();
  const [decay, setDecay] = useState<'on' | 'off'>('on');
  const [logX, setLogX] = useState(true);
  const [hist, setHist] = useState<GrokCheckpoint[]>([]);
  const [running, setRunning] = useState(false);
  const [timing, setTiming] = useState('');
  const stop = useRef(false);
  const engine = useRef<MlpEngine | null>(null);
  const R = GROK.seeds.length;

  useEffect(() => () => {
    stop.current = true;
    engine.current?.destroy();
  }, []);
  // a new setting means a new experiment
  useEffect(() => {
    stop.current = true;
    setHist([]);
    setTiming('');
  }, [decay, tier.kind]);

  const run = async () => {
    stop.current = false;
    setRunning(true);
    setHist([]);
    engine.current?.destroy();
    const e = await createMlpEngine(grokSpec(decay === 'on' ? GROK.weightDecay : 0, R));
    engine.current = e;
    const acc: GrokCheckpoint[] = [];
    const t0 = performance.now();
    try {
      await runGrokking(e, GROK.seeds, async (c) => {
        acc.push(c);
        setHist([...acc]);
        const s = (performance.now() - t0) / 1000;
        setTiming(`${c.step.toLocaleString()} steps × ${R} seeds in ${s.toFixed(1)} s`);
        await new Promise((r) => requestAnimationFrame(() => r(null)));
        return !stop.current;
      });
    } catch {
      /* engine destroyed mid-run */
    }
    setRunning(false);
  };

  const x = (step: number) =>
    PAD.l + (W - PAD.l - PAD.r) * (logX ? Math.log10(step / GROK.every) / Math.log10(GROK.steps / GROK.every) : (step - GROK.every) / (GROK.steps - GROK.every));
  const y = (a: number) => H - PAD.b - (H - PAD.t - PAD.b) * a;
  const path = (key: 'trainAcc' | 'valAcc', r: number) => hist.map((h, i) => `${i ? 'L' : 'M'}${x(h.step).toFixed(1)},${y(h[key][r]).toFixed(1)}`).join('');
  const mean = (key: 'trainAcc' | 'valAcc') => hist.map((h, i) => `${i ? 'L' : 'M'}${x(h.step).toFixed(1)},${y(h[key].reduce((s, v) => s + v, 0) / R).toFixed(1)}`).join('');
  const ticks = logX ? [100, 300, 1000, 3000, 6000] : [100, 1500, 3000, 4500, 6000];
  const last = hist.at(-1);
  const memo = firstReach(hist, 'trainAcc', 1, R);
  const grok = firstReach(hist, 'valAcc', 0.99, R);
  const pct = (v: number) => `${(100 * v).toFixed(0)}%`;

  let note = `Press Run. Four seeds of the same network learn (a + b) mod ${GROK.p} from half of the ${GROK.p * GROK.p} possible sums; the other half is held back to test.`;
  if (last) {
    const tr = last.trainAcc.reduce((s, v) => s + v, 0) / R;
    const va = last.valAcc.reduce((s, v) => s + v, 0) / R;
    note = `Step ${last.step.toLocaleString()}: training accuracy ${pct(tr)}, validation ${pct(va)} (averages over the four seeds; guessing scores ${pct(1 / GROK.p)}).`;
    if (memo.every((m) => m !== null) && va < 0.1 && grok.every((g) => g === null)) note += ' Every training sum is memorised, yet unseen sums are still mostly wrong.';
    if (grok.every((g) => g !== null))
      note += ` All four seeds now get unseen sums right: they reached 99% at steps ${grok.map((g) => g!.toLocaleString()).join(', ')}, between ${Math.min(...grok.map((g, r) => g! / memo[r]!)).toFixed(0)} and ${Math.max(...grok.map((g, r) => g! / memo[r]!)).toFixed(0)} times later than they memorised.`;
    else if (decay === 'off' && last.step >= 3000)
      note += ' Without weight decay, none of the seeds generalises in this window. (Gromov found this model can grok with no regularisation at all, given far longer training; decay makes it happen much sooner.)';
  }

  return (
    <Widget
      title="Grokking, live: memorise first, understand later"
      wide
      actions={<BackendChip tier={tier} timing={timing} />}
      caption={
        <>
          A real network (one-hot a and b in, {GROK.hidden} hidden units with a squaring activation, {GROK.p} outputs) trained full-batch with AdamW, the minimal
          model from Gromov’s 2023 paper. Thin lines are the four seeds, thick lines their average: <span className="c-a">training accuracy</span> and{' '}
          <span className="c-b">validation accuracy</span> on sums the network never trains on. Try it with weight decay on, then off. The step axis is logarithmic by
          default, as in grokking papers; switch to linear steps to see how long the climb takes.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 8 }}>
        <Segmented options={[{ value: 'on', label: 'weight decay on (2)' }, { value: 'off', label: 'weight decay off' }]} value={decay} onChange={(v) => setDecay(v as 'on' | 'off')} />
        <button className={`btn${running ? '' : ' primary'}`} disabled={tier.kind === 'checking'} onClick={() => (running ? (stop.current = true) : run())}>
          {running ? '❚❚ Stop' : hist.length ? '↻ Run again' : '▶ Run'}
        </button>
        <Segmented options={[{ value: 'log', label: 'log steps' }, { value: 'lin', label: 'linear steps' }]} value={logX ? 'log' : 'lin'} onChange={(v) => setLogX(v === 'log')} />
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, minWidth: 300, display: 'block' }} role="img" aria-label="Training and validation accuracy against steps">
          {[0, 0.25, 0.5, 0.75, 1].map((a) => (
            <g key={a}>
              <line className="plot-grid" x1={PAD.l} x2={W - PAD.r} y1={y(a)} y2={y(a)} />
              <text className="plot-tick" x={PAD.l - 5} y={y(a) + 3.5} textAnchor="end">
                {pct(a)}
              </text>
            </g>
          ))}
          {ticks.map((t) => (
            <text key={t} className="plot-tick" x={x(t)} y={H - PAD.b + 15} textAnchor={t === GROK.steps ? 'end' : 'middle'}>
              {t.toLocaleString()}
            </text>
          ))}
          <text className="plot-tick" x={(W + PAD.l) / 2} y={H - 2} textAnchor="middle">
            training steps
          </text>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(1 / GROK.p)} y2={y(1 / GROK.p)} stroke="var(--text-faint)" strokeDasharray="2 3" />
          <text className="plot-tick" x={W - PAD.r} y={y(1 / GROK.p) - 4} textAnchor="end">
            guessing
          </text>
          {hist.length > 0 &&
            GROK.seeds.map((_, r) => (
              <g key={r} fill="none" strokeWidth={1} opacity={0.45}>
                <path d={path('trainAcc', r)} stroke="var(--c-a)" />
                <path d={path('valAcc', r)} stroke="var(--c-b)" />
              </g>
            ))}
          {hist.length > 0 && (
            <g fill="none" strokeWidth={2.6}>
              <path d={mean('trainAcc')} stroke="var(--c-a)" />
              <path d={mean('valAcc')} stroke="var(--c-b)" />
            </g>
          )}
        </svg>
      </div>
      {hist.length > 0 && (
        <div style={{ overflowX: 'auto' }}>
          <table className="num-table" style={{ fontSize: 12.5 }}>
            <thead>
              <tr>
                <th>seed</th>
                <th>memorised (100% train)</th>
                <th>generalised (99% validation)</th>
              </tr>
            </thead>
            <tbody>
              {GROK.seeds.map((s, r) => (
                <tr key={s}>
                  <td>{s}</td>
                  <td>{memo[r] === null ? '…' : `step ${memo[r]!.toLocaleString()}`}</td>
                  <td>{grok[r] === null ? (last && last.step >= GROK.steps ? `not by ${GROK.steps.toLocaleString()}` : '…') : `step ${grok[r]!.toLocaleString()}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="live-note">{note}</p>
    </Widget>
  );
}
