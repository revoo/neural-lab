// authored-by: Claude Opus 5.5
/**
 * V2-WP3: live warmup. Six seeds of a deep ReLU network train on a spiral, each twice from the same
 * start on the same mini-batches: with a constant step size and with a 300-step linear warmup to it.
 * All twelve runs are replicas of one engine (WebGPU when available, else the CPU Float32 tier).
 */
import { useEffect, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { BackendChip, useComputeTier } from '../../components/BackendChip';
import { createMlpEngine } from '../../lib/compute/selectBackend';
import type { MlpEngine } from '../../lib/compute/types';
import { died, median, runWarmup, solved, WARM, warmSpec, type WarmCheckpoint, type WarmRun } from './warmup';

const W = 460;
const H = 220;
const PAD = { l: 40, r: 12, t: 12, b: 30 };
const LMIN = 0.01;
const LMAX = 10;

export function WarmupDemo() {
  const tier = useComputeTier();
  const [peak, setPeak] = useState<number>(0.03);
  const [hist, setHist] = useState<WarmCheckpoint[]>([]);
  const [early, setEarly] = useState<number[][] | null>(null);
  const [running, setRunning] = useState(false);
  const [timing, setTiming] = useState('');
  const stop = useRef(false);
  const engine = useRef<MlpEngine | null>(null);
  const seeds = WARM.seeds;
  const S = seeds.length;

  useEffect(
    () => () => {
      stop.current = true;
      engine.current?.destroy();
    },
    [],
  );
  useEffect(() => {
    stop.current = true;
    setHist([]);
    setEarly(null);
    setTiming('');
  }, [peak, tier.kind]);

  const run = async () => {
    stop.current = false;
    setRunning(true);
    setHist([]);
    setEarly(null);
    engine.current?.destroy();
    const e = await createMlpEngine(warmSpec(2 * S));
    engine.current = e;
    const acc: WarmCheckpoint[] = [];
    const t0 = performance.now();
    try {
      await runWarmup(e, seeds, peak, async (c, r: WarmRun) => {
        acc.push(c);
        setHist([...acc]);
        setEarly(r.earlyMax.map((v) => [...v]));
        setTiming(`${c.step.toLocaleString()} steps × ${2 * S} runs in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
        await new Promise((res) => requestAnimationFrame(() => res(null)));
        return !stop.current;
      });
    } catch {
      /* engine destroyed mid-run */
    }
    setRunning(false);
  };

  const x = (step: number) => PAD.l + ((W - PAD.l - PAD.r) * step) / WARM.steps;
  const y = (l: number) => {
    const v = Math.min(LMAX, Math.max(LMIN, l));
    return PAD.t + ((H - PAD.t - PAD.b) * (Math.log10(LMAX) - Math.log10(v))) / (Math.log10(LMAX) - Math.log10(LMIN));
  };
  const path = (pts: [number, number][]) => pts.map(([s, l], i) => `${i ? 'L' : 'M'}${x(s).toFixed(1)},${y(l).toFixed(1)}`).join('');
  const seedPath = (i: number, a: number) => path(hist.map((h) => [h.step, h.worst[i][a]]));
  const medPath = (a: number) => path(hist.map((h) => [h.step, median(h.worst.map((v) => v[a]))]));
  const last = hist.at(-1);
  const done = last?.step === WARM.steps;
  const pct = (v: number) => `${(100 * v).toFixed(1)}%`;
  const ARM = ['no warmup', 'warmup'];

  let note = `Press Run. Six seeds, each trained twice from the same random start on the same mini-batches: once at a constant step size of ${peak}, once ramping up to ${peak} over the first ${WARM.warmup} steps.`;
  if (early && last) {
    const em = [0, 1].map((a) => median(early.map((v) => v[a])));
    note = `Step ${last.step.toLocaleString()}. Worst mini-batch loss so far in the first ${Math.min(last.step, WARM.warmup)} steps, median over the six seeds: ${em[0].toFixed(2)} without warmup, ${em[1].toFixed(2)} with it.`;
    if (done) {
      const sv = [0, 1].map((a) => last.acc.filter((v) => solved(v[a])).length);
      const dd = [0, 1].map((a) => last.acc.filter((v) => died(v[a])).length);
      note += ` After ${WARM.steps.toLocaleString()} steps, ${sv[0]} of 6 runs without warmup and ${sv[1]} of 6 with it classify at least 99% of the spiral; ${dd[0]} and ${dd[1]} are stuck near guessing. Six seeds are too few to call that difference: the 20-seed results are in the text.`;
    }
  }

  return (
    <Widget
      title="Warmup, live: does a gentle start change the ending?"
      wide
      actions={<BackendChip tier={tier} timing={timing} />}
      caption={
        <>
          A real network: 2 inputs, eight hidden ReLU layers of 64 units (He initialisation), one sigmoid output, Adam, mini-batches of {WARM.batch}, trained on a
          400-point two-arm spiral. Thin lines are single seeds, thick lines the median over six: <span className="c-a">constant step size</span> and{' '}
          <span className="c-b">{WARM.warmup}-step linear warmup</span> to the same peak. Each point is the worst mini-batch loss in a window of 20 steps, on a logarithmic axis. Try each peak step size; the bigger it is, the rougher the start.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 8 }}>
        <Segmented options={WARM.peaks.map((p) => ({ value: String(p), label: `peak ${p}` }))} value={String(peak)} onChange={(v) => setPeak(Number(v))} />
        <button className={`btn${running ? '' : ' primary'}`} disabled={tier.kind === 'checking'} onClick={() => (running ? (stop.current = true) : run())}>
          {running ? '❚❚ Stop' : hist.length ? '↻ Run again' : '▶ Run'}
        </button>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, minWidth: 300, display: 'block' }} role="img" aria-label="Worst mini-batch loss against steps, with and without warmup">
          <rect x={x(0)} y={PAD.t} width={x(WARM.warmup) - x(0)} height={H - PAD.t - PAD.b} fill="var(--c-b)" opacity={0.07} />
          <text className="plot-tick" x={x(WARM.warmup) - 4} y={PAD.t + 11} textAnchor="end">
            warmup phase
          </text>
          {[0.01, 0.1, 1, 10].map((l) => (
            <g key={l}>
              <line className="plot-grid" x1={PAD.l} x2={W - PAD.r} y1={y(l)} y2={y(l)} />
              <text className="plot-tick" x={PAD.l - 5} y={y(l) + 3.5} textAnchor="end">
                {l}
              </text>
            </g>
          ))}
          {[0, 250, 500, 750, 1000].map((t) => (
            <text key={t} className="plot-tick" x={x(t)} y={H - PAD.b + 15} textAnchor={t === 1000 ? 'end' : 'middle'}>
              {t.toLocaleString()}
            </text>
          ))}
          <text className="plot-tick" x={(W + PAD.l) / 2} y={H - 2} textAnchor="middle">
            training steps
          </text>
          {hist.length > 0 &&
            seeds.map((s, i) => (
              <g key={s} fill="none" strokeWidth={1} opacity={0.35}>
                <path d={seedPath(i, 0)} stroke="var(--c-a)" />
                <path d={seedPath(i, 1)} stroke="var(--c-b)" />
              </g>
            ))}
          {hist.length > 0 && (
            <g fill="none" strokeWidth={2.6}>
              <path d={medPath(0)} stroke="var(--c-a)" />
              <path d={medPath(1)} stroke="var(--c-b)" />
            </g>
          )}
        </svg>
      </div>
      {early && last && (
        <div style={{ overflowX: 'auto' }}>
          <table className="num-table" style={{ fontSize: 12.5 }}>
            <thead>
              <tr>
                <th>seed</th>
                <th>worst early loss, {ARM[0]}</th>
                <th>worst early loss, {ARM[1]}</th>
                <th>accuracy, {ARM[0]}</th>
                <th>accuracy, {ARM[1]}</th>
              </tr>
            </thead>
            <tbody>
              {seeds.map((s, i) => (
                <tr key={s}>
                  <td>{s}</td>
                  <td>{early[i][0].toFixed(2)}</td>
                  <td>{early[i][1].toFixed(2)}</td>
                  <td>{pct(last.acc[i][0])}</td>
                  <td>{pct(last.acc[i][1])}</td>
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
