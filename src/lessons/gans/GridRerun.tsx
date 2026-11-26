// authored-by: Claude Opus 5.5
/**
 * V2-WP3: re-run the lab's whole 30-setting × 5-seed search live. On a GPU every one of the 150 duels
 * is one WGSL invocation (`ganGpu.ts`), all running at once; without one, the CPU runs them one
 * setting at a time. Either way the table compares the result with the recorded search.
 */
import { useEffect, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { BackendChip, useComputeTier } from '../../components/BackendChip';
import { getGpu } from '../../lib/compute/webgpuBackend';
import { evaluateLabSetting, type LabEvalResult, type LabSetting } from './gan';
import { evaluateSettingsGpu } from './ganGpu';
import { LAB_GRID_RESULTS, passesBrief1, passesBrief2 } from './results';

const SETTINGS: LabSetting[] = LAB_GRID_RESULTS.map((r) => ({ lrRatio: r.lrRatio, dSteps: r.dSteps, noiseDim: r.noiseDim }));

export function GridRerun() {
  const tier = useComputeTier();
  const [res, setRes] = useState<LabEvalResult[] | null>(null);
  const [progress, setProgress] = useState(0);
  const [running, setRunning] = useState(false);
  const [timing, setTiming] = useState('');
  const [ranOn, setRanOn] = useState<'webgpu' | 'cpu' | null>(null);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true; // (React's strict mode unmounts and remounts once in development)
    return () => {
      alive.current = false;
    };
  }, []);

  const run = async () => {
    setRunning(true);
    setRes(null);
    setProgress(0);
    const t0 = performance.now();
    const gpu = tier.kind === 'webgpu' ? await getGpu() : null;
    let out: LabEvalResult[];
    if (gpu) {
      out = await evaluateSettingsGpu(gpu.device, SETTINGS, (f) => alive.current && setProgress(f));
    } else {
      out = [];
      for (const s of SETTINGS) {
        out.push(evaluateLabSetting(s));
        if (!alive.current) return;
        setProgress(out.length / SETTINGS.length);
        await new Promise((r) => setTimeout(r, 0));
      }
    }
    if (!alive.current) return;
    setRes(out);
    setRanOn(gpu ? 'webgpu' : 'cpu');
    setTiming(`150 duels in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    setRunning(false);
  };

  const rows = res?.map((r, i) => ({ rec: LAB_GRID_RESULTS[i], now: r, same: passesBrief1(r) === passesBrief1(LAB_GRID_RESULTS[i]) && passesBrief2(r) === passesBrief2(LAB_GRID_RESULTS[i]) }));
  const b2 = rows?.filter((r) => passesBrief2(r.now)).length ?? 0;
  const same = rows?.filter((r) => r.same).length ?? 0;
  const close = rows?.filter((r) => Math.abs(r.now.meanW1 - r.rec.meanW1) < 0.01 && r.now.meanModes === r.rec.meanModes).length ?? 0;
  const fastest = res
    ?.map((r) => r.updatesToCover3)
    .filter((u): u is number => u !== null)
    .reduce((a, b) => Math.min(a, b), Infinity);

  let note = 'Press Run to repeat the lab’s entire search: 30 settings, 5 seeds each, 240 outer steps per duel.';
  if (running) note = `Training… ${Math.round(progress * 100)}% done.`;
  else if (rows) {
    note = `${close} of 30 settings reproduce the recorded averages (modes equal, W₁ within 0.01), and ${same} of 30 land on the same side of Briefs 1 and 2. ${b2} settings pass Brief 2 (the recorded search found 6), and the fewest updates to reach the bar is ${Number.isFinite(fastest) ? fastest : 'none'} (recorded: 40).`;
    note +=
      ranOn === 'webgpu'
        ? ' The GPU computes in 32-bit floats where the recorded search used 64-bit ones.'
        : ' This run used 64-bit floats like the recorded one, but your browser’s exp and tanh may round the last binary digit differently from the Node.js run that recorded it.';
    note +=
      ' Balanced duels shrug off differences that small; unbalanced ones, with the generator keeping pace with or outpacing the detective, amplify them until the run takes a different path. That sensitivity is the instability you were tuning against.';
  }

  return (
    <Widget
      title="Re-run the whole search, live"
      wide
      actions={<BackendChip tier={tier} timing={timing} />}
      caption={
        <>
          On a GPU, each of the 150 duels (30 settings × 5 seeds) is one shader invocation that does everything itself: draws its mini-batches, trains D and G with
          Adam, and records G’s samples every 20 steps. All 150 run at the same time. Without a GPU, the CPU runs them one setting after another. Compare the{' '}
          <em>recorded</em> columns (the search behind the debrief) with <em>this run</em>.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 8 }}>
        <button className={`btn${running ? '' : ' primary'}`} disabled={running || tier.kind === 'checking'} onClick={run}>
          {running ? `Training… ${Math.round(progress * 100)}%` : res ? '↻ Run again' : '▶ Run 150 duels'}
        </button>
      </div>
      {rows && (
        <div style={{ overflowX: 'auto', maxHeight: 420 }}>
          <table className="num-table" style={{ fontSize: 12.5 }}>
            <thead>
              <tr>
                <th>noise dim</th>
                <th>D-steps</th>
                <th>G / D rate</th>
                <th>modes, recorded</th>
                <th>modes, this run</th>
                <th>W₁, recorded</th>
                <th>W₁, this run</th>
                <th>briefs agree</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ rec, now, same: s }, i) => (
                <tr key={i}>
                  <td>{rec.noiseDim}</td>
                  <td>{rec.dSteps}</td>
                  <td>{rec.lrRatio}×</td>
                  <td>{rec.meanModes.toFixed(1)}</td>
                  <td>{now.meanModes.toFixed(1)}</td>
                  <td>{rec.meanW1.toFixed(3)}</td>
                  <td>{now.meanW1.toFixed(3)}</td>
                  <td style={{ color: s ? 'var(--c-good)' : 'var(--c-bad)' }}>{s ? '✓' : '✗'}</td>
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
