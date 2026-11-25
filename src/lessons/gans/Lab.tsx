// authored-by: Gemini Antigravity
import { useMemo, useState } from 'react';
import { LabShell, useLabProgress, type LabBrief } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { LiveCode } from '../../components/LabJob';
import { Segmented } from '../../components/controls';
import { GanDensityPlot } from './Widgets';
import { buildGanCode } from './realCode';
import {
  GAN_LAB_BRIEFS,
  GanRunner,
  LAB_BASE_LR_D,
  LAB_D_STEPS,
  LAB_LR_RATIOS,
  LAB_MAX_STEPS,
  LAB_NOISE_DIMS,
  evaluateLabSetting,
  ganLabHint,
  type LabEvalResult,
  type LabSetting,
} from './gan';
import { LAB_GRID_RESULTS, passesBrief1, passesBrief2, passesBrief3 } from './results';

const INIT_SETTING: LabSetting = { lrRatio: 4, dSteps: 1, noiseDim: 1 };

export function DuelBalanceLab() {
  const progress = useLabProgress('gan-duel-lab-v1');
  const log = useMachineLog();
  const [active, setActive] = useState(0);
  const [setting, setSetting] = useState<LabSetting>(INIT_SETTING);
  const [lastRun, setLastRun] = useState<{ setting: LabSetting; res: LabEvalResult } | null>(null);

  // Instant lookup from committed 30-config table for the currently selected dials (or live run result)
  const recorded = useMemo(
    () =>
      LAB_GRID_RESULTS.find(
        (r) => r.lrRatio === setting.lrRatio && r.dSteps === setting.dSteps && r.noiseDim === setting.noiseDim,
      )!,
    [setting],
  );

  // Live Seed-1 preview for the currently tested configuration (runs fast: 1 seed × 240 steps ≈ 35ms)
  const previewSnap = useMemo(() => {
    const r = new GanRunner({
      seed: 1,
      noiseDim: setting.noiseDim,
      lrD: LAB_BASE_LR_D,
      lrG: LAB_BASE_LR_D * setting.lrRatio,
      dSteps: setting.dSteps,
      batchSize: 36,
      objective: 'non-saturating',
    });
    r.step(LAB_MAX_STEPS);
    return r.snapshot();
  }, [setting]);

  const currentIsTested =
    lastRun !== null &&
    lastRun.setting.lrRatio === setting.lrRatio &&
    lastRun.setting.dSteps === setting.dSteps &&
    lastRun.setting.noiseDim === setting.noiseDim;

  const evalMetrics = currentIsTested ? lastRun!.res : null;

  const runFiveSeeds = () => {
    const res = evaluateLabSetting(setting);
    setLastRun({ setting: { ...setting }, res });
    const pass3 = passesBrief3(res);
    const pass2 = passesBrief2(res);
    log.push(
      `z=${setting.noiseDim}, D-steps=${setting.dSteps}, lrG/lrD=${setting.lrRatio}× → ${res.meanModes.toFixed(1)}/3 modes (${res.allThreeSeeds}/5 seeds all-3), W₁=${res.meanW1.toFixed(3)}, upd-to-3=${res.updatesToCover3 ?? '—'}`,
      pass3 || pass2 ? 'good' : undefined,
    );
  };

  const briefId = GAN_LAB_BRIEFS[active].id;
  const briefs: LabBrief[] = [
    {
      id: 'cover2',
      tab: '1 · Cover 2+ modes',
      title: 'Brief 1 — Break out of single-mode collapse',
      text: 'Choose an LR ratio (lrG / lrD), discriminator steps per generator step, and noise dimension, then press “Run 5 seeds”. Reach an average of at least 2.0 / 3 modes covered and W₁ ≤ 0.80 across the five fixed seeds.',
    },
    {
      id: 'cover3',
      tab: '2 · Cover all 3',
      title: 'Brief 2 — Reliably cover all three modes',
      text: 'Hold the duel balanced across all five seeds: reach an average of at least 2.8 / 3 covered modes and distribution error W₁ ≤ 0.45 by step 240.',
    },
    {
      id: 'fast',
      tab: '3 · Fewest updates',
      title: 'Brief 3 — Cover all three modes in the fewest total updates',
      text: (
        <>
          Every outer step costs <strong>D-steps + 1</strong> network updates. Hit the Brief 2 quality bar in as few total updates as possible (best found by searching the 30-setting grid: 40 updates).
        </>
      ),
    },
  ];

  const passing =
    evalMetrics !== null &&
    (briefId === 'cover2'
      ? passesBrief1(evalMetrics)
      : briefId === 'cover3'
        ? passesBrief2(evalMetrics)
        : passesBrief3(evalMetrics));

  const optimal = evalMetrics !== null && passesBrief3(evalMetrics);

  const statusText = (() => {
    if (!evalMetrics) {
      return 'Adjust the three controls above, inspect the Seed 1 preview on the left, then press “Run 5 seeds” to score your configuration fairly across all five seeds.';
    }
    const base = `5-seed average: ${evalMetrics.meanModes.toFixed(1)} / 3 modes (${evalMetrics.allThreeSeeds}/5 seeds hit all 3), W₁ = ${evalMetrics.meanW1.toFixed(3)}, updates to reach 3-mode bar: ${evalMetrics.updatesToCover3 ?? 'not reached'}.`;
    if (passing) return `✓ Passed! ${base}`;
    return `Not passing yet. ${base}`;
  })();

  const hintText = ganLabHint(briefId, {
    meanModes: evalMetrics ? evalMetrics.meanModes : recorded.meanModes,
    meanW1: evalMetrics ? evalMetrics.meanW1 : recorded.meanW1,
    updatesToCover3: evalMetrics ? evalMetrics.updatesToCover3 : recorded.updatesToCover3,
    meanOffMode: evalMetrics ? evalMetrics.meanOffMode : recorded.meanOffMode,
  });

  const b1Count = LAB_GRID_RESULTS.filter(passesBrief1).length;
  const b2Rows = LAB_GRID_RESULTS.filter(passesBrief2);

  const revealContent = (
    <div>
      <p style={{ margin: '0 0 8px' }}>
        <strong>Best found by searching all 30 settings</strong> (5 LR ratios × 3 D-steps × 2 noise dims, averaged over 5 fixed seeds):{' '}
        {b1Count}/30 pass Brief 1, <strong>{b2Rows.length}/30</strong> pass Brief 2, and the fastest to reach the 3-mode bar is{' '}
        <span className="mono">lrG/lrD = 0.5×, D-steps = 1, noiseDim = 2</span> in <strong>40 updates</strong> (3.0/3 modes, 5/5 seeds, W₁ = 0.396). Lowest final W₁ is{' '}
        <span className="mono">lrG/lrD = 0.25×, D-steps = 2, noiseDim = 2</span> (W₁ = 0.272, 120 updates).
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table className="num-table">
          <thead>
            <tr>
              <th>Noise</th>
              <th>D steps</th>
              <th>LR ratio</th>
              <th>Modes</th>
              <th>5-seed W1</th>
              <th>Upd to 3</th>
            </tr>
          </thead>
          <tbody>
            {b2Rows.map((r) => (
              <tr
                key={`${r.noiseDim}-${r.dSteps}-${r.lrRatio}`}
                style={passesBrief3(r) ? { fontWeight: 700, background: 'var(--accent-soft)' } : undefined}
              >
                <td className="mono">{r.noiseDim}D</td>
                <td className="mono">{r.dSteps}</td>
                <td className="mono">{r.lrRatio}×</td>
                <td className="mono">{r.meanModes.toFixed(1)} ({r.allThreeSeeds}/5)</td>
                <td className="mono">{r.meanW1.toFixed(3)}</td>
                <td className="mono">{r.updatesToCover3 ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  return (
    <LabShell
      title="Lab: Keep the Duel Balanced"
      caption="Tune the generator-to-discriminator learning-rate ratio, discriminator steps per generator step, and noise dimension. Every configuration is scored fairly across five fixed seeds."
      briefs={briefs}
      active={active}
      onSelect={setActive}
      progress={progress}
      passing={passing}
      optimal={optimal}
      status={statusText}
      hint={hintText}
      reveal={{ label: 'the 30-setting 5-seed grid search', content: revealContent }}
      log={{ lines: log.lines, onClear: log.clear, empty: 'Press “Run 5 seeds” to log a configuration.' }}
      actions={
        <button
          type="button"
          className="btn"
          onClick={() => {
            setSetting(INIT_SETTING);
            setLastRun(null);
          }}
        >
          Reset
        </button>
      }
    >
      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 340px', maxWidth: 460 }}>
          <GanDensityPlot snap={previewSnap} width={440} height={240} />
          <div style={{ fontSize: 12.5, color: 'var(--text-muted)', marginTop: 4 }}>
            Seed 1 preview at step 240: {previewSnap.metrics.modesCovered}/3 modes, W₁ = {previewSnap.metrics.w1.toFixed(3)}
          </div>

          {evalMetrics && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
              {evalMetrics.perSeed.map((ps) => (
                <span
                  key={ps.seed}
                  className="mono"
                  style={{
                    fontSize: 12,
                    padding: '3px 7px',
                    borderRadius: 6,
                    border: '1px solid var(--border)',
                    background:
                      ps.modes === 3
                        ? 'color-mix(in srgb, var(--c-good) 14%, var(--bg-sunken))'
                        : 'var(--bg-sunken)',
                  }}
                >
                  S{ps.seed}: {ps.modes}/3 · W₁ {ps.w1.toFixed(2)}
                </span>
              ))}
            </div>
          )}
        </div>

        <div style={{ flex: '1 1 250px', display: 'grid', gap: 12 }}>
          <Segmented
            label="LR ratio (lrG / lrD, with lrD = 0.015)"
            value={setting.lrRatio}
            onChange={(v) => setSetting((s) => ({ ...s, lrRatio: v }))}
            options={LAB_LR_RATIOS.map((r) => ({ value: r, label: `${r}×` }))}
          />
          <Segmented
            label="Discriminator steps per G step"
            value={setting.dSteps}
            onChange={(v) => setSetting((s) => ({ ...s, dSteps: v }))}
            options={LAB_D_STEPS.map((d) => ({ value: d, label: String(d) }))}
          />
          <Segmented
            label="Noise dimension (z)"
            value={setting.noiseDim}
            onChange={(v) => setSetting((s) => ({ ...s, noiseDim: v }))}
            options={LAB_NOISE_DIMS.map((n) => ({ value: n, label: `${n}D` }))}
          />

          <div className="btn-row">
            <button type="button" className="btn primary" onClick={runFiveSeeds}>
              Run 5 seeds (240 steps)
            </button>
          </div>

          <LiveCode note="Equivalent PyTorch loop" code={buildGanCode(setting)} />
        </div>
      </div>
    </LabShell>
  );
}
