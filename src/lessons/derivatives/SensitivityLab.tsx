// authored-by: Codex Meridian 6
import { useState } from 'react';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { Label, Plot, Seg } from '../../components/plot';
import { accepts, experiments, secant, survey, type Probe } from './sensitivityLab';

const key = 'neural-lab:sensitivity-settings';
const briefs = experiments.map((experiment, i) => ({
  id: String(i),
  tab: experiment.title,
  text:
    'Estimate the slope ' +
    (i === 2 ? 'on each side ' : '') +
    'at x = ' +
    experiment.at +
    ' within ±' +
    experiment.tolerance +
    '. Probe budget: ' +
    experiment.budget +
    ' per run.',
}));
const inputStyle = {
  maxWidth: '100%',
  border: '1px solid var(--border-strong)',
  padding: 6,
  background: 'var(--bg)',
  color: 'var(--text)',
};
type Settings = { x: string; probes: Probe[]; pairs: [string, string][] };

/** Probe pairs in the order survey() returns them. */
const METHOD_NAMES = ['forward: x and x + h', 'centred: x − h and x + h', 'backward: x − h and x'];

function ErrorPlot({ index }: { index: number }) {
  const rows = survey(index);
  const floor = (error: number) => Math.log10(Math.max(error, 1e-12));
  const methods = index === 2 ? [0, 2, 1] : [0, 1];
  const colours = ['var(--c-a)', 'var(--c-b)', 'var(--c-c)'];
  // Different widths, widest drawn first, so lines that coincide stay visible.
  const widths = [4.5, 1.5, 2.5];
  const evens = (lo: number, hi: number) => Array.from({ length: Math.floor((hi - lo) / 2) + 1 }, (_, i) => lo + 2 * i);
  const top = Math.ceil(Math.max(1, ...rows.flatMap((row) => row.methods.map((method) => floor(method.error)))));
  return (
    <>
      <p style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 16px', alignItems: 'center' }} aria-label="Legend">
        {[0, 1, 2]
          .filter((method) => methods.includes(method))
          .map((method) => (
            <span key={method} style={{ whiteSpace: 'nowrap' }}>
              <span aria-hidden="true" style={{ display: 'inline-block', width: 22, height: widths[method], background: colours[method], verticalAlign: 'middle', marginRight: 6 }} />
              {METHOD_NAMES[method]}
            </span>
          ))}
        <span style={{ whiteSpace: 'nowrap' }}>
          <span aria-hidden="true" style={{ display: 'inline-block', width: 22, borderTop: '2px dashed var(--text-faint)', verticalAlign: 'middle', marginRight: 6 }} />
          allowed error
        </span>
      </p>
      <p>Each line shows how far one kind of probe pair lands from the true slope, as the spacing shrinks. Exact or tiny errors sit at the bottom of the chart.</p>
      <Plot
        x={[-10, -1]}
        y={[-13, top]}
        axes={false}
        pad={44}
        ariaLabel="Error versus step on logarithmic axes"
        style={{ display: 'block', maxWidth: '100%', height: 'auto' }}
      >
        <Seg from={[-10, -13]} to={[-1, -13]} color="var(--c-axis)" />
        <Seg from={[-10, -13]} to={[-10, top]} color="var(--c-axis)" />
        {evens(-10, -2).map((t) => (
          <Label key={'x' + t} at={[t, -13]} dy={15} anchor="middle" size={11} weight={400} color="var(--text-muted)">
            {String(t).replace('-', '−')}
          </Label>
        ))}
        {evens(-12, top).map((t) => (
          <Label key={'y' + t} at={[-10, t]} dx={-6} dy={4} anchor="end" size={11} weight={400} color="var(--text-muted)">
            {t === -12 ? '≤ −12' : String(t).replace('-', '−')}
          </Label>
        ))}
        <Label at={[-5.5, -13]} dy={32} anchor="middle" size={12} weight={500} color="var(--text-muted)">
          log₁₀ step (smaller to the left)
        </Label>
        <Label at={[-10, top]} dx={6} dy={4} size={12} weight={500} color="var(--text-muted)">
          log₁₀ error
        </Label>
        <Seg from={[-10, floor(experiments[index].tolerance)]} to={[-1, floor(experiments[index].tolerance)]} dash="4 4" />
        {methods.flatMap((method) =>
          rows
            .slice(1)
            .map((row, i) => (
              <Seg
                key={method + '-' + i}
                from={[-i - 1, floor(rows[i].methods[method].error)]}
                to={[-i - 2, floor(row.methods[method].error)]}
                color={colours[method]}
                width={widths[method]}
              />
            )),
        )}
      </Plot>
      {index === 2 && <p>A centred probe at a corner mixes the two sides.</p>}
    </>
  );
}

export function SensitivityLab() {
  const [index, setIndex] = useState(0);
  const [saved, save] = useState<Record<number, Settings>>(() => {
    try {
      return JSON.parse(localStorage.getItem(key) ?? '{}') ?? {};
    } catch {
      return {};
    }
  });
  const progress = useLabProgress('derivatives-lab'),
    log = useMachineLog();
  const experiment = experiments[index];
  const initial: Settings = { x: String(experiment.at), probes: [], pairs: experiment.slopes.map(() => ['', '']) };
  const cfg = saved[index] ?? initial;
  const update = (next: Settings) => {
    const values = { ...saved, [index]: next };
    save(values);
    try {
      localStorage.setItem(key, JSON.stringify(values));
    } catch {
      /* Storage is optional. */
    }
  };
  const estimates = cfg.pairs.map(([a, b], side) => {
    if (!a || !b) return;
    const first = cfg.probes[Number(a) - 1],
      second = cfg.probes[Number(b) - 1];
    if (index === 2) {
      const bounds = [first.x, second.x];
      if (side === 0 ? Math.max(...bounds) > experiment.at : Math.min(...bounds) < experiment.at) return;
    }
    return secant(first, second);
  });
  const passing = accepts(index, estimates);
  const x = Number(cfg.x);
  const valid = cfg.x.trim() !== '' && Number.isFinite(x) && Number.isFinite(experiment.run(x));
  const probe = () => {
    const y = experiment.run(x);
    update({ ...cfg, probes: [...cfg.probes, { x, y }] });
    log.push('Probe at ' + x + ' → ' + y.toPrecision(10));
  };
  return (
    <LabShell
      title="Lab: The Sensitivity Probe"
      caption="Choose an input, probe the black box, then use two recorded points to calculate a slope."
      briefs={briefs}
      active={index}
      onSelect={setIndex}
      progress={progress}
      passing={passing}
      status={
        passing
          ? 'Your report meets the tolerance.'
          : estimates.some((v) => v !== undefined)
            ? 'At least one slope still needs a closer estimate.'
            : 'Choose two distinct probes for each requested slope.'
      }
      hint={
        index === 2
          ? 'Compare the rises on the two sides of the corner. What happens when a pair straddles it?'
          : 'Look at how far the probes lie from the requested setting. A long interval measures the slope of a larger stretch of curve.'
      }
      reveal={{ label: 'error versus probe spacing', content: <ErrorPlot index={index} /> }}
      log={{ lines: log.lines, onClear: log.clear }}
    >
      <div className="btn-row">
        <label>
          Probe x{' '}
          <input
            type="number"
            step="any"
            value={cfg.x}
            onChange={(e) => update({ ...cfg, x: e.target.value })}
            style={inputStyle}
          />
        </label>
        <button type="button" className="btn" disabled={!valid || cfg.probes.length >= experiment.budget} onClick={probe}>
          Probe
        </button>
        <button
          type="button"
          className="btn"
          onClick={() => {
            update(initial);
            log.push('Started a new probe run');
          }}
        >
          Restart run
        </button>
      </div>
      <ol>
        {cfg.probes.map(({ x, y }, i) => (
          <li key={i} className="mono" style={{ overflowWrap: 'anywhere' }}>
            x = {x}, output = {y.toPrecision(10)}
          </li>
        ))}
      </ol>
      {cfg.pairs.map((pair, side) => (
        <fieldset key={side} style={{ minWidth: 0, marginBlock: 8, border: '1px solid var(--border)' }}>
          <legend>{index === 2 ? (side === 0 ? 'Left slope' : 'Right slope') : 'Slope report'}</legend>
          <div className="btn-row">
            {pair.map((value, slot) => (
              <label key={slot}>
                {slot === 0 ? 'Probe A ' : 'Probe B '}
                <select
                  value={value}
                  style={inputStyle}
                  onChange={(e) => {
                    const pairs = cfg.pairs.map((p, i): [string, string] =>
                      i !== side ? p : slot === 0 ? [e.target.value, p[1]] : [p[0], e.target.value],
                    );
                    update({ ...cfg, pairs });
                  }}
                >
                  <option value="">Choose</option>
                  {cfg.probes.map((_, i) => (
                    <option key={i} value={i + 1}>
                      Probe {i + 1}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>
          <div className="readout">
            Rise ÷ run{' '}
            {estimates[side] !== undefined ? '= ' + estimates[side]!.toPrecision(8) : '· waiting for a valid pair'}
          </div>
        </fieldset>
      ))}
    </LabShell>
  );
}
