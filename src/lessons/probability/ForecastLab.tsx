// authored-by: Codex Meridian 6
import { useState } from 'react';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { Slider } from '../../components/controls';
import { correctTemperature, days, dryDays, fixedLogits, gap, normalise, ratio, scores, tolerance, shownPercent } from './forecastLab';

const labels = ['Sun', 'Cloud', 'Rain', 'Snow'];
const briefs = [
  {
    id: 'calibrate',
    tab: 'Match observations',
    text: 'Adjust the scores until all four probabilities are within two percentage points of the observed frequencies.',
  },
  { id: 'lock', tab: 'Lock sun', text: 'Match the same observations with the sun score fixed at zero.' },
  {
    id: 'temperature',
    tab: 'Temperature',
    text: 'The model scores are fixed. Match the observations by adjusting the temperature alone.',
  },
  {
    id: 'zero',
    tab: 'No snow',
    text: 'Get the biggest gap below half a percentage point, then explain whether the probability of snow could ever reach exactly zero.',
  },
];
const storage = 'neural-lab:forecast-settings';
const start = { logits: [0, 0, 0, 0], temperature: 1, answer: '' };
const percent = (p: number) => (100 * p).toFixed(2) + '%';
export function ForecastLab() {
  const [active, select] = useState(0);
  const [saved, save] = useState<Record<number, typeof start>>(() => {
    try {
      return JSON.parse(localStorage.getItem(storage) ?? '{}') ?? {};
    } catch {
      return {};
    }
  });
  const progress = useLabProgress('probability-lab'),
    log = useMachineLog();
  const state = saved[active] ?? start;
  const counts = active === 3 ? dryDays : days,
    targets = normalise(counts);
  const logits = active === 2 ? fixedLogits : state.logits;
  const probabilities = scores(logits, state.temperature);
  const largest = gap(probabilities, counts);
  const match = active === 3 ? largest < tolerance(active) : largest <= tolerance(active);
  const passing = match && (active !== 3 || state.answer === 'positive');
  const update = (next: typeof start, message: string) => {
    const all = { ...saved, [active]: next };
    save(all);
    try {
      localStorage.setItem(storage, JSON.stringify(all));
    } catch {
      /* Storage is optional. */
    }
    log.push(
      message + ' · max gap ' + percent(gap(scores(active === 2 ? fixedLogits : next.logits, next.temperature), counts)),
    );
  };
  return (
    <LabShell
      title="Lab: Calibrate the Forecaster"
      caption="Compare each forecast bar with the frequency observed over the same stretch of days. The ratio beside each pair compares that class with sun."
      briefs={briefs}
      active={active}
      onSelect={select}
      progress={progress}
      passing={passing}
      status={
        'Largest gap ' +
        (100 * largest).toFixed(2) +
        ' percentage points. ' +
        (passing
          ? 'This forecast meets the brief.'
          : match
            ? 'Now explain the remaining snow probability.'
            : 'Bring the bars closer.')
      }
      hint={
        active === 2
          ? 'Compare how big the gaps between the bars are. What happens to all the gaps as you change the temperature?'
          : 'Watch the ratios as you change one score. How does its distance from the sun score change?'
      }
      reveal={{
        label: 'the calibration',
        content: (
          <p>
            {active === 2
              ? 'At temperature ' + correctTemperature + ', dividing by temperature recovers the log probabilities exactly.'
              : active === 3
                ? 'Each exponential stays positive at any finite score. Moving the snow score further below the others makes its share arbitrarily small.'
                : 'Taking the logarithm of each target gives scores that softmax turns back into exactly the same probabilities. Adding a common constant preserves every ratio.'}
          </p>
        ),
      }}
      log={{ lines: log.lines, onClear: log.clear }}
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(min(230px,100%),1fr))', gap: 14 }}>
        <div className="controls">
          {labels.map((label, i) => (
            <div key={label}>
              {active === 2 || (active === 1 && i === 0) ? (
                <span>
                  {label} score {logits[i].toFixed(3)} · locked
                </span>
              ) : (
                <Slider
                  label={label + ' score'}
                  min={-8}
                  max={5}
                  step={0.05}
                  value={logits[i]}
                  onChange={(value) =>
                    update({ ...state, logits: logits.map((v, j) => (j === i ? value : v)) }, label + ' score ' + value)
                  }
                />
              )}
            </div>
          ))}
          {active === 2 ? (
            <Slider
              label="Temperature"
              min={0.2}
              max={3}
              step={0.05}
              value={state.temperature}
              onChange={(temperature) => update({ ...state, temperature }, 'Temperature ' + temperature)}
            />
          ) : (
            <span>Temperature {state.temperature.toFixed(2)} · locked</span>
          )}
        </div>
        <div>
          {labels.map((label, i) => (
            <div key={label} style={{ marginBottom: 10 }}>
              <strong>{label}</strong> · {counts[i]} observed days
              {i > 0 && <span> · ×{ratio(logits[i], logits[0], state.temperature).toPrecision(3)} versus sun</span>}
              {[probabilities[i], targets[i]].map((value, j) => (
                <div
                  key={j}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '7em minmax(0,1fr)',
                    alignItems: 'center',
                    gap: 6,
                    fontSize: 12,
                  }}
                >
                  <span>
                    {j ? 'Observed' : 'Forecast'} {shownPercent(value)}
                  </span>
                  <div style={{ height: 7, background: 'var(--bg-sunken)' }}>
                    <div style={{ height: '100%', width: percent(value), minWidth: value > 0 ? 2 : 0, background: j ? 'var(--c-b)' : 'var(--c-a)' }} />
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
      {active === 3 && match && (
        <fieldset style={{ border: '1px solid var(--border)', minWidth: 0 }}>
          <legend>Can a finite score give snow exactly zero probability?</legend>
          <div className="btn-row">
            {[
              ['yes', 'Yes, if its score is negative'],
              ['positive', 'No, the exponential is always positive'],
            ].map(([value, label]) => (
              <button
                key={value}
                type="button"
                className="btn"
                aria-pressed={state.answer === value}
                onClick={() => update({ ...state, answer: value }, label)}
              >
                {label}
              </button>
            ))}
          </div>
        </fieldset>
      )}
    </LabShell>
  );
}
