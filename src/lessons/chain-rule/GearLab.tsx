// authored-by: Codex Meridian 6
import { useState } from 'react';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { Slider } from '../../components/controls';
import { aim, initial, pipeline, ranked, report, searchWalks, revived, saturated, sensitivities, turn } from './gearLab';

const stages = ['Scale', 'Tanh', 'Shifted scale', 'Sigmoid', 'Output scale'];
const ids = stages.map((_, i) => i);
const briefs = [
  {
    id: 'rank',
    tab: 'Rank',
    text: 'Order the knobs from largest to smallest influence on the output. Either order works for ties.',
  },
  { id: 'shot', tab: 'One shot', text: 'Choose a single small adjustment that lands within ±0.01 of the target output.' },
  {
    id: 'dead',
    tab: 'Dead stage',
    text: 'Adjust the third knob until the magnitude of the first knob’s influence reaches ' + revived + '.',
  },
  {
    id: 'backward',
    tab: 'Backward walk',
    text: 'Report every sensitivity. Try to minimise the number of products by reusing downstream multipliers.',
  },
];
type State = {
  order: number[];
  selected: number;
  delta: number;
  tried: boolean;
  third: number;
  reports: Record<number, number>;
  cache: (number | null)[];
  products: number;
  reuse: boolean;
};
const start: State = {
  order: ids,
  selected: 0,
  delta: 0,
  tried: false,
  third: saturated[2],
  reports: {},
  cache: [],
  products: 0,
  reuse: true,
};
const storage = 'neural-lab:gear-settings';
const best = searchWalks().minimum;

export function GearLab() {
  const [active, select] = useState(0);
  const [saved, save] = useState<Record<number, State>>(() => {
    try {
      return JSON.parse(localStorage.getItem(storage) ?? '{}') ?? {};
    } catch {
      return {};
    }
  });
  const progress = useLabProgress('chain-rule-lab'),
    log = useMachineLog();
  const state = saved[active] ?? start;
  const { order, selected, delta, tried, third, reports, cache, products, reuse } = state;
  const knobs =
    active === 2
      ? turn(saturated, 2, third - saturated[2])
      : active === 1 && tried
        ? turn(initial, selected, delta)
        : initial;
  const { outputs, inputSlopes, knobSlopes } = pipeline(knobs);
  const influence = sensitivities(knobs),
    output = outputs[4];
  const passing = [
    ranked(order, knobs),
    tried && Math.abs(output - aim) <= 0.01,
    Math.abs(influence[0]) >= revived,
    Object.keys(reports).length === 5,
  ][active];
  const update = (next: Partial<State>) => {
    const all = { ...saved, [active]: { ...state, ...next } };
    save(all);
    try {
      localStorage.setItem(storage, JSON.stringify(all));
    } catch {
      /* Storage is optional. */
    }
  };
  const reorder = (from: number, to: number) => {
    if (from < 0 || from >= order.length || to < 0 || to >= order.length || from === to) return;
    const next = [...order];
    next.splice(to, 0, ...next.splice(from, 1));
    update({ order: next });
    log.push('Ranked ' + next.map((i) => stages[i]).join(' → '));
  };
  return (
    <LabShell
      title="Lab: Find the Slipping Gear"
      caption="Select a stage to inspect the slope of its input and the slope of its own knob."
      briefs={briefs}
      active={active}
      onSelect={select}
      progress={progress}
      passing={passing}
      optimal={active === 3 && passing && products === best}
      status={
        passing
          ? 'This meets the brief.'
          : active === 1 && tried
            ? 'This shot missed. Restart to take another.'
            : 'Use the local slopes to see which changes carry through.'
      }
      hint={
        [
          'Compare each knob’s slope with the slopes it still has to pass through.',
          'For a small move, the output change is roughly the adjustment times its sensitivity.',
          'Look at how close the sigmoid output is to an endpoint.',
          'Look for the same stretch of downstream stages in more than one report.',
        ][active]
      }
      reveal={{
        label: 'the sensitivity calculation',
        content: (
          <p>
            {active === 3
              ? 'Reusing suffixes takes seven products. Computing every report independently takes ten.'
              : 'The knob slope is multiplied by every input slope after that stage. The final knob’s report stops at its local slope.'}
          </p>
        ),
      }}
      log={{ lines: log.lines, onClear: log.clear }}
      actions={
        <button
          className="btn"
          type="button"
          onClick={() => {
            update(start);
            log.push('Reset this brief');
          }}
        >
          Reset
        </button>
      }
    >
      <div className="btn-row">
        {stages.map((label, i) => (
          <button
            key={label}
            type="button"
            className="btn"
            aria-pressed={selected === i}
            disabled={active === 1 && tried}
            onClick={() => update({ selected: i })}
          >
            {label} · {outputs[i].toFixed(3)}
          </button>
        ))}
      </div>
      <div className="readout" style={{ marginBlock: 10 }}>
        {stages[selected]} · knob {knobs[selected].toFixed(3)} · knob slope {knobSlopes[selected].toPrecision(4)} · input
        slope {inputSlopes[selected].toPrecision(4)}
      </div>
      {active === 0 && (
        <ol>
          {order.map((stage, i) => (
            <li
              key={stage}
              draggable
              onDragStart={(e) => e.dataTransfer.setData('application/x-gear-rank', String(i))}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const from = e.dataTransfer.getData('application/x-gear-rank');
                if (from) reorder(Number(from), i);
              }}
            >
              {stages[stage]}{' '}
              <button
                type="button"
                className="btn small"
                onClick={() => reorder(i, i - 1)}
                disabled={i === 0}
                aria-label={'Move ' + stages[stage] + ' up'}
              >
                ↑
              </button>{' '}
              <button
                type="button"
                className="btn small"
                onClick={() => reorder(i, i + 1)}
                disabled={i === 4}
                aria-label={'Move ' + stages[stage] + ' down'}
              >
                ↓
              </button>
            </li>
          ))}
        </ol>
      )}
      {active === 1 && (
        <div className="controls">
          <p>
            Current output {output.toFixed(5)} · target {aim.toFixed(5)} · selected sensitivity{' '}
            {influence[selected].toFixed(5)}
          </p>
          {!tried && (
            <>
              <Slider
                label="Adjustment"
                min={-0.2}
                max={0.2}
                step={0.01}
                value={delta}
                onChange={(v) => update({ delta: v })}
              />
              <p>Linear estimate {(output + delta * influence[selected]).toFixed(5)}</p>
              <button
                type="button"
                className="btn"
                onClick={() => {
                  update({ tried: true });
                  log.push(
                    stages[selected] +
                      ' adjusted by ' +
                      delta +
                      ' → ' +
                      pipeline(turn(initial, selected, delta)).outputs[4].toFixed(5),
                  );
                }}
              >
                Take the shot
              </button>
            </>
          )}
        </div>
      )}
      {active === 2 && (
        <div className="controls">
          <Slider
            label="Third knob"
            min={-2}
            max={2}
            step={0.05}
            value={third}
            onChange={(v) => {
              update({ third: v });
              log.push(
                'Third knob ' +
                  v +
                  ' · first sensitivity ' +
                  sensitivities(turn(saturated, 2, v - saturated[2]))[0].toPrecision(4),
              );
            }}
          />
          <p>
            Sigmoid output {outputs[3].toPrecision(6)} · first sensitivity magnitude {Math.abs(influence[0]).toPrecision(4)}
          </p>
        </div>
      )}
      {active === 3 && (
        <div>
          <label>
            <input type="checkbox" checked={reuse} onChange={(e) => update({ reuse: e.target.checked })} /> Reuse downstream
            products already computed
          </label>
          <p>Multiplications: {products}</p>
          <div className="btn-row">
            {ids.map((stage) => (
              <button
                type="button"
                key={stage}
                disabled={reports[stage] !== undefined}
                className="btn"
                onClick={() => {
                  const result = report(knobs, stage, cache, reuse);
                  update({
                    reports: { ...reports, [stage]: result.value },
                    cache: result.costs,
                    products: products + result.work,
                  });
                  log.push(stages[stage] + ' → ' + result.value.toPrecision(4) + ' · ' + result.work + ' multiplications');
                }}
              >
                {stages[stage]}
                {reports[stage] === undefined ? ' · calculate' : ' · ' + reports[stage].toPrecision(4)}
              </button>
            ))}
          </div>
        </div>
      )}
    </LabShell>
  );
}
