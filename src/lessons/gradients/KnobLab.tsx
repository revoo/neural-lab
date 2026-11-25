// authored-by: Codex Meridian 6
import { useState } from 'react';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { Slider } from '../../components/controls';
import { axes, contourError, gradient, loss, minTurns, move, start, sweep, targetLoss, unit, type Point } from './knobLab';

const key = 'neural-lab:knob-settings';
const rates = sweep();
const budget = Math.min(...rates.map((row) => row.turns));
const shortest = minTurns();
const briefs = [
  {
    id: 'turns',
    tab: 'One at a time',
    text: 'Get below ' + targetLoss + ' loss with one-knob turns. See how few turns you need.',
  },
  { id: 'combined', tab: 'Move together', text: 'Get below the same target in at most ' + budget + ' combined moves.' },
  {
    id: 'contour',
    tab: 'Walk a contour',
    text: 'Travel at least one unit while every part of the walk stays within ±0.001 of the starting loss.',
  },
];
type Setting = {
  point: Point;
  direction: Point;
  length: number;
  turns: number;
  walked: number;
  error: number;
  probed: boolean;
};
const initial: Setting = { point: start, direction: [0, 0, 0], length: 1, turns: 0, walked: 0, error: 0, probed: false };
export function KnobLab() {
  const [active, select] = useState(0);
  const [saved, save] = useState<Record<number, Setting>>(() => {
    try {
      return JSON.parse(localStorage.getItem(key) ?? '{}') ?? {};
    } catch {
      return {};
    }
  });
  const progress = useLabProgress('gradients-lab'),
    log = useMachineLog();
  const cfg = saved[active] ?? initial;
  const update = (next: Setting) => {
    const all = { ...saved, [active]: next };
    save(all);
    try {
      localStorage.setItem(key, JSON.stringify(all));
    } catch {
      /* Storage is optional. */
    }
  };
  const slopes = gradient(cfg.point);
  const cost = loss(cfg.point);
  const passing =
    active === 2 ? cfg.walked >= 1 - 1e-9 && cfg.error <= 0.001 : cost < targetLoss && (active === 0 || cfg.turns <= budget);
  const exhausted = active === 1 && cfg.turns >= budget;
  const advance = (direction: Point, length: number) => {
    const point = move(cfg.point, direction, length);
    const error = Math.max(cfg.error, contourError(cfg.point, point));
    update({ ...cfg, point, turns: cfg.turns + 1, walked: cfg.walked + length, error, probed: false });
    log.push('Move ' + (cfg.turns + 1) + ' → loss ' + loss(point).toFixed(5));
  };
  return (
    <LabShell
      title="Lab: Which Knob, and How Far?"
      caption="Read the current loss, probe to see the three individual slopes, and choose where to move."
      briefs={briefs}
      active={active}
      onSelect={select}
      progress={progress}
      passing={passing}
      optimal={active === 0 && passing && cfg.turns === shortest}
      status={
        passing
          ? 'This walk meets the brief.'
          : exhausted
            ? 'Move budget spent. Restart to try a different walk.'
            : active === 2 && cfg.error > 0.001
              ? 'This walk left the band. Restart to try a different direction.'
              : 'Try a move and watch the loss.'
      }
      hint={
        active === 2
          ? 'Compare the relative slopes. Which direction could change the knobs with very little change in loss?'
          : 'The partials measure how steeply the loss changes when you turn each knob. Their signs tell you which way is downhill.'
      }
      reveal={{
        label: 'the reference walks',
        content: (
          <p>
            {active === 0
              ? 'An exhaustive breadth-first search finds ' + shortest + ' turns.'
              : active === 1
                ? budget +
                  ' moves was the best found by searching fixed lengths from 0.01 to 2 in steps of 0.01, each time pointing downhill.'
                : 'The third axis starts perpendicular to the gradient. Travelling either way along it changes loss by only 0.0005 per unit from the start.'}
          </p>
        ),
      }}
      log={{ lines: log.lines, onClear: log.clear }}
    >
      <div className="readout">
        Knobs [{cfg.point.map((x) => x.toFixed(3)).join(', ')}] · loss {cost.toFixed(5)} · {cfg.turns} moves
        {active === 2 && (
          <div>
            Distance {cfg.walked.toFixed(3)} · largest loss drift {cfg.error.toFixed(5)}
          </div>
        )}
        {cfg.probed && <div>Partial slopes [{slopes.map((value) => value.toFixed(5)).join(', ')}]</div>}
      </div>
      <div className="btn-row" style={{ marginBlock: 10 }}>
        <button
          type="button"
          className="btn"
          onClick={() => {
            update({ ...cfg, probed: true });
            log.push('Probed slopes ' + slopes.map((s) => s.toFixed(5)).join(', '));
          }}
        >
          Probe
        </button>
        <button
          type="button"
          className="btn"
          onClick={() => {
            update(initial);
            log.push('Back to the starting position');
          }}
        >
          Restart
        </button>
      </div>
      {active === 0 ? (
        <div className="btn-row">
          {axes.map((axis, i) => (
            <button type="button" className="btn" key={i} onClick={() => advance(axis, 0.1)}>
              Knob {Math.floor(i / 2) + 1} {i % 2 ? '−' : '+'}0.1
            </button>
          ))}
        </div>
      ) : (
        <div className="controls">
          {cfg.direction.map((value, i) => (
            <Slider
              key={i}
              label={'Direction component ' + (i + 1)}
              min={-2}
              max={2}
              step={0.01}
              value={value}
              onChange={(v) => {
                const direction: Point = [...cfg.direction];
                direction[i] = v;
                update({ ...cfg, direction });
              }}
            />
          ))}
          <Slider
            label="Move length"
            min={0.01}
            max={2}
            step={0.01}
            value={cfg.length}
            onChange={(length) => update({ ...cfg, length })}
          />
          <button
            type="button"
            className="btn"
            disabled={exhausted || cfg.direction.every((value) => value === 0)}
            onClick={() => advance(unit(cfg.direction), cfg.length)}
          >
            Move
          </button>
        </div>
      )}
    </LabShell>
  );
}
