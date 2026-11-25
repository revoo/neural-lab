// authored-by: Codex Meridian 6
import { useState } from 'react';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { useMachineLog } from '../../components/MachineLog';
import { Plot, Seg } from '../../components/plot';
import {
  batch,
  compose,
  inspect,
  matches,
  maxLayers,
  parts,
  solvePipeline,
  swapLast,
  target,
  transforms,
  meets,
  wide,
} from './pipelineLab';

const title = 'Lab: Build the Pipeline';
const storage = 'neural-lab:pipeline-settings';
const briefs = [
  { id: 'scores', tab: 'Scores', text: 'Send the whole batch through a chain that outputs ten scores per image.' },
  {
    id: 'hidden',
    tab: 'Hidden width',
    text: 'Reach the same output through at least one hidden layer, with every hidden layer at least 64 wide. Then try to minimise the parameter count.',
  },
  {
    id: 'deeper',
    tab: 'Two layers',
    text: 'Use at least two hidden layers, each at least 64 wide. Then try to minimise the batch’s multiply-adds.',
  },
  {
    id: 'order',
    tab: 'Order matters',
    text: 'Recreate the target grid with as few tiles as possible. Read the operations from left to right.',
  },
];
const solution = solvePipeline();

function Grid({ matrix: [a, b, c, d], label, range }: { matrix: ReturnType<typeof compose>; label: string; range: number }) {
  const map = (x: number, y: number): [number, number] => [a * x + b * y, c * x + d * y];
  return (
    <div style={{ minWidth: 0 }}>
      <strong>{label}</strong>
      <Plot
        width={240}
        height={240}
        x={[-range, range]}
        y={[-range, range]}
        grid={false}
        tickLabels={false}
        ariaLabel={label}
        style={{ display: 'block', maxWidth: '100%', height: 'auto' }}
      >
        {[-1, 0, 1].map((n) => (
          <g key={n}>
            <Seg from={map(-1, n)} to={map(1, n)} color={n ? 'var(--text-faint)' : 'var(--c-a)'} />
            <Seg from={map(n, -1)} to={map(n, 1)} color={n ? 'var(--text-faint)' : 'var(--c-b)'} />
          </g>
        ))}
        <Seg from={[0, 0]} to={map(1, 1)} color="var(--c-c)" width={3} />
      </Plot>
    </div>
  );
}

export function PipelineLab() {
  const [active, setActive] = useState(0);
  const [saved, setSaved] = useState<Record<string, number[]>>(() => {
    try {
      return JSON.parse(localStorage.getItem(storage) ?? '{}') ?? {};
    } catch {
      return {};
    }
  });
  const progress = useLabProgress('matrix-multiplication-lab');
  const log = useMachineLog();
  const brief = briefs[active],
    chain = saved[brief.id] ?? [];
  const geometry = active === 3,
    limit = geometry ? 4 : maxLayers;
  const result = inspect(geometry ? [] : chain);
  const passing = geometry ? matches(chain) : meets(result, active);
  const optimal =
    passing &&
    (active === 1
      ? result.parameters === solution.parameters
      : active === 2
        ? result.multiplyAdds === solution.multiplyAdds
        : false);
  const labels = geometry ? transforms.map((t) => t.name) : parts.map((p) => p.join(' × '));
  const update = (next: number[], message: string) => {
    const values = { ...saved, [brief.id]: next };
    setSaved(values);
    try {
      localStorage.setItem(storage, JSON.stringify(values));
    } catch {
      /* Storage is optional. */
    }
    const stats = inspect(geometry ? [] : next);
    log.push(
      geometry
        ? message + ' · grid entries ' + compose(next).join(', ')
        : message + ' · ' + stats.parameters.toLocaleString() + ' parameters',
    );
  };
  const add = (id: number) => {
    if (chain.length < limit && labels[id]) update([...chain, id], 'Added ' + labels[id]);
  };
  const matrix = compose(geometry ? chain : []);
  const swapped = compose(geometry ? swapLast(chain) : []);
  const range =
    Math.max(
      2,
      ...[matrix, swapped, target].map(([a, b, c, d]) => Math.max(Math.abs(a) + Math.abs(b), Math.abs(c) + Math.abs(d))),
    ) * 1.15;
  const mismatch = result.widths.length < chain.length;
  const status = geometry
    ? passing
      ? 'Target matched with ' + chain.length + ' tiles.'
      : 'Compare the direction of the grid axes with the target.'
    : mismatch
      ? 'The inner sizes must match.'
      : passing
        ? 'This chain meets the brief.'
        : result.complete && active > 0 && result.hidden >= active && result.narrowest < wide
          ? 'A hidden layer only ' + result.narrowest + ' wide squeezes every image through ' + result.narrowest + ' numbers. Each hidden layer must be at least ' + wide + ' wide.'
          : 'The chain still needs to meet the output and hidden-layer requirements.';
  return (
    <LabShell
      title={title}
      caption="Drag a tile into the chain or select it to append. Select an installed tile to remove it."
      briefs={briefs}
      active={active}
      onSelect={setActive}
      progress={progress}
      passing={passing}
      optimal={optimal}
      status={status}
      hint={
        geometry
          ? 'Follow the marked diagonal through each operation. Does it land where the target’s diagonal lands?'
          : 'Compare the width before the first highlighted tile with the tile’s input size.'
      }
      reveal={{
        label: 'the exhaustive search',
        content: (
          <p>
            Of {solution.trials.toLocaleString()} tile chains, {solution.valid.length} reach the score shape. The smallest
            chain whose hidden layers are all at least {wide} wide has {solution.parameters.toLocaleString()} parameters (a narrow
            layer counts as hidden too). With two hidden layers, both at least {wide} wide, the fewest multiply-adds is {solution.multiplyAdds.toLocaleString()}. The target grid has {solution.exact.length}{' '}
            recipes of length {solution.shortest}, the shortest length in the search.
          </p>
        ),
      }}
      log={{ lines: log.lines, onClear: log.clear }}
    >
      <div className="btn-row" aria-label="Parts bin">
        {labels.map((label, id) => (
          <button
            key={label}
            type="button"
            className="btn"
            disabled={chain.length === limit}
            draggable={chain.length < limit}
            onDragStart={(e) => e.dataTransfer.setData('application/x-neural-part', String(id))}
            onClick={() => add(id)}
          >
            {label}
          </button>
        ))}
      </div>
      <div
        className="readout"
        style={{ marginTop: 12 }}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const part = e.dataTransfer.getData('application/x-neural-part');
          if (part) add(Number(part));
        }}
      >
        <div>{geometry ? 'Original grid' : 'Input · ' + batch + ' × 784'}</div>
        {chain.map((id, i) => (
          <div key={i} style={{ marginBlock: 6 }}>
            <button
              className="btn small"
              type="button"
              aria-label={'Remove tile ' + (i + 1) + ', ' + labels[id]}
              onClick={() =>
                update(
                  chain.filter((_, j) => j !== i),
                  'Removed ' + labels[id],
                )
              }
            >
              {labels[id]} ×
            </button>
            {!geometry &&
              (i < result.widths.length ? (
                ' → ' + batch + ' × ' + result.widths[i]
              ) : i === result.widths.length ? (
                <strong style={{ color: 'var(--c-bad)' }}> Inner sizes must match</strong>
              ) : (
                ' · waiting'
              ))}
          </div>
        ))}
        {chain.length < limit && <div>Drop a tile here</div>}
      </div>
      {geometry ? (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit,minmax(min(170px,100%),1fr))',
            gap: 8,
            marginTop: 12,
          }}
        >
          <Grid matrix={target} label="Target" range={range} />
          <Grid matrix={matrix} label="Your grid" range={range} />
          {chain.length >= 2 && <Grid matrix={swapped} label="Last two swapped" range={range} />}
        </div>
      ) : (
        <p>
          {result.parameters.toLocaleString()} parameters · {result.multiplyAdds.toLocaleString()} multiply-adds for this
          batch
        </p>
      )}
    </LabShell>
  );
}
