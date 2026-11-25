// authored-by: Codex Meridian 6
import { useState } from 'react';
import { Arrow, Clip, Dot, Handle, Label, Plot, Seg } from '../../components/plot';
import { Slider } from '../../components/controls';
import { LabShell, useLabProgress } from '../../components/LabShell';
import { StuckButton } from '../../components/AnswerReveal';
import { useMachineLog } from '../../components/MachineLog';
import {
  corners,
  determinant,
  identity,
  lineMargin,
  margin,
  offset,
  tilted,
  transform,
  type Matrix,
  type Pair,
} from './untangleLab';

const briefs = [
  {
    id: 'tilt',
    tab: 'Brief 1',
    title: 'Turn the cloud',
    text: 'Move every filled dot to the right of the vertical axis and every ring to the left.',
  },
  {
    id: 'area',
    tab: 'Brief 2',
    title: 'Keep the area',
    text: 'Make the same split while keeping the area of a transformed unit square between 0.9 and 1.1.',
  },
  {
    id: 'shift',
    tab: 'Brief 3',
    title: 'A different cloud',
    text: 'Separate these points around the vertical axis. Try your matrix and check the split.',
  },
  {
    id: 'xor',
    tab: 'Brief 4',
    title: 'Opposite corners',
    text: 'Separate the filled dots from the rings with any straight line. Drag the two handles on the dashed line to place it.',
  },
];
type Setting = { m: Matrix; b: Pair; kink: boolean; line: [Pair, Pair]; attempts: number };
const start: Setting = {
  m: identity,
  b: [0, 0],
  kink: false,
  line: [
    [-1, -2],
    [-1, 2],
  ],
  attempts: 0,
};
const storage = 'neural-lab:matrices-lab:settings';
const bound = (v: number, max: number) => Math.max(-max, Math.min(max, Math.round(v * 20) / 20));

function GridLine({ from, to, cfg }: { from: Pair; to: Pair; cfg: Setting }) {
  // A ReLU bends a line at an axis crossing. Split there before mapping endpoints.
  const a = transform(from, cfg.m, cfg.b, false),
    b = transform(to, cfg.m, cfg.b, false);
  const cuts = [0, 1];
  if (cfg.kink)
    for (const i of [0, 1]) {
      const t = a[i] / (a[i] - b[i]);
      if (t > 0 && t < 1) cuts.push(t);
    }
  cuts.sort((x, y) => x - y);
  const point = (t: number): Pair =>
    transform([from[0] + t * (to[0] - from[0]), from[1] + t * (to[1] - from[1])], cfg.m, cfg.b, cfg.kink);
  return (
    <>
      {cuts.slice(1).map((t, i) => (
        <Seg key={i} from={point(cuts[i])} to={point(t)} opacity={0.35} />
      ))}
    </>
  );
}

export function UntangleLab() {
  const [active, select] = useState(0);
  const [saved, save] = useState<Record<string, Setting>>(() => {
    try {
      return JSON.parse(localStorage.getItem(storage) ?? '{}') ?? {};
    } catch {
      return {};
    }
  });
  const brief = briefs[active],
    cfg = saved[brief.id] ?? start;
  const progress = useLabProgress('matrices-lab'),
    log = useMachineLog();
  const cloud = [tilted, tilted, offset, corners][active];
  const move = (p: Pair) => transform(p, cfg.m, cfg.b, cfg.kink);
  const gap = active === 3 ? lineMargin(cloud, move, cfg.line) : margin(cloud, (p) => move(p)[0]);
  const area = Math.abs(determinant(cfg.m));
  const passing = gap > 1e-9 && (active !== 1 || (area >= 0.9 - 1e-9 && area <= 1.1 + 1e-9));
  const unlocked = cfg.attempts >= 2;
  const update = (change: Partial<Setting>) => {
    const next = { ...saved, [brief.id]: { ...cfg, ...change } };
    save(next);
    try {
      localStorage.setItem(storage, JSON.stringify(next));
    } catch {
      /* storage is optional */
    }
  };
  const check = () => {
    log.push(
      'Closest signed clearance ' +
        gap.toFixed(2) +
        ', area factor ' +
        area.toFixed(2) +
        (passing ? ' · separated' : ' · try again'),
      passing ? 'good' : 'info',
    );
    if (!passing) update({ attempts: cfg.attempts + 1 });
  };
  const [a, b] = cfg.line;
  const extend = (t: number): Pair => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
  const extent = Math.max(
    4,
    ...[...cloud.positive, ...cloud.negative].flatMap((p) => move(p).map((x) => Math.ceil(Math.abs(x) + 0.5))),
  );
  const hints = [
    'Watch the points closest to the vertical axis as you turn a column.',
    'Look at the little grid cells. How much room does each one occupy?',
    'Compare the filled dot and the ring on the same ray from the origin.',
    'Imagine joining each pair of matching marks. Where do the two segments cross?',
  ];
  return (
    <LabShell
      title="Lab: Untangle the Data"
      caption="Drag the column tips or focus one and use the arrow keys. The faint mesh shows where the original grid went. A filled dot and a hollow ring are the two labels."
      briefs={briefs}
      active={active}
      onSelect={select}
      progress={progress}
      passing={passing}
      status={
        (passing ? 'Separated. ' : 'Still working. ') +
        'Closest signed clearance ' +
        gap.toFixed(2) +
        ' · area factor ' +
        area.toFixed(2)
      }
      hint={hints[active]}
      actions={
        <button className="btn" onClick={check}>
          Check this split
        </button>
      }
      reveal={
        active < 2 || unlocked
          ? {
              label: 'the geometry',
              content: [
                'Only the first row decides which side of the vertical axis a point reaches. The second row is free.',
                'A shear slides grid lines past one another while preserving their area.',
                'The ring and dot on the same ray always had x-coordinates of the same sign. A shift moves the threshold between them.',
                'Try columns (1, −1) and (−1, 1) with the kink. Opposite filled corners meet at the origin. The rings land on different axes.',
              ][active],
            }
          : undefined
      }
      log={{ lines: log.lines, onClear: log.clear }}
    >
      <Plot
        x={[-extent, extent]}
        y={[-extent, extent]}
        grid={false}
        style={{ width: '100%', maxWidth: 520, height: 'auto', display: 'block', margin: '0 auto' }}
        ariaLabel="Transformed mesh, class labels and draggable matrix columns"
      >
        <Clip>
          {[-3, -2, -1, 0, 1, 2, 3].map((k) => (
            <g key={k}>
              <GridLine from={[k, -4]} to={[k, 4]} cfg={cfg} />
              <GridLine from={[-4, k]} to={[4, k]} cfg={cfg} />
            </g>
          ))}
          {active === 3 && <Seg from={extend(-100)} to={extend(100)} color="var(--text)" dash="6 3" />}
          {cloud.negative.map((p, i) => (
            <Dot key={i} at={move(p)} r={8} color="var(--bg-elev)" stroke="var(--c-a)" />
          ))}
          {cloud.positive.map((p, i) => (
            <Dot key={i} at={move(p)} r={5} color="var(--c-b)" />
          ))}
          {[0, 1].map((col) => {
            const tip: Pair = [cfg.m[0][col], cfg.m[1][col]];
            const colour = col ? 'var(--c-bad)' : 'var(--c-c)';
            return (
              <g key={col}>
                <Arrow from={[0, 0]} to={tip} color={colour} />
                <Label at={tip} color={colour} dx={10} dy={-9}>
                  {col ? 'ĵ' : 'î'}
                </Label>
                <Handle
                  x={tip[0]}
                  y={tip[1]}
                  snap={0.05}
                  color={colour}
                  label={'Matrix column ' + (col + 1)}
                  onDrag={(x, y) => {
                    const m: Matrix = [[...cfg.m[0]], [...cfg.m[1]]];
                    m[0][col] = bound(x, 2);
                    m[1][col] = bound(y, 2);
                    update({ m });
                  }}
                />
              </g>
            );
          })}
          {active === 3 &&
            cfg.line.map(([x, y], i) => (
              <Handle
                key={i}
                x={x}
                y={y}
                snap={0.05}
                label={'Line handle ' + (i + 1)}
                onDrag={(u, v) => {
                  const line: [Pair, Pair] = [...cfg.line];
                  line[i] = [bound(u, 4), bound(v, 4)];
                  update({ line });
                }}
              />
            ))}
        </Clip>
      </Plot>
      <div className="readout">
        Matrix rows {cfg.m.map((row) => '[' + row.map((v) => v.toFixed(2)).join(', ') + ']').join(' ')} · shift [
        {cfg.b.join(', ')}]
      </div>
      {active >= 2 && !unlocked && (
        <StuckButton
          key={brief.id}
          stuckKey={'untangle-tools-' + brief.id}
          label="why this split is hard and the extra control"
          onStuck={() => update({ attempts: 2 })}
        />
      )}
      {active >= 2 && (unlocked || active === 3) && (
        <div className="controls">
          <p>
            {active === 2
              ? 'A matrix keeps the origin fixed. The dot and ring on the same ray need a threshold away from it. Try shifting the mesh.'
              : unlocked
                ? 'The diagonals share a midpoint even after a shift. Try a kink that clamps each negative coordinate to zero.'
                : 'Try sliding the corners before you place your line.'}
          </p>
          {[0, 1].map((i) => (
            <Slider
              key={i}
              label={i ? 'Vertical shift' : 'Horizontal shift'}
              min={-2}
              max={2}
              step={0.05}
              value={cfg.b[i]}
              onChange={(v) => {
                const bias: Pair = [...cfg.b];
                bias[i] = v;
                update({ b: bias });
              }}
            />
          ))}
          {active === 3 && unlocked && (
            <label>
              <input type="checkbox" checked={cfg.kink} onChange={(e) => update({ kink: e.target.checked })} /> Apply the ReLU
              kink
            </label>
          )}
        </div>
      )}
    </LabShell>
  );
}
