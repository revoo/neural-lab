// authored-by: Claude Opus 5.5
/**
 * V2-WP2: the concept map (`#/map`). Modules are rows, lessons are boxes, technology destinations
 * sit along the bottom. Click a node to light up everything it needs (upstream) or everything it
 * unlocks (downstream); the panel says how many of those prerequisites you have completed.
 */
import { useMemo, useState } from 'react';
import { ALL_LESSONS, findLesson, MODULES } from '../curriculum';
import { ALL_NODES, allEdges, DESTINATIONS, downstream, isDestination, nodeLabel, prereqStatus, upstream } from '../dependencies';
import { useProgress } from '../progress';
import { lessonHref, mapHref } from '../router';
import { AREAS, lessonArea, type Area } from '../areas';
import { AreaChip } from '../components/AreaChip';

type Mode = 'up' | 'down' | 'area';
type Pos = { x: number; y: number; w: number; h: number };

const W = 1000;
const GUTTER = 40;
const ROW = 50;
const BOX_H = 26;
const DEST_H = 40;

function wrap(label: string, max: number): string[] {
  if (label.length <= max) return [label];
  const words = label.split(' ');
  const lines: string[] = [''];
  for (const w of words) {
    const cur = lines[lines.length - 1];
    if (cur && (cur + ' ' + w).length > max) lines.push(w);
    else lines[lines.length - 1] = cur ? cur + ' ' + w : w;
  }
  return lines.slice(0, 2);
}

export function DependencyGraphPage({ focus }: { focus?: string }) {
  const progress = useProgress();
  const [mode, setMode] = useState<Mode>('up');
  const [area, setArea] = useState<Area>('foundational');
  const nodes = ALL_NODES;
  const edges = useMemo(() => allEdges(), []);
  const nodeSet = useMemo(() => new Set(nodes), [nodes]);
  const selected = focus && nodeSet.has(focus) ? focus : 'dest:llms';

  // Layout: one row per module, destinations at the bottom.
  const { pos, rows, height } = useMemo(() => {
    const pos = new Map<string, Pos>();
    const rows: { label: string; title: string; y: number }[] = [];
    let y = 26;
    for (const m of MODULES) {
      const ls = m.lessons.filter((l) => nodeSet.has(l.id));
      if (!ls.length) continue;
      const span = (W - GUTTER - 10) / Math.max(ls.length, 5);
      const left = GUTTER + (W - GUTTER - 10 - span * ls.length) / 2;
      ls.forEach((l, i) => pos.set(l.id, { x: left + span * i + 4, y, w: span - 8, h: BOX_H }));
      rows.push({ label: String(m.number), title: m.title, y });
      y += ROW;
    }
    y += 16;
    const span = (W - GUTTER - 10) / DESTINATIONS.length;
    DESTINATIONS.forEach((d, i) => pos.set(d.id, { x: GUTTER + span * i + 4, y, w: span - 8, h: DEST_H }));
    return { pos, rows, height: y + DEST_H + 14 };
  }, [nodeSet]);

  const lit = useMemo(
    () => new Set(mode === 'up' ? upstream(selected) : mode === 'down' ? downstream(selected) : ALL_LESSONS.filter((l) => lessonArea(l.id) === area).map((l) => l.id)),
    [mode, selected, area],
  );
  const status = prereqStatus(selected, progress.completed);
  const unlocked = mode === 'down' ? [...lit] : [];
  const isLit = (id: string) => (mode === 'area' ? lit.has(id) : id === selected || lit.has(id));
  const edgeLit = ([a, b]: [string, string]) =>
    mode === 'up' ? lit.has(a) && (b === selected || lit.has(b)) : mode === 'down' ? (a === selected || lit.has(a)) && lit.has(b) : lit.has(a) && lit.has(b);

  const select = (id: string) => {
    if (mode === 'area') setMode('up');
    window.location.hash = mapHref(id);
  };
  const lesson = isDestination(selected) ? undefined : findLesson(selected);
  const dest = DESTINATIONS.find((d) => d.id === selected);
  const ups = upstream(selected);

  return (
    <div className="page map-page">
      <h1>The concept map</h1>
      <p className="muted" style={{ maxWidth: 720 }}>
        The curriculum is a line, but the ideas form a web. Pick where you are heading, or click any lesson, to see everything it builds on, and how much of
        that you have already done. Flip to <em>What does this unlock?</em> to follow an idea forwards.
      </p>
      <div className="map-dests" role="group" aria-label="Destinations">
        <span className="faint" style={{ fontSize: 13 }}>Where are you heading?</span>
        {DESTINATIONS.map((d) => (
          <button key={d.id} className={`btn small${selected === d.id ? ' primary' : ''}`} onClick={() => select(d.id)} title={d.blurb}>
            {d.title}
          </button>
        ))}
      </div>
      <div className="map-controls">
        <div className="track-toggle" role="radiogroup" aria-label="Direction">
          <button role="radio" aria-checked={mode === 'up'} className={mode === 'up' ? 'on' : ''} onClick={() => setMode('up')}>
            What does this need?
          </button>
          <button role="radio" aria-checked={mode === 'down'} className={mode === 'down' ? 'on' : ''} onClick={() => setMode('down')}>
            What does this unlock?
          </button>
          <button role="radio" aria-checked={mode === 'area'} className={mode === 'area' ? 'on' : ''} onClick={() => setMode('area')}>
            Where does it belong?
          </button>
        </div>
      </div>
      {mode === 'area' && (
        <div className="map-dests" role="group" aria-label="Areas">
          {(Object.keys(AREAS) as Area[]).map((a) => (
            <button key={a} className={`area-pick${area === a ? ' on' : ''}`} onClick={() => setArea(a)} aria-pressed={area === a}>
              <AreaChip area={a} />
            </button>
          ))}
        </div>
      )}

      <div className="map-readout" aria-live="polite">
        <div className="map-readout-head">
          {mode === 'area' ? <strong>{AREAS[area].label}</strong> : <strong>{dest ? dest.title : lesson?.title}</strong>}
          {mode !== 'area' && lesson && <AreaChip area={lessonArea(lesson.id)} />}
          {mode !== 'area' && lesson && (
            <a href={lessonHref(lesson.id)} className="map-open">
              Open lesson →
            </a>
          )}
        </div>
        {mode !== 'area' && dest && <div className="muted" style={{ fontSize: 14 }}>{dest.blurb}</div>}
        {mode === 'area' ? (
          <p>
            {AREAS[area].blurb} <strong>{lit.size}</strong> lesson{lit.size === 1 ? '' : 's'}, of which you have completed{' '}
            <strong>{[...lit].filter((id) => progress.completed[id]).length}</strong>.
            {area === 'foundational' && ' A lesson gets this chip exactly when the map shows it feeding at least three of the four big branches: language, vision, generation and reinforcement learning.'}
          </p>
        ) : mode === 'up' ? (
          status.total === 0 ? (
            <p>Nothing before this: it is a starting point.</p>
          ) : (
            <p>
              Builds on <strong>{status.total}</strong> lesson{status.total === 1 ? '' : 's'}. You have completed <strong>{status.done}</strong>
              {status.next ? (
                <>
                  . Your next unfinished prerequisite: <a href={lessonHref(status.next)}>{findLesson(status.next)?.title}</a>.
                </>
              ) : (
                <> of them: you are ready{lesson ? ' for this lesson' : ''}.</>
              )}
            </p>
          )
        ) : unlocked.length === 0 ? (
          <p>Nothing later in the course builds directly on this one.</p>
        ) : (
          <p>
            Unlocks <strong>{unlocked.filter((u) => !isDestination(u)).length}</strong> later lesson{unlocked.filter((u) => !isDestination(u)).length === 1 ? '' : 's'} and{' '}
            <strong>{unlocked.filter(isDestination).length}</strong> of the {DESTINATIONS.length} destinations.
          </p>
        )}
        <ul className="map-chips">
          {(mode === 'up' ? ups : mode === 'down' ? unlocked : ALL_LESSONS.map((l) => l.id).filter((id) => lit.has(id))).map((id) => (
            <li key={id}>
              <button className={`map-chip${progress.completed[id] ? ' done' : ''}${isDestination(id) ? ' dest' : ''}`} onClick={() => select(id)}>
                {progress.completed[id] ? '✓ ' : ''}
                {nodeLabel(id)}
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="map-scroll">
        <svg viewBox={`0 0 ${W} ${height}`} className="map-svg" style={{ width: "100%", minWidth: 760, height: "auto" }} role="img" aria-label="Concept dependency map">
          {rows.map((r) => (
            <g key={r.label}>
              <title>{`Module ${r.label}: ${r.title}`}</title>
              <text x={12} y={r.y + BOX_H / 2 + 4} className="map-row-label">
                {r.label}
              </text>
            </g>
          ))}
          <g fill="none">
            {edges.map(([a, b]) => {
              const p = pos.get(a);
              const q = pos.get(b);
              if (!p || !q) return null;
              const on = edgeLit([a, b]);
              const x1 = p.x + p.w / 2;
              const y1 = p.y + p.h;
              const x2 = q.x + q.w / 2;
              const y2 = q.y;
              const dy = Math.max(18, (y2 - y1) * 0.45);
              return (
                <path
                  key={a + '>' + b}
                  d={`M${x1},${y1} C${x1},${y1 + dy} ${x2},${y2 - dy} ${x2},${y2}`}
                  className={on ? 'map-edge on' : 'map-edge'}
                />
              );
            })}
          </g>
          {nodes.map((id) => {
            const p = pos.get(id)!;
            const dst = isDestination(id);
            const done = !!progress.completed[id];
            const cls = ['map-node', dst ? 'dest' : '', id === selected ? 'sel' : '', isLit(id) ? 'lit' : 'dim', done ? 'done' : ''].filter(Boolean).join(' ');
            const lines = wrap(nodeLabel(id), dst ? 18 : 30);
            const full = dst ? DESTINATIONS.find((d) => d.id === id)!.title : ALL_LESSONS.find((l) => l.id === id)!.title;
            return (
              <g
                key={id}
                className={cls}
                data-node={id}
                role="button"
                tabIndex={0}
                aria-label={full}
                onClick={() => select(id)}
                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), select(id))}
              >
                <title>{full}</title>
                <rect x={p.x} y={p.y} width={p.w} height={p.h} rx={dst ? 10 : 6} />
                {lines.map((t, i) => (
                  <text key={i} x={p.x + p.w / 2} y={p.y + p.h / 2 + 4 + (i - (lines.length - 1) / 2) * 13} textAnchor="middle">
                    {t}
                  </text>
                ))}
                {done && !dst && <circle cx={p.x + p.w - 5} cy={p.y + 5} r={3} className="map-done-dot" />}
              </g>
            );
          })}
        </svg>
      </div>
      <p className="faint" style={{ fontSize: 13 }}>
        Rows are modules (number on the left); a line means the lower lesson uses an idea from the upper one. A small dot marks a lesson you have completed. On a narrow screen, swipe the map sideways; the list above it holds the same information.
      </p>
    </div>
  );
}
