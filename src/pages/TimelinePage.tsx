// authored-by: Claude Opus 5.5
import { useEffect, useMemo, useState } from 'react';
import { ERAS, HISTORY, MILESTONES, TIMELINE, assignLanes, timelineX, type Era } from '../history';
import { findLesson } from '../curriculum';
import { useProgress } from '../progress';
import { lessonHref, timelineHref } from '../router';
import { HistoryBody } from '../components/HistoryCard';

const ERA_FILL: Record<Era['kind'], string> = {
  founding: 'color-mix(in srgb, var(--accent) 12%, transparent)',
  winter: 'color-mix(in srgb, var(--c-b) 16%, transparent)',
  boom: 'color-mix(in srgb, var(--c-good) 14%, transparent)',
};

/**
 * The whole arc of the field on one axis, built from src/history.ts (the same
 * data as the per-lab cards). Era bands carry the mood of the moment; each
 * lab's idea is a dot; filled dots are lessons you have finished, and the
 * marker shows the latest idea you have reached. It is a map, not a badge.
 */
export function TimelinePage({ focus }: { focus?: string }) {
  const progress = useProgress();
  const [era, setEra] = useState<number | null>(null);
  const entries = useMemo(() => [...HISTORY].sort((a, b) => a.years[0] - b.years[0]), []);
  const xs = entries.map((h) => timelineX(h.years[0]));
  const lanes = assignLanes(xs, 72);
  const laneCount = Math.max(...lanes) + 1;
  const top = 34;
  const laneH = 24;
  const axisY = top + laneCount * laneH + 10;
  const H = axisY + 46;
  const reached = entries.filter((h) => progress.completed[h.lessonId]);
  const latest = reached.length ? Math.max(...reached.map((h) => h.years[0])) : null;

  useEffect(() => {
    if (focus) document.getElementById(`t-${focus}`)?.scrollIntoView({ block: 'center' });
  }, [focus]);

  const ticks = [1940, 1960, 1980, 2000, 2020];
  return (
    <div className="page timeline-page">
      <h1 style={{ marginTop: 0 }}>A timeline of AI</h1>
      <p className="muted">
        Every lab’s idea, placed where it happened, against the mood of its time: the founding optimism, two winters when neural networks were out of favour, and the deep-learning boom. Click an era for its mood, or a dot for its story. Filled dots are lessons you have finished. Every date and quote links to its source.
      </p>

      <div style={{ overflowX: 'auto', margin: '14px 0' }}>
        <svg viewBox={`0 0 ${TIMELINE.width} ${H}`} style={{ width: '100%', minWidth: 640, height: 'auto', display: 'block' }} role="img" aria-label="Timeline of the ideas behind each lab">
          {ERAS.map((e, i) => (
            <g key={e.label} style={{ cursor: 'pointer' }} onClick={() => setEra(era === i ? null : i)}>
              <rect x={timelineX(e.from)} y={top - 22} width={timelineX(e.to) - timelineX(e.from)} height={axisY - top + 22} fill={ERA_FILL[e.kind]} stroke={era === i ? 'var(--accent)' : 'none'} />
              <text x={timelineX(e.from) + 4} y={top - 10} fontSize={12.5} fill="var(--text-muted)">
                {e.label}
              </text>
            </g>
          ))}
          <line x1={8} x2={TIMELINE.width - 8} y1={axisY} y2={axisY} stroke="var(--border)" strokeWidth={1.5} />
          <line x1={TIMELINE.stubWidth - 6} x2={TIMELINE.stubWidth - 6} y1={axisY - 5} y2={axisY + 5} stroke="var(--text-faint)" />
          <text x={8} y={axisY + 16} fontSize={12} className="plot-tick">earlier</text>
          {ticks.map((t) => (
            <text key={t} x={timelineX(t)} y={axisY + 16} fontSize={12} textAnchor="middle" className="plot-tick">
              {t}
            </text>
          ))}
          {MILESTONES.map((m) => (
            <g key={m.label}>
              <line x1={timelineX(m.year)} x2={timelineX(m.year)} y1={axisY - 8} y2={axisY + 4} stroke="var(--text-muted)" strokeWidth={2} />
              <title>{`${m.year}: ${m.label}`}</title>
            </g>
          ))}
          {latest !== null && (
            <g>
              <line x1={timelineX(latest)} x2={timelineX(latest)} y1={top - 4} y2={axisY + 22} stroke="var(--accent)" strokeWidth={2} strokeDasharray="4 3" />
              <text x={timelineX(latest)} y={axisY + 34} fontSize={10.5} textAnchor="middle" fill="var(--accent-text)" fontWeight={700}>
                you are here
              </text>
            </g>
          )}
          {entries.map((h, i) => {
            const done = !!progress.completed[h.lessonId];
            const y = top + lanes[i] * laneH + 8;
            return (
              <a key={h.id} href={timelineHref(h.id)}>
                <line x1={xs[i]} x2={xs[i]} y1={y} y2={axisY} stroke="var(--border)" />
                <circle cx={xs[i]} cy={y} r={5.5} fill={done ? 'var(--accent)' : 'var(--bg)'} stroke="var(--accent)" strokeWidth={2} />
                <text x={xs[i] + 8} y={y + 4} fontSize={12} fill={focus === h.id ? 'var(--accent-text)' : 'var(--text-muted)'} fontWeight={focus === h.id ? 700 : 400}>
                  {h.years[0]}
                </text>
                <title>{`${h.when}: ${h.idea} (${h.lab})`}</title>
              </a>
            );
          })}
        </svg>
      </div>

      {era !== null && (
        <div className="live-note" style={{ marginBottom: 14 }}>
          <strong>
            {ERAS[era].label} ({ERAS[era].from}–{ERAS[era].to === 2026 ? 'now' : ERAS[era].to}).
          </strong>{' '}
          {ERAS[era].mood.text}{' '}
          <a href={ERAS[era].mood.source} target="_blank" rel="noreferrer">
            [source]
          </a>
        </div>
      )}

      <div className="timeline-list">
        {entries.map((h) => {
          const lesson = findLesson(h.lessonId);
          const done = !!progress.completed[h.lessonId];
          return (
            <section key={h.id} id={`t-${h.id}`} className={`timeline-item${focus === h.id ? ' hl' : ''}`}>
              <div className="timeline-item-head">
                <span className="history-when">{h.when}</span>
                <h3>{h.idea}</h3>
                {lesson && (
                  <a href={lessonHref(lesson.id)} className="timeline-lesson">
                    {done ? '✓ ' : ''}Lab: {h.lab} · {lesson.title} →
                  </a>
                )}
              </div>
              <HistoryBody h={h} />
            </section>
          );
        })}
      </div>
      <p className="muted" style={{ fontSize: 14 }}>
        The timeline places each lab’s idea; lessons without a lab will join it as their history is sourced. Field-wide milestones (ticks on the axis):{' '}
        {MILESTONES.map((m, i) => (
          <span key={m.label}>
            {i > 0 && '; '}
            <a href={m.source} target="_blank" rel="noreferrer">
              {m.year}, {m.label}
            </a>
          </span>
        ))}
        .
      </p>
    </div>
  );
}
