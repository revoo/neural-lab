// authored-by: Gemini Antigravity
/**
 * Compact, date-ordered strip of eras, milestones and lab history entries from
 * src/history.ts whose years fall in [from, to]. Used at the top of the four
 * era interludes so every date links straight to the full timeline.
 */
import { ERAS, HISTORY, MILESTONES, type Era } from '../history';
import { timelineHref } from '../router';

export type EraStripItem = {
  kind: 'era' | 'milestone' | 'history';
  year: number;
  yearLabel: string;
  label: string;
  href: string;
  id?: string;
  eraKind?: Era['kind'];
};

const KIND_ORDER: Record<EraStripItem['kind'], number> = {
  era: 0,
  milestone: 1,
  history: 2,
};

/**
 * Pure selector: returns the ERAS overlapping [from, to], plus MILESTONES and
 * HISTORY entries (placed at years[0], matching TimelinePage) whose year lies
 * in [from, to], sorted ascending by year.
 */
export function eraStripItems(from: number, to: number): EraStripItem[] {
  const items: EraStripItem[] = [];

  for (const e of ERAS) {
    if (e.to >= from && e.from <= to) {
      items.push({
        kind: 'era',
        year: e.from,
        yearLabel: `${e.from}–${e.to === 2026 ? 'now' : e.to}`,
        label: e.label,
        href: timelineHref(),
        eraKind: e.kind,
      });
    }
  }

  for (const m of MILESTONES) {
    if (m.year >= from && m.year <= to) {
      items.push({
        kind: 'milestone',
        year: m.year,
        yearLabel: String(m.year),
        label: m.label,
        href: timelineHref(),
      });
    }
  }

  for (const h of HISTORY) {
    const y = h.years[0];
    if (y >= from && y <= to) {
      items.push({
        kind: 'history',
        year: y,
        yearLabel: String(y),
        label: h.idea,
        href: timelineHref(h.id),
        id: h.id,
      });
    }
  }

  return items.sort(
    (a, b) =>
      a.year - b.year ||
      KIND_ORDER[a.kind] - KIND_ORDER[b.kind] ||
      (a.id ?? a.label).localeCompare(b.id ?? b.label),
  );
}

export function EraStrip({ from, to }: { from: number; to: number }) {
  const items = eraStripItems(from, to);
  const eras = items.filter((i) => i.kind === 'era');
  const events = items.filter((i) => i.kind !== 'era');

  return (
    <nav className="era-strip" aria-label={`Timeline strip for ${from} to ${to}`}>
      <div className="era-strip-head">
        <span className="era-strip-range">{from}–{to}</span>
        {eras.length > 0 && (
          <div className="era-strip-eras">
            {eras.map((e) => (
              <a
                key={`${e.yearLabel}-${e.label}`}
                href={e.href}
                className={`era-strip-era kind-${e.eraKind ?? 'founding'}`}
              >
                <span className="era-strip-year">{e.yearLabel}</span> {e.label}
              </a>
            ))}
          </div>
        )}
        <a href={timelineHref()} className="era-strip-all">
          Full timeline →
        </a>
      </div>

      {events.length > 0 && (
        <ol className="era-strip-list">
          {events.map((item) => (
            <li key={`${item.kind}-${item.year}-${item.id ?? item.label}`}>
              <a href={item.href} className={`era-strip-item is-${item.kind}`}>
                <span className="era-strip-year">{item.yearLabel}</span>
                <span className="era-strip-label">{item.label}</span>
              </a>
            </li>
          ))}
        </ol>
      )}
    </nav>
  );
}
