// authored-by: Claude Opus 5.5
import type { LessonMeta, Track } from '../curriculum';
import { TRACK_LABEL, trackStore, useTrack } from '../track';

/** Full Course vs Core Track switch (V2-WP1). `compact` is the sidebar version. */
export function TrackToggle({ compact = false }: { compact?: boolean }) {
  const track = useTrack();
  const opts: { t: Track; hint: string }[] = [
    { t: 'full', hint: 'Every lesson, including historical bridges and interludes' },
    { t: 'core', hint: 'Skip optional lessons in navigation and progress (they stay clickable)' },
  ];
  return (
    <div className={`track-toggle${compact ? ' compact' : ''}`} role="radiogroup" aria-label="Learning track">
      {opts.map(({ t, hint }) => (
        <button key={t} role="radio" aria-checked={track === t} className={track === t ? 'on' : ''} title={hint} onClick={() => trackStore.set(t)}>
          {TRACK_LABEL[t]}
        </button>
      ))}
    </div>
  );
}

/** The small "Optional" pill, with the lesson's reason as a tooltip. */
export function OptionalPill({ lesson }: { lesson: LessonMeta }) {
  if (!lesson.optional) return null;
  return (
    <span className="pill optional-pill" title={lesson.optionalNote} aria-label={`Optional: ${lesson.optionalNote}`}>
      Optional
    </span>
  );
}
