// authored-by: Claude Opus 5.5
import { AREAS, lessonArea, type Area } from '../areas';

/**
 * A "where this belongs" chip (V2-WP1, as revised by the owner): Foundational, or the lesson's
 * home such as "Computer Vision foundation". `short` uses the one-word label for tight lists.
 */
export function AreaChip({ area, short = false }: { area: Area; short?: boolean }) {
  const a = AREAS[area];
  return (
    <span className={`area-chip area-${area}`} style={{ ['--chip' as string]: a.colour }} title={`${a.label}: ${a.blurb}`}>
      {short ? a.short : a.label}
    </span>
  );
}

export const LessonChip = ({ id, short = false }: { id: string; short?: boolean }) => <AreaChip area={lessonArea(id)} short={short} />;
