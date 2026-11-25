// authored-by: Claude Opus 5.5
/** A source link for a historical claim written in lesson prose. */
export function Src({ href }: { href: string }) {
  return (
    <sup className="src-link">
      <a href={href} target="_blank" rel="noreferrer" aria-label="source">
        [source]
      </a>
    </sup>
  );
}
