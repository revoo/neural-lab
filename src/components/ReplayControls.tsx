// authored-by: Claude Opus 5.5
import type { Replay } from './motion';

/** Play / pause, step back and forward, and a scrubber for a useReplay() replay. */
export function ReplayControls({ r, label = 'step' }: { r: Replay; label?: string }) {
  return (
    <div className="replay-controls">
      <button type="button" className="btn small" onClick={r.reset} aria-label="Rewind to the start" disabled={r.i === 0}>
        ⏮
      </button>
      <button type="button" className="btn small" onClick={r.back} aria-label="Step back" disabled={r.i === 0}>
        ◀
      </button>
      <button type="button" className="btn small primary" onClick={r.playing ? r.pause : r.play} aria-label={r.playing ? 'Pause' : 'Play'}>
        {r.playing ? '❚❚' : '▶'}
      </button>
      <button type="button" className="btn small" onClick={r.step} aria-label="Step forward" disabled={r.i >= r.frames}>
        ▶|
      </button>
      <input type="range" min={0} max={r.frames} value={r.i} onChange={(e) => r.seek(Number(e.target.value))} aria-label={`Scrub to ${label}`} style={{ flex: 1, minWidth: 80 }} />
      <span className="mono" style={{ fontSize: 12, color: 'var(--text-muted)', minWidth: 70, textAlign: 'right' }}>
        {label} {r.i} / {r.frames}
      </span>
    </div>
  );
}
