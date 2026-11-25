// authored-by: Claude Opus 5.5
import { useCallback, useEffect, useRef, useState } from 'react';

export type LogTone = 'info' | 'good' | 'bad';
export type LogLine = { id: number; text: string; tone: LogTone };

/** A running log of what the machine did, one plain-English line per event. */
export function useMachineLog(max = 200) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const next = useRef(1);
  const push = useCallback(
    (text: string, tone: LogTone = 'info') => {
      const id = next.current++;
      setLines((ls) => [...ls, { id, text, tone }].slice(-max));
    },
    [max],
  );
  const clear = useCallback(() => setLines([]), []);
  return { lines, push, clear };
}

/**
 * A collapsible monospace console. The newest line is always visible in the
 * summary, so the log stays live even when collapsed; opened, it keeps the
 * newest line in view.
 */
export function MachineLog({
  lines,
  onClear,
  title = 'Machine log',
  empty = 'Nothing has happened yet.',
  defaultOpen = false,
}: {
  lines: LogLine[];
  onClear?: () => void;
  title?: string;
  empty?: string;
  defaultOpen?: boolean;
}) {
  const body = useRef<HTMLDivElement>(null);
  const last = lines[lines.length - 1];
  useEffect(() => {
    const el = body.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);
  return (
    <details className="machine-log" open={defaultOpen}>
      <summary>
        <span className="machine-log-title">{title}</span>
        <span className={`machine-log-last mono tone-${last?.tone ?? 'info'}`}>{last ? last.text : empty}</span>
      </summary>
      <div className="machine-log-body mono" ref={body} role="log" aria-live="polite">
        {lines.length === 0 ? (
          <div className="machine-log-line tone-info">{empty}</div>
        ) : (
          lines.map((l) => (
            <div key={l.id} className={`machine-log-line tone-${l.tone}`}>
              {l.text}
            </div>
          ))
        )}
      </div>
      {onClear && lines.length > 0 && (
        <div className="machine-log-foot">
          <button type="button" className="btn small" onClick={onClear}>
            Clear log
          </button>
        </div>
      )}
    </details>
  );
}
