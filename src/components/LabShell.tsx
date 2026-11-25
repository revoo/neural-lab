// authored-by: Claude Opus 5.5
import { useEffect, useState, type ReactNode } from 'react';
import { Widget } from './Widget';
import { AnswerReveal } from './AnswerReveal';
import { MachineLog, type LogLine } from './MachineLog';

export type LabBrief = { id: string; tab: string; title?: string; text: ReactNode };
export type LabProgressState = { solved: Record<string, boolean>; optimal: Record<string, boolean> };

/**
 * Which briefs are solved (and which hit the optional goal), saved in this
 * browser under `storageKey`. `seed` lets a lab carry over progress it saved
 * in an older format.
 */
export function useLabProgress(storageKey: string, seed?: () => Partial<LabProgressState>) {
  const [state, setState] = useState<LabProgressState>(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      if (raw) {
        const p = JSON.parse(raw);
        return { solved: p.solved ?? {}, optimal: p.optimal ?? {} };
      }
    } catch {
      /* ignore */
    }
    const s = seed?.() ?? {};
    return { solved: s.solved ?? {}, optimal: s.optimal ?? {} };
  });
  const save = (next: LabProgressState) => {
    setState(next);
    try {
      localStorage.setItem(storageKey, JSON.stringify(next));
    } catch {
      /* ignore */
    }
  };
  const mark = (id: string, optimal = false) => {
    if (state.solved[id] && (!optimal || state.optimal[id])) return;
    save({ solved: { ...state.solved, [id]: true }, optimal: optimal ? { ...state.optimal, [id]: true } : state.optimal });
  };
  return { ...state, mark };
}
export type LabProgress = ReturnType<typeof useLabProgress>;

/**
 * The shared lab frame: brief tabs with solved ticks and a count, the brief
 * text, the lab body, a status line, an optional hint (which should point at
 * evidence, never at a setting), a post-solve reveal (answers only after a
 * solve or a confirmed "I'm stuck"), and an optional machine log.
 */
export function LabShell({
  title,
  caption,
  briefs,
  active,
  onSelect,
  progress,
  passing,
  optimal = false,
  status,
  statusTone,
  hint,
  reveal,
  log,
  actions,
  children,
}: {
  title: string;
  caption?: ReactNode;
  briefs: LabBrief[];
  active: number;
  onSelect: (i: number) => void;
  progress: LabProgress;
  /** Whether the current brief's goal is met right now (recorded as solved). */
  passing: boolean;
  /** Whether the optional goal is met right now (recorded as ★). */
  optimal?: boolean;
  status: ReactNode;
  statusTone?: 'good' | 'bad' | 'neutral';
  hint?: ReactNode;
  reveal?: { label: string; content: ReactNode };
  log?: { lines: LogLine[]; onClear?: () => void; title?: string; empty?: string };
  /** Extra buttons in the action row, e.g. Reset. */
  actions?: ReactNode;
  children: ReactNode;
}) {
  const brief = briefs[active];
  const [showHint, setShowHint] = useState(false);
  const { mark } = progress;
  useEffect(() => {
    if (passing) mark(brief.id, optimal);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [passing, optimal, brief.id]);
  const solvedCount = briefs.filter((b) => progress.solved[b.id]).length;
  const tone = statusTone ?? (passing ? 'good' : 'neutral');
  return (
    <Widget title={title} wide caption={caption}>
      {briefs.length > 1 && (
        <div className="lab-shell-tabs">
          <div className="clinic-tabs" role="tablist" aria-label="Briefs">
            {briefs.map((b, i) => (
              <button
                key={b.id}
                type="button"
                role="tab"
                aria-selected={i === active}
                className={i === active ? 'on' : ''}
                onClick={() => {
                  setShowHint(false);
                  onSelect(i);
                }}
              >
                {progress.optimal[b.id] ? '★ ' : progress.solved[b.id] ? '✓ ' : ''}
                {b.tab}
              </button>
            ))}
          </div>
          <span className="lab-shell-count mono">
            {solvedCount} / {briefs.length} solved
          </span>
        </div>
      )}
      <div className="lab-shell-brief">
        {brief.title && <div className="lab-shell-brief-title">{brief.title}</div>}
        <div>{brief.text}</div>
      </div>
      {children}
      <div className={`live-note lab-shell-status tone-${tone}`}>{status}</div>
      {(hint || actions) && (
        <div className="btn-row" style={{ marginTop: 10 }}>
          {actions}
          {hint && (
            <button type="button" className="btn" onClick={() => setShowHint((h) => !h)}>
              {showHint ? 'Hide hint' : 'Hint'}
            </button>
          )}
        </div>
      )}
      {hint && showHint && <div className="lab-shell-hint">{hint}</div>}
      {reveal && (
        <AnswerReveal solved={!!progress.solved[brief.id] || passing} stuckKey={`${title}:${brief.id}`} label={reveal.label}>
          <div className="lab-shell-reveal">{reveal.content}</div>
        </AnswerReveal>
      )}
      {log && <MachineLog {...log} />}
    </Widget>
  );
}
