// authored-by: Claude Opus 5.5
import { useState, type ReactNode } from 'react';

const STUCK_PREFIX = 'neural-lab:stuck:';

function readStuck(key: string): boolean {
  try {
    return localStorage.getItem(STUCK_PREFIX + key) === '1';
  } catch {
    return false;
  }
}

/**
 * "I'm stuck" button: asks once more before giving up on a brief, then
 * remembers the choice (per `stuckKey`) and calls `onStuck`.
 */
export function StuckButton({ stuckKey, label, onStuck }: { stuckKey: string; label: string; onStuck: () => void }) {
  const [confirming, setConfirming] = useState(false);
  if (!confirming) {
    return (
      <button type="button" className="btn" onClick={() => setConfirming(true)}>
        I’m stuck
      </button>
    );
  }
  return (
    <>
      <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>This shows {label} for this brief. Sure?</span>
      <button
        type="button"
        className="btn"
        onClick={() => {
          try {
            localStorage.setItem(STUCK_PREFIX + stuckKey, '1');
          } catch {
            /* storage blocked: reveal for this visit only */
          }
          setConfirming(false);
          onStuck();
        }}
      >
        Yes, show me
      </button>
      <button type="button" className="btn" onClick={() => setConfirming(false)}>
        Keep trying
      </button>
    </>
  );
}

/** Whether the learner has given up on this brief before (re-read when the key changes). */
export function useStuck(stuckKey: string): [boolean, () => void] {
  const [state, setState] = useState(() => ({ key: stuckKey, stuck: readStuck(stuckKey) }));
  if (state.key !== stuckKey) {
    const next = { key: stuckKey, stuck: readStuck(stuckKey) };
    setState(next);
    return [next.stuck, () => setState({ key: stuckKey, stuck: true })];
  }
  return [state.stuck, () => setState({ key: stuckKey, stuck: true })];
}

/**
 * The answer comes after a solve. Before the brief is solved the learner sees
 * only an "I'm stuck" button, which asks once more before revealing and is
 * remembered per brief (`stuckKey`). After a solve, or after giving up, a plain
 * show/hide toggle appears.
 */
export function AnswerReveal({
  solved,
  stuckKey,
  label,
  children,
}: {
  solved: boolean;
  /** Unique per lab and brief, e.g. "lora:brief-1". */
  stuckKey: string;
  /** What is revealed, e.g. "the exhaustive optimum". */
  label: string;
  children: ReactNode;
}) {
  const [stuck, markStuck] = useStuck(stuckKey);
  const [open, setOpen] = useState(false);
  const [openKey, setOpenKey] = useState(stuckKey);
  if (openKey !== stuckKey) {
    setOpenKey(stuckKey);
    setOpen(false);
  }

  if (!(solved || stuck)) {
    return (
      <div className="btn-row" style={{ marginTop: 10, alignItems: 'center' }}>
        <StuckButton
          stuckKey={stuckKey}
          label={label}
          onStuck={() => {
            markStuck();
            setOpen(true);
          }}
        />
      </div>
    );
  }

  return (
    <>
      <div className="btn-row" style={{ marginTop: 10 }}>
        <button type="button" className="btn" onClick={() => setOpen((o) => !o)}>
          {open ? `Hide ${label}` : `Show ${label}`}
        </button>
      </div>
      {open && children}
    </>
  );
}
