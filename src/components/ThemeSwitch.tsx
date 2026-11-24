import { useEffect, useState } from 'react';
import type { Theme } from '../theme';

const LABELS: Record<Theme, string> = { light: 'Light', dark: 'Dark', matrix: 'Matrix' };

function Icon({ theme }: { theme: Theme }) {
  if (theme === 'light') return <span aria-hidden="true">☀</span>;
  if (theme === 'dark') return <span aria-hidden="true">☾</span>;
  // three falling columns
  return (
    <svg aria-hidden="true" width="13" height="13" viewBox="0 0 13 13" className="matrix-icon">
      <path d="M2 1v3M2 6v2M6.5 3v5M6.5 10v2M11 0v2M11 4v5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

/** Three-way theme picker for the top bar. Labels collapse to icons on phones. */
export function ThemeSwitch({ theme, onChange }: { theme: Theme; onChange: (t: Theme) => void }) {
  return (
    <div className="theme-switch" role="radiogroup" aria-label="Colour theme">
      {(['light', 'dark', 'matrix'] as Theme[]).map((t) => (
        <button
          key={t}
          role="radio"
          aria-checked={theme === t}
          className={theme === t ? 'on' : ''}
          onClick={() => onChange(t)}
          title={t === 'matrix' ? 'Matrix: green phosphor terminal with digital rain' : `${LABELS[t]} theme`}
        >
          <Icon theme={t} />
          <span className="theme-label">{LABELS[t]}</span>
        </button>
      ))}
    </div>
  );
}

const BOOT_LINES = [
  '> neural-lab --mode=matrix',
  '  allocating tensors ........ ok',
  '  initialising weights ...... ok',
  '  checking gradients ........ ok',
  '  σ(w·x + b) online. follow the gradient.',
];

/**
 * A one-second terminal "boot" shown when the learner switches into the matrix theme. Decorative:
 * click or any key dismisses it early; it is skipped entirely with prefers-reduced-motion.
 */
export function BootSequence({ onDone }: { onDone: () => void }) {
  const [shown, setShown] = useState(1);
  useEffect(() => {
    if (shown < BOOT_LINES.length) {
      const id = window.setTimeout(() => setShown(shown + 1), 190);
      return () => window.clearTimeout(id);
    }
    const id = window.setTimeout(onDone, 650);
    return () => window.clearTimeout(id);
  }, [shown, onDone]);
  useEffect(() => {
    const k = () => onDone();
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [onDone]);
  return (
    <div className="boot" onClick={onDone} aria-hidden="true">
      <pre className="boot-term">
        {BOOT_LINES.slice(0, shown).join('\n')}
        <span className="boot-cursor">█</span>
      </pre>
    </div>
  );
}
