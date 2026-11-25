// authored-by: Claude Opus 5.5
import type { ReactNode } from 'react';

/** The real engineering job a lab stands for: one line placed just above the lab widget. */
export function LabJob({ children }: { children: ReactNode }) {
  return (
    <div className="lab-job">
      <span className="lab-job-tag">The job</span>
      <span>{children}</span>
    </div>
  );
}

/** After the lab: where the same problem turns up in real AI work. */
export function ForReal({ children }: { children: ReactNode }) {
  return (
    <aside className="callout wild">
      <div className="callout-title">Where this happens for real</div>
      {children}
    </aside>
  );
}

/**
 * The real line of code the current lab settings correspond to, updated live.
 * `note` says which library and version the call was checked against.
 */
export function LiveCode({ code, note }: { code: string; note?: ReactNode }) {
  return (
    <div className="live-code">
      <div className="live-code-head">
        <span>The same settings in real code</span>
        {note && <span className="live-code-note">{note}</span>}
      </div>
      <pre className="mono">{code}</pre>
    </div>
  );
}
