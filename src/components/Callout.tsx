import type { ReactNode } from 'react';

type Kind = 'note' | 'analogy' | 'insight' | 'misconception' | 'wild' | 'takeaways';
const DEFAULT_TITLES: Record<Kind, string> = {
  note: 'Note',
  analogy: 'Think of it this way',
  insight: 'The key insight',
  misconception: 'A tempting misconception',
  wild: 'In the real world',
  takeaways: 'What to carry forward',
};

export function Callout({ kind = 'note', title, children }: { kind?: Kind; title?: string; children: ReactNode }) {
  return (
    <aside className={`callout ${kind}`}>
      <div className="callout-title">{title ?? DEFAULT_TITLES[kind]}</div>
      {children}
    </aside>
  );
}

export const Analogy = (p: { title?: string; children: ReactNode }) => <Callout kind="analogy" {...p} />;
export const Insight = (p: { title?: string; children: ReactNode }) => <Callout kind="insight" {...p} />;
export const Misconception = (p: { title?: string; children: ReactNode }) => <Callout kind="misconception" {...p} />;
export const InTheWild = (p: { title?: string; children: ReactNode }) => <Callout kind="wild" {...p} />;
export const Takeaways = (p: { title?: string; children: ReactNode }) => <Callout kind="takeaways" {...p} />;

/** Collapsible section for optional depth: maths, code, history, edge cases. */
export function Deeper({ title, tag = 'Go deeper', children, open = false }: { title: string; tag?: string; children: ReactNode; open?: boolean }) {
  return (
    <details className="deeper" open={open}>
      <summary>
        <span style={{ flex: 1 }}>{title}</span>
        <span className="tag">{tag}</span>
      </summary>
      <div className="deeper-body">{children}</div>
    </details>
  );
}
