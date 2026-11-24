import type { ReactNode } from 'react';

/** Frame around every interactive experiment: title bar, body, optional caption with guidance. */
export function Widget({
  title,
  children,
  caption,
  wide = false,
  actions,
}: {
  title: string;
  children: ReactNode;
  caption?: ReactNode;
  wide?: boolean;
  actions?: ReactNode;
}) {
  return (
    <figure className={`widget${wide ? ' wide' : ''}`} style={{ marginInline: 'auto' }}>
      <div className="widget-head">
        <span className="w-icon" aria-hidden />
        <span className="widget-title">{title}</span>
        {actions}
      </div>
      <div className="widget-body">{children}</div>
      {caption && <figcaption className="widget-caption">{caption}</figcaption>}
    </figure>
  );
}
