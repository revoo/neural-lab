import type { CSSProperties, ReactNode } from 'react';

export function Slider({
  label,
  value,
  min,
  max,
  step = 0.01,
  onChange,
  format = (v) => v.toFixed(2),
  color,
}: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  color?: string; // CSS variable name, e.g. 'c-a'
}) {
  const style = color ? ({ '--slider-color': `var(--${color})` } as CSSProperties) : undefined;
  return (
    <label className="slider" data-color={color} style={style}>
      <span className="slider-top">
        <span className="slider-label">{label}</span>
        <span className="slider-value" style={color ? { color: `var(--${color})` } : undefined}>
          {format(value)}
        </span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  label,
}: {
  options: { value: T; label: ReactNode }[] | readonly T[];
  value: T;
  onChange: (v: T) => void;
  label?: string;
}) {
  const opts = (options as unknown[]).map((o) =>
    typeof o === 'object' && o !== null ? (o as { value: T; label: ReactNode }) : { value: o as T, label: String(o) },
  );
  return (
    <div>
      {label && <div className="field-label">{label}</div>}
      <div className="seg" role="radiogroup" aria-label={label}>
        {opts.map((o) => (
          <button
            key={String(o.value)}
            type="button"
            role="radio"
            aria-checked={o.value === value}
            className={o.value === value ? 'on' : ''}
            onClick={() => onChange(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Toggle({ label, checked, onChange }: { label: ReactNode; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontSize: 13.5, cursor: 'pointer' }}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} style={{ accentColor: 'var(--accent)' }} />
      {label}
    </label>
  );
}
