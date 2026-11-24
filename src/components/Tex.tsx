import katex from 'katex';
import { memo } from 'react';

/** Render a LaTeX string with KaTeX. Used for live equations inside widgets. */
export const Tex = memo(function Tex({ children, block = false, className }: { children: string; block?: boolean; className?: string }) {
  const html = katex.renderToString(children, { displayMode: block, throwOnError: false, strict: false, trust: true });
  const Tag = block ? 'div' : 'span';
  return <Tag className={className ?? (block ? 'tex-block' : undefined)} dangerouslySetInnerHTML={{ __html: html }} />;
});

/** Format a number for display inside LaTeX (handles negative zero and sign). */
export function fmt(x: number, digits = 2): string {
  if (!Number.isFinite(x)) return x > 0 ? '\\infty' : x < 0 ? '-\\infty' : '\\text{NaN}';
  const s = x.toFixed(digits);
  return /^-0(\.0+)?$/.test(s) ? s.slice(1) : s;
}

/** Wrap a LaTeX fragment in a colour taken from a CSS variable name (e.g. 'c-a'). */
export function col(name: string, tex: string): string {
  return `\\htmlStyle{color: var(--${name})}{${tex}}`;
}

/** Parenthesise negative numbers so substitutions read correctly: 3 × (−2). */
export function pn(x: number, digits = 2): string {
  const s = fmt(x, digits);
  return x < 0 && s !== fmt(0, digits) ? `(${s})` : s;
}
