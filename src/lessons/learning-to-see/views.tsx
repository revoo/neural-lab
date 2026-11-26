// authored-by: Claude Opus 5.5
/**
 * Small drawing pieces for "learning to see": a theme-aware pixel grid (canvas; colours read from
 * CSS variables and redrawn when the theme changes) and a 28 × 28 draw pad.
 */
import { useEffect, useRef, useState } from 'react';

export function useThemeKey() {
  const [k, setK] = useState(() => document.documentElement.dataset.theme ?? '');
  useEffect(() => {
    const obs = new MutationObserver(() => setK(document.documentElement.dataset.theme ?? ''));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => obs.disconnect();
  }, []);
  return k;
}

export type GridMode = 'grey' | 'signed' | 'heat';

/**
 * Draws an n × n array. grey: 0 → background, 1 → text colour; signed: positive in --c-a, negative in
 * --c-b, strength by |v| / scale; heat: --c-bad with strength v / scale (negatives faint --c-a).
 */
export function PixelGrid({ v, n, mode, scale, size = 84, label, onPixel, highlight }: { v: ArrayLike<number>; n: number; mode: GridMode; scale?: number; size?: number | string; label: string; onPixel?: (i: number, e: React.PointerEvent) => void; highlight?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const theme = useThemeKey();
  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext('2d');
    if (!c || !ctx) return;
    const css = getComputedStyle(c);
    const col = (name: string, fb: string) => css.getPropertyValue(name).trim() || fb;
    const bg = col('--bg-elev', '#fff');
    const fg = col('--text', '#000');
    const pos = col('--c-a', 'blue');
    const neg = col('--c-b', 'orange');
    const hot = col('--c-bad', 'red');
    let s = scale ?? 0;
    if (!s) for (let i = 0; i < n * n; i++) s = Math.max(s, Math.abs(v[i]));
    s = s || 1;
    ctx.globalAlpha = 1;
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, n, n);
    for (let i = 0; i < n * n; i++) {
      const x = i % n;
      const y = (i - x) / n;
      const val = v[i];
      if (mode === 'grey') {
        ctx.globalAlpha = Math.max(0, Math.min(1, val));
        ctx.fillStyle = fg;
      } else if (mode === 'signed') {
        ctx.globalAlpha = Math.min(1, Math.abs(val) / s);
        ctx.fillStyle = val >= 0 ? pos : neg;
      } else {
        ctx.globalAlpha = Math.min(1, Math.abs(val) / s);
        ctx.fillStyle = val >= 0 ? hot : pos;
      }
      if (ctx.globalAlpha > 0.004) ctx.fillRect(x, y, 1, 1);
    }
    ctx.globalAlpha = 1;
  }, [v, n, mode, scale, theme]);
  const pick = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!onPixel) return;
    const r = e.currentTarget.getBoundingClientRect();
    const x = Math.floor(((e.clientX - r.left) / r.width) * n);
    const y = Math.floor(((e.clientY - r.top) / r.height) * n);
    if (x >= 0 && y >= 0 && x < n && y < n) onPixel(y * n + x, e);
  };
  return (
    <canvas
      ref={ref}
      width={n}
      height={n}
      role="img"
      aria-label={label}
      onPointerDown={onPixel ? (e) => (e.currentTarget.setPointerCapture(e.pointerId), pick(e)) : undefined}
      onPointerMove={onPixel ? (e) => e.buttons && pick(e) : undefined}
      style={{ width: size, height: size, imageRendering: 'pixelated', border: `1px solid ${highlight ? 'var(--accent)' : 'var(--border)'}`, borderRadius: 3, touchAction: onPixel ? 'none' : undefined, cursor: onPixel ? 'crosshair' : undefined, display: 'block' }}
    />
  );
}

/** A 28 × 28 pad: drag to paint with a soft round brush (Fashion-MNIST items are light on dark). */
export function DrawPad({ img, onChange, size = 168 }: { img: Float32Array; onChange: (next: Float32Array) => void; size?: number }) {
  const paint = (i: number, e: React.PointerEvent) => {
    const n = 28;
    const cx = i % n;
    const cy = (i - cx) / n;
    const next = Float32Array.from(img);
    const erase = e.shiftKey || e.button === 2;
    for (let y = cy - 2; y <= cy + 2; y++)
      for (let x = cx - 2; x <= cx + 2; x++) {
        if (x < 0 || y < 0 || x >= n || y >= n) continue;
        const d = Math.hypot(x - cx, y - cy);
        const w = Math.max(0, 1 - d / 2.4);
        const k = y * n + x;
        next[k] = erase ? next[k] * (1 - w) : Math.min(1, next[k] + w * 0.8);
      }
    onChange(next);
  };
  return <PixelGrid v={img} n={28} mode="grey" size={size} label="Drawing pad: drag to draw" onPixel={paint} highlight />;
}
