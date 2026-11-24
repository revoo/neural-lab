import { useEffect, useRef } from 'react';

/**
 * The matrix theme's backdrop: columns of falling glyphs on a fixed, full-viewport canvas behind the
 * app. Purely decorative (aria-hidden) — the reading column is painted opaque on top of it (see
 * `.page` in the matrix block of global.css), so the rain lives in the margins and never behind text.
 *
 * The glyphs are the course's own alphabet: Greek letters and operators from the equations, digits,
 * and hex. Now and then a column rains a real term ("BACKPROP", "∇L", "σ(w·x+b)") top to bottom, so
 * if you look closely the rain is made of the things you're learning.
 *
 * Cheap by design: ~24 fps, one fillText per lit cell, paused while the tab is hidden; with
 * `prefers-reduced-motion` (or `animate=false`) it draws a single still frame.
 */

const GLYPHS = '0123456789ABCDEFθλσΣ∂∇ηεμπφψωΔ∫√≈±×÷·→∞⊗αβγ'.split('');
const WORDS = [
  'NEURON', 'WEIGHT', 'BIAS', 'RELU', 'SIGMOID', 'SOFTMAX', 'BACKPROP', 'GRADIENT', 'ADAM', 'LOSS',
  'EPOCH', 'TENSOR', 'DROPOUT', 'ATTENTION', '∇L', 'σ(w·x+b)', 'w ← w − η∇L', 'Σ wᵢxᵢ', '∂L/∂w', 'y = Wx + b',
];

const CELL = 18; // px between columns and rows
const FONT = 15;
const FPS = 24;

interface Drop {
  head: number; // row of the leading glyph (fractional)
  speed: number; // rows per frame
  len: number; // trail length in rows
  word: string | null; // if set, the column spells this word
  seed: number; // per-drop offset into the glyph table
  wait: number; // frames to wait before (re)starting
}

export function DigitalRain({ animate = true }: { animate?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    const moving = animate && !reduced;

    let w = 0;
    let h = 0;
    let rows = 0;
    let drops: Drop[] = [];
    // A slowly mutating glyph grid, so trails flicker like the real thing without random() per cell per frame.
    let grid: string[] = [];
    const rand = Math.random;

    const newDrop = (initial: boolean): Drop => {
      // pad with spaces so a repeating word reads as separate words down the column
      const word = rand() < 0.12 ? WORDS[Math.floor(rand() * WORDS.length)] + '   ' : null;
      return {
        head: initial ? rand() * rows : -rand() * 20,
        speed: 0.18 + rand() * 0.42,
        len: word ? Math.max(word.length + 6, 14) : 8 + Math.floor(rand() * 18),
        word,
        seed: Math.floor(rand() * 1000),
        wait: initial ? 0 : Math.floor(rand() * 60),
      };
    };

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      w = window.innerWidth;
      h = window.innerHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const cols = Math.ceil(w / CELL);
      rows = Math.ceil(h / CELL) + 1;
      drops = Array.from({ length: cols }, () => newDrop(true));
      grid = Array.from({ length: cols * rows }, () => GLYPHS[Math.floor(rand() * GLYPHS.length)]);
    };

    const draw = () => {
      ctx.fillStyle = '#030805';
      ctx.fillRect(0, 0, w, h);
      ctx.font = `${FONT}px 'JetBrains Mono Variable', ui-monospace, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (let c = 0; c < drops.length; c++) {
        const d = drops[c];
        const head = Math.floor(d.head);
        const x = c * CELL + CELL / 2;
        for (let k = 0; k < d.len; k++) {
          const r = head - k;
          if (r < 0 || r >= rows) continue;
          const glyph = d.word ? d.word[((r % d.word.length) + d.word.length) % d.word.length] : grid[c * rows + r];
          if (glyph === ' ') continue;
          const fade = 1 - k / d.len;
          if (k === 0) {
            ctx.fillStyle = 'rgba(215, 255, 228, 0.95)';
          } else {
            const a = 0.75 * fade * fade;
            ctx.fillStyle = d.word ? `rgba(120, 255, 170, ${a + 0.08})` : `rgba(40, 220, 110, ${a})`;
          }
          ctx.fillText(glyph, x, r * CELL + CELL / 2);
        }
      }
    };

    const step = () => {
      for (let c = 0; c < drops.length; c++) {
        const d = drops[c];
        if (d.wait > 0) {
          d.wait--;
          continue;
        }
        d.head += d.speed;
        if (d.head - d.len > rows) drops[c] = newDrop(false);
      }
      // mutate a few glyphs per frame
      for (let i = 0; i < 24; i++) grid[Math.floor(rand() * grid.length)] = GLYPHS[Math.floor(rand() * GLYPHS.length)];
    };

    resize();
    draw();
    window.addEventListener('resize', resize);
    if (!moving) {
      const onResize = () => draw();
      window.addEventListener('resize', onResize);
      return () => {
        window.removeEventListener('resize', resize);
        window.removeEventListener('resize', onResize);
      };
    }

    let raf = 0;
    let last = 0;
    const loop = (t: number) => {
      raf = requestAnimationFrame(loop);
      if (document.hidden || t - last < 1000 / FPS) return;
      last = t;
      step();
      draw();
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    };
  }, [animate]);

  return <canvas ref={ref} className="digital-rain" aria-hidden="true" />;
}
