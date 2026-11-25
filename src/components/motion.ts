// authored-by: Claude Opus 5.5
import { useEffect, useRef, useState } from 'react';

/**
 * Shared motion for labs. Motion must carry meaning (a path drawn step by step,
 * bars easing to their new heights); under `prefers-reduced-motion` everything
 * here jumps straight to its end state. global.css also zeroes CSS transitions
 * inside widgets under the same media query.
 */
export function usePrefersReducedMotion(): boolean {
  const query = '(prefers-reduced-motion: reduce)';
  const get = () => typeof window !== 'undefined' && !!window.matchMedia && window.matchMedia(query).matches;
  const [reduced, setReduced] = useState(get);
  useEffect(() => {
    if (!window.matchMedia) return;
    const mq = window.matchMedia(query);
    const on = () => setReduced(mq.matches);
    mq.addEventListener?.('change', on);
    return () => mq.removeEventListener?.('change', on);
  }, []);
  return reduced;
}

const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

/**
 * Ease an array of numbers to new values over `ms` (150–400 ms reads as "this
 * moved that way"). Arrays of a different length snap. Instant when reduced.
 */
export function useTweenedArray(target: readonly number[], ms = 280): number[] {
  const reduced = usePrefersReducedMotion();
  const [shown, setShown] = useState<number[]>(() => [...target]);
  const fromRef = useRef<number[]>([...target]);
  const shownRef = useRef<number[]>(shown);
  shownRef.current = shown;
  const key = target.join(',');
  useEffect(() => {
    const from = shownRef.current;
    if (reduced || from.length !== target.length) {
      setShown([...target]);
      return;
    }
    fromRef.current = [...from];
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      const k = ease(t);
      setShown(target.map((v, i) => fromRef.current[i] + (v - fromRef.current[i]) * k));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, reduced, ms]);
  return shown;
}

/** Ease a single number (see useTweenedArray). */
export function useTweened(target: number, ms = 280): number {
  return useTweenedArray([target], ms)[0];
}

/**
 * Run `frames` steps of a process over time: `onFrame(i)` is called for
 * i = 1 … frames, every `frameMs`. Returns `start()`, `running`. Under reduced
 * motion `start()` calls only the last frame.
 */
export function useFramePlayer(frames: number, frameMs: number, onFrame: (i: number) => void) {
  const reduced = usePrefersReducedMotion();
  const [running, setRunning] = useState(false);
  const timer = useRef<number | null>(null);
  const cb = useRef(onFrame);
  cb.current = onFrame;
  useEffect(() => () => {
    if (timer.current !== null) window.clearInterval(timer.current);
  }, []);
  const start = () => {
    if (timer.current !== null) window.clearInterval(timer.current);
    if (reduced || frames <= 1) {
      cb.current(frames);
      setRunning(false);
      return;
    }
    let i = 0;
    setRunning(true);
    timer.current = window.setInterval(() => {
      i++;
      cb.current(i);
      if (i >= frames) {
        window.clearInterval(timer.current!);
        timer.current = null;
        setRunning(false);
      }
    }, frameMs);
  };
  return { start, running, reduced };
}
