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

/**
 * A replay of a recorded process, frame 0 … frames: play, pause, step and
 * reset. Under reduced motion, play jumps to the end. Changing `resetKey`
 * (a new run) rewinds to frame 0 and, if `autoplay`, starts again.
 */
export function useReplay(frames: number, frameMs = 260, resetKey: unknown = undefined, autoplay = true) {
  const reduced = usePrefersReducedMotion();
  const [i, setI] = useState(autoplay && !reduced ? 0 : frames);
  const [playing, setPlaying] = useState(autoplay && !reduced);
  const lastKey = useRef(resetKey);
  useEffect(() => {
    if (lastKey.current === resetKey) return;
    lastKey.current = resetKey;
    if (reduced || !autoplay) {
      setI(frames);
      setPlaying(false);
    } else {
      setI(0);
      setPlaying(true);
    }
  }, [resetKey, frames, reduced, autoplay]);
  useEffect(() => {
    if (!playing) return;
    if (i >= frames) {
      setPlaying(false);
      return;
    }
    const t = window.setTimeout(() => setI((v) => Math.min(frames, v + 1)), frameMs);
    return () => window.clearTimeout(t);
  }, [playing, i, frames, frameMs]);
  return {
    i: Math.min(i, frames),
    frames,
    playing,
    play: () => {
      if (reduced) {
        setI(frames);
        return;
      }
      if (i >= frames) setI(0);
      setPlaying(true);
    },
    pause: () => setPlaying(false),
    step: () => {
      setPlaying(false);
      setI((v) => Math.min(frames, v + 1));
    },
    back: () => {
      setPlaying(false);
      setI((v) => Math.max(0, v - 1));
    },
    reset: () => {
      setPlaying(false);
      setI(0);
    },
    seek: (v: number) => {
      setPlaying(false);
      setI(Math.max(0, Math.min(frames, v)));
    },
  };
}
export type Replay = ReturnType<typeof useReplay>;
