import { useCallback, useEffect, useState } from 'react';

/**
 * Colour themes. `light` and `dark` are the everyday themes; `matrix` is the green-phosphor terminal
 * theme with digital rain (see `components/DigitalRain.tsx` and the `[data-theme='matrix']` block in
 * `styles/global.css`). The choice is stored in localStorage under `nl-theme`; `index.html` applies it
 * before first paint so there's no flash.
 */
export type Theme = 'light' | 'dark' | 'matrix';
export const THEMES: Theme[] = ['light', 'dark', 'matrix'];

const KEY = 'nl-theme';
const RAIN_KEY = 'nl-rain';

function initialTheme(): Theme {
  const t = document.documentElement.dataset.theme;
  if (t === 'light' || t === 'dark' || t === 'matrix') return t;
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(initialTheme);
  const setTheme = useCallback((next: Theme) => {
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* ignore */
    }
    setThemeState(next);
  }, []);
  return { theme, setTheme };
}

/** Whether the digital rain animates (matrix theme only). Defaults to on; persisted. */
export function useRain() {
  const [on, setOn] = useState<boolean>(() => {
    try {
      return localStorage.getItem(RAIN_KEY) !== 'off';
    } catch {
      return true;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(RAIN_KEY, on ? 'on' : 'off');
    } catch {
      /* ignore */
    }
  }, [on]);
  return { rain: on, setRain: setOn };
}

export function prefersReducedMotion() {
  return !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}
