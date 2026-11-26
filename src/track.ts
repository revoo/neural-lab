// authored-by: Claude Opus 5.5
/**
 * V2-WP1: which guided track the learner follows, persisted in localStorage under `nl-track-v1`.
 * `full` (the default) is every lesson; `core` skips optional lessons in navigation and progress.
 */
import { useSyncExternalStore } from 'react';
import type { Track } from './curriculum';

const KEY = 'nl-track-v1';

function load(): Track {
  try {
    return localStorage.getItem(KEY) === 'core' ? 'core' : 'full';
  } catch {
    return 'full';
  }
}

let track: Track = typeof localStorage === 'undefined' ? 'full' : load();
const listeners = new Set<() => void>();

export const trackStore = {
  get: () => track,
  set(next: Track) {
    track = next;
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* storage unavailable */
    }
    listeners.forEach((l) => l());
  },
  subscribe(l: () => void) {
    listeners.add(l);
    return () => listeners.delete(l);
  },
};

export function useTrack(): Track {
  return useSyncExternalStore(trackStore.subscribe, trackStore.get, trackStore.get);
}

export const TRACK_LABEL: Record<Track, string> = { full: 'Full Course', core: 'Core Track' };
