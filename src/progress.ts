/**
 * Learner progress, persisted in localStorage.
 * Lesson completion + per-question review state (a simple Leitner spaced-repetition scheme).
 */
import { useSyncExternalStore } from 'react';

export type QuestionRecord = {
  box: number; // 1..5 — higher means better known
  due: number; // epoch ms when it should be reviewed again
  attempts: number;
  correct: number;
  last: number;
  lastCorrect: boolean;
};

export type ProgressState = {
  completed: Record<string, number>;
  questions: Record<string, QuestionRecord>;
};

const KEY = 'nl-progress-v1';
const DAY = 24 * 60 * 60 * 1000;
/** Review interval (days) for each Leitner box. Box 1 is "review at the next session". */
export const BOX_INTERVAL_DAYS = [0, 0, 1, 3, 8, 21];

function load(): ProgressState {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const s = JSON.parse(raw);
      return { completed: s.completed ?? {}, questions: s.questions ?? {} };
    }
  } catch {
    /* storage unavailable: run without persistence */
  }
  return { completed: {}, questions: {} };
}

let state: ProgressState = load();
const listeners = new Set<() => void>();

function set(next: ProgressState) {
  state = next;
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    /* ignore */
  }
  listeners.forEach((l) => l());
}

export const progressStore = {
  get: () => state,
  subscribe(l: () => void) {
    listeners.add(l);
    return () => listeners.delete(l);
  },
  setCompleted(lessonId: string, done: boolean) {
    const completed = { ...state.completed };
    if (done) completed[lessonId] = Date.now();
    else delete completed[lessonId];
    set({ ...state, completed });
  },
  recordAnswer(questionId: string, correct: boolean, now = Date.now()) {
    set({ ...state, questions: { ...state.questions, [questionId]: nextRecord(state.questions[questionId], correct, now) } });
  },
  reset() {
    set({ completed: {}, questions: {} });
  },
};

export function nextRecord(prev: QuestionRecord | undefined, correct: boolean, now: number): QuestionRecord {
  const box = correct ? Math.min(5, (prev?.box ?? 1) + 1) : 1;
  return {
    box,
    due: now + BOX_INTERVAL_DAYS[box] * DAY,
    attempts: (prev?.attempts ?? 0) + 1,
    correct: (prev?.correct ?? 0) + (correct ? 1 : 0),
    last: now,
    lastCorrect: correct,
  };
}

export function useProgress(): ProgressState {
  return useSyncExternalStore(progressStore.subscribe, progressStore.get, progressStore.get);
}
