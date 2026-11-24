import type { Rng } from '../lib/rng';

/**
 * Question formats. Questions are plain data (in each lesson's quiz.ts) so the review page can
 * collect them across the curriculum. Text fields support $math$, **bold** and `code`.
 */
type Base = {
  /** Globally unique, conventionally `<lessonId>.<slug>`. */
  id: string;
  concepts?: string[];
};

export type ChoiceQuestion = Base & {
  kind: 'choice';
  prompt: string;
  options: { text: string; correct?: boolean; /** Why this option is right, or which misconception it reflects. */ why?: string }[];
  explain?: string;
  shuffle?: boolean;
};

export type NumericQuestion = Base & {
  kind: 'numeric';
  /** Generates a (possibly randomised) instance. Use the rng for parameterised questions. */
  make: (rng: Rng) => { prompt: string; answer: number; explain: string };
  /** Absolute tolerance (default 0.01). */
  tolerance?: number;
};

export type OrderQuestion = Base & {
  kind: 'order';
  prompt: string;
  /** Items in the correct order. */
  items: string[];
  explain: string;
};

export type Question = ChoiceQuestion | NumericQuestion | OrderQuestion;
