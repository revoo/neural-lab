/**
 * Components available in every lesson's MDX without importing them.
 * Lesson-specific widgets are imported explicitly inside each lesson.
 */
import type { ComponentProps } from 'react';
import { Callout, Analogy, Insight, Misconception, InTheWild, Takeaways, Deeper } from './components/Callout';
import { Predict } from './components/Predict';
import { Quiz } from './components/Quiz';
import { Term, LessonLink } from './components/Term';
import { Tex } from './components/Tex';

export const mdxComponents = {
  Callout,
  Analogy,
  Insight,
  Misconception,
  InTheWild,
  Takeaways,
  Deeper,
  Predict,
  Quiz,
  Term,
  LessonLink,
  Tex,
  a: (p: ComponentProps<'a'>) => <a {...p} target={p.href?.startsWith('http') ? '_blank' : undefined} rel={p.href?.startsWith('http') ? 'noreferrer' : undefined} />,
};
