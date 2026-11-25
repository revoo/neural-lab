/**
 * Components available in every lesson's MDX without importing them.
 * Lesson-specific widgets are imported explicitly inside each lesson.
 */
import type { ComponentProps } from 'react';
import { Callout, Analogy, Insight, Misconception, InTheWild, Takeaways, Deeper, MathExhibit } from './components/Callout';
import { Predict } from './components/Predict';
import { Quiz } from './components/Quiz';
import { Term, LessonLink, ModuleRef } from './components/Term';
import { Src } from './components/Src';
import { MachineMap } from './components/MachineMap';
import { Bridge } from './components/Bridge';
import { Tex } from './components/Tex';
import { Eq } from './components/Eq';
import { LabJob, ForReal } from './components/LabJob';
import { HistoryCard } from './components/HistoryCard';
import { RivalTable } from './components/RivalTable';
import { EraStrip } from './components/EraStrip';

export const mdxComponents = {
  Callout,
  Analogy,
  Insight,
  Misconception,
  InTheWild,
  Takeaways,
  Deeper,
  MathExhibit,
  Predict,
  Quiz,
  Term,
  LessonLink,
  ModuleRef,
  Src,
  MachineMap,
  Bridge,
  Tex,
  Eq,
  LabJob,
  ForReal,
  HistoryCard,
  RivalTable,
  EraStrip,
  a: (p: ComponentProps<'a'>) => <a {...p} target={p.href?.startsWith('http') ? '_blank' : undefined} rel={p.href?.startsWith('http') ? 'noreferrer' : undefined} />,
};
