# Neural Lab

An interactive place to understand AI from first principles — Feynman-style explanations, 3Blue1Brown-style visuals, and Zachtronics-style labs where you build the machinery yourself. Written for software engineers and lifelong learners.

Every visualisation computes real numbers: real gradients, real softmax, real networks trained live in the browser.

## Running it

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # maths and logic tests (Vitest)
npm run build      # typecheck + production build into dist/ (static, hash-routed — deploy anywhere)
```

## How it's organised

```
src/
  curriculum.ts        modules and lessons (lazy-loaded); add new lessons here
  glossary.ts          plain-language definitions used by <Term> tooltips and the glossary page
  katexMacros.ts       colour macros (\ca, \cb, …) for colour-coded equations
  lib/                 shared maths: linear algebra, autodiff, MLP, optimizers, probability — all tested
  components/          teaching components (Widget, Eq, Predict, Quiz, MathExhibit, …) and SVG plotting
  lessons/<id>/        one self-contained folder per lesson: index.mdx + widgets + quiz.ts (+ tests)
  pages/               app shell: home, lesson page, glossary, spaced-repetition review
docs/
  STYLE_GUIDE.md       the editorial and design rules — read before writing content
  ROADMAP.md           what's done, what's next, decisions and known limitations
```

Lessons are MDX: prose in Markdown with interactive React components dropped in. Maths in `$…$` is rendered at build time by KaTeX.
