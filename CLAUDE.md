# Neural Lab — instructions for Claude

(Other coding agents: the same rules are in `AGENTS.md`. Keep the two files in sync.)

An interactive website teaching AI from first principles to software engineers and lifelong learners: Feynman-style explanation, 3Blue1Brown-style visuals, Zachtronics-style labs. Every visualisation computes real numbers.

## Read before doing any work

Start with **`HANDOFF.md`** — the step-by-step onboarding procedure, lesson checklist and end-of-session routine. In short:

1. `docs/VISION.md` — the owner's brief: audience, philosophy, full 18-milestone curriculum, and **the owner's clarifications** (concept first; maths first-class but unintimidating; puzzle labs).
2. `docs/STYLE_GUIDE.md` — voice, lesson structure, how to present maths, lab design rules, engineering conventions.
3. `docs/ROADMAP.md` — what's done, next steps, decisions, limitations. **Update it at the end of every session.**
4. Read one or two finished lessons before writing a new one, to match the voice — good references: `src/lessons/neuron/index.mdx`, `src/lessons/activation-functions/index.mdx`, `src/lessons/gradient-descent/index.mdx`.

## Non-negotiables

- **Concept before equation.** The learner should see the equation and think "of course it looks like that".
- **Key equations use `<Eq words="…" tex="…">` with a legend** giving every symbol a plain-English name, colour-matched with `\ca \cb \cc \cg \cp \cv` (never `\cr` — TeX reserves it). `<MathExhibit>` is only for optional deeper material.
- **Conversational, warm, precise prose.** Structural analogies, predict-before-reveal, let the learner feel the problem first, address real misconceptions. No jargon-for-its-own-sake, no forced enthusiasm.
- **No fake interactivity.** Widgets compute real values; captions say what to try; live notes explain the current state.
- **Labs/puzzles:** primitives + clear objective + observable machinery + multiple solutions; working solution first, optional optimisation goals after, with optimums computed rather than guessed. No badges or points.
- Verify numbers you quote in prose (run them) and pin them with tests; run every code snippet. Compare methods fairly and report honestly (see the style guide).
- Three themes (light, dark, matrix): colour widgets only via CSS variables and check all three plus phone width.
- **After every finished lesson: commit, push, and update the "Current state" block in `docs/ROADMAP.md`** — the session may end at any time and another agent must be able to continue (see `HANDOFF.md`).

## Commands

```bash
npm run dev     # http://localhost:5173
npm test        # Vitest: maths, simulations, lesson logic
npm run build   # typecheck + production build
node scripts/check-pages.cjs   # with dev server running: loads every lesson, reports console errors and phone-width overflow
node scripts/check-pages.cjs --theme=matrix   # the same in the matrix theme
node scripts/shoot-widgets.cjs <lessonId>   # screenshots of every widget, light + dark + matrix, into scratch/shots
node scripts/drive-widget.cjs <lessonId> <i> '<actions JSON>'   # click/drag a widget and screenshot it
```

## Where things live

- `src/curriculum.ts` — register lessons here (lazy-loaded). `src/glossary.ts` — glossary terms for `<Term id>`.
- `src/lessons/<id>/` — self-contained lesson: `index.mdx`, widget components, `quiz.ts`, lesson-specific logic + tests. Avoid importing from other lessons.
- `src/lib/` — shared, tested maths (linear algebra, autodiff `Value`, `MLP`, optimizers, softmax/entropy, activations, seeded RNG).
- `src/components/` — `Widget`, `Eq`, `Predict`, `Quiz`, `Callout`/`Deeper`/`MathExhibit`, `Term`, controls, SVG `Plot` primitives, `Heatmap` (+ `fieldImage` thumbnails), `NetDiagram` (clickable network drawing).
- MDX gotchas: no raw `{`, `}` or `<` in prose; JSX attribute strings don't process escapes, so `tex="\frac{a}{b}"` works as written.
