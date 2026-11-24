# Neural Lab — instructions for coding agents

This file is for any AI coding agent (Codex, Claude Code, Cursor, Aider, Gemini CLI, …). `CLAUDE.md` holds the same rules; if you change one, change the other.

Neural Lab is an interactive website that teaches AI from first principles to software engineers and lifelong learners: Feynman-style explanation, 3Blue1Brown-style visuals, Zachtronics-style labs. **Every visualisation computes real numbers.** Stack: Vite + React + TypeScript + MDX, KaTeX for maths, Vitest for tests, no ML framework (plain-array maths in `src/lib/`).

## Read before doing any work

1. **`HANDOFF.md`**: the onboarding procedure, the lesson checklist, the definition of done and the gotchas.
2. **`docs/ROADMAP.md`**: start with the **"Current state"** block (what's done, what's in progress, what's next), then **"Next steps"**, which are in priority order and detailed enough to start cold.
3. **`docs/VISION.md`**: the owner's brief. The "Owner's clarifications" section overrides anything that seems to conflict.
4. **`docs/STYLE_GUIDE.md`**: voice, lesson structure, how to present maths, lab design, and the rules on **fairness and honesty in experiments**.
5. One or two finished lessons, to match the voice: `src/lessons/neuron/index.mdx`, `src/lessons/overfitting/index.mdx`, `src/lessons/attention/index.mdx`. For a lab, read `src/lessons/diagnosis-lab/` or `src/lessons/training-lab/`.

## Non-negotiables

- **Concept before equation.** The learner should see the equation and think "of course it looks like that".
- **Key equations use `<Eq words="…" tex="…">` with a legend** giving every symbol a plain-English name. Colour-match the symbols with `\ca \cb \cc \cg \cp \cv`; never use `\cr`, which TeX reserves. Use `<MathExhibit>` only for optional deeper material.
- **Write conversational, warm, precise prose.** Use structural analogies, ask the learner to predict before revealing, let them feel the problem first, and address real misconceptions. British spelling. No jargon for its own sake, no forced enthusiasm.
- **No fake interactivity.** Widgets compute real values; captions say what to try; live notes explain the current state.
- **Labs and puzzles:** give primitives, a clear objective, observable machinery and multiple valid solutions. Ask for a working solution first, with optional optimisation goals after. Compute optimums exhaustively, or label them "best found by searching <stated space>", produced by a committed script and re-verified by a test. No badges or points.
- **Compare fairly and report honestly.** Each competing method gets its own best settings, found by search, and random effects are averaged over fixed seeds. If something doesn't behave as the textbook says in our small problems, say so. Re-check a "fix" with other seeds before crediting it.
- **Verify every number quoted in prose by running it, and pin it with a test.** Run every code snippet you put in a lesson (NumPy is installed; for PyTorch, `pip install torch --index-url https://download.pytorch.org/whl/cpu`).
- **Three themes: light, dark and matrix.** Colour widgets only through CSS variables (`var(--c-a)`, `var(--accent)`, …), never hard-coded hex. Check every widget in all three themes and at phone width (390px).
- **After every finished lesson:** commit, push, and update the "Current state" block, the status table and "Next steps" in `docs/ROADMAP.md`. Assume the session can end at any moment and someone else will continue from the repository alone.

## Commands

```bash
npm install
npm run dev                                    # http://localhost:5173 (leave running for the scripts below)
npm test                                       # Vitest: maths, simulations, lesson logic, quoted numbers
npm run build                                  # typecheck + production build
node scripts/check-pages.cjs                   # every page at 1300px and 390px: console errors, phone overflow
node scripts/check-pages.cjs --theme=matrix    # the same, in the matrix theme
node scripts/shoot-widgets.cjs <lessonId>      # screenshot every widget (light, dark, matrix) into scratch/shots
node scripts/drive-widget.cjs <lessonId> <i> '<actions JSON>'   # click/drag a widget and screenshot it
node --experimental-strip-types --import ./scripts/ts-hooks.mjs scratch/x.ts   # run TS that imports project modules
```

The scripts use Playwright with a preinstalled Chromium (`require('playwright')`). `scratch/` is git-ignored: put throwaway scripts and screenshots there.

## Where things live

- `src/curriculum.ts`: register lessons here (they are lazy-loaded). `src/glossary.ts`: glossary terms for `<Term id>`.
- `src/lessons/<id>/`: one self-contained lesson: `index.mdx`, widget components, `quiz.ts`, lesson logic in plain `.ts`, and `*.test.ts` beside it. Don't import from other lessons; duplicate small pieces instead.
- `src/lib/`: shared, tested maths: linear algebra, autodiff `Value`, `MLP` (with dropout), optimizers, softmax/entropy, activations, seeded RNG.
- `src/components/`: `Widget`, `Eq`, `Predict`, `Quiz`, `Callout`/`Deeper`/`MathExhibit`, `Term`, controls, SVG `Plot` primitives, `Heatmap`, `LogChart`, `NetDiagram`, `DigitalRain`, `ThemeSwitch`.
- `src/styles/global.css`: design tokens and all styles. The matrix theme block is near the end.

## Gotchas

- **MDX:** no raw `{`, `}` or `<` in prose. JSX attribute strings don't process escapes, so `tex="\frac{a}{b}"` works as written; in JS strings (e.g. `quiz.ts`), backslashes must be doubled.
- **Committed searches:** labs' recorded optimums live in files like `results.ts` and `fixes.ts`, produced by `search.run.ts` scripts. Re-run the script if you change the engine, and keep the tests that re-verify them.
- **Node vs browser floating point:** they can differ in the last bit of `Math.exp`, which long training runs amplify. Pin numbers with Node tests and quote them to sensible precision.
- **Branches:** work on the branch you were given, and don't push to `main` without the owner's say-so.

More gotchas are listed in `HANDOFF.md`.
