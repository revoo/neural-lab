# Handoff — start here

You're picking up **Neural Lab**, an interactive website that teaches AI from first principles: Feynman-style explanation, 3Blue1Brown-style visuals, Zachtronics-style labs. This file is the onboarding procedure. Follow it in order. (Agent rules in brief: `AGENTS.md` / `CLAUDE.md`.)

## 1. Get oriented (≈10 minutes)

Read, in this order:

1. **`docs/VISION.md`** — the owner's brief. Pay special attention to *"Owner's clarifications during development"*: they override anything that seems to conflict.
2. **`docs/STYLE_GUIDE.md`** — how to write prose, present maths, design widgets and labs.
3. **`docs/ROADMAP.md`** — what's done, what's next, known limitations.
4. **One finished lesson, start to finish**, to absorb the voice before writing anything:
   - `src/lessons/neuron/index.mdx` (concept → `Eq` → geometry → live training)
   - `src/lessons/gradient-descent/index.mdx` (feel-the-problem widget + puzzle lab)
   - `src/lessons/logic-lab/` (how a Zachtronics-style lab is built, with tested logic)

## 2. Set up and verify the baseline

```bash
npm install
npm test                          # expect all tests to pass
npm run build                     # typecheck + production build
npm run dev                       # leave running, then in another terminal:
node scripts/check-pages.cjs      # expect ✓ for every page at 1300px and 390px
node scripts/check-pages.cjs --theme=matrix   # same, in the matrix theme
```

Visual checks (dev server running; output goes to `scratch/shots/`, which is git-ignored):

```bash
node scripts/shoot-widgets.cjs <lessonId>            # every widget on a lesson, light + dark + matrix (--themes=dark to narrow)
node scripts/drive-widget.cjs <lessonId> <widgetIndex> '[{"click":"button:has-text(\"Next\")","times":2},{"click":"[data-node=\"1-0\"]"},{"shot":"after"}]'
```

`drive-widget.cjs` documents its actions (click, clickAt, drag, wait, eval, shot) at the top of the file. Then look at the PNGs.

If anything fails before you've changed a line, fix that first and note it in the roadmap.

## 3. Pick up the work

Take the first item under **"Next steps"** in `docs/ROADMAP.md` unless the owner asked for something else. Work in complete vertical slices: one finished, polished lesson beats three half-written ones.

### Checklist: adding a lesson

1. Create `src/lessons/<id>/` with `index.mdx`, widget component(s), and `quiz.ts`.
2. Register it in `src/curriculum.ts` (and remove it from that module's `planned` list).
3. Add any new terms to `src/glossary.ts`; use `<Term id="…">` on first mention.
4. Put non-trivial lesson logic in a plain `.ts` module with a `*.test.ts` beside it.
5. Follow the lesson arc: problem → analogy → hands-on widget (with a `<Predict>` where something surprises) → "Writing it down" with `<Eq>` blocks → code → `<Misconception>` → `<Quiz>` → `<Takeaways>` → `<InTheWild>` → a one-line bridge to the next lesson.

### Definition of done for a lesson

- [ ] Understandable conceptually before any equation; each key equation is an `<Eq>` with a word version and a legend naming every symbol.
- [ ] Every widget computes real values; its caption says what to try; a live note explains the current state.
- [ ] Any lab has a clear objective, automatic checking, multiple valid solutions, and optional optimisation goals whose optimum was **computed**, not guessed.
- [ ] Every number quoted in prose or code comments has been checked by running it.
- [ ] Quiz: mostly conceptual questions; wrong-answer feedback explains the misconception; numeric questions are parameterised.
- [ ] `npm test`, `npm run build` and `node scripts/check-pages.cjs` all pass.
- [ ] You looked at the widgets in a real browser (Playwright screenshots are fine) in light, dark **and** matrix themes, and at phone width.

### After every finished lesson (owner's instruction)

Assume the session can disconnect at any moment and someone else will continue from the repository alone:

1. Commit and push (`git push -u origin <branch>`).
2. Update `docs/ROADMAP.md`: the **Current state** block at the top (what's done, what's next), the status table, and **Next steps** (with enough of a plan that the next lesson can be started cold).
3. If you changed the procedure or found a new gotcha, update this file.

## 4. Before you end the session

1. Update **`docs/ROADMAP.md`**: the status table, labs built, new decisions, new limitations, and the next steps in order.
2. If the owner gave new direction, record it in **`docs/VISION.md`** under the clarifications, and in **`docs/STYLE_GUIDE.md`** if it affects how things are written.
3. Commit with clear messages and push.

## Gotchas that have already bitten someone

- **Three themes.** `data-theme` is `light`, `dark` or `matrix` (`src/theme.ts`). OS-dark media queries use `:root:not([data-theme])` so they only apply when no theme was chosen — don't write `:not([data-theme='light'])`, it leaks dark styles into matrix. Colour widgets through CSS variables only.

- **MDX:** no raw `{`, `}` or `<` in prose. JSX attribute strings don't process escapes, so `tex="\frac{a}{b}"` works as written; in JS strings (e.g. `quiz.ts`), backslashes must be doubled.
- **Colour macros:** use `\ca \cb \cc \cg \cp \cv`. Never name a macro `\cr` — TeX reserves it as "end of row", and matrices silently collapse.
- **One KaTeX version only.** `package.json` pins it via `overrides`; a second copy makes subscripts render at the baseline.
- **Heatmaps** recompute whenever their `f` prop changes identity — memoise it (`useMemo`) or define it at module scope.
- **Lessons stay independent.** Duplicate a small dataset rather than importing from another lesson's folder. (Exporting a small component from one lesson's own file to another file in the *same* lesson is fine.)
- **Shared network drawing:** use `src/components/NetDiagram.tsx` rather than writing another SVG network. Weight layout is `W[l][j][i]` (layer, to-neuron, from-neuron), the same as `src/lib/mlp.ts`.
- **Quick numeric checks:** `node --experimental-strip-types scratch/foo.ts` runs a TypeScript file that imports lesson `.ts` modules (use explicit `.ts` extensions in the import). Don't point Vitest at a directory outside the repo (e.g. `--root /`): it will crawl the whole disk.
- **Node vs browser floating point.** `Math.exp` (and so `tanh`) can differ in the last binary digit between Node and Chromium. Short computations agree to every printed digit, but a 4,000-step training run can drift in the third decimal place. Pin numbers with Node tests, quote them to sensible precision, and if a widget re-runs a recorded search live, compare with a tolerance and say why (see `regularization/Widgets.tsx`).
- **Verify code snippets in lessons by running them.** NumPy is installed. PyTorch isn't by default, but `pip install torch --index-url https://download.pytorch.org/whl/cpu` works in about a minute (it's not saved between sessions).
- **Shell safety check:** multi-line heredocs containing JavaScript template strings (`${…}`) can be refused by the agent's command-safety check. Write scripts with the file-writing tool into `scratch/` (git-ignored) and run them with a plain command.
- **`<Eq>` and `<Predict>` props:** `<Eq>` renders `words` through KaTeX (so wrap prose words in `\text{…}`) and takes its symbol legend as MDX `children` (not a `legend` prop). `<Predict>` takes its revealed explanation as MDX `children` (not an `explain` prop).
- **`.field-label` uppercase transform:** `Segmented`'s `label` prop renders with `text-transform: uppercase`, which turns lowercase Greek letters like `α` and `ε` into `Α` and `Ε`. Keep lowercase Greek symbols inside the option labels rather than the group heading. The same applies to `.num-table th` headers (π becomes Π, α/β become A/B): write words there instead.
- **Twenty seeds can still be luck.** When an outcome is decided by one early random draw (e.g. a policy that "marries" its first action), a 20-seed average mostly measures how those 20 first draws fell. Before quoting such a number, re-run it over a few hundred seeds and say so (see the policy-gradients lab debrief). `policy-gradients/pg.ts` also scrambles seeds with `mixSeed` before `createRng`.
- **Phone-width grids:** for side-by-side widget panels use `gridTemplateColumns: 'repeat(auto-fit, minmax(min(270px, 100%), 1fr))'` (the `min(…, 100%)` matters at 390px), and wrap wide `num-table`s in `<div style={{ overflowX: 'auto' }}>`.
