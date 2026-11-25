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
2. Update `docs/ROADMAP.md`: the **Current state** block at the top (what's done, what's next), the status table, **Next steps** (with enough of a plan that the next lesson can be started cold), and **Labs built so far**. Roadmap entries use `- **Done [<MODEL_NAME>]:** …` (see the owner's standing instructions below).
3. If you changed the procedure or found a new gotcha, update this file.

## Owner's standing instructions (from the session prompts; follow every session)

**Branch and pull requests.** Work on, commit to and push to `claude/cool-feynman-4hci4d`. It is the repository's default branch and there is no `main`: don't open a pull request and don't create other branches. If a push is rejected because the remote moved, `git fetch origin claude/cool-feynman-4hci4d && git merge --no-edit FETCH_HEAD`, then push again (another agent or the owner may have pushed in between).

**Work autonomously** from "Next steps" in `docs/ROADMAP.md`. When a lesson meets the definition of done, move straight on to the next one.

**Invisible authorship watermark** (the owner compares quality across models). It must never be visible on the rendered site or inside `dist/`.
- Use an exact `<MODEL_NAME>` identifying your model. Names already in use: "Neon High", "Gemini Antigravity", "Claude Opus 5.5". If you are one of those models, reuse its name exactly. Otherwise pick a new, unique name that doesn't appear in any lesson text in `dist/`.
- Top of every new `.mdx` file: `{/* authored-by: <MODEL_NAME> */}`
- Top of every new `.ts`/`.tsx` file: `// authored-by: <MODEL_NAME>`
- Git commit trailer on every commit: `Authored-by: <MODEL_NAME>`
- Roadmap entries: `- **Done [<MODEL_NAME>]:** …`
- After `npm run build`, check that `rg "<MODEL_NAME>" dist/` returns 0 matches.

**Non-negotiables, as the owner phrased them** (details in `docs/STYLE_GUIDE.md`):
- Concept before equation. Key equations use `<Eq words="…" tex="…">` with prose words wrapped in `\text{…}`, colour-matched with `\ca \cb \cc \cg \cp \cv` (never `\cr`), and a plain-English symbol legend passed as MDX children, not a `legend` prop. `<Predict>` also takes its revealed explanation as MDX children, not an `explain` prop.
- Conversational, warm, precise prose (British spelling); structural analogies; predict before reveal; let the learner feel the problem first; address real misconceptions.
- Every widget computes real numbers and colours only through CSS variables (never hard-coded hex; use `--c-good`/`--c-bad`, not `--good`/`--bad`). Labs give primitives, a clear objective, automatic checking, `localStorage` persistence and multiple valid solutions. Optimums are computed exhaustively, or labelled "best found by searching <stated space>" and produced by a committed script re-verified by a test.
- Compare competing methods fairly (each at its own searched best, averaged over fixed seeds) and report honestly when results don't match the textbook story.
- Verify every quoted number by running it and pin it with a Vitest test. Check every hand-written gradient against central finite differences. Run every code snippet you put in a lesson.
- Visually inspect every new widget in a real browser in light, dark and matrix themes and at 390px (`scripts/shoot-widgets.cjs`, `scripts/drive-widget.cjs`).

**Environment tips.**
- The dev server can stop between runs. If `check-pages` fails with `ERR_CONNECTION_REFUSED`, restart `npm run dev`; that isn't a page failure.
- If Playwright isn't in local `node_modules` and `require('playwright')` fails, run `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm install -g playwright` and invoke scripts as `NODE_PATH=$(npm root -g) node scripts/...`.

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
- **PyTorch install:** the CPU wheel index (`download.pytorch.org`) may be blocked by the network proxy; plain `pip install torch` from PyPI works (larger download, runs fine on CPU).
- **Labs that train networks live:** judge a design on several fixed seeds (the autoencoders lab uses 3) so one lucky run can't win the optional goal, and record the exhaustive search with a sharded script (`node … lab.run.ts <shard> <shards>` in parallel) whose output is regenerated into a committed results file.
- **Lessons that need a language model (Module 18):** no LLM runs in the browser. Make the computable part (retrieval, tool state, schema validation, success probabilities) real and exact, show the exact prompt a model would receive, and say in the caption what is simulated. The RAG lesson's hand-made concept dictionary is the pattern: a transparent stand-in with the same structure as the real component, labelled as a stand-in.
- **Exhaustive searches that are cheap** (under a second or two) can run live in the browser on demand (guidance, RAG labs); slow ones go in a sharded `*.run.ts` script whose output is committed and re-verified by a test (autoencoders, diffusion labs).
- **Percentages of a finite sample:** format consistently (e.g. `(100*v).toFixed(1)`) in widgets *and* pin the exact formatted string in tests; `Math.round` of 11.5 and of 88.5 can disagree so two complementary shares add to 101%.
- **Uppercase labels again:** `Segmented` labels are uppercased, so `(k)` becomes `(K)` and `ᾱ` becomes `Ā`. Keep symbols out of labels.
- **Heredocs with `${…}` or `rm`-like text can be refused by the command-safety check** even when nothing is deleted: write the script with the file tool instead.


## Lab improvements plan: progress (`docs/LAB_IMPROVEMENTS.md`)

The owner asked for the plan in `docs/LAB_IMPROVEMENTS.md` to be implemented in its "Suggested order of work", one step per commit. Tick a step when its commit is pushed; mark the step you are in the middle of as **IN PROGRESS** with a note of what is done so far, so another agent can resume after a disconnect.

- [x] 1. Quick fixes: two hints naming locked options; five early answer reveals gated behind a solve or an explicit "I'm stuck" step.
- [x] 2. A real job in every brief, "where this happens for real" debrief lines, live code lines in the six tuning labs.
- [x] 3. History cards: `src/history.ts` + test, collapsed card after each lab.
- [x] 4. Solution maps: Attention puzzles (11×11 query map), Build a previous-word head (180-speed strip).
- [x] 5. Shared pieces: `LabShell`, `MachineLog`, reduced-motion hook; Dice Workshop and Tune the Q-Learner moved onto them.
- [x] 6. Machinery for the four "needs work" labs: Q-Learner maze replay + curves + predict-then-train (6a), sampler live draws (6b), compute-budget map (6c), alignment reply bars + Goodhart dot + prediction + training log (6d). All four are on LabShell.
- [x] 7. Timeline page built from `src/history.ts`.
- [x] 8. Decision lines in the Logic workbench.
- [x] 9. Replay helper (`useReplay`, `ReplayControls`, `GhostPath`); reward hacking (9a), Wire the Policy (9b), golf labs (9c), VRAM Tetris falling blocks + roofline (9d).
- [ ] 10. **IN PROGRESS** Remaining rows of the three tables, one commit per lab (tick as done):
  - [x] Stabilise the policy gradient: 20 faint seed lines under the average
  - [x] Design a schedule: 20 seeded runs behind the average, best cosine ghost after a solve (moving marker not done)
  - [x] Prune it: the exhaustive minimum's kept neurons beside yours, log
  - [x] LoRA: per-layer captured vs missed heatmaps, log
  - [ ] Mixed-precision quantisation: selected block on a number line with the format's levels; evidence hint; per-block log
  - [ ] Shannon's betting game: per-word bits-lost strip (you, neural LM, unigram) + log
  - [ ] Compress the corpus: live re-segmenting sample sentence + merge log; settle the "optimum" label question
  - [ ] Build a previous-word head: clock hands under the bars
  - [ ] Superposition: rotatable cube for Brief 1, clash log
  - [ ] Write the backward pass: collection of the correct versions, wrong-but-trains call-out, gradient-check log
  - [ ] Training-lab challenges: reference network's neuron maps after a challenge
  - [ ] Rescue the deep network: per-layer log
  - [ ] Attention puzzles: worked q·k → ÷√2 → exp → share table
  - [ ] The model clinic: train.log tail per case; two-cause cases

Notes for whoever continues:
- Step 1 added `src/components/AnswerReveal.tsx` (`AnswerReveal`, `StuckButton`, `useStuck`): use it for any answer/optimum reveal so it appears only after a solve or an explicit, remembered "I'm stuck". Hints that depend on the brief live in the lesson's `.ts` (`rescueHint`, `alignmentHint`) with a `hints.test.ts` checking they never name a locked control.
- Step 2 added `src/components/LabJob.tsx`: `<LabJob>` (one line above a lab, in MDX), `<ForReal>` ("Where this happens for real", after the debrief) and `LiveCode` (live code line inside a widget). Each tuning lab's code builder lives in `src/lessons/<id>/realCode.ts` with a test; APIs were checked against library source on GitHub (TRL, vLLM, peft, llama.cpp, transformers).
- Step 3 added `src/history.ts` (25 entries, one per reviewed lab; every sentence a `Claim` with its own source URL, plus `ERAS` and `MILESTONES` for the timeline) and `<HistoryCard id>` placed after each lab's `<ForReal>`. `src/history.test.ts` checks sources, dates and that every entry's card is in its lesson. All † affiliations were confirmed on the papers' title pages; entries without a sourced mood line simply omit it.
- Step 4: `queryMap` (attention) and `speedStrip`/`speedPairMap` (positions) feed post-solve maps behind a "Show the map" button that appears once a puzzle is solved or declared impossible. The two-hand level gets a 180×180 canvas map (colours read from CSS variables, redrawn on theme change).
- Step 5 added the shared lab pieces. `src/components/LabShell.tsx`: `LabShell` (brief tabs with ✓/★ and a count, brief text, body, status line, hint toggle, `reveal` gated by AnswerReveal, `log`, `actions`) and `useLabProgress(key, seed?)` (solved/optimal per brief in localStorage; `seed` carries over progress saved in an older format). `src/components/MachineLog.tsx`: `useMachineLog()` + `<MachineLog>` (collapsible console, newest line shown in the summary). `src/components/motion.ts`: `usePrefersReducedMotion`, `useTweenedArray`/`useTweened`, `useFramePlayer`; global.css zeroes transitions inside widgets under reduced motion. The Dice Workshop (Roll 1,000 times + log) and Tune the Q-Learner (log, evidence hints, answer behind reveal) now use them.
- Step 7: `#/timeline` (`src/pages/TimelinePage.tsx`, sidebar link "Timeline of AI") draws `HISTORY`, `ERAS` and `MILESTONES` from `src/history.ts`: era bands (click for the sourced mood), a dot per lab idea (filled when its lesson is completed), a "you are here" marker at the latest idea reached, and every card below (shared `HistoryBody`). `timelineX`/`assignLanes` are tested. Each lesson's history card links to `#/timeline/<id>`, which highlights it.
- `scripts/drive-widget.cjs` takes `--title=<text>` to pick a widget by title instead of index.
