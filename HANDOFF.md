# Handoff — start here

You're picking up **Neural Lab**, an interactive website that teaches AI from first principles: Feynman-style explanation, 3Blue1Brown-style visuals, Zachtronics-style labs. This file is the onboarding procedure. Follow it in order. (Agent rules in brief: `AGENTS.md` / `CLAUDE.md`.)

## 0. Resume here: curriculum plan (WP0–WP9) and v2 plan (V2-WP1–WP6) complete

**v2 update [Claude Opus 5.5]:** the v2 release plan (`docs/V2_PLAN.md`) is finished: area chips, the `#/map` concept map, the two-tier compute library (CPU Float32 + WebGPU) with its upgrades, and three new lessons with labs (`moe`, `reasoning`, `multimodal-vit`). The site has 78 lessons and 60 history cards. Nothing further is queued; candidates are listed under "Next steps" in `docs/ROADMAP.md`.

`docs/CURRICULUM_PLAN.md` (WP0 → WP9) is **complete** and pushed on `claude/cool-feynman-4hci4d`: Claude Opus 5.5 did WP0–WP1.7, Labs 1–2, WP1.10, WP1.9 and WP2–WP5; Codex Meridian 6 did Labs 3–9; Gemini Antigravity completed WP6–WP9. The site now has **20 modules (0–19) and 75 lessons**, 57 sourced history cards, 12 sourced timeline milestones, and 150 passing test files (644 tests).

**Read first, in this order:** `AGENTS.md`, this file (especially "Owner's standing instructions" and "Gotchas"), and the "Current state" block at the top of `docs/ROADMAP.md` (which includes "Open questions for the owner").

**Your name for the authorship watermark.** Follow the owner's rule in "Owner's standing instructions": if your model already has a name ("Neon High", "Gemini Antigravity", "Claude Opus 5.5", "Codex Meridian 6"), reuse it exactly; otherwise pick a new, unique `<MODEL_NAME>`. Use it in file headers, the `Authored-by:` commit trailer and roadmap entries. After `npm run build`, `rg "<MODEL_NAME>" dist/` must find nothing. (An earlier note here said not to reuse "Claude Opus 5.5"; that conflicted with the owner's rule and is recorded as an open question in `docs/ROADMAP.md`.)

### Done so far

| Step | What exists now |
|---|---|
| WP0.1 | `ModuleRef` (renders the live module number, linked) in `src/components/Term.tsx`; `Src` (source link) in `src/components/Src.tsx`; both registered in `src/mdxComponents.tsx`. Test: `src/components/moduleRef.test.ts`. |
| WP0.2–0.3 | No lesson prose types "Module N" or "last/next/previous lesson/module" any more: they're `ModuleRef` / `LessonLink`. |
| WP0.4 | `src/curriculum.test.ts` enforces the above, checks every `LessonLink`/`ModuleRef` target, consecutive module numbers and one registration per lesson folder, and (test 6, added in WP1.4) that every toolkit lesson has a `<Bridge to>` a later module. The plain-summary test is live (WP2.4). `node scripts/forward-links.cjs` prints links that point later in the course; read it after any reorder. |
| WP1.1 | Module 1 is "The Maths Toolkit" (9 lessons). `statistics` sits between `loss-functions` and `sgd-minibatch`; `high-dimensions` opens Module 8 (`classical`). Their prose was re-tensed to fit. |
| WP1.2 | `src/components/MachineMap.tsx` (+ `machineMap.test.ts`): SVG of the learning machine; two-column layout under 560px. |
| WP1.3 | A `<MachineMap highlight="…" />` plus a "**Where this is going.**" paragraph before the second `##` of each toolkit lesson. |
| WP1.4 | `src/components/Bridge.tsx` ("Where you'll use this next"); one `<Bridge>` after `</InTheWild>` in each toolkit lesson. |
| WP1.5 | 16 new `<Predict>`s; every number pinned in `src/lessons/derivatives/predicts.test.ts` (one file for all nine lessons). |
| WP1.6 | `klDivergence` in `src/lib/prob.ts` (tests in `src/lib/lib.test.ts`); KL section + `KLExplorer.tsx` + `kl.ts`/`kl.test.ts` + quiz `information.kl` in `information`; glossary `kl-divergence` now points at `information`. |
| WP1.7 | "Play" animations in `TangentExplorer`, `DotExplorer`, `ChainExplorer`, `CompositionDemo` (all via `useFramePlayer`, so reduced motion jumps to the end); frame values pinned in `src/lessons/chain-rule/motion.test.ts`. |
| WP1.8 Lab 1 | `src/lessons/vectors/featureLab.ts`, `featureLab.test.ts`, `FeatureLab.tsx`; history entry `feature-workbench`. |
| WP1.8 Lab 2 | `src/lessons/dot-product/spamLab.ts`, `spamLab.test.ts`, `SpamLab.tsx`; history entry `spam-scorer`. This brought the history collection to 48 entries. |
| WP1.8 Lab 3 | `src/lessons/matrices/untangleLab.ts`, solver tests and `UntangleLab.tsx`. History entry `untangle`. |
| WP1.8 Labs 4–9 | Build the Pipeline, The Sensitivity Probe, Which Knob, and How Far?, Find the Slipping Gear, Calibrate the Forecaster, and Twenty Questions Compressor. Lab logic, tests and widgets are in their lesson directories. |
| WP1.10 | `src/components/RivalTable.tsx` (cards judged on the three tests; registered in `src/mdxComponents.tsx`, styles under `.rival-table` in `global.css`). Each toolkit lesson has `rivals.ts` + `rivals.test.ts` (every number in the cards and the section's `Predict` is computed there and pinned) and a `## Why this, and not something else?` section before its lab. The three tests are stated once, in `vectors`, as an `Insight`. `src/curriculum.test.ts` test 6 now also requires that heading, before the lab, mentioning the three tests. |
| WP2 | `rl` is Module 14 (before `finetuning`, `inference`); `similarity-search` opens `systems`; token notes in `one-hot-vs-dense` and `attention`; Appendix A summaries; summary test live. |
| WP3 | Module 0 `prologue` (`src/lessons/prologue/`): sourced history, `PrologueDemo` (25-parameter net on two circles), winter dates from `ERAS` via `eraSpan`. Home button "Start with the prologue". |
| WP4 | `src/lessons/perceptron/` after `logic-lab`: `perceptron.ts` (rule, `run`, `novikoffBound`, lab searches), `PerceptronStepper`, `MarkILab` (4 briefs), history `mark-i` (55 entries), glossary `perceptron`, `credit-assignment`. |
| WP5 | `era` on every module in `curriculum.ts` (test: every history card's years fall inside its module's era); shown on Home cards and in the lesson kicker (linked to the timeline); 12 sourced `MILESTONES` in `src/history.ts`. |
| WP1.9 | `minutes` +15 for each toolkit lesson; vectors' units section trimmed to point at Lab 1's brief 3; two closing teasers cut; consistent backprop cost (about three forward passes). Widget fixes: Sensitivity Probe chart legend and axes; Build the Pipeline counts every intermediate layer as hidden (`meets`, brief 2 optimum now 50,816); Calibrate the Forecaster's `shownPercent` never shows a positive probability as 0. |
| WP6 | WP6.1 sourced opening hooks in 12 lessons (`neuron`, `gradient-descent`, `loss-functions`, `activation-functions`, `backpropagation`, `vanishing-gradients`, `optimizers`, `overfitting`, `cnns`, `transformer-block`, `mdps`, `word-embeddings`); WP6.2 "You are standing where they stood" `Insight` notes in 6 remaining labs (`logic-lab`, `network-from-scratch`, `vanishing-gradients`, `attention`, `chain-rule`, `information`; `perceptron` already had its note). |
| WP7 | `src/components/EraStrip.tsx` (+ `eraStrip.test.ts`, registered in `mdxComponents.tsx`) and four 8-minute sourced interlude lessons (`interlude-first-winter` in `neuron`, `interlude-second-winter` in `classical`, `interlude-big-bang` in `vision`, `interlude-chatgpt` in `llms`). |
| WP8.1 | `src/lessons/svm/` in `classical` after `linear-regression`: exact 2D hard-margin solver, SMO soft-margin, 3D paraboloid lift, `WidenTheStreetLab` (4 briefs), history `svm` (56 entries), glossary `svm`, `support-vector`, `margin`, `kernel-trick`. |
| WP8.2 | `src/lessons/gans/` in `generative` after `autoencoders`: live 1D GAN on 3-mode Gaussian (`gan.ts`, exact finite-diff checked backprop), `GanDuelWidget`, `DuelBalanceLab` (30-setting × 5-seed search in `search.run.ts`/`results.ts`), history `gans` (57 entries), glossary `gan`, `mode-collapse`, `minimax-game`. |
| WP8.3 | Module 19 `epilogue` (`era: '2018 to today'`) and `src/lessons/open-questions/` ("What Nobody Understands Yet"): minimum-norm random Fourier features double-descent engine & `RandomFeaturesWidget` (peak pinned at $p = n = 24$), grokking, superposition, both sides of emergence, scaling laws, closing link to `prologue`, 3-question quiz. |
| WP9 | Final documentation updated across `docs/ROADMAP.md`, `HANDOFF.md`, `docs/VISION.md`, and `docs/CURRICULUM_AUDIT.md`. |

### What to do next

- **Curriculum plan (WP0–WP9) is complete.** Next up is the **v2 Release Plan** (`docs/V2_PLAN.md`, checklist at the bottom of this file):
  1. **V2-WP1:** Optional sections & guided tracks (`Full Course` vs `Core Track`).
  2. **V2-WP2:** Interactive Concept Dependency Graph (`#/map`).
  3. **V2-WP3:** Two-Tier Compute Offloading (Test `Float32Array` + Production WebGPU).
  4. **V2-WP4:** Mixture of Experts (MoE) & Sparse Routing (`moe`, Lab: Balance the Experts).
  5. **V2-WP5:** Reasoning Models & Test-Time Compute (`reasoning`, Lab: Spend the Test-Time Budget).
  6. **V2-WP6:** Multimodal Models & Vision Transformers (`multimodal-vit`, Lab: Wire the Multimodal Model).

### How each maths lab was built (copy this)

1. **Solver and test first.** Put the logic, level data and an exhaustive (or clearly labelled) solver in `src/lessons/<id>/<name>Lab.ts`, and the test in `<name>Lab.test.ts`. The test pins the counts quoted in the prose and checks that each brief is solvable, has more than one answer, and that a random setting usually fails.
2. **Tune the data, not the idea.** Labs 1 and 2 both needed it. Write a throwaway random-search script in `scratch/` and run it with `node --experimental-strip-types --import ./scripts/ts-hooks.mjs scratch/x.ts`. Examples: `scratch/lab1search.ts`, `scratch/lab2.ts` (`scratch/` is git-ignored and may be gone; the pattern is simple).
3. **UI on `LabShell`.** Use `useLabProgress('<lessonId>-lab')`, a `localStorage` key for the learner's settings, `useMachineLog`, an evidence-only `hint`, and a `reveal` (the solution map or search summary, shown only after a solve or a confirmed "I'm stuck"). `src/lessons/vectors/FeatureLab.tsx` and `src/lessons/dot-product/SpamLab.tsx` are the models.
4. **MDX.** Add `## Lab: <name>` just before `## In code`, containing: a short intro, `<LabJob>`, the widget, a `<Deeper title="Debrief: …" tag="Spoiler">` with the solver's numbers, `<ForReal>`, `<HistoryCard id="…" />`, and a "You've just…" line that links forward with `LessonLink`/`ModuleRef`.
5. **History.** Add the entry at the top of `HISTORY` in `src/history.ts`, with every sentence a `Claim` with its source, checked on the page itself. Fetch pages with `curl -sL "https://en.wikipedia.org/w/index.php?title=<Page>&action=raw"`. Leave out anything you can't confirm, and record "dropped: <claim>, reason" in `docs/ROADMAP.md`. Raise the count in `src/history.test.ts`.
6. **Checks, then screenshots.** Run the checks (below), then `node scripts/drive-widget.cjs <lessonId> 0 '[{"shot":"x"}]' --title="<widget title>"` in all three themes, plus `--width=390 --themes=light`. Look at the PNGs in `scratch/shots/`. Check that captions don't name colours that change between themes.

### Checks

Before every commit: `npm test` → `npm run build` → `rg "<MODEL_NAME>" dist/` (must be empty). Once per lab or work package (and after shared-component, CSS or curriculum changes), with `npm run dev` running: `node scripts/check-pages.cjs` and `node scripts/check-pages.cjs --theme=matrix`. Then `git fetch origin claude/cool-feynman-4hci4d && git merge --no-edit FETCH_HEAD`, commit as `Curriculum plan WPx.y: …` with your `Authored-by:` trailer, and push. No pull requests, no other branches.

### Gotchas found in this run

- **(V2-WP3) WebGPU in headless Chromium** needs `--enable-unsafe-webgpu` *and* a secure context (`http://localhost`, not `about:blank`); the adapter is SwiftShader (a CPU emulation, so slow). `node scripts/check-webgpu.cjs` runs every WGSL kernel against the CPU tier; `drive-widget.cjs --webgpu` screenshots a widget on the GPU tier. Without the flag, widgets fall back to the CPU tier (that's what `check-pages.cjs` exercises).
- **(V2-WP3) One buffer can't be bound read-only and read-write in the same dispatch** (WebGPU validation silently invalidates the dispatch and you read garbage). Use distinct dummy buffers.
- **(V2-WP3) Chaotic training amplifies last-bit differences.** The GAN duel's unbalanced settings come out differently on the GPU (f32) *and* on the browser's f64 CPU (its `exp`/`tanh` differ from Node's). Compare such runs only with a tolerance on the stable cases, and say so in prose.
- **(V2-WP3) React strict mode in dev** runs effect cleanups once on mount: an `alive` ref cleared in a cleanup must be set back to `true` in the effect body, or async results are silently dropped (`GridRerun.tsx`).
- **(V2-WP4–6) TypeScript parameter properties** (`constructor(readonly K: number)`) don't run under `node --experimental-strip-types`; declare fields explicitly.
- **(V2-WP4–6) ReLU kinks and finite differences:** with zero biases, many pre-activations are exactly 0 and a central difference straddles the kink. Move biases off zero in gradient tests rather than loosening the tolerance.
- **(V2-WP5) No LLM in the browser:** the reasoning lesson's stand-in model is exact (closed forms + enumeration) and labelled as a stand-in; keep claims about real models sourced and keep the stand-in's assumptions in the prose.
- **(V2-WP3) Tiny networks are launch-bound on a GPU.** Don't claim the GPU is faster for the training lab's 37-parameter nets; the caption says the CPU often wins there.

- **(WP2–WP5) Vitest only collects `src/**/*.test.ts`.** A probe test in `scratch/` is "not found". To print data from modules the TS runner can't load, put a throwaway `src/zz-probe.test.ts` in place, run it with `--silent=false`, then delete it.
- **(WP2–WP5) Make bulk-edit scripts idempotent, or run them once.** Re-running an earlier "replace X with Y" script silently undid a later hand edit (the 31-word summaries). After any scripted edit, re-run the relevant test.
- **(WP2–WP5) Screenshot scripts and themes:** the theme is stored in `localStorage` under `nl-theme` (`light`, `dark`, `matrix`).
- **(WP2–WP5) Uppercase text again:** `.lesson-kicker` and the Home card `.n` line are uppercased, so an era like `1670s` would show as `1670S`. Both render the era with `text-transform: none`. Keep anything with lowercase letters that matter out of those lines, or do the same.
- **(WP2–WP5) Sources that disagree:** Wikipedia's Mark I and Perceptron articles give different dates for the machine. Don't pick one silently: avoid the date in prose and record the disagreement in the history entry's `dispute`.
- **(WP2–WP5) A lab primitive can make the goal trivial.** In the Mark I lab, free starting weights made "fewest mistakes" equal 0 (set the knobs by hand), so Brief 2 locks them. Check that every optional goal still needs thought after you've listed the primitives.
- **(WP2–WP5) Search results are bounds, so say which way.** `bestMargin` can only find a margin at or below the true one, so `novikoffBound` errs on the safe side; pin such numbers with `toBeCloseTo`, not exact equality.
- **(WP6.2 note)** The perceptron lesson already has its "You are standing where they stood" `Insight`, at the end of "The wall" (after the lab), not directly before the history card. The logic lab now ends with a one-line link to the perceptron lesson; WP6.2's logic-lab note goes before its `HistoryCard`, above that line.

- **Heredocs.** The agent's command-safety check refused heredocs containing `${…}`. Write helper scripts as files instead.
- **`<Predict>` inside a list.** A `<Predict>` inserted between Markdown list items breaks the MDX parse ("Unexpected end of file before attribute name"). Put it before or after the list.
- **JSX attribute strings don't process escapes.** In `question="…"`, write `\mathbf` with a single backslash. If you generate Predicts with `json.dumps`, un-double the backslashes in the attribute.
- **Distance plots.** When a lab's point is that something is close or far, draw both axes on one scale (`equalScale` in `FeatureLab.tsx`), or the picture lies.
- **Dot-product rankings depend only on the direction of w.** A 2D search over w is therefore one angle. `spamLab.ts` sweeps it in 0.01° steps.
- **Flaky page check.** `check-pages.cjs` once reported a one-off horizontal overflow on `/learn/evaluation` at 390px; a rerun was clean.
- **`.tsx` imports and the TS runner.** `node --experimental-strip-types --import ./scripts/ts-hooks.mjs` can't load a `.tsx` file, so a scratch script that imports lesson data from a widget fails with `ERR_UNKNOWN_FILE_EXTENSION`. Explore through a throwaway Vitest file instead (`npx vitest run <file> --silent=false` prints `console.log`).
- **`Plot` draws its axes through the origin.** For a chart whose domain excludes 0 (log scales, say), pass `axes={false}` and draw the axes and tick labels yourself (`SensitivityLab.tsx` shows how), or they land at a corner.
- **Render-time computation.** `rivals()` in each toolkit lesson runs when the page renders; keep it cheap. `gradients/rivals.ts` stores its grid-searched step sizes in `BIG_BEST` and its test re-runs the search.
- **Never type module numbers or 'next lesson' in prose; use `ModuleRef`/`LessonLink`, and `src/curriculum.test.ts` enforces it.**
- **Every Module 1 lesson needs a `Bridge` to a later module** (and a `## Why this, and not something else?` section before its lab; enforced by `src/curriculum.test.ts`).
- **Module `era` strings in `src/curriculum.ts`** must match `YYYY[s] to YYYY[s]|today` (segments separated by `, then `), and every history card in the module must fall inside its era (`src/curriculum.test.ts`).
- **`.row` in `global.css` does not set `display: flex`** — use explicit inline flex (`display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap'`) for responsive two-column widget layouts so they stack cleanly at 390px.
- **Fetching Wikipedia raw pages** in cloud containers can require IPv4 and a User-Agent header: `curl -4 -sL -A "NeuralLabEducationalBot/1.0 (https://github.com/revoo/neural-lab)" "https://en.wikipedia.org/w/index.php?title=<Page>&action=raw"`.
- **Slow 2-core VMs and `bpe.test.ts`:** run the full Vitest suite with `npm test -- --testTimeout=20000 --no-isolate` (and never concurrently with Playwright) so `bpe.test.ts` does not time out at 5s.
- **Lessons on the other agent's list** (unticked in "Next task" below): make only minimal line edits there; never reformat them.

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
2. Register it in `src/curriculum.ts` (and remove it from that module's `planned` list), then add it to `PREREQS` and `SHORT` in `src/dependencies.ts` and to `LESSON_AREA` in `src/areas.ts` (its chip must agree with the graph: see `src/areas.test.ts`). Keep the `summary` plain: **at most 30 words and no symbols** (enforced by `src/curriculum.test.ts`).
3. Add any new terms to `src/glossary.ts`; use `<Term id="…">` on first mention.
4. Put non-trivial lesson logic in a plain `.ts` module with a `*.test.ts` beside it.
5. Every historical sentence carries a `<Src href="…" />` verified on the source page itself; use `<ModuleRef>` / `<LessonLink>` instead of typing module numbers or "next lesson".
6. Follow the lesson arc: problem → analogy → hands-on widget (with a `<Predict>` where something surprises) → "Writing it down" with `<Eq>` blocks → code → `<Misconception>` → `<Quiz>` → `<Takeaways>` → `<InTheWild>` → a one-line bridge to the next lesson.

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
- Verify every quoted number by running it, and pin numbers that code produces (solvers, optimums, simulations, quoted widget readouts) with a Vitest test; plain arithmetic in prose needs no test. Check every hand-written gradient against central finite differences. Run every code snippet you put in a lesson.
- Visually inspect every new widget in a real browser in light, dark and matrix themes and at 390px (`scripts/shoot-widgets.cjs`, `scripts/drive-widget.cjs`); for small tweaks, one theme plus 390px is enough.
- **Proportionate checks (owner, Nov 2026):** this is a static teaching site. Before every commit run `npm test` and `npm run build`; run both `check-pages` passes once per finished lesson or work package, and after changes to shared components, `global.css` or `curriculum.ts`. Small related steps may share a commit. Keep correctness strict (numbers, optimums, history); keep process light.

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
- [x] 10. Remaining rows of the three tables, one commit per lab (small leftovers listed in ROADMAP "Current state"):
  - [x] Stabilise the policy gradient: 20 faint seed lines under the average
  - [x] Design a schedule: 20 seeded runs behind the average, best cosine ghost after a solve (moving marker not done)
  - [x] Prune it: the exhaustive minimum's kept neurons beside yours, log
  - [x] LoRA: per-layer captured vs missed heatmaps, log
  - [x] Mixed-precision quantisation: selected block on a number line with the format's levels; evidence hint; per-block log
  - [x] Shannon's betting game: per-word bits-lost strip (you, neural LM, unigram) + log
  - [x] Compress the corpus: live re-segmenting sentence + merge log; "optimum" settled: beam search (width up to 128, pinned in bpe.test.ts) finds no total below 1,806 bits, goal relabelled "best total found"
  - [x] Build a previous-word head: clock hands under the bars
  - [x] Superposition: rotatable cube for Brief 1, clash log
  - [x] Write the backward pass: collection of the correct versions, wrong-but-trains call-out, gradient-check log
  - [x] Training-lab challenges: reference network's neuron maps after a challenge
  - [x] Rescue the deep network: per-layer log
  - [x] Attention puzzles: worked q·k → ÷√2 → exp → share table
  - [x] The model clinic: train.log tail per case; two-cause cases (11 and 12, fixes recorded by search, pinned in clinic.test.ts)

Notes for whoever continues:
- Step 1 added `src/components/AnswerReveal.tsx` (`AnswerReveal`, `StuckButton`, `useStuck`): use it for any answer/optimum reveal so it appears only after a solve or an explicit, remembered "I'm stuck". Hints that depend on the brief live in the lesson's `.ts` (`rescueHint`, `alignmentHint`) with a `hints.test.ts` checking they never name a locked control.
- Step 2 added `src/components/LabJob.tsx`: `<LabJob>` (one line above a lab, in MDX), `<ForReal>` ("Where this happens for real", after the debrief) and `LiveCode` (live code line inside a widget). Each tuning lab's code builder lives in `src/lessons/<id>/realCode.ts` with a test; APIs were checked against library source on GitHub (TRL, vLLM, peft, llama.cpp, transformers).
- Step 3 added `src/history.ts` (25 entries, one per reviewed lab; every sentence a `Claim` with its own source URL, plus `ERAS` and `MILESTONES` for the timeline) and `<HistoryCard id>` placed after each lab's `<ForReal>`. `src/history.test.ts` checks sources, dates and that every entry's card is in its lesson. All † affiliations were confirmed on the papers' title pages; entries without a sourced mood line simply omit it.
- Step 4: `queryMap` (attention) and `speedStrip`/`speedPairMap` (positions) feed post-solve maps behind a "Show the map" button that appears once a puzzle is solved or declared impossible. The two-hand level gets a 180×180 canvas map (colours read from CSS variables, redrawn on theme change).
- Step 5 added the shared lab pieces. `src/components/LabShell.tsx`: `LabShell` (brief tabs with ✓/★ and a count, brief text, body, status line, hint toggle, `reveal` gated by AnswerReveal, `log`, `actions`) and `useLabProgress(key, seed?)` (solved/optimal per brief in localStorage; `seed` carries over progress saved in an older format). `src/components/MachineLog.tsx`: `useMachineLog()` + `<MachineLog>` (collapsible console, newest line shown in the summary). `src/components/motion.ts`: `usePrefersReducedMotion`, `useTweenedArray`/`useTweened`, `useFramePlayer`; global.css zeroes transitions inside widgets under reduced motion. The Dice Workshop (Roll 1,000 times + log) and Tune the Q-Learner (log, evidence hints, answer behind reveal) now use them.
- Step 7: `#/timeline` (`src/pages/TimelinePage.tsx`, sidebar link "Timeline of AI") draws `HISTORY`, `ERAS` and `MILESTONES` from `src/history.ts`: era bands (click for the sourced mood), a dot per lab idea (filled when its lesson is completed), a "you are here" marker at the latest idea reached, and every card below (shared `HistoryBody`). `timelineX`/`assignLanes` are tested. Each lesson's history card links to `#/timeline/<id>`, which highlights it.
- New gotcha: the dev server stops between long sessions; `drive-widget.cjs` then fails with `Node.js v…` and no screenshot. Restart `npm run dev` (see Environment tips).
- `scripts/drive-widget.cjs` takes `--title=<text>` to pick a widget by title instead of index.

## Next task: the lab-improvement treatment for the labs added after the review (NOT STARTED)

The owner asked for this next, in a fresh session. `docs/LAB_IMPROVEMENTS.md` reviewed only the 25 labs that existed then. The 21 labs added since have none of what those 25 now have. Bring them up to the same standard, one lab per commit, ticking the list below as you go. If you are interrupted, mark the lab you are in the middle of **IN PROGRESS** with a note.

**Read first:** `docs/LAB_IMPROVEMENTS.md` ("Cross-cutting ideas", "What good looks like", "The card") and the step notes in the section above. Then look at one finished example of each piece:
- Real job + "where this happens for real" + history card in the MDX: `src/lessons/logic-lab/index.mdx` (search for `<LabJob>`, `<ForReal>`, `<HistoryCard`).
- Answer gating: `src/components/AnswerReveal.tsx`, used in `src/lessons/lora/Widgets.tsx` (LoraBudgetLab).
- Evidence-only hints with a test that they never name a locked control: `rescueHint` in `src/lessons/vanishing-gradients/vanish.ts` and `hints.test.ts` beside it.
- Shared frame: `LabShell` + `useLabProgress` (with `seed` to keep old saves), as in `DiceWorkshop` in `src/lessons/statistics/Widgets.tsx`.
- Machine log: `MachineLog`/`useMachineLog`, e.g. `src/lessons/kv-cache/Widgets.tsx` (VramTetrisLab).
- History data: `src/history.ts` (every sentence a `Claim` with its own source URL). `src/history.test.ts` currently expects exactly 25 entries: update that count as you add entries.

**Start with a review, not a checklist.** The last round worked because each lab was first opened in the running app and judged against the project's lab rules, then given the *one* upgrade that would teach most. Do the same for each of these labs before touching code, and write a short verdict in your commit message (strong / one upgrade / needs work). Ask:
- **Is the machinery on screen?** The weakest labs showed only scores ("pick settings from a grid and watch three tiles turn green"). The fix was always to draw the thing being trained, sampled or searched, usually by reusing a widget from earlier in the same lesson. Last round: the sampler shows the words it samples, the Q-learner shows its maze and all ten learning curves, the compute budget is a point on a loss map, and the alignment lab shows the policy over its eight named replies.
- **Does the learner predict before they press?** Where a lab has a run button or a small grid of settings, ask "will this pass?" first (or a conceptual prediction, like the Goodhart curve in the alignment lab), and show the guess next to the result. That turns clicking through settings into reasoning. Examples: the Q-learner's two Train buttons, and the alignment lab's Brief 1.
- **Does anything move, and does the motion mean something?** Animate the process, never decoration: a path drawn step by step, bars easing to new heights, an agent walking its plan, pieces dropping into place. Use `useReplay` + `ReplayControls` for processes (play, pause, step, scrub), `useTweenedArray` for bars, `useFramePlayer` for short runs, and `GhostPath` for "race your earlier run". All of these jump to the end under reduced motion. Examples: the reward-hacking replay, Wire the Policy's 20 episodes, the golf ghosts, VRAM Tetris.
- **After a solve, is there something new to see?** A map of the whole solution space (Attention puzzles' 11×11 query map, the previous-word head's 180-speed strip), the reference solution beside yours (Prune it, the training-lab neuron maps), or every method's best run together (Optimizer golf). The reward is understanding: never points, badges, timers or streaks (the vision rules these out).
- **Is there a "felt" moment?** The best moments last round made a number physical: a slip into the pit happening in front of you, the four arrows turning out to be cube diagonals, the backward-pass log showing every gradient exactly 4× too big, a broken model that still trains. Look for that moment in each lab and give it room.
- **Can a lab hide a deeper lesson?** For example, the model clinic's two-cause cases, where fixing one cause isn't enough. Add a brief like that only if it teaches something the existing briefs don't, and verify by search that it behaves as claimed. No briefs just for volume.

**Honesty rules that bit last round:** captions must be true in every theme (say "the coloured ticks", not "blue ticks"), so check every claim against a screenshot. Keep Greek letters out of uppercase labels (β becomes B). Compare what the optimiser actually compares, not a truncated version of it. Don't assume an intended or reference behaviour: read the brief (the reward-hacking boat's intended route goes *over* the pad). Pin every new number with a test, and if a search can settle an open question, run it and pin the result (as with the 1,806-bit BPE floor).

**For each lab, in priority order:**
1. **Audit for early answers (required; it's a project rule).** Any optimum, best setting, exhaustive summary or "show the answer" visible before a solve must be gated: after a solve, or behind a confirmed, remembered "I'm stuck" (`AnswerReveal`, or `StuckButton` + `useStuck`). A *target number* stated as the optional goal after a pass is fine; the *setting* that achieves it is not. Every one of these 21 labs mentions an optimum or exhaustive search, so check each.
2. **Hints point at evidence, never at a setting**, and never name a locked control. If a hint depends on the brief, move it to the lesson's `.ts` and add a `hints.test.ts`.
3. **A real job:** `<LabJob>…</LabJob>` just above the widget, and `<ForReal>` (a paragraph or two) after the lab's debrief, before the next `##` heading. Check every real-world claim (library names, APIs, numbers) against a source. Network access works: you can fetch raw files from GitHub and Wikipedia's `?action=raw` with curl, and arXiv PDFs with `pdftotext`.
4. **A history card:** add an entry to `src/history.ts` and `<HistoryCard id="…" />` after `<ForReal>`. Every date, place and quote needs its own source URL, checked on the page itself. Leave out the mood line rather than invent one. Name disputes plainly. `src/history.test.ts` checks that each entry's card is in its lesson. Add new milestones to the timeline only if they're sourced.
5. **The one upgrade from your review**: machinery on screen, a prediction, meaningful motion, or a post-solve map. This is where most of the teaching value is, so give it the most time.
6. **A machine log** where the lab has discrete actions (one plain line per event, with real numbers).
7. **Optional:** a live code line (`LiveCode` + a tested `realCode.ts`) where the settings map onto a real library call (e.g. scikit-learn for Module 8, FAISS for similarity search, the PyTorch `nn.LSTM` signature). Moving the lab onto `LabShell` (keep old saves via `useLabProgress(key, seed)`).

**Labs to do** (lesson id · lab):
- [x] linear-regression · Feature Engineering + Logistic Regression
- [x] knn · Rescue the kNN Classifier
- [x] decision-trees · Build the Tabular Ensemble
- [x] clustering-pca · Compress & Segment
- [x] images-as-tensors · Build the Vision Preprocessor
- [x] convolution · Design the Kernel
- [x] cnns · Architect the CNN
- [x] object-detection · Tune the Object Detector
- [x] one-hot-vs-dense · Choose & Size the Representation
- [x] word-embeddings · Train & Debug Word2Vec
- [x] similarity-search · Tune the Vector Index
- [x] contrastive-learning · Tune Contrastive Training
- [x] rnns · Tame BPTT
- [x] lstms-grus · Wire the Gated Cell
- [x] seq2seq-attention · Break the Bottleneck
- [x] autoencoders · The Face Printer
- [x] diffusion · Spend the Step Budget
- [x] guidance · Tune the Guidance
- [x] rag · Tune the Retriever
- [x] tool-calling · Harden the Agent Pipeline
- [x] evaluation · Size Your Eval Suite

**After each lab:** `npx tsc --noEmit -p .`, the lesson's tests, then screenshots with `node scripts/drive-widget.cjs <lessonId> 0 '[{"shot":"x"}]' --title="<widget title text>"` in light, dark and matrix, and at `--width=390`. Commit with the `Authored-by` trailer and push. **At the end:** run `npm test`, `npm run build` (then check the watermark is absent from `dist/`), `node scripts/check-pages.cjs` and `--theme=matrix`. Update `docs/ROADMAP.md` "Current state" and tick this list.


## Curriculum plan: progress (docs/CURRICULUM_PLAN.md)

The owner asked for `docs/CURRICULUM_PLAN.md` to be implemented in order, WP0 → WP9, one numbered step per commit. Tick a step in the same commit that finishes it; mark a step you are in the middle of as **IN PROGRESS** with a note.

- [x] WP0.1 `ModuleRef` and `Src` components
- [x] WP0.2 Replace hard-coded module numbers
- [x] WP0.3 Replace relative lesson references
- [x] WP0.4 `src/curriculum.test.ts` and `scripts/forward-links.cjs`
- [x] WP1.1 Move `statistics` and `high-dimensions` out of Module 1
- [x] WP1.2 The Machine Map component
- [x] WP1.3 "Where this is going" openers
- [x] WP1.4 `Bridge` sections
- [x] WP1.5 Predict-first moments
- [x] WP1.6 KL divergence in `information`
- [x] WP1.7 Motion: the 3Blue1Brown pass
- [x] WP1.8 Nine maths labs:
  - [x] Lab 1 vectors · The Feature Workbench
  - [x] Lab 2 dot-product · Tune the Spam Scorer
  - [x] Lab 3 matrices · Untangle the Data
  - [x] Lab 4 matrix-multiplication · Build the Pipeline
  - [x] Lab 5 derivatives · The Sensitivity Probe
  - [x] Lab 6 gradients · Which Knob, and How Far?
  - [x] Lab 7 chain-rule · Find the Slipping Gear
  - [x] Lab 8 probability · Calibrate the Forecaster
  - [x] Lab 9 information · Twenty Questions Compressor
- [x] WP1.10 "Why this, and not something else?"
- [x] WP1.9 Re-read Module 1 as a learner
- [x] WP2.1 Move RL before fine-tuning
- [x] WP2.2 Move `similarity-search` into Module 18
- [x] WP2.3 Tokens, not words: two short notes
- [x] WP2.4 Plain-language lesson summaries
- [x] WP3 The Prologue
- [x] WP4 New lesson: the Perceptron
- [x] WP5 Era labels and the timeline
- [x] WP6.1 Opening hooks
- [x] WP6.2 "You are standing where they stood"
- [x] WP7 Era interludes (EraStrip + four interludes)
- [x] WP8.1 `svm`
- [x] WP8.2 `gans`
- [x] WP8.3 `open-questions`
- [x] WP9 Final documentation

Notes for whoever continues: **WP0–WP9 are complete.** All items in `docs/CURRICULUM_PLAN.md` have been implemented, verified, and pushed to `claude/cool-feynman-4hci4d`. See `docs/ROADMAP.md` for the full summary, deviations, dropped claims, and open questions for the owner.
- Labs follow one pattern: logic and solver in `src/lessons/<id>/<name>Lab.ts` with a test written first; tune the level data (a scratch random-search script works well) until the test shows "solvable, more than one answer, a random setting usually fails"; then the UI on `LabShell`; then `## Lab: …` before "## In code" with `LabJob`, a spoiler `Deeper` debrief, `ForReal`, `HistoryCard`, and a "You've just…" line. Bump the count in `src/history.test.ts` for each new history entry.
- Draw 2D lab plots with both axes on the same scale when distance matters (`equalScale` in `src/lessons/vectors/FeatureLab.tsx`): otherwise the picture lies about which points are close.
- Writing scripts: heredocs containing `${…}` are refused by the command-safety check; write Python/TS helpers into `scratch/` with the file tool. A `<Predict>` placed between list items breaks the MDX parse: put it before or after the list.
- `scripts/check-pages.cjs` occasionally reports a one-off "horizontal overflow" on `/learn/evaluation` at 390px; a rerun is clean.

## v2 Release Plan: progress (docs/V2_PLAN.md)

Owner-approved next release (implement in order, one work package per commit or series of commits, same quality bar and verification rules):

- [x] V2-WP1 Optional sections & guided tracks — **revised by the owner**: replaced by Foundational / "where this belongs" chips (`src/areas.ts`, `AreaChip`); no track toggle
- [x] V2-WP2 Interactive Concept Dependency Graph (`#/map`)
- [x] V2-WP3 Two-Tier Compute Offloading (Test `Float32Array` + Production WebGPU): `src/lib/compute/`, `BackendChip`, training lab on WebGPU, CPU vs WebGPU benchmark (`kv-cache`), live grokking (`open-questions`), live warmup (`lr-schedules`), GAN grid re-run on the GPU (`gans/ganGpu.ts`, `GridRerun.tsx`). Details in `docs/ROADMAP.md`.
- [x] V2-WP4 Mixture of Experts (MoE) & Sparse Routing (`moe` + Lab: Balance the Experts) — see ROADMAP "Current state"
- [x] V2-WP5 Reasoning Models & Test-Time Compute (`reasoning` + Lab: Spend the Test-Time Budget) — see ROADMAP "Current state"
- [x] V2-WP6 Multimodal Models & Vision Transformers (`multimodal-vit` + Lab: Wire the Multimodal Model) — see ROADMAP "Current state". **The v2 plan is complete.**
