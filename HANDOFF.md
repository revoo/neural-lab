# Handoff — start here

You're picking up **Neural Lab**, an interactive website that teaches AI from first principles: Feynman-style explanation, 3Blue1Brown-style visuals, Zachtronics-style labs. This file is the onboarding procedure. Follow it in order.

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
```

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
- [ ] You looked at the widgets in a real browser (Playwright screenshots are fine) in light **and** dark mode.

## 4. Before you end the session

1. Update **`docs/ROADMAP.md`**: the status table, labs built, new decisions, new limitations, and the next steps in order.
2. If the owner gave new direction, record it in **`docs/VISION.md`** under the clarifications, and in **`docs/STYLE_GUIDE.md`** if it affects how things are written.
3. Commit with clear messages and push.

## Gotchas that have already bitten someone

- **MDX:** no raw `{`, `}` or `<` in prose. JSX attribute strings don't process escapes, so `tex="\frac{a}{b}"` works as written; in JS strings (e.g. `quiz.ts`), backslashes must be doubled.
- **Colour macros:** use `\ca \cb \cc \cg \cp \cv`. Never name a macro `\cr` — TeX reserves it as "end of row", and matrices silently collapse.
- **One KaTeX version only.** `package.json` pins it via `overrides`; a second copy makes subscripts render at the baseline.
- **Heatmaps** recompute whenever their `f` prop changes identity — memoise it (`useMemo`) or define it at module scope.
- **Lessons stay independent.** Duplicate a small dataset rather than importing from another lesson's folder.
