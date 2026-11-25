<!-- authored-by: Claude Opus 5.5 -->
# Curriculum Implementation Plan

_Written November 2026 from `docs/CURRICULUM_AUDIT.md`, plus the owner's follow-up: make the maths feel less like a textbook and more like 3Blue1Brown, give the maths lessons labs of their own, and show **exactly** how each piece of maths is used in the AI lessons that follow it._

**Who this is for:** an implementing agent. Every decision has already been made. Your job is to carry it out step by step, check each step, and record progress. If a step can't be done as written, read §0.6 ("When something doesn't fit") before improvising.

---

## Contents

- §0 Rules for the implementing agent (read all of it first)
- §1 Decisions already taken, and why (read once)
- WP0 Guard rails: robust cross-references and tests (do first)
- WP1 The Maths Toolkit upgrade (the owner's priority)
- WP2 Quick fixes to order and jargon
- WP3 The Prologue
- WP4 New lesson: the Perceptron
- WP5 Era labels and the timeline
- WP6 History in the main text: opening hooks and "where they stood"
- WP7 Era interludes
- WP8 New lessons: SVMs, GANs, Epilogue
- WP9 Final documentation
- Appendix A: replacement lesson summaries (verbatim)
- Appendix B: sources to verify

---

## §0 Rules for the implementing agent

### 0.1 Before you start

1. Read `CLAUDE.md`, `HANDOFF.md` (all of it, especially "Owner's standing instructions", "Checklist: adding a lesson" and "Definition of done for a lesson"), `docs/STYLE_GUIDE.md` and `docs/CURRICULUM_AUDIT.md`.
2. Read one finished lesson end to end: `src/lessons/statistics/index.mdx` and its `Widgets.tsx`. It has a lab built with `LabShell`, `useLabProgress`, `LabJob`, `ForReal`, `AnswerReveal` and `HistoryCard`. **Copy its patterns** for every lab in this plan.
3. Run the baseline: `npm install` (if needed), `npm test`, `npm run build`. Both must pass before you change anything. If they don't, stop and record that in `docs/ROADMAP.md`.
4. Add a progress checklist to `HANDOFF.md` under a new heading `## Curriculum plan: progress (docs/CURRICULUM_PLAN.md)`, with one unticked line per numbered step in this plan (WP0.1, WP0.2, …). Tick each line in the same commit that finishes the step.

### 0.2 Branch, commits and the other agent

- Work on `claude/cool-feynman-4hci4d` only. No pull requests, no other branches.
- **Another agent may be working at the same time.** The "lab-improvement treatment" (see `HANDOFF.md`, "Next task") is going through these lessons in order: object-detection, one-hot-vs-dense, word-embeddings, similarity-search, contrastive-learning, rnns, lstms-grus, seq2seq-attention, autoencoders, diffusion, guidance, rag, tool-calling, evaluation. To avoid conflicts:
  - Before **every** step: `git fetch origin claude/cool-feynman-4hci4d && git merge --no-edit FETCH_HEAD`.
  - In lessons on that list that are still unticked in `HANDOFF.md`, make **only the minimal line edits this plan asks for**. Never reformat, reorder or rewrap their files.
  - Commit and push straight after each step. Small commits merge cleanly.
  - If a merge conflicts, resolve it by keeping **both** sides' intent (their lab changes and your reference edits). Never discard their work.
- Commit message format: `Curriculum plan WPx.y: <what changed>`, followed by the trailers required in `HANDOFF.md` (`Authored-by: <MODEL_NAME>`) and any session attribution lines.
- Watermark every new `.mdx` / `.ts` / `.tsx` file as `HANDOFF.md` describes.

### 0.3 Checks after every step

Run these in order. All must pass before you commit:

```bash
npm test
```
```bash
npm run build
```

For any step that touches a lesson page, also start the dev server and run the page checker in two themes:

```bash
npm run dev
```
```bash
node scripts/check-pages.cjs
```
```bash
node scripts/check-pages.cjs --theme=matrix
```

For any step that adds or changes a widget, take screenshots and **look at them** (light, dark, matrix, and phone width):

```bash
node scripts/shoot-widgets.cjs <lessonId>
```

After `npm run build`, check that the watermark hasn't leaked: `rg "<MODEL_NAME>" dist/` must return nothing.

### 0.4 House rules you must not break

- Concept before equation. Every key equation uses `<Eq words="…" tex="…">` with a legend as MDX children, colour macros `\ca \cb \cc \cg \cp \cv` (never `\cr`), and prose words in `\text{…}`.
- Widgets compute real values. Colours come only from CSS variables (`--c-a`, `--c-b`, `--c-c`, `--c-grad`, `--accent`, `--c-good`, `--c-bad`, …), never hex.
- Every number in prose is computed by code and pinned by a Vitest test. Every lab optimum is **computed** (exhaustive search, or "best found by searching <stated space>"), never guessed.
- Every historical claim shown on the site has a source URL. Fetch the page and check the claim before you use it (Appendix B lists the pages). **If you can't confirm a claim, leave it out.** Never invent dates, quotes or anecdotes.
- British spelling. Warm, precise, conversational prose. No badges or points.
- MDX gotchas: no raw `{`, `}` or `<` in prose. JSX attribute strings don't process escapes.
- **Text copied from this plan into lesson prose:** wherever it says "Module N", write `<ModuleRef id="…" short />` (or the full form); wherever it names a lesson, use `<LessonLink to="…">`. Never paste "Module 4", "next lesson" or "last lesson" as plain text, because the WP0.4 tests will fail. Mentions inside a JSX attribute string must be reworded without numbers.

### 0.5 Updating the docs as you go

After each WP, update the "Current state" block in `docs/ROADMAP.md` (what's done, what's next). Use `- **Done [<MODEL_NAME>]:** …` entries.

### 0.6 When something doesn't fit

- **A line number in this plan is wrong**, because files have moved on. Search for the quoted phrase instead. Line numbers are hints, not addresses.
- **A lab level turns out unsolvable, or trivially solvable.** Change the level's data, not its idea, until your solver test shows it is solvable and not trivial. Record what you changed in `docs/ROADMAP.md`.
- **A historical claim can't be verified.** Drop it, and record "dropped: <claim>, reason" in `docs/ROADMAP.md`.
- **A step would need a judgement this plan doesn't cover.** Choose the smallest change that satisfies the step's acceptance checks, then write the question under "Open questions for the owner" in `docs/ROADMAP.md`.
- **Never** skip a test, weaken an assertion, or fake a result to get green.

---

## §1 Decisions already taken, and why

Read this once, so the steps make sense. Don't reopen these decisions.

1. **The maths stays as a section at the front, but shorter and connected.** The audit suggested spreading every maths lesson just before its first use. Checking the dependencies showed that `neuron` already uses dot products, matrices ("the panel of judges"), probability, derivatives and the chain-rule-style gradient `(a − y)x`. Spreading the maths would mean rewriting the core lessons. Instead:
   - Module 1 becomes **"The Maths Toolkit"**: 9 lessons, each ending with a precise bridge to the AI lesson that uses it, and each with its own lab.
   - The two maths lessons that are used much later move to where they are used: `statistics` goes into Module 4 (before `sgd-minibatch`) and `high-dimensions` goes to the start of Module 8 (before `knn`).
   - A shared **Machine Map** diagram shows, at the top of every maths lesson, which part of a learning machine the lesson builds. This answers the owner's question: "how does this relate to what we're trying to do with AI?"
2. **Module numbers are rendered, never typed.** Prose currently hard-codes "Module 13" in 39 places and says "last lesson" or "next lesson" in 37 places. These break silently when lessons move. WP0 replaces them with components and adds a test, so later reordering is safe.
3. **RL moves before fine-tuning.** `rlhf-dpo` uses policies, rewards and PPO-style penalties, which are taught in the RL module.
4. **KL divergence gets taught in `information`**, because `rlhf-dpo` uses it before `autoencoders` explains it.
5. **Tokenization stays where it is.** The audit suggested moving it, but `tokenization` is written as "the door into the transformer you just built" (it relies on n² attention cost and the embedding table). Moving it would mean a rewrite. Instead, the first embeddings lesson and the attention lesson get a short note saying that words are really tokens, linking forward.
6. **`similarity-search` moves to the start of Module 18**, right before `rag`, which uses it.
7. **History moves into the main text**: a sourced 2–4 sentence hook in the opening of key lessons; a "You are standing where they stood" note after labs that rediscover a historical result; four short era interludes; a Prologue; era labels on every module. The existing collapsed `HistoryCard`s stay as they are.
8. **New lessons, in priority order:** Prologue (WP3), Perceptron (WP4), interludes (WP7), then SVMs, GANs and the Epilogue (WP8).

The **final module order** after all WPs (ids in `code`):

| # | Module (`id`) | Lessons, in order (new ones in **bold**, moved ones marked ↪) |
|---|---|---|
| 0 | Prologue (`prologue`) **new** | **`prologue`** |
| 1 | The Maths Toolkit (`foundations`) | vectors, dot-product, matrices, matrix-multiplication, derivatives, gradients, chain-rule, probability, information |
| 2 | The Artificial Neuron (`neuron`) | neuron, activation-functions, logic-lab, **`perceptron`**, **`interlude-first-winter`** |
| 3 | Neural Network Architecture (`architecture`) | forward-propagation, network-playground, depth-and-width |
| 4 | Loss Functions & Optimization (`optimization`) | gradient-descent, loss-functions, ↪ statistics, sgd-minibatch, optimizers, lr-schedules |
| 5 | Backpropagation (`backprop`) | backpropagation, network-from-scratch, vanishing-gradients |
| 6 | Neural Network Training Lab (`training`) | training-lab |
| 7 | Generalization & Failure Modes (`generalization`) | overfitting, regularization, diagnosis-lab |
| 8 | Classical Machine Learning (`classical`) | **`interlude-second-winter`**, ↪ high-dimensions, linear-regression, **`svm`**, knn, decision-trees, clustering-pca |
| 9 | Computer Vision (`vision`) | **`interlude-big-bang`**, images-as-tensors, convolution, cnns, object-detection |
| 10 | Embeddings & Representations (`embeddings`) | one-hot-vs-dense, word-embeddings, contrastive-learning |
| 11 | Sequence Models (`sequences`) | rnns, lstms-grus, seq2seq-attention |
| 12 | Transformers & Attention (`transformers`) | attention, multi-head-attention, positions, transformer-block, tokenization |
| 13 | Large Language Models (`llms`) | **`interlude-chatgpt`**, next-token-prediction, sampling, scaling-laws |
| 14 | Reinforcement Learning (`rl`) ↪ | mdps, q-learning, policy-gradients, reward-hacking |
| 15 | Training & Fine-Tuning (`finetuning`) | lora, rlhf-dpo |
| 16 | Inference & Hardware (`inference`) | quantization, kv-cache |
| 17 | Generative Models & Diffusion (`generative`) | autoencoders, **`gans`**, diffusion, guidance |
| 18 | AI Systems Engineering (`systems`) | ↪ similarity-search, rag, tool-calling, evaluation |
| 19 | Epilogue (`epilogue`) **new** | **`open-questions`** |

Progress is stored by lesson id (`src/progress.ts`), and URLs are `#/learn/<id>`. Reordering therefore loses no learner progress and breaks no links.

---

## WP0 Guard rails (do first)

**Goal:** make module numbers and lesson neighbours safe to change, and add tests that keep them safe.

### WP0.1 Add `ModuleRef` and `Src` components

1. In `src/components/Term.tsx`, next to `LessonLink`, add:
   ```tsx
   /** "Module 14 · Reinforcement Learning", linked to the module's first lesson. Renders the live number, so reordering modules never breaks prose. */
   export function ModuleRef({ id, short = false, children }: { id: string; short?: boolean; children?: ReactNode }) {
     const m = MODULES.find((x) => x.id === id);
     if (!m) { if (import.meta.env.DEV) console.warn(`Unknown module: ${id}`); return <>{children ?? id}</>; }
     const label = children ?? (short ? `Module ${m.number}` : `Module ${m.number} · ${m.title}`);
     const first = m.lessons[0];
     return first ? <a href={lessonHref(first.id)}>{label}</a> : <>{label}</>;
   }
   ```
   Import `MODULES` from `../curriculum`.
2. Create `src/components/Src.tsx` (watermarked):
   ```tsx
   /** A source link for a historical claim written in lesson prose. */
   export function Src({ href }: { href: string }) {
     return <sup className="src-link"><a href={href} target="_blank" rel="noreferrer" aria-label="source">[source]</a></sup>;
   }
   ```
3. Register `ModuleRef` and `Src` in `src/mdxComponents.tsx`.
4. Add `.src-link a { font-size: 0.75em; }` to `src/styles/global.css`, so the link stays small in all three themes.

**Accept:** build passes, and `<ModuleRef id="rl" />` renders "Module 16 · Reinforcement Learning" (the current number) in a test page or test.

### WP0.2 Replace hard-coded module numbers

Find every occurrence:

```bash
grep -rnoE "Modules? [0-9]+" src/lessons --include=*.mdx --include=*.tsx
```

Current number → module id: 1 `foundations`, 2 `neuron`, 3 `architecture`, 4 `optimization`, 5 `backprop`, 6 `training`, 7 `generalization`, 8 `classical`, 9 `vision`, 10 `embeddings`, 11 `sequences`, 12 `transformers`, 13 `llms`, 14 `finetuning`, 15 `inference`, 16 `rl`, 17 `generative`, 18 `systems`.

Rules:

- In MDX prose, replace "Module N · Title", "Module N (Title)" or "**Module N · Title**" with `<ModuleRef id="…" />`. Replace a bare "Module N" with `<ModuleRef id="…" short />`. Keep any surrounding bold: `**<ModuleRef id="rl" />**`.
- **Inside a JSX attribute string** (for example `<Callout … title="Module 11 complete — direct bridge to Module 12">`), components can't be used. Rewrite the title in words, without numbers: `title="Sequence models complete: the bridge to transformers"`. Known cases: `seq2seq-attention` ("Module 11 complete…") and `contrastive-learning` ("Module 10 complete…").
- References to "Module 1's *High-Dimensional Geometry*" (`knn`, `seq2seq-attention`) become `<LessonLink to="high-dimensions">the High-Dimensional Geometry lesson</LessonLink>`, because that lesson moves in WP1.1.
- In `.tsx` widget text (`seq2seq-attention/Widgets.tsx` around line 251, `transformer-block/Widgets.tsx` around line 121), import `ModuleRef` from `../../components/Term` and use it in the JSX.
- **Leave** code comments in `.ts` files (for example `rnns/rnn.ts`) as they are. Readers don't see them.

**Accept:** the grep above, run on `*.mdx` and on non-comment lines of `*.tsx`, returns nothing.

### WP0.3 Replace relative lesson references

Replace each phrase below with a named link. Use `<LessonLink to="target">the <lesson's short name> lesson</LessonLink>`, or reword the sentence naturally ("In the dot product lesson, we…"). Keep the sentence's meaning.

| File (lesson) | Approx. line | Phrase | Target lesson id |
|---|---|---|---|
| dot-product | 7 | last lesson | vectors |
| matrices | 14 | last lesson | dot-product |
| matrix-multiplication | 8 | last lesson | matrices |
| gradients | 79, 109 | last lesson | derivatives |
| neuron | 151 | next lesson | activation-functions |
| activation-functions | 9 | last lesson | neuron |
| logic-lab | 37 | last lesson | **check the content**: "one neuron draws one line" is taught in `neuron`, so link `neuron` |
| network-playground | 6 | last lesson | forward-propagation |
| network-playground | 130 | next lesson | depth-and-width |
| gradient-descent | 116 | next lesson | **check**: `loss_and_gradient` is explained in `backpropagation`, so link that if so, otherwise `loss-functions` |
| sgd-minibatch | 82 | next lesson | optimizers |
| training-lab | 76, 163 | next module | `<ModuleRef id="generalization" />` |
| overfitting | 142, 156 | next lesson | regularization |
| regularization | 6, 14, 160 | last lesson | overfitting |
| decision-trees | 7 | last lesson | knn |
| convolution | 98 | very next lesson | cnns |
| cnns | 12, 47 | previous lesson | convolution |
| object-detection | 12 | previous lesson | cnns |
| one-hot-vs-dense | 83 | very next lesson | word-embeddings |
| word-embeddings | 12 | previous lesson | one-hot-vs-dense |
| multi-head-attention | 34 | last lesson | attention |
| next-token-prediction | 160 | next lesson | sampling |
| sampling | 7 | last lesson | next-token-prediction |
| quantization | 140 | very next lesson | kv-cache |
| kv-cache | 7 | last lesson | quantization |
| mdps | 149 | next lesson | q-learning |
| q-learning | 7 | previous lesson | mdps |
| diffusion | 7, 220 | last lesson | autoencoders |
| guidance | 7, 36, 52 | last lesson | diffusion |
| guidance | 190 | last module | `<ModuleRef id="systems" />` |
| tool-calling | 7 | last lesson | rag |

### WP0.4 Add `src/curriculum.test.ts`

Write these tests (watermarked file). Read MDX with the same `node:fs` trick that `src/history.test.ts` uses.

1. **No hard-coded module numbers.** For each `src/lessons/*/index.mdx`: remove fenced code blocks (```…```) and `{/* … */}` comments, then assert there is no match for `/\bModules? \d+\b/`.
2. **No relative lesson references.** Same files, same stripping: no match for `/\b(last|previous|next|very next|following) (lesson|module)\b/i`.
3. **Every `LessonLink to` and `ModuleRef id` points at something real.** Collect `to="…"` from `<LessonLink` and `id="…"` from `<ModuleRef`, and assert each is in `ALL_LESSONS` / `MODULES`.
4. **Modules are numbered 0…n or 1…n with no gaps**, and every lesson id appears exactly once.
5. **Summaries are plain.** For every lesson: `summary.split(/\s+/).length <= 30`, and the summary doesn't match `/[Ͱ-Ͽᴀ-ᶿ⁰-₟=^_~²³]/` (Greek letters, superscripts and subscripts, `=`, `^`, `_`, `~`). **This test fails until WP2.4.** Add it with `it.todo` now and switch it to `it` in WP2.4.

Also add `scripts/forward-links.cjs`. It prints every `LessonLink` whose target comes **later** in the course than the lesson containing it, with the surrounding sentence. It is a report for a human to read, not a test. Run it after every reorder and read the output: a forward link is fine if the sentence says "later" or "coming up", and a problem if it says "remember" or "you saw".

**Accept:** tests 1–4 pass; `node scripts/forward-links.cjs` runs and prints a list.

---

## WP1 The Maths Toolkit upgrade (owner's priority)

### The verdict that motivates this

The maths prose is **not** dry. It's problem-first, uses analogies, and has live explorers. But the *structure* is textbook-like:

- every lesson follows the same template (problem → mental model → widget → "Writing it down" → "In code");
- 9 of the 11 lessons have no lab;
- 6 have no `<Predict>`;
- 4 of the core widgets don't move (derivatives, dot product, chain rule, matrix multiplication);
- the AI connection is a promise about distant things ("a transformer layer has 4,096 × 14,336 matrices", "PyTorch's `.grad`") rather than "here is the exact line where you'll use this in two lessons";
- each lesson ends by pointing to the *next maths lesson*, not to the AI that uses it.

WP1 fixes all of that.

### WP1.1 Move `statistics` and `high-dimensions` out of Module 1

1. In `src/curriculum.ts`, cut the `statistics` entry and paste it into the `optimization` module, after `loss-functions` and before `sgd-minibatch`.
2. Cut `high-dimensions` and paste it as the **first** lesson of the `classical` module.
3. Change the `foundations` module: `title: 'The Maths Toolkit'`, and `blurb: 'Nine tools that every neural network is built from. Each lesson ends by showing exactly where the next modules use it — and each has a lab of its own.'`
4. In `statistics/index.mdx`, add one sentence to the opening section: *"You'll need this straight away in <LessonLink to="sgd-minibatch">the mini-batch lesson</LessonLink>: a mini-batch gradient is an average of noisy pieces, and how much an average wobbles is exactly what this lesson measures."*
5. In `high-dimensions/index.mdx`, rewrite the opening so it no longer assumes it's in the maths module. Add: *"Before we meet k-nearest neighbours, which judges similarity by distance, it's worth finding out how strange distance becomes when there are hundreds of numbers per point."* Then check every mention of neurons, embeddings and language models in the superposition lab. Anything not yet taught by Module 8 gets a short gloss or a `<Term>`.
6. Run `node scripts/forward-links.cjs` and fix any "remember"-style link that now points forward.

**Accept:** tests pass; the sidebar shows 9 lessons in Module 1; `statistics` sits before `sgd-minibatch`; `high-dimensions` opens Module 8.

### WP1.2 The Machine Map component

Create `src/components/MachineMap.tsx` (watermarked). It is a small SVG diagram of the learning machine that Modules 1–5 build. It is real structure, not decoration.

**Stages** (`id`: label → lesson):

| id | Box label | Small caption | Lesson |
|---|---|---|---|
| `features` | Email → `[3, 6]` | measurements as a vector | vectors |
| `weighted-sum` | `w · x` | weigh the evidence | dot-product |
| `layer` | `W x` | many neurons at once | matrices |
| `stack` | `W₂(W₁ x)` | layers in a pipeline | matrix-multiplication |
| `probabilities` | softmax | scores → probabilities | probability |
| `loss` | −log p | how surprised? | information |
| `slope` | f′(x) | which way is downhill? | derivatives |
| `gradient` | ∇L | every knob's slope | gradients |
| `backward` | chain rule | blame flows backwards | chain-rule |

**Layout:**

- Top row, left to right: the forward stages `features → weighted-sum → layer → stack → probabilities → loss`.
- Bottom row, right to left, with arrows pointing back: `backward ← gradient ← slope`, labelled "learning". A curved arrow goes from `loss` down to `slope`, and another from `backward` back up to the weights.
- At phone width (below 560px), stack it vertically as two columns. Never scroll horizontally.

**Props:** `highlight?: string` (one stage id) and `caption?: ReactNode`. The highlighted stage uses `--accent` for border and text. The others use `--border` / `--text-muted`. Each box is a link (`lessonHref`) to its lesson. Under the diagram, a one-line note reads: *"You are here: <stage caption>."*

Register it in `mdxComponents.tsx`. Screenshot it in all three themes and at 390px width.

**Accept:** renders in all themes with no overflow, and each box's link goes to the right lesson.

### WP1.3 "Where this is going" openers

In each of the 9 Module 1 lessons, insert the block below **immediately before the second `## ` heading** (so the opening problem comes first, then the map). Use the text exactly, then check that each sentence matches the target lesson (open it and confirm the thing named really is there; if a detail differs, correct the sentence to match the lesson).

```mdx
<MachineMap highlight="features" />

**Where this is going.** …text below…
```

| Lesson | highlight | Text |
|---|---|---|
| vectors | features | In Module 2 you'll build an artificial neuron, and the very first thing it does is receive a list of measurements, like `[3 links, 6 exclamation marks]`. This lesson is about that list: why it can be read as an arrow, and why the arrow reading is what makes learning possible. |
| dot-product | weighted-sum | A neuron's first step is to weigh each piece of evidence and add it up. That step is a dot product, and so is the "how relevant is this word?" score at the heart of every transformer. |
| matrices | layer | One neuron is one dot product. A layer of neurons is many dot products at once, and writing them as a single matrix is what lets a GPU compute a whole layer in one go. |
| matrix-multiplication | stack | A network is layers in a pipeline: the output of one matrix becomes the input of the next. Multiplying matrices is how you compose those stages, and the shape rules here are the ones that crash or run your model code. |
| derivatives | slope | Training means turning knobs to make a model less wrong. The derivative answers the only question training ever asks a knob: if I turn you slightly, does the error go up or down, and by how much? |
| gradients | gradient | A real model has millions of knobs. The gradient collects every knob's slope into one arrow that points straight uphill, and gradient descent, in Module 4, walks the other way. |
| chain-rule | backward | A network is a chain of stages, and a weight deep inside it affects the error only through everything after it. The chain rule is how blame flows backwards through that chain. It is the whole idea behind backpropagation, in Module 5. |
| probability | probabilities | A classifier shouldn't just shout "cat". It should say how sure it is. Softmax turns a network's raw scores into probabilities, and it's the last layer of nearly every classifier and language model. |
| information | loss | To learn, a model needs a single number saying how wrong it was. "How surprised were you by the right answer?" turns out to be the best such number, and it's the loss behind every classifier and language model. |

In the rendered text, use `<ModuleRef id="neuron" short />` (and so on) instead of typing "Module 2".

**Accept:** each of the 9 lessons shows the map with the right stage highlighted, and the sentence below it.

### WP1.4 `Bridge` sections: exactly where the maths is used next

1. Create `src/components/Bridge.tsx` (watermarked):
   ```tsx
   /** End-of-lesson card: where this maths is used next, with a link. */
   export function Bridge({ to, children }: { to: string; children: ReactNode }) {
     const l = findLesson(to);
     return (
       <aside className="callout bridge">
         <div className="callout-title">Where you'll use this next{l ? <> · <a href={lessonHref(to)}>{l.title} →</a></> : null}</div>
         {children}
       </aside>
     );
   }
   ```
   Add `.callout.bridge` styling in `global.css` that copies the `insight` callout but uses `--accent` for the left border. Register it in `mdxComponents.tsx`.
2. In each Module 1 lesson, add a `<Bridge>` **after `<Takeaways>` and before the final "next lesson" teaser paragraph**. Each Bridge has three parts:
   - (a) one sentence naming the exact place;
   - (b) the exact expression as it appears there, in `$…$` with the same colour macros;
   - (c) one sentence on what would break without this lesson's idea.

   Draft text is below. **Before using it, open the target lesson and copy its actual notation into (b).** If the target writes it differently, use the target's form.

| Lesson | `to` | (a) and (c) |
|---|---|---|
| vectors | neuron | The neuron lesson feeds exactly these email vectors into a neuron, and its decision boundary is a line in this same arrow space. *Without the arrow reading, "the neuron draws a line" would mean nothing.* Also mention: the units trap in this lesson returns in `knn` (feature scaling). |
| dot-product | neuron | The neuron's first step is the score `w · x + b`, your spam score plus a cut-off. *In attention (<ModuleRef id="transformers" />), the same dot product scores how much one word should look at another.* |
| matrices | forward-propagation | A layer computes `W x + b`: each row of `W` is one neuron's weight vector, the "panel of judges" from this lesson. *Without non-linearity, any stack of these is still one matrix, which is why activation functions exist.* Link `activation-functions`. |
| matrix-multiplication | forward-propagation | A batch of 32 inputs goes through a layer as one product `X Wᵀ`, and the shape rules decide whether it runs. *`depth-and-width` counts parameters as the sizes of these matrices.* |
| derivatives | gradient-descent | Gradient descent nudges each weight against its slope. *`network-from-scratch` checks its own backward pass by measuring slopes with the finite differences from this lesson.* |
| gradients | gradient-descent | One training step is `w ← w − η ∇L`: the gradient from this lesson, scaled by the learning rate. |
| chain-rule | backpropagation | Backpropagation is the chain rule applied in reverse order, reusing each product once, which turns billions of slope calculations into about three passes. *`vanishing-gradients` shows what happens when many small factors multiply.* |
| probability | loss-functions | The classifier's output is the softmax distribution from this lesson, and cross-entropy compares it with the truth. *`sampling` uses the same temperature dial to make language models more or less adventurous.* |
| information | loss-functions | Cross-entropy, `−ln p(correct)`, is the loss that trains every classifier and language model. *KL divergence returns as the "leash" in `rlhf-dpo` and the pull towards the bell in `autoencoders`.* |

**Accept:** add test 6 to `src/curriculum.test.ts`: every lesson in the `foundations` module contains `<Bridge to="X"`, where X is a lesson in a **different module that comes later** in `ALL_LESSONS`.

### WP1.5 Predict-first moments (at least 2 per maths lesson)

Add each `<Predict>` **before** the widget or section that reveals the answer, using the component's real API: `question`, `options`, `answer` (index), and the explanation as children. Pin every number in a test in that lesson's existing test file, or in a new `predicts.test.ts` in the lesson folder.

| Lesson | Question | Options (answer **bold**) | Explanation (children) |
|---|---|---|---|
| dot-product | Two weight arrows point in exactly perpendicular directions. What is their dot product? | **0** / 1 / it depends on their lengths | Perpendicular means no agreement at all: each arrow's shadow on the other has length zero, whatever their lengths. |
| dot-product | You double every number in the email vector. What happens to its spam score `w · x`? | **it doubles** / it stays the same / it quadruples | The dot product is linear in each argument, so twice the evidence gives twice the score. That's why the cosine section that follows divides by length. |
| matrices | A matrix sends î to (2, 0) and ĵ to (1, 1). Where does the point (1, 1) land? | **(3, 1)** / (2, 1) / (1, 2) | (1, 1) is one î plus one ĵ, so it lands at (2, 0) + (1, 1) = (3, 1). The columns are the whole recipe. |
| matrices | A matrix has columns (1, 2) and (2, 4). Can its effect be undone? | yes / **no** | Both columns point along the same line, so the whole plane is squashed onto it (determinant 0). Points that differed now coincide, and no matrix can separate them again. |
| matrix-multiplication | A batch of shape (32, 784) is multiplied by a weight matrix of shape (784, 128). What shape comes out? | **(32, 128)** / (784, 784) / (32, 784) | The inner 784s match and disappear; the outer sizes remain: 32 inputs, each now 128 numbers. |
| matrix-multiplication | "Rotate 90°, then shear" versus "shear, then rotate 90°": the same transformation? | same / **different** | Matrix multiplication isn't commutative: the order of stages matters, just as in any pipeline. Try both in the composition demo. |
| derivatives | What is the slope of x² at x = 3? | 3 / **6** / 9 | The derivative of x² is 2x, so 6. Check it with the explorer. |
| derivatives | You shrink the nudge h to 10⁻¹⁵. Does the slope estimate get better? | **no, it goes haywire** / yes, it's nearly exact / it stops changing | Floating-point numbers have about 16 significant digits. Subtracting two nearly equal values leaves mostly rounding noise, which is why gradient checks use h around 10⁻⁵ with central differences. |
| gradients | On a hillside, you walk exactly perpendicular to the gradient arrow. Does your height change, for a small step? | **no** / yes, it rises / yes, it falls | Change ≈ gradient · step, and a perpendicular step has zero dot product. You're walking along a contour line. |
| gradients | The gradient is (3, 4). How steep is the steepest climb, per unit step? | **5** / 7 / 12 | Its length: √(3² + 4²) = 5. The gradient's direction is the steepest way up, and its length is the steepness. |
| chain-rule | Three gears in a chain multiply turns by 2, 3 and 0.5. Turn the first gear once: how far does the last turn? | **3** / 5.5 / 6 | Sensitivities multiply along a chain: 2 × 3 × 0.5 = 3. |
| chain-rule | Ten sigmoid stages in a row, each with slope at most 0.25. At most, how much of a nudge at the start reaches the end? | about 1 / about 0.25 / **about one millionth** | 0.25¹⁰ ≈ 9.5 × 10⁻⁷. This is the vanishing gradient, and Module 5 is about rescuing it. |
| probability | What does softmax give for scores [2, 2, 2]? And for [102, 102, 102]? | **⅓ each, both times** / ⅓ then nearly all on one / different, because 102 is bigger | Softmax only cares about differences between scores. Adding 100 to all three changes nothing. |
| probability | You turn the temperature down towards 0. What happens to the probabilities? | they become equal / **almost all goes to the top score** / nothing | Dividing by a tiny temperature stretches the differences enormously, so the largest score takes nearly everything. |
| information | An event has probability 1/8. How surprising is it, in bits? | 1 / **3** / 8 | −log₂(1/8) = 3: as surprising as three fair coin flips all landing your way. |
| information | A model says 50/50 on a yes/no question, and the answer is yes. What is its cross-entropy, in bits? | 0 / 0.5 / **1** | −log₂ 0.5 = 1 bit. A model that always says 50/50 pays exactly one bit per question, the price of knowing nothing. |

(`vectors` already has one Predict, about units. Add a second one before "The double life of a vector" ends: *"Two emails are `[3, 6]` and `[6, 12]`. Do their arrows point in the same direction?"* Options: **yes, one is just longer** / no / only if they are both spam. Explanation: the second is the first scaled by 2, so same direction and twice the length. Direction-versus-length is exactly what cosine similarity separates in the dot product lesson.)

### WP1.6 KL divergence in `information`

1. In `src/lib/prob.ts`, add:
   ```ts
   /** KL divergence D(p‖q) = Σ p_i log(p_i / q_i): extra surprise from believing q when the truth is p. Infinity if some q_i = 0 < p_i. */
   export function klDivergence(p: number[], q: number[], base = 2): number
   ```
   Terms with `p_i = 0` contribute 0.
2. In `src/lib/lib.test.ts`, add tests:
   - `klDivergence(p, p) = 0`;
   - `klDivergence([0.9, 0.1], [0.5, 0.5]) ≈ 0.5310` (4 dp);
   - `klDivergence([0.5, 0.5], [0.9, 0.1]) ≈ 0.7370`;
   - it equals `crossEntropy(p, q, 2) − entropy(p, 2)` for random p, q (seeded);
   - it returns `Infinity` when q has a zero where p doesn't;
   - `klDivergence([0.5, 0.25, 0.125, 0.125], [0.25, 0.25, 0.25, 0.25]) = 0.25`.

   (These numbers were computed while writing this plan. Your tests re-check them.)
3. In `src/lessons/information/index.mdx`, add a new section **after** "Cross-entropy: the model's surprise at the truth" and **before** "In code":

   `## KL divergence: the price of believing the wrong thing`

   - Open with a concrete case: a weather app believes rain and sun are 50/50, but here it's sunny 90% of the time. On average, how much more surprised is it than it would be if it knew the truth?
   - The idea in words: cross-entropy splits into two parts: the surprise that's unavoidable (the entropy of the truth) plus the extra surprise caused by wrong beliefs. That extra part is the KL divergence.
   - The Eq:
     ```
     words="\text{extra surprise} = \text{surprise when believing the model} - \text{surprise when knowing the truth}"
     tex="\cv{D_{\text{KL}}(\ca{p} \,\|\, \cb{q})} = \cp{H(\ca{p}, \cb{q})} - \cc{H(\ca{p})}"
     ```
     The legend names p (**the truth**, blue), q (**the model's belief**, orange), H(p, q) (**cross-entropy**), H(p) (**entropy**) and D_KL (**the extra surprise**). Add a second line in the legend: "Written as one sum: Σ pᵢ log₂(pᵢ / qᵢ)."
   - A new widget `KLExplorer.tsx` in the lesson folder: two 3-outcome distributions, each with sliders (three per distribution, renormalised to sum to 1), drawn as side-by-side bars in `--c-a` (p) and `--c-b` (q). Live readouts: H(p), H(p, q), D(p‖q) **and** D(q‖p). The live note says which is bigger and why: being confidently wrong costs more than being vaguely wrong. Caption: *"Make q equal to p and watch both KL readouts go to zero. Then make p confident and q flat, and swap them."*
   - `<Predict>` before the widget: *"Is D(p‖q) the same as D(q‖p)?"* Options: yes / **no**. Explanation uses the computed numbers: 0.531 bits one way, 0.737 bits the other way, for p = 90/10 and q = 50/50.
   - `<Misconception>`: "KL is a distance between distributions." It isn't symmetric (as you just saw), and the triangle inequality doesn't hold.
   - A connection paragraph: training with cross-entropy minimises KL from the data to the model, because the data's entropy is a fixed constant. Two forward links: "KL returns as the leash in <LessonLink to="rlhf-dpo">RLHF</LessonLink> and as the pull towards the bell curve in <LessonLink to="autoencoders">autoencoders</LessonLink>."
   - Add two lines to `<Takeaways>`.
4. Add a parameterised numeric question to `information/quiz.ts`, id `information.kl`. Pick p at random from `[1/2,1/4,1/8,1/8]`, `[1/2,1/2,0,0]`, `[1/2,1/4,1/4,0]` and `[1/4,1/4,1/4,1/4]`; q is uniform over 4. Answer = 2 − H(p) bits (0.25, 1, 0.5, 0). The explanation describes a "20 questions" code built for the wrong belief.
5. In `src/glossary.ts`, change `kl-divergence`'s `lesson` to `'information'`.
6. In `rlhf-dpo/index.mdx`, make the first mention of KL a `<Term id="kl-divergence">`. In `autoencoders/index.mdx`, change the KL introduction to *"KL divergence, which you met in <LessonLink to="information">the information lesson</LessonLink>, …"*. Keep that edit to one sentence: autoencoders is on the other agent's list.

### WP1.7 Motion: the 3Blue1Brown pass

Four core widgets don't move. Add **one** animation to each, built with `useFramePlayer` / tweens from `src/components/motion.ts`, with a Play/Pause button and `ReplayControls` where they fit. Every frame computes real values. With `usePrefersReducedMotion()`, jump straight to the end state.

| Widget | Animation | Live note during playback |
|---|---|---|
| `derivatives/TangentExplorer.tsx` | "Play: shrink the nudge": h tweens from 1 to 0.001 on a log scale over about 3 s. The secant line visibly rotates into the tangent, and the slope readout converges. | "h = 0.25: secant slope 6.25 → …" (live values) |
| `dot-product/DotExplorer.tsx` | "Play: rotate": w sweeps 360° around x. The dot product is traced as a curve underneath, which turns out to be a cosine wave. | "angle 90°: dot product 0, no agreement" |
| `chain-rule/ChainExplorer.tsx` | "Play: send a nudge": a small input nudge travels stage by stage, each stage's output nudge drawn as a pulse sized by the running product. | "after stage 3: nudge × 2 × 3 = 6" |
| `matrix-multiplication/CompositionDemo.tsx` | "Play: first A, then B": the grid morphs through A, pauses, then morphs through B. A second button does B then A, so the learner sees the end states differ. | "after A: … after B: …" |

**Colour contract.** Check each Module 1 lesson. The same quantity must have the same colour in the `<Eq>` (via `\ca` etc., defined in `src/katexMacros.ts`) and in its widget. For example, if the widget draws the weight vector in `--c-a`, the Eq writes `\ca{\mathbf{w}}`. Fix whichever side is inconsistent. Write down what you changed in the commit message.

### WP1.8 Nine maths labs

General rules for every lab:

- Build it the way the existing labs are built (`statistics/Widgets.tsx` is the model): `LabShell` with briefs, `useLabProgress('<lessonId>-lab')`, `LabJob` above it, `AnswerReveal` for optimums and solution maps (only after a solve or "I'm stuck"), `MachineLog` for a running log, `ForReal` after it, and `HistoryCard` where an entry is given.
- Put the logic in `src/lessons/<id>/<name>Lab.ts`, with `<name>Lab.test.ts` beside it. The UI goes in `<Name>Lab.tsx`.
- Every brief has a pass condition checked automatically, and allows **more than one valid solution**.
- Every optimum is computed by a solver in the `.ts` file. The test runs the solver and pins the result. In the UI, an optimum is labelled "the best possible" only if the search was exhaustive; otherwise "best found by searching <space>".
- Write the solver and its test **first**. Then tune each level's data until the test shows the level is solvable but not trivial: a random setting should fail. That check can be part of the test (for example "fewer than 30% of random configurations pass").
- Place the lab section (`## Lab: <name>`) **after** the lesson's main widgets and **before** "In code".
- For new history entries, add them to `src/history.ts` with sourced claims, and raise the count in `src/history.test.ts`.
- Each lab ends with a one-line "You've just…" sentence pointing to its `Bridge`.

---

#### Lab 1 · vectors · "Lab: The Feature Workbench"

- **LabJob:** "A mail team wants every email turned into two numbers so that similar emails end up close together. Choose the measurements, and the units."
- **Data:** 12 hand-written emails (6 spam, 6 not), each with 6 precomputed integer features: links, exclamation marks, ALL-CAPS words, mentions of the recipient's name, length in characters, hour sent. Keep the data in the `.ts` file, with the email text shown on hover.
- **Primitives:** pick the x-axis feature and the y-axis feature (2 of 6, so 15 pairs); per axis, a units dial: ×0.01, ×0.1, ×1, ×10, or "standardise" (subtract the mean, divide by the SD).
- **Machinery on screen:** the 12 points plotted; for the email you select, a line to its nearest neighbour (Euclidean distance) and the distance value.
- **Score:** "neighbour agreement": the number of emails (out of 12) whose nearest other email has the same label.
- **Briefs:**
  1. Reach at least 10/12.
  2. Reach 12/12.
  3. *The units trap:* the axes are locked to (links, length in characters) with raw units. The score is poor because length dominates the distance. Reach 12/12 by changing **only units**.
  4. *(Optional optimisation)* Reach 12/12 with a pair that doesn't use `links`.
- **Solver:** enumerate all 15 pairs × 5 × 5 unit settings = 375 configurations and score each one. The test pins how many configurations reach 12/12, which must be ≥ 3 (multiple solutions), and asserts that the brief-3 raw setting scores ≤ 8.
- **AnswerReveal:** a grid of feature pairs × units, coloured by score.
- **History entry** `feature-workbench`: Fisher's iris data (1936): 150 flowers, each measured as 4 numbers. Source: `WP('Iris_flower_data_set')`.
- **Bridge line:** "You've just done feature engineering, and the nearest-neighbour score you used *is* the k-nearest-neighbours classifier, which you'll meet properly in Module 8."

#### Lab 2 · dot-product · "Lab: Tune the Spam Scorer"

- **LabJob:** "Rank incoming mail so that every spam email sits above every real one in the junk queue."
- **Data:** 10 emails with standardised features (links, exclamation marks), plus a third feature (mentions of your name) used in brief 3.
- **Primitives:** a draggable weight arrow `w` in 2D. In brief 3 this becomes three sliders.
- **Machinery:** each email's score `w · x` drawn as a bar, the list sorted live, and the projection "shadow" of each point onto `w`.
- **Briefs:**
  1. Perfect ranking: every spam scores above every non-spam.
  2. *Widest gap:* `w` is locked to length 1. Maximise (lowest spam score − highest non-spam score). Pass at ≥ 95% of the optimum.
  3. *Negative evidence:* with the third feature, a perfect ranking requires a negative weight on "mentions your name". The data must make that true; the solver test asserts it.
  4. *Length cheat:* an eleventh email is a copy of a real email with every count doubled (a long email). Its raw score doubles. Switch the scorer to cosine similarity and get a perfect ranking.
- **Solver:** sweep the angle of `w` over 0–360° in 0.01° steps for briefs 1–2 (exhaustive at that resolution; say so), and over a 3D grid for brief 3. The test pins the optimum gap and the range of angles that pass brief 1 (it must be a range, not a single angle).
- **AnswerReveal:** the band of angles that rank perfectly, drawn as a shaded wedge.
- **History entry** `spam-scorer`: Gerard Salton's vector space model for information retrieval (documents as vectors compared by cosine). Source: `WP('Vector_space_model')`, and check the dates on that page.
- **Bridge line:** "Add a cut-off and a squash to this score, and you have the neuron in Module 2."

#### Lab 3 · matrices · "Lab: Untangle the Data"

- **LabJob:** "Two kinds of points are mixed up on the screen. Find a transformation of space that moves all the orange ones to the right of the vertical axis and all the blue ones to the left."
- **Primitives:** drag the tips of î and ĵ (the matrix's columns). The 2×2 matrix is shown live, and the grid and points move together. Unlocked in brief 3: a shift vector `b`. Unlocked in brief 4: a "kink" toggle that applies ReLU after the matrix.
- **Pass check:** after the transform, every orange point has x > 0 and every blue point has x < 0. In brief 4 the goal changes: any straight line the learner draws (two draggable points) separates the colours.
- **Briefs:**
  1. The classes are separable by a tilted line through the origin. Solve it.
  2. Same data, but |det| must stay between 0.9 and 1.1 ("no stretching or squashing allowed"). Rotations and shears only.
  3. The classes are separable only by a line that **doesn't** pass through the origin. A matrix alone can't do it, because a matrix keeps the origin fixed; the brief text says this *after* two failed attempts or "I'm stuck". Then unlock the shift `b` and solve it. This is the neuron's bias.
  4. XOR corners: orange at (1, 1) and (−1, −1), blue at (1, −1) and (−1, 1). Impossible with a matrix and a shift, because the transformed x-coordinate is (first row) · p + b, a single straight line. Unlock the kink: ReLU(M p + b), then any line. A known solution, pinned in a test, is M = [[1, −1], [−1, 1]], b = 0: orange lands at (0, 0) and blue at (2, 0) and (0, 2), so the line x + y = 1 separates them.
- **Solver/tests:** brief 1, by grid search over matrix entries in [−2, 2] at 0.05: count the passing matrices (must be many). Brief 3: prove impossibility without `b`, by asserting the data has a point of each colour on both sides of every line through the origin (check all angles at 0.01°). Brief 4: show no (M, b) on the grid passes without ReLU, and the pinned solution passes with it.
- **History entry** `untangle`: Fisher's linear discriminant (1936): find the direction that best separates two classes. Source: `WP('Linear_discriminant_analysis')`.
- **Bridge line:** "Brief 3 was a neuron's bias; brief 4 was the reason networks need activation functions. Both return in Modules 2 and 3."

#### Lab 4 · matrix-multiplication · "Lab: Build the Pipeline"

First read `matrix-multiplication/ShapeLab.tsx`. If it already has a shape-chain puzzle, **extend it** with the briefs below instead of building a second widget.

- **LabJob:** "Wire up a digit reader: 784 pixel values in, 10 scores out, for a batch of 32 images, as cheaply as you can."
- **Primitives:** a parts bin of weight matrices, each drawn as a tile labelled with its shape (rows × columns): 784×128, 784×64, 784×10, 128×128, 128×64, 128×10, 64×128, 64×10, 10×128, 10×64. Drag tiles into a chain after the input `X` (32×784).
- **Machinery:** each intermediate shape shown under the chain, and the first mismatch highlighted in red with the rule "the inner sizes must match"; a parameter counter; a multiply-add counter (Σ 32 × n_in × n_out).
- **Briefs:**
  1. Any valid chain ending at 32×10.
  2. Valid, with at least one hidden width ≥ 64, using the **fewest parameters**.
  3. Valid, with at least two hidden layers, using the **fewest multiply-adds**.
  4. *Order matters:* a separate 2×2 panel. A target grid is drawn (for example "rotate 90°, then shear"). Build it from primitives (rotate 90°, shear-x by 1, flip-x, scale ×2) in the fewest pieces. The lab shows what you get if the last two pieces are swapped.
- **Solver:** enumerate chains of length up to 3 from the bin (exhaustive; pin the optima for briefs 2 and 3). BFS over primitive sequences up to length 4 for brief 4 (pin the minimum length and confirm that more than one minimal sequence exists, or else make the target allow it).
- **History entry** `pipeline`: Strassen (1969) showed that the schoolbook way of multiplying matrices isn't the fastest. Source: `WP('Strassen_algorithm')`.
- **Bridge line:** "This is the forward pass in Module 3: a batch flowing through weight matrices, with the bill counted in parameters and multiply-adds."

#### Lab 5 · derivatives · "Lab: The Sensitivity Probe"

- **LabJob:** "A colleague hands you a black-box function. You can run it, but you can't read it. How sensitive is its output at one particular setting?"
- **Primitives:** a "probe" button that evaluates f(x) at any x you type (each probe goes into the log and counts against the budget); a calculator row that computes (f(a) − f(b)) / (a − b) from two probes you choose.
- **Black boxes (one per brief):**
  1. a smooth cubic, for example 0.5x³ − x at x₀ = 1.3; report the slope within ±0.01;
  2. a one-weight "model loss" curve; report the slope within ±0.001, using **at most 2 probes**;
  3. a kink exactly at x₀ (a ReLU-like corner). The honest answer is two slopes: report the left slope and the right slope, each within ±0.01;
  4. *Floating point:* f is computed in float32 (wrap each arithmetic step in `Math.fround`). Get within 1e-4 using 2 probes. A very small h (such as 1e-7) gives garbage; the learner has to choose h well, and the central difference (x₀ − h, x₀ + h) helps.
- **Solver/tests:** for each brief, sweep h over 10⁻¹…10⁻¹⁰ for forward and central differences and assert that brief 2's tolerance is reachable with central differences. For brief 4, assert that h = 1e-7 fails, that some h passes, and record the passing range.
- **AnswerReveal:** the error-versus-h plot (a V shape on log-log axes), for both methods.
- **History entry** `probe`: the Leibniz–Newton dispute over who invented calculus. Source: `WP('Leibniz–Newton_calculus_controversy')`.
- **Bridge line:** "Measuring slopes like this is exactly how `network-from-scratch` checks its own backward pass, and it's this project's own rule for testing every gradient."

#### Lab 6 · gradients · "Lab: Which Knob, and How Far?"

- **LabJob:** "A tiny model has three knobs and a loss. Get the loss under the target in as few moves as possible."
- **Loss:** a fixed tilted quadratic bowl in 3 knobs (choose a positive-definite matrix with unequal axes). The learner sees the loss value, and partial derivatives only after pressing "probe" (each probe is logged).
- **Primitives:** brief 1: "turn one knob by ±0.1" buttons. Brief 2: one "move" = choose a direction (three sliders) and a step length; the move is applied to all knobs at once.
- **Briefs:**
  1. Get the loss under the target using only one-knob turns. Show the move count.
  2. Same target, in at most K combined moves. Moving along −gradient is the natural solution, but K is chosen so that some other directions pass too.
  3. *Contour walker:* move a total distance ≥ 1 while keeping the loss within ±0.001 of where you started. The trick is to move perpendicular to the gradient.
- **Solver/tests:** brief 1, BFS over the 0.1 grid (pin the minimum number of moves). Brief 2, pin K as the number of steepest-descent moves with the best fixed step length found by searching 0.01…2 in steps of 0.01 (label it "best found by searching"), and assert that a second, different direction sequence also passes. Brief 3, verify that a perpendicular path passes and an axis-aligned path fails.
- **History card:** none. Link to the existing descent-golf card instead: "Cauchy's 1847 rule is next door, in `gradient-descent`."
- **Bridge line:** "Brief 2's best move, straight downhill, is gradient descent, and Module 4 automates it."

#### Lab 7 · chain-rule · "Lab: Find the Slipping Gear"

- **LabJob:** "A five-stage pipeline produces the wrong output, and you may adjust exactly one stage's knob. Which knob moves the output most?"
- **Pipeline:** five stages, each with a knob kᵢ, for example y₁ = k₁x, y₂ = tanh(k₂y₁), y₃ = k₃y₂ + 1, y₄ = σ(k₄y₃), y₅ = k₅y₄. Clicking a stage shows its local slope.
- **Briefs:**
  1. *Rank the knobs* by influence on the output (drag them into order; use the existing `order` question pattern for the UI). The checker compares against products of local slopes.
  2. *One shot:* you get a single adjustment to hit the target output within ±0.01. Choose the knob and the amount using its sensitivity (a linear estimate). The amount must be small enough for the estimate to hold.
  3. *The dead stage:* stage 4's sigmoid is saturated (its slope is tiny), so every knob upstream of it has almost no influence. Change k₃ so that stage 4 leaves saturation, until knob 1's influence rises above a threshold.
  4. *The backward walk:* compute all five sensitivities, multiplying as little as possible. The counter shows multiplications used. Doing each knob separately repeats work; walking backwards once and reusing the running product is optimal.
- **Solver/tests:** check every sensitivity against central finite differences (the project rule). Pin the minimal number of multiplications for brief 4 by counting the backward-reuse strategy, and assert that the per-knob strategy uses more.
- **History entry** `slipping-gear`: Leibniz's early use of the chain rule (1676), plus reverse-mode differentiation (Linnainmaa, 1970) as the "after". Sources: `WP('Chain_rule')` and `WP('Seppo_Linnainmaa')`. Check both dates on the pages.
- **Bridge line:** "Brief 4 is backpropagation in miniature, and brief 3 is the vanishing gradient. Module 5 does both on real networks."

#### Lab 8 · probability · "Lab: Calibrate the Forecaster"

- **LabJob:** "A weather model outputs four raw scores: sun, cloud, rain and snow. Make its forecast match what actually happens."
- **Target:** observed counts over 100 days, for example 50 / 30 / 15 / 5.
- **Primitives:** four logit sliders and a temperature slider.
- **Machinery:** softmax bars next to the target bars; the largest gap shown; the logit differences shown with "× e^difference" ratios.
- **Briefs:**
  1. Match all four within ±0.02.
  2. Same, with one logit locked at 0. Shift invariance makes this always possible.
  3. *Temperature only:* the logits are fixed at c × ln(target) for some c ≠ 1. Find the temperature that matches. The answer is T = c; pin it.
  4. *The zero:* the target says snow is 0%. Softmax can never output exactly 0. Get the gap below 0.005, and answer a follow-up `Predict` explaining why "exactly 0" is impossible.
- **Solver/tests:** the exact solution is logits = ln(target) + any constant. Assert that briefs 1–3 are solvable, and check brief 4's bound.
- **History entry** `forecaster`: softmax is the Boltzmann distribution from statistical physics (temperature is literally temperature); the name "softmax" was coined by John Bridle (1990). Sources: `WP('Boltzmann_distribution')` and `WP('Softmax_function')`. Check both on the pages.
- **Bridge line:** "Cross-entropy, in the information lesson and in Module 4, scores exactly these bars against the truth, and temperature returns when language models sample."

#### Lab 9 · information · "Lab: Twenty Questions Compressor"

- **LabJob:** "Design a yes/no question strategy that identifies the hidden item in as few questions as possible, on average."
- **Primitives:** a tree builder. Start with all items in one box, click to split a box into a "yes" group and a "no" group, and repeat until every box holds one item. Show each item's depth, and the average depth weighted by probability.
- **Briefs:**
  1. 4 items with probabilities [1/2, 1/4, 1/8, 1/8]: average ≤ 1.75 questions, which equals the entropy, the best possible.
  2. 8 animals with [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/128]: reach the entropy, 1.984375 questions.
  3. 4 items with [0.4, 0.3, 0.2, 0.1]: the best possible is 1.9 (Huffman), while the entropy is 1.8464. Reach 1.9, and see the floor you can't reach with whole questions.
  4. *Wrong beliefs:* the strategy is locked to the one that's best for a uniform belief (2 questions each), but the truth is brief 1's distribution. Predict the extra cost, then measure it: 2 − 1.75 = 0.25 questions. That gap is the KL divergence from WP1.6, and the lab labels it so.
- **Solver/tests:** implement Huffman coding and assert its averages (1.75, 1.984375, 1.9) and the entropies. Check brute force over all trees for the 4-item cases (exhaustive), and assert that more than one optimal tree exists for brief 1 or 3 (if not, add an item so that one does).
- **History entry** `twenty-questions`: David Huffman's 1952 optimal code, found as an MIT graduate student when given the choice of a term paper or a final exam, improving on his professor's method. Check every detail against `WP('Huffman_coding')` and drop any part the page doesn't support. Add Kullback and Leibler (1951) as the "after", source `WP('Kullback–Leibler_divergence')`.
- **Bridge line:** "Cross-entropy loss is the cost of the code your model's beliefs imply, and KL is the waste. Both return in Module 4, in language models (perplexity), and in RLHF."

**Accept (each lab):** the HANDOFF "Definition of done" for labs; solver tests green; screenshots checked in all themes and at phone width; one commit per lab.

### WP1.9 Re-read Module 1 as a learner

After all of WP1, read Module 1 top to bottom in the browser, then fix the following:

- any closing teaser that now duplicates its `Bridge`;
- any section that is now too long because a lab repeats it (cut the prose, and keep the lab);
- the `minutes` values in `curriculum.ts` (add about 15 minutes per lab).

---

## WP2 Quick fixes to order and jargon

### WP2.1 Move RL before fine-tuning

1. In `src/curriculum.ts`, move the whole `rl` module object to just after `llms`. Renumber: `rl` 14, `finetuning` 15, `inference` 16. Every `number` field must stay consecutive; test 4 checks this.
2. In `rlhf-dpo/index.mdx`, at the start of section 1, add a recap paragraph: *"In <ModuleRef id="rl" /> an agent learned a policy, a rule for choosing actions, by chasing rewards, and the reward-hacking lab showed what happens when the reward is slightly wrong. RLHF is the same machine with a language model as the policy and a learned model of human preferences as the reward."* Link "policy" to `policy-gradients` and "reward-hacking lab" to `reward-hacking`.
3. In `policy-gradients/index.mdx`, find the ChatGPT/RLHF mention. If it implies RLHF was already taught, reword it to "coming up in <ModuleRef id="finetuning" />".
4. Run `node scripts/forward-links.cjs` and fix any new "remember"-style forward links.

### WP2.2 Move `similarity-search` into Module 18

1. In `curriculum.ts`, move `similarity-search` to be the first lesson of `systems`.
2. `grep -n "similarity" src/lessons/contrastive-learning/index.mdx src/lessons/word-embeddings/index.mdx src/lessons/one-hot-vs-dense/index.mdx`. Change any recap that lists it as part of the embeddings module, and any "next" pointer to it.
3. In `similarity-search/index.mdx`, add one opening sentence connecting it to the module: *"Retrieval-augmented generation starts with a question this lesson answers: given ten million vectors, how do you find the nearest ones without checking them all?"* (write it as `<LessonLink to="rag">Retrieval, in the RAG lesson</LessonLink>`, so it doesn't trip test 2).
4. In `rag/index.mdx`, where the vector index is first mentioned, link it to `similarity-search`.

### WP2.3 Tokens, not words: two short notes

1. In `one-hot-vs-dense/index.mdx`, at the end of the first section, add:
   ```mdx
   <Callout kind="note" title="Words, or pieces of words?">
   This module treats every word as one item, because it keeps the pictures simple. Real language models first chop text into sub-word pieces called **tokens** ("strawberry" may become two or three of them). How that chopping works is the subject of <LessonLink to="tokenization">Tokenization & Byte-Pair Encoding</LessonLink>, at the end of <ModuleRef id="transformers" />. Everything here carries over unchanged: a token gets its vector exactly the way a word does.
   </Callout>
   ```
   Before writing "two or three", check the claim against the `tokenization` lesson's own example (`st | r | a | w | berr | y | _` with all merges) and adjust the wording to match it.
2. In `attention/index.mdx`, after the first use of "word" as the unit, add one sentence: *"(We'll say "word" throughout; real models work on tokens, which <LessonLink to="tokenization">the tokenization lesson</LessonLink> explains.)"*

### WP2.4 Plain-language lesson summaries

Replace the `summary` of each lesson listed in **Appendix A** with the text given there, verbatim. Then switch test 5 in `curriculum.test.ts` from `it.todo` to `it`, and make it pass. If any other summary fails the test, rewrite it in the same style: name the problem and the surprise, not the method; at most 30 words; no symbols.

### WP2.5 Opening hooks: see WP6

---

## WP3 The Prologue

Create module `{ id: 'prologue', number: 0, title: 'Prologue', blurb: 'Before any maths: the question, the people, the hype, the winters, and the machine you are about to build.' }`, with one lesson:

`{ id: 'prologue', title: 'Can a Machine Learn?', summary: 'Seventy years of bold promises, dead ends and comebacks, and a working learning machine you can watch in your browser before we take it apart.', minutes: 12 }`

Renumber nothing else: Module 1 stays 1. Update test 4 to allow numbering from 0.

**Files:** `src/lessons/prologue/index.mdx`, `PrologueDemo.tsx`, `quiz.ts`, `prologue.test.ts`.

**Sections, in order** (every historical sentence carries `<Src href="…" />` pointing at a page you checked):

1. **`## A question older than the computer you're using`.** Turing's 1950 paper "Computing Machinery and Intelligence", and its closing section "Learning Machines": rather than program an adult mind, build a child's mind and educate it. Two or three sentences. Source: `WP('Computing_Machinery_and_Intelligence')`.
2. **`## One summer`.** The 1955 Dartmouth proposal expected "a significant advance" from a carefully chosen group of scientists working together "for a summer". Quote only words you've confirmed on the source page. Source: `WP('Dartmouth_workshop')`. Then: it took about seventy years, not one summer, and the people involved were among the best minds of their time. *This is the theme of the course.*
3. **`## Hype, then winter`.** The 1958 press report on Rosenblatt's perceptron ("the embryo of an electronic computer that [the Navy] expects will be able to walk, talk, see, write, reproduce itself and be conscious of its existence"; source `WP('Perceptron')`, checked word for word). The 1969 book *Perceptrons* and the first winter; the 1987 collapse and the second winter. Use the `ERAS` data in `src/history.ts` for the dates, so the page and the timeline agree.
4. **`## The long wait`.** Ideas that were right but early: backprop (1986), convolutional networks (1989), LSTMs (1997), all waiting for data and fast hardware. ImageNet and AlexNet (2012); ChatGPT (November 2022). One sentence each, sourced. End with Rich Sutton's 2019 essay "The Bitter Lesson" in one sentence, source `http://www.incompleteideas.net/IncIdeas/BitterLesson.html`.
5. **`## The machine you'll build`.** `<PrologueDemo />`: a small MLP (use `src/lib/mlp.ts`, seeded with `src/lib/rng.ts`) that learns the two-circles dataset live, with a decision-boundary heatmap (`src/components/heatmap.tsx`), a loss readout and Train/Pause/Reset. **Don't import from other lessons.** Under it, `<MachineMap />` with no highlight, and the sentence: *"Every box in this diagram is one lesson in <ModuleRef id="foundations" />. By the end of <ModuleRef id="backprop" short /> you'll be able to explain every number on this screen."* The demo's caption says what to try (press Train; then Reset and train again: the seed changes the boundary but not the outcome).
6. **`## How to read this course`.** Three short paragraphs:
   - (a) every idea here was once confusing to the people who invented it, and many were declared dead before they worked;
   - (b) labs come before history: you'll often rediscover something first, then find out who else did, and when; the "You are standing where they stood" notes mark those moments;
   - (c) the timeline (link `timelineHref()`) shows where each lesson sits in time.
7. `<Quiz>` with 3 `choice` questions, each with misconception feedback:
   - "What did the 1955 Dartmouth proposal expect?" (a big advance in a summer);
   - "Why did many good ideas wait decades?" (data and compute, not just cleverness);
   - "In this course, which comes first: the history or the experiment?"
8. `<Takeaways>`.

**Tests (`prologue.test.ts`):** the demo's training reaches a pinned accuracy within N steps for the fixed seed, and the loss decreases.

**Also:** in `src/pages/Home.tsx`, the hero's button text "Start with lesson 1" becomes "Start with the prologue". It is already computed from the first uncompleted lesson; only change the `done === 0` label.

---

## WP4 New lesson: the Perceptron

Insert into the `neuron` module after `logic-lab`:

`{ id: 'perceptron', title: 'The Perceptron: A Machine That Learns Its Own Weights', summary: 'In 1958 a machine with motor-driven knobs learned to tell shapes apart. The simple rule it used, why it always works when a line exists, and where it fails forever.', minutes: 30 }`

**Files:** `index.mdx`, `perceptron.ts` (rule, datasets, solver), `perceptron.test.ts`, `PerceptronStepper.tsx`, `MarkILab.tsx`, `quiz.ts`.

**Sections:**

1. **`## The problem: you set every weight by hand`.** In the logic lab you chose every weight yourself. Rosenblatt asked a machine to find them. The Mark I Perceptron: a camera of photocells, with weights held in motor-driven potentiometers. Check the details (count of photocells, 1958 or later) on `WP('Mark_I_Perceptron')`, and source every sentence.
2. **`## The rule in plain English`.** When the output is right, do nothing. When it should have fired and didn't, add the input to the weights. When it fired and shouldn't have, subtract it. Analogy: a tutor who only comments on mistakes.
3. **`<PerceptronStepper />`.** 2D points, the current line, and a "Next example" step button. Each update is written to a `MachineLog` ("example 3 misclassified: w ← w + x = …"). Datasets: OR, AND, a random separable set, and XOR. Caption: *"Step through OR until nothing changes. Then try XOR and keep stepping."*
4. **`<Predict>`** before XOR: *"Will the perceptron ever stop making mistakes on XOR?"* Options: yes, eventually / **no, never**. Explanation: no line separates XOR, and each fix to one corner breaks another, so the weights cycle forever.
5. **`## Writing it down`.** The `<Eq>`:
   ```
   words="\text{new weights} = \text{old weights} + \text{step} \times (\text{right answer} - \text{what it said}) \times \text{input}"
   tex="\cv{\mathbf{w}} \leftarrow \cv{\mathbf{w}} + \cg{\eta}\,(\cc{y} - \cp{\hat{y}})\,\ca{\mathbf{x}}"
   ```
   Legend for every symbol. Note that when the output is right, the bracket is 0 and nothing changes.
6. **`<Deeper title="Why it always works when a line exists">`.** The convergence theorem (Novikoff, 1962; check on `WP('Perceptron')`): if a separating line with margin γ exists and every point lies within radius R, the perceptron makes at most (R/γ)² mistakes. `perceptron.test.ts` checks the bound holds on every dataset in the lesson.
7. **`## Lab: Train the Mark I`** (`MarkILab.tsx`, `LabShell`). Primitives: the order of the examples (drag to reorder), the starting weights (three sliders including bias), and the step size. Briefs:
   1. learn OR;
   2. learn AND with the **fewest mistakes**, optimum computed by exhaustive search over all 24 orderings × starting weights on a stated grid × step sizes {0.1, 0.5, 1};
   3. *thin margin:* a separable set with one point very close to the line, so the run takes many mistakes; remove exactly one point to make it learn in ≤ N mistakes, with N computed;
   4. XOR: try for 50 updates, then the brief explains why it's impossible and unlocks the AnswerReveal ("no line exists; the hand-wired XOR network from the logic lab works, but this rule can't train its hidden neurons, because nothing tells a hidden neuron what it *should* have output").
8. **`## The wall`.** That last point is the **credit assignment problem**. Add a `<Term>`, and a new glossary entry `credit-assignment` if it doesn't exist. Then: it took until 1986 for the field to widely adopt a way through (backprop; <ModuleRef id="backprop" />). Link the existing `logic` HistoryCard's timeline entry.
9. `<Insight title="You are standing where they stood">`: *"In the last few minutes you found the perceptron's power and its limit. The field went from Rosenblatt's machine (1958) to the limit being proved in a book (1969) to a widely adopted way past it (1986)."* Check all three years against `src/history.ts`.
10. New history entry `mark-i` (lab "Train the Mark I"): who (Rosenblatt, Cornell Aeronautical Laboratory), mood (the 1958 press quote), before (hand-set threshold units, 1943), this (a learning rule), after (*Perceptrons*, 1969). Every claim sourced.
11. `<Quiz>`: 4 questions (2 choice, 1 numeric: "one update with these numbers gives which weights?", parameterised; and 1 on "why can't it train hidden neurons?"), then `<Takeaways>`, then a `Bridge`-style closing paragraph pointing to `forward-propagation`.

---

## WP5 Era labels and the timeline

1. Add `era?: string` to `ModuleMeta` in `curriculum.ts`. Set these values. They are approximate spans of each module's central ideas. Check each against the lessons' history cards, and if one disagrees, use the card's years.

| Module id | era |
|---|---|
| prologue | 1950 to today |
| foundations | 1670s to 1951 |
| neuron | 1943 to 1969 |
| architecture | 1960s to 1989 |
| optimization | 1847 to 2016 |
| backprop | 1970 to 2015 |
| training | 1989 to today |
| generalization | 1990s to 2019 |
| classical | 1805 to 2001 |
| vision | 1959 to 2016 |
| embeddings | 2003 to 2021 |
| sequences | 1986 to 2015 |
| transformers | 2014 to 2021 |
| llms | 1948, then 2018 to 2024 |
| rl | 1957 to 2017 |
| finetuning | 2017 to 2023 |
| inference | 2019 to 2023 |
| generative | 2013 to 2022 |
| systems | 2020 to today |
| epilogue | open questions |

2. Show it:
   - `Home.tsx` module cards: `Module {m.number} · {m.era}` in the `.n` line;
   - `LessonPage.tsx` kicker: `Module N · Title · <era> · Lesson k`, where the era is a link to `timelineHref(id)` using the first `HISTORY` entry whose `lessonId` is in this module (or plain `timelineHref()` if there is none);
   - nothing in the sidebar (no room).
3. Add `MILESTONES` to `src/history.ts`, each sourced, for the timeline:
   - 1950 Turing's "Computing Machinery and Intelligence";
   - 1958 the perceptron press reports;
   - 1969 *Perceptrons*;
   - 1973 the Lighthill report;
   - 1986 the backprop paper in *Nature*;
   - 1997 LSTM;
   - 2009 ImageNet;
   - 2017 the transformer;
   - 30 November 2022 ChatGPT;
   - 2024 Nobel Prizes (Physics: Hopfield and Hinton).

   Check each one on its page (Appendix B).

---

## WP6 History in the main text

### WP6.1 Opening hooks

For each lesson below, add a 2–4 sentence paragraph **at the end of its first section** (after the problem is posed, before the second `## `). Each claim gets `<Src>`. The hook poses the historical problem, so it gives the learner a reason to read on rather than trivia. Draft wording is given. **Check every fact on the source, and cut whatever you can't confirm.**

| Lesson | Hook (draft) | Source(s) |
|---|---|---|
| neuron | "The first artificial neuron was written down in 1943 by Warren McCulloch, a neurophysiologist, and Walter Pitts, a young logician. It was a claim about brains, but it became the blueprint for every network since." | `WP('Walter_Pitts')`, `WP('Warren_Sturgis_McCulloch')` |
| gradient-descent | "The rule you're about to meet was proposed by Augustin-Louis Cauchy in 1847 to make the hand calculation of astronomical orbits less laborious. Every model in this course is trained by a descendant of it." | `WP('Gradient_descent')`; confirm the astronomy context, otherwise drop that clause |
| loss-functions | "Squared error is older than computers. Legendre published least squares in 1805; Gauss said he'd been using it since 1795, and had used it to relocate the asteroid Ceres. The priority dispute was famously bitter." | `WP('Least_squares')` |
| activation-functions | "The kink that rescues deep networks, ReLU, is about as simple as a function can be, and it took decades to catch on: it was used in the late 1960s and only became the standard around 2010–2011." | `WP('Rectifier_(neural_networks)')` |
| backpropagation | "Backpropagation was discovered several times: Linnainmaa's reverse-mode differentiation (1970), Werbos's application to networks (1974 thesis; 1982 paper), and the 1986 *Nature* paper by Rumelhart, Hinton and Williams that made it famous. A great idea sat in plain sight for sixteen years." | `WP('Backpropagation')`, `WP('Seppo_Linnainmaa')`, `WP('Paul_Werbos')` |
| vanishing-gradients | "In 1991 a student named Sepp Hochreiter worked out, in a diploma thesis written in German, why deep networks wouldn't train. Practical fixes took about two more decades." | `WP('Vanishing_gradient_problem')`, `WP('Sepp_Hochreiter')` |
| optimizers | "Adam (2014) became the default optimiser almost overnight. In 2018 a paper showed that its convergence proof had an error, and that paper won a best-paper award, while Adam itself kept working in practice." | `https://arxiv.org/abs/1904.09237` (and confirm the ICLR 2018 best-paper award on `https://iclr.cc` or the paper's page; if you can't confirm the award, drop that clause) |
| overfitting | "The textbook story says bigger models overfit more. In 2019 Belkin and colleagues showed test error falling *again* as models grow past the point of fitting the data exactly, 'double descent', and the textbooks are still catching up." | `https://arxiv.org/abs/1812.11118` |
| cnns | "Convolutional networks read handwritten digits at Bell Labs in 1989. They became famous in 2012, when AlexNet, trained on two gaming graphics cards, won the ImageNet challenge by a wide margin." | `WP('LeNet')`, `WP('AlexNet')` |
| transformer-block | "Batch normalisation (2015) was explained as fixing 'internal covariate shift'. In 2018 Santurkar and colleagues argued that's not why it works. It's common in this field for a method to work before anyone agrees on why." | `https://arxiv.org/abs/1805.11604` |
| mdps | "Richard Bellman, who wrote down this lesson's central equation, later said he chose the name 'dynamic programming' partly to keep his mathematical research from sounding like mathematics to a funder who disliked the word 'research'." | `WP('Dynamic_programming')`; quote only what the page quotes |
| word-embeddings | "Word2vec (2013) made 'king − man + woman ≈ queen' famous, though the idea that a word's meaning comes from its company is far older." | `WP('Word2vec')`, `WP('Distributional_semantics')` |

### WP6.2 "You are standing where they stood"

After each lab listed below, directly before its `<HistoryCard>` (or before `ForReal` if there is no card), add:

```mdx
<Insight title="You are standing where they stood">
…one or two sentences…
</Insight>
```

| Lab (lesson) | Text (check the years against `src/history.ts`) |
|---|---|
| logic-lab | You just wired XOR out of hidden neurons. In 1969 the limit of a single neuron was proved in a book; a widely adopted way to *train* hidden neurons arrived in 1986. |
| network-from-scratch | You just wrote a backward pass and checked it by measurement. People found that algorithm at least three separate times between 1970 and 1986. |
| vanishing-gradients | You just rescued a 32-layer network. The problem was diagnosed in 1991, and the residual connections you used are from 2015. |
| attention (Attention puzzles) | You just steered attention by hand. In 2014 this was a new idea for translation; three years later it was the whole architecture. |
| chain-rule (new lab, WP1.8) | Your backward walk, reusing the running product, is reverse-mode differentiation, published by Linnainmaa in 1970 and still the core of every framework's `backward()`. |
| information (new lab, WP1.8) | You just found an optimal code. Huffman found it in 1952 as a student, after his professor's method had fallen short of it. |
| perceptron (WP4) | (Given in WP4 step 9.) |

---

## WP7 Era interludes

Four short lessons (8 minutes each). Each one has:

- a `<EraStrip from={…} to={…} />`;
- 3–5 short sections telling the story;
- every claim sourced with `<Src>`;
- a 2-question `<Quiz>`;
- `<Takeaways>`.

No lab is needed.

**First, build `src/components/EraStrip.tsx`:** a horizontal strip (wrapping on phones) showing the `ERAS`, `MILESTONES` and `HISTORY` entries whose years fall in [from, to], read from `src/history.ts`, in date order. Each item links to `timelineHref(id)`. Register it in `mdxComponents.tsx`. Write a test that the strip's item selection matches a hand-computed list for one range.

| Lesson id | Where | Title | Sections (claims to check) |
|---|---|---|---|
| `interlude-first-winter` | end of `neuron` module | "Interlude: The First Winter (1969–1980)" | (1) The promise: the 1958 headlines. (2) The book: *Perceptrons*, and what it actually proved (a single layer can't do XOR-like tasks); the existing `dispute` in `history.ts` says the "book killed the field" story is simplified, so repeat that nuance. (3) The report: Lighthill (1973) and the funding cuts (`WP('Lighthill_report')`, `WP('AI_winter')`). (4) The machines: mainframes, punched cards, memory in kilobytes; in that world symbolic AI looked like the practical route (`WP('History_of_artificial_intelligence')`). (5) Minsky's turn: he built SNARC, an early neural-network machine, in 1951, and later co-wrote the field's best-known critique (`WP('Stochastic_neural_analog_reinforcement_calculator')`; check the page title). |
| `interlude-second-winter` | first lesson of `classical` | "Interlude: When Neural Networks Were Unfashionable (1987–2006)" | (1) The expert-systems boom and the 1987 collapse of the Lisp-machine market (`ERAS` already sources this). (2) Backprop worked, but on 1990s data and CPUs, methods with tidy guarantees (support vector machines, later random forests) usually won (`WP('Support_vector_machine')`, `WP('Random_forest')`). (3) The quote already in `history.ts`: researchers "avoided the term artificial intelligence". (4) A small group kept going: the CIFAR programme (2004) and the 2006 deep-belief-net papers that helped bring back the name "deep learning" (`WP('Geoffrey_Hinton')`; drop anything not on the page). (5) Why this module exists: these methods are still the right tool for many tabular problems. |
| `interlude-big-bang` | first lesson of `vision` | "Interlude: Data, Graphics Cards and 2012" | (1) The data: ImageNet (2009), labelled with crowd workers (`WP('ImageNet')`). (2) The hardware: graphics cards built for games, and CUDA (2007) making them programmable for general arithmetic (`WP('CUDA')`). (3) The result: AlexNet's 2012 win and its error rate against the runner-up, quoting the page's numbers (`WP('AlexNet')`). (4) The lesson: most of the ideas were from 1989; what changed was the scale of data and arithmetic. Link `cnns`. |
| `interlude-chatgpt` | first lesson of `llms` | "Interlude: The Week Everyone Noticed (2022)" | (1) GPT (2018) to GPT-3 (2020): scale as a strategy (`WP('GPT-3')`). (2) ChatGPT's release on 30 November 2022 and its adoption numbers, quoting the page's figures (`WP('ChatGPT')`). (3) What was new and what wasn't: the transformer was from 2017, next-token prediction goes back to Shannon (1951), and the new part was post-training on human preferences, which this course covers in <ModuleRef id="finetuning" />. (4) 2024: Nobel Prizes for Hopfield and Hinton (Physics) and Hassabis and Jumper, with Baker (Chemistry). Source `https://www.nobelprize.org/prizes/physics/2024/summary/` and `https://www.nobelprize.org/prizes/chemistry/2024/summary/`. |

---

## WP8 New lessons: SVMs, GANs, Epilogue

Each one is a full lesson under the HANDOFF "Checklist: adding a lesson" and "Definition of done". These are the largest steps; do them last.

### WP8.1 `svm`: "Support Vector Machines: The Widest Street" (`classical`, after `linear-regression`)

- **Summary:** "Many lines separate the data; which one should you trust? The widest street, the few points that decide it, and a trick for drawing curves with straight lines."
- **Arc:**
  - many separating lines → `<Predict>` "which line generalises best?" → max margin;
  - the support vectors (drag a non-support point: nothing changes; drag a support point: the street moves);
  - the soft margin C (a slider; overlapping data);
  - the kernel trick, shown by lifting rings into 3D with an x² + y² feature. The learner did feature engineering in `linear-regression`'s lab, so link it.
- **The Eq** for the margin, 2/‖w‖, with a legend.
- **Lab "Widen the Street":** briefs:
  1. place a separating line;
  2. reach ≥ 95% of the maximum margin;
  3. pick the feature from a parts bin that makes the rings separable;
  4. choose C on noisy data to maximise validation accuracy, with the optimum computed over a stated grid.
- **Solver:** an exact hard-margin solution for small 2D sets, by enumerating candidate support-vector pairs and triples and checking constraints (exhaustive), cross-checked against a sub-gradient hinge-loss solver in a test.
- **History:** Vapnik and colleagues; the kernel method (1992); the soft margin (Cortes & Vapnik, 1995); dominance in the 1990s. Source `WP('Support_vector_machine')` and check each date.
- **`<Insight>`** on fashion: the method that out-competed neural networks for a decade.

### WP8.2 `gans`: "GANs: A Forger and a Detective" (`generative`, after `autoencoders`)

- **Summary:** "Two networks in a duel: one forges data, one tries to catch it. Why the duel can produce sharp samples, why it so often goes wrong, and how to keep it balanced."
- **Arc:**
  - the forger/detective analogy;
  - a live 1D GAN: real data from a mixture of 3 Gaussians; generator and discriminator are small MLPs from `src/lib/mlp.ts` (or `autodiff`), seeded;
  - histograms of real against generated data, updating live;
  - `<Predict>` "if the detective gets too good, what happens to the forger's learning signal?" (it vanishes);
  - mode collapse demonstrated on a pinned seed and settings;
  - the minimax `<Eq>` with a legend.
- **Lab "Keep the Duel Balanced":** primitives: learning-rate ratio, discriminator steps per generator step, and noise dimension (1 or 2). Briefs:
  1. cover at least 2 of the 3 modes by step N;
  2. cover all 3;
  3. do it in the fewest total updates.

  The optimum is "best found by searching <grid>", averaged over 5 fixed seeds (house rule for fair comparison).
- **History:** Goodfellow et al. (2014, `https://arxiv.org/abs/1406.2661`). Use the "invented in one night after an argument" story only if `WP('Ian_Goodfellow')` or another reliable page supports it. "After": diffusion models beat GANs on image quality (Dhariwal & Nichol, 2021, `https://arxiv.org/abs/2105.05233`). Link `diffusion`.
- **Tests:** determinism with the fixed seed; the collapse configuration covers fewer modes than the balanced one.

### WP8.3 `open-questions`: "What Nobody Understands Yet" (new module `epilogue`, last)

- **Summary:** "The field's best minds are still puzzled. Why big networks generalise, why abilities seem to appear suddenly, and what is really stored inside a model."
- **Sections:**
  1. **Why does it generalise at all?** Double descent (link `overfitting`). First grep `overfitting` and `regularization` for any existing double-descent content, and link rather than duplicate. Widget: a live random-features regression with the minimum-norm solution (`src/lib/linalg.ts`), plotting test error against the number of features, showing the second descent. Pin the curve's peak location in a test.
  2. **Grokking.** Generalisation long after memorising (`https://arxiv.org/abs/2201.02177`). Describe it from the paper's abstract only.
  3. **What's inside?** Superposition; link the high-dimensions lab and the 2023 dictionary-learning work already cited in `history.ts`.
  4. **Do abilities really "emerge"?** The debate (`https://arxiv.org/abs/2304.15004`). Present both sides honestly.
  5. **Why do scaling laws hold?** Link `scaling-laws`; say plainly that there is no agreed explanation.
  6. **Closing.** It ends where the prologue began: the best minds are still confused, and the learner now knows enough to be confused about the same things. Link the prologue.
- `<Quiz>` (3 questions) and `<Takeaways>`.

---

## WP9 Final documentation

1. In `docs/ROADMAP.md`, rewrite "Current state" to describe the new structure. Add a line per WP to the status table. Record any dropped claims and any "Open questions for the owner".
2. In `HANDOFF.md`:
   - tick every line in the progress checklist;
   - add two gotchas: "Never type module numbers or 'next lesson' in prose; use `ModuleRef`/`LessonLink`, and `src/curriculum.test.ts` enforces it", and "Every Module 1 lesson needs a `Bridge` to a later module";
   - update "Checklist: adding a lesson" with: summary ≤ 30 words with no symbols; historical sentences carry `<Src>`.
3. In `docs/VISION.md`, under "Owner's clarifications", add item 8: the owner asked for the maths to feel like 3Blue1Brown, with labs of its own and explicit bridges to the AI that uses it; and for history and "the struggle" to be part of the main text.
4. In `docs/CURRICULUM_AUDIT.md`, add a line at the top: "Implemented via `docs/CURRICULUM_PLAN.md`; see the ROADMAP for status."

---

## Appendix A: replacement lesson summaries (verbatim)

Paste these into `src/curriculum.ts`, keeping the quote style used there (escape apostrophes as `\'` if the file uses single quotes, or keep the existing curly ’).

| id | New summary |
|---|---|
| linear-regression | The oldest learning machine: fit a line by least squares, see why it fails at yes-or-no questions, fix it with logistic regression — and a feature-engineering lab. |
| knn | Classify by asking the neighbours: no training at all, one dial that trades memorising for smoothing, and the two ways it quietly breaks — and Lab: Rescue the kNN Classifier. |
| decision-trees | Twenty questions as a model: how a tree picks its questions, why one tree overfits and a forest of them doesn’t — and Lab: Build the Tabular Ensemble. |
| clustering-pca | Finding structure with no labels at all: grouping points into clusters, squashing many features into the few that matter — and Lab: Compress & Segment. |
| images-as-tensors | What a photo looks like to a network: grids of numbers, colour channels, and why shrinking an image the naive way invents patterns — and Lab: Build the Vision Preprocessor. |
| convolution | Slide one small pattern-detector across a whole image: edge finders, blurs, and the arithmetic of output sizes — and Lab: Design the Kernel. |
| cnns | Stack small detectors into big ones: how each layer sees a wider patch, and why a tiny CNN beats a much bigger plain network — and Lab: Architect the CNN. |
| object-detection | Not just what is in the picture but where: drawing boxes, scoring overlaps, removing duplicates in a crowd — and Lab: Tune the Object Detector. |
| one-hot-vs-dense | Why giving every word its own switch makes all words strangers, and how short dense vectors let similar words share what they learn — and Lab: Choose & Size the Representation. |
| word-embeddings | Learn what words mean from nothing but the company they keep, and watch analogies like king to queen emerge — and Lab: Train & Debug Word2Vec. |
| similarity-search | Finding the nearest of ten million vectors without checking them all: buckets, random hashes and compression, and what each shortcut costs — and Lab: Tune the Vector Index. |
| contrastive-learning | Learn representations without labels by pulling matching pairs together and pushing everything else apart, and why skipping the push collapses everything to one point — and Lab: Tune the Contrastive Objective. |
| rnns | A network with memory: reuse the same weights at every step, unroll it to train it, and watch long-range signals fade or explode — and Lab: Tame BPTT. |
| lstms-grus | Give memory a protected highway and gates that decide what to keep, then race plain, gated and LSTM cells fairly — and Lab: Wire the Gated Cell. |
| seq2seq-attention | Squeeze a whole sentence into one vector and translation falls apart; let the decoder look back at every word and it recovers — and Lab: Break the Bottleneck. |
| scaling-laws | How much better does a model get with more size, data and compute? The surprisingly regular answer, the famous correction, and a compute-budget lab. |
| lora | Fine-tune a giant model by training a tiny add-on: why the change a model needs is far smaller than the model — and a rank-budget lab. |
| rlhf-dpo | Teaching a model what people prefer: why optimising a learned score backfires, how a leash keeps it honest, a simpler method without the scorer — and an alignment lab. |
| quantization | How many bits can a number lose before the model notices? Flip bits in real number formats, watch one outlier ruin compression — and a mixed-precision lab. |
| kv-cache | Why generating text one token at a time leaves a GPU mostly idle, and how caching past work fixes the waste — and Lab: VRAM Tetris. |
| mdps | Decisions with consequences: how an agent weighs now against later, why uncertainty makes it cautious, and the equation that plans ahead — and Lab: Wire the Policy. |
| q-learning | Learning by trial and error: the explore-or-exploit dilemma, a value table that fills itself in, and a cliff that separates bold from careful — and Lab: Tune the Q-Learner. |
| policy-gradients | Skip the value table and adjust behaviour directly: why randomness helps, why a baseline costs nothing, how clipping keeps steps safe — and Lab: Stabilise the Policy Gradient. |
| reward-hacking | Write a reward, and an optimiser finds the loophole: the boat that circles forever, why bonuses backfire, and a lab where an exact optimiser attacks every reward you write. |
| autoencoders | Why averaging two faces makes a ghost, how a bottleneck finds the hidden knobs behind data, and how a VAE learns to make new faces — and Lab: The Face Printer. |
| diffusion | Dissolve data into noise step by step, then learn to run the film backwards: how a denoiser turns static into samples — and Lab: Spend the Step Budget. |
| guidance | Ask a generator for a particular class, turn the request up, and watch quality rise while variety drains away — and Lab: Tune the Guidance. |
| rag | Why models need to look things up, keyword versus meaning-based search measured on real questions, and how chunk size trades accuracy for cost — and Lab: Tune the Retriever. |
| tool-calling | How a text predictor acts on the world through tool calls, why each step’s small failure rate compounds, and what makes agents safe — and Lab: Harden the Agent Pipeline. |
| evaluation | Is model B really better, or did you just get lucky? Fair comparisons, the trap of tuning on your test, judges that play favourites — and Lab: Size Your Eval Suite. |

Before pasting, check every lab name in these summaries against the lab's actual heading in its lesson (the other agent may have renamed some). Use the lesson's current name.

---

## Appendix B: sources to verify

Fetch each page and confirm the exact claim before using it. `WP(x)` means `https://en.wikipedia.org/wiki/x`, as in `src/history.ts`.

| Claim | Source |
|---|---|
| Turing 1950, "Learning Machines" section | WP('Computing_Machinery_and_Intelligence') |
| Dartmouth 1955 proposal wording | WP('Dartmouth_workshop') |
| 1958 press quote on the perceptron; IBM 704 simulation; Novikoff 1962 | WP('Perceptron') |
| Mark I hardware (photocells, motor-driven potentiometers) | WP('Mark_I_Perceptron') |
| SNARC 1951 (Minsky) | WP('Stochastic_neural_analog_reinforcement_calculator') (check the exact title) |
| Lighthill report 1973 | WP('Lighthill_report') |
| Least squares: Legendre 1805, Gauss, Ceres | WP('Least_squares') |
| Cauchy 1847 | WP('Gradient_descent') |
| ReLU history | WP('Rectifier_(neural_networks)') |
| Backprop discoveries | WP('Backpropagation'), WP('Seppo_Linnainmaa'), WP('Paul_Werbos') |
| Hochreiter 1991 | WP('Vanishing_gradient_problem'), WP('Sepp_Hochreiter') |
| Adam's proof flaw | https://arxiv.org/abs/1904.09237 |
| BatchNorm explanation disputed | https://arxiv.org/abs/1805.11604 |
| Double descent | https://arxiv.org/abs/1812.11118 |
| Grokking | https://arxiv.org/abs/2201.02177 |
| Emergence debate | https://arxiv.org/abs/2304.15004 |
| LeNet / AlexNet | WP('LeNet'), WP('AlexNet') |
| ImageNet, CUDA | WP('ImageNet'), WP('CUDA') |
| Bellman and the name "dynamic programming" | WP('Dynamic_programming') |
| Word2vec; distributional semantics | WP('Word2vec'), WP('Distributional_semantics') |
| SVM dates | WP('Support_vector_machine') |
| Random forests | WP('Random_forest') |
| GANs | https://arxiv.org/abs/1406.2661, WP('Ian_Goodfellow') |
| Diffusion beats GANs | https://arxiv.org/abs/2105.05233 |
| GPT-3, ChatGPT | WP('GPT-3'), WP('ChatGPT') |
| 2024 Nobel Prizes | https://www.nobelprize.org/prizes/physics/2024/summary/, https://www.nobelprize.org/prizes/chemistry/2024/summary/ |
| The Bitter Lesson | http://www.incompleteideas.net/IncIdeas/BitterLesson.html |
| Iris data; Fisher's discriminant | WP('Iris_flower_data_set'), WP('Linear_discriminant_analysis') |
| Vector space model | WP('Vector_space_model') |
| Strassen | WP('Strassen_algorithm') |
| Newton–Leibniz dispute | WP('Leibniz–Newton_calculus_controversy') |
| Chain rule history | WP('Chain_rule') |
| Boltzmann; softmax naming | WP('Boltzmann_distribution'), WP('Softmax_function') |
| Huffman 1952; KL 1951 | WP('Huffman_coding'), WP('Kullback–Leibler_divergence') |
| McCulloch and Pitts | WP('Warren_Sturgis_McCulloch'), WP('Walter_Pitts') |
| Hinton, CIFAR, 2006 | WP('Geoffrey_Hinton') |
