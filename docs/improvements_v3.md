# Plan — V3 Intuition-First Upgrades (`docs/improvements_v3.md`)

**Status:** Drafted for owner approval / ready to hand to an implementer agent at **V3-1**.

---

## 1. Goal: Make Intuition King

Neural Lab already covers 21 modules (83 lessons, 66 sourced history cards) with real browser computation from first principles. After completing the Module 5 Backpropagation Bridge (`docs/BACKPROP_BRIDGE_PLAN.md`), the remaining opportunities are **not** more topics — they are **seven surgical intuition multipliers** plus a tighter Module 5 bridge so a learner can grasp deep learning viscerally without ever feeling gated by heavy maths:

1. **Intuition-first entry ramp & optional "Words-first" `<Eq>` toggle** — remove the psychological wall of 9 maths lessons before building a neuron, and let math-shy readers hide symbolic TeX behind a 1-click peek while keeping the plain-English word equation and legend visible.
2. **Rubber-Sheet Space Folding (`SpaceFoldWidget`)** — watch a real 2D coordinate grid warp, crease, and fold layer-by-layer through a trained `2 → 2 → 2 → 1` network on Concentric Rings and Two Moons until the final layer slices through with a single straight knife cut.
3. **"It's All Still `W·x + b`" Architecture Rosetta Stone (`RosettaMatrix`)** — morph one `6 × 6` linear weight matrix live into a dense MLP, a 1D Toeplitz convolution (weight sharing across space), an unrolled RNN (weight sharing across time), input-dependent Self-Attention (`W(x)`), and low-rank LoRA (`W₀ + A·B`).
4. **Why Big Networks Don't Get Trapped in Bad Local Minima (`SaddleExplorer`)** — dissolve the 1D roller-coaster misconception by showing analytically and interactively how flat points turn from traps (`50%` in 1D) into escape-ramp saddle points (`(1/2)^d` trap probability in $d$ dimensions), plus a live 2D saddle ridge trajectory.
5. **Tighter Module 5 Arc (`backpropagation` ↔ `vanishing-gradients`)** — link `vanishing-gradients`'s 16-layer `SignalTelephone` and `RescueDeepNetLab` directly back to `TinyMlpBackprop`'s *Saturated tanh* preset (`682×` single-layer attenuation) and `StepMicroscope`'s crease-line motion, showing how LayerNorm keeps pre-activations off the flat shelf and residual skips wire a `+1` highway around every valve.
6. **"Paint Your Own Dataset" Brush in the Training Lab** — let learners click/drag custom blue & orange dots (or drop a stubborn outlier) directly onto `TrainingLab`'s 2D canvas and watch a live network bend around *their* drawing.
7. **Sentence Microscope for a 4-Token Transformer (`SentenceMicroscope`)** — the LLM counterpart to `StepMicroscope`: click any word in `"The cat sat on"` and trace its single residual-stream vector step-by-step (`embedding → +position → +attention mix → +MLP → unembedding logits & p − y`), with toggles to disable attention, residual skips, or positions live.

---

## 2. Placement, Length & Preservation Rules

All target lessons have substantial word-count headroom under the ~3,300-word budget:

| Lesson / Surface | Current words | V3 Addition | Est. final words |
|---|---:|---|---:|
| `src/pages/Home.tsx` + `src/components/Eq.tsx` | — | Intuition-first entry card + optional `Words first` `<Eq>` toggle | — |
| `src/lessons/forward-propagation/index.mdx` | 2,550 | `SpaceFoldWidget` + `<Predict>` (~350 words) | ~2,900 |
| `src/lessons/depth-and-width/index.mdx` | 2,100 | `RosettaMatrix` + `<Predict>` + forward `<LessonLink>`s (~420 words) | ~2,520 |
| `src/lessons/gradient-descent/index.mdx` | 1,960 | `SaddleExplorer` + `<Predict>` (~380 words) | ~2,340 |
| `src/lessons/vanishing-gradients/index.mdx` | 1,961 | Direct Module 5 callback to `TinyMlpBackprop` (`682×` valve) & `StepMicroscope` (~220 words) | ~2,180 |
| `src/lessons/training-lab/index.mdx` | 2,113 | `Custom (paint)` brush mode in `TrainingLab.tsx` (~120 words) | ~2,230 |
| `src/lessons/transformer-block/index.mdx` | 1,724 | `SentenceMicroscope` + `<Predict>` (~420 words) | ~2,140 |

**Strict preservation rules:**
- Do **not** alter module ordering, lesson IDs, or any existing pinned test numbers (`presets.test.ts`, `lab.test.ts`, `vanish.test.ts`, `block.test.ts`, `pieces.test.ts`, `islands.test.ts`, `curriculum.test.ts`, `history.test.ts`).
- Keep default reading mode on `<Eq>` as **`full`** (both word equation and symbolic TeX visible) so existing pages look identical unless the learner opts into `Words first`.
- Every widget computes real numbers (no scripted plots or faked trajectories), colours exclusively via CSS variables (no hard-coded hex), respects `prefers-reduced-motion`, and is inspected visually in **light, dark, matrix, and 390px phone width**.
- Every analytic gradient is checked against central finite differences (`< 1e-8`); every number quoted in prose is pinned by a Vitest test.
- Follow `AGENTS.md` / `HANDOFF.md` / `docs/STYLE_GUIDE.md` (British spelling, concept before equation, `<Eq>` with colour-matched legend `\ca \cb \cc \cg \cp \cv` never `\cr`, invisible authorship watermark with 0 matches in `dist/`, commit & push directly to `claude/cool-feynman-4hci4d`, never open PRs or extra branches).

---

## 3. Detailed Specification (V3-1 – V3-8)

### V3-1 — Intuition-First Entry Ramp & Optional "Words-First" `<Eq>` Mode

1. **Home page intuition-first starter (`src/pages/Home.tsx`):**
   - Right beside the primary `"Start with the prologue"` / `"Continue"` CTA in `.hero` (only when `done < 10`, or as a subtle secondary link always available), add a secondary button/link:
     - `"Jump straight to neurons →"` linking to `lessonHref('neuron')`, plus a one-line helper note right under the Module 1 card blurb (or in the hero): *"Prefer to build a neural network first? Start at Module 2 and dip into the Maths Toolkit whenever a symbol looks unfamiliar."*
   - Keep `src/curriculum.ts` and `src/dependencies.ts` completely unchanged.
2. **Global optional `<Eq>` reading mode (`src/mathMode.ts`, `src/components/Eq.tsx`, `src/App.tsx`):**
   - Create `src/mathMode.ts` storing `'full' | 'words-first'` in `localStorage` under `'nl-math-mode'` (default `'full'`), matching `src/theme.ts`'s clean subscription pattern (`useMathMode()`, `setMathMode()`).
   - Update [`src/components/Eq.tsx`](../src/components/Eq.tsx):
     - When `mode === 'full'` (or when `words` is omitted): renders identically to today.
     - When `mode === 'words-first'` and `words` is present: shows `.eq-words` + `.eq-legend` visibly, and collapses `.eq-tex` behind a compact inline button (`"Show symbols"` / `"Hide symbols"`) so the learner can peek at any individual equation without leaving Words-first mode!
   - Add a compact, unobtrusive toggle button in the top bar (next to `Glossary` / `ThemeSwitch`, hidden cleanly or abbreviated on 390px phone width so the header never overflows!): `Σ Symbols` / `Words first`.
   - **Tests (`src/components/eq.test.ts`):** verify SSR/markup in `'full'` mode shows `.eq-tex`, in `'words-first'` mode hides `.eq-tex` by default when `words` is supplied, and always shows `.eq-tex` when `words` is omitted.

---

### V3-2 — Rubber-Sheet Space Folding (`SpaceFoldWidget` in `src/lessons/forward-propagation/`)

**Files:** `src/lessons/forward-propagation/spaceFold.ts`, `spaceFold.test.ts`, `SpaceFoldWidget.tsx`, update `index.mdx` and `quiz.ts`.

1. **Engine (`spaceFold.ts`):**
   - Architecture: a real `2 → 2 → 2 → 1` `tanh` MLP (15 parameters: `W1 (2×2), b1 (2), W2 (2×2), b2 (2), W3 (2×1), b3 (1)`).
   - Two crisp, deterministic 2D datasets (~48 points each, seeded):
     - **`rings`** (*Concentric rings*): inner radius `r ≈ 0.45` (class 1) vs outer ring `r ≈ 1.15` (class 0).
     - **`moons`** (*Two interlocking moons*): two curved crescents that no straight line can separate.
   - Train deterministically with seeded Adam or full-batch GD (`trainSpaceFold(datasetId)` — verify it runs in `< 40 ms` in Node/browser, or cache the trained weights once per dataset and verify them in `spaceFold.test.ts`!).
     - **Measure first:** Confirm both `rings` and `moons` achieve **≥ 96% accuracy** with `2 → 2 → 2 → 1` `tanh` (because all intermediate layers have width **2**, every intermediate activation `(h1_1, h1_2)` and `(h2_1, h2_2)` lives in **exact 2D** with zero projection!).
     - **Honest check:** Check whether `2 → 2 → 2 → 1` solves `rings` cleanly — remember a `2 → 2 → 1` network can only bend 2 creases (hard for a closed circle), whereas `2 → 2 → 2 → 1` has two chained 2D folds (measure on 10 seeds first!). If `rings` needs 3 hidden layers of width 2 (`2 → 2 → 2 → 2 → 1`, 21 params) to wrap a closed ring cleanly at ≥ 96% accuracy, use `2 → 2 → 2 → 2 → 1` and record the exact measured accuracy!
   - **Mesh deformation (`deformGrid(net, t)`):**
     - Generate a `13 × 13` regular coordinate mesh over `[-1.5, 1.5]²` (horizontal gridlines coloured via `--c-a` tint, vertical gridlines via `--c-b` tint, or subtle `--border-strong` + highlighted axes).
     - At `t = 0`: raw input `(x₁, x₂)`.
     - At `t = 1`: after Layer 1 `h⁽¹⁾ = tanh(W₁x + b₁)`.
     - At `t = 2`: after Layer 2 `h⁽²⁾ = tanh(W₂h⁽¹⁾ + b₂)` (and `t = 3` if 3 hidden layers).
     - For fractional `t`, linearly interpolate every mesh vertex and every data point between consecutive layers so the sheet **continuously morphs** as you scrub or press Play!
   - **Output knife cut (`outputBoundaryLine(net)`):**
     - Compute the exact straight line `w_out[0]·u + w_out[1]·v + b_out = 0` in the final hidden plane (`t = L`) — and draw its **inverse image** (the curved decision boundary `p(x₁, x₂) = 0.5`) back on the input plane (`t = 0`)!
   - **Separability metric at each layer:** compute linear accuracy of the best linear probe (or the trained output line at `t = L`) vs raw input linear baseline (`~50%` on rings, `~85%` on moons → `≥ 96%` after folding!).
   - **Tests (`spaceFold.test.ts`):**
     - Analytic backward pass checked against central finite differences (`< 1e-8`).
     - Pin raw-input linear accuracy vs final-layer accuracy for both `rings` and `moons`.
     - Verify `deformGrid` continuity at `t = 0, 0.5, 1, 1.5, 2`.
2. **Widget (`SpaceFoldWidget.tsx`):**
   - Controls: Dataset switcher (*Concentric rings* | *Two moons*), **Play / Pause** + **Layer scrubber** (`Input → Layer 1 → Layer 2 [→ Layer 3]`) using `useReplay` + `ReplayControls` (respecting `prefers-reduced-motion`), toggle **Show mesh grid**.
   - Main visual: Responsive 2D SVG showing the warping rubber mesh + data points + at final stage the crisp straight magenta knife line (`--c-grad`), paired with a compact side panel showing **Step-by-step transformation readout** (current layer, linear separability, and what this layer just did — e.g. *"Layer 1 pinched the ring into a U-shape; Layer 2 folded the two arms together so one straight line separates inner from outer."*).
3. **MDX integration (`src/lessons/forward-propagation/index.mdx`):**
   - Place right after `FeatureGallery` (or right before `## In code`) with a `<Predict>` (*"A `2 → 2 → 2 → 1` network classifies concentric rings with a single straight-line output neuron. How can a straight line separate a circle inside a ring?"*).
   - Add 1 conceptual quiz question to `src/lessons/forward-propagation/quiz.ts`.

---

### V3-3 — "It's All Still `W·x + b`" Architecture Rosetta Stone (`RosettaMatrix` in `src/lessons/depth-and-width/`)

**Files:** `src/lessons/depth-and-width/rosetta.ts`, `rosetta.test.ts`, `RosettaMatrix.tsx`, update `index.mdx` and `quiz.ts`.

1. **Engine (`rosetta.ts`):**
   - Fix a readable 6-element input vector `x = [0.2, 0.9, 1.0, 0.3, -0.4, -0.8]` (which can represent a 6-pixel 1D edge, 6 time steps, or 6 token scores — and let the learner nudge any entry `x_i` via compact sliders or click presets like *Step edge*, *Spike*, *Alternating*!).
   - Compute the **exact** `6 × 6` effective linear matrix `W_eff`, output `y = W_eff @ x`, **free parameter count**, and **effective multiply-adds** for all 5 architectures:
     1. **`dense` (Dense MLP):** arbitrary `6 × 6` matrix → **36** free weights, every cell independent.
     2. **`conv` (1D Convolution, kernel `k = [-1, 0, +1]`, same-padding):** exact `6 × 6` Toeplitz matrix whose 3 diagonals repeat `[-1, 0, +1]` and whose off-band entries are `0` → **3** free weights (`12×` fewer than dense!). Verify `matmul(W_conv, x)` equals direct 1D sliding convolution to `< 1e-12`!
     3. **`rnn` (Unrolled RNN, 3 steps of width 2):** block-lower-triangular `6 × 6` matrix where the same `2 × 2` recurrence block `W_h` repeats across steps `t = 1, 2, 3` → **4** shared recurrence weights (plus 4 input weights), showing weight tying across *time* instead of *space*.
     4. **`attention` (Self-Attention, dynamic `W(x)`):** compute real 1D/2D query-key projections `Q = x·w_q, K = x·w_k` (with a fixed 2D feature per position `[x_i, pos_i]`), exact `A = softmax(Q Kᵀ / √d)` (`6 × 6` row-stochastic matrix), and `y = A @ x`. **Key interactive magic:** when the learner drags `x_2`, the cells of `W_eff` *themselves* change live — whereas in dense/conv/RNN the matrix stays frozen!
     5. **`lora` (LoRA low-rank adapter `W₀ + A·B`, rank `r = 1`):** frozen base matrix `W₀` plus rank-1 outer product `a · bᵀ` → **12** trainable parameters (`6 + 6`) instead of `36` (`3×` fewer), showing exact matrix reconstruction `W₀ + ΔW`.
   - **Tests (`rosetta.test.ts`):**
     - Verify `conv` matrix equals direct 1D convolution (`< 1e-12`) across all input presets.
     - Verify `attention` rows each sum to `1.0` (`< 1e-12`) and changing `x` strictly changes `W_eff` (`max |W(x) - W(x')| > 0.05`), whereas `dense`/`conv`/`rnn`/`lora` `W_eff` is input-invariant.
     - Verify `lora` rank-1 `ΔW = A @ B` has exact parameter count `12` vs `36`.
     - Pin all quoted parameter ratios and outputs.
2. **Widget (`RosettaMatrix.tsx`):**
   - Controls: 5-way segmented switcher (*Dense MLP* | *1D Conv* | *RNN* | *Self-Attention* | *LoRA*), input preset buttons (*Step edge* | *Single spike* | *Wave*), plus inline sliders for `x₀..x₅` (or clickable bar-drag) so the learner immediately sees Attention's matrix morph when `x` changes!
   - Visual: Side-by-side `6 × 6` colour-coded heatmap of `W_eff` (highlighting shared weight groups with matching borders/badges in Conv & RNN, zero entries faded out, LoRA showing `W₀ + A·B` side by side) next to the `x → y` bar vectors and a **Parameter & structure badge** (`36 free` → `3 shared` → `dynamic from x` → `12 trainable`).
   - Live note explaining *what constraint was placed on `W`* and linking forward with `<LessonLink>` in prose to `convolution`, `rnns`, `attention`, and `lora`.
3. **MDX integration (`src/lessons/depth-and-width/index.mdx`):**
   - Add `## Zooming out: almost every architecture is still W·x + b` right before `<Quiz>` with a `<Predict>` (*"Which of these five architectures changes the entries of its effective weight matrix every time you feed it a new input?"* → Self-Attention!).
   - Add 1 quiz question to `src/lessons/depth-and-width/quiz.ts`.

---

### V3-4 — Saddle-Point Geometry in High Dimensions (`SaddleExplorer` in `src/lessons/gradient-descent/`)

**Files:** `src/lessons/gradient-descent/saddle.ts`, `saddle.test.ts`, `SaddleExplorer.tsx`, update `index.mdx` and `quiz.ts`.

1. **Engine (`saddle.ts`):**
   - **Exact combinatorics (`minProbability(d)`):** for a random critical point whose $d$ principal curvatures are independently positive or negative with probability $1/2$:
     - $P(\text{local min}) = (1/2)^d$, $P(\text{local max}) = (1/2)^d$, $P(\text{saddle}) = 1 - 2\cdot(1/2)^d = 1 - 2^{1-d}$ (for $d \ge 2$; `0` for $d = 1$).
     - Monte Carlo cross-check (`sampleCurvatures(d, trials, seed)`) verifying agreement on small $d$ and generating real random curvature bars for the widget's dimension slider `d ∈ [1, 30]`.
   - **Live 2D saddle trajectory (`runSaddleDescent(x0, y0, lr, steps)`):**
     - Surface: $f(x, y) = \tfrac{1}{2}(a x^2 - b y^2)$ (e.g. $a = 1.0, b = 0.6$), exact gradient $\nabla f = (a x, -b y)$ verified against finite differences (`< 1e-8`).
     - Default start `(x0 = 1.2, y0 = 0.02)` (almost dead-on the ridge!):
       - Measure and pin the exact plateau: steps `0 → 12` drop rapidly along `x` onto the saddle near `(0, 0)`, steps `12 → 25` appear to "stall" (`|Δf| < 0.01`, looking just like a flat loss curve plateau!), then steps `25 → 45` suddenly accelerate off the cliff along `y`!
       - Compare with exact ridge `(x0 = 1.2, y0 = 0.0)`: strictly trapped at `(0, 0)` forever (`y_t === 0` for all $t$), showing why a tiny perturbation (mini-batch SGD noise!) breaks every saddle!
   - **Tests (`saddle.test.ts`):**
     - Pin `minProbability(1) === 0.5`, `minProbability(2) === 0.25`, `minProbability(10) ≈ 0.0009766`, `minProbability(20) < 1e-6`.
     - Verify gradient vs central finite differences (`< 1e-8`).
     - Pin the plateau-then-escape trajectory from `(1.2, 0.02)` vs stuck `(1.2, 0.0)`.
2. **Widget (`SaddleExplorer.tsx`):**
   - Two coordinated panels:
     - **Left:** 2D contour `<Heatmap>` of the saddle $f(x, y) = \tfrac{1}{2}(x^2 - 0.6y^2)$ with clickable starting point (presets: *Near ridge `(1.2, 0.02)`*, *Dead on ridge `(1.2, 0.0)`*, *Off ridge `(1.2, 0.3)`*), animated gradient-descent path via `useReplay` + `ReplayControls`, and synchronized **loss-vs-step** mini curve showing the deceptive flat plateau before the escape drop!
     - **Right:** **Dimension slider `d = 1 .. 30`** showing `d` sampled curvature bars (`+` curving up in `--c-a`, `−` curving down in `--c-b`), a **"Resample critical point"** button, and live exact readout: *"In `d = 10` dimensions, only `0.10%` (1 in 1,024) of flat spots are local minima — `99.8%` are saddles with downhill escape ramps."*
3. **MDX integration (`src/lessons/gradient-descent/index.mdx`):**
   - Place right before `<Misconception>` (or right after the learning-rate lab) with a `<Predict>` (*"At a flat spot where every slope is zero in a 10-parameter network, if each direction is equally likely to curve up or down, roughly how often is it a true local-minimum trap?"* → `Less than 1 in 1,000` (`1 / 1,024`)!).
   - Add 1 quiz question to `src/lessons/gradient-descent/quiz.ts`.

---

### V3-5 — Tighter Module 5 Bridge (`backpropagation` ↔ `vanishing-gradients`)

**Files:** `src/lessons/vanishing-gradients/index.mdx`, optional pinned callback assertion in `src/lessons/vanishing-gradients/vanish.test.ts`.

1. **Opening bridge upgrade (`src/lessons/vanishing-gradients/index.mdx`):**
   - Replace the raw `[Backpropagation](#/learn/backpropagation)` link on line 7 with `<LessonLink to="backpropagation">Backpropagation, One Step at a Time</LessonLink>`.
   - Add a crisp 2-paragraph bridge right after the opening bullets (before `## 1. The signal telephone`) explicitly connecting Module 5's three lessons into one continuous mechanical story:
     - Recall `TinyMlpBackprop`'s *Saturated tanh* preset: just **one** hidden neuron sitting at $z_1 = 4.0$ had local slope $1 - h_1^2 = 0.0013$ and choked its incoming weights by **682×** relative to an active neighbour.
     - Recall `StepMicroscope`: when a hidden neuron's incoming gradients vanish, its crease line (`h_j = 0`) physically stops rotating — leaving the early layer frozen while the output layer tries to compensate alone.
     - Now stack 16 or 32 such valves in series: `SignalTelephone` measures what happens when those single-layer multipliers compound across depth, and `RescueDeepNetLab` tests the two architectural fixes — **normalisation** (which pulls pre-activations back into the active slope zone off the flat shelf) and **residual skips** (which wire a `+1` gradient bypass straight past every valve).
   - Also check `vanishing-gradients/index.mdx`'s existing `<Eq>` and `<Predict>` syntax: notice lines 27–32 use `legend={[...]}` and line 45 uses `explain="…"`. Do **not** break existing rendering if those components support them, or verify with `check-pages.cjs` that every existing widget and equation renders cleanly.
   - Add a short callback line in `vanishing-gradients/index.mdx`'s `## 4. Residual skip connections` / debrief tying back to the `682×` single-valve failure.

---

### V3-6 — "Paint Your Own Dataset" Brush in the Training Lab (`src/lessons/training-lab/`)

**Files:** `src/lessons/training-lab/customData.ts`, `customData.test.ts`, update `TrainingLab.tsx` and `index.mdx`.

1. **Engine helper (`customData.ts`):**
   - Pure, tested helpers for custom painted 2D points:
     - `clampPoint(x, y, bound = 1.0)`
     - `addPaintedPoint(pts, x, y, label, minSpacing = 0.06, maxPoints = 160)` (prevents accidental 1,000-point pileups while dragging!)
     - `hasBothClasses(pts)` (requires at least 2 points of class 0 and 2 of class 1 before training/scoring, showing a friendly prompt otherwise)
     - `seedWithOutlier(basePoints, x, y, label)` helper so learners can also drop a single stubborn opposite-class outlier onto an existing painted cloud!
     - `splitCustomTrainVal(pts)` (deterministic 80/20 stratified split so the Training Lab's train/test loss curves keep working cleanly on painted data!).
   - **Tests (`customData.test.ts`):** verify clamping, spacing deduplication, cap at 160 points, stratified split balance, and verify the real `Trainer` from `src/lessons/training-lab/lab.ts` converges to ≥ 95% accuracy on a synthetic painted 2-cluster dataset.
2. **UI integration (`src/lessons/training-lab/TrainingLab.tsx`):**
   - Add a **`Custom (paint)`** option to the dataset selector (strictly outside the 3 challenge briefs, which keep their locked datasets untouched!).
   - When `Custom (paint)` is active:
     - Show compact brush toolbar right above the plot: `Paint Class A (blue)` | `Paint Class B (orange)` | `Undo` | `Clear` | `Seed starter` (drops 12 clean points so the learner can immediately experiment or add a single outlier!).
     - Pointer click & drag on the 2D `<Plot>` paints points live (`onPointerDown` + `onPointerMove` while pressed), immediately resetting/feeding the live `Trainer`.
     - Live note adapts cleanly: *"Click or drag on the plot to paint Class A or Class B points — try drawing two islands, or drop one wrong-coloured dot inside a cluster to watch overfitting."*
   - Keep all 5 existing presets (`presets.ts` / `presets.test.ts`) and all 3 challenges (`challenges.ts` / `lab.test.ts`) 100% untouched.

---

### V3-7 — Sentence Microscope for a 4-Token Transformer (`SentenceMicroscope` in `src/lessons/transformer-block/`)

**Files:** `src/lessons/transformer-block/sentenceMicro.ts`, `sentenceMicro.test.ts`, `SentenceMicroscope.tsx`, update `index.mdx` and `quiz.ts`.

1. **Engine (`sentenceMicro.ts`):**
   - A tiny, completely real, inspectable 1-block causal Transformer (`d_model = 4`, `n_heads = 2`, `d_head = 2`, `d_ff = 8`, ReLU MLP, LayerNorm, causal mask, unembedding matrix `W_U`, softmax + cross-entropy).
   - Fixed 6-word vocabulary: `["The", "cat", "sat", "on", "mat", "dog"]`.
   - Prompt: `["The", "cat", "sat", "on"]` with expected next word `"mat"` at position 3 (`"on"`).
   - **Important:** Do **not** fake the weights with hand-waved probabilities — either train this tiny 4-dim Transformer cleanly on a small deterministic 6-sentence toy grammar (e.g. `"The cat sat on mat"`, `"The dog sat on mat"`, `"The cat ..."` using real backprop, or construct exact clean orthogonal/interpretable weights and verify every matrix multiplication, attention weight, LayerNorm, MLP activation, softmax, and `p − y` gradient in `sentenceMicro.test.ts`!).
     - **Even better (measure first!):** Construct/verify clean, interpretable weights where:
       - **Head 1 (Subject head):** at `"on"`, attends strongly (`> 0.65`) back to `"cat"` (carrying subject identity into the residual stream).
       - **Head 2 (Local/preposition head):** at `"on"`, attends strongly (`> 0.65`) to `"sat"` + `"on"` (carrying the `"sat on ___"` phrase context).
       - **Full block (`attention ON`, `residual ON`, `positions ON`):** predicts `"mat"` cleanly with high confidence (`p("mat") > 0.70`, top-1 = `"mat"`).
       - **Ablation 1 (`attention OFF`):** `"on"` can no longer see `"cat"` or `"sat"`, so `"mat"` probability drops sharply!
       - **Ablation 2 (`residual skip OFF`):** overwrites the token's own identity vector; prediction degrades or flips!
       - **Ablation 3 (`positions OFF`):** shuffling `"on sat cat The"` gives the exact same unpositioned bag attention at the last slot (or drops `"mat"` confidence measurably!).
     - Every single intermediate 4-number vector (`emb`, `pos`, `x0 = emb + pos`, `attnWeights[head][q][k]`, `attnOut`, `x1 = x0 + attnOut`, `mlpOut`, `x2 = x1 + mlpOut`, `logits[6]`, `probs[6]`, `dLogits = p - y`) is returned from `traceSentence(options)`.
   - **Tests (`sentenceMicro.test.ts`):**
     - Verify causal mask strictly zeroes `k > q` in every head (`attnWeights[h][q][k] === 0`).
     - Verify every attention row sums to `1.0` (`< 1e-12`).
     - Verify softmax probabilities sum to `1.0` and `p − y` matches central finite differences on `logits` (`< 1e-8`).
     - Pin exact `p("mat")` in full mode vs `attention OFF` vs `residual OFF` vs `positions OFF`.
2. **Widget (`SentenceMicroscope.tsx`):**
   - Controls:
     - Clickable 4-token sentence strip: `[ The ] [ cat ] [ sat ] [ on ] → ?` (default selected: `"on"`).
     - Three ablation toggles: `Attention: ON/off`, `Residual skips: ON/off`, `Positions: ON/off`.
   - Main visual (two responsive columns):
     - **Left: Residual-stream waterfall for the selected token** — 5 stacked horizontal 4-bar strips (`e_t`, `+ pos`, `+ attn` with live arrows showing *"68% from `cat`, 74% from `sat`"*, `+ MLP`, `= final stream`) so the learner literally watches a single 4-number vector get edited in place from top to bottom!
     - **Right: Next-token probability bars & `p − y` logit gradient** for all 6 vocab words (`mat` highlighted as target), updating live when you toggle Attention / Residual / Positions!
3. **MDX integration (`src/lessons/transformer-block/index.mdx`):**
   - Place right after the block anatomy widget (before parameter counting or `## In code`) with a `<Predict>` (*"If you switch off the attention layer inside `"The cat sat on"` while keeping the MLP and residual stream on, what can the last token `"on"` still know?"* → only its own word and position — never `"cat"` or `"sat"`!).
   - Add 1 quiz question to `src/lessons/transformer-block/quiz.ts`.

---

### V3-8 — Visual Verification, Cross-Theme/Mobile Polish & Wrap-Up

- Run `scripts/shoot-widgets.cjs` on every updated lesson (`forward-propagation`, `depth-and-width`, `gradient-descent`, `vanishing-gradients`, `training-lab`, `transformer-block`) in **light, dark, and matrix** themes.
- Drive interactive states & 390px phone views with `scripts/drive-widget.cjs`:
  - `Eq` words-first toggle ON + clicking `"Show symbols"` on a lesson page.
  - `SpaceFoldWidget` at `t = 0` vs `t = 2` (folded sheet + straight knife cut) on both `rings` and `moons`.
  - `RosettaMatrix` across `1D Conv`, `Self-Attention` (after moving an input slider), and `LoRA`.
  - `SaddleExplorer` on near-ridge plateau escape + `d = 10` curvature bars.
  - `TrainingLab` in `Custom (paint)` mode with painted dots and trained decision boundary.
  - `SentenceMicroscope` on `"on"` with full block vs `Attention: off`.
- Inspect every PNG visually; fix any contrast, label collision, uppercase `.num-table th` Greek symbol, or 390px overflow issues.
- Run both `node scripts/check-pages.cjs` and `node scripts/check-pages.cjs --theme=matrix` (0 console errors, 0 overflow at 1300px and 390px).
- Run full Vitest suite (`npx vitest run --testTimeout=20000 --no-isolate --exclude src/lessons/tiny-gpt/tinyGpt.test.ts`) + `npm run build` + `rg "<MODEL_NAME>" dist/` returning 0 matches.
- Update `docs/ROADMAP.md` (Current state, status table, Next steps, and any departures/surprises) and `HANDOFF.md`.

---

## 4. Implementation Checklist (`V3-1`–`V3-8`)

Make one commit per step and push directly to `claude/cool-feynman-4hci4d`. Mark the active step `IN PROGRESS` and tick it when pushed.

- [x] **V3-1:** Intuition-first entry ramp on `Home.tsx` + optional `Words first` reading mode (`src/mathMode.ts`, `src/components/Eq.tsx` + `eq.test.ts`, header toggle).
- [x] **V3-2:** Rubber-sheet space folding (`forward-propagation/spaceFold.ts`, `spaceFold.test.ts`, `SpaceFoldWidget.tsx`, `<Predict>`, MDX + quiz).
- [x] **V3-3:** `W·x + b` Architecture Rosetta Stone (`depth-and-width/rosetta.ts`, `rosetta.test.ts`, `RosettaMatrix.tsx`, `<Predict>`, MDX + quiz).
- [x] **V3-4:** High-dimensional saddle-point explorer (`gradient-descent/saddle.ts`, `saddle.test.ts`, `SaddleExplorer.tsx`, `<Predict>`, MDX + quiz).
- [ ] **V3-5:** Tighter Module 5 bridge in `vanishing-gradients/index.mdx` (linking `TinyMlpBackprop`'s `682×` saturated-`tanh` valve & `StepMicroscope` crease motion directly into `SignalTelephone` and `RescueDeepNetLab`).
- [ ] **V3-6:** `Custom (paint)` brush mode in `training-lab` (`customData.ts`, `customData.test.ts`, `TrainingLab.tsx`, MDX note).
- [ ] **V3-7:** 4-token Transformer `SentenceMicroscope` (`transformer-block/sentenceMicro.ts`, `sentenceMicro.test.ts`, `SentenceMicroscope.tsx`, `<Predict>`, MDX + quiz).
- [ ] **V3-8:** Visual screenshots (light, dark, matrix, 390px, driven states), both `check-pages.cjs` passes, full Vitest suite, `npm run build` + 0-match watermark check, update `docs/ROADMAP.md` & `HANDOFF.md`.
