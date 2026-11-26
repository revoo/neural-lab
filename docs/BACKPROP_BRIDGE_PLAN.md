# Plan — Deepening the Backpropagation & Multi-Layer Foundation (`docs/BACKPROP_BRIDGE_PLAN.md`)

**Status:** Approved by owner (26 November 2026). Ready to start at **B1**.

---

## 1. Why this work exists

Modules 2–7 already teach single neurons, activation kinks, hand-wired XOR, MLP forward passes, neuron ablation/pruning, scalar autograd on 1 neuron, batched matrix backward passes, a live Training Lab, and a 12-case Model Clinic.

Yet there is a real pedagogical jump right in the middle of **Module 5 (Backpropagation)**:

1. [`src/lessons/backpropagation/`](../src/lessons/backpropagation/) walks through a **1-neuron** scalar graph (`graph.ts`, 13 nodes, no hidden layer) and currently has **no hands-on lab** (`LabShell`).
2. The very next lesson, [`src/lessons/network-from-scratch/`](../src/lessons/network-from-scratch/), jumps straight to **batched matrix calculus** (`dh = dz2 @ W2.T`, `dW1 = X.T @ dz1`) on a 2-layer XOR network.
3. Nowhere yet does a learner watch backprop flow through a **tiny 2-layer scalar MLP** (seeing credit split across hidden neurons and *why* $W_2^\top$ appears), nor watch **one single training step under a microscope** (click a misclassified point → trace the error back → watch both hidden decision lines physically rotate and shift on the 2D plane).
4. Finally, `network-from-scratch` covers binary **sigmoid + BCE** (`dz2 = (p - y) / n`), leaving the multi-class **softmax + cross-entropy** cancellation (`dz = (p - y_one_hot) / n`) only lightly touched before CNNs and Transformers.

Because everything after Module 5 (CNNs, RNNs, Attention, Transformers, Diffusion, LoRA, RL) rests on multi-layer backpropagation, this plan closes all three gaps with real, inspectable machinery.

---

## 2. Placement, scope & length discipline

- **Primary target:** Upgrade **`src/lessons/backpropagation/`** (currently 2,001 words, 35 min, no `LabShell` lab) with:
  1. **Widget 1 (`TinyMlpBackprop`):** 2-layer scalar credit-assignment explorer (`2 → 2 → 2` MLP) showing multi-path fan-in, why $W_2^\top$ appears, symmetry breaking, and saturated-unit starvation.
  2. **Widget 2 (`StepMicroscope`):** Slow-motion **single-step geometry microscope** on a 9-parameter `2 → 2 → 1` MLP — click one point (or full batch), preview ghost lines, take 1 step, and watch the two hidden crease lines physically move as the loss drops.
  3. **Lab (`CreditLab` — *The Credit-Assignment Workbench*):** 3-brief `LabShell` lab testing sign-tracing, diagnosing & fixing broken gradient paths, and discovering why hidden-layer backprop is strictly required (frozen-hidden vs unfrozen-hidden on XOR).
  4. **History card & glossary:** New sourced history card `credit-assignment-backprop` in `src/history.ts` (falling inside Module 5's `1970 to 2015` era, complementing `backward-pass` in `network-from-scratch`) and any helpful glossary terms (e.g. `symmetry-breaking`, `dead-neuron`, `softmax-cross-entropy` if not already present — check `src/glossary.ts` first!).
- **Secondary target:** Add one focused section + interactive widget (**`SoftmaxBackprop`**) to **`src/lessons/network-from-scratch/`** (currently 2,016 words) right before `## In code`:
  - *"From yes-or-no to many classes: why softmax + cross-entropy still gives `p − y`"* (~350–450 words, `<Predict>`, `<Eq>`, interactive 3-class tiny MLP with live finite-difference verification, and an executed 15-line stdlib-Python verification snippet).
- **Length rule & split point:** Keep `backpropagation/index.mdx` under **~3,300 words** (matching `tiny-gpt` and `llm-serving`). If the upgraded `backpropagation` lesson clearly runs past ~3,400 words, **split cleanly** right after `## In code: an autograd engine in 30 lines`: keep single-neuron backprop + 30-line autograd + fading signal in `backpropagation`, and place `TinyMlpBackprop`, `StepMicroscope`, and *The Credit-Assignment Workbench* lab into a new lesson **`backprop-in-action`** (*Backpropagation in a Multi-Layer Network*) directly between `backpropagation` and `network-from-scratch` in Module 5 (updating `curriculum.ts`, `dependencies.ts`, `areas.ts` — chip `'foundational'` — and recording the split in `docs/ROADMAP.md`).
- **Preserve all existing lessons & tests:** Do not break or silently change any existing pinned numbers in `backpropagation/graph.test.ts` or `network-from-scratch/net.test.ts`. Keep all existing widgets (`BackpropCost`, `BackpropStepper`, `FadingSignal`, `CodeStepper`, `BackwardLab`) working.

---

## 3. Detailed design specification

### 3.1 Engine 1 — Two-layer scalar MLP & credit-assignment tracer (`src/lessons/backpropagation/tinyMlp.ts` + `tinyMlp.test.ts`)

Build a clean, self-contained engine using the shared autodiff `Value` (`src/lib/autodiff.ts`) or explicit float64 forward/backward (cross-checked against **both** `Value.backward()` and central finite differences to `< 1e-8`!):

1. **`2 → 2 → 2` Credit-Assignment Network (for `TinyMlpBackprop`):**
   - Inputs: $(x_1, x_2)$.
   - Layer 1 (`tanh` or `relu` selectable): $z^{(1)}_j = w^{(1)}_{j1} x_1 + w^{(1)}_{j2} x_2 + b^{(1)}_j$, $h_j = \phi(z^{(1)}_j)$ for $j \in \{1, 2\}$.
   - Layer 2 (linear outputs): $y_k = w^{(2)}_{k1} h_1 + w^{(2)}_{k2} h_2 + b^{(2)}_k$ for $k \in \{1, 2\}$.
   - Loss: $L = \tfrac{1}{2}\big((y_1 - t_1)^2 + (y_2 - t_2)^2\big)$.
   - **Why 2 outputs?** So each hidden neuron $h_j$ genuinely fans out to **two** downstream outputs:
     $$\frac{\partial L}{\partial h_j} \;=\; \underbrace{\frac{\partial L}{\partial y_1} w^{(2)}_{1j}}_{\text{path via } y_1} \;+\; \underbrace{\frac{\partial L}{\partial y_2} w^{(2)}_{2j}}_{\text{path via } y_2}$$
     Notice how the two weights reading $h_j$ during forward pass form **column $j$** of $W_2$ (or row $j$ in `(in × out)` layout), so the backward sum is literally multiplying `dy` by $W_2^\top$. The widget should expose this correspondence directly so the learner *never* wonders where the transpose in `dh = dz2 @ W2.T` comes from!
   - **Presets (all real computed numbers, pinned in `tinyMlp.test.ts`):**
     - **Balanced network:** clear non-zero paths from both $y_1$ and $y_2$ into $h_1$ and $h_2$ (including a case where one path pulls positive and the other pulls negative, partially cancelling!).
     - **Identical twin hidden neurons (symmetry trap):** $w^{(1)}_{1\cdot} = w^{(1)}_{2\cdot}$, $b^{(1)}_1 = b^{(1)}_2$, $w^{(2)}_{\cdot 1} = w^{(2)}_{\cdot 2}$. Stepping updates keeps $h_1 \equiv h_2$ bit-for-bit forever — pinning that 10 steps of gradient descent leave `|w1 - w2| === 0`!
     - **Saturated hidden neuron (`tanh`):** one hidden neuron has $|z^{(1)}_1| \approx 4$, so $1 - h_1^2 \approx 0.0013$ — watch its incoming weight gradients drop >500× relative to the active neuron.

2. **`2 → 2 → 1` Geometry Microscope Network (for `StepMicroscope`):**
   - 9 parameters total: hidden lines $h_1(x) = \tanh(w_{11}x_1 + w_{12}x_2 + b_1)$, $h_2(x) = \tanh(w_{21}x_1 + w_{22}x_2 + b_2)$, output $p(x) = \sigma(v_1 h_1 + v_2 h_2 + c)$, binary cross-entropy loss.
   - Dataset: a clean 4-point XOR (e.g. $(-1, -1)\to 0, (-1, 1)\to 1, (1, -1)\to 1, (1, 1)\to 0$ or $[0,1]^2$ — pick whichever sits nicest in `[-1.5, 1.5]^2` so both hidden zero-contour lines $w_{j1}x_1 + w_{j2}x_2 + b_j = 0$ visibly cross the plot!).
   - Helper `hiddenLineSegment(w1, w2, b, bounds)` computing the exact zero-contour line and unit normal vector for each hidden neuron so the 2D canvas draws **both** the filled probability heatmap AND the two crisp hidden crease lines (current solid + next-step preview dashed ghost!).
   - Supports **Single-point mode** (click any training point $i$: compute $\nabla L_i$, preview/apply 1 step on just that point, and show the change in loss on point $i$ *and* on the other points!) and **Full-batch mode** (all points averaged).
   - Helper `singleWeightBestStep()` or **Compare vs nudging 1 weight:** optionally show how much loss drops from 1 coordinated 9-weight backprop step vs nudging the single steepest weight by the same step size.

### 3.2 Widgets in `backpropagation`

1. **`TinyMlpBackprop`** (placed right after `BackpropStepper` & `Writing it down`, before the 30-line autograd or right after it — reorder if the narrative flows better!):
   - Interactive SVG network diagram (`2 → 2 → 2 → L`) with clickable nodes and edges.
   - Clicking a hidden neuron $h_1$ or $h_2$ highlights **both** outgoing paths to $y_1$ and $y_2$, displays the explicit two-term sum $\frac{\partial L}{\partial h_j} = \delta_1 w_{1j} + \delta_2 w_{2j}$, and displays a compact side-by-side **"Matrix view"** chip showing the exact row of $W_2^\top$ matching those two scalar edges.
   - Controls: preset switcher (*Balanced*, *Cancelling paths*, *Identical twins*, *Saturated tanh*), activation toggle (`tanh` / `ReLU`), **Take 1 update step** & **Reset** buttons (so on *Identical twins* the learner can click 5 times and see the twins stay mathematically fused!).
   - Live note explaining the currently selected node or preset.
   - Include a `<Predict>` right before the widget (e.g., *"Both hidden neurons start with the exact same weights, and we train for 100 steps with plain gradient descent. What happens?"* — options: they specialise automatically / they stay identical forever / gradients explode).

2. **`StepMicroscope`** (immediately after `TinyMlpBackprop`):
   - Left panel: 2D plot (`Plot` or crisp SVG with `fieldImage` heatmap) showing the `2 → 2 → 1` probability field, the data points (clickable!), solid lines for `h₁ = 0` and `h₂ = 0`, and **dashed ghost lines** showing where `h₁` and `h₂` will move on the next step (toggled or always shown on hover/preview).
   - Right panel: compact 9-parameter gradient table / mini network showing `w`, `grad`, and `Δw = −η·grad`, plus per-point loss bars (before → after).
   - Controls: click any point on the plot (or click **"Full batch"**), learning-rate slider `η`, **"Take 1 step"**, **"Run 25 steps"** (animated via `useFramePlayer` / `useReplay`, respecting `prefers-reduced-motion`), **"Reset"**.
   - **Delight moment:** When you click a single misclassified point and press **"Take 1 step"**, watch its loss drop sharply while a neighbouring opposite-class point's loss *rises slightly* — then switch to **"Full batch"** and watch the lines settle cleanly between all four points!

### 3.3 Lab — *The Credit-Assignment Workbench* (`src/lessons/backpropagation/CreditLab.tsx` + `creditLab.ts` + `search.run.ts` → `results.ts`)

Built on `LabShell` + `useLabProgress` + `AnswerReveal` + `MachineLog` + `LiveCode` (real PyTorch `loss.backward()` / `register_hook` / `requires_grad = False` verified code) + `<LabJob>` + `<ForReal>` + `<HistoryCard id="credit-assignment-backprop" />`.

**Three briefs (measure every number and verify solution counts exhaustively before fixing thresholds!):**

1. **Brief 1 — "Trace the signs without a calculator":**
   - A frozen `2 → 2 → 1` network (`tanh` hidden, linear/sigmoid output) evaluates one concrete input $(x_1, x_2)$ with target $y$. All forward activations ($x_1, x_2, h_1, h_2, p$) and weights ($W_1, W_2$) are visible — gradients are hidden.
   - The learner sets the predicted **direction of update** (`increase +`, `decrease −`, or `unchanged 0`) for **4 chosen weights** (1 output weight, 2 hidden weights including one behind a negative output weight so signs flip!, and 1 weight feeding a dead ReLU or zero input where gradient is `0`).
   - Pressing **Check & run backprop** reveals the true exact gradients, updates the log, and checks all 4 signs.
   - *Design note:* Ensure every sign is unambiguous (large non-zero magnitude, e.g. `|grad| > 0.05`, except the exact `0` case). Optional goal: get all 4 on the first try (or inspect the exact `|grad|` ranking: which of the 4 weights moves most?). *Wait: no badges/points; optional goal can be "also identify which weight has the largest |gradient|."*

2. **Brief 2 — "Unstick the broken hidden layer":**
   - A `2 → 2 → 1` network on XOR is broken at initialisation in a diagnosed way (e.g. **symmetric identical hidden weights + saturated scale**, OR **two distinct broken starts** — let the implementer test and pick the cleanest single broken start whose gradient telemetry tells the whole story: e.g., both hidden neurons start identical AND large `|w| = 3.5` under `tanh`, so BOTH symmetry AND saturation kill learning!).
   - Finite control grid (say `initMode`: `keep broken` / `break symmetry (small)` / `break symmetry (large)` / `zero init`; `activation`: `tanh` / `sigmoid` / `relu`; `lr`: `0.05` / `0.2` / `0.5` / `1.5` — e.g. 4 × 3 × 4 = 48 or 60 settings, averaged over fixed seeds or deterministic starts!).
   - Objective: reach **100% XOR accuracy** and **loss ≤ 0.08** within 150 steps.
   - **Verify exhaustively before fixing thresholds:**
     - `keep broken` and `zero init` score **0 passes** across every activation & learning rate (proving LR alone cannot break symmetry!).
     - Multiple valid combinations pass (across activations & LRs).
     - Optional goal: reach the target loss in the fewest steps (exact exhaustive optimum recorded in `results.ts` and re-verified in `creditLab.test.ts`).

3. **Brief 3 — "Do you actually need hidden-layer backprop?":**
   - A `2 → 3 → 1` network on XOR starts from a fixed seeded initialisation where the 3 random hidden neurons **cannot** linearly separate XOR on their own (verify this strictly: freezing all of layer 1 (`W1, b1` frozen) and training only layer 2 (`W2, b2`) across every LR fails to solve XOR — max accuracy ≤ 75%!).
   - Controls: **freeze/unfreeze toggles** for each of the 3 hidden neurons (`h₁`, `h₂`, `h₃`) + the output layer (`out`) — $2^4 = 16$ masks (or 3 hidden toggles × 2 LRs = 16–32 configs, exhaustively searched!).
   - Objective: train for 200 steps and reach **100% XOR accuracy** and **loss ≤ 0.08**.
   - **The payoff:**
     - Freezing all hidden neurons (output-only training, i.e., a random-feature linear classifier) **provably fails** (0 passing LRs).
     - Unfreezing **even 1 or 2** hidden neurons (measure which exact subsets work!) lets backprop bend those hidden lines into linear separability — and the optional goal asks for the **fewest unfrozen hidden neurons** (exhaustive minimum over all 16 masks!).
     - Even better: plot the **hidden space $(h_1, h_2)$** alongside the input space $(x_1, x_2)$ so the learner literally sees XOR untangle into linearly separable points when hidden backprop is turned on, and stay tangled when layer 1 is frozen!

- **Evidence-only hints (`hints.test.ts`):** Every hint points only at visible evidence (e.g. "Compare `h1` and `h2`'s gradients in the probe: notice they are identical to 4 decimal places" or "Check the output-only run in the hidden-space plot: can a single straight line separate the two classes there?") and **never** names a button, toggle, or setting.

### 3.4 Secondary Upgrade — Multi-Class Softmax + Cross-Entropy Bridge in `network-from-scratch`

Add a crisp ~350–450 word section right before `## In code` in [`src/lessons/network-from-scratch/index.mdx`](file:///home/antigravity/revoo/neural-lab/src/lessons/network-from-scratch/index.mdx):

- **Heading:** `## From yes-or-no to many classes: why softmax + cross-entropy still gives p − y`
- **`<Predict>`:** Suppose a 3-class classifier outputs probabilities `p = [0.20, 0.70, 0.10]` when the true class is class 0 (`y = [1, 0, 0]`). Which class logit gets pushed *up*, which get pushed *down*, and what do the three logit gradients add up to? (Answer: class 0 pushed up by `-0.80` grad, classes 1 and 2 pushed down proportionally to their wrong probabilities `+0.70` and `+0.10`, sum = `0`!)
- **`<Eq>`** with colour-coded legend showing $\frac{\partial L}{\partial z_k} = p_k - y_k$ (and `/ n` over a batch), plus a collapsed `<MathExhibit>` giving the 4-line algebraic cancellation between the softmax Jacobian and $-1/p_c$.
- **Widget (`SoftmaxBackprop` in `src/lessons/network-from-scratch/SoftmaxBackprop.tsx` + engine in `softmax.ts` + `softmax.test.ts`):**
  - 3 interactive logit sliders (`z₀`, `z₁`, `z₂`) + true-class selector (`Class A / B / C`) + one-click presets (*Confident & right*, *Confident & wrong*, *Flat tie*).
  - Displays: Softmax probability bars `p`, Cross-entropy loss $-\log p_{\text{true}}$, the exact analytic gradient `p − y`, the **live central finite-difference gradient** side-by-side (`max |diff| < 1e-8`), the sum-to-zero invariant ($\sum_k (p_k - y_k) = 0$), and a **"Take 1 logit step"** button so the learner sees probability mass flow from wrong classes straight into the true class.
- **Executed Python snippet:** 15-line stdlib Python verifying `p − y` against central finite differences on `[2.0, 1.0, -0.5]`, run directly with real stdout pasted.

---

## 4. Creative licence & honesty rules

You have creative licence to make these widgets and puzzles genuinely delightful and surprising:
- If the **hidden-space untangling view** in Brief 3 or an interactive **click-to-nudge a weight on the 2D plot** makes the geometry click harder, add it.
- If a planned preset or brief threshold behaves differently once measured (e.g. if a random `2 → 3 → 1` start accidentally *does* linearly separate XOR with frozen layer 1), **measure first**, pick a clean fixed seed where the failure is genuine and documented honestly, or report what actually happens.
- **Non-negotiables:**
  - Every number computed for real (no fake charts or hardcoded gradients).
  - Every analytic gradient checked against finite differences (`< 1e-8`).
  - Every quoted number in prose pinned by a Vitest test.
  - Exhaustive counts over the lab's finite control space (committed `search.run.ts` → `results.ts`, re-verified by test).
  - Evidence-only hints tested in `hints.test.ts`.
  - All widgets themed purely via CSS variables (no hard-coded hex), inspected in **light, dark, matrix, and 390px phone width** (`scripts/shoot-widgets.cjs` + `scripts/drive-widget.cjs`).
  - Respect `prefers-reduced-motion` in any step animation.
  - British spelling, concept before equation, `<Eq>` with legend (`\ca \cb \cc \cg \cp \cv`, never `\cr`), no badges/points.
  - Verify every historical claim on its source page itself (`<Src>` / `src/history.ts`).
  - Invisible authorship watermark per `HANDOFF.md` ("Owner's standing instructions"): file headers, `Authored-by:` commit trailer, roadmap tag, and `rg "<MODEL_NAME>" dist/` returning **0 matches**.
  - Commit and push directly to `claude/cool-feynman-4hci4d`; never create other branches or open pull requests.

---

## 5. Steps and checklist (`B1`–`B6`)

Make one commit per step and push to `claude/cool-feynman-4hci4d`. Tick each step when pushed and mark the active step `IN PROGRESS`.

- [x] **B1 Engines & tests (`tinyMlp.ts` + `tinyMlp.test.ts`, `softmax.ts` + `softmax.test.ts`):**
  - `2 → 2 → 2` scalar credit-assignment network on `Value` + explicit backward, finite-diff checked `< 1e-8`, with pinned presets (*Balanced*, *Cancelling paths*, *Identical twins* staying bit-identical over 10 steps, *Saturated tanh* >500× attenuation).
  - `2 → 2 → 1` geometry microscope network with exact zero-contour line solver, single-point vs full-batch step, and finite-diff check.
  - `network-from-scratch/softmax.ts`: 3-class softmax + cross-entropy analytic vs central finite-diff check (`< 1e-8`), sum-to-zero invariant.
- [ ] **B2 (IN PROGRESS) Lab engine, exhaustive search & hints (`creditLab.ts`, `search.run.ts` → `results.ts`, `creditLab.test.ts`, `hints.test.ts`, `realCode.ts` + `realCode.test.ts`):**
  - Measure all 3 briefs of *The Credit-Assignment Workbench* before fixing thresholds; ensure multiple valid routes and clear failing routes on Briefs 2 and 3.
  - Pin all exhaustive counts, optimums, failing routes, and evidence-only hint invariants in tests.
  - Verify PyTorch `LiveCode` snippets (run locally if PyTorch is installed, or verify against standard PyTorch `autograd` / `requires_grad` / `register_hook` API and document clearly).
- [ ] **B3 Widgets & Lab UI:**
  - `TinyMlpBackprop.tsx` (clickable `2 → 2 → 2` diagram, dual-path fan-in highlight, side-by-side $W_2^\top$ matrix correspondence, preset switcher, step button).
  - `StepMicroscope.tsx` (2D heatmap + solid/ghost hidden lines + clickable points + 9-weight gradient table + animated replay respecting `prefers-reduced-motion`).
  - `CreditLab.tsx` (`LabShell`, all 3 briefs including hidden-space untangling view in Brief 3, `MachineLog`, `LiveCode`, `AnswerReveal`).
  - `src/lessons/network-from-scratch/SoftmaxBackprop.tsx` (3-class logit sliders, live `p − y` vs finite-diff table, 1-step button).
- [ ] **B4 Prose, quizzes, history & wiring:**
  - Integrate widgets, `<Predict>`s, `<Eq>`s, *The Credit-Assignment Workbench* lab + debrief + `<ForReal>` + `<HistoryCard id="credit-assignment-backprop" />` into `backpropagation/index.mdx` (split cleanly into `backprop-in-action` if it runs past ~3,400 words).
  - Add the softmax + cross-entropy bridge section, `<Predict>`, `<Eq>`, `<MathExhibit>`, `<SoftmaxBackprop />`, and executed Python snippet into `network-from-scratch/index.mdx`.
  - Add 1–2 new quiz questions to `backpropagation/quiz.ts` (symmetry breaking / multi-path fan-in / why $W_2^\top$ appears) and 1 to `network-from-scratch/quiz.ts` (`p − y` sum-to-zero / wrong-class push).
  - Add verified history entry `credit-assignment-backprop` in `src/history.ts` (and bump `HISTORY.length` in `src/history.test.ts` from 65 → 66); add any new glossary terms to `src/glossary.ts`; update `backpropagation` summary & minutes in `src/curriculum.ts` (keeping summary ≤ 30 words, zero symbols).
- [ ] **B5 Visual & browser verification:**
  - Run `node scripts/shoot-widgets.cjs backpropagation` and `node scripts/shoot-widgets.cjs network-from-scratch` in light, dark, matrix, and at 390px; drive interactive states (symmetry-trap step, single-point microscope step, Brief 2/3 lab solves) with `scripts/drive-widget.cjs`.
  - Inspect every PNG visually and fix any overlap, contrast, wrapping, or mobile layout issues.
  - Run both `node scripts/check-pages.cjs` and `node scripts/check-pages.cjs --theme=matrix` (0 console errors, 0 overflow at 1300px and 390px).
- [ ] **B6 Wrap-up:**
  - Full Vitest suite (`npx vitest run --testTimeout=20000 --exclude src/lessons/tiny-gpt/tinyGpt.test.ts`) + `npm run build` + `rg "<MODEL_NAME>" dist/` returning 0.
  - Update `docs/ROADMAP.md` (Current state, status table, Next steps, Labs built so far, and any departures or unexpected measured results), tick B6 here, remove the `"NEXT TASK"` note from `HANDOFF.md` §0, commit and push.
