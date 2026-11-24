# Roadmap & Progress

_Last updated: session 3 (in progress)._

## Current state (read this first if you're picking up mid-session)

Session 3 (owner's order): first a **Matrix theme**, then resume "Next steps" with Module 7. Commit and push after every finished lesson; keep this block current. Anything not listed as done has not been started (or only exists as uncommitted scratch).

- **Done this session:** Matrix theme (third theme: light / dark / matrix — see "Platform features").
- **Done:** Module 7 · *Overfitting and Underfitting* (`overfitting`): capacity dial (exact polynomial least squares, degree 0–11, 12–100 points, train/test vs degree on a log chart), twenty parallel universes (bias² + variance + noise measured over 200 training sets, adds up exactly to the measured error), a 1→32→32→1 network overfitting over time (best test 0.13 at ~step 300, then rises; but far gentler than the degree-11 polynomial), and an optional double-descent widget (min-norm fits past degree 11). All quoted numbers pinned in `src/lessons/overfitting/fit.test.ts`.
- **Done:** Module 7 · *Regularisation: early stopping, weight decay and dropout* (`regularization`). **Dropout added to `src/lib/mlp.ts`** (`setDropout(p, rng)`, inverted dropout on hidden layers, per-example masks, `lastMasks`, explicit masks for a deterministic numerical gradient check in `lib.test.ts`). Widgets: early stopping with patience; exact ridge regression on the degree-11 polynomial (λ slider, coefficient bars); a cure bench training the 1,153-parameter net live with none / weight decay / dropout (showing the real dropout masks); a fair race where each cure's strength is chosen on validation error by a committed search (`search.ts`, `search.run.ts`, ~3 min, recorded in `results.ts`, spot-checked by `reg.test.ts`), which can be re-run live in the browser. Honest result: every cure beat no cure on all 10 datasets (0.251 → 0.123 / 0.128 / 0.135), differences between cures are within noise.
- **Done:** Module 7 · *Lab: The Model Clinic* (`diagnosis-lab`) — **Module 7 complete.** Ten real broken runs (`cases.ts`), each pinned by `clinic.test.ts`: step too large (saturation; and a "quiet death" of ReLUs), step too small, dead-on-arrival init (biases −3), underfitting, overfitting, a leaky test set (93% test vs 75% fresh), random labels, unscaled inputs, unshuffled sorted data (see-saw). Instruments: loss/accuracy curves incl. fresh "production" data, decision boundary, dead/saturated neurons now and at step 0, weight/grad RMS, input ranges, train/test duplicates, neighbour label agreement, batch class mix. Diagnose from a menu of 9 causes (wrong answers explain what evidence rules them out), then fix it; optional goal = find all one-change fixes, counted exhaustively by `search.ts`/`search.run.ts` (~1 min, recorded in `fixes.ts`). Honest finding written into the debrief: the memoriser's "ReLU fix" is seed luck.
- **Done:** Module 5 · *A Neural Network from Scratch* (`network-from-scratch`): a line-by-line debugger of a 2→3→1 XOR network in NumPy-style matrix code (every matrix and shape shown), the "shapes write the backward pass" insight with an `Eq` for Z = AW, and a lab where the learner fills five blanks of the backward pass from candidate expressions — NumPy-style shape errors, a numerical gradient check, 24 of 1,024 combinations correct (equivalent forms all accepted), two wrong-but-still-trains options, optional speed goal with an exhaustive par over the 48 menu settings (`GOLF_PAR`, re-derived in `net.test.ts`). The NumPy snippet in the lesson was run and prints what it claims.
- **Done:** Module 12 · *Attention from Scratch* (`attention`): soft-dictionary-lookup widget (drag a query among keys; colour values blend), a hand-built sentence ("it" → "animal" 98%; switching "tired"→"wide" honestly changes nothing in one layer — motivates stacking), the `Eq` for softmax(QKᵀ/√d)V, a measured √d study (max weight 38%→95% unscaled vs ~31% scaled), and attention puzzles on an integer grid with exhaustive optimums (one level provably impossible: a key that's the average of two others can never win). All pinned in `attention.test.ts`.
- **Done:** Module 12 · *Multi-Head Attention* (`multi-head-attention`): one head vs two on the hand-built sentence — one softmax is a single 100% budget (combined head: referent 91%, predicates 6%; best balance only ~46% each), two heads give 98% and 99% in separate output blocks; head-cost calculator showing parameters 4d² and attention arithmetic are independent of h (GPT-2 small: 2,362,368 per layer). NumPy snippet verified against a per-head loop. No lab in this lesson (the attention lesson's puzzles cover steering a head).
- **Next up:** Module 12 · *Positional information* (plan under "Next steps").

## Status at a glance

| # | Module | Status |
| --- | --- | --- |
| 1 | Mathematical Foundations | **9 lessons done**: vectors, dot product, matrices, matrix multiplication & tensors, derivatives, gradients, chain rule, probability & softmax, logarithms/entropy/cross-entropy |
| 2 | The Artificial Neuron | **3 lessons done**: the neuron, activation functions (with live XOR training), Lab: Wiring Logic by Hand |
| 3 | Network Architecture | **3 lessons done (module complete)**: layers and the forward pass (step-through of a hand-built 2→6→2→1 "two islands" network; per-neuron response thumbnails with a break-a-neuron experiment); the network playground (build/train/inspect any small MLP: per-neuron maps, contribution bars, ablation importance, every tensor at a probe point, parameter breakdown, pruning challenge with an exhaustively computed optimum); depth vs width (parameter calculator with ×2 width/depth buttons; exact linear-piece counting for 1D ReLU nets — folding layers give 2^L pieces vs n+1 for one wide layer, with honest fragility and random-network experiments) |
| 4 | Loss & Optimization | **5 lessons done (module complete)**: gradient descent (fog walker + descent-golf puzzle with computed par); loss functions (tug of war: springs vs ropes → mean vs median; exact L2 vs L1 line fits with a draggable glitch; "stuck when wrong": squared error through a sigmoid vs cross-entropy, steps-to-escape); SGD and mini-batches (polling the gradient: 30 batch estimates around the true gradient with measured vs predicted 1/√B spread; a race on a fixed budget of examples processed, log-scale excess loss); momentum/RMSProp/Adam (race on valley/plateau/banana with each optimizer at its grid-searched best step size, finish-line stops, play/step/reset; optimizer golf on the banana with par from a 2,340-run search); learning-rate schedules (fast-or-precise jitter clouds; four schedules each at its searched best peak, averaged over 20 seeds; schedule-designer puzzle with five draggable knots, goal = beat every constant step size, optional = beat the best cosine, with a brute-force knot search as reference) |
| 5 | Backpropagation | **2 lessons done**: backpropagation (step-through graph explorer, cost comparison, fading signal); a neural network from scratch (matrix-code debugger, write-the-backward-pass lab with gradient check). Planned: vanishing/exploding gradients in depth |
| 6 | Training Lab | **1 lesson done**: the training laboratory (TF-Playground-style: 6 datasets incl. draw-your-own, 7 input features, up to 4×8 hidden layers, activation/optimizer/step size/batch/L2/seed, live boundary + neuron maps, train/test loss curves, pause/step/reset, saved runs for comparison, 5 test-pinned presets, 3 challenges with fewest-parameter references from a committed search) |
| 12 | Transformers & Attention | **2 lessons done**: attention from scratch (soft lookup, hand-built sentence, √d study, attention puzzles); multi-head attention (softmax budget, heads are free). Next: positions, the block, tokenisation |
| 7 | Generalisation & Failure Modes | **3 lessons done (module complete)**: overfitting and underfitting (capacity dial, bias–variance universes, network overfitting over time, double descent); regularisation (early stopping, weight decay, dropout, fair race); Lab: the model clinic (10 broken runs to diagnose and fix, exhaustive one-change fix counts) |
| 8–11, 13–18 | Classical ML → AI systems | planned (titles listed in `src/curriculum.ts`) |

## Platform features (done)

- Curriculum navigation with per-lesson completion, progress dashboard, three themes: light, dark and **matrix** (session 3: green-phosphor terminal; `src/theme.ts`, `src/components/DigitalRain.tsx`, `src/components/ThemeSwitch.tsx`, matrix block at the end of `global.css`; digital rain of the course's own symbols in the margins only, pause button, boot sequence on switch, reduced-motion aware).
- Lesson components: `Widget`, `Eq` (word equation + colour-coded symbols + legend), `Predict`, `Quiz` (multiple choice with misconception feedback, parameterised numeric, ordering), `Deeper`, `MathExhibit`, callouts, glossary `Term` tooltips.
- Spaced-repetition review (Leitner boxes) across all lesson quizzes; parameterised questions generate fresh numbers.
- Searchable glossary.
- Shared maths library with tests: linear algebra, numerical calculus, softmax/entropy, activations + derivatives, scalar autodiff (`Value`), MLP with hand-written backprop (gradient-checked), SGD/momentum/RMSProp/Adam/AdamW.
- SVG plotting primitives (`Plot`, `Handle`, `Arrow`, `FnPath`, `Heatmap` with contours, `Clip`, …), plus `fieldImage` (render f(x, y) to a data-URL thumbnail).
- `NetDiagram` (`src/components/NetDiagram.tsx`): any feed-forward network as columns of nodes and signed, weighted edges; optional per-node values or thumbnails, click-to-select neurons/edges (`data-node="col-j"`, `data-edge="l-j-i"` attributes for scripted tests), dead/hatched nodes. Used by the forward-pass lesson and intended for the playground and training lab.
- `LogChart` (`src/components/LogChart.tsx`): small line chart with a log y-axis (optional log x-axis) for errors spanning orders of magnitude; used by the Module 7 lessons.
- `scripts/ts-hooks.mjs`: lets `node --experimental-strip-types --import ./scripts/ts-hooks.mjs scratch/x.ts` import project modules (extensionless imports) — use it to verify numbers quoted in prose.
- Dev scripts: `scripts/check-pages.cjs` (every page, console errors + phone overflow), `scripts/shoot-widgets.cjs <lesson>` (screenshot every widget in light + dark), `scripts/drive-widget.cjs <lesson> <widgetIndex> '<actions JSON>'` (click/drag/screenshot a widget step by step). Screenshots land in `scratch/` (git-ignored).

## Architectural decisions

- **Vite + React + TypeScript + MDX.** MDX keeps prose first-class while allowing widgets inline. KaTeX renders maths at build time (zero runtime cost for prose maths); widgets use KaTeX at runtime for live equations.
- **Hash routing, static build.** Deploys to any static host with no configuration.
- **Lessons are isolated folders**, lazy-loaded. Cross-lesson imports are avoided (small datasets are duplicated rather than shared) so lessons can evolve independently.
- **Plain arrays, no ML framework.** Networks are small; transparent code that matches what lessons teach matters more than speed.
- **Progress in localStorage** (`nl-progress-v1`; lab state in `nl-logic-lab-v1`).
- **Fair races and honest references (session 2).** Whenever methods are compared (optimizers, schedules, batch sizes), each competitor gets its own best setting found by a grid search, and results that depend on randomness are averaged over fixed seeds. Puzzle "par" values are computed: exhaustively where feasible (pruning, golf), otherwise by a committed search over a stated space, shown to learners as "best found by searching …", never as a proven optimum.
- **Node and browsers can disagree in the last bit of `Math.exp`** (session 3). Long training runs amplify it to the third decimal place, so numbers are pinned by Node tests and quoted to sensible precision; widgets that re-run recorded searches live compare with a tolerance and explain why.
- **Lesson presets and quoted behaviours are pinned by tests** (e.g. `training-lab/presets.test.ts`), so a future change to shared code can't silently make the prose wrong.
- **Maths presentation:** key equations in the main flow via `Eq` (word version + colour-coded symbols + plain-English legend); `MathExhibit` only for deeper derivations. See `docs/STYLE_GUIDE.md`.

## Known limitations

- Widget captions name colours as they appear in the light/dark themes ("the purple line"); in the matrix theme the accent is green. Acceptable for now; a future pass could say "the accent-coloured line" or show a colour swatch.
- Learning-rate warmup is explained but not demonstrated: we tested it (spirals, 2→16→16→1 ReLU, Adam, peaks 0.01–0.1, 8 seeds) and it made no difference, so the lesson says so honestly rather than faking a demo. A convincing warmup demo would need a larger model; revisit if a worker/WebGL training backend is added.
- Module 3 lessons come *before* gradient descent and backprop in the curriculum order, so the playground's Train button is a preview (as the XOR trainer in the activation lesson already was); the prose says so implicitly by not explaining training. Pruning-optimum search is exhaustive and only enabled up to 14 hidden neurons.
- Heatmaps re-render on the main thread; fine for small networks, will need a worker or WebGL for the full training lab.
- KaTeX is in the main bundle (~0.5 MB JS total, gzip ~165 KB). Could be split later.
- Inline maths inside running prose (e.g. `$\mathbf{w}\cdot\mathbf{x}$`) isn't colour-coded yet; only `Eq` blocks are.
- No automated UI tests; verification so far is unit tests on maths plus manual/scripted screenshots.

## Next steps (in order)

1. **Module 12 · Positional information** (`positions`): attention is order-blind (show "dog bites man" = "man bites dog" give identical outputs, permuted — a real, testable fact), then sinusoidal and learned position vectors added to embeddings; RoPE as rotating q and k so that q·k depends on relative offset (2D rotation widget, connects to the matrices lesson).
2. **Module 12 · The transformer block** (`transformer-block`): attention → add & norm → feed-forward → add & norm; residual stream as a shared whiteboard; follow one token through a tiny block with real numbers; causal mask for language models.
3. **Module 12 · Tokenization** (planned there; can come first in the module order): BPE merges on a small corpus, live.
4. Then LLMs (sampling simulator: temperature/top-k/top-p), inference/hardware calculators (KV cache, VRAM).
5. Module 5 · Vanishing and exploding gradients in depth (planned there); Module 1's two planned lessons (statistics, high-dimensional geometry).

## Labs built so far

- **Attention puzzles** (Module 12, attention): steer one integer query so the right tokens win; optimums (fewest units) found by trying all 121 grid queries; one puzzle is provably impossible and the debrief explains why (convex-hull argument).
- **Write the backward pass** (Module 5, network-from-scratch): fill five blanks of a matrix-form backward pass; shape errors and a numerical gradient check give automatic feedback; 24 correct combinations; optional fewest-steps goal with exhaustive par.
- **The model clinic** (Module 7, diagnosis-lab): ten broken training runs to diagnose from real evidence and then fix; the number of one-change fixes per case is computed exhaustively over the bench's menu of single changes.
- **Training-lab challenges** (Module 6): a circle with no hidden layer (fewest found: 3 params, inputs x², y²), XOR from raw coordinates (13 params; 2 hidden neurons never trained to 95% from starts 1–3 — the lesson uses this to teach representable ≠ learnable), the spiral at 90% (13 params). References come from `src/lessons/training-lab/search.ts` (run `search.run.ts`, ~40 s), tried smallest-first, re-verified in `lab.test.ts`, and labelled in the UI as best-found within a stated space.
- **Design a schedule** (Module 4, lr-schedules): drag five knots of a log-scale step-size schedule; goal: finish within 0.15 of the best loss (averaged over 20 seeded runs), which no constant step size can; optional: beat the best cosine (0.085); a brute-force search over 400+ knot schedules (0.056) is offered as a reference, explicitly labelled as not a proof of optimality.
- **Optimizer golf** (Module 4, optimizers): reach the bottom of Rosenbrock's banana in the fewest steps by choosing optimizer, step size and β values; par computed by searching every optimizer × 65 step sizes × β grid.
- **Prune it** (Module 3, network playground): switch off as many hidden neurons of a trained network as possible while keeping ≥95% accuracy; the true minimum is computed by trying every on/off combination smallest-first (enabled up to 14 hidden neurons).
- **Logic workbench** (Module 2): hand-wire step neurons into AND/OR/NAND/XOR/XNOR; optional "parts cost" goal with brute-force-verified optimum.
- **Descent golf** (Module 4): reach the valley floor in the fewest steps; par computed live by searching step sizes (and momentum).

## Ideas for more labs

- **Optimizer golf (extended):** reach a loss threshold on a nasty landscape in the fewest steps; choose optimizer and schedule.
- **Parameter budget:** classify spirals at ≥95% with the fewest weights.
- **Attention puzzle:** set a query vector so a chosen token gets most of the attention.
- **VRAM Tetris:** fit a model + KV cache on a given GPU by choosing quantisation and context length.
- **Reward-hacking sandbox:** design a reward that a gridworld agent can't exploit.
