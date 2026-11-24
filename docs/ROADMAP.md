# Roadmap & Progress

_Last updated: end of session 2._

## Current state (read this first if you're picking up mid-session)

Session 2 worked through Module 3 → Module 4 → Module 6 in order, committing and pushing after every finished lesson. Keep this block current: the last completed lesson is recorded here, and anything not listed as done has not been started (or only exists as uncommitted scratch).

- **Done this session:** Module 3 complete — *Layers and the Forward Pass* (`forward-propagation`), *The Network Playground* (`network-playground`), *Depth, Width and Parameter Counts* (`depth-and-width`).
- **Also done:** Module 4 complete — *Loss functions* (`loss-functions`), *SGD and mini-batches* (`sgd-minibatch`), *Momentum, RMSProp and Adam* (`optimizers`), *Learning-rate schedules* (`lr-schedules`).
- **Also done:** Module 6 · *The Training Laboratory* (`training-lab`).
- **Next up:** Module 7 · Generalisation & failure modes (plan under "Next steps").

## Status at a glance

| # | Module | Status |
| --- | --- | --- |
| 1 | Mathematical Foundations | **9 lessons done**: vectors, dot product, matrices, matrix multiplication & tensors, derivatives, gradients, chain rule, probability & softmax, logarithms/entropy/cross-entropy |
| 2 | The Artificial Neuron | **3 lessons done**: the neuron, activation functions (with live XOR training), Lab: Wiring Logic by Hand |
| 3 | Network Architecture | **3 lessons done (module complete)**: layers and the forward pass (step-through of a hand-built 2→6→2→1 "two islands" network; per-neuron response thumbnails with a break-a-neuron experiment); the network playground (build/train/inspect any small MLP: per-neuron maps, contribution bars, ablation importance, every tensor at a probe point, parameter breakdown, pruning challenge with an exhaustively computed optimum); depth vs width (parameter calculator with ×2 width/depth buttons; exact linear-piece counting for 1D ReLU nets — folding layers give 2^L pieces vs n+1 for one wide layer, with honest fragility and random-network experiments) |
| 4 | Loss & Optimization | **5 lessons done (module complete)**: gradient descent (fog walker + descent-golf puzzle with computed par); loss functions (tug of war: springs vs ropes → mean vs median; exact L2 vs L1 line fits with a draggable glitch; "stuck when wrong": squared error through a sigmoid vs cross-entropy, steps-to-escape); SGD and mini-batches (polling the gradient: 30 batch estimates around the true gradient with measured vs predicted 1/√B spread; a race on a fixed budget of examples processed, log-scale excess loss); momentum/RMSProp/Adam (race on valley/plateau/banana with each optimizer at its grid-searched best step size, finish-line stops, play/step/reset; optimizer golf on the banana with par from a 2,340-run search); learning-rate schedules (fast-or-precise jitter clouds; four schedules each at its searched best peak, averaged over 20 seeds; schedule-designer puzzle with five draggable knots, goal = beat every constant step size, optional = beat the best cosine, with a brute-force knot search as reference) |
| 5 | Backpropagation | **1 lesson done**: step-through graph explorer with a learner-computed gradient, cost comparison, fading-signal (vanishing/exploding) demo |
| 6 | Training Lab | **1 lesson done**: the training laboratory (TF-Playground-style: 6 datasets incl. draw-your-own, 7 input features, up to 4×8 hidden layers, activation/optimizer/step size/batch/L2/seed, live boundary + neuron maps, train/test loss curves, pause/step/reset, saved runs for comparison, 5 test-pinned presets, 3 challenges with fewest-parameter references from a committed search) |
| 7–18 | Generalisation → AI systems | planned (titles listed in `src/curriculum.ts`) |

## Platform features (done)

- Curriculum navigation with per-lesson completion, progress dashboard, three themes: light, dark and **matrix** (session 3: green-phosphor terminal; `src/theme.ts`, `src/components/DigitalRain.tsx`, `src/components/ThemeSwitch.tsx`, matrix block at the end of `global.css`; digital rain of the course's own symbols in the margins only, pause button, boot sequence on switch, reduced-motion aware).
- Lesson components: `Widget`, `Eq` (word equation + colour-coded symbols + legend), `Predict`, `Quiz` (multiple choice with misconception feedback, parameterised numeric, ordering), `Deeper`, `MathExhibit`, callouts, glossary `Term` tooltips.
- Spaced-repetition review (Leitner boxes) across all lesson quizzes; parameterised questions generate fresh numbers.
- Searchable glossary.
- Shared maths library with tests: linear algebra, numerical calculus, softmax/entropy, activations + derivatives, scalar autodiff (`Value`), MLP with hand-written backprop (gradient-checked), SGD/momentum/RMSProp/Adam/AdamW.
- SVG plotting primitives (`Plot`, `Handle`, `Arrow`, `FnPath`, `Heatmap` with contours, `Clip`, …), plus `fieldImage` (render f(x, y) to a data-URL thumbnail).
- `NetDiagram` (`src/components/NetDiagram.tsx`): any feed-forward network as columns of nodes and signed, weighted edges; optional per-node values or thumbnails, click-to-select neurons/edges (`data-node="col-j"`, `data-edge="l-j-i"` attributes for scripted tests), dead/hatched nodes. Used by the forward-pass lesson and intended for the playground and training lab.
- `scripts/ts-hooks.mjs`: lets `node --experimental-strip-types --import ./scripts/ts-hooks.mjs scratch/x.ts` import project modules (extensionless imports) — use it to verify numbers quoted in prose.
- Dev scripts: `scripts/check-pages.cjs` (every page, console errors + phone overflow), `scripts/shoot-widgets.cjs <lesson>` (screenshot every widget in light + dark), `scripts/drive-widget.cjs <lesson> <widgetIndex> '<actions JSON>'` (click/drag/screenshot a widget step by step). Screenshots land in `scratch/` (git-ignored).

## Architectural decisions

- **Vite + React + TypeScript + MDX.** MDX keeps prose first-class while allowing widgets inline. KaTeX renders maths at build time (zero runtime cost for prose maths); widgets use KaTeX at runtime for live equations.
- **Hash routing, static build.** Deploys to any static host with no configuration.
- **Lessons are isolated folders**, lazy-loaded. Cross-lesson imports are avoided (small datasets are duplicated rather than shared) so lessons can evolve independently.
- **Plain arrays, no ML framework.** Networks are small; transparent code that matches what lessons teach matters more than speed.
- **Progress in localStorage** (`nl-progress-v1`; lab state in `nl-logic-lab-v1`).
- **Fair races and honest references (session 2).** Whenever methods are compared (optimizers, schedules, batch sizes), each competitor gets its own best setting found by a grid search, and results that depend on randomness are averaged over fixed seeds. Puzzle "par" values are computed: exhaustively where feasible (pruning, golf), otherwise by a committed search over a stated space, shown to learners as "best found by searching …", never as a proven optimum.
- **Lesson presets and quoted behaviours are pinned by tests** (e.g. `training-lab/presets.test.ts`), so a future change to shared code can't silently make the prose wrong.
- **Maths presentation:** key equations in the main flow via `Eq` (word version + colour-coded symbols + plain-English legend); `MathExhibit` only for deeper derivations. See `docs/STYLE_GUIDE.md`.

## Known limitations

- Learning-rate warmup is explained but not demonstrated: we tested it (spirals, 2→16→16→1 ReLU, Adam, peaks 0.01–0.1, 8 seeds) and it made no difference, so the lesson says so honestly rather than faking a demo. A convincing warmup demo would need a larger model; revisit if a worker/WebGL training backend is added.
- Module 3 lessons come *before* gradient descent and backprop in the curriculum order, so the playground's Train button is a preview (as the XOR trainer in the activation lesson already was); the prose says so implicitly by not explaining training. Pruning-optimum search is exhaustive and only enabled up to 14 hidden neurons.
- Heatmaps re-render on the main thread; fine for small networks, will need a worker or WebGL for the full training lab.
- KaTeX is in the main bundle (~0.5 MB JS total, gzip ~165 KB). Could be split later.
- Inline maths inside running prose (e.g. `$\mathbf{w}\cdot\mathbf{x}$`) isn't colour-coded yet; only `Eq` blocks are.
- No automated UI tests; verification so far is unit tests on maths plus manual/scripted screenshots.

## Next steps (in order)

1. **Module 7 · Generalisation & failure modes.** The training lab already shows overfitting (preset "Few noisy points, big network"; test loss 1.4 → ~5 while train accuracy > 90%). Build on it: (a) over/underfitting and the train/test gap, with a capacity-vs-data sweep; (b) regularisation — L2 (already in the lab's Trainer), early stopping, dropout (needs adding to `lib/mlp.ts` with a gradient check); (c) a "diagnose the broken model" lab: hand the learner misconfigured runs (leaky test set, wrong step size, dead ReLUs, unshuffled data) and ask them to find the cause. Reuse `src/lessons/training-lab/lab.ts` ideas by duplicating, not importing.
2. Consider a Module 5 "network from scratch" lesson (planned there) — the lab's `Trainer` is a good code-to-visual mapping target.
3. Then, transformers (flagship), LLMs, inference/hardware calculators.

## Labs built so far

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
