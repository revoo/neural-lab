# Roadmap & Progress

_Last updated: session 2 (in progress — see "Current state" below)._

## Current state (read this first if you're picking up mid-session)

Session 2 is working through Module 3 → Module 4 → Module 6 in order, committing and pushing after every finished lesson. The last completed lesson is recorded here; anything not listed as done has not been started (or only exists as uncommitted scratch).

- **Done this session:** Module 3 · *Layers and the Forward Pass* (`forward-propagation`); Module 3 · *The Network Playground* (`network-playground`).
- **Next up:** Module 3 · *Depth, Width and Parameter Counts* (`depth-and-width`). Plan under "Next steps".

## Status at a glance

| # | Module | Status |
| --- | --- | --- |
| 1 | Mathematical Foundations | **9 lessons done**: vectors, dot product, matrices, matrix multiplication & tensors, derivatives, gradients, chain rule, probability & softmax, logarithms/entropy/cross-entropy |
| 2 | The Artificial Neuron | **3 lessons done**: the neuron, activation functions (with live XOR training), Lab: Wiring Logic by Hand |
| 3 | Network Architecture | **2 lessons done**: layers and the forward pass (step-through of a hand-built 2→6→2→1 "two islands" network; per-neuron response thumbnails with a break-a-neuron experiment); the network playground (build/train/inspect any small MLP: per-neuron maps, contribution bars, ablation importance, every tensor at a probe point, parameter breakdown, pruning challenge with an exhaustively computed optimum) |
| 4 | Loss & Optimization | **1 lesson done**: gradient descent (fog walker + descent-golf puzzle with computed par) |
| 5 | Backpropagation | **1 lesson done**: step-through graph explorer with a learner-computed gradient, cost comparison, fading-signal (vanishing/exploding) demo |
| 6–18 | Training lab → AI systems | planned (titles listed in `src/curriculum.ts`) |

## Platform features (done)

- Curriculum navigation with per-lesson completion, progress dashboard, light/dark themes.
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
- **Maths presentation:** key equations in the main flow via `Eq` (word version + colour-coded symbols + plain-English legend); `MathExhibit` only for deeper derivations. See `docs/STYLE_GUIDE.md`.

## Known limitations

- Heatmaps re-render on the main thread; fine for small networks, will need a worker or WebGL for the full training lab.
- KaTeX is in the main bundle (~0.5 MB JS total, gzip ~165 KB). Could be split later.
- Inline maths inside running prose (e.g. `$\mathbf{w}\cdot\mathbf{x}$`) isn't colour-coded yet; only `Eq` blocks are.
- No automated UI tests; verification so far is unit tests on maths plus manual/scripted screenshots.

## Next steps (in order)

1. Module 3 remainder:
   - **Depth, Width and Parameter Counts** (`depth-and-width`): parameter-count formula + calculator for real layers; the "folding" demo — a 1D ReLU net where each extra *wide* neuron adds one kink but each extra *layer* of two tent-map neurons doubles the number of linear pieces (count the pieces exactly from the function); predict-before-reveal on doubling width vs depth; trade-off (deep is harder to train → link to backprop's fading signal).
2. Module 4 remainder (each its own lesson, in this order):
   - **Loss functions** (`loss-functions`): MSE vs MAE — the best constant under MSE is the mean, under MAE the median, so drag an outlier and watch MSE's fit get dragged; cross-entropy vs MSE for classification — CE punishes confident wrong answers without bound, and with a sigmoid output MSE's gradient vanishes exactly when the model is confidently wrong.
   - **SGD and mini-batches** (`sgd-minibatch`): the full gradient is an average over every example; estimate it from a sample like an opinion poll; widget shows sampled mini-batch gradient arrows around the true one, spread shrinking like 1/√batch; race measured in examples processed, not steps; epochs.
   - **Momentum, RMSProp and Adam** (`optimizers`): race on the same landscapes (narrow valley, banana, saddle, badly scaled axes) using `lib/optimizers`; "optimizer golf" with par computed by grid search.
   - **Learning-rate schedules** (`lr-schedules`): constant vs step vs cosine vs warmup+cosine on a noisy (mini-batch) problem — constant LR stalls at a noise floor, decay lets it settle. Only show a warmup benefit if it is reproducible in a real run; otherwise explain it honestly in prose.
3. Module 6 training lab (`training-lab`, TensorFlow-Playground-style): datasets (circles, XOR, clusters, spirals, moons, click-to-draw), optional input features (x², xy, sin…), architecture, activation, optimizer, LR, batch size, L2, noise, train/test split, seed; live boundary + train/test loss curves; pause/step/reset. Objectives on top (e.g. "≥95% test accuracy on spirals with the fewest weights"), with the best-known value computed by an offline search script that is committed alongside, and described honestly as "best found by searching …".
4. Then generalisation, transformers (flagship), LLMs, inference/hardware calculators.

## Labs built so far

- **Prune it** (Module 3, network playground): switch off as many hidden neurons of a trained network as possible while keeping ≥95% accuracy; the true minimum is computed by trying every on/off combination smallest-first (enabled up to 14 hidden neurons).
- **Logic workbench** (Module 2): hand-wire step neurons into AND/OR/NAND/XOR/XNOR; optional "parts cost" goal with brute-force-verified optimum.
- **Descent golf** (Module 4): reach the valley floor in the fewest steps; par computed live by searching step sizes (and momentum).

## Ideas for more labs

- **Optimizer golf (extended):** reach a loss threshold on a nasty landscape in the fewest steps; choose optimizer and schedule.
- **Parameter budget:** classify spirals at ≥95% with the fewest weights.
- **Attention puzzle:** set a query vector so a chosen token gets most of the attention.
- **VRAM Tetris:** fit a model + KV cache on a given GPU by choosing quantisation and context length.
- **Reward-hacking sandbox:** design a reward that a gridworld agent can't exploit.
