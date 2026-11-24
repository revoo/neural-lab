# Roadmap & Progress

_Last updated: session 1._

## Status at a glance

| # | Module | Status |
| --- | --- | --- |
| 1 | Mathematical Foundations | **9 lessons done**: vectors, dot product, matrices, matrix multiplication & tensors, derivatives, gradients, chain rule, probability & softmax, logarithms/entropy/cross-entropy |
| 2 | The Artificial Neuron | **3 lessons done**: the neuron, activation functions (with live XOR training), Lab: Wiring Logic by Hand |
| 3 | Network Architecture | planned |
| 4 | Loss & Optimization | see below |
| 5 | Backpropagation | see below |
| 6–18 | Training lab → AI systems | planned (titles listed in `src/curriculum.ts`) |

## Platform features (done)

- Curriculum navigation with per-lesson completion, progress dashboard, light/dark themes.
- Lesson components: `Widget`, `Eq` (word equation + colour-coded symbols + legend), `Predict`, `Quiz` (multiple choice with misconception feedback, parameterised numeric, ordering), `Deeper`, `MathExhibit`, callouts, glossary `Term` tooltips.
- Spaced-repetition review (Leitner boxes) across all lesson quizzes; parameterised questions generate fresh numbers.
- Searchable glossary.
- Shared maths library with tests: linear algebra, numerical calculus, softmax/entropy, activations + derivatives, scalar autodiff (`Value`), MLP with hand-written backprop (gradient-checked), SGD/momentum/RMSProp/Adam/AdamW.
- SVG plotting primitives (`Plot`, `Handle`, `Arrow`, `FnPath`, `Heatmap` with contours, `Clip`, …).

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
- Lessons 1–9 were written before the `Eq` component existed; some key equations still need the word-equation treatment.
- No automated UI tests; verification so far is unit tests on maths plus manual/scripted screenshots.

## Next steps (in order)

1. Gradient descent lesson + puzzle (in progress).
2. Backpropagation step-through lesson.
3. Module 3: forward propagation and a full network playground (inspect every neuron and weight).
4. Module 4 remainder: loss functions, SGD/mini-batches, momentum/RMSProp/Adam race, learning-rate schedules.
5. Module 6 training lab (TensorFlow-Playground-style) with Zachtronics-style objectives: hit a target accuracy under a parameter budget.
6. Then generalisation, transformers (flagship), LLMs, inference/hardware calculators.

## Ideas for more labs

- **Optimizer golf:** reach a loss threshold on a nasty landscape in the fewest steps; choose optimizer and schedule.
- **Parameter budget:** classify spirals at ≥95% with the fewest weights.
- **Attention puzzle:** set a query vector so a chosen token gets most of the attention.
- **VRAM Tetris:** fit a model + KV cache on a given GPU by choosing quantisation and context length.
- **Reward-hacking sandbox:** design a reward that a gridworld agent can't exploit.
