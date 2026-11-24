# Neural Lab — Editorial & Design Style Guide

This guide keeps every lesson, widget, quiz and tooltip in one voice. Read it before writing or editing content.

## The one-sentence brief

> It should feel like having Richard Feynman explain a fascinating machine, followed by a Zachtronics-style laboratory where you can take that machine apart, rebuild it and experiment with it yourself.

The reader is an experienced software engineer and lifelong learner — **not** a university student. Understanding is the reward; discovery is the entertainment. No badges, points or streaks.

## 1. Concept first, always

The learner must understand an idea *conceptually* before seeing its equation, so that when the equation appears the reaction is: *"ah, of course it looks like that."*

Every major idea follows this path:

1. **The problem.** What were we trying to do, and why did the obvious approach fail? Let the learner *feel* the failure (try XOR with one line; try dividing scores by their sum).
2. **A familiar picture.** An analogy that preserves the real structure (gears for the chain rule, a receipt for the dot product, a factory of multiplying machines for linear collapse).
3. **Hands on.** A widget where the learner manipulates the real mechanism, with prose that guides them to the important observations.
4. **Write it down.** The equation, presented accessibly (see §3), as a transcription of what they already understand.
5. **In code.** Usually Python/NumPy, plain loops before library calls.
6. **Test, stretch, summarise, connect** — quiz, an edge case or challenge, takeaways, and "In the real world".

## 2. Prose

- Conversational, warm, precise. A knowledgeable engineer at a whiteboard, not a textbook or a press release.
- Simplify the idea, never the truth. Don't manufacture sophistication with jargon.
- Introduce a technical term as **a name for something the reader already understands** ("that tally has a name: the weighted sum").
- One new idea at a time. Put the central insight first; push edge cases into `<Deeper>`.
- Analogies must be **structural**: map the relationships, then hand over to the real mechanism, and state the analogy's limits where they matter.
- Address misconceptions that will actually trip people up (`<Misconception>`), explaining why they're tempting.
- Prefer concrete scenarios (spam, houses, weather, a cat photo) over abstract symbols.
- British spelling in prose (colour, optimise) — be consistent.

### Preferred vs. avoided

| Avoid | Prefer |
| --- | --- |
| "The activation function introduces nonlinearity, thereby increasing representational expressivity." | "Without activation functions, stacking layers gives you nothing new — the whole network collapses into one layer. A bend is what lets layers build on each other." |
| "Softmax normalises the logit vector into a probability simplex." | "Softmax makes every score positive, then shares out 100% of the probability in proportion to them." |
| Opening a section with a formula. | Opening with a question the reader wants answered. |

## 3. Mathematics: first-class, but never intimidating

Maths is part of the main reading path — it's how the learner will eventually read papers. But it must be **accessible**:

- **Use the `<Eq>` component for key equations.** It shows a *word equation* on top (plain-English names), the symbolic version below, and a legend explaining every symbol. Example: *output = response(agreement with preferences + baseline)* above $a = f(\mathbf{w}\cdot\mathbf{x} + b)$.
- **Give symbols descriptive names** in the legend and nearby prose: "$b$ — the **bias**: the neuron's default mood before any evidence arrives", never just "$b$ is the bias term".
- **Colour-match** symbols across word equation, symbolic equation and widget using the KaTeX macros `\ca{}` (blue), `\cb{}` (orange), `\cc{}` (green), `\cg{}` (magenta, gradients), `\cp{}` (purple), `\cv{}` (red). Defined in `src/katexMacros.ts`.
- Explain notation the first time it appears (Σ is a `for` loop; subscripts are indexes; ∂ means "holding the others fixed").
- Use small concrete numbers; show intermediate arithmetic in widgets.
- `<MathExhibit>` (collapsed, "museum piece") is for **deeper** material only — derivations, proofs, historical theorems — never for the core equation a lesson depends on.
- Quizzes may include calculations, but should mostly test what the maths *means*.

## 4. Interactivity

- Every widget computes real numbers. No fake animations, no decorative controls.
- Widgets live inside `<Widget>` with a caption that tells the learner what to try and what to notice.
- Prefer **live explanations** (`.live-note`) that react to the current state ("the tanh has saturated, so…").
- Use **Predict** before revealing a surprising result.
- Colour semantics are global: blue/orange for the two classes and for positive/negative weights, green for results, magenta for gradients, purple for the model/accent.

## 5. Labs and puzzles (the Zachtronics principle)

Where a concept can be *built*, make it a lab:

- **Primitives** the learner already understands (neurons with weights and biases, an optimizer with a step size…).
- **A clear objective** with automatic checking (a truth table, a target accuracy, a loss threshold).
- **Observable machinery**: every intermediate value can be inspected; click a case to watch it flow.
- **Many valid solutions.** Celebrate the diversity in the debrief.
- **Functional first, optimisation optional.** After solving, show honest metrics (parts cost, parameter count, steps, memory) and, where known, the true optimum — computed, not guessed.
- Don't be cryptic: rules and controls must be obvious. Hints explain reasoning, not answers.
- Persist the learner's work locally.

## 6. Engineering conventions

- Each lesson is a self-contained folder: `src/lessons/<id>/index.mdx`, its widgets, `quiz.ts`, and any lesson-specific maths with tests.
- Shared code only where there's genuine reuse: `src/lib` (maths), `src/components` (teaching components, plotting).
- Register lessons in `src/curriculum.ts`. Add glossary terms in `src/glossary.ts`.
- Test any non-trivial maths (`*.test.ts`, Vitest). Gradient code is always checked numerically.
- In MDX, avoid raw `{`, `}` and `<` in prose. JSX attribute strings don't process escapes, so `tex="\frac{a}{b}"` works as written.
