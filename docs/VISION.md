# Neural Lab — Product Vision & Brief

This is the project owner's brief, condensed from the original specification plus clarifications given during development. It is the source of truth for *what* we're building and *why*. `STYLE_GUIDE.md` covers *how* to write and design; `ROADMAP.md` covers *where we are*.

## Vision

A comprehensive, highly interactive website teaching AI, machine learning, neural networks and modern deep-learning architectures from first principles. It combines the explanatory philosophy of **3Blue1Brown**, the interactivity of **Brilliant.org** and the experimental freedom of **TensorFlow Playground**. Interactivity is the primary teaching mechanism — not static articles with occasional illustrations.

Long-term ambition: virtually every important AI concept can be understood through visualisation, experimentation, mathematical exploration and active problem-solving.

## Inspirations (owner's emphasis)

- **Richard Feynman** — clarity, curiosity, everyday analogies, first principles, making sophisticated ideas accessible without diminishing them.
- **Zachtronics games** (Opus Magnum, SHENZHEN I/O, EXAPUNKS) — systems-based puzzles, observable machinery, constrained problem-solving, multiple valid solutions, the satisfaction of independent discovery.
- **3Blue1Brown** — geometric intuition, visual storytelling, tangible abstract maths.
- **Brilliant.org** — active learning, progressive challenges, participation over consumption.
- **ChatGPT/Gemini at their best** — natural conversational prose, thoughtful metaphors, clear prioritisation, adapting to the reader's existing mental models.

> The ideal experience feels like having Richard Feynman explain a fascinating machine, followed by a Zachtronics-style laboratory where the learner can disassemble, modify, rebuild and experiment with that machine independently.

The learner should frequently forget they are studying. **Understanding is the reward; discovery is the entertainment.** Never gamify with badges, points, achievements or artificial reward schedules.

## Audience

An experienced software engineer and lifelong learner — *not* a university student. Understands programming, algorithms, data structures, architecture. Limited formal ML/maths background. Wants to understand how AI really works, learns best through experimentation and visual feedback, and eventually wants to read papers, reason about architectures and trade-offs, and run independent experiments.

- Don't teach elementary programming.
- Assume only school maths: arithmetic, fractions, percentages, basic algebra, simple graphs. Do **not** assume linear algebra, calculus, probability theory, logarithms, optimisation or paper notation — introduce each when it becomes useful.
- Leverage the programmer's existing intuitions: Σ is a loop, a vector is an array, a matrix multiply is nested loops, a computational graph is a dependency graph, backprop is a reverse traversal, broadcasting is implicit expansion.

## Owner's clarifications during development (important)

1. **Conceptual understanding trumps everything.** The learner must understand the concept so well that on seeing the equation they think *"ah yeah, of course it looks like that."*
2. **This is not a maths textbook** — it's a place to build intuition, for lifelong learners rather than college students.
3. **…but don't over-correct: maths is still first-class and necessary.** Make it accessible and unintimidating, not hidden. Use descriptive names for symbols rather than bare `x`, `y` (e.g. "bias — the neuron's default mood"). → Implemented as the `Eq` component: plain-English word equation on top, colour-coded symbols, legend underneath. `MathExhibit` is only for genuinely deeper optional material.
4. **Turn concepts into open-ended engineering puzzles** where appropriate: primitives, a well-defined objective, observable behaviour, freedom to find one's own solution. Functional solution first; optional optimisation objectives (accuracy, parameter count, compute, memory, training time) afterwards. Never artificially cryptic.

5. **Work so that anyone can pick up at any moment** (session 2). After each finished lesson: commit, push, and update `docs/ROADMAP.md` ("Current state" and "Next steps") and `HANDOFF.md` if the procedure changed. Assume the session may disconnect after any lesson and another agent will continue from the repository alone — leave plans, decisions and gotchas written down, not in your head.

6. **A third theme, "Matrix" (session 3).** Alongside light and dark: green-phosphor terminal, digital rain, glowing green accents, mainframe-hacker energy — geeky and very theme-specific, but never hard to read. → Implemented as `data-theme='matrix'` (tokens and rules at the end of `src/styles/global.css`), `src/components/DigitalRain.tsx` (rain made of the course's own symbols and terms, in the margins only), a short boot sequence on switching, a rain pause button, and reduced-motion support. Every new widget must look right in all three themes.

7. **Session 3 working order (owner).** Build the Matrix theme first, then work through the roadmap's "Next steps" in order (Module 7 → Module 5's network-from-scratch → Module 12), committing and pushing after every finished lesson, comparing methods fairly (each competitor at its own searched best, averaged over fixed seeds) and reporting honestly when a textbook story doesn't hold in our small problems.

8. **Proportionate checking (November 2026).** This is a static educational site: keep correctness strict (every number a learner reads is computed, lab optimums are searched, history is sourced), but keep process light. Tests and build before each commit; the full page checker once per lesson or work package; screenshots in every theme only for new widgets; no tests for arithmetic a reader can check in their head.

9. **3Blue1Brown-style maths labs and history in the main text (November 2026).** The maths toolkit at the front should feel like 3Blue1Brown, with Zachtronics-style labs of its own, explicit "Why this, and not something else?" comparisons, and clear bridges to the AI lessons that use each idea. History and "the struggle" (dead ends, winters, disputes, ideas waiting decades for data and hardware) belong in the main text — opening hooks, era interludes, a prologue and epilogue, and "You are standing where they stood" lab notes — not tucked away only in collapsed cards.

10. **Chips instead of tracks (v2, November 2026).** The owner replaced the planned Full Course / Core Track switch with visual chips: **Foundational** on concepts integral to the whole backbone of AI, and a "where this belongs" chip on every module and lesson (e.g. *Computer Vision Foundation*, *LLM & Transformers Backbone*). Chip labels use Title Case on every word. → `src/areas.ts`; Foundational is checked against the concept map (feeds at least three of the four big branches).
11. **Watching it come alive (V1, November 2026).** The owner wants V1 to include a tiny-GPT capstone and visual labs on how a network learns to see and to speak. These should make the learner *feel* a network coming alive: random noise turning, step by step, into something that recognises and speaks, a sense that something intelligent is being born under your cursor. Spectacle is wanted here, but it must stay honest: every visual is the real computation, each "moment" is detected by a measured threshold rather than staged, and the prose says plainly what has and hasn't emerged. → `HANDOFF.md`, "Next tasks (owner, November 2026)", "The feeling to aim for".

## Lesson methodology

Ideally each major lesson: (1) explain the problem, (2) build an intuitive mental model, (3) interactive visualisation, (4) let the learner manipulate variables, (5) explain the maths, (6) show code, (7) test understanding, (8) a challenging experiment exposing an edge case or limitation, (9) summarise transferable principles, (10) connect to real-world AI.

Also: progressive disclosure; motivate inventions (what problem existed, why previous approaches failed); frequently ask the learner to **predict** before running an experiment; let the learner **feel the problem** before the solution (XOR with one line, divergence before learning-rate advice, overfitting before regularisation); use thought experiments and counterfactuals ("what if we removed this?"); connect new ideas to ones already learned ("attention is another weighted sum, with weights computed from the input"); address misconceptions likely to interfere with later learning; use concrete recurring scenarios.

## Prose (first-class feature)

Conversational, warm, precise — an exceptionally articulate engineer at a whiteboard. Not academic, encyclopedic, documentation-like or machine-summary-like. Always simplify the *presentation*, never falsify the idea. Introduce terms as names for things already understood. Structural analogies (map the relationships, then hand over to the real mechanism, state limits when they matter). Prioritise ruthlessly: central insight first, details later in collapsibles. Natural sentences, rhythm, occasional humour, no forced enthusiasm or patronising reassurance. Prose should guide the widgets ("increase the learning rate to 0.9 and watch…") and widgets should give live, state-dependent explanations. See `STYLE_GUIDE.md` for examples.

## Mathematics

Mathematics serves understanding. Intuition before notation: problem → intuition → visualisation → experimentation → maths → implementation. Explain every symbol, subscript and convention; use small concrete numbers and worked arithmetic; keep plain English, visual, equation, numbers and code synchronised; make maths interactive (drag a vector, watch the dot product); explain *why* an operation was chosen and what breaks without it; build prerequisites just in time; use geometry heavily while flagging where 2D pictures mislead about high dimensions; support multiple depths (conceptual → computational → mathematical → deeper exploration); connect small operations to their scale in real systems (memory, compute, hardware). Assessments test what maths *means*, not formula recall.

## Curriculum (18 milestones)

1. **Mathematical foundations** — scalars/vectors/matrices/tensors, dot products, matrix multiplication, linear transformations, dimensionality, functions, derivatives, partials, chain rule, gradients, probability, statistics, logarithms, entropy. Draggable vectors, grid transformations, tensor shape/broadcasting explorer, tangent lines, computational graphs, manipulable distributions, high-dimensional geometry.
2. **The artificial neuron** — inputs, weights, bias, activations (sigmoid, tanh, ReLU, leaky ReLU, GELU, SiLU…) with derivatives; why non-linearity is necessary; neuron as decision boundary.
3. **Network architecture** — interactive playground: add/remove layers and neurons, edit weights, step through forward propagation, inspect tensors, parameter counts and complexity; clicking a neuron/connection reveals its contribution. Must reflect real calculations.
4. **Loss & optimisation** — MSE, MAE, BCE, multiclass CE; GD, SGD, mini-batch, momentum, RMSProp, Adam, AdamW, schedules; 2D/3D landscapes, trajectories, local minima/saddles, optimisers racing on the same problem.
5. **Backpropagation** (one of the most detailed areas) — forward, loss, local derivatives, chain rule, backward, accumulation, updates, autodiff; step-through with substituted numbers; learner computes gradients before reveal; vanishing/exploding gradients; network from scratch with code-to-visual mapping; how autodiff libraries work.
6. **Training laboratory** (TensorFlow-Playground-like) — datasets (linear, XOR, circles, spirals, clusters, regression, click-to-draw); configurable architecture, LR, activation, optimiser, batch size, noise, regularisation, data size, init; live boundaries, train/val loss, accuracy, activations, weight/gradient distributions; pause/step/reset; side-by-side comparisons; real computation.
7. **Generalisation & failure modes** — over/underfitting, bias–variance, train vs validation, leakage, imbalance, distribution shift, regularisation, dropout, early stopping, initialisation, normalisation, hyperparameter sensitivity; diagnostic challenges with broken models.
8. **Classical ML** — linear/logistic regression, kNN, trees, forests, boosting, SVMs, k-means, PCA, anomaly detection, feature engineering; compare boundaries, accuracy, cost; when classical beats neural.
9. **Computer vision** — pixels/channels, convolution, kernels, padding/stride, feature maps, pooling, receptive fields, CNNs, detection, IoU, NMS, segmentation, YOLO, ViTs; manipulate kernels on real images, animate convolution.
10. **Embeddings & representation learning** — one-hot vs dense, word embeddings, cosine/Euclidean, normalisation, latent spaces, autoencoders, contrastive learning, dimensionality reduction, vector DBs, nearest-neighbour search.
11. **Sequence models** — RNNs, hidden states, BPTT, LSTMs, GRUs, encoder–decoder, early attention; the bottlenecks that motivated transformers.
12. **Transformers & attention** (flagship) — tokenisation, BPE, embeddings, positional encoding, RoPE, Q/K/V, scaled dot-product attention, multi-head, causal masks, cross-attention, FFN, residuals, layer norm, blocks; follow a sequence through a small transformer; numerical walkthroughs; parallelism and sequence-length cost.
13. **LLMs** — autoregressive generation, pretraining, logits, softmax, temperature, top-k/top-p, greedy, beam search, context windows, perplexity, scaling laws, emergence, instruction following, hallucination, reasoning models; token generation simulator; real small models where practical.
14. **Training & fine-tuning** — pretraining, SFT, transfer, LoRA, QLoRA, RLHF, DPO, distillation, synthetic data, curriculum learning, catastrophic forgetting, distributed/data/model parallelism.
15. **Inference & hardware** — CPU vs GPU, tensor cores, FP32/FP16/BF16/INT8/INT4, quantisation, memory bandwidth, VRAM, compute- vs memory-bound, KV cache, prefill vs decode, batching, speculative decoding, offloading, MoE, local inference; calculators for memory, KV cache size, throughput/latency.
16. **Reinforcement learning** — MDPs, policies, value functions, exploration, Q-learning, DQN, policy gradients, actor-critic, PPO, reward shaping, reward hacking, credit assignment; trainable agents; poorly specified objectives.
17. **Generative AI & diffusion** — generative vs discriminative, AEs, VAEs, GANs, diffusion (forward/reverse, schedules, denoising), latent diffusion, conditioning, CFG, multimodal.
18. **AI systems engineering** — RAG, semantic search, vector DBs, chunking, reranking, tool calling, structured outputs, agents, planning, context management, memory, routing, multi-agent, evaluation, observability, reliability, latency/cost.

## Assessment system

Reusable and integrated: multiple choice, numerical, predict-the-output, predict-the-effect, find-the-wrong-operation, debug-a-model, match-graph-to-behaviour, order-the-steps, complete-the-algorithm, interactive challenges with measurable success. Explain the misconception behind wrong answers. Cumulative banks, difficulty progression, mastery tracking, review of missed concepts, spaced repetition, cross-topic and parameterised questions, practical exams.

## Experiment philosophy

Experiments are miniature laboratories: manipulable parameters, immediate feedback, real computation, inspectable intermediates, pause/step/reset, reproducible seeds, suggested experiments, edge cases, side-by-side comparisons, predict-before-reveal, explanations of observed behaviour. Prefer genuine simulations over scripted demos. **Never** fake simulations, non-functional controls, or visuals disconnected from real calculations; when a simplified model is used, say what was simplified and why.

## Visual & UX

A professional scientific workstation crossed with a premium educational product: persistent navigation, completion indicators, progress dashboard, searchable glossary, expandable maths, experimentation areas, dark/light themes, tooltips, cross-references, interactive diagrams, animated data flows, tensor/matrix inspectors, syntax-highlighted code, side-by-side comparisons. Colour communicates meaning (activation strength, gradient size, attention, sign, probability). Information density and legibility over decoration.

## Execution priorities

1. Mathematical and technical correctness. 2. Educational effectiveness. 3. Quality and clarity of prose. 4. Meaningful interactivity. 5. Visual clarity and polish. 6. Depth. 7. Breadth.

Develop through complete, working vertical slices. Keep architecture simple and lessons independent; shared abstractions only where reuse is genuine. Write focused tests for important maths and simulation behaviour, proportional to importance. Keep `ROADMAP.md` current (completed milestones, status, remaining content, decisions, limitations, ideas, continuation points). Work autonomously; after finishing a feature, move to the next meaningful educational enhancement. Spend effort on experiments, explanations, prose, challenges and visualisations rather than infrastructure. If capacity runs low, leave a working app and an up-to-date roadmap.
