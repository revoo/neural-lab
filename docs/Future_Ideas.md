<!-- authored-by: Claude Opus 5.5 -->
# Future ideas (post-v2, not scheduled)

Candidate topics and features for future releases, collected after the v2 plan was completed (November 2026). **Nothing here is approved.** Confirm with the owner before starting any of it. Two items from the original list are already **scheduled for V1** and specified in `HANDOFF.md`: the **tiny-GPT capstone** and the **"how a network learns to see / to speak" visual labs**. They are not repeated here.

Every idea must meet the usual standards: real computed numbers, hand-written gradients checked against finite differences, fair multi-seed comparisons, sourced history, all three themes, and 390px width.

## Suggested priority for a v3

1. Interpretability module
2. Training at scale
3. Faster inference

The rest are in no particular order.

---

## 1. Interpretability (a module of its own)

The epilogue mentions superposition, but no lesson teaches the tools.

- **Probing and the residual stream:** linear probes, and the "logit lens" (decode the residual stream after each layer to see what the model is predicting partway through).
- **Induction heads and in-context learning:** train a real two-layer attention-only model on sequences with repeats, then let the learner find the circuit: a previous-token head feeding a copying head. This also explains *why prompting works*, which the course doesn't cover today.
- **Sparse autoencoders:** pull readable features out of a superposed layer. This builds directly on the `high-dimensions` superposition lab.
- *Lab idea:* "Find the circuit": ablate heads to find the smallest set that keeps in-context copying working (exhaustive over head subsets).

## 2. Training at scale

The course stops at one GPU.

- Mixed precision and loss scaling (ties into `quantization`'s number formats).
- Recomputing activations to save memory (gradient checkpointing): the memory-versus-time trade.
- Splitting training across GPUs: data, tensor and pipeline parallelism, and ZeRO-style sharding of the optimiser state.
- FlashAttention as a memory-hierarchy story: tiling, and why reading memory is the bottleneck, not arithmetic.
- *Lab idea:* "Place the model": put a transformer on 8 simulated GPUs under memory and bandwidth limits, and minimise the time per step (exhaustive over a small placement space).

## 3. Faster and cheaper inference

- **Speculative decoding:** the acceptance probability is exact and computable, so it makes a good lab ("pick the draft model and draft length").
- Continuous batching and paged KV memory (follows `kv-cache`).
- Distillation and pruning.

## 4. Long context and alternatives to attention

- Linear attention and state-space models (Mamba): their memory cost against attention's, and what they give up on exact-recall tasks (a measured comparison, in the style of the "fair race" labs).
- Stretching RoPE for longer context: position interpolation (follows `positions`).

## 5. Data as a first-class topic

- Deduplication, quality filtering, and data mixtures.
- Test-set contamination (pairs with `evaluation`).
- Synthetic data and the "model collapse" debate, presented as contested, with sources for both sides.

## 6. Safety and security in practice

- Prompt injection against the existing tool-calling agent (`tool-calling`), jailbreaks, red-teaming.
- How `reward-hacking` and the Goodhart effects in RLHF and best-of-N generalise.
- *Lab idea:* defend an agent against injected instructions, scored on a fixed set of attack documents.

## 7. Search and planning

- Monte Carlo tree search and AlphaZero: a learned value function plus search at decision time. This bridges the RL module and the `reasoning` lesson's test-time compute.

## 8. Other modalities and learning signals

- Audio: waveforms, then spectrograms, then speech tokens (a short lesson).
- Self-supervised vision: masked autoencoders (MAE) and DINO-style distillation.
- Flow matching, as a sequel to `diffusion`.
- Graph neural networks (optional lesson).

## 9. Uncertainty and calibration in deep networks

- When to trust a network's confidence: reliability diagrams, temperature scaling, deep ensembles. This connects the probability toolkit to deployment.

---

## Platform features

- **Review prompts with spaced repetition,** drawn from existing quiz questions. Per the owner's rules: no points, no streaks, no badges.
- **"Open in a notebook":** export each lesson's verified Python snippet, plus a short continuation, as a runnable notebook.
- **Annotated paper walkthroughs:** for 10–15 landmark papers already cited in `src/history.ts`, show which lesson covers each figure and equation.
- **Accessibility:** keyboard control and screen-reader descriptions for the main widgets (most are mouse-driven today).
- **Offline / installable (PWA):** the site is static, so this is cheap.
- **A Web Worker for the CPU compute tier:** long CPU runs (the warmup demo takes about 50 s without a GPU) currently block the page. Recommended before or alongside the capstone.
