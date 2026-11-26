<!-- authored-by: Gemini Antigravity -->
# Neural Lab — v2 Release Plan

`docs/CURRICULUM_PLAN.md` (WP0–WP9) is complete (20 modules, 75 lessons). This document specifies the next major release (**v2**): platform upgrades and three frontier AI lessons chosen by the owner, built to the exact same standards as the existing course (real computed numbers, finite-difference-checked gradients, fair multi-seed comparisons, `LabShell` + `AnswerReveal` + evidence-only hints, sourced `<Src>` history, light/dark/matrix + 390px verification).

---

## V2-WP1: Optional Sections & Guided Tracks

**Goal:** Let busy software engineers take a focused "Modern AI Fast Path" without removing the rich historical lessons that explain *why* modern architectures exist.

- **Data model (`src/curriculum.ts`):**
  - Add `optional?: boolean` and `optionalNote?: string` (short plain-English reason, e.g. *"Historical bridge — skip if you are racing straight to Transformers"*) on `LessonMeta`.
  - Candidate lessons marked `optional: true`:
    - The four era interludes (`interlude-first-winter`, `interlude-second-winter`, `interlude-big-bang`, `interlude-chatgpt`)
    - Historical/rarely-deployed-from-scratch bridges: `perceptron`, `svm`, `knn`, `rnns`, `lstms-grus`, `gans`
    - (Keep `seq2seq-attention`, `word-embeddings`, `autoencoders`, `decision-trees`, `linear-regression`, `clustering-pca`, and all Maths Toolkit / Core DL / Transformer / LLM / RL / Systems lessons as core.)
- **UI (`Home.tsx`, `Sidebar.tsx`, `LessonPage.tsx`):**
  - Add a clean track toggle persisted in `localStorage` (`nl-track-v1`): **Full Course (all lessons)** vs **Core Track (skip optional)** — default **Full Course**.
  - Render a subtle theme-aware pill (`Optional`) on optional lessons in `Home`, `Sidebar`, and `LessonPage` kicker (with tooltip/note explaining why it is optional and what it teaches).
  - In **Core Track** mode: collapse or mute optional lessons in `Sidebar` and `Home` (still clickable anytime!), skip optional lessons in Next/Prev footer navigation, and compute module/overall progress bars and total minutes against the selected track.
- **Tests:** Extend `src/curriculum.test.ts` to verify every `optional: true` lesson has an `optionalNote`, and test track-filtered `neighbours()` and progress calculation.

---

## V2-WP2: Interactive Concept Dependency Graph (`#/map`)

**Goal:** Give learners a visual map showing exactly which maths and foundational concepts feed into which modern AI technologies, complementing the linear curriculum (`Home`) and chronological view (`#/timeline`).

- **Data (`src/dependencies.ts` + `src/dependencies.test.ts`):**
  - Define a typed directed acyclic graph (DAG) of prerequisite/feeds-into edges between lesson IDs (seeded from `<Bridge>` links and module dependencies) plus named **technology destinations** (e.g. *Transformers & LLMs*, *Reasoning Models*, *Diffusion & Generative AI*, *RAG & Agents*, *Computer Vision & Multimodal*, *Reinforcement Learning*).
  - Tests pin: every node ID exists in `ALL_LESSONS`, graph is strictly acyclic (topological sort succeeds), every lesson has at least one edge (no orphans), and every `<Bridge to="…">` in Module 1 appears as an edge.
- **Page (`src/pages/DependencyGraphPage.tsx`, route `#/map` in `src/router.ts` & `src/App.tsx`):**
  - Add a `Map` link alongside `Timeline` in `Sidebar.tsx`.
  - **Interactive highlighting:**
    - Click any **technology destination** (or any lesson node) → highlights its full upstream prerequisite tree and dims unrelated nodes, with a readout showing how many prerequisites you have already completed (`nl-progress-v1`) and your next uncompleted prerequisite.
    - Toggle **"What does this unlock?" (downstream)** vs **"What does this need?" (upstream)** when inspecting a node.
    - Respects the V2-WP1 **Core / Full** track filter and all three themes (CSS variables only) + 390px phone layout (stacked collapsible/filterable view or horizontal pan container with zero page overflow).

---

## V2-WP3: Tiered Compute Offloading (Float32 Worker → WASM SIMD → Optional WebGPU)

**Goal:** Remove the browser main-thread bottleneck (~5k params / ~500 steps) while preserving Neural Lab's transparency, static deployment (`base: './'`), and 100% deterministic Node/Vitest tests.

### Why a 3-tier backend instead of WebGPU-only

WebGPU is fast, but unavailable in headless CI, older Safari/Firefox setups, and many mobile devices; meanwhile, moving plain JS `number[][]` loops off the main thread onto contiguous `Float32Array` buffers in a Web Worker already removes UI jank and delivers a 5–15× speedup everywhere. A unified `ComputeBackend` interface gives every learner smooth 60 fps interactivity and unlocks heavier experiments cleanly:

| Tier | Technology | Browser coverage | Speedup vs JS `number[][]` | Role |
| --- | --- | --- | --- | --- |
| **Tier 0** | Synchronous `Float32Array` TS kernels | 100% + Node/Vitest | ~5–10× | Reference implementation & deterministic Vitest testing |
| **Tier 1** | Dedicated **Web Worker** (`type: 'module'`) + `Transferable` `ArrayBuffer` | 100% of browsers | ~5–15× + **0 ms UI freeze** | Default interactive training, live heatmaps (`ImageData`), multi-seed sweeps, instant slider cancellation |
| **Tier 2** | **WASM SIMD (`wasm32-simd128`)** GEMM / activation kernel | ~95%+ (Chrome, Firefox, Safari, mobile) | ~20–60× | Heavier in-browser training (live grokking, mini character Transformer, MoE, ViT) on any CPU |
| **Tier 3** | **WebGPU (`navigator.gpu`)** WGSL tiled matmul + grid shader | Modern desktop GPU browsers | ~50–200× | Optional turbo path + live CPU-vs-GPU roofline benchmark inside Module 16 (`kv-cache`) |

### Architecture & deliverables

1. **`src/lib/compute/` module:**
   - `tensor.ts`: minimal `Tensor2D` (`Float32Array` + `rows`/`cols`) with `matmul`, `addBias`, `relu`/`tanh`/`gelu`, `softmax`, `layerNorm`, and exact backward passes — tested against central finite differences and `src/lib/linalg.ts` in `compute.test.ts`.
   - `workerBridge.ts` + `compute.worker.ts`: typed message protocol (`start`, `step`, `cancel`, `frame`) using Vite's native `?worker` / `new URL(..., import.meta.url)` bundling so `npm run build` stays 100% static. Stale jobs abort cleanly when the user drags a slider.
   - `wasmGemm.ts` (+ small checked-in/built SIMD `.wasm` kernel): drop-in `matmul` accelerator with automatic feature detection (`WebAssembly.validate` for `v128`) and silent fallback to Tier 1.
   - `webgpuBackend.ts`: optional WGSL compute shader for large matmuls and 2D heatmap grid evaluation, auto-detected via `navigator.gpu?.requestAdapter()` with silent fallback to Tier 2/1.
   - All tiers pass the exact same equivalence test suite (within `Float32` tolerance `1e-4`), and heavy widgets display a small subtle backend pill (e.g. `Worker · Float32` / `WASM SIMD` / `WebGPU`).
2. **Immediate upgrades unlocked across existing lessons:**
   - Offload `training-lab` heatmaps + training loop and `gans` 5-seed lab evaluation onto the Web Worker (zero UI stutter!).
   - Upgrade `lr-schedules` with a **real live warmup experiment** on a deeper/wider network where warmup demonstrably prevents early divergence (removing the current limitation in `docs/ROADMAP.md`).
   - Upgrade `open-questions` (`epilogue`) with a **live Grokking demo** (modular addition network memorising in ~200 steps and visibly grokking to 100% validation accuracy around step ~3,000–5,000 in ~2 seconds inside the worker!).
   - Optional live CPU-vs-WebGPU throughput mini-benchmark inside Module 16 (`inference`).

---

## V2-WP4: New Lesson — Mixture of Experts (MoE) & Sparse Routing (`moe`)

- **Placement:** Module 12 (`transformers`), right after `transformer-block` (before `tokenization`), since sparse MoE directly replaces the dense FFN inside a transformer block — with explicit forward links to Module 16 (`inference`, VRAM vs active FLOPs).
- **Plain summary (≤30 words, no symbols):** *"Why wake up every neuron for every word? How a router picks a few specialists per token, why one expert hogs all the work, and how to balance them."* (29 words)
- **Conceptual arc & widgets:**
  1. **Dense FFN vs Sparse Specialists:** analogy of a hospital triage desk routing patients to 2 of 8 specialists instead of waking the whole building.
  2. **Router & Top-$k$ Gating Explorer:** drag/click input tokens from 4 distinct clusters, watch router logits → softmax → top-$k$ weights blend expert outputs, and see why sparse routing decouples **total parameters (VRAM)** from **active parameters (FLOPs)** (`<Eq>` with colour-matched legend for $\sum_{i \in \text{Top-}k} g_i(x) E_i(x)$).
  3. **`<Predict>` + Live Load-Balancing Collapse:** train a real tiny 4-expert MoE (finite-diff checked backprop in `moe.ts`) with $\alpha = 0$ and watch the rich-get-richer feedback loop send >85% of tokens to one expert while the other three starve; then turn on the auxiliary load-balancing loss $\mathcal{L}_{\text{aux}} = \alpha E \sum_{i=1}^E f_i P_i$ (`<Eq>` with legend) and watch utilisation equalise.
  4. **VRAM vs Active-FLOPs Calculator:** compare dense vs sparse models (e.g. Mixtral-style 8×7B and fine-grained 64-expert models) on active FLOPs/token vs GPU memory footprint and all-to-all communication cost.
- **Lab: "Balance the Experts"** (`LabShell`, `AnswerReveal`, `MachineLog`, `LiveCode`, evidence-only hints + `hints.test.ts`):
  - Primitives: number of active experts `top-k` (`1, 2`), auxiliary balance weight `α`, capacity factor `C` (token dropping threshold), and router noise / temperature.
  - 3 briefs scored fairly across 5 fixed seeds (committed `search.run.ts` → `results.ts`, verified in `moe.test.ts`):
    1. Break single-expert collapse (max expert load ≤ 35% and test accuracy ≥ target);
    2. Zero dropped tokens under tight capacity factor while preserving accuracy;
    3. Hit the accuracy target with the fewest active FLOPs per token (`top-1` vs `top-2` trade-off).
- **History & glossary:** Sourced `<HistoryCard>` (Jacobs et al. 1991 adaptive mixtures → Shazeer et al. 2017 Sparsely-Gated MoE → Fedus et al. 2021 Switch Transformer, every claim verified on arXiv/Wikipedia); glossary terms `moe`, `router-collapse`, `load-balancing-loss`; verified PyTorch/NumPy snippet; 4-question quiz.

---

## V2-WP5: New Lesson — Reasoning Models & Test-Time Compute (`reasoning`)

- **Placement:** Module 15 (`finetuning`), right after `rlhf-dpo` (relying on RL from Module 14, sampling from Module 13, and scaling laws from Module 13).
- **Plain summary (≤30 words, no symbols):** *"Letting a model think longer before it answers: trading training size for inference compute, majority voting, verifiable rewards, and why noisy checkers backfire."* (24 words)
- **Conceptual arc & widgets:**
  1. **System 1 vs System 2 Compute:** why a fixed forward pass spends the same FLOPs on `"2 + 2"` and a proof; test-time scaling as a second scaling axis alongside pretraining (`<LessonLink to="scaling-laws">`).
  2. **Parallel Search: Pass@$k$, Majority Voting & Verifier Goodharting:** exact closed-form & simulated widget on a solvable toy reasoning task — watch Pass@$N$ climb with $N$, see where Majority Voting helps (and fails when the wrong answer is the mode!), and `<Predict>` what happens to Best-of-$N$ as $N$ grows when the reward model / verifier is only 85% accurate (accuracy rises, peaks, then *falls* as $N$ mines false positives — Goodhart's law at test time!).
  3. **Sequential Search: Chain-of-Thought & Verifiable-Reward RL (RLVR / GRPO):** explain why outcome-verifiable tasks (maths, unit-tested code) remove the fragile learned reward model; `<Eq>` for GRPO's group-relative advantage $\hat{A}_i = (r_i - \bar{r}) / \sigma_r$ (no separate critic network, halving policy-training memory vs PPO!); live tiny RLVR/GRPO toy trainer on a multi-step symbolic task showing accuracy and step length co-evolving.
  4. **Iso-Compute Trade-off Map:** compare a small model with test-time search against a 10× larger model with 1-shot decoding across easy vs hard problems.
- **Lab: "Spend the Test-Time Budget"** (`LabShell`, `AnswerReveal`, `MachineLog`, `LiveCode`, evidence-only hints + `hints.test.ts`):
  - Primitives: base model tier (`Small`, `Medium`, `Large`), decoding strategy (`1-shot`, `Majority Vote`, `Best-of-N`), sample count `N`, verifier tier (`None`, `Noisy 85% RM`, `Exact Unit-Test Verifier`), and CoT thinking-token budget.
  - 3 briefs on a deterministic solvable math/logic benchmark (exhaustive search over the full config grid pinned in `reasoning.test.ts`):
    1. **Tight latency budget:** beat the single-shot Large model's accuracy at lower total FLOPs using a Small/Medium model + test-time compute;
    2. **The noisy-verifier trap:** on unverified open-ended math with an 85% RM, find why cranking $N$ too high hurts accuracy and hit the sweet spot;
    3. **Hard multi-step problems:** under an exact verifier + fixed token budget, balance thinking depth vs parallel attempts for maximum pass rate (optional: fewest tokens).
- **History & glossary:** Sourced `<HistoryCard>` (Wei et al. 2022 Chain-of-Thought, Wang et al. 2022 Self-Consistency, Snell et al. 2024 Scaling LLM Test-Time Compute, Shao et al. 2024 GRPO / DeepSeek-R1 2025, every claim verified directly on arXiv); glossary terms `test-time-compute`, `grpo`, `pass-at-k`, `best-of-n`; verified Python snippet; 4-question quiz.

---

## V2-WP6: New Lesson — Multimodal Models & Vision Transformers (`multimodal-vit`)

- **Placement:** End of **Module 12 (`transformers`)** after `tokenization` (or a dedicated bridge before `llms`). *Dependency note:* ViTs require self-attention & positional encodings (Module 12) and CLIP requires contrastive learning (Module 10) and image tensors/CNNs (Module 9), so placing `multimodal-vit` after Module 12 guarantees **zero forward dependency inversions** while cleanly unifying Modules 9, 10, and 12!
- **Plain summary (≤30 words, no symbols):** *"Chopping an image into patch tokens so a transformer can read it, why vision transformers need more data than CNNs, and aligning pictures with text."* (26 words)
- **Conceptual arc & widgets:**
  1. **Patches as Tokens (`PatchifyWidget`):** slice a real 16×16 / 24×24 sprite grid into $P \times P$ patches, flatten and linearly project each patch into a token vector, add 2D positional embeddings, and compute the exact $N = (H/P)(W/P)$ sequence length and $O(N^2)$ attention-matrix explosion as $P$ shrinks (`<Eq>` with legend). Include a **"Shuffle patches" button** to show live that without positional embeddings a ViT is completely bag-of-patches blind to layout!
  2. **`<Predict>` + Inductive Bias Race (CNN vs ViT):** fair multi-seed race (powered by V2-WP3's Float32/Worker engine!) comparing a small CNN against a tiny Patch-Attention ViT on small vs larger shifted-shape datasets — honestly demonstrating the core ViT finding: CNNs' built-in translation bias wins decisively on small data, while ViT closes the gap as data grows.
  3. **Aligning Two Worlds — Mini-CLIP Workbench:** live dual-encoder contrastive trainer (`vit.ts`, finite-diff checked InfoNCE gradients) embedding tiny visual sprites and text descriptions into one shared unit-circle/2D space; click any unseen `(image, caption)` pair to inspect the $N \times N$ similarity matrix and perform live **zero-shot classification** by writing/picking text prompts!
- **Lab: "Wire the Multimodal Model"** (`LabShell`, `AnswerReveal`, `MachineLog`, `LiveCode`, evidence-only hints + `hints.test.ts`):
  - 3 briefs (exhaustive/multi-seed search pinned in `vit.test.ts`):
    1. **Patch budget:** choose patch size $P$ and embedding dim $d$ on a 32×32 image task to preserve small-object detail without blowing the $O(N^2)$ attention FLOPs cap;
    2. **Rescue the small-data vision model:** choose architecture (`CNN` vs `ViT`), positional embeddings (`None` vs `2D`), and augmentation on a small shifted dataset (proving why raw ViT without pos-enc or data fails and what rescues it);
    3. **Zero-shot CLIP alignment:** tune contrastive temperature $\tau$, projection dimension, and negative batch size to hit ≥90% zero-shot accuracy on held-out visual-text pairs.
- **History & glossary:** Sourced `<HistoryCard>` (Dosovitskiy et al. 2020 *An Image is Worth 16×16 Words* `2010.11929`, Radford et al. 2021 CLIP `2103.00020`, checked directly on arXiv); glossary terms `vision-transformer`, `patch-embedding`, `clip`, `multimodal`; verified PyTorch/NumPy snippet; 4-question quiz.

---

## Recommended implementation order

1. **V2-WP1** (Optional sections & tracks) — small, immediate UX win, informs V2-WP2's graph filters.
2. **V2-WP2** (Concept Dependency Graph `#/map`) — ties the entire 75-lesson course together visually and automatically incorporates the new v2 lessons as they land.
3. **V2-WP3** (Tiered Compute Offloading: Float32 Worker → WASM SIMD → optional WebGPU) — builds the fast engine *before* the three heavy frontier lessons so MoE, Reasoning/GRPO, and ViT/CLIP can run rich live training loops at 60 fps.
4. **V2-WP4** (`moe`), **V2-WP5** (`reasoning`), **V2-WP6** (`multimodal-vit`) — one full lesson + lab + sourced history + tests + screenshots per work package.
