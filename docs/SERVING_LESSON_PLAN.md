<!-- authored-by: Claude Opus 5.5 -->
# Lesson plan: Serving LLMs (`llm-serving`)

**Status: approved by the owner (26 November 2026), not started.** This file is the full specification. The progress checklist is at the end: tick each step when its commit is pushed, and mark the step you are in the middle of **IN PROGRESS**, so another agent can resume after a disconnect.

This lesson takes the "Faster and cheaper inference" item from `docs/Future_Ideas.md` §3, except distillation and pruning, which stay unapproved. Read `CLAUDE.md`, `HANDOFF.md` (lesson checklist, definition of done, gotchas, owner's standing instructions) and `docs/STYLE_GUIDE.md` before starting. Then read the lesson this one follows, `src/lessons/kv-cache/index.mdx`, all the way through: this lesson must build on it without repeating it.

---

## 1. Why this lesson, and where it goes

**The gap.** `kv-cache` explains a single request well: prefill is limited by compute, decode is limited by memory bandwidth, the KV cache grows with context and batch size, and batching makes better use of the GPU. Then it stops. Continuous batching, PagedAttention and speculative decoding each get one bullet in a `<Deeper>`. Nothing in the course explains what happens when **many people use one model at once**, and that is the engineering behind every chat product.

**One question drives the whole lesson:** *your chatbot is quick for one user. On launch day 200 people arrive at once. What breaks, and how do real serving engines fix it?*

**Placement:**
- **Module 16, Inference & Hardware**, as its third lesson, after `kv-cache`: `quantization` → `kv-cache` → **`llm-serving`**.
- The module's era is currently `'2019 to 2023'`. Widen it to `'2019 to 2024'` only if a history card or sourced sentence dates something to 2024 (for example DistServe). Otherwise leave it.

**Wiring** (the HANDOFF lesson checklist, step 2):
- `src/curriculum.ts`: add after `kv-cache`.
  - `id: 'llm-serving'`
  - `title: 'Serving LLMs: Batching, Paging and Speculative Decoding'`
  - `minutes: 45`
  - Suggested `summary` (27 words, no symbols): *"Serving many users from one GPU: batching requests token by token, paging memory, and letting a small model draft for a big one. Lab: the serving desk."*
- `src/dependencies.ts`:
  - `PREREQS['llm-serving'] = ['kv-cache', 'sampling']`
  - `SHORT['llm-serving'] = 'Serving'`
  - add `'llm-serving'` to `dest:inference.needs`
- `src/areas.ts`: `LESSON_AREA['llm-serving'] = 'inference'`. `src/areas.test.ts` checks that it agrees with the graph.
- `kv-cache/index.mdx`: in its `<Deeper>`, change the PagedAttention and speculative-decoding bullets to point here with `<LessonLink to="llm-serving">`, and add a one-line bridge at the end of the lesson. Keep its numbers exactly as they are.
- `docs/ROADMAP.md`: update "Current state", the status table (Module 16 goes to 3 lessons; the heading count goes up by one), "Next steps" and "Labs built so far".

**Out of scope** (mention in one sentence at most, or in `<InTheWild>`):
- FlashAttention: give it a `<Deeper>` at most. It deserves its own lesson later.
- Tensor and pipeline parallelism across GPUs, and prefill/decode disaggregation: one `<InTheWild>` line.
- Distillation, pruning, and long-context or state-space models: not approved.

---

## 2. What the learner should come away with

1. Users feel two separate latencies. **Time to first token (TTFT)** is mostly prefill and queueing. **Time per output token (TPOT)** is decode. The operator pays for **throughput**. Improving one of the three usually costs another, and **goodput** (requests per second that meet *both* latency targets) is the number that matters.
2. **Static batching** wastes the GPU because a batch finishes only when its longest reply does. **Continuous (iteration-level) batching** lets requests join and leave at every step.
3. **A long prompt holds everyone up.** Prefilling a 16k-token document in one step stalls every user who is currently decoding. **Chunked prefill** caps the tokens processed per step and mixes prefill chunks in with decode tokens.
4. **Memory, not arithmetic, caps the batch size.** Reserving a full-length KV slab per request wastes most of it. **Paged KV memory** (PagedAttention) works like virtual memory: fixed-size blocks and a block table per sequence. Waste drops to less than one block per sequence, and identical prefixes (a shared system prompt) can be stored once (**prefix caching**).
5. Decode leaves the GPU's arithmetic mostly idle, and **speculative decoding** puts that idle arithmetic to work. A small model drafts k tokens and the big model checks them all in one pass. With the accept/reject rule, the output follows **exactly the big model's distribution**: it is lossless, not an approximation.
6. Every gain above shows up only in the memory-bound regime that `kv-cache` measured. The lesson keeps saying so, and shows one case where the trick does *not* help.

---

## 3. The lesson arc (section by section)

Follow the house arc: problem → analogy → hands-on widget (with `<Predict>`) → "Writing it down" with `<Eq>` → code → `<Misconception>` → `<Quiz>` → `<Takeaways>` → `<InTheWild>` → bridge. Target length: **2,800–3,300 words**. `tiny-gpt` is 3,301 words and is the upper bound.

**Split point, if needed:** if the draft goes clearly beyond 3,300 words, split at §3.6. Make that section, its widget and Lab 2 a separate lesson `speculative-decoding`, placed right after this one (prerequisites `['llm-serving']`). Record the split in the roadmap. Don't squeeze the prose to avoid splitting.

### 3.1 Opening: launch day (problem)

- Open with the question, not a definition. Recall the number from `kv-cache`: one user on an H100 gets the roofline ceiling of 205.2 tok/s with Llama-3 8B FP16 at 2k context. Quote it via `<LessonLink>`; don't re-derive it.
- `<Predict>`: "200 people send a message within the same second. You process them one at a time, in arrival order, each at full single-user speed. Roughly how long does the 200th person wait for their *first* token?" Options span "under a second", "a few seconds" and "minutes". Compute the answer with the lesson's engine on its fixed trace and pin it in a test. Don't estimate it.
- The reveal names the three numbers the lesson is about: TTFT, TPOT and throughput. Introduce each with `<Term>`.

**Widget A, `LatencyAnatomy`:** one request drawn as a timeline: queue wait, then prefill (one wide block), then decode (a row of thin ticks).
- Sliders: prompt length and reply length.
- It shows TTFT and TPOT from the engine's step-cost model (§4.1).
- Live note, for example: "TTFT is 92% queueing: the GPU is busy with someone else."
- This is where the first `<Eq>` belongs (§3.8).

### 3.2 The bus that waits for its last passenger (static vs continuous batching)

- **Analogy (structural):**
  - Static batching is a minibus that leaves the depot full and won't take anyone new until the passenger going furthest has got off. Seats empty out stop by stop and stay empty.
  - Continuous batching is a bus that lets people off and on at every stop.
  - Map it explicitly: a seat is a batch slot, a stop is one decode step, the journey is the reply length.
  - State the limit: unlike a bus, a new passenger must first be "prefilled", which takes a whole step of its own. That limit is the opening for §3.3.

**Widget B, `BatchingRace`:** the same fixed 24-request trace served by static batching and continuous batching, as two Gantt charts one above the other.
- Each row is a batch slot, each column is an engine step, and the colour shows which request is in the slot (prefill vs decode shaded differently; empty slots hatched).
- Readouts: throughput, TTFT p50/p95, TPOT p50/p95, and the share of slot-steps left idle.
- Scrubbable step cursor through `ReplayControls`.
- **Fairness rule:** static batching gets its own best batch size and wait timeout, found by grid search over a stated space by a committed script. Pin the result in a test, and say so in the caption ("static batching at its best settings, found by searching …").

`<Predict>` before the reveal: "Continuous batching does the same total arithmetic. Where does its gain come from?" The answer is in idle slots and queue time, not in faster maths.

### 3.3 One trolley holds up the whole queue (chunked prefill)

- Problem the learner feels: with continuous batching on, a request with a 16k-token prompt arrives mid-trace. Its prefill step is long, and **every user already decoding gets no token during that step**. The TPOT trace shows one big spike.
- Analogy: one customer with a full trolley at the only checkout. Split the trolley: put through a few items between every other customer's basket.
- Mechanism: a per-step **token budget**. Every step, decode tokens go first, then whatever budget is left goes to a chunk of a pending prefill. Chunks attend to the KV the earlier chunks already wrote, so the result is exactly the same as an unchunked prefill. Say this, and note that it is the same causal-mask argument as in `kv-cache` §1.

**Widget C, `ChunkedPrefill`:** one Gantt and a TPOT-over-time trace.
- Control: token budget (off / 256 / 512 / 1,024 / 2,048 / 4,096).
- It shows the trade: smaller chunks flatten the TPOT spike but stretch the long request's own TTFT, because each chunk re-reads the weights.
- Live note states the current worst TPOT and the long request's TTFT.
- **Honesty:** in the roofline model, a small chunk added to a decode step is almost free until the step crosses the ridge point. Let the numbers show where that knee falls for the chosen GPU. Don't assert it in advance.

### 3.4 Malloc for the KV cache (paged memory)

- Problem: the lesson's GPU holds N full-context KV slabs. Most replies are short, so most reserved memory is never written. The batch-size ceiling, which `kv-cache` showed caps throughput, is set by memory that sits empty.
- Analogy for this audience: **virtual memory**. Software engineers know page tables. Reserving `max_len` per request is like `malloc` of the worst case for every string. Paging hands out fixed-size blocks on demand, and a per-sequence block table maps logical token positions to physical blocks.
  - State the limit: here the "page fault" is planned. The scheduler allocates a block when a sequence crosses a block boundary. If no block is free it must **preempt** a sequence: pause it and free its blocks (recompute or swap). Show preemption in the lab's log.
- **Prefix caching:** 200 requests that share a 1,000-token system prompt can point at the same physical blocks. Reference counts, plus copy-on-write when the shared block is only partly full. This saves memory *and* prefill work.

**Widget D, `KvMemoryMap`:** the GPU's KV region drawn as a grid of blocks, coloured by owning request.
- Toggle: contiguous max-length reservation / contiguous reservation of the actual prompt that grows by reallocating (show the fragmentation) / paged, with block size 8 / 16 / 32 tokens / paged + prefix sharing.
- Readouts: sequences resident, bytes reserved vs bytes used, and the waste broken down as internal (the unfilled tail of each last block) vs reserved-but-unused.
- `<Predict>`: "With paging, what limits the batch now?" The answer is actual tokens resident, so throughput now rises until decode leaves the memory-bound regime. Link back to the roofline.
- **Honesty:** in our model, smaller blocks always waste less memory. Say plainly that real engines pick 16-ish blocks for *kernel* efficiency (memory access patterns, table overhead), which our roofline model does not represent. Don't invent a cost to make block size 16 "win".

### 3.5 Lab 1: The Serving Desk (see §5.1)

### 3.6 The junior drafts, the senior signs off (speculative decoding)

- Problem: after §3.4 a single user's decode is still memory-bound. Every token re-reads all the weights to do about 1 FLOP per byte, and `kv-cache` showed that more than 99% of the arithmetic sits idle. Checking 5 tokens in one pass costs about the same bytes as checking 1.
- Analogy: a senior editor whose time goes mainly on walking to the filing room (reading the weights), not on reading. A junior drafts five words; the senior walks over once and signs off on all five in the same trip, accepting a prefix and correcting the first wrong word. State the limit: the senior doesn't just mark words "wrong"; the replacement is drawn from a specific correction distribution, and *that* is what keeps it lossless.
- **Real models, no stand-in.** Use the two trained character-level models already committed in `src/lib/compute/`:
  - **draft:** `tinyGpt.cpu.bin`, `CPU_RUN.cfg` (d 32, 2 blocks, 29,696 parameters);
  - **target:** `tinyGpt.gpu.bin`, `GPU_RUN.cfg` (d 96, 3 blocks, 348,384 parameters).

  Both share the Grimm vocabulary. Introduce them as "two small storytellers trained on Grimm's fairy tales; you will build one yourself in the capstone", with `<LessonLink to="tiny-gpt">`.
  - Use `Decoder` from `src/lib/compute/gpt.ts`. Its KV cache makes rejection cheap: rolling back is just setting `len` back, because later entries get overwritten. That is a nice concrete point for the prose: "rejecting three drafted tokens is one assignment."
  - Check first (a test) that `Decoder` logits match `Gpt.nextLogits` after a rollback.
  - The context is only 64 characters. Keep prompt plus continuation ≤ 64, or re-prefill a sliding window and say so.
  - Also offer a **zero-cost draft**: a character bigram table counted from the training corpus. It is an honest rival, because "cheaper but dumber" is the real trade.

**Widget E, `SpeculativeStepper`:** step through rounds on a fixed prompt.
- Each round shows the k drafted characters with q(x) (draft probability) and p(x) (target probability), the accept test `u < p/q`, the first rejection, the residual distribution `max(0, p − q)` renormalised as a small bar chart, and the corrected character.
- A running tally: characters emitted per target pass.
- Controls: draft model (bigram / small GPT), k (1–8), temperature (0, i.e. greedy, or 0.7 / 1.0), seed.
- **Greedy check shown live:** at temperature 0 the speculative output string equals the target's own greedy string character for character. Pin it in a test.
- **Sampling check:** pin it in a test. Exactly: for one round, enumerate the probability of the emitted first character, sum over the draft's choices, and check it equals p(x) to 1e-12. Also run a seeded Monte Carlo over the full procedure, with a stated tolerance.

**The honest twist (keep it in the prose):**
- On the learner's CPU these models are tiny and not memory-bound, so verifying k tokens costs about k times one token. Speculation therefore makes **wall-clock time worse** here. Measure that in the widget (a small timing readout), show it, and explain that it is the roofline again: the trick helps only when one pass is limited by bytes, not FLOPs.
- The modelled speed-up uses the step-cost model in §4.1 with a stated (draft, target) pair at real-model scale and the **measured** acceptance rate. Label that clearly: "if a pair of production-sized models accepted drafts as often as ours do…".
- Don't claim the tiny pair's acceptance rate carries over to production pairs. Say it is the mechanism that carries over, not the number.

### 3.7 Lab 2: Tune the draft (see §5.2)

### 3.8 Writing it down (`<Eq>` blocks)

Put each next to its section, not all in one place. Every symbol gets a legend entry, colour-matched with `\ca \cb \cc \cg \cp \cv` (never `\cr`). Prose words in `words=` go inside `\text{…}`. The legend goes in MDX children.
1. **Request latency:** latency = TTFT + (output tokens − 1) × TPOT.
2. **Engine step time** (the roofline from `kv-cache`, now for a mixed batch): time = max(FLOPs ÷ peak, bytes ÷ bandwidth), where FLOPs ≈ 2N × (tokens processed this step) plus the attention term, and bytes ≈ weight bytes + Σ over running sequences of the KV bytes read. One sentence connecting it to `kv-cache`'s `Eq`.
3. **Paging waste:** contiguous waste per sequence = (max_len − actual length) × KV bytes per token; paged waste per sequence < block size × KV bytes per token.
4. **Speculative accept rule:** accept drafted x with probability min(1, p(x)/q(x)); on rejection draw from norm(max(0, p − q)). The full proof that the output is distributed as p goes in a `<MathExhibit>`: it is short and pleasing, but optional.
5. **Expected tokens per target pass:** E = (1 − α^{k+1}) / (1 − α), and modelled speed-up = E / (k·c + 1), where c is the draft-to-target cost ratio. **Honesty check to include:** the formula assumes every token is accepted with the same probability α. Compare its prediction with the measured tokens per pass over the lesson's fixed prompts, and report the gap. Acceptance is high mid-word and low at word starts, and the stepper can show that.

### 3.9 In code

- **Python/NumPy, run and verified** (HANDOFF rule: run every snippet; paste its real output):
  - (a) the speculative-sampling accept/reject step in about 15 lines, plus a Monte Carlo check that the output histogram matches p;
  - (b) a continuous-batching scheduler loop in about 25 lines: a waiting queue, a running set, a token budget, admit, step, retire. Plain loops before any library.
- **`LiveCode` / `realCode.ts`** (the lab-improvement pattern), from the learner's lab settings:
  - turn them into a vLLM launch line: chunked prefill and `--max-num-batched-tokens`, `--max-num-seqs`, `--block-size`, `--enable-prefix-caching`, and speculative decoding settings;
  - **verify every flag name against the current vLLM documentation before writing it.** They have changed between versions; the speculative decoding config in particular has moved. If a flag can't be verified, leave it out;
  - `realCode.test.ts` pins the mapping.

### 3.10 Misconceptions (`<Misconception>`, 2–3 of them)

- "Speculative decoding is an approximation that trades quality for speed." No: with the accept/reject rule it is exact. The greedy-match test and the distribution test show it.
- "A bigger batch is always better." Throughput rises, but TPOT rises too, and past the ridge point decode stops being nearly free. Memory runs out first in most real deployments.
- "Paging makes attention faster." It makes *more sequences fit*. The throughput gain comes from the larger batch, and the attention kernel does slightly more work to follow the block table.
- Optional `<Deeper>` if sourced: batching can change results in the last bits, because the order of a floating-point reduction can depend on batch size, so "the same prompt" can give different text under load. Include it only with a verified `<Src>`; otherwise skip it.

### 3.11 Quiz, takeaways, in the wild, bridge

- `quiz.ts`, 6–8 questions, mostly conceptual. Wrong-answer feedback explains the misconception. Numeric ones are parameterised, for example: "draft accepts with α = …, k = …; expected tokens per pass?".
- `<Takeaways>`: the six points of §2, one line each.
- `<InTheWild>`:
  - vLLM, SGLang and TensorRT-LLM all implement continuous batching, paged KV, prefix caching, chunked prefill and speculative decoding;
  - prefill/decode disaggregation (separate GPU pools) is the next step;
  - multi-token heads (Medusa-style) and self-speculation are variants.
  - Every product claim needs a `<Src>` checked on the source page.
- Bridge: a one-line `<LessonLink>` to the next lesson in curriculum order.

---

## 4. The engine (`src/lessons/llm-serving/serve.ts` + `serve.test.ts`)

Self-contained: **don't import from `kv-cache/`**; duplicate the few model and GPU constants needed. A test pins that the duplicated constants reproduce `kv-cache`'s quoted headline, 205.2 tok/s for Llama-3 8B FP16, H100, B = 1, 2k context, so the two lessons can't contradict each other.

### 4.1 Step-cost model

- `stepTime(batch)`. Inputs: the decode sequences (each with its current context length) and any prefill chunks (chunk length, and the context already cached for that request). Output: `max(flops / peak, bytes / bandwidth)`.
  - FLOPs: 2N per processed token, plus attention: about 4 · layers · d_model · context per token, since QKᵀ and AV are 2 FLOPs per multiply-add each. Derive it in a comment and test it.
  - Bytes: weights once per step, plus every running sequence's KV read, plus newly written KV.
- **No invented overheads.** If a fixed per-step overhead is included at all, make it a named constant that defaults to 0 and is stated in the caption. Real engines run at 70–85% of the ceiling, as `kv-cache`'s `<Deeper>` says: point back to it.
- **GPU and model.** For the lab, choose a configuration where memory actually binds. The recommended one is Llama-3 8B FP16 on the **RTX 4090 (24 GiB)** from `kv-cache`'s table: 14.96 GiB of weights leaves a tight KV budget. Copy its spec exactly. Widgets A–D may use the same setup for continuity. Reserve a stated runtime scratch allowance, if `kv-cache` uses one; match it.

### 4.2 Workload trace

- `trace.ts`: a seeded generator (`src/lib/rng.ts`, seed fixed before looking at any results, and recorded in a comment) and the committed trace it produces.
  - Arrivals: Poisson at a stated rate.
  - Prompt lengths: mostly short chat, a stated fraction of long documents (8k–16k).
  - Reply lengths: a heavy-tailed distribution. State the distribution and its parameters; don't claim they match any real service unless a source is cited.
  - Optional: all requests share a 1,000-token system prompt (for prefix caching).
- A test pins the trace's summary statistics, so regenerating it can't silently change the lab.

### 4.3 Scheduler simulator

- `simulate(trace, config) → { steps[], requests[], metrics }`. It is deterministic, discrete-event, one engine step per iteration, and the same code drives every widget and both labs.
- `config`:
  - `batching`: `'static' | 'continuous'`; for static, also `batchSize` and `maxWaitMs`
  - `maxRunning`
  - `tokenBudget` (`null` = no chunking)
  - `allocator`: `'reserve-max' | 'reserve-grow' | 'paged'`
  - `blockSize`
  - `prefixSharing`
  - `admission`: `'fcfs'`, plus optionally one more stated policy; keep it small
- Preemption when paged memory runs out: evict the most recently admitted sequence, free its blocks, re-queue it, recompute on resume. Use that one stated policy and log every preemption to `MachineLog`.
- `metrics`: throughput (output tok/s), TTFT p50/p95, TPOT p50/p95, SLO attainment, **goodput**, peak KV used vs reserved, preemptions.
- Tests:
  - **Conservation:** every request's output tokens are produced exactly once, and blocks allocated = blocks freed at the end.
  - **Equivalence:** chunked prefill produces the same KV contents. Check this on the tiny GPT with a real `Decoder`: prefill in chunks vs all at once, logits identical. The simulator itself only counts, so this test is what makes "chunking is exact" a checked claim.
  - **Fairness search for static batching** (§3.2): pin the best-found settings and their metrics.
  - **Every number quoted in the prose**, pinned (HANDOFF rule).

### 4.4 Speculative engine (`spec.ts` + `spec.test.ts`)

- `specRound(target: Decoder, draft: Drafter, k, temperature, rng)` returns `{ drafted, accepted, emitted, targetPasses }`. `Drafter` is either a GPT `Decoder` or the bigram table, behind one interface.
- Rollback via `Decoder.len`. Add a small `truncate(len)` helper to `gpt.ts` only if setting `len` directly turns out to be unsafe. If you change `src/lib/compute/gpt.ts`, run `gpt.test.ts`, `gptReference.test.ts` and the parity scripts.
- **Exactness tests:** greedy string equality; exact first-token distribution equality to 1e-12; seeded Monte Carlo with a stated tolerance.
- **Measured acceptance:** a committed script `spec.run.ts` measures acceptance over a **fixed set of prompts chosen before measuring** (say, 12 tale openings from `taleStarts`, fixed seeds) for every (drafter, k, temperature) in the lab's grid. It writes `specResults.ts`, and `spec.test.ts` re-verifies a sample of entries. The browser re-runs live and compares with a tolerance, for the Node-vs-browser `Math.exp` reason in HANDOFF.
- **Honesty:** if the small-GPT draft turns out worse than the bigram draft at some setting once its cost is counted, or speculation never pays at temperature 1, the prose says so.

---

## 5. Labs

Both labs use `LabShell` + `useLabProgress` (`localStorage`), `AnswerReveal` (answers only after a solve or a remembered "I'm stuck"), evidence-only hints tested in `hints.test.ts`, `MachineLog`, `<LabJob>` above, `<ForReal>` after, `LiveCode`, and a sourced `<HistoryCard>`. Each brief asks for a **working solution first**, with an **optional** optimisation goal after. No badges or points.

### 5.1 Lab 1: The Serving Desk

**`<LabJob>`:** "You are on call for launch day. One GPU, one model, a recorded morning of real-shaped traffic. Configure the serving engine so users get their first token quickly and a steady stream after that."

**Primitives:**
- the `config` knobs of §4.3, as `Segmented` controls with options limited to a finite grid (so the solution space can be counted);
- a **Run the morning** button.

**Observable machinery:**
- the step Gantt (click a step to see its batch: which requests, prefill chunk vs decode, FLOPs, bytes, whether it was compute- or memory-bound);
- the KV block map at the scrubbed step;
- a per-request table (click a request to highlight its path through the Gantt);
- latency histograms with the SLO lines drawn on;
- `MachineLog` for admissions, retirements and preemptions.

**Briefs:**
1. **"Nobody waits forever."** A chat-only trace. Objective: SLO attainment ≥ a stated target with TTFT ≤ X and TPOT ≤ Y. Static batching, even at its best searched settings, should fail. **Verify that it does before writing the brief.** If it doesn't, change the trace or the thresholds openly and record why in the roadmap. Optional: maximise goodput.
2. **"The long document."** The same trace plus long-document requests. Continuous batching without chunking breaks the TPOT target. Objective: meet the SLO. Optional: maximise goodput.
3. **"Fit the crowd."** A busier trace with a shared system prompt on the memory-tight GPU. Objective: goodput ≥ a stated requests/s. The intended routes are paging and/or prefix sharing, but count what actually works. Optional: the fewest preemptions at that goodput, or the maximum goodput.

**Solution space:**
- Exhaustive over the full config grid per brief. Keep the grid small enough to run (aim for a few hundred configs × 3 briefs).
- A committed `search.run.ts` (sharded if slow: `node … search.run.ts <shard> <shards>`, as in HANDOFF) writes `results.ts`: the count of passing configs per brief, the optimum (with ties listed) and the Pareto front of TTFT p95 vs throughput.
- `search.test.ts` re-runs the optimum and a few random configs and checks them.
- The debrief celebrates the variety ("23 of the 360 settings meet the SLO; here's how they differ"). Use the real count.

**Post-solve map:** a scatter of every config, throughput (x) against TTFT p95 (y), coloured pass/fail, with the learner's choice highlighted. The trade-off frontier is the takeaway.

**Hints** must not name locked options (test). Hint ladder, for example: "Look at the Gantt: what fraction of slot-steps is hatched?" → "What happens to a slot when its request finishes early?"

### 5.2 Lab 2: Tune the draft

**`<LabJob>`:** "You serve one user at a time on a memory-bound GPU and want replies faster without changing a single output token. You have a big model, a small one and a bigram table."

**Primitives:** drafter (bigram / small GPT), k (1–8), temperature (0 / 0.7 / 1.0, set by the brief), cost ratio c (shown, derived from the stated production-scale pair in §3.6; the learner doesn't pick it).

**Machinery:** the `SpeculativeStepper` rounds, the measured acceptance per position (mid-word vs word start), a measured-vs-formula E chart over k, and the modelled speed-up.

**Briefs:**
1. **Greedy (temperature 0):** reach a modelled speed-up ≥ a stated target while the output stays identical to the target's greedy output (checked live). Optional: find the best (drafter, k).
2. **Sampling (temperature 1):** acceptance falls. Reach any modelled speed-up > 1, or declare it impossible. If it is impossible for this pair, that is allowed and good: STYLE_GUIDE §5 on impossible puzzles. Verify the claim exhaustively in a test and give the learner a "declare impossible" button. Optional: the best setting.
3. **Optional brief:** the wall-clock readout on *your* machine. Can any setting beat plain decoding in real time on this CPU? Report what was measured. The expected answer is "no, and here's why" (§3.6), but **measure first and write whatever is true.**

**Optimum:** exhaustive over drafter × k × temperature (at most 48 cells), from `specResults.ts`, re-verified by test.

### History card (`src/history.ts`)

One card per lab, or one shared card. Candidates; check every date, venue and author on the source page itself (paper PDF or proceedings page) before writing:
- Orca and iteration-level scheduling (OSDI 2022)
- vLLM and PagedAttention (SOSP 2023)
- speculative decoding, two independent groups: Leviathan, Kalman & Matias (arXiv 2022, ICML 2023); Chen et al., DeepMind (arXiv 2023)
- Sarathi and chunked prefill (2023)

The mood of the moment: ChatGPT's launch made serving cost a front-page problem almost overnight. Link `interlude-chatgpt` if it fits.

---

## 6. Glossary (`src/glossary.ts`)

New terms, each with `lesson: 'llm-serving'` and sensible `related` links (`kv-cache`, `memory-bound`, `gqa`):
- `ttft`
- `tpot`
- `goodput`
- `continuous-batching`
- `chunked-prefill`
- `paged-attention` (aka PagedAttention, paged KV cache)
- `prefix-caching`
- `preemption`
- `speculative-decoding`
- `acceptance-rate`

Introduce each with `<Term>` on first mention.

---

## 7. Checks (from HANDOFF; keep them proportionate)

- Before every commit: `npm test` and `npm run build`.
- Once the lesson is finished: `node scripts/check-pages.cjs` and `node scripts/check-pages.cjs --theme=matrix`. Run both again after any change to a shared component, `global.css` or `curriculum.ts`.
- Every new widget: `node scripts/shoot-widgets.cjs llm-serving` in light, dark and matrix, and at 390px. Use `drive-widget.cjs` to exercise Lab 1 runs, Gantt clicks and the stepper.
  - Gantt and block-map grids must not overflow at 390px: scroll them horizontally inside the widget, using HANDOFF's phone-width grid gotcha.
  - Colours only through CSS variables.
  - Respect `prefers-reduced-motion` in replay animations.
- Watermarks: `{/* authored-by: <MODEL_NAME> */}` at the top of `.mdx`, `// authored-by: <MODEL_NAME>` at the top of `.ts`/`.tsx`, the `Authored-by:` commit trailer, and `rg "<MODEL_NAME>" dist/` returning 0 after the build.

---

## 8. Steps and checklist

One commit per step, pushed to `claude/cool-feynman-4hci4d`. Small related steps may share a commit.

- [x] **S1 Engine:** `serve.ts` (step cost, trace, simulator) with `serve.test.ts`. Includes the constants-consistency test against 205.2 tok/s, the conservation tests, and the static-batching fairness search script and its pinned result (`staticSearch.run.ts` → `staticBest.ts`).
- [x] **S2 Speculative engine:** `spec.ts`, `trainDraft.run.ts` → `draftGpt.bin` (5,424-param small GPT drafting for `tinyGpt.cpu.bin`), bigram drafter, rollback check, the exactness tests, `spec.run.ts` → `specResults.ts`, re-verification test in `spec.test.ts`. Chunked-prefill equivalence test on the real tiny GPT.
- [x] **S3 Lab 1 search:** `lab.ts`, `search.run.ts` → `results.ts` (counts, optimum, Pareto front, 56-config static sweep) + `search.test.ts` + `hints.test.ts`. Thresholds chosen after measuring (Brief 1: 90/300, Brief 2: 32/300, Brief 3: 18/300).
- [x] **S4 Widgets A–E** + Lab 1 (`ServingDeskLab.tsx`) + Lab 2 (`TuneDraftLab.tsx`), hints and `hints.test.ts`, `realCode.ts` + `realCode.test.ts` (browser screenshots in all 3 themes & 390px run right after S5 wires the route).
- [x] **S5 Prose:** Split cleanly at §3.6 into `llm-serving` (3,181 raw words, Lab: The Serving Desk) and `speculative-decoding` (2,079 raw words, Lab: Tune the Draft), with `index.mdx`, `quiz.ts`, 10 glossary entries, 2 verified history cards, wiring (`curriculum`, `dependencies`, `areas`), and `kv-cache` links/bridge. Every quoted number pinned; Python snippets run with real outputs pasted.
- [ ] **S6 Wrap-up (IN PROGRESS — Gemini Antigravity):** full checks, both `check-pages` runs, widget screenshots across all 3 themes & 390px, watermark check; update `docs/ROADMAP.md` (Current state, status table, Next steps, Labs built so far), `HANDOFF.md` §0, and this checklist; push.

## 9. Risks and decisions left to the implementer (record what you decide in the roadmap)

- **Thresholds and traces are not fixed here on purpose.** Pick them after measuring, so each brief has several valid solutions and a clear failing route, and write the measured numbers into the prose. Never tune a trace to make a technique look better than it is. If a technique doesn't help in our model, say so (STYLE_GUIDE §5).
- **Lesson length:** see the split point at the top of §3.
- **The roofline model is an upper bound,** not a benchmark. Every caption that shows a time or throughput says "roofline model of <GPU>", as `kv-cache` does.
- **vLLM flag names:** verify them, or leave them out (§3.9).
