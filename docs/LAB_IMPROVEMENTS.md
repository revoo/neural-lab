# Neural Lab — lab improvement suggestions

_Review written 25 November 2026. It covers the 25 labs that existed then (Modules 1–7 and 12–16); labs added later (Modules 8–10, 17, 18) were not reviewed, though the cross-cutting ideas apply to them too. Nothing in it has been implemented yet. Live, editable version: https://claude.ai/artifact/LgGJiBKRH3VkRNvxFuKmCD_

**For an implementing agent:** follow `HANDOFF.md`, `CLAUDE.md`/`AGENTS.md` and `docs/STYLE_GUIDE.md` as usual. Work through "Suggested order of work" at the end of this file, one step per commit. Every number or date a lab quotes must be re-run or re-checked and pinned by a test. Places marked † below still need a source.

## Summary

The labs are worth refining, and the work is uneven: the early puzzle labs (Modules 1–7 and 12) already meet the style guide, while most of the later tuning labs (Modules 13–16) are "pick settings from a small grid and watch three tiles turn green". I opened all 25 labs in the running app and read their code. Every suggestion below keeps the vision's rule: the reward is understanding, never points.

The later labs fall short on two of the project's own lab rules. **Observable machinery**: several show only scores, not the thing being trained or sampled. **Try first**: five labs put the full answer one click away before the learner has tried anything.

A second pass looked at motion, real-world goals and logs. Only 5 of 25 labs move at all, 16 open with no real-world job, and none keeps a running log of what the machine did. The "Motion, real-world goals and inside the machine" section below the lab table gives the details and a per-lab plan.

A third pass proposes history for every lab: when, where, who, and the mood of the moment, with every date checked against a source page. Checking caught real errors in what I first remembered: the backprop dates, the Mark I Perceptron's year, and a TD-Gammon date the sources don't support.

This report is about the labs themselves. Site-wide mechanics (calibration, share cards, hint ladders, a collection log) are already covered in `gamification-proposal.md`, and I only point to them where a lab is a natural first home.

### Top 5, in order

1. **Hide the answer until it's earned.** "Show exhaustive optimum" (LoRA, alignment, VRAM Tetris), "Show exhaustive summary" (Rescue the deep network) and "Inspect π\*" (Wire the Policy) work before any attempt. Show them only after the brief is solved, or after a deliberate "I'm stuck" step. *Stabilise the policy gradient* already does this right (`entry.solved && exhaustive`). Effort: small.
2. **Fix two hints that point at locked options.** *Rescue the deep network* Level 1 says "switch on Residual skip", and residual is locked there (`vanishing-gradients/Widgets.tsx:791`). *Tune the alignment pipeline* Brief 1 says "switch to DPO", and the method is locked to RLHF (`rlhf-dpo/Widgets.tsx:607`). Effort: tiny.
3. **Put the machinery on screen in the tuning labs.** *Tune the Q-Learner* never shows the maze, the agent or a learning curve. *Tune the sampler* never shows a single generated word. *Spend the compute budget* has two sliders and no map. In each case the widget earlier in the same lesson already draws the right picture, so the lab can reuse it. Effort: medium.
4. **Draw each neuron's decision line in the Logic workbench.** Showing the four input corners with each neuron's line makes "XOR needs two lines" visible instead of found by +/− trial. It also links straight back to the neuron lesson's boundary widget. Effort: small to medium.
5. **Show the solution space once a puzzle is solved.** Many labs already enumerate every setting, so the map is free. Two standouts: *Attention puzzles* as an 11×11 pass/fail grid of queries, which draws the convex-hull reason the impossible puzzle is impossible; and *Build a previous-word head* as a strip of all 180 speeds, which shows the periodic pattern. This is the lab-level version of idea 2 in the gamification proposal. Effort: small per lab.

## Lab-by-lab review

Of the 25 labs, 9 need only polish, 12 would gain from one upgrade, and 4 need the most work. The table follows curriculum order, and each row gives the one change I'd make first.

| Lab (module) | Verdict | Best improvement | Effort |
| --- | --- | --- | --- |
| The Dice Workshop (1) | Strong: polish only | Add a **Roll** button that throws the recipe 1,000 times and grows a histogram towards the exact bars, so the exact distribution and what you'd actually see meet. | Small |
| Superposition (1) | Good: one upgrade | In Brief 1 (3D), draw the four ±1 arrows inside a cube you can rotate: they turn out to be its diagonals. The later briefs keep the dot-product grid. | Medium |
| Logic workbench (2) | Good: one upgrade | Draw each neuron's line over the four input corners, and let learners drag the line as well as press +/−. XOR's "one line can't do it" becomes visible. | Medium |
| Prune it (3) | Strong: polish only | Once solved, show which neurons the exhaustive minimum keeps beside the learner's set, so they can compare *which* neurons survive, not just how many. | Small |
| Descent golf (4) | Good: one upgrade | Overlay ghost paths for your best run and, once you've matched par, the par run. Racing your own ghost is cheap fun and shows why fewer zig-zags win. | Small |
| Optimizer golf (4) | Good: one upgrade | The same ghosts, plus a post-solve view with every optimizer's best run on the banana together. | Small |
| Design a schedule (4) | Strong: polish only | Draw the 20 seeded runs faintly behind the average, and after a solve, overlay the best cosine schedule as a ghost. | Small |
| Training-lab challenges (6) | Strong: polish only | After a challenge, show the reference network's neuron maps next to yours. | Small |
| Write the backward pass (5) | Strong: polish only | A collection view of the 24 correct versions, and a call-out when the learner finds one of the two wrong-but-still-trains options (a natural home for idea 3 of the gamification proposal). | Small |
| Rescue the deep network (5) | Good: one upgrade | Fix the Level 1 hint that suggests the locked residual skip, and show the exhaustive summary only after a solve. | Small |
| The model clinic (7) | Strong: polish only | Add a few two-cause cases where fixing one cause isn't enough. The proposal's "Sabotage" game belongs here too. | Medium |
| Attention puzzles (12) | Strong: polish only | After a solve, show an 11×11 map of which queries pass. For the impossible puzzle the map draws the convex-hull argument. | Small |
| Build a previous-word head (12) | Good: one upgrade | After a solve, show a strip of all 180 speeds marked pass/fail, and draw each position's clock hand under the bars. | Small |
| Compress the corpus (12) | Good: one upgrade | Show a sentence of the corpus re-segmenting live as you merge. Also check the 1,806-bit "optimum": it covers greedy checkpoints only, while hand mode allows any merges. Relabel it, or make "beat greedy BPE" a goal if hand merges can win. | Medium |
| Shannon's betting game (13) | Good: one upgrade | At the end of a sentence, show a strip of bits lost per word for you, the neural LM and Unigram, so the learner sees exactly where they lost. | Small |
| Tune the sampler (13) | Needs work | Show 10 live continuations per prompt under the current settings, with tail junk highlighted, so the three numbers describe text the learner can read. | Medium |
| Spend the compute budget (13) | Needs work | Plot (N, D) as a draggable point on a loss contour map with the budget line and the optimum, reusing the IsoFLOP widget instead of two blind sliders. | Medium |
| LoRA rank budget (14) | Good: one upgrade | Show the optimum only after a solve, and draw each layer's captured vs. missed shift as two small heatmaps beside the singular-value bars. | Small |
| Tune the alignment pipeline (14) | Needs work | Fix the Brief 1 hint that suggests the locked DPO, and show the optimum only after a solve. Brief 1 has 7 choices: put β as a dot on the lesson's Goodhart curve and ask for a prediction before the first click. | Medium |
| Mixed-precision quantisation (15) | Good: one upgrade | Show the selected block's weights on a number line with the format's levels, reusing the outlier widget. That replaces the hint that names the worst block. | Medium |
| VRAM Tetris (15) | Good: one upgrade | Show the optimum only after a solve, and put the config on the lesson's roofline so the tok/s figure has a visible cause. | Medium |
| Wire the Policy (16) | Strong: polish only | Show π\* only after a solve. Add "Run 20 episodes", which animates sampled paths including slips, so an 18.5% pit risk is something you watch happen. | Medium |
| Tune the Q-Learner (16) | Needs work | Show the maze with one seed's greedy policy and all 10 learning curves. With only 36 settings, ask for a prediction before each Train. | Medium |
| Stabilise the policy gradient (16) | Good: one upgrade | Replace the row of per-seed numbers with 20 faint lines under the average, so "a few lucky seeds" can be seen at a glance. | Small |
| Reward-Hacking Sandbox (16) | Strong: polish only | Animate the plan step by step with a running reward counter and the pad's recharge timer, and show what your *intended* plan would have earned next to the hack. | Medium |

## Motion, real-world goals and inside the machine

The labs are mostly static, mostly abstract, and log nothing. All three gaps can be closed with a few shared pieces rather than 25 rewrites. I counted animation code (`requestAnimationFrame`, timers, CSS transitions) in each lab and read every brief.

- **Motion.** Only 5 of 25 labs move. Three animate the process itself: the playground and training lab train live, and descent golf draws its path step by step. VRAM Tetris and Wire the Policy slide a bar for 120–160 ms. The other 20 snap from one still frame to the next. None of the lab animations respects `prefers-reduced-motion`; only the matrix theme's decorations do.
- **Real-world goal.** 9 of 25 open with a real engineering job: model clinic, VRAM Tetris, compute budget, quantisation, reward hacking, sampler, backward pass, LoRA and alignment. The other 16 are abstract ("Four arrows in 3D", "Plain 16-layer stack", "Fast Maze Convergence") or a game story with no link to AI work (dice for a board game).
- **Inside the machine.** Every lab has a status line, and a few show real internals: the backward pass prints NumPy shape errors, the clinic has instrument panels, the playground lists every tensor at the probe point. No lab keeps a running log of what happened, step by step.

### What good looks like

1. **Motion that carries meaning.** Animate the process, never decoration: a path drawn step by step, probability mass flowing as a slider moves, two tokens sliding together, an agent walking its plan. State changes ease over 150–400 ms so the eye sees which way things moved. Processes get play, pause and step. One shared hook switches all of it to instant under reduced motion.
2. **A real job in the first line.** Each brief opens with the situation a practitioner would be in ("You're on call: a training job has sat at loss 0.693, a coin-flip, since step 1"). The debrief adds a "where this happens for real" line. For an audience of software engineers, the strongest link is **the real line of code the current settings correspond to**, updated live: `model.generate(temperature=0.7, top_p=0.9)`, `LoraConfig(r=2)`, `DPOConfig(beta=0.1)`, llama.cpp's `Q4_K_M`, vLLM's `--kv-cache-dtype fp8`. Check each against the library's current docs before shipping.
3. **Three layers of inside-the-machine.** First, a picture of the mechanism. Second, live numbers placed on that picture. Third, a **machine log**: a collapsible monospace console with one plain-English line per event, e.g. `merge 1: a+t → at · 531 → 504 tokens · table +9.0 bits · total 2,368 → 2,289 bits`. Engineers debug from logs, so this is their native way to see a system work, and it doubles as a record of what the learner tried. One shared `MachineLog` component covers every lab.

### Per lab: real-world goal, motion and inside the machine

Each row gives a one-line real-world job for the brief and the motion, visuals and log lines that would show the machine at work.

| Lab | Motion now | Real-world goal | Motion and inside the machine to add |
| --- | --- | --- | --- |
| The Dice Workshop | None | Keep the game briefs as warm-ups, then add "initialise a layer": choose weights so a sum of 256 random inputs comes out with SD ≈ 1. He initialisation is exactly this sum. | A Roll button that tumbles dice into a growing histogram. Log: `+d6: mean 10.5 → 14.0, variance 8.75 → 11.67`. |
| Superposition | None | "Store 10 features (*is code*, *is French*, *is a question*…) in 6 neurons": how real LLMs pack more features than they have neurons. | A rotatable cube for Brief 1. The clashing pair flashes in the dot-product grid when a cell flips. Log: `row 3, entry 2 flipped: now clashes with row 1`. |
| Logic workbench | None | XOR is the two-way light switch at the top and bottom of the stairs. AND is an alarm that sounds when the door is open *and* the system is armed. | The decision line sweeps across the four corners as weights change. Clicking a truth-table row sends a pulse along the wires. Log: `A=1, B=1: 1×1 + 1×1 − 1.5 = 0.5 → fires`. |
| Prune it | Live process | Ship this classifier to a smartwatch, where every neuron costs memory and battery. | A pruned neuron's map fades out while the boundary morphs. Log: `neuron 2 off: accuracy 97% → 94% ✗`. |
| Descent golf | Live process | Each step is a full pass over a large dataset, so fewer steps means a smaller compute bill. | Ghosts of your best run and the par run. Step log with position, gradient and step length, which shows the side-to-side bounce as numbers. |
| Optimizer golf | None | Choose the optimizer line for a real training script: show `torch.optim.Adam(params, lr=…, betas=(…))` live for the current settings. | Animate the path, which is currently drawn all at once. Draw a momentum arrow on the ball. Per-step log of step size and velocity. |
| Design a schedule | None | A run with a fixed budget of 400 steps. Real runs use warm-up then decay, and the knots are a hand-drawn version of that schedule. | A marker moves along the schedule and the loss curve together, with the 20 seeded runs drawn faintly behind. |
| Training-lab challenges | Live process | Circle: the right engineered feature beats a bigger model, the way "distance from home" helps a fraud model more than raw coordinates. | Already live. Add an event log that flags moments worth noticing, e.g. `step 300: test loss up 3 checks in a row`. |
| Write the backward pass | None | You're writing what `loss.backward()` does for you. Gradient checking is how framework authors test it (`torch.autograd.gradcheck`). | Each filled line lights up as its gradient flows back through the shapes. Check log per weight: `W1[0,2]: yours …, measured … ✓`. |
| Rescue the deep network | None | An on-call incident: "a 32-layer job has sat at loss 0.693 since step 1". This is why every transformer block has LayerNorm and a residual. | A pulse travels layer by layer through both RMS charts. Log per layer: `L12: gradient RMS 0.003, vanishing`. |
| The model clinic | None | Already real: complaints from engineers about their runs. | A `train.log` tail per case (step, loss, gradient norm, step size), which is the first evidence a real engineer reads. |
| Attention puzzles | None | Pronoun resolution: make "it" look at the cat, not the mat, as a translator must to pick the right word in French. | A small worked table per token: q·k → ÷√2 → exp → share. Bars glide as the query is dragged. |
| Build a previous-word head | None | Previous-token heads are real circuits found in trained language models, and part of how they copy and learn from context (induction heads). | Clock hands turn for each position as the slider moves. Log of the score at each offset. |
| Compress the corpus | None | Choose a new model's vocabulary size: more tokens shorten inputs (cheaper attention) but grow the embedding table. The tokenizer behind GPT-3.5 and GPT-4 has 100,258 tokens. | In a sample sentence, two tokens slide together on each merge. Log: `merge 1: a+t → at · 531 → 504 tokens · table +9.0 bits`. |
| Shannon's betting game | None | You are the suggestion bar on a phone keyboard, and perplexity is how surprised your users are. | The word flips in while a bits counter drains. Per-word log of you, the neural LM and Unigram, plus the end-of-sentence strip. |
| Tune the sampler | None | One setting must serve a support bot stating facts ("The capital of France is…") and a brainstorming tool that should vary ("I opened the box and found a…"). Show `generate(temperature=…, top_k=…, top_p=…)` live. | Probability bars reshape smoothly and cut tokens drop away. 10 live draws per prompt, with junk highlighted. |
| Spend the compute budget | None | Already real: research run, edge model, billion-user chatbot. | A point moving on the loss contour map with the budget line. Log: `N 150B, D 300B → C = 6ND = 2.7 × 10²³`. |
| LoRA rank budget | None | Fine-tune an 8B model on your company's documents on one GPU. Show the `peft` `LoraConfig` with per-layer ranks. | Each layer's leftover-error heatmap shrinks as its rank grows. Log: `layer 2: r 1 → 2, MSE … → …`. |
| Tune the alignment pipeline | None | Ship a chatbot update when the reward model loves flattery and bullet spam. | Bars for the engine's 8 named replies, from "Crisp Rayleigh explanation" to "Sycophantic bullet-spam", shifting as β changes. A training log of reward and KL. |
| Mixed-precision quantisation | None | Fit an 8B model on a laptop: show the file size implied by the average bits per weight, and the nearest llama.cpp preset name. | Each block's weights snap onto the chosen format's grid. Per-block error log. |
| VRAM Tetris | Some | Already real. Add the equivalent vLLM launch flags. | Weights, KV cache and runtime drop into the memory bar like Tetris pieces, which makes the name literal. A dot on the roofline. |
| Wire the Policy | Some | Route a warehouse robot across a wet floor, where a slip into the pit is a dropped parcel. | "Run 20 episodes" with animated paths and slips. Cells flash by how much their value changed after each arrow click. |
| Tune the Q-Learner | None | A robot vacuum learning a new flat in 25 practice runs. The cliff is the stairs, and the shifting goal is a moved charging dock. | A maze replay with the agent and Q-values filling in. A TD-update log in the same format as the lesson's stepper. |
| Stabilise the policy gradient | None | Choose which of three page layouts to show from noisy click rewards, like an online recommender. | A replay of one seed with logit bars nudging on each press, and 20 faint lines. Log: press, choice, reward, advantage, push. |
| Reward-Hacking Sandbox | None | Already real: the boat-race loop, a robot that breaks the vase, a cleaner that blinds its camera. | A boat replay with a reward ledger (`step 4: pad fires, +3`), and the intended plan's total shown beside the hack's. |

## History: time, place, people and the mood of the moment

Every concept and lab should carry its history: when it appeared, where, who made it, and what the field believed at the time. Today history shows up only as the odd aside: 20 of the 45 lessons mention no year at all, including backpropagation.

This fits the project's core rule, "let the learner feel the problem first", because history is a record of people feeling that problem. Several labs already replay a historical struggle without saying so. Told at the moment the learner has just hit the same wall, a date becomes "people were stuck here for years, and you just found the way out".

### The card

Each lesson and lab gets one short card, collapsed by default and under about 120 words, with five parts:

1. **When and where.** Year, institution and city, e.g. "1958 · Cornell Aeronautical Laboratory, Buffalo".
2. **Who.** The people, each linked to a source.
3. **The mood of the moment.** One sourced line on what the field believed then, e.g. how neural networks were regarded during an AI winter. This is what makes the date feel alive.
4. **Before → this → after.** What people did and what broke, what this made possible (one concrete number where there's a good one), and what built on it.
5. **Sources.** One link per claim.

### Where it goes

- **After the lab, not before.** The learner feels the problem, solves it, then learns who else did and when.
- **A site-wide timeline page** running from 1943 to the 2020s. It has a band for each era showing the mood of the moment, a lane of milestones, and each lesson placed on it, with a marker for where the learner has got to. It shows the whole arc of the field, and it is not a badge. A mock-up follows the table below.

### Caveats

- **Popular history is often wrong.** "Minsky and Papert killed neural networks" is a simplification, and credit for backprop is disputed. A card names such disputes plainly instead of picking a hero.
- **The mood of the moment is the easiest part to get wrong.** Each mood line needs a source: a contemporary account, or a reputable history looking back. "Out of favour, with funding cut" can be sourced. "Seen as a joke" needs a quote from someone who was there.
- **Dates are numbers.** Keep them in one data file (e.g. `src/history.ts`) with a source URL for each claim, and pin them with a test, the same way the project already handles numbers.
- **The idea, not the legend.** Tell the problem and the idea that solved it, not the myth around a person.

### Proposed history card per lab

Every date, place and quote below was checked on the linked page on 25 November 2026, and each card's sources sit in its own row. A dagger (†) marks a place I know but couldn't confirm on a source page; it needs a source before it ships. Where no sourced mood line exists, the cell says so rather than inventing one.

| Lab | When | Where and who | Mood of the moment | Before → this → after |
| --- | --- | --- | --- | --- |
| Logic workbench | 1943 · 1958 · 1969 | [McCulloch](https://en.wikipedia.org/wiki/Warren_Sturgis_McCulloch) (University of Illinois at Chicago) and Pitts (University of Chicago). [Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt), Cornell Aeronautical Laboratory, Buffalo. [Minsky and Papert](<https://en.wikipedia.org/wiki/Perceptrons_(book)>). | 1956: the [Dartmouth workshop](https://en.wikipedia.org/wiki/Dartmouth_workshop) names the field "artificial intelligence". 1958: the New York Times runs "New Navy Device Learns By Doing". In the 1970s, funding for neural-network work "was difficult to find" ([AI winter](https://en.wikipedia.org/wiki/AI_winter)). | A [threshold neuron](https://en.wikipedia.org/wiki/Artificial_neuron) can do AND and OR but not XOR → the [perceptron](https://en.wikipedia.org/wiki/Perceptron) learns its own weights → *Perceptrons* (1969) proves the single-layer limits. Wikipedia says "there are many mistakes" in the story that the book killed the field. |
| Write the backward pass | 1986 | Rumelhart, [Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton) (then a postdoc at UC San Diego) and Williams, "Learning representations by back-propagating errors", *Nature*. | The neural-network winter "came to an end in the middle 1980s" ([AI winter](https://en.wikipedia.org/wiki/AI_winter)). | Credit is genuinely shared: the [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) page lists Linnainmaa (reverse-mode differentiation, published 1976), Werbos (1971 PhD work) and many others → the 1986 paper makes it the standard way to train multilayer networks → every framework's `backward()`. |
| Training-lab challenges | 1988–1991 | George Cybenko; Hornik, Stinchcombe and White ([universal approximation](https://en.wikipedia.org/wiki/Universal_approximation_theorem)). | 1987: the LISP machine market collapses, and the second AI winter runs to about 2000 ([AI winter](https://en.wikipedia.org/wiki/AI_winter)). | Proof that a network *can* represent almost any function → the open question is whether training *finds* it, which is exactly the XOR challenge's lesson: representable ≠ learnable. |
| Rescue the deep network | 1991 → 2010 → 2015 | [Hochreiter](https://en.wikipedia.org/wiki/Sepp_Hochreiter), diploma thesis, Technical University of Munich. [Glorot and Bengio](https://proceedings.mlr.press/v9/glorot10a.html) (Bengio: Université de Montréal). He, Zhang, Ren and Sun (Microsoft Research†). | 1990s: AI researchers "deliberately called their work by other names", avoiding the term "for fear of being viewed as wild-eyed dreamers" ([history of AI](https://en.wikipedia.org/wiki/History_of_artificial_intelligence)). | Hochreiter identifies the [vanishing gradient](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) → new initialisations ([2010](https://proceedings.mlr.press/v9/glorot10a.html), [He 2015](https://arxiv.org/abs/1502.01852)) → [ResNet](https://arxiv.org/abs/1512.03385) trains 152 layers and wins ILSVRC 2015 at 3.57% error → [LayerNorm](https://arxiv.org/abs/1607.06450) (2016). |
| Descent golf | 1847 | [Augustin-Louis Cauchy](https://en.wikipedia.org/wiki/Gradient_descent). | Before computers: every step was done by hand. | A rule for walking downhill on a function → the same rule, nearly 180 years later, behind every training run in these labs. |
| Optimizer golf | 1960 → 2014 | [Howard Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function). [Kingma and Ba](https://arxiv.org/abs/1412.6980), *Adam* (ICLR 2015). | After AlexNet's 2012 win, "investment and interest in AI boomed" ([history of AI](https://en.wikipedia.org/wiki/History_of_artificial_intelligence)). | A test valley where "to find the valley is trivial" but converging is hard → adaptive optimizers → Adam is the baseline the [LoRA paper](https://arxiv.org/abs/2106.09685) compares against. |
| Design a schedule | 2016 | [Loshchilov and Hutter, *SGDR*](https://arxiv.org/abs/1608.03983). | The post-2012 boom, as above. | A fixed step size → schedules that change it over training, including warm restarts. |
| Prune it | 2015 → 2018 | [Han, Mao and Dally, *Deep Compression*](https://arxiv.org/abs/1510.00149) (Stanford†). [Frankle and Carbin, *Lottery Ticket*](https://arxiv.org/abs/1803.03635) (MIT†). | The post-2012 boom: networks winning, but too big to ship. | AlexNet shrinks 35×, from 240 MB to 6.9 MB, with no loss of accuracy → "winning tickets" under 10–20% of the original size on MNIST and CIFAR-10. |
| The model clinic | 2019 | [Andrej Karpathy, "A Recipe for Training Neural Networks"](http://karpathy.github.io/2019/04/25/recipe/). | Training has become an engineering craft, and the post's warning is "neural net training fails silently". | Folklore passed between colleagues → a written diagnostic recipe → the clinic's ten cases. |
| Attention puzzles | 2014 → 2017 | [Bahdanau, Cho and Bengio](https://arxiv.org/abs/1409.0473) (Université de Montréal). [Vaswani et al.](https://arxiv.org/abs/1706.03762), "[eight scientists and engineers working at Google](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need)". | 2017: researchers at Google Brain introduce the transformer ([history of AI](https://en.wikipedia.org/wiki/History_of_artificial_intelligence)). | Squeezing a whole sentence into one fixed-length vector, which Bahdanau et al. conjecture "is a bottleneck" → a soft search over source words → the transformer, "dispensing with recurrence and convolutions entirely" (28.4 BLEU English→German). |
| Build a previous-word head | 2021 → 2022 | [Su et al., RoFormer](https://arxiv.org/abs/2104.09864). [Olsson et al.](https://arxiv.org/abs/2209.11895) (Anthropic†). | No sourced mood line yet. | Rotary positions encode "absolute position with a rotation matrix" with "explicit relative position dependency" → induction heads, which "might constitute the mechanism for the majority of all in-context learning" (the authors call their evidence preliminary). |
| Compress the corpus | 1994 → 2015 | [Philip Gage](https://en.wikipedia.org/wiki/Byte-pair_encoding). [Sennrich, Haddow and Birch](https://arxiv.org/abs/1508.07909) (ACL 2016; Edinburgh†). | No sourced mood line yet. | A file-compression trick → subword units for "open-vocabulary translation" → byte-level BPE in GPT tokenizers (GPT-3.5 and GPT-4: 100,258 tokens). |
| Shannon's betting game | 1948 → 1951 | [Claude Shannon](https://en.wikipedia.org/wiki/Claude_Shannon), Bell Labs. | 1956: Shannon is one of the four organisers of the [Dartmouth workshop](https://en.wikipedia.org/wiki/Dartmouth_workshop). | [Information theory](https://en.wikipedia.org/wiki/A_Mathematical_Theory_of_Communication) (1948) → ["Prediction and Entropy of Printed English"](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf): people guess the next letter, and long-range context brings English down to "the order of one bit per letter" → next-token prediction. |
| Tune the sampler | 2019 | [Holtzman et al.](https://arxiv.org/abs/1904.09751) (University of Washington†). | No sourced mood line yet. | Using likelihood as the decoding goal "leads to text that is bland and strangely repetitive" → nucleus (top-p) sampling. |
| Spend the compute budget | 2020 → 2022 → 2024 | [Kaplan et al.](https://arxiv.org/abs/2001.08361) (OpenAI†). [Hoffmann et al.](https://arxiv.org/abs/2203.15556), [DeepMind](<https://en.wikipedia.org/wiki/Chinchilla_(language_model)>). [Meta AI](<https://en.wikipedia.org/wiki/Llama_(language_model)>). | The race to size: Chinchilla's rivals ran up to 530B parameters (Megatron-Turing NLG). | Loss follows power laws, some spanning "more than seven orders of magnitude" → Chinchilla (70B parameters, 1.4T tokens) beats Gopher (280B) and GPT-3 (175B) → Llama 3 (April 2024) trains on about 15 trillion tokens. |
| LoRA rank budget | 2019 → 2021 → 2023 | [Houlsby et al.](https://arxiv.org/abs/1902.00751). [Hu et al.](https://arxiv.org/abs/2106.09685) (Microsoft†). [Dettmers et al.](https://arxiv.org/abs/2305.14314). | GPT-3 (175B) is the giant everyone wants to adapt. | Adapters reach within 0.4% of full fine-tuning with 3.6% extra parameters → LoRA cuts GPT-3's trainable parameters 10,000× and GPU memory 3× → QLoRA fine-tunes 65B on one 48 GB GPU. |
| Tune the alignment pipeline | 2017 → 2022 → 2023 | [Christiano et al.](https://arxiv.org/abs/1706.03741). [Ouyang et al.](https://arxiv.org/abs/2203.02155) (OpenAI†). [Rafailov et al.](https://arxiv.org/abs/2305.18290) (Stanford†). | [Goodhart's law](https://en.wikipedia.org/wiki/Goodhart%27s_law) (1975, in Strathern's 1997 words): "When a measure becomes a target, it ceases to be a good measure." | Learning from human preferences between pairs, with feedback on under 1% of interactions → the 1.3B InstructGPT is preferred to the 175B GPT-3 → DPO replaces "complex and often unstable" RLHF with "a simple classification loss". |
| Mixed-precision quantisation | 2022 → 2023 | [Dettmers et al., LLM.int8()](https://arxiv.org/abs/2208.07339). [Frantar et al., GPTQ](https://arxiv.org/abs/2210.17323). Iwan Kawrakow, [llama.cpp k-quants](https://github.com/ggml-org/llama.cpp/pull/1684). | No sourced mood line yet. | 8-bit inference up to 175B "without any performance degradation", once emergent outliers are handled → 175B models at 3–4 bits in about four GPU hours → `Q4_K_M` merged into llama.cpp on 5 June 2023. |
| VRAM Tetris | 2019 → 2023 | [Noam Shazeer, multi-query attention](https://arxiv.org/abs/1911.02150). [Ainslie et al., GQA](https://arxiv.org/abs/2305.13245). [Kwon et al., vLLM](https://arxiv.org/abs/2309.06180) (UC Berkeley†). | No sourced mood line yet. | Sharing keys and values across heads cuts "memory bandwidth requirements of incremental decoding" → GQA keeps quality "close to multi-head" at MQA-like speed → PagedAttention gives 2–4× throughput over FasterTransformer and Orca. |
| Wire the Policy | 1953 → 1957 | [Richard Bellman](https://en.wikipedia.org/wiki/Richard_Bellman), RAND Corporation. | Bellman's RAND years, from 1949. | Dynamic programming (1953) → the book *Dynamic Programming* (1957, [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation)) → the value backups every RL method uses. |
| Tune the Q-Learner | 1988 → 1989 → 2013/2015 | Sutton ([TD learning](https://en.wikipedia.org/wiki/Temporal_difference_learning), 1988). Chris Watkins ([Q-learning](https://en.wikipedia.org/wiki/Q-learning)). [Mnih et al.](https://arxiv.org/abs/1312.5602), [DeepMind](https://en.wikipedia.org/wiki/Google_DeepMind), London. | 1989 sits inside the second AI winter (1987–2000). In 2014 Google buys DeepMind. | Q-learning (1989; convergence proof with Dayan, 1992) → TD-Gammon plays backgammon "at the level of expert human players" → Atari from raw pixels (2013) → 2015: "comparable to a professional human games tester" on 49 games. |
| Stabilise the policy gradient | 1992 → 2017 | Ronald J. Williams, [REINFORCE](https://en.wikipedia.org/wiki/Policy_gradient_method). [Schulman et al., PPO](https://arxiv.org/abs/1707.06347) (OpenAI†). | The second AI winter (1987–2000). | "The first policy gradient method" → baselines to cut variance → PPO's objective "enables multiple epochs of minibatch updates". |
| Reward-Hacking Sandbox | 1975 · 1999 · 2016 | [Goodhart](https://en.wikipedia.org/wiki/Goodhart%27s_law). [Ng, Harada and Russell](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf), UC Berkeley. [OpenAI](https://openai.com/index/faulty-reward-functions/). | Goodhart's law began as a 1975 point about UK monetary policy. | Potential-based shaping provably keeps the optimal policy, and "well-known bugs" come from shaping that isn't → 21 December 2016: the CoastRunners boat circles a lagoon knocking over targets and scores 20% higher than human players. |
| Superposition | 2022 | [Elhage et al., *Toy Models of Superposition*](https://arxiv.org/abs/2209.10652) (Anthropic†). | No sourced mood line yet. | One neuron per concept → models storing "additional sparse features in superposition" → interpretability research that tries to pull them apart. |
| The Dice Workshop | 18th–19th century | The [de Moivre–Laplace theorem](https://en.wikipedia.org/wiki/Central_limit_theorem), "the earliest version" of the central limit theorem. | Before computers. | Why sums of random parts form a bell → why He initialisation scales weights the way it does, and why benchmark gaps need error bars. |

### Mock-up: the timeline page

_(The drawn mock-up is in the live version linked at the top: a horizontal time axis from 1940 to 2026; tinted era bands for the AI winters of 1974–1980 and 1987–2000 and the post-2012 boom; 13 milestones from the history table, each labelled with its year and lab — 1943 threshold neuron, 1948 information theory, 1956 Dartmouth, 1958 perceptron, 1969 Perceptrons, 1986 backprop, 1989 Q-learning, 1991 vanishing gradient, 2012 AlexNet, 2015 ResNet, 2017 Transformer, 2022 Chinchilla, 2023 DPO; and an accent marker for where the learner has got to.)_

The real page would place every lesson, not this selection. Clicking an era band opens its mood line, e.g. the 1990s, when researchers avoided the term "AI" "for fear of being viewed as wild-eyed dreamers" ([history of AI](https://en.wikipedia.org/wiki/History_of_artificial_intelligence)). Clicking a milestone opens its card.

## Cross-cutting ideas

Five patterns would improve many labs at once. The first makes the other four cheap.

- **A shared lab frame.** Each of the 25 labs builds its own brief tabs, solved state, hint, reveal and `localStorage` handling (`src/components/` has `Widget` but nothing lab-specific). So the labs drift apart: some show "0 / 3 solved", some don't, and five reveal the answer early. A `LabShell` holding briefs, solved ticks, a status line, a hint and a post-solve reveal slot would make the rules below the default. Labs can move over to it one at a time.
- **Hints point at evidence, never at a setting.** A good hint says where to look ("which layer still has tall orange bars?"), not what to choose ("upgrade Attn QKV"). It should never name a locked control. A small test could check this: each level lists its locked controls, and its hint strings must not mention them.
- **The answer comes after a solve.** Optimums, π\* and exhaustive summaries appear only after the brief is passed, or after an explicit "I'm stuck, show me" step that the lab remembers. *Stabilise the policy gradient* and *Superposition* already work this way.
- **Predict before you press.** Labs with a single run button (Q-Learner, policy gradient, reward hacking, model clinic, backward pass) can ask "will this pass?" before running, then show the guess next to the result. That turns a 7- or 36-setting grid from click-through into reasoning. The same data feeds the calibration page in the gamification proposal.
- **Ghosts and replays.** One small animation helper (play, pause, step, and a faint previous run behind the current one) covers golf paths, MDP episodes, the reward-hacking boat and Q-learning rollouts. Watching an agent exploit your reward is the most memorable moment these labs can offer.

A solution-space map after each solve is the sixth pattern, already in the gamification proposal (idea 2). Attention puzzles and Build a previous-word head are the best places to try it first.

## Suggested order of work

Start with the hour of fixes, then build the lab frame before the bigger upgrades, so they land on it.

1. **Quick fixes (about an hour, small).** Fix the two hints that name locked options. Show the five early answer reveals only after a solve.
2. **A real job in every brief (small, text only).** Rewrite each abstract brief's first line and add a "where this happens for real" line to each debrief. In the six settings-tuning labs, also show the live code line (`generate(...)`, `LoraConfig(...)`, `DPOConfig(...)`, vLLM flags, llama.cpp preset, `torch.optim`).
3. **History cards (small to medium).** Put the sourced rows from the history table into `src/history.ts`, with a test that every entry has a source URL. Add a collapsed card after each lab, starting with the Logic workbench, Rescue the deep network and Shannon's betting game, which have the strongest stories. Find sources for the † places and the missing mood lines as you go.
4. **Solution maps for Attention puzzles and Build a previous-word head (small).** Both labs already enumerate every setting, so this is drawing only.
5. **Shared pieces (medium to large).** `LabShell`, `MachineLog`, and one motion hook that respects reduced motion. Move one strong lab (e.g. Dice Workshop) and one weak one (e.g. Tune the Q-Learner) onto them first, to prove the API.
6. **Machinery for the four "needs work" labs (medium each).** Q-Learner maze replay, live sampler text, compute-budget map, alignment reply bars, each with its log and moving onto the shared pieces as you go.
7. **The timeline page (medium).** Built from the same `src/history.ts`, so cards and timeline can't disagree.
8. **Decision lines in the Logic workbench (small to medium).**
9. **Replay helper, then reward hacking, Wire the Policy, the two golf labs and VRAM Tetris's falling blocks (medium).**
10. **The remaining rows in all three tables**, as time allows.

As always, any new number a lab quotes needs a run and a test pin, and every change needs a check in all three themes and at phone width. I didn't run `check-pages.cjs` for this review.

**Deliberately not recommended:**

- Timers, scores, streaks or leaderboards on labs. The vision rules them out, and the research summary in the gamification proposal backs that.
- Making grid labs continuous to add difficulty. They would lose the exhaustive counts that make their verdicts honest.
- More briefs just for volume. A visible mechanism in an existing brief teaches more than a fourth brief of tiles.
- An entropy-bonus brief for policy gradients. The roadmap already records that it made no difference in these small problems.

**Open question:** should Compress the corpus keep "optimum" for the best greedy checkpoint? If a hand-picked merge set can beat 1,806 bits, "beat greedy BPE" is a better optional goal than "match" it. That needs a search to settle.
