<!-- authored-by: Claude Opus 5.5 -->
# Curriculum Audit: Order, Jargon, History and the Human Story

_Written November 2026, at the owner's request. It covers all 66 registered lessons in `src/curriculum.ts`, the brief in `docs/VISION.md`, the history data in `src/history.ts` and the timeline page. Recommendations only: no lesson has been moved yet._

The owner's questions were:

1. Does the order build from basics, so a learner never meets jargon before it has been explained?
2. Does the course give enough history, and enough of what computing was like at the time, to fix each idea in its moment?
3. Does it follow the ideas in the order they were invented and discovered?
4. Does it take the mystique out of AI by showing that the best minds struggled too: they hit dead ends, waited decades, and found the intuition hard?

The goal behind all four is learning, motivation and curiosity.

---

## 1. Verdict in brief

| Question | Verdict |
|---|---|
| Builds from the basics | **Mostly yes, with five real inversions** where a lesson leans on ideas taught later (§3). |
| No jargon before it is explained | **Strong in Modules 1–7, weaker after that.** The early lesson summaries are inviting. The later ones read like abstracts ("ρᵏ vanishing/exploding Jacobian law", "cancels Z(x)", "B=0 & A~N breaks symmetry"), and some lessons use RL and KL vocabulary several modules before it is taught (§4). |
| History and the feel of the time | **Present but tucked away.** There are well-sourced History cards on 25 of the 66 lessons, plus a timeline page. The cards are collapsed, written as terse citations, and placed after the labs. 20 lessons name no historical person or year in their prose at all. There is no prologue, no telling of the AI winters, and almost nothing about what computers were like in each era (§5). |
| Follows the order of discovery | **The neural-network core does; the course as a whole doesn't.** Neuron → XOR → backprop → recurrence → attention → transformer → LLM is close to the historical order. But the oldest material in the course (least squares from 1805, PCA from 1901, Bellman's equation from 1957) sits in the middle and near the end, and the opening maths block is ordered like a textbook, not a story (§6). |
| Shows the struggle | **Rarely.** Where struggle appears, it is in one-line `dispute` fields. The best material is missing: Rosenblatt's 1958 hype, Minsky building a neural machine and later becoming the field's best-known critic, backprop being found again and again across 16 years, the 1990s when "neural network" was a dirty word, Adam's flawed convergence proof, BatchNorm's disputed explanation (§7). |

**The main recommendation** is a *historical spine with a conceptual order*. Keep dependencies as the hard constraint, and tell the course as a story in eras ("acts"), with the human history in the main text instead of only in collapsed cards. A strictly chronological course would teach worse, for reasons given in §6. But in most places the history and the best teaching order agree, and where they disagree the course should say so openly.

---

## 2. What is already working, and should be kept

- **Problem-first openings.** Every lesson I sampled opens with a problem the learner can feel: the spam filter in `vectors`, "billions of slopes" in `backpropagation`, "it was too tired / too wide" in `attention`. This is the right base for a historical hook, because the history is the story of who faced that problem first.
- **`src/history.ts` is a serious asset.** Every claim has a source, a test enforces it, there are "before → this → after" chains, and disputes are named instead of crowning a hero. The `TimelinePage` reads the same data. The infrastructure for everything proposed below already exists.
- **History cards after the lab is a good decision for labs.** The learner feels the problem, then learns who else felt it. Keep it. The proposals below add to it and don't replace it.
- **Honest reporting.** The house rule (each method tuned to its own best, results reported even when they contradict the textbook story) is the same attitude as "the experts were confused too". It is already in the course's character, but it hasn't yet been aimed at history.
- **The neural core follows prerequisites carefully.** For example, `neuron` explicitly picks up the spam score from `dot-product`, and `chain-rule` comes before `backpropagation`.

---

## 3. Ordering: prerequisite inversions

These are places where a lesson relies on ideas the learner hasn't met yet. They are listed by severity.

| # | Where | What goes wrong | Fix |
|---|---|---|---|
| 1 | **`rlhf-dpo` (M14) comes before the whole RL module (M16)** | RLHF is taught using *policy* (π), *reward model*, *reference policy* and the PPO-style "leash", and the RL module that defines those terms comes two modules later. The historical order is the reverse: Bellman 1957, Q-learning 1989, policy gradients 1992, PPO 2017, RLHF 2017–22. | Move Module 16 (RL) before Module 14. |
| 2 | **KL divergence is used before it is taught** | `rlhf-dpo` makes a KL penalty central ("KL distance from SFT reference"). The only lesson that explains KL is `autoencoders`, in Module 17. | Add KL to `information` (M1). It is a natural next step: KL = cross-entropy − entropy, "the extra surprise you pay for believing the wrong distribution". Generative models then also come before it is needed. |
| 3 | **`tokenization` is the *last* lesson of the transformer module** | The learner spends `word-embeddings`, three sequence lessons and four attention lessons treating text as "words" before learning how text actually becomes IDs. It is also the most concrete, least intimidating lesson in the module. | Move it before `one-hot-vs-dense` / `word-embeddings` in Module 10, or make it the first lesson of Module 12. It also brings the "strawberry" hook to the front. |
| 4 | **The `high-dimensions` superposition lab in Module 1** | Lesson 11 of the maths block uses *neurons*, *features*, *embeddings* (10 mentions) and *language models*, all before the first neuron. The interpretability result it shows (Anthropic, 2022) is one of the newest ideas in the course, presented as foundations. | Move `high-dimensions` to the start of Module 10 (embeddings), where "768 dimensions" means something. Alternatively, keep the geometry in M1 and move the superposition lab to an end-of-course "open questions" lesson (§8). |
| 5 | **`similarity-search` (IVF, LSH, PQ-64, 10M-vector RAM sizing) in Module 10** | This is infrastructure engineering among lessons about representation, about eight modules before the RAG lesson that needs it. | Move it into Module 18, directly before `rag`. |
| Minor | `network-playground` (M3) says "Train" before gradient descent (M4) | Handled acceptably: the lesson presents training as a black box to open later. Worth one explicit sentence: "how training finds these weights is Module 4's whole story". | Add that sentence. |
| Minor | `gradient-descent` comes before `loss-functions` | Gradient descent defines "loss" in one sentence, which is enough. | Leave as is, or swap the two. Both orders work. |

### The structural issue: 11 lessons (≈4 hours) of maths before the first neuron

Module 1 is a well-made set of lessons, but its position contradicts the course's own brief. `VISION.md` says "build prerequisites just in time" and "introduce each [maths topic] when it becomes useful". In practice the learner gets vectors → dot products → matrices → matmul & tensors → derivatives → gradients → chain rule → softmax → cross-entropy → statistics → high-dimensional geometry, and only then a neuron. That is a university-style "prerequisites first" order. It is also where self-learners most often give up. The forward references show the cost: Module 1 lessons mention *neuron* 30 times, *loss* 61 times and *logit* 38 times, to motivate maths whose purpose the learner can't see yet.

**Recommendation:** keep the Module 1 lessons, but spread them just ahead of the lessons that need them:

| Toolkit lesson | Place it directly before |
|---|---|
| vectors, dot-product | neuron |
| derivatives, gradients | gradient-descent |
| matrices, matrix-multiplication | forward-propagation (layers are matrices) |
| chain-rule | backpropagation |
| probability (softmax), information (+ KL) | loss-functions / classification |
| statistics | vanishing-gradients (initialisation is the CLT) or overfitting |
| high-dimensions | embeddings |

Keep a "Maths toolkit" index page that lists them all, so a learner who wants the maths first can still take that route. The lesson files hardly change. The prose changes are small, because these lessons already motivate themselves with AI examples. The learner reaches a working neuron in the third lesson instead of the twelfth.

---

## 4. Jargon on the way in

The early material does this well: terms are introduced as names for things already understood, `<Term>` links go to a 200-entry glossary, and summaries like "Why a list of numbers can also be an arrow" invite the learner in. From about Module 9 onwards, three things slip.

1. **Lesson summaries became abstracts.** The sidebar, home page and "next lesson" cards are the first thing a learner reads, and a learner thinking "should I continue?" meets:
   - `rnns`: "the ρᵏ vanishing/exploding Jacobian law, Pascanu's exploding cliff rescued by gradient clipping…"
   - `lora`: "factoring ΔW = BA compresses updates 128×, why B=0 & A~N breaks symmetry"
   - `rlhf-dpo`: "how DPO cancels Z(x) to drop the reward model, catch likelihood displacement"
   - `contrastive-learning`: "InfoNCE … on the unit hypersphere, temperature τ hard-negative focus"

   Compare these with Module 2's "Stack a hundred linear layers and you still have one. Then add a kink, and everything changes." **Fix:** rewrite the summaries from Module 8 onwards in the early style. Name the problem and the surprise, not the method. For example, for `lora`: *"Fine-tuning a giant model without copying it: why the change a model needs is far smaller than the model."* This is cheap and high-impact, and a test could flag summaries that contain symbols or more than about 25 words.
2. **Forward references to RL and KL** (see §3, items 1–2).
3. **Named-paper shorthand.** Phrases such as "Pascanu's cliff", "Bahdanau & Luong", "Chinchilla" and "k-quants" appear in summaries before the learner knows who these people or models are. Inside a lesson, once the story has been told, the names help. Before that they are noise. The history proposals below turn these names into characters the learner has met.

---

## 5. History and the feel of the time: what's there and what's missing

### What exists

- 25 `HistoryCard`s, one per lab, collapsed and placed about two-thirds of the way through each lesson, with sourced "who / mood / before → this → after / dispute" fields.
- A `#/timeline` page that collects them.
- Scattered prose mentions. A count of person names and years in lesson prose (outside the cards) puts **20 of the 66 lessons at zero historical references**, including `dot-product`, `derivatives`, `gradient-descent`, `backpropagation`, `forward-propagation`, `knn`, `multi-head-attention`, `kv-cache`, `rag` and `evaluation`. `backpropagation`, one of the most important stories in the field, has its history only in a collapsed card in the *next* lesson.

### What's missing

**No founding story.** Nowhere in the course do these appear: Turing ("Computing Machinery and Intelligence", 1950, with its section on "learning machines"), the 1955 Dartmouth proposal (which expected "a significant advance" from "a carefully selected group of scientists" working "together for a summer"), Hebb's 1949 learning rule, or ADALINE/LMS (Widrow & Hoff, 1960, which is essentially SGD on a linear neuron and the direct ancestor of Module 4). Rosenblatt appears only inside a collapsed card, and the course never shows the perceptron *learning*. `logic-lab` has hand-set weights only.

**The AI winters are only "mood" quotes.** The cards quote Wikipedia ("funding was difficult to find in the 1970s"), but the course never tells the story: the 1958 hype, Perceptrons (1969), the Lighthill report (1973), the expert-systems boom and the 1987 Lisp-machine collapse, and the 1990s when, as one card quotes, researchers "avoided the term artificial intelligence for fear of being viewed as wild-eyed dreamers". This is the most motivating part of the whole history: the ideas the learner is studying were declared dead twice.

**The big turning points are missing.** There is no telling of ImageNet (2009, labelled by crowd workers) or AlexNet (2012, trained on two gaming GPUs), though ImageNet is mentioned in passing in five lessons. Nothing covers the 2006 "deep learning" rebrand, DQN on Atari (2013) or AlphaGo (2016) as moments, GPT-2's staged release (2019), GPT-3 (2020) or ChatGPT (Nov 2022) as the event that put this course's subject in the news. The 2024 Nobel Prizes (Physics: Hopfield and Hinton; Chemistry: Hassabis, Jumper and Baker) aren't mentioned either, though they are a striking "how far it came" bookend.

**Almost nothing about what computing was like.** The course talks about modern GPUs a lot (19 lessons mention GPUs), but never about the machines each idea was born on. That context is what fixes an idea in its moment, and it explains *why* good ideas waited:

| Era | The machines | Why it mattered |
|---|---|---|
| 1940s–50s | Vacuum tubes; IBM 704; Rosenblatt's custom **Mark I Perceptron** (a camera of 400 photocells, with weights held in motor-driven potentiometers) | The weights were literally knobs turned by motors: "observable machinery", in the course's own spirit. |
| 1960s–70s | Mainframes, time-sharing, LISP; memory in kilobytes | Symbolic AI (logic, search) fitted these machines; learning from data didn't. |
| 1980s | Workstations (VAX, Sun), Lisp machines, the PDP books (1986) | Backprop works, but a "large" network has thousands of weights and trains overnight. |
| 1990s | PCs, the early web, MNIST (1998, 60,000 digits) | Datasets are small, so SVMs and trees win, and neural networks go out of fashion. |
| 2000s | Web-scale data, GPUs built for games, CUDA (2007), ImageNet (2009) | The two missing ingredients, data and parallel arithmetic, arrive. |
| 2010s | GPU deep learning, TPUs, frameworks with autodiff (Theano, TensorFlow, PyTorch) | Every engineer gets `backward()`: Module 5's machinery as a library call. |
| 2020s | Clusters of tens of thousands of GPUs; scaling laws; chips as geopolitics | Module 13's 6ND arithmetic becomes a budget line in billions. |

The thread running through this table is the course's best motivational story, and it is never told: **many of the central ideas were right decades early and waited for compute and data.** Backprop (1970/1986), CNNs (1980/1989) and LSTMs (1997) all fit this pattern. Sutton's essay "The Bitter Lesson" (2019) states it outright, and it gives the learner a reason to care about Modules 13 and 15.

---

## 6. Does it follow the order of discovery?

Here is each module's position against the dates of its central ideas:

| Module | Central ideas, by date | Placement in historical terms |
|---|---|---|
| 1 Foundations | calculus (1600s), CLT (1733–1812), Shannon (1948); superposition (2022) | Mostly old; one 2022 result placed at the very start |
| 2 Neuron | McCulloch–Pitts 1943, perceptron 1958, XOR/Perceptrons 1969 | ✓ chronological start |
| 3 Architecture | multilayer nets (1960s–80s), universal approximation 1989 | ✓ |
| 4 Optimisation | gradient descent 1847, SGD (Robbins–Monro) 1951, momentum 1964, RMSProp 2012, Adam 2014, SGDR 2016 | A 170-year toolbox; the dates jump around inside it, and the course doesn't say so |
| 5 Backprop | 1970 / 1974 / 1986; vanishing gradients 1991; ResNet 2015 | ✓ |
| 7 Generalisation | old statistics; dropout 2012–14; double descent 2019 | Mixed |
| 8 Classical ML | **least squares 1805**, PCA 1901, kNN 1951, k-means 1957, CART 1984, SVM 1992–95, random forests 2001 | **The oldest material in the course, placed in the middle** |
| 9 Vision | Hubel & Wiesel 1959, Neocognitron 1980, LeNet 1989, AlexNet 2012 | ✓ broadly |
| 10 Embeddings | LSH 1998, word2vec 2013, contrastive learning 2018–21 | ✓ broadly |
| 11 Sequences | Jordan/Elman 1986–90, LSTM 1997, seq2seq/attention 2014 | **Before 2013 in history; after it in the course** (a defensible teaching choice) |
| 12 Transformers | 2017; BPE 1994/2015; RoPE 2021 | ✓ |
| 13 LLMs | Shannon 1948–51; GPT 2018; scaling laws 2020–22 | ✓ |
| 14 Fine-tuning | 2017–2023 | ✓ but needs Module 16 |
| 15 Inference | 2019–2023 | ✓ |
| 16 RL | **Bellman 1957, Samuel 1959, TD 1988, Q-learning 1989**, DQN 2013, PPO 2017 | **Old ideas placed after the newest material** |
| 17 Generative | autoencoders 1980s/2006, VAE 2013, GAN 2014 (no lesson), diffusion 2015/2020, CFG 2022 | Comes after LLMs, though VAEs and GANs came before transformers |
| 18 Systems | RAG 2020, agents 2022–24 | ✓ |

### Should the course be strictly chronological? No.

History is often a poor teacher. Ideas were found out of order, by people who didn't understand them yet:

- **Backprop came before anyone understood why deep networks wouldn't train.** It was found in 1970/1974/1986; the vanishing gradient was diagnosed in 1991; the practical fixes (ReLU, good initialisation, residuals) arrived in 2010–2015. Teaching it in pure date order would hide the cause-and-effect the course is built on.
- **Some essential tools are very young.** LayerNorm (2016) is newer than the problems it fixes, and teaching transformers without it would be worse teaching.
- **Some old ideas only make sense after new ones.** Shannon's 1951 guessing game is the perfect introduction to *next-token prediction*, but it only lands once the learner knows what a model predicts.

What makes history useful in teaching is narrative motive: *who was stuck on what, and why the previous idea wasn't enough*. `VISION.md` already asks for this ("motivate inventions: what problem existed, why previous approaches failed"). So the recommendation is to **keep the order of dependencies, group lessons into historical eras where the dependencies allow, and say openly where the course breaks chronology and why.** A one-line note does the job: *"Historically, LSTMs (1997) came before word2vec (2013); we've swapped them, because a network that reads words needs word vectors first."* That keeps the learner's sense of time correct, and the swap itself teaches something.

---

## 7. The struggle: stories that take the mystique away

Each of these is small, true, and does what the owner asks: it shows that the inventors were confused, wrong, ignored or lucky. Each is mapped to the lesson it would anchor. **Every claim must be sourced in `src/history.ts` before it goes into a lesson, under the existing rule.** The items marked _(verify carefully)_ are well-known anecdotes whose details are often garbled in retelling.

| Story | What it shows | Lesson |
|---|---|---|
| Legendre (1805) and Gauss (1809) both publish least squares; Gauss says he had used it since 1795, and the priority dispute is bitter. Gauss had used it to find the lost asteroid Ceres (1801). | The first "learning algorithm" was invented to find a lost asteroid, and its inventors fought over the credit. | `loss-functions` / `linear-regression` |
| Cauchy proposes gradient descent in 1847 to solve the equations of astronomical orbits. | The algorithm behind every training run was a 19th-century trick for saving arithmetic done by hand. | `gradient-descent` |
| 1958: Rosenblatt's perceptron is reported in the press (NYT) as "the embryo of an electronic computer" expected to "walk, talk, see, write, reproduce itself and be conscious of its existence". | AI hype is older than the transistor computer, and the perceptron on that day was a single layer. | new `perceptron` lesson |
| Minsky built SNARC (1951), one of the first neural-network machines, then co-wrote *Perceptrons* (1969), the field's best-known critique. | Even the experts changed their minds, and the critique was mathematically right about what it actually claimed. | `logic-lab` / XOR interlude |
| Backprop is found again and again: Linnainmaa 1970, Werbos 1974 (thesis, largely unnoticed for years), Parker and LeCun mid-1980s, Rumelhart–Hinton–Williams 1986. | A "great idea" sat in plain sight for 16 years. Ideas need an audience and a machine, not only an inventor. | `backpropagation` (in the opening, not a collapsed card) |
| Hochreiter's 1991 diploma thesis, written in German, diagnoses vanishing gradients. LSTM follows in 1997 and is widely used only after about 2014. | Being right early is not the same as being heard. | `vanishing-gradients`, `lstms-grus` |
| The 1990s: SVMs (convex, provable, with tidy theory) beat neural networks, which stay out of fashion. LeCun's convnets read real cheques at Bell Labs all the same _(verify the figures carefully)_. CIFAR funds a small group to keep going (2004); "deep learning" is partly a rebrand (2006). | The field was led by fashion and proofs, not just results. The learner is studying ideas that survived being unpopular. | interlude before Module 8 / `cnns` |
| ReLU, max(0, x), probably the simplest possible fix, only became standard around 2010–11 (Nair & Hinton; Glorot, Bordes & Bengio). | Decades of very smart people missed the obvious. The learner may well see it before the history says it. | `activation-functions` |
| ImageNet (Fei-Fei Li et al., 2009) took years of crowd-sourced labelling and met with scepticism. AlexNet (2012) won by a wide margin on two consumer gaming GPUs. | The breakthrough was data plus hardware made for video games, not a new equation. | `cnns` |
| Adam (2014) became the default optimiser. In 2018 Reddi, Kale & Kumar showed that its convergence proof had an error, and the paper won a best-paper award. | A peer-reviewed proof behind the most-used optimiser in the world was wrong, and it still works in practice. | `optimizers` |
| BatchNorm (2015) was explained as fixing "internal covariate shift". In 2018 Santurkar et al. argued that this isn't why it helps. | A method can work while its inventors' explanation is wrong, and the field can use it for years before noticing. | `transformer-block` (LayerNorm) or a normalisation section |
| Classical bias–variance says bigger models overfit. "Double descent" (Belkin et al., 2019) shows test error falling *again* past the interpolation point. | The textbook story was incomplete as recently as 2019. | `overfitting` |
| Kaplan et al. (2020) and then Chinchilla (2022) on how to split compute between parameters and data. | Already in the course, and a model for the rest: the experts' first estimate was revised two years later. | `scaling-laws` (keep) |
| GANs (Goodfellow et al., 2014), reportedly first coded the night after an argument _(verify carefully)_. | Big ideas can be quick and informal. | new `gans` lesson |
| Grokking (Power et al., 2022): networks suddenly generalising long after they had memorised the data. The discovery is often described as accidental _(verify carefully)_. | Researchers still find basic phenomena by accident. | `regularization` or the epilogue |
| CoastRunners (2016), reward hacking. | Already in the course. A good model of how to tell these stories. | `reward-hacking` (keep) |

There is one more framing device the course could use: **"You are standing where they stood."** Many labs already have the learner rediscover something (XOR needs a hidden layer; the step size matters; residual skips rescue depth). Say so in the lesson: *"You just found in four minutes what took the field from 1969 to 1986. They didn't have this widget, the arithmetic, or the hindsight."* That is a true statement, and few things motivate a learner more.

---

## 8. Proposed curriculum: twelve acts on a historical spine

This version respects every dependency in §3, spreads out the maths block (§3), puts history in the main text (§5), and follows chronology wherever the dependencies allow. **New** marks a lesson that doesn't exist yet, and ↪ marks a moved lesson.

**Prologue: Can a machine learn?** _(new, about 10 minutes, no maths)_
Turing's 1950 question, the Dartmouth summer, the 1958 hype, two winters, ImageNet, ChatGPT. Ends with the course's thesis: *every idea here was once confusing to the people who invented it, and many were declared dead before they worked.* The timeline becomes the course map: "you are here: 1943".

**Act I: The first artificial neurons (1943–1969)**
↪ vectors · ↪ dot-product · neuron · logic-lab · **New: The perceptron learns** (Rosenblatt's error-driven update rule: learning *without calculus*, the Mark I's motor-driven weights, the 1958 headline) · _Interlude: the XOR wall and the first winter_ (Perceptrons, Lighthill).

**Act II: Learning as walking downhill (1805–1960)**
↪ derivatives · ↪ gradients · gradient-descent (Cauchy 1847) · loss-functions (with the Gauss/Legendre least-squares story) · **New or split: Least squares & ADALINE** (the line fit as the first learning machine; LMS (1960) as SGD on a linear neuron) · ↪ probability · ↪ information (+ KL) · logistic regression (split from `linear-regression`: "the neuron *is* logistic regression").

**Act III: Hidden layers and the backprop revival (1970–1989)**
↪ matrices · ↪ matrix-multiplication · forward-propagation · activation-functions · depth-and-width · ↪ chain-rule · backpropagation (history in the opening) · network-from-scratch · network-playground · _Interlude: invented four times, and the 1980s connectionists against symbolic AI_.

**Act IV: Making networks actually train (a 170-year toolbox)**
sgd-minibatch (Robbins–Monro 1951) · optimizers (Polyak 1964 → Adam 2014, and the flawed proof) · lr-schedules · ↪ statistics · vanishing-gradients (Hochreiter 1991 → ResNet 2015) · training-lab (universal approximation 1989).

**Act V: When neural networks were unfashionable (1990s–2000s)**
_Interlude: the second winter; SVMs rule; CIFAR keeps the faith._ · overfitting (+ double descent) · regularization · diagnosis-lab · linear-regression (rest: metrics, ROC, feature engineering) · knn · decision-trees · **New: Support vector machines** (the method that beat neural nets for a decade, and why: convexity and guarantees) · clustering-pca.
_This places classical ML in the era when it was the state of the art, so it reads as a rival tradition rather than a detour._

**Act VI: The deep learning big bang (2009–2015)**
_Interlude: ImageNet, gaming GPUs, CUDA, AlexNet._ · images-as-tensors · convolution · cnns · object-detection.

**Act VII: Meaning as geometry (1998–2021)**
↪ high-dimensions · ↪ tokenization · one-hot-vs-dense · word-embeddings (2013) · contrastive-learning.

**Act VIII: Memory and sequences (1986–2016)**
rnns · lstms-grus · seq2seq-attention. _One-line note: historically these came before word2vec; we needed word vectors first._

**Act IX: Attention is all you need (2017)**
attention · multi-head-attention · positions · transformer-block.

**Act X: Machines that imagine (2013–2022)**
autoencoders (+ VAE) · **New: GANs** (named in the M17 blurb and `VISION.md`, but there is no lesson) · diffusion · guidance.

**Act XI: Learning from consequences (1957–2017)**
↪ mdps (Bellman 1957, Samuel's checkers 1959) · q-learning (TD-Gammon 1992, DQN 2013) · policy-gradients (AlphaGo 2016, PPO 2017) · reward-hacking.
_Taught here because the LLM post-training lessons need it. Historically it is one of the oldest threads, and the act should open by saying so._

**Act XII: Large language models (2018–now)**
next-token-prediction (Shannon 1951) · sampling · scaling-laws · lora · rlhf-dpo (now after RL and KL) · quantization · kv-cache.

**Act XIII: Building with models**
↪ similarity-search · rag · tool-calling · evaluation.

**Epilogue: What nobody understands yet** _(new)_
Superposition and interpretability (the lab from `high-dimensions` could live here), grokking, why scaling works at all, open problems. It ends where the prologue started: the field's best minds are still confused, and the learner now knows enough to be confused about the same things.

(This adds one act to the existing 18 modules' grouping. The lesson files keep their ids, so `curriculum.ts` changes but URLs don't.)

---

## 9. How to tell history inside the lessons

Following the house style, history should be *narrative motive*, not trivia:

1. **An opening hook in the main text (2–4 sentences)** for every lesson whose central idea has a story. Pose the problem as the person who first faced it: *"In 1986 a network with three layers was a big network, and nobody had a reliable way to train its middle."* The facts come from `history.ts`, so the sourcing stays in one place. Add a `<HistoryHook id>` component that renders the entry's `before` claim as text, so prose and timeline can't disagree.
2. **Keep the collapsed card after the lab** for the full sourced record (who, where, before → this → after, dispute).
3. **Era interludes:** short non-lab pages (5 minutes each) between acts, about the machines, the money, the mood and who was wrong. Six or seven cover it: first winter, 1980s connectionism, second winter, the big bang, the transformer moment, the ChatGPT moment.
4. **A "You are standing where they stood" note** after any lab where the learner rediscovers a historical result (§7).
5. **Time-stamped timeline.** Highlight the learner's current era on the `#/timeline` page, and link each lesson's header to its year. The data is already there.
6. **Tone:** no hero worship and no gossip. Name the dispute, as `history.ts` already does. Use people's struggles to normalise the learner's own ("this took the field 17 years; it is fine that it took you 20 minutes").

---

## 10. Suggested phasing

**Phase 1: quick wins (no new lessons, low risk)**
1. Move `rl` (M16) before `finetuning` (M14). §3 #1.
2. Add KL divergence to `information`. §3 #2.
3. Move `tokenization` to before `one-hot-vs-dense`. §3 #3.
4. Move `similarity-search` to the start of Module 18. §3 #5.
5. Rewrite lesson summaries from Module 8 onwards in the plain style, and add a test that flags symbols in summaries. §4.
6. Add a historical hook to the opening of `backpropagation`, `gradient-descent`, `neuron`, `activation-functions` (ReLU), `cnns` (AlexNet), `optimizers` (Adam's proof) and `overfitting` (double descent).

**Phase 2: the story layer**
7. Write the Prologue.
8. Write a new `perceptron` lesson (learning rule + Mark I + 1958 hype + the XOR wall). It is the largest gap in both the history and the teaching sequence.
9. Write era interludes; add `HistoryHook`; highlight the current era on the timeline.

**Phase 3: the restructure**
10. Spread the Module 1 toolkit lessons just in time (§3), keeping a Maths Toolkit index.
11. Regroup into acts (§8), and split `linear-regression` into the Act II and Act V parts.
12. New lessons: `svm`, `gans`, epilogue.

**Cost to watch when moving lessons.** Lesson prose has about 60 "in the X lesson" references and 37 relative "last/next/previous lesson" references. Relative ones break silently when the order changes. Before any reorder, grep for `(last|previous|next) lesson` and `in the .* lesson` and rewrite them to name the lesson. Quizzes and the review page may also assume the current order, so check `src/pages/ReviewPage.tsx`. A Vitest check that every "the X lesson" reference names a lesson that comes *earlier* in `ALL_LESSONS` would make future reorders safe.

---

## Appendix: method

- Read `src/curriculum.ts` (66 lessons), `docs/VISION.md`, `docs/ROADMAP.md`, `HANDOFF.md`, `src/history.ts` (25 entries), `src/components/HistoryCard.tsx`, and the openings of `vectors`, `neuron`, `logic-lab`, `backpropagation` and `attention`.
- Counted historical person names and years (1600–2029) in each lesson's prose, excluding `HistoryCard` lines, with a fixed list of about 70 names. This undercounts lessons that mention history without naming anyone, so treat the counts as a guide.
- Counted forward-referenced terms (neuron, loss, logit, embedding, superposition, …) in Modules 1–3.
- Searched for key historical topics anywhere in lesson prose. Not found anywhere: Turing, Dartmouth, Hebb, Widrow/ADALINE, Rosenblatt (outside the card), SVM, Hopfield, the Nobel prizes, "bitter lesson". GANs are mentioned only in `autoencoders`.
- The historical dates in this report are standard and well documented. Under the house rule, each one still needs a source in `src/history.ts` before it appears on the site.
