# Gamification proposal for Neural Lab

_Brainstorm and research report, 25 November 2026. Nothing here has been built; it's a menu of ideas to choose from._

---

## TL;DR

`docs/VISION.md` already takes a position: *"Understanding is the reward; discovery is the entertainment. Never gamify with badges, points, achievements or artificial reward schedules."* The research mostly supports that rule. Points, badges and leaderboards (PBL) get people to do more of a task, but they don't make it more interesting. For tasks people already enjoy, which describes a curious engineer working through Neural Lab, rewards that depend on doing or finishing the task measurably reduce intrinsic motivation.

What does reliably improve engagement and learning is a different set of tools:

- **Game design, as opposed to game decoration.** That means real challenge, choice and feedback that tells you something (the "competence, autonomy, relatedness" that self-determination theory says makes games compelling).
- **The well-evidenced learning-science techniques, presented as play.** These are prediction, retrieval, spacing, interleaving, self-explanation and productive failure.

So this proposal suggests **game mechanics where the reward is information about your own understanding, never a token.** The six I'd build first:

1. **Confidence-rated predictions and a calibration page.** Your own predictions get scored in bits, using the cross-entropy the learner already met in the logarithms/entropy lesson.
2. **"Your solution vs every solution" histograms for labs.** These are Opus Magnum-style, but computed exhaustively from the solution spaces the labs already enumerate, so no server or other players are needed.
3. **A collection log of distinct lab solutions.** It rewards finding *different* solutions, which is the Zachtronics value, rather than grinding one.
4. **Spoiler-free share cards and solution links.** These are Wordle-style and encoded in the URL, with no accounts.
5. **Hint ladders on labs.** They are designed so the learner has to try first and can't click through to the answer.
6. **A "Sabotage" game.** One person breaks a model in a single change and sends the link; a friend diagnoses it. It's social, competitive and collaborative at once, still with no accounts.

Each item is detailed below, with evidence, how it would plug into the existing code, and its risks. There is also a list of things to avoid.

---

## 1. What the research says

### 1.1 Gamification works on average, but the average hides a lot

- Meta-analyses find positive average effects of gamification on learning, but they vary widely between studies. Sailer & Homner (2020) found a small-to-medium effect on cognitive outcomes, and that effect **held up in high-rigour studies**. The effects on motivation and behaviour were **less stable**. Two features moderated the behavioural effects: **game fiction** (a narrative frame) and **social interaction**, with competition *combined with* collaboration working best. [Sailer & Homner 2020](https://link.springer.com/article/10.1007/s10648-019-09498-w)
- Other pooled estimates range from g ≈ 0.26 (for intrinsic motivation) to g ≈ 0.8 or more. The wide confidence intervals and differences in study quality suggest a lot of publication noise. [Frontiers 2023](https://www.frontiersin.org/articles/10.3389/fpsyg.2023.1253549/full), [PMC 2023 meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10591086/), [Springer 2020 element-level study](https://link.springer.com/10.1007/s11423-020-09807-z)

**Takeaway:** "Gamification" is too broad a category to be good or bad as a whole. Which elements you use matters more than whether you gamify.

### 1.2 Points, badges and leaderboards make people do more, not care more

- In a controlled experiment, points, levels and leaderboards raised the *quantity* of output but **did not raise intrinsic motivation or perceived competence**. The authors concluded that the elements "functioned as extrinsic incentives". [Mekler et al. 2017](https://glassmanlab.seas.harvard.edu/annotated_works/mekler17.pdf)
- **Overjustification.** A meta-analysis of 128 experiments found that rewards which depend on taking part, finishing or reaching a performance level all *reduced* later free-choice intrinsic motivation (d = −0.40, −0.36, −0.28). The effect is strongest for tasks people already found interesting. For dull tasks rewards can help, which is why Duolingo-style mechanics make sense for vocabulary drilling, but they are a bad fit for a curiosity-driven course. [Deci, Koestner & Ryan 1999](https://pubmed.ncbi.nlm.nih.gov/10589297/), [overview](https://en.wikipedia.org/wiki/Overjustification_effect)
- **Leaderboards** give mixed results. They engage competitive people who are already near the top, and they drive away low performers and people who aren't competitive. [crowdsourcing study and review](https://trepo.tuni.fi/bitstream/10024/130150/1/GaLA_Conf_2020_paper_leaderboard.pdf), [systematic mapping of negative effects](https://arxiv.org/pdf/2305.08346)
- **Streaks** run on loss aversion; that's how Duolingo describes them itself. The evidence for them is mostly the company's own figures, which don't show causation, and they are exactly the "artificial reward schedule" the vision rules out. [Duolingo on streaks](https://blog.duolingo.com/how-duolingo-streak-builds-habit), [Duolingo streak research](https://blog.duolingo.com/duolingo-streak-research/)

### 1.3 What makes games themselves compelling

- Self-determination theory explains the pull of games as their ability to satisfy three needs: **competence** (I'm getting better and I can see it), **autonomy** (my choices matter, and there's more than one way) and **relatedness** (other people are involved). Games that meet these needs are enjoyable *and* tend to leave players feeling better. [Ryan, Rigby & Przybylski 2006](https://link.springer.com/content/pdf/10.1007%2Fs11031-006-9051-8.pdf)
- The key distinction from cognitive evaluation theory is that **feedback which tells you how you did builds competence, while feedback that feels like control erodes autonomy.** "Your solution uses 13 parameters; the smallest known uses 11" tells you something. "+50 XP!" is controlling.
- **Zachtronics.** Opus Magnum shows each solution against **histograms** of every player's cost, cycles and area rather than a ranked leaderboard. No single solution can top all three, so there's no one "winner", and different players can each be best on a different measure. [Opus Magnum](https://en.wikipedia.org/wiki/Opus_Magnum_(video_game)), [Zachtronics: open-ended puzzle design (GDC 2019)](https://gdcvault.com/play/1025715/Open-Ended-Puzzle-Design-at)
- **Wordle's share grid** spread because it **shows how you did without spoiling the answer**. It also came from players before the developer built it in. [The Spinoff](https://thespinoff.co.nz/media/13-01-2022/those-coloured-boxes-on-twitter-are-new-zealands-fault), [TechCrunch](https://techcrunch.com/?p=2256733)

### 1.4 Techniques with strong evidence that suit a game format

| Technique | Evidence | Already in Neural Lab? |
| --- | --- | --- |
| **Retrieval practice** (being quizzed rather than rereading) | g ≈ 0.50 vs restudy ([Rowland 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC10409703)); g ≈ 0.70 for practice tests ([Adesope et al. 2017](https://link.springer.com/10.1007/s10648-020-09572-8)) | Yes: quizzes plus the Review page |
| **Spacing** (spreading practice out over time) | Large benefit of spaced over massed retrieval, though smaller in maths (g ≈ 0.28; [Murray et al. 2025](https://eprints.whiterose.ac.uk/id/eprint/229807)) | Yes, as Leitner boxes in `src/progress.ts` |
| **Interleaving** (mixing similar-but-different problem types) | g = 0.42 overall, largest when the categories are easy to confuse ([Brunmair & Richter 2019](https://pubmed.ncbi.nlm.nih.gov/31556629/)) | Partly: "Mixed practice" shuffles questions but doesn't target confusable pairs |
| **Self-explanation** (explaining *why*, in your own words) | g = 0.55 ([Bisra et al. 2018](https://doi.org/https://doi.org/10.1007/s10648-018-9434-x)) | No |
| **Productive failure** (try the problem before being taught) | g ≈ 0.36, rising to 0.37–0.58 when done properly ([Sinha & Kapur 2021](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2022.956416/pdf)) | Yes in spirit ("feel the problem first") |
| **Prediction → surprise** | Making a prediction sparks curiosity about the true answer ([npj Sci. Learn. 2019](https://link.springer.com/article/10.1038/s41539-019-0056-y)). Curiosity improves memory, including for incidental material ([Gruber & Ranganath, PACE](https://pmc.ncbi.nlm.nih.gov/articles/PMC6891259)). Curiosity peaks when the gap in your knowledge is *small*, not huge ([Loewenstein's information-gap theory](https://shohamylab.zuckermaninstitute.columbia.edu/sites/default/files/2017-02/Curiosity.pdf)) | Yes: `<Predict>`, used 0–2 times per lesson; the choice isn't stored |
| **Hypercorrection** (confident mistakes are corrected best) | Errors made with high confidence are the most likely to be fixed after feedback, but they are also the most likely to come back later, so they need re-testing ([Butterfield & Metcalfe; Metcalfe 2011](https://pubmed.ncbi.nlm.nih.gov/21989771/), [classroom replication](https://pubmed.ncbi.nlm.nih.gov/29781391/)) | No confidence ratings |
| **Hints** | Click-through hint sequences invite "gaming the system", which is linked to worse learning ([Baker et al.](https://learninganalytics.upenn.edu/ryanbaker/PSCS-gaming-v6.pdf), [An et al. 2026, LAK](https://www.cs.cmu.edu/afs/cs.cmu.edu/Web/People/bmclaren/pubs/AnEtAl-RevisitingTheHintButton-LAK2026.pdf)) | Labs have no hint system |

Explorable explanations (Bret Victor, Nicky Case), the genre Neural Lab belongs to, have a strong design tradition but **almost no controlled studies** of their engagement or learning effects. That gap is worth knowing about: the techniques above are where the evidence actually is. [Victor](https://worrydream.com/ExplorableExplanations/), [Case](https://blog.ncase.me/explorable-explanations)

---

## 2. Design filter: rules every idea below must pass

1. **The reward is information about your understanding,** never a token. A good mechanic answers "how well do I understand this, and what should I try next?"
2. **Every number is real.** Histograms, pars and calibration scores are computed (exhaustively where feasible) and pinned by tests, following the existing "fair races and honest references" rule.
3. **Nothing punishes absence.** No streaks, decay, nagging or "you'll lose…" messages. Come back in three months and nothing is lost.
4. **Multiple solutions beat a single score.** Make it rewarding to explore the space of solutions, not just to push one number down.
5. **Optional and quiet.** Each mechanic can be ignored without blocking the lesson. No confetti; one sentence of feedback is enough.
6. **No accounts, no server.** State lives in `localStorage` and the URL. Sharing means sending a link.
7. **Three themes and phone width**, as for every widget.

---

## 3. The ideas

Effort is a rough estimate in agent-sessions (one session ≈ one finished lesson's worth of work).

### Tier 1: build these first (high value, low cost, clearly within the vision)

#### 1. Confidence-rated predictions and a calibration page

**What.** `<Predict>` gains a second click: *"How sure? — guessing · fairly sure · certain."* Each prediction and its confidence are stored. A new **"Your intuition"** page shows:

- a calibration chart: when you said "certain", how often were you right?
- your total prediction score **in bits**, against the "always guessing" baseline of log₂(number of options) per question. This is cross-entropy, which the logarithms/entropy lesson already teaches. The learner gets scored by the loss function they just learned, and the page can link back to that lesson.
- a short list of **"confident surprises"**: predictions you were sure about and got wrong.

**Why.** Prediction drives curiosity. Hypercorrection says confident errors are the most teachable moments, but also the ones most likely to return, so they belong in spaced review. Calibration gives a sense of competence that can't be gamed: the only way to improve the score is to understand more, or to be honest about what you don't know.

**How.** Extend `src/progress.ts` with a `predictions` record (keep it backward compatible or bump to `nl-progress-v2` with a migration). Give each `<Predict>` a stable `id` prop. Confident surprises enter the Leitner deck as a "re-predict" card. Add a `ReviewPage`-style page and a stat on `Home.tsx`. Pin the scoring maths with Vitest.

**Also:** most lessons have only 0–2 `<Predict>` blocks, and several have none. Adding more would also strengthen the pedagogy.

**Effort:** about 1 session. **Risk:** low. Keep the confidence step to one click so it doesn't slow lessons down.

#### 2. Solution-space histograms for labs ("you vs every possible solution")

**What.** When a lab is solved, show where the solution sits in the **distribution of all solutions**, Opus Magnum-style. The difference is that the distribution comes from enumerating the solution space, not from other players. The labs already compute these spaces:

- previous-word head: 91 of 180 single-speed settings work; 2,689 of 16,290 with two speeds
- write-the-backward-pass: 24 correct combinations out of 1,024
- attention puzzles: all 121 grid queries tried
- pruning: every on/off combination tried, smallest first
- optimizer golf: a 2,340-run search
- model clinic: every one-change fix counted

A caption would read: *"Your schedule finishes at 0.071. Of the 400+ schedules we searched, 18% did better. The best found was 0.056."* Where a lab has several measures (steps and parameters, say), draw the **Pareto front** (the solutions where you can't improve one measure without worsening another) and mark your solution on it.

**Why.** This feeds competence directly. It tells you how good your solution is, which is more informative than "you won". It turns the existing honest pars into something you can see, and it avoids every downside of leaderboards because there is nobody to lose to.

**How.** Most labs already have `search.ts` and `search.run.ts` with recorded results (`results.ts`, `fixes.ts`, `GOLF_PAR`). Record the whole distribution instead of only the optimum, add a small shared `SolutionHistogram` component (`src/components/`) built on the existing `Plot` primitives, and pin the distributions in the labs' existing tests.

**Effort:** about 1–2 sessions across all labs. **Risk:** low. Searched (not exhaustive) spaces need the existing "best found by searching…" wording.

#### 3. Collection log of distinct solutions

**What.** Labs with many valid solutions keep a local log: *"Solutions you've found: 3 of 24 correct backward passes, including one of the two that run but train wrongly."* Show the found ones as thumbnails (reuse `fieldImage` or `NetDiagram`), with "?" placeholders for the rest. Optionally add short notes on families of solutions you've discovered, for example *"you found the transposed-equivalent form"*.

**Why.** Autonomy and exploration. It rewards variety over grinding, which is the central Zachtronics value ("multiple valid solutions") and something a points score would undermine. Each new entry is a small, real discovery about how the mechanism works.

**How.** Each lab needs a canonical "solution signature" (so equivalent solutions count once) and a `localStorage` key next to the existing ones (`nl-logic-lab-v1` and others). Only where the space is enumerable.

**Effort:** about 1 session. **Risk:** it could drift towards completionism. Never show a percentage-complete bar; a count and the interesting variants are enough.

#### 4. Spoiler-free share cards and solution links

**What.** A "Share" button on labs produces:

- a **text card**, Wordle-style, for chat:
  ```
  Neural Lab · Optimizer golf · banana
  31 steps (par 27) · Adam
  ▁▂▃▅▇█▇▅▃▂  ← you're here: ▲ top 12%
  ```
- a **URL that reproduces the exact solution** (the state encoded in the hash route), so a friend can open it, see the machinery run, and try to beat it.

**Why.** Relatedness without accounts. The card shows how you did without giving the answer away, and the link makes it an invitation to play rather than a boast.

**How.** Hash routing already exists (`src/router.ts`). Add a query segment such as `#/lesson/optimizers?sol=<base64url>`, and give each lab a small `encode`/`decode` pair with round-trip tests. The card text comes from the histogram data in idea 2.

**Effort:** about 1 session. **Risk:** low. Validate decoded state strictly, since URLs are untrusted input.

#### 5. Hint ladders for labs (designed so they can't be clicked through)

**What.** Each lab gets 3–4 hints that go from general to specific:

1. a **nudge** ("What happens to the loss when the step size doubles?")
2. a **pointer to the instrument** that reveals the problem (highlight the gradient-norm readout)
3. a **partial worked step**
4. as a last resort, **one working solution**

The first hint unlocks only after a genuine attempt (for example, one run of the machine), and each later hint asks a one-line "what did you notice?" question before opening.

**Why.** It catches learners who would otherwise give up, while keeping productive failure. The attempt-first rule and the small cost on each hint deal with the click-through problem the tutoring-system research documents.

**How.** A shared `HintLadder` component that takes a list of steps and an `attempted` boolean. Record hint use locally, for the learner only and never as a score: *"solved with 1 hint"* is useful information to them.

**Effort:** about 1 session for the component and hints for the existing labs. **Risk:** writing good hints is the hard part, so match the voice in `docs/STYLE_GUIDE.md`.

#### 6. "Sabotage": break a model, challenge a friend

**What.** A new mode for the Model Clinic (`src/lessons/diagnosis-lab/`):

1. You make **one change** that breaks a healthy training run: step size, initialisation, shuffling, leaky test set, scaling…
2. You get a link.
3. Your friend opens the case with only the instruments visible and has to diagnose it and fix it.
4. The debrief shows both of you the evidence trail.

A good saboteur makes a break that's subtle but still diagnosable.

**Why.** It combines competition and collaboration, the strongest moderator in the meta-analysis, and it's social without a server. Designing a convincing failure needs deep understanding: to hide a symptom you have to know what the symptom looks like. That's the "build the attack to understand the defence" idea.

**How.** The clinic already has a menu of single changes and real runs. The link encodes (base config, seed, sabotage), and the case is re-simulated on open, so nothing needs storing.

**Effort:** about 1–1.5 sessions. **Risk:** some sabotages can't be told apart (two causes with the same evidence). The clinic's exhaustive search can detect those, and the game can warn the saboteur.

### Tier 2: medium effort, high engagement

#### 7. Puzzle of the day, seeded by date (no streaks)

A daily variant of an existing lab: a new starting point for descent golf, a new dataset for the training-lab challenges, a new sentence for the attention puzzles. It is generated from `seed = date`, so everyone gets the same puzzle, with par computed live or by a small in-browser search. It pairs naturally with the share card.

**Deliberately missing:** streak counters, "you missed yesterday" messages and notifications. Old puzzles stay playable from an archive. **Effort:** 1–2 sessions, mostly in making sure each daily instance is solvable and has a computed par. Generate the next 365 seeds and test them in CI.

#### 8. Black-box challenges ("model forensics")

These are deduction games built around curiosity. You see the behaviour of a hidden model and work out what's inside it:

- *Which activation function produced this decision boundary?*
- *Which optimizer made this trajectory?*
- *Which head is the previous-word head?*
- *What learning-rate schedule produced this loss curve?*

Each answer is checked against the real hidden setup. They work well as **interleaved review** questions, because telling apart look-alike behaviours is exactly where interleaving helps most. They extend the "match-graph-to-behaviour" assessment type already listed in the vision. **Effort:** about 1 session for a shared question type that generates its plot from a real simulation, plus 2–3 per module.

#### 9. "Explain it back" prompts and a Feynman notebook

At key moments: *"In one or two sentences: why did the loss explode at step size 0.9?"* A free-text box (optional, and skippable with one click) is followed by the expert explanation to compare with, plus the misconceptions to check your answer against. There is no automated grading. Entries collect into a **personal notebook** that can be exported as Markdown, so the learner ends the course with their own write-up of how AI works.

This is self-explanation (g = 0.55), and fitting for a project modelled on Feynman. **Effort:** about 1 session.

#### 10. A concept map that reflects what you actually remember

A visual "tech tree" of concepts and their prerequisites (vectors → dot product → attention…). Each node's colour comes from **retrieval performance** (Leitner box), not from lesson completion. A shaky prerequisite shows as faded and links to a 3-question review. It answers "what should I revise before the transformer lessons?"

**Risk:** it can turn into a progress meter that people chase for its own sake. Keep it informational: no percentages, no "fill the tree". **Effort:** 1–2 sessions, including tagging questions with concept IDs in each `quiz.ts`.

#### 11. Instruments that join your workbench as you build them

A progression you can see inside the product. After the optimizers lesson, the Adam option in the Training Lab is labelled *"you built this in Module 4"* and links back to it. After the clinic, its instruments (dead-neuron counter, gradient RMS) show up in the playground. The workbench grows with the learner, much as Zachtronics games add new parts. **Nothing gets locked**: everything stays available, and the link to its origin is the reward. **Effort:** small for each instrument, done gradually.

#### 12. End-of-module practical exams (capstones)

The vision already lists "practical exams". One integrating lab per module, with unfamiliar cases. For example, at the end of Module 4: *"train this unseen dataset to a target loss on a fixed budget of examples; you choose the optimizer, schedule and batch size."* The result shows on the solution histogram (idea 2) and the Pareto front. **Effort:** about 1 session per module, and a good fit for the existing "Labs built so far" pattern.

### Tier 3: bigger bets

#### 13. Light game fiction

Game fiction was one of the significant moderators in the meta-analysis, and Zachtronics uses it heavily: TIS-100's printed manual, the in-world memos in EXAPUNKS. The Model Clinic already has a gentle version (patients, case files). Options:

- a consistent "lab notebook" voice for lab briefings
- clients with problems ("a startup's model works on Tuesday and fails on Wednesday")
- a printable **field manual** per module: a one-page reference card, a lovely object to keep

Keep it light so it never gets in the way of the explanations. **Effort:** depends on scope; it's mostly writing.

#### 14. Puzzle editor and levels made by learners

Expose the level format of one or two labs (attention puzzles and the logic workbench are the simplest) in an editor. The level is encoded in the URL, the par is computed on save (the solver already exists), and the editor refuses levels it can't solve or prove impossible. The best community levels can be accepted by pull request into a "community" level pack. **Effort:** 2+ sessions.

#### 15. A low-cost community space

Turn on GitHub Discussions with one thread per lab ("post your share card"), and link it from each lab's debrief. There's no infrastructure to run, and it gives the relatedness that a no-accounts site otherwise lacks. **Effort:** minutes, plus moderation.

---

## 4. What to avoid, and why

| Avoid | Why |
| --- | --- |
| XP, points, levels | They add output but not intrinsic motivation, and they risk overjustification on a course people already find interesting. |
| Badges and achievements | Ruled out by the vision. Rewards for finishing are the kind that most reduced later intrinsic motivation in the meta-analysis. If you ever want some recognition, add one sentence to a lab's debrief that says something true ("you found the solution most people miss"). |
| Streaks, streak freezes, decay | They run on loss aversion and punish absence, so they fall under "artificial reward schedules". |
| Global leaderboards | They demotivate low performers and non-competitive learners, the audience is small and there's no server. Distribution histograms do the job better. |
| Timers, lives, hearts | They punish exploration and failure, and failing is how you learn on this site. |
| Random or surprise rewards | They are exactly the variable-ratio reward schedule the vision forbids. |
| Confetti and cheering on every action | Constant celebration feels patronising to this audience ("no forced enthusiasm" in the style guide). |
| Hint buttons with no friction | Learners click straight through to the answer ("gaming the system"). |

**On the user's question versus the vision:** the request asks for "gamification", and the brief forbids its most common forms. If the vision's rule has since loosened, the gentlest version that would still respect its spirit is **debrief remarks that say something true** (one sentence when you find a rare or optimal solution). It is better than a badge cabinet. Everything else in section 3 works without changing the rule at all.

---

## 5. Suggested order

1. **Phase 1, the foundations (about 4 sessions):**
   - confidence predictions and the calibration page (1)
   - solution histograms (2)
   - share cards and solution links (4)
   - hint ladders (5)

   Ideas 2 and 4 share their data. Idea 1 adds to `progress.ts`, which the other ideas also use, so do it first and settle the storage schema once.
2. **Phase 2, play (about 4 sessions):**
   - Sabotage (6)
   - collection log (3)
   - black-box review questions (8)
   - puzzle of the day (7)
3. **Phase 3, depth (ongoing):**
   - explain-it-back notebook (9)
   - concept map (10)
   - capstones (12)
   - workbench links (11)
   - then the editor (14), game fiction (13) and the community space (15)

Shared building blocks to write once: a `SolutionHistogram`/Pareto plot, a `HintLadder`, a URL-state codec (`encode`/`decode` with strict validation), and a versioned migration for `nl-progress`. Also an **export/import of progress as JSON**: without accounts, it's the only way to move between devices or keep a backup, and it lets learners keep their notebook and collection.

## 6. Checking whether it works (without analytics)

No accounts and no tracking means no A/B tests. Lighter-weight options that fit:

- **The learner's own data is the measure.** The Leitner records already show retention. A local "how have I done over time?" view (accuracy on reviews at 3, 8 and 21 days) tells each learner whether it's working for them.
- **Small user tests.** Ask a few friendly engineers to do one module with and without (say) the calibration page, and watch where they slow down, skip or replay.
- **Opt-in feedback** through GitHub Discussions (idea 15).

## 7. Notes for whoever builds this

- Several ideas touch shared code (`src/progress.ts`, `src/components/Predict.tsx`, `src/router.ts`). Another agent is currently finishing the Tokenization lesson, so start once that's committed to avoid merge conflicts.
- Follow the existing conventions:
  - pin every distribution, par and score with Vitest
  - label searched (non-exhaustive) references as "best found"
  - colour only via CSS variables
  - check light, dark, matrix and phone width with `scripts/check-pages.cjs` and `scripts/shoot-widgets.cjs`
- If any of these are adopted, record them in `docs/ROADMAP.md` ("Platform features" and "Next steps"), and add the design filter from section 2 to `docs/STYLE_GUIDE.md` so later agents don't reintroduce points and badges.

---

## Sources

- Sailer & Homner (2020), *The Gamification of Learning: a Meta-analysis*. https://link.springer.com/article/10.1007/s10648-019-09498-w · full text: https://opus.bibliothek.uni-augsburg.de/opus4/files/109056/109056.pdf
- Frontiers in Psychology (2023) gamification meta-analysis. https://www.frontiersin.org/articles/10.3389/fpsyg.2023.1253549/full
- Gamification effectiveness meta-analysis (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10591086/
- Element-level gamification meta-analysis (2020). https://link.springer.com/10.1007/s11423-020-09807-z
- Higher-education gamification meta-analysis (2026). https://journals.sagepub.com/doi/10.1177/21582440261421375
- Mekler et al. (2017), *Towards understanding the effects of individual gamification elements…*. https://glassmanlab.seas.harvard.edu/annotated_works/mekler17.pdf
- Deci, Koestner & Ryan (1999), meta-analysis of extrinsic rewards. https://pubmed.ncbi.nlm.nih.gov/10589297/ · https://en.wikipedia.org/wiki/Overjustification_effect
- Leaderboard effects. https://trepo.tuni.fi/bitstream/10024/130150/1/GaLA_Conf_2020_paper_leaderboard.pdf · https://arxiv.org/pdf/2305.08346
- Duolingo on streaks. https://blog.duolingo.com/how-duolingo-streak-builds-habit · https://blog.duolingo.com/duolingo-streak-research/
- Ryan, Rigby & Przybylski (2006), *The motivational pull of video games*. https://link.springer.com/content/pdf/10.1007%2Fs11031-006-9051-8.pdf
- Opus Magnum histograms. https://en.wikipedia.org/wiki/Opus_Magnum_(video_game) · Zach Barth, *Open-Ended Puzzle Design at Zachtronics* (GDC 2019): https://gdcvault.com/play/1025715/Open-Ended-Puzzle-Design-at
- Wordle share grid. https://thespinoff.co.nz/media/13-01-2022/those-coloured-boxes-on-twitter-are-new-zealands-fault · https://techcrunch.com/?p=2256733 · https://www.smithsonianmag.com/smart-news/heres-why-the-word-game-wordle-went-viral-180979439/
- Retrieval and spacing: Rowland (2014) and related work. https://pmc.ncbi.nlm.nih.gov/articles/PMC10409703 · https://link.springer.com/10.1007/s10648-020-09572-8 · https://eprints.whiterose.ac.uk/id/eprint/229807
- Brunmair & Richter (2019), *Similarity matters: interleaved learning*. https://pubmed.ncbi.nlm.nih.gov/31556629/
- Bisra et al. (2018), self-explanation meta-analysis. https://doi.org/10.1007/s10648-018-9434-x
- Sinha & Kapur (2021), productive failure. https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2022.956416/pdf
- Hypercorrection. https://pubmed.ncbi.nlm.nih.gov/21989771/ · https://pubmed.ncbi.nlm.nih.gov/29781391/
- Curiosity and memory. https://pmc.ncbi.nlm.nih.gov/articles/PMC6891259 · https://link.springer.com/article/10.1038/s41539-019-0056-y · https://shohamylab.zuckermaninstitute.columbia.edu/sites/default/files/2017-02/Curiosity.pdf
- Hints and gaming the system. https://learninganalytics.upenn.edu/ryanbaker/PSCS-gaming-v6.pdf · https://www.cs.cmu.edu/afs/cs.cmu.edu/Web/People/bmclaren/pubs/AnEtAl-RevisitingTheHintButton-LAK2026.pdf · https://learnlab.org/research/wiki/index.php/Help_abuse
- FSRS (a possible later upgrade to the Leitner scheduler). https://github.com/open-spaced-repetition · https://docs.ankiweb.net/faqs/what-spaced-repetition-algorithm
- Explorable explanations. https://worrydream.com/ExplorableExplanations/ · https://blog.ncase.me/explorable-explanations
