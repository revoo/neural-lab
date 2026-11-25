// authored-by: Claude Opus 5.5
/**
 * History of every lab's idea: when, where, who, the mood of the moment, and
 * before → this → after. Every claim carries its own source URL, and the test
 * in history.test.ts checks that. Dates and quotes were checked against the
 * linked pages in November 2026 (docs/LAB_IMPROVEMENTS.md, "History"); author
 * affiliations were checked on each paper's title page. The timeline page and
 * the per-lab cards both read this file, so they cannot disagree.
 */

export type Claim = { text: string; source: string };

export type HistoryEntry = {
  /** Unique id; also used by <HistoryCard id> in the lesson MDX. */
  id: string;
  lessonId: string;
  /** The lab's name as shown in the lesson. */
  lab: string;
  /** Short title for the idea, used on the timeline. */
  idea: string;
  /** Key years, ascending. The first is where the timeline places the entry. */
  years: number[];
  /** Display form of the years, e.g. "1943 · 1958 · 1969". */
  when: string;
  /** Where it happened and who did it. */
  who: Claim[];
  /** What the field believed at the time. Left out rather than invented when no source was found. */
  mood?: Claim;
  /** Before → this → after, one sourced step each. */
  story: { before: Claim; this: Claim; after: Claim };
  /** A disputed point, named plainly instead of picking a hero. */
  dispute?: Claim;
};

const WP = (page: string) => `https://en.wikipedia.org/wiki/${page}`;

export const HISTORY: HistoryEntry[] = [
  {
    id: 'dice',
    lessonId: 'statistics',
    lab: 'The Dice Workshop',
    idea: 'Sums of random parts form a bell',
    years: [1733, 1812],
    when: '1733 · 1812',
    who: [
      { text: 'Abraham de Moivre approximated the spread of coin-toss counts with the bell curve in a 1733 article.', source: WP('Central_limit_theorem') },
      { text: 'Pierre-Simon Laplace extended it in Théorie analytique des probabilités (1812).', source: WP('Central_limit_theorem') },
    ],
    story: {
      before: { text: 'De Moivre needed the distribution of the number of heads in many coin tosses, an exact sum that grows unwieldy by hand.', source: WP('Central_limit_theorem') },
      this: { text: 'The de Moivre–Laplace theorem, "the earliest version" of the central limit theorem: sums of many random parts come out bell-shaped.', source: WP('Central_limit_theorem') },
      after: { text: 'He initialisation (2015) scales random weights so that sums of hundreds of them keep a steady spread, layer after layer.', source: 'https://arxiv.org/abs/1502.01852' },
    },
  },
  {
    id: 'superposition',
    lessonId: 'high-dimensions',
    lab: 'Superposition',
    idea: 'Models pack more features than neurons',
    years: [2022],
    when: '2022',
    who: [{ text: 'Nelson Elhage and colleagues at Anthropic (with Harvard), Toy Models of Superposition.', source: 'https://arxiv.org/abs/2209.10652' }],
    story: {
      before: { text: 'The hope that each neuron stands for one concept, so a network can be read neuron by neuron.', source: 'https://arxiv.org/abs/2209.10652' },
      this: { text: 'Small models store "additional sparse features in superposition": more features than dimensions, at almost-perpendicular angles.', source: 'https://arxiv.org/abs/2209.10652' },
      after: { text: 'Interpretability research that pulls the packed features apart again, with sparse dictionary learning.', source: 'https://transformer-circuits.pub/2023/monosemantic-features' },
    },
  },
  {
    id: 'logic',
    lessonId: 'logic-lab',
    lab: 'Logic workbench',
    idea: 'Threshold neurons, the perceptron and XOR',
    years: [1943, 1958, 1969],
    when: '1943 · 1958 · 1969',
    who: [
      { text: 'Warren McCulloch (University of Illinois at Chicago) and Walter Pitts (University of Chicago) describe the threshold neuron in 1943.', source: WP('Warren_Sturgis_McCulloch') },
      { text: 'Frank Rosenblatt builds the perceptron at the Cornell Aeronautical Laboratory in Buffalo, New York (1958).', source: WP('Frank_Rosenblatt') },
      { text: 'Marvin Minsky and Seymour Papert publish Perceptrons (1969).', source: WP('Perceptrons_(book)') },
    ],
    mood: { text: 'For neural networks, funding "was difficult to find in the 1970s and early 1980s".', source: WP('AI_winter') },
    story: {
      before: { text: 'A threshold neuron with hand-set weights can compute AND and OR, but no single one computes XOR.', source: WP('Artificial_neuron') },
      this: { text: 'The perceptron learns its own weights from examples.', source: WP('Perceptron') },
      after: { text: 'Perceptrons (1969) proves what single-layer perceptrons cannot do; multilayer networks escape the limit, but nobody yet knew how to train them.', source: WP('AI_winter') },
    },
    dispute: { text: 'The popular story that the book single-handedly killed neural networks is a simplification: Wikipedia\'s article on the book says "there are many mistakes" in that story.', source: WP('Perceptrons_(book)') },
  },
  {
    id: 'prune',
    lessonId: 'network-playground',
    lab: 'Prune it',
    idea: 'Pruning: most of a trained network can go',
    years: [2015, 2018],
    when: '2015 → 2018',
    who: [
      { text: 'Song Han, Huizi Mao and William Dally (Stanford, NVIDIA and Tsinghua), Deep Compression.', source: 'https://arxiv.org/abs/1510.00149' },
      { text: 'Jonathan Frankle and Michael Carbin (MIT CSAIL), The Lottery Ticket Hypothesis.', source: 'https://arxiv.org/abs/1803.03635' },
    ],
    mood: { text: 'After 2012, deep networks were winning benchmarks and "investment and interest in AI boomed", but the winning networks were too big to ship to phones.', source: WP('History_of_artificial_intelligence') },
    story: {
      before: { text: 'AlexNet\'s weights take 240 MB.', source: 'https://arxiv.org/abs/1510.00149' },
      this: { text: 'Pruning, quantisation and coding shrink AlexNet 35×, to 6.9 MB, with no loss of accuracy.', source: 'https://arxiv.org/abs/1510.00149' },
      after: { text: '"Winning tickets": subnetworks under 10–20% of the original size that train as well as the whole network on MNIST and CIFAR-10.', source: 'https://arxiv.org/abs/1803.03635' },
    },
  },
  {
    id: 'descent-golf',
    lessonId: 'gradient-descent',
    lab: 'Descent golf',
    idea: 'Gradient descent',
    years: [1847],
    when: '1847',
    who: [{ text: 'Augustin-Louis Cauchy "first suggested it in 1847".', source: WP('Gradient_descent') }],
    story: {
      before: { text: 'Minimising a function of many variables, for which an exact algebraic solution is rarely available.', source: WP('Gradient_descent') },
      this: { text: 'A rule for walking downhill: take repeated steps against the gradient.', source: WP('Gradient_descent') },
      after: { text: 'The same rule, nearly 180 years later, behind every training run in these labs.', source: WP('Stochastic_gradient_descent') },
    },
  },
  {
    id: 'optimizer-golf',
    lessonId: 'optimizers',
    lab: 'Optimizer golf',
    idea: 'Adaptive optimizers and Adam',
    years: [1960, 2014],
    when: '1960 → 2014',
    who: [
      { text: 'Howard Rosenbrock introduces the banana-shaped test function in 1960.', source: WP('Rosenbrock_function') },
      { text: 'Diederik Kingma and Jimmy Ba, Adam (ICLR 2015).', source: 'https://arxiv.org/abs/1412.6980' },
    ],
    mood: { text: 'After AlexNet\'s 2012 win, "investment and interest in AI boomed".', source: WP('History_of_artificial_intelligence') },
    story: {
      before: { text: 'A test valley where finding the valley is easy but converging to the minimum is hard.', source: WP('Rosenbrock_function') },
      this: { text: 'Adam: momentum plus a per-parameter step size from a running average of squared gradients.', source: 'https://arxiv.org/abs/1412.6980' },
      after: { text: 'Adam becomes the default baseline: the LoRA paper, for example, trains with it.', source: 'https://arxiv.org/abs/2106.09685' },
    },
  },
  {
    id: 'schedule',
    lessonId: 'lr-schedules',
    lab: 'Design a schedule',
    idea: 'Learning-rate schedules and warm restarts',
    years: [2016],
    when: '2016',
    who: [{ text: 'Ilya Loshchilov and Frank Hutter (University of Freiburg), SGDR.', source: 'https://arxiv.org/abs/1608.03983' }],
    mood: { text: 'The post-2012 boom: "investment and interest in AI boomed".', source: WP('History_of_artificial_intelligence') },
    story: {
      before: { text: 'A fixed step size, or one cut by hand at a few chosen moments.', source: 'https://arxiv.org/abs/1608.03983' },
      this: { text: 'Cosine-shaped schedules, with warm restarts, that change the step size smoothly over training.', source: 'https://arxiv.org/abs/1608.03983' },
      after: { text: 'Warm-up then cosine decay becomes a standard recipe, built into training libraries.', source: 'https://huggingface.co/docs/transformers/main_classes/optimizer_schedules' },
    },
  },
  {
    id: 'training-lab',
    lessonId: 'training-lab',
    lab: 'Training-lab challenges',
    idea: 'Universal approximation',
    years: [1988, 1991],
    when: '1988 → 1991',
    who: [{ text: 'George Cybenko gives a rigorous proof for sigmoid networks (March 1988); Kurt Hornik, Maxwell Stinchcombe and Halbert White extend it.', source: WP('Universal_approximation_theorem') }],
    mood: { text: 'In 1987 the LISP-machine market collapsed, and the second AI winter ran to about 2000.', source: WP('AI_winter') },
    story: {
      before: { text: 'No guarantee that a network of simple units could represent an arbitrary function at all.', source: WP('Universal_approximation_theorem') },
      this: { text: 'Proof that one hidden layer with enough neurons can approximate any continuous function as closely as you like.', source: WP('Universal_approximation_theorem') },
      after: { text: 'The open question becomes whether training finds that network, which is exactly the challenges\' lesson: representable is not the same as learnable.', source: WP('Universal_approximation_theorem') },
    },
  },
  {
    id: 'backward-pass',
    lessonId: 'network-from-scratch',
    lab: 'Write the backward pass',
    idea: 'Backpropagation',
    years: [1986],
    when: '1986',
    who: [
      { text: 'David Rumelhart (UC San Diego), Geoffrey Hinton (Carnegie Mellon) and Ronald Williams, "Learning representations by back-propagating errors", Nature.', source: WP('David_Rumelhart') },
      { text: 'Hinton was part of the "Parallel Distributed Processing" group at Carnegie Mellon in the 1980s.', source: WP('Geoffrey_Hinton') },
    ],
    mood: { text: 'The neural-network winter "came to an end in the middle 1980s", when this work revived large-scale interest.', source: WP('AI_winter') },
    story: {
      before: { text: 'No practical way to train the hidden layers of a multilayer network.', source: WP('AI_winter') },
      this: { text: 'Backpropagation: the chain rule run backwards through the network, giving every weight its gradient in one pass.', source: WP('Backpropagation') },
      after: { text: 'Every framework\'s backward(), and every network trained since.', source: WP('Backpropagation') },
    },
    dispute: { text: 'Credit is genuinely shared: Seppo Linnainmaa published reverse-mode automatic differentiation in 1976, Paul Werbos applied it to multilayer networks in the now-standard way in 1982, and others found it independently.', source: WP('Backpropagation') },
  },
  {
    id: 'rescue',
    lessonId: 'vanishing-gradients',
    lab: 'Rescue the deep network',
    idea: 'Vanishing gradients, initialisation and ResNets',
    years: [1991, 2010, 2015],
    when: '1991 → 2010 → 2015',
    who: [
      { text: 'Sepp Hochreiter\'s 1991 diploma thesis at the Technical University of Munich formally identifies the vanishing gradient problem.', source: WP('Vanishing_gradient_problem') },
      { text: 'Xavier Glorot and Yoshua Bengio (Université de Montréal) analyse initialisation (2010).', source: 'https://proceedings.mlr.press/v9/glorot10a.html' },
      { text: 'Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun (Microsoft Research), ResNet (2015).', source: 'https://arxiv.org/abs/1512.03385' },
    ],
    mood: { text: 'Through the 1990s, many researchers and engineers "avoided the term artificial intelligence for fear of being viewed as wild-eyed dreamers".', source: WP('History_of_artificial_intelligence') },
    story: {
      before: { text: 'Deep networks freeze because gradients shrink layer by layer.', source: WP('Vanishing_gradient_problem') },
      this: { text: 'Variance-preserving initialisation (Glorot 2010, He 2015) and residual connections.', source: 'https://arxiv.org/abs/1502.01852' },
      after: { text: 'ResNet trains 152 layers and wins ILSVRC 2015 at 3.57% error; LayerNorm (2016) follows.', source: 'https://arxiv.org/abs/1512.03385' },
    },
  },
  {
    id: 'clinic',
    lessonId: 'diagnosis-lab',
    lab: 'The model clinic',
    idea: 'A written recipe for debugging training',
    years: [2019],
    when: '2019',
    who: [{ text: 'Andrej Karpathy, "A Recipe for Training Neural Networks".', source: 'http://karpathy.github.io/2019/04/25/recipe/' }],
    mood: { text: 'Training had become an engineering craft, and the post\'s warning was that "neural net training fails silently".', source: 'http://karpathy.github.io/2019/04/25/recipe/' },
    story: {
      before: { text: 'Debugging know-how passed between colleagues as folklore.', source: 'http://karpathy.github.io/2019/04/25/recipe/' },
      this: { text: 'A step-by-step diagnostic recipe: look at the data, start simple, overfit one batch, then regularise.', source: 'http://karpathy.github.io/2019/04/25/recipe/' },
      after: { text: 'The same checklist behind this clinic\'s ten cases.', source: 'http://karpathy.github.io/2019/04/25/recipe/' },
    },
  },
  {
    id: 'attention-puzzles',
    lessonId: 'attention',
    lab: 'Attention puzzles',
    idea: 'Attention and the transformer',
    years: [2014, 2017],
    when: '2014 → 2017',
    who: [
      { text: 'Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio (Université de Montréal).', source: 'https://arxiv.org/abs/1409.0473' },
      { text: 'Vaswani and colleagues, "eight scientists and engineers working at Google", Attention Is All You Need.', source: WP('Attention_Is_All_You_Need') },
    ],
    mood: { text: 'In 2017, researchers at Google Brain introduced the transformer architecture.', source: WP('History_of_artificial_intelligence') },
    story: {
      before: { text: 'Squeezing a whole sentence into one fixed-length vector, which Bahdanau et al. conjectured "is a bottleneck".', source: 'https://arxiv.org/abs/1409.0473' },
      this: { text: 'A soft search over the source words: attention.', source: 'https://arxiv.org/abs/1409.0473' },
      after: { text: 'The transformer, "dispensing with recurrence and convolutions entirely" (28.4 BLEU on English→German).', source: 'https://arxiv.org/abs/1706.03762' },
    },
  },
  {
    id: 'previous-word',
    lessonId: 'positions',
    lab: 'Build a previous-word head',
    idea: 'Rotary positions and induction heads',
    years: [2021, 2022],
    when: '2021 → 2022',
    who: [
      { text: 'Jianlin Su and colleagues, RoFormer (rotary position embedding).', source: 'https://arxiv.org/abs/2104.09864' },
      { text: 'Catherine Olsson and colleagues at Anthropic, In-context Learning and Induction Heads.', source: 'https://arxiv.org/abs/2209.11895' },
    ],
    story: {
      before: { text: 'Position added to each word as a fixed vector.', source: 'https://arxiv.org/abs/2104.09864' },
      this: { text: 'Rotary embeddings encode "absolute position with a rotation matrix" with "explicit relative position dependency".', source: 'https://arxiv.org/abs/2104.09864' },
      after: { text: 'Induction heads, built on previous-token heads, "might constitute the mechanism for the majority of all in-context learning" (the authors call their evidence preliminary).', source: 'https://arxiv.org/abs/2209.11895' },
    },
  },
  {
    id: 'compress',
    lessonId: 'tokenization',
    lab: 'Compress the corpus',
    idea: 'Byte-pair encoding',
    years: [1994, 2015],
    when: '1994 → 2015',
    who: [
      { text: 'Philip Gage describes byte-pair encoding as a data-compression method (1994).', source: WP('Byte-pair_encoding') },
      { text: 'Rico Sennrich, Barry Haddow and Alexandra Birch (University of Edinburgh) apply it to translation (ACL 2016).', source: 'https://arxiv.org/abs/1508.07909' },
    ],
    story: {
      before: { text: 'A file-compression trick: repeatedly replace the most common pair of bytes with a new symbol.', source: WP('Byte-pair_encoding') },
      this: { text: 'Subword units for "open-vocabulary" neural translation.', source: 'https://arxiv.org/abs/1508.07909' },
      after: { text: 'BPE tokenizers in GPT-2 and its successors.', source: WP('Byte-pair_encoding') },
    },
  },
  {
    id: 'shannon',
    lessonId: 'next-token-prediction',
    lab: "Shannon's betting game",
    idea: 'Information theory and the entropy of English',
    years: [1948, 1951],
    when: '1948 → 1951',
    who: [{ text: 'Claude Shannon, Bell Labs.', source: WP('Claude_Shannon') }],
    mood: { text: 'Shannon went on to be one of the four organisers of the 1956 Dartmouth workshop that named the field "artificial intelligence".', source: WP('Dartmouth_workshop') },
    story: {
      before: { text: 'No way to measure how much information a message carries.', source: WP('A_Mathematical_Theory_of_Communication') },
      this: { text: 'Information theory (1948), then "Prediction and Entropy of Printed English" (1951): people guess the next letter, and long-range context brings English down to "the order of one bit per letter".', source: 'https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf' },
      after: { text: 'Next-token prediction, scored in bits, is how every large language model is trained.', source: WP('Large_language_model') },
    },
  },
  {
    id: 'sampler',
    lessonId: 'sampling',
    lab: 'Tune the sampler',
    idea: 'Nucleus (top-p) sampling',
    years: [2019],
    when: '2019',
    who: [{ text: 'Ari Holtzman and colleagues (University of Washington and the Allen Institute for AI), The Curious Case of Neural Text Degeneration.', source: 'https://arxiv.org/abs/1904.09751' }],
    story: {
      before: { text: 'Decoding by likelihood "leads to text that is bland and strangely repetitive".', source: 'https://arxiv.org/abs/1904.09751' },
      this: { text: 'Nucleus sampling: sample only from the smallest set of tokens holding most of the probability.', source: 'https://arxiv.org/abs/1904.09751' },
      after: { text: 'top_p becomes a standard parameter of language-model APIs and libraries.', source: 'https://huggingface.co/docs/transformers/main_classes/text_generation' },
    },
  },
  {
    id: 'compute-budget',
    lessonId: 'scaling-laws',
    lab: 'Spend the compute budget',
    idea: 'Scaling laws and Chinchilla',
    years: [2020, 2022, 2024],
    when: '2020 → 2022 → 2024',
    who: [
      { text: 'Jared Kaplan and colleagues (OpenAI and Johns Hopkins), Scaling Laws for Neural Language Models.', source: 'https://arxiv.org/abs/2001.08361' },
      { text: 'Jordan Hoffmann and colleagues at DeepMind, Chinchilla.', source: 'https://arxiv.org/abs/2203.15556' },
      { text: 'Meta AI, Llama 3 (April 2024).', source: WP('Llama_(language_model)') },
    ],
    mood: { text: 'A race to size: Chinchilla\'s comparisons included models of up to 530B parameters.', source: 'https://arxiv.org/abs/2203.15556' },
    story: {
      before: { text: 'Loss follows power laws in model size, data and compute, some spanning "more than seven orders of magnitude".', source: 'https://arxiv.org/abs/2001.08361' },
      this: { text: 'Chinchilla (70B parameters, 1.4T tokens) outperforms Gopher (280B) and GPT-3 (175B) with the same compute.', source: 'https://arxiv.org/abs/2203.15556' },
      after: { text: 'Llama 3 trains small models on about 15 trillion tokens, far past compute-optimal, to make them cheap to serve.', source: WP('Llama_(language_model)') },
    },
  },
  {
    id: 'lora',
    lessonId: 'lora',
    lab: 'LoRA rank budget',
    idea: 'Adapters, LoRA and QLoRA',
    years: [2019, 2021, 2023],
    when: '2019 → 2021 → 2023',
    who: [
      { text: 'Neil Houlsby and colleagues, Parameter-Efficient Transfer Learning for NLP (adapters).', source: 'https://arxiv.org/abs/1902.00751' },
      { text: 'Edward Hu and colleagues (Microsoft), LoRA.', source: 'https://arxiv.org/abs/2106.09685' },
      { text: 'Tim Dettmers and colleagues, QLoRA.', source: 'https://arxiv.org/abs/2305.14314' },
    ],
    mood: { text: 'GPT-3 had 175B parameters, and deploying a separately fine-tuned copy for every task was, in the LoRA authors\' words, "prohibitively expensive".', source: 'https://arxiv.org/abs/2106.09685' },
    story: {
      before: { text: 'Adapters reach within 0.4% of full fine-tuning while adding 3.6% parameters per task.', source: 'https://arxiv.org/abs/1902.00751' },
      this: { text: 'LoRA cuts GPT-3\'s trainable parameters 10,000× and GPU memory 3×.', source: 'https://arxiv.org/abs/2106.09685' },
      after: { text: 'QLoRA fine-tunes a 65B model on a single 48 GB GPU.', source: 'https://arxiv.org/abs/2305.14314' },
    },
  },
  {
    id: 'alignment',
    lessonId: 'rlhf-dpo',
    lab: 'Tune the alignment pipeline',
    idea: 'RLHF and DPO',
    years: [2017, 2022, 2023],
    when: '2017 → 2022 → 2023',
    who: [
      { text: 'Paul Christiano and colleagues, Deep Reinforcement Learning from Human Preferences.', source: 'https://arxiv.org/abs/1706.03741' },
      { text: 'Long Ouyang and colleagues (OpenAI), InstructGPT.', source: 'https://arxiv.org/abs/2203.02155' },
      { text: 'Rafael Rafailov and colleagues (Stanford), Direct Preference Optimization.', source: 'https://arxiv.org/abs/2305.18290' },
    ],
    mood: { text: 'Goodhart\'s law, in Marilyn Strathern\'s 1997 words: "When a measure becomes a target, it ceases to be a good measure."', source: WP("Goodhart's_law") },
    story: {
      before: { text: 'Learning from human preferences between pairs, with feedback on less than 1% of the agent\'s interactions.', source: 'https://arxiv.org/abs/1706.03741' },
      this: { text: 'People prefer the 1.3B-parameter InstructGPT to the 175B GPT-3.', source: 'https://arxiv.org/abs/2203.02155' },
      after: { text: 'DPO replaces "complex and often unstable" RLHF with "a simple classification loss".', source: 'https://arxiv.org/abs/2305.18290' },
    },
  },
  {
    id: 'quant',
    lessonId: 'quantization',
    lab: 'Mixed-precision quantisation',
    idea: 'LLM quantisation: LLM.int8(), GPTQ and k-quants',
    years: [2022, 2023],
    when: '2022 → 2023',
    who: [
      { text: 'Tim Dettmers and colleagues, LLM.int8().', source: 'https://arxiv.org/abs/2208.07339' },
      { text: 'Elias Frantar and colleagues, GPTQ.', source: 'https://arxiv.org/abs/2210.17323' },
      { text: 'Iwan Kawrakow, llama.cpp k-quants.', source: 'https://github.com/ggml-org/llama.cpp/pull/1684' },
    ],
    story: {
      before: { text: '8-bit inference for models up to 175B "without any performance degradation", once emergent outlier features are handled.', source: 'https://arxiv.org/abs/2208.07339' },
      this: { text: 'GPTQ quantises 175B models to 3–4 bits in about four GPU hours.', source: 'https://arxiv.org/abs/2210.17323' },
      after: { text: 'k-quants, including Q4_K_M, are merged into llama.cpp on 5 June 2023, bringing mixed-precision files to laptops.', source: 'https://github.com/ggml-org/llama.cpp/pull/1684' },
    },
  },
  {
    id: 'vram',
    lessonId: 'kv-cache',
    lab: 'VRAM Tetris',
    idea: 'Shared KV heads and PagedAttention',
    years: [2019, 2023],
    when: '2019 → 2023',
    who: [
      { text: 'Noam Shazeer, multi-query attention.', source: 'https://arxiv.org/abs/1911.02150' },
      { text: 'Joshua Ainslie and colleagues, grouped-query attention (GQA).', source: 'https://arxiv.org/abs/2305.13245' },
      { text: 'Woosuk Kwon and colleagues (UC Berkeley, with Stanford and UC San Diego), vLLM and PagedAttention.', source: 'https://arxiv.org/abs/2309.06180' },
    ],
    story: {
      before: { text: 'Sharing keys and values across heads cuts the "memory bandwidth requirements of incremental decoding".', source: 'https://arxiv.org/abs/1911.02150' },
      this: { text: 'GQA keeps quality "close to multi-head" attention at speed comparable to multi-query.', source: 'https://arxiv.org/abs/2305.13245' },
      after: { text: 'PagedAttention gives 2–4× the throughput of FasterTransformer and Orca.', source: 'https://arxiv.org/abs/2309.06180' },
    },
  },
  {
    id: 'wire-policy',
    lessonId: 'mdps',
    lab: 'Wire the Policy',
    idea: 'Dynamic programming and the Bellman equation',
    years: [1953, 1957],
    when: '1953 → 1957',
    who: [{ text: 'Richard Bellman, at the RAND Corporation from 1949.', source: WP('Richard_Bellman') }],
    story: {
      before: { text: 'No general method for planning a sequence of decisions under uncertainty.', source: WP('Bellman_equation') },
      this: { text: 'Dynamic programming (1953), then the book Dynamic Programming (1957) and the Bellman equation.', source: WP('Bellman_equation') },
      after: { text: 'The value backups at the heart of every reinforcement-learning method.', source: WP('Bellman_equation') },
    },
  },
  {
    id: 'q-learner',
    lessonId: 'q-learning',
    lab: 'Tune the Q-Learner',
    idea: 'TD learning, Q-learning and DQN',
    years: [1988, 1989, 2013, 2015],
    when: '1988 → 1989 → 2013/2015',
    who: [
      { text: 'Richard Sutton, temporal-difference learning (1988).', source: WP('Temporal_difference_learning') },
      { text: 'Chris Watkins, Q-learning (1989; convergence proof with Peter Dayan, 1992).', source: WP('Q-learning') },
      { text: 'Volodymyr Mnih and colleagues at DeepMind, London.', source: 'https://arxiv.org/abs/1312.5602' },
    ],
    mood: { text: '1989 sits inside the second AI winter (about 1987–2000).', source: WP('AI_winter') },
    story: {
      before: { text: 'Temporal-difference learning updates a guess from the next guess, without waiting for the final outcome.', source: WP('Temporal_difference_learning') },
      this: { text: 'Q-learning: learn the value of each action from experience, without a model of the world.', source: WP('Q-learning') },
      after: { text: 'Deep Q-networks learn Atari games from raw pixels (2013); by 2015 they reach "a level comparable to that of a professional human games tester across a set of 49 games".', source: 'https://www.nature.com/articles/nature14236' },
    },
  },
  {
    id: 'policy-gradient',
    lessonId: 'policy-gradients',
    lab: 'Stabilise the policy gradient',
    idea: 'REINFORCE and PPO',
    years: [1992, 2017],
    when: '1992 → 2017',
    who: [
      { text: 'Ronald J. Williams, REINFORCE.', source: WP('Policy_gradient_method') },
      { text: 'John Schulman and colleagues (OpenAI), Proximal Policy Optimization.', source: 'https://arxiv.org/abs/1707.06347' },
    ],
    mood: { text: 'Inside the second AI winter (about 1987–2000).', source: WP('AI_winter') },
    story: {
      before: { text: 'REINFORCE, "the first policy gradient method", whose estimates are very noisy.', source: WP('Policy_gradient_method') },
      this: { text: 'Baselines that cut the variance without biasing the gradient.', source: WP('Policy_gradient_method') },
      after: { text: 'PPO\'s clipped objective "enables multiple epochs of minibatch updates", and became the workhorse of RLHF.', source: 'https://arxiv.org/abs/1707.06347' },
    },
  },
  {
    id: 'reward-hacking',
    lessonId: 'reward-hacking',
    lab: 'Reward-Hacking Sandbox',
    idea: 'Reward shaping and specification gaming',
    years: [1975, 1999, 2016],
    when: '1975 · 1999 · 2016',
    who: [
      { text: 'Charles Goodhart, on UK monetary policy (1975).', source: WP("Goodhart's_law") },
      { text: 'Andrew Ng, Daishi Harada and Stuart Russell (UC Berkeley), potential-based reward shaping.', source: 'https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf' },
      { text: 'OpenAI, "Faulty reward functions in the wild" (21 December 2016).', source: 'https://openai.com/index/faulty-reward-functions/' },
    ],
    mood: { text: 'Goodhart\'s law began as a 1975 point about UK monetary policy, long before anyone trained an agent.', source: WP("Goodhart's_law") },
    story: {
      before: { text: 'Hand-added shaping rewards, which cause "well-known bugs".', source: 'https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf' },
      this: { text: 'Potential-based shaping provably keeps the optimal policy unchanged.', source: 'https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf' },
      after: { text: 'The CoastRunners boat circles a lagoon knocking over targets and scores 20% higher than human players without finishing the race.', source: 'https://openai.com/index/faulty-reward-functions/' },
    },
  },
  {
    id: 'linear-regression',
    lessonId: 'linear-regression',
    lab: 'Feature Engineering + Logistic Regression',
    idea: 'Least squares, logistic regression and Ridge',
    years: [1805, 1944, 1958, 1970],
    when: '1805 · 1944 · 1958 · 1970',
    who: [
      { text: 'Adrien-Marie Legendre publishes the method of least squares in Paris (1805); Carl Friedrich Gauss publishes it in 1809 after using it at age 24 to relocate the lost asteroid Ceres in 1801.', source: WP('Least_squares') },
      { text: 'Joseph Berkson coins "logit" in 1944 and champions the logistic model; David Cox refines binary logistic regression in 1958.', source: WP('Logistic_regression') },
      { text: 'Arthur Hoerl and Robert Kennard introduce Ridge regression in Technometrics (1970) to stabilise collinear least squares.', source: WP('Ridge_regression') },
    ],
    mood: { text: 'Berkson\'s logit model "was initially dismissed as inferior to the probit model", only reaching equal footing in statistics journals between 1960 and 1970.', source: WP('Logistic_regression') },
    story: {
      before: { text: 'Least squares fits continuous curves (such as planetary orbits) in one linear solve, but breaks on binary outcomes and collinear features.', source: WP('Least_squares') },
      this: { text: 'Logistic regression models log-odds linearly through a sigmoid, while Ridge shifts the diagonal of XᵀX by λI to tame near-singular matrices.', source: WP('Ridge_regression') },
      after: { text: 'Paired with engineered features, regularised logistic regression became the fast, convex baseline across medicine, credit scoring and large-scale ad prediction.', source: WP('Logistic_regression') },
    },
    dispute: { text: 'Legendre published least squares first in 1805, while Gauss claimed in 1809 to have used it since 1795 — sparking a famous priority dispute (though Gauss went further by tying it to the normal distribution).', source: WP('Least_squares') },
  },
  {
    id: 'knn',
    lessonId: 'knn',
    lab: 'Rescue the kNN Classifier',
    idea: 'Nearest neighbours and the curse of dimensionality',
    years: [1951, 1957, 1967],
    when: '1951 · 1957 · 1967',
    who: [
      { text: 'Evelyn Fix and Joseph Hodges develop non-parametric nearest-neighbour discrimination in a 1951 report for the USAF School of Aviation Medicine in Texas.', source: WP('K-nearest_neighbors_algorithm') },
      { text: 'Richard Bellman coins "the curse of dimensionality" in Dynamic Programming (1957, RAND Corporation).', source: WP('Curse_of_dimensionality') },
      { text: 'Thomas Cover and Peter Hart publish "Nearest neighbor pattern classification" in IEEE Transactions on Information Theory (1967).', source: WP('K-nearest_neighbors_algorithm') },
    ],
    story: {
      before: { text: 'Classical discriminant analysis assumed each class followed a clean parametric distribution such as a Gaussian.', source: WP('K-nearest_neighbors_algorithm') },
      this: { text: 'Fix and Hodges classify by majority vote of the k closest stored examples; Cover and Hart prove that even 1-NN asymptotically has an error rate no worse than twice the optimal Bayes error rate.', source: WP('K-nearest_neighbors_algorithm') },
      after: { text: 'Raw-feature kNN hits Bellman\'s curse of dimensionality when noisy columns accumulate, so modern systems run approximate nearest-neighbour search over learned neural embeddings instead.', source: WP('Curse_of_dimensionality') },
    },
  },
];

/** Era bands for the timeline: the mood of each period, sourced. */
export type Era = { from: number; to: number; label: string; mood: Claim; kind: 'winter' | 'boom' | 'founding' };

export const ERAS: Era[] = [
  {
    from: 1956,
    to: 1974,
    label: 'The founding years',
    kind: 'founding',
    mood: { text: 'The 1956 Dartmouth workshop names the field "artificial intelligence", and early optimism runs high: Rosenblatt predicted the perceptron "may eventually be able to learn, make decisions, and translate languages".', source: WP('AI_winter') },
  },
  {
    from: 1974,
    to: 1980,
    label: 'First AI winter',
    kind: 'winter',
    mood: { text: 'Funding for neural-network work "was difficult to find in the 1970s and early 1980s".', source: WP('AI_winter') },
  },
  {
    from: 1987,
    to: 2000,
    label: 'Second AI winter',
    kind: 'winter',
    mood: { text: 'After the LISP-machine market collapsed in 1987, many engineers "avoided the term artificial intelligence for fear of being viewed as wild-eyed dreamers".', source: WP('History_of_artificial_intelligence') },
  },
  {
    from: 2012,
    to: 2026,
    label: 'The deep-learning boom',
    kind: 'boom',
    mood: { text: 'After AlexNet\'s 2012 win, "investment and interest in AI boomed".', source: WP('History_of_artificial_intelligence') },
  },
];

/** Field-wide milestones that aren't a lab's own idea but anchor the timeline. */
export const MILESTONES: { year: number; label: string; source: string }[] = [
  { year: 1956, label: 'Dartmouth workshop names "artificial intelligence"', source: WP('Dartmouth_workshop') },
  { year: 2012, label: 'AlexNet wins ImageNet', source: WP('History_of_artificial_intelligence') },
];

export function historyById(id: string): HistoryEntry | undefined {
  return HISTORY.find((h) => h.id === id);
}

/** Every claim in an entry, in reading order (who, mood, story, dispute). */
export function claimsOf(h: HistoryEntry): Claim[] {
  return [...h.who, ...(h.mood ? [h.mood] : []), h.story.before, h.story.this, h.story.after, ...(h.dispute ? [h.dispute] : [])];
}

// ------------------------------------------------------------------ Timeline layout (pure, tested)

/** The overview axis: a compressed "earlier" stub for pre-1940 ideas, then 1940 → 2026 linearly. */
export const TIMELINE = { stubWidth: 70, start: 1940, end: 2026, width: 900 } as const;

export function timelineX(year: number): number {
  const { stubWidth, start, end, width } = TIMELINE;
  if (year < start) {
    // pre-1940 ideas (1733 → 1847) share the stub, in order
    const t = (year - 1700) / (start - 1700);
    return 8 + Math.max(0, Math.min(1, t)) * (stubWidth - 20);
  }
  return stubWidth + ((year - start) / (end - start)) * (width - stubWidth - 12);
}

/**
 * Stack labelled dots into lanes so no two in a lane are closer than `gap`
 * pixels: the first lane with room wins (greedy, in x order).
 */
export function assignLanes(xs: number[], gap: number): number[] {
  const order = xs.map((x, i) => ({ x, i })).sort((a, b) => a.x - b.x);
  const laneEnd: number[] = [];
  const lanes = new Array<number>(xs.length);
  for (const { x, i } of order) {
    let lane = laneEnd.findIndex((e) => x - e >= gap);
    if (lane === -1) {
      lane = laneEnd.length;
      laneEnd.push(x);
    } else laneEnd[lane] = x;
    lanes[i] = lane;
  }
  return lanes;
}
