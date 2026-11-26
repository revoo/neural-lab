/**
 * The curriculum: modules (milestones) and their lessons.
 * Lessons are lazily loaded, so adding a lesson costs nothing until someone opens it.
 * To add a lesson: create src/lessons/<id>/index.mdx and add an entry here.
 */
import type { ComponentType } from 'react';

export type LessonMeta = {
  id: string;
  title: string;
  summary: string;
  minutes: number;
  load?: () => Promise<{ default: ComponentType }>;
};

export type ModuleMeta = {
  id: string;
  number: number;
  /**
   * Approximate span of the module's central ideas, e.g. "1943 to 1969". Segments are
   * joined with ", then "; "today" and decades ("1960s") are allowed. Every history
   * card of the module's lessons falls inside it (checked in curriculum.test.ts).
   */
  era?: string;
  title: string;
  blurb: string;
  lessons: LessonMeta[];
  /** Lessons not yet written, shown as the roadmap. */
  planned?: string[];
};

export const MODULES: ModuleMeta[] = [
  {
    id: 'prologue',
    number: 0,
    era: '1950 to today',
    title: 'Prologue',
    blurb: 'Before any maths: the question, the people, the hype, the winters, and the machine you are about to build.',
    lessons: [
      { id: 'prologue', title: 'Can a Machine Learn?', summary: 'Seventy years of bold promises, dead ends and comebacks, and a working learning machine you can watch in your browser before we take it apart.', minutes: 12, load: () => import('./lessons/prologue/index.mdx') },
    ],
  },
  {
    id: 'foundations',
    number: 1,
    era: '1670s to 1990',
    title: 'The Maths Toolkit',
    blurb: 'Nine tools that every neural network is built from. Each lesson ends by showing exactly where the next modules use it — and each has a lab of its own.',
    lessons: [
      { id: 'vectors', title: 'Vectors: Lists That Point Somewhere', summary: 'Why a list of numbers can also be an arrow, and why that double life matters for AI.', minutes: 30, load: () => import('./lessons/vectors/index.mdx') },
      { id: 'dot-product', title: 'The Dot Product: Measuring Agreement', summary: 'The single most important operation in deep learning, and the reason it measures alignment.', minutes: 35, load: () => import('./lessons/dot-product/index.mdx') },
      { id: 'matrices', title: 'Matrices: Machines That Transform Space', summary: 'A matrix as a recipe for moving every point in space — and how to read that recipe at a glance.', minutes: 35, load: () => import('./lessons/matrices/index.mdx') },
      { id: 'matrix-multiplication', title: 'Matrix Multiplication, Shapes & Tensors', summary: 'Many dot products at once, the shape rules that keep them straight, and what a tensor really is.', minutes: 40, load: () => import('./lessons/matrix-multiplication/index.mdx') },
      { id: 'derivatives', title: 'Derivatives: Measuring Sensitivity', summary: 'If I nudge this input a little, how much does the output move? That question is the derivative.', minutes: 35, load: () => import('./lessons/derivatives/index.mdx') },
      { id: 'gradients', title: 'Partial Derivatives & Gradients', summary: 'Sensitivity when there are many knobs — and the arrow that points uphill.', minutes: 35, load: () => import('./lessons/gradients/index.mdx') },
      { id: 'chain-rule', title: 'The Chain Rule: Sensitivity Through a Pipeline', summary: 'How sensitivities multiply through a chain of functions — the engine underneath backpropagation.', minutes: 35, load: () => import('./lessons/chain-rule/index.mdx') },
      { id: 'probability', title: 'Probability Distributions & Softmax', summary: 'Turning arbitrary scores into well-behaved probabilities, and what temperature actually does.', minutes: 35, load: () => import('./lessons/probability/index.mdx') },
      { id: 'information', title: 'Logarithms, Surprise & Cross-Entropy', summary: 'Why “how surprised were you?” is the loss function behind every classifier and language model.', minutes: 40, load: () => import('./lessons/information/index.mdx') },
    ],
  },
  {
    id: 'neuron',
    number: 2,
    era: '1943 to 1969',
    title: 'The Artificial Neuron',
    blurb: 'The tiny computational unit everything else is assembled from.',
    lessons: [
      { id: 'neuron', title: 'The Artificial Neuron', summary: 'Weighted sum, bias, activation — and why a single neuron is really a line drawn through space.', minutes: 25, load: () => import('./lessons/neuron/index.mdx') },
      { id: 'activation-functions', title: 'Why Networks Need Activation Functions', summary: 'Stack a hundred linear layers and you still have one. Then add a kink, and everything changes.', minutes: 25, load: () => import('./lessons/activation-functions/index.mdx') },
      { id: 'logic-lab', title: 'Lab: Wiring Logic by Hand', summary: 'A puzzle workbench: build AND, OR and finally XOR out of neurons whose weights you set yourself.', minutes: 30, load: () => import('./lessons/logic-lab/index.mdx') },
      { id: 'perceptron', title: 'The Perceptron: A Machine That Learns Its Own Weights', summary: 'A machine with motor-driven knobs that learned to tell shapes apart: the simple rule it used, why it always works when a line exists, and where it fails for ever.', minutes: 30, load: () => import('./lessons/perceptron/index.mdx') },
      { id: 'interlude-first-winter', title: 'Interlude: The First Winter (1969–1980)', summary: 'How the excitement around perceptrons cooled into the first AI winter: Minsky and Papert’s proof, the Lighthill report, hardware limits, and why symbolic AI won the 1970s.', minutes: 8, load: () => import('./lessons/interlude-first-winter/index.mdx') },
    ],
  },
  {
    id: 'architecture',
    number: 3,
    era: '1960s to 2018',
    title: 'Neural Network Architecture',
    blurb: 'Wiring neurons into layers, and layers into networks you can inspect neuron by neuron.',
    lessons: [
      { id: 'forward-propagation', title: 'Layers and the Forward Pass', summary: 'How neurons are organised into layers, and how an input flows through them — one matrix multiplication at a time.', minutes: 25, load: () => import('./lessons/forward-propagation/index.mdx') },
      { id: 'network-playground', title: 'The Network Playground', summary: 'Train a network, then take it apart: what each neuron responds to, what breaks without it, and how much of it you really need.', minutes: 30, load: () => import('./lessons/network-playground/index.mdx') },
      { id: 'depth-and-width', title: 'Depth, Width and Parameter Counts', summary: 'What a network costs, and why stacking layers can buy exponentially more than widening them.', minutes: 25, load: () => import('./lessons/depth-and-width/index.mdx') },
    ],
  },
  {
    id: 'optimization',
    number: 4,
    era: '1733 to 2016',
    title: 'Loss Functions & Optimization',
    blurb: 'How a model measures its mistakes, and how it walks downhill to make fewer of them.',
    lessons: [
      { id: 'gradient-descent', title: 'Gradient Descent: Walking Downhill in the Fog', summary: 'The algorithm behind nearly all of modern AI training, and why the step size can make or break it.', minutes: 25, load: () => import('./lessons/gradient-descent/index.mdx') },
      { id: 'loss-functions', title: 'Loss Functions: What Does “Wrong” Mean?', summary: 'Squared error, absolute error and cross-entropy — what each one really asks the model for, and why the choice matters.', minutes: 25, load: () => import('./lessons/loss-functions/index.mdx') },
      { id: 'statistics', title: 'Statistics: Mean, Variance & the Bell Curve', summary: 'Spread as an average of squares, why variances add, where the bell curve comes from, how much a test score wobbles — and Lab: The Dice Workshop.', minutes: 30, load: () => import('./lessons/statistics/index.mdx') },
      { id: 'sgd-minibatch', title: 'Stochastic Gradient Descent and Mini-Batches', summary: 'Why a noisy gradient from a handful of examples beats an exact one from millions.', minutes: 25, load: () => import('./lessons/sgd-minibatch/index.mdx') },
      { id: 'optimizers', title: 'Momentum, RMSProp and Adam', summary: 'Two simple fixes for gradient descent’s weaknesses, and the optimizer that combines them — racing on the same landscapes.', minutes: 30, load: () => import('./lessons/optimizers/index.mdx') },
      { id: 'lr-schedules', title: 'Learning-Rate Schedules', summary: 'Fast or precise? Why the best step size changes during training — and how to design a schedule yourself.', minutes: 25, load: () => import('./lessons/lr-schedules/index.mdx') },
    ],
  },
  {
    id: 'backprop',
    number: 5,
    era: '1970 to 2015',
    title: 'Backpropagation',
    blurb: 'Tracing an error backwards through a network to find out how every weight should change.',
    lessons: [
      { id: 'backpropagation', title: 'Backpropagation, One Step at a Time', summary: 'Walk a real computational graph forwards and backwards, number by number.', minutes: 35, load: () => import('./lessons/backpropagation/index.mdx') },
      { id: 'network-from-scratch', title: 'A Neural Network from Scratch', summary: 'A whole network in fifteen lines of matrix code — and a lab where you write the backward pass yourself, checked against measured gradients.', minutes: 35, load: () => import('./lessons/network-from-scratch/index.mdx') },
      { id: 'vanishing-gradients', title: 'Vanishing & Exploding Gradients', summary: 'Why deep networks freeze or blow up to NaN, how Xavier and He initialisation conserve variance, why residual skips rescue 32-layer stacks — and a deep-stack rescue lab.', minutes: 35, load: () => import('./lessons/vanishing-gradients/index.mdx') },
    ],
  },
  {
    id: 'training',
    number: 6,
    era: '1988 to today',
    title: 'Neural Network Training Lab',
    blurb: 'A full browser-based training sandbox with live decision boundaries.',
    lessons: [
      { id: 'training-lab', title: 'The Training Laboratory', summary: 'Every piece on the bench at once: data, features, architecture and settings — train live, compare runs, and take on the challenges.', minutes: 40, load: () => import('./lessons/training-lab/index.mdx') },
    ],
  },
  {
    id: 'generalization',
    number: 7,
    era: '1990s to 2019',
    title: 'Generalization & Failure Modes',
    blurb: 'Why a model that aces its training data can still be useless.',
    lessons: [
      { id: 'overfitting', title: 'Overfitting and Underfitting', summary: 'Why a perfect score on the training data can mean nothing, the two ways to be wrong, and how data and training time move the line between them.', minutes: 30, load: () => import('./lessons/overfitting/index.mdx') },
      { id: 'regularization', title: 'Regularisation: Early Stopping, Weight Decay and Dropout', summary: 'Three ways to keep a big network and lose its bad habits — and a fair race between them.', minutes: 35, load: () => import('./lessons/regularization/index.mdx') },
      { id: 'diagnosis-lab', title: 'Lab: The Model Clinic', summary: 'Twelve broken training runs (two with two causes), the instruments to examine them, and a menu of root causes. Diagnose each one, then fix it.', minutes: 45, load: () => import('./lessons/diagnosis-lab/index.mdx') },
    ],
  },
  {
    id: 'classical',
    number: 8,
    era: '1805 to 2022',
    title: 'Classical Machine Learning',
    blurb: 'Trees, nearest neighbours, SVMs, clustering — and when they beat neural networks.',
    lessons: [
      { id: 'interlude-second-winter', title: 'Interlude: When Neural Networks Were Unfashionable (1987–2006)', summary: 'After the 1987 LISP-machine crash, support vector machines and random forests ruled the 1990s while a small group kept neural nets alive. Why those classical methods still matter.', minutes: 8, load: () => import('./lessons/interlude-second-winter/index.mdx') },
      { id: 'high-dimensions', title: 'High-Dimensional Geometry', summary: 'Why almost all of an orange is peel in 768 dimensions, why random arrows are perpendicular, why neighbours look alike — and Lab: Superposition.', minutes: 30, load: () => import('./lessons/high-dimensions/index.mdx') },
      { id: 'linear-regression', title: 'Linear and Logistic Regression', summary: 'The oldest learning machine: fit a line by least squares, see why it fails at yes-or-no questions, fix it with logistic regression — and a feature-engineering lab.', minutes: 35, load: () => import('./lessons/linear-regression/index.mdx') },
      { id: 'svm', title: 'Support Vector Machines: The Widest Street', summary: 'Many lines separate the data; which one should you trust? The widest street, the few points that decide it, and a trick for drawing curves with straight lines.', minutes: 35, load: () => import('./lessons/svm/index.mdx') },
      { id: 'knn', title: 'k-Nearest Neighbours', summary: 'Classify by asking the neighbours: no training at all, one dial that trades memorising for smoothing, and the two ways it quietly breaks — and Lab: Rescue the kNN Classifier.', minutes: 30, load: () => import('./lessons/knn/index.mdx') },
      { id: 'decision-trees', title: 'Decision Trees and Random Forests', summary: 'Twenty questions as a model: how a tree picks its questions, why one tree overfits and a forest of them doesn’t — and Lab: Build the Tabular Ensemble.', minutes: 35, load: () => import('./lessons/decision-trees/index.mdx') },
      { id: 'clustering-pca', title: 'k-Means and PCA', summary: 'Finding structure with no labels at all: grouping points into clusters, squashing many features into the few that matter — and Lab: Compress & Segment.', minutes: 35, load: () => import('./lessons/clustering-pca/index.mdx') },
    ],
  },
  {
    id: 'vision',
    number: 9,
    era: '1949 to 2017',
    title: 'Computer Vision',
    blurb: 'From pixels to convolutions to object detectors.',
    lessons: [
      { id: 'interlude-big-bang', title: 'Interlude: Data, Graphics Cards and 2012', summary: 'How ImageNet, programmable graphics cards, and one 2012 convolutional network turned ideas from 1989 into the deep-learning boom.', minutes: 8, load: () => import('./lessons/interlude-big-bang/index.mdx') },
      { id: 'images-as-tensors', title: 'Images as Tensors', summary: 'What a photo looks like to a network: grids of numbers, colour channels, and why shrinking an image the naive way invents patterns — and Lab: Build the Vision Preprocessor.', minutes: 30, load: () => import('./lessons/images-as-tensors/index.mdx') },
      { id: 'convolution', title: 'Convolution and Kernels', summary: 'Slide one small pattern-detector across a whole image: edge finders, blurs, and the arithmetic of output sizes — and Lab: Design the Kernel.', minutes: 35, load: () => import('./lessons/convolution/index.mdx') },
      { id: 'cnns', title: 'CNNs and Feature Maps', summary: 'Stack small detectors into big ones: how each layer sees a wider patch, and why a tiny CNN beats a much bigger plain network — and Lab: Architect the CNN.', minutes: 35, load: () => import('./lessons/cnns/index.mdx') },
      { id: 'object-detection', title: 'Object Detection', summary: 'Not just what is in the picture but where: drawing boxes, scoring overlaps, removing duplicates in a crowd — and Lab: Tune the Object Detector.', minutes: 35, load: () => import('./lessons/object-detection/index.mdx') },
    ],
  },
  {
    id: 'embeddings',
    number: 10,
    era: '1954 to 2021',
    title: 'Embeddings & Representations',
    blurb: 'Meaning as geometry: learned spaces where similar things sit close together.',
    lessons: [
      { id: 'one-hot-vs-dense', title: 'One-Hot vs Dense Vectors', summary: 'Why giving every word its own switch makes all words strangers, and how dense vectors let similar words share what they learn — and Lab: Choose & Size the Representation.', minutes: 30, load: () => import('./lessons/one-hot-vs-dense/index.mdx') },
      { id: 'word-embeddings', title: 'Word Embeddings', summary: 'Learn what words mean from nothing but the company they keep, and watch analogies like king to queen emerge — and Lab: Train & Debug Word2Vec.', minutes: 35, load: () => import('./lessons/word-embeddings/index.mdx') },
      { id: 'contrastive-learning', title: 'Contrastive Learning & Representation Spaces', summary: 'Learn representations without labels by pulling matching pairs together and pushing everything else apart, and why skipping the push collapses everything to one point — and Lab: Tune Contrastive Training.', minutes: 35, load: () => import('./lessons/contrastive-learning/index.mdx') },
    ],
  },
  {
    id: 'sequences',
    number: 11,
    era: '1986 to 2017',
    title: 'Sequence Models',
    blurb: 'RNNs, LSTMs, and the bottleneck that motivated attention.',
    lessons: [
      { id: 'rnns', title: 'Recurrent Neural Networks & BPTT', summary: 'A network with memory: reuse the same weights at every step, unroll it to train it, and watch long-range signals fade or explode — and Lab: Tame BPTT.', minutes: 35, load: () => import('./lessons/rnns/index.mdx') },
      { id: 'lstms-grus', title: 'LSTMs and GRUs', summary: 'Give memory a protected highway and gates that decide what to keep, then race plain, gated and LSTM cells fairly — and Lab: Wire the Gated Cell.', minutes: 35, load: () => import('./lessons/lstms-grus/index.mdx') },
      { id: 'seq2seq-attention', title: 'Encoder–Decoder and Early Attention', summary: 'Squeeze a whole sentence into one vector and translation falls apart; let the decoder look back at every word and it recovers — and Lab: Break the Bottleneck.', minutes: 35, load: () => import('./lessons/seq2seq-attention/index.mdx') },
    ],
  },
  {
    id: 'transformers',
    number: 12,
    era: '1994 to 2022',
    title: 'Transformers & Attention',
    blurb: 'The architecture behind modern language models, taken apart one tensor at a time.',
    lessons: [
      { id: 'attention', title: 'Attention from Scratch', summary: 'How a word pulls meaning from its context: a soft dictionary lookup built from dot products, softmax and a weighted sum — and puzzles where you steer it.', minutes: 35, load: () => import('./lessons/attention/index.mdx') },
      { id: 'multi-head-attention', title: 'Multi-Head Attention', summary: 'Why one softmax can only answer one question at a time — and how splitting attention into heads asks several at once, for free.', minutes: 25, load: () => import('./lessons/multi-head-attention/index.mdx') },
      { id: 'positions', title: 'Positions: Teaching Attention About Order', summary: 'Attention can’t tell “dog bites man” from “man bites dog”. Clocks, rotations, and a lab where you build a head that finds the previous word.', minutes: 30, load: () => import('./lessons/positions/index.mdx') },
      { id: 'transformer-block', title: 'The Transformer Block', summary: 'Communicate, then compute, on a shared whiteboard: attention, feed-forward networks, residual connections, layer norm and the causal mask — assembled into a GPT.', minutes: 35, load: () => import('./lessons/transformer-block/index.mdx') },
      { id: 'moe', title: 'Mixture of Experts & Sparse Routing', summary: 'Why wake every neuron for every word? How a router picks a few specialist networks per token, why one expert can hog the work, and how to balance them.', minutes: 35, load: () => import('./lessons/moe/index.mdx') },
      { id: 'tokenization', title: 'Tokenization & Byte-Pair Encoding', summary: 'How raw text becomes integer IDs: why characters are too long, words are too many, how BPE discovers subwords — and why models struggle to count the “r”s in strawberry.', minutes: 30, load: () => import('./lessons/tokenization/index.mdx') },
      { id: 'multimodal-vit', title: 'Vision Transformers & Multimodal Models', summary: 'Chopping an image into patch tokens so a transformer can read it, why vision transformers need more data than convolutional networks, and aligning pictures with text.', minutes: 35, load: () => import('./lessons/multimodal-vit/index.mdx') },
    ],
  },
  {
    id: 'llms',
    number: 13,
    era: '1948 to 1951, then 2018 to 2024',
    title: 'Large Language Models',
    blurb: 'From next-token prediction to assistants: sampling, scaling, and their failure modes.',
    lessons: [
      { id: 'interlude-chatgpt', title: 'Interlude: The Week Everyone Noticed (2022)', summary: 'From GPT-3 to ChatGPT’s sudden public arrival in November 2022, what was old versus new inside it, and the 2024 Nobel Prizes that marked the moment.', minutes: 8, load: () => import('./lessons/interlude-chatgpt/index.mdx') },
      { id: 'next-token-prediction', title: 'Next-Token Prediction & Perplexity', summary: 'How a transformer writes one token at a time, what perplexity really means, why neural embeddings beat lookup tables — and Shannon’s betting game.', minutes: 30, load: () => import('./lessons/next-token-prediction/index.mdx') },
      { id: 'sampling', title: 'Sampling: Temperature, Top-k and Top-p', summary: 'Why greedy loops, why raw sampling falls into the long tail, how nucleus sampling adapts to confidence, beam search — and a sampler-tuning lab with an impossible level.', minutes: 30, load: () => import('./lessons/sampling/index.mdx') },
      { id: 'scaling-laws', title: 'Scaling Laws & Compute-Optimal Training', summary: 'How much better does a model get with more size, data and compute? The surprisingly regular answer, the famous correction, and a compute-budget lab.', minutes: 30, load: () => import('./lessons/scaling-laws/index.mdx') },
    ],
  },
  {
    id: 'rl',
    number: 14,
    era: '1953 to 2017',
    title: 'Reinforcement Learning',
    blurb: 'Agents that learn from consequences — and exploit badly specified rewards.',
    lessons: [
      { id: 'mdps', title: 'States, Actions, Rewards & Value', summary: 'Decisions with consequences: how an agent weighs now against later, why uncertainty makes it cautious, and the equation that plans ahead — and Lab: Wire the Policy.', minutes: 30, load: () => import('./lessons/mdps/index.mdx') },
      { id: 'q-learning', title: 'Q-Learning & Exploration', summary: 'Learning by trial and error: the explore-or-exploit dilemma, a value table that fills itself in, and a cliff that separates bold from careful — and Lab: Tune the Q-Learner.', minutes: 35, load: () => import('./lessons/q-learning/index.mdx') },
      { id: 'policy-gradients', title: 'Policy Gradients, Baselines & Actor-Critic', summary: 'Skip the value table and adjust behaviour directly: why randomness helps, why a baseline costs nothing, how clipping keeps steps safe — and Lab: Stabilise the Policy Gradient.', minutes: 40, load: () => import('./lessons/policy-gradients/index.mdx') },
      { id: 'reward-hacking', title: 'Lab: The Reward-Hacking Sandbox', summary: 'Write a reward, and an optimiser finds the loophole: the boat that circles forever, why bonuses backfire, and a lab where an exact optimiser attacks every reward you write.', minutes: 35, load: () => import('./lessons/reward-hacking/index.mdx') },
    ],
  },
  {
    id: 'finetuning',
    number: 15,
    era: '2017 to 2025',
    title: 'Training & Fine-Tuning',
    blurb: 'Pretraining, fine-tuning, LoRA, RLHF and the rest of the post-training toolkit.',
    lessons: [
      { id: 'lora', title: 'LoRA & Low-Rank Fine-Tuning', summary: 'Fine-tune a giant model by training a tiny add-on: why the change a model needs is far smaller than the model — and a rank-budget lab.', minutes: 30, load: () => import('./lessons/lora/index.mdx') },
      { id: 'rlhf-dpo', title: 'RLHF & Direct Preference Optimisation (DPO)', summary: 'Teaching a model what people prefer: why optimising a learned score backfires, how a leash keeps it honest, a simpler method without the scorer — and an alignment lab.', minutes: 35, load: () => import('./lessons/rlhf-dpo/index.mdx') },
      { id: 'reasoning', title: 'Reasoning Models & Test-Time Compute', summary: 'Letting a model think longer before it answers: trading model size for compute at answer time, voting, checkers, judges that backfire, and rewards you can verify.', minutes: 35, load: () => import('./lessons/reasoning/index.mdx') },
    ],
  },
  {
    id: 'inference',
    number: 16,
    era: '2019 to 2023',
    title: 'Inference & Hardware',
    blurb: 'What actually happens on the GPU, and why memory bandwidth rules everything.',
    lessons: [
      { id: 'quantization', title: 'Floating-Point Formats & Quantisation', summary: 'How many bits can a number lose before the model notices? Flip bits in real number formats, watch one outlier ruin compression — and a mixed-precision lab.', minutes: 30, load: () => import('./lessons/quantization/index.mdx') },
      { id: 'kv-cache', title: 'KV Caching & Memory-Bound Decoding', summary: 'Why generating text one token at a time leaves a GPU mostly idle, and how caching past work fixes the waste — and Lab: VRAM Tetris.', minutes: 35, load: () => import('./lessons/kv-cache/index.mdx') },
    ],
  },
  {
    id: 'generative',
    number: 17,
    era: '1986 to 2023',
    title: 'Generative Models & Diffusion',
    blurb: 'VAEs, GANs and diffusion: how machines learn to create.',
    lessons: [
      { id: 'autoencoders', title: 'Autoencoders and VAEs', summary: 'Why averaging two faces makes a ghost, how a bottleneck finds the hidden knobs behind data, and how a VAE makes new faces — and Lab: The Face Printer.', minutes: 40, load: () => import('./lessons/autoencoders/index.mdx') },
      { id: 'gans', title: 'GANs: A Forger and a Detective', summary: 'Two networks in a duel: one forges data, one tries to catch it. Why the duel produces sharp samples, why it breaks, and how to keep it balanced.', minutes: 35, load: () => import('./lessons/gans/index.mdx') },
      { id: 'diffusion', title: 'Diffusion: Adding and Removing Noise', summary: 'Dissolve data into noise step by step, then learn to run the film backwards: how a denoiser turns static into samples — and Lab: Spend the Step Budget.', minutes: 40, load: () => import('./lessons/diffusion/index.mdx') },
      { id: 'guidance', title: 'Guidance and Conditioning', summary: 'Ask a generator for a particular class, turn the request up, and watch quality rise while variety drains away — and Lab: Tune the Guidance.', minutes: 35, load: () => import('./lessons/guidance/index.mdx') },
    ],
  },
  {
    id: 'systems',
    number: 18,
    era: '1947 to today',
    title: 'AI Systems Engineering',
    blurb: 'Vector search, RAG, tools, agents and evaluation: models inside real software.',
    lessons: [
      { id: 'similarity-search', title: 'Similarity Search', summary: 'Finding the nearest of ten million vectors without checking them all: buckets, random hashes and compression, and what each shortcut costs — and Lab: Tune the Vector Index.', minutes: 35, load: () => import('./lessons/similarity-search/index.mdx') },
      { id: 'rag', title: 'Retrieval-Augmented Generation', summary: 'Why models need to look things up, keyword versus meaning-based search measured on real questions, and how chunk size trades accuracy for cost — and Lab: Tune the Retriever.', minutes: 35, load: () => import('./lessons/rag/index.mdx') },
      { id: 'tool-calling', title: 'Tool Calling and Agents', summary: 'How a text predictor acts on the world through tool calls, why each step’s small failure rate compounds, and what makes agents safe — and Lab: Harden the Agent Pipeline.', minutes: 35, load: () => import('./lessons/tool-calling/index.mdx') },
      { id: 'evaluation', title: 'Evaluation: Measuring AI Without Fooling Yourself', summary: 'Is model B really better, or just lucky? Fair comparisons, the trap of tuning on your test, judges that play favourites — and Lab: Size Your Eval Suite.', minutes: 35, load: () => import('./lessons/evaluation/index.mdx') },
    ],
  },
  {
    id: 'epilogue',
    number: 19,
    era: '2018 to today',
    title: 'Epilogue',
    blurb: 'Where the textbook stops and active research begins: five puzzles the field still cannot fully explain.',
    lessons: [
      {
        id: 'open-questions',
        title: 'What Nobody Understands Yet',
        summary: 'The field’s best minds are still puzzled. Why big networks generalise, why abilities seem to appear suddenly, and what is really stored inside a model.',
        minutes: 20,
        load: () => import('./lessons/open-questions/index.mdx'),
      },
    ],
  },
];

export const ALL_LESSONS: (LessonMeta & { module: ModuleMeta })[] = MODULES.flatMap((m) => m.lessons.map((l) => ({ ...l, module: m })));

export const findLesson = (id: string) => ALL_LESSONS.find((l) => l.id === id);

export function neighbours(id: string) {
  const i = ALL_LESSONS.findIndex((l) => l.id === id);
  return { prev: i > 0 ? ALL_LESSONS[i - 1] : undefined, next: i >= 0 && i < ALL_LESSONS.length - 1 ? ALL_LESSONS[i + 1] : undefined };
}
