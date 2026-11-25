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
  title: string;
  blurb: string;
  lessons: LessonMeta[];
  /** Lessons not yet written, shown as the roadmap. */
  planned?: string[];
};

export const MODULES: ModuleMeta[] = [
  {
    id: 'foundations',
    number: 1,
    title: 'Mathematical Foundations',
    blurb: 'The handful of mathematical ideas that nearly all of AI is built from — each one introduced through something you can drag, twist, and break.',
    lessons: [
      { id: 'vectors', title: 'Vectors: Lists That Point Somewhere', summary: 'Why a list of numbers can also be an arrow, and why that double life matters for AI.', minutes: 15, load: () => import('./lessons/vectors/index.mdx') },
      { id: 'dot-product', title: 'The Dot Product: Measuring Agreement', summary: 'The single most important operation in deep learning, and the reason it measures alignment.', minutes: 20, load: () => import('./lessons/dot-product/index.mdx') },
      { id: 'matrices', title: 'Matrices: Machines That Transform Space', summary: 'A matrix as a recipe for moving every point in space — and how to read that recipe at a glance.', minutes: 20, load: () => import('./lessons/matrices/index.mdx') },
      { id: 'matrix-multiplication', title: 'Matrix Multiplication, Shapes & Tensors', summary: 'Many dot products at once, the shape rules that keep them straight, and what a tensor really is.', minutes: 25, load: () => import('./lessons/matrix-multiplication/index.mdx') },
      { id: 'derivatives', title: 'Derivatives: Measuring Sensitivity', summary: 'If I nudge this input a little, how much does the output move? That question is the derivative.', minutes: 20, load: () => import('./lessons/derivatives/index.mdx') },
      { id: 'gradients', title: 'Partial Derivatives & Gradients', summary: 'Sensitivity when there are many knobs — and the arrow that points uphill.', minutes: 20, load: () => import('./lessons/gradients/index.mdx') },
      { id: 'chain-rule', title: 'The Chain Rule: Sensitivity Through a Pipeline', summary: 'How sensitivities multiply through a chain of functions — the engine underneath backpropagation.', minutes: 20, load: () => import('./lessons/chain-rule/index.mdx') },
      { id: 'probability', title: 'Probability Distributions & Softmax', summary: 'Turning arbitrary scores into well-behaved probabilities, and what temperature actually does.', minutes: 20, load: () => import('./lessons/probability/index.mdx') },
      { id: 'information', title: 'Logarithms, Surprise & Cross-Entropy', summary: 'Why “how surprised were you?” is the loss function behind every classifier and language model.', minutes: 25, load: () => import('./lessons/information/index.mdx') },
      { id: 'statistics', title: 'Statistics: Mean, Variance & the Bell Curve', summary: 'Spread as an average of squares, why variances add, where the bell curve comes from, how much a test score wobbles — and Lab: The Dice Workshop.', minutes: 30, load: () => import('./lessons/statistics/index.mdx') },
    ],
    planned: ['High-dimensional geometry'],
  },
  {
    id: 'neuron',
    number: 2,
    title: 'The Artificial Neuron',
    blurb: 'The tiny computational unit everything else is assembled from.',
    lessons: [
      { id: 'neuron', title: 'The Artificial Neuron', summary: 'Weighted sum, bias, activation — and why a single neuron is really a line drawn through space.', minutes: 25, load: () => import('./lessons/neuron/index.mdx') },
      { id: 'activation-functions', title: 'Why Networks Need Activation Functions', summary: 'Stack a hundred linear layers and you still have one. Then add a kink, and everything changes.', minutes: 25, load: () => import('./lessons/activation-functions/index.mdx') },
      { id: 'logic-lab', title: 'Lab: Wiring Logic by Hand', summary: 'A puzzle workbench: build AND, OR and finally XOR out of neurons whose weights you set yourself.', minutes: 30, load: () => import('./lessons/logic-lab/index.mdx') },
    ],
  },
  {
    id: 'architecture',
    number: 3,
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
    title: 'Loss Functions & Optimization',
    blurb: 'How a model measures its mistakes, and how it walks downhill to make fewer of them.',
    lessons: [
      { id: 'gradient-descent', title: 'Gradient Descent: Walking Downhill in the Fog', summary: 'The algorithm behind nearly all of modern AI training, and why the step size can make or break it.', minutes: 25, load: () => import('./lessons/gradient-descent/index.mdx') },
      { id: 'loss-functions', title: 'Loss Functions: What Does “Wrong” Mean?', summary: 'Squared error, absolute error and cross-entropy — what each one really asks the model for, and why the choice matters.', minutes: 25, load: () => import('./lessons/loss-functions/index.mdx') },
      { id: 'sgd-minibatch', title: 'Stochastic Gradient Descent and Mini-Batches', summary: 'Why a noisy gradient from a handful of examples beats an exact one from millions.', minutes: 25, load: () => import('./lessons/sgd-minibatch/index.mdx') },
      { id: 'optimizers', title: 'Momentum, RMSProp and Adam', summary: 'Two simple fixes for gradient descent’s weaknesses, and the optimizer that combines them — racing on the same landscapes.', minutes: 30, load: () => import('./lessons/optimizers/index.mdx') },
      { id: 'lr-schedules', title: 'Learning-Rate Schedules', summary: 'Fast or precise? Why the best step size changes during training — and how to design a schedule yourself.', minutes: 25, load: () => import('./lessons/lr-schedules/index.mdx') },
    ],
  },
  {
    id: 'backprop',
    number: 5,
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
    title: 'Neural Network Training Lab',
    blurb: 'A full browser-based training sandbox with live decision boundaries.',
    lessons: [
      { id: 'training-lab', title: 'The Training Laboratory', summary: 'Every piece on the bench at once: data, features, architecture and settings — train live, compare runs, and take on the challenges.', minutes: 40, load: () => import('./lessons/training-lab/index.mdx') },
    ],
  },
  {
    id: 'generalization',
    number: 7,
    title: 'Generalization & Failure Modes',
    blurb: 'Why a model that aces its training data can still be useless.',
    lessons: [
      { id: 'overfitting', title: 'Overfitting and Underfitting', summary: 'Why a perfect score on the training data can mean nothing, the two ways to be wrong, and how data and training time move the line between them.', minutes: 30, load: () => import('./lessons/overfitting/index.mdx') },
      { id: 'regularization', title: 'Regularisation: Early Stopping, Weight Decay and Dropout', summary: 'Three ways to keep a big network and lose its bad habits — and a fair race between them.', minutes: 35, load: () => import('./lessons/regularization/index.mdx') },
      { id: 'diagnosis-lab', title: 'Lab: The Model Clinic', summary: 'Ten broken training runs, the instruments to examine them, and a menu of root causes. Diagnose each one, then fix it.', minutes: 45, load: () => import('./lessons/diagnosis-lab/index.mdx') },
    ],
  },
  { id: 'classical', number: 8, title: 'Classical Machine Learning', blurb: 'Trees, nearest neighbours, SVMs, clustering — and when they beat neural networks.', lessons: [], planned: ['Linear and logistic regression', 'k-nearest neighbours', 'Decision trees and forests', 'k-means and PCA'] },
  { id: 'vision', number: 9, title: 'Computer Vision', blurb: 'From pixels to convolutions to object detectors.', lessons: [], planned: ['Images as tensors', 'Convolution and kernels', 'CNNs and feature maps', 'Object detection'] },
  { id: 'embeddings', number: 10, title: 'Embeddings & Representations', blurb: 'Meaning as geometry: learned spaces where similar things sit close together.', lessons: [], planned: ['One-hot vs dense vectors', 'Word embeddings', 'Similarity search', 'Autoencoders'] },
  { id: 'sequences', number: 11, title: 'Sequence Models', blurb: 'RNNs, LSTMs, and the bottleneck that motivated attention.', lessons: [], planned: ['Recurrent networks', 'LSTMs and GRUs', 'Encoder–decoder and early attention'] },
  {
    id: 'transformers',
    number: 12,
    title: 'Transformers & Attention',
    blurb: 'The architecture behind modern language models, taken apart one tensor at a time.',
    lessons: [
      { id: 'attention', title: 'Attention from Scratch', summary: 'How a word pulls meaning from its context: a soft dictionary lookup built from dot products, softmax and a weighted sum — and puzzles where you steer it.', minutes: 35, load: () => import('./lessons/attention/index.mdx') },
      { id: 'multi-head-attention', title: 'Multi-Head Attention', summary: 'Why one softmax can only answer one question at a time — and how splitting attention into heads asks several at once, for free.', minutes: 25, load: () => import('./lessons/multi-head-attention/index.mdx') },
      { id: 'positions', title: 'Positions: Teaching Attention About Order', summary: 'Attention can’t tell “dog bites man” from “man bites dog”. Clocks, rotations, and a lab where you build a head that finds the previous word.', minutes: 30, load: () => import('./lessons/positions/index.mdx') },
      { id: 'transformer-block', title: 'The Transformer Block', summary: 'Communicate, then compute, on a shared whiteboard: attention, feed-forward networks, residual connections, layer norm and the causal mask — assembled into a GPT.', minutes: 35, load: () => import('./lessons/transformer-block/index.mdx') },
      { id: 'tokenization', title: 'Tokenization & Byte-Pair Encoding', summary: 'How raw text becomes integer IDs: why characters are too long, words are too many, how BPE discovers subwords — and why models struggle to count the “r”s in strawberry.', minutes: 30, load: () => import('./lessons/tokenization/index.mdx') },
    ],
  },
  {
    id: 'llms',
    number: 13,
    title: 'Large Language Models',
    blurb: 'From next-token prediction to assistants: sampling, scaling, and their failure modes.',
    lessons: [
      { id: 'next-token-prediction', title: 'Next-Token Prediction & Perplexity', summary: 'How a transformer writes one token at a time, what perplexity really means, why neural embeddings beat lookup tables — and Shannon’s betting game.', minutes: 30, load: () => import('./lessons/next-token-prediction/index.mdx') },
      { id: 'sampling', title: 'Sampling: Temperature, Top-k and Top-p', summary: 'Why greedy loops, why raw sampling falls into the long tail, how nucleus sampling adapts to confidence, beam search — and a sampler-tuning lab with an impossible level.', minutes: 30, load: () => import('./lessons/sampling/index.mdx') },
      { id: 'scaling-laws', title: 'Scaling Laws & Compute-Optimal Training', summary: 'Why training costs 6ND FLOPs, how Chinchilla balances parameters against tokens, why Llama 3 breaks Chinchilla on purpose — and a compute-budget lab.', minutes: 30, load: () => import('./lessons/scaling-laws/index.mdx') },
    ],
  },
  {
    id: 'finetuning',
    number: 14,
    title: 'Training & Fine-Tuning',
    blurb: 'Pretraining, fine-tuning, LoRA, RLHF and the rest of the post-training toolkit.',
    lessons: [
      { id: 'lora', title: 'LoRA & Low-Rank Fine-Tuning', summary: 'Why full fine-tuning blows up VRAM, how factoring ΔW = BA compresses updates 128×, why B=0 & A~N breaks symmetry — and a rank-allocation budget lab.', minutes: 30, load: () => import('./lessons/lora/index.mdx') },
      { id: 'rlhf-dpo', title: 'RLHF & Direct Preference Optimisation (DPO)', summary: 'Watch a proxy reward model trigger Goodhart’s law, see how DPO cancels Z(x) to drop the reward model, catch likelihood displacement — and tune an alignment pipeline.', minutes: 35, load: () => import('./lessons/rlhf-dpo/index.mdx') },
    ],
  },
  {
    id: 'inference',
    number: 15,
    title: 'Inference & Hardware',
    blurb: 'What actually happens on the GPU, and why memory bandwidth rules everything.',
    lessons: [
      { id: 'quantization', title: 'Floating-Point Formats & Quantisation', summary: 'Flip the bits of FP16, BF16 and FP8, watch one outlier crush per-tensor INT4, and solve a mixed-precision quantisation budget lab.', minutes: 30, load: () => import('./lessons/quantization/index.mdx') },
      { id: 'kv-cache', title: 'KV Caching & Memory-Bound Decoding', summary: 'Why naive decoding wastes 400× work, how GQA shrinks the KV cache, why an H100 runs at 0.36% compute during single-user chat — and Lab: VRAM Tetris.', minutes: 35, load: () => import('./lessons/kv-cache/index.mdx') },
    ],
  },
  {
    id: 'rl',
    number: 16,
    title: 'Reinforcement Learning',
    blurb: 'Agents that learn from consequences — and exploit badly specified rewards.',
    lessons: [
      { id: 'mdps', title: 'States, Actions, Rewards & Value', summary: 'How discount factor γ controls myopia, how stochastic transitions create cautious detours, Bellman’s dynamic-programming equation — and Lab: Wire the Policy.', minutes: 30, load: () => import('./lessons/mdps/index.mdx') },
      { id: 'q-learning', title: 'Q-Learning & Exploration', summary: 'Multi-armed bandits and UCB1, model-free TD backups on a live Q-table, Q-learning vs SARSA on Cliff Walking — and Lab: Tune the Q-Learner.', minutes: 35, load: () => import('./lessons/q-learning/index.mdx') },
      { id: 'policy-gradients', title: 'Policy Gradients, Baselines & Actor-Critic', summary: 'Why some problems need random policies, the log-derivative push, why baselines are free, actor-critic’s one-step judgement, PPO’s clip — and Lab: Stabilise the Policy Gradient.', minutes: 40, load: () => import('./lessons/policy-gradients/index.mdx') },
      { id: 'reward-hacking', title: 'Lab: The Reward-Hacking Sandbox', summary: 'Why optimisers farm renewable bonuses, the boat-race loop solved exactly, naive vs potential-based shaping — and a lab where an exact optimiser attacks every reward you write.', minutes: 35, load: () => import('./lessons/reward-hacking/index.mdx') },
    ],
  },
  { id: 'generative', number: 17, title: 'Generative Models & Diffusion', blurb: 'VAEs, GANs and diffusion: how machines learn to create.', lessons: [], planned: ['Autoencoders and VAEs', 'Diffusion: adding and removing noise', 'Guidance and conditioning'] },
  { id: 'systems', number: 18, title: 'AI Systems Engineering', blurb: 'RAG, tools, agents and evaluation: models inside real software.', lessons: [], planned: ['Retrieval-augmented generation', 'Tool calling and agents', 'Evaluation'] },
];

export const ALL_LESSONS: (LessonMeta & { module: ModuleMeta })[] = MODULES.flatMap((m) => m.lessons.map((l) => ({ ...l, module: m })));

export const findLesson = (id: string) => ALL_LESSONS.find((l) => l.id === id);

export function neighbours(id: string) {
  const i = ALL_LESSONS.findIndex((l) => l.id === id);
  return { prev: i > 0 ? ALL_LESSONS[i - 1] : undefined, next: i >= 0 && i < ALL_LESSONS.length - 1 ? ALL_LESSONS[i + 1] : undefined };
}
