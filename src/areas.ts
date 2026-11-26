// authored-by: Claude Opus 5.5
/**
 * V2-WP1 (revised by the owner): instead of a Full Course / Core Track switch, every lesson and
 * module carries a chip saying where it belongs.
 *
 * - **Foundational**: integral to the whole backbone of AI. Checked against the concept map
 *   (`src/dependencies.ts`): a lesson is Foundational exactly when it feeds at least three of the
 *   four big branches (language, vision, generation, reinforcement learning). `areas.test.ts`
 *   enforces both directions, so a chip can't drift from the graph.
 * - Otherwise one **home** chip (e.g. "Computer Vision foundation", "LLM & Transformers backbone").
 *   When the home names a destination on the map, the test checks the lesson really leads there.
 */
import { MODULES, type ModuleMeta } from './curriculum';

export type Area =
  | 'foundational'
  | 'llm'
  | 'vision'
  | 'generative'
  | 'rl'
  | 'agents'
  | 'inference'
  | 'reasoning'
  | 'classical'
  | 'craft'
  | 'history'
  | 'frontier';

export type AreaInfo = {
  label: string;
  short: string;
  blurb: string;
  /** The concept-map destination this area leads to, if any. */
  dest?: string;
  /** CSS colour token for the chip. */
  colour: string;
};

export const AREAS: Record<Area, AreaInfo> = {
  foundational: { label: 'Foundational', short: 'Foundational', blurb: 'Integral to the whole backbone of AI: language, vision, generation and agents that learn all build on it.', colour: 'var(--accent)' },
  llm: { label: 'LLM & Transformers Backbone', short: 'LLM', blurb: 'Part of how language models and transformers work.', dest: 'dest:llms', colour: 'var(--c-a)' },
  vision: { label: 'Computer Vision Foundation', short: 'Vision', blurb: 'Part of how machines see: images, convolutions, detectors and picture–text models.', dest: 'dest:vision', colour: 'var(--c-b)' },
  generative: { label: 'Generative AI Foundation', short: 'Generative', blurb: 'Part of how machines make new images and data.', dest: 'dest:generative', colour: 'var(--c-pos)' },
  rl: { label: 'Reinforcement Learning Foundation', short: 'RL', blurb: 'Part of how agents learn from rewards and consequences.', dest: 'dest:rl', colour: 'var(--c-c)' },
  agents: { label: 'RAG & Agents Foundation', short: 'Agents', blurb: 'Part of putting models inside real software: search, tools and evaluation.', dest: 'dest:agents', colour: 'var(--c-grad)' },
  inference: { label: 'Efficient Training & Inference', short: 'Efficiency', blurb: 'Part of fitting and running big models on real hardware.', dest: 'dest:inference', colour: 'var(--c-warn)' },
  reasoning: { label: 'Reasoning & Alignment', short: 'Reasoning', blurb: 'Part of how models are post-trained to reason and to follow what people want.', dest: 'dest:reasoning', colour: 'var(--c-neg)' },
  classical: { label: 'Classical ML', short: 'Classical', blurb: 'Machine learning before deep learning, still the right tool for much tabular data.', colour: 'var(--text-muted)' },
  craft: { label: 'Training Craft', short: 'Craft', blurb: 'Practical know-how for getting any network to train well.', colour: 'var(--text-muted)' },
  history: { label: 'History & Context', short: 'History', blurb: 'How the field got here: the ideas, dead ends and turning points behind today’s models.', colour: 'var(--text-faint)' },
  frontier: { label: 'Research Frontier', short: 'Frontier', blurb: 'Open questions the field is still working on.', colour: 'var(--text-faint)' },
};

/** Each lesson's home. Foundational lessons are listed explicitly (and verified by the test). */
export const LESSON_AREA: Record<string, Area> = {
  prologue: 'history',
  vectors: 'foundational',
  'dot-product': 'foundational',
  matrices: 'foundational',
  'matrix-multiplication': 'foundational',
  derivatives: 'foundational',
  gradients: 'foundational',
  'chain-rule': 'foundational',
  probability: 'foundational',
  information: 'foundational',
  neuron: 'foundational',
  'activation-functions': 'foundational',
  'logic-lab': 'foundational',
  perceptron: 'history',
  'interlude-first-winter': 'history',
  'forward-propagation': 'foundational',
  'network-playground': 'foundational',
  'depth-and-width': 'craft',
  'gradient-descent': 'foundational',
  'loss-functions': 'foundational',
  statistics: 'foundational',
  'sgd-minibatch': 'foundational',
  optimizers: 'foundational',
  'lr-schedules': 'craft',
  backpropagation: 'foundational',
  'network-from-scratch': 'foundational',
  'vanishing-gradients': 'craft',
  'training-lab': 'foundational',
  overfitting: 'foundational',
  regularization: 'craft',
  'diagnosis-lab': 'craft',
  'interlude-second-winter': 'history',
  'high-dimensions': 'classical',
  'linear-regression': 'classical',
  svm: 'classical',
  knn: 'classical',
  'decision-trees': 'classical',
  'clustering-pca': 'classical',
  'interlude-big-bang': 'history',
  'images-as-tensors': 'vision',
  convolution: 'vision',
  cnns: 'vision',
  'object-detection': 'vision',
  'one-hot-vs-dense': 'llm',
  'word-embeddings': 'llm',
  'contrastive-learning': 'vision',
  rnns: 'history',
  'lstms-grus': 'history',
  'seq2seq-attention': 'llm',
  attention: 'llm',
  'multi-head-attention': 'llm',
  positions: 'llm',
  'transformer-block': 'llm',
  moe: 'inference',
  tokenization: 'llm',
  'multimodal-vit': 'vision',
  'interlude-chatgpt': 'history',
  'next-token-prediction': 'llm',
  sampling: 'llm',
  'scaling-laws': 'llm',
  mdps: 'rl',
  'q-learning': 'rl',
  'policy-gradients': 'rl',
  'reward-hacking': 'rl',
  lora: 'inference',
  'rlhf-dpo': 'reasoning',
  reasoning: 'reasoning',
  quantization: 'inference',
  'kv-cache': 'inference',
  autoencoders: 'generative',
  gans: 'generative',
  diffusion: 'generative',
  guidance: 'generative',
  'similarity-search': 'agents',
  rag: 'agents',
  'tool-calling': 'agents',
  evaluation: 'agents',
  'tiny-gpt': 'llm',
  'open-questions': 'frontier',
};

export const lessonArea = (id: string): Area => LESSON_AREA[id] ?? 'history';

/** A module's chip: the home most of its lessons share (ties go to the earliest lesson). */
export function moduleArea(m: ModuleMeta): Area {
  const counts = new Map<Area, number>();
  for (const l of m.lessons) counts.set(lessonArea(l.id), (counts.get(lessonArea(l.id)) ?? 0) + 1);
  let best: Area = lessonArea(m.lessons[0]?.id ?? '');
  for (const l of m.lessons) if ((counts.get(lessonArea(l.id)) ?? 0) > (counts.get(best) ?? 0)) best = lessonArea(l.id);
  return best;
}

export const MODULE_AREAS = Object.fromEntries(MODULES.map((m) => [m.id, moduleArea(m)])) as Record<string, Area>;

/** The four big branches used by the Foundational rule. */
export const BRANCHES: Record<string, string[]> = {
  language: ['dest:llms', 'dest:reasoning', 'dest:agents', 'dest:inference'],
  vision: ['dest:vision'],
  generation: ['dest:generative'],
  rl: ['dest:rl'],
};
