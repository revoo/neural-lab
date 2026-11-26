// authored-by: Claude Opus 5.5
/**
 * V2-WP2: the concept dependency graph behind `#/map`.
 *
 * A curated directed acyclic graph. `PREREQS[lesson]` lists the lessons whose ideas it uses
 * directly (seeded from the `<Bridge>` links of the Maths Toolkit and the `LessonLink`s in the
 * prose). Technology destinations (`dest:…`) are the places a learner might be heading. Everything
 * else (upstream trees, downstream "unlocks", the Foundational chips in `areas.ts`) is computed from these lists.
 */
import { ALL_LESSONS } from './curriculum';

export const PREREQS: Record<string, string[]> = {
  prologue: [],
  vectors: ['prologue'],
  'dot-product': ['vectors'],
  matrices: ['vectors'],
  'matrix-multiplication': ['matrices', 'dot-product'],
  derivatives: ['prologue'],
  gradients: ['derivatives', 'vectors'],
  'chain-rule': ['derivatives'],
  probability: ['vectors'],
  information: ['probability'],
  neuron: ['vectors', 'dot-product'],
  'activation-functions': ['neuron'],
  'logic-lab': ['neuron', 'activation-functions'],
  perceptron: ['logic-lab'],
  'interlude-first-winter': ['perceptron'],
  'forward-propagation': ['matrices', 'matrix-multiplication', 'activation-functions', 'logic-lab'],
  'network-playground': ['forward-propagation'],
  'depth-and-width': ['forward-propagation'],
  'gradient-descent': ['derivatives', 'gradients'],
  'loss-functions': ['probability', 'information'],
  statistics: ['probability'],
  'sgd-minibatch': ['gradient-descent', 'statistics'],
  optimizers: ['sgd-minibatch'],
  'lr-schedules': ['optimizers'],
  backpropagation: ['chain-rule', 'gradient-descent', 'loss-functions', 'forward-propagation'],
  'network-from-scratch': ['backpropagation'],
  'vanishing-gradients': ['backpropagation', 'depth-and-width'],
  'training-lab': ['network-from-scratch', 'optimizers', 'network-playground'],
  overfitting: ['training-lab'],
  regularization: ['overfitting'],
  'diagnosis-lab': ['regularization', 'vanishing-gradients', 'lr-schedules'],
  'interlude-second-winter': ['backpropagation'],
  'high-dimensions': ['dot-product', 'statistics'],
  'linear-regression': ['gradient-descent', 'loss-functions'],
  svm: ['linear-regression', 'dot-product'],
  knn: ['high-dimensions'],
  'decision-trees': ['information', 'overfitting'],
  'clustering-pca': ['matrices', 'statistics'],
  'interlude-big-bang': ['backpropagation'],
  'images-as-tensors': ['matrix-multiplication'],
  convolution: ['images-as-tensors', 'dot-product'],
  cnns: ['convolution', 'backpropagation', 'optimizers', 'overfitting'],
  'learning-to-see': ['cnns'],
  'object-detection': ['cnns'],
  'one-hot-vs-dense': ['neuron'],
  'word-embeddings': ['one-hot-vs-dense', 'loss-functions'],
  'contrastive-learning': ['word-embeddings', 'high-dimensions', 'sgd-minibatch'],
  rnns: ['backpropagation', 'word-embeddings'],
  'lstms-grus': ['rnns', 'vanishing-gradients'],
  'seq2seq-attention': ['lstms-grus'],
  attention: ['probability', 'word-embeddings', 'seq2seq-attention'],
  'multi-head-attention': ['attention'],
  positions: ['attention'],
  'transformer-block': ['multi-head-attention', 'positions', 'vanishing-gradients'],
  moe: ['transformer-block', 'optimizers'],
  tokenization: ['one-hot-vs-dense', 'information'],
  'multimodal-vit': ['transformer-block', 'cnns', 'contrastive-learning'],
  'interlude-chatgpt': ['transformer-block'],
  'next-token-prediction': ['transformer-block', 'tokenization'],
  sampling: ['next-token-prediction'],
  'scaling-laws': ['next-token-prediction', 'overfitting'],
  mdps: ['probability'],
  'q-learning': ['mdps'],
  'policy-gradients': ['q-learning', 'gradient-descent'],
  'reward-hacking': ['policy-gradients'],
  lora: ['transformer-block', 'matrix-multiplication'],
  'rlhf-dpo': ['policy-gradients', 'next-token-prediction'],
  reasoning: ['rlhf-dpo', 'sampling', 'scaling-laws'],
  quantization: ['transformer-block'],
  'kv-cache': ['transformer-block', 'sampling'],
  autoencoders: ['network-from-scratch', 'clustering-pca', 'optimizers', 'overfitting'],
  gans: ['autoencoders'],
  diffusion: ['autoencoders', 'statistics'],
  guidance: ['diffusion'],
  'similarity-search': ['contrastive-learning', 'high-dimensions'],
  rag: ['similarity-search', 'next-token-prediction'],
  'tool-calling': ['rag', 'sampling'],
  evaluation: ['statistics', 'tool-calling'],
  'tiny-gpt': ['transformer-block', 'tokenization', 'next-token-prediction', 'sampling', 'optimizers'],
  'open-questions': ['scaling-laws', 'overfitting', 'high-dimensions'],
};

export type Destination = { id: string; title: string; blurb: string; needs: string[] };

/** Where the learner might be heading. `needs` are the lessons that lead straight into it. */
export const DESTINATIONS: Destination[] = [
  { id: 'dest:llms', title: 'Transformers & LLMs', blurb: 'How a language model reads, predicts and writes text.', needs: ['next-token-prediction', 'sampling', 'scaling-laws', 'tiny-gpt'] },
  { id: 'dest:reasoning', title: 'Reasoning Models', blurb: 'Models that are trained to think longer before they answer.', needs: ['reasoning'] },
  { id: 'dest:generative', title: 'Diffusion & Generative AI', blurb: 'Machines that make new images and data.', needs: ['guidance', 'gans'] },
  { id: 'dest:agents', title: 'RAG & Agents', blurb: 'Models inside real software: search, tools and evaluation.', needs: ['rag', 'tool-calling', 'evaluation'] },
  { id: 'dest:vision', title: 'Computer Vision & Multimodal', blurb: 'Seeing: from pixels to detectors to pictures matched with words.', needs: ['object-detection', 'multimodal-vit', 'learning-to-see'] },
  { id: 'dest:rl', title: 'Reinforcement Learning', blurb: 'Agents that learn from consequences.', needs: ['reward-hacking'] },
  { id: 'dest:inference', title: 'Efficient Training & Inference', blurb: 'Fitting and running big models on real hardware.', needs: ['lora', 'quantization', 'kv-cache', 'moe'] },
];

export const isDestination = (id: string) => id.startsWith('dest:');

const LESSON_INDEX = new Map(ALL_LESSONS.map((l, i) => [l.id, i]));

/** Direct prerequisites of any node (lesson or destination), in the full graph. */
export function directPrereqs(id: string): string[] {
  if (isDestination(id)) return DESTINATIONS.find((d) => d.id === id)?.needs ?? [];
  return PREREQS[id] ?? [];
}

/** Every edge [from, to] of the full graph: `from` feeds into `to`. */
export function allEdges(): [string, string][] {
  const out: [string, string][] = [];
  for (const [to, from] of Object.entries(PREREQS)) for (const f of from) out.push([f, to]);
  for (const d of DESTINATIONS) for (const f of d.needs) out.push([f, d.id]);
  return out;
}

/** All node ids: lessons in course order, then destinations. */
export const ALL_NODES = [...ALL_LESSONS.map((l) => l.id), ...DESTINATIONS.map((d) => d.id)];

/** Kahn's algorithm. Returns null if the graph has a cycle. */
export function topoSort(nodes: string[] = ALL_NODES, edges: [string, string][] = allEdges()): string[] | null {
  const indeg = new Map(nodes.map((n) => [n, 0]));
  const out = new Map<string, string[]>(nodes.map((n) => [n, []]));
  for (const [a, b] of edges) {
    indeg.set(b, (indeg.get(b) ?? 0) + 1);
    out.get(a)?.push(b);
  }
  const queue = nodes.filter((n) => indeg.get(n) === 0);
  const order: string[] = [];
  while (queue.length) {
    const n = queue.shift()!;
    order.push(n);
    for (const m of out.get(n) ?? []) {
      indeg.set(m, indeg.get(m)! - 1);
      if (indeg.get(m) === 0) queue.push(m);
    }
  }
  return order.length === nodes.length ? order : null;
}

function closure(start: string, step: (id: string) => string[]): Set<string> {
  const seen = new Set<string>();
  const stack = [...step(start)];
  while (stack.length) {
    const n = stack.pop()!;
    if (seen.has(n)) continue;
    seen.add(n);
    stack.push(...step(n));
  }
  return seen;
}

const byCourseOrder = (a: string, b: string) => (LESSON_INDEX.get(a) ?? 1e9) - (LESSON_INDEX.get(b) ?? 1e9);

/** Everything `id` needs, directly or indirectly, in course order (lessons only). */
export function upstream(id: string): string[] {
  const edges = allEdges();
  const parents = new Map<string, string[]>();
  for (const [a, b] of edges) parents.set(b, [...(parents.get(b) ?? []), a]);
  return [...closure(id, (n) => parents.get(n) ?? [])].sort(byCourseOrder);
}

/** Everything that builds on `id`, directly or indirectly: lessons in course order, then destinations. */
export function downstream(id: string): string[] {
  const edges = allEdges();
  const children = new Map<string, string[]>();
  for (const [a, b] of edges) children.set(a, [...(children.get(a) ?? []), b]);
  return [...closure(id, (n) => children.get(n) ?? [])].sort(byCourseOrder);
}

/** How many of a node's prerequisites are done, and the earliest one in course order that isn't. */
export function prereqStatus(id: string, completed: Record<string, unknown>) {
  const ups = upstream(id);
  const done = ups.filter((u) => completed[u]).length;
  return { total: ups.length, done, next: ups.find((u) => !completed[u]) };
}

/** Short labels for the map's nodes (the full titles are too long to draw). */
export const SHORT: Record<string, string> = {
  prologue: 'Prologue',
  vectors: 'Vectors',
  'dot-product': 'Dot product',
  matrices: 'Matrices',
  'matrix-multiplication': 'Matrix multiply',
  derivatives: 'Derivatives',
  gradients: 'Gradients',
  'chain-rule': 'Chain rule',
  probability: 'Probability',
  information: 'Cross-entropy',
  neuron: 'Neuron',
  'activation-functions': 'Activations',
  'logic-lab': 'Logic lab',
  perceptron: 'Perceptron',
  'interlude-first-winter': 'First winter',
  'forward-propagation': 'Forward pass',
  'network-playground': 'Playground',
  'depth-and-width': 'Depth & width',
  'gradient-descent': 'Gradient descent',
  'loss-functions': 'Loss functions',
  statistics: 'Statistics',
  'sgd-minibatch': 'SGD',
  optimizers: 'Adam & co.',
  'lr-schedules': 'LR schedules',
  backpropagation: 'Backprop',
  'network-from-scratch': 'Net from scratch',
  'vanishing-gradients': 'Vanishing grads',
  'training-lab': 'Training lab',
  overfitting: 'Overfitting',
  regularization: 'Regularisation',
  'diagnosis-lab': 'Model clinic',
  'interlude-second-winter': '1990s interlude',
  'high-dimensions': 'High dimensions',
  'linear-regression': 'Regression',
  svm: 'SVMs',
  knn: 'k-NN',
  'decision-trees': 'Trees & forests',
  'clustering-pca': 'k-Means & PCA',
  'interlude-big-bang': '2012 interlude',
  'images-as-tensors': 'Images',
  convolution: 'Convolution',
  cnns: 'CNNs',
  'learning-to-see': 'Learning to see',
  'object-detection': 'Detection',
  'one-hot-vs-dense': 'One-hot vs dense',
  'word-embeddings': 'Word embeddings',
  'contrastive-learning': 'Contrastive',
  rnns: 'RNNs',
  'lstms-grus': 'LSTMs & GRUs',
  'seq2seq-attention': 'Seq2seq',
  attention: 'Attention',
  'multi-head-attention': 'Multi-head',
  positions: 'Positions',
  'transformer-block': 'Transformer block',
  moe: 'Mixture of experts',
  tokenization: 'Tokenization',
  'multimodal-vit': 'Vision transformers & CLIP',
  'interlude-chatgpt': 'ChatGPT interlude',
  'next-token-prediction': 'Next token',
  sampling: 'Sampling',
  'scaling-laws': 'Scaling laws',
  mdps: 'MDPs',
  'q-learning': 'Q-learning',
  'policy-gradients': 'Policy gradients',
  'reward-hacking': 'Reward hacking',
  lora: 'LoRA',
  'rlhf-dpo': 'RLHF & DPO',
  reasoning: 'Reasoning & test-time compute',
  quantization: 'Quantisation',
  'kv-cache': 'KV cache',
  autoencoders: 'Autoencoders',
  gans: 'GANs',
  diffusion: 'Diffusion',
  guidance: 'Guidance',
  'similarity-search': 'Similarity search',
  rag: 'RAG',
  'tool-calling': 'Tools & agents',
  evaluation: 'Evaluation',
  'tiny-gpt': 'Tiny GPT',
  'open-questions': 'Open questions',
};

export const nodeLabel = (id: string) => (isDestination(id) ? DESTINATIONS.find((d) => d.id === id)!.title : (SHORT[id] ?? id));
