/**
 * The case files: broken training runs, each a real configuration that fails in a characteristic
 * way (pinned by clinic.test.ts), plus the menu of root causes and the space of single changes
 * used for the optional "fix it with one change" goal.
 */
import { HEALTHY, type Config } from './clinic';

export type CauseId = 'lr-high' | 'lr-low' | 'init' | 'capacity' | 'overfit' | 'leak' | 'labels' | 'scale' | 'shuffle';

export const CAUSES: { id: CauseId; name: string }[] = [
  { id: 'lr-high', name: 'Step size too large' },
  { id: 'lr-low', name: 'Step size too small' },
  { id: 'init', name: 'Bad weight initialisation' },
  { id: 'capacity', name: 'Model too simple (underfitting)' },
  { id: 'overfit', name: 'Too little data for the model (overfitting)' },
  { id: 'leak', name: 'Test set contains training data (leakage)' },
  { id: 'labels', name: 'Wrong labels in the data' },
  { id: 'scale', name: 'Inputs on wildly different scales' },
  { id: 'shuffle', name: 'Training data not shuffled' },
];

export type Case = {
  id: string;
  title: string;
  /** the message from the colleague who brought the case in */
  brief: string;
  cfg: Config;
  cause: CauseId;
  /** Two-cause cases: a second root cause that must also be named (and fixed). */
  secondCause?: CauseId;
  /** why the right answer is right, pointing at the evidence */
  explain: string;
  /** for each tempting wrong answer: what evidence rules it out */
  ruleOut: Partial<Record<CauseId, string>>;
  /** fresh-data accuracy a fix must reach */
  target: number;
};

const moons = (p: Partial<Config>): Config => ({ ...HEALTHY, ...p });
const noisyCircle: Partial<Config> = { dataset: 'circle', n: 60, noise: 0.25, hidden: [16, 16, 16], steps: 4000 };

export const CASES: Case[] = [
  {
    id: 'flatline',
    title: 'Flatlined after one step',
    brief: 'Two moons, two hidden layers of 8 tanh neurons, momentum. The loss shot up straight away and has been flat ever since. More training changes nothing.',
    cfg: moons({ optimizer: 'momentum', lr: 3 }),
    cause: 'lr-high',
    target: 0.95,
    explain:
      'Within five steps the loss had jumped from 0.69 to over 4, and it ended above 8: the first updates were so large that they flung the weights to big values. By step 50, 94% of the tanh neurons were saturated, and by the end every one is, pinned at ±1 on nearly every point, where its slope is almost zero, so the gradients (10⁻⁸ and smaller) carry almost no learning signal. A step size of 3 with momentum is far too large for this network.',
    ruleOut: {
      init: 'At step 0 none of the neurons were saturated (see “trouble at step 0”). Something during training broke them, not the starting weights.',
      capacity: 'This network handles the moons easily with a sensible step size. And an underfitting model still learns something; this one is stuck at chance.',
      scale: 'Both inputs span roughly the same range (about −1.6 to 1.6).',
    },
  },
  {
    id: 'tortoise',
    title: 'The tortoise',
    brief: 'Plain SGD, 2,000 steps. It works, sort of: 87% on fresh data. The loss is still going down when training ends. My deadline is tomorrow.',
    cfg: moons({ optimizer: 'sgd', lr: 0.0003 }),
    cause: 'lr-low',
    target: 0.95,
    explain:
      'Nothing is broken: the loss falls smoothly, no neuron is dead or saturated, and training and fresh accuracy agree. It is simply moving very slowly. With plain SGD, a step size of 0.0003 means each step barely changes the weights, so 2,000 steps cover only a fraction of the distance to a good solution.',
    ruleOut: {
      capacity: 'The same network reaches about 99% with a bigger step size. And the loss is still falling at the end: it hasn’t hit a floor.',
      overfit: 'Training and fresh accuracy are close together; overfitting shows a big gap.',
    },
  },
  {
    id: 'doa',
    title: 'Dead on arrival',
    brief: 'ReLU network, Adam, perfectly ordinary settings. The loss sits at exactly 0.693 from the first step to the last. It’s as if nothing is connected.',
    cfg: moons({ activation: 'relu', init: 'negative-bias' }),
    cause: 'init',
    target: 0.95,
    explain:
      'Every hidden ReLU outputs zero for every training point — and they already did at step 0, before any training. A dead ReLU passes back no gradient, so nothing can ever revive them: the gradients are exactly zero and the output sits at 0.5 for every point, a loss of ln 2 ≈ 0.693. The culprit is the starting weights: every hidden bias was initialised to −3, which puts each neuron’s switch-on point outside the data.',
    ruleOut: {
      'lr-high': 'The neurons were dead before the first update, so no step, large or small, killed them. (Compare the last case in the clinic.)',
      capacity: 'Sixteen hidden neurons are plenty for the moons; here not one of them is contributing.',
    },
  },
  {
    id: 'stiff',
    title: 'Too stiff',
    brief: 'Circle data. I kept it simple: inputs straight into one output neuron. Training and test accuracy are both stuck around 40%, and the loss stopped improving almost at once.',
    cfg: moons({ dataset: 'circle', hidden: [] }),
    cause: 'capacity',
    target: 0.9,
    explain:
      'A single neuron can only draw a straight line, and no straight line separates a ring from the disc inside it. The training loss plateaus immediately at about 0.69 and training accuracy is as bad as test accuracy: the model is wrong in the same way everywhere. That is underfitting, high bias. It needs hidden neurons (or better input features).',
    ruleOut: {
      'lr-low': 'The loss flattened at once rather than creeping down; a bigger step size would reach the same plateau, just sooner.',
      overfit: 'Training accuracy is as bad as test accuracy. Overfitting needs a model that does well on its training data.',
    },
  },
  {
    id: 'memoriser',
    title: 'The memoriser',
    brief: 'Only 30 labelled examples (the rest went into the test set), so I used a big network to squeeze the most out of them. Training accuracy 100%. Test accuracy 70%.',
    cfg: moons(noisyCircle),
    cause: 'overfit',
    target: 0.8,
    explain:
      'Training accuracy is perfect and test accuracy is 30 points lower: a huge gap. Three layers of 16 neurons (609 parameters) for 30 noisy points means the network can draw islands around individual examples, and the decision boundary is full of them. The training loss kept falling while the test loss climbed. That is overfitting: too much capacity for too little data.',
    ruleOut: {
      leak: 'The test set has no copies of training points (the duplicate check reads 0%), and the test score is disappointing, not suspiciously high.',
      labels: 'Neighbouring training points mostly agree on their labels; the noise is in the positions, which is ordinary measurement noise.',
      capacity: 'The model fits its training data almost perfectly; it isn’t too simple.',
    },
  },
  {
    id: 'too-good',
    title: 'Too good to be true',
    brief: 'Same small, noisy dataset, but this time the test score was great: 93%! Then we deployed it, and on real traffic it gets 75%. What happened?',
    cfg: moons({ ...noisyCircle, leak: 0.9 }),
    cause: 'leak',
    target: 0.8,
    explain:
      '90% of the “test” points are exact copies of training points: the duplicate check gives it away. The model memorised its training data (it is the same overfitting network as in the previous case), so of course it scores well on copies of it. The test score was measuring memory, not generalisation. Removing the duplicates won’t make the model better, but it makes the scoreboard honest: then you can see the overfitting and treat it.',
    ruleOut: {
      overfit: 'The model is overfitting, but that alone would show up as a low test score. The puzzle here is why the test score was high; the root cause of the misleading number is the contaminated test set.',
      labels: 'The labels are fine: neighbouring training points mostly agree.',
    },
  },
  {
    id: 'random',
    title: 'Nothing to learn',
    brief: 'A new labelling script tagged our two-moons data. Bigger network, longer training, and still about 50% on everything. Is the network broken?',
    cfg: moons({ labelNoise: 1, hidden: [16, 16], steps: 3000 }),
    cause: 'labels',
    target: 0.95,
    explain:
      'The network is fine; the labels are not. Look at the label consistency check: only about half of the training points agree with the majority of their nearest neighbours — exactly what you’d expect if the labels were coin flips. Two points almost on top of each other are as likely to disagree as agree. There is no pattern to learn, so the best any model can do on new data is chance. No setting can fix this; only fixing the labels can.',
    ruleOut: {
      capacity: 'A bigger, longer-trained network was already tried. And with clean labels, this exact network gets 99%.',
      'lr-low': 'The loss is not creeping down; it sits at about 0.69, the loss of a model that has nothing better to say than 50–50.',
    },
  },
  {
    id: 'units',
    title: 'Lost in translation',
    brief: 'We merged data from two sensors. Momentum, step size 0.01, standard network. It learns a bit, then gets stuck at about two-thirds accuracy. The boundary looks odd: just vertical stripes.',
    cfg: moons({ xScale: 100, optimizer: 'momentum', lr: 0.01 }),
    cause: 'scale',
    target: 0.95,
    explain:
      'Check the input ranges: the first input runs from about −160 to 160, the second from about −1 to 1. One sensor recorded in different units. With inputs a hundred times larger, the weights on x get gradients a hundred times larger and the first-layer neurons saturate on x alone (all 8 are saturated), so the network can only cut along x: vertical stripes. Standardising each input (subtract the mean, divide by the spread) puts them on equal footing.',
    ruleOut: {
      'lr-high': 'The step size is a normal 0.01, and the problem is lopsided: everything to do with x is huge, everything to do with y is normal.',
      init: 'The starting weights are standard; it’s the inputs they multiply that are a hundred times too big.',
    },
  },
  {
    id: 'seesaw',
    title: 'The see-saw',
    brief: 'Big dataset (1,000 training points), plain SGD with step size 0.3, batches of 10. The loss bounces up and down in a regular rhythm, and the final model is barely better than a coin.',
    cfg: moons({ n: 2000, shuffle: false, optimizer: 'sgd', lr: 0.3 }),
    cause: 'shuffle',
    target: 0.95,
    explain:
      'Look at the batch composition: long runs of batches that are entirely class 0, then entirely class 1. The training file is sorted by class and is read in order, so for half of every pass the network only ever sees one class, learns that “everything is class 0”, and then gets dragged the other way. The rhythm of the loss is exactly one pass through the data. Shuffling the data mixes every batch.',
    ruleOut: {
      'lr-high': 'A large step size would make the loss jumpy, but not in a regular rhythm matching one pass through the data. And the same step size works with shuffled data.',
      labels: 'Neighbouring points agree on their labels; the data itself is fine.',
    },
  },
  {
    id: 'quiet',
    title: 'The quiet death',
    brief: 'ReLU network, Adam. I turned the step size up to 1 to train faster. No explosion, no NaNs, but it plateaued at around 65% and most of the neurons seem to be doing nothing.',
    cfg: moons({ activation: 'relu', lr: 1 }),
    cause: 'lr-high',
    target: 0.95,
    explain:
      'Most hidden ReLUs are dead now — but at step 0 none of them were (see “trouble at step 0”). The first few huge steps knocked their biases and weights so far that they output zero for every input, and a dead ReLU gets no gradient to recover. Same symptom as “Dead on arrival”, different cause: there, the starting weights were bad; here, the step size killed healthy neurons. Divergence is loud; this kind of failure is quiet.',
    ruleOut: {
      init: 'At step 0 every neuron was alive. The deaths happened during training.',
      capacity: 'With a sensible step size this network gets about 99%. The capacity is there; most of it has been killed.',
    },
  },
  {
    id: 'two-sensors-sorted',
    title: 'Two causes: two sensors, one sorted file',
    brief: 'Two sensors, one big file of 1,000 training points, momentum with step size 0.01. It is stuck at about two-thirds accuracy. Someone already tried a few things; nothing on its own got us to 95%.',
    cfg: moons({ xScale: 100, n: 2000, shuffle: false, optimizer: 'momentum', lr: 0.01 }),
    cause: 'scale',
    secondCause: 'shuffle',
    target: 0.95,
    explain:
      'Two faults, each with its own fingerprint. The input ranges show the first input running from −160 to 160 while the second stays within about ±1.1: one sensor records in different units, and 7 of the 8 first-layer neurons are saturated. The batch strip shows long runs of single-class batches: the file is sorted by class and read in order. Fix only the scale and fresh accuracy reaches 93.0%; fix only the shuffling and it stays at 65.7%; fix both and it reaches 97.8%. No single change in the fix bench reaches 95%: after you find the first cause, the evidence for the second is still on the screen.',
    ruleOut: {
      'lr-high': 'The step size is a normal 0.01 with momentum, and nothing diverges: the loss settles near 0.69 almost at once.',
      capacity: 'Two hidden layers of 8 neurons handle the moons easily once the data is in order.',
      labels: 'Neighbouring points agree on their labels; the data itself is fine.',
    },
  },
  {
    id: 'line-wild-steps',
    title: 'Two causes: a straight line taking wild steps',
    brief: 'Circle data, a model with no hidden layer at all, momentum with step size 3. The loss went up instead of down and the accuracy is a coin flip.',
    cfg: moons({ dataset: 'circle', hidden: [], optimizer: 'momentum', lr: 3 }),
    cause: 'capacity',
    secondCause: 'lr-high',
    target: 0.95,
    explain:
      'Two faults. A model with no hidden layer can only draw a straight line, and no straight line separates a circle from what surrounds it. And a step size of 3 with momentum is far too large: the loss climbs above its starting 0.73 straight away and ends at 1.91, worse than a coin flip’s 0.69. Add hidden layers alone and it still fails (50.5% on fresh data), because the steps are still wild; lower the step size alone and it fails too (36.7%), because a line can’t do it. Both together: 96.7%.',
    ruleOut: {
      overfit: 'Training accuracy (48%) is as bad as test accuracy (52%): nothing has been memorised.',
      scale: 'Both inputs span roughly −1.4 to 1.4.',
      init: 'There are no hidden neurons to start badly, and the trouble begins with the very first, oversized steps.',
    },
  },
];

// ---------------------------------------------------------------- the single-change fix space

export type Change = { knob: string; label: string; apply: (c: Config) => Config };

const choices = <T,>(knob: string, values: T[], get: (c: Config) => T, set: (c: Config, v: T) => Config, fmt: (v: T) => string): ((c: Config) => Change[]) => (c) =>
  values.filter((v) => JSON.stringify(v) !== JSON.stringify(get(c))).map((v) => ({ knob, label: fmt(v), apply: (cfg) => set(cfg, v) }));

export const KNOBS = {
  lr: [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
  optimizer: ['sgd', 'momentum', 'adam'] as const,
  activation: ['tanh', 'relu'] as const,
  hidden: [[], [4], [8, 8], [16, 16], [16, 16, 16]] as number[][],
  l2: [0, 0.001, 0.01, 0.1],
  batch: [1, 10, 50],
  steps: [500, 2000, 8000],
};

const hiddenLabel = (h: number[]) => (h.length ? h.join(' × ') : 'no hidden layer');

/** Every configuration that differs from `cfg` in exactly one knob. */
export function singleChanges(cfg: Config): Change[] {
  return [
    ...choices('step size', KNOBS.lr, (c) => c.lr, (c, v) => ({ ...c, lr: v }), (v) => `step size ${v}`)(cfg),
    ...choices('optimizer', [...KNOBS.optimizer], (c) => c.optimizer, (c, v) => ({ ...c, optimizer: v }), (v) => `optimizer ${v}`)(cfg),
    ...choices('activation', [...KNOBS.activation], (c) => c.activation, (c, v) => ({ ...c, activation: v }), (v) => `activation ${v}`)(cfg),
    ...choices('hidden layers', KNOBS.hidden, (c) => c.hidden, (c, v) => ({ ...c, hidden: v }), (v) => `hidden ${hiddenLabel(v)}`)(cfg),
    ...choices('weight decay', KNOBS.l2, (c) => c.l2, (c, v) => ({ ...c, l2: v }), (v) => `weight decay ${v}`)(cfg),
    ...choices('batch size', KNOBS.batch, (c) => c.batch, (c, v) => ({ ...c, batch: v }), (v) => `batch size ${v}`)(cfg),
    ...choices('steps', KNOBS.steps, (c) => c.steps, (c, v) => ({ ...c, steps: v }), (v) => `${v} steps`)(cfg),
    ...(cfg.init !== 'standard' ? [{ knob: 'initialisation', label: 'standard initialisation', apply: (c: Config) => ({ ...c, init: 'standard' as const }) }] : []),
    { knob: 'data', label: 'collect 4× more data', apply: (c: Config) => ({ ...c, n: c.n * 4 }) },
    ...(cfg.labelNoise > 0 ? [{ knob: 'labels', label: 'fix the labels', apply: (c: Config) => ({ ...c, labelNoise: 0 }) }] : []),
    { knob: 'normalise', label: cfg.normalise ? 'stop normalising inputs' : 'normalise the inputs', apply: (c: Config) => ({ ...c, normalise: !c.normalise }) },
    { knob: 'shuffle', label: cfg.shuffle ? 'stop shuffling' : 'shuffle the training data', apply: (c: Config) => ({ ...c, shuffle: !c.shuffle }) },
    ...(cfg.leak > 0 ? [{ knob: 'test set', label: 'remove duplicates from the test set', apply: (c: Config) => ({ ...c, leak: 0 }) }] : []),
  ];
}
