/** Ready-made setups for the experiments described in the lesson text. */
import type { Config, DatasetName } from './lab';

type Preset = { label: string; dataset: Exclude<DatasetName, 'draw'>; noise: number; n: number; trainFrac: number; cfg: Partial<Config> };
export const PRESETS: Record<string, Preset> = {
  linear: { label: 'A single neuron on the circle', dataset: 'circle', noise: 0.1, n: 200, trainFrac: 0.5, cfg: { features: ['x', 'y'], hidden: [] } },
  squares: { label: 'A single neuron with x² and y²', dataset: 'circle', noise: 0.1, n: 200, trainFrac: 0.5, cfg: { features: ['x2', 'y2'], hidden: [] } },
  spiralRaw: { label: 'Spiral from raw inputs', dataset: 'spiral', noise: 0.05, n: 300, trainFrac: 0.5, cfg: { features: ['x', 'y'], hidden: [8, 8, 8], activation: 'relu', lr: 0.01 } },
  hot: { label: 'Step size too big', dataset: 'moons', noise: 0.1, n: 200, trainFrac: 0.5, cfg: { features: ['x', 'y'], hidden: [6, 6], activation: 'tanh', optimizer: 'momentum', lr: 1 } },
  overfit: { label: 'Few noisy points, big network', dataset: 'circle', noise: 0.45, n: 100, trainFrac: 0.3, cfg: { features: ['x', 'y'], hidden: [8, 8, 8, 8], activation: 'relu', lr: 0.03 } },
};

