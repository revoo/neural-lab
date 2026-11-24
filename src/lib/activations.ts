/** Activation functions with their derivatives. Shared by the neuron, activation, and training lessons. */
export type ActivationName = 'linear' | 'step' | 'sigmoid' | 'tanh' | 'relu' | 'leakyRelu' | 'gelu' | 'silu' | 'softplus';

export type Activation = {
  name: ActivationName;
  label: string;
  f: (x: number) => number;
  df: (x: number) => number;
  /** LaTeX formula */
  tex: string;
};

const sigmoid = (x: number) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));

/** Abramowitz–Stegun 7.1.26 approximation, max error ~1.5e-7. */
export function erf(x: number): number {
  const s = Math.sign(x);
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}
const phi = (x: number) => 0.5 * (1 + erf(x / Math.SQRT2)); // standard normal CDF
const pdf = (x: number) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

export const ACTIVATIONS: Record<ActivationName, Activation> = {
  linear: { name: 'linear', label: 'Linear (none)', f: (x) => x, df: () => 1, tex: 'f(x) = x' },
  step: {
    name: 'step', label: 'Step', f: (x) => (x >= 0 ? 1 : 0), df: () => 0,
    tex: 'f(x) = \\begin{cases} 1 & x \\ge 0 \\\\ 0 & x < 0 \\end{cases}',
  },
  sigmoid: {
    name: 'sigmoid', label: 'Sigmoid', f: sigmoid, df: (x) => sigmoid(x) * (1 - sigmoid(x)),
    tex: '\\sigma(x) = \\frac{1}{1 + e^{-x}}',
  },
  tanh: { name: 'tanh', label: 'Tanh', f: Math.tanh, df: (x) => 1 - Math.tanh(x) ** 2, tex: '\\tanh(x) = \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}' },
  relu: { name: 'relu', label: 'ReLU', f: (x) => (x > 0 ? x : 0), df: (x) => (x > 0 ? 1 : 0), tex: '\\mathrm{ReLU}(x) = \\max(0, x)' },
  leakyRelu: {
    name: 'leakyRelu', label: 'Leaky ReLU', f: (x) => (x > 0 ? x : 0.1 * x), df: (x) => (x > 0 ? 1 : 0.1),
    tex: '\\mathrm{LeakyReLU}(x) = \\max(0.1x,\\, x)',
  },
  gelu: {
    name: 'gelu', label: 'GELU', f: (x) => x * phi(x), df: (x) => phi(x) + x * pdf(x),
    tex: '\\mathrm{GELU}(x) = x \\cdot \\Phi(x)',
  },
  silu: {
    name: 'silu', label: 'SiLU / Swish', f: (x) => x * sigmoid(x), df: (x) => sigmoid(x) * (1 + x * (1 - sigmoid(x))),
    tex: '\\mathrm{SiLU}(x) = x \\cdot \\sigma(x)',
  },
  softplus: {
    name: 'softplus', label: 'Softplus', f: (x) => (x > 20 ? x : Math.log1p(Math.exp(x))), df: sigmoid,
    tex: '\\mathrm{softplus}(x) = \\ln(1 + e^{x})',
  },
};

export { sigmoid };
