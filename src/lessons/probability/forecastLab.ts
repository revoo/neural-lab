// authored-by: Codex Meridian 6
import { softmax } from '../../lib/prob';

export const days = [50, 30, 15, 5];
export const dryDays = [55, 30, 15, 0];
export const correctTemperature = 1.7;
export const normalise = (counts: number[]) => counts.map((count) => count / counts.reduce((sum, value) => sum + value, 0));
export const exact = (counts: number[], reference = 0) => normalise(counts).map((p) => Math.log(p) - reference);
export const fixedLogits = exact(days).map((value) => value * correctTemperature);
export const tolerance = (brief: number) => (brief === 3 ? 0.005 : 0.02);
export const scores = (logits: number[], temperature: number) => softmax(logits, temperature);
export const gap = (forecast: number[], counts: number[]) =>
  Math.max(...normalise(counts).map((value, i) => Math.abs(value - forecast[i])));
export const ratio = (a: number, b: number, temperature: number) => Math.exp((a - b) / temperature);
export function roundLogits(logits: number[]) {
  return logits.map((value) => Math.round(value * 20) / 20);
}

/**
 * A probability as a percentage for display. Exactly 0 shows as "0%"; a positive
 * probability never rounds to zero: below 0.01% it gets two significant figures,
 * and below 0.0001% it shows as "< 0.0001%".
 */
export function shownPercent(p: number): string {
  if (p === 0) return '0%';
  const v = 100 * p;
  if (v >= 0.01) return v.toFixed(2) + '%';
  if (v >= 1e-4) return v.toPrecision(2) + '%';
  return '< 0.0001%';
}
