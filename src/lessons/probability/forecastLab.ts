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
