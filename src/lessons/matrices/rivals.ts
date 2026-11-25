// authored-by: Claude Opus 5.5
/**
 * "Why this, and not something else?" for matrices: three ways to get several
 * scores from one email (the lesson's panel of three judges on [4, 5, 0]),
 * and the same three for a real first layer, 784 pixels in and 128 features out.
 */
import type { Rival } from '../../components/RivalTable';

/** The panel of judges from the lesson: spammy, urgent, personal. */
export const JUDGES = [
  [1, 2, -3],
  [0, 1, 4],
  [-1, 0, 2],
];
export const EMAIL = [4, 5, 0];

/** Rival (a): one separate dot product per judge. Returns the scores and how many separate operations ran. */
export function loopOfDots(M: number[][], x: number[]): { out: number[]; operations: number; multiplyAdds: number } {
  const out: number[] = [];
  let multiplyAdds = 0;
  for (const row of M) {
    let s = 0;
    for (let i = 0; i < x.length; i++) {
      s += row[i] * x[i];
      multiplyAdds++;
    }
    out.push(s);
  }
  return { out, operations: M.length, multiplyAdds };
}

/** Rival (c): the matrix, as one operation (here, the columns view: a blend of the columns). */
export function matrixTimes(M: number[][], x: number[]): number[] {
  const out = M.map(() => 0);
  x.forEach((xj, j) => M.forEach((row, i) => (out[i] += row[j] * xj)));
  return out;
}

/** Rival (b): a lookup table with one row per possible input. Counts from 0 to `top` on each of `inputs` measurements. */
export function tableRows(inputs: number, levels: number): number {
  return levels ** inputs;
}
/** log10 of the number of rows, for tables too big to count. */
export const tableRowsLog10 = (inputs: number, levels: number) => inputs * Math.log10(levels);

export const LAYER = { inputs: 784, outputs: 128 };
export const layerMultiplyAdds = () => LAYER.inputs * LAYER.outputs;

const fmt = (n: number) => n.toLocaleString('en-GB');

export function rivals(): Rival[] {
  const loop = loopOfDots(JUDGES, EMAIL);
  const rows = tableRows(3, 10);
  return [
    {
      name: 'A loop of dot products',
      result: `One dot product per judge gives the same [${loop.out.join(', ')}], from the same ${loop.multiplyAdds} multiply-adds, but as ${loop.operations} separate jobs. For a first layer turning 784 pixels into 128 features that is ${LAYER.outputs} separate jobs of ${LAYER.inputs} multiply-adds each (${fmt(layerMultiplyAdds())} in all), where the matrix hands the GPU one.`,
      verdicts: ['pass', 'pass', 'partly'],
    },
    {
      name: 'A lookup table',
      result: `Write down the three scores for every possible email. If each count runs from 0 to 9, that is ${fmt(rows)} rows, ${fmt(rows * 3)} numbers against the matrix’s 9, and an email with 10 links isn’t in it at all. For 784-pixel greyscale images the table needs about 10^${Math.round(tableRowsLog10(784, 256))} rows. Learning one row teaches it nothing about its neighbours, so there is no slope to share.`,
      verdicts: ['fail', 'fail', 'fail'],
    },
    {
      name: 'A matrix',
      result: `Nine numbers handle every email, including ones nobody has seen, such as [12, 0, 1] → [${matrixTimes(JUDGES, [12, 0, 1]).join(', ')}]. Each score’s slope with respect to each entry is just the matching input, and the whole layer is one grid of multiply-adds. Its limit: lines stay lines, so it can never bend the data. The lab runs straight into that.`,
      verdicts: ['pass', 'pass', 'pass'],
      winner: true,
    },
  ];
}
