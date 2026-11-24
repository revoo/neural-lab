/** Numerical derivatives. Used both in widgets (to show slopes) and in tests (to check analytic gradients). */
export const derivative = (f: (x: number) => number, x: number, h = 1e-5): number =>
  (f(x + h) - f(x - h)) / (2 * h);

export const partials = (f: (v: number[]) => number, v: number[], h = 1e-5): number[] =>
  v.map((_, i) => {
    const up = v.slice();
    const dn = v.slice();
    up[i] += h;
    dn[i] -= h;
    return (f(up) - f(dn)) / (2 * h);
  });
