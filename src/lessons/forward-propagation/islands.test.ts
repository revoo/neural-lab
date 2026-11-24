import { describe, expect, it } from 'vitest';
import { forward, onLand, paramCount, multiplyAdds, LAYERS } from './islands';

describe('two-islands network', () => {
  it('says land in the middle of each island and sea far away', () => {
    expect(forward([-0.9, 0]).a[3][0]).toBeGreaterThan(0.9);
    expect(forward([1, 0.45]).a[3][0]).toBeGreaterThan(0.9);
    expect(forward([0, 1.8]).a[3][0]).toBeLessThan(0.1);
    expect(forward([1.8, -1.5]).a[3][0]).toBeLessThan(0.1);
  });
  it('agrees with the exact geometry away from the coastlines', () => {
    let agree = 0;
    let total = 0;
    for (let x = -2.2; x <= 2.2; x += 0.05)
      for (let y = -2.2; y <= 2.2; y += 0.05) {
        const out = forward([x, y]).a[3][0];
        if (Math.abs(out - 0.5) < 0.3) continue; // skip the blurry shoreline
        total++;
        if (out > 0.5 === onLand(x, y)) agree++;
      }
    expect(agree / total).toBeGreaterThan(0.99);
  });
  it('has the stated sizes', () => {
    expect(LAYERS.map((L) => [L.W.length, L.W[0].length])).toEqual([[6, 2], [2, 6], [1, 2]]);
    expect(paramCount()).toBe(6 * 2 + 6 + 2 * 6 + 2 + 2 + 1);
    expect(paramCount()).toBe(35);
    expect(multiplyAdds()).toBe(26);
  });
});
