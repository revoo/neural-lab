// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { bytes, crossover, flops, gflops, intensity, repsFor } from './bench';

describe('V2-WP3 throughput benchmark arithmetic', () => {
  it('counts FLOPs and bytes', () => {
    expect(flops('mm', 64)).toBe(524_288);
    expect(flops('mv', 1024)).toBe(2_097_152);
    expect(bytes('mm', 1024)).toBe(12_582_912);
  });
  it('matrix products get more FLOPs per byte as they grow; matrix-vector products stay near ½', () => {
    expect(intensity('mm', 1024)).toBeCloseTo(1024 / 6, 6);
    expect(intensity('mm', 1024).toFixed(0)).toBe('171');
    expect(intensity('mm', 32).toFixed(1)).toBe('5.3');
    for (const n of [256, 4096]) expect(intensity('mv', n)).toBeGreaterThan(0.49), expect(intensity('mv', n)).toBeLessThan(0.5);
  });
  it('converts timings to GFLOP/s and finds the crossover', () => {
    expect(gflops('mm', 1000, 10, 2000)).toBeCloseTo(10, 9);
    expect(crossover([{ n: 32, cpu: 2, gpu: 0.5 }, { n: 64, cpu: 2, gpu: 1.9 }, { n: 128, cpu: 2, gpu: 8 }])).toBe(128);
    expect(crossover([{ n: 32, cpu: 2 }])).toBeNull();
    expect(repsFor(10, 4, 60, 512)).toBe(24);
    expect(repsFor(0, 4, 60, 512)).toBe(512);
  });
});
