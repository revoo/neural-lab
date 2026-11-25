// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import { BRIEFS, allDesigns, designKey, labData, labTrainer, paramCount, printAndJudge, verdict, type Design } from './lab';
import { LAB_RESULTS, labOptimum, passingDesigns, passingSeed1 } from './labResults';
import { REAL_TOL, evaluate, nearestFace, type VAE } from './vae';

const key = (d: Design) => designKey(d);
const rec = (latent: number, hidden: number, beta: number) => LAB_RESULTS[key({ latent, hidden, beta })].seeds;

describe('the recorded exhaustive search', () => {
  it('covers every design with three seeds', () => {
    expect(allDesigns()).toHaveLength(54);
    for (const d of allDesigns()) expect(LAB_RESULTS[key(d)].seeds).toHaveLength(3);
  });

  it('Brief 1: 6 designs pass every seed; the smallest is 2 numbers, width 16, β = 0.3, and it is unique', () => {
    const pass = passingDesigns('printer');
    expect(pass.map(key).sort()).toEqual(['2-16-0.3', '2-32-0.3', '2-32-1', '3-16-1', '3-32-0.3', '3-32-1'].sort());
    expect(pass.every((d) => d.beta === 0.3 || d.beta === 1)).toBe(true);
    const opt = labOptimum('printer');
    expect(key(opt.design)).toBe('2-16-0.3');
    expect(opt.params).toBe(4884);
    expect(opt.ties).toBe(1);
    // judged on seed 1 alone, a lucky smaller design would have won
    expect(passingSeed1('printer')).toHaveLength(9);
    expect(key(opt.seed1Smallest)).toBe('3-8-0.1');
    expect(paramCount(opt.seed1Smallest)).toBe(2542);
    expect(verdict(BRIEFS[0], rec(3, 8, 0.1)[1]).pass).toBe(false);
    expect(verdict(BRIEFS[0], rec(3, 8, 0.1)[2]).pass).toBe(false);
  });

  it('Brief 2: 7 designs pass; the smallest is 2 numbers, width 16, β = 0.1; one plain autoencoder passes', () => {
    const pass = passingDesigns('compressor');
    expect(pass).toHaveLength(7);
    const opt = labOptimum('compressor');
    expect(key(opt.design)).toBe('2-16-0.1');
    expect(opt.ties).toBe(1);
    expect(pass.filter((d) => d.beta === 0).map(key)).toEqual(['3-32-0']);
  });

  it('one-number codes and β ≥ 3 never pass either brief', () => {
    for (const b of BRIEFS) for (const d of passingDesigns(b.id)) {
      expect(d.latent).toBeGreaterThan(1);
      expect(d.beta).toBeLessThan(3);
    }
  });

  it('quoted ranges: plain autoencoders print 4.5–46.5% real faces; two-number β = 0.1 prints 74.5–97%', () => {
    const ae = allDesigns().filter((d) => d.beta === 0).flatMap((d) => LAB_RESULTS[key(d)].seeds.map((s) => s.realShare));
    expect(Math.min(...ae)).toBe(0.045);
    expect(Math.max(...ae)).toBe(0.465);
    const b01 = allDesigns().filter((d) => d.beta === 0.1 && d.latent === 2).flatMap((d) => LAB_RESULTS[key(d)].seeds.map((s) => s.realShare));
    expect(Math.min(...b01)).toBe(0.745);
    expect(Math.max(...b01)).toBe(0.97);
  });

  it('β dial numbers (2 numbers, width 32, seed 1)', () => {
    expect(rec(2, 32, 0)[0].recon).toBeCloseTo(0.014, 3);
    expect(rec(2, 32, 1)[0].recon).toBeCloseTo(0.027, 3);
    expect(rec(2, 32, 0.1)[0].realShare).toBe(0.885);
    expect(rec(2, 32, 3)[0].cells).toBe(4);
    expect(rec(2, 32, 10)[0].cells).toBe(1);
  });
});

function windowShare(net: VAE, view: [number, number, number, number], n = 28) {
  let real = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const x = view[0] + ((j + 0.5) * (view[1] - view[0])) / n;
      const y = view[2] + ((i + 0.5) * (view[3] - view[2])) / n;
      if (nearestFace(net.decode([x, y]).out).dist <= REAL_TOL) real++;
    }
  return real / (n * n);
}

describe('fresh training reproduces the records and the prose', () => {
  it('re-trains 2 numbers, width 16, β = 0.3, seed 2 exactly as recorded', () => {
    const t = labTrainer({ latent: 2, hidden: 16, beta: 0.3 }, 2);
    t.advance(Infinity);
    const r = printAndJudge(t.net);
    const want = rec(2, 16, 0.3)[1];
    expect(r.recon).toBeCloseTo(want.recon, 5);
    expect(r.realShare).toBe(want.realShare);
    expect(r.cells).toBe(want.cells);
  }, 60_000);

  it('the plain autoencoder: codes span about −2…15 by −0.3…10; 31% of the −3…3 window and 70% of the zoomed window decode to faces; a fitted bell gives 92%', () => {
    const t = labTrainer({ latent: 2, hidden: 32, beta: 0 }, 1);
    t.advance(Infinity);
    const codes = labData().imgs.map((x) => t.net.code(x));
    const xs = codes.map((c) => c[0]);
    const ys = codes.map((c) => c[1]);
    const [ax, bx, ay, by] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    expect(ax).toBeCloseTo(-1.88, 1);
    expect(bx).toBeCloseTo(14.89, 1);
    expect(ay).toBeCloseTo(-0.32, 1);
    expect(by).toBeCloseTo(10.27, 1);
    expect(Math.round(100 * windowShare(t.net, [-3, 3, -3, 3]))).toBe(31);
    const half = Math.max(bx - ax, by - ay) / 2 + 0.3;
    const cx = (ax + bx) / 2;
    const cy = (ay + by) / 2;
    expect(Math.round(100 * windowShare(t.net, [cx - half, cx + half, cy - half, cy + half]))).toBe(70);
    expect(Math.round(100 * evaluate(t.net, t.data).realShare)).toBe(92);
  }, 60_000);

  it('the VAE at β = 1: the whole −3…3 window decodes to faces', () => {
    const t = labTrainer({ latent: 2, hidden: 32, beta: 1 }, 1);
    t.advance(Infinity);
    expect(windowShare(t.net, [-3, 3, -3, 3])).toBe(1);
  }, 60_000);

  it('a stalled plain autoencoder: codes grow from ~0.5 to ~5 and saturate every decoder tanh unit', () => {
    const t = labTrainer({ latent: 2, hidden: 16, beta: 0 }, 2);
    const rms = () => {
      const v = labData().imgs.flatMap((x) => t.net.code(x));
      return Math.sqrt(v.reduce((s, a) => s + a * a, 0) / v.length);
    };
    expect(rms()).toBeLessThan(0.7);
    t.advance(Infinity);
    expect(rms()).toBeGreaterThan(4);
    let sat = 0;
    let n = 0;
    for (const x of labData().imgs) for (const h of t.net.decode(t.net.code(x)).h) {
      n++;
      if (Math.abs(h) > 0.99) sat++;
    }
    expect(sat / n).toBe(1);
    expect(rec(2, 16, 0)[1].recon).toBeGreaterThan(0.11); // it drew the average face
  }, 60_000);
});
