// authored-by: Claude Opus 5.5
/**
 * V2-WP3: the GAN lab's grid search on WebGPU.
 *
 * `evaluateLabSetting` (gan.ts) trains 5 seeds of one setting on the CPU. Here every (setting, seed)
 * pair of the lab's grid is one WGSL invocation that runs the *whole* duel itself: mini-batch
 * sampling (mulberry32 reproduced with u32 maths, Box–Muller in f32), D's forward/backward and Adam
 * update, G's update through the frozen D, and G's outputs on the fixed evaluation noise at every
 * 20-step checkpoint. The start weights come from the CPU `GanRunner`s, so both tiers begin from
 * identical networks; the GPU computes in 32-bit floats where the CPU uses 64-bit, so long runs
 * drift apart (a GAN duel amplifies tiny differences) and the browser check compares the two with
 * a tolerance (`scripts/check-webgpu.cjs`).
 *
 * The training is split into one dispatch per checkpoint window, so no single dispatch runs long
 * enough to trip a GPU watchdog. The CPU turns the returned outputs into the lab's metrics with the
 * same functions as `evaluateLabSetting` (`labResultFromFakes`, tested in Node against it).
 */
import { createRng } from '../../lib/rng';
import type { MLP } from '../../lib/mlp';
import {
  evaluateModes,
  GanRunner,
  LAB_BASE_LR_D,
  LAB_CHECK_INTERVAL,
  LAB_MAX_STEPS,
  LAB_SEEDS,
  sampleNoise,
  sampleReal,
  wasserstein1D,
  type LabEvalResult,
  type LabSetting,
} from './gan';

export const EVAL_N = 360;
export const CHECKPOINTS = LAB_MAX_STEPS / LAB_CHECK_INTERVAL;
const H = 14;
const BATCH = 36;
/** Flat parameter counts, G laid out with room for 2 noise inputs whatever the setting. */
const GP = H * 2 + H * H + H + H + H + 1; // 267
const DP = H + H * H + H + H + H + 1; // 253
/** Per run: G params, m, v, grad; then D params, m, v, grad. */
const STRIDE = 4 * GP + 4 * DP;

const EVAL_REAL = sampleReal(EVAL_N, createRng(7777));
export const evalNoise = (noiseDim: number) => sampleNoise(EVAL_N, noiseDim, createRng(9991 + noiseDim * 13));

/** Flatten an MLP as [W0 rows, W1 rows, W2 rows, b0, b1, b2]; G's first layer is padded to 2 inputs. */
function flatten(net: MLP, padIn: number) {
  const out: number[] = [];
  net.W.forEach((L, l) => L.forEach((row) => out.push(...row, ...(l === 0 ? new Array(padIn - row.length).fill(0) : []))));
  net.b.forEach((r) => out.push(...r));
  return out;
}

export type GanJob = { setting: LabSetting; seed: number };

export const gridJobs = (settings: LabSetting[]): GanJob[] => settings.flatMap((setting) => LAB_SEEDS.map((seed) => ({ setting, seed })));

/** Everything the GPU needs, packed: start weights, per-run config and RNG state, evaluation noise. */
export function packJobs(jobs: GanJob[]) {
  const n = jobs.length;
  const state = new Float32Array(n * STRIDE);
  const cfgU = new Uint32Array(n * 8);
  const cfgF = new Float32Array(cfgU.buffer);
  jobs.forEach(({ setting, seed }, r) => {
    const runner = new GanRunner({ seed, noiseDim: setting.noiseDim, lrD: LAB_BASE_LR_D, lrG: LAB_BASE_LR_D * setting.lrRatio, dSteps: setting.dSteps, batchSize: BATCH, objective: 'non-saturating' });
    state.set(flatten(runner.G, 2), r * STRIDE);
    state.set(flatten(runner.D, 1), r * STRIDE + 4 * GP);
    // the runner's own training stream starts at seed·3037 + 53 (see GanRunner)
    cfgU[r * 8] = (seed * 3037 + 53) >>> 0;
    cfgU[r * 8 + 1] = setting.noiseDim;
    cfgU[r * 8 + 2] = setting.dSteps;
    cfgU[r * 8 + 3] = 0; // Adam step count for D
    cfgU[r * 8 + 4] = 0; // Adam step count for G
    cfgF[r * 8 + 5] = LAB_BASE_LR_D;
    cfgF[r * 8 + 6] = LAB_BASE_LR_D * setting.lrRatio;
  });
  const ez = new Float32Array(2 * EVAL_N * 2);
  evalNoise(1).forEach((z, e) => (ez[e * 2] = z[0]));
  evalNoise(2).forEach((z, e) => ez.set(z, EVAL_N * 2 + e * 2));
  return { n, state, cfg: cfgU, ez };
}

/**
 * The lab's metrics for one setting from G's outputs on the evaluation noise:
 * `fakes[seedIndex][checkpoint]` is that seed's 360 outputs after (checkpoint + 1) × 20 outer steps.
 * Mirrors `evaluateLabSetting` exactly.
 */
export function labResultFromFakes(setting: LabSetting, fakes: ArrayLike<number>[][]): LabEvalResult {
  let updatesToCover3: number | null = null;
  const per = (f: ArrayLike<number>) => {
    const xs = Array.from(f);
    return { ...evaluateModes(xs), w1: wasserstein1D(EVAL_REAL, xs) };
  };
  const S = fakes.length;
  for (let c = 0; c < CHECKPOINTS; c++) {
    let sm = 0;
    let sw = 0;
    let so = 0;
    for (let s = 0; s < S; s++) {
      const m = per(fakes[s][c]);
      sm += m.modesCovered;
      sw += m.w1;
      so += m.offModeShare;
    }
    if (updatesToCover3 === null && sm / S >= 2.8 && sw / S <= 0.45 && so / S <= 0.25) updatesToCover3 = (c + 1) * LAB_CHECK_INTERVAL * (setting.dSteps + 1);
  }
  const perSeed = fakes.map((f, s) => {
    const m = per(f[CHECKPOINTS - 1]);
    return { seed: LAB_SEEDS[s], modes: m.modesCovered, w1: m.w1, offMode: m.offModeShare };
  });
  const mean = (k: 'modes' | 'w1' | 'offMode') => perSeed.reduce((a, p) => a + p[k], 0) / S;
  return {
    setting,
    meanModes: mean('modes'),
    minModes: Math.min(...perSeed.map((p) => p.modes)),
    allThreeSeeds: perSeed.filter((p) => p.modes === 3).length,
    meanW1: mean('w1'),
    meanOffMode: mean('offMode'),
    updatesToCover3,
    finalUpdates: LAB_MAX_STEPS * (setting.dSteps + 1),
    perSeed,
  };
}

/** CPU reference for the same outputs (64-bit `GanRunner`): fakes[checkpoint] for one job. */
export function cpuFakes({ setting, seed }: GanJob): number[][] {
  const r = new GanRunner({ seed, noiseDim: setting.noiseDim, lrD: LAB_BASE_LR_D, lrG: LAB_BASE_LR_D * setting.lrRatio, dSteps: setting.dSteps, batchSize: BATCH, objective: 'non-saturating' });
  const ez = evalNoise(setting.noiseDim);
  const out: number[][] = [];
  for (let c = 0; c < CHECKPOINTS; c++) {
    r.step(LAB_CHECK_INTERVAL);
    out.push(ez.map((z) => r.G.predict(z)[0]));
  }
  return out;
}

// ------------------------------------------------------------------ WGSL

const WGSL = /* wgsl */ `
const H: u32 = 14u;
const B: u32 = 36u;
const GP: u32 = ${GP}u;
const DP: u32 = ${DP}u;
const STRIDE: u32 = ${STRIDE}u;
const EVAL_N: u32 = ${EVAL_N}u;
// G offsets (W0 is 14 × 2, row-major)
const GW1: u32 = 28u; const GW2: u32 = 224u; const GB0: u32 = 238u; const GB1: u32 = 252u; const GB2: u32 = 266u;
// D offsets
const DW1: u32 = 14u; const DW2: u32 = 210u; const DB0: u32 = 224u; const DB1: u32 = 238u; const DB2: u32 = 252u;

struct U { runs: u32, ck: u32, steps: u32, pad: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> st: array<f32>;
@group(0) @binding(2) var<storage, read_write> cfg: array<u32>;
@group(0) @binding(3) var<storage, read> ez: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;

var<private> rs: u32;
var<private> base: u32;

// mulberry32, exactly as src/lib/rng.ts (u32 arithmetic wraps like Math.imul)
fn next_u() -> u32 {
  rs = rs + 0x6d2b79f5u;
  var t = rs;
  t = (t ^ (t >> 15u)) * (t | 1u);
  t = t ^ (t + (t ^ (t >> 7u)) * (t | 61u));
  return t ^ (t >> 14u);
}
fn next_f() -> f32 { return f32(next_u()) * 2.3283064365386963e-10; }
fn normal() -> f32 {
  var a = next_u();
  loop { if (a != 0u) { break; } a = next_u(); }
  let uu = f32(a) * 2.3283064365386963e-10;
  let v = next_f();
  return sqrt(-2.0 * log(uu)) * cos(6.283185307179586 * v);
}
fn th(z: f32) -> f32 { return tanh(clamp(z, -15.0, 15.0)); }
fn sig(z: f32) -> f32 { return 1.0 / (1.0 + exp(-clamp(z, -80.0, 80.0))); }

// parameter, m, v, grad blocks: G at base, D at base + 4·GP
fn gp(i: u32) -> u32 { return base + i; }
fn gg(i: u32) -> u32 { return base + 3u * GP + i; }
fn dp(i: u32) -> u32 { return base + 4u * GP + i; }
fn dg(i: u32) -> u32 { return base + 4u * GP + 3u * DP + i; }

fn g_forward(z0: f32, z1: f32, h1: ptr<function, array<f32, 14>>, h2: ptr<function, array<f32, 14>>) -> f32 {
  for (var j = 0u; j < H; j++) { (*h1)[j] = th(st[gp(j * 2u)] * z0 + st[gp(j * 2u + 1u)] * z1 + st[gp(GB0 + j)]); }
  for (var j = 0u; j < H; j++) {
    var s = st[gp(GB1 + j)];
    for (var i = 0u; i < H; i++) { s += st[gp(GW1 + j * H + i)] * (*h1)[i]; }
    (*h2)[j] = th(s);
  }
  var x = st[gp(GB2)];
  for (var i = 0u; i < H; i++) { x += st[gp(GW2 + i)] * (*h2)[i]; }
  return x;
}
fn d_forward(x: f32, a1: ptr<function, array<f32, 14>>, a2: ptr<function, array<f32, 14>>) -> f32 {
  for (var j = 0u; j < H; j++) { (*a1)[j] = th(st[dp(j)] * x + st[dp(DB0 + j)]); }
  for (var j = 0u; j < H; j++) {
    var s = st[dp(DB1 + j)];
    for (var i = 0u; i < H; i++) { s += st[dp(DW1 + j * H + i)] * (*a1)[i]; }
    (*a2)[j] = th(s);
  }
  var z = st[dp(DB2)];
  for (var i = 0u; i < H; i++) { z += st[dp(DW2 + i)] * (*a2)[i]; }
  return sig(z);
}
/** Backprop d3 = dLoss/dz_out through D. Accumulates D's gradient when acc; returns dLoss/dx. */
fn d_backward(x: f32, d3: f32, a1: ptr<function, array<f32, 14>>, a2: ptr<function, array<f32, 14>>, acc: bool) -> f32 {
  var d2: array<f32, 14>;
  for (var i = 0u; i < H; i++) {
    if (acc) { st[dg(DW2 + i)] += d3 * (*a2)[i]; }
    d2[i] = st[dp(DW2 + i)] * d3 * (1.0 - (*a2)[i] * (*a2)[i]);
  }
  if (acc) { st[dg(DB2)] += d3; }
  var d1: array<f32, 14>;
  for (var j = 0u; j < H; j++) {
    if (acc) {
      st[dg(DB1 + j)] += d2[j];
      for (var i = 0u; i < H; i++) { st[dg(DW1 + j * H + i)] += d2[j] * (*a1)[i]; }
    }
  }
  for (var i = 0u; i < H; i++) {
    var s = 0.0;
    for (var j = 0u; j < H; j++) { s += st[dp(DW1 + j * H + i)] * d2[j]; }
    d1[i] = s * (1.0 - (*a1)[i] * (*a1)[i]);
  }
  var dx = 0.0;
  for (var i = 0u; i < H; i++) {
    if (acc) { st[dg(DB0 + i)] += d1[i]; st[dg(i)] += d1[i] * x; }
    dx += st[dp(i)] * d1[i];
  }
  return dx;
}
fn adam(p0: u32, n: u32, lr: f32, t: u32, scale: f32) {
  let c1 = 1.0 - pow(0.5, f32(t));
  let c2 = 1.0 - pow(0.999, f32(t));
  for (var i = 0u; i < n; i++) {
    let g = st[p0 + 3u * n + i] * scale;
    let m = 0.5 * st[p0 + n + i] + 0.5 * g;
    let v = 0.999 * st[p0 + 2u * n + i] + 0.001 * g * g;
    st[p0 + n + i] = m;
    st[p0 + 2u * n + i] = v;
    st[p0 + i] -= (lr * (m / c1)) / (sqrt(v / c2) + 1e-8);
    st[p0 + 3u * n + i] = 0.0;
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= u.runs) { return; }
  base = r * STRIDE;
  rs = cfg[r * 8u];
  let nz = cfg[r * 8u + 1u];
  let dSteps = cfg[r * 8u + 2u];
  var tD = cfg[r * 8u + 3u];
  var tG = cfg[r * 8u + 4u];
  let lrD = bitcast<f32>(cfg[r * 8u + 5u]);
  let lrG = bitcast<f32>(cfg[r * 8u + 6u]);
  var h1: array<f32, 14>; var h2: array<f32, 14>; var a1: array<f32, 14>; var a2: array<f32, 14>;
  var reals: array<f32, 36>;
  var zs: array<f32, 72>;
  for (var step = 0u; step < u.steps; step++) {
    for (var k = 0u; k < dSteps; k++) {
      // sampleReal(B) then sampleNoise(B, nz), in the CPU's order
      for (var i = 0u; i < B; i++) {
        var c = 0.0;
        if (i % 3u == 0u) { c = -2.4; } else if (i % 3u == 2u) { c = 2.4; }
        reals[i] = c + 0.28 * normal();
      }
      for (var i = 0u; i < B; i++) { for (var q = 0u; q < nz; q++) { zs[i * 2u + q] = normal(); } }
      for (var i = 0u; i < B; i++) {
        let x = reals[i];
        let p = d_forward(x, &a1, &a2);
        _ = d_backward(x, p - 1.0, &a1, &a2, true);
      }
      for (var i = 0u; i < B; i++) {
        var z1 = 0.0; if (nz == 2u) { z1 = zs[i * 2u + 1u]; }
        let x = g_forward(zs[i * 2u], z1, &h1, &h2);
        let p = d_forward(x, &a1, &a2);
        _ = d_backward(x, p, &a1, &a2, true);
      }
      tD += 1u;
      adam(base + 4u * GP, DP, lrD, tD, 1.0 / 72.0);
    }
    // generator step, non-saturating loss −log D(G(z)), through the updated D
    for (var i = 0u; i < B; i++) { for (var q = 0u; q < nz; q++) { zs[i * 2u + q] = normal(); } }
    for (var s = 0u; s < B; s++) {
      var z1 = 0.0; if (nz == 2u) { z1 = zs[s * 2u + 1u]; }
      let z0 = zs[s * 2u];
      let x = g_forward(z0, z1, &h1, &h2);
      let p = d_forward(x, &a1, &a2);
      let dx = d_backward(x, p - 1.0, &a1, &a2, false);
      var e2: array<f32, 14>;
      for (var i = 0u; i < H; i++) {
        st[gg(GW2 + i)] += dx * h2[i];
        e2[i] = st[gp(GW2 + i)] * dx * (1.0 - h2[i] * h2[i]);
      }
      st[gg(GB2)] += dx;
      for (var j = 0u; j < H; j++) {
        st[gg(GB1 + j)] += e2[j];
        for (var i = 0u; i < H; i++) { st[gg(GW1 + j * H + i)] += e2[j] * h1[i]; }
      }
      for (var i = 0u; i < H; i++) {
        var a = 0.0;
        for (var j = 0u; j < H; j++) { a += st[gp(GW1 + j * H + i)] * e2[j]; }
        let e1 = a * (1.0 - h1[i] * h1[i]);
        st[gg(GB0 + i)] += e1;
        st[gg(i * 2u)] += e1 * z0;
        if (nz == 2u) { st[gg(i * 2u + 1u)] += e1 * z1; }
      }
    }
    tG += 1u;
    adam(base, GP, lrG, tG, 1.0 / 36.0);
  }
  cfg[r * 8u] = rs;
  cfg[r * 8u + 3u] = tD;
  cfg[r * 8u + 4u] = tG;
  // G on the fixed evaluation noise at this checkpoint
  let eb = select(0u, EVAL_N * 2u, nz == 2u);
  for (var e = 0u; e < EVAL_N; e++) {
    var z1 = 0.0; if (nz == 2u) { z1 = ez[eb + e * 2u + 1u]; }
    outp[(r * ${CHECKPOINTS}u + u.ck) * EVAL_N + e] = g_forward(ez[eb + e * 2u], z1, &h1, &h2);
  }
}
`;

/**
 * Train every job on the GPU. Resolves to fakes[job][checkpoint] (360 outputs each).
 * `onProgress` is called after each checkpoint window with the fraction done.
 */
export async function runGanJobsGpu(device: GPUDevice, jobs: GanJob[], onProgress?: (frac: number) => void): Promise<Float32Array[][]> {
  const { n, state, cfg, ez } = packJobs(jobs);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const mk = (bytes: number) => device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage });
  const stB = mk(state.byteLength);
  const cfgB = mk(cfg.byteLength);
  const ezB = mk(ez.byteLength);
  const outFloats = n * CHECKPOINTS * EVAL_N;
  const outB = mk(outFloats * 4);
  device.queue.writeBuffer(stB, 0, state);
  device.queue.writeBuffer(cfgB, 0, cfg);
  device.queue.writeBuffer(ezB, 0, ez);
  const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: WGSL }), entryPoint: 'main' } });
  const extra: GPUBuffer[] = [];
  try {
    for (let c = 0; c < CHECKPOINTS; c++) {
      const ub = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      extra.push(ub);
      device.queue.writeBuffer(ub, 0, new Uint32Array([n, c, LAB_CHECK_INTERVAL, 0]));
      const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [ub, stB, cfgB, ezB, outB].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(n / 64));
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      onProgress?.((c + 1) / CHECKPOINTS);
    }
    const staging = device.createBuffer({ size: outB.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    extra.push(staging);
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(outB, 0, staging, 0, outB.size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(staging.getMappedRange().slice(0, outFloats * 4));
    staging.unmap();
    return jobs.map((_, r) => Array.from({ length: CHECKPOINTS }, (_, c) => all.subarray((r * CHECKPOINTS + c) * EVAL_N, (r * CHECKPOINTS + c + 1) * EVAL_N)));
  } finally {
    [stB, cfgB, ezB, outB, ...extra].forEach((b) => b.destroy());
  }
}

/** Run settings on the GPU and score each one exactly as the lab does. */
export async function evaluateSettingsGpu(device: GPUDevice, settings: LabSetting[], onProgress?: (frac: number) => void): Promise<LabEvalResult[]> {
  const jobs = gridJobs(settings);
  const fakes = await runGanJobsGpu(device, jobs, onProgress);
  const S = LAB_SEEDS.length;
  return settings.map((s, i) => labResultFromFakes(s, fakes.slice(i * S, (i + 1) * S)));
}

/**
 * Browser check (run by `scripts/check-webgpu.cjs`): the WGSL duel matches the 64-bit CPU runner
 * to within 0.01 (outputs span about ±2.4) before chaos can set in, after the first 20 steps, and the full grid on the GPU keeps
 * what the lesson says it keeps: every recorded Brief 2 pass still passes, and the fastest route
 * to the bar is still 40 updates.
 */
export async function ganGpuParity(device: GPUDevice) {
  const { LAB_GRID_RESULTS, passesBrief2 } = await import('./results');
  const early = gridJobs([
    { lrRatio: 0.5, dSteps: 1, noiseDim: 2 },
    { lrRatio: 4, dSteps: 4, noiseDim: 1 },
  ]);
  const f = await runGanJobsGpu(device, early);
  let worst = 0;
  early.forEach((j, r) => {
    const c = cpuFakes(j)[0];
    for (let i = 0; i < EVAL_N; i++) worst = Math.max(worst, Math.abs(f[r][0][i] - c[i]));
  });
  const settings = LAB_GRID_RESULTS.map((r) => ({ lrRatio: r.lrRatio, dSteps: r.dSteps, noiseDim: r.noiseDim }));
  const res = await evaluateSettingsGpu(device, settings);
  const keptB2 = LAB_GRID_RESULTS.every((r, i) => !passesBrief2(r) || passesBrief2(res[i]));
  const fastest = Math.min(...res.map((r) => r.updatesToCover3 ?? Infinity));
  // settings where G learns more slowly than D stay close to the recorded search (quoted in the lesson)
  const slow = LAB_GRID_RESULTS.map((r, i) => ({ r, g: res[i] })).filter(({ r }) => r.lrRatio < 1);
  const drift = Math.max(...slow.map(({ r, g }) => Math.max(Math.abs(g.meanModes - r.meanModes) / 0.2, Math.abs(g.meanW1 - r.meanW1) / 0.06)));
  return [
    { name: 'GAN grid: 12 slower-G settings within 0.2 modes and 0.06 W1 of the record (scaled)', ok: slow.length === 12 && drift <= 1 + 1e-9, worst: drift, tol: 1 },
    { name: 'GAN duel: G samples after 20 steps vs CPU', ok: worst < 1e-2, worst, tol: 1e-2 },
    { name: 'GAN grid: recorded Brief 2 passes still pass', ok: keptB2, worst: keptB2 ? 0 : 1, tol: 0 },
    { name: 'GAN grid: fastest route to the bar is 40 updates', ok: fastest === 40, worst: Math.abs(fastest - 40), tol: 0 },
  ];
}
