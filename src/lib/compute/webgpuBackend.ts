// authored-by: Claude Opus 5.5
/**
 * V2-WP3, production tier: WGSL compute shaders behind the same `MlpEngine` interface as the CPU
 * reference.
 *
 * What makes it fast on small models is what it *doesn't* do. Weights, gradients and the
 * optimiser's moments live in GPU buffers for the engine's whole life. One `train()` call records
 * every kernel of K steps × R replicas into a single command encoder, and the only thing read back
 * is K × R loss values. A naive design that read the weights back after every step would spend
 * most of its time waiting on the bus and lose to the CPU.
 *
 * Kernels: one generic batched, tiled GEMM (16×16 tiles in workgroup memory) with fused epilogues
 * (bias + activation for the forward pass, + L2 for weight gradients, × activation derivative for
 * the backward pass), plus small kernels for gathering a mini-batch, the output-layer error, bias
 * gradients, the loss, the optimiser update and a step counter. Parity with the CPU tier is checked
 * in a real browser by `scripts/check-webgpu.cjs`.
 */
import { paramLayout, type ActName, type MlpEngine, type MlpSpec, type TrainCall, type Weights } from './types';

// ------------------------------------------------------------------ device

export type GpuInfo = { device: GPUDevice; name: string };
let devicePromise: Promise<GpuInfo | null> | null = null;

/** The shared WebGPU device, or null when the browser has no usable adapter. */
export function getGpu(): Promise<GpuInfo | null> {
  if (devicePromise) return devicePromise;
  devicePromise = (async () => {
    try {
      const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
      if (!gpu) return null;
      const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return null;
      const device = await adapter.requestDevice();
      device.lost.then(() => {
        devicePromise = null;
      });
      const info = adapter.info;
      const name = [info?.vendor, info?.architecture || info?.device].filter(Boolean).join(' ') || 'GPU';
      return { device, name };
    } catch {
      return null;
    }
  })();
  return devicePromise;
}

// ------------------------------------------------------------------ shaders

const ACT_ID: Record<ActName, number> = { linear: 0, relu: 1, tanh: 2, sigmoid: 3, gelu: 4, square: 5 };

const ACT_WGSL = /* wgsl */ `
fn actf(id: u32, z: f32) -> f32 {
  switch id {
    case 1u: { return max(z, 0.0); }
    case 2u: { return tanh(clamp(z, -30.0, 30.0)); }
    case 3u: { return 1.0 / (1.0 + exp(-clamp(z, -80.0, 80.0))); }
    case 4u: { let zz = clamp(z, -30.0, 30.0); return 0.5 * zz * (1.0 + tanh(0.7978845608 * (zz + 0.044715 * zz * zz * zz))); }
    case 5u: { return z * z; }
    default: { return z; }
  }
}
fn actg(id: u32, z: f32, a: f32) -> f32 {
  switch id {
    case 1u: { return select(0.0, 1.0, z > 0.0); }
    case 2u: { return 1.0 - a * a; }
    case 3u: { return a * (1.0 - a); }
    case 4u: {
      let zz = clamp(z, -30.0, 30.0);
      let t = tanh(0.7978845608 * (zz + 0.044715 * zz * zz * zz));
      return 0.5 * (1.0 + t) + 0.5 * zz * (1.0 - t * t) * 0.7978845608 * (1.0 + 0.134145 * zz * zz);
    }
    case 5u: { return 2.0 * z; }
    default: { return 1.0; }
  }
}`;

/** Modes of the GEMM epilogue. */
export const MODE = { plain: 0, forward: 1, weightGrad: 2, inputGrad: 3 } as const;

export const GEMM_WGSL = /* wgsl */ `
struct U {
  M: u32, N: u32, K: u32, transA: u32,
  transB: u32, lda: u32, ldb: u32, ldc: u32,
  strideA: u32, strideB: u32, strideC: u32, offA: u32,
  offB: u32, offC: u32, mode: u32, act: u32,
  auxOff: u32, auxStride: u32, l2: f32, pad: u32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> X1: array<f32>;
@group(0) @binding(5) var<storage, read_write> X2: array<f32>;
${ACT_WGSL}
var<workgroup> tA: array<array<f32, 16>, 16>;
var<workgroup> tB: array<array<f32, 16>, 16>;
fn getA(r: u32, row: u32, col: u32) -> f32 {
  if (row >= u.M || col >= u.K) { return 0.0; }
  let base = u.offA + r * u.strideA;
  if (u.transA == 1u) { return A[base + col * u.lda + row]; }
  return A[base + row * u.lda + col];
}
fn getB(r: u32, row: u32, col: u32) -> f32 {
  if (row >= u.K || col >= u.N) { return 0.0; }
  let base = u.offB + r * u.strideB;
  if (u.transB == 1u) { return B[base + col * u.ldb + row]; }
  return B[base + row * u.ldb + col];
}
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let r = wg.z;
  let row = wg.y * 16u + l.y;
  let col = wg.x * 16u + l.x;
  var acc = 0.0;
  let tiles = (u.K + 15u) / 16u;
  for (var t = 0u; t < tiles; t++) {
    tA[l.y][l.x] = getA(r, row, t * 16u + l.x);
    tB[l.y][l.x] = getB(r, t * 16u + l.y, col);
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) { acc += tA[l.y][k] * tB[k][l.x]; }
    workgroupBarrier();
  }
  if (row >= u.M || col >= u.N) { return; }
  let ci = u.offC + r * u.strideC + row * u.ldc + col;
  switch u.mode {
    case 1u: {
      let z = acc + X1[u.auxOff + r * u.auxStride + col];
      C[ci] = z;
      X2[ci] = actf(u.act, z);
    }
    case 2u: { C[ci] = acc + u.l2 * X1[u.auxOff + r * u.auxStride + row * u.ldc + col]; }
    case 3u: { C[ci] = acc * actg(u.act, X1[ci], X2[ci]); }
    default: { C[ci] = acc; }
  }
}`;

const MISC_STRUCT = /* wgsl */ `
struct MU {
  R: u32, B: u32, P: u32, din: u32,
  dout: u32, lossKind: u32, useIdx: u32, width: u32,
  off: u32, wTotal: u32, optKind: u32, pad: u32,
  l2: f32, b1: f32, b2: f32, eps: f32,
  wd: f32, pad1: f32, pad2: f32, pad3: f32,
};`;

const GATHER_WGSL = /* wgsl */ `${MISC_STRUCT}
@group(0) @binding(0) var<uniform> u: MU;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Y: array<f32>;
@group(0) @binding(3) var<storage, read> idx: array<u32>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
@group(0) @binding(5) var<storage, read_write> A0: array<f32>;
@group(0) @binding(6) var<storage, read_write> Yb: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; let r = g.y;
  if (i >= u.B) { return; }
  var src = i;
  if (u.useIdx == 1u) { src = idx[(st[0] * u.R + r) * u.B + i]; }
  let dst = r * u.B + i;
  for (var c = 0u; c < u.din; c++) { A0[dst * u.din + c] = X[src * u.din + c]; }
  for (var c = 0u; c < u.dout; c++) { Yb[dst * u.dout + c] = Y[src * u.dout + c]; }
}`;

/** Output-layer error δ = dLoss/dZ (already ÷ B) and each row's loss. */
const DELTA_WGSL = /* wgsl */ `${MISC_STRUCT}
@group(0) @binding(0) var<uniform> u: MU;
@group(0) @binding(1) var<storage, read> Z: array<f32>;
@group(0) @binding(2) var<storage, read> Aout: array<f32>;
@group(0) @binding(3) var<storage, read> Yb: array<f32>;
@group(0) @binding(4) var<storage, read_write> D: array<f32>;
@group(0) @binding(5) var<storage, read_write> lossRow: array<f32>;
${ACT_WGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; let r = g.y;
  if (i >= u.B) { return; }
  let base = (r * u.B + i) * u.dout;
  let invB = 1.0 / f32(u.B);
  var loss = 0.0;
  if (u.lossKind == 1u) {
    var mx = -3.4e38;
    for (var j = 0u; j < u.dout; j++) { mx = max(mx, Z[base + j]); }
    var s = 0.0;
    for (var j = 0u; j < u.dout; j++) { s += exp(Z[base + j] - mx); }
    for (var j = 0u; j < u.dout; j++) {
      let p = exp(Z[base + j] - mx) / s;
      let y = Yb[base + j];
      if (y > 0.0) { loss -= y * log(max(p, 1e-30)); }
      D[base + j] = (p - y) * invB;
    }
  } else {
    for (var j = 0u; j < u.dout; j++) {
      let o = Aout[base + j];
      let y = Yb[base + j];
      if (u.lossKind == 0u) {
        let q = clamp(o, 1e-7, 1.0 - 1e-7);
        loss -= y * log(q) + (1.0 - y) * log(1.0 - q);
        D[base + j] = (o - y) * invB;
      } else {
        let d = o - y;
        loss += 0.5 * d * d;
        D[base + j] = d * actg(u.width, Z[base + j], o) * invB;
      }
    }
  }
  lossRow[r * u.B + i] = loss;
}`;

const BIASGRAD_WGSL = /* wgsl */ `${MISC_STRUCT}
@group(0) @binding(0) var<uniform> u: MU;
@group(0) @binding(1) var<storage, read> D: array<f32>;
@group(0) @binding(2) var<storage, read_write> G: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let j = g.x; let r = g.y;
  if (j >= u.width) { return; }
  var s = 0.0;
  for (var i = 0u; i < u.B; i++) { s += D[(r * u.B + i) * u.width + j]; }
  G[r * u.P + u.off + j] = s;
}`;

const LOSS_WGSL = /* wgsl */ `${MISC_STRUCT}
@group(0) @binding(0) var<uniform> u: MU;
@group(0) @binding(1) var<storage, read> lossRow: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<f32>;
@group(0) @binding(3) var<storage, read> st: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let r = g.x;
  if (r >= u.R) { return; }
  var s = 0.0;
  for (var i = 0u; i < u.B; i++) { s += lossRow[r * u.B + i]; }
  s = s / f32(u.B);
  if (u.l2 > 0.0) {
    var w2 = 0.0;
    for (var k = 0u; k < u.wTotal; k++) { let w = P[r * u.P + k]; w2 += w * w; }
    s += 0.5 * u.l2 * w2;
  }
  out[st[0] * u.R + r] = s;
}`;

const OPT_WGSL = /* wgsl */ `${MISC_STRUCT}
@group(0) @binding(0) var<uniform> u: MU;
@group(0) @binding(1) var<storage, read_write> P: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> Mm: array<f32>;
@group(0) @binding(4) var<storage, read_write> Vv: array<f32>;
@group(0) @binding(5) var<storage, read> lr: array<f32>;
@group(0) @binding(6) var<storage, read> st: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x + g.y * 65535u * 256u;
  if (k >= u.R * u.P) { return; }
  let r = k / u.P;
  let rate = lr[st[0] * u.R + r];
  let gj = G[k];
  if (u.optKind == 0u) { P[k] -= rate * gj; return; }
  if (u.optKind == 1u) { Mm[k] = u.b1 * Mm[k] + gj; P[k] -= rate * Mm[k]; return; }
  let t = f32(st[1u + r] + 1u);
  Mm[k] = u.b1 * Mm[k] + (1.0 - u.b1) * gj;
  Vv[k] = u.b2 * Vv[k] + (1.0 - u.b2) * gj * gj;
  if (u.optKind == 3u && u.wd > 0.0) { P[k] -= rate * u.wd * P[k]; }
  let mh = Mm[k] / (1.0 - pow(u.b1, t));
  let vh = Vv[k] / (1.0 - pow(u.b2, t));
  P[k] -= rate * mh / (sqrt(vh) + u.eps);
}`;

const TICK_WGSL = /* wgsl */ `${MISC_STRUCT}
@group(0) @binding(0) var<uniform> u: MU;
@group(0) @binding(1) var<storage, read_write> st: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let r = g.x;
  if (r < u.R) { st[1u + r] += 1u; }
  if (r == 0u) { st[0] += 1u; }
}`;

const SOFTMAX_WGSL = /* wgsl */ `${MISC_STRUCT}
@group(0) @binding(0) var<uniform> u: MU;
@group(0) @binding(1) var<storage, read> Z: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; let r = g.y;
  if (i >= u.B) { return; }
  let base = (r * u.B + i) * u.dout;
  var mx = -3.4e38;
  for (var j = 0u; j < u.dout; j++) { mx = max(mx, Z[base + j]); }
  var s = 0.0;
  for (var j = 0u; j < u.dout; j++) { s += exp(Z[base + j] - mx); }
  for (var j = 0u; j < u.dout; j++) { O[base + j] = exp(Z[base + j] - mx) / s; }
}`;

// ------------------------------------------------------------------ helpers

const pipelineCache = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();
export function pipeline(device: GPUDevice, code: string): GPUComputePipeline {
  let m = pipelineCache.get(device);
  if (!m) pipelineCache.set(device, (m = new Map()));
  let p = m.get(code);
  if (!p) {
    p = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
    m.set(code, p);
  }
  return p;
}

const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
export function buffer(device: GPUDevice, floats: number, usage = S) {
  return device.createBuffer({ size: Math.max(16, Math.ceil((floats * 4) / 16) * 16), usage });
}
export function uniform(device: GPUDevice, data: ArrayBuffer) {
  const b = device.createBuffer({ size: Math.ceil(data.byteLength / 16) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(b, 0, data);
  return b;
}
export function bind(device: GPUDevice, p: GPUComputePipeline, buffers: GPUBuffer[]) {
  return device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
}

export type GemmU = {
  M: number; N: number; K: number; transA?: number; transB?: number; lda: number; ldb: number; ldc: number;
  strideA: number; strideB: number; strideC: number; offA?: number; offB?: number; offC?: number;
  mode: number; act?: number; auxOff?: number; auxStride?: number; l2?: number;
};
export function gemmUniform(u: GemmU) {
  const buf = new ArrayBuffer(80);
  const a = new Uint32Array(buf);
  const vals = [u.M, u.N, u.K, u.transA ?? 0, u.transB ?? 0, u.lda, u.ldb, u.ldc, u.strideA, u.strideB, u.strideC, u.offA ?? 0, u.offB ?? 0, u.offC ?? 0, u.mode, u.act ?? 0, u.auxOff ?? 0, u.auxStride ?? 0];
  vals.forEach((v, i) => (a[i] = v));
  new Float32Array(buf)[18] = u.l2 ?? 0;
  return buf;
}
type MiscU = Partial<Record<'R' | 'B' | 'P' | 'din' | 'dout' | 'lossKind' | 'useIdx' | 'width' | 'off' | 'wTotal' | 'optKind', number>> &
  Partial<Record<'l2' | 'b1' | 'b2' | 'eps' | 'wd', number>>;
function miscUniform(u: MiscU) {
  const buf = new ArrayBuffer(80);
  const a = new Uint32Array(buf);
  const f = new Float32Array(buf);
  (['R', 'B', 'P', 'din', 'dout', 'lossKind', 'useIdx', 'width', 'off', 'wTotal', 'optKind'] as const).forEach((k, i) => (a[i] = u[k] ?? 0));
  (['l2', 'b1', 'b2', 'eps', 'wd'] as const).forEach((k, i) => (f[12 + i] = u[k] ?? 0));
  return buf;
}

export type Op = { p: GPUComputePipeline; g: GPUBindGroup; x: number; y: number; z: number };

export async function readBack(device: GPUDevice, src: GPUBuffer, floats: number, encoder?: GPUCommandEncoder): Promise<Float32Array> {
  const staging = device.createBuffer({ size: Math.max(16, Math.ceil((floats * 4) / 16) * 16), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = encoder ?? device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, staging.size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0, floats * 4));
  staging.unmap();
  staging.destroy();
  return out;
}

// ------------------------------------------------------------------ the MLP engine

/** Per-batch-size buffers and pre-built dispatch lists. */
type Workspace = {
  B: number;
  A: GPUBuffer[];
  Z: GPUBuffer[];
  D: GPUBuffer[];
  Yb: GPUBuffer;
  lossRow: GPUBuffer;
  soft?: GPUBuffer;
  forward: Op[];
  trainStep: Op[];
  /** The gather op's uniform buffer (useIdx changes between calls). */
  gatherU?: GPUBuffer;
};

export class GpuMlpEngine implements MlpEngine {
  readonly kind = 'webgpu' as const;
  readonly replicas: number;
  private P: number;
  private layout: ReturnType<typeof paramLayout>;
  private params: GPUBuffer;
  private grads: GPUBuffer;
  private m: GPUBuffer;
  private v: GPUBuffer;
  private state: GPUBuffer;
  private X: GPUBuffer | null = null;
  private Y: GPUBuffer | null = null;
  private idx: GPUBuffer | null = null;
  private lr: GPUBuffer | null = null;
  private lossOut: GPUBuffer | null = null;
  private cap = 0;
  private workspaces = new Map<string, Workspace>();
  private destroyed = false;
  private chain: Promise<unknown> = Promise.resolve();

  readonly spec: MlpSpec;
  private device: GPUDevice;
  constructor(spec: MlpSpec, device: GPUDevice) {
    this.spec = spec;
    this.device = device;
    this.replicas = spec.replicas ?? 1;
    this.layout = paramLayout(spec.sizes);
    this.P = this.layout.total;
    const RP = this.replicas * this.P;
    this.params = buffer(device, RP);
    this.grads = buffer(device, RP);
    this.m = buffer(device, RP);
    this.v = buffer(device, RP);
    this.state = buffer(device, 1 + this.replicas);
    if (spec.loss === 'softmax-ce' && spec.acts.at(-1) !== 'linear') throw new Error('softmax-ce needs a linear output layer');
    if (spec.loss === 'bce' && spec.acts.at(-1) !== 'sigmoid') throw new Error('bce needs a sigmoid output layer');
  }

  /** Serialise async work so calls from a widget never interleave on the shared buffers. */
  private run<T>(f: () => Promise<T>): Promise<T> {
    const p = this.chain.then(f, f);
    this.chain = p.catch(() => undefined);
    return p;
  }

  setWeights(r: number, w: Weights) {
    const flat = new Float32Array(this.P);
    w.W.forEach((Wl, l) => flat.set(Wl, this.layout.wOff[l]));
    w.b.forEach((bl, l) => flat.set(bl, this.layout.bOff[l]));
    const q = this.device.queue;
    q.writeBuffer(this.params, r * this.P * 4, flat);
    const zero = new Float32Array(this.P);
    q.writeBuffer(this.m, r * this.P * 4, zero);
    q.writeBuffer(this.v, r * this.P * 4, zero);
    q.writeBuffer(this.state, (1 + r) * 4, new Uint32Array([0]));
  }

  setData(X: Float32Array, Y: Float32Array, n: number) {
    this.X?.destroy();
    this.Y?.destroy();
    this.X = buffer(this.device, X.length);
    this.Y = buffer(this.device, Y.length);
    this.device.queue.writeBuffer(this.X, 0, X);
    this.device.queue.writeBuffer(this.Y, 0, Y);
    void n;
    for (const w of this.workspaces.values()) for (const b of [...w.A, ...w.Z, ...w.D, w.Yb, w.lossRow, w.soft]) b?.destroy();
    this.workspaces.clear();
  }

  private workspace(B: number, broadcastInput: boolean): Workspace {
    const key = `${B}/${broadcastInput}`;
    const hit = this.workspaces.get(key);
    if (hit) return hit;
    const d = this.device;
    const { sizes, acts } = this.spec;
    const L = sizes.length - 1;
    const R = this.replicas;
    const P = this.P;
    const { wOff, bOff } = this.layout;
    const A = sizes.map((s, l) => buffer(d, (l === 0 && broadcastInput ? 1 : R) * B * s));
    const Z = sizes.slice(1).map((s) => buffer(d, R * B * s));
    const D = sizes.slice(1).map((s) => buffer(d, R * B * s));
    const Yb = buffer(d, R * B * sizes[L]);
    const lossRow = buffer(d, R * B);
    const dummy = buffer(d, 4);
    const gemm = pipeline(d, GEMM_WGSL);
    const tile = (n: number) => Math.ceil(n / 16);
    const lossKind = this.spec.loss === 'bce' ? 0 : this.spec.loss === 'softmax-ce' ? 1 : 2;
    const outAct = ACT_ID[acts[L - 1]];

    const forward: Op[] = [];
    for (let l = 0; l < L; l++) {
      const u = gemmUniform({
        M: B, N: sizes[l + 1], K: sizes[l], transB: 1, lda: sizes[l], ldb: sizes[l], ldc: sizes[l + 1],
        strideA: l === 0 && broadcastInput ? 0 : B * sizes[l], strideB: P, strideC: B * sizes[l + 1], offB: wOff[l],
        mode: MODE.forward, act: ACT_ID[acts[l]], auxOff: bOff[l], auxStride: P,
      });
      forward.push({ p: gemm, g: bind(d, gemm, [uniform(d, u), A[l], this.params, Z[l], this.params, A[l + 1]]), x: tile(sizes[l + 1]), y: tile(B), z: R });
    }
    let soft: GPUBuffer | undefined;
    if (lossKind === 1) {
      soft = buffer(d, R * B * sizes[L]);
      const sp = pipeline(d, SOFTMAX_WGSL);
      forward.push({ p: sp, g: bind(d, sp, [uniform(d, miscUniform({ R, B, dout: sizes[L] })), Z[L - 1], soft]), x: Math.ceil(B / 64), y: R, z: 1 });
    }

    const trainStep: Op[] = [];
    let gatherU: GPUBuffer | undefined;
    if (!broadcastInput && this.X && this.Y) {
      const gp = pipeline(d, GATHER_WGSL);
      gatherU = uniform(d, miscUniform({ R, B, din: sizes[0], dout: sizes[L], useIdx: 1 }));
      if (!this.idx) this.ensureCapacity(1, B);
      trainStep.push({ p: gp, g: bind(d, gp, [gatherU, this.X, this.Y, this.idx!, this.state, A[0], Yb]), x: Math.ceil(B / 64), y: R, z: 1 });
      trainStep.push(...forward.slice(0, L));
      const dp = pipeline(d, DELTA_WGSL);
      trainStep.push({ p: dp, g: bind(d, dp, [uniform(d, miscUniform({ R, B, dout: sizes[L], lossKind, width: outAct })), Z[L - 1], A[L], Yb, D[L - 1], lossRow]), x: Math.ceil(B / 64), y: R, z: 1 });
      const bg = pipeline(d, BIASGRAD_WGSL);
      for (let l = L - 1; l >= 0; l--) {
        const dw = gemmUniform({
          M: sizes[l + 1], N: sizes[l], K: B, transA: 1, lda: sizes[l + 1], ldb: sizes[l], ldc: sizes[l],
          strideA: B * sizes[l + 1], strideB: B * sizes[l], strideC: P, offC: wOff[l], mode: MODE.weightGrad, auxOff: wOff[l], auxStride: P, l2: this.spec.l2 ?? 0,
        });
        trainStep.push({ p: gemm, g: bind(d, gemm, [uniform(d, dw), D[l], A[l], this.grads, this.params, dummy]), x: tile(sizes[l]), y: tile(sizes[l + 1]), z: R });
        trainStep.push({ p: bg, g: bind(d, bg, [uniform(d, miscUniform({ R, B, P, width: sizes[l + 1], off: bOff[l] })), D[l], this.grads]), x: Math.ceil(sizes[l + 1] / 64), y: R, z: 1 });
        if (l > 0) {
          const da = gemmUniform({
            M: B, N: sizes[l], K: sizes[l + 1], lda: sizes[l + 1], ldb: sizes[l], ldc: sizes[l],
            strideA: B * sizes[l + 1], strideB: P, strideC: B * sizes[l], offB: wOff[l], mode: MODE.inputGrad, act: ACT_ID[acts[l - 1]],
          });
          trainStep.push({ p: gemm, g: bind(d, gemm, [uniform(d, da), D[l], this.params, D[l - 1], Z[l - 1], A[l]]), x: tile(sizes[l]), y: tile(B), z: R });
        }
      }
      this.ensureLossOut(1);
    }
    const ws: Workspace = { B, A, Z, D, Yb, lossRow, soft, forward, trainStep, gatherU };
    this.workspaces.set(key, ws);
    return ws;
  }

  private ensureCapacity(steps: number, B: number) {
    const need = steps * this.replicas * B;
    if (this.idx && this.cap >= need) return false;
    // rebuild the training workspaces: they hold bind groups that reference the old idx buffer
    this.idx?.destroy();
    this.cap = Math.max(need, 1024);
    this.idx = buffer(this.device, this.cap);
    for (const [k, w] of this.workspaces) if (w.gatherU) this.workspaces.delete(k);
    return true;
  }
  private lossCap = 0;
  private ensureLossOut(steps: number) {
    const need = steps * this.replicas;
    if (this.lossOut && this.lossCap >= need && this.lr) return;
    this.lossOut?.destroy();
    this.lr?.destroy();
    this.lossCap = Math.max(need, 256);
    this.lossOut = buffer(this.device, this.lossCap);
    this.lr = buffer(this.device, this.lossCap);
  }

  train(call: TrainCall): Promise<Float32Array> {
    return this.run(async () => {
      if (!this.X) throw new Error('setData first');
      const d = this.device;
      const R = this.replicas;
      const { steps, batch: B } = call;
      this.ensureCapacity(steps, B);
      this.ensureLossOut(steps);
      const ws = this.workspace(B, false);
      d.queue.writeBuffer(ws.gatherU!, 24, new Uint32Array([call.idx ? 1 : 0]));
      if (call.idx) d.queue.writeBuffer(this.idx!, 0, call.idx);
      d.queue.writeBuffer(this.lr!, 0, call.lr);
      d.queue.writeBuffer(this.state, 0, new Uint32Array([0]));
      const ms: MiscU = {
        R, B, P: this.P, wTotal: this.layout.bOff[0], l2: this.spec.l2 ?? 0,
        optKind: { sgd: 0, momentum: 1, adam: 2, adamw: 3 }[this.spec.opt.kind],
        b1: this.spec.opt.beta1 ?? 0.9, b2: this.spec.opt.beta2 ?? 0.999, eps: this.spec.opt.eps ?? 1e-8, wd: this.spec.opt.weightDecay ?? 0,
      };
      const lp = pipeline(d, LOSS_WGSL);
      const op = pipeline(d, OPT_WGSL);
      const tp = pipeline(d, TICK_WGSL);
      const mu = uniform(d, miscUniform(ms));
      const lossG = bind(d, lp, [mu, ws.lossRow, this.params, this.state, this.lossOut!]);
      const optG = bind(d, op, [mu, this.params, this.grads, this.m, this.v, this.lr!, this.state]);
      const tickG = bind(d, tp, [mu, this.state]);
      const groups = Math.ceil((R * this.P) / 256);
      const tail: Op[] = [
        { p: lp, g: lossG, x: Math.ceil(R / 64), y: 1, z: 1 },
        { p: op, g: optG, x: Math.min(groups, 65535), y: Math.ceil(groups / 65535), z: 1 },
        { p: tp, g: tickG, x: Math.ceil((R + 1) / 64), y: 1, z: 1 },
      ];
      const enc = d.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let k = 0; k < steps; k++)
        for (const o of [...ws.trainStep, ...tail]) {
          pass.setPipeline(o.p);
          pass.setBindGroup(0, o.g);
          pass.dispatchWorkgroups(o.x, o.y, o.z);
        }
      pass.end();
      const out = await readBack(d, this.lossOut!, steps * R, enc);
      mu.destroy();
      return out;
    });
  }

  predict(X: Float32Array, n: number): Promise<Float32Array> {
    return this.run(async () => {
      const d = this.device;
      const { sizes } = this.spec;
      const L = sizes.length - 1;
      const R = this.replicas;
      const dout = sizes[L];
      const CH = 4096;
      const result = new Float32Array(R * n * dout);
      for (let s = 0; s < n; s += CH) {
        const rows = Math.min(CH, n - s);
        const ws = this.workspace(CH, true);
        const chunk = new Float32Array(CH * sizes[0]);
        chunk.set(X.subarray(s * sizes[0], (s + rows) * sizes[0]));
        d.queue.writeBuffer(ws.A[0], 0, chunk);
        const enc = d.createCommandEncoder();
        const pass = enc.beginComputePass();
        for (const o of ws.forward) {
          pass.setPipeline(o.p);
          pass.setBindGroup(0, o.g);
          pass.dispatchWorkgroups(o.x, o.y, o.z);
        }
        pass.end();
        const out = await readBack(d, ws.soft ?? ws.A[L], R * CH * dout, enc);
        for (let r = 0; r < R; r++) result.set(out.subarray(r * CH * dout, (r * CH + rows) * dout), (r * n + s) * dout);
      }
      return result;
    });
  }

  readWeights(): Promise<Weights[]> {
    return this.run(async () => {
      const all = await readBack(this.device, this.params, this.replicas * this.P);
      const { sizes } = this.spec;
      const { wOff, bOff } = this.layout;
      return Array.from({ length: this.replicas }, (_, r) => ({
        W: sizes.slice(1).map((o, l) => all.slice(r * this.P + wOff[l], r * this.P + wOff[l] + sizes[l] * o)),
        b: sizes.slice(1).map((o, l) => all.slice(r * this.P + bOff[l], r * this.P + bOff[l] + o)),
      }));
    });
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const b of [this.params, this.grads, this.m, this.v, this.state, this.X, this.Y, this.idx, this.lr, this.lossOut]) b?.destroy();
    for (const w of this.workspaces.values()) for (const b of [...w.A, ...w.Z, ...w.D, w.Yb, w.lossRow, w.soft]) b?.destroy();
  }
}

// ------------------------------------------------------------------ plain GEMM (benchmark)

/**
 * Time `reps` multiplications C = A·B (A: M×K, B: K×N) on the GPU and return the last result (for
 * checking) and the wall-clock time in ms, including read-back.
 * - `batched` (the design the engine uses): all reps recorded in one command encoder, one read-back.
 * - otherwise (the naive design): one submission and a full read-back of C per multiplication.
 */
export async function gpuGemm(device: GPUDevice, A: Float32Array, B: Float32Array, dims: { M: number; N: number; K: number }, reps: number, batched = true) {
  const { M, N, K } = dims;
  const a = buffer(device, M * K);
  const b = buffer(device, K * N);
  const c = buffer(device, M * N);
  device.queue.writeBuffer(a, 0, A);
  device.queue.writeBuffer(b, 0, B);
  const gemm = pipeline(device, GEMM_WGSL);
  // two distinct dummies: one buffer can't be bound read-only and read-write in the same dispatch
  const dummy = buffer(device, 4);
  const dummy2 = buffer(device, 4);
  const u = uniform(device, gemmUniform({ M, N, K, lda: K, ldb: N, ldc: N, strideA: 0, strideB: 0, strideC: 0, mode: MODE.plain }));
  const g = bind(device, gemm, [u, a, b, c, dummy, dummy2]);
  const record = (enc: GPUCommandEncoder, k: number) => {
    const pass = enc.beginComputePass();
    for (let i = 0; i < k; i++) {
      pass.setPipeline(gemm);
      pass.setBindGroup(0, g);
      pass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(M / 16), 1);
    }
    pass.end();
  };
  // warm-up: compile and run once so we time the work, not the shader compiler
  {
    const enc = device.createCommandEncoder();
    record(enc, 1);
    await readBack(device, c, 1, enc);
  }
  const t0 = performance.now();
  let out: Float32Array = new Float32Array(0);
  if (batched) {
    const enc = device.createCommandEncoder();
    record(enc, reps);
    out = await readBack(device, c, M * N, enc);
  } else
    for (let i = 0; i < reps; i++) {
      const enc = device.createCommandEncoder();
      record(enc, 1);
      out = await readBack(device, c, M * N, enc);
    }
  const ms = performance.now() - t0;
  for (const x of [a, b, c, dummy, dummy2, u]) x.destroy();
  return { out, ms };
}
