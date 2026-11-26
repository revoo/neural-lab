// authored-by: Claude Opus 5.5
/**
 * The tiny GPT on WebGPU (V1 capstone, step A2): the same model, maths and AdamW update as the CPU
 * reference in `gpt.ts`, as WGSL compute shaders.
 *
 * Like `GpuMlpEngine`, its speed comes from what it doesn't do: parameters, gradients and AdamW
 * moments live in GPU buffers for the engine's whole life, K training steps are recorded into one
 * command encoder, and only K loss values come back. Linear layers reuse the tiled GEMM kernel of
 * `webgpuBackend.ts` (with its fused bias + GELU epilogue); attention, layer norm, the loss, the
 * embedding gradients, the gradient norm and AdamW have their own kernels below.
 *
 * Numerics differ from the CPU tier in the last bits (float32 accumulation order, the GPU's own
 * `exp`), so parity is checked with a tolerance by `scripts/check-webgpu.cjs` (`gptParity`).
 */
import { gptLayout, type AdamWOpts, type GptConfig, type GptLayout } from './gpt';
import { bind, buffer, GEMM_WGSL, gemmUniform, MODE, pipeline, readBack, uniform, type GemmU, type Op } from './webgpuBackend';

const STRUCT = /* wgsl */ `
struct GU {
  N: u32, T: u32, B: u32, d: u32,
  H: u32, hd: u32, V: u32, ctx: u32,
  P: u32, o0: u32, o1: u32, flag: u32,
  width: u32, len: u32, pad0: u32, pad1: u32,
  scale: f32, b1: f32, b2: f32, eps: f32,
  wd: f32, clip: f32, pad2: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> u: GU;`;

/** x₀[n, j] = W_te[token, j] + W_pe[t, j], tokens read from this step's slice of the batch buffer. */
const EMBED = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> tok: array<u32>;
@group(0) @binding(3) var<storage, read> st: array<u32>;
@group(0) @binding(4) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + g.y * 65535u * 256u;
  if (i >= u.N * u.d) { return; }
  let n = i / u.d; let j = i % u.d;
  let b = n / u.T; let t = n % u.T;
  let k = tok[st[0] * u.B * (u.T + 1u) + b * (u.T + 1u) + t];
  X[i] = P[u.o0 + k * u.d + j] + P[u.o1 + t * u.d + j];
}`;

/** Layer norm, one thread per row: Y = γ ⊙ x̂ + β, keeping x̂ and 1/σ. o0 = γ offset, o1 = β offset. */
const LN_FWD = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> Xh: array<f32>;
@group(0) @binding(5) var<storage, read_write> Rs: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let n = g.x;
  if (n >= u.N) { return; }
  let o = n * u.d;
  var mean = 0.0;
  for (var j = 0u; j < u.d; j++) { mean += X[o + j]; }
  mean /= f32(u.d);
  var v = 0.0;
  for (var j = 0u; j < u.d; j++) { let c = X[o + j] - mean; v += c * c; }
  v /= f32(u.d);
  let r = 1.0 / sqrt(v + 1e-5);
  Rs[n] = r;
  for (var j = 0u; j < u.d; j++) {
    let h = (X[o + j] - mean) * r;
    Xh[o + j] = h;
    Y[o + j] = P[u.o0 + j] * h + P[u.o1 + j];
  }
}`;

/** Out = A + B, elementwise over len floats. */
const ADD = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + g.y * 65535u * 256u;
  if (i >= u.len) { return; }
  O[i] = A[i] + B[i];
}`;

/** Causal attention, one thread per (sequence, head, query position). */
const ATT_FWD = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read> QKV: array<f32>;
@group(0) @binding(2) var<storage, read_write> Att: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let id = g.x;
  if (id >= u.B * u.H * u.T) { return; }
  let i = id % u.T; let bh = id / u.T; let h = bh % u.H; let b = bh / u.H;
  let d3 = 3u * u.d;
  let qo = (b * u.T + i) * d3 + h * u.hd;
  let row = bh * u.T * u.T + i * u.T;
  var mx = -3.4e38;
  for (var j = 0u; j <= i; j++) {
    let ko = (b * u.T + j) * d3 + u.d + h * u.hd;
    var s = 0.0;
    for (var e = 0u; e < u.hd; e++) { s += QKV[qo + e] * QKV[ko + e]; }
    s *= u.scale;
    Att[row + j] = s;
    mx = max(mx, s);
  }
  var sum = 0.0;
  for (var j = 0u; j <= i; j++) { let e = exp(Att[row + j] - mx); Att[row + j] = e; sum += e; }
  for (var j = 0u; j <= i; j++) { Att[row + j] /= sum; }
  for (var j = i + 1u; j < u.T; j++) { Att[row + j] = 0.0; }
  let yo = (b * u.T + i) * u.d + h * u.hd;
  for (var e = 0u; e < u.hd; e++) {
    var acc = 0.0;
    for (var j = 0u; j <= i; j++) { acc += Att[row + j] * QKV[(b * u.T + j) * d3 + 2u * u.d + h * u.hd + e]; }
    Y[yo + e] = acc;
  }
}`;

/** Softmax cross-entropy per row: loss row and dlogits = (p − onehot)/N. */
const CE = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read> L: array<f32>;
@group(0) @binding(2) var<storage, read> tok: array<u32>;
@group(0) @binding(3) var<storage, read> st: array<u32>;
@group(0) @binding(4) var<storage, read_write> D: array<f32>;
@group(0) @binding(5) var<storage, read_write> lossRow: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let n = g.x;
  if (n >= u.N) { return; }
  let b = n / u.T; let t = n % u.T;
  let y = tok[st[0] * u.B * (u.T + 1u) + b * (u.T + 1u) + t + 1u];
  let o = n * u.V;
  var mx = -3.4e38;
  for (var k = 0u; k < u.V; k++) { mx = max(mx, L[o + k]); }
  var s = 0.0;
  for (var k = 0u; k < u.V; k++) { s += exp(L[o + k] - mx); }
  let inv = 1.0 / f32(u.N);
  for (var k = 0u; k < u.V; k++) {
    let p = exp(L[o + k] - mx) / s;
    D[o + k] = (p - select(0.0, 1.0, k == y)) * inv;
  }
  lossRow[n] = -(L[o + y] - mx - log(s));
}`;

const LOSS_SUM = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read> lossRow: array<f32>;
@group(0) @binding(2) var<storage, read> st: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3u) {
  var s = 0.0;
  for (var n = l.x; n < u.N; n += 256u) { s += lossRow[n]; }
  part[l.x] = s;
  workgroupBarrier();
  if (l.x == 0u) {
    var tot = 0.0;
    for (var k = 0u; k < 256u; k++) { tot += part[k]; }
    out[st[0]] = tot / f32(u.N);
  }
}`;

/** Layer-norm backward for the input, one thread per row; flag = 1 adds into dX, 0 overwrites. */
const LN_BWD_X = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> dY: array<f32>;
@group(0) @binding(3) var<storage, read> Xh: array<f32>;
@group(0) @binding(4) var<storage, read> Rs: array<f32>;
@group(0) @binding(5) var<storage, read_write> dX: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let n = g.x;
  if (n >= u.N) { return; }
  let o = n * u.d;
  var s1 = 0.0; var s2 = 0.0;
  for (var j = 0u; j < u.d; j++) { let gh = dY[o + j] * P[u.o0 + j]; s1 += gh; s2 += gh * Xh[o + j]; }
  let r = Rs[n] / f32(u.d);
  for (var j = 0u; j < u.d; j++) {
    let v = r * (f32(u.d) * dY[o + j] * P[u.o0 + j] - s1 - Xh[o + j] * s2);
    if (u.flag == 1u) { dX[o + j] += v; } else { dX[o + j] = v; }
  }
}`;

/** Layer-norm backward for γ (at o0) and β (at o1), one thread per feature. */
const LN_BWD_P = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read> Xh: array<f32>;
@group(0) @binding(3) var<storage, read_write> G: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let j = g.x;
  if (j >= u.d) { return; }
  var sg = 0.0; var sb = 0.0;
  for (var n = 0u; n < u.N; n++) { let v = dY[n * u.d + j]; sg += v * Xh[n * u.d + j]; sb += v; }
  G[u.o0 + j] = sg;
  G[u.o1 + j] = sb;
}`;

/** Bias gradient: column sums of D [N × width] into G at o0. */
const BIAS = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read> D: array<f32>;
@group(0) @binding(2) var<storage, read_write> G: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let j = g.x;
  if (j >= u.width) { return; }
  var s = 0.0;
  for (var n = 0u; n < u.N; n++) { s += D[n * u.width + j]; }
  G[u.o0 + j] = s;
}`;

/** Attention backward, part 1 (one thread per query): dS_ij = P_ij (dP_ij − Σ P dP)·scale, and dq_i. */
const ATT_BWD_Q = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read> QKV: array<f32>;
@group(0) @binding(2) var<storage, read> Att: array<f32>;
@group(0) @binding(3) var<storage, read> dY: array<f32>;
@group(0) @binding(4) var<storage, read_write> dS: array<f32>;
@group(0) @binding(5) var<storage, read_write> dQKV: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let id = g.x;
  if (id >= u.B * u.H * u.T) { return; }
  let i = id % u.T; let bh = id / u.T; let h = bh % u.H; let b = bh / u.H;
  let d3 = 3u * u.d;
  let row = bh * u.T * u.T + i * u.T;
  let yo = (b * u.T + i) * u.d + h * u.hd;
  var dot = 0.0;
  for (var j = 0u; j <= i; j++) {
    let vo = (b * u.T + j) * d3 + 2u * u.d + h * u.hd;
    var s = 0.0;
    for (var e = 0u; e < u.hd; e++) { s += dY[yo + e] * QKV[vo + e]; }
    dS[row + j] = s;
    dot += s * Att[row + j];
  }
  for (var j = 0u; j <= i; j++) { dS[row + j] = Att[row + j] * (dS[row + j] - dot) * u.scale; }
  for (var j = i + 1u; j < u.T; j++) { dS[row + j] = 0.0; }
  let qo = (b * u.T + i) * d3 + h * u.hd;
  for (var e = 0u; e < u.hd; e++) {
    var acc = 0.0;
    for (var j = 0u; j <= i; j++) { acc += dS[row + j] * QKV[(b * u.T + j) * d3 + u.d + h * u.hd + e]; }
    dQKV[qo + e] = acc;
  }
}`;

/** Attention backward, part 2 (one thread per key position j): dk_j = Σᵢ dS_ij q_i, dv_j = Σᵢ P_ij dy_i. */
const ATT_BWD_KV = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read> QKV: array<f32>;
@group(0) @binding(2) var<storage, read> Att: array<f32>;
@group(0) @binding(3) var<storage, read> dY: array<f32>;
@group(0) @binding(4) var<storage, read> dS: array<f32>;
@group(0) @binding(5) var<storage, read_write> dQKV: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let id = g.x;
  if (id >= u.B * u.H * u.T) { return; }
  let j = id % u.T; let bh = id / u.T; let h = bh % u.H; let b = bh / u.H;
  let d3 = 3u * u.d;
  let base = bh * u.T * u.T;
  let ko = (b * u.T + j) * d3 + u.d + h * u.hd;
  for (var e = 0u; e < u.hd; e++) {
    var dk = 0.0; var dv = 0.0;
    for (var i = j; i < u.T; i++) {
      dk += dS[base + i * u.T + j] * QKV[(b * u.T + i) * d3 + h * u.hd + e];
      dv += Att[base + i * u.T + j] * dY[(b * u.T + i) * u.d + h * u.hd + e];
    }
    dQKV[ko + e] = dk;
    dQKV[ko + u.d + e] = dv;
  }
}`;

/** Token-embedding gradient from the input side, added to the output-layer part already in G. */
const EMB_TOK = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read> dX: array<f32>;
@group(0) @binding(2) var<storage, read> tok: array<u32>;
@group(0) @binding(3) var<storage, read> st: array<u32>;
@group(0) @binding(4) var<storage, read_write> G: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + g.y * 65535u * 64u;
  if (i >= u.V * u.d) { return; }
  let v = i / u.d; let j = i % u.d;
  var s = 0.0;
  let base = st[0] * u.B * (u.T + 1u);
  for (var n = 0u; n < u.N; n++) {
    let b = n / u.T; let t = n % u.T;
    if (tok[base + b * (u.T + 1u) + t] == v) { s += dX[n * u.d + j]; }
  }
  G[u.o0 + i] += s;
}`;

/** Position-embedding gradient (rows past the current sequence length get 0). */
const EMB_POS = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read> dX: array<f32>;
@group(0) @binding(2) var<storage, read_write> G: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= u.ctx * u.d) { return; }
  let t = i / u.d; let j = i % u.d;
  var s = 0.0;
  if (t < u.T) { for (var b = 0u; b < u.B; b++) { s += dX[(b * u.T + t) * u.d + j]; } }
  G[u.o1 + i] = s;
}`;

/** Sum of squared gradients: one partial sum per workgroup of 256. */
const NORM_PART = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) w: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let chunk = (u.P + u.len - 1u) / u.len;
  let start = w.x * chunk;
  var s = 0.0;
  for (var k = start + l.x; k < min(start + chunk, u.P); k += 256u) { let v = G[k]; s += v * v; }
  sh[l.x] = s;
  workgroupBarrier();
  if (l.x == 0u) {
    var t = 0.0;
    for (var k = 0u; k < 256u; k++) { t += sh[k]; }
    part[w.x] = t;
  }
}`;
const NORM_FINAL = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read> part: array<f32>;
@group(0) @binding(2) var<storage, read_write> fst: array<f32>;
@compute @workgroup_size(1)
fn main() {
  var s = 0.0;
  for (var k = 0u; k < u.len; k++) { s += part[k]; }
  fst[0] = sqrt(s);
}`;

/** AdamW in PyTorch's form, with clipping by the global norm in fst[0]; mask = 1 for matrices (decayed). */
const ADAMW = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read_write> Pm: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> M: array<f32>;
@group(0) @binding(4) var<storage, read_write> Vv: array<f32>;
@group(0) @binding(5) var<storage, read> lr: array<f32>;
@group(0) @binding(6) var<storage, read> st: array<u32>;
@group(0) @binding(7) var<storage, read> fst: array<f32>;
@group(0) @binding(8) var<storage, read> mask: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x + g.y * 65535u * 256u;
  if (k >= u.P) { return; }
  let rate = lr[st[0]];
  let norm = fst[0];
  var c = 1.0;
  if (u.clip > 0.0 && norm > u.clip) { c = u.clip / (norm + 1e-6); }
  let gk = G[k] * c;
  let t = f32(st[1] + 1u);
  M[k] = u.b1 * M[k] + (1.0 - u.b1) * gk;
  Vv[k] = u.b2 * Vv[k] + (1.0 - u.b2) * gk * gk;
  var p = Pm[k];
  if (mask[k] == 1u) { p *= 1.0 - rate * u.wd; }
  let stepSize = rate / (1.0 - pow(u.b1, t));
  let c2 = sqrt(1.0 - pow(u.b2, t));
  Pm[k] = p - stepSize * M[k] / (sqrt(Vv[k]) / c2 + u.eps);
}`;

const TICK = /* wgsl */ `${STRUCT}
@group(0) @binding(1) var<storage, read_write> st: array<u32>;
@compute @workgroup_size(1)
fn main() {
  if (u.flag == 1u) { st[1] += 1u; }
  st[0] += 1u;
}`;

type GU = Partial<Record<'N' | 'T' | 'B' | 'd' | 'H' | 'hd' | 'V' | 'ctx' | 'P' | 'o0' | 'o1' | 'flag' | 'width' | 'len', number>> &
  Partial<Record<'scale' | 'b1' | 'b2' | 'eps' | 'wd' | 'clip', number>>;
const U_KEYS = ['N', 'T', 'B', 'd', 'H', 'hd', 'V', 'ctx', 'P', 'o0', 'o1', 'flag', 'width', 'len'] as const;
const F_KEYS = ['scale', 'b1', 'b2', 'eps', 'wd', 'clip'] as const;
function gu(v: GU): ArrayBuffer {
  const buf = new ArrayBuffer(96);
  const a = new Uint32Array(buf);
  const f = new Float32Array(buf);
  U_KEYS.forEach((k, i) => (a[i] = v[k] ?? 0));
  F_KEYS.forEach((k, i) => (f[16 + i] = v[k] ?? 0));
  return buf;
}

type LayerBufs = {
  xin: GPUBuffer; xh1: GPUBuffer; rs1: GPUBuffer; a1: GPUBuffer; qkv: GPUBuffer; att: GPUBuffer; y: GPUBuffer;
  xmid: GPUBuffer; xh2: GPUBuffer; rs2: GPUBuffer; a2: GPUBuffer; hpre: GPUBuffer; hact: GPUBuffer;
};
type Workspace = { B: number; T: number; forward: Op[]; backward: Op[]; owned: GPUBuffer[] };

const PARTS = 64;

export class GptGpu {
  readonly kind = 'webgpu' as const;
  readonly cfg: GptConfig;
  readonly layout: GptLayout;
  private device: GPUDevice;
  private P: number;
  private params: GPUBuffer;
  private grads: GPUBuffer;
  private m: GPUBuffer;
  private v: GPUBuffer;
  private mask: GPUBuffer;
  /** st[0]: step index within this submission; st[1]: AdamW step count (persists). */
  private st: GPUBuffer;
  /** fst[0]: the global gradient norm of the current step. */
  private fst: GPUBuffer;
  private part: GPUBuffer;
  private tok: GPUBuffer | null = null;
  private tokCap = 0;
  private lr: GPUBuffer | null = null;
  private lossOut: GPUBuffer | null = null;
  private stepCap = 0;
  private workspaces = new Map<string, Workspace>();
  private chain: Promise<unknown> = Promise.resolve();
  private destroyed = false;
  private dummyA: GPUBuffer;
  private dummyB: GPUBuffer;
  private uniforms: GPUBuffer[] = [];

  constructor(cfg: GptConfig, device: GPUDevice) {
    this.cfg = cfg;
    this.device = device;
    this.layout = gptLayout(cfg);
    this.P = this.layout.total;
    const P = this.P;
    this.params = buffer(device, P);
    this.grads = buffer(device, P);
    this.m = buffer(device, P);
    this.v = buffer(device, P);
    this.mask = buffer(device, P);
    const mask = new Uint32Array(P);
    for (const [o, n] of this.layout.decay) mask.fill(1, o, o + n);
    device.queue.writeBuffer(this.mask, 0, mask);
    this.st = buffer(device, 4);
    this.fst = buffer(device, 4);
    this.part = buffer(device, PARTS);
    this.dummyA = buffer(device, 4);
    this.dummyB = buffer(device, 4);
  }

  private run<T>(f: () => Promise<T>): Promise<T> {
    const p = this.chain.then(f, f);
    this.chain = p.catch(() => undefined);
    return p;
  }

  /** Upload parameters (flat layout of `gptLayout`) and reset the optimiser. */
  setParams(p: Float32Array) {
    const q = this.device.queue;
    q.writeBuffer(this.params, 0, p);
    const zero = new Float32Array(this.P);
    q.writeBuffer(this.m, 0, zero);
    q.writeBuffer(this.v, 0, zero);
    q.writeBuffer(this.st, 0, new Uint32Array(4));
  }

  readParams(): Promise<Float32Array> {
    return this.run(() => readBack(this.device, this.params, this.P));
  }

  private u(v: GU) {
    const b = uniform(this.device, gu(v));
    this.uniforms.push(b);
    return b;
  }
  private gemmOp(g: GemmU, bufs: GPUBuffer[]): Op {
    const d = this.device;
    const p = pipeline(d, GEMM_WGSL);
    const ub = uniform(d, gemmUniform(g));
    this.uniforms.push(ub);
    return { p, g: bind(d, p, [ub, ...bufs]), x: Math.ceil(g.N / 16), y: Math.ceil(g.M / 16), z: 1 };
  }
  private op(code: string, bufs: GPUBuffer[], threads: number, wg: number): Op {
    const p = pipeline(this.device, code);
    const groups = Math.ceil(threads / wg);
    return { p, g: bind(this.device, p, bufs), x: Math.min(groups, 65535), y: Math.ceil(groups / 65535), z: 1 };
  }

  private ensureSteps(K: number, B: number, T: number) {
    const tokNeed = K * B * (T + 1);
    if (!this.tok || this.tokCap < tokNeed) {
      this.tok?.destroy();
      this.tokCap = Math.max(tokNeed, 4096);
      this.tok = buffer(this.device, this.tokCap);
      this.clearWorkspaces(); // their bind groups reference the old token buffer
    }
    if (!this.lr || this.stepCap < K) {
      this.lr?.destroy();
      this.lossOut?.destroy();
      this.stepCap = Math.max(K, 256);
      this.lr = buffer(this.device, this.stepCap);
      this.lossOut = buffer(this.device, this.stepCap);
      this.clearWorkspaces();
      this.optCache.clear();
    }
  }
  private clearWorkspaces() {
    for (const w of this.workspaces.values()) w.owned.forEach((b) => b.destroy());
    this.workspaces.clear();
  }

  private workspace(B: number, T: number): Workspace {
    const key = `${B}x${T}`;
    const hit = this.workspaces.get(key);
    if (hit) return hit;
    const d0 = this.device;
    const { cfg, layout } = this;
    const { d, heads: H, vocab: V, layers: L, ctx } = cfg;
    const F = (cfg.ff ?? 4) * d;
    const hd = d / H;
    const N = B * T;
    const owned: GPUBuffer[] = [];
    const A = (n: number) => {
      const b = buffer(d0, n);
      owned.push(b);
      return b;
    };
    const layers: LayerBufs[] = Array.from({ length: L }, () => ({
      xin: A(N * d), xh1: A(N * d), rs1: A(N), a1: A(N * d), qkv: A(N * 3 * d), att: A(B * H * T * T), y: A(N * d),
      xmid: A(N * d), xh2: A(N * d), rs2: A(N), a2: A(N * d), hpre: A(N * F), hact: A(N * F),
    }));
    const xL = A(N * d);
    const x0 = L > 0 ? layers[0].xin : xL; // the embedding writes straight into the first block's input
    const xhf = A(N * d), rsf = A(N), af = A(N * d), logits = A(N * V), dlog = A(N * V), lossRow = A(N);
    const tmp = A(N * Math.max(3 * d, F, V)), scratch = A(N * Math.max(3 * d, F, V));
    const dx = A(N * d), dtmp = A(N * d), dF = A(N * F), dqkv = A(N * 3 * d), dS = A(B * H * T * T);
    const base: GU = { N, T, B, d, H, hd, V, ctx, P: this.P };
    const { params: Pb, grads: G, st } = this;
    const tok = this.tok!;
    const fwd: Op[] = [];
    const bwd: Op[] = [];

    /** Y = X·Wᵀ (+ b), optional GELU into `act` (else the epilogue writes scratch). */
    const linear = (X: GPUBuffer, din: number, wOff: number, bOff: number, dout: number, Y: GPUBuffer, act?: GPUBuffer) =>
      bOff < 0
        ? this.gemmOp({ M: N, N: dout, K: din, transB: 1, lda: din, ldb: din, ldc: dout, strideA: 0, strideB: 0, strideC: 0, offB: wOff, mode: MODE.plain }, [X, Pb, Y, this.dummyA, this.dummyB])
        : this.gemmOp({ M: N, N: dout, K: din, transB: 1, lda: din, ldb: din, ldc: dout, strideA: 0, strideB: 0, strideC: 0, offB: wOff, mode: MODE.forward, act: act ? 4 : 0, auxOff: bOff }, [X, Pb, Y, Pb, act ?? scratch]);
    /** dW = dYᵀ·X (overwrite), db = Σ dY (overwrite), dX = dY·W (overwrite; × GELU′ when given). */
    const linearBack = (dY: GPUBuffer, X: GPUBuffer, din: number, wOff: number, bOff: number, dout: number, dX: GPUBuffer | null, gelu?: { z: GPUBuffer; a: GPUBuffer }) => {
      bwd.push(this.gemmOp({ M: dout, N: din, K: N, transA: 1, lda: dout, ldb: din, ldc: din, strideA: 0, strideB: 0, strideC: 0, offC: wOff, mode: MODE.plain }, [dY, X, G, this.dummyA, this.dummyB]));
      if (bOff >= 0) bwd.push(this.op(BIAS, [this.u({ ...base, width: dout, o0: bOff }), dY, G], dout, 64));
      if (dX)
        bwd.push(
          gelu
            ? this.gemmOp({ M: N, N: din, K: dout, lda: dout, ldb: din, ldc: din, strideA: 0, strideB: 0, strideC: 0, offB: wOff, mode: MODE.inputGrad, act: 4 }, [dY, Pb, dX, gelu.z, gelu.a])
            : this.gemmOp({ M: N, N: din, K: dout, lda: dout, ldb: din, ldc: din, strideA: 0, strideB: 0, strideC: 0, offB: wOff, mode: MODE.plain }, [dY, Pb, dX, this.dummyA, this.dummyB]),
        );
    };
    const lnF = (X: GPUBuffer, g: number, b: number, Y: GPUBuffer, Xh: GPUBuffer, Rs: GPUBuffer) =>
      fwd.push(this.op(LN_FWD, [this.u({ ...base, o0: g, o1: b }), Pb, X, Y, Xh, Rs], N, 64));
    const lnB = (dY: GPUBuffer, g: number, b: number, Xh: GPUBuffer, Rs: GPUBuffer, dX: GPUBuffer, accumulate: boolean) => {
      bwd.push(this.op(LN_BWD_P, [this.u({ ...base, o0: g, o1: b }), dY, Xh, G], d, 64));
      bwd.push(this.op(LN_BWD_X, [this.u({ ...base, o0: g, flag: accumulate ? 1 : 0 }), Pb, dY, Xh, Rs, dX], N, 64));
    };
    const add = (a: GPUBuffer, b: GPUBuffer, out: GPUBuffer, len: number, list: Op[]) => list.push(this.op(ADD, [this.u({ ...base, len }), a, b, out], len, 256));
    const scale = 1 / Math.sqrt(hd);

    // ---- forward
    fwd.push(this.op(EMBED, [this.u({ ...base, o0: layout.wte, o1: layout.wpe }), Pb, tok, st, x0], N * d, 256));
    for (let l = 0; l < L; l++) {
      const c = layers[l];
      const bl = layout.blocks[l];
      lnF(c.xin, bl.ln1g, bl.ln1b, c.a1, c.xh1, c.rs1);
      fwd.push(linear(c.a1, d, bl.wqkv, bl.bqkv, 3 * d, c.qkv));
      fwd.push(this.op(ATT_FWD, [this.u({ ...base, scale }), c.qkv, c.att, c.y], B * H * T, 64));
      fwd.push(linear(c.y, d, bl.wo, bl.bo, d, tmp));
      add(c.xin, tmp, c.xmid, N * d, fwd);
      lnF(c.xmid, bl.ln2g, bl.ln2b, c.a2, c.xh2, c.rs2);
      fwd.push(linear(c.a2, d, bl.w1, bl.b1, F, c.hpre, c.hact));
      fwd.push(linear(c.hact, F, bl.w2, bl.b2, d, tmp));
      add(c.xmid, tmp, l + 1 < L ? layers[l + 1].xin : xL, N * d, fwd);
    }
    lnF(xL, layout.lnfg, layout.lnfb, af, xhf, rsf);
    fwd.push(linear(af, d, layout.wte, -1, V, logits));
    fwd.push(this.op(CE, [this.u(base), logits, tok, st, dlog, lossRow], N, 64));
    fwd.push(this.op(LOSS_SUM, [this.u(base), lossRow, st, this.lossOut!], 256, 256));

    // ---- backward
    linearBack(dlog, af, d, layout.wte, -1, V, dtmp);
    lnB(dtmp, layout.lnfg, layout.lnfb, xhf, rsf, dx, false);
    for (let l = L - 1; l >= 0; l--) {
      const c = layers[l];
      const bl = layout.blocks[l];
      linearBack(dx, c.hact, F, bl.w2, bl.b2, d, dF, { z: c.hpre, a: c.hact });
      linearBack(dF, c.a2, d, bl.w1, bl.b1, F, dtmp);
      lnB(dtmp, bl.ln2g, bl.ln2b, c.xh2, c.rs2, dx, true);
      linearBack(dx, c.y, d, bl.wo, bl.bo, d, dtmp);
      bwd.push(this.op(ATT_BWD_Q, [this.u({ ...base, scale }), c.qkv, c.att, dtmp, dS, dqkv], B * H * T, 64));
      bwd.push(this.op(ATT_BWD_KV, [this.u({ ...base, scale }), c.qkv, c.att, dtmp, dS, dqkv], B * H * T, 64));
      linearBack(dqkv, c.a1, d, bl.wqkv, bl.bqkv, 3 * d, dtmp);
      lnB(dtmp, bl.ln1g, bl.ln1b, c.xh1, c.rs1, dx, true);
    }
    bwd.push(this.op(EMB_TOK, [this.u({ ...base, o0: layout.wte }), dx, tok, st, G], V * d, 64));
    bwd.push(this.op(EMB_POS, [this.u({ ...base, o1: layout.wpe }), dx, G], ctx * d, 64));
    const ws: Workspace = { B, T, forward: fwd, backward: bwd, owned };
    this.workspaces.set(key, ws);
    return ws;
  }

  private optCache = new Map<string, Op[]>();
  private optimiserOps(o: AdamWOpts): Op[] {
    const key = JSON.stringify(o);
    const hit = this.optCache.get(key);
    if (hit) return hit;
    const base: GU = { P: this.P, len: PARTS };
    const uo = this.u({ ...base, b1: o.beta1 ?? 0.9, b2: o.beta2 ?? 0.99, eps: o.eps ?? 1e-8, wd: o.weightDecay ?? 0.1, clip: o.clip ?? 0 });
    const ops = [
      { ...this.op(NORM_PART, [this.u(base), this.grads, this.part], PARTS * 256, 256), x: PARTS, y: 1 },
      this.op(NORM_FINAL, [this.u(base), this.part, this.fst], 1, 1),
      this.op(ADAMW, [uo, this.params, this.grads, this.m, this.v, this.lr!, this.st, this.fst, this.mask], this.P, 256),
      this.op(TICK, [this.u({ flag: 1 }), this.st], 1, 1),
    ];
    this.optCache.set(key, ops);
    return ops;
  }

  /**
   * K AdamW steps. `batches` holds K batches of B rows of T + 1 tokens ([K × B × (T+1)]), `lr` one
   * learning rate per step. Resolves to the K training losses.
   */
  train(batches: Int32Array, K: number, B: number, T: number, lr: Float32Array, o: AdamWOpts = {}): Promise<Float32Array> {
    return this.run(async () => {
      const d = this.device;
      this.ensureSteps(K, B, T);
      const ws = this.workspace(B, T);
      d.queue.writeBuffer(this.tok!, 0, new Uint32Array(batches.buffer, batches.byteOffset, K * B * (T + 1)));
      d.queue.writeBuffer(this.lr!, 0, lr);
      d.queue.writeBuffer(this.st, 0, new Uint32Array([0]));
      const opt = this.optimiserOps(o);
      const enc = d.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let k = 0; k < K; k++)
        for (const op of [...ws.forward, ...ws.backward, ...opt]) {
          pass.setPipeline(op.p);
          pass.setBindGroup(0, op.g);
          pass.dispatchWorkgroups(op.x, op.y, op.z);
        }
      pass.end();
      return readBack(d, this.lossOut!, K, enc);
    });
  }

  /** Mean loss on B rows of T + 1 tokens, without updating anything. */
  evalLoss(batch: Int32Array, B: number, T: number): Promise<number> {
    return this.run(async () => {
      const d = this.device;
      this.ensureSteps(1, B, T);
      const ws = this.workspace(B, T);
      d.queue.writeBuffer(this.tok!, 0, new Uint32Array(batch.buffer, batch.byteOffset, B * (T + 1)));
      d.queue.writeBuffer(this.st, 0, new Uint32Array([0]));
      const enc = d.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (const op of ws.forward) {
        pass.setPipeline(op.p);
        pass.setBindGroup(0, op.g);
        pass.dispatchWorkgroups(op.x, op.y, op.z);
      }
      pass.end();
      return (await readBack(d, this.lossOut!, 1, enc))[0];
    });
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.clearWorkspaces();
    for (const b of [this.params, this.grads, this.m, this.v, this.mask, this.st, this.fst, this.part, this.tok, this.lr, this.lossOut, this.dummyA, this.dummyB, ...this.uniforms]) b?.destroy();
  }
}
