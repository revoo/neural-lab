// authored-by: Claude Opus 5.5
/**
 * V2-WP3 (required): CPU vs WebGPU throughput, measured live in the learner's browser.
 * CPU: the Float32 reference tier's matmul on the main thread. GPU: the WGSL tiled GEMM, either
 * batched (many multiplications per command submission, one read-back: how the training engine
 * works) or naive (one submission and a full read-back per multiplication).
 */
import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { LogChart } from '../../components/LogChart';
import { BackendChip, useComputeTier } from '../../components/BackendChip';
import { matmul, tensor } from '../../lib/compute/cpuBackend';
import { getGpu, gpuGemm } from '../../lib/compute/webgpuBackend';
import { createRng } from '../../lib/rng';
import { CPU_MAX_MM, crossover, gflops, intensity, repsFor, SHAPE_LABEL, SIZES, type BenchRow, type Shape } from './bench';

const yieldToPage = () => new Promise((r) => setTimeout(r, 0));
const fmt = (v?: number) => (v === undefined ? '—' : v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));

export function ThroughputBench() {
  const tier = useComputeTier();
  const [shape, setShape] = useState<Shape>('mm');
  const [rows, setRows] = useState<Record<Shape, BenchRow[]>>({ mm: [], mv: [] });
  const [busy, setBusy] = useState('');
  const [ranOn, setRanOn] = useState<Record<Shape, string>>({ mm: '', mv: '' });

  const run = async () => {
    const out: BenchRow[] = [];
    const gpu = tier.kind === 'webgpu' ? await getGpu() : null;
    setRows((r) => ({ ...r, [shape]: [] }));
    for (const n of SIZES[shape]) {
      const r = createRng(n);
      const K = n;
      const N = shape === 'mm' ? n : 1;
      const A = Float32Array.from({ length: n * K }, () => r.normal());
      const B = Float32Array.from({ length: K * N }, () => r.normal());
      const row: BenchRow = { n };
      let ref: Float32Array | null = null;
      if (shape === 'mv' || n <= CPU_MAX_MM) {
        setBusy(`CPU, n = ${n}…`);
        await yieldToPage();
        const tA = tensor(n, K, A);
        const tB = tensor(K, N, B);
        let reps = 0;
        const t0 = performance.now();
        do {
          ref = matmul(tA, tB).data;
          reps++;
        } while (performance.now() - t0 < 25);
        row.cpu = gflops(shape, n, reps, performance.now() - t0);
      }
      if (gpu) {
        setBusy(`WebGPU, n = ${n}…`);
        await yieldToPage();
        const trial = await gpuGemm(gpu.device, A, B, { M: n, N, K }, 1);
        const reps = repsFor(trial.ms, 1, 60, 512);
        const res = await gpuGemm(gpu.device, A, B, { M: n, N, K }, reps);
        row.gpu = gflops(shape, n, reps, res.ms);
        const nreps = Math.min(reps, 32);
        const naive = await gpuGemm(gpu.device, A, B, { M: n, N, K }, nreps, false);
        row.naive = gflops(shape, n, nreps, naive.ms);
        if (ref) {
          let worst = 0;
          for (let i = 0; i < ref.length; i++) worst = Math.max(worst, Math.abs(res.out[i] - ref[i]) / Math.max(1, Math.abs(ref[i])));
          row.err = worst;
        }
      }
      out.push(row);
      setRows((rs) => ({ ...rs, [shape]: [...out] }));
    }
    setBusy('');
    setRanOn((o) => ({ ...o, [shape]: gpu ? `WebGPU on ${tier.gpuName}` : 'CPU only' }));
  };

  const data = rows[shape];
  const all = data.flatMap((r) => [r.cpu, r.gpu, r.naive]).filter((v): v is number => v !== undefined && v > 0);
  const yDomain: [number, number] = all.length ? [10 ** Math.floor(Math.log10(Math.min(...all))), 10 ** Math.ceil(Math.log10(Math.max(...all)))] : [0.01, 100];
  const sizes = SIZES[shape];
  const series = [
    { label: 'CPU (Float32, main thread)', color: 'var(--c-warn)', dots: true, points: data.filter((r) => r.cpu).map((r) => [r.n, r.cpu!] as [number, number]) },
    { label: 'WebGPU, batched submissions', color: 'var(--c-good)', dots: true, points: data.filter((r) => r.gpu).map((r) => [r.n, r.gpu!] as [number, number]) },
    { label: 'WebGPU, read back after every multiply', color: 'var(--c-a)', dashed: true, dots: true, points: data.filter((r) => r.naive).map((r) => [r.n, r.naive!] as [number, number]) },
  ].filter((s) => s.points.length);
  const cross = crossover(data);
  const done = data.length === sizes.length && !busy;
  const last = data.at(-1);

  let note = 'Press Run. Each size is timed for at least 25 ms on the CPU and about 60 ms on the GPU; the page may stutter while the CPU runs.';
  if (done && !data.some((r) => r.gpu))
    note = `CPU only: ${tier.gpuAvailable ? 'you switched to the CPU (use the chip to switch back)' : 'this browser does not expose WebGPU'}. The CPU managed ${fmt(last?.cpu)} GFLOP/s at n = ${last?.n}.`;
  else if (done && shape === 'mm')
    note =
      cross === null
        ? `The CPU won at every size here. That happens on software GPUs (headless browsers use SwiftShader, which runs "GPU" code on the CPU). On a real graphics card, expect the GPU to pull ahead as n grows.`
        : `The batched GPU overtook the CPU at n = ${cross}. Below that, launching kernels and reading results back costs more than the arithmetic. At n = ${last?.n} each byte moved feeds ${intensity('mm', last!.n).toFixed(0)} FLOPs, so the GPU's parallel units stay busy.`;
  else if (done)
    note = `Matrix × vector does only ${intensity('mv', 1024).toFixed(2)} FLOPs per byte at every size: each weight is used once. Both processors spend their time moving bytes, so the GFLOP/s stay far below the matrix × matrix numbers. That is decoding's problem in a nutshell.`;

  return (
    <Widget
      title="CPU vs WebGPU, measured in your browser"
      wide
      actions={<BackendChip tier={tier} />}
      caption={
        <>
          Real timings on your machine, so your numbers will differ from anyone else’s. Pick a shape and press <strong>Run</strong>. The solid GPU line records
          many multiplications into one command submission and reads back once, which is how this site’s training engine works; the dashed line submits each
          multiplication separately and copies its result back to the page, the naive design. Results are checked against the CPU’s answer. Both GPU lines use
          one general-purpose tiled kernel (and it wastes most of each tile on a single vector); tuned vendor libraries are several times faster, so treat the
          GPU numbers as a floor.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10 }}>
        <Segmented options={(['mm', 'mv'] as Shape[]).map((s) => ({ value: s, label: SHAPE_LABEL[s] }))} value={shape} onChange={setShape} />
        <button className="btn primary" disabled={!!busy || tier.kind === 'checking'} onClick={run}>
          {busy ? busy : '▶ Run'}
        </button>
      </div>
      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 360px', minWidth: 0, maxWidth: 460 }}>
          <LogChart
            series={series.length ? series : [{ points: [], color: 'var(--text-faint)' }]}
            xDomain={[sizes[0], sizes.at(-1)!]}
            yDomain={yDomain}
            xLabel="n"
            xTicks={sizes}
            logX
            ariaLabel="GFLOP/s against matrix size"
            width={440}
            height={230}
          />
          <div className="faint" style={{ fontSize: 12 }}>
            vertical: GFLOP/s (log scale) · horizontal: n {shape === 'mm' ? '(n × n times n × n)' : '(n × n times a length-n vector)'}
          </div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 12, marginTop: 4 }}>
            {series.map((s) => (
              <span key={s.label}>
                <span style={{ display: 'inline-block', width: 16, borderTop: `2px ${s.dashed ? 'dashed' : 'solid'} ${s.color}`, verticalAlign: 'middle', marginRight: 5 }} />
                {s.label}
              </span>
            ))}
          </div>
        </div>
        <div style={{ flex: '1 1 260px', minWidth: 0, overflowX: 'auto' }}>
          <table className="num-table" style={{ fontSize: 12.5 }}>
            <thead>
              <tr>
                <th>n</th>
                <th>FLOPs per byte</th>
                <th>CPU</th>
                <th>GPU batched</th>
                <th>GPU naive</th>
              </tr>
            </thead>
            <tbody>
              {sizes.map((n) => {
                const r = data.find((x) => x.n === n);
                return (
                  <tr key={n}>
                    <td>{n}</td>
                    <td>{intensity(shape, n).toFixed(intensity(shape, n) < 10 ? 2 : 0)}</td>
                    <td>{r ? (r.cpu === undefined ? 'skipped' : fmt(r.cpu)) : ''}</td>
                    <td>{r ? fmt(r.gpu) : ''}</td>
                    <td>{r ? fmt(r.naive) : ''}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="faint" style={{ fontSize: 12 }}>
            GFLOP/s. The CPU skips matrix products above n = {CPU_MAX_MM}: one 1024 × 1024 product is 2.1 billion FLOPs.
            {data.some((r) => r.err !== undefined) && ` GPU vs CPU answers agree to ${Math.max(...data.map((r) => r.err ?? 0)).toExponential(0)} (relative).`}
            {ranOn[shape] && ` Last run: ${ranOn[shape]}.`}
          </div>
        </div>
      </div>
      <p className="live-note">{note}</p>
    </Widget>
  );
}
