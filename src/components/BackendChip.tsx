// authored-by: Claude Opus 5.5
/**
 * V2-WP3: the compute-tier status chip shown on heavy widgets. It always says where the numbers are
 * being computed: "⚡ WebGPU · <adapter>" when the GPU tier is active, "CPU · main thread" when the
 * Float32 reference tier is running on the page's main thread. When a GPU is available, a button
 * switches between the two so the learner can compare them live (with the timing beside it).
 */
import { activeKind, computePref, useComputePref, useGpuStatus } from '../lib/compute/selectBackend';

export type ComputeTier = ReturnType<typeof useComputeTier>;

export function useComputeTier() {
  const status = useGpuStatus();
  const pref = useComputePref();
  const kind = activeKind(status, pref);
  return { kind, pref, gpuName: status.state === 'ready' ? status.name : '', gpuAvailable: status.state === 'ready' };
}

/** `cpuWhere` names where the CPU tier runs when it isn't the main thread (the tiny GPT uses a Web Worker). */
export function BackendChip({ tier, timing, cpuWhere = 'main thread' }: { tier: ComputeTier; timing?: string; cpuWhere?: string }) {
  const { kind, gpuName, gpuAvailable, pref } = tier;
  const label = kind === 'webgpu' ? `⚡ WebGPU · ${gpuName}` : kind === 'cpu' ? `◼ CPU · ${cpuWhere}` : '… checking for a GPU';
  const title =
    kind === 'webgpu'
      ? 'Training and evaluation run in WGSL compute shaders on your GPU; weights and optimiser state stay in GPU memory.'
      : gpuAvailable
        ? `You switched to the CPU: the same maths in 32-bit floats, on the ${cpuWhere === 'main thread' ? 'browser’s main thread' : cpuWhere}.`
        : `This browser does not expose WebGPU, so the same maths runs in 32-bit floats on the CPU, ${cpuWhere === 'main thread' ? 'on the page’s main thread' : `in a ${cpuWhere} (the page stays responsive)`}.`;
  return (
    <span className="backend-row">
      <span className={`backend-chip ${kind}`} title={title} data-backend={kind}>
        {label}
      </span>
      {timing && <span className="backend-timing mono">{timing}</span>}
      {gpuAvailable && (
        <button className="backend-toggle" onClick={() => computePref.set(pref === 'cpu' ? 'auto' : 'cpu')} title="Run the same computation on the other tier to compare">
          {pref === 'cpu' ? 'use WebGPU' : 'compare on CPU'}
        </button>
      )}
    </span>
  );
}
