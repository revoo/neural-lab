// authored-by: Claude Opus 5.5
/**
 * The honest scorecard: validation loss on the held-out tales for the baselines, the recorded
 * reference runs, and the learner's own run, with one sample from each model.
 */
import { Widget } from '../../components/Widget';
import { CPU_REFERENCE } from '../../lib/compute/gptReferenceResults';
import { GPU_REFERENCE } from '../../lib/compute/gptReferenceGpuResults';
import { PROMPT } from '../../lib/compute/gptReference';
import { useLiveRun } from './runStore';

const fmt = (n: number) => n.toLocaleString('en-GB');
const pct = (v: number) => `${(100 * v).toFixed(0)}%`;

export function Scorecard() {
  const live = useLiveRun();
  const b = CPU_REFERENCE.baselines;
  const cpu = CPU_REFERENCE.checkpoints.at(-1)!;
  const gpu = GPU_REFERENCE.checkpoints.at(-1)!;
  const yours = live && live.checkpoints.length > 1 ? live.checkpoints.at(-1)! : null;
  const rows: { name: string; how: string; loss: number; real?: number; sample?: string; mark?: boolean }[] = [
    { name: 'blind guess', how: 'every character equally likely', loss: b.uniform },
    { name: 'letter counts', how: 'how often each character appears', loss: b.unigram },
    { name: 'letter-pair counts', how: 'which character follows which', loss: b.bigram },
    { name: `tiny GPT, ${fmt(CPU_REFERENCE.params)} parameters`, how: `recorded run, step ${fmt(cpu.step)}`, loss: cpu.valLoss, real: cpu.stats.realShare, sample: cpu.samples[0] },
    { name: `larger GPT, ${fmt(GPU_REFERENCE.params)} parameters`, how: `recorded run, step ${fmt(gpu.step)}`, loss: gpu.valLoss, real: gpu.stats.realShare, sample: gpu.samples[0] },
  ];
  if (yours) rows.push({ name: `your ${live!.label}`, how: `${live!.tier === 'webgpu' ? 'GPU' : 'CPU'} run, step ${fmt(yours.step)}`, loss: yours.valLoss, real: yours.stats.realShare, sample: yours.samples[0], mark: true });
  return (
    <Widget
      title="The honest scorecard"
      wide
      caption={
        <>
          Validation loss in nats per character on the four held-out tales (lower is better), and the share of real words in four 200-character samples per model. The
          baselines are counted on the training tales. The sample is the first of the four, from the prompt “{PROMPT}”, at temperature 1.
        </>
      }
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="num-table" style={{ fontSize: 13 }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>model</th>
              <th style={{ textAlign: 'left' }}>how it predicts</th>
              <th style={{ textAlign: 'right' }}>validation loss</th>
              <th style={{ textAlign: 'right' }}>real words</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name} style={r.mark ? { background: 'var(--accent-soft)' } : undefined}>
                <td style={{ textAlign: 'left' }}>{r.name}</td>
                <td style={{ textAlign: 'left' }}>{r.how}</td>
                <td className="mono" style={{ textAlign: 'right' }}>{r.loss.toFixed(3)}</td>
                <td className="mono" style={{ textAlign: 'right' }}>{r.real === undefined ? '' : pct(r.real)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows
        .filter((r) => r.sample)
        .map((r) => (
          <div key={r.name} style={{ marginTop: 8 }}>
            <div className="plot-tick">{r.name}:</div>
            <div className="mono gpt-sample">
              {PROMPT}
              <b>{r.sample}</b>
            </div>
          </div>
        ))}
    </Widget>
  );
}
