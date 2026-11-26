// authored-by: Claude Opus 5.5
/**
 * "Assemble": the tiny GPT as a stack of parts you have already built, with the real parameter
 * count of each (from `gptLayout`, so the numbers are the engine's own).
 */
import { useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { LessonLink } from '../../components/Term';
import { gptParamCount } from '../../lib/compute/gpt';
import { CPU_RUN, GPU_RUN } from '../../lib/compute/gptReference';
import { parameterGroups } from './tinyGpt';

const FROM: Record<string, { to: string; name: string }> = {
  wte: { to: 'one-hot-vs-dense', name: 'dense vectors' },
  wpe: { to: 'positions', name: 'positions' },
  ln1: { to: 'transformer-block', name: 'the block' },
  att: { to: 'multi-head-attention', name: 'multi-head attention' },
  ln2: { to: 'transformer-block', name: 'the block' },
  mlp: { to: 'transformer-block', name: 'the block' },
  lnf: { to: 'transformer-block', name: 'the block' },
  out: { to: 'next-token-prediction', name: 'next-token prediction' },
};

export function AnatomyWidget() {
  const [which, setWhich] = useState<'small' | 'large'>('small');
  const cfg = (which === 'small' ? CPU_RUN : GPU_RUN).cfg;
  const groups = parameterGroups(cfg);
  const total = gptParamCount(cfg);
  const fmt = (n: number) => n.toLocaleString('en-GB');
  const rows = groups.flatMap((g) => (g.perBlock ? [] : [g]));
  const block = groups.filter((g) => g.perBlock);
  const blockTotal = block.reduce((s, g) => s + g.count, 0);
  const bar = (n: number) => (
    <span style={{ display: 'inline-block', height: 8, width: `${Math.max(n ? 1.5 : 0, (100 * n) / total)}%`, background: 'var(--accent)', borderRadius: 3, verticalAlign: 'middle' }} />
  );
  const row = (g: (typeof groups)[number], times = 1) => (
    <tr key={g.id + times}>
      <td style={{ textAlign: 'left' }}>{g.label}</td>
      <td className="mono" style={{ whiteSpace: 'nowrap' }}>{g.formula}</td>
      <td className="mono" style={{ textAlign: 'right' }}>{fmt(g.count * times)}</td>
      <td style={{ width: '28%', minWidth: 60 }}>{bar(g.count * times)}</td>
      <td style={{ fontSize: 12.5, textAlign: 'left' }}>
        <LessonLink to={FROM[g.id].to}>{FROM[g.id].name}</LessonLink>
      </td>
    </tr>
  );
  const emb = rows[0].count + rows[1].count;
  return (
    <Widget
      title="Assemble the machine: where the parameters live"
      wide
      caption={
        <>
          The model in this lesson, part by part, in the order a character travels through it. Counts are read from the engine’s own parameter layout. Each part links to
          the lesson where you built it. Switch to the larger model and watch where the growth goes: everything inside a block grows with the square of the width.
        </>
      }
    >
      <Segmented
        options={[
          { value: 'small', label: `tiny: width ${CPU_RUN.cfg.d}, ${CPU_RUN.cfg.layers} blocks` },
          { value: 'large', label: `larger: width ${GPU_RUN.cfg.d}, ${GPU_RUN.cfg.layers} blocks` },
        ]}
        value={which}
        onChange={(v) => setWhich(v)}
      />
      <div style={{ overflowX: 'auto', marginTop: 8 }}>
        <table className="num-table" style={{ fontSize: 13, width: '100%' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>part</th>
              <th>shape</th>
              <th style={{ textAlign: 'right' }}>parameters</th>
              <th>share</th>
              <th>built in</th>
            </tr>
          </thead>
          <tbody>
            {row(rows[0])}
            {row(rows[1])}
            <tr>
              <td colSpan={5} style={{ fontWeight: 600, paddingTop: 8, textAlign: 'left' }}>
                × {cfg.layers} transformer blocks ({fmt(blockTotal)} parameters each)
              </td>
            </tr>
            {block.map((g) => row(g, cfg.layers))}
            {row(rows[2])}
            {row(rows[3])}
            <tr>
              <td style={{ fontWeight: 600, textAlign: 'left' }}>total</td>
              <td />
              <td className="mono" style={{ textAlign: 'right', fontWeight: 600 }}>{fmt(total)}</td>
              <td colSpan={2} />
            </tr>
          </tbody>
        </table>
      </div>
      <p className="live-note">
        {which === 'small'
          ? `The ${cfg.layers} blocks hold ${((100 * blockTotal * cfg.layers) / total).toFixed(0)}% of the ${fmt(total)} parameters; the two embedding tables hold ${((100 * emb) / total).toFixed(0)}%. Tying the output layer to the token embedding saves another ${fmt(cfg.vocab * cfg.d)}.`
          : `Three times the width means about nine times the parameters per block: ${fmt(blockTotal)} instead of ${fmt(parameterGroups(CPU_RUN.cfg).filter((g) => g.perBlock).reduce((s, g) => s + g.count, 0))}. The embeddings only grow three-fold, so the blocks now hold ${((100 * blockTotal * cfg.layers) / total).toFixed(0)}% of the ${fmt(total)}.`}
      </p>
    </Widget>
  );
}
