// authored-by: Claude Opus 5.5
/**
 * "Characters or BPE?": the same sentence cut into characters and into BPE pieces learned on the
 * tales, with the trade-off measured on the held-out tales.
 */
import { useMemo, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented } from '../../components/controls';
import { encodeBpe } from '../../lib/corpora/bpe';
import { VOCAB } from '../../lib/corpora/grimm';
import { CPU_RUN } from '../../lib/compute/gptReference';
import { MERGES } from './bpeMerges';
import { cleanPrompt, useCorpus } from './TrainWidget';
import { showChar } from './tinyGpt';

const SIZES = [0, 100, 250, 500] as const;

export function TokeniserWidget() {
  const lang = useCorpus();
  const [text, setText] = useState('There was once a king who had three daughters.');
  const [merges, setMerges] = useState<(typeof SIZES)[number]>(500);
  const clean = cleanPrompt(text);
  const bpe = useMemo(() => ({ merges: MERGES.slice(0, merges), vocab: VOCAB + merges }), [merges]);
  const pieces = merges === 0 ? [...clean] : encodeBpe(bpe, clean);
  const d = CPU_RUN.cfg.d;
  const T = CPU_RUN.cfg.ctx;
  const table = useMemo(() => {
    if (!lang) return null;
    const val = lang.corpus.text.slice(lang.corpus.split);
    return SIZES.map((n) => {
      const toks = n === 0 ? val.length : encodeBpe({ merges: MERGES.slice(0, n), vocab: VOCAB + n }, val).length;
      return { n, vocab: VOCAB + n, toks, cpt: val.length / toks, emb: (VOCAB + n) * d };
    });
  }, [lang, d]);
  const fmt = (n: number) => n.toLocaleString('en-GB');
  return (
    <Widget
      title="Characters or BPE pieces?"
      wide
      caption={
        <>
          Type a sentence and change how many merges the tokeniser has learned from the training tales (the same byte-pair encoding as in the tokenization lesson, with
          GPT-2’s rule that a word keeps its leading space). The table measures every choice on the four held-out tales.
        </>
      }
    >
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <input className="text-input" value={text} onChange={(e) => setText(e.target.value)} maxLength={90} style={{ flex: '1 1 260px' }} aria-label="Sentence to tokenise" />
        <Segmented options={SIZES.map((n) => ({ value: n, label: n === 0 ? 'characters' : `${n} merges` }))} value={merges} onChange={(v) => setMerges(v)} />
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, margin: '10px 0' }} aria-label="Tokens">
        {pieces.map((p, i) => (
          <span key={i} className="mono" style={{ padding: '2px 5px', borderRadius: 4, background: i % 2 ? 'var(--accent-soft)' : 'var(--bg-sunken)', border: '1px solid var(--border)', fontSize: 13 }}>
            {[...p].map(showChar).join('')}
          </span>
        ))}
      </div>
      <p className="live-note" style={{ marginTop: 0 }}>
        {pieces.length} tokens for {clean.length} characters
        {merges ? `: about ${(clean.length / Math.max(1, pieces.length)).toFixed(1)} characters per token.` : ': one token per character.'}
      </p>
      {table && (
        <div style={{ overflowX: 'auto' }}>
          <table className="num-table" style={{ fontSize: 13 }}>
            <thead>
              <tr>
                <th>tokeniser</th>
                <th style={{ textAlign: 'right' }}>vocabulary</th>
                <th style={{ textAlign: 'right' }}>tokens in the held-out tales</th>
                <th style={{ textAlign: 'right' }}>characters per token</th>
                <th style={{ textAlign: 'right' }}>a {T}-token window sees</th>
                <th style={{ textAlign: 'right' }}>embedding table at width {d}</th>
              </tr>
            </thead>
            <tbody>
              {table.map((r) => (
                <tr key={r.n} style={r.n === merges ? { background: 'var(--accent-soft)' } : undefined}>
                  <td>{r.n === 0 ? 'characters' : `BPE, ${r.n} merges`}</td>
                  <td className="mono" style={{ textAlign: 'right' }}>{fmt(r.vocab)}</td>
                  <td className="mono" style={{ textAlign: 'right' }}>{fmt(r.toks)}</td>
                  <td className="mono" style={{ textAlign: 'right' }}>{r.cpt.toFixed(2)}</td>
                  <td className="mono" style={{ textAlign: 'right' }}>≈ {Math.round(T * r.cpt)} characters</td>
                  <td className="mono" style={{ textAlign: 'right' }}>{fmt(r.emb)} parameters</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Widget>
  );
}
