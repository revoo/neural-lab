// authored-by: Claude Opus 5.5
import { describe, expect, it } from 'vitest';
import text from './grimm.txt?raw';
import { prepareCorpus, VOCAB } from './grimm';
import { chunksOf, encodeBpe, trainBpe } from './bpe';

describe('characters versus BPE on the Grimm corpus', () => {
  const c = prepareCorpus(text);
  const train = c.text.slice(0, c.split);
  const val = c.text.slice(c.split);
  const bpe = trainBpe(train, 1000, VOCAB);
  it('pre-splits like GPT-2: each word keeps its leading space', () => {
    expect(chunksOf(" The king's daughter, said 'no'.")).toEqual([' The', " king's", ' daughter', ',', ' said', " 'no'", '.']);
  });
  it('token counts on the held-out tales (47,267 characters)', () => {
    const counts = [0, 100, 250, 500, 1000].map((n) => encodeBpe({ merges: bpe.merges.slice(0, n), vocab: VOCAB + n }, val).length);
    expect(val.length).toBe(47267);
    expect(counts).toEqual([47267, 24557, 19750, 16881, 14558]);
  });
  it('the first merges are the commonest pairs, and a 500-merge BPE splits a sentence into word pieces', () => {
    expect(bpe.merges.slice(0, 4)).toEqual([['h', 'e'], [' ', 't'], [' ', 'a'], [' t', 'he']]);
    const b500 = { merges: bpe.merges.slice(0, 500), vocab: VOCAB + 500 };
    expect(encodeBpe(b500, 'There was once a king who had three daughters.')).toEqual(['The', 're', ' was', ' once', ' a', ' king', ' who', ' had', ' three', ' daught', 'ers', '.']);
  });
});
