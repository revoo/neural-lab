// authored-by: Claude Opus 5.5
import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget } from '../../components/Widget';
import { Segmented, Slider } from '../../components/controls';
import { QUESTIONS, type Question } from './corpus';
import { buildIndex, chunkCorpus, containsAnswer, evaluate } from './rag';
import { LAB_BRIEFS, LAB_KS, LAB_LAMBDAS, LAB_SIZES, labExhaustive, type LabBrief } from './lab';

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const PRESETS = [
  'How soon do I get my money back?',
  'What does E2 mean?',
  'Can I use vinegar to remove limescale?',
  'How heavy is the K2?',
];

// ------------------------------------------------------------------ 1. Two ways to search

export function TwoSearches() {
  const [query, setQuery] = useState(PRESETS[0]);
  const index = useMemo(() => buildIndex(chunkCorpus(1)), []);
  const lex = useMemo(() => index.search(query, 0).slice(0, 4), [index, query]);
  const sem = useMemo(() => index.search(query, 1).slice(0, 4), [index, query]);
  const words = useMemo(() => index.queryTokens(query), [index, query]);
  const gold = QUESTIONS.find((q) => q.q === query);

  const List = ({ rows, which }: { rows: typeof lex; which: 'lex' | 'sem' }) => (
    <div style={{ display: 'grid', gap: 6 }}>
      {rows.map((r, i) => {
        const s = which === 'lex' ? r.lex : r.sem;
        const hit = gold ? containsAnswer(r.chunk, gold) : false;
        return (
          <div key={r.chunk.id} style={{ fontSize: 13, lineHeight: 1.45, padding: '6px 8px', borderRadius: 6, background: 'var(--bg-sunken)', border: `1px solid ${hit ? 'var(--c-good)' : 'var(--border)'}`, opacity: s > 0 ? 1 : 0.5 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--text-muted)' }}>
              <span>
                #{i + 1} · {r.chunk.title}
                {hit ? ' · the answer' : ''}
              </span>
              <span>{s.toFixed(3)}</span>
            </div>
            {r.chunk.sentences[0]}
          </div>
        );
      })}
    </div>
  );

  return (
    <Widget
      title="Two ways to search"
      wide
      caption={
        <>
          The knowledge base is 16 support pages for a fictional espresso machine, split into single sentences. Type a question (or pick one) and both searches run for real. <strong>Lexical</strong> search compares the words themselves, weighting rare words more (TF-IDF). <strong>Semantic</strong> search first maps words to meanings with a small hand-made dictionary of 38 concepts (“refund”, “money” and “paid” all mean MONEY), standing in for a learned embedding model. Scores are cosine similarities; a green border marks the sentence that answers a known question.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 8 }}>
        {PRESETS.map((p) => (
          <button key={p} type="button" className={`btn small${p === query ? ' primary' : ''}`} onClick={() => setQuery(p)}>
            {p}
          </button>
        ))}
      </div>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        aria-label="Your question"
        style={{ width: '100%', padding: '8px 10px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', color: 'var(--text)', fontSize: 14.5, marginBottom: 10 }}
      />
      <div className="readout" style={{ marginBottom: 10 }}>
        <span className="k">words that count </span>
        {words.length === 0 ? '(none)' : words.map((w) => `${w.word} (rarity ${w.idf.toFixed(1)}${w.concept ? ` → ${w.concept}` : ''})`).join(', ')}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16 }}>
        <div>
          <div className="field-label">Lexical (matching words)</div>
          <List rows={lex} which="lex" />
        </div>
        <div>
          <div className="field-label">Semantic (matching meanings)</div>
          <List rows={sem} which="sem" />
        </div>
      </div>
      <div className="live-note" style={{ marginTop: 12 }}>
        {words.length === 0
          ? 'Every word was a stop word (like “the” or “how”), so neither search has anything to go on.'
          : words.every((w) => !w.concept)
            ? 'None of these words is in the concept dictionary, so semantic search sees an empty vector and every score is 0. An embedding model has the same blind spot for identifiers it never learned, like product codes and error numbers. Lexical search matches them exactly.'
            : words.some((w) => !w.concept)
              ? `Semantic search ignores the words it has no meaning for (${words
                  .filter((w) => !w.concept)
                  .map((w) => w.word)
                  .join(', ')}); lexical search uses every word, rare ones most.`
              : 'Every word has a meaning in the dictionary, so semantic search can match sentences that use different words for the same idea. Lexical search only scores sentences that share the exact words.'}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 2. Score the retriever

export function RetrieverBench() {
  const [size, setSize] = useState(1);
  const [lambda, setLambda] = useState(0.5);
  const [k, setK] = useState(3);
  const index = useMemo(() => buildIndex(chunkCorpus(size)), [size]);
  const rows = useMemo(
    () =>
      QUESTIONS.map((q) => {
        const rank = (l: number) => {
          const res = index.search(q.q, l);
          const i = res.findIndex((r) => containsAnswer(r.chunk, q));
          return i >= 0 && res[i].score > 0 ? i + 1 : Infinity;
        };
        return { q, lex: rank(0), sem: rank(1), mix: rank(lambda) };
      }),
    [index, lambda],
  );
  const recall = (key: 'lex' | 'sem' | 'mix') => rows.filter((r) => r[key] <= k).length / rows.length;
  const r = useMemo(() => evaluate(size, lambda, k), [size, lambda, k]);
  const fmt = (v: number) => (v === Infinity ? '—' : String(v));
  const cell = (v: number) => ({ color: v <= k ? 'var(--c-good)' : 'var(--c-bad)' });

  return (
    <Widget
      title="Score the retriever"
      wide
      caption={
        <>
          34 questions, each labelled with the one sentence that answers it. A question counts as found if a chunk containing that sentence is in the top <strong>k</strong>. Choose how many sentences go in each chunk, how much weight the hybrid gives to semantic search (λ = 0 is pure lexical, 1 pure semantic) and k. The table shows where the answer ranked for each question (— means never: it scored 0).
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 8 }}>
        <Segmented label="Sentences per chunk" value={size} onChange={setSize} options={[1, 2, 3, 4, 5].map((s) => ({ value: s, label: String(s) }))} />
        <Segmented label="Chunks retrieved" value={k} onChange={setK} options={[1, 2, 3, 4, 5].map((s) => ({ value: s, label: String(s) }))} />
      </div>
      <Slider label="hybrid weight λ on semantic" value={lambda} min={0} max={1} step={0.25} onChange={setLambda} format={(v) => v.toFixed(2)} color="c-p" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(280px, 100%), 1fr))', gap: 16, marginTop: 10, alignItems: 'start' }}>
        <div style={{ overflowX: 'auto', maxHeight: 360, overflowY: 'auto' }}>
          <table className="num-table" style={{ width: '100%' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>Question</th>
                <th>Lexical</th>
                <th>Semantic</th>
                <th>Hybrid</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.q.q}>
                  <td style={{ textAlign: 'left', fontFamily: 'var(--font-ui)', fontSize: 12.5 }}>{row.q.q}</td>
                  <td style={cell(row.lex)}>{fmt(row.lex)}</td>
                  <td style={cell(row.sem)}>{fmt(row.sem)}</td>
                  <td style={cell(row.mix)}>{fmt(row.mix)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ display: 'grid', gap: 10 }}>
          <div className="readout">
            <span className="k">found in top {k}: lexical </span>
            {pct(recall('lex'))}
            {'\n'}
            <span className="k">found in top {k}: semantic </span>
            {pct(recall('sem'))}
            {'\n'}
            <span className="k">found in top {k}: hybrid </span>
            {pct(recall('mix'))}
            {'\n'}
            <span className="k">hybrid MRR (average of 1 / rank) </span>
            {r.mrr.toFixed(3)}
            {'\n'}
            <span className="k">retrieved text in the prompt </span>
            {r.promptWords.toFixed(0)} words
          </div>
          <div className="live-note">
            {recall('mix') >= Math.max(recall('lex'), recall('sem')) && recall('mix') > Math.min(recall('lex'), recall('sem'))
              ? `The hybrid finds ${pct(recall('mix'))}: lexical search rescues the error-code questions that semantic search can’t see, and semantic search rescues the reworded ones that share no words with the answer. `
              : 'With this weighting the hybrid is no better than its best component. '}
            {size > 1
              ? `Bigger chunks make it easier to find the answer (it only has to be somewhere in the chunk), but every retrieved chunk costs prompt space: ${r.promptWords.toFixed(0)} words here.`
              : 'One-sentence chunks keep the prompt short, but a chunk that is only one sentence has fewer words to match.'}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 3. What the model would see

export function PromptView() {
  const [qi, setQi] = useState(4);
  const [useRag, setUseRag] = useState(true);
  const q: Question = QUESTIONS[qi];
  const index = useMemo(() => buildIndex(chunkCorpus(2)), []);
  const top = useMemo(() => index.search(q.q, 0.5).slice(0, 3), [index, q]);
  const found = top.some((r) => containsAnswer(r.chunk, q));

  return (
    <Widget
      title="What the model would see"
      wide
      caption={
        <>
          Retrieval-augmented generation is, in the end, string concatenation: the retrieved chunks are pasted into the prompt above the question. This shows the exact prompt for a question, with the hybrid retriever (2-sentence chunks, λ = 0.5, top 3). No language model runs on this page; the point is to see what one would be given.
        </>
      }
    >
      <div className="btn-row" style={{ marginBottom: 10, gap: 12 }}>
        <select value={qi} onChange={(e) => setQi(Number(e.target.value))} aria-label="Question" style={{ padding: '6px 8px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-sunken)', color: 'var(--text)', maxWidth: '100%' }}>
          {QUESTIONS.map((x, i) => (
            <option key={x.q} value={i}>
              {x.q}
            </option>
          ))}
        </select>
        <Segmented
          value={useRag ? 1 : 0}
          onChange={(v) => setUseRag(v === 1)}
          options={[
            { value: 0, label: 'Question only' },
            { value: 1, label: 'With retrieval' },
          ]}
        />
      </div>
      <div className="readout" style={{ whiteSpace: 'pre-wrap', fontSize: 12.5, lineHeight: 1.6 }}>
        {useRag ? (
          <>
            <span className="k">You are a support assistant for Kestrel. Answer using only the sources below. If they don’t contain the answer, say you don’t know.{'\n\n'}</span>
            {top.map((r, i) => (
              <span key={r.chunk.id}>
                <span className="k">[Source {i + 1}: {r.chunk.title}]{'\n'}</span>
                {r.chunk.sentences.map((s, j) => (
                  <span key={j} style={r.chunk.doc === q.doc && r.chunk.first + j === q.sent ? { color: 'var(--c-good)', fontWeight: 600 } : undefined}>
                    {s}{' '}
                  </span>
                ))}
                {'\n\n'}
              </span>
            ))}
            <span className="k">Question: </span>
            {q.q}
          </>
        ) : (
          <>
            <span className="k">Question: </span>
            {q.q}
          </>
        )}
      </div>
      <div className="live-note" style={{ marginTop: 10 }}>
        {!useRag
          ? 'Without retrieval, the model has only its training data, and the Kestrel K2 isn’t in it. A capable model should say it doesn’t know; a less careful one will produce a plausible answer from what espresso machines usually do (a hallucination that sounds exactly like a real answer).'
          : found
            ? 'The answer sentence (in green) is in the prompt. Now the model’s job is reading comprehension, which language models are very good at. Notice the instruction to say “I don’t know” when the sources don’t help: retrieval doesn’t remove the need for it.'
            : 'The retriever missed: the answer is not in any of these sources. The model will either admit it doesn’t know or, worse, answer confidently from the wrong source. Retrieval failures turn directly into answer failures.'}
      </div>
    </Widget>
  );
}

// ------------------------------------------------------------------ 4. Lab: Tune the retriever

const LAB_KEY = 'nl-rag-lab-v1';
type Entry = { size: number; lambda: number; k: number; solved: boolean; best: number | null };
type LabState = Record<LabBrief['id'], Entry>;
const START: Entry = { size: 1, lambda: 0, k: 1, solved: false, best: null };

function loadLab(): LabState {
  const d: LabState = { everything: { ...START }, tight: { ...START }, nolex: { ...START, lambda: 1 } };
  try {
    const raw = localStorage.getItem(LAB_KEY);
    if (!raw) return d;
    const parsed = JSON.parse(raw) as Partial<LabState>;
    for (const b of LAB_BRIEFS) {
      const v = parsed[b.id];
      if (!v || !(LAB_SIZES as readonly number[]).includes(v.size) || !(LAB_LAMBDAS as readonly number[]).includes(v.lambda) || !(LAB_KS as readonly number[]).includes(v.k)) continue;
      d[b.id] = { size: v.size, lambda: b.fixedLambda ?? v.lambda, k: v.k, solved: !!v.solved, best: typeof v.best === 'number' ? v.best : null };
    }
  } catch {
    // ignore storage errors
  }
  return d;
}

export function RetrieverLab() {
  const [lab, setLab] = useState<LabState>(() => loadLab());
  const [briefIdx, setBriefIdx] = useState(0);
  const [showOpt, setShowOpt] = useState(false);
  const brief = LAB_BRIEFS[briefIdx];
  const e = lab[brief.id];
  const lambda = brief.fixedLambda ?? e.lambda;
  const r = useMemo(() => evaluate(e.size, lambda, e.k), [e.size, lambda, e.k]);
  const pass = r.recall >= brief.minRecall - 1e-9;
  const ex = useMemo(() => (showOpt ? labExhaustive(brief) : null), [showOpt, brief]);
  const saved = useRef(lab);
  saved.current = lab;
  const persist = (next: LabState) => {
    setLab(next);
    try {
      localStorage.setItem(LAB_KEY, JSON.stringify(next));
    } catch {
      // ignore storage errors
    }
  };
  const update = (patch: Partial<Entry>) => persist({ ...lab, [brief.id]: { ...e, ...patch } });
  const words = Math.round(r.promptWords * 10) / 10;

  useEffect(() => {
    if (!pass) return;
    const cur = saved.current[brief.id];
    if (cur.solved && cur.best !== null && cur.best <= words) return;
    persist({ ...saved.current, [brief.id]: { ...cur, solved: true, best: cur.best === null ? words : Math.min(cur.best, words) } });
  }, [pass, brief.id, words]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Widget
      title="Lab: Tune the Retriever"
      wide
      caption={
        <>
          Every word of retrieved text costs money and context space on every request, so the job is to find the answers with as little text as possible. Choose chunk size, hybrid weight and k; recall is measured on all 34 labelled questions. Meet the brief’s recall; optionally, use as few words as you can. Your settings are saved in this browser.
        </>
      }
    >
      <Segmented
        value={briefIdx}
        onChange={(i) => {
          setBriefIdx(i);
          setShowOpt(false);
        }}
        options={LAB_BRIEFS.map((b, i) => ({ value: i, label: `${lab[b.id].solved ? '✓ ' : ''}${b.title.split(' · ')[0]}` }))}
      />
      <div style={{ margin: '12px 0' }}>
        <div style={{ fontWeight: 650, marginBottom: 4 }}>{brief.title}</div>
        <div style={{ fontSize: 14.5, lineHeight: 1.55 }}>{brief.story}</div>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 8 }}>
        <Segmented label="Sentences per chunk" value={e.size} onChange={(size) => update({ size })} options={LAB_SIZES.map((s) => ({ value: s as number, label: String(s) }))} />
        <Segmented label="Chunks retrieved" value={e.k} onChange={(k) => update({ k })} options={LAB_KS.map((s) => ({ value: s as number, label: String(s) }))} />
      </div>
      {brief.fixedLambda === undefined ? (
        <Slider label="hybrid weight λ on semantic" value={e.lambda} min={0} max={1} step={0.25} onChange={(v) => update({ lambda: v })} format={(v) => v.toFixed(2)} color="c-p" />
      ) : (
        <div style={{ fontSize: 13.5, color: 'var(--text-muted)' }}>This brief fixes λ = {brief.fixedLambda} (semantic search only).</div>
      )}
      <div style={{ display: 'grid', gap: 10, marginTop: 12 }}>
        <div className="btn-row">
          <span className={`status ${pass ? 'good' : 'bad'}`}>{pass ? 'Meets the brief' : 'Not yet'}</span>
          {e.solved && <span className="status good">Solved{e.best !== null ? ` · fewest words so far: ${e.best}` : ''}</span>}
        </div>
        <div className="readout">
          <span style={{ color: pass ? 'var(--c-good)' : 'var(--c-bad)' }}>{pass ? '✓' : '✗'}</span> recall {pct(r.recall)} (need ≥ {pct(brief.minRecall)}){'\n'}
          <span className="k">retrieved text per question </span>
          {words} words ({r.chunks} chunks in the index)
        </div>
        <div className="live-note">
          {pass
            ? `It works with ${words} words of retrieved text per question. Optional: can you find them all with fewer?`
            : `${QUESTIONS.length - Math.round(r.recall * QUESTIONS.length)} question${QUESTIONS.length - Math.round(r.recall * QUESTIONS.length) === 1 ? '' : 's'} still miss. Check the table in “Score the retriever” with these settings to see which, and whether they are reworded questions or exact codes.`}
        </div>
        {e.solved && (
          <div>
            <button type="button" className="btn" onClick={() => setShowOpt((s) => !s)}>{showOpt ? 'Hide' : 'Show'} the exhaustive search</button>
            {showOpt && ex && (
              <div className="live-note" style={{ marginTop: 10 }}>
                All {ex.total} settings were tried: {ex.passing} meet this brief. The fewest words is <strong>{ex.bestWords}</strong>, with {ex.best.map((b) => `${b.size}-sentence chunks, λ = ${b.lambda}, k = ${b.k}`).join('; or ')}.
              </div>
            )}
          </div>
        )}
      </div>
    </Widget>
  );
}
