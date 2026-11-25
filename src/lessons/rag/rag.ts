// authored-by: Claude Opus 5.5
/**
 * Retrieval, computed. Two ways to turn text into vectors, both small enough to inspect:
 *  - lexical: TF-IDF over words (a word that appears in few chunks counts for more), cosine similarity
 *  - "semantic": a hand-made dictionary maps words to about 40 concepts ("refund", "money" and
 *    "paid" all map to MONEY), and a chunk's vector counts its concepts. It is a stand-in for a learned
 *    embedding model with the same structure: different words, same meaning, nearby vectors.
 * A hybrid score mixes the two. Retrieval is scored on labelled questions: did a chunk containing the
 * answer sentence come back in the top k?
 */
import { DOCS, QUESTIONS, type Question } from './corpus';

const STOP = new Set(
  'a an the and or but if of to in on at for from by with without as is are was were be been being it its this that these those i you your my me we our us do does did can could should would will what which who how when where why there their them they he she his her not no so than then too very just about into over under up down out more most less any all every each some such only own same also once while after before during again further here off'.split(
    ' ',
  ),
);

/** Lowercase words, numbers and codes like "e2"; a plural "s" is dropped so "shots" matches "shot". */
export function tokens(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .split(' ')
    .filter((w) => w && !STOP.has(w))
    .map((w) => (w.length > 3 && w.endsWith('s') && !w.endsWith('ss') ? w.slice(0, -1) : w));
}

/** The hand-made concept dictionary (after `tokens`' plural stripping). */
const CONCEPTS: Record<string, string[]> = {
  WARRANTY: ['warranty', 'guarantee', 'covered', 'claim'],
  REPAIR: ['repair', 'fix', 'fault', 'broken', 'break', 'breaks', 'service', 'damage'],
  FREE: ['free'],
  LIMESCALE: ['limescale', 'scale', 'chalk', 'descale', 'descaling', 'build', 'hard'],
  RETURN: ['return', 'returned', 'send', 'back'],
  MONEY: ['refund', 'money', 'paid', 'cost', 'price', 'euro', 'fee', 'cheap', 'expensive'],
  USED: ['used', 'unused', 'opened', 'unopened'],
  SPEED: ['soon', 'quickly', 'fast', 'faster', 'quick', 'slow', 'slower', 'forever', 'long', 'second', 'minute', 'hour', 'day', 'working'],
  DELIVERY: ['delivery', 'deliver', 'shipping', 'ship', 'parcel', 'courier', 'express', 'tracking', 'order', 'warehouse'],
  NEXTDAY: ['next', 'express', 'overnight', 'tomorrow'],
  COUNTRY: ['switzerland', 'norway', 'eu', 'united', 'kingdom', 'country', 'abroad'],
  VINEGAR: ['vinegar', 'liquid', 'sachet', 'acid'],
  WASH: ['dishwasher', 'wash', 'clean', 'cleaning', 'wipe', 'rinse', 'backflush', 'purge'],
  WAND: ['wand'],
  COFFEE: ['coffee', 'espresso', 'shot', 'brew', 'brewing', 'double'],
  AMOUNT: ['gram', 'dose', 'amount', 'quantity', 'sachet'],
  GRIND: ['grind', 'ground', 'grinder', 'finer', 'coarser', 'fine', 'coarse', 'pre'],
  TASTE: ['sour', 'bitter', 'weak', 'watery', 'taste'],
  MILK: ['milk', 'oat', 'plant', 'froth', 'froths', 'foam', 'jug', 'barista'],
  HEAT: ['hot', 'heat', 'heats', 'temperature', 'degree', 'warm', 'overheating', 'cool', 'thermoblock'],
  ERROR: ['error', 'code', 'display', 'show', 'light', 'flashe'],
  PUMP: ['pump', 'bar'],
  WATER: ['water', 'tank', 'reservoir', 'litre'],
  LEAK: ['leak', 'drip', 'drips', 'puddle'],
  GASKET: ['gasket', 'seal', 'seals'],
  POWER: ['electricity', 'power', 'watt', 'energy', 'standby', 'idle', 'plugged', 'socket', 'extension', 'cord', 'lead', 'plug'],
  WEIGHT: ['heavy', 'weigh', 'weighs', 'weight', 'kilogram'],
  SIZE: ['big', 'size', 'hold', 'holds', 'wide', 'tall', 'deep', 'centimetre', 'dimension'],
  APP: ['app', 'bluetooth', 'phone', 'io', 'android'],
  UPDATE: ['firmware', 'update', 'install', 'installed'],
  NEED: ['need', 'needed', 'required'],
  BASKET: ['basket', 'portafilter', 'pressurised', 'wall'],
  INVOICE: ['invoice', 'receipt', 'download', 'pdf'],
  CASING: ['casing', 'cover', 'open', 'inside'],
  SELF: ['yourself', 'myself', 'home'],
  SAFETY: ['burn', 'children', 'unplug', 'safe', 'safety'],
  START: ['start', 'begin', 'button', 'press'],
  NOTHING: ['nothing', 'empty'],
};

const WORD_TO_CONCEPT = new Map<string, number>();
export const CONCEPT_NAMES = Object.keys(CONCEPTS);
CONCEPT_NAMES.forEach((c, i) => CONCEPTS[c].forEach((w) => WORD_TO_CONCEPT.set(w, i)));
export const conceptOf = (w: string) => {
  const i = WORD_TO_CONCEPT.get(w);
  return i === undefined ? null : CONCEPT_NAMES[i];
};

// ------------------------------------------------------------------ chunks

export type Chunk = { id: string; doc: string; title: string; first: number; sentences: string[]; text: string; words: number };

/** Split every document into chunks of `size` consecutive sentences (the last chunk may be shorter). */
export function chunkCorpus(size: number): Chunk[] {
  const out: Chunk[] = [];
  for (const d of DOCS)
    for (let i = 0; i < d.sentences.length; i += size) {
      const s = d.sentences.slice(i, i + size);
      const text = `${d.title}. ${s.join(' ')}`;
      out.push({ id: `${d.id}#${i}`, doc: d.id, title: d.title, first: i, sentences: s, text, words: text.split(/\s+/).length });
    }
  return out;
}

export const containsAnswer = (c: Chunk, q: Question) => c.doc === q.doc && q.sent >= c.first && q.sent < c.first + c.sentences.length;

// ------------------------------------------------------------------ vectors

type Sparse = Map<string, number>;

function norm(v: Sparse) {
  let s = 0;
  v.forEach((x) => (s += x * x));
  return Math.sqrt(s);
}

export function cosine(a: Sparse, b: Sparse) {
  const na = norm(a);
  const nb = norm(b);
  if (!na || !nb) return 0;
  let s = 0;
  a.forEach((x, k) => {
    const y = b.get(k);
    if (y) s += x * y;
  });
  return s / (na * nb);
}

/** A searchable index over a set of chunks. */
export function buildIndex(chunks: Chunk[]) {
  const df = new Map<string, number>();
  const toks = chunks.map((c) => tokens(c.text));
  for (const t of toks) new Set(t).forEach((w) => df.set(w, (df.get(w) ?? 0) + 1));
  const N = chunks.length;
  const idf = (w: string) => Math.log((N + 1) / ((df.get(w) ?? 0) + 1)) + 1;
  const lexVec = (ts: string[]): Sparse => {
    const v: Sparse = new Map();
    for (const w of ts) v.set(w, (v.get(w) ?? 0) + 1);
    v.forEach((tf, w) => v.set(w, tf * idf(w)));
    return v;
  };
  const semVec = (ts: string[]): Sparse => {
    const v: Sparse = new Map();
    for (const w of ts) {
      const c = conceptOf(w);
      if (c) v.set(c, (v.get(c) ?? 0) + 1);
    }
    return v;
  };
  const lex = toks.map(lexVec);
  const sem = toks.map(semVec);
  return {
    chunks,
    idf,
    /** Scores for every chunk: lexical, semantic and the hybrid mix (λ = weight on semantic). */
    search(query: string, lambda: number) {
      const qt = tokens(query);
      const ql = lexVec(qt);
      const qs = semVec(qt);
      return chunks
        .map((c, i) => {
          const l = cosine(ql, lex[i]);
          const s = cosine(qs, sem[i]);
          return { chunk: c, lex: l, sem: s, score: (1 - lambda) * l + lambda * s };
        })
        .sort((a, b) => b.score - a.score || a.chunk.id.localeCompare(b.chunk.id));
    },
    queryTokens: (q: string) => tokens(q).map((w) => ({ word: w, idf: idf(w), concept: conceptOf(w) })),
  };
}

// ------------------------------------------------------------------ evaluation

/** Rank (1-based) of the first chunk containing the answer; Infinity if no chunk scored above 0. */
export function answerRank(index: ReturnType<typeof buildIndex>, q: Question, lambda: number) {
  const res = index.search(q.q, lambda);
  const i = res.findIndex((r) => containsAnswer(r.chunk, q));
  return i >= 0 && res[i].score > 0 ? i + 1 : Infinity;
}

export function evaluate(chunkSize: number, lambda: number, k: number, qs: Question[] = QUESTIONS) {
  const index = buildIndex(chunkCorpus(chunkSize));
  const ranks = qs.map((q) => answerRank(index, q, lambda));
  const hits = ranks.filter((r) => r <= k).length;
  const avgWords = index.chunks.reduce((s, c) => s + c.words, 0) / index.chunks.length;
  return {
    ranks,
    recall: hits / qs.length,
    mrr: ranks.reduce((s, r) => s + (r === Infinity ? 0 : 1 / r), 0) / qs.length,
    /** words of retrieved text put into the prompt (average chunk length × k) */
    promptWords: avgWords * k,
    chunks: index.chunks.length,
  };
}
