// authored-by: Claude Opus 5.5
/**
 * Lab: Steer the sentence. The trained tiny GPT continues a prompt greedily (always its most likely
 * next character). The learner may force characters at chosen positions (each forcing costs 1) so that
 * a target appears within the first WINDOW characters. `searchSteer` finds the fewest forcings over a
 * stated space: every plan with 0, 1 or 2 forcings, each forced character one of FORCE_ALPHABET at any
 * position in the window. If none works, the "spell it out" plan (force only the target characters the
 * model doesn't write itself, at the best starting position) gives the best found; when it needs only 3
 * forcings it is a proven minimum, since no plan with fewer works.
 */
import { Decoder, Gpt } from '../../lib/compute/gpt';
import { CHARS, encode } from '../../lib/corpora/grimm';

export const WINDOW = 30;
export const FORCE_ALPHABET = 'abcdefghijklmnopqrstuvwxyz ';
export const STEER_PROMPT = 'There was once';
export const TARGETS = [' king', ' daughter', ' wolf'] as const;

export type Forcing = { pos: number; ch: string };

const argmax = (l: ArrayLike<number>) => {
  let b = 0;
  for (let k = 1; k < l.length; k++) if (l[k] > l[b]) b = k;
  return b;
};

/** Greedy continuation of `prompt` for WINDOW characters, with characters forced at given positions. */
export function steer(model: Gpt, prompt: string, forced: readonly Forcing[], n = WINDOW): string {
  const dec = new Decoder(model);
  let logits = dec.prefill(encode(prompt));
  let out = '';
  for (let pos = 0; pos < n; pos++) {
    const f = forced.find((x) => x.pos === pos);
    const id = f ? CHARS.indexOf(f.ch) : argmax(logits);
    out += CHARS[id];
    if (pos + 1 < n) logits = dec.step(id);
  }
  return out;
}

export type SteerSearch = { cost: number; plan: Forcing[]; exact: boolean; plansTried: number };

export function searchSteer(model: Gpt, prompt: string, target: string, n = WINDOW): SteerSearch {
  const alpha = [...FORCE_ALPHABET].map((c) => CHARS.indexOf(c));
  let tried = 0;
  // the greedy path, keeping the decoder state before each position
  const states: Decoder[] = [];
  const greedy: number[] = [];
  {
    const dec = new Decoder(model);
    let logits = dec.prefill(encode(prompt));
    for (let pos = 0; pos < n; pos++) {
      states.push(dec.clone());
      const id = argmax(logits);
      greedy.push(id);
      (states[pos] as Decoder & { logits?: Float64Array }).logits = logits;
      if (pos + 1 < n) logits = dec.step(id);
    }
  }
  const text = (ids: number[]) => ids.map((i) => CHARS[i]).join('');
  tried++;
  if (text(greedy).includes(target)) return { cost: 0, plan: [], exact: true, plansTried: tried };
  /** Roll out greedily from a decoder positioned before `pos`, having fixed ids[0..pos). */
  const finish = (dec: Decoder, ids: number[], pos: number, first: number) => {
    const out = ids.slice(0, pos);
    let id = first;
    let logits: Float64Array;
    for (let p = pos; p < n; p++) {
      out.push(id);
      if (p + 1 < n) {
        logits = dec.step(id);
        id = argmax(logits);
      }
    }
    return out;
  };
  // one forcing
  for (let p1 = 0; p1 < n; p1++)
    for (const c1 of alpha) {
      if (c1 === greedy[p1]) continue;
      tried++;
      const ids = finish(states[p1].clone(), greedy, p1, c1);
      if (text(ids).includes(target)) return { cost: 1, plan: [{ pos: p1, ch: CHARS[c1] }], exact: true, plansTried: tried };
    }
  // two forcings: after the first, walk greedily, branching at every later position
  for (let p1 = 0; p1 < n; p1++)
    for (const c1 of alpha) {
      if (c1 === greedy[p1]) continue;
      const dec = states[p1].clone();
      const ids = greedy.slice(0, p1);
      ids.push(c1);
      let logits = p1 + 1 < n ? dec.step(c1) : new Float64Array(0);
      for (let p2 = p1 + 1; p2 < n; p2++) {
        const nat = argmax(logits);
        for (const c2 of alpha) {
          if (c2 === nat) continue;
          tried++;
          const out = finish(dec.clone(), ids, p2, c2);
          if (text(out).includes(target)) return { cost: 2, plan: [{ pos: p1, ch: CHARS[c1] }, { pos: p2, ch: CHARS[c2] }], exact: true, plansTried: tried };
        }
        ids.push(nat);
        if (p2 + 1 < n) logits = dec.step(nat);
      }
    }
  // spell it out: start the target at position s, force only characters the model wouldn't write
  let best: Forcing[] | null = null;
  const T = [...target].map((c) => CHARS.indexOf(c));
  for (let s = 0; s + T.length <= n; s++) {
    const plan: Forcing[] = [];
    const dec = states[s].clone();
    let logits = (states[s] as Decoder & { logits?: Float64Array }).logits!;
    for (let k = 0; k < T.length; k++) {
      if (argmax(logits) !== T[k]) plan.push({ pos: s + k, ch: CHARS[T[k]] });
      if (s + k + 1 < n) logits = dec.step(T[k]);
    }
    tried++;
    if (!best || plan.length < best.length) best = plan;
  }
  // every plan with 0, 1 or 2 forcings failed, so a 3-forcing plan is a proven minimum
  return { cost: best!.length, plan: best!, exact: best!.length <= 3, plansTried: tried };
}
