// authored-by: Claude Opus 5.5
/**
 * Builds `grimm.txt`, the tiny GPT's corpus, from Project Gutenberg eBook #2591, "Grimms' Fairy
 * Tales" (Jacob and Wilhelm Grimm, translated by Edgar Taylor and Marian Edwardes; public domain).
 *
 *   curl -L -o /tmp/pg2591.txt https://gutenberg.pglaf.org/cache/epub/2591/pg2591.txt
 *   node --experimental-strip-types src/lib/corpora/buildGrimm.run.ts /tmp/pg2591.txt
 *
 * Steps: keep only the tales (from the first tale's heading to the "*****" before the closing
 * biographical note, so no Project Gutenberg header, footer or licence text remains); unwrap the
 * hard line breaks so each paragraph is one line; map curly quotes and dashes to ASCII, drop the
 * handful of other rare characters; collapse runs of spaces.
 */
import { readFileSync, writeFileSync } from 'node:fs';

const raw = readFileSync(process.argv[2], 'utf8').replace(/\r/g, '');
const start = raw.indexOf('THE BROTHERS GRIMM FAIRY TALES');
const bodyStart = raw.indexOf('THE GOLDEN BIRD', start);
const bodyEnd = raw.indexOf('\n*****', bodyStart);
if (start < 0 || bodyStart < 0 || bodyEnd < 0) throw new Error('markers not found');
let body = raw.slice(bodyStart, bodyEnd);

const MAP: Record<string, string> = { '‘': "'", '’': "'", '“': '"', '”': '"', '—': '-', 'ä': 'a', 'é': 'e' };
body = [...body].map((c) => MAP[c] ?? c).join('');

const paragraphs = body
  .split(/\n\s*\n/)
  .map((p) => p.split('\n').map((l) => l.trim()).filter(Boolean).join(' '))
  .map((p) => p.replace(/[^A-Za-z0-9 .,;:!?'"()\-]/g, '').replace(/ {2,}/g, ' ').trim())
  .filter(Boolean);
const text = paragraphs.join('\n') + '\n';
writeFileSync(new URL('./grimm.txt', import.meta.url), text);
const chars = [...new Set(text)].sort();
console.log(`${text.length} characters, ${paragraphs.length} paragraphs, vocabulary ${chars.length}: ${JSON.stringify(chars.join(''))}`);
