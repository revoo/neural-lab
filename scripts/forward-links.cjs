// authored-by: Claude Opus 5.5
/*
 * Report every <LessonLink to="…"> whose target comes LATER in the course than
 * the lesson containing it, with the surrounding sentence. A forward link is
 * fine if the sentence says "later" or "coming up"; it's a problem if it says
 * "remember" or "you saw". This is a report for a human, not a test.
 *
 *   node scripts/forward-links.cjs
 */
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const cur = fs.readFileSync(path.join(root, 'src/curriculum.ts'), 'utf8');
const order = [...cur.matchAll(/\{ id: '([a-z0-9-]+)', title: .*?load:/g)].map((m) => m[1]);
const pos = new Map(order.map((id, i) => [id, i]));

const BACKWARD_WORDS = /\b(remember|recall|you saw|you met|you built|we saw|as before|earlier|already)\b/i;
let count = 0;
let suspicious = 0;
for (const id of order) {
  const file = path.join(root, 'src/lessons', id, 'index.mdx');
  if (!fs.existsSync(file)) continue;
  const src = fs.readFileSync(file, 'utf8').replace(/```[\s\S]*?```/g, '');
  for (const m of src.matchAll(/<LessonLink\s+to="([^"]+)"/g)) {
    const to = m[1];
    if (!pos.has(to) || pos.get(to) <= pos.get(id)) continue;
    // the sentence around the link: back to the previous full stop / line start, forward to the next
    const before = src.slice(0, m.index);
    const start = Math.max(before.lastIndexOf('. '), before.lastIndexOf('\n')) + 1;
    const afterIdx = src.slice(m.index).search(/(\.\s|\n)/);
    const sentence = src.slice(start, afterIdx < 0 ? undefined : m.index + afterIdx + 1).trim();
    const flag = BACKWARD_WORDS.test(sentence);
    if (flag) suspicious++;
    count++;
    console.log(`${flag ? '!!' : '  '} ${id} (#${pos.get(id) + 1}) → ${to} (#${pos.get(to) + 1})\n     ${sentence.slice(0, 300)}`);
  }
}
console.log(`\n${count} forward link(s); ${suspicious} marked !! use backward-looking words ("remember", "you saw"…) and need a human look.`);
