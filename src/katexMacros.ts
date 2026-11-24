/**
 * Colour macros shared by build-time (rehype-katex) and runtime (Tex) maths, so a symbol can be
 * colour-matched to its plain-English name and to the same quantity in a visualisation.
 *   \ca{…} blue   \cb{…} orange   \cc{…} green   \cg{…} magenta (gradients)   \cp{…} purple (accent)   \cv{…} red
 */
export const KATEX_MACROS: Record<string, string> = {
  '\\ca': '\\htmlStyle{color: var(--c-a)}{#1}',
  '\\cb': '\\htmlStyle{color: var(--c-b)}{#1}',
  '\\cc': '\\htmlStyle{color: var(--c-c)}{#1}',
  '\\cg': '\\htmlStyle{color: var(--c-grad)}{#1}',
  '\\cp': '\\htmlStyle{color: var(--accent-text)}{#1}',
  '\\cv': '\\htmlStyle{color: var(--c-bad)}{#1}',
};
