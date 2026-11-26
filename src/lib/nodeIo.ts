// authored-by: Claude Opus 5.5
/**
 * File access for Node-only `*.run.ts` scripts. The app's tsconfig has no Node types, so `node:fs`
 * is imported through a computed specifier (as `src/history.test.ts` does) and typed by hand here.
 */
export type NodeFs = {
  readFileSync: ((p: string | URL, enc: 'utf8') => string) & ((p: string | URL) => Uint8Array);
  writeFileSync: (p: string | URL, data: string | Uint8Array) => void;
};
export async function nodeFs(): Promise<NodeFs> {
  const spec = 'node:' + 'fs';
  return (await import(/* @vite-ignore */ spec)) as NodeFs;
}
/** Command-line arguments after the script name. */
export const argv = (): string[] => ((globalThis as unknown as { process?: { argv: string[] } }).process?.argv ?? []).slice(2);
