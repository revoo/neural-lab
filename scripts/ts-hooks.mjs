// Lets plain Node run TypeScript exploration scripts that import project modules with extensionless
// specifiers (as Vite allows). Usage:
//   node --experimental-strip-types --import ./scripts/ts-hooks.mjs scratch/explore.ts
// Use it for quick numeric checks of claims made in lesson prose.
import { register } from 'node:module';

register(
  'data:text/javascript,' +
    encodeURIComponent(`
export async function resolve(specifier, context, next) {
  try {
    return await next(specifier, context);
  } catch (err) {
    if ((specifier.startsWith('.') || specifier.startsWith('/')) && !/\\.[cm]?[jt]sx?$/.test(specifier)) {
      for (const ext of ['.ts', '.tsx', '/index.ts']) {
        try { return await next(specifier + ext, context); } catch {}
      }
    }
    throw err;
  }
}`),
  import.meta.url,
);
