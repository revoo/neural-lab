import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import mdx from '@mdx-js/rollup';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import remarkSmartypants from 'remark-smartypants';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';

export default defineConfig({
  base: './',
  plugins: [
    {
      enforce: 'pre',
      ...mdx({
        providerImportSource: '@mdx-js/react',
        remarkPlugins: [remarkMath, remarkGfm, [remarkSmartypants, { dashes: 'oldschool' }]],
        rehypePlugins: [[rehypeKatex, { strict: false }], rehypeHighlight],
      }),
    },
    react({ include: /\.(mdx|tsx?)$/ }),
  ],
  build: { chunkSizeWarningLimit: 900 },
  test: {
    include: ['src/**/*.test.ts'],
  },
});
