// authored-by: Claude Opus 5.5
// V2-WP3: run the WebGPU kernels in a real Chromium and compare them with the CPU reference tier.
// Usage (dev server running): node scripts/check-webgpu.cjs [baseUrl]
// Headless Chromium exposes WebGPU only with --enable-unsafe-webgpu (adapter: SwiftShader, on the CPU).
const { chromium } = require('playwright');
const base = process.argv[2] || 'http://localhost:5173';
(async () => {
  const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu'] });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  await page.goto(`${base}/#/glossary`, { waitUntil: 'networkidle' });
  const res = await page.evaluate(async () => (await import('/src/lib/compute/parity.ts')).runParity());
  console.log(`adapter: ${res.adapter}`);
  let bad = 0;
  for (const c of res.checks) {
    if (!c.ok) bad++;
    console.log(`${c.ok ? '✓' : '✗'} ${c.name}: worst ${c.worst.toExponential(2)} (tolerance ${c.tol})`);
  }
  if (errors.length) console.log('console errors:\n' + errors.join('\n'));
  await browser.close();
  process.exit(bad || errors.length ? 1 : 0);
})();
