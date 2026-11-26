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
  const res = await page.evaluate(async () => {
    const r = await (await import('/src/lib/compute/parity.ts')).runParity();
    // V2-WP3: the GAN lab's whole-duel kernel (src/lessons/gans/ganGpu.ts)
    const { getGpu } = await import('/src/lib/compute/webgpuBackend.ts');
    const gpu = await getGpu();
    if (gpu) r.checks.push(...(await (await import('/src/lessons/gans/ganGpu.ts')).ganGpuParity(gpu.device)));
    // V1 capstone A2: the tiny GPT on the GPU vs the CPU tier, and the CPU tier's bit-exactness vs Node
    const gp = await import('/src/lib/compute/gptParity.ts');
    if (gpu) r.checks.push(...(await gp.gptParity(gpu.device)));
    const hash = gp.cpuDeterminismHash();
    const pinned = (await import('/src/lib/compute/gptPinned.ts')).CPU_DETERMINISM_HASH;
    r.checks.push({ name: `GPT CPU tier in the browser is bit-identical to Node (hash ${hash}, pinned ${pinned})`, worst: hash === pinned ? 0 : 1, tol: 0, ok: hash === pinned });
    // the same reference run through the CPU Web Worker the lesson uses
    const pin = await import('/src/lib/compute/gptPinned.ts');
    const { createGptTrainer } = await import('/src/lib/compute/gptTrainer.ts');
    const s = pin.determinismSetup();
    const tr = await createGptTrainer(s.cfg, 'cpu');
    await tr.setParams(s.init);
    await tr.train(s.batches, s.K, s.B, s.T, s.lr, s.opts);
    const wh = pin.paramHash(await tr.readParams());
    r.checks.push({ name: `GPT CPU tier in a Web Worker (${tr.where}) is bit-identical to Node (hash ${wh})`, worst: wh === pinned ? 0 : 1, tol: 0, ok: wh === pinned && tr.where === 'worker' });
    tr.destroy();
    return r;
  });
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
