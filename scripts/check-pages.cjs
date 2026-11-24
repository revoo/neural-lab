// Loads every lesson (plus the main pages) in headless Chromium against the running dev server
// and reports console errors and horizontal overflow at phone width.
// Usage: npm run dev   (in another terminal), then: node scripts/check-pages.cjs [baseUrl]
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const base = process.argv[2] || 'http://localhost:5173';
const curriculum = fs.readFileSync(path.join(__dirname, '../src/curriculum.ts'), 'utf8');
const lessons = [...curriculum.matchAll(/id: '([^']+)', title:/g)].map((m) => m[1]);
const routes = ['', 'glossary', 'review', ...lessons.map((l) => `learn/${l}`)];

(async () => {
  const browser = await chromium.launch();
  let failures = 0;
  for (const width of [1300, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    for (const r of routes) {
      const errors = [];
      const onErr = (e) => errors.push(e.message);
      const onConsole = (m) => m.type() === 'error' && errors.push(m.text());
      page.on('pageerror', onErr);
      page.on('console', onConsole);
      await page.goto(`${base}/#/${r}`, { waitUntil: 'networkidle' });
      await page.waitForTimeout(300);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
      page.off('pageerror', onErr);
      page.off('console', onConsole);
      if (errors.length || overflow) {
        failures++;
        console.log(`✕ [${width}px] /${r}`, overflow ? 'horizontal overflow' : '', errors.join(' | '));
      } else console.log(`✓ [${width}px] /${r}`);
    }
    await page.close();
  }
  await browser.close();
  process.exit(failures ? 1 : 0);
})();
