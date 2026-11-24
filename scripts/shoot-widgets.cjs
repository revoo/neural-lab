// Screenshots every widget on one lesson page, in light and dark mode, for visual checks.
// Usage (dev server running): node scripts/shoot-widgets.cjs <lessonId> [outDir] [--width=1300]
// Writes <outDir>/<lessonId>-<theme>-<n>.png (default outDir: scratch/shots, which is git-ignored).
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const flags = Object.fromEntries(
  process.argv
    .slice(2)
    .filter((a) => a.startsWith('--'))
    .map((a) => a.slice(2).split('=')),
);
const lesson = args[0];
const outDir = args[1] || path.join(__dirname, '../scratch/shots');
const width = Number(flags.width || 1300);
const base = flags.base || 'http://localhost:5173';
if (!lesson) {
  console.error('usage: node scripts/shoot-widgets.cjs <lessonId> [outDir]');
  process.exit(1);
}
fs.mkdirSync(outDir, { recursive: true });

(async () => {
  const browser = await chromium.launch();
  for (const theme of ['light', 'dark']) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } });
    await page.addInitScript((t) => localStorage.setItem('nl-theme', t), theme);
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
    await page.goto(`${base}/#/learn/${lesson}`, { waitUntil: 'networkidle' });
    await page.evaluate((t) => (document.documentElement.dataset.theme = t), theme);
    await page.waitForTimeout(400);
    const widgets = await page.$$('figure.widget');
    for (let i = 0; i < widgets.length; i++) {
      await widgets[i].scrollIntoViewIfNeeded();
      await page.waitForTimeout(150);
      const file = path.join(outDir, `${lesson}-${theme}-${i + 1}.png`);
      await widgets[i].screenshot({ path: file });
      console.log(file);
    }
    if (errors.length) console.log(`[${theme}] errors:`, errors.join(' | '));
    await page.close();
  }
  await browser.close();
})();
