// Drive one widget through a sequence of actions and screenshot it along the way, in each colour theme (light, dark, matrix).
// Usage (dev server running):
//   node scripts/drive-widget.cjs <lessonId> <widgetIndex> '<actions JSON>' [--width=1300] [--themes=light,dark,matrix] [--title=Lab]
//   (--webgpu launches Chromium with WebGPU enabled (SwiftShader adapter); without it widgets fall back to the CPU tier)
//   (--open-details opens every collapsed <details> first, for widgets inside a Deeper)
//   (--title=<text> selects the first widget whose title contains <text>; widgetIndex is then ignored, pass 0)
// Actions (array, run in order inside the widget's <figure>):
//   {"click": "button:has-text('Next')", "times": 3}   click a Playwright selector (scoped to the widget)
//   {"clickAt": [x, y], "on": "svg >> nth=0"}            click at element-relative coordinates
//   {"drag": [[x1, y1], [x2, y2]], "on": "svg >> nth=0"} drag between element-relative coordinates
//   {"wait": 500}                                        wait (ms)
//   {"shot": "name"}                                     screenshot -> scratch/shots/<lessonId>-<name>-<theme>.png
//   {"eval": "js expression"}                            evaluate in the page and print the result
// Console errors are printed at the end.
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
const [lesson, idx, json] = args;
const actions = JSON.parse(json || '[]');
const width = Number(flags.width || 1300);
const themes = (flags.themes || 'light,dark,matrix').split(',');
const base = flags.base || 'http://localhost:5173';
const outDir = path.join(__dirname, '../scratch/shots');
fs.mkdirSync(outDir, { recursive: true });

(async () => {
  const browser = await chromium.launch('webgpu' in flags ? { args: ['--enable-unsafe-webgpu'] } : {});
  for (const theme of themes) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } });
    await page.addInitScript((t) => localStorage.setItem('nl-theme', t), theme);
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
    await page.goto(`${base}/#/learn/${lesson}`, { waitUntil: 'networkidle' });
    await page.evaluate((t) => (document.documentElement.dataset.theme = t), theme);
    // --open-details opens every <Deeper>/<MathExhibit> first, for widgets placed inside one
    if ('open-details' in flags) await page.evaluate(() => document.querySelectorAll('details').forEach((d) => (d.open = true)));
    await page.waitForTimeout(300);
    // --title=<text> picks the first widget whose title contains <text>, instead of the index.
    const widget = flags.title
      ? page.locator('figure.widget', { has: page.locator('.widget-title', { hasText: flags.title }) }).first()
      : page.locator('figure.widget').nth(Number(idx));
    await widget.scrollIntoViewIfNeeded();
    for (const a of actions) {
      if (a.click) for (let i = 0; i < (a.times || 1); i++) await widget.locator(a.click).first().click();
      if (a.clickAt) {
        const box = await widget.locator(a.on || 'svg').first().boundingBox();
        await page.mouse.click(box.x + a.clickAt[0], box.y + a.clickAt[1]);
      }
      if (a.drag) {
        const box = await widget.locator(a.on || 'svg').first().boundingBox();
        await page.mouse.move(box.x + a.drag[0][0], box.y + a.drag[0][1]);
        await page.mouse.down();
        await page.mouse.move(box.x + a.drag[1][0], box.y + a.drag[1][1], { steps: 8 });
        await page.mouse.up();
      }
      if (a.wait) await page.waitForTimeout(a.wait);
      if (a.eval) console.log(theme, '→', await page.evaluate(a.eval));
      if (a.shot) {
        await page.waitForTimeout(120);
        const file = path.join(outDir, `${lesson}-${a.shot}-${theme}.png`);
        await widget.screenshot({ path: file });
        console.log(file);
      }
    }
    if (errors.length) console.log(`[${theme}] errors:`, errors.join(' | '));
    await page.close();
  }
  await browser.close();
})();
