// 可选的浏览器回归：node scripts/check_atlas.cjs <服务地址> [截图目录]（需要 playwright）。
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const [url = 'http://127.0.0.1:8000', out] = process.argv.slice(2);

(async () => {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({viewport: {width: 1536, height: 900}});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('response', r => { if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`); });
    await page.goto(url);
    const ready = async () => {
      await page.waitForFunction(() => S.items.length && cache.size && [...cache.values()].every(e => e.ok || e.failed), null, {timeout: 60000});
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    };
    const shot = async name => { if (out) { fs.mkdirSync(out, {recursive:true}); await page.screenshot({path:path.join(out, name + '.png')}); } };
    await ready();
    assert.equal(await page.inputValue('#reduce'), 'rms');
    assert.equal(await page.inputValue('#scope'), 'role');
    await shot('weight-atlas');
    async function probe() {
      const p = await page.evaluate(() => {
        const i = S.items.find(i => i.tensor?.source.endsWith('.mlp.down_proj.weight')) || S.items[0];
        const box = canvas.getBoundingClientRect();
        return {x:box.left + (i.x + i.w / 2 - S.view.x) * S.view.scale,
          y:box.top + (i.y + i.h / 2 - S.view.y) * S.view.scale};
      });
      await page.mouse.move(p.x, p.y);
      await page.waitForFunction(() => document.getElementById('status').textContent.includes('基准 RMS'));
      return p;
    }
    const pt = await probe();
    console.log('probe:', await page.textContent('#status'));
    await page.mouse.click(pt.x, pt.y);
    if (await page.isVisible('#comparePick')) {
      const source = await page.evaluate(() => S.hover.it.tensor.source);
      await page.click('#comparePick');
      assert.equal(await page.evaluate(() => S.items[0].tensor.source), source);
      await page.click('#modeAtlas');
    }
    await page.mouse.dblclick(pt.x, pt.y);
    await page.click('#fitBtn');
    await page.selectOption('#reduce','maxabs');
    assert.equal(await page.inputValue('#range'), '6');
    await page.selectOption('#scope','tensor');
    if (await page.isEnabled('#modeStack')) {
      await page.click('#modeStack');
      await page.selectOption('#agg','std');
      assert.ok(await page.isDisabled('#reduce'));
      assert.ok(await page.locator('#scaleLegend').evaluate(el => el.classList.contains('unsigned')));
      await page.click('#modeAtlas');
    }
    const recording = await page.evaluate(() => recordings.find(r => r.attention && r.lens)?.id);
    await page.click('[data-view="attn"]');
    if (recording) {
      await page.selectOption('#recSelect', recording);
      await page.locator('#attnBody').waitFor({state:'visible'});
    }
    await page.click('[data-view="lens"]');
    if (recording) {
      await page.waitForFunction(() => document.querySelector('#lensTable td'));
      await page.click('#tokens button:nth-child(2)');
    }
    await page.click('[data-view="weights"]');
    if (recording) {
      assert.ok(await page.isVisible('#thrBox'));
      await ready();
      await page.selectOption('#recSelect', '');
      assert.ok(!await page.isVisible('#thrBox'));
    }
    await page.selectOption('#reduce','rms');
    await page.selectOption('#scope','role');
    await page.setViewportSize({width:390,height:844});
    await ready();
    await page.click('#panelToggle');
    assert.ok(await page.isVisible('aside'));
    await page.click('#panelToggle');
    assert.ok(!await page.isVisible('aside'));
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    assert.deepEqual(errors, []);
    console.log('OK: atlas / probe / zoom / scale / stack / tabs / mobile settings');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
