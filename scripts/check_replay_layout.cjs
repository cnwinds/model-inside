// 需要一条带注意力和逐层预测的录制。可选第二个参数保存截图。
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const [url = 'http://127.0.0.1:8000', out] = process.argv.slice(2);

(async () => {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({viewport:{width:1920,height:911}});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(url);
    await page.waitForFunction(() => document.querySelector('#recSelect').options.length > 1);
    const id = await page.evaluate(() => recordings.find(r => r.attention && r.lens)?.id);
    assert.ok(id, '需要含 attention 和 lens 的录制');
    await page.selectOption('#recSelect', id);
    const frame = () => page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
    async function bounds() {
      await frame();
      const bad = await page.evaluate(() => {
        const nodes = document.querySelectorAll('html,body,main,#stage,.pane,.view-body,.attn-grid,.attn-left,.attn-right,.lens-grid,.lens-sheet,.lens-table-wrap,.lens-side,.lens-rank,#player,.controls');
        return [...nodes].filter(n => n.clientHeight && (n.scrollHeight > n.clientHeight+1 || n.scrollWidth > n.clientWidth+1)).map(n => [n.id || n.className || n.tagName, n.clientWidth, n.scrollWidth, n.clientHeight, n.scrollHeight]);
      });
      assert.deepEqual(bad, [], '主工作区不应溢出或裁掉内容');
      assert.ok(await page.locator('#playBtn').isVisible());
    }
    async function shot(name) {
      if (out) { fs.mkdirSync(out, {recursive:true}); await page.screenshot({path:path.join(out,name+'.png')}); }
    }
    for (const [w,h] of [[1920,911],[1920,657],[1366,657],[1024,768],[390,844],[320,568],[844,390]]) {
      await page.setViewportSize({width:w,height:h});
      await page.click('[data-view="lens"]');
      const compact = await page.isVisible('#lensBody .view-switch');
      if (compact) await page.click('#lensBody [data-panel="table"]');
      await page.locator('#lensTable td').first().waitFor();
      await bounds();
      if (w === 1920 || w === 390) await shot(`lens-${w}x${h}`);
      // 两个轴都能走到边界；翻页后，当前行列交点也必须可见。
      for (const key of ['nextLayers','nextSteps','prevLayers','prevSteps']) {
        let clicks = 0;
        while (await page.isEnabled('#'+key)) {
          await page.click('#'+key); await frame();
          assert.ok(++clicks < 300, '翻页没有前进');
          assert.ok(await page.isVisible('#lensTable tr.sel td.col'));
        }
      }
      if (compact) {
        for (const name of ['rank','picks']) {
          await page.click(`#lensBody [data-panel="${name}"]`); await bounds();
          assert.ok(await page.isVisible(name === 'rank' ? '#lensChart' : '#lensTop .cand:last-child'));
        }
      }
      await page.click('[data-view="attn"]');
      await page.waitForFunction(() => Views.state().A.layers.every(Boolean));
      if (await page.isVisible('#attnBody .view-switch')) {
        for (const name of ['arcs','matrix','heads']) { await page.click(`#attnBody [data-panel="${name}"]`); await bounds(); }
      }
      await bounds();
      if (w === 1920 || w === 390) await shot(`attention-${w}x${h}`);
      const corner = await page.evaluate(() => {
        const a = Views.state().A, {lab,top,cs} = a.ovGeo, c = document.querySelector('#attnOverview'), r = c.getBoundingClientRect();
        return {x:r.left+c.clientLeft+lab+(a.H+.5)*cs, y:r.top+c.clientTop+top+(a.L-.5)*cs};
      });
      await page.mouse.click(corner.x,corner.y); await frame();
      assert.ok(await page.evaluate(() => {const a=Views.state().A;return a.layer===a.L-1 && a.head===a.H-1;}));
      console.log(`${w}×${h}: no overflow, table navigation and overview hit test OK`);
    }
    assert.deepEqual(errors, []);
  } finally { await browser.close(); }
})().catch(e => {console.error(e);process.exitCode=1;});
