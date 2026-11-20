// 两个回放视图：注意力、逐层预测。
// 依赖 app.js 里的全局 S（状态）和 setStep()；app.js 在换录制、换步、切标签页时调用 Views.*。
const Views = (() => {
  'use strict';
  const el = (id) => document.getElementById(id);
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const ACCENT = [70, 255, 110];
  const BG = [14, 17, 24];
  const MONO = '12px Consolas, "Cascadia Mono", monospace';
  // 服务端把注意力存成 round(255·√权重)，这里按平方还原
  const LUT = Float32Array.from({ length: 256 }, (_, q) => (q / 255) ** 2);

  const vis = (t) => (String(t).replace(/\n/g, '↵').replace(/\t/g, '⇥') || '∅');
  const pct = (p) => (p >= 0.1 ? `${Math.round(p * 100)}%` : p >= 0.01 ? `${(p * 100).toFixed(1)}%` : p >= 1e-4 ? `${(p * 100).toFixed(2)}%` : '<0.01%');

  // 给元素填内容：字符串是普通文字，{b: '...'} 是强调。一律走 textContent，token 里有什么字符都不会被当成 HTML。
  function say(node, ...parts) {
    node.replaceChildren(...parts.map((p) => {
      if (typeof p === 'string') return document.createTextNode(p);
      const b = document.createElement('b');
      b.textContent = p.b;
      return b;
    }));
  }

  // 按设备像素比准备画布，返回 CSS 像素坐标系下的绘图环境
  function setup(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return null;
    const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
    if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph; }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function clip(ctx, text, maxW) {
    if (ctx.measureText(text).width <= maxW) return text;
    let s = text;
    while (s.length > 1 && ctx.measureText(s + '…').width > maxW) s = s.slice(0, -1);
    return s + '…';
  }

  function rrect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, r); else ctx.rect(x, y, w, h);
  }

  function emptyMessage(kind) {
    if (!S.rec) return '还没有选中录制。<br>在左侧输入提示词并点击「运行并记录」，或在「回放」里选一条已有的录制。';
    if (!S.rec[kind]) return '这条录制里没有这项数据（可能是在加入这个视图之前录的）。<br>请重新点击「运行并记录」。';
    return null;
  }

  function scheduler(fn) {
    let queued = false;
    return () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; fn(); });
    };
  }

  // ======================================================================
  // 注意力
  // ======================================================================

  const A = { id: null, layers: [], L: 0, H: 0, n: 0, layer: -1, head: -1, token: 0, geo: null, mat: null, ov: null };

  const shade = (w) => {
    const t = Math.sqrt(w);
    return [BG[0] + (ACCENT[0] - BG[0]) * t, BG[1] + (ACCENT[1] - BG[1]) * t, BG[2] + (ACCENT[2] - BG[2]) * t];
  };

  // raw 是一层的数据（头, 查询位置, 被看位置）；head < 0 表示各头平均
  function weight(raw, head, i, j) {
    const n = A.n;
    if (head >= 0) return LUT[raw[(head * n + i) * n + j]];
    let s = 0;
    for (let k = 0; k < A.H; k++) s += LUT[raw[(k * n + i) * n + j]];
    return s / A.H;
  }

  function queryRow(raw, head, i) {
    const row = new Float32Array(A.n);
    for (let j = 0; j <= i; j++) row[j] = weight(raw, head, i, j);
    return row;
  }

  function startAttention(meta) {
    const info = meta && meta.attention;
    const token = ++A.token;
    A.id = info ? meta.id : null;
    A.L = info ? info.layers : 0;
    A.H = info ? info.heads : 0;
    A.n = meta ? meta.tokens.length : 0;
    A.layers = Array(A.L).fill(null);
    A.mat = A.ov = null;
    if (!info) return;

    A.layer = A.layer < 0 || A.layer >= A.L ? Math.floor(A.L / 2) : A.layer;
    if (A.head >= A.H) A.head = -1;
    el('attnLayer').max = A.L - 1;
    el('attnHead').replaceChildren(new Option(`全部 ${A.H} 个头的平均`, -1), ...Array.from({ length: A.H }, (_, h) => new Option(`头 ${h}`, h)));
    syncAttnControls();

    // 选中的那一层先加载，其余的并行补上
    const order = [A.layer, ...Array.from({ length: A.L }, (_, l) => l).filter((l) => l !== A.layer)];
    let next = 0;
    const worker = async () => {
      while (next < order.length && token === A.token) {
        const l = order[next++];
        try {
          const res = await fetch(`/api/attention/${meta.id}/${l}`);
          if (!res.ok) throw new Error(res.statusText);
          const data = new Uint8Array(await res.arrayBuffer());
          if (token !== A.token) return;
          A.layers[l] = data;
        } catch (err) {
          if (token === A.token) A.layers[l] = false;
          console.warn('注意力加载失败', l, err);
        }
        renderAttnSoon();
      }
    };
    for (let i = 0; i < 4; i++) worker();
  }

  function syncAttnControls() {
    el('attnLayer').value = A.layer;
    el('attnHead').value = A.head;
    el('attnLayerLabel').textContent = `第 ${A.layer} 层（共 ${A.L} 层）`;
  }

  function drawArcs(raw) {
    const g = setup(el('attnArcs'));
    if (!g) return;
    const { ctx, w, h } = g;
    ctx.font = MONO;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';
    if (!raw) {
      ctx.fillStyle = '#7f899c';
      ctx.fillText(raw === false ? '这一层加载失败' : '加载中…', w / 2, h / 2);
      return;
    }
    const n = A.n, q = clamp(S.step, 0, n - 1), tokens = S.rec.tokens;
    const slot = Math.min(58, Math.max(26, (w - 12) / 4)), count = clamp(Math.floor((w - 8) / slot), 1, n);
    const a = clamp(q - count + 1 + Math.min(3, count - 1), 0, n - count);   // 窗口末尾多留几个「未来的词」，对比出它们看不到
    const x0 = (w - count * slot) / 2, ty = h - 44, th = 28;
    const cx = (k) => x0 + (k - a + 0.5) * slot;
    const row = queryRow(raw, A.head, q);

    let older = 0;
    const arcs = [];
    for (let k = 0; k < q; k++) {
      if (k < a) older += row[k];
      else if (row[k] >= 0.01) arcs.push(k);
    }
    arcs.sort((p, r) => row[p] - row[r]);   // 粗的盖在细的上面
    for (const k of arcs) {
      const wgt = row[k], x1 = cx(k), x2 = cx(q);
      const rise = Math.min(ty - 22, 16 + Math.abs(x2 - x1) * 0.4);
      ctx.beginPath();
      ctx.moveTo(x1, ty);
      ctx.quadraticCurveTo((x1 + x2) / 2, ty - rise * 2, x2, ty);
      ctx.lineWidth = 1 + 12 * Math.sqrt(wgt);
      ctx.strokeStyle = `rgba(${ACCENT}, ${0.22 + 0.78 * Math.sqrt(wgt)})`;
      ctx.stroke();
      if (wgt >= 0.08) {
        ctx.fillStyle = '#d5dae5';
        ctx.fillText(pct(wgt), (x1 + x2) / 2, ty - rise - 9);
      }
    }
    if (row[q] >= 0.01) {   // 看自己
      ctx.beginPath();
      ctx.arc(cx(q), ty - 11, 9, 0, Math.PI * 2);
      ctx.lineWidth = 1 + 8 * Math.sqrt(row[q]);
      ctx.strokeStyle = `rgba(${ACCENT}, ${0.25 + 0.75 * Math.sqrt(row[q])})`;
      ctx.stroke();
      if (row[q] >= 0.08) { ctx.fillStyle = '#d5dae5'; ctx.fillText(`自己 ${pct(row[q])}`, cx(q), ty - 30); }
    }
    ctx.textAlign = 'left';
    ctx.fillStyle = '#7f899c';
    if (older >= 0.01) ctx.fillText(`← 更早的词合计 ${pct(older)}`, 10, 14);

    ctx.textAlign = 'center';
    for (let k = a; k < a + count; k++) {
      const x = x0 + (k - a) * slot + 2;
      rrect(ctx, x, ty, slot - 4, th, 5);
      if (k === q) {
        ctx.fillStyle = `rgb(${ACCENT})`;
        ctx.fill();
        ctx.fillStyle = '#06210d';
      } else if (k < q) {
        ctx.fillStyle = `rgba(${ACCENT}, ${0.06 + 0.7 * Math.sqrt(row[k])})`;
        ctx.fill();
        ctx.fillStyle = '#d5dae5';
      } else {
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = '#2a3042';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#4b5366';
      }
      ctx.fillText(clip(ctx, vis(tokens[k]), slot - 12), x + (slot - 4) / 2, ty + th / 2 + 1);
      ctx.fillStyle = '#4b5366';
      ctx.fillText(String(k + 1), x + (slot - 4) / 2, ty + th + 11);
    }
  }

  function updateCaption(raw) {
    const cap = el('attnCaption');
    if (!raw) { cap.textContent = ''; return; }
    const q = clamp(S.step, 0, A.n - 1), tokens = S.rec.tokens;
    const row = queryRow(raw, A.head, q);
    const top = [...row.keys()].filter((k) => row[k] >= 0.01).sort((p, r) => row[r] - row[p]).slice(0, 4);
    const who = A.head < 0 ? `全部 ${A.H} 个头的平均` : `头 ${A.head}`;
    const parts = [`第 ${A.layer} 层 · ${who}：处理第 ${q + 1} 个词「`, { b: vis(tokens[q]) }, '」时，最关注 '];
    top.forEach((k, i) => parts.push(i ? '、' : '', k === q ? '自己' : `「${vis(tokens[k])}」`, ' ', { b: pct(row[k]) }));
    say(cap, ...parts);
  }

  function matrixImage(raw) {
    const key = `${A.id}|${A.layer}|${A.head}`;
    if (A.mat && A.mat.key === key) return A.mat.canvas;
    const n = A.n;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = n;
    const img = new ImageData(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const o = (i * n + j) * 4;
        const [r, g, b] = j > i ? [18, 21, 29] : shade(weight(raw, A.head, i, j));
        img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
      }
    }
    canvas.getContext('2d').putImageData(img, 0, 0);
    A.mat = { key, canvas };
    return canvas;
  }

  function drawMatrix(raw) {
    const g = setup(el('attnMatrix'));
    if (!g) return;
    const { ctx, w, h } = g;
    A.geo = null;
    if (!raw) return;
    const n = A.n, tokens = S.rec.tokens, q = clamp(S.step, 0, n - 1);
    let left = 84, top = 70;
    let size = Math.max(40, Math.min(w - left - 12, h - top - 12));
    const labelled = size / n >= 13;
    if (!labelled) { left = top = 12; size = Math.max(40, Math.min(w, h) - 24); }
    const labelWidth = left;
    left += Math.max(0, (w - left - 12 - size) / 2);
    const cell = size / n;
    A.geo = { left, top, cell };

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(matrixImage(raw), left, top, size, size);
    ctx.strokeStyle = '#232836';
    ctx.lineWidth = 1;
    ctx.strokeRect(left - 0.5, top - 0.5, size + 1, size + 1);
    ctx.fillStyle = `rgba(${ACCENT}, 0.10)`;
    ctx.fillRect(left, top + q * cell, size, cell);
    ctx.strokeStyle = `rgb(${ACCENT})`;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(left, top + q * cell, size, cell);

    ctx.font = '11px Consolas, "Cascadia Mono", monospace';
    ctx.fillStyle = '#7f899c';
    ctx.textBaseline = 'middle';
    if (labelled) {
      ctx.textAlign = 'right';
      for (let i = 0; i < n; i++) {
        ctx.fillStyle = i === q ? `rgb(${ACCENT})` : '#9aa4b8';
        ctx.fillText(clip(ctx, vis(tokens[i]), labelWidth - 12), left - 6, top + (i + 0.5) * cell);
      }
      ctx.textAlign = 'left';
      for (let j = 0; j < n; j++) {
        ctx.save();
        ctx.translate(left + (j + 0.5) * cell, top - 6);
        ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = '#9aa4b8';
        ctx.fillText(clip(ctx, vis(tokens[j]), top - 14), 0, 0);
        ctx.restore();
      }
      ctx.fillStyle = '#7f899c';
      ctx.fillText('行 = 谁在看，列 = 被看的词；右上角空白是看不到的「未来」', 8, 12);
    }
  }

  function drawOverview() {
    const canvas = el('attnOverview');
    const cols = A.H + 1;   // 第 0 列 = 各头平均
    const g = setup(canvas);
    if (!g) return;
    const { ctx, w, h } = g;
    const cell = Math.min((w - 32) / cols, (h - 24) / A.L);
    if (!(cell > 0)) { A.ovGeo = null; return; }
    const lab = 28 + Math.max(0, (w - 32 - cols * cell) / 2), top = 18;
    const cs = clamp(Math.ceil(cell), 1, 40);  // 缩略图采样分辨率与实际点击区域分开。

    const loaded = A.layers.filter(Boolean).length;
    const key = `${A.id}|${loaded}|${cs}`;
    if (!A.ov || A.ov.key !== key) {
      const wpx = cols * cs, hpx = A.L * cs, n = A.n;
      const off = document.createElement('canvas');
      off.width = wpx; off.height = hpx;
      const img = new ImageData(wpx, hpx);
      for (let l = 0; l < A.L; l++) {
        const raw = A.layers[l];
        for (let c = 0; c < cols; c++) {
          for (let py = 0; py < cs; py++) {
            const i = Math.min(n - 1, Math.floor((py * n) / cs));
            for (let px = 0; px < cs; px++) {
              const j = Math.min(n - 1, Math.floor((px * n) / cs));
              const [r, gg, b] = !raw || j > i ? [18, 21, 29] : shade(weight(raw, c - 1, i, j));
              const o = ((l * cs + py) * wpx + c * cs + px) * 4;
              img.data[o] = r; img.data[o + 1] = gg; img.data[o + 2] = b; img.data[o + 3] = 255;
            }
          }
        }
      }
      off.getContext('2d').putImageData(img, 0, 0);
      A.ov = { key, canvas: off };
    }
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(A.ov.canvas, lab, top, cols * cell, A.L * cell);
    A.ovGeo = { lab, top, cs: cell };

    ctx.font = '10px Consolas, "Cascadia Mono", monospace';
    ctx.fillStyle = '#99a4b6';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    for (let l = 0; l < A.L; l += Math.max(1, Math.ceil(13 / cell))) ctx.fillText(String(l), lab - 5, top + (l + .5) * cell);
    ctx.textAlign = 'center';
    for (let c = 0; c < cols; c += Math.max(1, Math.ceil(15 / cell))) ctx.fillText(c === 0 ? '均' : String(c - 1), lab + (c + .5) * cell, top - 8);

    ctx.strokeStyle = 'rgb(' + ACCENT + ')';
    ctx.lineWidth = 2;
    ctx.strokeRect(lab + (A.head + 1) * cell + .5, top + A.layer * cell + .5, Math.max(1,cell - 1), Math.max(1,cell - 1));
  }

  function renderAttention() {
    const msg = emptyMessage('attention');
    el('attnEmpty').hidden = !msg;
    el('attnBody').hidden = !!msg;
    if (msg) { el('attnEmpty').innerHTML = msg; return; }
    const raw = A.layers[A.layer];
    syncAttnControls();
    drawArcs(raw);
    updateCaption(raw);
    drawMatrix(raw);
    drawOverview();
  }
  const renderAttnSoon = scheduler(() => { if (S.tab === 'attn') renderAttention(); });

  el('attnLayer').oninput = (e) => { A.layer = +e.target.value; renderAttnSoon(); };
  el('attnHead').onchange = (e) => { A.head = +e.target.value; renderAttnSoon(); };

  function matrixCell(e) {
    if (!A.geo) return null;
    const r = e.currentTarget.getBoundingClientRect();
    const i = Math.floor((e.clientY - r.top - e.currentTarget.clientTop - A.geo.top) / A.geo.cell);
    const j = Math.floor((e.clientX - r.left - e.currentTarget.clientLeft - A.geo.left) / A.geo.cell);
    return i >= 0 && j >= 0 && i < A.n && j < A.n ? { i, j } : null;
  }
  el('attnMatrix').addEventListener('pointermove', (e) => {
    const c = matrixCell(e), raw = A.layers[A.layer];
    const tokens = S.rec ? S.rec.tokens : [];
    if (!c || !raw) return say(el('attnHover'), '');
    say(el('attnHover'), c.j > c.i
      ? `「${vis(tokens[c.i])}」看不到它后面的「${vis(tokens[c.j])}」`
      : `「${vis(tokens[c.i])}」→「${vis(tokens[c.j])}」 ${pct(weight(raw, A.head, c.i, c.j))}　（点击跳到这一行的词）`);
  });
  el('attnMatrix').addEventListener('pointerleave', () => say(el('attnHover'), ''));
  el('attnMatrix').addEventListener('click', (e) => { const c = matrixCell(e); if (c) setStep(c.i); });
  el('attnOverview').addEventListener('click', (e) => {
    const g = A.ovGeo;
    if (!g) return;
    const r = e.currentTarget.getBoundingClientRect();
    const l = Math.floor((e.clientY - r.top - e.currentTarget.clientTop - g.top) / g.cs), c = Math.floor((e.clientX - r.left - e.currentTarget.clientLeft - g.lab) / g.cs);
    if (l < 0 || l >= A.L || c < 0 || c > A.H) return;
    A.layer = l;
    A.head = c - 1;
    renderAttnSoon();
  });

  // ======================================================================
  // 逐层预测
  // ======================================================================

  const N = { id: null, data: null, status: '', layer: 0, token: 0, rows: [], cols: [], selRow: null };

  // 状态 l：0 = 第一层之前（词嵌入），l >= 1 = 第 l-1 层的输出。和权重图里「第 N 层」的编号一致。
  const layerName = (l) => (l === 0 ? '输入（词嵌入）' : l === N.data.layers - 1 ? `第 ${l - 1} 层·最终输出` : `第 ${l - 1} 层输出`);

  async function startLens(meta) {
    const token = ++N.token;
    N.id = meta && meta.lens ? meta.id : null;
    N.data = null;
    N.status = '';
    N.page = null;
    N.rows = [];
    N.cols = [];
    N.selRow = null;
    el('lensTable').replaceChildren();
    if (!N.id) return;
    N.status = '加载中…';
    try {
      const res = await fetch(`/api/lens/${meta.id}`);
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      if (token !== N.token) return;
      N.data = data;
      N.status = '';
      N.layer = data.layers - 1;
      el('lensLayer').replaceChildren(...Array.from({length: data.layers}, (_, i) => {
        const l = data.layers - i - 1;
        return new Option(layerName(l), l);
      }));
    } catch (err) {
      if (token === N.token) N.status = `加载失败：${err.message}`;
    }
    renderLensSoon();
  }

  function mk(tag, text, cls) {
    const e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }

  // 只生成当前窗口能读清的单元格。翻页、选曲线或播放都由同一层号/位置驱动。
  function layoutLensTable() {
    const wrap = el('lensTable').parentElement, w = wrap.clientWidth, h = wrap.clientHeight;
    if (!w || !h) return;
    const d = N.data, titleWidth = Math.min(116, Math.floor(w * .30));
    const cols = clamp(Math.floor((w - titleWidth) / 55), 1, d.steps);
    const rows = clamp(Math.floor(h / 21) - 2, 1, d.layers);
    // 末页保留上文，不让少数几个格子孤零零占一屏；在同页点选时保持视野不动。
    function windowStart(index, count, capacity, previous) {
      if (previous !== undefined && index >= previous && index < previous + capacity) return previous;
      return Math.min(Math.floor(index / capacity) * capacity, count - capacity);
    }
    const p0 = windowStart(S.step, d.steps, cols, N.page?.cols === cols ? N.page.p0 : undefined), p1 = p0 + cols;
    const r0 = windowStart(d.layers-1-N.layer, d.layers, rows, N.page?.rows === rows ? N.page.r0 : undefined), r1 = r0 + rows;
    const key = [p0,p1,r0,r1,titleWidth].join('/');
    if (N.page?.key !== key) buildLensTable(p0, p1, r0, r1, titleWidth);
    N.page = {key, cols, rows, p0, r0};
    const table = el('lensTable');
    table.style.width = (titleWidth + Math.min(102, (w - titleWidth) / cols) * (p1 - p0)) + 'px';
    table.style.setProperty('--cell-height', clamp(Math.floor(h / (rows + 2)), 1, 30) + 'px');
    el('stepRange').textContent = '位置 ' + (p0+1) + '–' + p1 + ' / ' + d.steps;
    const short = l => l === 0 ? '输入' : String(l-1);
    el('layerRange').textContent = '层 ' + short(d.layers-1-r0) + '–' + short(d.layers-r1);
    el('prevSteps').disabled = p0 === 0; el('nextSteps').disabled = p1 === d.steps;
    el('prevLayers').disabled = r0 === 0; el('nextLayers').disabled = r1 === d.layers;
    markLens();
  }

  function turnPage(axis, delta) {
    const pg = N.page, d = N.data;
    if (!pg || !d) return;
    const layers = axis === 'r0', size = layers ? pg.rows : pg.cols, total = layers ? d.layers : d.steps;
    const old = pg[axis], offset = (layers ? total - 1 - N.layer : S.step) - old;
    const next = clamp((Math.ceil(old / size) + delta) * size, 0, total - size);
    pg[axis] = next;
    if (layers) { N.layer = total - 1 - next - offset; renderLensSoon(); }
    else setStep(next + offset);
  }
  for (const [prev,next,axis] of [['prevLayers','nextLayers','r0'],['prevSteps','nextSteps','p0']]) {
    el(prev).onclick = () => turnPage(axis, -1);
    el(next).onclick = () => turnPage(axis, 1);
  }
  el('lensLayer').onchange = e => { N.layer = +e.target.value; renderLensSoon(); };

  function buildLensTable(p0, p1, rowStart, rowEnd, titleWidth) {
    const d = N.data, tokens = S.rec.tokens;
    const group = mk('colgroup'), title = mk('col'), cols = mk('col');
    title.style.width = titleWidth + 'px'; cols.span = p1-p0; group.append(title,cols);
    const r1 = mk('tr'), r2 = mk('tr');
    r1.append(mk('th', '输入的词', 'row'));
    r2.append(mk('th', '最终预测 →', 'row'));
    for (let p = p0; p < p1; p++) {
      const a = mk('th', vis(tokens[p])), b = mk('th', vis(d.vocab[d.final[p]]));
      a.dataset.p = b.dataset.p = p;
      a.title = `位置 ${p + 1} · ${vis(tokens[p])}`;
      b.title = `位置 ${p + 1} 的最终预测：${vis(d.vocab[d.final[p]])}`;
      r1.append(a);
      r2.append(b);
    }
    const tbody = mk('tbody');
    N.rows = [];
    for (let l = d.layers - 1 - rowStart; l >= d.layers - rowEnd; l--) {
      const tr = mk('tr');
      tr.dataset.l = l;
      tr.append(mk('th', layerName(l), 'row'));
      for (let p = p0; p < p1; p++) {
        const best = d.top[l][p][0];
        const td = mk('td', vis(d.vocab[best]), best === d.final[p] ? 'hit' : '');
        td.dataset.p = p;
        td.style.background = `rgba(${ACCENT}, ${(Math.sqrt(d.final_prob[l][p]) * 0.8).toFixed(3)})`;
        td.title = `${layerName(l)} · 位置 ${p + 1}\n第一候选「${vis(d.vocab[best])}」${pct(d.prob[l][p][0])}\n最终答案「${vis(d.vocab[d.final[p]])}」排第 ${d.final_rank[l][p]}`;
        tr.append(td);
      }
      tbody.append(tr);
      N.rows[l] = tr;
    }
    const thead = mk('thead');
    thead.append(r1, r2);
    el('lensTable').replaceChildren(group, thead, tbody);
  }

  function markLens() {
    for (const e of N.cols) e.classList.remove('col');
    N.cols = [...el('lensTable').querySelectorAll(`[data-p="${S.step}"]`)];
    for (const e of N.cols) e.classList.add('col');
    if (N.selRow) N.selRow.classList.remove('sel');
    N.selRow = N.rows[N.layer] || null;
    if (N.selRow) N.selRow.classList.add('sel');

  }

  function drawChart(ranks) {
    const g = setup(el('lensChart'));
    if (!g) return;
    const { ctx, w, h } = g, d = N.data;
    const padL = 34, padR = 14, padT = 12, padB = 26;
    const tick = (r) => (r >= 1e5 ? `${r / 1e4}万` : r >= 1e4 ? '1万' : r >= 1e3 ? '1千' : String(r));
    const lg = Math.log10(d.vocab_size);
    const x = (l) => padL + (d.layers === 1 ? 0.5 : l / (d.layers - 1)) * (w - padL - padR);
    const y = (r) => padT + (Math.log10(r) / lg) * (h - padT - padB);

    ctx.font = '10px Consolas, "Cascadia Mono", monospace';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    for (let e = 0; 10 ** e <= d.vocab_size; e++) {
      ctx.strokeStyle = '#1b202c';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, y(10 ** e)); ctx.lineTo(w - padR, y(10 ** e)); ctx.stroke();
      ctx.fillStyle = '#7f899c';
      ctx.fillText(tick(10 ** e), padL - 5, y(10 ** e));
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#7f899c';
    const last = d.layers - 1;
    const tickEvery = Math.max(1, Math.ceil(last / Math.max(1, Math.floor((w - padL - padR) / 48))));
    for (let l = 0; l < last && x(last) - x(l) > 36; l += tickEvery) {
      ctx.fillText(l === 0 ? '输入' : String(l - 1), x(l), h - padB + 6);
    }
    ctx.textAlign = 'right';
    ctx.fillText(last ? '第 ' + (last - 1) + ' 层' : '输入', w - 4, h - padB + 6);

    ctx.strokeStyle = '#46ff6e55';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x(N.layer), padT); ctx.lineTo(x(N.layer), h - padB); ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = `rgb(${ACCENT})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ranks.forEach((r, l) => (l ? ctx.lineTo(x(l), y(r)) : ctx.moveTo(x(l), y(r))));
    ctx.stroke();
    ranks.forEach((r, l) => {
      ctx.fillStyle = l === N.layer ? '#ffffff' : `rgb(${ACCENT})`;
      ctx.beginPath(); ctx.arc(x(l), y(r), l === N.layer ? 4.5 : 2.5, 0, Math.PI * 2); ctx.fill();
    });
    N.chart = { padL, padR, x };
  }

  function renderLensSide() {
    const d = N.data, p = clamp(S.step, 0, d.steps - 1), l = N.layer, tokens = S.rec.tokens;
    const answer = d.vocab[d.final[p]];
    const ranks = d.final_rank.map((r) => r[p]);

    say(el('lensTitle'), `位置 ${p + 1}：读到「${vis(tokens[p])}」，最终预测「${vis(answer)}」`);

    const first10 = ranks.findIndex((r) => r <= 10);
    let stable = d.layers - 1;
    while (stable > 0 && ranks[stable - 1] === 1) stable--;
    const summary = [`答案「${vis(answer)}」：`];
    if (stable === 0) summary.push('从一开始就排第 1。');
    else if (first10 === stable) summary.push('此前一直不在前 10 名，从 ', { b: layerName(stable) }, ' 起一步登顶，排第 1。');
    else {
      if (first10 === 0) summary.push('一开始就在前 10 名；');
      else if (first10 > 0) summary.push('从 ', { b: layerName(first10) }, ' 起进入前 10 名；');
      if (stable === d.layers - 1) summary.push('直到 ', { b: layerName(stable) }, ' 才排到第 1。');
      else summary.push('从 ', { b: layerName(stable) }, ' 起稳定排第 1。');
    }
    say(el('lensSummary'), ...summary);

    drawChart(ranks);

    say(el('lensPick'), `${layerName(l)}：答案排第 `, { b: String(ranks[l]) }, ` 名，概率 ${pct(d.final_prob[l][p])}。此刻最想说的前 ${d.k} 个词：`);
    el('lensTop').replaceChildren(...d.top[l][p].map((vi, i) => {
      const row = mk('div', undefined, vi === d.final[p] ? 'cand ans' : 'cand');
      const bar = mk('div', undefined, 'bar');
      const fill = mk('i');
      fill.style.width = `${Math.min(100, d.prob[l][p][i] * 100)}%`;
      bar.title = vis(d.vocab[vi]);
      bar.append(fill, mk('span', bar.title));
      row.append(bar, mk('div', pct(d.prob[l][p][i]), 'pct'));
      return row;
    }));
  }

  function renderLens() {
    const msg = emptyMessage('lens') || (N.data ? null : N.status || '加载中…');
    el('lensEmpty').hidden = !msg;
    el('lensBody').hidden = !!msg;
    if (msg) { el('lensEmpty').innerHTML = msg; return; }
    layoutLensTable();
    el('lensLayer').value = N.layer;
    if (el('lensLayer').clientWidth) renderLensSide();
  }
  const renderLensSoon = scheduler(() => { if (S.tab === 'lens') renderLens(); });

  el('lensTable').addEventListener('click', (e) => {
    const cell = e.target.closest('[data-p]');
    if (!cell || !S.rec) return;
    const row = cell.closest('tr[data-l]');
    if (row) N.layer = +row.dataset.l;
    setStep(+cell.dataset.p);   // 会触发 stepChanged 重绘
  });
  el('lensChart').addEventListener('click', (e) => {
    if (!N.data || !N.chart) return;
    const r = e.currentTarget.getBoundingClientRect();
    const f = (e.clientX - r.left - N.chart.padL) / (r.width - N.chart.padL - N.chart.padR);
    N.layer = clamp(Math.round(f * (N.data.layers - 1)), 0, N.data.layers - 1);
    renderLensSoon();
  });

  // ======================================================================

  if (window.ResizeObserver) {
    new ResizeObserver(() => renderAttnSoon()).observe(el('attnView'));
    const lensObserver = new ResizeObserver(renderLensSoon);
    lensObserver.observe(el('lensView'));
    lensObserver.observe(el('lensTable').parentElement);
  }

  const refresh = () => { renderAttnSoon(); renderLensSoon(); };
  for (const bar of document.querySelectorAll('.view-switch')) bar.onclick = e => {
    const button = e.target.closest('button[data-panel]');
    if (!button) return;
    bar.parentElement.dataset.panel = button.dataset.panel;
    for (const b of bar.children) b.setAttribute('aria-pressed', String(b === button));
    refresh();
  };
  return {
    recordingChanged(meta) { startAttention(meta); startLens(meta); refresh(); },
    stepChanged: refresh,
    tabChanged: refresh,
    state: () => ({ A, N }),   // 只给调试和自动化检查读取
  };
})();
