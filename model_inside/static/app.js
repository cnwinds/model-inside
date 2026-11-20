// 画布上的一切都用"世界坐标"：1 个单位 = 1 个权重。
// view.scale 是每个权重占多少 CSS 像素；绘制时统一换算到设备像素，避免瓦片接缝。
const $ = (id) => document.getElementById(id);
const canvas = $('view');
const ctx = canvas.getContext('2d');
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const MAX_SCALE = 48;
const CACHE_LIMIT = 3000;

const S = {
  model: null,
  tab: 'weights',       // weights | attn | lens
  items: [],            // 当前要画的图：{x, y, w, h, zmax, label, tensor, url(z, tx, ty)}
  world: { w: 1, h: 1 },
  columns: [],          // 全景模式下每层一列的范围，用来标层号
  view: { x: 0, y: 0, scale: 1 },
  minScale: 1e-6,
  mode: 'atlas', reduce: 'rms', scope: 'role', range: 2, autoFit: true, marks: [],
  group: 0, agg: 'single', layer: 0,
  rec: null, recTensors: new Set(), step: 0, thr: 3,
  playing: false, speed: 4,
  hover: null, pinned: false, groupFor: new Map(),
};

// ---------- 瓦片缓存 ----------

const cache = new Map();        // url -> {img, ok, failed}
const lastOverlay = new Map();  // 瓦片位置 -> 上一张已加载的激活图，切步时避免闪烁
let pendingOverlays = 0;

function getTile(url) {
  let e = cache.get(url);
  if (e) {
    cache.delete(url);
    cache.set(url, e);
    return e;
  }
  e = { img: new Image(), ok: false, failed: false };
  e.img.onload = () => { e.ok = true; requestDraw(); };
  e.img.onerror = () => { e.failed = true; requestDraw(); };
  e.img.src = url;
  cache.set(url, e);
  if (cache.size > CACHE_LIMIT) {
    const oldest = cache.keys().next().value;
    cache.get(oldest).img.src = '';
    cache.delete(oldest);
  }
  return e;
}

const q = () => `range=${S.range}&v=${S.model.id}`;
const tileUrl = (tid) => (z, x, y) => `/api/tile/${tid}/${z}/${x}/${y}.png?mode=${S.reduce}&scope=${S.scope}&${q()}`;
const stackUrl = (gid) => (z, x, y) => `/api/stack/${gid}/${z}/${x}/${y}.png?agg=${S.agg}&${q()}`;
const actUrl = (tid, z, x, y) => `/api/act/${S.rec.id}/${S.step}/${tid}/${z}/${x}/${y}.png?thr=${S.thr}`;

// ---------- 要画哪些图 ----------

const AGG_NAMES = { mean: '各层平均', maxabs: '各层绝对值最大', std: '层间标准差' };

function rebuildItems() {
  setHover(null);
  AtlasUI.sync(S);
  const m = S.model;
  if (S.mode === 'atlas') {
    S.items = m.tensors.map((t) => ({ x: t.x, y: t.y, w: t.w, h: t.h, zmax: t.lodmax, label: t.name, tensor: t, url: tileUrl(t.id) }));
    S.marks = AtlasUI.captions(S.items);
    S.world = m.world;
    const cols = new Map();
    for (const t of m.tensors) {
      if (t.layer === null) continue;
      const c = cols.get(t.layer) || { layer: t.layer, x0: Infinity, x1: -Infinity, y1: -Infinity };
      c.x0 = Math.min(c.x0, t.x); c.x1 = Math.max(c.x1, t.x + t.w); c.y1 = Math.max(c.y1, t.y + t.h);
      cols.set(t.layer, c);
    }
    S.columns = [...cols.values()];
  } else {
    const g = m.groups[S.group];
    S.marks = [];
    const base = { x: 0, y: 0, w: g.w, h: g.h, zmax: g.lodmax };
    if (S.agg === 'single') {
      const t = m.tensors[g.members[S.layer]];
      S.items = [{ ...base, label: t.name, tensor: t, url: tileUrl(t.id) }];
    } else {
      S.items = [{ ...base, label: `${g.role} · ${AGG_NAMES[S.agg]}（${g.members.length} 层）`, tensor: null, url: stackUrl(g.id) }];
    }
    S.world = { w: g.w, h: g.h };
    S.columns = [];
  }
  requestDraw();
}

function fit(rect) {
  setHover(null);
  S.autoFit = !rect;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (!cw || !ch) return;
  const pad = Math.min(40, cw * .07, ch * .1);
  const scaleFor = (r) => Math.min((cw - 2 * pad) / r.w, (ch - 2 * pad) / r.h);
  const world = { x: 0, y: 0, ...S.world };
  S.minScale = scaleFor(world) * 0.3;
  rect = rect || world;
  const scale = clamp(scaleFor(rect), S.minScale, MAX_SCALE);
  S.view = { scale, x: rect.x - (cw / scale - rect.w) / 2, y: rect.y - (ch / scale - rect.h) / 2 };
  requestDraw();
}

// ---------- 绘制 ----------

let drawQueued = false;
function requestDraw() {
  if (!drawQueued) {
    drawQueued = true;
    requestAnimationFrame(draw);
  }
}

function level(it, k) { return clamp(Math.round(Math.log2(1 / k)), 0, it.zmax); }

function draw() {
  drawQueued = false;
  if (!S.model) return;
  if (S.tab !== 'weights') { pendingOverlays = 0; return; }   // 别的标签页下不画瓦片，也别让播放等着它
  const dpr = window.devicePixelRatio || 1;
  const W = Math.round(canvas.clientWidth * dpr), H = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#0c0e14';
  ctx.fillRect(0, 0, W, H);
  ctx.imageSmoothingEnabled = false;

  const k = S.view.scale * dpr;                 // 每个权重占多少设备像素
  const px = (wx) => Math.round((wx - S.view.x) * k);
  const py = (wy) => Math.round((wy - S.view.y) * k);
  const T = S.model.tile;
  let pending = 0;

  for (const it of S.items) {
    const x0 = px(it.x), y0 = py(it.y), x1 = px(it.x + it.w), y1 = py(it.y + it.h);
    if (x0 > W || y0 > H || x1 < 0 || y1 < 0) continue;
    if (x1 <= x0 || y1 <= y0) continue;
    const z = level(it, k);
    const s = 2 ** z, span = T * s;
    const tx0 = Math.max(0, Math.floor((S.view.x - it.x) / span));
    const ty0 = Math.max(0, Math.floor((S.view.y - it.y) / span));
    const tx1 = Math.min(Math.ceil(it.w / span) - 1, Math.floor((S.view.x + W / k - it.x) / span));
    const ty1 = Math.min(Math.ceil(it.h / span) - 1, Math.floor((S.view.y + H / k - it.y) / span));
    const overlay = S.rec && it.tensor && S.recTensors.has(it.tensor.source);

    for (let ty = ty0; ty <= ty1; ty++) {
      for (let tx = tx0; tx <= tx1; tx++) {
        // 裁掉边缘像素超出张量的部分，窄偏置不应被画成整块宽带。
        const cols = Math.min(span, it.w - tx * span) / s;
        const rows = Math.min(span, it.h - ty * span) / s;
        const dx = px(it.x + tx * span), dy = py(it.y + ty * span);
        const dw = px(it.x + tx * span + cols * s) - dx, dh = py(it.y + ty * span + rows * s) - dy;
        if (!dw || !dh) continue;

        ctx.globalAlpha = S.rec && it.tensor ? 0.22 : 1;  // 回放时压暗底图，让被激活的权重突出
        const e = getTile(it.url(z, tx, ty));
        if (e.ok) {
          ctx.drawImage(e.img, 0, 0, cols, rows, dx, dy, dw, dh);
        } else if (!drawAncestor(it, z, tx, ty, cols, rows, dx, dy, dw, dh)) {
          ctx.fillStyle = '#161a23';
          ctx.fillRect(dx, dy, dw, dh);
        }
        ctx.globalAlpha = 1;

        if (overlay) {
          const pos = `${it.tensor.id}/${z}/${tx}/${ty}`;
          const a = getTile(actUrl(it.tensor.id, z, tx, ty));
          if (a.ok) lastOverlay.set(pos, a.img);
          else if (!a.failed) pending++;
          const img = a.ok ? a.img : lastOverlay.get(pos);
          if (img) ctx.drawImage(img, 0, 0, cols, rows, dx, dy, dw, dh);
        }
      }
    }
  }
  pendingOverlays = pending;
  drawLabels(dpr, k, px, py);
  drawHoverCell(k, px, py);
  const side = 2 ** Math.max(0, Math.round(Math.log2(1 / k)));
  $('zoomInfo').textContent = k >= 1 ? '1 权重 ≈ ' + k.toFixed(1) + ' px' : '统计块 ≤ ' + side + ' × ' + side;
}

// 目标瓦片还没到时，先用已经加载的更粗一级瓦片顶上，缩放时不会闪黑。
function drawAncestor(it, z, tx, ty, cols, rows, dx, dy, dw, dh) {
  for (let zz = z + 1; zz <= it.zmax; zz++) {
    const d = zz - z;
    const e = cache.get(it.url(zz, tx >> d, ty >> d));
    if (!e || !e.ok) continue;
    const f = 2 ** d, T = S.model.tile;
    const sx = (tx - ((tx >> d) << d)) * T / f, sy = (ty - ((ty >> d) << d)) * T / f;
    ctx.drawImage(e.img, sx, sy, cols / f, rows / f, dx, dy, dw, dh);
    return true;
  }
  return false;
}

function drawLabels(dpr, k, px, py) {
  ctx.fillStyle = '#8b95a7';
  ctx.font = `${11 * dpr}px Consolas, monospace`;
  ctx.textBaseline = 'bottom';
  const room = S.mode === 'stack' || S.model.gap * k >= 14 * dpr;
  if (room) {
    for (const it of S.items) {
      const x0 = px(it.x), x1 = px(it.x + it.w), y0 = py(it.y);
      const width = x1 - x0;
      if (width < 60 * dpr || x1 < 0 || x0 > canvas.width || y0 < 0 || y0 > canvas.height + 20) continue;
      let text = `${it.label}  ${it.h}×${it.w}`;
      while (text.length > 4 && ctx.measureText(text).width > width) text = '…' + text.slice(Math.ceil(text.length * 0.15) + 1);
      ctx.fillText(text, x0, y0 - 3 * dpr);
    }
  }
  if (!room) AtlasUI.drawCaptions(ctx, S.marks, dpr, k, px, py);
  ctx.textBaseline = 'top';
  ctx.textAlign = 'center';
  for (const c of S.columns) {
    const x0 = px(c.x0), x1 = px(c.x1);
    if (x1 - x0 < 14 * dpr) continue;
    ctx.fillText(x1 - x0 < 48 * dpr ? String(c.layer) : `第 ${c.layer} 层`, (x0 + x1) / 2, py(c.y1) + 4 * dpr);
  }
  ctx.textAlign = 'left';
}

function drawHoverCell(k, px, py) {
  const h = S.hover;
  if (!h) return;
  const s = 2 ** h.z;
  const x = px(h.it.x + h.c), y = py(h.it.y + h.r);
  const w = px(h.it.x + Math.min(h.c + s, h.it.w)) - x;
  const height = py(h.it.y + Math.min(h.r + s, h.it.h)) - y;
  ctx.lineWidth = window.devicePixelRatio || 1;
  ctx.strokeStyle = '#eaf5fb';
  if (Math.min(w, height) >= 5) ctx.strokeRect(x, y, w, height);
  else ctx.strokeRect(px(h.it.x), py(h.it.y), h.it.w * k, h.it.h * k);
}

// ---------- 交互 ----------

function zoomAt(mx, my, factor) {
  const v = S.view;
  S.autoFit = false;
  setHover(null);
  const ns = clamp(v.scale * factor, S.minScale, MAX_SCALE);
  const wx = v.x + mx / v.scale, wy = v.y + my / v.scale;
  S.view = { scale: ns, x: wx - mx / ns, y: wy - my / ns };
  requestDraw();
}

function itemAt(mx, my) {
  const dpr = window.devicePixelRatio || 1, k = S.view.scale * dpr;
  const ix = Math.floor(mx * dpr) + .5, iy = Math.floor(my * dpr) + .5;
  const coord = (u, axis) => Math.round((u - S.view[axis]) * k);
  const it = S.items.find(i => ix >= coord(i.x, 'x') && ix < coord(i.x + i.w, 'x') && iy >= coord(i.y, 'y') && iy < coord(i.y + i.h, 'y'));
  if (!it) return null;
  const z = level(it, k), s = 2 ** z, span = S.model.tile * s;
  // 以屏幕像素中心和 drawImage 的整数边界确定实际采到的统计块。
  function cell(p, origin, size, axis) {
    let tile = clamp(Math.floor(((p / k + S.view[axis]) - origin) / span), 0, Math.ceil(size / span) - 1);
    if (p < coord(origin + tile * span, axis)) tile--;
    else if (p >= coord(origin + (tile + 1) * span, axis)) tile++;
    const start = tile * span, end = Math.min(size, start + span);
    const lo = coord(origin + start, axis), hi = coord(origin + end, axis);
    return start + Math.floor((p - lo) / (hi - lo) * (end - start) / s) * s;
  }
  return {it, z, r: cell(iy, it.y, it.h, 'y'), c: cell(ix, it.x, it.w, 'x')};
}

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * 0.0015));
}, { passive: false });

let drag = null;
canvas.addEventListener('pointerdown', e => {
  if (!e.isPrimary || e.button !== 0) return;
  drag = {x: e.clientX, y: e.clientY, moved: false};
  canvas.setPointerCapture(e.pointerId);
});
function endDrag() { drag = null; canvas.classList.remove('dragging'); }
canvas.addEventListener('pointercancel', endDrag);
canvas.addEventListener('pointerup', e => {
  if (!e.isPrimary || !drag) return;
  if (!drag.moved) {
    const hit = itemAt(e.offsetX, e.offsetY);
    const same = S.pinned && hit && S.hover && hit.it === S.hover.it && hit.r === S.hover.r && hit.c === S.hover.c;
    setHover(hit && !same ? hit : null, !!hit && !same);
  }
  endDrag();
});
canvas.addEventListener('pointermove', e => {
  if (!e.isPrimary) return;
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < 3) return;
    if (!drag.moved) { setHover(null); canvas.classList.add('dragging'); }
    S.autoFit = false;
    S.view.x -= dx / S.view.scale; S.view.y -= dy / S.view.scale;
    drag = {x: e.clientX, y: e.clientY, moved: true};
    requestDraw();
  } else if (!S.pinned) setHover(itemAt(e.offsetX, e.offsetY));
});
canvas.addEventListener('pointerleave', () => { if (!S.pinned) setHover(null); });
canvas.addEventListener('dblclick', (e) => {
  const hit = itemAt(e.offsetX, e.offsetY);
  if (hit) fit(hit.it);
});
new ResizeObserver(() => {
  if (S.model && S.tab === 'weights' && canvas.clientWidth) {
    if (S.autoFit) fit(); else { setHover(null); requestDraw(); }
  }
}).observe(canvas);

let hoverTimer = 0, hoverSeq = 0, hoverFetch = null;
function setHover(h, pinned = false) {
  if (pinned === S.pinned && h && S.hover && h.it === S.hover.it && h.z === S.hover.z && h.r === S.hover.r && h.c === S.hover.c) return;
  S.hover = h;
  S.pinned = pinned;
  $('comparePick').hidden = !pinned || S.mode !== 'atlas' || !S.groupFor.has(h?.it.tensor?.id);
  requestDraw();
  clearTimeout(hoverTimer);
  if (hoverFetch) hoverFetch.abort();
  const seq = ++hoverSeq;
  $('probeTitle').textContent = h ? (pinned ? '已固定 · ' : '') + h.it.label : '权重探针';
  $('status').textContent = h ? '读取统计块…' : AtlasUI.idle;
  if (!h) return;
  if (!h.it.tensor) {
    const s = 2 ** h.z;
    $('status').textContent = '跨层合成图 · 行 ' + h.r + '–' + (Math.min(h.r+s,h.it.h)-1) + ' / 列 ' + h.c + '–' + (Math.min(h.c+s,h.it.w)-1) + ' · 切换「逐层查看」可检查原权重';
    return;
  }
  hoverTimer = setTimeout(async () => {
    hoverFetch = new AbortController();
    try {
      const params = new URLSearchParams({z: h.z, r: h.r, c: h.c, scope: S.scope});
      const res = await fetch('/api/block/' + h.it.tensor.id + '?' + params, {signal: hoverFetch.signal});
      if (!res.ok) throw new Error();
      const b = await res.json();
      if (seq === hoverSeq) $('status').textContent = AtlasUI.readout(b, S.reduce, S.scope);
    } catch (err) {
      if (seq === hoverSeq && err.name !== 'AbortError') $('status').textContent = '读取失败，请移开后重试';
    }
  }, 80);
}

// ---------- 标签页 ----------

function setTab(tab) {
  S.tab = tab;
  for (const b of document.querySelectorAll('#tabs [data-view]')) b.classList.toggle('on', b.dataset.view === tab);
  $('weightsPane').hidden = tab !== 'weights';
  $('attnView').hidden = tab !== 'attn';
  $('lensView').hidden = tab !== 'lens';
  $('weightsSide').hidden = tab !== 'weights';    // 缩放、叠加这些设置只对权重图有意义
  $('thrBox').hidden = !S.rec || tab !== 'weights';
  setHover(null);
  togglePanel(false);
  requestDraw();
  Views.tabChanged();
}
function togglePanel(open) {
  document.body.classList.toggle('controls-open', open);
  $('panelToggle').setAttribute('aria-expanded', String(open));
}
$('panelToggle').onclick = () => togglePanel(!document.body.classList.contains('controls-open'));
document.addEventListener('keydown', e => { if (e.key === 'Escape') { togglePanel(false); setHover(null); } });
for (const b of document.querySelectorAll('#tabs [data-view]')) b.onclick = () => setTab(b.dataset.view);

// 方向键逐个 token 前后切换（在输入框、下拉框、滑块上时让它们自己处理）
window.addEventListener('keydown', (e) => {
  if (!S.rec || e.altKey || e.ctrlKey || e.metaKey) return;
  if (['TEXTAREA', 'INPUT', 'SELECT'].includes(e.target.tagName)) return;
  const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
  if (!d) return;
  e.preventDefault();
  setStep(clamp(S.step + d, 0, S.rec.tokens.length - 1));
});

// ---------- 侧栏 ----------

function setMode(mode) {
  S.mode = mode;
  $('modeAtlas').classList.toggle('on', mode === 'atlas');
  $('modeStack').classList.toggle('on', mode === 'stack');
  $('stackBox').hidden = mode !== 'stack';
  syncStackControls();
  rebuildItems();
  fit();
}

function syncStackControls() {
  const g = S.model.groups[S.group];
  if (!g) return;
  S.layer = clamp(S.layer, 0, g.members.length - 1);
  $('layer').max = g.members.length - 1;
  $('layer').value = S.layer;
  $('layerRow').hidden = S.agg !== 'single';
  $('layerLabel').textContent = `第 ${S.model.tensors[g.members[S.layer]].layer} 层（共 ${g.members.length} 层）`;
}

$('comparePick').onclick = () => {
  const tid = S.hover?.it.tensor?.id, g = S.groupFor.get(tid);
  if (!g) return;
  S.group = g.id; S.layer = g.members.indexOf(tid); S.agg = 'single';
  $('group').value = g.id; $('agg').value = S.agg;
  setMode('stack');
};
$('modeAtlas').onclick = () => setMode('atlas');
$('modeStack').onclick = () => setMode('stack');
$('fitBtn').onclick = () => fit();
for (const [id, factor] of [['zoomOut', 0.5], ['zoomIn', 2]]) {
  $(id).onclick = () => zoomAt(canvas.clientWidth / 2, canvas.clientHeight / 2, factor);
}
for (const key of ['reduce', 'scope', 'range']) $(key).onchange = e => {
  S[key] = key === 'range' ? +e.target.value : e.target.value;
  if (key === 'reduce') { S.range = {rms: 2, maxabs: 6, mean: 1}[S.reduce]; $('range').value = S.range; }
  setHover(null); AtlasUI.sync(S); requestDraw();
};
$('group').onchange = (e) => { S.group = +e.target.value; syncStackControls(); rebuildItems(); fit(); };
$('agg').onchange = (e) => { S.agg = e.target.value; syncStackControls(); rebuildItems(); };
$('layer').oninput = (e) => { S.layer = +e.target.value; syncStackControls(); rebuildItems(); };

// ---------- 录制与回放 ----------

let playTimer = 0;

function setStep(step) {
  S.step = step;
  $('stepSlider').value = step;
  const n = S.rec.tokens.length;
  $('stepLabel').textContent = `位置 ${step + 1} / ${n}`;
  const buttons = $('tokens').children;
  for (let i = 0; i < buttons.length; i++) buttons[i].classList.toggle('cur', i === step);
  // 只移动 token 条，切步和播放不应带着整个界面跳动。
  const strip = $('tokens'), area = strip.getBoundingClientRect(), active = buttons[step].getBoundingClientRect();
  if (active.left < area.left) strip.scrollLeft += active.left - area.left;
  else if (active.right > area.right) strip.scrollLeft += active.right - area.right;
  requestDraw();
  Views.stepChanged();
}

function setPlaying(on) {
  S.playing = on;
  $('playBtn').textContent = on ? '暂停' : '播放';
  clearTimeout(playTimer);
  if (on) playTimer = setTimeout(tick, 1000 / S.speed);
}

function tick() {
  if (!S.playing || !S.rec) return;
  // 这一步的激活图还没算完就先等着，保证画面和进度一致
  if (pendingOverlays === 0) {
    if (S.step >= S.rec.tokens.length - 1) return setPlaying(false);
    setStep(S.step + 1);
  }
  playTimer = setTimeout(tick, 1000 / S.speed);
}

function selectRecording(meta) {
  setPlaying(false);
  lastOverlay.clear();
  S.rec = meta;
  S.recTensors = new Set(meta ? meta.modules.map((m) => m.tensor) : []);
  $('player').hidden = !meta;
  $('stepHint').hidden = !meta;
  $('thrBox').hidden = !meta || S.tab !== 'weights';
  $('recSelect').value = meta ? meta.id : '';
  if (meta) {
    $('stepSlider').max = meta.tokens.length - 1;
    $('tokens').replaceChildren(...meta.tokens.map((tok, i) => {
      const b = document.createElement('button');
      b.textContent = tok.replace(/\n/g, '↵') || '∅';
      b.title = i < meta.n_prompt ? '提示词' : '模型生成';
      if (i >= meta.n_prompt) b.className = 'gen';
      b.onclick = () => setStep(i);
      return b;
    }));
    setStep(0);
  }
  Views.recordingChanged(meta);
  requestDraw();
}

let recordings = [];
async function loadRecordings() {
  recordings = await (await fetch('/api/recordings')).json();
  $('recSelect').replaceChildren(new Option('不显示', ''), ...recordings.map((r) => {
    const text = r.prompt.length > 18 ? r.prompt.slice(0, 18) + '…' : r.prompt;
    return new Option(`${r.id}  ${text}`, r.id);
  }));
  $('recSelect').value = S.rec ? S.rec.id : '';
}

$('recSelect').onchange = (e) => selectRecording(recordings.find((r) => r.id === e.target.value) || null);
$('playBtn').onclick = () => {
  if (!S.playing && S.step >= S.rec.tokens.length - 1) setStep(0);
  setPlaying(!S.playing);
};
$('stepSlider').oninput = (e) => setStep(+e.target.value);
$('speed').onchange = (e) => { S.speed = +e.target.value; };

let thrTimer = 0;
function showThr() { $('thrLabel').textContent = `激活阈值：贡献 > 典型值的 ${S.thr} 倍`; }
$('thr').oninput = (e) => {
  S.thr = +e.target.value;
  showThr();
  clearTimeout(thrTimer);
  thrTimer = setTimeout(requestDraw, 150);
};

$('recordBtn').onclick = async () => {
  const prompt = $('prompt').value;
  if (!prompt.trim()) { $('recMsg').textContent = '请先输入提示词'; return; }
  $('recordBtn').disabled = true;
  $('recMsg').textContent = '运行中…首次需要把模型载入内存，会慢一些';
  try {
    const res = await fetch('/api/record', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, max_new_tokens: +$('newTokens').value || 0 }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.statusText);
    const extra = [body.attention && '注意力', body.lens && '逐层预测'].filter(Boolean);
    $('recMsg').textContent = `已记录 ${body.tokens.length} 个位置、${body.modules.length} 个线性层${extra.length ? '，以及' + extra.join('、') : ''}`;
    await loadRecordings();
    selectRecording(body);
  } catch (err) {
    $('recMsg').textContent = `失败：${err.message}`;
  } finally {
    $('recordBtn').disabled = !S.model.can_record;
  }
};

// ---------- 启动 ----------

function formatCount(n) {
  if (n >= 1e8) return `${(n / 1e8).toFixed(2)} 亿`;
  if (n >= 1e4) return `${(n / 1e4).toFixed(1)} 万`;
  return String(n);
}

async function init() {
  const response = await fetch('/api/model');
  if (!response.ok) throw new Error('读取模型信息失败，请检查本地服务后刷新。');
  const m = await response.json();
  if (!Number.isInteger(m.source_count)) throw new Error('服务仍是旧版。请重启 model_inside 后刷新本页。');
  S.model = m;
  document.title = `${m.name} · model-inside`;
  $('modelName').textContent = m.name;
  $('modelStats').textContent = `${formatCount(m.total_params)} 个权重 · ${m.source_count} 个张量 · ${m.tensors.length} 个显示块`;
  AtlasUI.mass(m);
  $('group').replaceChildren(...m.groups.map((g) => new Option(`${g.role}（${g.members.length} 层，${g.h}×${g.w}）`, g.id)));
  for (const g of m.groups) for (const id of g.members) S.groupFor.set(id, g);
  if (m.groups.length) {
    const first = m.groups.find(g => g.role.endsWith('.q_proj.weight')) || m.groups.reduce((a,b) => a.h*a.w >= b.h*b.w ? a : b);
    S.group = first.id; $('group').value = first.id;
  }
  $('modeStack').disabled = m.groups.length === 0;
  if (!m.can_record) {
    $('recordBtn').disabled = true;
    $('recMsg').textContent = '模型目录里没有 config.json，只能查看权重';
  }
  showThr();
  await loadRecordings();
  rebuildItems();
  fit();
}

init().catch(error => {
  $('modelName').textContent = '无法打开权重图';
  $('modelStats').textContent = $('status').textContent = error.message;
  $('controlsPanel').querySelectorAll('input, select, button, textarea').forEach(el => el.disabled = true);
});
