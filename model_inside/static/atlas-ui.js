// 全景的说明层：只解释当前的数值编码，不参与瓦片计算。
const AtlasUI = (() => {
  const el = id => document.getElementById(id);
  const labels = { rms: '块 RMS', mean: '均值', maxabs: '带符号极值' };
  const idle = '悬停读数 · 单击固定 · 双击放大张量 · 拖动平移';
  const number = v => v == null ? '非有限值' : Number(v.toPrecision(5)).toString();

  function sync(s) {
    const stack = s.mode === 'stack', composite = stack && s.agg !== 'single';
    const unsigned = composite ? s.agg === 'std' : s.reduce === 'rms';
    el('reduce').disabled = el('scope').disabled = composite;
    el('massBox').hidden = stack;
    el('scaleLegend').classList.toggle('unsigned', unsigned);
    el('scaleLow').textContent = unsigned ? '0' : `−${s.range}×`;
    el('scaleMid').textContent = unsigned ? '' : '0';
    el('scaleHigh').textContent = `${unsigned ? '' : '+'}${s.range}×`;
    const note = composite ? `使用同类色标。先逐位置计算${{ std: '层间标准差', mean: '各层均值', maxabs: '各层带符号极值' }[s.agg]}，再缩小。` :
      ({ rms: '越亮，块内权重的整体幅度越大；不区分正负。', maxabs: '显示绝对值最大的权重。蓝为负，暖色为正。', mean: '显示块内均值。蓝为负，暖色为正；正负可能抵消。' })[s.reduce];
    el('scaleHint').textContent = note;
    el('atlasTitle').textContent = stack ? (composite ? '跨层统计' : '逐层对照') : '参数全景';
  }

  function mass(model) {
    const bins = [
      ['词嵌入', '#8eacc3', /(?:^|\.)(?:embed_tokens|word_embeddings|wte|embed)\.weight$/i],
      ['前馈', '#cda879', /(?:^|\.)(?:mlp|feed_forward|ffn)\./i],
      ['注意力', '#85b6a6', /(?:^|\.)(?:self_attn|attention|attn)\./i],
      ['其他', '#9e99ad', /.*/],
    ].map(([name, color, pattern]) => ({name, color, pattern, n: 0}));
    for (const t of model.tensors) bins.find(b => b.pattern.test(t.source)).n += t.h * t.w;
    for (const b of bins.filter(b => b.n)) {
      const pct = 100 * b.n / model.total_params;
      const bar = document.createElement('i');
      bar.style.cssText = `background:${b.color};width:${pct}%`;
      el('massBar').append(bar);
      const label = document.createElement('span'), swatch = document.createElement('i');
      swatch.style.background = b.color;
      label.append(swatch, `${b.name} ${pct < 0.1 ? '<0.1' : pct.toFixed(1)}%`);
      label.title = `${b.n.toLocaleString()} 个权重（按张量名归类）`;
      el('massKey').append(label);
    }
  }

  function captions(items) {
    const marks = [], globals = new Map();
    const layers = items.filter(it => it.tensor.layer !== null);
    const first = layers[0]?.tensor.layer;
    for (const it of items) {
      const {source, layer} = it.tensor;
      if (layer === null) {
        const g = globals.get(source) || {x: it.x, y: it.y, w: 0, h: it.h, count: 0};
        g.w = it.x + it.w - g.x; g.count++;
        globals.set(source, g);
      } else if (layer === first) {
        const m = source.match(/\.(down|up|gate|q|k|v|o)_proj\.weight$/);
        if (m) marks.push({...it, text: m[1].toUpperCase(), vertical: it.h > 2 * it.w});
      }
    }
    for (const [name, g] of globals) {
      const base = /embed_tokens|word_embeddings|wte\.weight|^embed\.weight/.test(name) ? '词嵌入' : name;
      marks.push({...g, above: true, text: base + (g.count > 1 ? ` · ${g.count} 段` : '')});
    }
    return marks;
  }

  function drawCaptions(ctx, marks, dpr, k, px, py) {
    ctx.save();
    ctx.font = `600 ${11 * dpr}px 'Segoe UI', 'Microsoft YaHei', sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (const m of marks) {
      const w = m.w * k, h = m.h * k, tw = ctx.measureText(m.text).width + 8 * dpr;
      if ((m.vertical ? h : w) < tw || (m.vertical ? w : h) < 15 * dpr) continue;
      const x = px(m.x + m.w / 2), y = m.above ? py(m.y) - 11 * dpr : py(m.y + m.h / 2);
      ctx.save(); ctx.translate(x, y);
      if (m.vertical) ctx.rotate(-Math.PI / 2);
      if (!m.above) {
        ctx.fillStyle = 'rgba(12,14,20,.82)'; ctx.fillRect(-tw / 2, -7 * dpr, tw, 15 * dpr);
      }
      ctx.fillStyle = '#dce6eb'; ctx.fillText(m.text, 0, 0); ctx.restore();
    }
    ctx.restore();
  }

  function readout(b, mode, scope) {
    const unit = `基准 RMS ${number(b.unit)}`;
    if (b.count === 1) return `[${b.index.join(', ')}]  权重 ${number(b.mean)}  ·  ${unit}`;
    const dims = `行 ${b.rows.join('–')} / 列 ${b.cols.join('–')}（${b.count.toLocaleString()} 个权重）`;
    const stats = [mode, ...Object.keys(labels).filter(k => k !== mode)].map(k => `${labels[k]} ${number(b[k])}`);
    return `${dims}  ·  ${stats.join('  ·  ')}  ·  ${scope === 'role' ? '同类' : '张量'}${unit}`;
  }
  return {sync, mass, captions, drawCaptions, readout, idle};
})();
