"""把权重区域渲染成瓦片（PNG）。

坐标约定：第 z 级的一个像素覆盖 2^z × 2^z 个权重；每块瓦片 256 × 256 像素。
z = 0 时一个像素就是一个权重。大区域按行分带读取，内存占用有上限。
"""
from __future__ import annotations

import io
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from .store import ModelStore, TensorInfo

TILE = 256
_BAND_ELEMS = 8_000_000  # 每次从 mmap 读入并处理的权重数上限
_ACT_SAMPLES = 4  # 激活图缩小显示时，每个像素块每个方向的采样数


def zmax(h: int, w: int) -> int:
    """最粗的一级：整张图能放进一块瓦片。"""
    return max(0, int(np.ceil(np.log2(max(h, w) / TILE))))


def lodmax(h: int, w: int) -> int:
    """全景可继续聚合到 1×1，不能在刚好装入一块瓦片时就停止。"""
    return (max(h, w) - 1).bit_length()


def tile_region(t_h: int, t_w: int, z: int, x: int, y: int):
    if z < 0 or z > 40:
        return None
    span = TILE << z
    r0, c0 = y * span, x * span
    if z < 0 or x < 0 or y < 0 or r0 >= t_h or c0 >= t_w:
        return None
    return r0, min(r0 + span, t_h), c0, min(c0 + span, t_w)


def _bands(r0: int, r1: int, ncols: int, s: int, layers: int = 1):
    rows = max(1, _BAND_ELEMS // (ncols * layers)) // s * s or s
    for a in range(r0, r1, rows):
        yield a, min(a + rows, r1)


def block_reduce(a: np.ndarray, s: int, mode: str) -> np.ndarray:
    """RMS 表示幅度；mean、maxabs 保留符号。边缘只统计实际存在的元素。"""
    if s == 1:
        return np.abs(a) if mode == "rms" else a
    h, w = a.shape
    ri, ci = np.arange(0, h, s), np.arange(0, w, s)
    if mode in ("mean", "rms"):
        values = np.square(a, dtype=np.float64) if mode == "rms" else a
        sums = np.add.reduceat(values, ci, axis=1, dtype=np.float64)
        count = np.outer(np.minimum(s, h - ri), np.minimum(s, w - ci))
        mean = np.add.reduceat(sums, ri, axis=0) / count
        return np.sqrt(mean) if mode == "rms" else mean
    # reduceat 不需要把一个 1×N 的边缘块补成 s×s。
    mx = np.maximum.reduceat(np.maximum.reduceat(a, ci, axis=1), ri, axis=0)
    mn = np.minimum.reduceat(np.minimum.reduceat(a, ci, axis=1), ri, axis=0)
    return np.where(mx >= -mn, mx, mn)


def _reduce_region(region, s, mode, read, layers=1):
    """read(start, end) 读取一带。即使单个统计块比内存预算还大，也不一次载入它。"""
    r0, r1, c0, c1 = region
    ncols = c1 - c0
    budget_rows = max(1, _BAND_ELEMS // (ncols * layers))
    out = []
    for a, b in _bands(r0, r1, ncols, s, layers):
        if b - a <= budget_rows:
            out.append(block_reduce(read(a, b), s, mode))
            continue
        # 一带只有一个输出块行；子带同宽，用实际行数合并残缺块。
        acc = None
        for i in range(a, b, budget_rows):
            j = min(i + budget_rows, b)
            v = block_reduce(read(i, j), s, mode)
            if mode == "maxabs":
                if acc is None:
                    acc = v
                else:
                    acc = np.where((np.abs(v) > np.abs(acc)) | ((v == -acc) & (v > 0)), v, acc)
            else:
                part = (v * v if mode == "rms" else v) * (j - i)
                acc = part if acc is None else acc + part
        if mode != "maxabs":
            acc /= b - a
            if mode == "rms":
                acc = np.sqrt(acc)
        out.append(acc)
    return np.vstack(out)


# ---- 颜色 ----

_T = np.array([0.0, 0.5, 1.0])
_POS = np.array([[12, 14, 20], [232, 84, 32], [255, 238, 176]], dtype=np.float32)
_NEG = np.array([[12, 14, 20], [34, 104, 236], [176, 240, 255]], dtype=np.float32)
_SEQ = np.array([[12, 14, 20], [46, 116, 145], [246, 226, 156]], dtype=np.float32)


def _ramp(t: np.ndarray, stops: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t, _T, stops[:, c]) for c in range(3)], axis=-1)


def colorize(v: np.ndarray, signed: bool = True) -> np.ndarray:
    """v 已按数值范围归一化。正值暖色、负值冷色、零接近黑色。"""
    v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
    t = np.clip(np.abs(v), 0, 1) ** 0.7
    if not signed:
        return _ramp(t, _SEQ).astype(np.uint8)
    return np.where((v >= 0)[..., None], _ramp(t, _POS), _ramp(t, _NEG)).astype(np.uint8)


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", compress_level=1)
    return buf.getvalue()


# ---- 渲染 ----


def weight_values(store: ModelStore, t: TensorInfo, z: int, x: int, y: int, mode: str) -> np.ndarray | None:
    region = tile_region(t.h, t.w, z, x, y)
    if region is None:
        return None
    _, _, c0, c1 = region
    return _reduce_region(region, 1 << z, mode, lambda a, b: store.read(t, a, b, c0, c1))


def inspect_block(store, t, z, r, c, scope):
    """与瓦片网格对齐，只读命中的块。大块分带统计，极值平手取正值。"""
    s = 1 << z
    r0, c0 = r // s * s, c // s * s
    r1, c1 = min(r0 + s, t.h), min(c0 + s, t.w)
    n = (r1 - r0) * (c1 - c0)
    total = squares = 0.0
    lo, hi = np.inf, -np.inf
    for a, b in _bands(r0, r1, c1 - c0, 1):
        w = store.read(t, a, b, c0, c1)
        total += w.sum(dtype=np.float64)
        squares += np.square(w, dtype=np.float64).sum()
        lo, hi = np.minimum(lo, w.min()), np.maximum(hi, w.max())
    def finite(v):
        return float(v) if np.isfinite(v) else None
    return {"rows": [t.row0 + r0, t.row0 + r1 - 1], "cols": [c0, c1 - 1], "count": n,
            "rms": finite(np.sqrt(squares / n)), "mean": finite(total / n),
            "maxabs": finite(hi if hi >= -lo else lo), "unit": store.scale(t, scope),
            "index": store.original_index(t, r, c) if n == 1 else None}


def render_weight_tile(store: ModelStore, t: TensorInfo, z: int, x: int, y: int, mode: str, rng: float, scope="tensor"):
    v = weight_values(store, t, z, x, y, mode)
    if v is None:
        return None
    return _png(colorize(v / (rng * store.scale(t, scope)), signed=mode != "rms"))


def stack_values(store: ModelStore, members: list[TensorInfo], z: int, x: int, y: int, agg: str):
    """跨层聚合：先在原始分辨率上对各层求 mean / std / maxabs，再缩小。"""
    t0 = members[0]
    region = tile_region(t0.h, t0.w, z, x, y)
    if region is None:
        return None
    _, _, c0, c1 = region

    def layer_band(a, b):
        cube = np.stack([store.read(t, a, b, c0, c1) for t in members])
        if agg == "std":
            return cube.std(axis=0)
        if agg == "maxabs":
            idx = np.abs(cube).argmax(axis=0)
            return np.take_along_axis(cube, idx[None], axis=0)[0]
        return cube.mean(axis=0)

    return _reduce_region(region, 1 << z, "maxabs" if agg == "maxabs" else "mean", layer_band, len(members))


def render_stack_tile(store: ModelStore, members: list[TensorInfo], z: int, x: int, y: int, agg: str, rng: float):
    v = stack_values(store, members, z, x, y, agg)
    if v is None:
        return None
    scale = store.scale(members[0], "role")
    return _png(colorize(v / (rng * scale), signed=agg != "std"))


def activation_alpha(
    store: ModelStore, t: TensorInfo, vec: np.ndarray, axis: int, z: int, x: int, y: int, thr: float
) -> np.ndarray | None:
    """每个权重在这一步的贡献 |w · x|，超过阈值的算"被激活"。

    vec 是该层这一步的输入向量，axis 是它在显示坐标里对应的轴（1 = 每列一个输入）。
    贡献先除以"典型贡献"（权重均方根 × 输入均方根），thr 是这个比值的阈值。
    返回 0..1 的亮度；缩小显示时是块内被激活权重的平均亮度。

    激活图每一步都不同、没法预先缓存，所以每块瓦片的计算量必须固定：
    一个像素覆盖超过 4×4 个权重时，每个方向只均匀取 4 个样本来估计，而不是全部读出。
    """
    region = tile_region(t.h, t.w, z, x, y)
    if region is None:
        return None
    r0, r1, c0, c1 = region
    s = 1 << z
    step = max(1, s // _ACT_SAMPLES)
    vec = np.abs(vec.astype(np.float32))
    rms = float(np.sqrt(np.mean(vec.astype(np.float64) ** 2)))
    unit = store.scale(t) * rms
    thr = max(thr, 1e-6)
    w = np.abs(store.read(t, r0, r1, c0, c1, step))
    x = vec[None, c0:c1:step] if axis == 1 else vec[r0:r1:step, None]
    ratio = w * x / unit if unit > 0 else w * 0
    lit = np.where(ratio > thr, 0.3 + 0.7 * np.clip((ratio - thr) / (3 * thr), 0, 1), 0).astype(np.float32)
    return np.sqrt(block_reduce(lit, s // step, "mean"))


def render_activation_tile(store, t, vec, axis, z, x, y, thr):
    a = activation_alpha(store, t, vec, axis, z, x, y, thr)
    if a is None:
        return None
    rgba = np.empty((*a.shape, 4), dtype=np.uint8)
    glow = (a * a)[..., None]
    rgba[..., :3] = np.array([70, 255, 110]) * (1 - glow) + 255 * glow
    rgba[..., 3] = a * 255
    return _png(rgba)


class TileCache:
    """内存 LRU；计算代价高的缩小级别（z >= 2）另外落盘，下次启动直接用。"""

    def __init__(self, root: Path, max_items: int = 2000):
        self.root = root
        self.max_items = max_items
        self._mem: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str, z: int, make, persist: bool = True):
        with self._lock:
            data = self._mem.get(key)
            if data is not None:
                self._mem.move_to_end(key)
                return data
        path = self.root / f"{key}.png"
        on_disk = persist and z >= 2
        if on_disk and path.is_file():
            data = path.read_bytes()
        else:
            data = make()
            if data is None:
                return None
            if on_disk:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
                tmp.write_bytes(data)
                tmp.replace(path)
        with self._lock:
            self._mem[key] = data
            while len(self._mem) > self.max_items:
                self._mem.popitem(last=False)
        return data
