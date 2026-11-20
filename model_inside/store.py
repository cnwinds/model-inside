"""safetensors 权重的内存映射索引。

设计目标是 7B/27B 级模型：启动时只读文件头，权重数据永远通过 mmap 按需读取，
不会整体载入内存。每个张量被看作一张二维图（h × w），并在一张"全景图"上分配位置。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import threading
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,  # numpy 没有 bfloat16，按 uint16 读出后再转换
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}

# 没有层号的"全局"张量（如词嵌入）如果又高又窄，就按行切成几段并排显示，
# 否则一个 15 万行的词表会把全景图拉得极长。
_SPLIT_ASPECT = 4
_SCALE_SAMPLES = 1024  # 估计数值范围时每个方向最多采样的行/列数


@dataclass
class TensorInfo:
    """全景图上的一张图：一个完整的张量，或者被切段的张量中的一段。"""

    id: int
    name: str  # 显示用的名字；切段时带行号范围
    source: str  # 权重文件里的张量名
    row0: int  # 这一段在原张量里的起始行
    dtype: str
    file: Path
    offset: int  # 这一段数据在文件中的绝对字节偏移
    h: int
    w: int
    layer: int | None  # 名字里的第一个数字段，视为层号
    role: str  # 把层号换成 * 之后的名字，同 role 的张量可以叠加
    x: int = 0  # 在全景图中的位置
    y: int = 0


def _natural_key(name: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def _split_layer(name: str) -> tuple[int | None, str]:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p.isdigit():
            return int(p), ".".join(parts[:i] + ["*"] + parts[i + 1 :])
    return None, name


def _shape_2d(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        return 1, shape[0]
    return shape[0], int(np.prod(shape[1:]))


def _segment_rows(h: int, w: int, grid_h: int) -> int:
    """又高又窄的张量每段放多少行：有分层结构时和各层排成的网格等高，否则让各段拼起来接近方形。"""
    target = max(grid_h or int((h * w) ** 0.5), w)
    segments = -(-h // target)
    return -(-h // segments)


def _shelf_pack(sizes: list[tuple[int, int]], gap: int, max_w: int):
    """货架式排布一组 (h, w)：从左到右放，超过 max_w 就另起一行。返回 (各项的 (x, y), 总宽, 总高)。"""
    pos = []
    x = y = shelf_h = width = 0
    for h, w in sizes:
        if x and x + w > max_w:
            x, y, shelf_h = 0, y + shelf_h + gap, 0
        pos.append((x, y))
        x += w + gap
        shelf_h = max(shelf_h, h)
        width = max(width, x - gap)
    return pos, width, y + shelf_h


def _grid_columns(n: int, cell_w: int, cell_h: int, aspect: float = 1.6) -> int:
    """n 个同样大的格子每行放几个，整体宽高比最接近屏幕。"""
    return min(range(1, n + 1), key=lambda c: abs(math.log(c * cell_w / (-(-n // c) * cell_h) / aspect)))


def bf16_to_f32(a: np.ndarray) -> np.ndarray:
    """bfloat16 就是 float32 的高 16 位。"""
    return (a.astype(np.uint32) << 16).view(np.float32)


class ModelStore:
    def __init__(self, path: str | Path):
        path = Path(path)
        if path.is_dir():
            files = sorted(path.glob("*.safetensors"))
            self.model_dir: Path | None = path
            self.name = path.name
        else:
            files = [path]
            self.model_dir = path.parent
            self.name = path.stem
        if not files or not all(f.is_file() for f in files):
            raise FileNotFoundError(f"在 {path} 没有找到 .safetensors 文件")

        self.files = files
        self._maps: dict[Path, np.memmap] = {}
        self._lock = threading.Lock()
        self._scale_lock = threading.RLock()
        self._scales: dict[str, float] = {}
        self._shared_scales: dict[int, float] = {}

        entries = []
        for f in files:
            entries.extend(self._read_header(f))
        entries.sort(key=lambda e: _natural_key(e[0]))
        self.sources = {name: shape for name, _, shape, _, _ in entries}  # 原始张量名 -> 形状

        dims = sorted(max(_shape_2d(shape)) for shape in self.sources.values())
        self.gap = max(16, int(0.08 * dims[len(dims) // 2]))

        # 每层的张量先在层内紧凑排布成一个"块"，所有块再排成接近屏幕比例的网格
        by_layer: dict[int, list[tuple[int, int]]] = {}
        for name, shape in self.sources.items():
            layer, _ = _split_layer(name)
            if layer is not None:
                by_layer.setdefault(layer, []).append(_shape_2d(shape))
        self._blocks = {
            layer: _shelf_pack(sizes, self.gap, max(w for _, w in sizes)) for layer, sizes in sorted(by_layer.items())
        }
        self._cell_w = max((b[1] for b in self._blocks.values()), default=0) + 2 * self.gap
        block_h = max((b[2] for b in self._blocks.values()), default=0)
        self._cell_h = block_h + max(2 * self.gap, block_h // 10)  # 行间留出标层号的位置
        self._per_row = _grid_columns(len(self._blocks), self._cell_w, self._cell_h) if self._blocks else 1
        grid_rows = -(-len(self._blocks) // self._per_row)
        grid_h = (grid_rows - 1) * self._cell_h + block_h if self._blocks else 0

        self.tensors: list[TensorInfo] = []
        for name, dtype, shape, file, offset in entries:
            h, w = _shape_2d(shape)
            layer, role = _split_layer(name)
            rows = _segment_rows(h, w, grid_h) if layer is None and h > _SPLIT_ASPECT * w else h
            row_bytes = w * np.dtype(_DTYPES[dtype]).itemsize
            for row0 in range(0, h, rows):
                n = min(rows, h - row0)
                label = name if rows == h else f"{name} [行 {row0}–{row0 + n - 1}]"
                self.tensors.append(
                    TensorInfo(len(self.tensors), label, name, row0, dtype, file, offset + row0 * row_bytes, n, w, layer, role)
                )
        self.total_params = sum(t.h * t.w for t in self.tensors)

        sig = "|".join(f"{f.resolve()}:{f.stat().st_size}:{f.stat().st_mtime_ns}" for f in files)
        self.model_id = hashlib.sha1(sig.encode()).hexdigest()[:12]  # 只认权重文件，录制文件靠它对应模型
        # 瓦片缓存按张量编号存放，而编号取决于怎么切段，所以缓存标识要把切段结果也算进去
        pieces = repr([(t.source, t.row0, t.h, t.w) for t in self.tensors])
        self.view_id = hashlib.sha1(("atlas-v2:" + sig + pieces).encode()).hexdigest()[:12]

        self._layout()
        self.world_w = max(t.x + t.w for t in self.tensors)
        self.world_h = max(t.y + t.h for t in self.tensors)
        self.groups = self._build_groups()
        self._group_for = {i: g for g in self.groups for i in g["members"]}

    # ---- 文件头 ----

    @staticmethod
    def _read_header(file: Path):
        with open(file, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        base = 8 + n
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if info["dtype"] not in _DTYPES:
                continue
            shape = tuple(info["shape"])
            if int(np.prod(shape)) == 0:
                continue
            yield name, info["dtype"], shape, file, base + info["data_offsets"][0]

    # ---- 布局 ----

    def _layout(self):
        """左边是全局张量（切段的各段并排），右边是各层的块排成的网格。"""
        x = 0
        for t in self.tensors:
            if t.layer is None:
                t.x, t.y = x, 0
                x += t.w + self.gap
        x += self.gap if x else 0
        for i, (layer, (pos, _, _)) in enumerate(self._blocks.items()):
            bx = x + (i % self._per_row) * self._cell_w
            by = (i // self._per_row) * self._cell_h
            for t, (tx, ty) in zip([t for t in self.tensors if t.layer == layer], pos):
                t.x, t.y = bx + tx, by + ty

    def _build_groups(self) -> list[dict]:
        """同 role 且形状相同的张量组成一组，可以像素对齐地叠加。"""
        by_key: dict[tuple, list[TensorInfo]] = {}
        for t in self.tensors:
            if t.layer is not None:
                by_key.setdefault((t.role, t.h, t.w), []).append(t)
        groups = []
        for (role, h, w), members in by_key.items():
            if len(members) >= 2:
                members.sort(key=lambda t: t.layer)
                groups.append({"id": len(groups), "role": role, "h": h, "w": w, "members": [t.id for t in members]})
        return groups

    # ---- 读数据 ----

    def _map(self, file: Path) -> np.memmap:
        with self._lock:
            m = self._maps.get(file)
            if m is None:
                m = self._maps[file] = np.memmap(file, dtype=np.uint8, mode="r")
            return m

    def _stored(self, t: TensorInfo) -> np.ndarray:
        dt = np.dtype(_DTYPES[t.dtype])
        raw = self._map(t.file)[t.offset : t.offset + t.h * t.w * dt.itemsize]
        return raw.view(dt).reshape(t.h, t.w)

    def _to_f32(self, t: TensorInfo, block: np.ndarray) -> np.ndarray:
        if t.dtype == "BF16":
            return bf16_to_f32(np.ascontiguousarray(block))
        return np.asarray(block, dtype=np.float32)

    def read(self, t: TensorInfo, r0: int, r1: int, c0: int, c1: int, step: int = 1) -> np.ndarray:
        """读取一块区域，返回 float32 副本。step > 1 时每隔 step 个取一个。"""
        return self._to_f32(t, self._stored(t)[r0:r1:step, c0:c1:step])

    def original_index(self, t: TensorInfo, r: int, c: int) -> tuple[int, ...]:
        """图上的 (r, c) 对应的原始张量下标。"""
        shape = self.sources[t.source]
        if len(shape) <= 1:
            return (c,) if shape else ()
        return (t.row0 + r, *[int(i) for i in np.unravel_index(c, shape[1:])])

    def _source_rms(self, t: TensorInfo) -> float:
        # 在原始张量上取样：切成几个显示块不应改变同一张表的颜色。
        if t.source not in self._scales:
            h, w = _shape_2d(self.sources[t.source])
            row_bytes = w * np.dtype(_DTYPES[t.dtype]).itemsize
            whole = replace(t, h=h, offset=t.offset - t.row0 * row_bytes)
            a = self._stored(whole)[:: -(-h // _SCALE_SAMPLES), :: -(-w // _SCALE_SAMPLES)]
            v = self._to_f32(t, a)
            v = v[np.isfinite(v)].astype(np.float64)
            self._scales[t.source] = float(np.sqrt(np.mean(v * v))) if v.size else 0.0
        return self._scales[t.source]

    def scale(self, t: TensorInfo, scope: str = "tensor") -> float:
        """单张量 RMS，或同角色且同形状的所有层合并后的 RMS（估计值）。"""
        group = self._group_for.get(t.id) if scope == "role" else None
        with self._scale_lock:
            if group is None:
                return self._source_rms(t) or 1.0
            gid = group["id"]
            if gid not in self._shared_scales:
                squares = [self._source_rms(self.tensors[i]) ** 2 for i in group["members"]]
                self._shared_scales[gid] = math.sqrt(sum(squares) / len(squares)) or 1.0
            return self._shared_scales[gid]
