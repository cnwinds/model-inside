"""HTTP 服务：模型元信息、权重瓦片、叠加瓦片、录制与回放瓦片。"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, tiles
from .recorder import Recorder
from .store import ModelStore

STATIC = Path(__file__).parent / "static"
_PNG_HEADERS = {"Cache-Control": "public, max-age=86400"}


class RecordRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=16, ge=0, le=256)


def create_app(model_path: str | Path, data_dir: str | Path = ".", dtype: str = "float32") -> FastAPI:
    data_dir = Path(data_dir)
    store = ModelStore(model_path)
    cache = tiles.TileCache(data_dir / ".cache" / store.view_id)
    recorder = Recorder(store, data_dir / "recordings", dtype=dtype)

    app = FastAPI(title="model-inside", version=__version__)

    def tensor(tid: int):
        if not 0 <= tid < len(store.tensors):
            raise HTTPException(404, "没有这个张量")
        return store.tensors[tid]

    def png(data: bytes | None) -> Response:
        if data is None:
            raise HTTPException(404, "瓦片超出范围")
        return Response(data, media_type="image/png", headers=_PNG_HEADERS)

    @app.get("/api/model")
    def model_info():
        return {
            "id": store.view_id,
            "name": store.name,
            "total_params": store.total_params,
            "source_count": len(store.sources),
            "world": {"w": store.world_w, "h": store.world_h},
            "gap": store.gap,
            "tile": tiles.TILE,
            "can_record": recorder.available,
            "tensors": [
                {
                    "id": t.id,
                    "name": t.name,
                    "source": t.source,
                    "dtype": t.dtype,
                    "h": t.h,
                    "w": t.w,
                    "x": t.x,
                    "y": t.y,
                    "zmax": tiles.zmax(t.h, t.w),
                    "lodmax": tiles.lodmax(t.h, t.w),
                    "layer": t.layer,
                }
                for t in store.tensors
            ],
            "groups": [{**g, "zmax": tiles.zmax(g["h"], g["w"]), "lodmax": tiles.lodmax(g["h"], g["w"])} for g in store.groups],
        }

    @app.get("/api/tile/{tid}/{z}/{x}/{y}.png")
    def weight_tile(tid: int, z: int, x: int, y: int, mode: str = "maxabs", range: float = 3.0, scope: str = "tensor"):
        t = tensor(tid)
        if mode not in ("rms", "maxabs", "mean") or scope not in ("role", "tensor") or not 0.1 <= range <= 100:
            raise HTTPException(400, "参数不合法")
        key = f"w/{tid}/{mode}-{scope}-{range:g}/{z}_{x}_{y}"
        return png(cache.get(key, z, lambda: tiles.render_weight_tile(store, t, z, x, y, mode, range, scope)))

    @app.get("/api/stack/{gid}/{z}/{x}/{y}.png")
    def stack_tile(gid: int, z: int, x: int, y: int, agg: str = "mean", range: float = 3.0):
        if not 0 <= gid < len(store.groups):
            raise HTTPException(404, "没有这个分组")
        if agg not in ("mean", "std", "maxabs") or not 0.1 <= range <= 100:
            raise HTTPException(400, "参数不合法")
        members = [store.tensors[i] for i in store.groups[gid]["members"]]
        key = f"s/{gid}/{agg}-{range:g}/{z}_{x}_{y}"
        # 跨层聚合要读 N 层数据，任何级别都值得落盘
        return png(cache.get(key, 2, lambda: tiles.render_stack_tile(store, members, z, x, y, agg, range)))

    @app.get("/api/block/{tid}")
    def block(tid: int, z: int, r: int, c: int, scope: str = "tensor"):
        t = tensor(tid)
        if scope not in ("tensor", "role") or not 0 <= z <= tiles.lodmax(t.h, t.w):
            raise HTTPException(400, "参数不合法")
        if not (0 <= r < t.h and 0 <= c < t.w):
            raise HTTPException(404, "坐标超出范围")
        return tiles.inspect_block(store, t, z, r, c, scope)

    @app.get("/api/value/{tid}")
    def value(tid: int, r: int, c: int):
        t = tensor(tid)
        if not (0 <= r < t.h and 0 <= c < t.w):
            raise HTTPException(404, "坐标超出范围")
        return {"value": float(store.read(t, r, r + 1, c, c + 1)[0, 0]), "index": store.original_index(t, r, c)}

    @app.get("/api/recordings")
    def recordings():
        return recorder.list()

    @app.post("/api/record")
    def record(req: RecordRequest):
        if not recorder.available:
            raise HTTPException(400, "模型目录里没有 config.json，无法运行模型")
        try:
            return recorder.record(req.prompt, req.max_new_tokens)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/act/{rec_id}/{step}/{tid}/{z}/{x}/{y}.png")
    def activation_tile(rec_id: str, step: int, tid: int, z: int, x: int, y: int, thr: float = 3.0):
        t = tensor(tid)
        rec = recorder.get(rec_id)
        got = rec.input(t.source, step) if rec else None
        if got is None or not 0 <= thr <= 1000:
            raise HTTPException(404, "没有这一步的记录")
        vec, axis = got
        if axis == 0:  # 每行一个输入：被切段的张量只取这一段对应的行
            vec = vec[t.row0 : t.row0 + t.h]
        if vec.shape[0] != (t.w if axis == 1 else t.h):
            raise HTTPException(409, "记录与权重形状不匹配")
        key = f"a/{rec_id}/{step}/{tid}/{thr:g}/{z}_{x}_{y}"
        make = lambda: tiles.render_activation_tile(store, t, vec, axis, z, x, y, thr)  # noqa: E731
        return png(cache.get(key, z, make, persist=False))

    @app.get("/api/attention/{rec_id}/{layer}")
    def attention(rec_id: str, layer: int):
        """某一层所有头的注意力：uint8，形状 (头, 查询位置, 被看位置)，值 = round(255 · √权重)。"""
        rec = recorder.get(rec_id)
        data = rec.attention_layer(layer) if rec else None
        if data is None:
            raise HTTPException(404, "这条录制没有这一层的注意力")
        return Response(data.tobytes(), media_type="application/octet-stream", headers=_PNG_HEADERS)

    @app.get("/api/lens/{rec_id}")
    def lens(rec_id: str):
        rec = recorder.get(rec_id)
        path = rec.lens_file() if rec else None
        if path is None:
            raise HTTPException(404, "这条录制没有逐层预测")
        return FileResponse(path, media_type="application/json", headers=_PNG_HEADERS)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
