import io
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from safetensors.torch import save_file

from model_inside import tiles
from model_inside.server import create_app
from model_inside.store import ModelStore


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """一个小的假模型：两层同形状的线性层、一个又高又窄的全局张量、三种浮点格式。"""
    g = torch.Generator().manual_seed(0)
    src = {
        "embed.weight": torch.randn(700, 40, generator=g).to(torch.bfloat16),
        "layers.0.fc.weight": torch.randn(300, 520, generator=g),
        "layers.1.fc.weight": torch.randn(300, 520, generator=g),
        "layers.0.fc.bias": torch.randn(300, generator=g).to(torch.float16),
        "layers.1.fc.bias": torch.randn(300, generator=g).to(torch.float16),
        "conv.weight": torch.randn(4, 3, 2, 2, generator=g),
    }
    path = tmp_path_factory.mktemp("model") / "model.safetensors"
    save_file(src, str(path))
    return ModelStore(path), {k: v.float().numpy() for k, v in src.items()}


def pick(store, name):
    (t,) = [t for t in store.tensors if t.source == name]
    return t


def test_reads_match_source_for_all_dtypes(model):
    store, src = model
    for name in ("layers.0.fc.weight", "layers.1.fc.bias"):
        t = pick(store, name)
        np.testing.assert_array_equal(store.read(t, 0, t.h, 0, t.w), src[name].reshape(t.h, t.w))


def test_tall_global_tensor_is_split_into_side_by_side_segments(model):
    store, src = model
    segs = [t for t in store.tensors if t.source == "embed.weight"]
    column_h = max(t.y + t.h for t in store.tensors if t.layer is not None)
    assert len(segs) == 3 and all(t.h <= column_h and t.w == 40 for t in segs)
    assert [t.row0 for t in segs] == [0, 234, 468] and sum(t.h for t in segs) == 700
    assert len({t.y for t in segs}) == 1 and len({t.x for t in segs}) == 3  # 并排，不是上下堆叠

    np.testing.assert_array_equal(np.vstack([store.read(t, 0, t.h, 0, t.w) for t in segs]), src["embed.weight"])
    np.testing.assert_array_equal(store.read(segs[2], 3, 9, 10, 35), src["embed.weight"][471:477, 10:35])
    assert store.original_index(segs[1], 5, 7) == (239, 7)
    assert store.total_params == sum(v.size for v in src.values())  # 切段不改变权重总数


def test_many_layers_wrap_into_a_screen_shaped_grid(tmp_path):
    g = torch.Generator().manual_seed(0)
    src = {"embed.weight": torch.randn(5000, 64, generator=g)}
    for i in range(24):
        src[f"layers.{i}.down.weight"] = torch.randn(64, 320, generator=g)
        src[f"layers.{i}.gate.weight"] = torch.randn(320, 64, generator=g)
        src[f"layers.{i}.up.weight"] = torch.randn(320, 64, generator=g)
    save_file(src, str(tmp_path / "model.safetensors"))
    store = ModelStore(tmp_path)

    layered = [t for t in store.tensors if t.layer is not None]
    assert 1 < len({t.y for t in layered if t.role == "layers.*.down.weight"}) < 24  # 折成了多行
    assert 1.0 < store.world_w / store.world_h < 2.6  # 整体接近屏幕比例，而不是一长条
    # 层内：两个又高又窄的矩阵并排放在同一行，而不是上下堆叠
    gate, up = pick(store, "layers.3.gate.weight"), pick(store, "layers.3.up.weight")
    assert gate.y == up.y and gate.x != up.x
    # 词嵌入切成的各段与整个网格等高
    grid_h = max(t.y + t.h for t in layered)
    segs = [t for t in store.tensors if t.source == "embed.weight"]
    assert len(segs) > 1 and max(t.h for t in segs) <= grid_h

    rects = [(t.x, t.y, t.x + t.w, t.y + t.h) for t in store.tensors]
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def test_tile_cache_is_invalidated_when_segmentation_changes(model, monkeypatch):
    store, _ = model
    monkeypatch.setattr("model_inside.store._SPLIT_ASPECT", 1000)  # 不再切段，张量编号随之改变
    unsplit = ModelStore(store.files[0])
    assert len(unsplit.tensors) < len(store.tensors)
    assert unsplit.view_id != store.view_id  # 瓦片缓存不能混用
    assert unsplit.model_id == store.model_id  # 录制文件仍然对得上


def test_4d_tensor_is_flattened_and_index_maps_back(model):
    store, src = model
    t = pick(store, "conv.weight")
    assert (t.h, t.w) == (4, 12)
    idx = store.original_index(t, 2, 7)
    assert store.read(t, 2, 3, 7, 8)[0, 0] == src["conv.weight"][idx]


def test_layout_has_no_overlaps(model):
    store, _ = model
    rects = [(t.x, t.y, t.x + t.w, t.y + t.h) for t in store.tensors]
    for i, a in enumerate(rects):
        assert a[2] <= store.world_w and a[3] <= store.world_h
        for b in rects[i + 1 :]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def test_level0_tile_is_one_pixel_per_weight(model):
    store, src = model
    t = pick(store, "layers.0.fc.weight")
    v = tiles.weight_values(store, t, 0, 1, 1, "maxabs")  # 右下角那块，尺寸不足 256
    np.testing.assert_array_equal(v, src[t.name][256:300, 256:512])
    assert tiles.weight_values(store, t, 0, 3, 0, "maxabs") is None


@pytest.mark.parametrize("mode", ["maxabs", "mean", "rms"])
def test_zoomed_out_tile_matches_brute_force(model, mode, monkeypatch):
    monkeypatch.setattr(tiles, "_BAND_ELEMS", 3000)  # 强制分多带读取
    store, src = model
    t = pick(store, "layers.0.fc.weight")
    a, s = src[t.name], 4
    v = tiles.weight_values(store, t, 2, 0, 0, mode)
    assert v.shape == (75, 130)
    for i, j in [(0, 0), (10, 57), (74, 129)]:
        blk = a[i * s : (i + 1) * s, j * s : (j + 1) * s]
        want = {"mean": blk.mean(), "maxabs": blk.flat[np.abs(blk).argmax()], "rms": np.sqrt(np.mean(blk**2))}[mode]
        assert v[i, j] == pytest.approx(want, rel=1e-5)


def test_ragged_edge_block_mean_ignores_padding():
    a = np.arange(1, 8, dtype=np.float32).reshape(1, 7)
    np.testing.assert_allclose(tiles.block_reduce(a, 4, "mean"), [[2.5, 6.0]])
    np.testing.assert_array_equal(tiles.block_reduce(-a, 4, "maxabs"), [[-4, -7]])


@pytest.mark.parametrize("agg", ["mean", "std", "maxabs"])
def test_stack_aggregates_across_layers(model, agg):
    store, src = model
    (group,) = [g for g in store.groups if g["role"] == "layers.*.fc.weight"]
    members = [store.tensors[i] for i in group["members"]]
    cube = np.stack([src[t.name] for t in members])
    v = tiles.stack_values(store, members, 0, 0, 0, agg)
    want = {
        "mean": cube.mean(0),
        "std": cube.std(0),
        "maxabs": np.where(np.abs(cube[0]) >= np.abs(cube[1]), cube[0], cube[1]),
    }[agg][:256, :256]
    np.testing.assert_allclose(v, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("axis", [0, 1])
def test_activation_marks_weights_whose_contribution_exceeds_threshold(model, axis):
    store, src = model
    t = pick(store, "layers.0.fc.weight")
    w = src[t.name]
    vec = np.random.default_rng(1).normal(size=t.w if axis == 1 else t.h).astype(np.float32)
    thr = 2.0
    alpha = tiles.activation_alpha(store, t, vec, axis, 0, 0, 0, thr)

    x = np.abs(vec)[None, :256] if axis == 1 else np.abs(vec)[:256, None]
    unit = store.scale(t) * float(np.sqrt(np.mean(vec.astype(np.float64) ** 2)))
    ratio = np.abs(w[:256, :256]) * x / unit
    np.testing.assert_array_equal(alpha > 0, ratio > thr)
    assert 0 < (alpha > 0).mean() < 0.5
    # 贡献越大越亮
    lit = alpha > 0
    assert alpha[lit][ratio[lit].argmax()] == alpha.max()

    # 缩小一级后：亮度是块内的平均，全暗的块仍然全暗
    alpha2 = tiles.activation_alpha(store, t, vec, axis, 1, 0, 0, thr)
    dark = ~lit[:256, :256].reshape(128, 2, 128, 2).any(axis=(1, 3))
    np.testing.assert_array_equal(alpha2[:128, :128] == 0, dark)


def test_far_zoomed_out_activation_samples_a_fixed_grid_per_pixel(model):
    store, src = model
    t = pick(store, "layers.0.fc.weight")
    vec = np.random.default_rng(2).normal(size=t.w).astype(np.float32)
    alpha = tiles.activation_alpha(store, t, vec, 1, 3, 0, 0, 2.0)  # 一个像素 = 8×8 个权重
    assert alpha.shape == (38, 65)

    # 每隔一个取一个，即每个像素 4×4 个样本
    unit = store.scale(t) * float(np.sqrt(np.mean(vec.astype(np.float64) ** 2)))
    ratio = np.abs(src[t.name][::2, ::2]) * np.abs(vec)[None, ::2] / unit
    assert (alpha[:37, :65] > 0).tolist() == (ratio[:148] > 2.0).reshape(37, 4, 65, 4).any(axis=(1, 3)).tolist()

    # 采样估计出的整体亮度应和逐个权重精确计算的接近
    exact = tiles.activation_alpha(store, t, vec, 1, 0, 0, 0, 2.0)
    assert (alpha[:32, :32] ** 2).mean() == pytest.approx((exact**2).mean(), rel=0.25)


def test_api_serves_model_tiles_and_values(model, tmp_path):
    store, src = model
    client = TestClient(create_app(store.files[0], tmp_path))
    info = client.get("/api/model").json()
    assert info["total_params"] == sum(v.size for v in src.values())
    by_name = {t["name"]: t for t in info["tensors"]}
    t = by_name["layers.1.fc.weight"]
    assert t["zmax"] == 2

    r = client.get(f"/api/tile/{t['id']}/0/1/0.png")
    assert r.status_code == 200
    assert Image.open(io.BytesIO(r.content)).size == (256, 256)
    r = client.get(f"/api/tile/{t['id']}/2/0/0.png")
    assert Image.open(io.BytesIO(r.content)).size == (130, 75)
    assert (tmp_path / ".cache").is_dir()  # 缩小级别落盘
    assert client.get(f"/api/tile/{t['id']}/0/9/9.png").status_code == 404

    v = client.get(f"/api/value/{t['id']}", params={"r": 7, "c": 500}).json()
    assert v == {"value": float(src["layers.1.fc.weight"][7, 500]), "index": [7, 500]}

    gid = next(g["id"] for g in info["groups"] if g["role"] == "layers.*.fc.weight")
    assert client.get(f"/api/stack/{gid}/0/0/0.png?agg=std").status_code == 200
    assert info["can_record"] is False


def test_api_replays_recorded_activations_on_the_right_weights(model, tmp_path):
    store, _ = model
    rng = np.random.default_rng(3)
    inputs = {"embed.weight": (0, rng.normal(size=(2, 700))), "layers.0.fc.weight": (1, rng.normal(size=(2, 520)))}
    rec_dir = tmp_path / "recordings" / "rec1"
    rec_dir.mkdir(parents=True)
    modules = []
    for i, (name, (in_axis, arr)) in enumerate(inputs.items()):
        np.save(rec_dir / f"m{i}.npy", arr.astype(np.float16))
        modules.append({"tensor": name, "in_axis": in_axis, "file": f"m{i}.npy"})
    meta = {"id": "rec1", "model_id": store.model_id, "prompt": "p", "n_prompt": 1, "tokens": ["a", "b"], "modules": modules}
    (rec_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    client = TestClient(create_app(store.files[0], tmp_path))
    assert [r["id"] for r in client.get("/api/recordings").json()] == ["rec1"]

    def lit(t, step):
        r = client.get(f"/api/act/rec1/{step}/{t.id}/0/0/0.png", params={"thr": 1.5})
        assert r.status_code == 200
        return np.asarray(Image.open(io.BytesIO(r.content)).getchannel("A")) > 0

    # 每行一个输入的张量被切段后，第三段要用输入向量里它自己那几行
    seg = [t for t in store.tensors if t.source == "embed.weight"][2]
    vec = inputs["embed.weight"][1][1].astype(np.float16).astype(np.float32)
    want = tiles.activation_alpha(store, seg, vec[seg.row0 : seg.row0 + seg.h], 0, 0, 0, 0, 1.5) > 0
    assert want.any() and not want.all()
    np.testing.assert_array_equal(lit(seg, 1), want)

    # 每列一个输入的普通线性层
    t = pick(store, "layers.0.fc.weight")
    vec = inputs[t.source][1][0].astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(lit(t, 0), tiles.activation_alpha(store, t, vec, 1, 0, 0, 0, 1.5) > 0)

    assert client.get(f"/api/act/rec1/2/{t.id}/0/0/0.png").status_code == 404  # 没有这一步
    assert client.get(f"/api/act/rec1/0/{pick(store, 'conv.weight').id}/0/0/0.png").status_code == 404  # 这层没录
    assert client.get(f"/api/act/..%2F..%2Fx/0/{t.id}/0/0/0.png").status_code == 404


@pytest.mark.parametrize("mode", ["rms", "mean", "maxabs"])
def test_coarse_lod_stays_in_budget_and_respects_ragged_edges(monkeypatch, mode):
    a = np.arange(-226, 225, dtype=np.float32).reshape(41, 11)
    a[2, 0], a[30, 1] = -999, 999  # 极值绝对值相同，跨子带也要取正值
    monkeypatch.setattr(tiles, "_BAND_ELEMS", 22)
    def read(i, j):
        assert j - i <= 2
        return a[i:j]
    got = tiles._reduce_region((0, 41, 0, 11), 32, mode, read)
    np.testing.assert_allclose(got, tiles.block_reduce(a, 32, mode), rtol=1e-7)
    if mode == "maxabs":
        assert got[0, 0] == 999


def test_shared_rms_and_tile_cache_scope(tmp_path):
    src = {f"layers.{i}.fc.weight": torch.full((12, 11), v) for i, v in enumerate([0., -1., 4.])}
    save_file(src, str(tmp_path / "weights.safetensors"))
    store = ModelStore(tmp_path)
    t = pick(store, "layers.1.fc.weight")
    shared = np.sqrt(17 / 3)
    for member in store.tensors:
        assert store.scale(member, "role") == pytest.approx(shared)
    client = TestClient(create_app(tmp_path, tmp_path))
    for scope, unit in [("role", shared), ("tensor", 1), ("role", shared)]:
        r = client.get(f"/api/tile/{t.id}/2/0/0.png?mode=rms&range=2&scope={scope}")
        assert r.status_code == 200
        expected = tiles.colorize(np.ones((3, 3)) / (2 * unit), signed=False)
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(r.content))), expected)


def test_inspector_matches_visible_block_and_source_coordinates(model, tmp_path):
    store, src = model
    segs = [t for t in store.tensors if t.source == "embed.weight"]
    t = segs[-1]
    client = TestClient(create_app(store.files[0], tmp_path))
    info = client.get("/api/model").json()
    assert info["source_count"] == len(src) < len(info["tensors"])
    for seg in segs:
        assert store.scale(seg) == pytest.approx(np.sqrt((src[t.source]**2).mean()))
    z, r, c = 5, t.h - 1, 39  # 不足 32×32 的右下角块；z 超过塞入一个瓦片所需级别
    b = client.get(f"/api/block/{t.id}?r={r}&c={c}&z={z}&scope=role").json()
    a = src[t.source][t.row0 + r//32*32:t.row0 + t.h, 32:40]
    assert b["rows"] == [t.row0 + r//32*32, t.row0 + t.h - 1]
    assert b["cols"] == [32,39] and b["count"] == a.size and b["index"] is None
    assert b["rms"] == pytest.approx(np.sqrt(np.mean(a**2)))
    for mode in ("rms", "mean", "maxabs"):
        v = tiles.weight_values(store, t, z, 0, 0, mode)
        assert b[mode] == pytest.approx(v[-1, -1])
    single = client.get(f"/api/block/{t.id}?r=1&c=2&z=0").json()
    assert single["index"] == [t.row0+1, 2] and single["mean"] == float(src[t.source][t.row0+1, 2])
    for params, code in [("z=-1&r=0&c=0",400),("z=99&r=0&c=0",400),("z=0&r=-1&c=0",404),("z=0&r=0&c=0&scope=bad",400)]:
        assert client.get(f"/api/block/{t.id}?{params}").status_code == code


def test_colors_distinguish_sign_and_magnitude():
    rgb = tiles.colorize(np.array([[-1.0, -0.2, 0.0, 0.2, 1.0]]))[0].astype(int)
    assert rgb[0][2] > rgb[0][0] and rgb[4][0] > rgb[4][2]  # 负值偏蓝，正值偏红
    assert rgb[1].sum() < rgb[0].sum() and rgb[3].sum() < rgb[4].sum()
    assert rgb[2].sum() < 60  # 零接近黑
