"""对正在运行的服务做一次端到端检查并计时。

用法: python scripts/smoke_test.py [http://127.0.0.1:8000] [--atlas] [--record]
"""
import io
import json
import sys
import time
import urllib.request

from PIL import Image

base = next((a for a in sys.argv[1:] if a.startswith("http")), "http://127.0.0.1:8000")
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 本机服务，不走代理


def get(path, data=None):
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    body = opener.open(req, timeout=600).read()
    return body, time.perf_counter() - t


def tile(path):
    body, dt = get(path)
    img = Image.open(io.BytesIO(body))
    return img, dt, len(body)


def main():
    info = json.loads(get("/api/model")[0])
    print(f"模型 {info['name']}: {info['total_params']:,} 个权重, {info['source_count']} 个张量, {len(info['tensors'])} 个显示块, "
          f"{len(info['groups'])} 个可叠加分组, 全景 {info['world']['w']:,} × {info['world']['h']:,}")

    biggest = max(info["tensors"], key=lambda t: t["h"] * t["w"])
    for label, t in (("最大的张量", biggest), ("第一个分层张量", next(t for t in info["tensors"] if t["layer"] is not None))):
        print(f"{label}: {t['name']} {t['h']}×{t['w']} zmax={t['zmax']}")
        for z in sorted({t["zmax"], 0}, reverse=True):
            img, dt, n = tile(f"/api/tile/{t['id']}/{z}/0/0.png?v=smoke{time.time()}")
            print(f"  z={z}: {img.size[0]}×{img.size[1]} {img.mode}  {dt * 1000:.0f} ms  {n / 1024:.0f} KB")

    g = max(info["groups"], key=lambda g: g["h"] * g["w"] * len(g["members"]))
    for agg in ("mean", "std", "maxabs"):
        img, dt, _ = tile(f"/api/stack/{g['id']}/{g['zmax']}/0/0.png?agg={agg}")
        print(f"叠加 {g['role']} × {len(g['members'])} 层 [{agg}]: {img.size[0]}×{img.size[1]}  {dt * 1000:.0f} ms")

    if "--atlas" in sys.argv:  # 打开全景视图时要算的东西：每张图最粗的一级
        total = 0.0
        for t in info["tensors"]:
            total += tile(f"/api/tile/{t['id']}/{t['zmax']}/0/0.png")[1]
        print(f"全景: {len(info['tensors'])} 张图的最粗一级共 {total:.1f} s（顺序请求；算过一次后走磁盘缓存）")

    if "--record" in sys.argv:
        body = json.dumps({"prompt": "The capital of France is", "max_new_tokens": 8}).encode()
        meta, dt = get("/api/record", body)
        meta = json.loads(meta)
        print(f"录制: {len(meta['tokens'])} 个位置, {len(meta['modules'])} 个线性层, {dt:.1f} s")
        print("  tokens:", meta["tokens"])
    else:
        recs = json.loads(get("/api/recordings")[0])
        meta = recs[0] if recs else None

    if meta and meta.get("attention"):
        a = meta["attention"]
        n = len(meta["tokens"])
        body, dt = get(f"/api/attention/{meta['id']}/{a['layers'] // 2}")
        assert len(body) == a["heads"] * n * n, "注意力数据大小和 (头, 位置, 位置) 对不上"
        print(f"注意力: {a['layers']} 层 × {a['heads']} 个头 × {n}×{n}，单层 {len(body) / 1024:.0f} KB, {dt * 1000:.0f} ms")
    if meta and meta.get("lens"):
        body, dt = get(f"/api/lens/{meta['id']}")
        lens = json.loads(body)
        last = lens["steps"] - 1
        answer = lens["vocab"][lens["final"][last]]
        ranks = [lens["final_rank"][l][last] for l in range(lens["layers"])]
        print(f"逐层预测: {lens['layers']} 层状态, {len(body) / 1024:.0f} KB, {dt * 1000:.0f} ms; "
              f"最后一个位置的答案 {answer!r} 的排名 输入层 {ranks[0]} → 最后一层 {ranks[-1]}")

    if meta:
        recorded = {m["tensor"] for m in meta["modules"]}
        shown = [t for t in info["tensors"] if t["source"] in recorded]
        step = len(meta["tokens"]) - 1
        total = 0.0
        for t in shown:
            img, dt, _ = tile(f"/api/act/{meta['id']}/{step}/{t['id']}/{t['zmax']}/0/0.png?thr=3")
            total += dt
            assert img.mode == "RGBA"
        print(f"回放: 第 {step} 步全部 {len(shown)} 张图的最粗一级激活图共 {total * 1000:.0f} ms")
        t = next(t for t in shown if t["layer"] is not None)
        img, dt, _ = tile(f"/api/act/{meta['id']}/{step}/{t['id']}/0/0/0.png?thr=3")
        lit = sum(1 for a in img.getchannel("A").getdata() if a) / (img.size[0] * img.size[1])
        print(f"  {t['name']} z=0 瓦片: 阈值 3 时 {lit:.1%} 的权重被点亮, {dt * 1000:.0f} ms")


if __name__ == "__main__":
    main()
