"""运行模型并记录三类数据，供回放使用：

1. 每个线性层在每个 token 位置的输入向量。"哪些权重被激活"在回放时由 |权重 × 输入| 按需算出，
   所以录制文件不会随权重数量膨胀。
2. 每层每个注意力头的注意力矩阵：每个词在看前面的哪些词。
3. 逐层预测（logit lens）：把每一层之后的中间状态直接当作最终状态去查词表，看模型"此刻最想输出什么词"，
   以及最终答案在每一层的概率和排名。

录制文件是独立的：meta.json + 若干 .npy/.json，可以在别的机器上录好再拷过来回放。
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import numpy as np

from .store import ModelStore

_ID_RE = re.compile(r"^[\w\-]+$")
_LENS_TOP_K = 5
_MAX_ATTENTION_BYTES = 256 << 20  # 注意力矩阵随序列长度平方增长，太大就不存


def _sig(x) -> float:
    """保留 4 位有效数字：概率可能小到 1e-7，固定小数位会把它们全变成 0。"""
    return float(f"{float(x):.4g}")


def quantize_attention(a: np.ndarray) -> np.ndarray:
    """注意力权重 0..1 → uint8。先开方再量化，小权重也有分辨率；前端按平方还原。"""
    return np.rint(np.sqrt(a.astype(np.float32)) * 255).astype(np.uint8)


class Recording:
    def __init__(self, path: Path):
        self.path = path
        self.meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        self.modules = {m["tensor"]: m for m in self.meta["modules"]}
        self._arrays: dict[str, np.ndarray] = {}

    def _array(self, file: str) -> np.ndarray:
        arr = self._arrays.get(file)
        if arr is None:
            arr = self._arrays[file] = np.load(self.path / file, mmap_mode="r")
        return arr

    def input(self, tensor_name: str, step: int) -> tuple[np.ndarray, int] | None:
        """返回 (该步的输入向量, 输入对应的存储轴)。"""
        m = self.modules.get(tensor_name)
        if m is None or not 0 <= step < len(self.meta["tokens"]):
            return None
        return np.asarray(self._array(m["file"])[step], dtype=np.float32), m["in_axis"]

    def attention_layer(self, layer: int) -> np.ndarray | None:
        """某一层所有头的注意力，uint8，形状 (头, 查询位置, 被看位置)。"""
        info = self.meta.get("attention")
        if info is None or not 0 <= layer < info["layers"]:
            return None
        return quantize_attention(self._array(info["file"])[layer])

    def lens_file(self) -> Path | None:
        info = self.meta.get("lens")
        return self.path / info["file"] if info else None


class Recorder:
    def __init__(self, store: ModelStore, root: Path, dtype: str = "float32"):
        self.store = store
        self.root = root
        self.dtype = dtype
        self._model = None
        self._tok = None
        self._targets: list[tuple[object, str, int]] = []  # (模块, 权重张量名, 输入轴)
        self._lock = threading.Lock()
        self._loaded: dict[str, Recording] = {}

    @property
    def available(self) -> bool:
        d = self.store.model_dir
        return d is not None and (d / "config.json").is_file()

    # ---- 读取 ----

    def list(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        out = []
        for p in sorted(self.root.iterdir(), reverse=True):
            rec = self.get(p.name)
            if rec is not None and rec.meta.get("model_id") == self.store.model_id:
                out.append(rec.meta)
        return out

    def get(self, rec_id: str) -> Recording | None:
        if not _ID_RE.match(rec_id):
            return None
        rec = self._loaded.get(rec_id)
        if rec is None:
            path = self.root / rec_id
            if not (path / "meta.json").is_file():
                return None
            rec = self._loaded[rec_id] = Recording(path)
        return rec

    # ---- 录制 ----

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        d = self.store.model_dir
        self._tok = AutoTokenizer.from_pretrained(d)
        dtype = getattr(torch, self.dtype)
        try:
            # 只有 eager 实现才会把注意力权重交出来
            self._model = AutoModelForCausalLM.from_pretrained(d, dtype=dtype, attn_implementation="eager").eval()
        except Exception:  # noqa: BLE001 个别架构不支持 eager，退回默认实现，只是没有注意力视图
            self._model = AutoModelForCausalLM.from_pretrained(d, dtype=dtype).eval()

        # 同一份权重可能有多个名字（如输出层与词嵌入共享），按内存地址归并后再去权重文件里找。
        aliases: dict[int, list[str]] = {}
        for name, p in self._model.named_parameters(remove_duplicate=False):
            aliases.setdefault(p.data_ptr(), []).append(name)

        for mod in self._model.modules():
            w = getattr(mod, "weight", None)
            if isinstance(mod, torch.nn.Linear):
                in_axis = 1  # 权重形状 (输出, 输入)
            elif type(mod).__name__ == "Conv1D" and w is not None and w.dim() == 2:
                in_axis = 0  # GPT-2 的 Conv1D 权重形状是 (输入, 输出)
            else:
                continue
            source = next((s for n in aliases.get(w.data_ptr(), []) if (s := self._resolve(n))), None)
            if source is not None and tuple(self.store.sources[source]) == tuple(w.shape):
                self._targets.append((mod, source, in_axis))

    def _resolve(self, param_name: str) -> str | None:
        """模型里的参数名和文件里的名字可能差一个前缀，按后缀匹配。"""
        sources = self.store.sources
        parts = param_name.split(".")
        for i in range(len(parts)):
            name = ".".join(parts[i:])
            if name in sources:
                return name
        hits = [name for name in sources if name.endswith("." + param_name)]
        return hits[0] if len(hits) == 1 else None

    def _lens_parts(self):
        """逐层预测需要的三样东西：解码器层列表、最终归一化、输出层。找不到就返回 None。"""
        model = self._model
        base = model.base_model
        layers = getattr(base, "layers", None) or getattr(base, "h", None)
        norm = next((getattr(base, n) for n in ("norm", "ln_f", "final_layernorm", "final_layer_norm") if hasattr(base, n)), None)
        head = model.get_output_embeddings()
        if not layers or norm is None or head is None:
            return None
        return layers, norm, head

    def record(self, prompt: str, max_new_tokens: int) -> dict:
        import torch

        with self._lock:
            self._load()
            tok, model = self._tok, self._model
            enc = tok(prompt, return_tensors="pt")
            ids = enc.input_ids
            n_prompt = ids.shape[1]
            if n_prompt == 0:
                raise ValueError("提示词为空")

            with torch.no_grad():
                if max_new_tokens > 0:
                    ids = model.generate(
                        ids,
                        attention_mask=enc.attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
                steps = ids.shape[1]

                captured: dict[str, np.ndarray] = {}

                def make_hook(name):
                    def hook(_mod, inputs, _out):
                        x = inputs[0].detach()
                        x = x.reshape(-1, x.shape[-1])
                        if x.shape[0] == steps:
                            captured[name] = x.float().clamp(-6e4, 6e4).numpy().astype(np.float16)

                    return hook

                handles = [mod.register_forward_hook(make_hook(name)) for mod, name, _ in self._targets]

                # 每层的输出（残差流）：第 0 项是第一层的输入，之后依次是每一层之后的状态
                lens = self._lens_parts()
                states: list = []
                if lens is not None:
                    layers, _, _ = lens

                    def first_in(_mod, args, kwargs):
                        states.append((args[0] if args else kwargs["hidden_states"]).detach()[0])

                    def layer_out(_mod, _args, out):
                        states.append((out[0] if isinstance(out, tuple) else out).detach()[0])

                    handles.append(layers[0].register_forward_pre_hook(first_in, with_kwargs=True))
                    handles += [layer.register_forward_hook(layer_out) for layer in layers]

                try:
                    out = model(ids, output_attentions=True)
                    lens_data = self._lens(states, lens, tok) if lens is not None and len(states) == len(lens[0]) + 1 else None
                finally:
                    for h in handles:
                        h.remove()

            attn = None
            if out.attentions:
                attn = torch.stack([a[0] for a in out.attentions]).float().numpy().astype(np.float16)  # (层, 头, 位置, 位置)
                if attn.nbytes > _MAX_ATTENTION_BYTES:
                    attn = None

            # ID 精确到秒；同一秒内再录一次就加序号，绝不覆盖已有的录制
            stamp = time.strftime("%Y%m%d-%H%M%S")
            rec_id, n = stamp, 1
            self.root.mkdir(parents=True, exist_ok=True)
            while True:
                path = self.root / rec_id
                try:
                    path.mkdir()
                    break
                except FileExistsError:
                    n += 1
                    rec_id = f"{stamp}-{n}"
            modules = []
            for i, (_, name, in_axis) in enumerate(self._targets):
                if name in captured:
                    np.save(path / f"m{i}.npy", captured[name])
                    modules.append({"tensor": name, "in_axis": in_axis, "file": f"m{i}.npy"})
            meta = {
                "id": rec_id,
                "model_id": self.store.model_id,
                "model": self.store.name,
                "prompt": prompt,
                "n_prompt": n_prompt,
                "tokens": [tok.decode([i]) for i in ids[0].tolist()],
                "modules": modules,
            }
            if attn is not None:
                np.save(path / "attn.npy", attn)
                meta["attention"] = {"file": "attn.npy", "layers": attn.shape[0], "heads": attn.shape[1]}
            if lens_data is not None:
                (path / "lens.json").write_text(json.dumps(lens_data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
                meta["lens"] = {"file": "lens.json", "layers": len(states)}
            (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            return meta

    def _lens(self, states, parts, tok) -> dict:
        """逐层预测。states[l] 是第 l 层之后每个位置的状态，形状 (位置, 隐藏维度)。

        每一层都做同一件事：套上最终归一化，再乘输出层的权重，得到对整个词表的打分。
        最后一层的结果就是模型真实的输出，前面各层是"如果现在就必须回答"的答案。
        """
        import torch

        _, norm, head = parts
        k = _LENS_TOP_K
        top_ids, top_prob, final_prob, final_rank = [], [], [], []
        final = None
        vocab_size = 0
        for state in reversed(states):  # 先算最后一层，拿到每个位置的最终答案，再回头算其余各层
            probs = head(norm(state)).float().softmax(-1)
            vocab_size = probs.shape[-1]
            if final is None:
                final = probs.argmax(-1)
            p_final = probs.gather(1, final[:, None])[:, 0]
            top = probs.topk(k, dim=-1)
            top_ids.append(top.indices.tolist())
            top_prob.append([[_sig(v) for v in row] for row in top.values.tolist()])
            final_prob.append([_sig(v) for v in p_final.tolist()])
            final_rank.append(((probs > p_final[:, None]).sum(-1) + 1).tolist())
            del probs
        for series in (top_ids, top_prob, final_prob, final_rank):
            series.reverse()

        used = sorted({i for layer in top_ids for row in layer for i in row} | set(final.tolist()))
        index = {tid: j for j, tid in enumerate(used)}
        return {
            "layers": len(states),
            "steps": len(final),
            "k": k,
            "vocab_size": vocab_size,  # 排名最差是这个数，图表的对数刻度要用
            "vocab": [tok.decode([t]) for t in used],
            "top": [[[index[i] for i in row] for row in layer] for layer in top_ids],
            "prob": top_prob,
            "final": [index[i] for i in final.tolist()],
            "final_prob": final_prob,
            "final_rank": final_rank,
        }
