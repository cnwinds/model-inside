"""用一个很小的随机 Qwen2 模型，把录制 → 落盘 → 接口读取整条链路和模型自己的输出逐项对照。"""
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from model_inside.server import create_app

WORDS = "the cat sat on a mat and the dog ran to big red house in town".split()
PROMPT = "the cat sat on a mat and the dog"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    # 目录里一旦有 config.json（model_type=qwen2），AutoTokenizer 就会换成 Qwen2 的切词规则，套在玩具词表上会切错。
    # 真实模型的分词器和配置是配套的，没有这个问题，所以这里让它原样读回我们保存的分词器。
    patch = pytest.MonkeyPatch()
    patch.setattr(AutoTokenizer, "from_pretrained", PreTrainedTokenizerFast.from_pretrained)
    d = tmp_path_factory.mktemp("tiny")
    vocab = {w: i for i, w in enumerate(["[UNK]", "[EOS]", *dict.fromkeys(WORDS)])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", eos_token="[EOS]").save_pretrained(d)

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=True,
    )
    Qwen2ForCausalLM(cfg).save_pretrained(d)

    data = d / "data"
    client = TestClient(create_app(d, data))
    meta = client.post("/api/record", json={"prompt": PROMPT, "max_new_tokens": 3}).json()
    reference = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32, attn_implementation="eager").eval()
    yield client, meta, reference, d
    patch.undo()


def real_forward(reference, tiny_dir, meta):
    tok = PreTrainedTokenizerFast.from_pretrained(tiny_dir)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    ids = reference.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=tok.eos_token_id)
    assert [tok.decode([i]) for i in ids[0].tolist()] == meta["tokens"]  # 录制的 token 序列和独立跑出来的一致
    with torch.no_grad():
        return reference(ids, output_attentions=True)


def test_recording_describes_what_was_captured(tiny):
    _, meta, _, _ = tiny
    n = len(meta["tokens"])
    assert n == 9 + 3 and meta["n_prompt"] == 9
    assert meta["attention"] == {"file": "attn.npy", "layers": 3, "heads": 4}
    assert meta["lens"] == {"file": "lens.json", "layers": 4}  # 第一层之前 + 每层之后
    assert any(m["tensor"].endswith("q_proj.weight") for m in meta["modules"])


def test_attention_endpoint_matches_the_model_and_is_causal(tiny):
    client, meta, reference, d = tiny
    real = real_forward(reference, d, meta).attentions
    n = len(meta["tokens"])
    for layer in range(3):
        r = client.get(f"/api/attention/{meta['id']}/{layer}")
        assert r.status_code == 200
        q = np.frombuffer(r.content, dtype=np.uint8).reshape(4, n, n)
        got = (q / 255.0) ** 2  # 前端的还原方式
        want = real[layer][0].numpy()
        np.testing.assert_allclose(got, want, atol=0.012)  # 量化误差上限
        assert (np.triu(q, 1) == 0).all()  # 不能偷看后面的词
        np.testing.assert_allclose(got.sum(-1), 1.0, atol=0.08)  # 每个词的关注度加起来约为 1
    assert client.get(f"/api/attention/{meta['id']}/3").status_code == 404
    assert client.get(f"/api/attention/..%2F..%2Fx/0").status_code == 404


def test_small_attention_weights_survive_quantization(tiny):
    """开方量化的意义：0.001 量级的小权重不会被抹成 0。"""
    from model_inside.recorder import quantize_attention

    a = np.array([0.0, 0.0005, 0.001, 0.01, 0.5, 1.0])
    q = quantize_attention(a)
    assert q[0] == 0 and q[1] > 0 and q[2] > q[1]
    np.testing.assert_allclose((q / 255.0) ** 2, a, atol=0.006)


def test_lens_last_layer_is_exactly_the_models_own_answer(tiny):
    client, meta, reference, d = tiny
    lens = client.get(f"/api/lens/{meta['id']}").json()
    out = real_forward(reference, d, meta)
    probs = out.logits[0].softmax(-1)
    n, k = len(meta["tokens"]), lens["k"]
    assert (lens["layers"], lens["steps"]) == (4, n)
    assert lens["vocab_size"] == reference.config.vocab_size
    assert max(max(row) for row in lens["final_rank"]) <= lens["vocab_size"]
    vocab = lens["vocab"]

    # 最后一层的逐层预测 == 模型真实输出
    last = lens["layers"] - 1
    for p in range(n):
        want = int(probs[p].argmax())
        assert vocab[lens["top"][last][p][0]] == meta_word(want, d)
        assert vocab[lens["final"][p]] == meta_word(want, d)
        assert lens["prob"][last][p][0] == pytest.approx(float(probs[p].max()), rel=2e-3)
        assert lens["final_rank"][last][p] == 1
        assert lens["final_prob"][last][p] == pytest.approx(float(probs[p].max()), rel=2e-3)

    # 每一层：候选按概率从高到低；答案的排名和概率与候选表一致
    for layer in range(lens["layers"]):
        for p in range(n):
            pr = lens["prob"][layer][p]
            assert len(pr) == k and pr == sorted(pr, reverse=True)
            if lens["top"][layer][p][0] == lens["final"][p]:
                assert lens["final_rank"][layer][p] == 1
                assert lens["final_prob"][layer][p] == pr[0]
            else:
                assert lens["final_rank"][layer][p] > 1


def test_lens_middle_layer_matches_manual_decode(tiny):
    """第 1 层之后的状态，套上最终归一化和输出层，应当和录制出来的一致。"""
    client, meta, reference, d = tiny
    lens = client.get(f"/api/lens/{meta['id']}").json()
    tok = PreTrainedTokenizerFast.from_pretrained(d)
    ids = torch.tensor([[tok.convert_tokens_to_ids(t) for t in meta["tokens"]]])
    with torch.no_grad():
        hs = reference(ids, output_hidden_states=True).hidden_states  # hs[0] 是嵌入，hs[i] 是第 i 层之后
        for layer in (0, 1, 2):  # 最后一层的 hs 已带归一化，不能直接比，上面的测试已经覆盖
            logits = reference.lm_head(reference.model.norm(hs[layer]))[0]
            pr = logits.softmax(-1)
            for p in range(len(meta["tokens"])):
                assert lens["vocab"][lens["top"][layer][p][0]] == tok.convert_ids_to_tokens(int(pr[p].argmax()))
                assert lens["prob"][layer][p][0] == pytest.approx(float(pr[p].max()), rel=2e-3)


def meta_word(token_id, d):
    return PreTrainedTokenizerFast.from_pretrained(d).decode([token_id])


def test_second_recording_works_and_old_recordings_still_load(tiny):
    client, meta, _, d = tiny
    second = client.post("/api/record", json={"prompt": "the dog", "max_new_tokens": 0}).json()
    assert len(second["tokens"]) == 2 and second["attention"]["layers"] == 3  # hook 没有残留，也没有串到上一次

    # 同一秒内录两次：各自独立，不覆盖；第一条录制的数据仍然完好
    third = client.post("/api/record", json={"prompt": "the cat sat", "max_new_tokens": 0}).json()
    assert len({meta["id"], second["id"], third["id"]}) == 3
    assert all(rid in [r["id"] for r in client.get("/api/recordings").json()] for rid in (meta["id"], second["id"], third["id"]))
    n = len(meta["tokens"])
    assert len(client.get(f"/api/attention/{meta['id']}/0").content) == 4 * n * n
    assert len(client.get(f"/api/attention/{third['id']}/0").content) == 4 * 3 * 3

    # 没有 attention / lens 字段的旧录制：列表里能看到，两个接口返回 404 而不是报错
    old = d / "data" / "recordings" / "20200101-000000"
    old.mkdir(parents=True)
    legacy = {k: v for k, v in meta.items() if k not in ("attention", "lens")} | {"id": "20200101-000000"}
    (old / "meta.json").write_text(json.dumps(legacy), encoding="utf-8")
    ids = [r["id"] for r in client.get("/api/recordings").json()]
    assert "20200101-000000" in ids
    assert client.get("/api/attention/20200101-000000/0").status_code == 404
    assert client.get("/api/lens/20200101-000000").status_code == 404
