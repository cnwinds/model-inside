"""离线生成一个小型测试模型：沿用现有模型的配置和分词器，但缩小尺寸、随机初始化。

权重是随机的，模型没有语言能力，只用来在没有真实权重时验证工具的各个环节。

用法: python scripts/make_test_model.py <含 config.json 和分词器的目录> <输出目录>
"""
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

src, out = Path(sys.argv[1]), Path(sys.argv[2])

cfg = AutoConfig.from_pretrained(src)
cfg.hidden_size, cfg.intermediate_size = 512, 2048
cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads = 8, 8, 2
if getattr(cfg, "layer_types", None):  # 新版配置里每层有一个类型，长度要和层数一致
    cfg.layer_types = cfg.layer_types[: cfg.num_hidden_layers]

torch.manual_seed(0)
model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16)
model.save_pretrained(out)
AutoTokenizer.from_pretrained(src).save_pretrained(out)
print(f"已生成 {sum(p.numel() for p in model.parameters()):,} 个参数 -> {out}")
