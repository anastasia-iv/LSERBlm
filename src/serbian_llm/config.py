import copy
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 512
    n_layers: int = 24
    d_model: int = 384
    n_heads: int = 6
    n_kv_heads: int = 2
    d_ff: int = 1024
    positional: str = "rope"
    rope_theta: float = 10000.0
    norm: str = "rmsnorm"
    ffn: str = "swiglu"
    tie_embeddings: bool = True
    dropout: float = 0.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    gradient_checkpointing: bool = False
    attention_backend: str = "sdpa"
    qk_norm: bool = False
    block_repeats: int = 1

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.n_layers, self.d_model, self.n_heads, self.n_kv_heads, self.d_ff) <= 0:
            raise ValueError("Model dimensions must be positive.")
        if self.d_model % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("d_model must divide into n_heads; n_heads must divide into n_kv_heads groups.")
        if self.positional == "rope" and (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension.")
        if self.positional not in {"rope", "absolute"} or self.norm not in {"rmsnorm", "layernorm"}:
            raise ValueError("Unsupported positional encoding or normalization.")
        if self.ffn not in {"swiglu", "gelu"} or self.attention_backend not in {"sdpa", "eager"}:
            raise ValueError("Unsupported FFN or attention backend.")
        if not 0 <= self.dropout < 1 or self.rope_theta <= 0 or self.norm_eps <= 0 or self.init_std <= 0:
            raise ValueError("Invalid dropout, RoPE, normalization, or initialization setting.")
        if not isinstance(self.qk_norm, bool):
            raise ValueError("qk_norm must be boolean.")
        if (not isinstance(self.block_repeats, int) or isinstance(self.block_repeats, bool)
                or self.block_repeats < 1):
            raise ValueError("block_repeats must be a positive integer.")


def load_config(path, overrides=()):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    if isinstance(config.get("model"), dict):
        config["model"].setdefault("qk_norm", False)
        config["model"].setdefault("block_repeats", 1)
    for item in overrides:
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        keys = key.split(".")
        target = config
        for component in keys[:-1]:
            if component not in target or not isinstance(target[component], dict):
                raise KeyError(f"Unknown configuration path: {key}")
            target = target[component]
        if keys[-1] not in target:
            raise KeyError(f"Unknown configuration setting: {key}")
        target[keys[-1]] = value
    return config
