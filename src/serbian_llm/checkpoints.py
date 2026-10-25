import os
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .config import ModelConfig
from .model import SmallLM
from .utils import environment_versions, rng_state


FORMAT = "serbian-small-lm-v1"


def save_checkpoint(path, model, optimizer, scaler, state, sampler, config, identity, tokenizer, lineage):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {"format": FORMAT, "model_config": asdict(model.config), "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
               "state": dict(state), "sampler": sampler.state_dict(), "rng": rng_state(),
               "config": config, "identity": identity, "tokenizer_json": tokenizer.to_str(),
               "lineage": lineage, "environment": environment_versions()}
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_checkpoint(path, mmap=False):
    # Training checkpoints contain optimizer/RNG Python objects. Load your own files.
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    if payload.get("format") != FORMAT:
        raise ValueError("Not a checkpoint produced by this project.")
    if payload.get("lineage", {}).get("origin") != "random_initialization":
        raise ValueError("Checkpoint does not record scratch-training lineage.")
    # Legacy checkpoints have the same weights/optimizer layout with these defaults.
    for model_config in [payload["model_config"], payload["config"]["model"]]:
        model_config.setdefault("qk_norm", False)
        model_config.setdefault("block_repeats", 1)
    return payload


def load_for_inference(path, device="cpu"):
    payload = read_checkpoint(path, mmap=True)
    model = SmallLM(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    tokenizer = Tokenizer.from_str(payload["tokenizer_json"])
    return model, tokenizer, payload


def checkpoint_metadata(path):
    payload = read_checkpoint(path, mmap=True)
    return {key: payload[key] for key in ["model_config", "state", "lineage"]}
