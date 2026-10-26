import time
from dataclasses import asdict

import torch

from .config import ModelConfig
from .model import SmallLM
from .training import make_optimizer
from .utils import (
    environment_versions,
    precision_context,
    resolve_device,
    seed_everything,
    write_json,
)


def hardware():
    result = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "gpus": []}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(i)
            result["gpus"].append(
                {
                    "index": i,
                    "name": properties.name,
                    "memory_gib": properties.total_memory / 1024**3,
                }
            )
        result["bf16_supported"] = torch.cuda.is_bf16_supported()
    result["environment"] = environment_versions()
    return result


def describe(config):
    model_config = ModelConfig(**config["model"])
    # meta tensors avoid allocating large model weights merely to count them.
    with torch.device("meta"):
        model = SmallLM(model_config)
    report = model.parameter_report()
    report["model_config"] = asdict(model_config)
    report["layer_3"] = (
        {"parameters": sum(p.numel() for p in model.blocks[2].parameters())}
        if model_config.n_layers >= 3
        else None
    )
    if report["layer_3"]:
        p = report["layer_3"]["parameters"]
        report["layer_3"].update(
            {"fp32_gradients_bytes": p * 4, "adam_fp32_two_moments_bytes": p * 8}
        )
    report["effective_batch_sequences"] = (
        config["train"]["batch_size"] * config["train"]["gradient_accumulation"]
    )
    report["tokens_per_optimizer_step"] = (
        report["effective_batch_sequences"] * model_config.context_length
    )
    report["planned_pretrain_tokens"] = (
        report["tokens_per_optimizer_step"] * config["train"]["max_steps"]
    )
    return report


def speed(config, output, device_name="auto", warmup=3, steps=10):
    if warmup < 0 or steps < 1:
        raise ValueError("Need nonnegative warmup and positive measured steps.")
    device = resolve_device(device_name)
    cfg = config["train"]
    seed_everything(cfg["seed"])
    _, precision = precision_context(device, cfg["precision"])
    model = SmallLM(ModelConfig(**config["model"])).to(device).train()
    forward_model = torch.compile(model) if cfg["compile"] else model
    optimizer = make_optimizer(model, cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    shape = (cfg["batch_size"], model.config.context_length)
    x = torch.randint(model.config.vocab_size, shape, device=device)
    y = torch.randint(model.config.vocab_size, shape, device=device)
    tokens = x.numel() * cfg["gradient_accumulation"]
    elapsed = []
    for step in range(warmup + steps):
        if step == warmup and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg["gradient_accumulation"]):
            context, _ = precision_context(device, precision)
            with context:
                _, loss = forward_model(x, labels=y)
            scaler.scale(loss / cfg["gradient_accumulation"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if step >= warmup:
            elapsed.append(time.perf_counter() - started)
    mean_time = sum(elapsed) / len(elapsed)
    report = {
        "precision": precision,
        "device": str(device),
        "parameters": model.parameter_report()["parameters"],
        "unique_blocks": model.config.n_layers,
        "block_applications": model.config.n_layers * model.config.block_repeats,
        "qk_norm": model.config.qk_norm,
        "tokens_per_step": tokens,
        "mean_step_seconds": mean_time,
        "tokens_per_second": tokens / mean_time,
        "estimated_training_hours_excluding_eval_and_io": cfg["max_steps"] * mean_time / 3600,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "sdpa_requested": model.config.attention_backend == "sdpa",
        "backend_note": "SDPA is an API, not a guarantee that FlashAttention was selected.",
        "peak_allocated_gpu_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "config": config,
        "environment": environment_versions(),
    }
    write_json(output, report)
    return report
