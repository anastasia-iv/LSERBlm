import copy
import json
import math
import signal
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .batching import PackedSampler
from .checkpoints import read_checkpoint, save_checkpoint
from .config import ModelConfig
from .model import SmallLM
from .utils import (environment_versions, file_sha256, precision_context, read_json,
                    resolve_device, restore_rng, seed_everything, write_json)


def learning_rate(step, train_config):
    maximum = train_config["learning_rate"]
    minimum = train_config["min_learning_rate"]
    total = train_config["max_steps"]
    warmup = min(train_config["warmup_steps"], max(0, total - 1))
    if step < warmup:
        return maximum * (step + 1) / warmup
    if train_config["schedule"] == "wsd":
        decay_start = max(warmup, int(total * (1 - train_config["decay_fraction"])))
        if step < decay_start:
            return maximum
        ratio = (step - decay_start) / max(1, total - 1 - decay_start)
    elif train_config["schedule"] == "cosine":
        ratio = (step - warmup) / max(1, total - 1 - warmup)
    else:
        raise ValueError("schedule must be wsd or cosine")
    return minimum + 0.5 * (maximum - minimum) * (1 + math.cos(math.pi * min(1, ratio)))


def make_optimizer(model, config, device):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [{"params": decay, "weight_decay": config["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=config["learning_rate"], betas=tuple(config["betas"]),
                             eps=config["eps"], fused=device.type == "cuda")


def to_device(batch, device):
    return {name: tensor.to(device) if tensor is not None else None for name, tensor in batch.items()}


@torch.no_grad()
def validation_loss(model, sampler_factory, batches, device, precision):
    was_training = model.training
    model.eval()
    numerator = denominator = 0
    sampler = sampler_factory()
    try:
        for _ in range(batches):
            batch = to_device(sampler.batch(), device)
            context, _ = precision_context(device, precision)
            with context:
                _, loss = model(**batch)
            count = int((batch["labels"] != -100).sum().item())
            numerator += loss.item() * count
            denominator += count
    finally:
        model.train(was_training)
    return numerator / denominator


def train(config, tokenizer_folder, output, resume=None, init_from=None,
          stop_after=None, device_name="auto"):
    config = copy.deepcopy(config)
    cfg = config["train"]
    if cfg["max_steps"] < 1 or cfg["batch_size"] < 1 or cfg["gradient_accumulation"] < 1:
        raise ValueError("Positive steps, batch size and gradient accumulation are required.")
    if cfg["eval_interval"] < 1 or cfg["save_interval"] < 1 or cfg["eval_batches"] < 1 or cfg["log_interval"] < 1:
        raise ValueError("Logging/evaluation/checkpoint intervals must be positive.")
    if not 0 < cfg["learning_rate"] or not 0 <= cfg["min_learning_rate"] <= cfg["learning_rate"]:
        raise ValueError("Invalid learning rates.")
    if not 0 < cfg["decay_fraction"] <= 1:
        raise ValueError("decay_fraction must be in (0, 1].")
    if cfg["mode"] not in {"pretrain", "continued_pretrain"}:
        raise ValueError("mode must be pretrain or continued_pretrain")
    mode = cfg["mode"]
    if mode == "pretrain" and init_from:
        raise ValueError("Pretraining starts from scratch. Use continued_pretrain for --init-from.")
    if mode == "continued_pretrain" and not (resume or init_from):
        raise ValueError("Continued pretraining requires --init-from or --resume.")
    if resume and init_from:
        raise ValueError("Use resume or init_from, not both.")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive.")
    device = resolve_device(device_name)
    _, precision = precision_context(device, cfg["precision"])
    seed_everything(cfg["seed"])
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = cfg["allow_tf32"]
    folder, output = Path(tokenizer_folder), Path(output)
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    if config.get("tokenizer"):
        expected = config["tokenizer"]
        actual_algorithm = json.loads(tokenizer.to_str())["model"]["type"].lower()
        if (actual_algorithm != expected["algorithm"]
                or tokenizer.get_vocab_size() != expected["vocab_size"]):
            raise ValueError(
                f"Config requires {expected['algorithm']} tokenizer with {expected['vocab_size']} tokens; "
                f"got {actual_algorithm} with {tokenizer.get_vocab_size()}. Pass the matching --tokenizer directory.")
    token_hash = file_sha256(folder / "tokenizer.json")
    stats = read_json(folder / "token_stats.json")
    if stats["tokenizer_sha256"] != token_hash:
        raise ValueError("Tokenizer and encoded data identities disagree.")
    checkpoint_data = read_checkpoint(resume or init_from) if resume or init_from else None
    if checkpoint_data is not None:
        embedded_tokenizer = Tokenizer.from_str(checkpoint_data["tokenizer_json"])
        if embedded_tokenizer.to_str() != tokenizer.to_str():
            raise ValueError("Checkpoint and current tokenizer are different.")
    values = dict(config["model"])
    values["vocab_size"] = tokenizer.get_vocab_size()
    model_config = ModelConfig(**values)
    if mode == "continued_pretrain":
        parent_values = dict(checkpoint_data["model_config"])
        current_values = asdict(model_config)
        for key in ["gradient_checkpointing", "attention_backend"]:
            parent_values.pop(key)
            current_values.pop(key)
        if parent_values != current_values:
            raise ValueError("Continued pretraining must preserve the parent architecture and context length.")
    identity = {"mode": mode, "tokenizer_sha256": token_hash, "corpus_files": stats["corpus_files"]}
    if mode == "continued_pretrain" and init_from and config.get("continuation", {}).get("require_same_validation", False):
        parent_validation = checkpoint_data["identity"].get("corpus_files", {}).get("validation")
        if parent_validation != stats["corpus_files"].get("validation"):
            raise ValueError("Continuation config requires the parent's frozen validation documents.")
    for split in ["train", "validation"]:
        actual = file_sha256(folder / (split + ".bin"))
        if actual != stats["splits"][split]["bin_sha256"]:
            raise ValueError("Encoded data changed: " + split)
    config["model"] = asdict(model_config)
    config["train"]["precision"] = precision
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.iterdir()):
        raise FileExistsError("Run directory is not empty. Choose a new --output or use --resume.")
    if resume:
        if checkpoint_data["config"] != config or checkpoint_data["identity"] != identity:
            raise ValueError("Exact resume requires the same effective config, tokenizer and data. Use the original settings.")
        if checkpoint_data["environment"] != environment_versions():
            raise ValueError("Environment versions changed; exact resume requires the recorded environment.")

    model = SmallLM(model_config).to(device)
    if checkpoint_data:
        model.load_state_dict(checkpoint_data["model"])
    optimizer = make_optimizer(model, cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    def make_sampler(split, seed):
        return PackedSampler(folder / (split + ".bin"), model_config.context_length, cfg["batch_size"], seed)
    sampler = make_sampler("train", cfg["seed"] + 1)
    validation_factory = lambda: make_sampler("validation", cfg["seed"] + 2)
    state = {"step": 0, "processed_input_tokens": 0, "supervised_tokens": 0,
             "optimizer_updates": 0, "best_validation_loss": float("inf"), "training_seconds": 0.0}
    lineage = {"origin": "random_initialization", "stage": mode, "parent_checkpoint_sha256": None}
    if init_from:
        lineage["parent_checkpoint_sha256"] = file_sha256(init_from)
        parent_lineage = checkpoint_data["lineage"]
        lineage.update(parent_checkpoint=str(Path(init_from).resolve()),
                       parent_stage=parent_lineage["stage"],
                       parent_processed_input_tokens=checkpoint_data["state"]["processed_input_tokens"],
                       ancestor_processed_input_tokens=(parent_lineage.get("ancestor_processed_input_tokens", 0)
                                                       + checkpoint_data["state"]["processed_input_tokens"]),
                       optimizer_policy="reset", schedule_policy="restart at stage step 0")
    if resume:
        optimizer.load_state_dict(checkpoint_data["optimizer"])
        scaler.load_state_dict(checkpoint_data["scaler"])
        state = checkpoint_data["state"]
        sampler.load_state_dict(checkpoint_data["sampler"])
        lineage = checkpoint_data["lineage"]
        restore_rng(checkpoint_data["rng"])
    del checkpoint_data
    forward_model = torch.compile(model) if cfg["compile"] else model
    write_json(output / "config.json", config)
    write_json(output / "identity.json", identity)
    write_json(output / "lineage.json", lineage)
    write_json(output / "environment.json", environment_versions())
    write_json(output / "parameter_report.json", model.parameter_report())
    tokenizer.save(str(output / "tokenizer.json"))
    print(json.dumps(model.parameter_report(), indent=2))
    print("Precision:", precision, "SDPA requested:", model_config.attention_backend == "sdpa")

    writer = None
    if cfg["tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output / "tensorboard"), purge_step=state["step"] if resume else None)
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested: finishing the current optimizer step and saving last.pt.", flush=True)
    previous_handlers = {}
    for name in [signal.SIGINT, signal.SIGTERM]:
        previous_handlers[name] = signal.signal(name, request_stop)

    def save(name):
        save_checkpoint(output / name, model, optimizer, scaler, state, sampler, config, identity, tokenizer, lineage)

    def log(row):
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if writer:
            for key, value in row.items():
                if isinstance(value, (float, int)) and key != "step":
                    writer.add_scalar(key, value, row["step"])
        print(json.dumps(row), flush=True)

    def evaluate_and_sample():
        loss = validation_loss(model, validation_factory, cfg["eval_batches"], device, precision)
        log({"step": state["step"], "validation_loss": loss, "validation_perplexity": math.exp(min(50, loss)),
             "processed_input_tokens": state["processed_input_tokens"]})
        if loss < state["best_validation_loss"]:
            state["best_validation_loss"] = loss
            save("best.pt")
        if cfg["sample_prompts"]:
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(cfg["seed"])
                for prompt in cfg["sample_prompts"]:
                    ids = [tokenizer.token_to_id("[BOS]")] + tokenizer.encode(prompt).ids
                    inputs = torch.tensor([ids], device=device)
                    context, _ = precision_context(device, precision)
                    with context:
                        result = model.generate(inputs, max_new_tokens=cfg["sample_new_tokens"], temperature=0.8,
                            eos_id=tokenizer.token_to_id("[EOS]"), forbidden_ids=[tokenizer.token_to_id("[PAD]"), tokenizer.token_to_id("[BOS]")])
                    record = {"step": state["step"], "prompt": prompt,
                              "completion": tokenizer.decode(result[0, len(ids):].tolist())}
                    with (output / "samples.jsonl").open("a", encoding="utf-8") as file:
                        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        if state["step"] == 0:
            evaluate_and_sample()
            save("last.pt")
        while state["step"] < cfg["max_steps"]:
            if stop_after is not None and state["step"] >= stop_after:
                break
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batches = [sampler.batch() for _ in range(cfg["gradient_accumulation"])]
            supervised = sum(int((b["labels"] != -100).sum()) for b in batches)
            input_tokens = sum(int(b["attention_mask"].sum()) if b["attention_mask"] is not None else b["input_ids"].numel() for b in batches)
            if supervised == 0:
                raise ValueError("No supervised tokens in a training step.")
            step_loss = 0.0
            lr = learning_rate(state["step"], cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            for batch in batches:
                active = int((batch["labels"] != -100).sum())
                batch = to_device(batch, device)
                context, _ = precision_context(device, precision)
                with context:
                    _, loss = forward_model(**batch)
                    weighted_loss = loss * active / supervised
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss. Resume the previous checkpoint after investigating.")
                step_loss += loss.item() * active / supervised
                scaler.scale(weighted_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            if precision != "fp16" and not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite gradients. Resume the previous checkpoint after investigating.")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            state["step"] += 1
            state["processed_input_tokens"] += input_tokens
            state["supervised_tokens"] += supervised
            state["optimizer_updates"] += int(not skipped)
            state["training_seconds"] += elapsed
            if state["step"] % cfg["log_interval"] == 0:
                row = {"step": state["step"], "train_loss": step_loss, "learning_rate": lr,
                       "gradient_norm": float(grad_norm), "optimizer_update_skipped": skipped,
                       "tokens_per_second": input_tokens / elapsed, "step_seconds": elapsed,
                       "processed_input_tokens": state["processed_input_tokens"]}
                if device.type == "cuda":
                    row["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated(device)
                log(row)
            final = state["step"] == cfg["max_steps"] or stop_requested
            final = final or (stop_after is not None and state["step"] >= stop_after)
            if state["step"] % cfg["eval_interval"] == 0 or final:
                evaluate_and_sample()
            if state["step"] % cfg["save_interval"] == 0 or final:
                save("last.pt")
            if stop_requested:
                break
        save("last.pt")
    finally:
        if writer:
            writer.close()
        for name, handler in previous_handlers.items():
            signal.signal(name, handler)
    return {"state": state, "output": str(output), "parameters": model.parameter_report()["parameters"]}
