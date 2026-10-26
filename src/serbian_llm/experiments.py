import json
from pathlib import Path

from .checkpoints import checkpoint_metadata
from .utils import read_json


RUN_NAMES = (
    "absolute_41m",
    "rope_41m",
    "serbian_283m_1",
    "serbian_283m_2",
    "serbian_283m_3",
    "deep_continue_A",
    "deep_continue_B",
)


def read_metrics(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return rows


def run_summary(folder):
    folder = Path(folder)
    config = read_json(folder / "config.json")
    parameters = read_json(folder / "parameter_report.json")
    rows = read_metrics(folder / "metrics.jsonl")
    training = [r for r in rows if "train_loss" in r]
    validation = [r for r in rows if "validation_loss" in r]
    last = checkpoint_metadata(folder / "last.pt")
    best = checkpoint_metadata(folder / "best.pt")
    state = last["state"]
    step = max([state["step"]] + [r["step"] for r in rows])
    return {
        "run": folder.name,
        "parameters": parameters["parameters"],
        "layers": config["model"]["n_layers"],
        "width": config["model"]["d_model"],
        "step": step,
        "max_steps": config["train"]["max_steps"],
        "complete": state["step"] == config["train"]["max_steps"],
        "training_seconds": state["training_seconds"],
        "timing_checkpoint_step": state["step"],
        "processed_input_tokens": max(
            [state["processed_input_tokens"]] + [r["processed_input_tokens"] for r in rows]
        ),
        "best_step": best["state"]["step"],
        "best_validation_loss": best["state"]["best_validation_loss"],
        "latest_validation": validation[-1] if validation else None,
        "tokens_per_second": training[-1]["tokens_per_second"] if training else None,
        "weights_bytes": parameters["fp32_weights_bytes"],
        "checkpoint_bytes": (folder / "best.pt").stat().st_size,
        "lineage": last["lineage"],
    }
