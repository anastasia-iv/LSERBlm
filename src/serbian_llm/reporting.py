import csv
from pathlib import Path

from .utils import read_json, read_jsonl, write_json


def summarize_runs(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for folder in map(Path, runs):
        config = read_json(folder / "config.json")
        params = read_json(folder / "parameter_report.json")
        metrics = list(read_jsonl(folder / "metrics.jsonl"))
        # After a crash, logs can include attempts beyond last.pt. The latest
        # occurrence at a step supersedes the earlier attempt in these plots.
        train = {r["step"]: r for r in metrics if "train_loss" in r}
        validation = {r["step"]: r for r in metrics if "validation_loss" in r}
        for ax, records, field in [(axes[0], train, "train_loss"), (axes[1], validation, "validation_loss")]:
            points = [records[key] for key in sorted(records)]
            ax.plot([r["processed_input_tokens"] for r in points], [r[field] for r in points], label=folder.name)
            ax.set(xlabel="processed input tokens", ylabel=field)
        values = list(train.values())
        rows.append({"run": str(folder), "mode": config["train"]["mode"], "parameters": params["parameters"],
                     "position": config["model"]["positional"], "vocab_size": config["model"]["vocab_size"],
                     "best_logged_validation_loss": min(r["validation_loss"] for r in validation.values()),
                     "last_logged_tokens": max(r["processed_input_tokens"] for r in validation.values()),
                     "mean_logged_tokens_per_second": sum(r["tokens_per_second"] for r in values) / max(1, len(values))})
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)
    if rows:
        with (output / "runs.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    write_json(output / "runs.json", rows)
    return rows
