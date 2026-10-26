"""Prepare grouped Cyrillic EuroAlpaca instructions for response-only SFT."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import unicodedata

DATASET = "Telefonica/EuroAlpaca"
TASKS = {
    "question_answering",
    "information_extraction",
    "paraphrasing",
    "summarization",
    "creative_writing",
    "classification",
    "procedural_instructions",
    "style_transfer",
}
FILES = [
    "sr/train-00000-of-00001.parquet",
    "annotations/source_annotations_and_policies.parquet",
    "annotations/validation/sr.parquet",
]
SERBIAN = set("абвгдђежзијклљмнњопрстћуфхцчџш")


def clean(value):
    return unicodedata.normalize("NFC", str(value or "")).strip()


def canonical(value):
    return " ".join(clean(value).casefold().split())


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def script_ok(text):
    letters = [c for c in text.casefold() if c.isalpha()]
    if not letters:
        return True
    cyrillic = [c for c in letters if "\u0400" <= c <= "\u04ff"]
    return (
        len(cyrillic) / len(letters) >= 0.8
        and sum(c not in SERBIAN for c in cyrillic) / len(letters) <= 0.01
    )


def post_edit_applied(row, check):
    try:
        edit = json.loads(check.get("post_edit") or "{}")
    except (TypeError, ValueError):
        return False
    changes = [
        field for field in ["instruction", "input", "output"] if edit.get(field + "_changed")
    ]
    return bool(changes) and all(
        field + "_original" in edit and row.get(field) != edit[field + "_original"]
        for field in changes
    )


def prepare_rows(
    records,
    annotations,
    validation,
    tokenizer,
    blocked,
    max_response_tokens,
    include_post_edited=False,
):
    from serbian_llm.batching import format_question, sft_example

    accepted, conflicts, counts = {}, set(), Counter()
    rejected_samples = []
    for row in records:
        counts["scanned"] += 1
        sid = int(row["sample_id"])
        annotation, check = annotations.get(sid, {}), validation.get(sid, {})
        question, context, answer = [clean(row.get(k)) for k in ["instruction", "input", "output"]]
        prompt = format_question(question, context)
        normalized = canonical(prompt)
        domains = {annotation.get("primary_domain"), *annotation.get("secondary_domains", [])}
        reason = None
        edited = post_edit_applied(row, check)
        if check.get("validation_status") != "validated" or not (
            check.get("is_coherent") is True or include_post_edited and edited
        ):
            reason = "upstream_coherence"
        elif annotation.get("primary_task") not in TASKS or domains & {
            "mathematics",
            "programming",
        }:
            reason = "task_or_domain"
        elif not question or not answer:
            reason = "empty"
        elif any(
            marker in (prompt + answer).casefold() for marker in ["```", "http://", "https://"]
        ):
            reason = "code_or_url"
        elif any(token in prompt + answer for token in ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]):
            reason = "special_token_literal"
        elif not all(script_ok(field) for field in [question, context, answer]):
            reason = "script"
        elif any(item in normalized for item in blocked):
            reason = "benchmark_exact_overlap"
        prefix_tokens = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
        answer_tokens = len(tokenizer.encode(" " + answer, add_special_tokens=False).ids)
        if reason is None and (
            prefix_tokens > 320
            or not answer_tokens
            or answer_tokens > max_response_tokens
            or prefix_tokens + answer_tokens + 1 > 512
        ):
            reason = "too_long_or_empty_tokens"
        if reason is None and normalized in conflicts:
            reason = "conflicting_duplicate"
        if reason is None and normalized in accepted:
            if accepted[normalized]["answer"] != answer:
                del accepted[normalized]
                conflicts.add(normalized)
                reason = "conflicting_duplicate"
            else:
                reason = "exact_duplicate"
        if reason:
            counts[reason] += 1
            if len(rejected_samples) < 100:
                rejected_samples.append(
                    {"id": sid, "reason": reason, "question": question, "answer": answer[:1000]}
                )
            continue
        group_key = (
            "context:" + canonical(context) if context else "question:" + canonical(question)
        )
        record = {
            "id": f"euro_sr_{sid}",
            "group_id": "euro_sr_" + digest(group_key),
            "question": question,
            "context": context,
            "answer": answer,
            "source": DATASET,
            "source_sample_id": sid,
            "task": annotation["primary_task"],
            "upstream_is_coherent": check.get("is_coherent"),
            "upstream_post_edited": edited,
        }
        x, y = sft_example(record, tokenizer, 512)
        assert len(x) <= 512 and sum(t != -100 for t in y) == answer_tokens + 1
        accepted[normalized] = record
    counts["accepted_before_limit"] = len(accepted)
    counts["conflicting_prompts_removed"] = len(conflicts)
    return list(accepted.values()), dict(counts), rejected_samples


def split_rows(rows, seed, limits):
    rows = sorted(rows, key=lambda row: row["id"])
    random.Random(seed).shuffle(rows)
    splits = {name: [] for name in limits}
    for row in rows:
        bucket = int(digest(str(seed) + row["group_id"])[:8], 16) % 100
        split = "test" if bucket < 5 else "validation" if bucket < 10 else "train"
        if len(splits[split]) < limits[split]:
            splits[split].append(row)
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="data/serbian_mix_v2/bpe8k")
    parser.add_argument("--base-config", default="runs/deep_continue_B/config.json")
    parser.add_argument("--sft-template", default="configs/sft.json")
    parser.add_argument("--output", default="data/sft_euroalpaca_v1")
    parser.add_argument("--config-output", default="configs/sft_deep_lr1e5.json")
    parser.add_argument("--benchmark", action="append", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--include-post-edited", action="store_true")
    parser.add_argument("--max-train", type=int, default=50000)
    parser.add_argument("--max-validation", type=int, default=2000)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--train-volumes", type=float, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        min(
            args.max_train,
            args.max_validation,
            args.max_test,
            args.max_response_tokens,
            args.train_volumes,
            args.learning_rate,
        )
        <= 0
    ):
        raise ValueError("Positive sizes, train volumes and learning rate are required.")
    output, config_path = Path(args.output), Path(args.config_output)
    if output.exists() or config_path.exists():
        raise FileExistsError(
            "Choose new output/config paths; existing artifacts are not overwritten."
        )

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from tokenizers import Tokenizer
    from serbian_llm.batching import sft_example, validate_sft_splits
    from serbian_llm.utils import file_sha256, read_jsonl, write_json

    blocked = set()
    for path in args.benchmark:
        for row in read_jsonl(path):
            for key in ["question", "prompt"]:
                if canonical(row.get(key)):
                    blocked.add(canonical(row[key]))
    revision = args.revision or HfApi().dataset_info(DATASET).sha
    local = {
        name: hf_hub_download(DATASET, name, repo_type="dataset", revision=revision)
        for name in FILES
    }
    records = pq.read_table(local[FILES[0]]).to_pylist()
    annotations = {int(r["sample_id"]): r for r in pq.read_table(local[FILES[1]]).to_pylist()}
    validation = {int(r["sample_id"]): r for r in pq.read_table(local[FILES[2]]).to_pylist()}
    tokenizer_path = Path(args.tokenizer) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if any(tokenizer.token_to_id(t) is None for t in ["[PAD]", "[BOS]", "[EOS]"]):
        raise ValueError("Missing tokenizer special tokens.")
    rows, counts, rejected = prepare_rows(
        records,
        annotations,
        validation,
        tokenizer,
        blocked,
        args.max_response_tokens,
        args.include_post_edited,
    )
    splits = split_rows(
        rows,
        args.seed,
        dict(train=args.max_train, validation=args.max_validation, test=args.max_test),
    )
    if len(splits["train"]) < 100 or min(len(splits[s]) for s in ["validation", "test"]) < 20:
        raise ValueError("Too few usable train or held-out examples.")
    template = json.loads(Path(args.sft_template).read_text())
    template["model"] = json.loads(Path(args.base_config).read_text())["model"]
    if template["model"]["context_length"] != 512:
        raise ValueError("Preparation expects context_length=512.")
    steps = math.ceil(args.train_volumes * len(splits["train"]) / 32)
    template["train"].update(
        seed=args.seed,
        max_steps=steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.learning_rate / 10,
        warmup_steps=min(steps - 1, max(1, round(0.03 * steps))),
    )
    output.mkdir(parents=True)
    for name, items in splits.items():
        with (output / f"{name}.jsonl").open("x", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    audits = {}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        audits[f"{a}:{b}"] = validate_sft_splits(output / f"{a}.jsonl", output / f"{b}.jsonl")
    stats = {}
    for name, items in splits.items():
        encoded = [sft_example(item, tokenizer, 512) for item in items]
        stats[name] = {
            "examples": len(items),
            "input_tokens": sum(len(x) for x, _ in encoded),
            "supervised_tokens": sum(sum(t != -100 for t in y) for _, y in encoded),
            "max_input_tokens": max(len(x) for x, _ in encoded),
            "upstream_quality": dict(
                Counter("coherent" if r["upstream_is_coherent"] else "post_edited" for r in items)
            ),
            "tasks": dict(Counter(r["task"] for r in items)),
        }
    samples = random.Random(args.seed).sample(splits["train"], min(100, len(splits["train"])))
    with (output / "audit_samples.jsonl").open("x", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_json(output / "rejected_samples.json", rejected)
    write_json(config_path, template)
    report = {
        "dataset": DATASET,
        "configuration": "sr",
        "revision": revision,
        "license": "CC-BY-NC-4.0",
        "source_files": {name: file_sha256(p) for name, p in local.items()},
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "filters": counts,
        "splits": stats,
        "include_post_edited": args.include_post_edited,
        "max_response_tokens": args.max_response_tokens,
        "max_steps": steps,
        "effective_batch": 32,
        "planned_example_draws": steps * 32,
        "train_volumes": args.train_volumes,
        "split_protocol": "90/5/5 by hashed context/question group; capped example counts; sampling with replacement during training",
        "file_sha256": {s: file_sha256(output / f"{s}.jsonl") for s in splits},
        "split_audits": audits,
        "benchmarks": {p: file_sha256(p) for p in args.benchmark},
        "limitations": "Upstream machine coherence metadata and script filters are not human or factual verification. Exact benchmark matching does not detect semantic paraphrases. No answers are truncated; test is reserved for final evaluation.",
    }
    write_json(output / "preparation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
