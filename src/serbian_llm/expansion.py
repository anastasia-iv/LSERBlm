"""Expand train while preserving evaluation files byte for byte."""
import json
import shutil
from collections import Counter
from pathlib import Path

from datasketch import MinHashLSH
from tqdm import tqdm

from .data import audit_splits, clean_document, normalize_text, sketch_of, text_signature
from .utils import file_sha256, read_json, read_jsonl, write_json


def long_paragraphs(text, minimum=200):
    return {text_signature(p) for p in text.splitlines() if len(p.strip()) >= minimum}


def expand_corpus(base_corpus, inputs, output, config):
    base, output = Path(base_corpus), Path(output)
    config = dict(config)
    base_manifest = read_json(base / "manifest.json")
    for split, digest in base_manifest["file_sha256"].items():
        if file_sha256(base / f"{split}.jsonl") != digest:
            raise ValueError("Base corpus changed: " + split)
    identity = {"pipeline_version": "frozen-holdout-expansion-v1", "config": config,
                "base_corpus_files": base_manifest["file_sha256"],
                "inputs": [{"path": str(Path(p).resolve()), "sha256": file_sha256(p)} for p in inputs]}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("Different expansion exists; choose a new output directory.")
        for split, digest in manifest["file_sha256"].items():
            if file_sha256(output / f"{split}.jsonl") != digest:
                raise ValueError("Expanded corpus changed: " + split)
        return manifest
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Expansion output must be empty; partial data is not trusted.")
    # Protect original AND cleaned forms of the frozen holdout. No holdout labels
    # or frequencies are used for tokenizer/model fitting.
    protected, paragraphs = set(), set()
    holdout_lsh = MinHashLSH(threshold=config["near_duplicate_threshold"],
                             num_perm=config["minhash_permutations"])
    for split in ["validation", "test"]:
        for row in read_jsonl(base / f"{split}.jsonl"):
            clean, _ = clean_document({**row, "language": "srp"}, config)
            for text in {normalize_text(row["text"]), clean}:
                signature = text_signature(text)
                paragraphs.update(long_paragraphs(text))
                if signature not in protected and config["near_dedup"]:
                    holdout_lsh.insert(signature, sketch_of(text, config))
                protected.add(signature)
    output.mkdir(parents=True, exist_ok=True)
    for split in ["validation", "test"]:
        shutil.copyfile(base / f"{split}.jsonl", output / f"{split}.jsonl")
    seen = set()
    train_lsh = MinHashLSH(threshold=config["near_duplicate_threshold"],
                          num_perm=config["minhash_permutations"])
    rejected, origins, source_counts, source_bytes = Counter(), Counter(), Counter(), Counter()
    counts = {"train": 0, "validation": 0, "test": 0}
    sizes = dict.fromkeys(counts, 0)
    samples = []
    scanned = 0
    partial = output / "train.jsonl.partial"
    with partial.open("x", encoding="utf-8") as target:
        for path, is_base in [(base / "train.jsonl", True)] + [(Path(p), False) for p in inputs]:
            for row in tqdm(read_jsonl(path), desc="Expand " + path.name, unit="doc"):
                scanned += 1
                # The base corpus already carries verified pipeline provenance;
                # new local inputs must supply their own language evidence.
                candidate = {**row, "language": "srp"} if is_base else row
                text, reason = clean_document(candidate, config)
                signature = text_signature(text)
                sketch = None
                if reason is None:
                    if signature in protected:
                        reason = "holdout_exact_document"
                    elif long_paragraphs(text) & paragraphs:
                        reason = "holdout_shared_long_paragraph"
                    elif signature in seen:
                        reason = "exact_duplicate"
                    elif config["near_dedup"]:
                        sketch = sketch_of(text, config)
                        if holdout_lsh.query(sketch):
                            reason = "holdout_near_duplicate"
                        elif train_lsh.query(sketch):
                            reason = "approximate_duplicate"
                if reason:
                    rejected[reason] += 1
                    if len(samples) < 100:
                        samples.append({"reason": reason, "input": str(path), "id": row.get("id"),
                                        "text": text[:1500]})
                    continue
                record = {**row, "text": text, "id": signature,
                          "expansion_input": str(path), "base_train": is_base}
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen.add(signature)
                if config["near_dedup"]:
                    train_lsh.insert(signature, sketch)
                size = len(text.encode("utf-8"))
                counts["train"] += 1
                sizes["train"] += size
                origins["base_train" if is_base else "additional_inputs"] += 1
                name = row.get("source", "local")
                source_counts[name] += 1
                source_bytes[name] += size
    if not counts["train"]:
        raise ValueError("No accepted training documents; inspect language metadata and filters.")
    partial.replace(output / "train.jsonl")
    for split in ["validation", "test"]:
        for row in read_jsonl(output / f"{split}.jsonl"):
            counts[split] += 1
            sizes[split] += len(row["text"].encode("utf-8"))
    manifest = {"identity": identity, "scanned_documents": scanned, "split_documents": counts,
                "split_text_bytes": sizes, "clean_text_bytes": sum(sizes.values()),
                "source_documents": dict(source_counts), "source_text_bytes": dict(source_bytes),
                "source_statistics_scope": "train only; holdout files are copied unchanged",
                "accepted_origins": dict(origins), "rejected": dict(rejected),
                "frozen_holdout": {s: base_manifest["file_sha256"][s] for s in ["validation", "test"]},
                "file_sha256": {s: file_sha256(output / f"{s}.jsonl") for s in counts},
                "notes": ["All input candidates are train-only; validation/test are frozen.",
                          "Exact documents, >=200-character lines and approximate document matches to holdout are excluded.",
                          "MinHash is approximate; semantic overlap still needs manual inspection."]}
    audit = audit_splits(output)
    write_json(output / "rejected_samples.json", samples)
    write_json(output / "manifest.json", manifest)
    if audit["exact_cross_split_documents"]:
        raise RuntimeError("Unexpected exact split overlap; do not use this corpus.")
    return manifest
