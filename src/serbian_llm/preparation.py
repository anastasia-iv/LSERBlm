"""Collect source candidates and prepare the Wikipedia/FineWeb mixture."""
import copy
import json
from collections import Counter
from contextlib import closing
from pathlib import Path

from tqdm import tqdm

from .data import audit_splits, clean_document, prepare
from .utils import file_sha256, read_json, write_json
from .streaming import STREAM_VERSION, stream_documents

WIKIPEDIA = "wikimedia/wikipedia"
FINEWEB = "HuggingFaceFW/fineweb-2"


def adapt_document(row, spec):
    """Keep source provenance and distinguish metadata from classifier evidence."""
    if spec["dataset"] not in {WIKIPEDIA, FINEWEB}:
        raise ValueError("Unsupported source adapter: " + spec["dataset"])
    wikipedia = spec["dataset"] == WIKIPEDIA
    if wikipedia and spec["subset"] != "20231101.sr":
        raise ValueError("Wikipedia adapter requires the verified Serbian edition config.")
    return {
        "text": row.get("text") or "", "id": row.get("id"), "url": row.get("url"),
        "title": row.get("title"), "language": "srp" if wikipedia else row.get("language"),
        "language_score": None if wikipedia else row.get("language_score"),
        "language_evidence": "serbian_wikipedia_edition" if wikipedia else "source_glotlid",
        "source": spec["dataset"], "source_revision": spec["revision"], "source_subset": spec["subset"],
    }


def source_filters(base_config, spec):
    filters = copy.deepcopy(base_config)
    filters.update(spec.get("filter_overrides", {}))
    # Explicitly trust this edition's language metadata, without inventing LID.
    filters["trusted_metadata_language_sources"] = [WIKIPEDIA]
    return filters


def collect_source(spec, base_config, output_file):
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = path.with_suffix(".manifest.json")
    filters = source_filters(base_config, spec)
    identity = {"adapter_version": 2, "stream_version": STREAM_VERSION, "source": spec, "filters": filters}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity or file_sha256(path) != manifest["file_sha256"]:
            raise ValueError("Source candidates/config changed. Use a new output directory.")
        return manifest
    partial = path.with_suffix(".partial")
    if path.exists() or partial.exists():
        raise FileExistsError("Incomplete source download; use a new output directory: " + str(path))
    if spec["target_text_mb"] <= 0:
        raise ValueError("Source byte target must be positive.")
    print("Opening pinned source:", spec["dataset"], spec["subset"], flush=True)
    rows = stream_documents(spec["dataset"], spec["subset"], spec["revision"],
                            filters["seed"], spec["shuffle_buffer"])
    target = round(spec["target_text_mb"] * 1_000_000)
    scanned = accepted = accepted_bytes = raw_bytes = 0
    rejected = Counter()
    rejected_samples = []
    with closing(rows), partial.open("x", encoding="utf-8") as file, tqdm(total=target, unit="B", unit_scale=True,
                                                          desc=spec["name"]) as bar:
        for row in rows:
            scanned += 1
            record = adapt_document(row, spec)
            raw_bytes += len(record["text"].encode("utf-8"))
            text, reason = clean_document(record, filters)
            if reason is not None:
                rejected[reason] += 1
                if len(rejected_samples) < 100:
                    rejected_samples.append({"reason": reason, "text": record["text"][:1500],
                                             "url": record["url"]})
            else:
                # Keep original accepted text here; prepare() applies normalization.
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                size = len(text.encode("utf-8"))
                accepted += 1
                accepted_bytes += size
                bar.update(size)
            if scanned % 5000 == 0:
                file.flush()
                bar.set_postfix(scanned=scanned, accepted=accepted)
            if accepted_bytes >= target or scanned >= spec["max_scanned_documents"]:
                break
    if not accepted:
        raise ValueError("No accepted documents for " + spec["name"])
    partial.replace(path)
    report = {"identity": identity, "scanned_documents": scanned, "scanned_raw_text_bytes": raw_bytes,
              "accepted_candidate_documents": accepted, "candidate_clean_text_bytes": accepted_bytes,
              "target_reached": accepted_bytes >= target, "rejected": dict(rejected),
              "rejected_samples": rejected_samples, "file_sha256": file_sha256(path),
              "note": "Candidates before global deduplication. Source-specific rejection counters are first-match reasons."}
    write_json(manifest_path, report)
    return report


def assemble_candidates(paths, base_config, output, candidate_documents):
    """All sources enter one deduplication pass before document-level splitting."""
    output = Path(output)
    combined = output / "raw/combined_candidates.jsonl"
    combined.parent.mkdir(parents=True, exist_ok=True)
    temporary = combined.with_suffix(".partial")
    with temporary.open("wb") as target:
        # Wikipedia is first, so its cleaner extraction wins shared duplicates.
        for path in paths:
            with Path(path).open("rb") as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
    temporary.replace(combined)
    filters = copy.deepcopy(base_config)
    filters["trusted_metadata_language_sources"] = [WIKIPEDIA]
    filters["process_all_input"] = True
    filters["min_cyrillic_fraction"] = min(0.85, filters["min_cyrillic_fraction"])
    filters["max_scanned_documents"] = candidate_documents + 1
    write_json(output / "effective_cleaning.json", filters)
    manifest = prepare(filters, output / "corpus", combined)
    audit = audit_splits(output / "corpus")
    return {"corpus": str(output / "corpus"), "clean_text_bytes": manifest["clean_text_bytes"],
            "source_text_bytes": manifest["source_text_bytes"], "split_documents": manifest["split_documents"],
            "split_audit": audit}


def prepare_project_data(base_config, mixture_config, output, fineweb_mb=None, wiki_mb=None, candidates_only=False):
    """Prepare both sources; run separately for pilot and main corpus."""
    output = Path(output)
    mixture = copy.deepcopy(mixture_config)
    overrides = {"fineweb": fineweb_mb, "wikipedia": wiki_mb}
    if any(value is not None and value < 0 for value in overrides.values()):
        raise ValueError("Byte targets cannot be negative; use 0 to disable a source.")
    paths, reports = [], {}
    for spec in mixture["sources"]:
        if overrides.get(spec["name"]) is not None:
            spec["target_text_mb"] = overrides[spec["name"]]
        if spec["target_text_mb"] <= 0:
            continue
        path = output / ("raw/" + spec["name"] + ".jsonl")
        reports[spec["name"]] = collect_source(spec, base_config, path)
        paths.append(path)
    if not paths:
        raise ValueError("Enable at least one source.")
    documents = sum(report["accepted_candidate_documents"] for report in reports.values())
    cleaning = copy.deepcopy(base_config)
    cleaning["target_text_mb"] = sum(spec["target_text_mb"] for spec in mixture["sources"])
    if candidates_only:
        result = {"candidate_paths": [str(p) for p in paths], "candidate_documents": documents,
                  "candidate_clean_text_bytes": sum(r["candidate_clean_text_bytes"] for r in reports.values()),
                  "stage": "source-filtered candidates only; expand-corpus performs global dedup and frozen-holdout checks"}
    else:
        result = assemble_candidates(paths, cleaning, output, documents)
    result["source_targets_reached"] = {name: report["target_reached"] for name, report in reports.items()}
    result["note"] = "Byte targets refer to cleaned candidates BEFORE global deduplication; final size and proportions can differ."
    write_json(output / "mixture_config.json", mixture)
    write_json(output / "preparation_result.json", result)
    return result
