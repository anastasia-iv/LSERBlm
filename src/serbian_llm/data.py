import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from .utils import file_sha256, read_json, read_jsonl, write_json
from .streaming import stream_documents

SERBIAN_LETTERS = set("абвгдђежзијклљмнњопрстћуфхцчџш")
CYRILLIC = set(chr(i) for i in range(0x0400, 0x0530))


def words_of(text):
    return re.findall(r"[^\W\d_]+", text.casefold())


def normalize_text(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(c for c in text if not unicodedata.category(c).startswith("C") or c in "\n\t")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def text_signature(text):
    # Only whitespace and Unicode normalization define exact duplicates.
    # Keep numbers, punctuation and case: different dates/facts are not exact copies.
    normalized = " ".join(normalize_text(text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def wikipedia_quality_reason(text, title):
    """Conservative prose heuristics, applied only to the Wikipedia source."""
    title = (title or "").casefold().strip()
    lead = text.split("\n", 1)[0].casefold().strip()
    if ("(вишезначна одредница)" in title or "(разврставање)" in title
            or (len(lead) < 250 and lead.endswith(":") and re.search(
                r"\b(?:има више значења|може (?:да се односи|се односити) на|може означавати)\b", lead))):
        return "wikipedia_disambiguation"
    if title.startswith(("списак ", "spisak ")):
        return "wikipedia_list"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lengths = [len(words_of(line)) for line in lines]
    if len(lines) >= 8 and sum(n < 20 for n in lengths) / len(lines) >= 0.8:
        prose_words = sum(n for line, n in zip(lines, lengths)
                          if n >= 20 and re.search(r"[.!?][\s\"»”’)]*$", line))
        if prose_words / max(1, sum(lengths)) < 0.35:
            return "wikipedia_list"
    return None


def strip_wikipedia_tail(text):
    """Cut only standalone terminal-section headings, never inline mentions."""
    headings = {"референце", "извори", "литература", "спољашње везе", "види још",
                "напомене", "белешке", "reference", "izvori", "literatura",
                "spoljašnje veze", "vidi još", "napomene", "beleške"}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        heading = line.strip().strip("=# ").rstrip(":").casefold()
        if heading in headings:
            return "\n".join(lines[:index]).rstrip()
    return text


def web_noise_reason(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return None
    noisy = sum(bool(re.match(r"^(?:https?://|www\.)", line)) or line.count("|") >= 3
                or (len(words_of(line)) < 8 and not re.search(r"[.!?]$", line)) for line in lines)
    return "web_navigation_or_table" if noisy / len(lines) >= 0.7 else None


def clean_document(row, config):
    text = normalize_text(row.get("text") or "")
    if row.get("source") == "wikimedia/wikipedia" and config.get("strip_wikipedia_tail", False):
        text = strip_wikipedia_tail(text)
    if not config["min_characters"] <= len(text) <= config["max_characters"]:
        return text, "length"
    if row.get("language") not in {"srp", "srp_Cyrl"}:
        return text, "source_language"
    score = row.get("language_score")
    trusted = row.get("source") in config.get("trusted_metadata_language_sources", [])
    if score is None and not trusted:
        return text, "missing_language_score"
    if score is not None and float(score) < config["min_language_score"]:
        return text, "source_language_score"
    letters = [c for c in text.casefold() if c.isalpha()]
    if not letters:
        return text, "no_letters"
    fraction = sum(c in CYRILLIC for c in letters) / len(letters)
    if fraction < config["min_cyrillic_fraction"]:
        return text, "script"
    other = sum(c in CYRILLIC and c not in SERBIAN_LETTERS for c in letters)
    if other / len(letters) > config["max_other_cyrillic_fraction"]:
        return text, "other_cyrillic_letters"
    if len(words_of(text)) < config["min_words"]:
        return text, "few_words"
    if row.get("source") == "wikimedia/wikipedia" and config.get("filter_wikipedia_lists", True):
        reason = wikipedia_quality_reason(text, row.get("title"))
        if reason:
            return text, reason
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 30]
    if len(lines) >= 5 and 1 - len(set(lines)) / len(lines) > 0.4:
        return text, "repeated_lines"
    if config.get("filter_web_noise", False):
        reason = web_noise_reason(text)
        if reason:
            return text, reason
    return text, None


def sketch_of(text, config):
    words = words_of(text)
    shingles = {" ".join(words[i:i + 5]) for i in range(max(0, len(words) - 4))}
    if not shingles:
        shingles = {" ".join(words)}
    shingles = sorted(shingles, key=lambda s: hashlib.blake2b(s.encode(), digest_size=8).digest())
    shingles = shingles[:config["max_shingles"]]
    sketch = MinHash(num_perm=config["minhash_permutations"], seed=config["seed"])
    sketch.update_batch([s.encode("utf-8") for s in shingles])
    return sketch


def choose_split(signature, config):
    value = int(signature[:12], 16) % 10000
    test_boundary = round(config["test_fraction"] * 10000)
    validation_boundary = test_boundary + round(config["validation_fraction"] * 10000)
    if value < test_boundary:
        return "test"
    if value < validation_boundary:
        return "validation"
    return "train"


def prepare(config, output, input_jsonl=None):
    config = dict(config)
    if config["validation_fraction"] <= 0 or config["test_fraction"] <= 0:
        raise ValueError("Validation and test fractions must be positive.")
    if config["validation_fraction"] + config["test_fraction"] >= 1:
        raise ValueError("A nonempty training fraction is required.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_info = {"kind": "local", "sha256": file_sha256(input_jsonl)} if input_jsonl else None
    if source_info is None:
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(config["dataset"], revision=config["revision"]).sha
        source_info = {"kind": "huggingface", "dataset": config["dataset"], "subset": config["subset"],
                       "revision": revision, "source_split": "train"}
    identity = {"pipeline_version": "serbian-clean-v4", "config": config, "source": source_info}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("Output contains a different corpus. Choose a new --output directory.")
        for split, digest in manifest["file_sha256"].items():
            if file_sha256(output / (split + ".jsonl")) != digest:
                raise ValueError("Corpus file was modified; use a new directory.")
        return manifest
    paths = {s: output / (s + ".jsonl") for s in ["train", "validation", "test"]}
    if any(p.exists() for p in paths.values()):
        raise RuntimeError("Incomplete preparation detected. Use a new --output directory; partial data is not trusted.")
    if input_jsonl:
        source = read_jsonl(input_jsonl)
    else:
        source = stream_documents(config["dataset"], config["subset"], source_info["revision"],
                                  config["seed"], config["shuffle_buffer"])
    seen = set()
    lsh = MinHashLSH(threshold=config["near_duplicate_threshold"], num_perm=config["minhash_permutations"])
    rejected, counts, byte_counts, domains = Counter(), Counter(), Counter(), Counter()
    source_counts, source_bytes, split_sources = Counter(), Counter(), {s: Counter() for s in paths}
    raw_words, train_words = Counter(), Counter()
    raw_lengths, train_lengths, audit, rejected_examples = [], [], [], []
    total_bytes = raw_bytes = scanned = 0
    target_bytes = round(config["target_text_mb"] * 1_000_000)
    files = {s: p.open("x", encoding="utf-8") for s, p in paths.items()}
    bar = tqdm(total=target_bytes, unit="B", unit_scale=True, desc="Clean text")
    try:
        for row in source:
            scanned += 1
            raw = row.get("text") or ""
            raw_bytes += len(raw.encode("utf-8"))
            raw_lengths.append(len(raw))
            raw_words.update(words_of(raw))
            text, reason = clean_document(row, config)
            if reason is None:
                signature = text_signature(text)
                if signature in seen:
                    reason = "exact_duplicate"
                elif config["near_dedup"]:
                    sketch = sketch_of(text, config)
                    if lsh.query(sketch):
                        reason = "approximate_duplicate"
            if reason is not None:
                rejected[reason] += 1
                if len(rejected_examples) < 100:
                    rejected_examples.append({"reason": reason, "text": text[:1500]})
            else:
                seen.add(signature)
                if config["near_dedup"]:
                    lsh.insert(signature, sketch)
                split = choose_split(signature, config)
                domain = urlparse(row.get("url") or "").netloc
                record = {"id": signature, "text": text, "source_url": row.get("url"), "domain": domain,
                          "source_id": row.get("id"), "language_score": row.get("language_score")}
                for key in ["source", "source_revision", "source_subset", "title", "language_evidence"]:
                    if key in row:
                        record[key] = row[key]
                files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                size = len(text.encode("utf-8"))
                total_bytes += size
                counts[split] += 1
                byte_counts[split] += size
                source_name = row.get("source", config["dataset"])
                source_counts[source_name] += 1
                source_bytes[source_name] += size
                split_sources[split][source_name] += 1
                if split == "train":
                    train_words.update(words_of(text))
                    train_lengths.append(len(text))
                    domains[domain] += 1
                    if len(audit) < 100:
                        audit.append(record)
                bar.update(size)
            if scanned % 5000 == 0:
                for handle in files.values():
                    handle.flush()
                write_json(output / "progress.json", {"scanned": scanned, "text_bytes": total_bytes, "rejected": dict(rejected)})
            byte_limit = total_bytes >= target_bytes and not config.get("process_all_input", False)
            if byte_limit or scanned >= config["max_scanned_documents"]:
                break
    finally:
        source.close()
        for handle in files.values():
            handle.close()
        bar.close()
    if any(counts[s] == 0 for s in paths):
        raise RuntimeError("A split is empty; increase corpus size in a new output directory.")
    for filename, records in [("audit_samples.jsonl", audit), ("rejected_samples.jsonl", rejected_examples)]:
        with (output / filename).open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output / "train_word_frequencies.tsv").open("w", encoding="utf-8") as file:
        file.write("word\tcount\n")
        for word, count in train_words.most_common():
            file.write(f"{word}\t{count}\n")
    np.savez_compressed(output / "lengths.npz", raw=np.array(raw_lengths), train=np.array(train_lengths))
    manifest = {"identity": identity, "scanned_documents": scanned, "scanned_raw_text_bytes": raw_bytes,
                "clean_text_bytes": total_bytes, "target_reached": total_bytes >= target_bytes,
                "split_documents": dict(counts), "split_text_bytes": dict(byte_counts), "rejected": dict(rejected),
                "source_documents": dict(source_counts), "source_text_bytes": dict(source_bytes),
                "split_source_documents": {s: dict(c) for s, c in split_sources.items()},
                "train_words": sum(train_words.values()), "train_word_types": len(train_words),
                "train_hapax_types": sum(n == 1 for n in train_words.values()),
                "raw_top_words": raw_words.most_common(50), "train_top_domains": domains.most_common(30),
                "file_sha256": {s: file_sha256(p) for s, p in paths.items()},
                "notes": ["Raw statistics cover all scanned records; clean frequencies cover train only.",
                          "Source classifier scores or explicitly trusted edition metadata plus script heuristics are not independent language verification.",
                          "MinHash is approximate and does not prove zero near-duplicate leakage."]}
    write_json(manifest_path, manifest)
    plot_corpus(output)
    return manifest


def plot_corpus(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    lengths = np.load(folder / "lengths.npz")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ["raw", "train"]):
        ax.hist(np.log10(lengths[key] + 1), bins=50)
        ax.set(title=key, xlabel="log10(characters + 1)", ylabel="documents")
    fig.tight_layout()
    fig.savefig(folder / "lengths.png", dpi=150)
    plt.close(fig)
    frequencies = []
    with (folder / "train_word_frequencies.tsv").open(encoding="utf-8") as file:
        next(file)
        for line in file:
            frequencies.append(int(line.rsplit("\t", 1)[1]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(np.arange(1, len(frequencies) + 1), frequencies)
    ax.set(xlabel="word rank", ylabel="train frequency")
    fig.tight_layout()
    fig.savefig(folder / "word_frequencies.png", dpi=150)
    plt.close(fig)


def audit_splits(folder, minimum_paragraph_characters=200, output=None):
    folder = Path(folder)
    seen_ids, seen_paragraphs = {}, {}
    exact = paragraph_collisions = 0
    examples = []
    pairs = Counter()
    for split in ["train", "validation", "test"]:
        for row in read_jsonl(folder / (split + ".jsonl")):
            signature = text_signature(row["text"])
            previous = seen_ids.get(signature)
            exact += int(previous is not None and previous != split)
            seen_ids[signature] = split
            for paragraph in row["text"].split("\n"):
                if len(paragraph.strip()) < minimum_paragraph_characters:
                    continue
                key = text_signature(paragraph)
                previous = seen_paragraphs.get(key)
                collision = previous is not None and previous[0] != split
                paragraph_collisions += int(collision)
                if collision and len(examples) < 100:
                    examples.append({"first_split": previous[0], "first_document": previous[1],
                                     "second_split": split, "second_document": row.get("id", signature),
                                     "paragraph_sha256": key, "paragraph": paragraph[:2000]})
                if collision:
                    pairs[f"{previous[0]}:{split}"] += 1
                seen_paragraphs.setdefault(key, (split, row.get("id", signature)))
    report = {"exact_cross_split_documents": exact, "shared_long_paragraph_occurrences": paragraph_collisions,
              "minimum_paragraph_characters": minimum_paragraph_characters,
              "shared_paragraph_examples": examples,
              "shared_paragraph_split_pairs": dict(pairs),
              "note": "Shared paragraphs can be quotations or boilerplate; inspect before interpreting as leakage."}
    write_json(output or folder / "split_audit.json", report)
    return report
