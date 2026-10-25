import json
import shutil
import unicodedata
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tqdm import tqdm

from .data import words_of
from .utils import file_sha256, read_json, read_jsonl, write_json

SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]


def texts(path):
    for row in read_jsonl(path):
        yield row["text"]


def train_tokenizer(corpus, output, vocab_size=8192):
    if not 260 <= vocab_size < 65536:
        raise ValueError("ByteLevel tokenizers require 260 <= vocab_size < 65536.")
    corpus, output = Path(corpus), Path(output)
    manifest = read_json(corpus / "manifest.json")
    output.mkdir(parents=True, exist_ok=True)
    identity = {"train_sha256": file_sha256(corpus / "train.jsonl"), "target_vocab_size": vocab_size,
                "normalizer": "NFC", "algorithm": "ByteLevel BPE", "min_frequency": 2}
    if identity["train_sha256"] != manifest["file_sha256"]["train"]:
        raise ValueError("Training corpus changed since preparation.")
    path = output / "tokenizer.json"
    if path.exists():
        if not (output / "tokenizer_meta.json").exists() or read_json(output / "tokenizer_meta.json")["identity"] != identity:
            raise ValueError("Different or incomplete tokenizer exists. Use a new output folder.")
        if file_sha256(path) != read_json(output / "tokenizer_meta.json")["tokenizer_sha256"]:
            raise ValueError("Tokenizer file changed after training. Use a new output folder.")
        return read_json(output / "tokenizer_meta.json")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer_options = dict(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                           initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    trainer = trainers.BpeTrainer(min_frequency=2, **trainer_options)
    tokenizer.train_from_iterator(texts(corpus / "train.jsonl"), trainer=trainer,
                                   length=manifest["split_documents"]["train"])
    for text in ["Здраво! Како си?", "Љубав, њива, ђак, ћирилица, џез.", "Београд је главни град Србије."]:
        assert tokenizer.decode(tokenizer.encode(text).ids, skip_special_tokens=False) == unicodedata.normalize("NFC", text)
    tokenizer.save(str(path))
    meta = {"identity": identity, "vocab_size": tokenizer.get_vocab_size(), "tokenizer_sha256": file_sha256(path),
            "special_ids": {name: tokenizer.token_to_id(name) for name in SPECIAL_TOKENS}}
    write_json(output / "tokenizer_meta.json", meta)
    return meta


def encode_corpus(corpus, tokenizer_folder, include_test=False, output=None, reuse_tokenizer=False):
    corpus, source = Path(corpus), Path(tokenizer_folder)
    folder = Path(output) if output else source
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    tokenizer_hash = file_sha256(source / "tokenizer.json")
    manifest = read_json(corpus / "manifest.json")
    meta = read_json(source / "tokenizer_meta.json")
    if meta["identity"]["train_sha256"] != manifest["file_sha256"]["train"]:
        if not reuse_tokenizer:
            raise ValueError("Tokenizer was trained on another corpus. For continuation use --reuse-tokenizer --output NEW_DIRECTORY.")
        if folder.resolve() == source.resolve():
            raise ValueError("Encoding another corpus requires a separate --output directory.")
    if meta["tokenizer_sha256"] != tokenizer_hash:
        raise ValueError("Tokenizer changed after training.")
    if tokenizer.get_vocab_size() >= 65536:
        raise ValueError("uint16 storage requires vocab_size < 65536.")
    folder.mkdir(parents=True, exist_ok=True)
    if folder.resolve() != source.resolve():
        for name in ["tokenizer.json", "tokenizer_meta.json"]:
            destination = folder / name
            if destination.exists() and file_sha256(destination) != file_sha256(source / name):
                raise ValueError("Different tokenizer exists in encoding output. Use a new directory.")
            if not destination.exists():
                shutil.copyfile(source / name, destination)
    report_path = folder / "token_stats.json"
    if report_path.exists():
        report = read_json(report_path)
        if report["tokenizer_sha256"] != tokenizer_hash or report["corpus_files"] != manifest["file_sha256"]:
            raise ValueError("Existing encoded data has another identity. Use a new output folder.")
    else:
        report = {"tokenizer_sha256": tokenizer_hash, "vocab_size": tokenizer.get_vocab_size(),
                  "corpus_files": manifest["file_sha256"], "dtype": "uint16", "splits": {}}
    report["document_format"] = "BOS + content + EOS"
    if reuse_tokenizer:
        report["tokenizer_training_corpus_sha256"] = meta["identity"]["train_sha256"]
        report["reused_tokenizer"] = True
    for split in (["train", "validation", "test"] if include_test else ["train", "validation"]):
        path = corpus / (split + ".jsonl")
        if file_sha256(path) != manifest["file_sha256"][split]:
            raise ValueError(f"Modified corpus: {split}")
        counts = np.zeros(tokenizer.get_vocab_size(), dtype=np.int64)
        tokens = content_tokens = byte_count = word_count = unk_count = documents = 0
        temporary = folder / (split + ".bin.partial")
        with temporary.open("wb") as file:
            for text in tqdm(texts(path), total=manifest["split_documents"][split], desc="Encode " + split):
                normalized = unicodedata.normalize("NFC", text)
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                content_tokens += len(ids)
                unk_count += ids.count(tokenizer.token_to_id("[UNK]"))
                byte_count += len(normalized.encode("utf-8"))
                word_count += len(words_of(normalized))
                ids = [tokenizer.token_to_id("[BOS]")] + ids + [tokenizer.token_to_id("[EOS]")]
                values = np.asarray(ids, dtype=np.uint16)
                values.tofile(file)
                counts += np.bincount(values.astype(np.int64), minlength=len(counts))
                tokens += len(ids)
                documents += 1
        temporary.replace(folder / (split + ".bin"))
        report["splits"][split] = {"tokens": tokens, "content_tokens": content_tokens, "text_bytes": byte_count,
                                  "words": word_count, "documents": documents,
                                  "tokens_per_word": content_tokens / max(1, word_count),
                                  "bytes_per_token": byte_count / max(1, content_tokens),
                                  "unknown_fraction": unk_count / max(1, content_tokens),
                                  "bin_sha256": file_sha256(folder / (split + ".bin"))}
        if split == "train":
            special_ids = set(meta["special_ids"].values())
            lexical = np.array([n for i, n in enumerate(counts) if i not in special_ids])
            report["unused_vocabulary_fraction"] = float(np.mean(lexical == 0))
            report["vocabulary_fraction_below_5_occurrences"] = float(np.mean(lexical < 5))
            with (folder / "token_frequencies.jsonl").open("w", encoding="utf-8") as file:
                for token_id, count in enumerate(counts):
                    file.write(json.dumps({"id": token_id, "token": tokenizer.id_to_token(token_id),
                                           "count": int(count)}, ensure_ascii=False) + "\n")
        write_json(report_path, report)
    return report
