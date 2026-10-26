import numpy as np
import torch

from .utils import read_jsonl


class PackedSampler:
    """nanoGPT-style random windows with replacement from a uint16 token stream."""

    def __init__(self, path, context_length, batch_size, seed):
        self.data = np.memmap(path, mode="r", dtype=np.uint16)
        self.context_length, self.batch_size = context_length, batch_size
        self.rng = np.random.default_rng(seed)
        if len(self.data) <= context_length:
            raise ValueError(f"{path} needs at least context_length + 1 tokens.")

    def batch(self):
        starts = self.rng.integers(0, len(self.data) - self.context_length, size=self.batch_size)
        rows = np.stack([self.data[i : i + self.context_length + 1] for i in starts]).astype(
            np.int64
        )
        return {
            "input_ids": torch.from_numpy(rows[:, :-1].copy()),
            "labels": torch.from_numpy(rows[:, 1:].copy()),
            "attention_mask": None,
        }

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]


def format_question(question, context=None):
    prefix = "" if not context else "Контекст:\n" + context.strip() + "\n\n"
    return prefix + "Питање: " + question.strip() + "\nОдговор:"


def sft_example(row, tokenizer, context_length):
    prompt = row.get("prompt")
    if prompt is None:
        prompt = format_question(row["question"], row.get("context"))
    response = row.get("response", row.get("answer"))
    if not isinstance(prompt, str) or not isinstance(response, str) or not response.strip():
        raise ValueError("SFT rows need prompt/response or question/answer strings.")
    prefix = [tokenizer.token_to_id("[BOS]")] + tokenizer.encode(
        prompt, add_special_tokens=False
    ).ids
    answer = tokenizer.encode(" " + response.strip(), add_special_tokens=False).ids + [
        tokenizer.token_to_id("[EOS]")
    ]
    ids = prefix + answer
    if len(ids) - 1 > context_length:
        raise ValueError(
            f"SFT example {row.get('id')} exceeds context. Shorten it explicitly; no silent truncation."
        )
    labels = [-100] * len(prefix) + answer
    return ids[:-1], labels[1:]


class SFTSampler:
    def __init__(self, path, tokenizer, context_length, batch_size, seed):
        self.rows = list(read_jsonl(path))
        if not self.rows:
            raise ValueError("Empty SFT file.")
        self.examples = [sft_example(row, tokenizer, context_length) for row in self.rows]
        self.batch_size = batch_size
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.rng = np.random.default_rng(seed)

    def batch(self):
        indices = self.rng.integers(0, len(self.examples), size=self.batch_size)
        return self.collate([self.examples[i] for i in indices])

    def iter_batches(self):
        for start in range(0, len(self.examples), self.batch_size):
            yield self.collate(self.examples[start : start + self.batch_size])

    def collate(self, selected):
        length = max(len(x) for x, _ in selected)
        size = len(selected)
        inputs = torch.full((size, length), self.pad_id, dtype=torch.long)
        labels = torch.full((size, length), -100, dtype=torch.long)
        mask = torch.zeros(size, length, dtype=torch.bool)
        for i, (x, y) in enumerate(selected):
            inputs[i, : len(x)] = torch.tensor(x)
            labels[i, : len(y)] = torch.tensor(y)
            mask[i, : len(x)] = True
        return {"input_ids": inputs, "labels": labels, "attention_mask": mask}

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]


def validate_sft_splits(train_path, validation_path):
    train, validation = list(read_jsonl(train_path)), list(read_jsonl(validation_path))
    if not train or not validation:
        raise ValueError("SFT train and validation must both be nonempty.")
    for key in ["id", "group_id", "source_document_id", "template_id"]:
        a = {str(row[key]) for row in train if row.get(key) is not None}
        b = {str(row[key]) for row in validation if row.get(key) is not None}
        if a & b:
            raise ValueError(f"Cross-split SFT overlap in {key}.")

    def prompt(row):
        text = row.get("prompt") or format_question(row["question"], row.get("context"))
        return " ".join(text.casefold().split())

    if {prompt(row) for row in train} & {prompt(row) for row in validation}:
        raise ValueError("Identical SFT prompts in train and validation.")
    contexts = lambda rows: {
        " ".join(row["context"].casefold().split()) for row in rows if row.get("context")
    }
    if contexts(train) & contexts(validation):
        raise ValueError("Identical SFT source contexts in train and validation.")
    return {
        "train_examples": len(train),
        "validation_examples": len(validation),
        "note": "Checks exact prompts and supplied group IDs; semantic paraphrase leakage still needs auditing.",
    }
