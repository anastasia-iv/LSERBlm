import numpy as np
import torch



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
        rows = np.stack([self.data[i:i + self.context_length + 1] for i in starts]).astype(np.int64)
        return {"input_ids": torch.from_numpy(rows[:, :-1].copy()),
                "labels": torch.from_numpy(rows[:, 1:].copy()), "attention_mask": None}

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]


def format_question(question, context=None):
    prefix = "" if not context else "Контекст:\n" + context.strip() + "\n\n"
    return prefix + "Питање: " + question.strip() + "\nОдговор:"
