import torch

from .batching import format_question
from .checkpoints import load_for_inference
from .utils import precision_context, resolve_device


class Generator:
    def __init__(self, checkpoint, device="auto", precision="auto"):
        self.device = resolve_device(device)
        self.model, self.tokenizer, self.checkpoint = load_for_inference(checkpoint, self.device)
        _, self.precision = precision_context(self.device, precision)

    def generate(self, prompt, max_new_tokens=100, temperature=0.8, top_k=40, seed=42):
        tokenizer = self.tokenizer
        ids = [tokenizer.token_to_id("[BOS]")] + tokenizer.encode(prompt, add_special_tokens=False).ids
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            inputs = torch.tensor([ids], device=self.device)
            context, _ = precision_context(self.device, self.precision)
            with context:
                output = self.model.generate(inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                    top_k=top_k, eos_id=tokenizer.token_to_id("[EOS]"),
                    forbidden_ids=[tokenizer.token_to_id("[PAD]"), tokenizer.token_to_id("[BOS]")])
        completion = tokenizer.decode(output[0, len(ids):].tolist(), skip_special_tokens=True)
        return {"prompt": prompt, "completion": completion, "prompt_tokens": len(ids),
                "prompt_was_truncated": len(ids) > self.model.config.context_length,
                "generated_tokens": output.size(1) - len(ids)}

    def answer(self, question, context=None, **kwargs):
        return self.generate(format_question(question, context), **kwargs)
