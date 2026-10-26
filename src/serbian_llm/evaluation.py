import math
import unicodedata

import torch
from torch.nn import functional as F

from .inference import Generator
from .utils import file_sha256, precision_context, read_jsonl, write_json


@torch.inference_mode()
def score_text(model, tokenizer, text, device, precision="fp32", stride=None):
    """Score each content token once, including the first token conditioned on BOS.

    EOS is not scored; its absence from the byte denominator avoids mixing text
    likelihood and document-ending probability. Context resets at documents.
    """
    text = unicodedata.normalize("NFC", text)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids:
        return {"nll": 0.0, "tokens": 0, "bytes": 0}
    context_length = model.config.context_length
    stride = context_length // 2 if stride is None else stride
    if not 1 <= stride <= context_length:
        raise ValueError("Evaluation stride must be between 1 and context_length.")
    sequence = [tokenizer.token_to_id("[BOS]")] + ids
    nll = 0.0
    model.eval()
    for start in range(0, len(ids), stride):
        end = min(start + stride, len(ids))
        left = max(0, end - context_length)
        x = torch.tensor([sequence[left:end]], device=device)
        y = torch.tensor([sequence[left + 1 : end + 1]], device=device)
        context, _ = precision_context(device, precision)
        with context:
            logits, _ = model(x)
        offset = start - left
        logits = logits[:, offset:, :].float()
        y = y[:, offset:]
        nll += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
    return {"nll": nll, "tokens": len(ids), "bytes": len(text.encode("utf-8"))}


def evaluate_lm(
    checkpoint, corpus_jsonl, output, device="auto", precision="auto", limit=100, stride=None
):
    generator = Generator(checkpoint, device, precision)
    nll, tokens, byte_count, documents = 0.0, 0, 0, 0
    for row in read_jsonl(corpus_jsonl):
        result = score_text(
            generator.model,
            generator.tokenizer,
            row["text"],
            generator.device,
            generator.precision,
            stride,
        )
        nll += result["nll"]
        tokens += result["tokens"]
        byte_count += result["bytes"]
        documents += 1
        if limit and documents >= limit:
            break
    if not tokens or not byte_count:
        raise ValueError("Evaluation corpus contains no content tokens.")
    report = {
        "documents": documents,
        "content_tokens": tokens,
        "utf8_text_bytes": byte_count,
        "nll_nats": nll,
        "nll_per_token": nll / tokens,
        "perplexity_per_token": math.exp(min(50, nll / tokens)),
        "bits_per_byte": nll / (math.log(2) * byte_count),
        "context_length": generator.model.config.context_length,
        "stride": stride or generator.model.config.context_length // 2,
        "corpus_sha256": file_sha256(corpus_jsonl),
        "checkpoint_sha256": file_sha256(checkpoint),
        "precision": generator.precision,
        "limit": limit,
        "protocol": "NFC text, per-document BOS, rolling token windows, every content token scored once, EOS excluded.",
        "comparison_note": "Compare BPB on identical documents. Token-level perplexity is not comparable across tokenizers.",
    }
    write_json(output, report)
    return report


@torch.inference_mode()
def evaluate_sft(checkpoint, data, output, device="auto", precision="auto", batch_size=8):
    from .batching import SFTSampler

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = Generator(checkpoint, device, precision)
    sampler = SFTSampler(
        data, generator.tokenizer, generator.model.config.context_length, batch_size, 0
    )
    nll, tokens = 0.0, 0
    for batch in sampler.iter_batches():
        batch = {k: v.to(generator.device) for k, v in batch.items()}
        context, _ = precision_context(generator.device, generator.precision)
        with context:
            _, loss = generator.model(**batch)
        count = int((batch["labels"] != -100).sum())
        nll += loss.item() * count
        tokens += count
    report = {
        "examples": len(sampler.rows),
        "answer_tokens_including_eos": tokens,
        "nll_nats": nll,
        "nll_per_answer_token": nll / tokens,
        "perplexity_answer_tokens": math.exp(min(50, nll / tokens)),
        "checkpoint_step": generator.checkpoint["state"]["step"],
        "checkpoint_sha256": file_sha256(checkpoint),
        "data_sha256": file_sha256(data),
        "precision": generator.precision,
        "protocol": "Every example once; loss on answer and EOS only; prompt and right padding masked; no truncation.",
    }
    write_json(output, report)
    return report
