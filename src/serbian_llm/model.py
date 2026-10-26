"""Decoder-only architectures used in the Serbian runs."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dimension, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(q, k, theta):
    dimension = q.size(-1)
    positions = torch.arange(q.size(-2), device=q.device, dtype=torch.float32)
    frequencies = theta ** (-torch.arange(0, dimension, 2, device=q.device).float() / dimension)
    angles = positions[:, None] * frequencies[None, :]
    angles = torch.cat((angles, angles), dim=-1)[None, None, :, :]
    cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        batch, length, _ = x.shape
        c = self.config
        q = self.q_proj(x).view(batch, length, c.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, c.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, c.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if c.positional == "rope":
            q, k = apply_rope(q, k, c.rope_theta)

        mask = None
        if attention_mask is not None:
            causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
            mask = causal[None, None, :, :] & attention_mask[:, None, None, :].bool()

        if c.attention_backend == "sdpa":
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=c.dropout if self.training else 0.0,
                is_causal=mask is None,
                enable_gqa=c.n_heads != c.n_kv_heads,
            )
        else:
            groups = c.n_heads // c.n_kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
            if mask is None:
                mask = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()[
                    None, None
                ]
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~mask, float("-inf"))
            probabilities = F.softmax(scores.float(), dim=-1).to(q.dtype)
            y = F.dropout(probabilities, p=c.dropout, training=self.training) @ v
        y = y.transpose(1, 2).contiguous().view(batch, length, c.d_model)
        return self.dropout(self.out_proj(y))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kind = config.ffn
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.gate_proj = (
            nn.Linear(config.d_model, config.d_ff, bias=False) if self.kind == "swiglu" else None
        )
        self.out_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.kind == "swiglu":
            hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        else:
            hidden = F.gelu(self.up_proj(x))
        return self.dropout(self.out_proj(hidden))


def make_norm(config):
    if config.norm == "rmsnorm":
        return RMSNorm(config.d_model, config.norm_eps)
    return nn.LayerNorm(config.d_model, eps=config.norm_eps, bias=False)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = make_norm(config)
        self.attention = Attention(config)
        self.norm2 = make_norm(config)
        self.ffn = FeedForward(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attention(self.norm1(x), attention_mask)
        return x + self.ffn(self.norm2(x))


class SmallLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positions = (
            nn.Embedding(config.context_length, config.d_model)
            if config.positional == "absolute"
            else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = make_norm(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        for name, parameter in self.named_parameters():
            if parameter.ndim >= 2:
                std = config.init_std
                if name.endswith("out_proj.weight"):
                    std /= math.sqrt(2 * config.n_layers)
                nn.init.normal_(parameter, mean=0.0, std=std)

    def forward(self, input_ids, labels=None, attention_mask=None, last_only=False):
        if input_ids.ndim != 2 or input_ids.size(1) > self.config.context_length:
            raise ValueError("Expected [batch, time] input within context_length.")
        if input_ids.size(1) == 0:
            raise ValueError("Input cannot be empty.")
        x = self.embedding(input_ids)
        if self.positions is not None:
            positions = torch.arange(input_ids.size(1), device=input_ids.device)
            x = x + self.positions(positions)
        x = self.dropout(x)
        for block in self.blocks:
            for _ in range(self.config.block_repeats):
                if self.config.gradient_checkpointing and self.training:
                    x = checkpoint(block, x, attention_mask, use_reentrant=False)
                else:
                    x = block(x, attention_mask)
        x = self.final_norm(x)
        if last_only:
            if labels is not None:
                raise ValueError("last_only is only for generation.")
            x = x[:, -1:, :]
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            # labels are already shifted by the batch builder. Never shift twice.
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
            )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40,
        eos_id=None,
        forbidden_ids=(),
    ):
        if input_ids.size(0) != 1:
            raise ValueError("This simple generation method accepts one prompt at a time.")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                window = input_ids[:, -self.config.context_length :]
                logits, _ = self(window, last_only=True)
                logits = logits[:, -1, :].float()
                for token_id in forbidden_ids:
                    logits[:, token_id] = float("-inf")
                if temperature <= 0:
                    token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k and top_k > 0:
                        values = logits.topk(min(top_k, logits.size(-1))).values
                        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                    token = torch.multinomial(F.softmax(logits, dim=-1), 1)
                input_ids = torch.cat((input_ids, token), dim=1)
                if eos_id is not None and token.item() == eos_id:
                    break
        finally:
            self.train(was_training)
        return input_ids

    def parameter_report(self):
        total = sum(p.numel() for p in self.parameters())
        embeddings = self.embedding.weight.numel()
        positions = 0 if self.positions is None else self.positions.weight.numel()
        return {
            "parameters": total,
            "unique_blocks": len(self.blocks),
            "block_repeats": self.config.block_repeats,
            "block_applications": len(self.blocks) * self.config.block_repeats,
            "qk_norm": self.config.qk_norm,
            "token_embedding_parameters": embeddings,
            "position_parameters": positions,
            "non_embedding_parameters": total - embeddings - positions,
            "fp32_weights_bytes": total * 4,
            "fp32_weights_gradients_adam_moments_bytes": total * 16,
            "note": "16P excludes activations, temporary tensors, optimizer step counters, and allocator overhead.",
        }
