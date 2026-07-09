"""Minimal GPT for the Phase 3 scale sweep (decoder-only, learned positions)."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 16
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.1


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.attn = nn.MultiheadAttention(c.n_embd, c.n_head, dropout=c.dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(c.n_embd, 4 * c.n_embd), nn.GELU(),
            nn.Linear(4 * c.n_embd, c.n_embd), nn.Dropout(c.dropout),
        )
        mask = torch.triu(torch.ones(c.block_size, c.block_size, dtype=torch.bool), 1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        T = x.size(1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:T, :T], need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, pad_id: int | None = None):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-1 if pad_id is None else pad_id)
        return logits, loss

    @torch.no_grad()
    def sample(self, bos_id: int, eos_id: int, n: int, temperature: float,
               device, max_len: int | None = None) -> list[list[int]]:
        self.eval()
        max_len = max_len or self.c.block_size
        idx = torch.full((n, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            nxt[finished] = eos_id
            idx = torch.cat([idx, nxt], 1)
            finished |= nxt.squeeze(1) == eos_id
            if finished.all():
                break
        out = []
        for row in idx.tolist():
            seq = []
            for t in row[1:]:
                if t == eos_id:
                    break
                seq.append(t)
            out.append(seq)
        return out

    @torch.no_grad()
    def seq_logprob(self, ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        """Mean per-token log-prob of each padded sequence (BOS included in input)."""
        self.eval()
        logits, _ = self(ids[:, :-1])
        logp = F.log_softmax(logits, -1)
        tgt = ids[:, 1:]
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        mask = tgt != pad_id
        return (tok_lp * mask).sum(1) / mask.sum(1)
