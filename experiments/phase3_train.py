"""Phase 3: train one small GPT on the synthetic inventions domain and evaluate
provable novelty (metrics M5/M6 + the continuous mirage-check, see README).

Every generated sentence is parsed and its triple exactly classified:
  train-dup   — triple was in training data (memorization, M6)
  L1          — valid, held out at triple level  → provable P-creative output
  L2          — valid, its (device,mechanism) pair NEVER seen  → deeper novelty
  invalid     — grammatical but breaks the latent compatibility rule
  unparseable — not even grammatical

Chance floor: a grammar-respecting model with no rule knowledge scores ~16% valid.

Also records mean per-token log-prob assigned to held-out-valid vs matched invalid
triples (continuous companion metric, H5).

Usage: uv run python experiments/phase3_train.py --size s2 [--steps 3000] [--device cuda]
Writes results/phase3/run_<size>.json (+ samples file for reading).
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from original_thought.domain import (BOS, EOS, PAD, build_domain, build_vocab,
                                     encode, sample_invalid)
from original_thought.model import GPT, GPTConfig

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "phase3"
BLOCK = 12

SIZES = {  # name -> (n_layer, n_head, n_embd)
    "s0": (1, 1, 8),
    "s1": (1, 2, 16),
    "s2": (2, 2, 32),
    "s3": (2, 4, 64),
    "s4": (4, 4, 128),
    "s5": (6, 8, 256),
    "s6": (8, 8, 384),
}
N_SAMPLES = 4000  # generations per eval


def get_device(arg):
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataset(dom, vocab, rng):
    rows = []
    for t in dom.train:
        for tpl in range(3):
            rows.append(encode(dom.render(t, tpl), vocab, BLOCK))
    rng.shuffle(rows)
    return torch.tensor(rows, dtype=torch.long)


def evaluate(model, dom, vocab, inv_vocab, device, temperature=1.0):
    counts = {"train_dup": 0, "l1": 0, "l2": 0, "invalid": 0, "unparseable": 0}
    uniq = {"l1": set(), "l2": set()}
    samples = []
    done = 0
    while done < N_SAMPLES:
        batch = min(1000, N_SAMPLES - done)
        for seq in model.sample(vocab[BOS], vocab[EOS], batch, temperature, device,
                                max_len=BLOCK):
            toks = [inv_vocab[i] for i in seq if inv_vocab[i] != PAD]
            triple = dom.parse(toks)
            if triple is None:
                counts["unparseable"] += 1
                kind = "unparseable"
            elif triple in dom.train:
                counts["train_dup"] += 1
                kind = "train_dup"
            elif triple in dom.l1:
                counts["l1"] += 1
                uniq["l1"].add(triple)
                kind = "l1"
            elif triple in dom.l2:
                counts["l2"] += 1
                uniq["l2"].add(triple)
                kind = "l2"
            else:
                counts["invalid"] += 1
                kind = "invalid"
            if len(samples) < 60:
                samples.append((kind, " ".join(toks)))
        done += batch

    n = float(N_SAMPLES)
    parsed = n - counts["unparseable"]
    novel = counts["l1"] + counts["l2"] + counts["invalid"]  # non-memorized, parseable
    metrics = {
        "rate": {k: v / n for k, v in counts.items()},
        "grammar_rate": parsed / n,
        # M5 headline: of novel (non-training) grammatical outputs, how many obey
        # the hidden rule? chance floor ≈ 0.16 for L1+L2 combined
        "novel_valid_precision": (counts["l1"] + counts["l2"]) / novel if novel else 0.0,
        "unique_l1": len(uniq["l1"]),
        "unique_l2": len(uniq["l2"]),
        "l1_coverage": len(uniq["l1"]) / len(dom.l1),
        "l2_coverage": len(uniq["l2"]) / len(dom.l2),
    }
    return metrics, samples


def logprob_gap(model, dom, vocab, device):
    """Continuous companion: mean log-prob on held-out valid vs invalid triples."""
    rng = random.Random(7)
    held = rng.sample(sorted(dom.l1 | dom.l2), 500)
    inv = sample_invalid(dom, 500, rng)
    out = {}
    for name, triples in [("heldout_valid", held), ("invalid", inv)]:
        ids = torch.tensor([encode(dom.render(t, i % 3), vocab, BLOCK)
                            for i, t in enumerate(triples)], device=device)
        lps = []
        for i in range(0, len(ids), 250):
            lps.append(model.seq_logprob(ids[i:i + 250], vocab[PAD]))
        out[name] = float(torch.cat(lps).mean())
    out["gap"] = out["heldout_valid"] - out["invalid"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=list(SIZES))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = get_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    dom = build_domain()
    vocab = build_vocab(dom)
    inv_vocab = {v: k for k, v in vocab.items()}
    data = make_dataset(dom, vocab, rng).to(device)

    L, H, E = SIZES[args.size]
    model = GPT(GPTConfig(vocab_size=len(vocab), block_size=BLOCK,
                          n_layer=L, n_head=H, n_embd=E)).to(device)
    n_params = model.n_params()
    print(f"size={args.size} params={n_params:,} device={device} rows={len(data)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        i = torch.randint(0, len(data), (args.batch,), device=device)
        batch = data[i]
        _, loss = model(batch[:, :-1], batch[:, 1:], pad_id=vocab[PAD])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"  step {step} loss {loss.item():.4f} ({time.time()-t0:.0f}s)")

    metrics, samples = evaluate(model, dom, vocab, inv_vocab, device)
    lp = logprob_gap(model, dom, vocab, device)

    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"size": args.size, "n_params": n_params, "steps": args.steps,
           "seed": args.seed, "final_loss": float(loss.item()),
           "metrics": metrics, "logprob": lp,
           "train_secs": round(time.time() - t0, 1)}
    (OUT / f"run_{args.size}_seed{args.seed}.json").write_text(json.dumps(rec, indent=2))
    with (OUT / f"samples_{args.size}_seed{args.seed}.txt").open("w") as f:
        for kind, s in samples:
            f.write(f"[{kind}] {s}\n")

    print(json.dumps({k: metrics[k] for k in
                      ("rate", "novel_valid_precision", "unique_l1", "unique_l2")},
                     indent=2))
    print("logprob gap:", round(lp["gap"], 4))


if __name__ == "__main__":
    main()
