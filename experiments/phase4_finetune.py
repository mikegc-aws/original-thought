"""Phase 4: is mode collapse trained-in, and is it reversible?

Design — one base model (s4, 816k params: high precision AND high pre-collapse
novelty in Phase 3), pretrained 6k steps, then three continuations of 3k steps
from IDENTICAL weights:

  control    — continue on the full training set (Phase 3 regime; expected:
               collapse deepens with training, per Phase 3 result 3)
  preference — continue on a "preferred" subset: 2% of training triples, one
               template (simulates preference-tuning's narrow high-reward
               distribution; expected: severe collapse)
  diversity  — iterative self-training: sample the model, keep only grammatical
               outputs that are NOT training duplicates and not already kept,
               fine-tune on them, repeat. NO validity oracle is used — the model
               gets no signal about the hidden rule, only "be new and
               grammatical". The open question: does rule-adherence (M5
               precision) survive, or does diversity training destroy validity?

Eval after each 500-step chunk with the Phase 3 evaluator (same metrics), so we
get trajectories, not just endpoints. Runs all three arms for several seeds.

Usage: uv run python experiments/phase4_finetune.py [--seeds 2] [--device cuda]
Writes results/phase4/traj_<arm>_seed<k>.json
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
                                     encode)
from original_thought.model import GPT, GPTConfig

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "phase4"
BLOCK = 12
SIZE = (4, 4, 128)  # s4
PRETRAIN_STEPS = 6000
FT_STEPS = 3000
EVAL_EVERY = 500
BATCH = 256
N_EVAL = 4000
PREF_TRIPLE_FRAC = 0.02
SELF_TRAIN_POOL = 8000  # generations sampled per diversity round


def get_device(arg):
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rows_to_tensor(rows, device):
    return torch.tensor(rows, dtype=torch.long, device=device)


def make_rows(dom, vocab, triples, templates=(0, 1, 2)):
    return [encode(dom.render(t, tpl), vocab, BLOCK) for t in triples for tpl in templates]


def train_steps(model, data, steps, opt, vocab, device):
    model.train()
    for _ in range(steps):
        i = torch.randint(0, len(data), (BATCH,), device=device)
        b = data[i]
        _, loss = model(b[:, :-1], b[:, 1:], pad_id=vocab[PAD])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return float(loss.item())


def evaluate(model, dom, vocab, inv_vocab, device):
    counts = {"train_dup": 0, "l1": 0, "l2": 0, "invalid": 0, "unparseable": 0}
    uniq = set()
    done = 0
    while done < N_EVAL:
        batch = min(1000, N_EVAL - done)
        for seq in model.sample(vocab[BOS], vocab[EOS], batch, 1.0, device, max_len=BLOCK):
            toks = [inv_vocab[i] for i in seq if inv_vocab[i] != PAD]
            triple = dom.parse(toks)
            if triple is None:
                counts["unparseable"] += 1
            elif triple in dom.train:
                counts["train_dup"] += 1
            elif triple in dom.l1 or triple in dom.l2:
                counts["l1" if triple in dom.l1 else "l2"] += 1
                uniq.add(triple)
            else:
                counts["invalid"] += 1
        done += batch
    n = float(N_EVAL)
    novel = counts["l1"] + counts["l2"] + counts["invalid"]
    return {
        "rate": {k: v / n for k, v in counts.items()},
        "novel_valid_precision": (counts["l1"] + counts["l2"]) / novel if novel else 0.0,
        "unique_heldout": len(uniq),
    }


def diversity_round(model, dom, vocab, inv_vocab, device, seen_selftrain):
    """Sample; keep grammatical non-training-duplicate outputs (no validity oracle)."""
    kept = []
    for seq in model.sample(vocab[BOS], vocab[EOS], SELF_TRAIN_POOL, 1.2, device,
                            max_len=BLOCK):
        toks = [inv_vocab[i] for i in seq if inv_vocab[i] != PAD]
        triple = dom.parse(toks)  # grammar check only; validity NOT consulted
        if triple is None or triple in dom.train or triple in seen_selftrain:
            continue
        seen_selftrain.add(triple)
        kept.append(triple)
    return kept


def run_arm(arm, seed, dom, vocab, inv_vocab, device, base_state, full_data):
    torch.manual_seed(seed + 1000)
    L, H, E = SIZE
    model = GPT(GPTConfig(vocab_size=len(vocab), block_size=BLOCK,
                          n_layer=L, n_head=H, n_embd=E)).to(device)
    model.load_state_dict(base_state)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    rng = random.Random(seed)
    if arm == "preference":
        pref = rng.sample(sorted(dom.train), int(PREF_TRIPLE_FRAC * len(dom.train)))
        data = rows_to_tensor(make_rows(dom, vocab, pref, templates=(0,)), device)
    elif arm == "control":
        data = full_data
    else:  # diversity: data refreshed each chunk
        data = None

    seen_selftrain: set = set()
    traj = [{"step": 0, **evaluate(model, dom, vocab, inv_vocab, device)}]
    for chunk_end in range(EVAL_EVERY, FT_STEPS + 1, EVAL_EVERY):
        if arm == "diversity":
            kept = diversity_round(model, dom, vocab, inv_vocab, device, seen_selftrain)
            # mix self-generated novelties with a slice of real data to anchor grammar
            anchor = rng.sample(sorted(dom.train), max(len(kept) // 4, 50))
            rows = make_rows(dom, vocab, kept) + make_rows(dom, vocab, anchor)
            if not rows:
                print(f"    diversity round produced nothing to train on")
                rows = make_rows(dom, vocab, anchor)
            data = rows_to_tensor(rows, device)
        train_steps(model, data, EVAL_EVERY, opt, vocab, device)
        m = evaluate(model, dom, vocab, inv_vocab, device)
        traj.append({"step": chunk_end, **m,
                     **({"selftrain_pool": len(seen_selftrain)} if arm == "diversity" else {})})
        print(f"  [{arm} s{seed}] step {chunk_end}: dup={m['rate']['train_dup']:.2f} "
              f"novel_prec={m['novel_valid_precision']:.2f} uniq={m['unique_heldout']}")
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = get_device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    dom = build_domain()
    vocab = build_vocab(dom)
    inv_vocab = {v: k for k, v in vocab.items()}

    for seed in range(args.seeds):
        rng = random.Random(seed)
        rows = make_rows(dom, vocab, sorted(dom.train))
        rng.shuffle(rows)
        full_data = rows_to_tensor(rows, device)

        # shared pretrained base for the three arms
        torch.manual_seed(seed)
        L, H, E = SIZE
        base = GPT(GPTConfig(vocab_size=len(vocab), block_size=BLOCK,
                             n_layer=L, n_head=H, n_embd=E)).to(device)
        opt = torch.optim.AdamW(base.parameters(), lr=3e-4, weight_decay=0.01)
        t0 = time.time()
        loss = train_steps(base, full_data, PRETRAIN_STEPS, opt, vocab, device)
        print(f"seed {seed}: pretrained {PRETRAIN_STEPS} steps, "
              f"loss {loss:.3f} ({time.time()-t0:.0f}s)")
        base_state = {k: v.clone() for k, v in base.state_dict().items()}

        for arm in ("control", "preference", "diversity"):
            out = OUT / f"traj_{arm}_seed{seed}.json"
            if out.exists():
                print(f"skip {arm} seed{seed}")
                continue
            traj = run_arm(arm, seed, dom, vocab, inv_vocab, device,
                           base_state, full_data)
            out.write_text(json.dumps({"arm": arm, "seed": seed, "traj": traj}, indent=2))


if __name__ == "__main__":
    main()
