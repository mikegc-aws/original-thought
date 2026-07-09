"""Phase 1 generation sweep.

Repeatedly asks frontier models on Bedrock for "an original and novel idea" across
sampling configs and prompt interventions, to quantify the mode collapse Mike observed
and test whether it is recoverable (hypotheses H1/H2 in docs/01_definitions.md).

Writes one JSONL record per generation to data/phase1/generations.jsonl, flushed
incrementally — safe to interrupt and rerun; completed (model, config, index) triples
are skipped on resume.
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from original_thought.bedrock import generate

OUT = Path(__file__).parent.parent / "data" / "phase1" / "generations.jsonl"

MODELS = {
    "claude-sonnet-4.6": "us.anthropic.claude-sonnet-4-6",
    "claude-haiku-4.5": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "nova-pro": "us.amazon.nova-pro-v1:0",
    "llama-3.3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
}

BASE_PROMPT = (
    "Write an original and novel idea that has not been done before. "
    "Describe the idea in a short paragraph."
)

DIVERSIFY_PROMPT = (
    "First, silently pick a random domain of human activity (science, art, cooking, "
    "transport, education, sport, medicine, agriculture, music, law, ...) and a random "
    "object or process within it. Then write an original and novel idea in that domain "
    "that has not been done before. Describe the idea in a short paragraph. "
    "Do not explain your choice of domain; just give the idea."
)

SEED_WORDS = [
    "lantern", "tide", "gravel", "violin", "compost", "orbit", "loom", "fungus",
    "ledger", "glacier", "whistle", "kelp", "circuit", "saddle", "prism", "yeast",
    "anchor", "meadow", "turbine", "ink", "hive", "canyon", "pendulum", "moss",
    "radar", "quilt", "ember", "reef", "chalk", "bellows", "sundial", "resin",
    "furrow", "gasket", "mica", "pollen", "stylus", "trellis", "vane", "wharf",
    "zephyr", "brine", "cog", "dune", "eaves", "flint", "gourd", "hearth",
    "isthmus", "jetty",
]


def seeded_prompt(idx: int) -> str:
    rng = random.Random(idx)  # deterministic per sample index, reproducible
    w1, w2 = rng.sample(SEED_WORDS, 2)
    return (
        f"Here are two random words: '{w1}' and '{w2}'. Using them only as loose "
        "inspiration, write an original and novel idea that has not been done before. "
        "Describe the idea in a short paragraph."
    )


# config -> (prompt for sample i, temperature, top_p)
CONFIGS = {
    "baseline_t0.7": lambda i: (BASE_PROMPT, 0.7, 0.95),
    "greedy_t0.0": lambda i: (BASE_PROMPT, 0.0, 1.0),
    "hot_t1.0": lambda i: (BASE_PROMPT, 1.0, 0.99),
    "diversify_prompt": lambda i: (DIVERSIFY_PROMPT, 0.7, 0.95),
    "random_seed_words": lambda i: (seeded_prompt(i), 0.7, 0.95),
}


def done_keys() -> set:
    if not OUT.exists():
        return set()
    keys = set()
    with OUT.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                keys.add((r["model"], r["config"], r["index"]))
            except json.JSONDecodeError:
                continue
    return keys


def one(model_name, model_id, config_name, idx):
    prompt, temp, top_p = CONFIGS[config_name](idx)
    r = generate(model_id, prompt, temperature=temp, top_p=top_p)
    return {
        "model": model_name,
        "config": config_name,
        "index": idx,
        "prompt": prompt,
        "temperature": temp,
        "top_p": top_p,
        "text": r["text"],
        "usage": r["usage"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=50, help="samples per model×config")
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys()

    jobs = [
        (m, MODELS[m], c, i)
        for m in args.models
        for c in args.configs
        for i in range(args.samples)
        if (m, c, i) not in done
    ]
    print(f"{len(done)} already done, {len(jobs)} to run")
    if not jobs:
        return

    failures = 0
    with OUT.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, *j): j for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            try:
                rec = fut.result()
            except Exception as e:
                failures += 1
                tqdm.write(f"FAIL {futs[fut][:3]}: {e}")
                continue
            f.write(json.dumps(rec) + "\n")
            f.flush()
    print(f"done; {failures} failures (rerun to fill gaps)")


if __name__ == "__main__":
    main()
