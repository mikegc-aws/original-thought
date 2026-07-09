"""Phase 1 coherence/value judging (metric M4).

Scores a random subsample of generations per (model, config) cell for coherence and
plausible value, using a judge model NOT in the generator pool (Claude Opus 4.5) to
avoid self-preference. Purpose: verify that diversity interventions (hot sampling,
diversify prompts) do not degrade idea quality — hypothesis H2.

Writes data/phase1/judgments.jsonl (incremental/resumable) and
results/phase1/judge_summary.csv.
"""

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from original_thought.bedrock import generate

ROOT = Path(__file__).parent.parent
GEN = ROOT / "data" / "phase1" / "generations.jsonl"
OUT = ROOT / "data" / "phase1" / "judgments.jsonl"
RESULTS = ROOT / "results" / "phase1"

JUDGE_MODEL = "us.anthropic.claude-opus-4-5-20251101-v1:0"  # not in generator pool
PER_CELL = 15  # judged samples per model×config

RUBRIC = """You are scoring a short "novel idea" produced by a language model.

Score two dimensions, each 1-10:
- coherence: Is it a well-formed, understandable idea (10) or confused/word-salad (1)?
- value: If implemented/pursued, could it plausibly be useful or interesting to someone
  (10 = clearly worthwhile, 1 = pointless or nonsensical)?

Do NOT reward length or style. Judge the idea itself.

Idea:
---
{idea}
---

Reply with ONLY a JSON object: {{"coherence": <int>, "value": <int>}}"""


def judge_one(rec):
    r = generate(JUDGE_MODEL, RUBRIC.format(idea=rec["text"][:3000]),
                 temperature=0.0, max_tokens=50)
    raw = r["text"].strip()
    raw = raw[raw.find("{"): raw.rfind("}") + 1]
    scores = json.loads(raw)
    return {"key": rec["key"], "model": rec["model"], "config": rec["config"],
            "coherence": int(scores["coherence"]), "value": int(scores["value"])}


def main():
    rows = [json.loads(l) for l in GEN.open()]
    for r in rows:
        r["key"] = f'{r["model"]}|{r["config"]}|{r["index"]}'
    df = pd.DataFrame(rows).drop_duplicates("key")

    done = set()
    if OUT.exists():
        done = {json.loads(l)["key"] for l in OUT.open()}

    rng = random.Random(0)
    todo = []
    for _, g in df.groupby(["model", "config"]):
        recs = g.to_dict("records")
        rng.shuffle(recs)
        todo += [r for r in recs[:PER_CELL] if r["key"] not in done]
    print(f"{len(done)} judged already, {len(todo)} to judge")

    with OUT.open("a") as f, ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(judge_one, r) for r in todo]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            try:
                f.write(json.dumps(fut.result()) + "\n")
                f.flush()
            except Exception as e:
                tqdm.write(f"FAIL: {e}")

    j = pd.DataFrame([json.loads(l) for l in OUT.open()])
    summary = j.groupby(["model", "config"])[["coherence", "value"]].mean().round(2)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / "judge_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
