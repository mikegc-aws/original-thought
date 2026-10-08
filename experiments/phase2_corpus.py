"""Phase 2 reference corpus builder.

Assembles a corpus of *real, existing* ideas to measure model-output novelty against
(novelty metric M3, see README and the writeup at blog.mikegchambers.com):

  - YC companies (public yc-oss API): name + one-liner + description = a product idea.
    Chosen because Phase 1 showed models' "novel ideas" have a startup-pitch genre.
  - arXiv abstracts from idea-adjacent categories (cs.HC, cs.CY): research ideas in
    the human-tech space the models favor.

Writes data/phase2/corpus.jsonl [{id, source, text}] and embeds everything with the
same Titan v2 512-d space used for Phase 1 generations (data/phase2/corpus_emb.jsonl,
incremental/resumable).
"""

import json
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from original_thought.bedrock import embed

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "phase2"
CORPUS = OUT_DIR / "corpus.jsonl"
EMB = OUT_DIR / "corpus_emb.jsonl"

ARXIV_CATS = ["cs.HC", "cs.CY"]
ARXIV_PER_CAT = 2000
ARXIV_PAGE = 200


def fetch_yc():
    with urllib.request.urlopen("https://yc-oss.github.io/api/companies/all.json") as r:
        companies = json.load(r)
    rows = []
    for c in companies:
        desc = (c.get("long_description") or c.get("one_liner") or "").strip()
        one = (c.get("one_liner") or "").strip()
        if len(desc) < 40:
            continue
        text = f"{one}. {desc}" if one and one not in desc else desc
        rows.append({"id": f"yc/{c['slug']}", "source": "yc", "text": text[:2500]})
    return rows


def fetch_arxiv():
    ns = {"a": "http://www.w3.org/2005/Atom"}
    rows = []
    for cat in ARXIV_CATS:
        for start in range(0, ARXIV_PER_CAT, ARXIV_PAGE):
            url = (f"https://export.arxiv.org/api/query?search_query=cat:{cat}"
                   f"&start={start}&max_results={ARXIV_PAGE}"
                   f"&sortBy=submittedDate&sortOrder=descending")
            with urllib.request.urlopen(url) as r:
                tree = ET.fromstring(r.read())
            entries = tree.findall("a:entry", ns)
            if not entries:
                break
            for e in entries:
                eid = e.find("a:id", ns).text.rsplit("/", 1)[-1]
                title = " ".join(e.find("a:title", ns).text.split())
                abstract = " ".join(e.find("a:summary", ns).text.split())
                rows.append({"id": f"arxiv/{eid}", "source": f"arxiv-{cat}",
                             "text": f"{title}. {abstract}"[:2500]})
            print(f"  {cat}: {start + len(entries)}")
            time.sleep(3)  # arXiv API politeness
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CORPUS.exists():
        print("fetching YC companies...")
        rows = fetch_yc()
        print(f"  {len(rows)} YC ideas")
        print("fetching arXiv abstracts...")
        rows += fetch_arxiv()
        seen = set()
        with CORPUS.open("w") as f:
            for r in rows:
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                f.write(json.dumps(r) + "\n")
        print(f"corpus: {len(seen)} items")

    corpus = [json.loads(l) for l in CORPUS.open()]
    done = set()
    if EMB.exists():
        done = {json.loads(l)["id"] for l in EMB.open()}
    todo = [r for r in corpus if r["id"] not in done]
    print(f"embedding {len(todo)} of {len(corpus)}")
    with EMB.open("a") as f, ThreadPoolExecutor(max_workers=8) as ex:
        futs = zip(todo, ex.map(embed, (r["text"] for r in todo)))
        for r, v in tqdm(futs, total=len(todo)):
            f.write(json.dumps({"id": r["id"], "v": v}) + "\n")


if __name__ == "__main__":
    main()
