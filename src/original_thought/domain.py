"""Phase 3 synthetic "inventions" domain: a closed combinatorial world where
originality is provable by construction.

World: triples (device, mechanism, purpose), 50 pseudo-words of each type.
Each element belongs to one of 5 latent classes (never stated in the data).
A triple is VALID iff class-compat(device, mechanism) AND class-compat(mechanism,
purpose), where each class is compatible with exactly 2 of the 5 classes on the
other side. Valid fraction of the 125,000-triple space ≈ 16%.

Splits of the valid set:
  train       — what the model sees (rendered as sentences, all 3 templates)
  L1 holdout  — valid triples excluded from training, but whose (d,m) pair occurs
                in training via other triples (novel triple, familiar parts)
  L2 holdout  — ALL triples containing certain (d,m) pairs are excluded, so the
                pair itself is unseen; producing one requires the latent class
                rule, not pair memory (deeper compositional novelty)

Everything is deterministic given SEED. The validity checker is exact: this is the
ground truth that frontier-model studies can never have.
"""

import itertools
import random
from dataclasses import dataclass, field

SEED = 42
N_WORDS = 50          # per element type
N_CLASSES = 5
COMPAT = 2            # compatible classes per class
L2_PAIR_FRAC = 0.08   # fraction of valid (d,m) pairs fully held out
L1_TRIPLE_FRAC = 0.10 # fraction of remaining valid triples held out

TEMPLATES = [
    ("the", "{d}", "uses", "{m}", "to", "{p}", "."),
    ("with", "{m}", "the", "{d}", "can", "{p}", "."),
    ("a", "{d}", "is", "able", "to", "{p}", "using", "{m}", "."),
]

_ONSETS = list("bdfgklmnprstvz") + ["br", "cl", "dr", "fl", "gr", "kr", "pl", "sk", "tr", "vl"]
_VOWELS = ["a", "e", "i", "o", "u", "ar", "el", "ir", "or", "un"]


def _make_words(n: int, rng: random.Random, taken: set) -> list[str]:
    words = []
    while len(words) < n:
        w = "".join(rng.choice(_ONSETS) + rng.choice(_VOWELS)
                    for _ in range(rng.choice([2, 3])))
        if w not in taken and 4 <= len(w) <= 12:
            taken.add(w)
            words.append(w)
    return words


@dataclass
class Domain:
    devices: list[str]
    mechanisms: list[str]
    purposes: list[str]
    cls: dict[str, int]              # word -> latent class
    dm_compat: list[list[bool]]      # device class x mechanism class
    mp_compat: list[list[bool]]      # mechanism class x purpose class
    train: set = field(default_factory=set)   # sets of (d,m,p) word triples
    l1: set = field(default_factory=set)
    l2: set = field(default_factory=set)
    l2_pairs: set = field(default_factory=set)

    def valid(self, d: str, m: str, p: str) -> bool:
        return (d in self.cls and m in self.cls and p in self.cls
                and self.dm_compat[self.cls[d]][self.cls[m]]
                and self.mp_compat[self.cls[m]][self.cls[p]])

    def render(self, triple, template_idx: int) -> list[str]:
        d, m, p = triple
        sub = {"{d}": d, "{m}": m, "{p}": p}
        return [sub.get(t, t) for t in TEMPLATES[template_idx]]

    def parse(self, tokens: list[str]):
        """Match a token sequence against the templates; return (d,m,p) or None."""
        for tpl in TEMPLATES:
            if len(tokens) != len(tpl):
                continue
            out = {}
            for tok, slot in zip(tokens, tpl):
                if slot in ("{d}", "{m}", "{p}"):
                    out[slot] = tok
                elif tok != slot:
                    break
            else:
                d, m, p = out["{d}"], out["{m}"], out["{p}"]
                if d in set(self.devices) and m in set(self.mechanisms) and p in set(self.purposes):
                    return (d, m, p)
        return None


def _compat_matrix(rng: random.Random) -> list[list[bool]]:
    m = []
    for _ in range(N_CLASSES):
        cols = rng.sample(range(N_CLASSES), COMPAT)
        m.append([j in cols for j in range(N_CLASSES)])
    return m


def build_domain() -> Domain:
    rng = random.Random(SEED)
    taken: set = set()
    devices = _make_words(N_WORDS, rng, taken)
    mechanisms = _make_words(N_WORDS, rng, taken)
    purposes = _make_words(N_WORDS, rng, taken)

    cls = {}
    for words in (devices, mechanisms, purposes):
        for i, w in enumerate(words):
            cls[w] = i % N_CLASSES  # balanced classes

    dom = Domain(devices, mechanisms, purposes, cls,
                 _compat_matrix(rng), _compat_matrix(rng))

    all_valid = [(d, m, p)
                 for d, m, p in itertools.product(devices, mechanisms, purposes)
                 if dom.valid(d, m, p)]
    rng.shuffle(all_valid)

    valid_pairs = sorted({(d, m) for d, m, _ in all_valid})
    rng.shuffle(valid_pairs)
    dom.l2_pairs = set(valid_pairs[: int(L2_PAIR_FRAC * len(valid_pairs))])
    dom.l2 = {t for t in all_valid if (t[0], t[1]) in dom.l2_pairs}

    rest = [t for t in all_valid if t not in dom.l2]
    n_l1 = int(L1_TRIPLE_FRAC * len(rest))
    dom.l1 = set(rest[:n_l1])
    dom.train = set(rest[n_l1:])

    # sanity: every word must appear in training data
    seen = {w for t in dom.train for w in t}
    assert seen >= set(devices) | set(mechanisms) | set(purposes), "vocab gap in train"
    return dom


def sample_invalid(dom: Domain, n: int, rng: random.Random) -> list[tuple]:
    out = []
    while len(out) < n:
        t = (rng.choice(dom.devices), rng.choice(dom.mechanisms), rng.choice(dom.purposes))
        if not dom.valid(*t):
            out.append(t)
    return out


# ---- tokenizer -------------------------------------------------------------

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


def build_vocab(dom: Domain) -> dict[str, int]:
    words = [PAD, BOS, EOS]
    words += sorted({t for tpl in TEMPLATES for t in tpl if not t.startswith("{")})
    words += dom.devices + dom.mechanisms + dom.purposes
    return {w: i for i, w in enumerate(words)}


def encode(sent: list[str], vocab: dict, block: int) -> list[int]:
    ids = [vocab[BOS]] + [vocab[w] for w in sent] + [vocab[EOS]]
    assert len(ids) <= block, sent
    return ids + [vocab[PAD]] * (block - len(ids))
