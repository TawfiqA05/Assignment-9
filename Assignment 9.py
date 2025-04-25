#!/usr/bin/env python3
"""
INFO-B 211 — Assignment 9  •  NLTK analysis script
Run:
    python main.py            # core output
    python main.py --show-ner # include named-entity counts
"""

from pathlib import Path
import argparse, re, sys
from collections import Counter
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk, ngrams, FreqDist
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from tabulate import tabulate

# === CHANGE THESE PATHS ONLY IF YOU RENAME THE FILES/FOLDER ===
FILES = [
    "texts/Martin.txt",
    "texts/RJ_Lovecraft.txt",
    "texts/RJ_Martin.txt",
    "texts/RJ_Tolkein.txt",
]

stemmer, lemmatizer = SnowballStemmer("english"), WordNetLemmatizer()


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def ensure_nltk():
    """Download required NLTK assets on first run (quietly)."""
    for pkg in [
        "punkt",
        "averaged_perceptron_tagger",
        "maxent_ne_chunker",
        "words",
        "wordnet",
    ]:
        try:
            nltk.data.find(
                f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}"
            )
        except LookupError:
            nltk.download(pkg, quiet=True)


def load(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def preprocess(txt: str) -> list[str]:
    txt = re.sub(r"[^\w\s]", " ", txt.lower())
    return word_tokenize(txt)


def top20(toks: list[str]) -> list[tuple[str, int]]:
    return Counter(toks).most_common(20)


def stems_lemmas(tok20: list[tuple[str, int]]) -> list[tuple[str, str]]:
    return [(stemmer.stem(w), lemmatizer.lemmatize(w)) for w, _ in tok20]


def ner_count(txt: str) -> int:
    return sum(
        1
        for sent in nltk.sent_tokenize(txt)
        for node in ne_chunk(pos_tag(word_tokenize(sent)))
        if hasattr(node, "label")
    )


def trigram_table(toks: list[str], k: int = 10) -> list[tuple[str, int]]:
    return [
        (" ".join(g), c)
        for g, c in FreqDist(ngrams(toks, 3)).most_common(k)
    ]


def subject_guess(tok20: list[tuple[str, int]]) -> str:
    return ", ".join([w for w, _ in tok20 if w.isalpha()][:3])


def compare_author(tri_sets: dict[str, set[str]]) -> str:
    base = {f: s for f, s in tri_sets.items() if "Tolkein" not in f}
    target = tri_sets["texts/RJ_Tolkein.txt"]
    best = max(base, key=lambda f: len(base[f] & target))
    return (
        f"\nAuthorship hint ➜ Text 4 most resembles **{Path(best).name}** "
        f"(shared trigrams = {len(base[best] & target)})"
    )


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main(show_ner: bool = False):
    ensure_nltk()
    tri_sets, summary = {}, []

    for fp in FILES:
        raw, toks = load(fp), preprocess(load(fp))
        top = top20(toks)
        tri_sets[fp] = set(" ".join(g) for g, _ in trigram_table(toks, 25))

        print(f"\n=== {Path(fp).name} ===")
        print(tabulate(top, headers=["token", "count"]))
        print("\nStem → Lemma")
        print(tabulate(stems_lemmas(top)))
        print("\nTop 10 trigrams")
        print(tabulate(trigram_table(toks), headers=["trigram", "count"]))

        summary.append(
            {
                "file": Path(fp).name,
                "named_entities": ner_count(raw) if show_ner else "-",
                "subject_guess": subject_guess(top),
            }
        )

    print("\n----- SUMMARY -----")
    print(tabulate(summary, headers="keys"))
    print(compare_author(tri_sets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-ner", action="store_true", help="include named-entity counts"
    )
    try:
        main(parser.parse_args().show_ner)
    except Exception as exc:
        sys.exit(f"Error: {exc}")