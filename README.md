# Assignment-9
Uses NLTK to compare four supplied texts (`Martin.txt`, `RJ_Lovecraft.txt`, `RJ_Martin.txt`, `RJ_Tolkein.txt`).

**Outputs**

* 20 most-common tokens (+ stems & lemmas)
* optional named-entity counts (`--show-ner`)
* top 3-gram table for each text
* simple authorship guess: which of the first three writers most resembles **Text 4**

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate   # create venv  (Windows:  .venv\Scripts\activate)
pip install -r requirements.txt                     # install deps

python main.py              # run analysis
python main.py --show-ner   # include NE counts

pytest                      # run 2 sanity tests
