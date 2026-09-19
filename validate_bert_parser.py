"""Check `bert_parser.BertSyntaxParser` against gold UD annotation.

Scores word-level POS accuracy / UAS / LAS on the last 20% of
`ru_taiga-ud-train-a.conllu` -- the validation split of the training notebook
(`texts[int(len*0.8):]`), which the model has never seen. The notebook reports
POS 0.9946 / UAS 0.9593 / LAS 0.9464 measured on its shifted token-position
labels; this script measures the same thing after decoding to real word ids, so
matching numbers mean the decoding in `bert_parser` is right.

    conda activate py38
    python validate_bert_parser.py --conllu ru_taiga-ud-train-a.conllu [--limit 500]

Get the file from https://github.com/UniversalDependencies/UD_Russian-Taiga
"""

import argparse

from tqdm import tqdm

from bert_parser import BertSyntaxParser


def read_conllu(path):
    """Yield sentences as lists of (form, upos, head, deprel) for word rows."""
    sentence = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                if sentence:
                    yield sentence
                sentence = []
            elif not line.startswith("#"):
                cols = line.split("\t")
                if cols[0].isdigit():
                    sentence.append((cols[1], cols[3], int(cols[6]), cols[7]))
        if sentence:
            yield sentence


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conllu", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N validation sentences")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="the relation tensor is batch x 51 x L x L, so keep this small for long sentences")
    args = ap.parse_args()

    sentences = list(read_conllu(args.conllu))
    val = sentences[int(len(sentences) * 0.8):]
    if args.limit:
        val = val[:args.limit]
    val.sort(key=len)   # similar lengths per batch: less padding and memory; the metrics are order-independent

    parser = BertSyntaxParser()
    total = pos_ok = uas_ok = las_ok = 0
    for i in tqdm(range(0, len(val), args.batch_size), desc="parsing"):
        batch = val[i:i + args.batch_size]
        for gold, pred in zip(batch, parser.parse_batch([[w[0] for w in s] for s in batch])):
            for (_, upos, head, deprel), p in zip(gold, pred):
                total += 1
                if p is None:
                    continue
                pos_ok += p.upos == upos
                uas_ok += p.head == head
                las_ok += p.head == head and p.deprel == deprel

    print(f"sentences={len(val)}  words={total}")
    print(f"POS acc={pos_ok / total:.4f}  UAS={uas_ok / total:.4f}  LAS={las_ok / total:.4f}")


if __name__ == "__main__":
    main()
