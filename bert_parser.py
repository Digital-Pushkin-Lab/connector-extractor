"""Syntactic parsing with the fine-tuned ruBERT biaffine parser
`IuliiaPr/my-ruBert-parser-model` -- an alternative to stanza's `pos` +
`depparse` processors (see `pipeline.build_engine(parser="bert")`).

The model predicts UPOS, the head and the deprel of every word jointly. It does
NOT predict lemmas, so tokenization and lemmatization stay with stanza.

Things worth knowing (all verified against the training notebook):

* The architecture is re-declared here (same parameter names as
  `modeling_biaffine.py` in the HF repo) instead of using
  `AutoModel.from_pretrained(..., trust_remote_code=True)`: no remote code is
  executed, and it works with the transformers 4.20 that ships with Python 3.8,
  which cannot read `.safetensors` checkpoints itself.
* The HF repo has no label maps. The training notebook built them as the sorted
  unique UPOS / DEPREL values of `ru_taiga-ud-train-a.conllu`; the result is
  stored in `data/taiga_labels.json`.
* Heads are NOT word ids. The training labels are token positions in the
  padded subword sequence, and they are shifted by one word: a dependent whose
  1-based CoNLL head is `h` was labelled with the last subtoken of the
  0-based word `h` (position 0 = [CLS] = root). So a predicted position `p`
  decodes to the CoNLL head `owner_word(p)` (see `parse_batch`). A side effect:
  a sentence's last word can never be a head (its label collapses to the root
  label), which is harmless for UD-style input ending in punctuation, so a "."
  is appended when the input does not end in one.
* Words must be tokenized UD-style (punctuation as separate tokens), like in
  the training data -- the notebook's `text.split()` demo does not do this.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, NamedTuple, Optional

DEFAULT_REPO = "IuliiaPr/my-ruBert-parser-model"
LABELS_PATH = Path(__file__).parent / "data" / "taiga_labels.json"
# inputs are one sentence at a time; avoids the "process got forked" warning from tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MAX_SUBTOKENS = 512     # BERT limit; training used 128, longer inputs are out of distribution


class WordParse(NamedTuple):
    upos: str
    head: int           # CoNLL id of the head, 0 = root
    deprel: str


def _has_letters_or_digits(word: str) -> bool:
    return any(ch.isalnum() for ch in word)


class BertSyntaxParser:
    """Loads the checkpoint once and parses lists of already-tokenized sentences."""

    def __init__(self, repo_id: str = DEFAULT_REPO, device: Optional[str] = None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from huggingface_hub import hf_hub_download, snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        self.upos_labels = labels["upos"]
        self.deprel_labels = labels["deprel"]

        config = json.loads(Path(hf_hub_download(repo_id, "config.json")).read_text(encoding="utf-8"))
        if (config["num_pos"], config["num_deprel"]) != (len(self.upos_labels), len(self.deprel_labels)):
            raise ValueError(f"{LABELS_PATH.name} does not match the checkpoint: "
                             f"{len(self.upos_labels)} UPOS / {len(self.deprel_labels)} deprels vs "
                             f"{config['num_pos']} / {config['num_deprel']} in the model config")

        # transformers 4.20's own downloader cannot follow the Hub's current
        # redirects, so fetch the base model's config/vocab with huggingface_hub.
        base_dir = snapshot_download(config["model_name"], allow_patterns=["*.json", "*.txt"])
        # The ruBERT vocab is cased but the repo ships no tokenizer_config, and transformers 4.20
        # then defaults to do_lower_case=True -- which feeds the model text it never saw in training.
        self.tokenizer = AutoTokenizer.from_pretrained(base_dir, do_lower_case=False)

        class Biaffine(nn.Module):
            def __init__(self, dim, out):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(out, dim, dim))
                self.bias = nn.Parameter(torch.zeros(out, dim + dim))   # unused, but in the checkpoint

            def forward(self, x_head, x_dep):
                return torch.einsum("bdi,oij,bej->boed", x_dep, self.weight, x_head)

        class Parser(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = AutoModel.from_config(AutoConfig.from_pretrained(base_dir))
                hidden, mlp = self.bert.config.hidden_size, config["mlp_dim"]
                self.pos_head = nn.Linear(hidden, config["num_pos"])
                self.mlp_arc_head = nn.Linear(hidden, mlp)
                self.mlp_arc_dep = nn.Linear(hidden, mlp)
                self.arc_attn = Biaffine(mlp, 1)
                self.mlp_rel_head = nn.Linear(hidden, mlp)
                self.mlp_rel_dep = nn.Linear(hidden, mlp)
                self.rel_attn = Biaffine(mlp, config["num_deprel"])

            def forward(self, input_ids, attention_mask):
                seq = self.bert(input_ids, attention_mask=attention_mask)[0]
                return {
                    "pos_logits": self.pos_head(seq),
                    # [batch, position, candidate head position]
                    "arc_logits": self.arc_attn(F.leaky_relu(self.mlp_arc_head(seq)),
                                                F.leaky_relu(self.mlp_arc_dep(seq))).squeeze(1),
                    # [batch, deprel, position, candidate head position]
                    "rel_logits": self.rel_attn(F.leaky_relu(self.mlp_rel_head(seq)),
                                                F.leaky_relu(self.mlp_rel_dep(seq))),
                }

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = Parser()
        weights_path = Path(hf_hub_download(repo_id, "model.safetensors"))
        self.repo_id = repo_id
        self.revision = weights_path.parent.name    # HF cache snapshot dir = commit sha
        state = load_file(str(weights_path))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        # position_ids is a non-persistent buffer in newer transformers versions
        missing = [k for k in missing if not k.endswith("position_ids")]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self.model.to(self.device).eval()

    def parse_batch(self, sentences: List[List[str]]) -> List[List[Optional[WordParse]]]:
        """Parse tokenized sentences. Words cut off by the subtoken limit get None."""
        torch = self.torch
        padded = [ws + ["."] if ws and _has_letters_or_digits(ws[-1]) else ws for ws in sentences]
        results: List[List[Optional[WordParse]]] = [[] for _ in sentences]
        todo = [i for i, ws in enumerate(padded) if ws]
        if not todo:
            return results

        enc = self.tokenizer([padded[i] for i in todo], is_split_into_words=True, truncation=True,
                             max_length=MAX_SUBTOKENS, padding=True, return_tensors="pt")
        with torch.no_grad():
            out = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
        pos_pred = out["pos_logits"].argmax(-1).cpu()
        arc = out["arc_logits"].cpu()
        rel = out["rel_logits"].cpu()

        for row, i in enumerate(todo):
            word_ids = enc.word_ids(row)
            first, last = {}, {}
            for t, w in enumerate(word_ids):
                if w is not None:
                    first.setdefault(w, t)
                    last[w] = t
            # Valid head targets in training: [CLS] (root) and the last subtoken of words 1..n-1.
            allowed = torch.full((arc.size(-1),), float("-inf"))
            allowed[0] = 0.0
            for w, t in last.items():
                if w >= 1:
                    allowed[t] = 0.0

            n_words = len(sentences[i])
            parsed: List[Optional[WordParse]] = []
            for w in range(n_words):
                if w not in first:
                    parsed.append(None)
                    continue
                t = first[w]
                p = int(torch.argmax(arc[row, t] + allowed))
                head = 0 if p == 0 else word_ids[p]
                deprel = self.deprel_labels[int(rel[row, :, t, p].argmax())]
                parsed.append(WordParse(self.upos_labels[int(pos_pred[row, t])], head, deprel))
            results[i] = parsed
        return results


# -- drop-in replacement for the stanza pipeline object ------------------------

class _Token:
    """Mimics the only part of a stanza Token the extractor uses: `to_dict()[0]`.
    Returns a fresh copy each call because callers add flags to `misc`."""

    def __init__(self, word: dict):
        self._word = word

    def to_dict(self) -> List[dict]:
        return [dict(self._word)]


class _Sentence:
    def __init__(self, text: str, tokens: List[_Token]):
        self.text = text
        self.tokens = tokens


class _Doc:
    def __init__(self, sentences: List[_Sentence]):
        self.sentences = sentences


class BertParsedPipeline:
    """Wraps a stanza pipeline (tokenize, pos, lemma[, depparse]) and overwrites
    `head` and `deprel` (and `upos` if `replace_pos`) with the BERT parser's
    predictions. Used like `nlp(text)` by `pipeline.parse_sentences`."""

    def __init__(self, nlp, parser: BertSyntaxParser, replace_pos: bool = True):
        self.nlp = nlp
        self.parser = parser
        self.replace_pos = replace_pos

    def __call__(self, text: str) -> _Doc:
        doc = self.nlp(text)
        sentences = []
        for sentence in doc.sentences:
            words = [tok.to_dict()[0] for tok in sentence.tokens]
            parsed = self.parser.parse_batch([[w["text"] for w in words]])[0]
            tokens = []
            for word, new in zip(words, parsed):
                if new is not None:     # None: beyond the subtoken limit, keep stanza's values
                    # Taiga has subtyped labels (e.g. "parataxis:discourse") but rules.py compares
                    # deprels to base names ("parataxis", "discourse", ...), so it would silently
                    # ignore them. Reduce to the base relation, as the rules expect.
                    word["head"], word["deprel"] = new.head, new.deprel.split(":")[0]
                    if self.replace_pos:
                        word["upos"] = new.upos
                tokens.append(_Token(word))
            sentences.append(_Sentence(sentence.text, tokens))
        return _Doc(sentences)
