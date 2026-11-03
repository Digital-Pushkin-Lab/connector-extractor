"""Match linker patterns against stanza-parsed sentence tokens.

A pattern may consist of several parts that must appear in order with a
non-empty gap between them (e.g. "если ... то"). Ported from the notebooks
`linkers (1).ipynb` and `connectors within linkers.ipynb`.
"""

from typing import List, Optional, Sequence, Tuple

# Для индексации паттернов
from collections import defaultdict
from bisect import bisect_left

Part = Tuple[str, ...]
Pattern = Tuple[Part, ...]

class Token:
    def __init__(self, token_stanza, tokens):
        self.ts = token_stanza
        self.text = token_stanza.text
        self.s = self.text.lower()
        self.tokens = tokens
        self.info = None # это поле вычисляеся один раз при первом обращении к нему. Для внешних функций, наш токен выглядит как этот словарь (есть функция .get())

    def index(self):
        return self.tokens.index(self)

    def prev(self):
        i = self.index()
        if i == 0:
            return None
        return self.tokens[i - 1]

    def next(self):
        i = self.index()
        if i == len(self.tokens) - 1:
            return None
        return self.tokens[i + 1]
    
    # для совместимости временно
    def to_dict(self):
        return self.ts.to_dict()  
    
    def get_info(self) -> dict:
        if self.info is None:
            self.assemble_token_info()
        return self.info
    
    # для совместимости. Для обращений как к словарю .get("start_char")
    def get(self, key, default=None):
        return self.get_info().get(key, default)    

    # для совместимости. Для обращений как к словарю ["start_char"]
    def __getitem__(self, key):
        return self.get_info()[key]    
    
    # эту функцию мы сделали методом класса и пока не меняли. Она будет переработана позже. 
    def assemble_token_info(self) -> dict:
        """Return token info with extra `misc` flags, caching the result."""
        if self.info is not None:
            return self.info

        token_position = self.index()
        tokens = self.tokens
        basic_info_dict = self.to_dict()[0]

        def add_misc(flag):
            if basic_info_dict.get("misc", None):
                basic_info_dict["misc"] += f";{flag}"
            else:
                basic_info_dict["misc"] = flag

        if (token_position >= 2) and tokens[token_position - 1].to_dict()[0]["upos"] == "PUNCT":
            add_misc("PunctBefore=Yes")

        if (token_position >= 1) and (token_position < len(tokens) - 2):
            prev_lemma = tokens[token_position - 1].to_dict()[0]["lemma"]
            next_lemma = tokens[token_position + 1].to_dict()[0]["lemma"]
            next_next_upos = tokens[token_position + 2].to_dict()[0]["upos"]
            if prev_lemma in {"а", "и"} or (next_lemma == "же" and next_next_upos == "PUNCT"):
                add_misc("IntroductoryPotential=Yes")

        if (token_position == len(tokens) - 1) or tokens[token_position + 1].to_dict()[0]["upos"] == "PUNCT":
            add_misc("SpaceAfter=No")

        # Зависимые ПОСЛЕ текущего токена (пунктуацию не считаем).
        this_id = basic_info_dict.get("id")
        head_id = basic_info_dict.get("head", 0)

        for j in range(token_position + 1, len(tokens)):
            td = tokens[j].to_dict()[0]
            if td.get("upos") == "PUNCT":
                continue
            if this_id is not None and td.get("head") == this_id:
                add_misc("DependentAfter=Yes")
                break

        for j in range(token_position + 1, len(tokens)):
            td = tokens[j].to_dict()[0]
            if head_id and td.get("id") == head_id and td.get("upos") in {"NOUN", "PROPN"}:
                add_misc("HeadIsFollowingNoun=Yes")
                break

        self.info = basic_info_dict
        return basic_info_dict

_cached_pattern_index = None

def build_pattern_index(patterns):
    global _cached_pattern_index
    if _cached_pattern_index is not None:
        return _cached_pattern_index

    ordered_patterns = sorted(
        patterns,
        key=lambda p: (
            len(p),
            sum(len(part) for part in p),
        ),
        reverse=True,
    )

    index = defaultdict(list)

    for pattern in ordered_patterns:
        first_word = pattern[0][0].lower()
        index[first_word].append(pattern)

    pattern_index = dict(index)

    _cached_pattern_index = pattern_index

    return pattern_index



def pattern_surface(pattern: Pattern) -> str:
    return "...".join(" ".join(part) for part in pattern)


def match_part(tokens, start: int, part: Part) -> Optional[int]:
    """Check whether `part` (a sequence of literal words) starts at `start`.
    Returns the index right after the matched words, or None."""
    if start + len(part) > len(tokens):
        return None

    passed = 0 # считаем сколько пропустили
    for i, text in enumerate(part):
        matched = False
        for token in tokens[start + i + passed:]: #делаем вложенный цикл, так как нам надо двигаться по токенам, чтобы пропустить запятую

            # token_dict = token.to_dict()
            # token_text = token_dict[0]["text"].lower()
            token_text = token.s # текст в нижнем регистре
            if i>0 and token_text == ",": #запятую пропускаем для не первого токена в шаблоне
                passed +=1  # считаем пропущенные
                continue 
            if token_text != text:
                return None
            
            matched=True
            break #проверка пройдена
        if not matched:
            return None

    return start + len(part)+passed






def match_pattern(tokens, start: int, pattern: Pattern) -> Optional[List[Tuple[int, int]]]:
    """Try to match all parts of `pattern` starting at `start`, allowing a
    non-punctuation gap between parts. Returns the list of (start, end) token
    spans for each part, or None if the pattern doesn't match."""
    spans = []

    for idx, part in enumerate(pattern):
        end = match_part(tokens, start, part)
        if end is None:
            return None

        spans.append((start, end))

        if idx < len(pattern) - 1:
            found = False

            for j in range(end + 1, len(tokens)):
                next_end = match_part(tokens, j, pattern[idx + 1])
                if next_end:
                    gap = tokens[end:j]
                    if any(t.ts.to_dict()[0]["upos"] != "PUNCT" for t in gap):                        
                        start = j
                        found = True
                        break
            if not found:
                return None


    return spans


def extract_entities_from_sentence(tokens_stanza, patterns: Sequence[Pattern], debug = False) -> List[dict]:
    """Greedily find non-overlapping pattern matches in a sentence's tokens,
    longest (most parts, then most words) first."""

    # создаём список токенов
    tokens = []
    for token_stanza in tokens_stanza:
        tokens.append(Token(token_stanza, tokens))

    # для точки останова
    if debug:
        pass

    used = [False] * len(tokens)
    results = []


    #функция готовит паттерны один раз, и не на предложение, а вообще, так как внутри используется кэш
    pattern_index = build_pattern_index(patterns) 

    for i in range(len(tokens)):


        if used[i]:
            continue

        first_word = tokens[i].s

        candidates = pattern_index.get(first_word, ())

        for pattern in candidates:

            # для точки останова. Проверяем сочетание метки debug (которая зависит от предложения) и элемента паттерна, т.е. смотрим в конкретное место двойного цикла
            if debug and pattern[0][0] == 'т.е.':
                pass

            spans = match_pattern(tokens, i, pattern)
            if spans:
                parts_json = {}
                char_spans = []
                for idx, (s, e) in enumerate(spans, start=1):
                    
                    part_tokens = tokens[s:e]

                    parts_json[f"part{idx}"] = part_tokens
                    char_spans.append((part_tokens[0]["start_char"], part_tokens[-1]["end_char"]))
                    for k in range(s, e):
                        used[k] = True

                results.append({
                    "surface": pattern_surface(pattern),
                    "parts": parts_json,
                    "spans": char_spans,
                })
                break

    return results


def sentence_to_json(sentence: str, tokens, patterns: Sequence[Pattern], debug = False) -> dict:
    return {
        "sentence": sentence,
        "entities": extract_entities_from_sentence(tokens, patterns, debug),
    }
