"""Match linker patterns against stanza-parsed sentence tokens.

A pattern may consist of several parts that must appear in order with a
non-empty gap between them (e.g. "если ... то"). Ported from the notebooks
`linkers (1).ipynb` and `connectors within linkers.ipynb`.
"""

from typing import List, Optional, Sequence, Tuple

# Для индексации паттернов
from collections import defaultdict
#from bisect import bisect_left

from rules import Token, Phrase

Part = Tuple[str, ...]
Pattern = Tuple[Part, ...]




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
                    if any(t.st.to_dict()[0]["upos"] != "PUNCT" for t in gap):                        
                        start = j
                        found = True
                        break
            if not found:
                return None


    return spans


def extract_entities_from_sentence(tokens_stanza, patterns: Sequence[Pattern], debug = False) -> List[dict]:
    """Greedily find non-overlapping pattern matches in a sentence's tokens,
    longest (most parts, then most words) first."""

    # Создаём список наших токенов, далее работаем с ними в том числе и в правилах
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
                if debug:
                    pass

                parts = [tokens[s:e] for s, e in spans]
                char_spans = [(part[0]["start_char"], part[-1]["end_char"]) for part in parts]

                for s, e in spans:
                    for k in range(s, e):
                        used[k] = True

                results.append(Phrase(parts, char_spans, pattern))
                break

    return results

#временно
def pattern_surface(pattern: Pattern) -> str:
    return "...".join(" ".join(part) for part in pattern)


def sentence_to_json(sentence: str, tokens, patterns: Sequence[Pattern], debug = False) -> dict:
    return {
        "sentence": sentence,
        "entities": extract_entities_from_sentence(tokens, patterns, debug),
    }
