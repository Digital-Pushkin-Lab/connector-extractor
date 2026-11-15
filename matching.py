"""Match linker patterns against stanza-parsed sentence tokens.

A pattern may consist of several parts that must appear in order with a
non-empty gap between them (e.g. "если ... то"). Ported from the notebooks
`linkers (1).ipynb` and `connectors within linkers.ipynb`.
"""

from typing import List, Optional, Sequence, Tuple
from collections import defaultdict

from patterns import Part, Pattern
from rules import Token, PhraseC




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


def extract_entities_from_sentence(engine, tokens_stanza, debug: bool = False) -> List[dict]:
    """Находит все вхождения паттернов из словаря в предложении (включая вложенные)."""
    tokens = []
    for token_stanza in tokens_stanza:
        tokens.append(Token(token_stanza, tokens))

    # для точки останова
    if debug:
        pass

    results = []

    pattern_index = engine.patterns.get("linker", {})

    for i in range(len(tokens)):
        first_word = tokens[i].s
        candidates = pattern_index.get(first_word, ())

        for pattern in candidates:
            # для точки останова
            if debug and pattern[0][0] == 'т.е.':
                pass

            spans = match_pattern(tokens, i, pattern)

            if spans:
                parts = [tokens[s:e] for s, e in spans]
                char_spans = [(part[0]["start_char"], part[-1]["end_char"]) for part in parts]
                results.append(PhraseC(parts, char_spans, pattern, engine))

    return results


#временно
def pattern_surface(pattern: Pattern) -> str:
    return "...".join(" ".join(part) for part in pattern)


