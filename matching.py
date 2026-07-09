"""Match linker/introductory-word patterns against stanza-parsed sentence tokens.

A pattern may consist of several parts that must appear in order with a
non-empty gap between them (e.g. "если ... то"). Ported from the notebooks
`linkers (1).ipynb` and `connectors within linkers.ipynb`.

Linkers and introductory words are matched in separate passes (see
`pipeline.analyze_parsed`), each with its own token-consumption state, so a
match in one category can never suppress a match in the other -- exactly
like the two source notebooks, which never shared state either.
"""

from typing import List, Optional, Sequence, Tuple

Part = Tuple[str, ...]
Pattern = Tuple[Part, ...]


def pattern_surface(pattern: Pattern) -> str:
    return "...".join(" ".join(part) for part in pattern)


def assemble_token_info(token_position: int, tokens) -> dict:
    """Return a stanza token's dict form, annotated with extra `misc` flags
    used by the rule-based scorer: PunctBefore, IntroductoryPotential,
    SpaceAfter."""
    basic_info_dict = tokens[token_position].to_dict()[0]

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

    return basic_info_dict


def match_part(tokens, start: int, part: Part) -> Optional[int]:
    """Check whether `part` (a sequence of literal words) starts at `start`.
    Returns the index right after the matched words, or None."""
    if start + len(part) > len(tokens):
        return None

    for i, text in enumerate(part):
        if tokens[start + i].to_dict()[0]["text"].lower() != text:
            return None

    return start + len(part)


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
                    if any(t.to_dict()[0]["upos"] != "PUNCT" for t in gap):
                        start = j
                        found = True
                        break
            if not found:
                return None

    return spans


def extract_entities_from_sentence(tokens, patterns: Sequence[Pattern]) -> List[dict]:
    """Greedily find non-overlapping pattern matches in a sentence's tokens,
    longest (most parts, then most words) first. `patterns` should be a
    single category's pattern list (linkers OR introductory words) -- run
    this separately per category so the two never compete for tokens."""
    used = [False] * len(tokens)
    results = []

    ordered_patterns = sorted(
        patterns,
        key=lambda p: (len(p), sum(len(part) for part in p)),
        reverse=True,
    )

    for i in range(len(tokens)):
        if used[i]:
            continue

        for pattern in ordered_patterns:
            spans = match_pattern(tokens, i, pattern)
            if spans:
                parts_json = {}
                for idx, (s, e) in enumerate(spans, start=1):
                    parts_json[f"part{idx}"] = [assemble_token_info(k, tokens) for k in range(s, e)]
                    for k in range(s, e):
                        used[k] = True

                results.append({
                    "surface": pattern_surface(pattern),
                    "parts": parts_json,
                })
                break

    return results


def sentence_to_json(sentence: str, tokens, patterns: Sequence[Pattern]) -> dict:
    return {
        "sentence": sentence,
        "entities": extract_entities_from_sentence(tokens, patterns),
    }
