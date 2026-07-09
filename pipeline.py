"""High-level analysis pipeline: split text into sentences, parse each with
stanza, match linker/introductory-word patterns, and score them.

Ported from `get_stats` / `get_stats_with_examples` / `get_final_stats` in
`linkers (1).ipynb`, with results reshaped into the flat column layout the
downstream spreadsheets expect (see `summarize_linkers` / `summarize_intro` /
`combine_stats`).

Linkers and introductory words are matched in two fully independent passes
(`analyze_parsed` is called once per category) -- exactly like the two
source notebooks, which never shared pattern lists or token-consumption
state. The stanza parse itself is still done only once per text
(`parse_sentences`) and its output is reused for both passes, since parsing
is the expensive step and is identical regardless of which patterns are
matched against it.
"""

from decimal import ROUND_HALF_UP, Decimal
from typing import List, Sequence, Tuple

from razdel import sentenize

from matching import Pattern, sentence_to_json

DEFAULT_THRESHOLD = 0.4


def round_half_up(value: float, ndigits: int = 1) -> float:
    """Round a float using 'round half up' (not Python's default 'round half
    to even')."""
    if not isinstance(value, (int, float)):
        raise TypeError("Input must be a float or int")

    decimal_value = Decimal(str(value))
    quantize_exp = Decimal("0.1") ** ndigits
    rounded_decimal = decimal_value.quantize(quantize_exp, rounding=ROUND_HALF_UP)
    return float(rounded_decimal)


def parse_sentences(text: str, nlp) -> Tuple[List[tuple], int]:
    """Split `text` into sentences and parse each with stanza once.

    Returns:
        parsed_sentences: list of (sentence_text, tokens) tuples, reusable
            across any number of independent matching passes.
        word_count: number of non-punctuation tokens in `text`, used to
            normalize the "per 100 words" metrics.
    """
    sentences = [s.text for s in sentenize(text)]
    parsed_sentences = []
    word_count = 0

    for sentence_text in sentences:
        parsed = nlp(sentence_text)
        doc_sentence = parsed.sentences[0]
        word_count += sum(1 for t in doc_sentence.tokens if t.to_dict()[0]["upos"] != "PUNCT")
        parsed_sentences.append((doc_sentence.text, doc_sentence.tokens))

    return parsed_sentences, word_count


def analyze_parsed(parsed_sentences: List[tuple], checker, patterns: Sequence[Pattern]) -> List[list]:
    """Run one independent match+score pass of `patterns` over sentences
    already parsed by `parse_sentences`."""
    sentence_results = []
    for sentence_text, tokens in parsed_sentences:
        sentence_json = sentence_to_json(sentence_text, tokens, patterns)
        sentence_results.append(checker.score_sentence(sentence_json))
    return sentence_results


def summarize_linkers(linker_sentence_results: List[list], threshold: float, word_count: int) -> dict:
    unique_linkers = set()
    linkers_by_appearance = []
    linker_count = 0

    for sentence in linker_sentence_results:
        kept = []
        for match in sentence:
            if round_half_up(match["probability"]) >= threshold:
                unique_linkers.add(match["linker"])
                linker_count += 1
                kept.append(match["linker"])
        linkers_by_appearance.append(kept)

    unique_linker_count = len(unique_linkers)

    return {
        "linkers_result": linker_sentence_results,
        "unique_linker_count": unique_linker_count,
        "linker_count": linker_count,
        "unique_linkers": sorted(unique_linkers),
        "linkers_by_appearance": linkers_by_appearance,
        "linkers_per_100": (linker_count / word_count * 100) if word_count else 0.0,
        "unique_linkers_per_100": (unique_linker_count / word_count * 100) if word_count else 0.0,
    }


def summarize_intro(intro_sentence_results: List[list], threshold: float, word_count: int) -> dict:
    unique_intro_words = set()
    intro_words_by_appearance = []
    intro_count = 0

    for sentence in intro_sentence_results:
        kept = []
        for match in sentence:
            if round_half_up(match["probability"]) >= threshold:
                unique_intro_words.add(match["linker"])
                intro_count += 1
                kept.append(match["linker"])
        intro_words_by_appearance.append(kept)

    unique_intro_count = len(unique_intro_words)

    return {
        "intro_result": intro_sentence_results,
        "unique_intro_count": unique_intro_count,
        "intro_count": intro_count,
        "unique_intro_words": sorted(unique_intro_words),
        "intro_words_by_appearance": intro_words_by_appearance,
        "intro_per_100": (intro_count / word_count * 100) if word_count else 0.0,
        "unique_intro_per_100": (unique_intro_count / word_count * 100) if word_count else 0.0,
    }


def combine_stats(linker_stats: dict, intro_stats: dict, word_count: int) -> dict:
    """Merge linker and introductory-word stats into one row, inserting the
    combined `linker_and_intro_per_100` metric between `intro_per_100` and
    `unique_intro_per_100`."""
    total_count = linker_stats["linker_count"] + intro_stats["intro_count"]

    combined = dict(linker_stats)
    combined["intro_result"] = intro_stats["intro_result"]
    combined["unique_intro_count"] = intro_stats["unique_intro_count"]
    combined["intro_count"] = intro_stats["intro_count"]
    combined["unique_intro_words"] = intro_stats["unique_intro_words"]
    combined["intro_words_by_appearance"] = intro_stats["intro_words_by_appearance"]
    combined["intro_per_100"] = intro_stats["intro_per_100"]
    combined["linker_and_intro_per_100"] = (total_count / word_count * 100) if word_count else 0.0
    combined["unique_intro_per_100"] = intro_stats["unique_intro_per_100"]
    return combined
