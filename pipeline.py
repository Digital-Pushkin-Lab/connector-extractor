"""High-level analysis pipeline: split text into sentences, parse each with
stanza, match linker patterns, and score them.

Ported from `get_stats` / `get_stats_with_examples` / `get_final_stats` in
`linkers (1).ipynb`, with results reshaped into the flat column layout the
downstream spreadsheets expect (see `summarize_linkers`).

All connectors (including former introductory words) come from a single
list, `data/linkers.tsv`, and are tagged `linker`.
"""

from __future__ import annotations

import math
import re
import uuid
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import List, NamedTuple, Sequence, Tuple, Dict, Iterable

from dataclasses import dataclass, field
from tqdm.auto import tqdm

from razdel import sentenize

from matching import Pattern, sentence_to_json
from patterns import build_patterns_from_csv

from collections import defaultdict #для сортирвоки

DEFAULT_THRESHOLD = 0.4

# Значение средних по части текста, в которой нет ни одного предложения.
NOT_APPLICABLE = "N/A"

# Абзац короче этого числа слов (не-пунктуационных токенов) присоединяется
# к предыдущему абзацу (первый абзац -- к следующему).
MIN_PARAGRAPH_WORDS = 5

# Абзац -- непустая строка текста: абзацы разделяются любым переводом строки
# (так их отдаёт чтение .docx), пустые строки просто пропускаются.
PARAGRAPH_RE = re.compile(r"[^\r\n]+")

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_LINKERS_CSV = DATA_DIR / "linkers.tsv"


def load_patterns(linkers_csv: str = None) -> dict:
    """Return {"linker": [...]}: the pattern list built from `linkers_tsv`."""
    linkers_csv = str(linkers_csv or DEFAULT_LINKERS_CSV)
    return {"linker": build_patterns_from_csv(linkers_csv)}



# Новый класс для оптимизации. Кэширует разбор от stanza, содержит функцию пакетной обработки. 
# (Пакетная обработка увеличила скорость разбора датасетов сразу на порядок)
# а также в него перенесена функция parse_sentences() которая теперь смотрит в кэш, в остальном работает как раньше
@dataclass
class Engine:
    """Stanza, scorer, паттерны и кэш разбора предложений."""

    # nlp: object
    # checker: object
    # patterns_by_type: dict
    # threshold: float = DEFAULT_THRESHOLD

    # # Кэш разметки Stanza
    # _sentence_cache: Dict[str, object] = field(
    #     default_factory=dict,
    #     init=False,
    #     repr=False,
    #     compare=False,
    # )
    def __init__(
        self,
        nlp: object,
        checker: object,
        patterns_by_type: dict,
        threshold=DEFAULT_THRESHOLD,        
    ) -> None:
        self.nlp = nlp
        self.checker = checker
        self.patterns_by_type = patterns_by_type
        self.threshold = threshold

        # Кэш разметки Stanza
        self._sentence_cache: Dict[str, object] = {}


    def preparse_texts(
        self,
        texts: Iterable[str],
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        """Пакетно разобрать отсутствующие в кэше предложения.

        Разбиение на абзацы и предложения совпадает с parse_sentences.
        В кэш сохраняются только результаты Stanza.
        Если verbose == True включаестя полоска tqdm
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        # Словарь одновременно сохраняет порядок и убирает дубликаты.
        pending = {}

        for text in texts:
            if not text or not text.strip():
                continue

            for para in PARAGRAPH_RE.finditer(text):
                for sub in sentenize(para.group()):
                    sentence_text = sub.text

                    if sentence_text not in self._sentence_cache:
                        pending.setdefault(sentence_text, None)

        sentence_texts = list(pending)

        progress = (
            tqdm(
                total=len(sentence_texts),
                desc="Parsing sentences",
                unit="text",
            )
            if verbose
            else None
        )

        try:
            for start in range(0, len(sentence_texts), batch_size):
                batch = sentence_texts[start:start + batch_size]

                parsed_docs = list(self.nlp.bulk_process(batch))

                if len(parsed_docs) != len(batch):
                    raise RuntimeError(
                        "Stanza returned an unexpected number of documents: "
                        f"expected {len(batch)}, got {len(parsed_docs)}"
                    )

                self._sentence_cache.update(zip(batch, parsed_docs))

                if progress is not None:
                    progress.update(len(batch))
        finally:
            if progress is not None:
                progress.close()


    def parse_sentences(
        self,
        text: str,
    ) -> Tuple[List[tuple], int]:
        """Split `text` into paragraphs and sentences and parse each sentence
        with stanza once.

        Paragraphs are non-empty lines (see PARAGRAPH_RE). Sentences are found
        with razdel inside each paragraph, so a sentence never spans two
        paragraphs. razdel decides the sentence boundaries. stanza's own tokenizer may split
        a razdel sentence further; in that case every stanza part is kept (and
        matched separately, since token ids/heads are per stanza sentence), but
        all parts still belong to the one razdel sentence -- their connectors
        are counted as coming from the same sentence.

        Returns:
            parsed_sentences: list of (sentence_text, parts, sentence_start,
                paragraph, n_words) tuples, one per razdel sentence, reusable
                across any number of matching passes. `parts` is a list of
                stanza token lists (usually just one). `sentence_start` is the
                sentence's character offset in `text`, needed to turn a token's
                start_char/end_char (relative to the razdel sentence) into an
                absolute offset (see `extract_spans`). `paragraph` is the
                0-based index of the sentence's paragraph and `n_words` its
                number of non-punctuation tokens (see `summarize_linkers`).
            word_count: number of non-punctuation tokens in `text`, used to
                normalize the "per 100 words" metrics.
        """
        parsed_sentences = []
        word_count = 0
        paragraph = 0

        for para in PARAGRAPH_RE.finditer(text):
            has_sentences = False

            for sub in sentenize(para.group()):
                sentence_text = sub.text

                if sentence_text not in self._sentence_cache:
                    # Обычный одиночный вызов остаётся рабочим,
                    # даже если preparse_texts не вызывался.
                    self._sentence_cache[sentence_text] = self.nlp(
                        sentence_text
                    )

                parsed = self._sentence_cache[sentence_text]

                if parsed.sentences:
                    parts = [
                        doc_sentence.tokens
                        for doc_sentence in parsed.sentences
                    ]

                    n_words = sum(
                        1
                        for tokens in parts
                        for token in tokens
                        if token.to_dict()[0]["upos"] != "PUNCT"
                    )

                    word_count += n_words

                    parsed_sentences.append((
                        sentence_text,
                        parts,
                        para.start() + sub.start,
                        paragraph,
                        n_words,
                    ))

                    has_sentences = True

            if has_sentences:
                paragraph += 1

        return parsed_sentences, word_count

    def clear_parse_cache(self) -> None:
        """Освободить результаты разбора."""
        self._sentence_cache.clear()

    # фцнкция возвращает размеченную скобками [] текстовую строку и список найденных паттернов. Перенесена из markup.py
    def mark_text(self, text: str) -> Tuple[str, List[dict]]:
        """
        Размечает предложение и возвращает кортеж:
        - размеченное предложение
        - список найденных паттернов

        В список паттернов входят как принятые, так и отклонённые span'ы.
        """
        if not text or not text.strip():
            return text, []

        parsed_sentences, _ = self.parse_sentences(text)

        spans = extract_spans(
            parsed_sentences,
            self.checker,
            self.patterns_by_type,
            return_rejected=True
        )

        spans = dedupe_spans(spans)


        patterns = []

        # принятые паттерны
        seen_composite = set()

        for sp in spans:
            surface = sp["surface"]
            pattern_type = sp["type"]

            if "..." in surface:
                key = (surface, pattern_type)

                # составной паттерн добавляем только один раз
                if key in seen_composite:
                    continue

                seen_composite.add(key)

            patterns.append({
                "surface": surface,
                "type": pattern_type if not sp["is_rejected"] else "other",
                "start": sp["start"],
                "end": sp["end"],
                "accepted": not sp["is_rejected"],

            })



        # сортируем паттерны по их позиции в исходном тексте
        patterns.sort(key=lambda p: (p["start"], p["end"]))

        if not spans :
            return text, patterns, []

        events = []

        # Разделяем на обычные и составные (по наличию "..." в surface)
        continuous = []
        groups = defaultdict(list)

        for sp in spans:
            if not sp["is_rejected"]:
                if "..." in sp["surface"]:
                    key = (sp["surface"], sp["type"])
                    groups[key].append(sp)
                else:
                    continuous.append(sp)

        # 1. Обычные (односоставные) — всегда без нуля
        for sp in continuous:
            cat = sp["type"]
            events.append((sp["start"], True, sp["end"], cat))
            events.append((sp["end"], False, sp["start"], cat))

        # 2. Составные — части нумеруются с 1
        for key, group in groups.items():
            group_sorted = sorted(group, key=lambda x: x["start"])

            for i, sp in enumerate(group_sorted):
                cat = f"{sp['type']}{i + 1}"
                events.append((sp["start"], True, sp["end"], cat))
                events.append((sp["end"], False, sp["start"], cat))

        # 3. Отклонённые
        for sp in spans:
            if sp["is_rejected"]:
                events.append((sp["start"], True, sp["end"], "other"))
                events.append((sp["end"], False, sp["start"], "other"))
        # Сортировка событий
        events.sort(
            key=lambda ev: (ev[0], 0 if not ev[1] else 1, -ev[2])
        )

        # Сборка размеченного текста
        parts = []
        pos = 0

        for idx, is_open, other, cat in events:
            parts.append(text[pos:idx])

            if is_open:
                parts.append("[")
            else:
                parts.append(f"]={cat}")

            pos = idx

        parts.append(text[pos:])
        marked_line = "".join(parts)




        return marked_line, patterns, spans




# Данная функция осталась здесь временно для совместимости как функция-заглушка. 
# Функционал перенесён внутрь класса Engine() для оптимизации парсинга через пакетную обработку
def parse_sentences(text: str, nlp) -> Tuple[List[tuple], int]:
 
    print("Вызвана устаревшая функция-обёртка parse_sentences(text: str, nlp). Требуется переписать через engine.parse_sentences(text) !!!")

    # создаём упрощённый Engine без чекера и паттернов, только Stanza
    engine = Engine(nlp=nlp, checker=None, patterns_by_type=None)

    # вызываем функцию из engine
    return engine.parse_sentences(text)



def build_engine(linkers_csv: str = None, nlp=None, threshold=DEFAULT_THRESHOLD) -> Engine:
    """Construct the extractor engine. Downloads the stanza `ru` model on
    first use if it is missing. Pass `nlp` to reuse an existing pipeline."""
    from rules import build_default_checker

    if nlp is None:
        import stanza
        processors = "tokenize,pos,lemma,depparse"
        try:
            nlp = stanza.Pipeline("ru", processors=processors,
                                  download_method=None, logging_level="ERROR")
        except Exception:
            stanza.download("ru")
            nlp = stanza.Pipeline("ru", processors=processors,
                                  logging_level="ERROR")

    return Engine(
        nlp=nlp,
        checker=build_default_checker(),
        patterns_by_type=load_patterns(linkers_csv),
        threshold = threshold,
    )


def round_half_up(value: float, ndigits: int = 1) -> float:
    """Round a float using 'round half up' (not Python's default 'round half
    to even')."""
    if not isinstance(value, (int, float)):
        raise TypeError("Input must be a float or int")

    decimal_value = Decimal(str(value))
    quantize_exp = Decimal("0.1") ** ndigits
    rounded_decimal = decimal_value.quantize(quantize_exp, rounding=ROUND_HALF_UP)
    return float(rounded_decimal)

#вызывается из extract.py
def analyze_parsed(parsed_sentences: List[tuple], checker, patterns: Sequence[Pattern]) -> List[list]:
    """Run one match+score pass of `patterns` over sentences already parsed
    by `parse_sentences`. Returns one result list per razdel sentence, with
    the matches from all of its stanza parts pooled together."""
    sentence_results = []
    for sentence_text, parts, *_ in parsed_sentences:
        scored = []
        for tokens in parts:
            sentence_json = sentence_to_json(sentence_text, tokens, patterns)
            scored.extend(checker.score_sentence(sentence_json))
        sentence_results.append(scored)
    return sentence_results


# эту функцию вызывает напрямую фронтенд Градио
# её же вызывает Engine.mark_text() - это мы сделали сознательно для максимального использования этого фрагмента
def extract_spans(
    parsed_sentences: List[tuple],
    checker,
    patterns_by_type: dict,
    threshold: float = DEFAULT_THRESHOLD,
    return_rejected: bool = False,          # новый параметр
    debug: bool = False,                    # новый параметр
) -> List[dict] | tuple[List[dict], List[dict]]:
    results = []


    for sentence_text, parts, sentence_start, *_ in parsed_sentences:

        # # для отладки
        # if "Екатерина" in sentence_text:
        #     debug=True    # 'этот параметр инициирует детальный вывод 
        #     pass

        # Все части одного razdel-предложения (если stanza разбила его
        # дальше) обрабатываются по отдельности; смещения токенов у всех
        # частей отсчитываются от начала razdel-предложения.
        for tokens in parts:

            for type_name, patterns in patterns_by_type.items():
                sentence_json = sentence_to_json(sentence_text, tokens, patterns, debug)
                phrases = checker.score_sentence(sentence_json, debug=debug)
                for phrase in phrases:
                    # Общий group_id для всех частей одного (возможно, разрывного)
                    # коннектора -- чтобы, например, "если" и "то" из "если...то"
                    # получили один и тот же id при отображении в gradio_app.
                    group_id = str(uuid.uuid4())[:8]
                    item = {
                        "start": None,          # заполним ниже
                        "end": None,
                        "surface": phrase.surface,
                        "type": type_name,
                        "probability": phrase.probability,
                        "group_id": group_id,
                        "is_rejected": phrase.probability < threshold, # добавили указание на отвергнутость
                        "phrase": phrase, # добавим сюда нашу фразу-первоисточник
                        
                    }

                    for s, e in phrase.spans:
                        span_item = item.copy()
                        span_item["start"] = sentence_start + s
                        span_item["end"] = sentence_start + e

                        if not span_item["is_rejected"] or return_rejected:
                            results.append(span_item)

    # добавляем эти поля в словарь, они нужны нам в evaluate.py для сравнения
    for sp in results:
        sp["text"] = sentence_text[sp["start"]:sp["end"]]
        sp["label"] = sp["type"] if not sp["is_rejected"] else "other"

    return results


def dedupe_spans(spans: List[dict]) -> List[dict]:
    """Collapse exact-duplicate (start, end) matches, keeping whichever has
    the higher scored probability. Order of first appearance is preserved.
    With a single pattern list this is a safety net rather than a necessity."""
    best: dict = {}
    order: List[Tuple[int, int]] = []
    for sp in spans:
        pos = (sp["start"], sp["end"])
        if pos not in best:
            best[pos] = sp
            order.append(pos)
        elif sp["probability"] > best[pos]["probability"]:
            best[pos] = sp
    return [best[pos] for pos in order]




def split_into_thirds(n_sentences: int) -> Tuple[int, int, int]:
    """Number of sentences in the beginning, middle and end of a text.

    Beginning and middle get ceil(n/3) sentences each and the end gets the
    rest, so the end is the shortest part (5 -> 2+2+1). Short texts:
    1 -> 1+0+0, 2 -> 1+0+1 (no middle), 4 -> 2+1+1."""
    special = {0: (0, 0, 0), 1: (1, 0, 0), 2: (1, 0, 1), 4: (2, 1, 1)}
    if n_sentences in special:
        return special[n_sentences]
    k = math.ceil(n_sentences / 3)
    return k, k, n_sentences - 2 * k


def text_parts(parsed_sentences: List[tuple]) -> Tuple[List[int], List[int], List[int]]:
    """Indices of the sentences in the beginning, middle and end of a text.

    If the text has several paragraphs, it is divided by paragraphs:
    paragraphs shorter than MIN_PARAGRAPH_WORDS words are first merged into
    the previous paragraph (a short first paragraph -- into the next one),
    then the paragraphs are divided into thirds with `split_into_thirds`,
    and each third takes all sentences of its paragraphs. A text with a
    single paragraph (also after merging) is divided by sentences."""
    # Абзацы в порядке следования: номер абзаца -> (индексы предложений, число слов)
    paragraphs: List[Tuple[List[int], int]] = []
    for i, (*_, paragraph, n_words) in enumerate(parsed_sentences):
        if paragraph == len(paragraphs):
            paragraphs.append(([], 0))
        sentences, words = paragraphs[paragraph]
        sentences.append(i)
        paragraphs[paragraph] = (sentences, words + n_words)

    groups: List[List[int]] = []
    pending: List[int] = []   # короткие абзацы в начале текста
    for sentences, words in paragraphs:
        if words < MIN_PARAGRAPH_WORDS:
            if groups:
                groups[-1].extend(sentences)
            else:
                pending.extend(sentences)
        else:
            groups.append(pending + sentences)
            pending = []
    if pending:  # в тексте нет ни одного абзаца нужной длины
        groups.append(pending)

    if len(groups) <= 1:
        units = [[i] for i in range(len(parsed_sentences))]
    else:
        units = groups

    n_begin, n_middle, _n_end = split_into_thirds(len(units))
    flatten = lambda chunk: [i for unit in chunk for i in unit]
    return (
        flatten(units[:n_begin]),
        flatten(units[n_begin:n_begin + n_middle]),
        flatten(units[n_begin + n_middle:]),
    )


def _mean_linkers(per_sentence_counts: List[int]):
    """Mean number of linkers per sentence, or NOT_APPLICABLE if there are
    no sentences."""
    if not per_sentence_counts:
        return NOT_APPLICABLE
    return sum(per_sentence_counts) / len(per_sentence_counts)


def summarize_linkers(linker_sentence_results: List[list], threshold: float, word_count: int,
                      parsed_sentences: List[tuple] = None) -> dict:
    """Per-text statistics. `parsed_sentences` (from `parse_sentences`)
    gives the paragraph layout used to divide the text into beginning,
    middle and end (see `text_parts`); without it the text is divided by
    sentences."""
    unique_linkers = set()
    linkers_by_appearance = []
    linker_count = 0

    for sentence in linker_sentence_results:
        kept = []
        for match in sentence:
            if round_half_up(match["probability"]) >= threshold:
                unique_linkers.add(match["surface"])
                linker_count += 1
                kept.append(match["surface"])
        linkers_by_appearance.append(kept)

    unique_linker_count = len(unique_linkers)

    # Предложения здесь -- предложения razdel (см. parse_sentences).
    counts = [len(kept) for kept in linkers_by_appearance]
    if parsed_sentences is None:
        # без разметки абзацев весь текст считается одним абзацем,
        # что сводится к делению по предложениям
        parsed_sentences = [(None, None, None, 0, 0)] * len(counts)
    begin, middle, end = text_parts(parsed_sentences)

    return {
        "linkers_result": linker_sentence_results,
        "unique_linker_count": unique_linker_count,
        "linker_count": linker_count,
        "unique_linkers": sorted(unique_linkers),
        "linkers_by_appearance": linkers_by_appearance,
        "linkers_per_100": (linker_count / word_count * 100) if word_count else 0.0,
        "unique_linkers_per_100": (unique_linker_count / word_count * 100) if word_count else 0.0,
        "linkers_per_sentence": _mean_linkers(counts),
        "linkers_per_sentence_beginning": _mean_linkers([counts[i] for i in begin]),
        "linkers_per_sentence_middle": _mean_linkers([counts[i] for i in middle]),
        "linkers_per_sentence_end": _mean_linkers([counts[i] for i in end]),
    }
