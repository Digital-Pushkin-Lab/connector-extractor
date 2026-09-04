
"""
simple_markup.py

Читает текст из input_file_name,

Пример результата:
[Без сомненья]=intro, [если]=linker ты придёшь, [то]=linker я буду рад, [но]=linker если нет — тоже хорошо.


Изменения в этой версии:
- Введена единая функция evaluate_row для оценки одной строки (tp/fp/fn, статус, span-метрики).
- compare_row, evaluate_markup и debug_report переписаны как обёртки над evaluate_row, чтобы убрать дублирование.
- evaluate_markup теперь делает один проход по датафрейму вместо двух.
- Логика оценки в метриках и в debug_report гарантированно одинаковая.
"""


import os, sys, re
import pandas as pd
from pathlib import Path
from io import StringIO
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Iterable, Any, Optional

import txt

# ---------------------------------------------------------------------------
# Пути к extractor (точно как в gradio_app.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ""))

import stanza
from extract import load_patterns
from pipeline import extract_spans, parse_sentences
from rules import build_default_checker

LINKERS_CSV = os.path.join(os.path.dirname(__file__), "", "data", "linkers.csv")
INTRO_CSV   = os.path.join(os.path.dirname(__file__), "", "data", "intro_words.csv")


print("Loading stanza pipeline (tokenize,pos,lemma,depparse)...", file=sys.stderr)


def get_stanza():
    """
    Инициализирует stanza-пайплайн для русского языка.
    При отсутствии модели — скачивает её.
    """
    try:
        st = stanza.Pipeline(
            'ru',
            processors="tokenize,pos,lemma,depparse",
            download_method=None,
            logging_level='ERROR'
        )
        print("Модель stanza ru используется локально")
        return st
    except Exception as e:
        if "model not found" in str(e).lower() or "download" in str(e).lower():
            print("Скачиваем модель ru...")
            stanza.download("ru")
            st = stanza.Pipeline('ru', processors="tokenize,pos,lemma,depparse")
            return st
        else:
            raise


NLP = get_stanza()

# NLP = stanza.Pipeline("ru", processors="tokenize,pos,lemma,depparse")

CHECKER = build_default_checker()
PATTERNS_BY_TYPE = load_patterns("both", LINKERS_CSV, INTRO_CSV)


def print_sent(parsed_sentences) -> None:
    """Печатает синтаксические зависимости Stanza в виде простой таблицы."""
    for sent_i, (sentence_text, tokens, _start) in enumerate(parsed_sentences, start=1):
        print(f"\n=== [{sent_i}] {sentence_text}")
        print(f"{'id':>4}  {'word':<20} {'upos':<8} {'head':>4}  {'deprel':<12}  head_word")
        print("-" * 70)

        # id → текст (для колонки head_word)
        id2text = {}
        rows = []
        for tok in tokens:
            # token.to_dict() — list слов; у MWT их несколько
            for w in tok.to_dict():
                wid = w.get("id")
                # id бывает int или tuple у MWT-диапазонов — берём только «обычные» слова
                if not isinstance(wid, int):
                    continue
                id2text[wid] = w.get("text", "")
                rows.append(w)

        for w in rows:
            wid = w["id"]
            head = w.get("head", 0)
            deprel = w.get("deprel", "")
            upos = w.get("upos", "")
            text = w.get("text", "")
            if head == 0:
                head_word = "ROOT"
            else:
                head_word = id2text.get(head, "?")
            print(
                f"{wid:>4}  {text:<20} {upos:<8} {head:>4}  {deprel:<12}  {head_word}"
            )


# =============================================================================
# БЛОК РАЗМЕТКИ (аналогично gradio_app.py)
# =============================================================================


def dedupe_spans(spans: List[dict]) -> List[dict]:
    """
    Оставляет только лучший (по probability) span при полном совпадении границ.
    Используется для устранения дубликатов от extract_spans.
    """
    best: Dict[Tuple[int, int], dict] = {}
    order: List[Tuple[int, int]] = []
    for sp in spans:
        pos = (sp["start"], sp["end"])
        if pos not in best:
            best[pos] = sp
            order.append(pos)
        elif sp["probability"] > best[pos]["probability"]:
            best[pos] = sp
    return [best[pos] for pos in order]


def mark_text(text: str) -> str:
    """
    Размечает текст коннекторами и вводными словами.
    Возвращает строку с конструкциями вида [фрагмент]=label.
    """
    if not text or not text.strip():
        return text

    parsed_sentences, _ = parse_sentences(text, NLP)

    spans, rejected = extract_spans(
        parsed_sentences,
        CHECKER,
        PATTERNS_BY_TYPE,
        return_rejected=True
    )

    spans = dedupe_spans(spans)
    rejected = dedupe_spans(rejected)

    if not spans and not rejected:
        return text

    events = []

    # Разделяем на обычные и составные (по наличию "..." в surface)
    continuous = []
    groups = defaultdict(list)

    for sp in spans:
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

    # 2. Составные — первой части добавляем 0
    for key, group in groups.items():
        group_sorted = sorted(group, key=lambda x: x["start"])
        for i, sp in enumerate(group_sorted):
            cat = sp["type"] + ("0" if i == 0 else "")
            events.append((sp["start"], True, sp["end"], cat))
            events.append((sp["end"], False, sp["start"], cat))

    # 3. Отклонённые
    for sp in rejected:
        events.append((sp["start"], True, sp["end"], "other"))
        events.append((sp["end"], False, sp["start"], "other"))

    # Сортировка событий для корректной сборки разметки
    events.sort(key=lambda ev: (ev[0], 0 if not ev[1] else 1, -ev[2]))

    # Сборка текста с разметкой
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
    return "".join(parts)


# =============================================================================
# БЛОК ЗАГРУЗКИ БЕНЧМАРКА (эталонной разметки)
# =============================================================================


def load_benchmark(path: str) -> pd.DataFrame:
    """
    Читает .tsv или .xlsx с эталонной разметкой.
    Пропускает строки-комментарии (text начинается с '!').
    """
    text = txt.load(path)  # Читаем файл в строку text (даже xlsx → текст с табуляциями)

    df = pd.read_csv(
        StringIO(text),
        sep="\t",
        dtype=str,
        keep_default_na=False,  # пустые ячейки остаются пустыми строками
    )

    # Нормализуем имена колонок на случай лишних пробелов
    df.columns = [c.strip() for c in df.columns]

    text_col = "text"
    if text_col not in df.columns:
        raise ValueError(f"Колонка '{text_col}' не найдена. Есть: {list(df.columns)}")

    # Пропускаем комментарии: text начинается с '!'
    mask_comment = df[text_col].str.startswith("!", na=False)
    df = df[~mask_comment].copy()

    # Убираем полностью пустые строки (нет номера и нет текста)
    df = df[df[text_col].str.strip() != ""].copy()

    df.reset_index(drop=True, inplace=True)
    return df


def strip_markup(text: str) -> str:
    """
    Убирает разметку коннекторов из текста.

    Разметка: [фрагмент]=метка  или  [фрагмент]==метка  и т.п.
    Метка — маленькие латинские буквы, цифры и знаки '='.

    Примеры:
        "[Правда]=linker, результаты..." → "Правда, результаты..."
        "[В результате]=other=linker формируются..." → "В результате формируются..."
        "[В отличие от]==linker устной..." → "В отличие от устной..."
    """
    if not isinstance(text, str):
        return text

    # [  + содержимое +  ]  +  один или больше =метка
    # метка: [a-z0-9=]+
    pattern = re.compile(r"\[([^\]]*)\]=[a-z0-9=]*")
    # Заменяем на содержимое скобок (группа 1)
    clean = pattern.sub(r"\1", text)
    return clean


def add_clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку с текстом без разметки (clean_text)."""
    df["clean_text"] = df["text"].map(strip_markup)
    return df


def add_result_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку с автоматически размеченным текстом (result_text)."""
    df["result_text"] = df["clean_text"].map(mark_text)
    return df


# =============================================================================
# БЛОК СРАВНЕНИЯ РЕЗУЛЬТАТОВ РАЗМЕТКИ С ЭТАЛОННОЙ РАЗМЕТКОЙ
# =============================================================================

# [текст] + хвост из одного или нескольких =метка / ==метка
MARKUP_RE = re.compile(r"\[([^\]]*)\]((?:=+[a-z0-9]*)*)")


def parse_markup(text: str) -> List[Tuple[str, str]]:
    """
    Парсит размеченную строку.

    Возвращает список (span_text, raw_label),
    где raw_label — последняя непустая метка после '='.
    """
    if not isinstance(text, str) or not text:
        return []

    entities: List[Tuple[str, str]] = []
    for m in MARKUP_RE.finditer(text):
        span = m.group(1)
        suffix = m.group(2) or ""

        # Из «=linker=intro» / «==linker» / «=other=linker» достаём метки
        labels = re.findall(r"=+([a-z0-9]*)", suffix)
        labels = [lb for lb in labels if lb]  # убираем пустые от «==»
        raw_label = labels[-1] if labels else ""

        entities.append((span, raw_label))
    return entities


def normalize_label(raw_label: str) -> Optional[str]:
    """
    Оставляет только интересующие сущности.
    linker0 / linker1 → linker
    intro → intro
    other / пусто → None (не участвует в метриках)
    """
    if not raw_label:
        return None
    if raw_label.startswith("linker"):
        return "linker"
    if raw_label.startswith("intro"):
        return "intro"
    return None


def entities_for_metrics(text: str) -> List[Tuple[str, str]]:
    """
    Список (span, label) только для linker и intro.
    Используется для расчёта метрик.
    """
    result = []
    for span, raw in parse_markup(text):
        label = normalize_label(raw)
        if label is not None:
            # лёгкая нормализация пробелов внутри span
            result.append((span.strip(), label))
    return result


# =============================================================================
# ЕДИНАЯ ФУНКЦИЯ ОЦЕНКИ ОДНОЙ СТРОКИ (ключевое упрощение)
# =============================================================================


# =============================================================================
# ОЦЕНКА РЕЗУЛЬТАТОВ И ОТЧЁТ
# =============================================================================


def evaluate_row(ref_text: str, pred_text: str) -> dict:
    """
    Оценивает одну пару строк.

    Это единственное место, где:
    - извлекаются сущности из эталона и результата;
    - считаются tp/fp/fn;
    - определяется статус OK / PART / ERR;
    - считаются span'ы для общей метрики connector.

    Результат затем сохраняется в DataFrame.
    Поэтому evaluate_markup и debug_report ничего не пересчитывают.
    """
    ref_ents = entities_for_metrics(ref_text)
    pred_ents = entities_for_metrics(pred_text)

    ref_c = Counter(ref_ents)
    pred_c = Counter(pred_ents)

    tp_mult = ref_c & pred_c
    fp_mult = pred_c - ref_c
    fn_mult = ref_c - pred_c

    # Сворачиваем Counter вида {(span, label): count}
    # в Counter вида {label: count}.
    def by_label(items: Counter) -> Counter:
        result = Counter()
        for (_, label), count in items.items():
            result[label] += count
        return result

    tp = by_label(tp_mult)
    fp = by_label(fp_mult)
    fn = by_label(fn_mult)

    # Для connector тип linker/intro не важен:
    # важен только сам найденный текстовый фрагмент.
    ref_spans = Counter(span for span, _ in ref_ents)
    pred_spans = Counter(span for span, _ in pred_ents)

    if not fp and not fn:
        status = "OK"
    elif ref_spans == pred_spans:
        # Границы всех сущностей совпали,
        # но хотя бы одна метка linker/intro перепутана.
        status = "PART"
    else:
        status = "ERR"

    return {
        "status": status,

        # Списки нужны только для удобного отчёта по конкретной строке.
        "tp_items": list(tp_mult.elements()),
        "fp_items": list(fp_mult.elements()),
        "fn_items": list(fn_mult.elements()),

        # Числа нужны для суммарных метрик.
        "tp_linker": tp["linker"],
        "fp_linker": fp["linker"],
        "fn_linker": fn["linker"],

        "tp_intro": tp["intro"],
        "fp_intro": fp["intro"],
        "fn_intro": fn["intro"],

        # Метрики connector: linker и intro объединяются.
        "tp_connector": sum((ref_spans & pred_spans).values()),
        "fp_connector": sum((pred_spans - ref_spans).values()),
        "fn_connector": sum((ref_spans - pred_spans).values()),
    }


def add_evaluation_columns(
    df: pd.DataFrame,
    ref_col: str = "text",
    pred_col: str = "result_text",
) -> pd.DataFrame:
    """
    Добавляет в DataFrame результаты оценки каждой строки.

    Это ключевое изменение:
    оценка выполняется один раз и сохраняется в колонках DataFrame.

    Добавляемые колонки:
    - status: OK / PART / ERR;
    - tp_items, fp_items, fn_items: сущности для debug-отчёта;
    - tp_*, fp_*, fn_*: числовые результаты для общих метрик.
    """
    if ref_col not in df.columns:
        raise KeyError(
            f"Колонка эталона '{ref_col}' не найдена. "
            f"Доступные: {list(df.columns)}"
        )

    if pred_col not in df.columns:
        raise KeyError(
            f"Колонка результата '{pred_col}' не найдена. "
            f"Доступные: {list(df.columns)}"
        )

    # apply возвращает Series из словарей;
    # pd.DataFrame разворачивает эти словари в отдельные колонки.
    row_results = df.apply(
        lambda row: evaluate_row(
            row.get(ref_col, "") or "",
            row.get(pred_col, "") or "",
        ),
        axis=1,
    )

    evaluation_df = pd.DataFrame(
        row_results.tolist(),
        index=df.index,
    )

    # join добавляет результаты алгоритма к исходному датасету.
    return df.join(evaluation_df)


def evaluate_markup(df: pd.DataFrame) -> dict:
    """
    Считает общие метрики из уже сохранённых колонок DataFrame.

    Здесь больше нет:
    - обхода text/result_text;
    - вызовов parse_markup;
    - вызовов entities_for_metrics;
    - повторного вычисления tp/fp/fn.

    Функция только суммирует результаты, которые уже вычислила
    add_evaluation_columns().
    """
    required_cols = [
        "tp_linker", "fp_linker", "fn_linker",
        "tp_intro", "fp_intro", "fn_intro",
        "tp_connector", "fp_connector", "fn_connector",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            "Не найдены колонки оценки: "
            f"{missing}. Сначала вызовите add_evaluation_columns(df)."
        )

    metrics = {}

    for label in ("linker", "intro", "connector"):
        tp = int(df[f"tp_{label}"].sum())
        fp = int(df[f"fp_{label}"].sum())
        fn = int(df[f"fn_{label}"].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return metrics


def print_metrics(metrics: dict) -> None:
    """Выводит P, R, F1 и F2 для linker, intro и connector."""
    print("Метрики (span-level, micro):")
    print("-" * 50)

    for label in ("linker", "intro", "connector"):
        m = metrics[label]
        p = m["precision"]
        r = m["recall"]

        f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
        f2 = (5 * p * r) / (4 * p + r) if (4 * p + r) else 0.0

        print(
            f"{label:10s}  "
            f"P = {p:.4f}  "
            f"R = {r:.4f}  "
            f"F1 = {f1:.4f}  "
            f"F2 = {f2:.4f}  "
            f"(tp={m['tp']}, fp={m['fp']}, fn={m['fn']})"
        )

    print("-" * 50)


def debug_report(
    df: pd.DataFrame,
    n: int = 10,
    file_name: str = "",
    status: str = "",
    fragments=(),
    ref_col: str = "text",
    pred_col: str = "result_text",
) -> None:
    """
    Печатает debug-отчёт на основе готовых колонок DataFrame.

    Важно: эта функция не запускает алгоритм и не пересчитывает оценку.
    Она только читает result_text, status, tp_items, fp_items и fn_items,
    добавленные ранее функцией add_evaluation_columns().
    """
    required_cols = ["status", "tp_items", "fp_items", "fn_items"]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            "Не найдены колонки отчёта: "
            f"{missing}. Сначала вызовите add_evaluation_columns(df)."
        )

    fh = open(file_name, "w", encoding="utf-8") if file_name else None

    def printf(text: str = "", verbose: bool = True) -> None:
        if verbose:
            print(text)
        if fh:
            fh.write(text + "\n")

    try:
        printf(f"=== debug: всего {len(df)} строк, на экран первые {n} ===")

        shown = 0

        for idx, row in df.iterrows():
            current_status = row["status"]

            # Фильтр по статусу.
            if status and current_status != status:
                continue

            # Фильтр по фрагментам очищенного текста.
            if fragments:
                clean_text = (row.get("clean_text", "") or "").lower()
                if not any(fragment.lower() in clean_text for fragment in fragments):
                    continue

            eid = row.get("№ примера", idx)
            ref_text = row.get(ref_col, "") or ""
            pred_text = row.get(pred_col, "") or ""

            # Первые n строк печатаются и в терминал, и в файл.
            # Остальные — только в файл.
            show_in_terminal = shown < n
            shown += 1

            parts = [f"{eid:>4} [{current_status}]"]

            if row["tp_items"]:
                parts.append(f"tp={row['tp_items']}")
            if row["fp_items"]:
                parts.append(f"fp={row['fp_items']}")
            if row["fn_items"]:
                parts.append(f"fn={row['fn_items']}")

            printf("  ".join(parts), show_in_terminal)

            if current_status == "OK":
                printf(f"pred: {pred_text}", show_in_terminal)
            else:
                printf(f"ref:  {ref_text}", show_in_terminal)
                printf(f"pred: {pred_text}", show_in_terminal)

        printf("=== end debug ===")

    finally:
        if fh:
            fh.close()


# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ МОДУЛЯ
# =============================================================================


def process_benchmark(file_name: str, n: int = 10) -> pd.DataFrame:
    """
    Полный и линейный пайплайн:

    1. Загружаем эталонный датасет.
    2. Убираем эталонную разметку -> clean_text.
    3. Запускаем алгоритм разметки -> result_text.
    4. Один раз оцениваем каждый результат -> колонки status/tp/fp/fn.
    5. Считаем общие метрики из готовых колонок.
    6. Печатаем лог из готовых колонок.
    """
    df = load_benchmark(file_name)
    df = add_clean_text_column(df)

    # Алгоритм работает здесь: clean_text -> result_text.
    df = add_result_text_column(df)

    # Оценка результата алгоритма выполняется здесь и только один раз.
    # После этого все данные для метрик и отчёта уже лежат в df.
    df = add_evaluation_columns(df)

    print(f"Загружено примеров: {len(df)} из файла {file_name}")

    # Метрики не оценивают текст заново: только суммируют числовые колонки df.
    metrics = evaluate_markup(df)
    print_metrics(metrics)

    # Отчёт не запускает алгоритм и не сравнивает разметку:
    # он использует result_text и результаты оценки, уже сохранённые в df.
    debug_report(df, n=n, file_name="src/debug.txt")

    return df 

def main() -> pd.DataFrame:
    df = process_benchmark("src/benchmark2.xlsx", n=0)

    # Дополнительные отчёты тоже ничего не пересчитывают:
    # они лишь выбирают нужные строки из готового df.
    debug_report(df, n=100, status="PART")

    debug_report(df, n=100, status="ERR")
    # debug_report(df, n=100, fragments=["как правило"])
    # debug_report(df, n=100, fragments=["при этом", "в результате"])

    return df


if __name__ == "__main__":
    main()


