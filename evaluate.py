"""
evaluate.py — оценка экстрактора коннекторов на эталонном датасете.

Что делает:
  1. читает бенчмарк с ручной разметкой (.tsv / .xlsx), где линкеры
     заключены в квадратные скобки с меткой:
         [Если]=linker0 мы сделаем уроки, [то]=linker пойдём гулять.
  2. убирает эталонную разметку и прогоняет экстрактор
     (`pipeline.predict_spans`) на очищенном тексте;
  3. сопоставляет предсказанные спаны с эталонными по символьным
     смещениям (с запасным сравнением по поверхностной форме);
  4. считает precision / recall / F1 / F2 для `linker` и печатает
     debug-отчёт.

Экстрактор и его правила живут в linker_extraction; здесь — только
эталонный парсер и метрики. Модуль не строит разметку для показа и не
дублирует логику фронтенда (gradio_app).

Формат эталонной разметки (см. шапку src/benchmark.tsv):
  [x]=linker / =other            — метка фрагмента (other в метрики не идёт);
  [x]=linker0                    — первая часть составной конструкции;
  [x]=other=linker               — двойная метка, верна последняя;
  [x]==linker                    — baseline не нашёл даже омонима;
  метка intro из старых бенчмарков читается как linker;
  строки, где text начинается с "!" — комментарии, пропускаются.

Запуск:
  python evaluate.py                         # src/benchmark3_all_linkers.xlsx
  python evaluate.py --benchmark src/benchmark.tsv --threshold 0.4
  python evaluate.py --status ERR --limit 100 --report src/debug.txt
"""

import argparse
import re
import sys
from typing import List, Optional, Tuple

import pandas as pd

from pipeline import DEFAULT_THRESHOLD, build_engine, predict_spans
from tables import read_table

DEFAULT_BENCHMARK = "src/benchmark3_all_linkers.xlsx"


# =============================================================================
# РАЗБОР ЭТАЛОННОЙ РАЗМЕТКИ
# =============================================================================

# [текст] + хвост из одной или нескольких меток «=label» / «==label»
MARKUP_RE = re.compile(r"\[([^\]]*)\]((?:=+[a-z0-9]*)*)")


def _raw_label(suffix: str) -> str:
    """Последняя непустая метка из хвоста «=other=linker» / «==linker»."""
    labels = [lb for lb in re.findall(r"=+([a-z0-9]*)", suffix) if lb]
    return labels[-1] if labels else ""


def normalize_label(raw_label: str) -> Optional[str]:
    """linker0 / linker1 / intro* → linker; other / пусто → None.

    Вводные слова больше не выделяются в отдельный тип: метка intro из
    старых бенчмарков засчитывается как linker."""
    if not raw_label:
        return None
    if raw_label.startswith(("linker", "intro")):
        return "linker"
    return None


def parse_reference(marked_text: str) -> Tuple[str, List[dict]]:
    """Разбирает размеченную строку.

    Возвращает `(clean_text, entities)`, где:
      - clean_text — текст без разметки (именно он подаётся экстрактору);
      - entities   — список {"start", "end", "surface", "label"} со
                     смещениями в clean_text; label == "linker"
                     (метка other и пустые отброшены).
    """
    if not isinstance(marked_text, str) or not marked_text:
        return marked_text or "", []

    clean_parts: List[str] = []
    entities: List[dict] = []
    clean_len = 0
    pos = 0

    for m in MARKUP_RE.finditer(marked_text):
        # текст до совпадения — как есть
        gap = marked_text[pos:m.start()]
        clean_parts.append(gap)
        clean_len += len(gap)

        surface = m.group(1)
        label = normalize_label(_raw_label(m.group(2) or ""))

        start = clean_len
        clean_parts.append(surface)
        clean_len += len(surface)

        if label is not None:
            entities.append({
                "start": start,
                "end": clean_len,
                "surface": surface,
                "label": label,
            })
        pos = m.end()

    clean_parts.append(marked_text[pos:])
    return "".join(clean_parts), entities


def strip_markup(marked_text: str) -> str:
    """Только очищенный текст (обёртка над parse_reference)."""
    return parse_reference(marked_text)[0]


# =============================================================================
# ЗАГРУЗКА БЕНЧМАРКА
# =============================================================================

def load_benchmark(path: str) -> pd.DataFrame:
    """Читает .tsv / .xlsx с эталонной разметкой.

    Требуется колонка `text`. Строки-комментарии (text начинается с «!»)
    и полностью пустые строки отбрасываются.
    """
    df = read_table(path, as_str=True)
    df.columns = [c.strip() for c in df.columns]

    if "text" not in df.columns:
        raise ValueError(f"Колонка 'text' не найдена. Есть: {list(df.columns)}")

    df = df[~df["text"].str.startswith("!", na=False)].copy()
    df = df[df["text"].str.strip() != ""].copy()
    df.reset_index(drop=True, inplace=True)
    return df


# =============================================================================
# ПРЕДСКАЗАНИЕ
# =============================================================================

def predict_entities_(clean_text: str, engine, threshold: float) -> List[dict]:
    """Спаны экстрактора в том же формате, что и parse_reference:
    {"start", "end", "surface", "label"} со смещениями в clean_text.

    `surface` берётся срезом из clean_text (а не dict-форма экстрактора),
    чтобы сравнение с эталоном и debug-отчёт были согласованы."""
    spans = predict_spans(clean_text, engine, threshold=threshold)
    return [
        {
            "start": sp["start"],
            "end": sp["end"],
            "surface": clean_text[sp["start"]:sp["end"]],
            "label": sp["type"],
        }
        for sp in spans
        if sp["type"] == "linker"
    ]


def predict_entities(clean_text: str, engine, threshold: float) -> List[dict]:
    """Спаны экстрактора в том же формате, что и parse_reference:
    {"start", "end", "surface", "label"} со смещениями в clean_text.
    
    AS2: Возвращает как активные линкеры, так и отклонённые для отладки с защитой от None."""
    
    # Получаем кортеж. Если predict_spans вернул None вместо кортежа, подстрахуемся
    raw_res = predict_spans(clean_text, engine, threshold=threshold, with_rejected=True)
    
    if raw_res is None:
        return []
        
    # Разворачиваем кортеж
    spans, rejected = raw_res
    
    # AS2: Защита от сценария, когда списки пришли как NoneType
    if spans is None:
        spans = []
    if rejected is None:
        rejected = []
        
    result = []
    
    # 1. Действующие линкеры (is_rejected=False)
    for sp in spans:
        if isinstance(sp, dict) and sp.get("type") == "linker":
            result.append({
                "start": sp["start"],
                "end": sp["end"],
                "surface": clean_text[sp["start"]:sp["end"]],
                "label": "linker",
                "is_rejected": False
            })
            
    # 2. Отклонённые элементы (is_rejected=True)
    for sp in rejected:
        if isinstance(sp, dict):
            result.append({
                "start": sp["start"],
                "end": sp["end"],
                "surface": clean_text[sp["start"]:sp["end"]],
                "label": "other",      # Имя тега для лога ошибок
                "is_rejected": True    # Сигнал для математики
            })
        
    return result



# =============================================================================
# СОПОСТАВЛЕНИЕ И ОЦЕНКА ОДНОЙ СТРОКИ
# =============================================================================

def _norm_surface(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _overlap(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def _match(ref: List[dict], pred: List[dict]):
    """Жадное сопоставление ref↔pred.

    Проход 1: точное совпадение границ.
    Проход 2: запасной — перекрытие границ + совпадение поверхностной
              формы.

    Возвращает (tp_pairs, fp_entities, fn_entities).
    """
    ref_pool = list(ref)
    pred_pool = list(pred)
    tp: List[Tuple[dict, dict]] = []

    def take(matched):
        for p in list(pred_pool):
            for r in list(ref_pool):
                if matched(r, p):
                    tp.append((r, p))
                    ref_pool.remove(r)
                    pred_pool.remove(p)
                    break

    take(lambda r, p: r["start"] == p["start"] and r["end"] == p["end"])
    take(lambda r, p: _overlap(r, p)
         and _norm_surface(r["surface"]) == _norm_surface(p["surface"]))

    return tp, pred_pool, ref_pool


def evaluate_row(ref_ents: List[dict], pred_ents: List[dict]) -> dict:
    """Единственное место, где считаются tp/fp/fn, статус и span-метрики
    для одной пары (эталон, предсказание). Результат кладётся в DataFrame,
    поэтому метрики и debug-отчёт ничего не пересчитывают."""
    tp, fp, fn = _match(ref_ents, pred_ents)
    status = "OK" if not fp and not fn else "ERR"

    def items(seq, is_pair=False):
        if is_pair:
            return [(r["surface"].strip(), r["label"]) for r, _ in seq]
        return [(e["surface"].strip(), e["label"]) for e in seq]

    return {
        "status": status,
        "tp_items": items(tp, is_pair=True),
        "fp_items": items(fp),
        "fn_items": items(fn),

        "tp_linker": len(tp),
        "fp_linker": len(fp),
        "fn_linker": len(fn),
    }


# =============================================================================
# РАЗМЕТКА ДЛЯ DEBUG-ОТЧЁТА (только показ, не участвует в оценке)
# =============================================================================

def render_markup(clean_text: str, entities: List[dict]) -> str:
    """Вставляет [surface]=label в clean_text — для читаемого отчёта."""
    if not entities:
        return clean_text

    events = []
    for e in entities:
        events.append((e["start"], 1, e["end"], e["label"]))
        events.append((e["end"], 0, e["start"], e["label"]))
    events.sort(key=lambda ev: (ev[0], ev[1], -ev[2]))

    parts, pos = [], 0
    for idx, is_open, _other, label in events:
        parts.append(clean_text[pos:idx])
        parts.append("[" if is_open else f"]={label}")
        pos = idx
    parts.append(clean_text[pos:])
    return "".join(parts)


# =============================================================================
# ПРОГОН ПО DATAFRAME
# =============================================================================

def add_evaluation_columns(df: pd.DataFrame, engine, threshold: float) -> pd.DataFrame:
    """Прогоняет экстрактор по каждой строке и сохраняет результат оценки
    в колонки df: clean_text, pred_markup, status, tp/fp/fn_items, tp/fp/fn_*."""
    records = []
    for text in df["text"].fillna(""):

        # для отладки
        if "Дело в том" in text:
            pass 

        clean, ref_ents = parse_reference(text)
#        pred_ents = predict_entities(clean, engine, threshold)
        # AS2: Получаем полный список сущностей (включая отклонённые)
        pred_ents_all = predict_entities(clean, engine, threshold)

        # AS2: Для расчёта метрик и выявления ошибок [ERR] оставляем только НЕотклонённые
        pred_ents = [e for e in pred_ents_all if not e.get("is_rejected", False)]


        row = evaluate_row(ref_ents, pred_ents)
        row["clean_text"] = clean
        row["pred_markup"] = render_markup(clean, pred_ents_all) #AS2 передаём с отклонёнными
        records.append(row)

    evaluation_df = pd.DataFrame(records, index=df.index)
    return df.join(evaluation_df)


#AS2 функция называлась ранее evaluate_markup
def get_metrics(df: pd.DataFrame) -> dict:
    """Суммирует числовые колонки оценки и считает P/R/F1/F2 по label."""
    required = [f"{p}_linker" for p in ("tp", "fp", "fn")]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Нет колонок оценки: {missing}. "
                      "Сначала вызовите add_evaluation_columns(df).")

    metrics = {}
    for label in ("linker",):
        tp = int(df[f"tp_{label}"].sum())
        fp = int(df[f"fp_{label}"].sum())
        fn = int(df[f"fn_{label}"].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        
        #AS2: Перенесли расчет F1 и F2 из функции печати внутрь логики оценки
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) else 0.0
        
        #AS2: Добавили f1 и f2 в результирующий словарь для сохранения целостности метрик
        metrics[label] = {
            "precision": precision, 
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "tp": tp, 
            "fp": fp, 
            "fn": fn
        }
    return metrics

def print_metrics(metrics: dict) -> None:
    print("Метрики (span-level, micro):")
    print("-" * 60)
    for label in ("linker",):
        m = metrics[label]
        #AS2: Избавились от дублирования формул, теперь просто забираем готовые значения из словаря
        p, r, f1, f2 = m["precision"], m["recall"], m["f1"], m["f2"]
        print(f"{label:10s}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}  F2={f2:.4f}  "
              f"(tp={m['tp']}, fp={m['fp']}, fn={m['fn']})")
    print("-" * 60)


def debug_report(df: pd.DataFrame, n: int = 10, file_name: str = "",
                 status: str = "", fragments=()) -> None:
    """Печатает debug-отчёт из готовых колонок df (алгоритм не запускает)."""
    required = ["status", "tp_items", "fp_items", "fn_items",
               "clean_text", "pred_markup"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Нет колонок отчёта: {missing}. "
                      "Сначала вызовите add_evaluation_columns(df).")

    fh = open(file_name, "w", encoding="utf-8") if file_name else None

    def printf(text: str = "", to_screen: bool = True) -> None:
        if to_screen:
            print(text)
        if fh:
            fh.write(text + "\n")

    try:
        if n:
            fragments_str = f"; отбор по fragments: {fragments}" if fragments else ""
            status_str = f"; отбор по status: {status}" if status else ""
            printf(f"=== debug: всего {len(df)} строк, на экран первые {n}{fragments_str}{status_str} ===")

        shown = 0
        for idx, row in df.iterrows():
            if status and row["status"] != status:
                continue
            if fragments:
                clean = (row.get("clean_text", "") or "").lower()
                if not any(f.lower() in clean for f in fragments):
                    continue

            eid = row.get("№ примера", row.get("num", idx+1)) #AS2 разрешим колонке называться num, а индекс-заглушку надо увеличть на 1, так как она итдёт от 0
            to_screen = shown < n
            shown += 1

            parts = [f"{str(eid):>4} [{row['status']}]"]
            if row["tp_items"]:
                parts.append(f"tp={row['tp_items']}")
            if row["fp_items"]:
                parts.append(f"fp={row['fp_items']}")
            if row["fn_items"]:
                parts.append(f"fn={row['fn_items']}")
            printf("  ".join(parts), to_screen)

            if row["status"] != "OK":
                printf(f"ref:  {row['text']}", to_screen)
            printf(f"pred: {row['pred_markup']}", to_screen)

        if n:
            printf("=== end debug ===")
    finally:
        if fh:
            fh.close()


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

def process_benchmark(path: str, engine, threshold: float = DEFAULT_THRESHOLD,
                      n: int = 10, report_path: str = "src/debug.txt",
                      status: str = "") -> pd.DataFrame:
    df = load_benchmark(path)
    df = add_evaluation_columns(df, engine, threshold)
    print(f"Загружено примеров: {len(df)} из файла {path}")

    print_metrics(get_metrics(df))
    debug_report(df, n=n, file_name=report_path, status=status)
    return df


def main() -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK,
                       help=f"Файл с эталонной разметкой (default: {DEFAULT_BENCHMARK}).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                       help=f"Порог вероятности экстрактора (default: {DEFAULT_THRESHOLD}).")
    parser.add_argument("--limit", type=int, default=0,
                       help="Сколько строк отчёта печатать на экран (0 — только в файл).")
    parser.add_argument("--status", default="", choices=["", "OK", "ERR"],
                       help="Фильтр строк отчёта по статусу.")
    parser.add_argument("--report", default="src/debug.txt",
                       help="Куда писать полный debug-отчёт.")
    args = parser.parse_args()

    print("Loading stanza pipeline (tokenize,pos,lemma,depparse)...", file=sys.stderr)
    engine = build_engine()

    # #AS2 Просто для ускорения отладки
#    args.limit=30
    args.benchmark = "src/books200.xlsx"
    # #AS2


    df = process_benchmark(args.benchmark, engine, threshold=args.threshold,
                             n=args.limit, report_path=args.report,
                             status=args.status)
    
    debug_report(df, n=100, status="ERR")
    # debug_report(df, n=100, fragments=["тогда же"])    
    # debug_report(df, n=100, fragments=["не как"])    
    # debug_report(df, n=100, fragments=["именно"])    
    # debug_report(df, n=100, fragments=["также"])    
#    debug_report(df, n=100, fragments=["затем"])    
#    debug_report(df, n=100, fragments=["дело в том"])    
#    debug_report(df, n=100, status="ERR", fragments=["при этом", "в результате"])

    return df


if __name__ == "__main__":
    main()
