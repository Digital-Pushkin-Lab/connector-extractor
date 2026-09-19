"""Сравнение экстрактора с разными синтаксическими парсерами на эталонном
бенчмарке (см. evaluate.py):

  stanza     pos + depparse из stanza (как раньше);
  bert       upos/head/deprel из ruBERT-парсера IuliiaPr/my-ruBert-parser-model;
  bert-deps  head/deprel из ruBERT-парсера, upos из stanza (отделяет вклад POS).

Токенизация и леммы во всех вариантах — stanza; правила, списки и порог те же,
поэтому разница в метриках объясняется только синтаксической разметкой.

Печатает и сохраняет в --out-dir (по умолчанию results/parser_comparison/):
  summary.txt        метрики P/R/F1/F2 по каждому эксперименту (блоки
                     «ЭКСПЕРИМЕНТ n: <parser>») + доля совпадений разметки;
  metrics.csv        те же метрики в виде таблицы (колонка experiment);
  debug_<parser>.txt полный debug-отчёт по строкам бенчмарка, по одному на
                     эксперимент (как `evaluate.py --report`);
  differences.txt    строки, где предсказания разных парсеров разошлись.
Повторный запуск перезаписывает файлы в этом каталоге.

Запуск (нужен py38: conda activate py38):
  python compare_parsers.py
  python compare_parsers.py --parsers stanza bert --out-dir results/my_run
"""

import argparse
import contextlib
import datetime
import io
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from evaluate import (DEFAULT_BENCHMARK, add_evaluation_columns, debug_report,
                      evaluate_markup, load_benchmark)
from pipeline import DEFAULT_THRESHOLD, PARSERS, build_engine, parse_sentences

LABELS = ("linker", "intro", "connector")

DESCRIPTIONS = {
    "stanza": "upos, head, deprel из stanza (pos + depparse) — базовый вариант",
    "bert": "upos, head, deprel из ruBERT-парсера (IuliiaPr/my-ruBert-parser-model)",
    "bert-deps": "head, deprel из ruBERT-парсера; upos из stanza",
}


def prf(r: dict):
    """(precision, recall, F1, F2) из записи evaluate_markup."""
    p, rec = r["precision"], r["recall"]
    f1 = (2 * p * rec) / (p + rec) if (p + rec) else 0.0
    f2 = (5 * p * rec) / (4 * p + rec) if (4 * p + rec) else 0.0
    return p, rec, f1, f2


def git_revision() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return f"{branch} @ {rev}" + (" (+ незакоммиченные изменения)" if dirty else "")
    except Exception:
        return "неизвестно"


def build_summary(args, n_examples, engines, metrics, seconds, agreement) -> List[str]:
    bert_parser = next((getattr(e.nlp, "parser", None) for e in engines.values()
                        if hasattr(e.nlp, "parser")), None)
    lines = [
        "СРАВНЕНИЕ СИНТАКСИЧЕСКИХ ПАРСЕРОВ В ЭКСТРАКТОРЕ ЛИНКЕРОВ / ВВОДНЫХ СЛОВ",
        "=" * 78,
        f"дата:        {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"код:         {git_revision()}",
        f"бенчмарк:    {args.benchmark} ({n_examples} примеров)",
        f"порог:       {args.threshold}",
        f"команда:     python compare_parsers.py {' '.join(sys.argv[1:])}".rstrip(),
        "",
        "Общее для всех экспериментов: токенизация и леммы — stanza (ru); словари,",
        "правила rules.py и порог одни и те же. Меняется только источник upos/head/deprel.",
    ]
    if bert_parser is not None:
        lines += [f"Модель BERT: {bert_parser.repo_id} (ревизия {bert_parser.revision[:10]}); "
                  "метки deprel с подтипом приведены к базовым (parataxis:discourse -> parataxis)."]
    for n, name in enumerate(args.parsers, start=1):
        lines += ["", "=" * 78, f"ЭКСПЕРИМЕНТ {n}: {name}" + ("  (база для сравнения)" if n == 1 else ""),
                  f"  {DESCRIPTIONS[name]}", "-" * 78]
        for label in LABELS:
            r = metrics[name][label]
            p, rec, f1, f2 = prf(r)
            lines.append(f"  {label:10s} P={p:.4f}  R={rec:.4f}  F1={f1:.4f}  F2={f2:.4f}  "
                         f"(tp={r['tp']}, fp={r['fp']}, fn={r['fn']})")
        lines.append(f"  время прогона: {seconds[name]:.0f} с")
    if agreement:
        lines += ["", "=" * 78, f"СОВПАДЕНИЕ РАЗМЕТКИ С ЭКСПЕРИМЕНТОМ 1 ({args.parsers[0]}), по токенам бенчмарка",
                  "-" * 78]
        for name, ag in agreement.items():
            lines.append(f"  {name:10s} POS {ag['pos']:.4f}  UAS {ag['uas']:.4f}  LAS {ag['las']:.4f}"
                         f"  ({ag['tokens']} токенов"
                         + (f", пропущено предложений с разной токенизацией: {ag['skipped_sentences']}"
                            if ag["skipped_sentences"] else "") + ")")
    return lines


def write_metrics_csv(path: Path, metrics: Dict[str, dict]) -> None:
    rows = []
    for name, m in metrics.items():
        for label in LABELS:
            r = m[label]
            p, rec, f1, f2 = prf(r)
            rows.append({"experiment": name, "description": DESCRIPTIONS[name], "label": label,
                         "precision": round(p, 4), "recall": round(rec, 4),
                         "f1": round(f1, 4), "f2": round(f2, 4),
                         "tp": r["tp"], "fp": r["fp"], "fn": r["fn"]})
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_agreement(texts, stanza_engine, other_engine) -> dict:
    """Доля токенов, на которых две разметки совпадают (токены одни и те же)."""
    total = same_pos = same_head = same_las = skipped = 0
    for text in texts:
        if not text.strip():
            continue
        a, _ = parse_sentences(text, stanza_engine.nlp)
        b, _ = parse_sentences(text, other_engine.nlp)
        for (_, toks_a, _), (_, toks_b, _) in zip(a, b):
            if len(toks_a) != len(toks_b):
                skipped += 1
                continue
            for ta, tb in zip(toks_a, toks_b):
                da, db = ta.to_dict()[0], tb.to_dict()[0]
                total += 1
                same_pos += da["upos"] == db["upos"]
                same_head += da["head"] == db["head"]
                same_las += da["head"] == db["head"] and da["deprel"] == db["deprel"]
    return {"tokens": total, "pos": same_pos / total, "uas": same_head / total,
            "las": same_las / total, "skipped_sentences": skipped}


def write_diff_report(path: Path, dfs: Dict[str, pd.DataFrame]) -> int:
    base_name, *others = dfs
    base = dfs[base_name]
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for idx, row in base.iterrows():
            variants = {o: dfs[o].loc[idx] for o in others}
            if all(v["pred_markup"] == row["pred_markup"] for v in variants.values()):
                continue
            n += 1
            statuses = " ".join(f"{name}={df.loc[idx]['status']}" for name, df in dfs.items())
            fh.write(f"{row.get('№ примера', idx)}  [статус по экспериментам: {statuses}]\n")
            fh.write(f"ref:  {row['text']}\n")
            for name, df in dfs.items():
                fh.write(f"{name + ':':10s} {df.loc[idx]['pred_markup']}\n")
            fh.write("\n")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--parsers", nargs="+", choices=PARSERS, default=list(PARSERS),
                    help="Первый из списка — база для сравнения (по умолчанию stanza).")
    ap.add_argument("--out-dir", default="results/parser_comparison",
                    help="Каталог для summary.txt, metrics.csv, debug_<parser>.txt, differences.txt.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = load_benchmark(args.benchmark)
    print(f"Загружено примеров: {len(benchmark)} из файла {args.benchmark}", file=sys.stderr)

    engines, dfs, metrics, seconds = {}, {}, {}, {}
    for name in args.parsers:
        print(f"[{name}] загрузка и прогон...", file=sys.stderr)
        engines[name] = build_engine("both", parser=name)
        start = time.perf_counter()
        dfs[name] = add_evaluation_columns(benchmark, engines[name], args.threshold)
        seconds[name] = time.perf_counter() - start
        metrics[name] = evaluate_markup(dfs[name])

        debug_path = out_dir / f"debug_{name}.txt"
        with contextlib.redirect_stdout(io.StringIO()):   # debug_report печатает шапку даже при n=0
            debug_report(dfs[name], n=0, file_name=str(debug_path))
        debug_path.write_text(f"# ЭКСПЕРИМЕНТ: {name} — {DESCRIPTIONS[name]}\n"
                              f"# бенчмарк: {args.benchmark}, порог: {args.threshold}\n\n"
                              + debug_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Совпадение считаем по очищенному тексту (без эталонной разметки [..]=label).
    base = args.parsers[0]
    clean_texts = dfs[base]["clean_text"]
    agreement = {name: parse_agreement(clean_texts, engines[base], engines[name])
                 for name in args.parsers[1:]}

    lines = build_summary(args, len(benchmark), engines, metrics, seconds, agreement)
    if len(dfs) > 1:
        n = write_diff_report(out_dir / "differences.txt", dfs)
        lines += ["", f"Строк с разными предсказаниями между экспериментами: {n} из {len(benchmark)} "
                      f"(см. differences.txt)"]
    text = "\n".join(lines)
    print(text)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    write_metrics_csv(out_dir / "metrics.csv", metrics)
    print(f"\nСохранено в {out_dir}/: summary.txt, metrics.csv, "
          + ", ".join(f"debug_{n}.txt" for n in args.parsers)
          + (", differences.txt" if len(dfs) > 1 else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
