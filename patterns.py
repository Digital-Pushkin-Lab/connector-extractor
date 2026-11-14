"""Build matching patterns from the linker CSV word list.

Ported from the `linker_list_to_patterns` helper shared by both source
notebooks.
"""

from typing import List, Tuple

import pandas as pd
import re
from itertools import product

Part = Tuple[str, ...]
Pattern = Tuple[Part, ...]


def mult_strings(slots: List[List[str]]) -> List[str]:
    """Перемножает варианты слов по слотам и склеивает их через пробел.

    Пример:
        [['в', 'на'], ['самом'], ['деле', 'конце']]
        -> ['в самом деле', 'в самом конце', 'на самом деле', 'на самом конце']
    """
    if not slots:
        return []
    return [" ".join(word for word in combo if word) for combo in product(*slots)]

def expand_expressions(expr: str) -> List[str]:
    """
    Функция "разворачивает" строку: удаляет комментарий, делит по запятым, применяет правила. 
    Поделит "и так далее, и т. д." на ["и так далее", "и т. д."] 
    Затем части, поделённые по запятым, далее делятся по пробелам, а потом по слешам с учётом ковычек. 
    Результат перемножеается. Пример, строка: 'тогда/когда/сразу /же/"же ш"' 
    поделится на: "тогда", "тогда же", "тогда же ш", "когда", "когда же", "когда же ш", "сразу", "сразу же", "сразу же ш"
    Приеняется для загрузки линкеров и устойчивых выражений. 
    
    Возвращает список непустых фраз. Сам список может быть пустой.
    """
    # 1. Удаляем комментарий (всё после #, если он не внутри кавычек)
    clean_expr = re.sub(r'#(?=(?:[^"\']*["\'][^"\']*["\'])*[^"\']*$).*', '', expr).strip()
    if not clean_expr:
        return []

    # 2. Делим по запятым, игнорируя запятые внутри кавычек
    raw_items = re.split(r',(?=(?:[^"\']*["\'][^"\']*["\'])*[^"\']*$)', clean_expr)

    results: List[str] = []

    for item in raw_items:
        item = item.strip().lower()
        if not item:
            continue

        # Добавляем пробелы до и после многоточий, если они не внутри кавычек
        item = re.sub(r'\.\.\.(?=(?:[^"\']*["\'][^"\']*["\'])*[^"\']*$)', ' ... ', item)

        # 3. Разбиваем на токены-слоты по пробелам, сохраняя фрагменты в кавычках целиком.
        # Конструкция вида `тогда/когда/сразу же/"же ш"` или `"как будто"/будто` останется единым слотом,
        # так как регулярка захватит и текст в кавычках, и примыкающие к нему непробельные символы со слэшами.
        raw_slots = re.findall(r'(?:[^\s"\']+|"[^"]*"|\'[^\']*\')+', item)

        slots: List[List[str]] = []
        for slot in raw_slots:
            # 4. Бьём слот по "/" (учитывая, что "/" внутри кавычек не должен ломать фразу)
            alternatives = re.split(r'/(?=(?:[^"\']*["\'][^"\']*["\'])*[^"\']*$)', slot)

            cleaned_alts = [
                alt.strip().strip('"\'').strip()
                for alt in alternatives
            ]

            if cleaned_alts:
                slots.append(cleaned_alts)


        # 5. Перемножаем варианты слотов. Пустые результаты отфильтровываем и удаляем пробелы до и после многоточий, если они не в кавычках
        expanded = []
        for phrase in mult_strings(slots):
            phrase = phrase.replace("... ", "...").replace(" ...", "...").removesuffix("...").removesuffix("...")

            if phrase:
                expanded.append(phrase)
           
        results.extend(expanded)

    return results

def linker_list_to_patterns(raw_expressions: List[str]) -> List[Pattern]:
    """Turn raw expressions like 'если ... то' into part tuples, e.g.
    (('если',), ('то',)). Deduplicates while preserving first-seen order."""
    patterns: List[Pattern] = []

    for raw_expr in raw_expressions:
        for expr in expand_expressions(raw_expr):
            expr = expr.replace("…", "...")

            #AS Добавим пробел перед дефисом в списках паттернов для -то (есть "во-первых", "во-вторых" в них пробел вставлять не надо)
            expr = expr.replace("-то", " -то").replace("  ", " ")

            parts = []
            for part in expr.split("..."):
                words = tuple(part.strip().split())
                if words:
                    parts.append(words)

            if parts:
                patterns.append(tuple(parts))

    unique: List[Pattern] = []
    seen = set()
    for pattern in patterns:
        if pattern not in seen:
            seen.add(pattern)
            unique.append(pattern)

    return unique


def build_patterns_from_csv(csv_path: str, column: str = "linker") -> List[Pattern]:

    separator = "\t" if csv_path.endswith(".tsv") else ","

    df = pd.read_csv(csv_path, sep=separator)
    return linker_list_to_patterns(df[column].dropna().tolist())
