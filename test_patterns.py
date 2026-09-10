# Модуль extract2
# цель модуля - извлечение данных по коннекторам и вводным словам по списку текстов с целью улучшения экстрактора коннекторов 

import stanza
import sys
from pathlib import Path
import json, re

from extract import (
    parse_args,
    load_patterns,
    build_default_checker,
    run,
    get_ext,

)


#функция слуджит для того, чтобы возвращать структуру args, заданную программно, а не через командную строку
def get_args(**kwargs):
    args = parse_args()
    
    # Обновляем атрибуты args значениями из kwargs
    for key, value in kwargs.items():
        setattr(args, key, value)

    #если input_csv задан, а output не задан, то зададим имя выходного файла
    #при этом проверим расширение, для csv добавим _result к имени, 
    # а для остальных расширений добавим .tsv к имени, чтобы получить результат в файле с табляциями
    if args.input_csv and not args.output:
        if get_ext(args.input_csv)=='.csv':
            args.output = args.input_csv[:-4]+'_result.csv' 
        else:
            args.output = args.input_csv+'.tsv' #файл с разделителями колонок табуляция

    
    return args



def print_patterns(patterns):
    for key in patterns:
        print(f"=== {key} ===")
        for i, tpl in enumerate(patterns[key], 1):
            # tpl[0], tpl[1], ... — кортежи строк
            parts = ["… ".join(" ".join(t) for t in tpl)]
            print(f"{i}. {parts[0]}")
        print()


def get_nested_linkers(filepath="nested_linkers.json"):
    """
    Считывает nested_linkers.json и возвращает структуру:
    список кортежей кортежей строк.

    Каждый элемент result — это:
      - для простого случая:  (("слово1", "слово2", ...),)
      - для сложного случая:  (("слово1", ...), ("словоN", ...))

    Сложный случай определяется по наличию многоточия '...' в ключe.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = []

    for lexeme, fields in data.items():
        # lexeme — это само словосочетание, например "а" или "а... в особенности"
        # fields нам не нужны, игнорируем

        # Разбиваем по многоточию: если есть '...', считаем сложным случаем
        if "..." in lexeme:
            parts = re.split(r"\s*\.\.\.\s*", lexeme)
            # parts — список строк, например ["а", "в особенности"]
            # каждую часть разбиваем на слова
            tuple1 = tuple(parts[0].split())
            tuple2 = tuple(parts[1].split()) if len(parts) > 1 else ()
            result.append((tuple1, tuple2))
        else:
            # Простой случай: одно словосочетание без '...'
            words = tuple(lexeme.split())
            result.append((words,))

    return result



# функция сравнивает список linker со списком intro или nested_linkers. 
# Проверяем совпадения, но если параметр inequality равен True - то проверяем различия
def test_patterns(patterns, nested_linkers=False, inequality=False):
    linker = patterns.get("linker", [])
    intro = patterns.get("intro", [])

    if nested_linkers:
        other = patterns.get("nested_linkers", [])
    else:
        other = intro

    def flatten(tpl):
        # tpl — кортеж кортежей строк → одна строка
        return "…  ".join(" ".join(t) for t in tpl)

    linker_flat = [flatten(tpl) for tpl in linker]
    other_flat = [flatten(tpl) for tpl in other]

    matches = []
    for i, l in enumerate(linker_flat, 1):
        for j, r in enumerate(other_flat, 1):
            if l == r:
                matches.append((i, j, l))

    name_other = "nested_linkers" if nested_linkers else "intro"

    if inequality:
        # Выводим элементы linker, не вошедшие в совпадения
        matched_linker_indices = {m[0] for m in matches}
        non_matched_linker = [
            (i, linker_flat[i - 1])
            for i in range(1, len(linker_flat) + 1)
            if i not in matched_linker_indices
        ]

        # Выводим элементы other, не вошедшие в совпадения
        matched_other_indices = {m[1] for m in matches}
        non_matched_other = [
            (j, other_flat[j - 1])
            for j in range(1, len(other_flat) + 1)
            if j not in matched_other_indices
        ]

        print(f"Элементы linker, не имеющие совпадений с {name_other}:")
        if non_matched_linker:
            for i, seq in non_matched_linker:
                print(f"linker[{i}]: {seq}")
        else:
            print("  (нет)")

        print(f"Элементы {name_other}, не имеющие совпадений с linker:")
        if non_matched_other:
            for j, seq in non_matched_other:
                print(f"{name_other}[{j}]: {seq}")
        else:
            print("  (нет)")
    else:
        # Обычный режим: вывод совпадений
        if not matches:
            print(f"Совпадений между linker и {name_other} не найдено.")
            return

        print(f"Найдены совпадения между linker и {name_other}:")
        for i, j, seq in matches:
            print(f"linker[{i}] == {name_other}[{j}]: {seq}")



def main():

    args = get_args()
    patterns = load_patterns(args.mode, args.linkers_csv, args.intro_csv)

    patterns["nested_linkers"] = get_nested_linkers("../nested_linkers.json")  

#   print_patterns(patterns)

    test_patterns(patterns)
#    test_patterns(patterns, nested_linkers=True, inequality=True) Выводим другое


    # а это варианты разметки текста
    # print("Loading stanza pipeline (tokenize,pos,lemma,depparse)...", file=sys.stderr)
    # nlp = stanza.Pipeline("ru", processors="tokenize,pos,lemma,depparse")
    # checker = build_default_checker()

    # run(get_args(input_csv = "src/texts.tsv"), patterns, nlp, checker)




if __name__ == "__main__":
    main()
