#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
markup.py

Модуль создаёт датасеты старого типа (функция  make_old_dataset) и нового типа (класс NewDataset)

Основаня идея разметки - берём сырые тексты, оборачиваем найденные паттерны в квадратные скобки и ставим метку
Пример разметки предложения:
[Без сомненья]=linker, [если]=linker1 ты придёшь, [то]=linker2 я буду рад, [но]=linker [если]=linker нет — тоже хорошо.

Основные отличия нового датасета (NewDataset) от старого:
Он умеет "дописывать" файл с примером, продолжая его. 
Сортирвока файла осуществлена по паттерну, а не по номеру примера, как раньше.
Программа
- извлекает из текста примеры, содержащие один паттерн (пример или контрпример), не считая паттернов "а", "и", "но";
- обрабатывает нескоько файлов из папки, несколько папок;
- учуитывает разметку Википедии.
... и другое.


Модуль исходно был создан по аналогии с gradio_app.py, так как тоже запускает разметку из ядра программы, а потом её интерпретирует.
"""

import os, sys, re
import pandas as pd
from pathlib import Path

from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# Пути к extractor (точно как в gradio_app.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ""))

import stanza
from extract import load_patterns
from pipeline import extract_spans, parse_sentences
from rules import build_default_checker

LINKERS_CSV = os.path.join(os.path.dirname(__file__),  "data", "linkers.tsv")


print("Loading stanza pipeline (tokenize,pos,lemma,depparse)...", file=sys.stderr)

def get_stanza():
    try:
        st = stanza.Pipeline('ru', processors="tokenize,pos,lemma,depparse", download_method=None, logging_level='ERROR')
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

#NLP = stanza.Pipeline("ru", processors="tokenize,pos,lemma,depparse")

CHECKER = build_default_checker()
PATTERNS_BY_TYPE = load_patterns(LINKERS_CSV)



# БЛОК РАЗМЕТКИ. СДЕЛАНО ПО АНАЛОГИИ С gradio_app.py 

def dedupe_spans(spans: List[dict]) -> List[dict]:
    """Оставляет только лучший (по probability) span при полном совпадении границ."""
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

from collections import defaultdict #для сортирвоки


def mark_text(text: str) -> Tuple[str, List[dict]]:
    """
    Размечает предложение и возвращает кортеж:
    - размеченное предложение
    - список найденных паттернов

    В список паттернов входят как принятые, так и отклонённые span'ы.
    """
    if not text or not text.strip():
        return text, []

    parsed_sentences, _ = parse_sentences(text, NLP)

    spans, rejected = extract_spans(
        parsed_sentences,
        CHECKER,
        PATTERNS_BY_TYPE,
        return_rejected=True
    )

    spans = dedupe_spans(spans)
    rejected = dedupe_spans(rejected)


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
            "type": pattern_type,
            "start": sp["start"],
            "end": sp["end"],
            "accepted": True,
        })

    # отклонённые паттерны тоже считаем паттернами
    for sp in rejected:
        patterns.append({
            "surface": sp["surface"],
            "type": sp.get("type", "other"),
            "start": sp["start"],
            "end": sp["end"],
            "accepted": False,
        })

    # сортируем паттерны по их позиции в исходном тексте
    patterns.sort(key=lambda p: (p["start"], p["end"]))

    if not spans and not rejected:
        return text, patterns

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

    # 2. Составные — части нумеруются с 1
    for key, group in groups.items():
        group_sorted = sorted(group, key=lambda x: x["start"])

        for i, sp in enumerate(group_sorted):
            cat = f"{sp['type']}{i + 1}"
            events.append((sp["start"], True, sp["end"], cat))
            events.append((sp["end"], False, sp["start"], cat))

    # 3. Отклонённые
    for sp in rejected:
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

    return marked_line, patterns




def make_old_dataset(input_file_name: str, output_file_name: str) -> None:
    """Читает файл → размечает → сохраняет."""
    with open(input_file_name, "r", encoding="utf-8") as f:
        text = f.read()

    marked_lines = []
    num=0
    max_num= 500
    marked = "num\ttext\n"
    for line in text.splitlines():
        if '[' in line or ']' in line:
            continue # просто проигнорируем любые строки с символами []
        marked_line = mark_text(line)[0]



        cleaned = marked_line.lower().replace('[и]', '').replace('[а]', '').replace('[но]', '') #уберём банальное из проверки
        if '[' in cleaned:  # добавим в датасет только предложения с разметкой,
            marked_lines.append(marked_line)
            num+=1
            print(num, marked_line)
            marked += f"{num}\t{marked_line}\n"

            if num>=max_num:
                break

#    marked = "\n".join(marked_lines)

    with open(output_file_name, "w", encoding="utf-8") as f:
        f.write(marked)

    print(f"Готово: {output_file_name}", file=sys.stderr)


import csv

# функция использовалася джля конвертации линкеров из csv в tsv
def csv2tsv(input_path: str = "data/linkers.csv", output_path: str = "data/linkers.tsv") -> None:
    """Читает CSV-файл и сохраняет его в формат TSV."""
    with open(input_path, mode="r", encoding="utf-8") as infile, \
         open(output_path, mode="w", encoding="utf-8", newline="") as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile, delimiter="\t")
        
        for row in reader:
            writer.writerow(row)

### НОВАЯ РАЗМЕТКА #####


import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
from nltk.tokenize import sent_tokenize
import string

class NewDataset:
    DATASET_COLUMNS = [
        "num",
        "main_pattern",
        "text",
        "comment50",
        "comment",
    ]
    WORK_COLUMNS = DATASET_COLUMNS + ["type"]
    END_PUNCTUATION = set(".!?…:;»\"'”’)]")    

    # эти символы мы вырезаем перед проверкой похожести предложений
    PUNCTUATION_AND_SYMBOLS = string.punctuation + "«»“”„…—–−·•◦▪▫●○◆◇■□★☆➤➤➔→←↑↓"

    MARKUP_RE = re.compile(
        r"\[[^\]]*\](?:=+[A-Za-zА-Яа-я0-9_]+)*"
    )

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.df = self.load()

    # ------------------------------------------------------------------
    # Загрузка и сохранение
    # ------------------------------------------------------------------


    def load(self) -> pd.DataFrame:
        """
        Загружает датасет из TSV.

        Если файла нет — создаёт пустой датасет.
        Если файл есть, но не читается — вызывает ошибку.
        """
        if not self.dataset_path.exists():
            print(
                f"Датасет {self.dataset_path} не найден, создаём новый.",
                file=sys.stderr
            )
            df = pd.DataFrame(columns=self.DATASET_COLUMNS)
            # внутренняя колонка для сортировки
            df["type"] = ""            
        else:
            try:
                df = pd.read_csv(
                    self.dataset_path,
                    sep="\t",
                    encoding="utf-8",
                    dtype={"num": "Int64"}
                )
            except Exception as e:
                raise RuntimeError(
                    f"Не удалось прочитать существующий датасет: "
                    f"{self.dataset_path}"
                ) from e

            for col in self.DATASET_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.NA

            df = df[self.DATASET_COLUMNS]

            # внутренняя колонка для сортировки;
            # в TSV она не сохраняется
            df["type"] = ""

            if "comment50" not in df.columns:
                df["comment50"] = ""

            df["comment50"] = (
                df["comment50"]
                .fillna("")
                .astype(str)
                .str.slice(0, 50)
            )

            print(
                f"Датасет загружен: {self.dataset_path}, строк: {len(df)}",
                file=sys.stderr
            )
        
      
        # пересчитываем main_pattern и type
        # перебором содержательных строк
        for idx, row in df.iterrows():
            text = row["text"]

            if not isinstance(text, str):
                continue

            text = text.strip()

            if not text or text.startswith("#") or text.startswith("!"):
                continue

            main_pattern, pattern_type = (
                self.parse_main_pattern_from_text(text)
            )

            df.at[idx, "main_pattern"] = main_pattern
            df.at[idx, "type"] = pattern_type



        return df[self.WORK_COLUMNS]



    def save(self) -> None:
        """
        Сохраняет датасет в TSV.

        Перед сохранением добавляет строки-заглушки
        для уникальных паттернов, затем сортирует таблицу:
        1. по main_pattern; пустой main_pattern идёт первым
        2. внутри одного main_pattern — по type;
           пустой type идёт первым

        Колонка type используется только внутри программы
        и в файл не записывается.
        """
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

        # добавляем комментарий к каждому паттерну, если ещё не добавлен
        self.add_pattern_stub_rows()

        # пустые значения должны быть строками,
        # иначе NaN при сортировке уходит в конец
        self.df["main_pattern"] = self.df["main_pattern"].fillna("").astype(str)
        self.df["type"] = self.df["type"].fillna("").astype(str)

        # сортируем по паттерну и типу (тип внутренняя колонка содердит одно из трёх ["","neg", "pos"], то есть сначала будут комментарии к паттерну, потом контрпримеры, потом примеры)
        self.df = self.df.sort_values(
            by=["main_pattern", "type"],
            kind="stable"
        )

        self.df.to_csv(
            self.dataset_path,
            sep="\t",
            index=False,
            encoding="utf-8",
            columns=self.DATASET_COLUMNS
        )

        print(
            f"Готово: {self.dataset_path}, "
            f"строк в датасете: {len(self.df)}",
            file=sys.stderr
        )

    # ------------------------------------------------------------------
    # Вспомогательные функции
    # ------------------------------------------------------------------

    @staticmethod
    def strip_markup(marked_text: str) -> str:
        """
        Удаляет разметку вида:
        [что-то]=linker
        [что-то]=linker0
        [что-то]==other
        [что-то]=other=linker
        """
        cleaned = NewDataset.MARKUP_RE.sub("", marked_text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
    
    @staticmethod
    def clean_md_line(line: str) -> str:
        """
        Очищает строку Markdown перед разметкой.
        """
        if line.lstrip().startswith("#"):
            return ""

        # удаляем маркер списка "- "
        line = re.sub(r"^\s*-\s+", "", line)

        # удаляем нумерацию в начале строки: "2) ", "12) " и т. п.
        line = re.sub(r"^\s*\d+\)\s+", "", line)

        # удаляем сноски и ссылки вида [3]
        line = re.sub(r"\[\d+\]", "", line)

        # удаляем знаки ударения
        line = re.sub(r"[\u0301]", "", line)
        line = line.replace("́", "")

        return line.strip()

    @classmethod
    def ends_with_punctuation(cls, text: str) -> bool:
        """
        Проверяет, заканчивается ли текст на знак препинания.
        """
        if not text:
            return False

        return text.rstrip().endswith(tuple(cls.END_PUNCTUATION))
    
    
    def get_next_num(self) -> int:
        """
        Возвращает следующий номер для колонки num.
        """
        if self.df.empty or self.df["num"].isna().all():
            return 1

        return int(self.df["num"].max()) + 1

    def get_main_pattern(self, patterns: List[dict]):
        """
        Возвращает единственный основной паттерн
        или None, если после отбрасывания 'а', 'и', 'но'
        паттернов не ровно один.
        """
        filtered = [
            p for p in patterns
            if p["surface"].strip().lower() not in {"а", "и","но"}
        ]

        if len(filtered) != 1:
            return None

        return filtered[0]

    @classmethod
    def normalize_for_dedupe(cls, text: str) -> str:
        """
        Приводит текст к виду для поиска дублей:

        - нижний регистр
        - удаление пунктуации
        - удаление распространённой псевдографики
        - замена ё на е
        - удаление цифр
        - схлопывание повторных пробелов
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = text.replace("ё", "е")

        # удаляем пунктуацию и псевдографику
        text = text.translate(
            str.maketrans("", "", cls.PUNCTUATION_AND_SYMBOLS)
        )

        # удаляем цифры
        text = re.sub(r"\d+", " ", text)

        # убираем любые другие не-буквенные/не-цифровые символы,
        # кроме пробелов
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

        # схлопываем пробелы
        text = re.sub(r"\s+", " ", text).strip()

        return text
    
    def stub_row_exists(self, pattern: str) -> bool:
        """
        Проверяет, есть ли уже строка-заглушка для паттерна.

        Заглушка:
        - num пустой
        - main_pattern равен pattern
        - type пустой или NaN
        - text пустой или начинается с '!' или '#'
        - comment50 и comment могут быть любыми
        """
        pattern = str(pattern).strip()

        num_is_empty = self.df["num"].isna()

        main_pattern_is_equal = (
            self.df["main_pattern"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq(pattern)
        )

        type_is_empty = (
            self.df["type"].isna()
            | self.df["type"]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
        )

        text_series = self.df["text"].fillna("").astype(str).str.strip()

        text_is_comment_or_empty = (
            text_series.eq("")
            | text_series.str.startswith("!")
            | text_series.str.startswith("#")
        )

        mask = (
            num_is_empty
            & main_pattern_is_equal
            & type_is_empty
            & text_is_comment_or_empty
        )

        return bool(mask.any())


    def add_pattern_stub_rows(self) -> None:
        """
        Добавляет строки-заглушки для уникальных паттернов,
        если такой заглушки ещё нет.
        """
        patterns = (
            self.df["main_pattern"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        patterns = sorted(patterns.unique())

        new_rows = []

        for pattern in patterns:
            if self.stub_row_exists(pattern):
                continue

            new_rows.append({
                "num": pd.NA,
                "main_pattern": pattern,
                "type": "",
                "text": "# ",
                "comment50": "",
                "comment": "",
            })

        if new_rows:
            self.df = pd.concat(
                [
                    self.df,
                    pd.DataFrame(new_rows, columns=self.WORK_COLUMNS)
                ],
                ignore_index=True
            )

            print(
                f"Добавлено строк-заглушек: {len(new_rows)}",
                file=sys.stderr
            )


    def parse_patterns_from_text(self, marked_text: str) -> List[Tuple[str, str]]:
        """
        Извлекает из размеченного текста все паттерны и их метки.

        Составные паттерны собираются обратно:
        [как]=linker1 ... [так и]=linker2
        превращается в ("как...так и", "linker").

        Паттерны приводятся к нижнему регистру.

        Возвращает список кортежей:
        (surface, last_label)
        """
        if not isinstance(marked_text, str):
            return []

        text = marked_text.strip()

        if not text:
            return []

        # один и два знака равенства означают одно и то же
        text = text.replace("==", "=")

        matches = re.findall(
            r"\[([^\]]+)\]((?:=[A-Za-zА-Яа-я0-9_]+)*)",
            text
        )

        raw_patterns = []

        for surface, labels_part in matches:
            surface_clean = surface.strip().lower()

            labels = [
                label.strip().lower()
                for label in labels_part.split("=")
                if label.strip()
            ]

            last_label = labels[-1] if labels else ""

            raw_patterns.append((surface_clean, last_label))

        patterns = []
        i = 0

        while i < len(raw_patterns):
            surface, label = raw_patterns[i]

            # начало составного паттерна
            if label == "linker1":
                parts = [surface]
                expected_part = 2

                i += 1

                # ищем linker2, linker3, ...
                while i < len(raw_patterns):
                    next_surface, next_label = raw_patterns[i]

                    # linker1 финализирует текущий составной паттерн
                    if next_label == "linker1":
                        break

                    # нужная следующая часть составного паттерна
                    if next_label == f"linker{expected_part}":
                        parts.append(next_surface)
                        expected_part += 1
                        i += 1
                        continue

                    # обычный linker, other, intro и т. п.
                    # не разрывают составной паттерн
                    i += 1

                composite_surface = "...".join(parts)

                patterns.append((composite_surface, "linker"))
                continue

            patterns.append((surface, label))
            i += 1

        return patterns



    def parse_main_pattern_from_text(self, marked_text: str):
        """
        Определяет основной паттерн и тип примера.

        Сначала получает все паттерны через parse_patterns_from_text(),
        затем:
        - пропускает паттерны 'а', 'и', 'но'
        - берёт первый оставшийся содержательный паттерн
        - определяет type по его метке

        Возвращает кортеж:
        (main_pattern, type)
        """
        patterns = self.parse_patterns_from_text(marked_text)

        for surface, label in patterns:
            # пропускаем банальные паттерны
            if surface in {"а", "и", "но"}:
                continue

            main_pattern = surface.replace(",", "")

            if label == "linker":
                pattern_type = "pos"
            elif label == "other":
                pattern_type = "neg"
            else:
                pattern_type = ""

            return main_pattern, pattern_type

        return "", ""



    # ------------------------------------------------------------------
    # Пополнение датасета
    # ------------------------------------------------------------------

    def make_new_dataset(
        self,
        source_dir: str,
        max_num: int = 50
    ) -> pd.DataFrame:
        """
        Дополняет датасет до max_num строк.

        max_num — желаемое общее число строк в датасете.
        Уже существующие строки не изменяются и не удаляются.
        """
        source_dir = Path(source_dir)

        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"Папка-первоисточник не найдена: {source_dir}"
            )

        # посчитаем содержательные примеры
        text_series = self.df["text"].fillna("").astype(str).str.strip()
        mask = text_series.ne("") & ~text_series.str.startswith("#") & ~text_series.str.startswith("!")
        num_examples =  int(mask.sum())

        remaining = max_num - num_examples

        if remaining <= 0:
            print(
                f"Датасет уже содержит {num_examples} примеров; "
                f"цель {max_num} достигнута.",
                file=sys.stderr
            )
            return self.df

        md_files = sorted(source_dir.glob("*.md"))

        if not md_files:
            print(
                f"В папке {source_dir} нет .md файлов",
                file=sys.stderr
            )
            return self.df

        next_num = self.get_next_num()

        existing_texts = set(
            self.df["text"]
            .dropna()
            .astype(str)
            .apply(self.strip_markup)
            .apply(self.normalize_for_dedupe)
        )


        # количество уже имеющихся примеров по паттерну
        pattern_counts = (
            self.df["main_pattern"]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
            .to_dict()
        )

        # количество уже имеющихся примеров
        # по комбинации паттерн + тип
        pattern_type_counts = (
            self.df[["main_pattern", "type"]]
            .dropna()
            .assign(
                main_pattern=lambda x: x["main_pattern"].astype(str).str.strip(),
                type=lambda x: x["type"].astype(str).str.strip()
            )
            .value_counts()
            .to_dict()
        )


        new_rows = []

        # цикл по файлам
        for md_file in md_files:
            if remaining <= 0:
                break

            print(f"Обрабатываем файл: {md_file}", file=sys.stderr)

            with open(md_file, "r", encoding="utf-8") as f:
                file_text = f.read()

            # цикл по строкам
            for line in file_text.splitlines():
                if remaining <= 0:
                    break

                line = self.clean_md_line(line)

                if not line:
                    continue

                if "[" in line or "]" in line:
                    continue

                # цикл по предложениям
                for sentence in sent_tokenize(line, language="russian"):
                    if remaining <= 0:
                        break

                    if not sentence.strip() or not self.ends_with_punctuation(sentence):
                        continue

                    # Здесь мы проверяем, что предложение удовлетворяет условию по длине и отсутствует в датасете
                    if len(sentence) < 30 or len(sentence) > 150:
                        continue

                    normalized_text = self.normalize_for_dedupe(sentence)

                    if normalized_text in existing_texts:
                        continue

                    (
                        marked_line,
                        patterns
                    ) = mark_text(sentence)



                    main_pattern = self.get_main_pattern(patterns)

                    if main_pattern is None:
                        continue

                    pattern_surface = main_pattern["surface"].strip()

                    if main_pattern["accepted"]:
                        example_type = "pos"
                    else:
                        example_type = "neg"

                    pattern_key = pattern_surface
                    pattern_type_key = (pattern_surface, example_type)

                    current_pattern_count = pattern_counts.get(pattern_key, 0)
                    current_type_count = pattern_type_counts.get(pattern_type_key, 0)

                    # Примеров одного вида не должно быть слишком много
                    # обычный лимит: не более 7 примеров одного паттерна
                    if current_pattern_count >= 7:
                        # исключение: если такой паттерн+тип пока встречается менее 3 раз, пример всё равно добавляем                        
                        if current_type_count >= 3:
                            continue

                    # обновляем счётчики
                    pattern_counts[pattern_key] = current_pattern_count + 1
                    pattern_type_counts[pattern_type_key] = (current_type_count + 1)


                    # добавляем предложение в массив проверок 
                    existing_texts.add(normalized_text)

                    # добавляем предложение в финальную строку
                    new_rows.append({
                        "num": next_num,
                        "main_pattern": pattern_surface,
                        "type": example_type,
                        "text": marked_line,
                    })

                    # увеличиваем счётчики
                    next_num += 1
                    remaining -= 1

                    print(
                        next_num - 1,
                        pattern_surface,
                        example_type,
                        marked_line
                    )
        
        # закончился цикл по файлам
        if new_rows:
            self.df = pd.concat(
                [
                    self.df,
                    pd.DataFrame(new_rows, columns=self.WORK_COLUMNS)
                ],
                ignore_index=True
            )

        print(
            f"Добавлено строк из {source_dir}: {len(new_rows)}; "
            f"теперь примеров в датасете: {num_examples + len(new_rows)};",
            f"всего строк с комментариями: {len(self.df)}",
            file=sys.stderr
        )

        return self.df



def main():

    # # Так формируем старый датасет 
    # input_file_name  = "src/books.tsv"
    # output_file_name = "src/books_result.tsv"
    # make_old_dataset(input_file_name, output_file_name)



    #  Так формирвем датасет нового типа
    source_dir = "../../../Sources/wiki.ru/"

    dataset = NewDataset("src/wiki.tsv")

    # dataset.make_new_dataset(source_dir + "Категория_Астрономия", max_num=300 )
    # dataset.make_new_dataset(source_dir + "Категория_Архитектура", max_num=600 )

    dataset.save()

if __name__ == "__main__":
    main()
   
