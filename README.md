# Извлечение линкеров/вводных слов

Извлекает из текста русскоязычные линкеры и/или вводные слова (интро-слова)
с помощью синтаксического парсинга stanza и скоринга на основе правил. Код
очищен и собран воедино на основе `linkers (1).ipynb` и
`connectors within linkers.ipynb`.

## Установка

```bash
pip install -r requirements.txt
python -c "import stanza; stanza.download('ru')"   # разовая загрузка модели
```

## Файлы

- `data/linkers.csv` — словарь линкеров (из `linkers.csv`).
- `data/intro_words.csv` — словарь вводных слов (из
  `Коннекторы и вводные слова - вводные конструкции все.csv`).
- `patterns.py` — превращает словари из CSV в паттерны для сопоставления,
  помеченные типом источника (`linker`/`intro`).
- `matching.py` — сопоставляет паттерны с токенами предложений, разобранных
  stanza.
- `rules.py` — скорер на основе правил (`RuleBasedLinkerChecker`), который
  оценивает, насколько найденный фрагмент похож на настоящий линкер/вводное
  слово, а не на случайное совпадение.
- `pipeline.py` — связывает парсинг, сопоставление и скоринг воедино, а также
  считает итоговую статистику по каждому типу.
- `extract.py` — точка входа для запуска из командной строки.

## Использование

```bash
# Одно предложение, только линкеры, JSON выводится в stdout
python extract.py --mode linkers --text "Более того, к Швеции отошли города Ивангород и Копорье."

# Текстовый файл, только вводные слова
python extract.py --mode intro --input-file article.txt --output result.json

# Линкеры и вводные слова за один проход
python extract.py --mode both --text "Если ты придёшь, то я буду рад."

# Пакетный режим: анализ каждой строки колонки в CSV, запись CSV с добавленными колонками статистики
python extract.py --mode both --input-csv texts.csv --text-column text --output results.csv
```

`--mode` определяет, какой словарь (или словари) используется для сопоставления с текстом:
- `linkers` — сопоставление только с `data/linkers.csv`.
- `intro` — сопоставление только с `data/intro_words.csv`.
- `both` — сопоставление с обоими словарями за один проход; каждое найденное
  совпадение помечается тем, из какого словаря (или словарей) оно взято
  (выражение, присутствующее в обоих списках, попадает в статистику обоих
  типов).

Прочие параметры:
- `--threshold` (по умолчанию `0.4`) — минимальная оценённая вероятность,
  при которой совпадение считается достоверным.
- `--linkers-csv` / `--intro-csv` — указать альтернативные словари.

## Формат вывода

Результат — это набор колонок (для `--input-csv` они добавляются к исходному
CSV, для `--text`/`--input-file` — тот же набор полей выводится как JSON).
Показатели "на 100 слов" считаются от числа не-пунктуационных токенов в
тексте.

При `--mode linkers`:

| Колонка | Описание |
|---|---|
| `linkers_result` | сырые результаты по каждому предложению: список найденных линкеров с оценённой вероятностью |
| `unique_linker_count` | количество уникальных линкеров (с вероятностью ≥ `--threshold`) |
| `linker_count` | общее количество найденных линкеров (с повторами) |
| `unique_linkers` | список уникальных линкеров |
| `linkers_by_appearance` | список найденных линкеров по каждому предложению |
| `linkers_per_100` | `linker_count`, нормализованный на 100 слов |
| `unique_linkers_per_100` | `unique_linker_count`, нормализованный на 100 слов |

При `--mode intro` — та же структура, но для вводных слов:
`intro_result`, `unique_intro_count`, `intro_count`, `unique_intro_words`,
`intro_words_by_appearance`, `intro_per_100`, `unique_intro_per_100`.

При `--mode both` колонки объединяются в следующем порядке:

```
linkers_result, unique_linker_count, linker_count, unique_linkers,
linkers_by_appearance, linkers_per_100, unique_linkers_per_100,
intro_result, unique_intro_count, intro_count, unique_intro_words,
intro_words_by_appearance, intro_per_100, linker_and_intro_per_100,
unique_intro_per_100
```

`linker_and_intro_per_100` — суммарное количество линкеров и вводных слов,
нормализованное на 100 слов (доступно только при `--mode both`).
