"""Build matching patterns from the linker CSV/TSV word list."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
import re
from typing import ClassVar, Dict, Iterable, List, Optional, Tuple, Union, overload
import pandas as pd

Part = Tuple[str, ...]


class FeatureVocab:
    """Словарь признаков со стабильной нумерацией битов (0, 1, 2, ...)."""

    def __init__(self) -> None:
        self.tag_to_bit: Dict[str, int] = {}
        self.bit_to_tag: Dict[int, str] = {}
        self.tag_to_idx: Dict[str, int] = {}

    def get_or_add(self, tag: str) -> int:
        if tag not in self.tag_to_bit:
            idx = len(self.tag_to_bit)
            bit = 1 << idx
            self.tag_to_bit[tag] = bit
            self.bit_to_tag[bit] = tag
            self.tag_to_idx[tag] = idx
        return self.tag_to_bit[tag]

    def encode(self, tags: Iterable[str]) -> int:
        mask = 0
        for tag in tags:
            mask |= self.get_or_add(tag)
        return mask

    def decode(self, mask: int) -> List[str]:
        return [tag for bit, tag in self.bit_to_tag.items() if mask & bit]


class _TagBitFlag:
    """Базовый дескриптор для доступа к битовой маске по строковому имени тега."""

    def __init__(self, tag: str, field_name: str, primary_only: bool = False):
        self.tag = tag
        self.field_name = field_name
        self.primary_only = primary_only
        self.bit: int = 0
        self.owner_cls = None

    def __set_name__(self, owner, name):
        self.owner_cls = owner
        # Регистрируем тег в словаре класса при его создании — порядок битов строго фиксируется!
        self.bit = owner.vocab.get_or_add(self.tag)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.field_name == "semfield":
            mask = instance.semfield1 if self.primary_only else (instance.semfield1 | instance.semfield2)
        else:
            mask = getattr(instance, self.field_name)
        return bool(mask & self.bit)


class SemFlag(_TagBitFlag):
    """Булев признак семантического поля (проверяет semfield1 | semfield2)."""

    def __init__(self, tag: str, primary_only: bool = False):
        super().__init__(tag=tag, field_name="semfield", primary_only=primary_only)


class PragFlag(_TagBitFlag):
    """Булев признак прагматики (проверяет pragmatics)."""

    def __init__(self, tag: str):
        super().__init__(tag=tag, field_name="pragmatics")


@dataclass(frozen=True)
class Pattern(Sequence):
    parts: Tuple[Part, ...]
    surface: str = ""
    semfield1: int = 0
    semfield2: int = 0
    pragmatics: int = 0

    vocab: ClassVar[FeatureVocab] = FeatureVocab()

    # ФИКСИРОВАННЫЕ ПРИЗНАКИ ДЛЯ ML И ПРАВИЛ (индексы 0..22 заданы жестко)
    # Если изменить порядок ниже, изменится нумерация признаков. 
    # В этой ситуации модель ML нужно переобучить, а правила менять не нужно,
    # правила обращаются к этим данным по имени признака, прммер:
    # if phrase.pattern.Time:

    # 1. Семантические поля (semfield1 / semfield2)
    Time = Время = SemFlag("время")
    Disjunction = Дизъюнкция = SemFlag("дизъюнкция")
    Substitution = Замещение = SemFlag("замещение")

    ImplicationCause = ИмпликацияПричина = SemFlag("импликация(причина)")
    ImplicationConsequence = ИмпликацияСледствие = SemFlag("импликация(следствие)")
    ImplicationCondition = ИмпликацияУсловие = SemFlag("импликация(условие)")
    ImplicationPurpose = ИмпликацияЦель = SemFlag("импликация(цель)")

    Paraphrase = Парафраз = SemFlag("парафраз")

    AdditionHighlight = ПрисоединениеВыделение = SemFlag("присоединение(выделение)")
    AdditionSupplement = ПрисоединениеДополнение = SemFlag("присоединение(дополнение)")
    AdditionIllustration = ПрисоединениеИллюстрация = SemFlag("присоединение(иллюстрация)")
    AdditionConjunction = ПрисоединениеКонъюнкция = SemFlag("присоединение(конъюнкция)")
    AdditionGeneralization = ПрисоединениеОбобщение = SemFlag("присоединение(обобщение)")
    AdditionSpecification = ПрисоединениеУточнение = SemFlag("присоединение(уточнение)")

    Opposition = Противление = SemFlag("противление")
    Comparison = Сравнение = SemFlag("сравнение")
    Means = Средство = SemFlag("средство")
    Degree = Степень = SemFlag("степень")
    Concession = Уступка = SemFlag("уступка")

    # 2. Прагматика (pragmatics)
    Accentuation = Акцентуализация = PragFlag("акцентуализация")
    Source = Источник = PragFlag("источник")
    Evaluation = Оценка = PragFlag("оценка")
    Epistemic = Эпистемика = PragFlag("эпистемика")

    def __init__(
        self,
        parts: Iterable[Iterable[str]],
        surface: str = "",
        semfield1: int = 0,
        semfield2: int = 0,
        pragmatics: int = 0,
    ):
        normalized = tuple(tuple(p) for p in parts)
        if not surface:
            surface = " ... ".join(" ".join(part) for part in normalized)

        # Pattern - это специальный класс (frozen=True), совместимый с кортежем. 
        # Поля класса заблокированы, поэтому здесь при инициализации поля присваиваются специальным способом (ниже)
        object.__setattr__(self, "parts", normalized)
        object.__setattr__(self, "surface", surface)
        object.__setattr__(self, "semfield1", semfield1)
        object.__setattr__(self, "semfield2", semfield2)
        object.__setattr__(self, "pragmatics", pragmatics)

    @classmethod
    def from_expr(
        cls,
        expr: str,
        semfield1: int = 0,
        semfield2: int = 0,
        pragmatics: int = 0,
    ) -> Optional["Pattern"]:
        """Создаёт объект Pattern из подготовленной строки выражения."""
        surface = expr.strip()
        normalized_expr = surface.replace("…", "...")
        normalized_expr = normalized_expr.replace("-то", " -то").replace("  ", " ")

        parts = [
            words
            for part in normalized_expr.split("...")
            if (words := tuple(part.strip().split()))
        ]
        if not parts:
            return None
        return cls(
            parts,
            surface=surface,
            semfield1=semfield1,
            semfield2=semfield2,
            pragmatics=pragmatics,
        )

    @property
    def all_semfields(self) -> int:
        return self.semfield1 | self.semfield2

    def to_ml_features(self) -> List[int]:
        """Возвращает детерминированный бинарный вектор признаков для ML-модели.
        Первые N бит — semfield1, затем semfield2, затем pragmatics.
        """
        n_tags = len(self.vocab.tag_to_bit)
        sf1_vec = [int(bool(self.semfield1 & (1 << i))) for i in range(n_tags)]
        sf2_vec = [int(bool(self.semfield2 & (1 << i))) for i in range(n_tags)]
        prag_vec = [int(bool(self.pragmatics & (1 << i))) for i in range(n_tags)]
        return sf1_vec + sf2_vec + prag_vec

    def get_semfield1_tags(self) -> List[str]:
        return self.vocab.decode(self.semfield1)

    def get_semfield2_tags(self) -> List[str]:
        return self.vocab.decode(self.semfield2)

    def get_pragmatics_tags(self) -> List[str]:
        return self.vocab.decode(self.pragmatics)

    @overload
    def __getitem__(self, index: int) -> Part: ...
    @overload
    def __getitem__(self, index: slice) -> Tuple[Part, ...]: ...

    def __getitem__(self, index: Union[int, slice]):
        return self.parts[index]

    def __len__(self) -> int:
        return len(self.parts)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Pattern):
            return self.parts == other.parts
        if isinstance(other, tuple):
            return self.parts == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.parts)



def mult_strings(slots: List[List[str]]) -> List[str]:
    """Перемножает варианты слов по слотам и склеивает их через пробел."""
    if not slots:
        return []
    return [" ".join(word for word in combo if word) for combo in product(*slots)]

def expand_expressions(expr: str) -> List[str]:
    """Разворачивает строку: удаляет комментарий, префикс до ':', делит по запятым, применяет правила."""
    outside_quotes = r'(?=(?:[^"\']*["\'][^"\']*["\'])*[^"\']*$)'
    slot_re = re.compile(r'(?:[^\s"\']+|"[^"]*"|\'[^\']*\')+')

    # 1. Удаляем комментарий (# вне кавычек)
    clean_expr = re.sub(r'#' + outside_quotes + r'.*', '', expr).strip()
    if not clean_expr:
        return []

    # 2. Отбрасываем префикс до первого двоеточия (вне кавычек), пример: "же: тогда/когда же" -> "тогда/когда же"
    # такая часть "же: " может применяться для удобной сортирвоки, чтобы связанные патерны были рядом
    colon_match = re.search(r':' + outside_quotes, clean_expr)
    if colon_match:
        clean_expr = clean_expr[colon_match.end():].strip()
        if not clean_expr:
            return []

    # 3. Делим по запятым вне кавычек
    raw_items = re.split(r',' + outside_quotes, clean_expr)
    results: List[str] = []

    for item in raw_items:
        item = item.strip().lower()
        if not item:
            continue

        # 4. Отделяем многоточия пробелами и разбиваем на слоты с учётом кавычек
        item = re.sub(r'\.\.\.' + outside_quotes, ' ... ', item)
        raw_slots = slot_re.findall(item)

        slots: List[List[str]] = []
        for slot in raw_slots:
            alternatives = re.split(r'/' + outside_quotes, slot)
            cleaned_alts = [alt.strip().strip('"\'').strip() for alt in alternatives]
            if cleaned_alts:
                slots.append(cleaned_alts)

        # 5. Перемножаем варианты слотов и убираем лишние пробелы вокруг многоточий
        for phrase in mult_strings(slots):
            phrase = (
                phrase.replace("... ", "...")
                .replace(" ...", "...")
                .removesuffix("...")
                .removesuffix("...")
            )
            if phrase:
                results.append(phrase)

    return results

def _parse_tags(cell: object) -> List[str]:
    """Разбирает ячейку с тегами, разделёнными ';' или ','."""
    if pd.isna(cell):
        return []
    text = str(cell).strip()
    if not text:
        return []
    return [tag.strip() for tag in re.split(r"[;,]", text) if tag.strip()]


def build_patterns_from_csv(
    csv_path: str,
    column: str = "linker",
    vocab: Optional[FeatureVocab] = None,
) -> List[Pattern]:
    """Загружает паттерны и их битовые признаки из CSV/TSV-файла."""
    separator = "\t" if csv_path.endswith(".tsv") else ","
    df = pd.read_csv(csv_path, sep=separator)

    target_vocab = vocab if vocab is not None else Pattern.vocab

    unique_patterns: List[Pattern] = []
    seen: set[Pattern] = set()

    for row in df.itertuples(index=False):
        raw_expr = getattr(row, column, None)
        if pd.isna(raw_expr):
            continue

        sf1_mask = target_vocab.encode(_parse_tags(getattr(row, "semfield1", None)))
        sf2_mask = target_vocab.encode(_parse_tags(getattr(row, "semfield2", None)))
        prag_mask = target_vocab.encode(_parse_tags(getattr(row, "pragmatics", None)))

        for expr in expand_expressions(str(raw_expr)):
            pattern = Pattern.from_expr(
                expr,
                semfield1=sf1_mask,
                semfield2=sf2_mask,
                pragmatics=prag_mask,
            )
            if pattern is None:
                continue

            if pattern in seen:
                print(f"Дубликат паттерна отброшен: {pattern.parts} (из строки '{raw_expr}')")
                continue

            seen.add(pattern)
            unique_patterns.append(pattern)

    return unique_patterns
