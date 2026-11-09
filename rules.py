"""Rule-based scorer that estimates how likely a matched span is to actually
be functioning as a linker (vs. a coincidental word sequence).

Ported from `RuleBasedLinkerChecker` and its `rule_*` functions in
`linkers (1).ipynb`. Each rule inspects the matched token parts and votes
-1/0/+1; the average vote (clamped at 0) is the reported probability.
"""

from typing import Callable, Dict, List


from pyroaring import BitMap


class BitFlag:
    """Булев атрибут, хранящий значение в bitmap."""

    def __init__(self, feature_id: int):
        self.feature_id = feature_id

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return self.feature_id in instance._bits

    def __set__(self, instance, value: bool):
        if not isinstance(value, bool):
            raise TypeError("Значение признака должно быть bool")

        if value:
            instance._bits.add(self.feature_id)
        else:
            instance._bits.discard(self.feature_id)


class TokenFeatures:
    """Бинарные признаки токена с доступом через атрибуты."""

    __slots__ = ("_bits",)

    # ID признаков должны оставаться постоянными.
    Is_PunctBefore = ЕстьПунктуацияПеред = BitFlag(0)
    Is_IntroductoryPotential = ПотенциальноВводное = BitFlag(1)
    Is_NoSpaceAfter = НетПробелаПосле = BitFlag(2)
    Is_DependentAfter = ЕстьЗависимоеПосле = BitFlag(3)
    Is_HeadFollowingNoun = ГлавноеСуществительноеСправа = BitFlag(4)

    def __init__(self):
        self._bits = BitMap()



class Token:
    # Известные флаги, пока оставим для истории.
    # _MISC_FEATURES = {
    #     "PunctBefore=Yes": "Is_PunctBefore",
    #     "IntroductoryPotential=Yes": "Is_IntroductoryPotential",
    #     "SpaceAfter=No": "Is_NoSpaceAfter",
    #     "DependentAfter=Yes": "Is_DependentAfter",
    #     "HeadIsFollowingNoun=Yes": "Is_HeadFollowingNoun",
    # }

    def __init__(self, token_stanza, tokens):
        self.st = token_stanza    # Stanza или Stanza Token (для Spacy Token, впрочем, будет sp)
        self.std_ = token_stanza.to_dict()
        self.std = self.std_[0]   # Stanza Dict - станза-токен, превращённый словарь

        self.text = token_stanza.text
        self.s = self.text.lower()   #часто нужно в нижем регистре, возможно тут требуется дополнительно отчистить от, например, знаков ударения
        self.tokens = tokens         # ключевой момент. Токены исходного предложения с типом Token

        # Вычисляются вместе при первом обращении.
        self.info = None # здесь были признаки токена. Оставили для совместимости. Фактически здесь теперь self.std
        self._f = None  # это признаки токена. Они зависят от соседей, поэтомы мы их вычислить в конструкторе не можем

    @property
    def f(self) -> TokenFeatures:
        """Признаки токена с ленивым вычислением."""
        if self._f is None:
            self.assemble_token_info()
        return self._f

    def index(self):
        return self.tokens.index(self)

    @property
    def prev(self):
        i = self.index()
        if i == 0:
            return None
        return self.tokens[i - 1]

    @property
    def next(self):
        i = self.index()
        if i == len(self.tokens) - 1:
            return None
        return self.tokens[i + 1]

    @property
    def lemma(self):
        return self.st.words[0].lemma if self.st.words else None

    @property
    def upos(self):
        return self.st.words[0].upos if self.st.words else None

    @property
    def id(self): #возможно назвать свойство st_id (то есть stanza.id)
        return self.st.words[0].id if self.st.words else None # возможно сделать 0 или 1 

    @property
    def head(self):
        return self.st.words[0].head if self.st.words else None # возможно сделать 0 или 1 
    
    @property
    def deprel(self): 
        return self.st.words[0].deprel if self.st.words else None
    
    def __repr__(self):
        return (
            f'Token({self.id!r}, {self.text!r}, '
            f'{self.lemma!r}, '
            f'{self.upos!r}, '
            f'{self.head!r}, '
            f'{self.deprel!r})'
        )
    
    def __str__(self):
        return self.__repr__()

    # Для совместимости с существующим кодом.
    def to_dict(self):
        return self.std_

    def get_info(self) -> dict:
        return self.assemble_token_info()

    def get(self, key, default=None):
        return self.get_info().get(key, default)

    def __getitem__(self, key):
        return self.get_info()[key]

    def assemble_token_info(self) -> dict:
        """Возвращает словарь Stanza и вычисляет признаки токена.

        Кэширует:
        - self.info — копию исходного словаря Stanza;
        - self._f — объект TokenFeatures с bitmap внутри.

        misc не читается и не изменяется.
        """
        if self.info is not None:
            return self.info

        token_position = self.index()
        tokens = self.tokens

        basic_info_dict = self.std.copy()
        features = TokenFeatures()

        prev_token = self.prev
        next_token = self.next

        # Условие сохранено из исходного класса: позиция >= 2.
        if token_position >= 2 and prev_token.upos == "PUNCT":
            features.Is_PunctBefore = True

        if token_position >= 1 and token_position < len(tokens) - 2:
            prev_lemma = prev_token.lemma
            next_lemma = next_token.lemma
            next_next_upos = next_token.next.upos

            if (
                prev_lemma in {"а", "и"}
                or (next_lemma == "же" and next_next_upos == "PUNCT")
            ):
                features.Is_IntroductoryPotential = True

        if next_token is None or next_token.upos == "PUNCT":
            features.Is_NoSpaceAfter = True

        this_id = basic_info_dict.get("id")
        head_id = basic_info_dict.get("head", 0)

        # Есть зависимый токен справа, не считая пунктуации.
        for j in range(token_position + 1, len(tokens)):
            std = tokens[j].std

            if std.get("upos") == "PUNCT":
                continue

            if this_id is not None and std.get("head") == this_id:
                features.Is_DependentAfter = True
                break

        # Главное слово справа — существительное или имя собственное.
        for j in range(token_position + 1, len(tokens)):
            std = tokens[j].std

            if (
                head_id
                and std.get("id") == head_id
                and std.get("upos") in {"NOUN", "PROPN"}
            ):
                features.Is_HeadFollowingNoun = True
                break

        self._f = features
        self.info = basic_info_dict

        return self.info



class Phrase:
    def __init__(self, parts, spans, pattern):
        self.parts = parts
        self.spans = spans
        self.pattern = pattern
        self.surface = "...".join(" ".join(part) for part in pattern)
        self.words = [w for part in parts.values() for w in part]  #токены фразы
        self.sent_tokens = self.words[0].tokens #токены предложения

    def parts_list(self) -> list:
        return list(self.parts.values())

    def get(self, key, default=None):
        return getattr(self, key, default)
    

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)








# не используется
# def rule_single_part_only_part(phrase):
#     words = phrase.words
#     if len(words) == 1 and words[0].upos == "PART":
#         return -1
#     return 0


def rule_sentence_initial(phrase):
    for part in phrase.parts_list():
        if any(w["id"] == 1 for w in part):
            return +1
    return 0


def rule_dependency_signal(phrase):
    parts_vals = phrase.parts_list()

    for w in phrase.words:
        if w["deprel"] in {"parataxis", "discourse", "cc", "mark", "fixed"}:
            return +1

    if len(parts_vals) > 1:
        ok = True
        for part in parts_vals:
            if not any(w["deprel"] in {"cc", "mark", "fixed"} for w in part):
                ok = False
                break
        if ok:
            return +1

    return 0


def rule_no_space_after(phrase):
    for w in phrase.words:
        if w.f.Is_NoSpaceAfter:
            return +1
    return 0


def rule_pos_per_part(phrase):
    for part in phrase.parts_list():
        if not any(w.upos in {"SCONJ", "CCONJ", "ADV"} for w in part):
            return 0
    return +1


def rule_punct_before(phrase):
    for part in phrase.parts_list():
        if part and part[0].f.Is_PunctBefore:
            return +1
    return 0


def rule_pron_after_adp(phrase):
    for part in phrase.parts_list():
        for i, w in enumerate(part):
            if w["id"] >= 2:
                if w["upos"] == "PRON" and part[i - 1]["upos"] == "ADP":
                    return +1
    return 0


# Не используется
# def rule_my_head_is_next_tokens_id(phrase):
#     for part in parts_list(phrase.parts):
#         for i, w in enumerate(part):
#             if i < len(part) - 1:
#                 if w["head"] == part[i + 1]["id"]:
#                     return +1
#     return 0


def rule_introductory_potential(phrase):
    """A trailing particle 'же' (up to the following comma), or a leading
    'а'/'и', is tolerated around an introductory construction."""
    for part in phrase.parts_list():
        if not part:
            continue

        if (
            part[0].f.Is_IntroductoryPotential
            or part[-1].f.Is_IntroductoryPotential
        ):
            return +1

    return 0
         


# используем также в markup.py
def in_triggers(words, triggers):
    """Проверяет string на попадание в triggers.

    Триггеры разделяются переводами строк или запятыми.
    Всё после # до конца строки считается комментарием.
    """
    if isinstance(words, str):    
        words = tuple(words.split())


    triggers_set = {
        tuple(trigger.split())
        for line in triggers.splitlines()
        for trigger in line.split("#", 1)[0].split(",")
        if trigger.strip()
    }
    return words in triggers_set



#AS Добавляем новое правило
# правило согласования "При этом". Отрабатывает конструкции типа "При этом", "В результате", "Как првило" и т.п. 
# зависимое слово сильно роняет вероятность ("При этом императоре"). Отсутствие зависимого слова сильно повышает вероятность. 

def rule_pri_etom_dependent(phrase):
    """«при этом» + зависимое справа → штраф; без зависимого → бонус."""
    words = phrase.words
    texts = tuple(w.get("text", "").lower() for w in words)  # tuple, не list

    #AS2 добавили: как 
    triggers = """
    при этом
    в результате, в итоге,
    как, как правило
    в то же время
    к тому же
    впоследствии
    между тем
    лучше
    тем временем
    с другой #  бывает "с другой стороны" (отдельный паттерн), а бывает "с другой — " и тут надо проверить отсутствие связи, чтобы исключить фразы типа "с другой книгой"
    
#    же # В Германии же books256 - линкер (нет связи) такие же books269- не линкер (связь)
    """

    if not in_triggers(texts, triggers):
        return 0

    last = words[-1]

    has_dep = last.f.Is_DependentAfter or last.f.Is_HeadFollowingNoun

    if has_dep:
        return -1
    return +1

        

def rule_phrase_in_list(phrase):
    """Правило даёт +1 балл, если оборот в списке. """

    # Это немного "читтерское" правило. Оно даёт оборотам +1 балл (0.2) что позволяет таким оборотам "сработать" при одном сработавшем обычном правиле
    # Смысл тут в том, что некоторые обороты являются всегда коннекторами




    words = phrase.words
    texts = tuple(w.get("text", "").lower() for w in words)  # tuple, не list

    #AS2 добавляем "однако" и другие
    triggers = """
    поэтому
    вследствие чего
    после чего
    однако
    фактически
    например
    тогда же
    сразу же
    именно
    в противном случае
    благодаря
    в это же время
    как будто
    едва
    потом
    всё чаще
    в том числе, в том числе и
    впредь
    в дальнейшем
    интересно
    же
    в тот же
    тогда как
#    наряду с  # надо разбираться со случаями употребления
    """



    if in_triggers(texts, triggers):
        return 1
    
    #AS2 новый блок +2
    triggers = """
    даже
    также
    затем
    с целью
    на сей раз
    вот # см. books446
    """
    if in_triggers(texts, triggers):
        return 2
    

    #AS2 Добавляем блок. Эти слова штрафуем
    triggers = """
    что
    """
    if in_triggers(texts, triggers):
        return -2

    return 0



def rule_conjunction_i(phrase):
    """Отдельный союз «и, или, либо» → сильный бонус; иначе 0."""
    conjunctions = "и, или, либо"

    words = phrase.words
    if len(words) != 1:
        return 0

    word = words[0]
    if word.s not in conjunctions.split(", "):
        return 0

    # типичные признаки сочинительного союза у Stanza
    if word.upos == "CCONJ" or word.get("deprel") in {"cc", "discourse", "fixed"}:
        return +2  # сдвиг p на 2/n (при n=5 → +0.4)core_entity
    return 0


def rule_special(phrase):
    """Специфические контекстные правила для сложных одиночных коннекторов."""
    words = phrase.words
    if not words:
        return 0

    string =  " ".join(w.get("text", "").lower() for w in words)


    if len(words)==1 and words[0].s=="же":
        w = words[0]
        w_id = w.get("id", 1)

        prev = w.prev

        if prev:

            # Жесткие текстовые исключения с добавлением лемм "том" и "тех"
            if prev.lemma in {
                "тот", "том", "тех", "такой", "так", "один", "то",
                "тут", "тот же", "такой же", "факт",
            }:
                return -3

            if prev.upos in {"NOUN", "PRON", "PROPN", "DET"} or prev.lemma == "сам":
                if w_id <= 4:
                    return +2


            if prev.prev and prev.prev.text in [":", "—", ";", ","]:
                if prev.upos in {"NOUN", "PRON", "PROPN", "DET"}:
                        return +2

            return 0

    return 0

def rule_time_connector(phrase):
    """Единое правило валидации синтаксического контекста для временных 
    дейктических слов (тогда, теперь, сейчас, впоследствии, ныне, затем и др.)
    """
    words = phrase.words
    if not words:
        return 0

    time_deictic_lemmas = {
        "тогда", "теперь", #"сейчас", "впоследствии",   "ныне", "нынче", "затем", "потом", "прежде", "раньше"
    }
    

    # 1. Находим конкретный токен дейктика внутри списка слов спана
    word = None
    for w in words:
        if w.lemma in time_deictic_lemmas:
            word = w
            break
            
    if not word:
        return 0

    if words[0].text == "Если": # первый токен предложения
#        print(tokens2sent(words[0].tokens))
        pass    

    # 2. Безусловный бонус для известных устойчивых сочетаний
    surface_lower = " ".join(w.s for w in words)
    if surface_lower in {"тогда же", "теперь же", "сейчас же", "сразу же"}:
        return +1

    # Безопасно извлекаем базовые свойства найденного токена
    w_id = word.id
    # deprel = word.deprel
    # upos = word.upos
    

    w_idx = w_id - 1  # 0-based индекс в tokens

    # ПРИОРИТЕТНЫЙ БЛОК: ЖЕЛЕЗНЫЕ ПРИЗНАКИ НАЧАЛА (Решает стр. 388)
    if w_id == 1:
        return +2

    tokens = words[0].tokens # пока так

    # СИНТАКСИЧЕСКИЙ АНАЛИЗ КОНТЕКСТА 
    if 0 <= w_idx < len(tokens):
        
        if word.prev and word.prev.lemma == "то":
            return +2

        # ИСПРАВЛЕНО: берём первый элемент списка words -> words[0]
        head_id = word.get("head", 0)
        head_idx = head_id - 1
        if 0 <= head_idx < len(tokens) and tokens[head_idx].st.words:
            head_word = tokens[head_idx].st.words[0]
            if head_word.feats and "VerbForm=Part" in str(head_word.feats):
                return -3  # Сильный штраф гарантированно превратит случай в [OK] как other

        # Исправление стр. 413: Вытягиваем БСП (поиск знака препинания двоеточия/тире слева)
        for i in range(w_idx - 1, -1, -1):
            if tokens[i].upos == "PUNCT":
                punct_text = tokens[i].text
                if punct_text in {":", "—", ";"}:
                    return +2  # Это начало новой части в БСП, даем весомый бонус
                if punct_text in {".", "!", "?"}:
                    break  # Граница предложения, останавливаемся

    # # СТАНДАРТНЫЕ ФОЛБЕКИ (для обратной совместимости)
    if w_id == 2 or word.f.Is_IntroductoryPotential or word.f.Is_PunctBefore:
        return +1
  
    return 0

class RuleBasedLinkerChecker:
    def __init__(self):
        self.rules: List[Callable[[Dict[str, list]], int]] = []
        self.extra_rules: List[Callable[[Dict[str, list]], int]] = []  #AS Добавляем экстра правила, они не попадают в знаменатель дроби

    def add_rule(self, rule_fn):
        self.rules.append(rule_fn)

    def add_extra_rule(self, rule_fn):  #AS экстра-парвила
        """Правило, которое при 0 не влияет на probability."""
        self.extra_rules.append(rule_fn)


    #AS Метод доработан. 1. Добавлены экстра-правила 2. Добавлены отладочные сообщения    
    def score_entity(self, phrase, debug: bool = False) -> float:
        """entity: {"surface": ..., "parts": {part1: [...], ...}}"""
        n = len(self.rules)
        votes = []
        for rule in self.rules:
            v = rule(phrase)
            votes.append((rule.__name__, v))

        score = sum(v for _, v in votes)
        p = score / n if n else 0.0

        extra_votes = []
        step = 1.0 / n if n else 1.0
        for rule in getattr(self, "extra_rules", []):
            v = rule(phrase) 
            extra_votes.append((rule.__name__, v))
            if v != 0:
                p += v * step

        p = max(0.0, min(1.0, p))

        if debug:
            print(f"  surface={phrase.surface!r}  p={p:.3f}")
            for name, v in votes:
                print(f"    {name}: {v:+d}")
            for name, v in extra_votes:
                print(f"    {name} [extra]: {v:+d}")

        return p

    def score_sentence(self, sentence_json: dict, debug: bool = False) -> List[dict]:
        results = []
        if debug:
            print(sentence_json.get('sentence'))

        phrases = sentence_json.get("entities", []);
        for phrase in phrases:
            phrase.probability = self.score_entity(phrase, debug=debug)

        return phrases


def build_default_checker() -> RuleBasedLinkerChecker:
    """The rule set actually used to score the real-data runs in the
    notebook (see cell enabling/disabling rules before `get_stats`)."""
    checker = RuleBasedLinkerChecker()
    checker.add_rule(rule_sentence_initial) #+1 если оборот в начале предложения
    checker.add_rule(rule_dependency_signal) #+1 если есть маркеры "parataxis", "discourse", "cc", "mark", "fixed"
    checker.add_rule(rule_no_space_after) #+1 если пунктуация после оборота
    checker.add_rule(rule_punct_before) #+1 если пунктуация перед оборотом
    checker.add_rule(rule_introductory_potential) # "а" или "и" перед оборотом, либо "же," после оборота


    checker.add_extra_rule(rule_pri_etom_dependent)  # AS Добавляем экстра-правило. Также см комментарий к функции rule_pri_etom_dependent
    checker.add_extra_rule(rule_conjunction_i)# AS Добавляем экстра-правило для и
    checker.add_extra_rule(rule_phrase_in_list)# AS Добавляем правило, усиливающее оценку конкретных оборотов на +1 (0.2)
    checker.add_extra_rule(rule_special)
    checker.add_extra_rule(rule_time_connector)     

    return checker


# Функция возвращает текст по списку тоуенов Stanza. Создана исключительно для упрощения отладки
def tokens2sent(tokens) -> str:
    """Восстанавливает исходный текст предложения из списка объектов-токенов Stanza
    с корректной расстановкой пробелов вокруг знаков препинания и кавычек.
    """
    if not tokens:
        return ""
        
    result = []
    for i, token in enumerate(tokens):
        text = token.text
        # Безопасно приводим misc к строке, даже если там None или dict
        misc = str(token.st.misc) if token.st.misc else ""
        
        result.append(text)
        

        # Базовая проверка Stanza на отсутствие пробела ПОСЛЕ текущего токена
        if token.f.Is_NoSpaceAfter:
            continue        
            
        # Умная лингвистическая подстраховка для знаков препинания:
        # Если следующий токен — это запятая, точка, двоеточие, закрывающая кавычка или скобка,
        # пробел перед ними ставить НЕ нужно.
        if i + 1 < len(tokens):
            next_text = tokens[i + 1].text
            if next_text in {",", ".", "!", "?", ":", ";", ")", "]", "»"}:
                continue
            # Если текущий токен — открывающая кавычка или скобка, после неё пробел не нужен
            if text in {"(", "[", "«"}:
                continue
                
        result.append(" ")
        
    return "".join(result).strip()
