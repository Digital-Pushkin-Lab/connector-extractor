"""Rule-based scorer that estimates how likely a matched span is to actually
be functioning as a linker (vs. a coincidental word sequence).

Ported from `RuleBasedLinkerChecker` and its `rule_*` functions in
`linkers (1).ipynb`. Each rule inspects the matched token parts and votes
-1/0/+1; the average vote (clamped at 0) is the reported probability.
"""

from typing import Callable, Dict, List


def all_words(parts: Dict[str, list]) -> list:
    return [w for part in parts.values() for w in part]


def parts_list(parts: Dict[str, list]) -> list:
    return list(parts.values())


def rule_single_part_only_part(parts):
    words = all_words(parts)
    if len(words) == 1 and words[0]["upos"] == "PART":
        return -1
    return 0


def rule_sentence_initial(parts):
    for part in parts_list(parts):
        if any(w["id"] == 1 for w in part):
            return +1
    return 0


def rule_dependency_signal(parts):
    parts_vals = parts_list(parts)

    for w in all_words(parts):
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


def rule_no_space_after(parts):
    for w in all_words(parts):
        if w.get("misc", None):
            if "SpaceAfter=No" in w["misc"]:
                return +1
    return 0


def rule_pos_per_part(parts):
    for part in parts_list(parts):
        if not any(w["upos"] in {"SCONJ", "CCONJ", "ADV"} for w in part):
            return 0
    return +1


def rule_punct_before(parts):
    for part in parts_list(parts):
        if part[0].get("misc", None):
            if "PunctBefore=Yes" in part[0]["misc"]:
                return +1
    return 0


def rule_pron_after_adp(parts):
    for part in parts_list(parts):
        for i, w in enumerate(part):
            if w["id"] >= 2:
                if w["upos"] == "PRON" and part[i - 1]["upos"] == "ADP":
                    return +1
    return 0


def rule_my_head_is_next_tokens_id(parts):
    for part in parts_list(parts):
        for i, w in enumerate(part):
            if i < len(part) - 1:
                if w["head"] == part[i + 1]["id"]:
                    return +1
    return 0


def rule_introductory_potential(parts):
    """A trailing particle 'же' (up to the following comma), or a leading
    'а'/'и', is tolerated around an introductory construction."""
    for part in parts_list(parts):
        if part[0].get("misc", None):
            if "IntroductoryPotential=Yes" in part[0]["misc"]:
                return +1
        elif part[-1].get("misc", None):
            if "IntroductoryPotential=Yes" in part[-1]["misc"]:
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

def rule_pri_etom_dependent(parts):
    """«при этом» + зависимое справа → штраф; без зависимого → бонус."""
    words = all_words(parts)
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
    misc = last.get("misc") or ""

    has_dep = (
        "DependentAfter=Yes" in misc  # кто-то зависимый справа смотрит на нас  
        or "HeadIsFollowingNoun=Yes" in misc  # после нас существительное, от которого мы зависим: При этом императоре
    )

    if has_dep:
        return -1
    return +1


        

def rule_phrase_in_list(parts):
    """Правило даёт +1 балл, если оборот в списке. """

    # Это немного "читтерское" правило. Оно даёт оборотам +1 балл (0.2) что позволяет таким оборотам "сработать" при одном сработавшем обычном правиле
    # Смысл тут в том, что некоторые обороты являются всегда коннекторами




    words = all_words(parts)
    texts = tuple(w.get("text", "").lower() for w in words)  # tuple, не list

    #AS2 добавляем "однако" и другие
    triggers = """
    поэтому
    вследствие чего
    после чего
    однако
    с целью
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
    в той же, в том же, в тех же, в тот же, в ту же, в того же, в этот же, в этом же, в этой же, в этих же # ряд однотипных выражений. не все из них пока в коннекторах. Вероятно, надо смотреть лемму
    """



    if in_triggers(texts, triggers):
        return 1
    
    #AS2 новый блок +2
    triggers = """
    даже
    также
    затем
    тут же
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



def rule_conjunction_i(parts):
    """Отдельный союз «и, или, либо» → сильный бонус; иначе 0."""
    conjunctions = "и, или, либо"

    words = all_words(parts)
    if len(words) != 1:
        return 0

    w = words[0]
    if w.get("text", "").lower() not in conjunctions.split(", "):
        return 0

    # типичные признаки сочинительного союза у Stanza
    if w.get("upos") == "CCONJ" or w.get("deprel") in {"cc", "discourse", "fixed"}:
        return +2  # сдвиг p на 2/n (при n=5 → +0.4)core_entity
    return 0


def rule_special(parts):
    """Специфические контекстные правила для сложных одиночных коннекторов."""
    words = all_words(parts)
    if not words:
        return 0

    for w in words:
        word_text = w.get("text", "").lower()
        w_id = w.get("id", 1)

        
       # =================================================================
        # БЛОК 1: ЧАСТИЦА "ЖЕ"
        # =================================================================
        if word_text == "же":
            prev = w.prev()

            if prev is not None:
                prev_token = prev.ts

                if prev_token.words:
                    prev_word = prev_token.words[0]
                    prev_lemma = prev_word.lemma.lower() if prev_word.lemma else ""
                    prev_upos = prev_word.upos

                    prev_prev = prev.prev()
                    prev_prev_token = (
                        prev_prev.ts if prev_prev is not None else None
                    )

                    # Контекстный перехват "В том же..." до блока штрафов.
                    if prev_lemma in {"тот", "том", "тех", "то"}:
                        if (
                            prev_prev_token is not None
                            and prev_prev_token.text.lower() == "в"
                        ):
                            return +2

                    # 1. Жесткие текстовые исключения.
                    if prev_lemma in {
                        "тот", "такой", "так", "один", "то",
                        "тут", "тот же", "такой же", "факт",
                    }:
                        return -3

                    # 2. Главный критерий линкера: стоит строго
                    # на 2, 3 или 4 месте после имени/местоимения.
                    if prev_upos in {"NOUN", "PRON", "PROPN", "DET"} or prev_lemma == "сам":
                        if w_id <= 4:
                            return +2

                    # 3. Подстраховка для БСП: перед предыдущим
                    # словом стоит знак препинания.
                    if prev_prev_token is not None and prev_prev_token.words:
                        punct_word = prev_prev_token.words[0]
                        if (
                            punct_word.upos == "PUNCT"
                            and prev_prev_token.text in {":", "—", ";", ","}
                        ):
                            if prev_upos in {"NOUN", "PRON", "PROPN", "DET"}:
                                return +2

            return 0

    return 0



def rule_time_connector(parts):
    """Единое правило валидации синтаксического контекста для временных 
    дейктических слов (тогда, теперь, сейчас, впоследствии, ныне, затем и др.)
    """
    words = all_words(parts)
    if not words:
        return 0

    time_deictic_lemmas = {
        "тогда", "теперь", #"сейчас", "впоследствии",   "ныне", "нынче", "затем", "потом", "прежде", "раньше"
    }
    

    # 1. Находим конкретный токен дейктика внутри списка слов спана
    target_word = None
    for w in words:
        if w.get("lemma", "").lower() in time_deictic_lemmas:
            target_word = w
            break
            
    if not target_word:
        return 0

    if words[0].tokens[0].text == "Если":
        print(tokens2sent(words[0].tokens))
        pass    

    # 2. Безусловный бонус для известных устойчивых сочетаний
    surface_lower = " ".join(w.get("text", "").lower() for w in words)
    if surface_lower in {"тогда же", "теперь же", "сейчас же", "сразу же"}:
        return +1

    # Безопасно извлекаем базовые свойства найденного токена
    misc = target_word.get("misc") or ""
    w_id = target_word.get("id", 1)
    deprel = target_word.get("deprel", "")
    upos = target_word.get("upos", "")
    
    w_idx = w_id - 1  # 0-based индекс в tokens

    # -----------------------------------------------------------------
    # ПРИОРИТЕТНЫЙ БЛОК: ЖЕЛЕЗНЫЕ ПРИЗНАКИ НАЧАЛА (Решает стр. 388)
    # -----------------------------------------------------------------
    if w_id == 1:
        return +2

    tokens = words[0].tokens # пока так
    # -----------------------------------------------------------------
    # СИНТАКСИЧЕСКИЙ АНАЛИЗ КОНТЕКСТА ЧЕРЕЗ TOKENS
    # -----------------------------------------------------------------
    if 0 <= w_idx < len(tokens):
        
        # ИСПРАВЛЕНО: берём первый элемент списка words -> words[0]
        if w_idx > 0 and tokens[w_idx - 1].ts.words:
            prev_word = tokens[w_idx - 1].ts.words[0]
            if prev_word.lemma and prev_word.lemma.lower() == "то":
                return +2

        # ИСПРАВЛЕНО: берём первый элемент списка words -> words[0]
        head_id = target_word.get("head", 0)
        head_idx = head_id - 1
        if 0 <= head_idx < len(tokens) and tokens[head_idx].ts.words:
            head_word = tokens[head_idx].ts.words[0]
            if head_word.feats and "VerbForm=Part" in str(head_word.feats):
                return -3  # Сильный штраф гарантированно превратит случай в [OK] как other

        # Исправление стр. 413: Вытягиваем БСП (поиск знака препинания двоеточия/тире слева)
        for i in range(w_idx - 1, -1, -1):
            if tokens[i].ts.words and tokens[i].ts.words[0].upos == "PUNCT":
                punct_text = tokens[i].text
                if punct_text in {":", "—", ";"}:
                    return +2  # Это начало новой части в БСП, даем весомый бонус
                if punct_text in {".", "!", "?"}:
                    break  # Граница предложения, останавливаемся

    # -----------------------------------------------------------------
    # СТАНДАРТНЫЕ ФОЛБЕКИ (для обратной совместимости)
    # -----------------------------------------------------------------
    if upos == "ADJ" or "HeadIsFollowingNoun=Yes" in misc or "DependentAfter=Yes" in misc:
        return -1

    if w_id == 2 or "IntroductoryPotential=Yes" in misc or "PunctBefore=Yes" in misc:
        return +1

    if deprel in {"parataxis", "discourse", "cc", "mark", "fixed"}:
        return +1

    if w_id > 4 and deprel not in {"parataxis", "discourse", "cc", "mark", "fixed"}:
        return -1

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


    #AS Метод доработан. 1. Добавлены экстра--правила 2. Добавлены отладочные сообщения    
    def score_entity(self, entity: dict, debug: bool = False) -> float:
        """entity: {"surface": ..., "parts": {part1: [...], ...}}"""
        n = len(self.rules)
        votes = []
        for rule in self.rules:
            v = rule(entity["parts"])
            votes.append((rule.__name__, v))

        score = sum(v for _, v in votes)
        p = score / n if n else 0.0

        extra_votes = []
        step = 1.0 / n if n else 1.0
        for rule in getattr(self, "extra_rules", []):
            v = rule(entity["parts"]) 
            extra_votes.append((rule.__name__, v))
            if v != 0:
                p += v * step

        p = max(0.0, min(1.0, p))

        if debug:
            print(f"  surface={entity.get('surface')!r}  p={p:.3f}")
            for name, v in votes:
                print(f"    {name}: {v:+d}")
            for name, v in extra_votes:
                print(f"    {name} [extra]: {v:+d}")

        return p

    def score_sentence(self, sentence_json: dict, tokens, debug: bool = False) -> List[dict]:
        results = []
        if debug:
            print(sentence_json.get('sentence'))

        for entity in sentence_json.get("entities", []):
            results.append({
                "linker": entity["surface"],
                "probability": self.score_entity(entity, debug=debug),
            })
        return results


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
        misc = str(token.ts.misc) if token.ts.misc else ""
        
        result.append(text)
        
        # Базовая проверка Stanza на отсутствие пробела ПОСЛЕ текущего токена
        if "SpaceAfter=No" in misc:
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
