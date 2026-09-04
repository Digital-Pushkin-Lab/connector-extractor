"""Rule-based scorer that estimates how likely a matched span is to actually
be functioning as a linker/introductory word (vs. a coincidental word sequence).

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


#AS Добавляем новое правило
# правило согласования "При этом". Отрабатывает конструкции типа "При этом", "В результате", "Как првило" и т.п. 
# зависимое слово сильно роняет вероятность ("При этом императоре"). Отсутствие зависимого слова сильно повышает вероятность. 

def rule_pri_etom_dependent(parts):
    """«при этом» + зависимое справа → штраф; без зависимого → бонус."""
    words = all_words(parts)
    texts = tuple(w.get("text", "").lower() for w in words)  # tuple, не list

    triggers = """
    при этом
    в результате
    как правило
    в то же время
    """

    triggers_set = {
        tuple(line.split())
        for line in triggers.splitlines()
        if line.strip()
    }

    if texts not in triggers_set:
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

    triggers = """
    затем
    поэтому
    вследствие чего
    после чего
    """

    triggers_set = {
        tuple(line.split())
        for line in triggers.splitlines()
        if line.strip()
    }

    if texts not in triggers_set:
        return 0

    return 1



def rule_conjunction_i(parts):
    """Отдельное «и»-союз → сильный бонус; иначе 0 (не влияет на знаменатель)."""
    words = all_words(parts)
    if len(words) != 1:
        return 0

    w = words[0]
    if w.get("text", "").lower() != "и":
        return 0

    # типичные признаки сочинительного союза у Stanza
    if w.get("upos") == "CCONJ" or w.get("deprel") in {"cc", "discourse", "fixed"}:
        return +2   # сдвиг p на 2/n (при n=5 → +0.4)
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


    #AS Старый метод
    def score_entity_(self, entity: dict) -> float:
        """entity: {"surface": ..., "parts": {part1: [...], ...}}"""
        n = len(self.rules) #AS Добавляем n
        score = sum(rule(entity["parts"]) for rule in self.rules) if n else 0
        p = score / n if n else 0.0

        #AS Применяем экстра-правмила extra: 0 игнорируем; ±1 сдвигает p на один «шаг» обычного правила
        step = 1.0 / n if n else 1.0
        for rule in self.extra_rules:
            v = rule(entity["parts"])
            if v != 0:
                p += v * step    
        #AS

        return max(0.0, p)

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

    def score_sentence(self, sentence_json: dict, debug: bool = False) -> List[dict]:
        results = []
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

    return checker
