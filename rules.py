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


class RuleBasedLinkerChecker:
    def __init__(self):
        self.rules: List[Callable[[Dict[str, list]], int]] = []

    def add_rule(self, rule_fn):
        self.rules.append(rule_fn)

    def score_entity(self, entity: dict) -> float:
        """entity: {"surface": ..., "parts": {part1: [...], ...}}"""
        score = sum(rule(entity["parts"]) for rule in self.rules)
        p = score / len(self.rules) if self.rules else 0
        return max(0.0, p)

    def score_sentence(self, sentence_json: dict) -> List[dict]:
        """sentence_json: output of matching.sentence_to_json"""
        results = []
        for entity in sentence_json.get("entities", []):
            results.append({
                "linker": entity["surface"],
                "probability": self.score_entity(entity),
            })
        return results


def build_default_checker() -> RuleBasedLinkerChecker:
    """The rule set actually used to score the real-data runs in the
    notebook (see cell enabling/disabling rules before `get_stats`)."""
    checker = RuleBasedLinkerChecker()
    checker.add_rule(rule_sentence_initial)
    checker.add_rule(rule_dependency_signal)
    checker.add_rule(rule_no_space_after)
    checker.add_rule(rule_punct_before)
    checker.add_rule(rule_introductory_potential)
    return checker
