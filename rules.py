"""Rule-based scorer that estimates how likely a matched span is to actually
be functioning as a linker (vs. a coincidental word sequence).

Ported from `RuleBasedLinkerChecker` and its `rule_*` functions in
`linkers (1).ipynb`. Each rule inspects the matched token parts and votes
-1/0/+1; the average vote (clamped at 0) is the reported probability.
"""

from typing import Callable, Dict, List
from pipeline import Engine

from pyroaring import BitMap
from pipeline import Engine, load_patterns

class BitFlag:
    """Булев атрибут, хранящий значение в bitmap с автоматической проверкой типа."""

    def __init__(self, feature_id: int):
        self.feature_id = feature_id
        self.owner_cls = None  # Определится автоматически чуть позже

    def __set_name__(self, owner, name):
        # Python автоматически вызовет этот метод и передаст сюда класс-владелец
        self.owner_cls = owner

    def _validate(self, instance):
        if self.owner_cls and not isinstance(instance, self.owner_cls):
            raise TypeError(
                f"Признак с ID {self.feature_id} принадлежит классу {self.owner_cls.__name__}, "
                f"но был вызван для объекта {type(instance).__name__}"
            )
 
    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        self._validate(instance) # проверяем тип
        return self.feature_id in instance._bits
    
    def __set__(self, instance, value: bool):
        if not isinstance(value, bool):
            raise TypeError("Значение признака должно быть bool")
        self._validate(instance)  # Проверяем тип перед записью

        if value:
            instance._bits.add(self.feature_id)
        else:
            instance._bits.discard(self.feature_id)


# class TokenFeatures:
#     """Бинарные признаки токена с доступом через атрибуты."""

#     __slots__ = ("_bits",)

#     # ID признаков должны оставаться постоянными.
# #    Is_PunctBefore = ЕстьПунктуацияПеред = BitFlag(0)
#     Is_IntroductoryPotential = ПотенциальноВводное = BitFlag(1)
#     Is_NoSpaceAfter = НетПробелаПосле = BitFlag(2)
#     Is_DependentAfter = ЕстьЗависимоеПосле = BitFlag(3)
#     Is_HeadFollowingNoun = ГлавноеСуществительноеСправа = BitFlag(4)

#     def __init__(self):
#         self._bits = BitMap()



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
        self.st = token_stanza    # Stanza или Stanza Token (для Spacy Token будет sp)
        st_dict = token_stanza.to_dict()
        self.info =st_dict[0]   # Первое слово из токена Stanza превращённого в словарь. Для доступа к инфорамции Stanaza через интерфейс словаря

        self.text = token_stanza.text
        self.s = self.text.lower()   #часто нужно в нижем регистре, возможно тут требуется дополнительно отчистить от, например, знаков ударения
        self.tokens = tokens         # ключевой момент. Токены исходного предложения с типом Token


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
        return self.st.words[0].id if self.st.words else 0 # такого id не бывает и он конвертируется в False

    @property
    def head_id(self):
        return self.st.words[0].head if self.st.words else 0 # такого id не бывает и он конвертируется в False

    @property
    def head(self):
        """Возвращает родительский объект Token, используя относительное смещение IDs."""
        if not self.head_id or not self.id or not self.tokens:
            return None
        
        # наша последовательность self.tokens должна содержать одно предложение, но не обязательно.
        # Для нажёжности считаем относительную, а не абсолютную адресацию
        offset = self.head_id - self.id         # Вычисляем относительное смещение в нумерации ID Stanza
        target_idx = self.index() + offset      # Находим целевой индекс в списке tokens предложения
        
        if 0 <= target_idx < len(self.tokens):
            return self.tokens[target_idx]
        return None
    
    @property
    def deprel(self): 
        return self.st.words[0].deprel if self.st.words else None

    @property
    def feats(self) -> str:
        """Возвращает морфологические признаки токена в виде строки."""
        if self.st.words and self.st.words[0].feats:
            return str(self.st.words[0].feats)
        return ""
    
    def __repr__(self):
        """Выводит представление токена в отладку"""
        # Выводим (lemma) только если s и lemma не совпадают
        # .s - всегда в нижнем регистре, а .lemma может быть в верхнем, если это имя, поэтому переведём её в нижний
        lemma_str = f"({self.lemma!r})" if self.s != self.lemma.lower() else ""  
        
        return (
            f"Token({self.text!r}{lemma_str}="
            f"{self.upos},"     
            f"{self.deprel}, "     
            f"id={self.id!r},"
            f"h={self.head_id!r})"
        )
        
    def __str__(self):
        return self.__repr__()

    def get(self, key, default=None):
        return self.info.get(key, default)

    def __getitem__(self, key):
        return self.info[key]


class Phrase:
    """Универсальный класс для найденной фразы (возможно, разрывной)."""

    def __init__(self, parts, spans, pattern, engine=None):
        self.engine = engine
        self.parts = parts
        self.spans = spans
        self.pattern = pattern
        self.surface = "...".join(" ".join(part) for part in pattern)
        self.words = [word for part in parts for word in part]  # токены фразы
        self.sent_tokens = self.words[0].tokens if self.words else []  # токены предложения
        self.verbose = False # для отладки. Если установить флаг, то в правилах выведется большое отладочное сообщение

        # Хранилище бинарных признаков для дескрипторов BitFlag в наследниках
        self._bits = set()

        # Хук для вычисления признаков в классах-наследниках
        self._calc_features()

    def _calc_features(self):
        """Переопределяется в наследниках для вычисления предметных признаков."""
        pass

    @property
    def first(self):
        """Первый токен фразы."""
        return self.words[0] if self.words else None

    @property
    def last(self):
        """Последний токен фразы."""
        return self.words[-1] if self.words else None

    @property
    def prev(self):
        """Токен, идущий строго перед фразой."""
        return self.first.prev if self.first else None

    @property
    def next(self):
        """Токен, идущий строго после фразы."""
        return self.last.next if self.last else None

    def find_lemma(self, lemmas):
        """Ищет первый токен во фразе, лемма которого входит в указанный список/множество лемм.

        Принимает: строку (одну лемму) или коллекцию лемм (set, list, tuple).
        Возвращает: Объект Token или None.
        """
        lemmas = {lemmas} if isinstance(lemmas, str) else set(lemmas)
        return next((w for w in self.words if w.lemma in lemmas), None)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.surface}')"

class PhraseC(Phrase):
    """Фраза-коннектор с набором бинарных синтаксических и пунктуационных признаков."""
    # PhraseC - PhraseConnector - предназначен для системы распознавания коннекторов, 
    # хранит не только коннекторы, но и отклонённые элементы и любые нужные фразы

    # Чтобы self._bits заполнился автоматически, мы должны здесь перечислить эти значения
    StartPosition = СтартоваяПозиция = BitFlag(0)
    EndPosition = КонечнаяПозиция = BitFlag(1)

    SentenceInitial = НачалоПредложения = BitFlag(2)

    Parataxis = ВводнаяКонструкция = BitFlag(3) # Отношение между главным предложением и соседней независимой конструкцией. Сюда относятся вводные слова, пояснения или прямая речь (например, «как мне кажется», «сказал он»).
    Discourse = ДискурсивныйМаркер = BitFlag(4) # Элементы речи, которые не несут синтаксической нагрузки, но важны для структуры диалога или выражения эмоций. Это междометия, частицы, маркеры ответа и привлечения внимания (например, «ну», «ого», «ведь»).
    CC = Coordinating = СочинительныйCоюз = BitFlag(5)  # Отношение между элементом предложения и сочинительным союзом, который его связывает (например, союзы «и», «а», «но»).
    Mark = Marker  = ПодчинительныйМаркер  = BitFlag(6) # Отношение между придаточным предложением и подчинительным союзом (или инфинитивной частицей), который его вводит (например, «чтобы», «если», «потому что»).
    Fixed  = ФиксированноеВыражение = BitFlag(7) # Используется для связи слов внутри застывших, неразложимых грамматических конструкций и многословных союзов. Первое слово становится главным, а остальные привязываются к нему через fixed (например, «в то время как», «в связи с»).    

    Parenthetical = Поясняющий = BitFlag(8) # Вводные элементы: вводные слова (parataxis) и частицы (discourse) 
    Connective = Связующий = BitFlag(9)  # Связующие элементы cc mark fixed

    Structural = Структурынй = BitFlag(10)  # Структурные элементы: есть вводные слова или связующие элементы

    Complex  = Комплексный = BitFlag(11) # Комплексный паттерн, состоящий из нескольких слов
    ConnectiveInEachPart  = СвязующийВКаждойЧасти = BitFlag(12) # Все части сложного выражения содержат связующие элементы

    Len1  = Длина1 = BitFlag(13) # Длина фразы 1 токен

    PunctAfter = ПунктуацияПосле = BitFlag(20)
    PunctInside = ПунктуацияВнутри = BitFlag(21)
    PunctBefore = ПунктуацияПеред = BitFlag(22)
    PunctBeforeAnyPart = ПунктуацияПередЛюбойЧастью = BitFlag(23)

    # к закрывающей пунктуации не относится "открывающая": "(«" и т.п.
    ClosedPunctAfter = ЗакрывающаяПунктуацияПосле = BitFlag(24)


    Prev_A_I = Предшествует_А_И = BitFlag(30)
    Next_ZheComma = СледуетЖеЗапятая = BitFlag(31) # Сейчас этот признак не используется в правилах

    BspSignalBefore = БспСигналПеред = BitFlag(41) # аббревиатуру БСП (Asyndetic Compound Sentence) в русскоязычных NLP-проектах часто оставляют транслитом для простоты
    DependentAfter = ЗависимоеПосле = BitFlag(42) # Зависимое от последнего токена фразы слово идёт сразу после фразы не считая пунктуации
 
    LastHeadAfter = ПоследнегоГлавноеСправа = BitFlag(50) # Главное слово последнего токена фразы справа
#    FirstHeadLeft = ПервогоГлавноеСлева = BitFlag(60) # Главное слово первого токена фразы слева

    # ПРИЗНАКИ С ПРАВИЛ, КОТОРЫЕ БОЛЬШЕ НЕ ИСПОЛЬЗУЮТСЯ
    LinkInEachPart = СвязьВКаждойЧасти = BitFlag(70) 
    SingleParticle = ОднаЧастица = BitFlag(71) 
    PronAfterAdp = МестоимениеПослеПредлога = BitFlag(72) 
    HeadNextToken = ВершинаСледующийТокен = BitFlag(73) 

    # ЕЩЁ ПРИЗНАКИ
    PhraseInSet = ФразаВУстойчивомВыражении = BitFlag(80) 

    # ПРИЗНАКИ ДЛЯ СОЮЗОВ И СОЮЗНЫХ СЛОВ (выделены из rule_conj)
    Next_By = СледующеетБы = BitFlag(81)  # Сразу после фразы идёт частица «бы» (например, «как бы», «что бы»)
    HeadIsInf = ГлавноеИнфинитив = BitFlag(82)  # Синтаксическая вершина первого слова фразы — инфинитив (VerbForm=Inf, например «как закупать»)
    CcompPerception = ИзъяснительноеВосприятия = BitFlag(83)  # Вершина фразы вводит изъяснительное придаточное (ccomp) при глаголе восприятия/мысли/речи («видели, как она летала»)
    RelativeSubject = СоюзноеСловоПодлежащее = BitFlag(84)  # Первое слово фразы — относительное местоимение-подлежащее (PRON + nsubj, например присоединительное «..., что помогло...»)
    CorrelativeBefore = КоррелятПеред = BitFlag(85)  # Слева перед запятой/фразой стоит соотносительное указательное слово («всё, что», «то/том, что», «тех, что»)


     # Признаки сочинительных союзов
    FirstCconj = ПервоеСочинительныйСоюз = BitFlag(86)      # Первое слово фразы имеет часть речи CCONJ
    FirstConjISignal = СигналСоюзаИ = BitFlag(87)            # Синтаксический сигнал союза «и»: deprel in {cc, fixed}
    FirstConjIliSignal = СигналСоюзаИли = BitFlag(88)        # Синтаксический сигнал разделительного союза «или» / «либо»: deprel in {cc, discourse, fixed}


    # Свойства синтаксической вершины (head) первого слова фразы
    HeadIsInf = ГлавноеИнфинитив = BitFlag(90)      # Синтаксическая вершина — инфинитив (VerbForm=Inf)
    HeadIsPart = ГлавноеПричастие = BitFlag(91)      # Синтаксическая вершина — причастие (VerbForm=Part, например «лучше организованы», «только испечён»)
    HeadIsConv = ГлавноеДеепричастие = BitFlag(92)   # Синтаксическая вершина — деепричастие (VerbForm=Conv, например «фактически превратив»)
    HeadIsAdj = ГлавноеПрилагательное = BitFlag(93)  # Синтаксическая вершина — прилагательное (upos == ADJ)

    def print_debug(self, label: str = "PHRASE", verbose: bool = True) -> None:
        """Выводит отладочную информацию о фразе и её морфосинтаксических признаках."""

        self.verbose = verbose # устанавливаем флаг, который выведет дополнительную отадку. Если она не нужна, запустите .print_debug(verbose=False)
        if not self.first:
            print(f"DEBUG {label}: <пустая фраза> | sent={tokens2text(self.sent_tokens)}")
            return

        head_info = f"{self.first.head.s}({self.first.head.upos},{self.first.head.deprel})" if self.first.head else "None"
        last_info = f", last={self.last.s!r}" if not self.Len1 and self.last else ""

        print(
            f"DEBUG {label}: surface={self.surface!r}, Len1={self.Len1}{last_info}, "
            f"upos={self.first.upos}, deprel={self.first.deprel}, head={head_info}, "
            f"init={self.SentenceInitial}, bits={sorted(self._bits)} | "
            f"sent={tokens2text(self.sent_tokens)}"
        )

    def _calc_features(self):
        """Вычисляет и устанавливает бинарные признаки для фразы."""
        if not self.words:
            return

        first= self.first
        last= self.last

        # СтартоваяПозиция
        self.StartPosition = not self.prev

        # КонечнаяПозиция
        self.EndPosition = not self.next

        # НачалоПредложения (Проверяем id. Теоретически поток токенов может содержать несколько предложений с точки зрения Stanza, хотя мы по предложениям токены делим)
        self.SentenceInitial = (bool(first) and first.id == 1)

        # Вводная конструкция
        self.Parataxis = any(w["deprel"] == "parataxis" for w in self.words)
        self.Discourse= any(w["deprel"] == "discourse" for w in self.words)
        self.Coordinating = any(w["deprel"] == "cc" for w in self.words)
        self.Marker = any(w["deprel"] == "mark" for w in self.words)
        self.Fixed = any(w["deprel"] == "fixed" for w in self.words)

        # Поясняющий
        self.Parenthetical = self.Parataxis or self.Discourse
        # Связующий
        self.Connective = self.Coordinating or self.Marker or self.Fixed
        
        self.Structural = self.Parenthetical or self.Connective

     
        self.Complex = len(self.parts) > 1
        self.Len1 = len(self.words) == 1

        # All parts of a split phrase contain a core connective relation
        connective_deprel = {"cc", "mark", "fixed"}
        self.ConnectiveInEachPart = len(self.parts) > 1 and all(any(w["deprel"] in connective_deprel for w in part) for part in self.parts)


        open_punct_str = "([«"

        #ПунктуацияПосле
        self.PunctAfter = bool(self.next) and self.next.upos == "PUNCT" 
        self.ClosedPunctAfter = self.PunctAfter and self.next.text not in open_punct_str # Для правила NoSpaseAfter

        #ПунктуацияВнутри
        self.PunctInside = any(word.next and word.next.upos == "PUNCT" for word in self.words[:-1]) # Для правила NoSpaseAfter

        #ПунктуацияПеред
        self.PunctBefore = bool(self.prev) and self.prev.upos == "PUNCT" # 

        #ПунктуацияПередЛюбойЧастью
        self.PunctBeforeAnyPart =  any(part and part[0].prev and part[0].prev.upos == "PUNCT" for part in self.parts)


        self.Prev_A_I = bool(self.prev) and self.prev.lemma in {"а", "и"}

        # этотпризнак сейчс не используется
        self.Next_ZheComma = bool(self.next) and bool(self.next.next) and self.next.lemma == "же" and self.next.next.upos == "PUNCT"


        # Анализ знаков препинания слева от фразы (Бессоюзное сложное предложение / Границы)
        self.BspSignalBefore = False
        curr = self.prev
        while curr:
            if curr.upos == "PUNCT":
                if curr.text in {":", "—", ";"}:
                    self.BspSignalBefore = True # Начало новой части в БСП
                    break
                if curr.text in {".", "!", "?"}:
                    break      # Граница предложения, останавливаем поиск
            curr = curr.prev

        # Есть зависимый токен справа, не считая пунктуации.

        self.DependentAfter = False
        curr = last.next
        while curr:
            if curr.upos == "PUNCT": #пропускаем пунктуацию
                curr = curr.next
                continue

            if curr.head_id and curr.head_id == last.id:
                self.DependentAfter = True
                break

            curr = curr.next

        self.LastHeadAfter = bool(last) and bool(last.id) and bool(last.head_id) and last.head_id > last.id

        # ПРИЗНАКИ ПРАВИЛ, КОТОРЫЕ НЕ ИСПОЛЬЗУЮТСЯ
        # СвязьВКаждойЧасти
        self.LinkInEachPart = True if all(any(w.upos in {"SCONJ", "CCONJ", "ADV"} for w in part) for part in self.parts) else False
        # ОднаЧастица
        self.SingleParticle = self.Len1 and first.upos == "PART"
        # Местоимение после предлога
        self.PronAfterAdp = any(w.prev and w.upos == "PRON" and w.prev.upos == "ADP" for w in self.words)

        #ВершинаСледующийТокен
        self.HeadNextToken = any(
            bool(w.next) and w.head_id == w.next.id
            for part in self.parts
            for w in part )
        
        # Фраза входит в устойчивое выражение
        self.PhraseInSet = self._check_phrase_in_set()      

        # ПРИЗНАКИ ДЛЯ СОЮЗОВ И СОЮЗНЫХ СЛОВ (из rule_conj)
        # Следующий токен — частица «бы» («как бы», «что бы»)
        self.Next_By = bool(self.next) and self.next.s == "бы"

        # Вершина первого токена — инфинитив (характерно для вопросительных/инфинитивных оборотов: «как закупать», «что делать»)
        self.HeadIsInf = bool(first) and bool(first.head) and "VerbForm=Inf" in first.head.feats

        # Изъяснительное придаточное (ccomp) при глаголах восприятия, знания и речи («видели, как она летала»)
        speech_perception_lemmas = {
            "видеть", "увидеть", "слышать", "услышать", "знать", "узнать",
            "понимать", "понять", "помнить", "вспомнить", "заметить",
            "наблюдать", "смотреть", "показать", "рассказать", "описать",
        }
        first_head = first.head if first else None
        head_parent = first_head.head if first_head else None
        self.CcompPerception = (
            bool(first_head)
            and first_head.deprel == "ccomp"
            and bool(head_parent)
            and head_parent.lemma in speech_perception_lemmas
        )

        # Союзное слово в роли подлежащего (PRON + nsubj — характерно для присоединительного «..., что привело к...»)
        self.RelativeSubject = bool(first) and first.upos == "PRON" and first.deprel == "nsubj"

        # Наличие соотносительного слова (коррелята) слева перед знаками препинания («всё, что», «с тем, что», «тех, что»)
        correlative_lemmas = {"всё", "все", "тот", "то", "такой", "один"}
        prev_word = self.prev
        while prev_word and prev_word.upos == "PUNCT":
            prev_word = prev_word.prev
        self.CorrelativeBefore = bool(prev_word) and prev_word.lemma in correlative_lemmas        

         # Признаки сочинительных союзов (морфология и синтаксическая связь первого токена)
        self.FirstCconj = bool(first) and first.upos == "CCONJ"
        self.FirstConjISignal = bool(first) and first.deprel in {"cc", "fixed"}
        self.FirstConjIliSignal = bool(first) and first.deprel in {"cc", "discourse", "fixed"}

        first_head = first.head if first else None

        self.HeadIsInf = bool(first_head) and "VerbForm=Inf" in first_head.feats
        self.HeadIsPart = bool(first_head) and "VerbForm=Part" in first_head.feats
        self.HeadIsConv = bool(first_head) and "VerbForm=Conv" in first_head.feats
        self.HeadIsAdj = bool(first_head) and first_head.upos == "ADJ"

    def _check_phrase_in_set(self) -> bool:
        """Проверяет, входит ли хотя бы одно слово фразы в устойчивое выражение из engine.sets."""
        if not self.engine or not getattr(self.engine, "sets_by_word", None):
            return False

        sent_words = [t.s for t in self.sent_tokens]
        n_sent = len(sent_words)

        for w in self.words:
            candidates = self.engine.sets_by_word.get(w.s)
            if not candidates:
                continue

            tok_idx = w.index()
            for expr, pos_in_expr in candidates:
                start = tok_idx - pos_in_expr
                end = start + len(expr)

                if start >= 0 and end <= n_sent:
                    if tuple(sent_words[start:end]) == expr:
                        return True

        return False
    

def rule_sentence_initial(phrase:Phrase):
    '''Фраза в начале предложения'''
    return +1 if phrase.SentenceInitial else 0

def rule_punct_before(phrase:Phrase):
    '''Пунктуация перед фразой или её отдельной частью'''
    return +1 if phrase.PunctBeforeAnyPart else 0

def rule_dependency_signal(phrase:Phrase):
    '''Вводные слова или связующие элементы в простой фразе, либо связующие элементы в каждой части сложной фразы'''
    # Any word has a target dependency relation or All parts of a split phrase contain a core connective relation
    return +1 if phrase.Structural or phrase.ConnectiveInEachPart else 0

def rule_no_space_after(phrase:Phrase):
    '''Правило проверяет, что оборот в конце цепочки, либо после оборота пунктуация, либо пунктуация внутри'''
    return +1 if phrase.EndPosition or phrase.ClosedPunctAfter or phrase.PunctInside else 0

# НЕИСПОЛЬЗУЕМЕ
# сейчас не используется. Ранее называлась rule_single_part_only_part
def rule_single_particle(phrase):
     return -1 if phrase.SingleParticle else 0

# сейчас не используется. Ранее назыалась rule_pos_per_part
def rule_link_in_each_part(phrase:Phrase):
    return +1 if Phrase.LinkInEachPart else 0

# сейчас не используется
def rule_pron_after_adp(phrase:Phrase):
    return +1 if phrase.PronAfterAdp else 0

# сейчас не используется
def rule_my_head_is_next_tokens_id(phrase):
    return +1 if phrase.HeadNextToken else 0

# Это правило упрощено. Тут ещё была проверка на "же," после оборота, но она фактически не давала эффекта. 
# Анализ "же," добавлен в признаки, но удалён из этого правила
def rule_introductory_potential(phrase:Phrase):
    """A trailing a leading 'а'/'и', is tolerated around an introductory construction."""
    return +1 if phrase.Prev_A_I else 0



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
    в итоге,
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

    # есть заависимое от последнего слова фразы после него
    if phrase.DependentAfter: 
        return -1
    
    # есть главное справа от последнего и оно существительное или имя собственное
    last = phrase.last
    last_head = last.head if last and last.head else None

    if phrase.LastHeadAfter and last_head and last_head.upos in {"NOUN", "PROPN"}:
        return -1

    return +1

        
def rule_phrase_in_list(phrase: PhraseC):
    """Правило даёт +1 или +2 балла, если оборот в списке."""

    texts = tuple(word.s for word in phrase.words)

    # Блок +1 балл (+0.2 к вероятности)
    triggers = """
    поэтому, вследствие чего, и потому                                  # следствие
    однако, тогда как                                                   # противопоставление
    после чего, потом, впредь, в дальнейшем                             # последовательность и время
    тогда же, сразу же, в это же время, в тот же                        # одновременность
    например, в том числе, в том числе и, наряду с                      # иллюстрация, включение, сопоставление
    фактически, в результате                                            # итог, реальное положение дел
    именно, тем более                                                   # выделение, акцентирование
    в противном случае, благодаря                                       # условие, причина
    как будто, едва, интересно, же, всё чаще, заодно и                  # модальность, оценка, частицы
    """

    if in_triggers(texts, triggers):
        return 1

    # Блок +2 балла (+0.4 к вероятности)
    triggers = """
    даже, также, заодно и, всё же        # присоединение, усиление, уступка
    затем                                # последовательность
    с целью                              # цель
    на сей раз, вот                      # указание, ситуация
    """
    if in_triggers(texts, triggers):
        return 2

    # Штраф -1
    triggers = """
    будто
    """
    if in_triggers(texts, triggers):
        return -1

    # Штраф -2
    triggers = """
    что
    """
    if in_triggers(texts, triggers):
        return -2

    return 0

def rule_special(phrase:PhraseC):
    """Специфические контекстные правила для коннекторов."""

    first = phrase.first
    if not first:
        return 0    

    if phrase.PhraseInSet:
        return -3
        

    # Правило на одиночную "же"
    if phrase.Len1 and first.s=="же":
        prev = first.prev

        if prev:

            # Жесткие текстовые исключения с добавлением лемм "том" и "тех"
            if prev.lemma in {"тот", "такой", "так", "один", "то", "тут", "факт"}:
                return -3
            
            if prev.upos in {"NOUN", "PRON", "PROPN", "DET"} or prev.lemma == "сам":
                if first.id <= 4:
                    return +2

            if prev.prev and prev.prev.text in [":", "—", ";", ","]:
                if prev.upos in {"NOUN", "PRON", "PROPN", "DET"}:
                    return +2

            return 0
        
    if not phrase.SentenceInitial and phrase.surface == "оказалось что":
        if phrase.first.deprel not in {"parataxis", "discourse"}:
            return -1    

    # Предложный оборот «в связи с этим» в середине предложения без обособления слева
    # («Историк Ключевский писал в связи с этим, что...» — пример 38) выступает как дополнение при глаголе.
    if phrase.surface == "в связи с этим":
        if not phrase.SentenceInitial and not phrase.PunctBefore and not phrase.BspSignalBefore:
            return -2
        
    

    # Вводное «как правило» в начале предложения должно закрываться запятой («Как правило, ...» vs first 92)
    if phrase.surface == "как правило":
        if phrase.SentenceInitial and not phrase.ClosedPunctAfter:
            return -2

        
    # Вводное слово «Правда, ...» требует наличия полноценной предикативной основы (глагола) в предложении;
    # в назывных конструкциях без глагола («Правда, только правда и ничего кроме правды» — first 13) это не коннектор.
    if phrase.surface == "правда" and phrase.SentenceInitial:
#        phrase.print_debug()
        has_verb = any(t.upos in {"VERB", "AUX"} for t in phrase.sent_tokens)
        if not has_verb:
            return -3
                
    # Предикатив-сказуемое главного предложения «Видно, что...» (first 24)
    # В отличие от вводного слова «Видно, ...» (deprel in {parataxis, discourse}),
    # здесь «видно» является вершиной всего предложения (deprel == "root").
    if phrase.surface == "видно":
#        phrase.print_debug()
        if phrase.first.deprel == "root":
            return -3
                        
    return 0

def rule_time_connector(phrase:Phrase):
    """Единое правило валидации синтаксического контекста для временных 
    дейктических слов (тогда, теперь, сейчас, впоследствии, ныне, затем и др.)
    """
    words = phrase.words
    if not words:
        return 0


    # 1. Находим конкретный токен дейктика внутри списка слов спана
    time_deictic_lemmas = {
        "тогда", "теперь", #"сейчас", "впоследствии",   "ныне", "нынче", "затем", "потом", "прежде", "раньше"
    }
    word = phrase.find_lemma(time_deictic_lemmas)
            
    if not word:
        return 0

    # if phrase.sent_tokens[0].text == "Если": # первый токен предложения
    #     print(tokens2text(phrase.sent_tokens))
    #     pass    

    # 2. Безусловный бонус для известных устойчивых сочетаний
    if phrase.surface in {"тогда же", "теперь же", "сейчас же", "сразу же"}:
        return +1

    # ПРИОРИТЕТНЫЙ БЛОК: ЖЕЛЕЗНЫЕ ПРИЗНАКИ НАЧАЛА (Решает books388)
    if phrase.SentenceInitial:
        return +2

    # --- СИНТАКСИЧЕСКИЙ АНАЛИЗ КОНТЕКСТА ---
    
    # Если перед дейктиком стоит частица "то" (например, "если... то тогда")
    if word.prev and word.prev.lemma == "то":
        return +2

    # Проверка на причастие (зависимость от VerbForm=Part)
    if word.head and "VerbForm=Part" in word.head.feats:
        return -3

    # Анализ знаков препинания слева от фразы (Бессоюзное сложное предложение / Границы)
    # books413
    if phrase.BspSignalBefore:
        return +2  # Начало новой части в БСП

    # Дополнительно работает в этом правиле
    if phrase.Prev_A_I or phrase.PunctBefore:
        return +1
  
    return 0


def rule_complex(phrase: PhraseC) -> int:
    """Контекстное правило для составных (разрывных) коннекторов."""
    
    if not phrase.Complex:
        return 0

    first = phrase.first
    last = phrase.last
    if not first or not last:
        return 0

    # 1. Ложные разрывные паттерны вроде «что...что», «что...то», «что...если»,
    # когда первое «что» — обычный изъяснительный союз (примеры 24, 61, 81, 278, 441).
    if first.s == "что" and first.upos == "SCONJ":
        return -3

    # 2. Случайные склейки союза «и» с частицей «только» или наречием «так»
    # через полпредложения (примеры 99, 206, 307, 309).
    if phrase.surface in {"и...только", "и...так"}:
        return -3

    # 3. Защита союза «то есть» (пример 382):
    # если начало или конец разрывного паттерна попадает на слово «то» внутри союза «то есть».
    if (last.s == "то" and last.next and last.next.s == "есть") or (
        first.s == "то" and first.next and first.next.s == "есть"
    ):
        return -3

    # 4. Наречие «так» в конструкции «так, чтобы» / «так, что» не должно становиться частью
    # разрывного коннектора (пример 485: «Если ... изображать фигуры так, чтобы ..., то ...»).
    for part in phrase.parts:
        if len(part) == 1 and part[0].s == "так":
            next_word = part[0].next
            while next_word and next_word.upos == "PUNCT":
                next_word = next_word.next
            if next_word and next_word.s in {"чтобы", "что"}:
                return -3

    # 5. Ложные парные склейки с ограничительной частицей «только» (без «не» перед ней)
    # или со сравнительной степенью наречия («чем...лучше» — примеры 38, 205).
    if phrase.surface == "только...а":
        if not (first.prev and first.prev.s == "не"):
            return -3

    if phrase.surface == "чем...лучше":
        return -3
   
    
    return 0


def rule_conj(phrase: Phrase) -> int:
    """Контекстное правило для одиночных союзов/союзных слов («как», «что»)."""
    def not_upper(token) -> bool:
        return bool(token) and token.text.upper() != token.text


    if not phrase.Len1:
        return 0

    word = phrase.first
    if not word:
        return 0

    if word.s == "что":
        # Присоединительное придаточное после запятой: «что» выступает как относительное
        # местоимение-подлежащее (RelativeSubject) и не опирается на коррелят слева
        # вроде «всё, что», «с тем, что», «тех, что» (примеры 16, 103, 145 vs 170, 357, 424).
        if phrase.PunctBefore and phrase.RelativeSubject and not phrase.CorrelativeBefore:
            return +4

        # Изъяснительный союз «что» (SCONJ) или местоимение при корреляте
        return 0

    if word.s == "как":


        # Вопросительное наречие в начале вопросительного предложения («Как правило рождает исключение?» — first 92)
        sent_ends_with_question = bool(phrase.sent_tokens) and phrase.sent_tokens[-1].text == "?"
        if phrase.SentenceInitial and sent_ends_with_question:
            return -3
        
        # Сочетание «как бы» — не одиночный линкер (пример 319)
        if phrase.Next_By:
            return -3

        # Перечисления после двоеточия или инфинитивные вопросы («советы: как закупать...» — пример 482)
        if phrase.HeadIsInf or (phrase.BspSignalBefore and word.upos == "ADV"):
            return -2

        # Изъяснительное «как» после глаголов восприятия/знания («видели, как она летала» — пример 359)
        if phrase.CcompPerception:
            return -2

        # Обособленный сравнительный или пояснительный оборот после запятой
        # («не на восемь, как в Италии», «таких, как...» — примеры 239, 241, 369)
        if (
            phrase.PunctBefore
            and not phrase.BspSignalBefore
            and not phrase.HeadIsInf
            and not phrase.CcompPerception
        ):
            return +1

        # Необособленное «как» в значении «в качестве» («как личность», «известен как...» — примеры 5, 112, 160)
        if not phrase.PunctBefore and not phrase.SentenceInitial:
            return -1

    if word.s == "и":
        if not (phrase.FirstCconj or phrase.FirstConjISignal):
            return 0

        # Корректируем ошибку Stanza отнесения заглавной «И» (инициала имени) к союзу (пример books14)
        if word.text == "И" and word.id > 1:
            if (
                word.prev and (not_upper(word.prev) or not_upper(word.prev.prev))
                or word.next and (not_upper(word.next) or (word.prev and not_upper(word.prev.next)))
            ):
                return 0

        # Внутриименная связка «эти и другие», «те и другие», «такие и другие» (пример 189)
        if (
            word.prev
            and word.prev.lemma in {"этот", "тот", "такой"}
            and word.next
            and word.next.lemma == "другой"
        ):
            return 0

        return +2

    if word.s in {"или", "либо"}:
        if phrase.FirstCconj or phrase.FirstConjIliSignal:
            return +2

        
    return 0


def rule_adverb(phrase: PhraseC) -> int:
    """Точечное правило для наречий образа действия при деепричастиях и полных причастиях."""
    if not phrase.Len1:
        return 0
    
    word = phrase.first
    if not word or word.upos != "ADV":
        return 0

    # Исключаем временные и дискурсивные наречия-коннекторы, которые могут стоять перед сказуемым/деепричастием
    time_and_discourse_adverbs = {
        "затем", "тогда", "теперь", "потом", "впоследствии", "впредь",
        "далее", "едва", "однако", "впрочем", "также", "притом", "причём",
    }
    if word.lemma in time_and_discourse_adverbs:
        return 0

    # Если наречие обособлено запятыми как вводное — не трогаем
    if (phrase.PunctBefore or phrase.SentenceInitial) and phrase.ClosedPunctAfter:
        return 0

    head = word.head
    if not head:
        return 0

    # 1. Наречие при деепричастии («фактически превратив Францию» — пример 125)
    # или при полном причастии (исключаем краткие причастия Variant=Short, которые служат сказуемым)
    is_full_participle = phrase.HeadIsPart and "Variant=Short" not in head.feats
    if phrase.HeadIsConv or is_full_participle:
        return -2

    # 2. Сравнительная степень наречия («лучше», «больше», «чаще») при не-глагольном сказуемом
    if "Degree=Cmp" in word.feats and not phrase.SentenceInitial:
        return -2

    return 0



class EngineChecker(Engine):
    def __init__(
        self,
        nlp: object = None,
        patterns_by_type: dict = None,
        threshold: float = None,
    ) -> None:
        super().__init__(
            nlp=nlp,
            patterns_by_type=patterns_by_type,
            threshold=threshold,
        )

        self.rules: List[Callable] = []
        self.extra_rules: List[Callable] = []  # Экстра-правила не попадают в знаменатель дроби

    def add_rule(self, rule_fn):
        self.rules.append(rule_fn)

    def add_extra_rule(self, rule_fn):  # экстра-парвила
        """Правило, которое при 0 не влияет на probability."""
        self.extra_rules.append(rule_fn)


    #AS Экстра правила
    def score_entity(self, phrase:PhraseC, debug: bool = False) -> float:
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

        if debug or phrase.verbose:
            print(f"  surface={phrase.surface!r}  p={p:.3f}")
            for name, v in votes:
                print(f"    {name}: {v:+d}")
            for name, v in extra_votes:
                print(f"    {name} [extra]: {v:+d}")

        return p

    def score_sentence(self, sentence_json: dict, debug: bool = False) -> List[dict]:

        if debug:
            print(sentence_json.get('sentence'))

        phrases = sentence_json.get("entities", []);
        for phrase in phrases:
            phrase.probability = self.score_entity(phrase, debug=debug)

        return phrases


# функция используется для создания экземпляра класса EngineChecher во всех вызовах, кроме Gradio (Gradio вызывает build_default_checker, пока это сохранено)
def build_engine(linkers_csv: str = None, nlp=None, threshold: float = None) -> EngineChecker:
    patterns = load_patterns(linkers_csv) if linkers_csv else None  
    return build_default_checker(nlp=nlp, patterns_by_type=patterns, threshold=threshold)



#Данная функция вызывается из Gradio её интерфейс сохранён для совместимости
def build_default_checker(
    nlp: object = None,
    patterns_by_type: dict = None,
    threshold: float = None,
) -> EngineChecker:
    """Создаёт экземпляр EngineChecker и регистрирует стандартный набор правил."""

    checker = EngineChecker(
        nlp=nlp,
        patterns_by_type=patterns_by_type,
        threshold=threshold,
    )

    
    checker.add_rule(rule_sentence_initial)         # +1 если оборот в начале предложения
    checker.add_rule(rule_punct_before)             # +1 если пунктуация перед любой частью оборота
    checker.add_rule(rule_dependency_signal)        # +1 если есть маркеры "parataxis", "discourse", "cc", "mark", "fixed"
    checker.add_rule(rule_no_space_after)           # +1 если пунктуация после оборота, внутри оборота или оборот в конце текста
    checker.add_rule(rule_introductory_potential)   # +1 если "а", "и" перед оборотом

    checker.add_extra_rule(rule_pri_etom_dependent) # экстра-правило по типу фразы "при этом". См. комментарий к функции rule_pri_etom_dependent
    checker.add_extra_rule(rule_phrase_in_list)     # правило, усиливающее оценку конкретных оборотов 
    checker.add_extra_rule(rule_special)            # правило, работающее с разными коннекторами
    checker.add_extra_rule(rule_time_connector)     # правило для коннекторов времени
    checker.add_extra_rule(rule_conj)               # союзы "что, как, и, или" и другие
    checker.add_extra_rule(rule_complex)            # cложные (из разных частей) паттерны
    checker.add_extra_rule(rule_adverb)             # наречия

    return checker


# Функция возвращает текст по списку тоуенов Stanza. Создана исключительно для упрощения отладки
def tokens2text(tokens) -> str:
    """Восстанавливает исходный текст предложения из списка объектов-токенов Stanza
    с корректной расстановкой пробелов вокруг знаков препинания и кавычек.
    """
    if not tokens:
        return ""
        
    result = []
    for token in tokens:
        result.append(token.text)
        
        if token.next:
            # пробел перед этими знаками препинания ставить не нужно.
            if token.next.text in ",.!?:;)]»": continue

            # Если текущий токен — открывающая кавычка или скобка, после неё пробел не нужен
            if token.text in "([«": continue
                
        result.append(" ")
        
    return "".join(result).strip()
