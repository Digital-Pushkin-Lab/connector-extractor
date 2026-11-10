"""Rule-based scorer that estimates how likely a matched span is to actually
be functioning as a linker (vs. a coincidental word sequence).

Ported from `RuleBasedLinkerChecker` and its `rule_*` functions in
`linkers (1).ipynb`. Each rule inspects the matched token parts and votes
-1/0/+1; the average vote (clamped at 0) is the reported probability.
"""

from typing import Callable, Dict, List


from pyroaring import BitMap


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
        self.st = token_stanza    # Stanza или Stanza Token (для Spacy Token, впрочем, будет sp)
        st_dict = token_stanza.to_dict()
        self.info =st_dict[0]   # Первое слово из токена Stanza превращённого в словарь. Для доступа к инфорамции Stanaza через интерфейс словаря

        self.text = token_stanza.text
        self.s = self.text.lower()   #часто нужно в нижем регистре, возможно тут требуется дополнительно отчистить от, например, знаков ударения
        self.tokens = tokens         # ключевой момент. Токены исходного предложения с типом Token


    # @property
    # def f(self) -> TokenFeatures:
    #     """Признаки токена с ленивым вычислением."""
    #     if self._f is None:
    #         self.assemble_token_info()
    #     return self._f

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
        
        # Вычисляем относительное смещение в нумерации ID Stanza
        offset = self.head_id - self.id
        # Находим целевой индекс в списке tokens предложения
        target_idx = self.index() + offset
        
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

    # # Для совместимости с существующим кодом.
    # def to_dict(self):
    #     return self.std_


    def get(self, key, default=None):
        return self.info.get(key, default)

    def __getitem__(self, key):
        return self.info[key]



class Phrase:

    # Чтобы self._bits заполнился автоматически, мы должны здесь перечислить эти значения
    StartPosition = СтартоваяПозиция = BitFlag(0)
    EndPosition = КонечнаяПозиция = BitFlag(1)

    SentenceInitial = НачалоПредложения = BitFlag(2)

    Parataxis = ВводнаяКонструкция = BitFlag(3) # Отношение между главным предложением и соседней независимой конструкцией. Сюда относятся вводные слова, пояснения или прямая речь (например, «как мне кажется», «сказал он»).
    Discourse = ДискурсивныйМаркер = BitFlag(4) # Элементы речи, которые не несут синтаксической нагрузки, но важны для структуры диалога или выражения эмоций. Это междометия, частицы, маркеры ответа и привлечения внимания (например, «ну», «ого», «ведь»).
    CC = Coordinating = СочинительныйCоюз = BitFlag(5)  # Отношение между элементом предложения и сочинительным союзом, который его связывает (например, союзы «и», «а», «но»).
    Marker  = ПодчинительныйМаркер  = BitFlag(6) # Отношение между придаточным предложением и подчинительным союзом (или инфинитивной частицей), который его вводит (например, «чтобы», «если», «потому что»).
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


    def __init__(self, parts, spans, pattern):
        self.parts = parts
        self.spans = spans
        self.pattern = pattern
        self.surface = "...".join(" ".join(part) for part in pattern)
        self.words = [word for part in parts for word in part]  #токены фразы
        self.sent_tokens = self.words[0].tokens #токены предложения
     
        # Инициализируем хранилище битов (можно заменить на BitMap())
        self._bits = set() 
        
        # 2. Запускаем вычисление признаков при создании объекта
        self._calc_features()


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


        #ПунктуацияПосле
        self.PunctAfter = bool(self.next) and self.next.upos == "PUNCT" # Для правила NoSpaseAfter

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

        # ПРИЗНАКИ ПРАВИЛ, КОТОРЫЕНЕ ИСПОЛЬЗУЮТСЯ
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
        if isinstance(lemmas, str):
            lemmas = {lemmas}
        else:
            lemmas = set(lemmas)

        return next((w for w in self.words if w.lemma in lemmas), None)  
      
    def get(self, key, default=None):
        return getattr(self, key, default)
    

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __repr__(self):
        return f"Phrase('{self.surface}')"




def rule_sentence_initial(phrase:Phrase):
    '''Фраза в начале предложения'''
    if phrase.SentenceInitial:
        return +1
    return 0

def rule_punct_before(phrase:Phrase):
    '''Пунктуация перед фразой или её отдельной частью'''

    if phrase.PunctBeforeAnyPart:
        return +1
    return 0


def rule_dependency_signal(phrase:Phrase):
    '''Вводные слова или связующие элементы в простой фразе, либо связующие элементы в каждой части сложной фразы'''

    # 1. Any word has a target dependency relation
    if phrase.Structural:
        return +1

    # 2. All parts of a split phrase contain a core connective relation
    if phrase.ConnectiveInEachPart:
        return +1

    return 0

def rule_no_space_after(phrase:Phrase):
    if phrase.EndPosition or phrase.PunctAfter or phrase.PunctInside:
        return +1
    return 0

# сейчас не используется
# Ранее называлась rule_single_part_only_part
def rule_single_particle(phrase):
 
    if phrase.SingleParticle:
        return -1
    return 0



# сейчас не используется
# ранее назыалась rule_pos_per_part
def rule_link_in_each_part(phrase:Phrase):
    return +1 if Phrase.LinkInEachPart else 0




# сейчас не используется
def rule_pron_after_adp(phrase:Phrase):
    return +1 if phrase.PronAfterAdp else 0

# сейчас не используется
def rule_my_head_is_next_tokens_id(phrase):
    return +1 if phrase.HeadNextToken else 0

# Это правило упрощено. Тут ещё была проверка на "же," после оборота, но она фактически не давала эффекта. 
# Такой прихнак на "же," добавлен в признаки, но удалён из этого правила
def rule_introductory_potential(phrase:Phrase):
    
    """A trailing a leading 'а'/'и', is tolerated around an introductory construction."""

    if phrase.Prev_A_I:
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


    # есть заависимое от последнего слова фразы после него
    if phrase.DependentAfter: 
        return -1
    
    # есть главное справо справа от последнего и оно существительное или имя собственное
    last = phrase.last
    last_head = last.head if last and last.head else None

    if phrase.LastHeadAfter and last_head and last_head.upos in {"NOUN", "PROPN"}:
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
    и потому
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



def rule_conjunction_i(phrase:Phrase):
    """Отдельный союз «и, или, либо» → сильный бонус; иначе 0."""
    
    word = phrase.first
    if not phrase.Len1 or not word.lemma in {"и", "или", "либо"}:
        return 0

    # типичные признаки сочинительного союза у Stanza
    if word.upos == "CCONJ" or word.deprel in {"cc", "discourse", "fixed"}:
        return +2  # сдвиг p на 2/n (при n=5 → +0.4)core_entity
    return 0


def rule_special(phrase:Phrase):
    """Специфические контекстные правила для сложных одиночных коннекторов."""
    if not phrase.words:
        return 0
    first = phrase.first

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
    checker.add_rule(rule_punct_before) #+1 если пунктуация перед любой частью оборота
    checker.add_rule(rule_dependency_signal) #+1 если есть маркеры "parataxis", "discourse", "cc", "mark", "fixed"
    checker.add_rule(rule_no_space_after) #+1 если пунктуация после оборота, внутри оборота или оборот в конце текста

    checker.add_rule(rule_introductory_potential) # "а" или "и" перед оборотом, либо "же," после оборота


    checker.add_extra_rule(rule_pri_etom_dependent)  # AS Добавляем экстра-правило. Также см комментарий к функции rule_pri_etom_dependent
    checker.add_extra_rule(rule_conjunction_i)# AS Добавляем экстра-правило для и
    checker.add_extra_rule(rule_phrase_in_list)# AS Добавляем правило, усиливающее оценку конкретных оборотов на +1 (0.2)
    checker.add_extra_rule(rule_special)
    checker.add_extra_rule(rule_time_connector)     

    return checker


# Функция возвращает текст по списку тоуенов Stanza. Создана исключительно для упрощения отладки
def tokens2text(tokens) -> str:
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
