"""Optional spell check for OCR / translation text. Uses pyenchant if available.

The checker is chosen from the language the text is supposed to be in instead of the
system default locale: checking English text with, say, a Russian dictionary either
flags everything or nothing, which is what made spell check look broken (issue #12).
"""
import re

from utils.logger import logger as LOGGER

_spell_checker = None
_spell_available = None
_checker_cache = {}
_last_error = ''

WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)

# Language key (as used by the translator language selectors) -> candidate enchant tags.
# Languages without word delimiters / hunspell dictionaries (Chinese, Japanese, Thai...)
# are deliberately absent: spell checking them word by word is meaningless.
LANG_DICT_TAGS = {
    'English': ('en_US', 'en_GB', 'en'),
    'Français': ('fr_FR', 'fr'),
    'Deutsch': ('de_DE', 'de'),
    'Español': ('es_ES', 'es'),
    'Português': ('pt_PT', 'pt'),
    'Brazilian Portuguese': ('pt_BR', 'pt'),
    'Italiano': ('it_IT', 'it'),
    'Nederlands': ('nl_NL', 'nl'),
    'Polski': ('pl_PL', 'pl'),
    'русский язык': ('ru_RU', 'ru'),
    'украї́нська мо́ва': ('uk_UA', 'uk'),
    'magyar nyelv': ('hu_HU', 'hu'),
    'limba română': ('ro_RO', 'ro'),
    'čeština': ('cs_CZ', 'cs'),
    'Türk dili': ('tr_TR', 'tr'),
    'Tiếng Việt': ('vi_VN', 'vi'),
    '한국어': ('ko_KR', 'ko'),
    'Arabic': ('ar', 'ar_SA'),
    'Hindi': ('hi_IN', 'hi'),
    'Malayalam': ('ml_IN', 'ml'),
    'Tamil': ('ta_IN', 'ta'),
}

DEFAULT_DICT_TAGS = ('en_US', 'en_GB', 'en')


def _init_enchant():
    """True when pyenchant and at least one usable dictionary are installed."""
    global _spell_checker, _spell_available, _last_error
    if _spell_available is not None:
        return _spell_available
    try:
        import enchant  # noqa: F401
    except Exception as e:
        _last_error = f'pyenchant is not installed ({e})'
        LOGGER.debug("Spell check (enchant) not available: %s", e)
        _spell_available = False
        return False

    _spell_checker = _get_checker()
    if _spell_checker is None:
        _spell_available = False
        return False
    _spell_available = True
    return True


def _get_checker(lang: str = None):
    """Return an enchant dictionary for `lang` (a translator language key), or None."""
    global _last_error
    key = lang or ''
    if key in _checker_cache:
        return _checker_cache[key]

    try:
        import enchant
    except Exception as e:
        _last_error = f'pyenchant is not installed ({e})'
        _checker_cache[key] = None
        return None

    if lang:
        tags = LANG_DICT_TAGS.get(lang)
        if tags is None:
            # Unknown language, or one that cannot be spell checked word by word.
            _checker_cache[key] = None
            return None
    else:
        tags = DEFAULT_DICT_TAGS

    checker = None
    for tag in tags:
        try:
            if enchant.dict_exists(tag):
                checker = enchant.Dict(tag)
                break
        except Exception as e:
            _last_error = f'{tag}: {e}'
    if checker is None:
        _last_error = f'no dictionary installed for {"/".join(tags)}'
        LOGGER.debug('Spell check: %s', _last_error)
    _checker_cache[key] = checker
    return checker


def spell_check_available(lang: str = None) -> bool:
    """True when misspellings can be looked up for `lang` (None = default dictionary)."""
    if not _init_enchant():
        return False
    return _get_checker(lang) is not None


def spell_check_status(lang: str = None) -> str:
    """Human readable reason why spell check is unavailable ('' when it works)."""
    if spell_check_available(lang):
        return ''
    if lang and lang not in LANG_DICT_TAGS:
        return f'Spell check is not supported for {lang}.'
    return _last_error or 'spell check is unavailable'


def _suggest(checker, word: str):
    try:
        return checker.suggest(word) or []
    except Exception as e:
        LOGGER.debug('Spell check suggest failed for %r: %s', word, e)
        return []


def _edit_distance_at_most_1(a: str, b: str) -> bool:
    """True when `a` becomes `b` with one insert/delete/substitution or one transposition."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        if len(diffs) == 2 and diffs[1] == diffs[0] + 1:
            i, j = diffs
            return a[i] == b[j] and a[j] == b[i]     # transposition, e.g. teh -> the
        return False
    # one insertion / deletion
    longer, shorter = (a, b) if la > lb else (b, a)
    i = j = 0
    skipped = False
    while i < len(longer) and j < len(shorter):
        if longer[i] == shorter[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            i += 1
    return True


def pick_auto_correction(word: str, suggestions) -> str:
    """
    Best replacement for an auto-correctable typo, or '' when nothing is safe enough.

    Only near-identical suggestions are accepted: a suggestion that adds punctuation
    or splits the word ("Ths" -> "Th's") is a different word, not a typo fix.
    """
    if not suggestions:
        return ''
    candidate = suggestions[0]
    if any(ch in candidate for ch in " -'’") and not any(ch in word for ch in " -'’"):
        return ''
    lowered, lowered_candidate = word.lower(), candidate.lower()
    if lowered == lowered_candidate:
        return ''
    if len(word) < 3 or not _edit_distance_at_most_1(lowered, lowered_candidate):
        return ''
    if word.isupper():
        return candidate.upper()
    if word[:1].isupper() and candidate[:1].islower():
        return candidate[:1].upper() + candidate[1:]
    return candidate


def spell_check_line(line: str, lang_hint: str = None):
    """Return the line with unambiguous typos corrected (e.g. "teh" -> "the")."""
    if not line or not line.strip():
        return line
    if not _init_enchant():
        return line
    checker = _get_checker(lang_hint)
    if checker is None:
        return line
    try:
        out = []
        last = 0
        for m in WORD_RE.finditer(line):
            word = m.group(0)
            if checker.check(word):
                continue
            correction = pick_auto_correction(word, _suggest(checker, word))
            if not correction:
                continue
            out.append(line[last:m.start()])
            out.append(correction)
            last = m.end()
        if not out:
            return line
        out.append(line[last:])
        return "".join(out)
    except Exception as e:
        LOGGER.debug("Spell check error: %s", e)
        return line


def _ocr_lang_hint() -> str:
    try:
        from utils.config import pcfg
        return getattr(pcfg.module, 'translate_source', '') or ''
    except Exception:
        return ''


def spell_check_textblocks(textblocks, **kwargs):
    """Postprocess hook: run spell check on each block's text if pcfg.ocr_spell_check is True."""
    from utils.config import pcfg
    if not getattr(pcfg, "ocr_spell_check", False):
        return
    if not _init_enchant():
        return
    lang_hint = _ocr_lang_hint()
    if _get_checker(lang_hint) is None:
        return
    for blk in textblocks:
        if not getattr(blk, "text", None) or not isinstance(blk.text, list):
            continue
        blk.text = [spell_check_line(line, lang_hint) for line in blk.text]


def get_spell_issues(text: str, lang: str = None) -> list:
    """Return list of (word, start_idx, end_idx, suggestions) for misspelled words."""
    if not text or not _init_enchant():
        return []
    checker = _get_checker(lang)
    if checker is None:
        return []
    issues = []
    for m in WORD_RE.finditer(text):
        word = m.group(0)
        try:
            if checker.check(word):
                continue
        except Exception as e:
            LOGGER.debug('Spell check failed for %r: %s', word, e)
            continue
        # Keep misspellings even when dictionary returns no suggestions,
        # so the Spell Check panel can still highlight/report the word.
        issues.append((word, m.start(), m.end(), _suggest(checker, word)))
    return issues


def reset_spell_check_cache():
    """Forget cached dictionaries (e.g. after installing pyenchant or a new dictionary)."""
    global _spell_checker, _spell_available, _last_error
    _checker_cache.clear()
    _spell_checker = None
    _spell_available = None
    _last_error = ''
