"""Spell check behavior (issue #12: enabling spell check appeared to do nothing)."""
import ast
import os.path as osp
import sys
import types

import pytest

# utils.config pulls in cv2 through utils.io_utils; stub it for headless test runs.
cv2_stub = sys.modules.get("cv2") or types.ModuleType("cv2")
cv2_stub.IMREAD_COLOR = getattr(cv2_stub, "IMREAD_COLOR", 1)
cv2_stub.IMREAD_GRAYSCALE = getattr(cv2_stub, "IMREAD_GRAYSCALE", 0)
cv2_stub.COLOR_GRAY2RGB = getattr(cv2_stub, "COLOR_GRAY2RGB", 0)
cv2_stub.cvtColor = getattr(cv2_stub, "cvtColor", lambda img, code: img)
sys.modules["cv2"] = cv2_stub

from utils.ocr_spellcheck import (
    LANG_DICT_TAGS,
    _edit_distance_at_most_1,
    get_spell_issues,
    pick_auto_correction,
    spell_check_available,
    spell_check_line,
    spell_check_status,
)


def test_edit_distance_at_most_1():
    assert _edit_distance_at_most_1("teh", "the")        # transposition
    assert _edit_distance_at_most_1("wht", "what")       # insertion
    assert _edit_distance_at_most_1("thee", "the")       # deletion
    assert _edit_distance_at_most_1("cat", "car")        # substitution
    assert not _edit_distance_at_most_1("cat", "dogs")


def test_pick_auto_correction_accepts_close_typo():
    assert pick_auto_correction("teh", ["the", "tech"]) == "the"
    assert pick_auto_correction("Teh", ["the"]) == "The"
    assert pick_auto_correction("THEER", ["there"]) == "THERE"


def test_pick_auto_correction_rejects_different_word():
    # A suggestion that adds an apostrophe / splits the word is not a typo fix.
    assert pick_auto_correction("Ths", ["Th's", "This"]) == ""
    assert pick_auto_correction("exmaple", ["ex maple"]) == ""
    # Too short / too far away to guess.
    assert pick_auto_correction("ab", ["abs"]) == ""
    assert pick_auto_correction("qwerty", ["query"]) == ""
    assert pick_auto_correction("word", []) == ""


def test_languages_without_word_boundaries_are_not_checked():
    for lang in ('简体中文', '日本語'):
        assert lang not in LANG_DICT_TAGS
        assert not spell_check_available(lang)
        assert spell_check_status(lang)


def test_unknown_language_never_uses_the_wrong_dictionary():
    # Checking text with a dictionary of another language flags everything / nothing.
    assert not spell_check_available('Klingon')


def test_inline_highlight_setting_exists_and_defaults_on():
    from utils.config import ProgramConfig
    assert 'spell_check_highlight' in ProgramConfig.__dataclass_fields__
    assert ProgramConfig().spell_check_highlight is True


def _source(*parts):
    return open(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), *parts), encoding='utf-8').read()


def test_text_editors_mark_misspellings():
    """SourceTextEdit paints misspellings itself; the panel alone left them unmarked."""
    tree = ast.parse(_source('ui', 'textedit_area.py'))
    methods = {
        node.name
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == 'SourceTextEdit'
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }
    for expected in ('update_spellcheck_marks', 'schedule_spellcheck', 'spellcheck_issue_at', '_add_spellcheck_actions'):
        assert expected in methods, f'SourceTextEdit.{expected} is missing'


@pytest.mark.skipif(not spell_check_available('English'), reason='no English dictionary installed')
def test_english_issues_and_autocorrect():
    issues = get_spell_issues("Ths is a smiple exmaple.", 'English')
    assert [i[0] for i in issues] == ['Ths', 'smiple', 'exmaple']
    # positions must point at the word so the editor can underline it
    for word, start, end, _suggs in issues:
        assert "Ths is a smiple exmaple."[start:end] == word
    assert not get_spell_issues("This is a simple example.", 'English')
    assert spell_check_line("I havv teh answer", 'English') == "I have the answer"
