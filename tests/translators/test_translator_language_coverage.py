"""Language coverage of translators whose lang_map is written by hand.

A translator that overwrites ``self.lang_map`` with a literal dict silently drops
every language missing from that dict (issue #151: Hungarian was not selectable
for the Gemini translator).  LLM based translators can translate to any language
in the global list, so they must cover all of it.
"""
import sys
import types

cv2_stub = types.ModuleType("cv2")
cv2_stub.IMREAD_COLOR = 1
cv2_stub.IMREAD_GRAYSCALE = 0
cv2_stub.COLOR_GRAY2RGB = 0
cv2_stub.INTER_LINEAR = 1
cv2_stub.INTER_AREA = 3
cv2_stub.INTER_NEAREST = 0
cv2_stub.INTER_CUBIC = 2
cv2_stub.INTER_LANCZOS4 = 4
cv2_stub.BORDER_CONSTANT = 0
cv2_stub.BORDER_REFLECT = 2
cv2_stub.BORDER_REPLICATE = 1
cv2_stub.copyMakeBorder = lambda *args, **kwargs: None
cv2_stub.cvtColor = lambda img, code: img
sys.modules.setdefault("cv2", cv2_stub)

textblock_stub = types.ModuleType("utils.textblock")
textblock_stub.TextBlock = type("TextBlock", (), {})
sys.modules.setdefault("utils.textblock", textblock_stub)

from modules.translators.base import LANGMAP_GLOBAL
from modules.translators.trans_neverliie_sdk import _build_lang_map


def _expected_languages():
    return {lang for lang in LANGMAP_GLOBAL if lang != 'Auto'}


def test_neverliie_lang_map_covers_global_languages():
    lang_map = _build_lang_map()
    missing = _expected_languages() - {k for k, v in lang_map.items() if v}
    assert not missing, f"Gemini/Mistral translator is missing languages: {sorted(missing)}"


def test_neverliie_lang_map_has_hungarian():
    assert _build_lang_map().get('magyar nyelv') == 'Hungarian'
