"""Behavior ported from upstream dmMaze/BallonsTranslator after our fork point."""
import sys
import types

import numpy as np
import pytest

cv2_stub = sys.modules.get("cv2") or types.ModuleType("cv2")
for _name, _value in (
    ("IMREAD_COLOR", 1), ("IMREAD_GRAYSCALE", 0), ("COLOR_GRAY2RGB", 0),
    ("INTER_LINEAR", 1), ("INTER_AREA", 3), ("INTER_NEAREST", 0), ("INTER_CUBIC", 2),
    ("INTER_LANCZOS4", 4), ("BORDER_CONSTANT", 0), ("BORDER_REFLECT", 2), ("BORDER_REPLICATE", 1),
):
    if not hasattr(cv2_stub, _name):
        setattr(cv2_stub, _name, _value)
cv2_stub.cvtColor = getattr(cv2_stub, "cvtColor", lambda img, code: img)
cv2_stub.copyMakeBorder = getattr(cv2_stub, "copyMakeBorder", lambda *a, **k: None)
sys.modules["cv2"] = cv2_stub

from utils.imgproc_utils import hex2bgr  # noqa: E402
from utils.text_processing import capitalize_sentences  # noqa: E402


def test_hex2bgr_keeps_the_least_significant_bit():
    """The old 254 bitmask silently zeroed the LSB of red and green (upstream 6ba4cb3)."""
    assert hex2bgr(np.array([0xFFFFFF])).tolist() == [[255, 255, 255]]
    assert hex2bgr(np.array([0x010101])).tolist() == [[1, 1, 1]]


def test_capitalize_sentences():
    assert capitalize_sentences('hello WORLD. "next ONE!" final?') == 'Hello world. "Next one!" Final?'
    assert capitalize_sentences('WHAT?! NO WAY') == 'What?! No way'
    assert capitalize_sentences('') == ''


def test_ocr_letter_case_modes_are_validated():
    from utils.config import ModuleConfig, OCRTextPostprocess
    assert ModuleConfig().ocr_text_postprocess == OCRTextPostprocess.NONE
    assert ModuleConfig(ocr_text_postprocess='nonsense').ocr_text_postprocess == OCRTextPostprocess.NONE
    assert ModuleConfig(ocr_text_postprocess='uppercase').ocr_text_postprocess == 'uppercase'


def test_filter_mask_by_bboxes_setting_exists():
    from utils.config import ModuleConfig
    assert ModuleConfig().filter_mask_by_bboxes is False


def test_model_downloads_do_not_use_dead_github_release_urls():
    """The manga-image-translator release assets moved to Hugging Face (upstream 4b10c18)."""
    from pathlib import Path
    dead = 'manga-image-translator/releases/download'
    offenders = []
    for name in ('modules/ocr/ocr_mit.py', 'modules/textdetector/detector_ctd.py',
                 'modules/inpaint/base.py', 'modules/prepare_local_files.py'):
        text = Path(name).read_text(encoding='utf-8')
        for line in text.splitlines():
            if dead in line and "'url'" in line:
                offenders.append(f'{name}: {line.strip()}')
    assert not offenders, offenders


@pytest.mark.parametrize('mode,text,expected', [
    ('none', 'hELLO there', 'hELLO there'),
    ('capitalize', 'hELLO there. bye', 'Hello there. Bye'),
    ('uppercase', 'hello there', 'HELLO THERE'),
])
def test_apply_ocr_letter_case(mode, text, expected, monkeypatch):
    from utils.config import pcfg
    from modules.ocr.base import apply_ocr_letter_case

    monkeypatch.setattr(pcfg.module, 'ocr_text_postprocess', mode, raising=False)
    blk = types.SimpleNamespace(text=[text])
    apply_ocr_letter_case(blk)
    assert blk.text == [expected]


def _load_adapt_unsupported_params():
    """Load LLMApiTranslator._adapt_unsupported_params without importing the heavy module."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path('modules/translators/trans_llm_api.py').read_text(encoding='utf-8'))
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == '_adapt_unsupported_params'
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, 'trans_llm_api.py', 'exec'), namespace)
    fn_obj = namespace['_adapt_unsupported_params']
    return getattr(fn_obj, '__func__', fn_obj)


def test_newer_openai_models_get_supported_parameters():
    """GPT-5 style models reject max_tokens/temperature (upstream #1251)."""
    adapt = _load_adapt_unsupported_params()
    args = {'model': 'gpt-5.5', 'messages': [], 'temperature': 0.2, 'top_p': 0.9, 'max_tokens': 2048}

    adapted, changed = adapt(args, Exception(
        "Error code: 400 - Unsupported parameter: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead."))
    assert 'max_tokens' not in adapted and adapted['max_completion_tokens'] == 2048
    assert 'max_completion_tokens' in changed

    adapted, _ = adapt(adapted, Exception(
        "Unsupported value: 'temperature' does not support 0.2 with this model. "
        "Only the default (1) value is supported."))
    assert 'temperature' not in adapted

    # Unrelated failures must propagate untouched.
    assert adapt(args, Exception('rate limit exceeded')) is None
