import ast
import copy
from pathlib import Path
from types import SimpleNamespace


def _load_reset_method_callable(fake_pcfg, fake_save_config):
    src_path = Path("ui/mainwindow.py")
    tree = ast.parse(src_path.read_text(encoding="utf-8"), filename=str(src_path))

    reset_fn = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "_reset_modules_to_core_defaults":
                    reset_fn = copy.deepcopy(child)
                    break
    assert reset_fn is not None, "_reset_modules_to_core_defaults not found"

    mod = ast.Module(body=[reset_fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    env = {
        "pcfg": fake_pcfg,
        "save_config": fake_save_config,
    }
    exec(compile(mod, str(src_path), "exec"), env)
    return env["_reset_modules_to_core_defaults"]


class _FallbackModuleManager:
    """Simulates module_manager fallback behavior when requested core defaults are unavailable."""

    def __init__(self):
        self.calls = []

    def setTextDetector(self, key):
        self.calls.append(("textdetector", key))

    def setOCR(self, key):
        # Simulate fallback away from manga_ocr
        self.calls.append(("ocr", "mit48px" if key == "manga_ocr" else key))

    def setInpainter(self, key):
        self.calls.append(("inpainter", key))

    def setTranslator(self, key):
        self.calls.append(("translator", key))



def test_reset_modules_to_core_defaults_delegates_to_module_manager_for_fallbacks():
    fake_pcfg = SimpleNamespace(module=SimpleNamespace(
        textdetector="old_detector",
        ocr="old_ocr",
        inpainter="old_inpainter",
        translator="old_translator",
    ))
    saved = {"count": 0}

    def fake_save_config():
        saved["count"] += 1

    reset = _load_reset_method_callable(fake_pcfg, fake_save_config)

    fake_self = SimpleNamespace(module_manager=_FallbackModuleManager())
    reset(fake_self)

    # Config is reset to core defaults first.
    assert fake_pcfg.module.textdetector == "ctd"
    assert fake_pcfg.module.ocr == "mit48px"
    assert fake_pcfg.module.inpainter == "aot"
    assert fake_pcfg.module.translator == "google"
    assert saved["count"] == 1

    # Runtime setter calls are delegated, allowing module_manager-level fallback behavior.
    assert fake_self.module_manager.calls == [
        ("textdetector", "ctd"),
        ("ocr", "mit48px"),
        ("inpainter", "aot"),
        ("translator", "google"),
    ]
