"""Module selection must survive restarts (issue #152).

Two rules are checked here without starting Qt:
  1. MainWindow._persist_module_selection writes the picked module to the config and
     schedules a save, but leaves the config alone for a fallback module.
  2. module_manager no longer overwrites the configured module when it has to fall
     back to another one (e.g. model files not downloaded yet).
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace


def _load_method(class_name: str, method_name: str, env: dict):
    src_path = Path("ui/mainwindow.py")
    tree = ast.parse(src_path.read_text(encoding="utf-8"), filename=str(src_path))

    fn = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    fn = copy.deepcopy(child)
                    break
    assert fn is not None, f"{class_name}.{method_name} not found"

    mod = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, str(src_path), "exec"), env)
    return env[method_name]


def _fake_self(pcfg):
    scheduled = []
    return SimpleNamespace(schedule_config_save=lambda *a, **k: scheduled.append(True)), scheduled


def test_persist_module_selection_saves_user_choice():
    pcfg = SimpleNamespace(module=SimpleNamespace(ocr="mit48px"))
    persist = _load_method("MainWindow", "_persist_module_selection", {"pcfg": pcfg})
    fake_self, scheduled = _fake_self(pcfg)

    persist(fake_self, "ocr", "manga_ocr")

    assert pcfg.module.ocr == "manga_ocr"
    assert scheduled, "picking a module should schedule a config save"


def test_persist_module_selection_ignores_fallback_module():
    pcfg = SimpleNamespace(module=SimpleNamespace(inpainter="lama_large_512px"))
    persist = _load_method("MainWindow", "_persist_module_selection", {"pcfg": pcfg})
    fake_self, scheduled = _fake_self(pcfg)

    persist(fake_self, "inpainter", "aot", is_fallback=True)

    assert pcfg.module.inpainter == "lama_large_512px"
    assert not scheduled


def test_persist_module_selection_skips_save_when_unchanged():
    pcfg = SimpleNamespace(module=SimpleNamespace(textdetector="ctd"))
    persist = _load_method("MainWindow", "_persist_module_selection", {"pcfg": pcfg})
    fake_self, scheduled = _fake_self(pcfg)

    persist(fake_self, "textdetector", "ctd")

    assert not scheduled


def test_module_manager_does_not_overwrite_config_on_fallback():
    src = Path("ui/module_manager.py").read_text(encoding="utf-8")
    for assignment in (
        "cfg_module.textdetector = fallback",
        "cfg_module.ocr = fallback",
        "cfg_module.inpainter = fallback",
        "cfg_module.translator = fallback",
    ):
        assert assignment not in src, (
            f"{assignment!r} discards the user's saved module selection when the "
            "configured module is temporarily unavailable (issue #152)"
        )
