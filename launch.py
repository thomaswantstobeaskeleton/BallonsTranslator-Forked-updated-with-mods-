from pathlib import Path
import sys
import argparse
import os.path as osp
import os
import importlib
import subprocess
import shlex
import warnings
import json
from platform import platform
from utils.gpu_runtime import (
    GPU_PROFILE_AUTO, GPU_PROFILE_CPU, GPU_PROFILE_NVIDIA_CUDA, GPU_PROFILE_AMD_DIRECTML,
    GPU_PROFILE_AMD_ROCM_PREVIEW, CPU_TORCH_COMMAND, build_gpu_install_plan, format_gpu_install_plan,
    has_amd_gpu, has_nvidia_gpu, classify_amd_gpu, detect_physical_gpus,
)

# Suppress requests' urllib3/chardet version mismatch warning (harmless for typical use)
warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")

# Disable Paddle oneDNN before any Paddle import (avoids ConvertPirAttribute2RuntimeAttribute error on Windows)
os.environ["FLAGS_use_mkldnn"] = "0"

# Skip slow "Checking connectivity to the model hosters" check (avoids long wait; set to "0" to re-enable)
if "DISABLE_MODEL_SOURCE_CHECK" not in os.environ:
    os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Prefer software-backed Qt rendering when system OpenGL is unavailable/misconfigured.
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

# Reduce verbose logging from Hugging Face / transformers (HTTP request lines, etc.)
import logging
for _name in (
    "huggingface_hub",
    "huggingface_hub._client",
    "huggingface_hub.file_download",
    "httpx",
    "httpcore",
    "transformers",
    "urllib3",
):
    logging.getLogger(_name).setLevel(logging.WARNING)

# PyTorch allocator fragmentation mitigation (Section 7: fewer CUDA OOMs from fragmentation)
if "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512"

# Faster HuggingFace downloads (Section 8): enable Xet when HF token is present
try:
    from utils.model_manager import enable_hf_xet_if_token_in_env
    enable_hf_xet_if_token_in_env()
except Exception:
    pass

BRANCH = 'main'
VERSION = '1.7.0'

python = sys.executable
git = os.environ.get('GIT', "git")
skip_install = False
index_url = os.environ.get('INDEX_URL', "")
QT_APIS = ['pyqt6', 'pyside6', 'pyqt5', 'pyside2']
stored_commit_hash = None

REQ_WIN = [
    'pywin32'
]

PATH_ROOT=Path(__file__).parent
PATH_FONTS=str(PATH_ROOT/'fonts')
FONT_EXTS = {'.ttf','.otf','.ttc','.pfb'}

IS_WIN7 = "Windows-7" in platform()

import utils.shared as shared # Earlier import of shared to use default for config_path argument

parser = argparse.ArgumentParser()
parser.add_argument("--reinstall-torch", action='store_true', help="launch.py argument: install the appropriate version of torch even if you have some version already installed")
parser.add_argument("--proj-dir", default='', type=str, help='Open project directory or .json project file on startup')
parser.add_argument("path", nargs='?', default='', help='Optional: path to project folder or .json file (same as --proj-dir)')
if IS_WIN7:
    parser.add_argument("--qt-api", default='pyqt5', choices=QT_APIS, help='Set qt api')
else:
    parser.add_argument("--qt-api", default='pyqt6', choices=QT_APIS, help='Set qt api')
parser.add_argument("--debug", action='store_true')
parser.add_argument("--requirements", default='requirements.txt')
parser.add_argument("--headless", action='store_true', help='run without GUI')
parser.add_argument("--headless_continuous", action='store_true', help='like headless but after finishing --exec_dirs prompts for new dirs (comma-separated) until you enter "exit"')
parser.add_argument("--exec_dirs", default='', help='translation queue (project directories) separated by comma')
parser.add_argument("--ldpi", default=None, type=float, help='logical dots perinch')
parser.add_argument("--export-translation-txt", action='store_true', help='save translation to txt file once RUN completed')
parser.add_argument("--export-source-txt", action='store_true', help='save source to txt file once RUN completed')
parser.add_argument("--frozen", action='store_true', help='run without checking requirements')
parser.add_argument("--update", action='store_true', help="Update the repository before launching") # Add argument --update
parser.add_argument("--config_path", default=shared.CONFIG_PATH, help='Config file to use for translation') # Named config_path to avoid conflict with existing name config
parser.add_argument('--nightly', action='store_true', help="Compatibility alias for --gpu-profile amd-rocm-preview")
parser.add_argument('--gpu-profile', default=os.environ.get('BT_GPU_PROFILE', GPU_PROFILE_AUTO), choices=[GPU_PROFILE_AUTO, GPU_PROFILE_CPU, GPU_PROFILE_NVIDIA_CUDA, GPU_PROFILE_AMD_DIRECTML, GPU_PROFILE_AMD_ROCM_PREVIEW], help='GPU runtime profile: auto, cpu, nvidia-cuda, amd-directml, or amd-rocm-preview')
parser.add_argument('--gpu-report', action='store_true', help='Print GPU runtime detection/install plan and exit')
args, _ = parser.parse_known_args()

if args.gpu_report:
    print(format_gpu_install_plan(build_gpu_install_plan(args.gpu_profile, nightly=args.nightly)))
    sys.exit(0)


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def _lrelease_candidates(ts_path: str, qm_path: str):
    return [
        ["pyside6-lrelease", ts_path, "-qm", qm_path],
        ["lrelease", ts_path, "-qm", qm_path],
        ["lrelease-qt6", ts_path, "-qm", qm_path],
        [sys.executable, "-m", "pylrelease6", ts_path, "-qm", qm_path],
        [sys.executable, "-m", "PySide6.scripts.pyside_tool", "lrelease", ts_path, "-qm", qm_path],
    ]


def ensure_qm_for_display_language(lang: str, translate_dir: str, logger=None) -> bool:
    """
    Ensure `<translate_dir>/<lang>.qm` exists.
    If missing and `<translate_dir>/<lang>.ts` exists, try compiling it with available lrelease tooling.
    Returns True when qm exists (already existed or compiled), otherwise False.
    """
    if not lang or lang in ('en_US', 'English'):
        return True

    qm_path = osp.join(translate_dir, f"{lang}.qm")
    if osp.exists(qm_path):
        return True

    ts_path = osp.join(translate_dir, f"{lang}.ts")
    if not osp.exists(ts_path):
        return False

    candidates = _lrelease_candidates(ts_path, qm_path)

    log = logger.info if logger is not None else print
    log(f"Display language qm missing for {lang}; attempting to compile {ts_path}.")

    for cmd in candidates:
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except FileNotFoundError:
            continue
        except Exception as exc:
            if logger is not None:
                logger.debug("lrelease command failed: %s | err=%s", cmd, exc, exc_info=True)
            continue

        if result.returncode == 0 and osp.exists(qm_path):
            log(f"Compiled display language qm successfully: {qm_path}")
            return True
        if logger is not None:
            logger.debug(
                "lrelease attempt failed: cmd=%s rc=%s stdout=%s stderr=%s",
                cmd, result.returncode, (result.stdout or "").strip(), (result.stderr or "").strip()
            )
    return osp.exists(qm_path)


def install_qt_base_translators(app, lang: str, logger=None) -> list:
    """
    Install Qt's own translations (qtbase/qt) for `lang` so standard widgets --
    QMessageBox buttons such as "Show details...", QDialogButtonBox, QColorDialog, ... --
    are localized instead of staying English (issue #150).

    Returns the list of installed QTranslator objects; the caller must keep a reference
    to them, otherwise Qt drops the translations when they are garbage collected.
    """
    installed = []
    if not lang or lang in ('en_US', 'English'):
        return installed

    from qtpy.QtCore import QTranslator, QLibraryInfo

    try:
        if hasattr(QLibraryInfo, 'path'):
            translations_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        else:   # Qt5
            translations_path = QLibraryInfo.location(QLibraryInfo.LibraryLocation.TranslationsPath)
    except Exception as exc:
        if logger is not None:
            logger.debug('Could not resolve Qt translations path: %s', exc)
        return installed

    if not translations_path or not osp.isdir(translations_path):
        return installed

    # e.g. zh_CN -> ('zh_CN', 'zh'); Qt ships some catalogs per language only.
    candidates = [lang]
    if '_' in lang:
        candidates.append(lang.split('_', 1)[0])

    for catalog in ('qtbase', 'qt'):
        for code in candidates:
            if not osp.isfile(osp.join(translations_path, f'{catalog}_{code}.qm')):
                continue
            translator = QTranslator()
            if translator.load(f'{catalog}_{code}', translations_path) and app.installTranslator(translator):
                installed.append(translator)
                if logger is not None:
                    logger.info(f'Loaded Qt standard translations: {catalog}_{code}.qm')
                break

    if not installed and logger is not None:
        logger.debug(f'No Qt standard translation catalog found for {lang} in {translations_path}.')
    return installed


def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    if desc is not None:
        print(desc)

    argv = shlex.split(command) if isinstance(command, str) else command

    if live:
        result = subprocess.run(argv, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def run_pip(args, desc=None):
    if skip_install:
        return

    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {args} --prefer-binary{index_url_line} --disable-pip-version-check --no-warn-script-location', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=True)


def commit_hash():
    global stored_commit_hash

    if stored_commit_hash is not None:
        return stored_commit_hash

    try:
        stored_commit_hash = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        stored_commit_hash = "<none>"

    return stored_commit_hash


BT = None
APP = None

def restart():
    global BT
    print('restarting...\n')
    if BT:
        BT.close()
    os.execv(sys.executable, ['python'] + sys.argv)


def setup_locks():
    from utils.lock import RUNTIME_LOCKS
    from qtpy.QtCore import QMutex
    RUNTIME_LOCKS['model_loading'] = QMutex()


def main():

    if args.debug:
        os.environ['BALLOONTRANS_DEBUG'] = '1'

    os.environ['QT_API'] = args.qt_api

    # Faster HuggingFace downloads (Section 8): enable Xet when HF token is present in env
    try:
        from utils.model_manager import enable_hf_xet_if_token_in_env
        enable_hf_xet_if_token_in_env()
    except Exception:
        pass

    commit = commit_hash()

    print('Python version: ', sys.version)
    print('Python executable: ', sys.executable)
    print(f'Version: {VERSION}')
    print(f'Branch: {BRANCH}')
    print(f"Commit hash: {commit}")

    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(APP_DIR)

    prepare_environment()

    from utils.zluda_config import enable_zluda_config
    enable_zluda_config()

    if args.update:
        if getattr(sys, 'frozen', False):
            print('Running as app, skipping update.')
        else:
            print('Checking for updates...')
            try:
                current_commit = commit_hash()
                run(f"{git} fetch origin {BRANCH}", desc="Fetching updates from git...", errdesc="Failed to fetch updates.")
                latest_commit = run(f"{git} rev-parse origin/{BRANCH}").strip()

                if current_commit != latest_commit:
                    print("New updates found. Updating repository...")
                    run(f"{git} pull origin {BRANCH}", desc="Updating repository...", errdesc="Failed to update repository.")
                    print("Repository updated. Restarting to apply updates...")
                    restart()
                    return
                else:
                    print("No updates found.")
            except Exception as e:
                print(f"Update check failed: {e}")
                print("Continuing with the current version.")


    from utils.logger import setup_logging, logger as LOGGER
    from utils.io_utils import find_all_files_recursive
    from utils import config as program_config
    from utils.model_packages import validate_manifest_on_startup

    from qtpy.QtCore import QTranslator, QLocale, Qt
    shared.args = args
    # Use system language for default display language; fallback by language when exact locale .qm missing (#1145)
    sys_locale = QLocale.system().name().replace('en_CN', 'zh_CN')
    trans_dir = shared.TRANSLATE_DIR
    # Prefer system script over region: e.g. Simplified Chinese in Macau should get zh_CN, not zh_TW
    try:
        sys_script = QLocale.system().script()
        traditional_codes = ('zh_TW', 'zh_HK', 'zh_MO')
        simplified_script = getattr(QLocale.Script, 'SimplifiedChineseScript', None)
        if simplified_script is not None and sys_script == simplified_script and sys_locale in traditional_codes:
            if osp.exists(osp.join(trans_dir, 'zh_CN.qm')):
                sys_locale = 'zh_CN'
    except Exception:
        pass
    # Without a matching .qm the app is English anyway; keeping the raw locale name
    # (e.g. "C" on a system without a locale) only produces a missing-file warning
    # and a display language the selector cannot show.
    untranslated_fallback = 'English'
    if osp.exists(osp.join(trans_dir, sys_locale + '.qm')):
        shared.DEFAULT_DISPLAY_LANG = sys_locale
    else:
        try:
            lang = QLocale.system().language()
            if lang == QLocale.Language.Chinese:
                for code in ('zh_CN', 'zh_TW'):
                    if osp.exists(osp.join(trans_dir, code + '.qm')):
                        shared.DEFAULT_DISPLAY_LANG = code
                        break
                else:
                    shared.DEFAULT_DISPLAY_LANG = untranslated_fallback
            else:
                shared.DEFAULT_DISPLAY_LANG = untranslated_fallback
        except Exception:
            shared.DEFAULT_DISPLAY_LANG = untranslated_fallback
    shared.HEADLESS = args.headless
    shared.HEADLESS_CONTINUOUS = getattr(args, 'headless_continuous', False)
    shared.load_cache()
    program_config.load_config(args.config_path)
    config = program_config.pcfg

    if args.headless or getattr(args, 'headless_continuous', False):
        config.module.load_model_on_demand = True
        config.module.empty_runcache = False

    if sys.platform == 'win32':
        import ctypes
        myappid = u'BallonsTranslatorPro'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    import qtpy
    from qtpy.QtWidgets import QApplication
    from qtpy.QtGui import QIcon, QFontDatabase, QGuiApplication, QFont
    from qtpy.QtCore import qInstallMessageHandler, QtMsgType
    from qtpy import API, QT_VERSION

    shared.STARTUP_HEALTH = []
    def _stage(name: str, status: str = "ok", details: str = ""):
        try:
            shared.STARTUP_HEALTH.append({"stage": name, "status": status, "details": details})
        except Exception:
            pass
    LOGGER.info(f'QT_API: {API}, QT Version: {QT_VERSION}')
    _stage("Qt ready", "ok", f"{API} / Qt {QT_VERSION}")

    shared.DEBUG = args.debug
    shared.USE_PYSIDE6 = API == 'pyside6'
    if qtpy.API_NAME[-1] == '6':
        shared.FLAG_QT6 = True
    else:
        shared.FLAG_QT6 = False
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True) #enable high dpi scaling
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True) #use high dpi icons
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    os.chdir(shared.PROGRAM_PATH)

    setup_logging(shared.LOGGING_PATH)
    if not validate_manifest_on_startup():
        LOGGER.warning('Model manifest validation failed; falling back to built-in model package definitions.')

    from utils.logger import apply_dev_mode_logging
    apply_dev_mode_logging(getattr(config, 'dev_mode', False))

    app_args = sys.argv
    if args.headless or getattr(args, 'headless_continuous', False):
        app_args = sys.argv + ['-platform', 'offscreen']
    # Keep dialogs Qt-rendered so they inherit app stylesheet in dark/light modes.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)
    app = QApplication(app_args)
    # Filter noisy Qt warnings (QPainter / QFont spam) so the console stays readable.
    # This does NOT change behavior, only hides repeated benign messages.
    def _qt_message_filter(msg_type, context, message):
        try:
            text = str(message)
        except Exception:
            text = message if isinstance(message, str) else repr(message)
        # Known noisy-but-benign messages we want to hide
        _suppress_substrings = (
            "QPainter::begin: A paint device can only be painted by one painter at a time.",
            "QPainter::begin: Painter already active",
            "Painter not active",
            "Unbalanced save/restore",
            "QWidgetEffectSourcePrivate::pixmap",
            "QFont::setPointSize: Point size <= 0",
        )
        for sub in _suppress_substrings:
            if sub in text:
                return
        # Otherwise, forward to stderr (approximate default handler)
        try:
            sys.__stderr__.write(text + "\n")
        except Exception:
            pass
    qInstallMessageHandler(_qt_message_filter)
    app.setApplicationName('BallonsTranslatorPro')
    app.setApplicationVersion(VERSION)
    app.setWindowIcon(QIcon(shared.ICON_PATH))

    # import msl.loadlib (required by translators/trans_eztrans) before init QApplication
    # yield QWindowsContext: OleInitialize() failed on py3.10, 
    from modules.base import init_module_registries
    from modules.prepare_local_files import prepare_local_files_forall
    init_module_registries()

    # First launch: let user choose which model packages to download (Issue #15)
    if getattr(shared, 'FIRST_RUN_NO_CONFIG', False) and not args.headless and not getattr(args, 'headless_continuous', False):
        from ui.model_package_selector_dialog import ModelPackageSelectorDialog
        dialog = ModelPackageSelectorDialog()
        dialog.exec()
        config.model_packages_enabled = dialog.get_selected_package_ids()
        config.offline_local_only_mode = dialog.is_offline_local_only_selected()
        if config.offline_local_only_mode:
            LOGGER.info("First-run mode selected: local-only (skip all model downloads).")
        config.model_package_preset_ids = dialog.get_selected_preset_ids()
        try:
            from utils.config import save_config
            save_config()
        except Exception:
            pass
        shared.FIRST_RUN_NO_CONFIG = False
        # Force welcome/start screen after first-setup flow so users land on onboarding UI.
        shared.FORCE_SHOW_WELCOME_ON_STARTUP = True

    # Download selected model packages; defer to after window is shown (GUI) so user can retry from Tools if it fails
    if args.headless or getattr(args, 'headless_continuous', False):
        LOGGER.info('Downloading selected model packages (this may take a few minutes)...')
        prepare_local_files_forall()
        _stage("Model check", "ok", "headless download path")
    else:
        shared.DEFER_INITIAL_MODEL_DOWNLOAD = True
        _stage("Model check", "ok", "deferred until UI shown")

    if not args.headless and not getattr(args, 'headless_continuous', False):
        ps = QGuiApplication.primaryScreen()
        shared.LDPI = ps.logicalDotsPerInch()
        shared.SCREEN_W = ps.geometry().width()
        shared.SCREEN_H = ps.geometry().height()

    lang = config.display_lang
    ensure_qm_for_display_language(lang, shared.TRANSLATE_DIR, LOGGER)
    langp = osp.join(shared.TRANSLATE_DIR, lang + '.qm')
    if not osp.exists(langp) and lang.startswith('zh_'):
        for code in ('zh_CN', 'zh_TW'):
            if code != lang and osp.exists(osp.join(shared.TRANSLATE_DIR, code + '.qm')):
                lang = code
                langp = osp.join(shared.TRANSLATE_DIR, lang + '.qm')
                break
    # Keep references on the app object: Qt drops translations that get garbage collected.
    app._bt_translators = install_qt_base_translators(app, lang, LOGGER)
    if osp.exists(langp):
        translator = QTranslator()
        translator.load(lang, osp.dirname(osp.abspath(__file__)) + "/translate")
        app.installTranslator(translator)
        app._bt_translators.append(translator)
    elif lang not in ('en_US', 'English'):
        LOGGER.warning(f'target display language file {langp} doesnt exist.')
    LOGGER.info(f'set display language to {lang}')
    _stage("Display language", "ok", str(lang))

    # Fonts
    # Load custom fonts if they exist
    if osp.exists(PATH_FONTS):
        for fp in find_all_files_recursive(PATH_FONTS, FONT_EXTS):
            fnt_idx = QFontDatabase.addApplicationFont(fp)
            if fnt_idx >= 0:
                shared.CUSTOM_FONTS.append(QFontDatabase.applicationFontFamilies(fnt_idx)[0])

    if sys.platform == 'win32' and (args.headless or getattr(args, 'headless_continuous', False)):
        # font database does not initialise on windows with qpa -offscreen:
        # whttps://github.com/dmMaze/BallonsTranslator/issues/519
        from qtpy.QtCore import QStandardPaths
        font_dir_list = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.FontsLocation)
        for fd in font_dir_list:
            fp_list = find_all_files_recursive(fd, FONT_EXTS)
            for fp in fp_list:
                fnt_idx = QFontDatabase.addApplicationFont(fp)

    if shared.FLAG_QT6:
        shared.FONT_FAMILIES = set(f for f in QFontDatabase.families())
    else:
        fdb = QFontDatabase()
        shared.FONT_FAMILIES = set(fdb.families())

    app_font = QFont('Microsoft YaHei UI')  # default UI font name (fallback: system font)
    if not app_font.exactMatch() or sys.platform == 'darwin':
        app_font = app.font()
    if app_font.pointSizeF() <= 0:
        app_font.setPointSizeF(10.0)
    if app_font.pointSize() <= 0:
        app_font.setPointSize(10)
    app_font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias)
    # Patch app setFont so any font with point size <= 0 is sanitized (stops QFont::setPointSize spam from any caller).
    _original_app_set_font = QGuiApplication.setFont
    def _app_set_font_sanitized(*args, **kwargs):
        # Can be called as app.setFont(font) -> (self, font) or QGuiApplication.setFont(font) -> (font,)
        if args and hasattr(args[0], 'pointSizeF'):
            font = args[0]
            rest = ()
        elif len(args) >= 2:
            font = args[1]
            rest = (args[0],)
        else:
            return _original_app_set_font(*args, **kwargs)
        if font.pointSizeF() <= 0 or font.pointSize() <= 0:
            f = QFont(font)
            f.setPointSizeF(10.0)
            f.setPointSize(10)
            font = f
        return _original_app_set_font(*(rest + (font,)), **kwargs)
    QGuiApplication.setFont = _app_set_font_sanitized
    QGuiApplication.setFont(app_font)
    shared.DEFAULT_FONT_FAMILY = app_font.family()
    shared.APP_DEFAULT_FONT = app_font.family()
    # Patch QWidget.setFont so any font with point size <= 0 is sanitized before reaching Qt (stops QFont::setPointSize spam).
    from qtpy.QtWidgets import QWidget as _QWidget
    _original_set_font = _QWidget.setFont
    def _set_font_sanitized(self, font):
        if font.pointSizeF() <= 0 or font.pointSize() <= 0:
            f = QFont(font)
            f.setPointSizeF(10.0)
            f.setPointSize(10)
            font = f
        return _original_set_font(self, font)
    _QWidget.setFont = _set_font_sanitized
    # Patch QFont.setPointSize/setPointSizeF so any caller passing <= 0 (e.g. Windows default -1) never triggers Qt warning.
    _original_set_pt = QFont.setPointSize
    _original_set_ptf = QFont.setPointSizeF
    def _set_point_size_clamped(self, size):
        return _original_set_pt(self, max(1, int(size)))
    def _set_point_size_f_clamped(self, size):
        return _original_set_ptf(self, max(1.0, float(size)))
    QFont.setPointSize = _set_point_size_clamped
    QFont.setPointSizeF = _set_point_size_f_clamped
    # Note: Qt6 removed QFont/QFontDatabase insertSubstitution; use a CJK-capable font in text style if you see empty squares (□).

    if args.ldpi is not None:
        shared.LDPI = args.ldpi
    elif getattr(config, 'logical_dpi', 0) > 0:
        shared.LDPI = float(config.logical_dpi)

    setup_locks()

    from ui.mainwindow import MainWindow
    open_path = (args.proj_dir or getattr(args, 'path', '') or '').strip()
    ballontrans = MainWindow(app, config, open_dir=open_path, **vars(args))
    # Destroy the Qt window tree before SIP performs interpreter-exit cleanup,
    # which otherwise crashes on quit (upstream dmMaze/BallonsTranslator@f313daa).
    delete_on_close = getattr(Qt, 'WidgetAttribute', Qt).WA_DeleteOnClose
    ballontrans.setAttribute(delete_on_close, True)
    global BT
    BT = ballontrans
    BT.restart_signal.connect(restart)

    if not args.headless and not getattr(args, 'headless_continuous', False):
        if shared.SCREEN_W > 1707 and sys.platform == 'win32':
            # https://github.com/dmMaze/BallonsTranslator/issues/220
            BT.comicTransSplitter.setHandleWidth(7)

        ballontrans.setWindowIcon(QIcon(shared.ICON_PATH))
        ballontrans.show()
        ballontrans.resetStyleSheet()
        _stage("UI shown", "ok", "mainwindow.show()")

        # Optional: offer Windows context menu on first launch (once per config)
        if sys.platform == 'win32' and not getattr(config, 'windows_context_menu_offered', False):
            from qtpy.QtCore import QTimer
            from qtpy.QtWidgets import QMessageBox

            def offer_context_menu():
                reply = QMessageBox.question(
                    ballontrans,
                    "Context menu",
                    "Add \"Open in BallonsTranslator\" to the right-click context menu for .json files and folders?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                config.windows_context_menu_offered = True
                try:
                    from utils.config import save_config
                    save_config()
                except Exception:
                    pass
                if reply == QMessageBox.StandardButton.Yes:
                    from utils.windows_context_menu import install as install_context_menu
                    ok, msg = install_context_menu()
                    if ok:
                        QMessageBox.information(ballontrans, "Context menu", msg)
                    else:
                        QMessageBox.warning(ballontrans, "Context menu", msg)

            QTimer.singleShot(500, offer_context_menu)

    # Let this frame release Qt objects before SIP's interpreter-exit cleanup
    # (upstream dmMaze/BallonsTranslator@d249515).
    return app.exec()

def is_amd_gpu():
    return has_amd_gpu(detect_physical_gpus())


def supported_amd_nightly_gpu():
    return classify_amd_gpu(detect_physical_gpus()) or "None"


def is_nvidia_gpu_present():
    """Best-effort physical NVIDIA GPU probe before importing PyTorch."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return False
    return has_nvidia_gpu(detect_physical_gpus())


def _probe_installed_torch():
    """Return installed torch diagnostics without importing torch in this launcher.

    Importing torch in the parent process before replacing wheels can keep
    torch/torchvision DLLs loaded on Windows. That may leave a half-upgraded
    install and cause entry-point errors such as ``torch_cuda.dll`` or
    ``torchvision/_C.pyd`` symbol failures after the CUDA wheel is installed.
    Probe in a short-lived child process instead so pip can safely uninstall
    the old wheel.
    """
    probe_code = """
import json
import torch
payload = {
    'ok': True,
    'version': getattr(torch, '__version__', None),
    'cuda': getattr(getattr(torch, 'version', None), 'cuda', None),
    'hip': getattr(getattr(torch, 'version', None), 'hip', None),
    'cuda_available': bool(torch.cuda.is_available()),
}
print(json.dumps(payload))
"""
    try:
        result = subprocess.run(
            [python, "-c", probe_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            env=os.environ,
        )
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if result.returncode != 0:
        return {
            "ok": False,
            "error": (result.stderr or result.stdout or f"probe exited with {result.returncode}").strip(),
        }
    try:
        return json.loads((result.stdout or "{}").strip().splitlines()[-1])
    except Exception as exc:
        return {"ok": False, "error": f"invalid probe output: {exc}; stdout={result.stdout!r}"}


def installed_torch_is_cpu_only():
    """Return True when installed torch cannot expose CUDA/HIP-style devices."""
    info = _probe_installed_torch()
    if not info.get("ok"):
        print(f"Unable to inspect installed PyTorch without importing it: {info.get('error')}")
        return True
    return not bool(info.get("cuda_available"))


def prepare_environment():

    try:
        import packaging
    except ModuleNotFoundError:
        run_pip(f"install packaging", "install packaging")

    from utils.package import check_req_file, check_reqs

    if getattr(sys, 'frozen', False):
        print('Running as app, skip dependency installation')
        return

    if args.frozen:
        return

    req_updated = False
    if sys.platform == 'win32':
        for req in REQ_WIN:
            if not check_reqs([req]):
                run_pip(f"install {req}", req)
                req_updated = True

    gpu_plan = build_gpu_install_plan(args.gpu_profile, nightly=args.nightly)
    print(format_gpu_install_plan(gpu_plan))
    torch_command = gpu_plan.torch_command
    cuda_torch_reinstall_needed = (
        not os.environ.get("BT_SKIP_CUDA_TORCH_REINSTALL")
        and os.environ.get("BT_CUDA_TORCH_REINSTALL_ATTEMPTED") != "1"
        and gpu_plan.resolved_profile == GPU_PROFILE_NVIDIA_CUDA
        and is_nvidia_gpu_present()
        and is_installed("torch")
        and installed_torch_is_cpu_only()
    )
    rocm_torch_reinstall_needed = (
        not os.environ.get("BT_SKIP_AMD_ROCM_TORCH_REINSTALL")
        and os.environ.get("BT_AMD_ROCM_TORCH_REINSTALL_ATTEMPTED") != "1"
        and gpu_plan.resolved_profile == GPU_PROFILE_AMD_ROCM_PREVIEW
        and is_installed("torch")
        and installed_torch_is_cpu_only()
    )
    if cuda_torch_reinstall_needed:
        print("NVIDIA GPU detected but installed PyTorch is CPU-only or cannot use CUDA; installing CUDA-enabled PyTorch.")
        os.environ["BT_CUDA_TORCH_REINSTALL_ATTEMPTED"] = "1"
    if rocm_torch_reinstall_needed:
        print("AMD ROCm preview profile selected but installed PyTorch cannot use a HIP/CUDA-style device; installing AMD ROCm preview PyTorch.")
        os.environ["BT_AMD_ROCM_TORCH_REINSTALL_ATTEMPTED"] = "1"

    required_missing = any(not is_installed(pkg) for pkg in gpu_plan.required_packages)
    if args.reinstall_torch or required_missing or cuda_torch_reinstall_needed or rocm_torch_reinstall_needed:
        install_torch_command = torch_command
        if (cuda_torch_reinstall_needed or rocm_torch_reinstall_needed) and "TORCH_COMMAND" not in os.environ and "--force-reinstall" not in install_torch_command:
            install_torch_command = f"{install_torch_command} --force-reinstall"
        try:
            run(f'"{python}" -m {install_torch_command}', "Installing GPU runtime packages", "Couldn't install GPU runtime packages", live=True)
        except RuntimeError as exc:
            # The GPU wheel index may not publish a build for this interpreter (issue #154).
            # Don't abort startup: fall back to the CPU wheels so the app still runs.
            if gpu_plan.resolved_profile == GPU_PROFILE_CPU or "TORCH_COMMAND" in os.environ:
                raise
            print(exc)
            print(
                "GPU PyTorch installation failed (no wheel for this Python version or index unreachable). "
                "Falling back to CPU PyTorch wheels; GPU acceleration will be unavailable.\n"
                "To use the GPU, install a Python version supported by the GPU wheels (3.12 is recommended) "
                "or set TORCH_COMMAND to a pip command that works on this interpreter."
            )
            run(f'"{python}" -m {CPU_TORCH_COMMAND}', "Installing CPU runtime packages", "Couldn't install CPU runtime packages", live=True)
        req_updated = True

    if not check_req_file(args.requirements):
        run_pip(f"install -r {args.requirements}", "requirements")
        req_updated = True

    if req_updated:
        import site
        importlib.reload(site)





if __name__ == '__main__':
    sys.exit(main())
