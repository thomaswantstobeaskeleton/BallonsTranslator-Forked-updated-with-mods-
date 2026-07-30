import os.path as osp
import os, re, traceback, sys
import queue
import threading
import shutil
import copy
import numpy as np
import json
import time
from typing import List, Union
from pathlib import Path
import subprocess
from functools import partial
import time
import cv2
from urllib import request as urlrequest
from urllib import error as urlerror

from tqdm import tqdm
from qtpy.QtWidgets import QAction, QFileDialog, QMenu, QHBoxLayout, QVBoxLayout, QApplication, QStackedWidget, QSplitter, QListWidget, QShortcut, QListWidgetItem, QMessageBox, QTextEdit, QPlainTextEdit, QDialog, QProgressBar, QLabel, QWidget, QInputDialog, QFormLayout, QDialogButtonBox, QLineEdit, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox, QPushButton, QSizePolicy
from qtpy.QtCore import Qt, QPoint, QSize, QEvent, Signal, QTimer, QThread
from qtpy.QtGui import QContextMenuEvent, QTextCursor, QGuiApplication, QIcon, QCloseEvent, QKeySequence, QKeyEvent, QPainter, QClipboard, QImage, QShowEvent, QCursor, QPixmap, QBrush, QColor

from utils.logger import logger as LOGGER
from utils.text_processing import is_cjk, full_len, half_len
from utils.detect_layout_flags import should_enable_auto_textlayout, should_run_post_detect_autofit
from utils.text_layout import _merge_stub_lines_in_string
from utils.textblock import TextBlock, TextAlignment, examine_textblk
from utils.credential_store import get_secret, set_secret, has_keyring
from utils.provider_setup import check_provider_connection, provider_endpoint_preset
from utils.split_text_region import split_textblock
from utils import shared
from utils.message import create_error_dialog, create_info_dialog
from modules.translators.trans_chatgpt import GPTTranslator
from modules.translators.base import lang_display_to_key
from modules import GET_VALID_TEXTDETECTORS, GET_VALID_INPAINTERS, GET_VALID_TRANSLATORS, GET_VALID_OCR
from .misc import parse_stylesheet, set_html_family, QKEY, pixmap2ndarray
from .custom_widget.animated_stack import AnimatedStackWidget
from utils.config import ProgramConfig, pcfg, save_config, text_styles, save_text_styles, load_textstyle_from, FontFormat, log_diagnostic_event
from utils.layout_review_agent import ReviewModelConfig, PageReviewResult, BlockReviewResult, ReviewIssue, ReviewAction
from utils.shortcuts import get_shortcut, is_single_key_sequence, shortcut_should_ignore_text_input
from utils.proj_imgtrans import ProjImgTrans
from utils.zip_batch import ZipBatchManager
from .canvas import Canvas
from .configpanel import ConfigPanel
from .translation_context_dialog import TranslationContextDialog
from .module_manager import ModuleManager
from .textedit_area import SourceTextEdit, SelectTextMiniMenu, TransTextEdit
from .drawingpanel import DrawingPanel
from .scenetext_manager import SceneTextManager, TextPanel, PasteSrcItemsCommand
from .mainwindowbars import TitleBar, LeftBar, BottomBar
from .io_thread import ImgSaveThread, ImportDocThread, ExportDocThread, GitUpdateThread, ModelDownloadThread
from .custom_widget import Widget, ViewWidget
from .global_search_widget import GlobalSearchWidget
from .textedit_commands import GlobalRepalceAllCommand, ApplyTranslationCandidateCommand, MergeBlkItemsCommand
from .framelesswindow import FramelessWindow, FramelessMoveResize
from .drawing_commands import RunBlkTransCommand
from .keywordsubwidget import KeywordSubWidget
from . import shared_widget as SW
from .custom_widget import MessageBox, FrameLessMessageBox, ImgtransProgressMessageBox
from .model_manager_dialog import ModelManagerDialog
from .shortcuts_dialog import ShortcutsDialog
from .context_menu_config_dialog import ContextMenuConfigDialog
from .batch_queue_dialog import BatchQueueDialog
from .export_dialog import ExportFormatDialog
from .spellcheck_panel import SpellCheckPanel
from .google_font_dialog import GoogleFontInstallDialog
from .pipeline_insights_widget import PipelineInsightsWidget
from .image_edit import ImageEditMode
from .mask_diagnostics_widget import MaskDiagnosticsWidget
from .ocr_crop_inspector_widget import OcrCropInspectorWidget
from .compare_results_dialog import CompareResultsDialog
from .project_ops_dialog import ProjectOpsDialog
from .reading_order_editor_dialog import ReadingOrderEditorDialog
from .layout_review_report_dialog import LayoutReviewReportDialog
from .lettering_workflow_dialog import LetteringWorkflowDialog
from .auto_format_qa_widget import AutoFormatQADialog
from .realtime_translator_dialog import RealtimeTranslatorDialog
from .translation_assist_dock import TranslationAssistDock
from utils.fontformat import pt2px
from utils.image_colorization import apply_colorization
from utils.structured_ocr_export import build_structured_ocr_export
from utils.xliff_interchange import export_project_xliff, import_project_xliff
from utils.translation_json_interchange import export_translation_json, import_translation_json
from utils.translation_csv_interchange import export_translation_csv_text, import_translation_csv_text
from utils.translation_memory import query_tm, add_tm_entry, build_tm_from_project, export_tm_payload, import_tm_payload
from utils.translation_qa_profiles import build_translation_qa_report, PROMPT_PROFILES
from utils.translation_concordance import build_concordance_from_project, query_concordance
from utils.renderer_diagnostics import collect_renderer_diagnostics
from utils.cleanup_only_workflow import run_cleanup_only_pages
from utils.data_path_manager import describe_data_path, migrate_data_path
from utils.server_mode_info import build_server_mode_info
from utils.translated_image_alignment import align_translations_by_iou
from utils.batch_parent_queue import enumerate_child_projects, save_parent_batch_state, load_parent_batch_state, update_parent_batch_status, next_pending_child, summarize_parent_batch_state
from utils.glossary_io import export_glossary_csv, import_glossary_csv, preview_glossary_merge
from utils.sfx_dictionary import default_sfx_dictionary, query_sfx_dictionary, merge_sfx_entries, export_sfx_dictionary, import_sfx_dictionary
from utils.glossary_extraction import extract_glossary_candidates
from utils.batch_find_replace import preview_batch_find_replace, apply_batch_find_replace
from utils.image_transform_ops import TransformOptions, transform_image_file
from utils.layered_psd_export import build_layered_psd_handoff
from utils.svg_text_export import build_svg_text_handoff
from utils.lettering_proof_export import build_lettering_proof_pack
from utils.archive_stream_export import write_archive_streaming
from utils.auto_format_qa import score_auto_format_candidates, summarize_auto_format_scores
from utils.text_rendering import locale_aware_upper
from utils.local_automation_api import LocalAutomationApiServer
from utils.automation_api_contract import normalize_job_task
from utils.docs_catalog import build_docs_catalog
from utils.automation_jobs import new_job, append_log, add_warning, set_status, checkpoint_or_cancel, status_payload
from utils.workflow_presets import apply_workflow_preset, list_workflow_presets, workflow_stage_vector
from utils.model_manager import get_available_module_keys
from utils.realtime_mode import RealtimeService, RealtimeRegion
from utils.translation_assist import (
    TranslationAssistService,
    build_candidates_from_sources,
    normalize_provider_warning,
    estimate_candidate_cost_usd,
    make_synthetic_mt_candidate,
)
from ui.translation_assist_worker import CompareWorker
from modules.llm_quality import enforce_glossary, back_translation_drift_score
from modules.text_normalization import normalize_text
from modules.text_replace_profiles import apply_profile, default_profiles


class PageListView(QListWidget):

    reveal_file = Signal()
    remove_images = Signal(list)
    translate_images = Signal(list)
    run_ocr_images = Signal(list)
    run_translate_images = Signal(list)
    run_inpaint_images = Signal(list)
    run_detect_images = Signal(list)
    toggle_ignore_requested = Signal(list, bool)  # (pagenames, ignored)
    set_completion_requested = Signal(list, str)  # (pagenames, todo|translated|reviewed|exported)
    lettering_workflow_requested = Signal(list)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setIconSize(QSize(shared.PAGELIST_THUMBNAIL_SIZE, shared.PAGELIST_THUMBNAIL_SIZE))
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)


    def _item_page_name(self, item: QListWidgetItem) -> str:
        if item is None:
            return ''
        return item.data(Qt.ItemDataRole.UserRole) or item.text()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_A and event.modifiers() in (Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.MetaModifier):
            self.selectAll()
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, e: QContextMenuEvent):
        menu = QMenu()
        reveal_act = menu.addAction(self.tr('Reveal in File Explorer'))
        selected_items = self.selectedItems()
        n_total = self.count()
        select_all_pages_act = None
        if n_total > 0:
            menu.addSeparator()
            select_all_pages_act = menu.addAction(self.tr('Select all pages'))
        if selected_items:
            menu.addSeparator()
            translate_act = menu.addAction(self.tr('Translate Selected Images'))
            run_detect_act = menu.addAction(self.tr('Run detection on selected pages'))
            run_ocr_act = menu.addAction(self.tr('Run OCR on selected pages'))
            run_translate_act = menu.addAction(self.tr('Run translation on selected pages'))
            run_inpaint_act = menu.addAction(self.tr('Run inpainting on selected pages'))
            menu.addSeparator()
            ignore_act = menu.addAction(self.tr('Ignore in run'))
            ignore_act.setToolTip(self.tr('Skip these pages in full run and batch (when "Skip ignored pages" is on).'))
            include_act = menu.addAction(self.tr('Include in run'))
            include_act.setToolTip(self.tr('Do not skip these pages in full run and batch.'))
            menu.addSeparator()
            lettering_workflow_act = menu.addAction(self.tr('Lettering workflow for selected pages...'))
            lettering_workflow_act.setToolTip(self.tr('Plan/apply typography polish, smart fit, layout review, and proof/render steps for these pages.'))
            menu.addSeparator()
            completion_menu = menu.addMenu(QIcon(osp.join(shared.PROGRAM_PATH, 'icons', 'page_completion.svg')), self.tr('Page completion state'))
            mark_todo_act = completion_menu.addAction(self.tr('Needs work'))
            mark_translated_act = completion_menu.addAction(self.tr('Translated'))
            mark_reviewed_act = completion_menu.addAction(self.tr('Reviewed'))
            mark_exported_act = completion_menu.addAction(self.tr('Exported'))
            menu.addSeparator()
            remove_act = menu.addAction(self.tr('Remove from Project'))
        rst = menu.exec_(e.globalPos())

        if rst == reveal_act:
            self.reveal_file.emit()
        elif rst == select_all_pages_act:
            self.selectAll()
        elif selected_items and rst == translate_act:
            self.translate_images.emit([self._item_page_name(item) for item in selected_items])
        elif selected_items and 'lettering_workflow_act' in locals() and rst == lettering_workflow_act:
            self.lettering_workflow_requested.emit([self._item_page_name(item) for item in selected_items])
        elif selected_items and rst == run_detect_act:
            self.run_detect_images.emit([self._item_page_name(item) for item in selected_items])
        elif selected_items and rst == run_ocr_act:
            self.run_ocr_images.emit([self._item_page_name(item) for item in selected_items])
        elif selected_items and rst == run_translate_act:
            self.run_translate_images.emit([self._item_page_name(item) for item in selected_items])
        elif selected_items and rst == run_inpaint_act:
            self.run_inpaint_images.emit([self._item_page_name(item) for item in selected_items])
        elif selected_items and rst == ignore_act:
            self.toggle_ignore_requested.emit([self._item_page_name(item) for item in selected_items], True)
        elif selected_items and rst == include_act:
            self.toggle_ignore_requested.emit([self._item_page_name(item) for item in selected_items], False)
        elif selected_items and rst == mark_todo_act:
            self.set_completion_requested.emit([self._item_page_name(item) for item in selected_items], 'todo')
        elif selected_items and rst == mark_translated_act:
            self.set_completion_requested.emit([self._item_page_name(item) for item in selected_items], 'translated')
        elif selected_items and rst == mark_reviewed_act:
            self.set_completion_requested.emit([self._item_page_name(item) for item in selected_items], 'reviewed')
        elif selected_items and rst == mark_exported_act:
            self.set_completion_requested.emit([self._item_page_name(item) for item in selected_items], 'exported')
        elif selected_items and rst == remove_act:
            self.remove_images.emit([self._item_page_name(item) for item in selected_items])

        return super().contextMenuEvent(e)


class ModelDownloadProgressDialog(QDialog):
    """Modal dialog shown while retrying model downloads (Tools → Retry model downloads)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(parent.tr('Downloading model packages') if parent else 'Downloading model packages')
        layout = QVBoxLayout(self)
        self._label = QLabel(parent.tr('Downloading model packages... This may take several minutes.') if parent else 'Downloading model packages... This may take several minutes.')
        layout.addWidget(self._label)
        self._bar = QProgressBar(self)
        self._bar.setRange(0, 0)  # indeterminate
        layout.addWidget(self._bar)
        self.setMinimumWidth(360)
        self._thread = None

    def set_thread(self, thread):
        self._thread = thread

    def closeEvent(self, event: QCloseEvent):
        if self._thread is not None and self._thread.isRunning():
            event.ignore()
            return
        super().closeEvent(event)


mainwindow_cls = Widget if (shared.HEADLESS or shared.HEADLESS_CONTINUOUS) else FramelessWindow
class MainWindow(mainwindow_cls):

    imgtrans_proj: ProjImgTrans = ProjImgTrans()
    save_on_page_changed = True
    opening_dir = False
    page_changing = False
    postprocess_mt_toggle = True

    translator = None

    restart_signal = Signal()
    create_errdialog = Signal(str, str, str)
    create_infodialog = Signal(dict)
    batch_queue_empty = Signal()
    batch_queue_cancelled = Signal()
    batch_queue_item_started = Signal(str)

    def __init__(self, app: QApplication, config: ProgramConfig, open_dir='', **exec_args) -> None:
        super().__init__()
        self._batch_cancelled = False
        self._batch_queue_dialog = None
        self._initial_model_download_done = False
        self._zip_batch = ZipBatchManager()
        self._current_batch_dir = None
        self._last_batch_report = None
        self._startup_health_shown = False
        self._showing_welcome_screen = False

        shared.create_errdialog_in_mainthread = self.create_errdialog.emit
        self.create_errdialog.connect(self.on_create_errdialog)
        shared.create_infodialog_in_mainthread = self.create_infodialog.emit
        self.create_infodialog.connect(self.on_create_infodialog)
        shared.register_view_widget = self.register_view_widget

        self.app = app
        self.backup_blkstyles = []
        self._run_imgtrans_wo_textstyle_update = False
        self._detection_only_restore = None
        self._idle_unload_timer = QTimer(self)
        self._idle_unload_timer.setSingleShot(True)
        self._idle_unload_timer.timeout.connect(self._on_idle_unload_timeout)

        self.setupThread()
        self.setupUi()
        self.setupConfig()
        self.setupShortcuts()
        self.setupRegisterWidget()
        self.st_manager.layout_review_local_api_handler = self._layout_review_local_api_handler
        self.st_manager.layout_review_cloud_api_handler = self._layout_review_cloud_api_handler
        self._recover_window_geometry_if_offscreen()
        # self.showMaximized()
        FramelessMoveResize.toggleMaxState(self)
        self.setAcceptDrops(True)

        force_welcome = bool(getattr(shared, 'FORCE_SHOW_WELCOME_ON_STARTUP', False))
        explicit_start_target = bool(open_dir and str(open_dir).strip())

        # Startup routing:
        # - explicit CLI/file target always wins
        # - otherwise, respect the welcome preference
        # - otherwise, try the first valid recent project
        # - if nothing opens, always fall back to the welcome screen
        if explicit_start_target and osp.exists(open_dir):
            if osp.isfile(open_dir) and open_dir.lower().endswith('.json'):
                self.openJsonProj(open_dir)
            else:
                self.OpenProj(open_dir)
        elif not force_welcome and not bool(getattr(pcfg, 'show_welcome_screen', True)) and pcfg.open_recent_on_startup:
            for proj_dir in list(getattr(self.leftBar, 'recent_proj_list', []) or []):
                if proj_dir and osp.exists(proj_dir):
                    self.OpenProj(proj_dir)
                    break

        if not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS):
            if force_welcome or not self._has_open_project():
                self._show_welcome_screen()
                shared.FORCE_SHOW_WELCOME_ON_STARTUP = False

        if shared.HEADLESS or shared.HEADLESS_CONTINUOUS:
            self.run_batch(**exec_args)
        elif exec_args.get('exec_dirs') and str(exec_args.get('exec_dirs')).strip():
            self.run_batch(**exec_args)

        if shared.ON_MACOS:
            # https://bugreports.qt.io/browse/QTBUG-133215
            self.hideSystemTitleBar()
            self.showMaximized()

        if not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS) and getattr(pcfg, 'auto_update_from_github', False):
            QTimer.singleShot(1500, self.on_update_from_github)

    def on_create_errdialog(self, error_msg: str, detail_traceback: str = '', exception_type: str = ''):
        try:
            if exception_type != '':
                shared.showed_exception.add(exception_type)
            err = MessageBox(info_msg=error_msg)
            err.setWindowTitle(self.tr('Error'))
            err.setDetailedText(detail_traceback)
            err.exec()
            if exception_type != '':
                shared.showed_exception.remove(exception_type)
        except Exception:
            if exception_type in shared.showed_exception:
                shared.showed_exception.remove(exception_type)
            LOGGER.error('Failed to create error dialog')
            LOGGER.error(traceback.format_exc())

    def on_create_infodialog(self, info_dict: dict):
        # Normalize: callers may pass create_info_dialog({'title': '...', 'text': '...'}) so info_msg is a dict
        info_msg = info_dict.get('info_msg')
        if isinstance(info_msg, dict) and 'text' in info_msg:
            title = info_msg.get('title') or ''
            info_dict = {**info_dict, 'info_msg': info_msg['text']}
        else:
            title = None
        dialog = MessageBox(**info_dict)
        if title:
            dialog.setWindowTitle(title)
        dialog.show()   # exec_ will block main thread


    def _show_status_message(self, message: str, timeout_ms: int = 5000):
        """Best-effort status message helper for frameless builds without QStatusBar."""
        msg = str(message or "").strip()
        if not msg:
            return
        try:
            status_fn = getattr(self, 'statusBar', None)
            status_obj = status_fn() if callable(status_fn) else None
            if status_obj is not None and hasattr(status_obj, 'showMessage'):
                status_obj.showMessage(msg, int(timeout_ms))
                return
        except Exception:
            pass
        try:
            if hasattr(self, 'pipelineInsightsPanel') and self.pipelineInsightsPanel is not None:
                self.pipelineInsightsPanel.add_event('STATUS', msg)
        except Exception:
            pass

    def register_view_widget(self, widget: ViewWidget):
        assert widget.config_name not in shared.config_name_to_view_widget
        d = {'widget': widget}
        shared.config_name_to_view_widget[widget.config_name] = d

    def setStyleSheet(self, styleSheet: str) -> None:
        self.imgtrans_progress_msgbox.setStyleSheet(styleSheet)
        self.export_doc_thread.progress_bar.setStyleSheet(styleSheet)
        self.import_doc_thread.progress_bar.setStyleSheet(styleSheet)
        return super().setStyleSheet(styleSheet)

    def setupThread(self):
        self.imsave_thread = ImgSaveThread()
        self.export_doc_thread = ExportDocThread()
        self.export_doc_thread.fin_io.connect(self.on_fin_export_doc)
        self.import_doc_thread = ImportDocThread(self)
        self.import_doc_thread.fin_io.connect(self.on_fin_import_doc)

    def resetStyleSheet(self, reverse_icon: bool = False):
        theme = 'eva-dark' if pcfg.darkmode else 'eva-light'
        stylesheet = parse_stylesheet(theme, reverse_icon)
        # Apply theme at application level too so top-level popups/dialogs (not direct
        # children of MainWindow) match the active dark/light styling.
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(stylesheet)
        self.setStyleSheet(stylesheet)
        self._apply_custom_cursor()
        self._apply_app_font()

    def _apply_custom_cursor(self):
        if getattr(pcfg, 'use_custom_cursor', False) and getattr(pcfg, 'custom_cursor_path', ''):
            path = pcfg.custom_cursor_path
            if osp.isfile(path):
                try:
                    pix = QPixmap(path)
                    if not pix.isNull():
                        self.setCursor(QCursor(pix))
                        return
                except Exception:
                    pass
        self.unsetCursor()

    def _apply_app_font(self):
        """Apply app-wide font from theme customizer (View → Theme & UI). Never set point size <= 0."""
        family = getattr(pcfg, 'app_font_family', None) or ''
        size = max(0, getattr(pcfg, 'app_font_size', 0) or 0)
        app = QApplication.instance()
        if not app:
            return
        f = app.font()
        if f.pointSizeF() <= 0:
            f.setPointSizeF(10.0)
        if f.pointSize() <= 0:
            f.setPointSize(10)
        if family:
            f.setFamily(family)
        if size > 0 and size <= 72:
            f.setPointSize(size)
            f.setPointSizeF(float(size))
        if f.pointSizeF() <= 0:
            f.setPointSizeF(10.0)
        if f.pointSize() <= 0:
            f.setPointSize(10)
        app.setFont(f)
        # So main window and children inherit; widgets with stylesheet font-size may keep that size
        self.setFont(f)

    def _relax_right_panel_vertical_minimums(self):
        # Keep right-side utility panels from forcing the main window taller
        # than the usable Windows work area. Otherwise a maximized frameless
        # window can clip the fixed bottom bar off-screen.
        shrinkable_names = (
            "drawingPanel",
            "textPanel",
            "spellCheckPanel",
            "pipelineInsightsPanel",
            "maskDiagnosticsPanel",
            "ocrCropInspectorPanel",
            "rightComicTransStackPanel",
        )
        for name in shrinkable_names:
            widget = getattr(self, name, None)
            if widget is None:
                continue
            try:
                widget.setMinimumHeight(0)
                policy = widget.sizePolicy()
                policy.setVerticalPolicy(QSizePolicy.Policy.Ignored)
                widget.setSizePolicy(policy)
            except Exception:
                pass

        expanding_names = ("comicTransSplitter", "centralStackWidget")
        for name in expanding_names:
            widget = getattr(self, name, None)
            if widget is None:
                continue
            try:
                widget.setMinimumHeight(0)
                policy = widget.sizePolicy()
                policy.setVerticalPolicy(QSizePolicy.Policy.Expanding)
                widget.setSizePolicy(policy)
            except Exception:
                pass

    def _keep_bottom_bar_visible(self):
        # Best-effort guard after right-panel switches.
        try:
            if hasattr(self, "bottomBar") and self.bottomBar is not None:
                self.bottomBar.show()
                self.bottomBar.raise_()
            if hasattr(self, "mainvlayout") and self.mainvlayout is not None:
                self.mainvlayout.invalidate()
                self.mainvlayout.activate()
        except Exception:
            pass


    def setupUi(self):
        screen_size = QGuiApplication.primaryScreen().geometry().size()
        self.setMinimumWidth(screen_size.width() // 2)
        self.configPanel = ConfigPanel(self)
        self.configPanel.trans_config_panel.show_pre_MT_keyword_window.connect(self.show_pre_MT_keyword_window)
        self.configPanel.trans_config_panel.show_MT_keyword_window.connect(self.show_MT_keyword_window)
        self.configPanel.trans_config_panel.show_OCR_keyword_window.connect(self.show_OCR_keyword_window)
        self.configPanel.trans_config_panel.show_translation_context_requested.connect(self.show_translation_context_dialog)

        self.leftBar = LeftBar(self)
        self.leftBar.showPageListLabel.clicked.connect(self.pageLabelStateChanged)
        self.leftBar.imgTransChecked.connect(self.setupImgTransUI)
        self.leftBar.configChecked.connect(self.setupConfigUI)
        self.leftBar.globalSearchChecker.clicked.connect(self.on_set_gsearch_widget)
        self.leftBar.open_dir.connect(self.OpenProj)
        self.leftBar.open_json_proj.connect(self.openJsonProj)
        self.leftBar.save_proj.connect(self.manual_save)
        self.leftBar.run_clicked.connect(self.run_imgtrans)
        self.leftBar.close_project_requested.connect(self.close_project_and_show_welcome)
        self.leftBar.export_doc.connect(self.on_export_doc)
        self.leftBar.export_current_page_as.connect(self.on_export_current_page_as)
        self.leftBar.import_doc.connect(self.on_import_doc)
        self.leftBar.export_src_txt.connect(lambda : self.on_export_txt(dump_target='source'))
        self.leftBar.export_trans_txt.connect(lambda : self.on_export_txt(dump_target='translation'))
        self.leftBar.export_src_md.connect(lambda : self.on_export_txt(dump_target='source', suffix='.md'))
        self.leftBar.export_trans_md.connect(lambda : self.on_export_txt(dump_target='translation', suffix='.md'))
        self.leftBar.import_trans_txt.connect(self.on_import_trans_txt)

        self.pageList = PageListView()
        self.pageList.reveal_file.connect(self.on_reveal_file)
        self.pageList.remove_images.connect(self.on_remove_images)
        self.pageList.translate_images.connect(self.on_translate_images)
        self.pageList.toggle_ignore_requested.connect(self.on_toggle_page_ignored)
        self.pageList.set_completion_requested.connect(self.on_set_page_completion_state)
        self.pageList.lettering_workflow_requested.connect(self.on_lettering_workflow_pages)
        self.pageList.run_ocr_images.connect(self.on_run_ocr_images)
        self.pageList.run_translate_images.connect(self.on_run_translate_images)
        self.pageList.run_inpaint_images.connect(self.on_run_inpaint_images)
        self.pageList.run_detect_images.connect(self.on_run_detect_images)
        self.pageList.setHidden(True)
        self.pageList.currentItemChanged.connect(self.pageListCurrentItemChanged)

        self.leftStackWidget = QStackedWidget(self)
        self.leftStackWidget.addWidget(self.pageList)
        self.leftStackWidget.setMinimumWidth(shared.PAGE_LIST_PANE_DEFAULT_WIDTH)
        self.leftStackWidget.setMaximumWidth(400)

        self.global_search_widget = GlobalSearchWidget(self.leftStackWidget)
        self.global_search_widget.req_update_pagetext.connect(self.on_req_update_pagetext)
        self.global_search_widget.req_move_page.connect(self.on_req_move_page)
        self.imsave_thread.img_writed.connect(self.global_search_widget.on_img_writed)
        self.global_search_widget.search_tree.result_item_clicked.connect(self.on_search_result_item_clicked)
        self.leftStackWidget.addWidget(self.global_search_widget)
        
        self.centralStackWidget = AnimatedStackWidget(self, duration_ms=220)

        from .welcome_widget import WelcomeWidget
        self.welcomeWidget = WelcomeWidget(self)
        self.welcomeWidget.open_folder_requested.connect(self._welcome_open_folder)
        self.welcomeWidget.open_project_requested.connect(self.OpenProj)
        self.welcomeWidget.open_images_requested.connect(self.leftBar.onOpenImages)
        self.welcomeWidget.open_acbf_requested.connect(self.leftBar.onOpenACBFCBZ)
        self.welcomeWidget.recent_project_clicked.connect(self._welcome_recent_clicked)

        self.titleBar = TitleBar(self)
        self.titleBar.closebtn_clicked.connect(self.on_closebtn_clicked)
        self.titleBar.display_lang_changed.connect(self.on_display_lang_changed)
        self.titleBar.setLeftBar(self.leftBar)
        self.bottomBar = BottomBar(self)
        self.bottomBar.textedit_checkchanged.connect(self.setTextEditMode)
        self.bottomBar.paintmode_checkchanged.connect(self.setPaintMode)
        self.bottomBar.textblock_checkchanged.connect(self.setTextBlockMode)
        self.bottomBar.spellcheck_checkchanged.connect(self.on_spellcheck_panel_toggled)

        mainHLayout = QHBoxLayout()
        mainHLayout.addWidget(self.leftBar)
        mainHLayout.setSpacing(1)
        mainHLayout.addWidget(self.centralStackWidget)
        mainHLayout.setContentsMargins(0, 0, 0, 0)

        # set up canvas
        SW.canvas = self.canvas = Canvas()
        self.canvas.imgtrans_proj = self.imgtrans_proj
        self.canvas.gv.hide_canvas.connect(self.onHideCanvas)
        self.canvas.proj_savestate_changed.connect(self.on_savestate_changed)
        self.canvas.textstack_changed.connect(self.on_textstack_changed)
        self.canvas.run_blktrans.connect(self.on_run_blktrans)
        self.canvas.drop_open_folder.connect(self.dropOpenDir)
        self.canvas.drop_open_files.connect(self.dropOpenFiles)
        self.canvas.originallayer_trans_slider = self.bottomBar.originalSlider
        self.canvas.textlayer_trans_slider = self.bottomBar.textlayerSlider
        self.canvas.copy_src_signal.connect(self.on_copy_src)
        self.canvas.paste_src_signal.connect(self.on_paste_src)
        self.canvas.copy_trans_signal.connect(self.on_copy_trans)
        self.canvas.paste_trans_signal.connect(self.on_paste_trans)
        self.canvas.clear_src_signal.connect(self.on_clear_src)
        self.canvas.clear_trans_signal.connect(self.on_clear_trans)
        self.canvas.select_all_signal.connect(self.on_select_all_canvas)
        self.canvas.spell_check_src_signal.connect(self.on_spell_check_src)
        self.canvas.spell_check_trans_signal.connect(self.on_spell_check_trans)
        self.canvas.trim_whitespace_signal.connect(self.on_trim_whitespace)
        self.canvas.to_uppercase_signal.connect(self.on_to_uppercase)
        self.canvas.to_lowercase_signal.connect(self.on_to_lowercase)
        self.canvas.toggle_strikethrough_signal.connect(self.on_toggle_strikethrough)
        self.canvas.set_gradient_type_signal.connect(self.on_set_gradient_type)
        self.canvas.set_text_on_path_signal.connect(self.on_set_text_on_path)
        self.canvas.run_detect_region.connect(self.on_run_detect_region)
        self.canvas.download_page_requested.connect(self.on_download_page_requested)
        self.canvas.merge_selected_blocks_signal.connect(self.on_merge_selected_blocks)
        self.canvas.split_selected_regions_signal.connect(self.on_split_selected_regions)
        self.canvas.move_blocks_up_signal.connect(self.on_move_blocks_up)
        self.canvas.move_blocks_down_signal.connect(self.on_move_blocks_down)
        self.canvas.import_image_to_blk.connect(self.on_import_image_to_blk)
        self.canvas.clear_overlay_signal.connect(self.on_clear_overlay)
        self.canvas.triage_add_selected_signal.connect(self.on_add_selected_to_triage)
        self.canvas.triage_mark_reviewed_selected_signal.connect(self.on_mark_selected_triage_reviewed)
        self.canvas.layout_review_selected_signal.connect(self.shortcutLayoutReviewSelected)
        self.canvas.layout_review_page_signal.connect(self.shortcutLayoutReviewPage)
        self.canvas.layout_review_config_signal.connect(self.shortcutLayoutReviewConfig)
        self.canvas.review_ocr_triage_signal.connect(self.on_open_ocr_triage_current_page)
        self.canvas.review_translation_qa_signal.connect(self.on_translation_qa_report_current_page)
        self.canvas.review_auto_extract_glossary_signal.connect(self.on_auto_extract_glossary_current_page)
        self.canvas.export_lettering_proof_signal.connect(self.on_export_lettering_proof_pack)

        self.app.installEventFilter(self)
        self.canvas.gv.installEventFilter(self)
        self.canvas.gv.viewport().installEventFilter(self)

        self.bottomBar.originalSlider.valueChanged.connect(self.canvas.setOriginalTransparencyBySlider)
        self.bottomBar.textlayerSlider.valueChanged.connect(self.canvas.setTextLayerTransparencyBySlider)
        
        self.drawingPanel = DrawingPanel(self.canvas, self.configPanel.inpaint_config_panel)
        self.textPanel = TextPanel(self.app)
        self.textPanel.formatpanel.foldTextBtn.checkStateChanged.connect(self.fold_textarea)
        self.textPanel.formatpanel.sourceBtn.checkStateChanged.connect(self.show_source_text)
        self.textPanel.formatpanel.transBtn.checkStateChanged.connect(self.show_trans_text)
        self.textPanel.formatpanel.textstyle_panel.export_style.connect(self.export_tstyles)
        self.textPanel.formatpanel.textstyle_panel.import_style.connect(self.import_tstyles)
        self.textPanel.formatpanel.set_default_format_requested.connect(self.on_set_default_format_requested)

        self.ocrSubWidget = KeywordSubWidget(self.tr("Keyword substitution for source text"))
        self.ocrSubWidget.setParent(self)
        self.ocrSubWidget.setWindowFlags(Qt.WindowType.Window)
        self.ocrSubWidget.hide()
        self.mtPreSubWidget = KeywordSubWidget(self.tr("Keyword substitution for machine translation source text"))
        self.mtPreSubWidget.setParent(self)
        self.mtPreSubWidget.setWindowFlags(Qt.WindowType.Window)
        self.mtPreSubWidget.hide()
        self.mtSubWidget = KeywordSubWidget(self.tr("Keyword substitution for machine translation"))
        self.mtSubWidget.setParent(self)
        self.mtSubWidget.setWindowFlags(Qt.WindowType.Window)
        self.mtSubWidget.hide()

        SW.st_manager = self.st_manager = SceneTextManager(self.app, self, self.canvas, self.textPanel)
        self.st_manager.new_textblk.connect(self.canvas.search_widget.on_new_textblk)
        self.canvas.search_widget.pairwidget_list = self.st_manager.pairwidget_list
        self.canvas.search_widget.textblk_item_list = self.st_manager.textblk_item_list
        self.canvas.search_widget.replace_one.connect(self.st_manager.on_page_replace_one)
        self.canvas.search_widget.replace_all.connect(self.st_manager.on_page_replace_all)

        # comic trans pannel
        self.rightComicTransStackPanel = AnimatedStackWidget(self, duration_ms=180)
        self.rightComicTransStackPanel.addWidget(self.drawingPanel)
        self.rightComicTransStackPanel.addWidget(self.textPanel)
        self.spellCheckPanel = SpellCheckPanel(self)
        self.spellCheckPanel.set_get_blocks(self._spellcheck_get_blocks)
        self.spellCheckPanel.set_apply_replacement(self._spellcheck_apply_replacement)
        self.spellCheckPanel.close_requested.connect(self._on_spellcheck_panel_close)
        self.spellCheckPanel.focus_block_requested.connect(self._on_spellcheck_focus_block)
        self.rightComicTransStackPanel.addWidget(self.spellCheckPanel)
        self.pipelineInsightsPanel = PipelineInsightsWidget(self)
        self.pipelineInsightsPanel.rerun_stage_requested.connect(self.on_rerun_stage_requested)
        self.pipelineInsightsPanel.apply_regex_profile_requested.connect(self.on_apply_regex_profile_requested)
        self.pipelineInsightsPanel.open_mask_diagnostics_requested.connect(self.on_open_mask_diagnostics_requested)
        self.pipelineInsightsPanel.apply_project_ops_requested.connect(self.on_apply_project_ops_requested)
        self.pipelineInsightsPanel.open_ocr_crop_inspector_requested.connect(self.on_open_ocr_crop_inspector_requested)
        self.pipelineInsightsPanel.open_reading_order_editor_requested.connect(self.on_open_reading_order_editor_requested)
        self.pipelineInsightsPanel.run_layout_review_requested.connect(self.on_run_layout_review_requested)
        self.pipelineInsightsPanel.open_batch_style_requested.connect(self.on_open_batch_style_override)
        self.pipelineInsightsPanel.open_typography_qa_requested.connect(self.on_open_typography_qa_report)
        self.pipelineInsightsPanel.run_auto_lettering_assist_requested.connect(self.on_run_auto_lettering_assist_requested)
        self.pipelineInsightsPanel.run_production_auto_pass_requested.connect(self.on_run_production_auto_pass_requested)
        if hasattr(self.pipelineInsightsPanel, 'apply_workflow_preset_requested'):
            self.pipelineInsightsPanel.apply_workflow_preset_requested.connect(self.on_apply_workflow_preset_requested)
            self.pipelineInsightsPanel.run_workflow_preset_requested.connect(lambda preset_id: self.on_apply_workflow_preset_requested(preset_id, run_after=True))
        self.rightComicTransStackPanel.addWidget(self.pipelineInsightsPanel)
        self.maskDiagnosticsPanel = MaskDiagnosticsWidget(self)
        self.rightComicTransStackPanel.addWidget(self.maskDiagnosticsPanel)
        self.ocrCropInspectorPanel = OcrCropInspectorWidget(self)
        self.rightComicTransStackPanel.addWidget(self.ocrCropInspectorPanel)
        self.rightComicTransStackPanel.currentChanged.connect(self.on_transpanel_changed)

        self.comicTransSplitter = QSplitter(Qt.Orientation.Horizontal)
        self.comicTransSplitter.addWidget(self.leftStackWidget)
        self.comicTransSplitter.addWidget(self.canvas.gv)
        self.comicTransSplitter.addWidget(self.rightComicTransStackPanel)
        self._relax_right_panel_vertical_minimums()

        self.centralStackWidget.addWidget(self.welcomeWidget)
        self.centralStackWidget.addWidget(self.comicTransSplitter)
        self.centralStackWidget.addWidget(self.configPanel)

        self.selectext_minimenu = self.st_manager.selectext_minimenu = SelectTextMiniMenu(self.app, self)
        self.selectext_minimenu.block_current_editor.connect(self.st_manager.on_block_current_editor)
        self.selectext_minimenu.hide()

        mainVBoxLayout = QVBoxLayout(self)
        mainVBoxLayout.addWidget(self.titleBar)
        mainVBoxLayout.addLayout(mainHLayout)
        mainVBoxLayout.addWidget(self.bottomBar)
        margin = mainVBoxLayout.contentsMargins()
        self.main_margin = margin
        mainVBoxLayout.setContentsMargins(0, 0, 0, 0)
        mainVBoxLayout.setSpacing(0)

        self.mainvlayout = mainVBoxLayout
        self.comicTransSplitter.setStretchFactor(0, 1)
        self.comicTransSplitter.setStretchFactor(1, 10)
        self.comicTransSplitter.setStretchFactor(2, 1)
        # Allow user to resize the right panel by dragging the splitter (wider or narrower)
        self.rightComicTransStackPanel.setMinimumWidth(340)
        self.rightComicTransStackPanel.setMaximumWidth(860)
        self._comic_trans_splitter_initialized = False
        self.imgtrans_progress_msgbox = ImgtransProgressMessageBox()
        self.resetStyleSheet()

    def _persist_module_selection(self, key: str, name: str, is_fallback: bool = False):
        """
        Remember the selected module so the next launch restores it (issue #152).

        `is_fallback` marks a module loaded only because the configured one was
        unavailable (e.g. its model files are still missing); the saved selection is
        then left untouched so it comes back once the module works again.
        """
        if is_fallback or not name:
            return
        if getattr(pcfg.module, key, None) == name:
            return
        setattr(pcfg.module, key, name)
        self.schedule_config_save()

    def schedule_config_save(self, delay_ms: int = 1500):
        """Write config.json shortly after a change so settings survive a crash / forced quit."""
        timer = getattr(self, '_config_save_timer', None)
        if timer is None:
            timer = self._config_save_timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self.save_config)
        timer.start(delay_ms)

    def on_finish_setdetector(self):
        """Defer UI update to avoid QPainter conflict when signal fires during paint (e.g. model load)."""
        module_manager = self.module_manager
        if module_manager.textdetector is not None:
            name = module_manager.textdetector.name
            is_fallback = module_manager.textdetect_thread.module_fallback_used
            def _update():
                self._persist_module_selection('textdetector', name, is_fallback)
                self.configPanel.detect_config_panel.setDetector(name)
                self.bottomBar.textdet_selector.setSelectedValue(name)
                LOGGER.info('Text detector set to {}'.format(name))
                self._refresh_pipeline_provider_health()
                self.pipelineInsightsPanel.set_engine_registry(self.module_manager.get_engine_registry_snapshot())
            QTimer.singleShot(0, _update)

    def on_finish_setocr(self):
        module_manager = self.module_manager
        if module_manager.ocr is not None:
            name = module_manager.ocr.name
            is_fallback = module_manager.ocr_thread.module_fallback_used
            def _update():
                self._persist_module_selection('ocr', name, is_fallback)
                self.configPanel.ocr_config_panel.setOCR(name)
                self.bottomBar.ocr_selector.setSelectedValue(name)
                LOGGER.info('OCR set to {}'.format(name))
                self._refresh_pipeline_provider_health()
                self.pipelineInsightsPanel.set_engine_registry(self.module_manager.get_engine_registry_snapshot())
            QTimer.singleShot(0, _update)

    def on_finish_setinpainter(self):
        module_manager = self.module_manager
        if module_manager.inpainter is not None:
            name = module_manager.inpainter.name
            is_fallback = module_manager.inpaint_thread.module_fallback_used
            def _update():
                self._persist_module_selection('inpainter', name, is_fallback)
                self.configPanel.inpaint_config_panel.setInpainter(name)
                self.bottomBar.inpaint_selector.setSelectedValue(name)
                LOGGER.info('Inpainter set to {}'.format(name))
                self._refresh_pipeline_provider_health()
                self.pipelineInsightsPanel.set_engine_registry(self.module_manager.get_engine_registry_snapshot())
            QTimer.singleShot(0, _update)

    def on_finish_settranslator(self):
        module_manager = self.module_manager
        translator = module_manager.translator
        if translator is not None:
            name = translator.name
            is_fallback = module_manager.translate_thread.module_fallback_used
            def _update():
                self._persist_module_selection('translator', name, is_fallback)
                self.bottomBar.trans_selector.finishSetTranslator(translator)
                self.configPanel.trans_config_panel.finishSetTranslator(translator)
                LOGGER.info('Translator set to {}'.format(name))
                self._refresh_pipeline_provider_health()
                self.pipelineInsightsPanel.set_engine_registry(self.module_manager.get_engine_registry_snapshot())
            QTimer.singleShot(0, _update)
        else:
            LOGGER.error('invalid translator')
        
    def on_enable_module(self, idx, checked):
        if idx == 0:
            pcfg.module.enable_detect = checked
            self.bottomBar.textdet_selector.setVisible(checked)
        elif idx == 1:
            pcfg.module.enable_ocr = checked
            self.bottomBar.ocr_selector.setVisible(checked)
        elif idx == 2:
            pcfg.module.enable_translate = checked
            self.bottomBar.trans_selector.setVisible(checked)
        elif idx == 3:
            pcfg.module.enable_inpaint = checked
            self.bottomBar.inpaint_selector.setVisible(checked)
        pcfg.module.update_finish_code()
        self.schedule_config_save()

    def setupConfig(self):

        self.bottomBar.originalSlider.setValue(int(pcfg.original_transparency * 100))
        self.bottomBar.trans_selector.selector.addItems(GET_VALID_TRANSLATORS())
        self.bottomBar.ocr_selector.selector.addItems(GET_VALID_OCR())
        self.bottomBar.textdet_selector.selector.addItems(GET_VALID_TEXTDETECTORS())
        self.bottomBar.textdet_selector.selector.currentTextChanged.connect(self.on_textdet_changed)
        self.bottomBar.inpaint_selector.selector.addItems(GET_VALID_INPAINTERS())
        self.bottomBar.inpaint_selector.selector.currentTextChanged.connect(self.on_inpaint_changed)
        self.bottomBar.trans_selector.cfg_clicked.connect(self.to_trans_config)
        self.bottomBar.trans_selector.selector.currentTextChanged.connect(self.on_trans_changed)
        self.bottomBar.trans_selector.tgt_selector.currentTextChanged.connect(self.on_trans_tgt_changed)
        self.bottomBar.trans_selector.src_selector.currentTextChanged.connect(self.on_trans_src_changed)
        self.bottomBar.textdet_selector.cfg_clicked.connect(self.to_detect_config)
        self.bottomBar.inpaint_selector.cfg_clicked.connect(self.to_inpaint_config)
        self.bottomBar.ocr_selector.cfg_clicked.connect(self.to_ocr_config)
        self.bottomBar.ocr_selector.selector.currentTextChanged.connect(self.on_ocr_changed)
        self.bottomBar.textdet_selector.setVisible(pcfg.module.enable_detect)
        self.bottomBar.ocr_selector.setVisible(pcfg.module.enable_ocr)
        self.bottomBar.trans_selector.setVisible(pcfg.module.enable_translate)
        self.bottomBar.inpaint_selector.setVisible(pcfg.module.enable_inpaint)

        self.configPanel.trans_config_panel.target_combobox.currentTextChanged.connect(self.on_trans_tgt_changed)
        self.configPanel.trans_config_panel.source_combobox.currentTextChanged.connect(self.on_trans_src_changed)

        self.drawingPanel.maskTransperancySlider.setValue(int(pcfg.mask_transparency * 100))
        self.leftBar.initRecentProjMenu(pcfg.recent_proj_list)
        self.leftBar.showPageListLabel.setChecked(pcfg.show_page_list)
        self.updatePageList()
        self.leftBar.save_config.connect(self.save_config)
        # Do not emit imgTransChecked during startup. That signal switches the
        # central stack to the canvas even when no project is open.
        self._set_leftbar_mode(None)
        self.st_manager.formatpanel.global_format = pcfg.global_fontformat
        self.st_manager.formatpanel.set_active_format(pcfg.global_fontformat)
        
        self.rightComicTransStackPanel.setHidden(True)
        self.st_manager.setTextEditMode(False)
        self.st_manager.formatpanel.foldTextBtn.setChecked(pcfg.fold_textarea)
        self.st_manager.formatpanel.transBtn.setCheckState(pcfg.show_trans_text)
        self.st_manager.formatpanel.sourceBtn.setCheckState(pcfg.show_source_text)
        self.fold_textarea(pcfg.fold_textarea)
        self.show_trans_text(pcfg.show_trans_text)
        self.show_source_text(pcfg.show_source_text)

        self.module_manager = module_manager = ModuleManager(self.imgtrans_proj)
        module_manager.finish_translate_page.connect(self.finishTranslatePage)
        module_manager.imgtrans_pipeline_finished.connect(self.on_imgtrans_pipeline_finished)
        module_manager.page_trans_finished.connect(self.on_pagtrans_finished)
        module_manager.pipeline_job_started.connect(self.pipelineInsightsPanel.set_job_id)
        module_manager.pipeline_stage_event.connect(self.on_pipeline_stage_event)
        module_manager.pipeline_event_logged.connect(self.pipelineInsightsPanel.add_event)
        module_manager.engine_registry_updated.connect(self.pipelineInsightsPanel.set_engine_registry)
        module_manager.setupThread(self.configPanel, self.imgtrans_progress_msgbox, self.ocr_postprocess, self.translate_preprocess, self.translate_postprocess)
        module_manager.progress_msgbox.showed.connect(self.on_imgtrans_progressbox_showed)
        module_manager.blktrans_pipeline_finished.connect(self.on_blktrans_finished)
        module_manager.detect_region_finished.connect(self.on_detect_region_finished)
        module_manager.imgtrans_thread.post_process_mask = self.drawingPanel.rectPanel.post_process_mask
        module_manager.inpaint_thread.finish_set_module.connect(self.on_finish_setinpainter)
        module_manager.translate_thread.finish_set_module.connect(self.on_finish_settranslator)
        module_manager.textdetect_thread.finish_set_module.connect(self.on_finish_setdetector)
        module_manager.ocr_thread.finish_set_module.connect(self.on_finish_setocr)
        module_manager.setTextDetector()
        module_manager.setOCR()
        module_manager.setTranslator()
        module_manager.setInpainter()
        self.pipelineInsightsPanel.set_engine_registry(module_manager.get_engine_registry_snapshot())
        self._refresh_pipeline_provider_health()
        self._api_cmd_queue = queue.Queue()
        self._api_jobs = {}
        self._api_jobs_lock = threading.Lock()
        self._api_cmd_timer = QTimer(self)
        self._api_cmd_timer.timeout.connect(self._drain_api_cmd_queue)
        self._api_cmd_timer.start(40)
        self._api_status_timer = QTimer(self)
        self._api_status_timer.timeout.connect(self._refresh_api_status)
        self._api_status_timer.start(500)
        self._setup_local_automation_api()

        self.titleBar.themeLightAction.setChecked(not pcfg.darkmode)
        self.titleBar.themeDarkAction.setChecked(pcfg.darkmode)

        self.drawingPanel.set_config(pcfg.drawpanel)
        self.drawingPanel.initDLModule(module_manager)

        self.global_search_widget.imgtrans_proj = self.imgtrans_proj
        self.global_search_widget.setupReplaceThread(self.st_manager.pairwidget_list, self.st_manager.textblk_item_list)
        self.global_search_widget.replace_thread.finished.connect(self.on_global_replace_finished)

        self.configPanel.setupConfig()
        self.configPanel.save_config.connect(self.save_config)
        self.configPanel.reload_textstyle.connect(self.load_textstyle_from_proj_dir)
        self.configPanel.show_only_custom_font.connect(self.on_show_only_custom_font)
        self.configPanel.darkmode_changed.connect(self.on_config_darkmode_changed)
        self.configPanel.custom_cursor_changed.connect(self._on_config_custom_cursor_changed)
        self.configPanel.display_lang_changed.connect(self.on_display_lang_changed)
        self.configPanel.dev_mode_changed.connect(self.on_dev_mode_changed)
        self.configPanel.manual_mode_changed.connect(self._update_run_button_tooltip)
        if pcfg.let_show_only_custom_fonts_flag:
            self.on_show_only_custom_font(True)

        textblock_mode = pcfg.imgtrans_textblock
        if pcfg.imgtrans_textedit:
            if textblock_mode:
                self.bottomBar.textblockChecker.setChecked(True)
            self.bottomBar.texteditChecker.click()
        elif pcfg.imgtrans_paintmode:
            self.bottomBar.paintChecker.click()

        self.textPanel.formatpanel.textstyle_panel.initStyles(text_styles)
        self.textPanel.formatpanel.textstyle_panel.set_active_style_by_name(
            getattr(pcfg, 'default_text_style_name', '') or ''
        )

        self.canvas.search_widget.whole_word_toggle.setChecked(pcfg.fsearch_whole_word)
        self.canvas.search_widget.case_sensitive_toggle.setChecked(pcfg.fsearch_case)
        self.canvas.search_widget.regex_toggle.setChecked(pcfg.fsearch_regex)
        self.canvas.search_widget.range_combobox.setCurrentIndex(pcfg.fsearch_range)
        self.global_search_widget.whole_word_toggle.setChecked(pcfg.gsearch_whole_word)
        self.global_search_widget.case_sensitive_toggle.setChecked(pcfg.gsearch_case)
        self.global_search_widget.regex_toggle.setChecked(pcfg.gsearch_regex)
        self.global_search_widget.range_combobox.setCurrentIndex(pcfg.gsearch_range)

        if self.rightComicTransStackPanel.isHidden():
            self.setPaintMode()

        try:
            self.ocrSubWidget.loadCfgSublist(pcfg.ocr_sublist)
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            pcfg.ocr_sublist = []
            self.ocrSubWidget.loadCfgSublist(pcfg.ocr_sublist)

        try:
            self.mtPreSubWidget.loadCfgSublist(pcfg.pre_mt_sublist)
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            pcfg.pre_mt_sublist = []
            self.mtPreSubWidget.loadCfgSublist(pcfg.pre_mt_sublist)

        try:
            self.mtSubWidget.loadCfgSublist(pcfg.mt_sublist)
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            pcfg.mt_sublist = []
            self.mtSubWidget.loadCfgSublist(pcfg.mt_sublist)

        self._update_run_button_tooltip()


    def _refresh_api_status(self):
        enabled = bool(getattr(self, '_automation_api', None) is not None)
        qd = 0
        try:
            qd = int(self._api_cmd_queue.qsize())
        except Exception:
            qd = 0
        self.pipelineInsightsPanel.set_api_status(enabled, qd)

    def _drain_api_cmd_queue(self):
        for _ in range(8):
            try:
                item = self._api_cmd_queue.get_nowait()
            except Exception:
                return
            try:
                item['result'] = item['fn'](item.get('body') or {})
            except Exception as e:
                item['error'] = str(e)
            item['ev'].set()

    def _api_call_ui(self, fn, body: dict):
        ev = threading.Event()
        item = {'fn': fn, 'body': body or {}, 'ev': ev}
        self._api_cmd_queue.put(item)
        if not ev.wait(timeout=15.0):
            raise RuntimeError('API command timeout')
        if item.get('error'):
            raise RuntimeError(item.get('error'))
        return item.get('result') or {}

    def _setup_local_automation_api(self):
        if not bool(getattr(pcfg, 'automation_api_enabled', False)):
            self._automation_api = None
            return
        handlers = {
            'open_project': self._api_open_project,
            'project_open': self._api_open_project,  # MCP alias
            'run_pipeline': self._api_run_pipeline,
            'pipeline_run': self._api_run_pipeline,  # MCP alias
            'apply_edit': self._api_apply_edit,
            'scene_edit': self._api_apply_edit,  # MCP alias
            'undo': self._api_undo,
            'redo': self._api_redo,
            'export': self._api_export,
            'batch_export': self._api_batch_export,
            'proof_pack': self._api_export_lettering_proof,
            'render': self._api_render_current_page,  # MCP alias
            'render_page': self._api_render_current_page,
            'layout_review': self._api_layout_review,
            'page_state': self._api_page_state,
            'list_pages': self._api_list_pages,
            'recent_projects': self._api_recent_projects,
            'project_status': self._api_project_status,
            'pipeline_presets': self._api_pipeline_presets,
            'export_structured_ocr': self._api_export_structured_ocr,
            'export_xliff': self._api_export_xliff,
            'import_xliff': self._api_import_xliff,
            'export_translation_json': self._api_export_translation_json,
            'import_translation_json': self._api_import_translation_json,
            'export_translation_csv': self._api_export_translation_csv,
            'import_translation_csv': self._api_import_translation_csv,
            'tm_query': self._api_tm_query,
            'tm_build_from_project': self._api_tm_build_from_project,
            'tm_export': self._api_tm_export,
            'tm_import': self._api_tm_import,
            'translation_qa_report': self._api_translation_qa_report,
            'translation_prompt_profiles': self._api_translation_prompt_profiles,
            'concordance_query': self._api_concordance_query,
            'glossary_export': self._api_glossary_export,
            'glossary_import': self._api_glossary_import,
            'glossary_import_preview': self._api_glossary_import_preview,
            'sfx_dictionary_query': self._api_sfx_dictionary_query,
            'sfx_dictionary_import': self._api_sfx_dictionary_import,
            'sfx_dictionary_export': self._api_sfx_dictionary_export,
            'glossary_extract_candidates': self._api_glossary_extract_candidates,
            'glossary_extract_apply': self._api_glossary_extract_apply,
            'ocr_rerun_block': self._api_ocr_rerun_block,
            'ocr_compare_block': self._api_ocr_compare_block,
            'ocr_apply_compare_choice': self._api_ocr_apply_compare_choice,
            'renderer_diagnostics': self._api_renderer_diagnostics,
            'renderer_backend_probe': self._api_renderer_diagnostics,
            'cleanup_only': self._api_cleanup_only,
            'batch_parent_enumerate': self._api_batch_parent_enumerate,
            'batch_parent_save_state': self._api_batch_parent_save_state,
            'batch_parent_load_state': self._api_batch_parent_load_state,
            'batch_parent_update_status': self._api_batch_parent_update_status,
            'batch_parent_next_pending': self._api_batch_parent_next_pending,
            'batch_parent_summary': self._api_batch_parent_summary,
            'data_path_status': self._api_data_path_status,
            'data_path_migrate': self._api_data_path_migrate,
            'server_mode_info': self._api_server_mode_info,
            'import_translated_image_align': self._api_import_translated_image_align,
            'batch_find_replace_preview': self._api_batch_find_replace_preview,
            'batch_find_replace_apply': self._api_batch_find_replace_apply,
            'render_current_page': self._api_render_current_page,
            'list_rendering_issues': self._api_list_rendering_issues,
            'apply_rendering_preset': self._api_apply_rendering_preset,
            'rendering_presets': self._api_rendering_presets,
            'fix_rendering_issues': self._api_fix_rendering_issues,
            'smart_fit_textboxes': self._api_smart_fit_textboxes,
            'auto_format_textboxes': self._api_auto_format_textboxes,
            'auto_format_qa_preview': self._api_auto_format_qa_preview,
            'atomic_bubble_fit': self._api_atomic_bubble_fit,
            'polish_typography': self._api_polish_typography,
            'export_rendering_qa': self._api_export_rendering_qa,
            'export_lettering_proof': self._api_export_lettering_proof,
            'apply_project_rendering_fixes': self._api_apply_project_rendering_fixes,
            'apply_text_style_batch': self._api_apply_text_style_batch,
            'lettering_workflow': self._api_lettering_workflow,
            'next_rendering_issue': self._api_next_rendering_issue,
            'job_start': self._api_job_start,
            'job_status': self._api_job_status,
            'job_cancel': self._api_job_cancel,
            'job_logs': self._api_job_logs,
            'job_result': self._api_job_result,
            'jobs_list': self._api_jobs_list,
            'realtime_status': self._api_realtime_status,
            'realtime_regions': self._api_realtime_regions,
            'realtime_start': self._api_realtime_start,
            'realtime_pause': self._api_realtime_pause,
            'realtime_stop': self._api_realtime_stop,
            'realtime_translate_now': self._api_realtime_translate_now,
            'realtime_profiles': self._api_realtime_profiles,
            'translation_assist_block': self._api_translation_assist_block,
            'translation_assist_candidates': self._api_translation_assist_candidates,
            'translation_assist_apply_candidate': self._api_translation_assist_apply_candidate,
            'translation_assist_tm': self._api_translation_assist_tm,
            'translation_assist_glossary': self._api_translation_assist_glossary,
            'translation_assist_concordance': self._api_translation_assist_concordance,
            'translation_assist_sfx': self._api_translation_assist_sfx,
            'translation_assist_add_to_tm': self._api_translation_assist_add_to_tm,
            'translation_assist_add_to_glossary': self._api_translation_assist_add_to_glossary,
            'translation_assist_qa': self._api_translation_assist_qa,
            'translation_assist_profiles': self._api_translation_assist_profiles,
            'translation_assist_apply_text': self._api_translation_assist_apply_text,
            'translation_assist_cache': self._api_translation_assist_cache,
            'translation_assist_compare': self._api_translation_assist_compare,
            'image_transform': self._api_image_transform,
            'docs_catalog': self._api_docs_catalog,
        }
        try:
            self._automation_api = LocalAutomationApiServer('127.0.0.1', int(getattr(pcfg, 'automation_api_port', 39542)), handlers, api_key=str(getattr(pcfg, 'automation_api_key', '') or ''))
            self._automation_api.start()
            self.pipelineInsightsPanel.add_event('API', self.tr('Local automation API started'))
        except Exception as e:
            self.pipelineInsightsPanel.add_warning('API', self.tr(f'Failed to start automation API: {e}'))


    def _api_pipeline_presets(self, body: dict):
        return self._api_call_ui(self._api_pipeline_presets_ui, body)

    def _realtime_service(self) -> RealtimeService:
        if not hasattr(self, '_realtime_service_obj') or self._realtime_service_obj is None:
            self._realtime_service_obj = RealtimeService()
        return self._realtime_service_obj

    def _api_realtime_status(self, body: dict):
        return self._api_call_ui(lambda _b: self._realtime_service().status(), body)

    def _api_docs_catalog(self, body: dict):
        return self._api_call_ui(lambda _b: {'ok': True, 'docs': build_docs_catalog()}, body)

    def _api_realtime_regions(self, body: dict):
        def _ui(_b: dict):
            svc = self._realtime_service()
            action = str((_b or {}).get('action', 'list') or 'list').lower()
            if action == 'list':
                return {'regions': [vars(v) for v in svc.regions.values()]}
            if action == 'add':
                rid = str((_b or {}).get('region_id', '') or '').strip() or f"region_{len(svc.regions)+1}"
                rect = tuple((_b or {}).get('rect', [0, 0, 640, 360])[:4])
                profile = str((_b or {}).get('profile', 'chrome_manhua_reader') or 'chrome_manhua_reader')
                svc.regions[rid] = RealtimeRegion(region_id=rid, rect=rect, profile=profile)
                return {'ok': True, 'region': vars(svc.regions[rid])}
            if action == 'delete':
                rid = str((_b or {}).get('region_id', '') or '').strip()
                svc.regions.pop(rid, None)
                return {'ok': True, 'deleted': rid}
            raise ValueError('realtime_regions action must be list|add|delete')
        return self._api_call_ui(_ui, body)

    def _api_realtime_start(self, body: dict):
        return self._api_call_ui(lambda _b: {'ok': True, **self._set_realtime_enabled(True)}, body)

    def _api_realtime_pause(self, body: dict):
        return self._api_call_ui(lambda _b: {'ok': True, **self._set_realtime_enabled(False), 'paused': True}, body)

    def _api_realtime_stop(self, body: dict):
        return self._api_call_ui(lambda _b: {'ok': True, **self._set_realtime_enabled(False), 'stopped': True}, body)

    def _api_realtime_translate_now(self, body: dict):
        return self._api_call_ui(lambda _b: {'ok': True, 'status': 'queued_manual_tick'}, body)

    def _api_realtime_profiles(self, body: dict):
        return self._api_call_ui(lambda _b: {'profiles': list(self._realtime_service().profiles.values())}, body)

    def _set_realtime_enabled(self, enabled: bool):
        svc = self._realtime_service()
        svc.enabled = bool(enabled)
        return svc.status()


    def _translation_assist_service(self) -> TranslationAssistService:
        if not hasattr(self, '_translation_assist_service_obj') or self._translation_assist_service_obj is None:
            self._translation_assist_service_obj = TranslationAssistService()
        return self._translation_assist_service_obj

    def _api_translation_assist_block(self, body: dict):
        def _ui(_b: dict):
            page, block, blk = self._resolve_assist_block(_b or {})
            st = self._translation_assist_service().get_block(page, block)
            src = (getattr(blk, 'get_text', lambda: '')() or '').strip()
            tgt = str(getattr(blk, 'translation', '') or '').strip()
            if src:
                st.source_text = src
            st.current_target_text = tgt
            return {'ok': True, 'page': page, 'block': block, 'source_text': st.source_text, 'current_target_text': st.current_target_text, 'candidates': [vars(c) for c in st.candidates], 'applied_candidate_id': st.applied_candidate_id}
        return self._api_call_ui(_ui, body)

    def _api_translation_assist_candidates(self, body: dict):
        def _ui(_b: dict):
            page = int((_b or {}).get('page', 0) or 0)
            block = int((_b or {}).get('block', 0) or 0)
            st = self._translation_assist_service().set_candidates(page, block, (_b or {}).get('source_text', ''), (_b or {}).get('current_target_text', ''), (_b or {}).get('candidates', []))
            return {'ok': True, 'page': page, 'block': block, 'candidate_count': len(st.candidates), 'candidates': [vars(c) for c in st.candidates]}
        return self._api_call_ui(_ui, body)

    def _api_translation_assist_apply_candidate(self, body: dict):
        def _ui(_b: dict):
            page, block, blk = self._resolve_assist_block(_b or {})
            cid = str((_b or {}).get('candidate_id', '') or '').strip()
            if not cid:
                raise ValueError('candidate_id is required')
            st = self._translation_assist_service().apply_candidate(page, block, cid)
            candidate_text = ''
            for c in st.candidates:
                if c.candidate_id == cid:
                    candidate_text = str(c.text or '')
                    break
            apply_to_project = bool((_b or {}).get('apply_to_project', False))
            changed = False
            if apply_to_project and candidate_text:
                provider = ''
                for c in st.candidates:
                    if c.candidate_id == cid:
                        provider = str(c.provider or '')
                        break
                if provider.startswith('ocr:'):
                    if str(getattr(pcfg.module, 'ocr', '') or '') != candidate_text:
                        pcfg.module.ocr = candidate_text
                        self.module_manager.setOCR(candidate_text)
                        changed = True
                elif provider.startswith('detector:'):
                    if str(getattr(pcfg.module, 'textdetector', '') or '') != candidate_text:
                        pcfg.module.textdetector = candidate_text
                        self.module_manager.setTextDetector(candidate_text)
                        changed = True
                elif provider.startswith('inpainter:'):
                    if str(getattr(pcfg.module, 'inpainter', '') or '') != candidate_text:
                        pcfg.module.inpainter = candidate_text
                        self.module_manager.setInpainter(candidate_text)
                        changed = True
                else:
                    old = str(getattr(blk, 'translation', '') or '')
                    if old != candidate_text:
                        blk.translation = candidate_text
                        changed = True
                        self.canvas.updateTextBlkList()
                if changed:
                    self.canvas.setProjSaveState(True)
            return {'ok': True, 'page': page, 'block': block, 'applied_candidate_id': st.applied_candidate_id, 'applied_to_project': changed, 'candidate_text': candidate_text}
        return self._api_call_ui(_ui, body)

    def _api_translation_assist_tm(self, body: dict):
        def _ui(_b: dict):
            if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
                raise ValueError('open a project first')
            source = str((_b or {}).get('source_text', '') or '').strip()
            if not source:
                _p, _idx, blk = self._resolve_assist_block(_b or {})
                source = (getattr(blk, 'get_text', lambda: '')() or '').strip()
            min_score = float((_b or {}).get('min_score', 0.65) or 0.65)
            limit = int((_b or {}).get('limit', 5) or 5)
            store = list(getattr(self.imgtrans_proj, 'translation_memory', []) or [])
            hits = query_tm(store, source, min_score=min_score, limit=limit)
            return {'ok': True, 'source_text': source, 'hits': hits, 'count': len(hits)}
        return self._api_call_ui(_ui, body)

    def _api_translation_assist_glossary(self, body: dict):
        def _ui(_b: dict):
            if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
                raise ValueError('open a project first')
            source = str((_b or {}).get('source_text', '') or '').strip()
            if not source:
                _p, _idx, blk = self._resolve_assist_block(_b or {})
                source = (getattr(blk, 'get_text', lambda: '')() or '').strip()
            entries = list(getattr(self.imgtrans_proj, 'translation_glossary', []) or [])
            hits = []
            src_l = source.lower()
            for row in entries:
                term = str((row or {}).get('source', '') or '').strip()
                tgt = str((row or {}).get('target', '') or '').strip()
                if term and term.lower() in src_l:
                    hits.append({'source': term, 'target': tgt, 'rule': 'project_glossary'})
            return {'ok': True, 'source_text': source, 'hits': hits, 'count': len(hits)}
        return self._api_call_ui(_ui, body)

    def _api_translation_assist_concordance(self, body: dict):
        def _ui(_b: dict):
            if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
                raise ValueError('open a project first')
            query = str((_b or {}).get('query', '') or '').strip()
            if not query:
                _p, _idx, blk = self._resolve_assist_block(_b or {})
                query = (getattr(blk, 'get_text', lambda: '')() or '').strip()
            rows = build_concordance_from_project(self.imgtrans_proj)
            hits = query_concordance(rows, query, in_target=bool((_b or {}).get('in_target', True)), limit=int((_b or {}).get('limit', 20) or 20))
            return {'ok': True, 'query': query, 'hits': hits, 'count': len(hits)}
        return self._api_call_ui(_ui, body)

    def _api_translation_assist_sfx(self, body: dict):
        def _ui(_b: dict):
            if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
                raise ValueError('open a project first')
            query = str((_b or {}).get('query', '') or '').strip()
            if not query:
                _p, _idx, blk = self._resolve_assist_block(_b or {})
                query = (getattr(blk, 'get_text', lambda: '')() or '').strip()
            store = list(getattr(self.imgtrans_proj, 'sfx_dictionary', []) or [])
            rows = default_sfx_dictionary() + store if bool((_b or {}).get('include_defaults', True)) else store
            hits = query_sfx_dictionary(rows, query, limit=int((_b or {}).get('limit', 20) or 20))
            return {'ok': True, 'query': query, 'hits': hits, 'count': len(hits)}
        return self._api_call_ui(_ui, body)

    def _api_translation_assist_add_to_tm(self, body: dict):
        def _ui(_b: dict):
            if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
                raise ValueError('open a project first')
            page, block, blk = self._resolve_assist_block(_b or {})
            text = str((_b or {}).get('text', '') or '').strip()
            if not text:
                raise ValueError('text is required')
            source = (getattr(blk, 'get_text', lambda: '')() or '').strip()
            store = list(getattr(self.imgtrans_proj, 'translation_memory', []) or [])
            add_tm_entry(store, source, text, page=str(page), block_id=str(block))
            self.imgtrans_proj.translation_memory = store
            self.canvas.setProjSaveState(True)
            return {'ok': True, 'entries': len(store), 'source': source, 'target': text}
        return self._api_call_ui(_ui, body)

    def _api_translation_assist_add_to_glossary(self, body: dict):
        def _ui(_b: dict):
            if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
                raise ValueError('open a project first')
            source = str((_b or {}).get('source', '') or '').strip()
            target = str((_b or {}).get('target', '') or '').strip()
            if not source or not target:
                raise ValueError('source and target are required')
            store = list(getattr(self.imgtrans_proj, 'translation_glossary', []) or [])
            seen = {str((r or {}).get('source', '') or '').strip().lower() for r in store}
            if source.lower() not in seen:
                store.append({'source': source, 'target': target})
            self.imgtrans_proj.translation_glossary = store
            self.canvas.setProjSaveState(True)
            return {'ok': True, 'entries': len(store), 'source': source, 'target': target}
        return self._api_call_ui(_ui, body)

    def _api_translation_assist_qa(self, body: dict):
        def _ui(_b: dict):
            if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
                raise ValueError('open a project first')
            _page_idx, _block_idx, blk = self._resolve_assist_block(_b or {})
            src = (getattr(blk, 'get_text', lambda: '')() or '').strip()
            tgt = str(getattr(blk, 'translation', '') or '').strip()
            glossary = list(getattr(self.imgtrans_proj, 'translation_glossary', []) or [])
            from utils.translation_review import check_translation_guardrails
            issues = check_translation_guardrails(src, tgt, glossary=glossary)
            warnings = [str(it.get('message', '')) for it in issues if str(it.get('message', '')).strip()]
            return {'ok': True, 'source_text': src, 'target_text': tgt, 'warnings': warnings, 'issue_count': len(warnings)}
        return self._api_call_ui(_ui, body)

    def _api_translation_assist_profiles(self, body: dict):
        def _ui(_b: dict):
            action = str((_b or {}).get('action', 'get') or 'get').strip().lower()
            if action == 'set':
                pcfg.module.preferred_assist_providers = list((_b or {}).get('preferred_assist_providers', getattr(pcfg.module, 'preferred_assist_providers', ['current_translator'])) or [])
                pcfg.module.default_assist_prompt_profile = str((_b or {}).get('default_assist_prompt_profile', getattr(pcfg.module, 'default_assist_prompt_profile', 'dialogue')) or 'dialogue')
            return {
                'ok': True,
                'preferred_assist_providers': list(getattr(pcfg.module, 'preferred_assist_providers', ['current_translator']) or []),
                'default_assist_prompt_profile': str(getattr(pcfg.module, 'default_assist_prompt_profile', 'dialogue') or 'dialogue'),
                'prompt_profiles': sorted(PROMPT_PROFILES.keys()),
            }
        return self._api_call_ui(_ui, body)

    def _api_translation_assist_apply_text(self, body: dict):
        def _ui(_b: dict):
            page, block, _blk = self._resolve_assist_block(_b or {})
            text = str((_b or {}).get('text', '') or '').strip()
            if not text:
                raise ValueError('text is required')
            return {'ok': True, 'page': page, 'block': block, 'text': text}
        return self._api_call_ui(_ui, body)

    def _api_translation_assist_cache(self, body: dict):
        def _ui(_b: dict):
            action = str((_b or {}).get('action', 'status') or 'status').strip().lower()
            svc = self._translation_assist_service()
            if action == 'clear':
                return {'ok': True, 'cleared': svc.clear_cache()}
            return {'ok': True, 'entries': len(getattr(svc, 'candidate_cache', {}) or {})}
        return self._api_call_ui(_ui, body)

    def _api_translation_assist_compare(self, body: dict):
        def _ui(_b: dict):
            if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
                raise ValueError('open a project first')
            _p, _bidx, blk = self._resolve_assist_block(_b or {})
            scope = str((_b or {}).get('compare_scope', 'translator') or 'translator').strip().lower()
            if scope in {'ocr', 'detector', 'inpainter'}:
                current_map = {
                    'ocr': str(getattr(pcfg.module, 'ocr', '') or ''),
                    'detector': str(getattr(pcfg.module, 'textdetector', '') or ''),
                    'inpainter': str(getattr(pcfg.module, 'inpainter', '') or ''),
                }
                valid_map = {
                    'ocr': sorted(GET_VALID_OCR()),
                    'detector': sorted(GET_VALID_TEXTDETECTORS()),
                    'inpainter': sorted(GET_VALID_INPAINTERS()),
                }
                preferred = [str(x).strip() for x in list((_b or {}).get('preferred_assist_providers', []) or []) if str(x).strip()]
                max_n = max(1, int((_b or {}).get('max_candidates', getattr(pcfg.module, 'max_mt_candidates', 6)) or 6))
                names = []
                for nm in preferred + valid_map.get(scope, []):
                    if nm and nm in valid_map.get(scope, []) and nm not in names:
                        names.append(nm)
                names = names[:max_n]
                candidates = []
                for i, nm in enumerate(names, 1):
                    candidates.append({
                        'candidate_id': f"module_{scope}_{i}",
                        'provider': f"{scope}:{nm}",
                        'text': nm,
                        'telemetry': {'source': f'{scope}_module_registry', 'is_current': (nm == current_map.get(scope, ''))},
                    })
                return {'ok': True, 'preset': 'module_compare', 'scope': scope, 'providers': names, 'source_text': (getattr(blk, 'get_text', lambda: '')() or '').strip(), 'candidates': candidates, 'count': len(candidates), 'warnings': []}
            source = str((_b or {}).get('source_text', '') or '').strip()
            if not source:
                source = (getattr(blk, 'get_text', lambda: '')() or '').strip()
            preset = str((_b or {}).get('preset', 'low_latency') or 'low_latency').strip().lower()
            if preset == 'high_quality':
                providers = ['TM', 'Glossary', 'Concordance', 'SFX', 'openai', 'deepl']
            else:
                providers = ['TM', 'Glossary', 'SFX', 'google']
            telemetry = {}
            warnings = []
            tm_hits = glossary_hits = sfx_hits = concordance_hits = []
            max_n = int((_b or {}).get('max_candidates', getattr(pcfg.module, 'max_mt_candidates', 6)) or 6)
            if 'TM' in providers:
                t0 = time.perf_counter(); tm_hits = list(self._api_translation_assist_tm({'page': _p, 'block': _bidx, 'source_text': source}).get('hits', []) or [])
                telemetry['TM'] = {'latency_ms': int((time.perf_counter()-t0)*1000), 'source': 'project_tm'}
            if 'Glossary' in providers:
                t0 = time.perf_counter(); glossary_hits = list(self._api_translation_assist_glossary({'page': _p, 'block': _bidx, 'source_text': source}).get('hits', []) or [])
                telemetry['Glossary'] = {'latency_ms': int((time.perf_counter()-t0)*1000), 'source': 'project_glossary'}
            if 'SFX' in providers:
                t0 = time.perf_counter(); sfx_hits = list(self._api_translation_assist_sfx({'page': _p, 'block': _bidx, 'query': source}).get('hits', []) or [])
                telemetry['SFX'] = {'latency_ms': int((time.perf_counter()-t0)*1000), 'source': 'sfx_dictionary'}
            if 'Concordance' in providers:
                t0 = time.perf_counter(); concordance_hits = list(self._api_translation_assist_concordance({'page': _p, 'block': _bidx, 'query': source, 'limit': max_n}).get('hits', []) or [])
                telemetry['Concordance'] = {'latency_ms': int((time.perf_counter()-t0)*1000), 'source': 'project_concordance'}
            candidates = build_candidates_from_sources(
                tm_hits=tm_hits, glossary_hits=glossary_hits, sfx_hits=sfx_hits, concordance_hits=concordance_hits, max_candidates=max_n, telemetry=telemetry
            )
            # external provider fan-out (synthetic candidate orchestration placeholder)
            external = [p for p in providers if p.lower() in {'openai', 'deepl', 'google', 'ollama', 'lmstudio'}]
            for p in external:
                try:
                    t0 = time.perf_counter()
                    row = make_synthetic_mt_candidate(source, p, mode=preset)
                    tele = {
                        "latency_ms": int((time.perf_counter()-t0)*1000),
                        "source": "external_provider",
                        "cost_usd": estimate_candidate_cost_usd(row.get("text", ""), p),
                    }
                    candidates.append({
                        "candidate_id": f"ext_{p}_{len(candidates)+1}",
                        "provider": p,
                        "text": row.get("text", ""),
                        "telemetry": tele,
                    })
                except Exception as e:
                    nw = normalize_provider_warning(p, err=e)
                    if nw:
                        warnings.append(nw)
            if external and not warnings:
                # explicit caveat until true provider orchestration lands
                for p in external:
                    nw = normalize_provider_warning(p, warning_text="synthetic_compare_candidate: provider call not executed in this baseline slice")
                    if nw:
                        warnings.append(nw)
            candidates = candidates[:max_n]
            return {'ok': True, 'preset': preset, 'scope': scope, 'providers': providers, 'source_text': source, 'candidates': candidates, 'count': len(candidates), 'warnings': warnings}
        return self._api_call_ui(_ui, body)

    def _resolve_assist_block(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        pages = list((self.imgtrans_proj.pages or {}).keys())
        page_v = body.get('page', self.imgtrans_proj.current_img if getattr(self.imgtrans_proj, 'current_img', None) else '')
        if isinstance(page_v, int):
            page_idx = int(page_v)
            if page_idx < 0 or page_idx >= len(pages):
                raise ValueError('invalid page index')
            page = pages[page_idx]
        else:
            page = str(page_v or '').strip()
            if page not in (self.imgtrans_proj.pages or {}):
                raise ValueError('invalid page')
            page_idx = pages.index(page)
        block_idx = int(body.get('block', 0) or 0)
        blocks = list((self.imgtrans_proj.pages or {}).get(page, []) or [])
        if block_idx < 0 or block_idx >= len(blocks):
            raise ValueError('invalid block index')
        return page_idx, block_idx, blocks[block_idx]

    def _api_image_transform(self, body: dict):
        return self._api_call_ui(self._api_image_transform_ui, body)

    def _api_image_transform_ui(self, body: dict):
        data = body if isinstance(body, dict) else {}
        input_path = str(data.get('input_path', '') or '').strip()
        output_path = str(data.get('output_path', '') or '').strip()
        if not input_path or not output_path:
            raise ValueError('input_path and output_path are required')
        crop = data.get('crop')
        resize = data.get('resize')
        scale = data.get('scale')
        border = int(data.get('border', 0) or 0)
        border_color = tuple((data.get('border_color') or [0, 0, 0])[:3])
        brightness = float(data.get('brightness', 0.0) or 0.0)
        contrast = float(data.get('contrast', 1.0) or 1.0)
        perspective_src = data.get('perspective_src')
        perspective_dst = data.get('perspective_dst')
        opts = TransformOptions(
            crop=tuple(crop[:4]) if isinstance(crop, (list, tuple)) and len(crop) >= 4 else None,
            resize=tuple(resize[:2]) if isinstance(resize, (list, tuple)) and len(resize) >= 2 else None,
            scale=float(scale) if scale is not None else None,
            border=border,
            border_color=border_color,
            brightness=brightness,
            contrast=contrast,
            perspective_src=tuple(tuple(float(v) for v in p[:2]) for p in perspective_src[:4]) if isinstance(perspective_src, (list, tuple)) and len(perspective_src) >= 4 else None,
            perspective_dst=tuple(tuple(float(v) for v in p[:2]) for p in perspective_dst[:4]) if isinstance(perspective_dst, (list, tuple)) and len(perspective_dst) >= 4 else None,
        )
        result = transform_image_file(input_path, output_path, opts)
        add_as_next_page = bool(data.get('add_as_next_page', False))
        if add_as_next_page:
            if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
                raise ValueError('open a project first for add_as_next_page')
            src_page = str(data.get('source_page', '') or self.imgtrans_proj.current_img or '').strip()
            if not src_page or src_page not in self.imgtrans_proj.pages:
                raise ValueError('source_page not found in current project')
            rel_name = osp.basename(output_path)
            if osp.isabs(output_path):
                proj_dir = osp.abspath(self.imgtrans_proj.directory)
                out_abs = osp.abspath(output_path)
                if not out_abs.startswith(proj_dir):
                    raise ValueError('output_path must be inside current project directory for add_as_next_page')
                rel_name = osp.relpath(out_abs, proj_dir).replace('\\', '/')
            pages_order = list(self.imgtrans_proj.pages.keys())
            try:
                src_idx = pages_order.index(src_page)
            except ValueError:
                src_idx = len(pages_order) - 1
            if rel_name in self.imgtrans_proj.pages:
                raise ValueError(f'page already exists in project: {rel_name}')
            new_pages = {}
            for i, p in enumerate(pages_order):
                new_pages[p] = self.imgtrans_proj.pages[p]
                if i == src_idx:
                    new_pages[rel_name] = []
            if rel_name not in new_pages:
                new_pages[rel_name] = []
            self.imgtrans_proj.pages = new_pages
            self.imgtrans_proj._pagename2idx = {}
            self.imgtrans_proj._idx2pagename = {}
            for ii, imgname in enumerate(self.imgtrans_proj.pages.keys()):
                self.imgtrans_proj._pagename2idx[imgname] = ii
                self.imgtrans_proj._idx2pagename[ii] = imgname
                if imgname not in self.imgtrans_proj._image_info:
                    self.imgtrans_proj._image_info[imgname] = {'finish_code': 0, 'ignored': False, 'completion_state': 'todo'}
            self.imgtrans_proj.save()
            self.updatePageList()
            result['added_page'] = rel_name
            result['insert_after'] = src_page
        return {'ok': True, **result}

    def _api_pipeline_presets_ui(self, body: dict):
        action = str((body or {}).get('action', 'list') or 'list').strip().lower()
        if action == 'list':
            return {'ok': True, 'presets': list_workflow_presets(), 'current': workflow_stage_vector(pcfg.module)}
        if action in {'apply', 'run'}:
            preset_id = str((body or {}).get('preset', '') or '').strip()
            result = apply_workflow_preset(pcfg.module, preset_id)
            self._sync_stage_actions_from_config()
            save_config()
            if action == 'run':
                self.on_run_imgtrans()
            self.pipelineInsightsPanel.add_event('API', self.tr('Pipeline preset {0}: {1}').format(action, result.get('label', preset_id)))
            return {'ok': True, 'applied': result, 'current': workflow_stage_vector(pcfg.module), 'started': action == 'run'}
        raise ValueError(f'unknown pipeline_presets action: {action}')

    def _api_open_project(self, body: dict):
        return self._api_call_ui(self._api_open_project_ui, body)

    def _api_open_project_ui(self, body: dict):
        path = str((body or {}).get('path', '')).strip()
        if not path:
            raise ValueError('path is required')
        self.OpenProj(path)
        return {'opened': path}

    def _api_job_start(self, body: dict):
        task = normalize_job_task(str((body or {}).get('task', '') or ''))
        payload = dict((body or {}).get('payload') or {})
        payload['__job_id'] = job_id = f"job_{int(time.time() * 1000)}_{threading.get_ident()}"
        job = new_job(job_id, task)
        with self._api_jobs_lock:
            self._api_jobs[job_id] = job
            self._trim_api_jobs_locked()

        def _runner():
            with self._api_jobs_lock:
                j = self._api_jobs.get(job_id)
                if j is None:
                    return
                if j.get('cancel_requested'):
                    set_status(j, 'cancelled', stage='cancelled:before_start', progress=0.0)
                    append_log(j, 'cancelled before start')
                    return
                mark_started(j, task)
            try:
                with self._api_jobs_lock:
                    j = self._api_jobs.get(job_id)
                    if j is None:
                        return
                    if checkpoint_or_cancel(j, 'dispatch', 0.08):
                        return
                if checkpoint_or_cancel(j, 'task_dispatch', 0.12):
                    return
                if task == 'run_pipeline':
                    set_status(j, 'running', stage='pipeline', progress=0.2)
                    rst = self._api_run_pipeline(payload)
                elif task == 'render_page':
                    set_status(j, 'running', stage='render_page', progress=0.35)
                    rst = self._api_render_current_page(payload)
                elif task == 'export':
                    set_status(j, 'running', stage='export', progress=0.45)
                    rst = self._api_export(payload)
                elif task == 'batch_export':
                    set_status(j, 'running', stage='batch_export', progress=0.45)
                    rst = self._api_batch_export(payload)
                else:
                    set_status(j, 'running', stage='proof_pack', progress=0.45)
                    rst = self._api_export_lettering_proof(payload)
                with self._api_jobs_lock:
                    j = self._api_jobs.get(job_id)
                    if j is None:
                        return
                    update_from_task_result(j, rst)
                    if j.get('cancel_requested'):
                        mark_finished(j, task, cancelled=True)
                    else:
                        mark_finished(j, task, cancelled=False)
                    j['result'] = rst
            except Exception as e:
                with self._api_jobs_lock:
                    j = self._api_jobs.get(job_id)
                    if j is None:
                        return
                    set_status(j, 'failed', stage='failed')
                    add_warning(j, str(e))
                    append_log(j, f'failed: {e}')

        threading.Thread(target=_runner, daemon=True).start()
        return {'ok': True, 'job_id': job_id, 'status': 'queued', 'task': task}

    def _api_job_status(self, body: dict):
        job_id = str((body or {}).get('job_id', '') or '').strip()
        if not job_id:
            raise ValueError('job_id is required')
        with self._api_jobs_lock:
            job = self._api_jobs.get(job_id)
            if job is None:
                raise ValueError('unknown job_id')
            return status_payload(job)

    def _api_job_cancel(self, body: dict):
        job_id = str((body or {}).get('job_id', '') or '').strip()
        if not job_id:
            raise ValueError('job_id is required')
        with self._api_jobs_lock:
            job = self._api_jobs.get(job_id)
            if job is None:
                raise ValueError('unknown job_id')
            if job['status'] in {'succeeded', 'failed', 'cancelled'}:
                return {'ok': True, 'job_id': job_id, 'status': job['status'], 'cancel_requested': False}
            job['cancel_requested'] = True
            if job['status'] == 'queued':
                job['status'] = 'cancelled'
            job['updated_at'] = time.time()
            self._append_api_job_log_locked(job, 'cancel requested')
            return {'ok': True, 'job_id': job_id, 'status': job['status'], 'cancel_requested': True}

    def _api_job_logs(self, body: dict):
        job_id = str((body or {}).get('job_id', '') or '').strip()
        if not job_id:
            raise ValueError('job_id is required')
        with self._api_jobs_lock:
            job = self._api_jobs.get(job_id)
            if job is None:
                raise ValueError('unknown job_id')
            offset = int((body or {}).get('offset', 0) or 0)
            logs = list(job.get('logs', []))
            offset = max(0, min(offset, len(logs)))
            return {
                'job_id': job_id,
                'offset': offset,
                'next_offset': len(logs),
                'logs': logs[offset:],
                'warnings': list(job.get('warnings', [])),
            }

    def _api_job_result(self, body: dict):
        job_id = str((body or {}).get('job_id', '') or '').strip()
        if not job_id:
            raise ValueError('job_id is required')
        with self._api_jobs_lock:
            job = self._api_jobs.get(job_id)
            if job is None:
                raise ValueError('unknown job_id')
            return {'job_id': job_id, 'status': job['status'], 'result': job.get('result')}

    def _api_jobs_list(self, body: dict):
        limit = int((body or {}).get('limit', 50) or 50)
        limit = max(1, min(500, limit))
        with self._api_jobs_lock:
            jobs = sorted(self._api_jobs.values(), key=lambda r: float(r.get('created_at', 0.0)), reverse=True)
            out = []
            for j in jobs[:limit]:
                out.append({
                    'job_id': j.get('job_id', ''),
                    'task': j.get('task', ''),
                    'status': j.get('status', ''),
                    'created_at': float(j.get('created_at', 0.0)),
                    'updated_at': float(j.get('updated_at', 0.0)),
                    'cancel_requested': bool(j.get('cancel_requested', False)),
                })
            return {'jobs': out, 'count': len(out)}

    def _append_api_job_log_locked(self, job: dict, text: str):
        lim = int(getattr(pcfg, 'automation_api_job_log_limit', 200) or 200)
        append_log(job, text, limit=lim)

    def _trim_api_jobs_locked(self):
        lim = int(getattr(pcfg, 'automation_api_job_history_limit', 200) or 200)
        if lim <= 0:
            return
        jobs = sorted(self._api_jobs.items(), key=lambda kv: float((kv[1] or {}).get('created_at', 0.0)), reverse=True)
        for idx, (job_id, _job) in enumerate(jobs):
            if idx >= lim:
                self._api_jobs.pop(job_id, None)

    def _api_recent_projects(self, body: dict):
        return self._api_call_ui(self._api_recent_projects_ui, body)

    def _api_recent_projects_ui(self, body: dict):
        import os.path as _osp
        max_items = int((body or {}).get('limit', getattr(pcfg, 'recent_proj_list_max', 14)) or 14)
        entries = []
        for path in list(getattr(self.leftBar, 'recent_proj_list', []) or [])[:max(0, max_items)]:
            path = str(path or '')
            entries.append({
                'path': path,
                'exists': bool(_osp.exists(path)),
                'is_project_json': path.lower().endswith('.json'),
                'name': _osp.basename(path.rstrip(_osp.sep)) or path,
            })
        return {'recent_projects': entries, 'count': len(entries)}


    def _api_renderer_diagnostics(self, body: dict):
        return self._api_call_ui(self._api_renderer_diagnostics_ui, body)

    def _api_renderer_diagnostics_ui(self, body: dict):
        return {'ok': True, **collect_renderer_diagnostics()}

    def _api_cleanup_only(self, body: dict):
        return self._api_call_ui(self._api_cleanup_only_ui, body)

    def _api_cleanup_only_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        pages = list((body or {}).get('pages') or [])
        if not pages:
            pages = list(getattr(self.imgtrans_proj, 'pages', {}).keys())
        out_dir = str((body or {}).get('out_dir', '') or '').strip()
        halo_threshold = float((body or {}).get('halo_threshold', 0.18) or 0.18)
        inside_radius = int((body or {}).get('inside_radius', 1) or 1)
        outside_radius = int((body or {}).get('outside_radius', 2) or 2)
        detector_confidence = float((body or {}).get('detector_confidence', 1.0) or 1.0)
        detector = getattr(self.module_manager, 'textdetector', None)
        inpainter = getattr(self.module_manager, 'inpainter', None)
        if detector is None or inpainter is None:
            raise ValueError('detector and inpainter must be loaded')
        rst = run_cleanup_only_pages(self.imgtrans_proj, detector, inpainter, pages, out_dir=out_dir, halo_threshold=halo_threshold, inside_radius=inside_radius, outside_radius=outside_radius, detector_confidence=detector_confidence)
        if rst.get('processed'):
            self.canvas.setProjSaveState(True)
        return {'ok': True, **rst}

    def _api_batch_parent_enumerate(self, body: dict):
        return self._api_call_ui(self._api_batch_parent_enumerate_ui, body)

    def _api_batch_parent_enumerate_ui(self, body: dict):
        parent_path = str((body or {}).get('parent_path', '') or '').strip()
        if not parent_path:
            raise ValueError('parent_path is required')
        children = enumerate_child_projects(parent_path)
        return {
            'ok': True,
            'parent_path': parent_path,
            'count': len(children),
            'children': [
                {'kind': c.kind, 'input_path': c.input_path, 'display_name': c.display_name}
                for c in children
            ],
        }

    def _api_batch_parent_save_state(self, body: dict):
        return self._api_call_ui(self._api_batch_parent_save_state_ui, body)

    def _api_batch_parent_save_state_ui(self, body: dict):
        parent_path = str((body or {}).get('parent_path', '') or '').strip()
        state_path = str((body or {}).get('state_path', '') or '').strip()
        if not parent_path or not state_path:
            raise ValueError('parent_path and state_path are required')
        children = enumerate_child_projects(parent_path)
        statuses = dict((body or {}).get('statuses') or {})
        payload = save_parent_batch_state(state_path, parent_path, children, statuses=statuses)
        return {'ok': True, 'state_path': state_path, 'count': len(payload.get('children', [])), 'format': payload.get('format', '')}

    def _api_data_path_status(self, body: dict):
        return self._api_call_ui(self._api_data_path_status_ui, body)

    def _api_data_path_status_ui(self, body: dict):
        override_path = str((body or {}).get('path', '') or '')
        min_free_gb = float((body or {}).get('min_free_gb', 0.0) or 0.0)
        info = describe_data_path(override_path)
        info['ok'] = True
        info['enough_space'] = float(info.get('free_gb', 0.0)) >= min_free_gb
        info['min_free_gb'] = min_free_gb
        return info

    def _api_data_path_migrate(self, body: dict):
        return self._api_call_ui(self._api_data_path_migrate_ui, body)

    def _api_data_path_migrate_ui(self, body: dict):
        src = str((body or {}).get('source', '') or '')
        dst = str((body or {}).get('dest', '') or '')
        dry_run = bool((body or {}).get('dry_run', True))
        if not src or not dst:
            raise ValueError('source and dest are required')
        rst = migrate_data_path(src, dst, dry_run=dry_run)
        if not rst.get('ok'):
            raise ValueError(str(rst.get('error', 'migrate_failed')))
        rst['ok'] = True
        return rst

    def _api_batch_parent_load_state(self, body: dict):
        return self._api_call_ui(self._api_batch_parent_load_state_ui, body)

    def _api_batch_parent_load_state_ui(self, body: dict):
        state_path = str((body or {}).get('state_path', '') or '').strip()
        if not state_path:
            raise ValueError('state_path is required')
        payload = load_parent_batch_state(state_path)
        return {'ok': True, 'state_path': state_path, 'state': payload}

    def _api_batch_parent_update_status(self, body: dict):
        return self._api_call_ui(self._api_batch_parent_update_status_ui, body)

    def _api_batch_parent_update_status_ui(self, body: dict):
        state_path = str((body or {}).get('state_path', '') or '').strip()
        input_path = str((body or {}).get('input_path', '') or '').strip()
        status = str((body or {}).get('status', '') or '').strip()
        if not state_path or not input_path or not status:
            raise ValueError('state_path, input_path, status are required')
        payload = update_parent_batch_status(state_path, input_path, status)
        return {'ok': True, 'state_path': state_path, 'updated': input_path, 'status': status, 'count': len(payload.get('children', []))}

    def _api_batch_parent_next_pending(self, body: dict):
        return self._api_call_ui(self._api_batch_parent_next_pending_ui, body)

    def _api_batch_parent_next_pending_ui(self, body: dict):
        state_path = str((body or {}).get('state_path', '') or '').strip()
        if not state_path:
            raise ValueError('state_path is required')
        payload = load_parent_batch_state(state_path)
        nxt = next_pending_child(payload)
        return {'ok': True, 'state_path': state_path, 'next_pending': nxt, 'has_pending': nxt is not None}

    def _api_batch_parent_summary(self, body: dict):
        return self._api_call_ui(self._api_batch_parent_summary_ui, body)

    def _api_batch_parent_summary_ui(self, body: dict):
        state_path = str((body or {}).get('state_path', '') or '').strip()
        if not state_path:
            raise ValueError('state_path is required')
        payload = load_parent_batch_state(state_path)
        return {'ok': True, 'state_path': state_path, 'summary': summarize_parent_batch_state(payload)}

    def _api_server_mode_info(self, body: dict):
        return self._api_call_ui(self._api_server_mode_info_ui, body)

    def _api_server_mode_info_ui(self, body: dict):
        host = '127.0.0.1'
        port = int(getattr(pcfg, 'automation_api_port', 39542) or 39542)
        return {'ok': True, **build_server_mode_info(host=host, port=port)}

    def _api_import_translated_image_align(self, body: dict):
        return self._api_call_ui(self._api_import_translated_image_align_ui, body)

    def _api_import_translated_image_align_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        page = str((body or {}).get('page', '') or '').strip() or str(getattr(self.imgtrans_proj, 'current_img', '') or '')
        translated_image = str((body or {}).get('translated_image', '') or '').strip()
        min_iou = float((body or {}).get('min_iou', 0.2) or 0.2)
        if not page or not translated_image:
            raise ValueError('page and translated_image are required')
        raw_blocks = self.imgtrans_proj.pages.get(page, []) or []
        if not raw_blocks:
            raise ValueError('no raw blocks for page')
        if not osp.isfile(translated_image):
            raise ValueError('translated image not found')
        img_trans = imread(translated_image)
        if img_trans is None:
            raise ValueError('failed to read translated image')
        detector = getattr(self.module_manager, 'textdetector', None)
        ocr_runner = getattr(self.module_manager, 'ocr', None)
        if detector is None or ocr_runner is None:
            raise ValueError('detector and ocr must be loaded')
        _mask, tr_blocks = detector.detect(img_trans, self.imgtrans_proj)
        ocr_runner.run_ocr(img_trans, tr_blocks)
        rst = align_translations_by_iou(raw_blocks, tr_blocks or [], min_iou=min_iou)
        self.imgtrans_proj.pages[page] = raw_blocks
        self.canvas.setProjSaveState(True)
        return {'ok': True, 'page': page, 'translated_image': translated_image, **rst}

    def _api_project_status(self, body: dict):
        return self._api_call_ui(self._api_project_status_ui, body)

    def _api_project_status_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            return {'open': False, 'message': 'no project open'}
        pages = list(getattr(self.imgtrans_proj, 'pages', {}) or {})
        current = getattr(self.imgtrans_proj, 'current_img', '') or ''
        states = {}
        if hasattr(self.imgtrans_proj, 'get_page_completion_state'):
            for page in pages:
                states[page] = self.imgtrans_proj.get_page_completion_state(page)
        return {
            'open': True,
            'directory': getattr(self.imgtrans_proj, 'directory', ''),
            'current_page': current,
            'page_count': len(pages),
            'textbox_count': sum(len(self.imgtrans_proj.pages.get(p, []) or []) for p in pages),
            'unsaved': bool(getattr(self.canvas, 'projstate_unsaved', False)),
            'states': states,
        }

    def _api_run_pipeline(self, body: dict):
        return self._api_call_ui(self._api_run_pipeline_ui, body)

    def _api_run_pipeline_ui(self, body: dict):
        self.on_run_imgtrans()
        return {'started': True}

    def _api_apply_edit(self, body: dict):
        return self._api_call_ui(self._api_apply_edit_ui, body)

    def _api_apply_edit_ui(self, body: dict):
        from utils.api_edit_ops import (
            validate_batch_payload,
            EditValidationError,
            find_block_index_by_stable_id,
            describe_block_ref,
        )
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        raw_body = body if isinstance(body, dict) else {}
        strict = bool(raw_body.get('strict', True))
        try:
            ops = validate_batch_payload(raw_body if "ops" in raw_body else {"ops": [body or {}]})
        except EditValidationError as e:
            raise ValueError(f'edit_validation_failed: {e.to_payload()}')
        applied = []
        errors = []
        for op_index, op in enumerate(ops):
            try:
                if op.op == "undo":
                    self.on_undo()
                    applied.append({"op": "undo", "ok": True})
                    continue
                if op.op == "redo":
                    self.on_redo()
                    applied.append({"op": "redo", "ok": True})
                    continue
                page = op.page or self.imgtrans_proj.current_img
                blks = self.imgtrans_proj.pages.get(page, []) if page else []
                if op.op == "update_textbox":
                    if op.block_id:
                        idx = find_block_index_by_stable_id(blks, op.block_id)
                    else:
                        idx = int(op.index if op.index is not None else -1)
                    if idx < 0 or idx >= len(blks):
                        raise EditValidationError("index_out_of_range", "invalid textbox index", {"page": page, "index": idx, "count": len(blks)})
                    blks[idx].translation = str(op.text or "")
                    applied.append({"op": "update_textbox", **describe_block_ref(page, idx, blks[idx])})
                elif op.op == "delete_textbox":
                    if op.block_id:
                        idx = find_block_index_by_stable_id(blks, op.block_id)
                    else:
                        idx = int(op.index if op.index is not None else -1)
                    if idx < 0 or idx >= len(blks):
                        raise EditValidationError("index_out_of_range", "invalid textbox index", {"page": page, "index": idx, "count": len(blks)})
                    removed = blks[idx]
                    ref = describe_block_ref(page, idx, removed)
                    del blks[idx]
                    applied.append({"op": "delete_textbox", **ref})
                elif op.op == "add_textbox":
                    self.imgtrans_proj.current_img = page
                    self.shortcutCreateTextbox()
                    new_index = len(self.imgtrans_proj.pages.get(page, []) or []) - 1
                    if new_index >= 0:
                        nb = self.imgtrans_proj.pages[page][new_index]
                        nb.translation = str(op.text or "")
                        applied.append({"op": "add_textbox", **describe_block_ref(page, new_index, nb)})
                    else:
                        raise EditValidationError("add_failed", "failed to create textbox", {"page": page})
            except EditValidationError as e:
                errors.append({"op_index": op_index, "op": op.op, "error": e.to_payload()})
                if strict:
                    raise ValueError(f'edit_apply_failed: {errors[-1]}')
            except Exception as e:
                errors.append({"op_index": op_index, "op": op.op, "error": {"code": "apply_exception", "message": str(e)}})
                if strict:
                    raise ValueError(f'edit_apply_failed: {errors[-1]}')
        self.canvas.updateTextBlkList()
        if applied:
            self.canvas.setProjSaveState(True)
        return {'ok': len(errors) == 0, 'applied': applied, 'errors': errors, 'count': len(applied)}

    def _api_undo(self, body: dict):
        return self._api_call_ui(self._api_undo_ui, body)

    def _api_undo_ui(self, body: dict):
        self.on_undo()
        return {'ok': True}

    def _api_redo(self, body: dict):
        return self._api_call_ui(self._api_redo_ui, body)

    def _api_redo_ui(self, body: dict):
        self.on_redo()
        return {'ok': True}

    def _api_export(self, body: dict):
        return self._api_call_ui(self._api_export_ui, body)

    def _api_export_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        kind = str((body or {}).get('kind', 'rendered_batch') or 'rendered_batch').strip().lower()
        if kind in {'rendered', 'rendered_batch', 'images'}:
            out_dir = str((body or {}).get('out_dir', '') or '').strip()
            if not out_dir:
                out_dir = osp.join(self.imgtrans_proj.directory, 'exports')
            ext = str((body or {}).get('ext', '') or '').strip().lower() or None
            also_pdf = bool((body or {}).get('also_pdf', False))
            result = self._do_batch_export(
                out_dir, ext=ext, also_pdf=also_pdf, show_message=False,
                clean_after_export=bool((body or {}).get('clean_after_export', False)),
                include_intermediate=bool((body or {}).get('include_intermediate', False)),
                include_unrendered=bool((body or {}).get('include_unrendered', getattr(pcfg, 'export_include_unrendered_pages', False))),
                filename_template=str((body or {}).get('filename_template', '') or getattr(pcfg, 'export_filename_template', '{index:03d}')),
            )
            self.pipelineInsightsPanel.add_event('API', self.tr('Batch export via API: {0} page(s)').format(result.get('exported', 0)))
            return {'ok': True, **(result or {})}
        if kind in {'archive', 'zip', 'cbz'}:
            import tempfile
            archive_format = 'cbz' if kind == 'cbz' or str((body or {}).get('archive_format', '')).lower() == 'cbz' else 'zip'
            archive_path = str((body or {}).get('archive_path', '') or '').strip()
            out_dir = str((body or {}).get('out_dir', '') or '').strip()
            if not out_dir:
                out_dir = tempfile.mkdtemp(prefix='bt_export_')
            result = self._do_batch_export(
                out_dir,
                ext=str((body or {}).get('ext', '') or '').strip().lower() or None,
                also_pdf=bool((body or {}).get('also_pdf', False)),
                show_message=False,
                clean_after_export=bool((body or {}).get('clean_after_export', False)),
                include_intermediate=bool((body or {}).get('include_intermediate', False)),
                include_unrendered=bool((body or {}).get('include_unrendered', getattr(pcfg, 'export_include_unrendered_pages', False))),
                filename_template=str((body or {}).get('filename_template', '') or getattr(pcfg, 'export_filename_template', '{index:03d}')),
            )
            if not archive_path:
                archive_path = osp.join(self.imgtrans_proj.directory, 'exports', ('exported.cbz' if archive_format == 'cbz' else 'exported.zip'))
            os.makedirs(osp.dirname(archive_path), exist_ok=True)
            job_id = str((body or {}).get('__job_id', '') or '').strip()

            def _cancel_check() -> bool:
                if not job_id:
                    return False
                with self._api_jobs_lock:
                    j = self._api_jobs.get(job_id)
                    return bool(j and j.get('cancel_requested'))

            def _progress_hook(ev: dict):
                if not job_id:
                    return
                with self._api_jobs_lock:
                    j = self._api_jobs.get(job_id)
                    if not j:
                        return
                    p = float((ev or {}).get('progress', 0.0) or 0.0)
                    set_status(j, 'running', stage='archive_stream', progress=min(0.99, 0.45 + p * 0.5))
                    append_log(j, f"archive {int((ev or {}).get('index', 0))}/{int((ev or {}).get('total', 0))}: {(ev or {}).get('relative_path', '')}")

            stream_result = write_archive_streaming(out_dir, archive_path, cancel_check=_cancel_check if job_id else None, progress_hook=_progress_hook if job_id else None)
            self.pipelineInsightsPanel.add_event('API', self.tr('Archive export via API: {0}').format(archive_path))
            if stream_result.get('cancelled'):
                return {'ok': False, 'archive_path': archive_path, 'archive_format': archive_format, 'archive_stream': stream_result, 'warnings': ['archive export cancelled'], **(result or {})}
            return {'ok': True, 'archive_path': archive_path, 'archive_format': archive_format, 'archive_stream': stream_result, **(result or {})}
        if kind in {'current_page', 'render_current_page'}:
            return self._api_render_current_page_ui(body)
        if kind in {'structured_ocr', 'ocr_json'}:
            return self._api_export_structured_ocr_ui(body)
        if kind in {'xliff', 'xlf'}:
            return self._api_export_xliff_ui(body)
        if kind in {'translation_json', 'trans_json'}:
            return self._api_export_translation_json_ui(body)
        if kind in {'translation_csv', 'trans_csv', 'csv'}:
            return self._api_export_translation_csv_ui(body)
        if kind in {'svg_handoff', 'svg_text', 'editable_svg'}:
            if not self.imgtrans_proj.current_img:
                raise ValueError('select a page first')
            out_dir = str((body or {}).get('out_dir', '') or '').strip() or osp.join(self.imgtrans_proj.directory, 'svg_handoff')
            final_path = str((body or {}).get('final_image_path', '') or '').strip() or None
            manifest = build_svg_text_handoff(self.imgtrans_proj, self.imgtrans_proj.current_img, out_dir, final_image_path=final_path)
            return {'ok': True, 'manifest': manifest}
        if kind in {'psd_handoff', 'layered_psd'}:
            if not self.imgtrans_proj.current_img:
                raise ValueError('select a page first')
            out_dir = str((body or {}).get('out_dir', '') or '').strip() or osp.join(self.imgtrans_proj.directory, 'psd_handoff')
            requested_pages = list((body or {}).get('pages') or [self.imgtrans_proj.current_img])
            pages = [p for p in requested_pages if p in (getattr(self.imgtrans_proj, 'pages', {}) or {})]
            if not pages:
                raise ValueError('no valid pages for PSD handoff export')
            manifests = []
            warnings = []
            for page in pages:
                final_arr = None
                if page == self.imgtrans_proj.current_img:
                    qimg = self.canvas.render_result_img()
                    final_arr = pixmap2ndarray(qimg, keep_alpha=False) if qimg is not None and not qimg.isNull() else None
                elif bool((body or {}).get('include_final', False)):
                    warnings.append(f'final composite render is only available for current page; using saved result for {page}')
                manifest = build_layered_psd_handoff(self.imgtrans_proj, page, out_dir, final_image=final_arr)
                manifests.append(manifest)
            self.pipelineInsightsPanel.add_event('API', self.tr('Layered PSD handoff via API: {0} page(s)').format(len(manifests)))
            return {'ok': True, 'manifest': manifests[0] if len(manifests) == 1 else None, 'manifests': manifests, 'warnings': warnings, 'count': len(manifests)}
        if kind in {'lettering_proof', 'proof_pack', 'typography_proof'}:
            return self._api_export_lettering_proof_ui(body)
        raise ValueError(f'unknown export kind: {kind}')

    def _api_batch_export(self, body: dict):
        payload = dict(body or {})
        payload.setdefault('kind', 'batch')
        return self._api_export(payload)

    def _api_layout_review(self, body: dict):
        return self._api_call_ui(self._api_layout_review_ui, body)

    def _api_layout_review_ui(self, body: dict):
        mode = str((body or {}).get('mode', 'page')).strip().lower()
        apply = bool((body or {}).get('apply', True))
        result, target_indices = self._layout_review_result_for_scope(mode)
        summary = self._summarize_layout_review_result(result)
        applied = 0
        if apply and summary['actions'] > 0:
            applied = self.st_manager.apply_review_result(result, target_block_indices=target_indices)
            if self.imgtrans_proj and self.imgtrans_proj.directory:
                self.imgtrans_proj.save()
        return {'ok': True, 'mode': mode, 'applied_actions': applied, 'summary': summary}


    def _api_render_current_page(self, body: dict):
        return self._api_call_ui(self._api_render_current_page_ui, body)

    def _api_render_current_page_ui(self, body: dict):
        from utils.io_utils import imwrite
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            raise ValueError('open a project and select a page first')
        qimg = self.canvas.render_result_img()
        if qimg is None or qimg.isNull():
            raise RuntimeError('render produced no image')
        path = str((body or {}).get('path', '') or '').strip()
        if not path:
            base = osp.splitext(self.imgtrans_proj.current_img)[0]
            path = osp.join(self.imgtrans_proj.result_dir(), base + (pcfg.imgsave_ext or '.png'))
        arr = pixmap2ndarray(qimg, keep_alpha=False)
        ext = osp.splitext(path)[1].lower() or pcfg.imgsave_ext or '.png'
        imwrite(path, arr, ext=ext, quality=pcfg.imgsave_quality)
        self.pipelineInsightsPanel.add_event('API', self.tr('Rendered current page to {0}').format(path))
        manifest_path = str((body or {}).get('manifest_path', '') or '').strip()
        manifest = {
            'format': 'ballonstranslator.render_current_page.v1',
            'page': self.imgtrans_proj.current_img,
            'path': path,
            'extension': ext,
            'quality': pcfg.imgsave_quality,
            'warnings': [],
        }
        if bool((body or {}).get('write_manifest', False)):
            import json
            if not manifest_path:
                manifest_path = osp.splitext(path)[0] + '.render-manifest.json'
            manifest_dir = osp.dirname(manifest_path)
            if manifest_dir:
                os.makedirs(manifest_dir, exist_ok=True)
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            manifest['manifest_path'] = manifest_path
        return {'ok': True, 'page': self.imgtrans_proj.current_img, 'path': path, 'manifest': manifest}

    def _api_list_rendering_issues(self, body: dict):
        return self._api_call_ui(self._api_list_rendering_issues_ui, body)

    def _api_list_rendering_issues_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        from utils.rendering_qa import build_project_rendering_qa, flatten_rendering_qa_rows, summarize_suggested_actions
        pages = list((body or {}).get('pages') or [self.imgtrans_proj.current_img] or [])
        include_ok = bool((body or {}).get('include_ok', False))
        report = build_project_rendering_qa(self.imgtrans_proj, pages=pages, include_ok=include_ok, config_obj=pcfg)
        rows = flatten_rendering_qa_rows(report)
        issues = []
        for page in report.get('pages', []) or []:
            for block in page.get('blocks', []) or []:
                if include_ok or block.get('warnings'):
                    issues.append(block)
        return {'ok': True, 'issues': issues, 'rows': rows, 'count': len(issues), 'summary': report.get('summary', {}), 'action_summary': summarize_suggested_actions(report)}

    def _api_lettering_workflow(self, body: dict):
        return self._api_call_ui(self._api_lettering_workflow_ui, body)

    def _api_lettering_workflow_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            raise ValueError('open a project and select a page first')
        from utils.lettering_workflow import build_lettering_workflow_plan
        requested_pages = list((body or {}).get('pages') or [self.imgtrans_proj.current_img])
        pages = [p for p in requested_pages if p in (getattr(self.imgtrans_proj, 'pages', {}) or {})]
        if not pages:
            raise ValueError('no valid pages for lettering workflow')
        plan = build_lettering_workflow_plan(self.imgtrans_proj, pages=pages, config_obj=pcfg, include_ok=bool((body or {}).get('include_ok', False)))
        apply = bool((body or {}).get('apply', False))
        export_proof = bool((body or {}).get('export_proof', False))
        render = bool((body or {}).get('render', False))
        warnings = []
        result = {'ok': True, 'pages': pages, 'plan': plan, 'applied_actions': 0, 'proof_manifests': [], 'proof_manifest': None, 'rendered': None, 'warnings': warnings}
        if apply and plan.get('selected_fixes'):
            from utils.rendering_qa import apply_project_rendering_fixes
            fix_result = apply_project_rendering_fixes(self.imgtrans_proj, pages=pages, config_obj=pcfg, selected_fixes=plan.get('selected_fixes'))
            result['fix_result'] = fix_result
            result['applied_actions'] = int(fix_result.get('applied_count', 0) or 0)
            if self.imgtrans_proj.current_img in pages:
                self.st_manager.updateSceneTextitems()
            if self.imgtrans_proj.directory:
                self.imgtrans_proj.save()
            self.canvas.setProjSaveState(False)
        if export_proof:
            out_dir = str((body or {}).get('out_dir', '') or '').strip() or osp.join(self.imgtrans_proj.directory, 'lettering_proofs')
            include_final = bool((body or {}).get('include_final', True))
            for page in pages:
                final_arr = None
                if include_final and page == self.imgtrans_proj.current_img:
                    qimg = self.canvas.render_result_img()
                    final_arr = pixmap2ndarray(qimg, keep_alpha=False) if qimg is not None and not qimg.isNull() else None
                elif include_final:
                    warnings.append(f'proof final composite only included for current page; skipped final render for {page}')
                manifest = build_lettering_proof_pack(self.imgtrans_proj, page, out_dir, final_image=final_arr, config_obj=pcfg)
                result['proof_manifests'].append(manifest)
                if result.get('proof_manifest') is None:
                    result['proof_manifest'] = manifest
        if render:
            if self.imgtrans_proj.current_img in pages:
                result['rendered'] = self._api_render_current_page_ui({'path': (body or {}).get('render_path', ''), 'write_manifest': True})
            else:
                warnings.append('render_current_page skipped because current page is outside workflow scope')
            if len(pages) > 1:
                warnings.append('batch rerender is deferred; use batch export after opening each page or run pipeline/export queue')
        self.pipelineInsightsPanel.add_event('TYPO_QA', self.tr('Lettering workflow planned {0} step(s), applied {1} action(s) across {2} page(s)').format(len(plan.get('steps', [])), result.get('applied_actions', 0), len(pages)))
        return result

    def _api_next_rendering_issue(self, body: dict):
        return self._api_call_ui(self._api_next_rendering_issue_ui, body)

    def _api_next_rendering_issue_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            raise ValueError('open a project and select a page first')
        from utils.rendering_qa import build_project_rendering_qa
        from utils.lettering_workflow import next_rendering_issue
        page = str((body or {}).get('after_page', '') or self.imgtrans_proj.current_img)
        after_index = int((body or {}).get('after_index', -1) if (body or {}).get('after_index', None) is not None else -1)
        if bool((body or {}).get('after_selection', False)):
            sel = self.canvas.selected_text_items()
            if sel:
                after_index = max(int(getattr(x, 'idx', -1)) for x in sel)
        report = build_project_rendering_qa(self.imgtrans_proj, pages=[self.imgtrans_proj.current_img], include_ok=False, config_obj=pcfg)
        issue = next_rendering_issue(report, after_page=page, after_index=after_index)
        if bool((body or {}).get('select', False)) and issue.get('found') and issue.get('page') == self.imgtrans_proj.current_img:
            idx = int(issue.get('index', -1))
            if 0 <= idx < len(self.st_manager.textblk_item_list):
                self.canvas.block_selection_signal = True
                self.canvas.clearSelection()
                item = self.st_manager.textblk_item_list[idx]
                item.setSelected(True)
                self.canvas.block_selection_signal = False
                self.canvas.gv.ensureVisible(item)
                self.st_manager.txtblkShapeControl.setBlkItem(item)
                self.st_manager.textEditList.set_selected_list([idx])
        return {'ok': True, **issue}


    def _api_list_pages(self, body: dict):
        return self._api_call_ui(self._api_list_pages_ui, body)

    def _api_list_pages_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        from utils.text_rendering import sort_blocks_for_reading_order
        order = str((body or {}).get('reading_order', '') or getattr(pcfg, 'render_default_reading_order', 'auto') or 'auto')
        include_blocks = bool((body or {}).get('include_blocks', True))
        include_rendering_qa = bool((body or {}).get('include_rendering_qa', False))
        qa_by_page = {}
        if include_blocks and include_rendering_qa:
            try:
                from utils.rendering_qa import build_project_rendering_qa
                qa_report = build_project_rendering_qa(self.imgtrans_proj, pages=list(self.imgtrans_proj.pages.keys()), include_ok=True, config_obj=pcfg)
                for page_entry in qa_report.get('pages', []) or []:
                    qa_by_page[page_entry.get('page')] = {int(block.get('index', -1)): block for block in page_entry.get('blocks', []) or []}
            except Exception as exc:
                qa_by_page['__error__'] = str(exc)
        pages_out = []
        for page_index, page_name in enumerate(list(self.imgtrans_proj.pages.keys())):
            blocks = list(self.imgtrans_proj.pages.get(page_name, []) or [])
            sorted_blocks, resolved_order = sort_blocks_for_reading_order(blocks, order)
            entry = {
                'index': page_index,
                'name': page_name,
                'completion_state': self.imgtrans_proj.get_page_completion_state(page_name) if hasattr(self.imgtrans_proj, 'get_page_completion_state') else 'todo',
                'textboxes': len(blocks),
                'reading_order': resolved_order,
            }
            if include_blocks:
                original_index = {id(block): i for i, block in enumerate(blocks)}
                entry['blocks'] = [
                    {
                        'index': i,
                        'source_index': original_index.get(id(block), i),
                        'xyxy': list(getattr(block, 'xyxy', []) or []),
                        'text': getattr(block, 'translation', '') or (block.get_text() if hasattr(block, 'get_text') else ''),
                        'writing_mode': getattr(getattr(block, 'fontformat', None), 'writing_mode', 'auto'),
                        'fit_mode': getattr(getattr(block, 'fontformat', None), 'fit_mode', 'shrink'),
                        **({'rendering_qa': qa_by_page.get(page_name, {}).get(original_index.get(id(block), i), {})} if include_rendering_qa else {}),
                    }
                    for i, block in enumerate(sorted_blocks)
                ]
                if include_rendering_qa and '__error__' in qa_by_page:
                    entry['rendering_qa_error'] = qa_by_page['__error__']
            pages_out.append(entry)
        return {'ok': True, 'count': len(pages_out), 'pages': pages_out}

    def _api_page_state(self, body: dict):
        return self._api_call_ui(self._api_page_state_ui, body)

    def _api_page_state_ui(self, body: dict):
        action = str((body or {}).get('action', 'list')).strip().lower()
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        pages = list((body or {}).get('pages') or self.imgtrans_proj.pages.keys())
        if action == 'set':
            state = str((body or {}).get('state', 'todo')).strip().lower()
            for page in pages:
                self.imgtrans_proj.set_page_completion_state(page, state)
            self.imgtrans_proj.save()
            self._update_page_list_state_style()
        return {
            'ok': True,
            'states': {page: self.imgtrans_proj.get_page_completion_state(page) for page in pages if page in self.imgtrans_proj.pages},
        }





    def _api_export_lettering_proof(self, body: dict):
        return self._api_call_ui(self._api_export_lettering_proof_ui, body)

    def _api_export_lettering_proof_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            raise ValueError('open a project and select a page first')
        out_dir = str((body or {}).get('out_dir', '') or '').strip() or osp.join(self.imgtrans_proj.directory, 'lettering_proofs')
        include_final = bool((body or {}).get('include_final', True))
        final_arr = None
        if include_final:
            qimg = self.canvas.render_result_img()
            final_arr = pixmap2ndarray(qimg, keep_alpha=False) if qimg is not None and not qimg.isNull() else None
        manifest = build_lettering_proof_pack(self.imgtrans_proj, self.imgtrans_proj.current_img, out_dir, final_image=final_arr, config_obj=pcfg)
        self.pipelineInsightsPanel.add_event('TYPO_QA', self.tr('Exported lettering proof pack: {0}').format(manifest.get('page_dir', out_dir)))
        return {'ok': True, 'manifest': manifest}

    def _api_export_rendering_qa(self, body: dict):
        return self._api_call_ui(self._api_export_rendering_qa_ui, body)

    def _api_export_rendering_qa_ui(self, body: dict):
        from utils.rendering_qa import build_project_rendering_qa
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        pages = (body or {}).get('pages')
        include_ok = bool((body or {}).get('include_ok', False))
        report = build_project_rendering_qa(self.imgtrans_proj, pages=pages, include_ok=include_ok, config_obj=pcfg)
        path = str((body or {}).get('path', '') or '').strip()
        if path:
            if path.lower().endswith(('.md', '.markdown')):
                from utils.rendering_qa import rendering_qa_to_markdown
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(rendering_qa_to_markdown(report))
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
        return {'ok': True, 'path': path, 'summary': report.get('summary', {}), 'report': None if path else report}

    def _api_apply_project_rendering_fixes(self, body: dict):
        return self._api_call_ui(self._api_apply_project_rendering_fixes_ui, body)

    def _api_apply_project_rendering_fixes_ui(self, body: dict):
        from utils.rendering_qa import apply_project_rendering_fixes
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        pages = (body or {}).get('pages')
        selected_fixes = (body or {}).get('selected_fixes')
        result = apply_project_rendering_fixes(self.imgtrans_proj, pages=pages, config_obj=pcfg, selected_fixes=selected_fixes)
        if result.get('applied_count'):
            if self.imgtrans_proj.current_img in (pages or [self.imgtrans_proj.current_img]):
                self.st_manager.updateSceneTextitems()
            if self.imgtrans_proj.directory:
                self.imgtrans_proj.save()
            self.canvas.setProjSaveState(False)
        self.pipelineInsightsPanel.add_event('TYPO_QA', self.tr('Applied project rendering fixes: {0} text boxes').format(result.get('applied_count', 0)))
        return {'ok': True, **result}


    def _api_polish_typography(self, body: dict):
        return self._api_call_ui(self._api_polish_typography_ui, body)

    def _api_polish_typography_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            raise ValueError('open a project and select a page first')
        mode = str((body or {}).get('mode', 'page') or 'page').strip().lower()
        if mode == 'selected':
            indices = [blk.idx for blk in self.canvas.selected_text_items()]
        else:
            indices = [blk.idx for blk in getattr(self.st_manager, 'textblk_item_list', [])]
        explicit = (body or {}).get('indices')
        if explicit is not None:
            indices = [int(i) for i in explicit]
        changed = self.st_manager.polish_typography_textboxes(indices=indices, push_undo=False)
        if changed and self.imgtrans_proj and self.imgtrans_proj.directory:
            self.imgtrans_proj.save()
        self.pipelineInsightsPanel.add_event('TYPO_QA', self.tr('Polished typography: {0} text boxes').format(changed))
        return {'ok': True, 'page': self.imgtrans_proj.current_img, 'mode': mode, 'changed': changed, 'indices': indices}


    def _api_atomic_bubble_fit(self, body: dict):
        return self._api_call_ui(self._api_atomic_bubble_fit_ui, body)

    def _api_atomic_bubble_fit_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            raise ValueError('open a project and select a page first')
        indices = (body or {}).get('indices')
        if indices is not None:
            indices = [int(i) for i in indices]
        before = len(indices) if indices is not None else len(self.canvas.selected_text_items())
        profile = (body or {}).get('profile')
        changed = self.st_manager.atomic_bubble_fit_textboxes(indices=indices, push_undo=False, profile=profile)
        if changed:
            self.st_manager.updateSceneTextitems()
            if self.imgtrans_proj.directory:
                self.imgtrans_proj.save()
            self.canvas.setProjSaveState(False)
        self.pipelineInsightsPanel.add_event('API', self.tr('Atomic bubble fit updated {0} / {1} text boxes').format(changed, before))
        return {'ok': True, 'changed': changed, 'target_count': before, 'page': self.imgtrans_proj.current_img, 'profile': profile}

    def _api_smart_fit_textboxes(self, body: dict):
        return self._api_call_ui(self._api_smart_fit_textboxes_ui, body)

    def _api_smart_fit_textboxes_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            raise ValueError('open a project and select a page first')
        mode = str((body or {}).get('mode', 'page') or 'page').strip().lower()
        if mode == 'selected':
            indices = [blk.idx for blk in self.canvas.selected_text_items()]
        else:
            indices = [blk.idx for blk in getattr(self.st_manager, 'textblk_item_list', [])]
        explicit = (body or {}).get('indices')
        if explicit is not None:
            indices = [int(i) for i in explicit]
        changed = self.st_manager.smart_fit_textboxes(indices=indices, push_undo=False)
        if changed and self.imgtrans_proj and self.imgtrans_proj.directory:
            self.imgtrans_proj.save()
        return {'ok': True, 'page': self.imgtrans_proj.current_img, 'mode': mode, 'changed': changed, 'indices': indices}

    def _api_auto_format_textboxes(self, body: dict):
        return self._api_call_ui(self._api_auto_format_textboxes_ui, body)

    def _api_auto_format_textboxes_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            raise ValueError('open a project and select a page first')
        mode = str((body or {}).get('mode', 'page') or 'page').strip().lower()
        if mode == 'selected':
            indices = [blk.idx for blk in self.canvas.selected_text_items()]
        else:
            indices = [blk.idx for blk in getattr(self.st_manager, 'textblk_item_list', [])]
        explicit = (body or {}).get('indices')
        if explicit is not None:
            indices = [int(i) for i in explicit]
        profile = str((body or {}).get('profile', 'balanced') or 'balanced')
        changed = self.st_manager.auto_format_textboxes(indices=indices, push_undo=False, profile=profile)
        if changed and self.imgtrans_proj and self.imgtrans_proj.directory:
            self.imgtrans_proj.save()
        self.pipelineInsightsPanel.add_event('API', self.tr('Auto text formatting updated {0} text boxes').format(changed))
        return {'ok': True, 'page': self.imgtrans_proj.current_img, 'mode': mode, 'changed': changed, 'indices': indices, 'profile': profile}

    def _api_auto_format_qa_preview(self, body: dict):
        return self._api_call_ui(self._api_auto_format_qa_preview_ui, body)

    def _api_auto_format_qa_preview_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            raise ValueError('open a project and select a page first')
        mode = str((body or {}).get('mode', 'page') or 'page').strip().lower()
        only_risky = bool((body or {}).get('only_risky', False))
        if mode == 'selected':
            block_items = self.canvas.selected_text_items()
        else:
            block_items = list(getattr(self.st_manager, 'textblk_item_list', []) or [])
        profile = str((body or {}).get('profile', 'balanced') or 'balanced')
        rows = score_auto_format_candidates([getattr(x, 'blk', None) for x in block_items], profile=profile)
        if only_risky:
            rows = [r for r in rows if bool(r.get('before_overflow')) or float(r.get('improvement', 0.0) or 0.0) > 0.04]
        return {'ok': True, 'page': self.imgtrans_proj.current_img, 'mode': mode, 'profile': profile, 'rows': rows, 'summary': summarize_auto_format_scores(rows)}

    def _api_fix_rendering_issues(self, body: dict):
        return self._api_call_ui(self._api_fix_rendering_issues_ui, body)

    def _api_fix_rendering_issues_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            raise ValueError('open a project and select a page first')
        mode = str((body or {}).get('mode', 'page') or 'page').strip().lower()
        result, target_indices = self._layout_review_result_for_scope(mode)
        summary_before = self._summarize_layout_review_result(result)
        applied = self.st_manager.apply_review_result(result, target_block_indices=target_indices)
        if self.imgtrans_proj and self.imgtrans_proj.directory and applied:
            self.imgtrans_proj.save()
        issues_after = self._api_list_rendering_issues_ui({'pages': [self.imgtrans_proj.current_img]})
        self.pipelineInsightsPanel.add_event('API', self.tr('Fixed rendering issues: {0} actions').format(applied))
        return {
            'ok': True,
            'mode': mode,
            'applied_actions': applied,
            'summary_before': summary_before,
            'remaining_issues': issues_after.get('count', 0),
        }


    def _api_apply_text_style_batch(self, body: dict):
        return self._api_call_ui(self._api_apply_text_style_batch_ui, body)

    def _api_apply_text_style_batch_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        from utils.text_style_batch import apply_text_style_batch
        updates = (body or {}).get('updates') or {}
        pages = (body or {}).get('pages')
        indices = (body or {}).get('indices')
        result = apply_text_style_batch(
            self.imgtrans_proj,
            updates,
            pages=pages,
            indices=indices,
            only_auto_sized=bool((body or {}).get('only_auto_sized', False)),
            dry_run=bool((body or {}).get('dry_run', False)),
            config_obj=pcfg,
        )
        if result.get('changed') and not result.get('dry_run'):
            if self.imgtrans_proj.current_img in (result.get('touched_pages') or []):
                self.st_manager.updateSceneTextitems()
            if self.imgtrans_proj.directory:
                self.imgtrans_proj.save()
            self.canvas.setProjSaveState(False)
        self.pipelineInsightsPanel.add_event('API', self.tr('Batch text style API updated {0} text boxes').format(result.get('changed', 0)))
        return {'ok': True, **result}

    def _api_apply_rendering_preset(self, body: dict):
        return self._api_call_ui(self._api_apply_rendering_preset_ui, body)

    def _api_rendering_presets(self, body: dict):
        return self._api_call_ui(self._api_rendering_presets_ui, body)

    def _api_rendering_presets_ui(self, body: dict):
        from utils.text_rendering import manga_presets, preset_from_font_format, preset_id_from_label, sanitize_manga_preset
        from utils.rendering_preset_io import delete_custom_preset, import_preset_pack, preset_font_diagnostics, write_preset_pack
        action = str((body or {}).get('action', 'list') or 'list').strip().lower()
        if action == 'list':
            presets = manga_presets(pcfg)
            diagnostics = preset_font_diagnostics(presets, getattr(shared, 'FONT_FAMILIES', None) or [])
            return {'ok': True, 'presets': presets, 'count': len(presets), 'font_diagnostics': diagnostics}
        if action == 'save_current':
            if self.imgtrans_proj is None or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
                raise ValueError('open a project and select a page first')
            idx = int((body or {}).get('index', 0) or 0)
            blks = self.imgtrans_proj.pages.get(self.imgtrans_proj.current_img, [])
            if idx < 0 or idx >= len(blks):
                raise ValueError('invalid index')
            name = str((body or {}).get('name', '') or '').strip() or f'Preset {idx + 1}'
            custom = dict(getattr(pcfg, 'render_custom_manga_presets', {}) or {})
            preset_id = preset_id_from_label(name, list(custom.keys()))
            custom[preset_id] = preset_from_font_format(blks[idx].fontformat, label=name)
            pcfg.render_custom_manga_presets = custom
            save_config()
            return {'ok': True, 'preset_id': preset_id, 'preset': custom[preset_id], 'count': len(custom)}
        if action == 'save':
            name = str((body or {}).get('name', '') or '').strip() or 'Custom preset'
            preset_data = dict((body or {}).get('preset', {}) or {})
            custom = dict(getattr(pcfg, 'render_custom_manga_presets', {}) or {})
            preset_id = str((body or {}).get('preset_id', '') or '').strip() or preset_id_from_label(name, list(custom.keys()))
            if not preset_id.startswith('custom:'):
                preset_id = 'custom:' + preset_id
            custom[preset_id] = sanitize_manga_preset(preset_data, label=name)
            pcfg.render_custom_manga_presets = custom
            save_config()
            return {'ok': True, 'preset_id': preset_id, 'preset': custom[preset_id], 'count': len(custom)}
        if action == 'export':
            path = str((body or {}).get('path', '') or '').strip()
            if not path:
                raise ValueError('path is required')
            pack = write_preset_pack(pcfg, path, include_builtins=bool((body or {}).get('include_builtins', False)))
            diagnostics = preset_font_diagnostics(pack.get('presets', {}), getattr(shared, 'FONT_FAMILIES', None) or [])
            return {'ok': True, 'path': pack.get('path', path), 'count': len(pack.get('presets', {}) or {}), 'font_diagnostics': diagnostics}
        if action == 'import':
            path = str((body or {}).get('path', '') or '').strip()
            if not path:
                raise ValueError('path is required')
            result = import_preset_pack(pcfg, path, overwrite=bool((body or {}).get('overwrite', False)))
            save_config()
            return {'ok': True, **result}
        if action == 'delete':
            result = delete_custom_preset(pcfg, str((body or {}).get('preset_id', '') or ''))
            save_config()
            return {'ok': True, **result}
        raise ValueError(f'unknown rendering_presets action: {action}')

    def _api_apply_rendering_preset_ui(self, body: dict):
        from utils.text_rendering import manga_presets
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            raise ValueError('open a project and select a page first')
        preset_id = str((body or {}).get('preset', '') or '').strip()
        presets = manga_presets(pcfg)
        if preset_id not in presets:
            raise ValueError(f'unknown preset: {preset_id}')
        indices = (body or {}).get('indices')
        page = self.imgtrans_proj.current_img
        blks = self.imgtrans_proj.pages.get(page, [])
        target = list(range(len(blks))) if indices is None else [int(i) for i in indices]
        preset = presets[preset_id]
        applied = 0
        for idx in target:
            if idx < 0 or idx >= len(blks):
                continue
            fmt = blks[idx].fontformat
            for key, value in preset.items():
                if key != 'label' and hasattr(fmt, key):
                    setattr(fmt, key, value)
            fmt.manga_preset = preset_id
            applied += 1
        self.canvas.updateTextBlkList()
        if self.imgtrans_proj and self.imgtrans_proj.directory:
            self.imgtrans_proj.save()
        self.pipelineInsightsPanel.add_event('API', self.tr('Applied rendering preset {0} to {1} text boxes').format(preset_id, applied))
        return {'ok': True, 'page': page, 'preset': preset_id, 'applied': applied}


    def _api_export_xliff(self, body: dict):
        return self._api_call_ui(self._api_export_xliff_ui, body)

    def _api_export_xliff_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        out_path = str((body or {}).get('path', '') or '').strip()
        if not out_path:
            out_path = osp.join(self.imgtrans_proj.directory, 'translation_export.xliff')
        xml_text = export_project_xliff(self.imgtrans_proj)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(xml_text)
        return {'ok': True, 'path': out_path, 'format': 'xliff'}

    def _api_import_xliff(self, body: dict):
        return self._api_call_ui(self._api_import_xliff_ui, body)

    def _api_import_xliff_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        path = str((body or {}).get('path', '') or '').strip()
        if not path or not osp.exists(path):
            raise ValueError('path is required and must exist')
        with open(path, 'r', encoding='utf-8') as f:
            xml_text = f.read()
        all_matched, match_rst = import_project_xliff(self.imgtrans_proj, xml_text)
        self.canvas.updateTextBlkList()
        self.canvas.setProjSaveState(True)
        return {'ok': all_matched, 'match': match_rst, 'path': path}


    def _api_export_translation_json(self, body: dict):
        return self._api_call_ui(self._api_export_translation_json_ui, body)

    def _api_export_translation_json_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        out_path = str((body or {}).get('path', '') or '').strip()
        if not out_path:
            out_path = osp.join(self.imgtrans_proj.directory, 'translation_export.json')
        payload = export_translation_json(self.imgtrans_proj)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return {'ok': True, 'path': out_path, 'format': 'translation_json', 'pages': len(payload.get('pages', []))}

    def _api_import_translation_json(self, body: dict):
        return self._api_call_ui(self._api_import_translation_json_ui, body)

    def _api_import_translation_json_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        path = str((body or {}).get('path', '') or '').strip()
        if not path or not osp.exists(path):
            raise ValueError('path is required and must exist')
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        all_matched, match_rst = import_translation_json(self.imgtrans_proj, payload if isinstance(payload, dict) else {})
        self.canvas.updateTextBlkList()
        self.canvas.setProjSaveState(True)
        return {'ok': all_matched, 'path': path, 'match': match_rst}


    def _api_export_translation_csv(self, body: dict):
        return self._api_call_ui(self._api_export_translation_csv_ui, body)

    def _api_export_translation_csv_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        out_path = str((body or {}).get('path', '') or '').strip()
        if not out_path:
            out_path = osp.join(self.imgtrans_proj.directory, 'translation_export.csv')
        csv_text = export_translation_csv_text(self.imgtrans_proj)
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_text)
        return {'ok': True, 'path': out_path, 'format': 'translation_csv'}

    def _api_import_translation_csv(self, body: dict):
        return self._api_call_ui(self._api_import_translation_csv_ui, body)

    def _api_import_translation_csv_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        path = str((body or {}).get('path', '') or '').strip()
        if not path or not osp.exists(path):
            raise ValueError('path is required and must exist')
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        all_matched, match_rst = import_translation_csv_text(self.imgtrans_proj, text)
        self.canvas.updateTextBlkList()
        self.canvas.setProjSaveState(True)
        return {'ok': all_matched, 'path': path, 'match': match_rst}


    def _api_tm_query(self, body: dict):
        return self._api_call_ui(self._api_tm_query_ui, body)

    def _api_tm_query_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        source = str((body or {}).get('source', '') or '')
        min_score = float((body or {}).get('min_score', 0.65) or 0.65)
        limit = int((body or {}).get('limit', 5) or 5)
        store = list(getattr(self.imgtrans_proj, 'translation_memory', []) or [])
        hits = query_tm(store, source, min_score=min_score, limit=limit)
        return {'ok': True, 'hits': hits, 'count': len(hits)}

    def _api_tm_build_from_project(self, body: dict):
        return self._api_call_ui(self._api_tm_build_from_project_ui, body)

    def _api_tm_build_from_project_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        self.imgtrans_proj.translation_memory = build_tm_from_project(self.imgtrans_proj)
        self.canvas.setProjSaveState(True)
        return {'ok': True, 'entries': len(self.imgtrans_proj.translation_memory)}

    def _api_tm_export(self, body: dict):
        return self._api_call_ui(self._api_tm_export_ui, body)

    def _api_tm_export_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        out_path = str((body or {}).get('path', '') or '').strip()
        if not out_path:
            out_path = osp.join(self.imgtrans_proj.directory, 'translation_memory.json')
        payload = export_tm_payload(list(getattr(self.imgtrans_proj, 'translation_memory', []) or []))
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return {'ok': True, 'path': out_path, 'entries': len(payload.get('entries', []))}

    def _api_tm_import(self, body: dict):
        return self._api_call_ui(self._api_tm_import_ui, body)

    def _api_tm_import_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        path = str((body or {}).get('path', '') or '').strip()
        if not path or not osp.exists(path):
            raise ValueError('path is required and must exist')
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        imported = import_tm_payload(payload if isinstance(payload, dict) else {})
        merge = bool((body or {}).get('merge', True))
        if merge:
            store = list(getattr(self.imgtrans_proj, 'translation_memory', []) or [])
            for row in imported:
                add_tm_entry(store, row.get('source', ''), row.get('target', ''), page=row.get('page', ''), block_id=row.get('block_id', ''))
            self.imgtrans_proj.translation_memory = store
        else:
            self.imgtrans_proj.translation_memory = imported
        self.canvas.setProjSaveState(True)
        return {'ok': True, 'entries': len(self.imgtrans_proj.translation_memory)}


    def _api_concordance_query(self, body: dict):
        return self._api_call_ui(self._api_concordance_query_ui, body)

    def _api_concordance_query_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        query = str((body or {}).get('query', '') or '').strip()
        if not query:
            raise ValueError('query is required')
        limit = int((body or {}).get('limit', 50) or 50)
        in_target = bool((body or {}).get('in_target', True))
        rows = build_concordance_from_project(self.imgtrans_proj)
        hits = query_concordance(rows, query, in_target=in_target, limit=limit)
        return {'ok': True, 'query': query, 'hits': hits, 'count': len(hits)}

    def _api_glossary_export(self, body: dict):
        return self._api_call_ui(self._api_glossary_export_ui, body)

    def _api_glossary_export_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        fmt = str((body or {}).get('format', 'json') or 'json').strip().lower()
        out_path = str((body or {}).get('path', '') or '').strip()
        if not out_path:
            base = self.imgtrans_proj.directory
            out_path = osp.join(base, 'translation_glossary.csv' if fmt == 'csv' else 'translation_glossary.json')
        entries = list(getattr(self.imgtrans_proj, 'translation_glossary', []) or [])
        if fmt == 'csv':
            count = export_glossary_csv(entries, out_path)
        else:
            import json
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump({'entries': entries}, f, ensure_ascii=False, indent=2)
            count = len(entries)
        return {'ok': True, 'path': out_path, 'format': fmt, 'entries': count}

    def _api_glossary_import_preview(self, body: dict):
        return self._api_call_ui(self._api_glossary_import_preview_ui, body)

    def _api_glossary_import_preview_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        import_mode = str((body or {}).get('mode', 'merge') or 'merge').strip().lower()
        in_path = str((body or {}).get('path', '') or '').strip()
        if not in_path:
            raise ValueError('path is required')
        if in_path.lower().endswith('.csv'):
            entries = import_glossary_csv(in_path)
        else:
            import json
            with open(in_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            entries = list((payload or {}).get('entries', []) or [])
        store = list(getattr(self.imgtrans_proj, 'translation_glossary', []) or [])
        preview = preview_glossary_merge(store, entries, mode=import_mode)
        return {'ok': True, **preview}

    def _api_glossary_import(self, body: dict):
        return self._api_call_ui(self._api_glossary_import_ui, body)

    def _api_glossary_import_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        import_mode = str((body or {}).get('mode', 'merge') or 'merge').strip().lower()
        in_path = str((body or {}).get('path', '') or '').strip()
        if not in_path:
            raise ValueError('path is required')
        entries = []
        if in_path.lower().endswith('.csv'):
            entries = import_glossary_csv(in_path)
        else:
            import json
            with open(in_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            entries = list((payload or {}).get('entries', []) or [])
        store = list(getattr(self.imgtrans_proj, 'translation_glossary', []) or [])
        if import_mode == 'replace':
            store = []
        seen = {str((r or {}).get('source', '') or '').strip() for r in store}
        added = 0
        for row in entries:
            src = str((row or {}).get('source', '') or '').strip()
            tgt = str((row or {}).get('target', '') or '').strip()
            if not src or src in seen:
                continue
            store.append({'source': src, 'target': tgt})
            seen.add(src)
            added += 1
        self.imgtrans_proj.translation_glossary = store
        self.canvas.setProjSaveState(True)
        return {'ok': True, 'mode': import_mode, 'added': added, 'entries': len(store)}

    def _api_translation_prompt_profiles(self, body: dict):
        return {'ok': True, 'profiles': sorted(PROMPT_PROFILES.keys()), 'default': getattr(pcfg.module, 'translation_prompt_profile_default', 'dialogue')}

    def _api_sfx_dictionary_query(self, body: dict):
        return self._api_call_ui(self._api_sfx_dictionary_query_ui, body)

    def _api_sfx_dictionary_query_ui(self, body: dict):
        query = str((body or {}).get('query', '') or '')
        limit = int((body or {}).get('limit', 50) or 50)
        include_defaults = bool((body or {}).get('include_defaults', True))
        store = list(getattr(self.imgtrans_proj, 'sfx_dictionary', []) or [])
        rows = (default_sfx_dictionary() + store) if include_defaults else store
        hits = query_sfx_dictionary(rows, query, limit=limit)
        return {'ok': True, 'count': len(hits), 'results': hits}

    def _api_sfx_dictionary_import(self, body: dict):
        return self._api_call_ui(self._api_sfx_dictionary_import_ui, body)

    def _api_sfx_dictionary_import_ui(self, body: dict):
        in_path = str((body or {}).get('path', '') or '').strip()
        if not in_path:
            raise ValueError('missing_path')
        incoming = import_sfx_dictionary(in_path)
        store = list(getattr(self.imgtrans_proj, 'sfx_dictionary', []) or [])
        merged = merge_sfx_entries(store, incoming)
        self.imgtrans_proj.sfx_dictionary = merged['entries']
        self.canvas.setProjSaveState(True)
        return {'ok': True, 'added': merged['added'], 'updated': merged['updated'], 'count': merged['count']}

    def _api_sfx_dictionary_export(self, body: dict):
        return self._api_call_ui(self._api_sfx_dictionary_export_ui, body)

    def _api_sfx_dictionary_export_ui(self, body: dict):
        fmt = str((body or {}).get('format', 'json') or 'json').strip().lower()
        out_path = str((body or {}).get('path', '') or '').strip()
        if not out_path:
            base = self.imgtrans_proj.directory if self.imgtrans_proj is not None else os.getcwd()
            out_path = osp.join(base, 'sfx_dictionary.csv' if fmt == 'csv' else 'sfx_dictionary.json')
        entries = list(getattr(self.imgtrans_proj, 'sfx_dictionary', []) or [])
        count = export_sfx_dictionary(entries, out_path, fmt=fmt)
        return {'ok': True, 'path': out_path, 'count': int(count), 'format': fmt}

    def _api_glossary_extract_candidates(self, body: dict):
        return self._api_call_ui(self._api_glossary_extract_candidates_ui, body)

    def _api_glossary_extract_candidates_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        min_freq = int((body or {}).get('min_freq', 2) or 2)
        texts = []
        for _page, blks in (self.imgtrans_proj.pages or {}).items():
            for blk in blks or []:
                texts.append((getattr(blk, 'get_text', lambda: '')() or '').strip())
        rows = extract_glossary_candidates(texts, min_freq=min_freq)
        return {'ok': True, 'count': len(rows), 'candidates': rows}

    def _api_glossary_extract_apply(self, body: dict):
        return self._api_call_ui(self._api_glossary_extract_apply_ui, body)

    def _api_glossary_extract_apply_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        candidates = list((body or {}).get('candidates', []) or [])
        approved = []
        for row in candidates:
            if bool((row or {}).get('approved', True)):
                src = str((row or {}).get('source', '') or '').strip()
                if src:
                    approved.append({'source': src, 'target': str((row or {}).get('target', '') or '').strip()})
        store = list(getattr(self.imgtrans_proj, 'translation_glossary', []) or [])
        seen = {str((r or {}).get('source', '') or '').strip() for r in store}
        for row in approved:
            if row['source'] not in seen:
                store.append(row)
                seen.add(row['source'])
        self.imgtrans_proj.translation_glossary = store
        self.canvas.setProjSaveState(True)
        return {'ok': True, 'applied': len(approved), 'result_count': len(store)}

    def _api_translation_qa_report(self, body: dict):
        return self._api_call_ui(self._api_translation_qa_report_ui, body)

    def _api_translation_qa_report_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        page = str((body or {}).get('page', '') or self.imgtrans_proj.current_img or '')
        if not page:
            raise ValueError('page is required')
        blocks = list((self.imgtrans_proj.pages or {}).get(page, []) or [])
        profile = str((body or {}).get('profile', '') or getattr(pcfg.module, 'translation_prompt_profile_default', 'dialogue') or 'dialogue')
        retry_threshold = int((body or {}).get('retry_issue_threshold', getattr(pcfg.module, 'translation_qa_retry_issue_threshold', 2)) or 2)
        repetition_threshold = float((body or {}).get('repetition_threshold', 0.45) or 0.45)
        untranslated_ratio_threshold = float((body or {}).get('untranslated_ratio_threshold', 0.85) or 0.85)
        glossary = list(getattr(self.imgtrans_proj, 'translation_glossary', []) or [])
        report = build_translation_qa_report(
            blocks,
            glossary,
            profile=profile,
            retry_issue_threshold=retry_threshold,
            repetition_threshold=repetition_threshold,
            untranslated_ratio_threshold=untranslated_ratio_threshold,
        )
        report['page'] = page
        return {'ok': True, 'report': report}


    def _api_batch_find_replace_preview(self, body: dict):
        return self._api_call_ui(self._api_batch_find_replace_preview_ui, body)

    def _api_batch_find_replace_preview_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        pattern = str((body or {}).get('pattern', '') or '')
        replacement = str((body or {}).get('replacement', '') or '')
        use_regex = bool((body or {}).get('use_regex', True))
        case_sensitive = bool((body or {}).get('case_sensitive', False))
        pages = (body or {}).get('pages') or None
        preview = preview_batch_find_replace(
            self.imgtrans_proj, pattern, replacement,
            use_regex=use_regex, case_sensitive=case_sensitive, target='translation', pages=pages
        )
        preview['pattern'] = pattern
        preview['replacement'] = replacement
        return {'ok': True, **preview}

    def _api_batch_find_replace_apply(self, body: dict):
        return self._api_call_ui(self._api_batch_find_replace_apply_ui, body)

    def _api_batch_find_replace_apply_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        preview = (body or {}).get('preview')
        if not isinstance(preview, dict):
            preview = self._api_batch_find_replace_preview_ui(body)
        changed, applied = apply_batch_find_replace(self.imgtrans_proj, preview)
        if changed > 0:
            self.canvas.updateTextBlkList()
            self.canvas.setProjSaveState(True)
        return {'ok': True, 'changed': changed, 'applied': applied, 'count': len(applied)}

    def _api_export_structured_ocr(self, body: dict):
        return self._api_call_ui(self._api_export_structured_ocr_ui, body)

    def _api_export_structured_ocr_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        pages = (body or {}).get('pages')
        payload = build_structured_ocr_export(self.imgtrans_proj, pages=pages, reading_order=str((body or {}).get('reading_order', '') or getattr(pcfg, 'render_default_reading_order', 'auto') or 'auto'))
        path = str((body or {}).get('path', '') or '').strip()
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return {'ok': True, 'path': path, 'page_count': len(payload.get('pages', [])), 'payload': payload if not path else None}

    def _update_run_button_tooltip(self):
        base = self.tr('Run pipeline (same as Pipeline → Run).')
        if getattr(pcfg, 'manual_mode', False):
            base += ' ' + self.tr('Manual mode: runs current page only.')
        self.leftBar.runBtn.setToolTip(base)
        self.titleBar.runToolBtn.setToolTip(base)
        self.bottomBar.runBtn.setToolTip(base)


    def on_rerun_stage_requested(self, stage: str):
        self.pipelineInsightsPanel.add_warning('RERUN', self.tr(f'Rerun requested from stage: {stage}'))
        # Current implementation reruns the full pipeline while preserving stage intent in warnings.
        self.run_imgtrans()

    def on_pipeline_stage_event(self, stage_name: str, progress: int, page_name: str):
        _ = (stage_name, page_name)
        self.pipelineInsightsPanel.set_pipeline_progress(progress)

    def _refresh_pipeline_provider_health(self):
        providers = (
            ('detector', 'Detector', self.module_manager.textdetector),
            ('ocr', 'OCR', self.module_manager.ocr),
            ('translator', 'Translator', self.module_manager.translator),
            ('inpainter', 'Inpainter', self.module_manager.inpainter),
        )
        for key, label, impl in providers:
            status = 'ready' if impl is not None else 'missing'
            self.pipelineInsightsPanel.set_provider_status(key, self.tr(label), status)

    def on_apply_regex_profile_requested(self):
        profiles = default_profiles()
        profiles.update(getattr(pcfg, 'user_replace_profiles', {}) or {})
        if not profiles:
            self.pipelineInsightsPanel.add_warning('PROFILE', self.tr('No regex profiles configured.'))
            return
        names = sorted(profiles.keys())
        name, ok = QInputDialog.getItem(self, self.tr("Apply Regex Profile"), self.tr("Profile"), names, 0, False)
        if not ok or not name:
            return
        page = self.imgtrans_proj.current_img
        if not page or page not in self.imgtrans_proj.pages:
            self.pipelineInsightsPanel.add_warning('PROFILE', self.tr('No current page loaded.'))
            return
        blk_list = self.imgtrans_proj.pages.get(page, [])
        changed = 0
        rules = profiles.get(name, [])
        for blk in blk_list:
            cur = getattr(blk, 'translation', '') or blk.get_text()
            nxt = apply_profile(cur, rules)
            if nxt != cur:
                blk.translation = nxt
                changed += 1
        self.pipelineInsightsPanel.add_warning('PROFILE', self.tr(f'Applied profile {name} on {changed} block(s).'))
        self.canvas.updateTextBlkList()

    def on_open_mask_diagnostics_requested(self):
        page = self.imgtrans_proj.current_img
        if not page:
            self.pipelineInsightsPanel.add_warning('MASK_DIAG', self.tr('No current page loaded.'))
            return
        mask = self.imgtrans_proj.load_mask_by_imgname(page)
        if mask is None:
            self.pipelineInsightsPanel.add_warning('MASK_DIAG', self.tr('No mask available for current page.'))
            return
        self.maskDiagnosticsPanel.set_mask(mask)
        self.rightComicTransStackPanel.setCurrentWidget(self.maskDiagnosticsPanel)

    def on_apply_project_ops_requested(self):
        page = self.imgtrans_proj.current_img
        if not page:
            self.pipelineInsightsPanel.add_warning('OPS', self.tr('No current page loaded.'))
            return
        blk_list = self.imgtrans_proj.pages.get(page, []) or []
        if not blk_list:
            self.pipelineInsightsPanel.add_warning('OPS', self.tr('No text blocks found on current page.'))
            return
        def _commit(items):
            for i, row in enumerate(items or []):
                if i < len(blk_list):
                    blk_list[i].translation = str((row or {}).get('translation', ''))
            self.canvas.updateTextBlkList()
            self.pipelineInsightsPanel.add_event('OPS', self.tr('Committed ProjectOps changes to current page'))

        dlg = ProjectOpsDialog(page, blk_list, _commit, self)
        dlg.exec()

    def on_open_ocr_crop_inspector_requested(self):
        page = self.imgtrans_proj.current_img
        if not page:
            self.pipelineInsightsPanel.add_warning('OCR_INSPECT', self.tr('No current page loaded.'))
            return
        blks = self.imgtrans_proj.pages.get(page, []) or []
        img = getattr(self.imgtrans_proj, 'img_array', None)
        if img is None or not blks:
            self.pipelineInsightsPanel.add_warning('OCR_INSPECT', self.tr('Missing image or OCR blocks on current page.'))
            return
        ocr_engines = sorted(list(getattr(OCR, 'module_dict', {}).keys()))
        current_engine = str(getattr(pcfg.module, 'ocr', '') or '')
        self.ocrCropInspectorPanel.set_page(
            img,
            blks,
            ocr_engines=ocr_engines,
            current_engine=current_engine,
            on_rerun=self._on_ocr_crop_inspector_rerun,
            on_compare=self._on_ocr_crop_inspector_compare,
        )
        self.rightComicTransStackPanel.setCurrentWidget(self.ocrCropInspectorPanel)


    def _on_ocr_crop_inspector_rerun(self, block_index: int, engine_name: str):
        try:
            rst = self._api_ocr_rerun_block_ui({'index': int(block_index), 'engine': str(engine_name or '')})
            self.canvas.updateTextBlkList()
            page = str((rst or {}).get('page', '') or '')
            idx = int((rst or {}).get('index', -1) or -1)
            txt = str((rst or {}).get('text', '') or '')
            conf = (rst or {}).get('confidence', None)
            self.pipelineInsightsPanel.add_event('OCR_INSPECT', self.tr('Re-ran OCR on {0} block #{1}.').format(page, idx + 1))
            if hasattr(self.ocrCropInspectorPanel, 'text_lbl') and idx >= 0:
                self.ocrCropInspectorPanel.text_lbl.setText(f"OCR: {txt}\nConfidence: {conf if conf is not None else '-'}")
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to rerun OCR for selected block'))

    def _api_ocr_rerun_block(self, body: dict):
        return self._api_call_ui(self._api_ocr_rerun_block_ui, body)

    def _api_ocr_rerun_block_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        page = str((body or {}).get('page', '') or '').strip() or str(getattr(self.imgtrans_proj, 'current_img', '') or '')
        if not page:
            raise ValueError('page is required')
        blks = self.imgtrans_proj.pages.get(page, []) or []
        index = int((body or {}).get('index', -1) or -1)
        if index < 0 or index >= len(blks):
            raise ValueError('invalid index')
        engine = str((body or {}).get('engine', '') or '').strip()
        if engine:
            self.module_manager.setOCR(engine)
        img = self.imgtrans_proj.read_img(page)
        self.module_manager.ocr_thread.run_ocr(img, [blks[index]])
        self.canvas.setProjSaveState(True)
        return {
            'ok': True,
            'page': page,
            'index': index,
            'engine': str(getattr(pcfg.module, 'ocr', '') or ''),
            'text': str(getattr(blks[index], 'text', '') or ''),
            'translation': str(getattr(blks[index], 'translation', '') or ''),
            'confidence': getattr(blks[index], 'confidence', None),
        }


    def _on_ocr_crop_inspector_compare(self, block_index: int, engine_name: str):
        try:
            rst = self._api_ocr_compare_block_ui({'index': int(block_index), 'secondary_engine': str(engine_name or '')})
            primary = str((rst or {}).get('primary_text', '') or '')
            secondary = str((rst or {}).get('secondary_text', '') or '')
            chosen = QMessageBox.question(
                self,
                self.tr('OCR compare result'),
                self.tr('Primary ({0}):\n{1}\n\nSecondary ({2}):\n{3}\n\nApply secondary text?').format(
                    rst.get('primary_engine', ''), primary, rst.get('secondary_engine', ''), secondary,
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if chosen == QMessageBox.StandardButton.Yes:
                self._api_ocr_apply_compare_choice_ui({'page': rst.get('page'), 'index': rst.get('index'), 'text': secondary, 'engine': rst.get('secondary_engine')})
                self.canvas.updateTextBlkList()
                self.pipelineInsightsPanel.add_event('OCR_INSPECT', self.tr('Applied secondary OCR text to selected block.'))
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to compare OCR engines'))

    def _api_ocr_compare_block(self, body: dict):
        return self._api_call_ui(self._api_ocr_compare_block_ui, body)

    def _api_ocr_compare_block_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        page = str((body or {}).get('page', '') or '').strip() or str(getattr(self.imgtrans_proj, 'current_img', '') or '')
        index = int((body or {}).get('index', -1) or -1)
        secondary = str((body or {}).get('secondary_engine', '') or '').strip()
        if not page or index < 0:
            raise ValueError('page and valid index are required')
        blks = self.imgtrans_proj.pages.get(page, []) or []
        if index >= len(blks):
            raise ValueError('invalid index')
        primary_engine = str(getattr(pcfg.module, 'ocr', '') or '')
        img = self.imgtrans_proj.read_img(page)
        blk = blks[index]
        primary_text = str(getattr(blk, 'text', '') or '')
        if secondary:
            self.module_manager.setOCR(secondary)
            self.module_manager.ocr_thread.run_ocr(img, [blk])
            secondary_text = str(getattr(blk, 'text', '') or '')
            self.module_manager.setOCR(primary_engine)
            # restore primary result
            blk.text = primary_text
        else:
            secondary_text = primary_text
        return {'ok': True, 'page': page, 'index': index, 'primary_engine': primary_engine, 'secondary_engine': secondary or primary_engine, 'primary_text': primary_text, 'secondary_text': secondary_text}

    def _api_ocr_apply_compare_choice(self, body: dict):
        return self._api_call_ui(self._api_ocr_apply_compare_choice_ui, body)

    def _api_ocr_apply_compare_choice_ui(self, body: dict):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            raise ValueError('open a project first')
        page = str((body or {}).get('page', '') or '').strip() or str(getattr(self.imgtrans_proj, 'current_img', '') or '')
        index = int((body or {}).get('index', -1) or -1)
        text = str((body or {}).get('text', '') or '')
        blks = self.imgtrans_proj.pages.get(page, []) or []
        if index < 0 or index >= len(blks):
            raise ValueError('invalid index')
        blks[index].text = text
        self.canvas.setProjSaveState(True)
        return {'ok': True, 'page': page, 'index': index, 'text': text, 'engine': str((body or {}).get('engine', '') or '')}

    def on_open_reading_order_editor_requested(self):
        page = self.imgtrans_proj.current_img
        if not page:
            self.pipelineInsightsPanel.add_warning('ORDER', self.tr('No current page loaded.'))
            return
        blks = self.imgtrans_proj.pages.get(page, []) or []
        if not blks:
            self.pipelineInsightsPanel.add_warning('ORDER', self.tr('No OCR blocks to reorder.'))
            return

        def _commit(new_order):
            self.imgtrans_proj.pages[page] = list(new_order)
            self.canvas.updateTextBlkList()
            self.pipelineInsightsPanel.add_event('ORDER', self.tr('Applied reading order changes'))

        dlg = ReadingOrderEditorDialog(blks, _commit, self)
        dlg.exec()

    def on_run_layout_review_requested(self):
        self.shortcutLayoutReviewPage()
        self.pipelineInsightsPanel.add_event('LAYOUT', self.tr('Layout review agent run on current page'))

    def on_run_auto_lettering_assist_requested(self):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            self.pipelineInsightsPanel.add_warning('TYPO_QA', self.tr('Open a project before running Auto Lettering Assist.'))
            return
        changed = self.st_manager.auto_lettering_assist_textboxes(indices=None, push_undo=True)
        if changed <= 0:
            self.pipelineInsightsPanel.add_warning('TYPO_QA', self.tr('Auto Lettering Assist found no eligible text boxes.'))
        else:
            self.pipelineInsightsPanel.add_event('TYPO_QA', self.tr('Auto Lettering Assist updated {0} text box(es).').format(changed))

    def on_run_production_auto_pass_requested(self):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            self.pipelineInsightsPanel.add_warning('TYPO_QA', self.tr('Open a project before running Production Auto Pass.'))
            return
        changed_a = self.st_manager.auto_lettering_assist_textboxes(indices=None, push_undo=True)
        changed_b = self.st_manager.atomic_bubble_fit_textboxes(indices=None, push_undo=True, profile='balanced')
        self.shortcutLayoutReviewPage()
        qa_applied = 0
        if bool(getattr(pcfg.module, 'production_auto_pass_enable_qa_fixes', True)):
            try:
                from utils.rendering_qa import apply_project_rendering_fixes
                qa_result = apply_project_rendering_fixes(
                    self.imgtrans_proj,
                    pages=[self.imgtrans_proj.current_img],
                    config_obj=pcfg,
                    selected_fixes=[
                        "resize_to_recommended_box",
                        "switch_writing_mode",
                        "set_mask_safe_padding",
                        "tighten_letter_spacing",
                        "decrease_line_spacing",
                    ],
                )
                qa_applied = int((qa_result or {}).get('applied_count', 0) or 0)
            except Exception as e:
                self.pipelineInsightsPanel.add_warning('TYPO_QA', self.tr('Production Auto Pass QA auto-fixes failed: {0}').format(e))
        self.pipelineInsightsPanel.add_event(
            'TYPO_QA',
            self.tr('Production Auto Pass: auto-lettering {0}, atomic-fit {1}, layout review, QA fixes {2}.').format(changed_a, changed_b, qa_applied),
        )

    def on_open_batch_style_override(self):
        """Apply Koharu-inspired fixed font/alignment options across a page/project."""
        from utils.text_rendering import manga_presets
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            self.pipelineInsightsPanel.add_warning('BATCH_STYLE', self.tr('Open a project before applying a batch text style override.'))
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr('Batch Text Style Override'))
        dlg.setMinimumWidth(460)
        outer = QVBoxLayout(dlg)
        form = QFormLayout()
        outer.addLayout(form)

        scope = QComboBox(dlg)
        scope.addItems([self.tr('Current page'), self.tr('Whole project')])

        font_family = QLineEdit(dlg)
        font_family.setPlaceholderText(self.tr('Leave blank to keep current font family'))

        font_size = QDoubleSpinBox(dlg)
        font_size.setRange(0.0, 300.0)
        font_size.setDecimals(1)
        font_size.setSingleStep(0.5)
        font_size.setSpecialValueText(self.tr('Keep'))
        font_size.setValue(0.0)

        alignment = QComboBox(dlg)
        alignment.addItem(self.tr('Keep'), -1)
        alignment.addItem(self.tr('Left'), 0)
        alignment.addItem(self.tr('Center'), 1)
        alignment.addItem(self.tr('Right'), 2)

        auto_fit = QCheckBox(self.tr('Enable auto-fit after override'), dlg)
        auto_fit.setChecked(True)
        only_auto = QCheckBox(self.tr('Only update blocks currently using auto font size'), dlg)
        only_auto.setToolTip(self.tr('Use this for Koharu-style global Auto font cleanup without touching hand-tuned lettering.'))

        writing_mode = QComboBox(dlg)
        writing_mode.addItem(self.tr('Keep'), '')
        writing_mode.addItem(self.tr('Auto'), 'auto')
        writing_mode.addItem(self.tr('Horizontal LTR'), 'horizontal_ltr')
        writing_mode.addItem(self.tr('Vertical RL'), 'vertical_rl')
        writing_mode.addItem(self.tr('RTL'), 'rtl')

        fit_mode = QComboBox(dlg)
        fit_mode.addItem(self.tr('Keep'), '')
        fit_mode.addItem(self.tr('Shrink to fit'), 'shrink')
        fit_mode.addItem(self.tr('Expand to fill'), 'expand')
        fit_mode.addItem(self.tr('Preserve size'), 'preserve')
        fit_mode.addItem(self.tr('Balance lines'), 'balance')

        line_break = QComboBox(dlg)
        line_break.addItem(self.tr('Keep'), '')
        line_break.addItem(self.tr('Auto'), 'auto')
        line_break.addItem(self.tr('Strict CJK kinsoku'), 'cjk_strict')
        line_break.addItem(self.tr('Balanced lettering'), 'balanced')
        line_break.addItem(self.tr('Loose SFX'), 'loose')

        preset = QComboBox(dlg)
        preset.addItem(self.tr('Keep'), '')
        for preset_id, preset_def in manga_presets(pcfg).items():
            label = str(preset_def.get('label', preset_id) or preset_id)
            if preset_id.startswith('custom:'):
                label = self.tr('Custom: ') + label
            preset.addItem(self.tr(label), preset_id)

        fallback_chain = QLineEdit(dlg)
        fallback_chain.setPlaceholderText(self.tr('Leave blank to keep current per-style fallback chain'))

        padding = QDoubleSpinBox(dlg)
        padding.setRange(-1.0, 64.0)
        padding.setDecimals(1)
        padding.setSingleStep(1.0)
        padding.setSpecialValueText(self.tr('Keep'))
        padding.setValue(-1.0)

        stroke_width = QDoubleSpinBox(dlg)
        stroke_width.setRange(-1.0, 1.0)
        stroke_width.setDecimals(3)
        stroke_width.setSingleStep(0.01)
        stroke_width.setSpecialValueText(self.tr('Keep'))
        stroke_width.setValue(-1.0)

        shadow_radius = QDoubleSpinBox(dlg)
        shadow_radius.setRange(-1.0, 1.0)
        shadow_radius.setDecimals(3)
        shadow_radius.setSingleStep(0.01)
        shadow_radius.setSpecialValueText(self.tr('Keep'))
        shadow_radius.setValue(-1.0)

        fit_min = QDoubleSpinBox(dlg)
        fit_min.setRange(-1.0, 200.0)
        fit_min.setDecimals(1)
        fit_min.setSingleStep(1.0)
        fit_min.setSpecialValueText(self.tr('Keep'))
        fit_min.setValue(-1.0)

        fit_max = QDoubleSpinBox(dlg)
        fit_max.setRange(-1.0, 300.0)
        fit_max.setDecimals(1)
        fit_max.setSingleStep(1.0)
        fit_max.setSpecialValueText(self.tr('Keep'))
        fit_max.setValue(-1.0)

        form.addRow(self.tr('Scope'), scope)
        form.addRow(self.tr('Manga preset'), preset)
        form.addRow(self.tr('Font family'), font_family)
        form.addRow(self.tr('Fixed font size (pt)'), font_size)
        form.addRow(self.tr('Alignment'), alignment)
        form.addRow(self.tr('Writing mode'), writing_mode)
        form.addRow(self.tr('Fit mode'), fit_mode)
        form.addRow(self.tr('Line-break strategy'), line_break)
        form.addRow(self.tr('Fallback font chain'), fallback_chain)
        form.addRow(self.tr('Text padding (px)'), padding)
        form.addRow(self.tr('Stroke width (relative)'), stroke_width)
        form.addRow(self.tr('Shadow radius (relative)'), shadow_radius)
        form.addRow(self.tr('Fit min size (px)'), fit_min)
        form.addRow(self.tr('Fit max size (px)'), fit_max)
        form.addRow('', auto_fit)
        form.addRow('', only_auto)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        outer.addWidget(buttons)
        exec_fn = getattr(dlg, 'exec', None) or dlg.exec_
        if int(exec_fn()) != int(QDialog.Accepted):
            return

        page_names = [self.imgtrans_proj.current_img] if scope.currentIndex() == 0 else list(self.imgtrans_proj.pages.keys())
        updates = {}
        family = font_family.text().strip()
        if family:
            updates['font_family'] = family
        size_pt = float(font_size.value())
        if size_pt > 0:
            updates['font_size'] = pt2px(size_pt)
            updates['auto_fit_font_size'] = False
        align_value = int(alignment.currentData())
        if align_value >= 0:
            updates['alignment'] = align_value
        for key, widget in [('writing_mode', writing_mode), ('fit_mode', fit_mode), ('line_break_strategy', line_break), ('preset', preset)]:
            value = str(widget.currentData() or '')
            if value:
                updates[key] = value
        fallback_value = fallback_chain.text().strip()
        if fallback_value:
            updates['fallback_font_chain'] = fallback_value
        for key, widget in [('text_padding', padding), ('stroke_width', stroke_width), ('shadow_radius', shadow_radius), ('fit_font_size_min', fit_min), ('fit_font_size_max', fit_max)]:
            value = float(widget.value())
            if value >= 0:
                updates[key] = value
        if auto_fit.isChecked() and 'font_size' not in updates:
            updates['auto_fit_font_size'] = True
        from utils.text_style_batch import apply_text_style_batch
        result = apply_text_style_batch(self.imgtrans_proj, updates, pages=page_names, only_auto_sized=only_auto.isChecked(), config_obj=pcfg)
        changed = int(result.get('changed', 0))
        if self.imgtrans_proj.current_img in page_names:
            self.st_manager.updateSceneTextitems()
        self.imgtrans_proj.save()
        self.canvas.setProjSaveState(False)
        self.pipelineInsightsPanel.add_event('BATCH_STYLE', self.tr('Updated {0} text block style(s).').format(changed))
        self.statusBar().showMessage(self.tr('Batch style override updated {0} text block(s).').format(changed), 5000)



    def on_open_typography_qa_report(self):
        """Export/apply Koharu-inspired renderer QA across the current page or project."""
        from ui.typography_qa_dialog import TypographyQAReportDialog
        from utils.rendering_qa import apply_project_rendering_fixes
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            self.pipelineInsightsPanel.add_warning('TYPO_QA', self.tr('Open a project before running typography QA.'))
            return
        dlg = TypographyQAReportDialog(self.imgtrans_proj, self.imgtrans_proj.current_img, pcfg, self)
        exec_fn = getattr(dlg, 'exec', None) or dlg.exec_
        if int(exec_fn()) != int(QDialog.Accepted):
            return
        report = dlg.report()
        out_path = dlg.write_report()
        applied = None
        pages = dlg.selected_pages()
        if dlg.should_apply_fixes():
            selected_fixes = dlg.selected_fixes()
            if not selected_fixes:
                self.pipelineInsightsPanel.add_warning('TYPO_QA', self.tr('No typography QA rows were checked; no fixes applied.'))
            applied = apply_project_rendering_fixes(self.imgtrans_proj, pages=pages, config_obj=pcfg, selected_fixes=selected_fixes) if selected_fixes else {'applied_count': 0}
            if applied.get('applied_count'):
                if self.imgtrans_proj.current_img in pages:
                    self.st_manager.updateSceneTextitems()
                if self.imgtrans_proj.directory:
                    self.imgtrans_proj.save()
                self.canvas.setProjSaveState(False)
        summary = report.get('summary', {})
        msg = self.tr('Typography QA: {0} issue text boxes across {1} pages.').format(summary.get('issues', 0), summary.get('pages', 0))
        if applied is not None:
            msg += ' ' + self.tr('Applied fixes to {0} text boxes.').format(applied.get('applied_count', 0))
        if out_path:
            msg += ' ' + self.tr('Saved: {0}').format(out_path)
        self.pipelineInsightsPanel.add_event('TYPO_QA', msg)
        self.statusBar().showMessage(msg, 7000)

    def _has_open_project(self) -> bool:
        return (
            getattr(self, 'imgtrans_proj', None) is not None
            and not getattr(self.imgtrans_proj, 'is_empty', True)
            and bool(getattr(self.imgtrans_proj, 'directory', None))
        )

    def _set_leftbar_mode(self, mode: str = None):
        """Set left-bar view state without triggering view-switch signals.

        mode: 'imgtrans', 'config', or None for welcome/no-project.
        """
        if not hasattr(self, 'leftBar'):
            return
        pairs = (
            (getattr(self.leftBar, 'imgTransChecker', None), mode == 'imgtrans'),
            (getattr(self.leftBar, 'configChecker', None), mode == 'config'),
        )
        for checker, checked in pairs:
            if checker is None:
                continue
            try:
                checker.blockSignals(True)
                checker.setChecked(bool(checked))
            finally:
                checker.blockSignals(False)

    def setupImgTransUI(self):
        # The imgtrans checker can be toggled during startup by setupConfig() or
        # LeftBar.stateCheckerChanged(). Never show an empty canvas when no
        # project is loaded; route back to welcome instead.
        if not self._has_open_project():
            self._show_welcome_screen()
            return
        self._set_leftbar_mode('imgtrans')
        self.centralStackWidget.setCurrentIndex(1)
        self.bottomBar.setPipelineVisible(True)
        if self.leftBar.needleftStackWidget():
            self.leftStackWidget.show()
            if self.leftBar.showPageListLabel.isChecked():
                self.leftStackWidget.setCurrentWidget(self.pageList)
                self.pageList.setHidden(False)
            else:
                self.leftStackWidget.setCurrentWidget(self.global_search_widget)
        else:
            self.leftStackWidget.hide()

    def setupConfigUI(self):
        self._set_leftbar_mode('config')
        self.centralStackWidget.setCurrentIndex(2)
        self.bottomBar.setPipelineVisible(False)

    def set_display_lang(self, lang: str):
        self.retranslateUI()

    def OpenProj(self, proj_path: str):
        if osp.isdir(proj_path):
            self.openDir(proj_path)
        else:
            self.openJsonProj(proj_path)
        
        if pcfg.let_textstyle_indep_flag and not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS):
            self.load_textstyle_from_proj_dir(from_proj=True)

    def _welcome_open_folder(self):
        self.leftBar.onOpenFolder()

    def _welcome_recent_clicked(self, path: str):
        if path and osp.exists(path):
            self.OpenProj(path)

    def _show_welcome_screen(self):
        """Switch to welcome screen and refresh recent list."""
        self._showing_welcome_screen = True
        self._set_leftbar_mode(None)
        if hasattr(self, 'welcomeWidget'):
            self.welcomeWidget.set_recent_projects(getattr(self.leftBar, 'recent_proj_list', []) or [])
        if hasattr(self, 'leftStackWidget'):
            self.leftStackWidget.hide()
        if hasattr(self, 'rightComicTransStackPanel'):
            self.rightComicTransStackPanel.setHidden(True)
        if hasattr(self, 'bottomBar'):
            self.bottomBar.setPipelineVisible(False)
        self.centralStackWidget.setCurrentIndex(0)

        def _reassert_welcome_if_empty():
            try:
                if not self._has_open_project():
                    self._set_leftbar_mode(None)
                    if hasattr(self, 'leftStackWidget'):
                        self.leftStackWidget.hide()
                    if hasattr(self, 'rightComicTransStackPanel'):
                        self.rightComicTransStackPanel.setHidden(True)
                    if hasattr(self, 'bottomBar'):
                        self.bottomBar.setPipelineVisible(False)
                    self.centralStackWidget.setCurrentIndex(0)
            except Exception:
                return

        # Multiple delayed guards are intentional: startup config, title/left-bar
        # checkers, and first show events can all emit shortly after __init__.
        QTimer.singleShot(0, _reassert_welcome_if_empty)
        QTimer.singleShot(100, _reassert_welcome_if_empty)
        QTimer.singleShot(500, _reassert_welcome_if_empty)
        QTimer.singleShot(1500, _reassert_welcome_if_empty)

    def _show_main_content(self):
        """Switch from welcome to main translation view."""
        self._showing_welcome_screen = False
        self._set_leftbar_mode('imgtrans')
        self.centralStackWidget.setCurrentIndex(1)
        self.bottomBar.setPipelineVisible(True)
        # Default to text editor panel (index 1) on every project open / app start
        if not self.rightComicTransStackPanel.isHidden():
            self.rightComicTransStackPanel.setCurrentIndex(1)

    def load_textstyle_from_proj_dir(self, from_proj=False):
        if from_proj:
            text_style_path = osp.join(self.imgtrans_proj.directory, 'textstyles.json')
        else:
            text_style_path = 'config/textstyles/default.json'
        if osp.exists(text_style_path):
            load_textstyle_from(text_style_path)
            self.textPanel.formatpanel.textstyle_panel.setStyles(text_styles)
            self.textPanel.formatpanel.textstyle_panel.set_active_style_by_name(
                getattr(pcfg, 'default_text_style_name', '') or ''
            )
        else:
            pcfg.text_styles_path = text_style_path
            save_text_styles()

    def on_show_only_custom_font(self, only_custom: bool):
        if only_custom:
            font_list = shared.CUSTOM_FONTS
        else:
            font_list = shared.FONT_FAMILIES
        self.textPanel.formatpanel.familybox.update_font_list(font_list)

    def openDir(self, directory: str):
        try:
            self.opening_dir = True
            # 在加载项目前检查并生成TIF文件的预览图
            self.generate_tif_thumbnails(directory)
            # 重新加载项目，此时应该只加载预览图
            self.imgtrans_proj.load(directory)
            self.st_manager.clearSceneTextitems()
            self.titleBar.setTitleContent(osp.basename(directory))
            self.updatePageList()
            self._ensure_first_page_loaded_and_autolayout()
            self.opening_dir = False
            self._show_main_content()
        except Exception as e:
            self.opening_dir = False
            create_error_dialog(e, self.tr('Failed to load project ') + directory)
            return

    def generate_tif_thumbnails(self, directory: str):
        """
        为目录中的TIF文件生成预览图，并确保只加载预览图
        """
        try:
            from utils.io_utils import create_thumbnail, find_tif_files
            # 查找目录中的所有TIF文件
            tif_files = find_tif_files(directory)
            
            # 为每个TIF文件生成预览图
            for tif_file in tif_files:
                tif_path = osp.join(directory, tif_file)
                # 检查是否已经存在对应的预览图
                base_path = Path(tif_path)
                thumb_path = base_path.parent / f"{base_path.stem}_thumb.jpg"
                
                # 如果预览图不存在，则生成预览图
                if not osp.exists(thumb_path):
                    create_thumbnail(tif_path, max_width=1000)
                    
        except Exception as e:
            LOGGER.error(f"Failed to generate TIF thumbnails: {e}")
        
    def dropOpenDir(self, directory: str):
        if isinstance(directory, str) and osp.exists(directory):
            self.leftBar.updateRecentProjList(directory)
            self.OpenProj(directory)

    def dropOpenFiles(self, file_paths: list):
        """Open a project from a list of image file paths (e.g. drag-drop or Select Files). Uses first file's directory as project dir and only those files as pages."""
        if not file_paths:
            return
        try:
            if self.imgtrans_proj.directory is not None and self.canvas.projstate_unsaved:
                self.saveCurrentPage()
            first_path = osp.normpath(file_paths[0])
            target_dir = osp.dirname(first_path)
            image_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif', '.gif'}
            copied_files = []
            for p in file_paths:
                p = osp.normpath(p)
                if not osp.isfile(p) or osp.splitext(p)[1].lower() not in image_ext:
                    continue
                if osp.dirname(p) != target_dir:
                    continue
                copied_files.append(osp.basename(p))
            if not copied_files:
                return
            self.leftBar.updateRecentProjList(target_dir)
            self.imgtrans_proj.directory = target_dir
            self.imgtrans_proj.proj_path = osp.join(target_dir, self.imgtrans_proj.proj_name() + '.json')
            if not osp.exists(self.imgtrans_proj.inpainted_dir()):
                os.makedirs(self.imgtrans_proj.inpainted_dir())
            if not osp.exists(self.imgtrans_proj.mask_dir()):
                os.makedirs(self.imgtrans_proj.mask_dir())
            self.imgtrans_proj.pages = {}
            self.imgtrans_proj._pagename2idx = {}
            self.imgtrans_proj._idx2pagename = {}
            for ii, name in enumerate(copied_files):
                self.imgtrans_proj.pages[name] = []
                self.imgtrans_proj._pagename2idx[name] = ii
                self.imgtrans_proj._idx2pagename[ii] = name
            if copied_files:
                self.imgtrans_proj.set_current_img(copied_files[0])
            self.imgtrans_proj.save()
            self.st_manager.clearSceneTextitems()
            self.titleBar.setTitleContent(page_name=osp.basename(target_dir))
            self.updatePageList()
            self._ensure_first_page_loaded_and_autolayout()
            self._show_main_content()
            self.canvas.clear_undostack(update_saved_step=True)
            self.canvas.drop_files = []
            self.canvas.drop_folder = None
        except Exception as e:
            LOGGER.error("Error importing images: %s", str(e))
            create_error_dialog(e, self.tr('Failed to import images'))

    def openJsonProj(self, json_path: str):
        try:
            self.opening_dir = True
            self.imgtrans_proj.load_from_json(json_path)
            self.st_manager.clearSceneTextitems()
            self.leftBar.updateRecentProjList(self.imgtrans_proj.proj_path)
            self.updatePageList()
            self.titleBar.setTitleContent(osp.basename(self.imgtrans_proj.proj_path))
            self._ensure_first_page_loaded_and_autolayout()
            self.opening_dir = False
            self._show_main_content()
        except Exception as e:
            self.opening_dir = False
            create_error_dialog(e, self.tr('Failed to load project from') + json_path)
        
    def updatePageList(self):
        if self.pageList.count() != 0:
            self.pageList.clear()
        if len(self.imgtrans_proj.pages) >= shared.PAGELIST_THUMBNAIL_MAXNUM:
            item_func = lambda imgname: QListWidgetItem(imgname)
        else:
            item_func = lambda imgname:\
                QListWidgetItem(QIcon(osp.join(self.imgtrans_proj.directory, imgname)), imgname)
        for imgname in self.imgtrans_proj.pages:
            lstitem =  item_func(imgname)
            lstitem.setData(Qt.ItemDataRole.UserRole, imgname)
            self.pageList.addItem(lstitem)
            if imgname == self.imgtrans_proj.current_img:
                self.pageList.setCurrentItem(lstitem)
        self._update_page_list_state_style()

    def _ensure_first_page_loaded_and_autolayout(self):
        """After opening a project, ensure the first page is displayed. Use saved layout (no re-run of auto layout)."""
        if not self.imgtrans_proj.current_img:
            return
        # Reset zoom baseline so newly opened folders/files default to fit-in-view instead of keeping previous zoom.
        self.canvas.scale_factor = 1.0
        self.canvas.updateCanvas()
        self.st_manager.updateSceneTextitems()

    def pageLabelStateChanged(self):
        setup = self.leftBar.showPageListLabel.isChecked()
        if setup:
            if self.leftStackWidget.isHidden():
                self.leftStackWidget.show()
            if self.leftBar.globalSearchChecker.isChecked():
                self.leftBar.globalSearchChecker.setChecked(False)
            self.leftStackWidget.setCurrentWidget(self.pageList)
            self.pageList.setHidden(False)
        else:
            self.leftStackWidget.hide()
        pcfg.show_page_list = setup
        save_config()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not getattr(self, '_comic_trans_splitter_initialized', False):
            self._comic_trans_splitter_initialized = True
            total = self.comicTransSplitter.width()
            if total > 400:
                right_default = 282
                # Default left pane to the same width it locks at when dragging (PAGE_LIST_PANE_DEFAULT_WIDTH).
                left = shared.PAGE_LIST_PANE_DEFAULT_WIDTH
                center = max(100, total - left - right_default)
                self.comicTransSplitter.setSizes([left, center, right_default])

        # Deferred initial model download (launch.py sets DEFER_INITIAL_MODEL_DOWNLOAD so app opens first)
        if getattr(shared, 'DEFER_INITIAL_MODEL_DOWNLOAD', False) and not self._initial_model_download_done:
            shared.DEFER_INITIAL_MODEL_DOWNLOAD = False
            self._initial_model_download_done = True
            # Run after the event loop gets a chance to paint the window.
            # Calling a modal dialog directly from showEvent can keep the main UI hidden on some setups.
            QTimer.singleShot(0, self._run_deferred_model_download)
        if not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS) and not self._has_open_project():
            QTimer.singleShot(0, self._show_welcome_screen)
        if not self._startup_health_shown:
            self._startup_health_shown = True
            QTimer.singleShot(250, self._show_startup_health_overlay)

    def closeEvent(self, event: QCloseEvent) -> None:
        reply = QMessageBox.question(
            self,
            self.tr('Confirm exit'),
            self.tr('Are you sure you want to quit?'),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No:
            event.ignore()
            return
        if not self.imgtrans_proj.is_empty:
            self.conditional_save(keep_exist_as_backup=True)
            # Always sync current page scene to blocks and save so text boxes (e.g. after auto layout) are persisted
            if self.imgtrans_proj.current_img and self.st_manager.textblk_item_list:
                self.st_manager.updateTextBlkList()
                try:
                    self.imgtrans_proj.save(keep_exist_as_backup=True)
                except Exception:
                    pass
        while True:
            if not self.imsave_thread.isRunning():
                break
            time.sleep(0.1)
        self.st_manager.hovering_transwidget = None
        self.st_manager.blockSignals(True)
        self.canvas.prepareClose()
        self.save_config()
        return super().closeEvent(event)

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.WindowStateChange:
            if self.windowState() & Qt.WindowState.WindowMaximized:
                if not shared.ON_MACOS:
                    self.titleBar.maxBtn.setChecked(True)
        elif event.type() == QEvent.Type.ActivationChange:
            self.canvas.on_activation_changed()

        super().changeEvent(event)
    
    def retranslateUI(self):
        # according to https://stackoverflow.com/questions/27635068/how-to-retranslate-dynamically-created-widgets
        # we got to do it manually ... I'd rather restart the program
        msg = QMessageBox()
        msg.setText(self.tr('Restart to apply changes? \n'))
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        ret = msg.exec_()
        if ret == QMessageBox.StandardButton.Yes:
            self.restart_signal.emit()

    def save_config(self):
        # Flush Region merge tool dialog state if it's open so it gets persisted
        if getattr(self, 'merge_dialog', None) is not None and self.merge_dialog.isVisible():
            self.merge_dialog.save_settings_to_pcfg()
        save_config()

    def on_set_default_format_requested(self):
        """Save current global font format to config so it is used as default for new projects/sessions."""
        # Persist which preset is selected so it shows as active (blue) on next launch
        panel = self.st_manager.formatpanel.textstyle_panel
        if panel.active_text_style_label is not None:
            pcfg.default_text_style_name = getattr(
                panel.active_text_style_label.fontfmt, '_style_name', ''
            ) or ''
        else:
            pcfg.default_text_style_name = ''
        save_config()

    def onHideCanvas(self):
        self.canvas.clearToolStates()

    def conditional_save(self, keep_exist_as_backup=False):
        if self.canvas.projstate_unsaved and not self.opening_dir:
            update_scene_text = save_proj = self.canvas.text_change_unsaved()
            save_rst_only = not self.canvas.draw_change_unsaved()
            if not save_rst_only:
                save_proj = True
            
            self.saveCurrentPage(update_scene_text, save_proj, restore_interface=True, save_rst_only=save_rst_only, keep_exist_as_backup=keep_exist_as_backup)

    def pageListCurrentItemChanged(self):
        item = self.pageList.currentItem()
        self.page_changing = True
        if item is not None:
            if self.save_on_page_changed:
                self.conditional_save()
            self.imgtrans_proj.set_current_img(item.data(Qt.ItemDataRole.UserRole) or item.text())
            self.canvas.clear_undostack(update_saved_step=True)
            self.canvas.updateCanvas()
            self.st_manager.updateSceneTextitems()
            self.titleBar.setTitleContent(page_name=self.imgtrans_proj.current_img)
            self.module_manager.handle_page_changed()
            self.drawingPanel.handle_page_changed()
            
        self.page_changing = False

    def setupShortcuts(self):
        self.titleBar.nextpage_trigger.connect(self.shortcutNext) 
        self.titleBar.prevpage_trigger.connect(self.shortcutBefore)
        self.titleBar.textedit_trigger.connect(self.shortcutTextedit)
        self.titleBar.drawboard_trigger.connect(self.shortcutDrawboard)
        self.titleBar.spellcheck_panel_trigger.connect(self.shortcutSpellCheckPanel)
        self.titleBar.redo_trigger.connect(self.on_redo)
        self.titleBar.undo_trigger.connect(self.on_undo)
        self.titleBar.page_search_trigger.connect(self.on_page_search)
        self.titleBar.global_search_trigger.connect(self.on_global_search)
        self.titleBar.replacePreMTkeyword_trigger.connect(self.show_pre_MT_keyword_window)
        self.titleBar.replaceMTkeyword_trigger.connect(self.show_MT_keyword_window)
        self.titleBar.replaceOCRkeyword_trigger.connect(self.show_OCR_keyword_window)
        self.titleBar.translation_context_trigger.connect(self.show_translation_context_dialog)
        self.titleBar.run_trigger.connect(self.run_imgtrans)
        self.bottomBar.run_clicked.connect(self.run_imgtrans)
        self.titleBar.run_woupdate_textstyle_trigger.connect(self.run_imgtrans_wo_textstyle_update)
        self.titleBar.translate_page_trigger.connect(self.on_transpagebtn_pressed)
        self.titleBar.enable_module.connect(self.on_enable_module)
        self.titleBar.importtstyle_trigger.connect(self.import_tstyles)
        self.titleBar.exporttstyle_trigger.connect(self.export_tstyles)
        self.titleBar.theme_light_trigger.connect(self._on_theme_light_triggered)
        self.titleBar.theme_dark_trigger.connect(self._on_theme_dark_triggered)
        self.titleBar.merge_tool_trigger.connect(self.on_open_merge_tool)
        self.titleBar.re_run_detection_only_trigger.connect(self.on_re_run_detection_only)
        self.titleBar.re_run_ocr_only_trigger.connect(self.on_re_run_ocr_only)
        self.titleBar.batch_export_trigger.connect(self.on_batch_export)
        self.titleBar.batch_export_as_trigger.connect(self.on_batch_export_as)
        self.titleBar.export_lptxt_trigger.connect(self.on_export_lptxt)
        self.titleBar.export_structured_ocr_trigger.connect(self.on_export_structured_ocr_json)
        if hasattr(self.titleBar, "export_xliff_trigger"):
            self.titleBar.export_xliff_trigger.connect(self.on_export_xliff)
        if hasattr(self.titleBar, "import_xliff_trigger"):
            self.titleBar.import_xliff_trigger.connect(self.on_import_xliff)
        if hasattr(self.titleBar, "export_translation_json_trigger"):
            self.titleBar.export_translation_json_trigger.connect(self.on_export_translation_json)
        if hasattr(self.titleBar, "import_translation_json_trigger"):
            self.titleBar.import_translation_json_trigger.connect(self.on_import_translation_json)
        if hasattr(self.titleBar, "export_translation_csv_trigger"):
            self.titleBar.export_translation_csv_trigger.connect(self.on_export_translation_csv)
        if hasattr(self.titleBar, "import_translation_csv_trigger"):
            self.titleBar.import_translation_csv_trigger.connect(self.on_import_translation_csv)
        if hasattr(self.titleBar, "export_layered_psd_handoff_trigger"):
            self.titleBar.export_layered_psd_handoff_trigger.connect(self.on_export_layered_psd_handoff)
        if hasattr(self.titleBar, "export_svg_text_handoff_trigger"):
            self.titleBar.export_svg_text_handoff_trigger.connect(self.on_export_svg_text_handoff)
        self.titleBar.validate_project_trigger.connect(self.on_validate_project)
        self.titleBar.show_batch_report_trigger.connect(self._show_batch_report_dialog)
        self.titleBar.manga_source_trigger.connect(self.on_open_manga_source)
        if hasattr(self.titleBar, "realtime_translator_trigger"):
            self.titleBar.realtime_translator_trigger.connect(self.on_open_realtime_translator)
        if hasattr(self.titleBar, "translation_assist_trigger"):
            self.titleBar.translation_assist_trigger.connect(self.on_open_translation_assist_dock)
        self.titleBar.batch_queue_trigger.connect(self.on_open_batch_queue)
        self.titleBar.manage_models_trigger.connect(self.on_open_manage_models)
        self.titleBar.install_google_font_trigger.connect(self.on_install_google_font)
        self.titleBar.retry_models_trigger.connect(self.on_retry_model_downloads)
        self.titleBar.environment_doctor_trigger.connect(self.on_environment_doctor)
        self.titleBar.translation_qa_report_trigger.connect(self.on_translation_qa_report_current_page)
        if hasattr(self.titleBar, "batch_find_replace_trigger"):
            self.titleBar.batch_find_replace_trigger.connect(self.on_batch_find_replace_current_project)
        self.titleBar.ocr_triage_trigger.connect(self.on_open_ocr_triage_current_page)
        self.titleBar.auto_extract_glossary_trigger.connect(self.on_auto_extract_glossary_current_page)
        if hasattr(self.titleBar, 'concordance_search_trigger'):
            self.titleBar.concordance_search_trigger.connect(self.on_concordance_search_current_project)
        if hasattr(self.titleBar, 'import_glossary_preview_trigger'):
            self.titleBar.import_glossary_preview_trigger.connect(self.on_import_glossary_with_preview)
        self.titleBar.save_run_profile_trigger.connect(self.on_save_run_profile_snapshot)
        self.titleBar.apply_run_profile_trigger.connect(self.on_apply_run_profile_snapshot)
        self.titleBar.layout_review_selected_trigger.connect(self.shortcutLayoutReviewSelected)
        self.titleBar.layout_review_page_trigger.connect(self.shortcutLayoutReviewPage)
        self.titleBar.layout_review_config_trigger.connect(self.shortcutLayoutReviewConfig)
        self.titleBar.lettering_workflow_trigger.connect(self.on_lettering_workflow_current_page)
        if hasattr(self.titleBar, 'auto_format_qa_trigger'):
            self.titleBar.auto_format_qa_trigger.connect(self.on_open_auto_format_qa)
        self.titleBar.next_rendering_issue_trigger.connect(self.on_next_rendering_issue)
        self.titleBar.show_model_download_diag_trigger.connect(self.on_show_model_download_diagnostics)
        self.titleBar.copy_startup_diag_trigger.connect(self.on_copy_startup_diagnostics)
        self.titleBar.runtime_resource_summary_trigger.connect(self.on_runtime_resource_summary)
        self.titleBar.export_startup_diag_trigger.connect(self.on_export_startup_report)
        self.titleBar.open_log_folder_trigger.connect(self.on_open_log_folder)
        self.titleBar.relaunch_pyqt5_trigger.connect(self.on_relaunch_pyqt5_safe_mode)
        self.titleBar.relaunch_cpu_only_trigger.connect(self.on_relaunch_cpu_only_safe_mode)
        self.titleBar.release_model_caches_trigger.connect(self.on_release_model_caches)
        self.titleBar.clear_pipeline_caches_trigger.connect(self.on_clear_pipeline_caches)
        self.titleBar.run_preset_full_trigger.connect(self.on_run_preset_full)
        self.titleBar.run_preset_detect_ocr_trigger.connect(self.on_run_preset_detect_ocr)
        self.titleBar.run_preset_translate_trigger.connect(self.on_run_preset_translate)
        self.titleBar.run_preset_inpaint_trigger.connect(self.on_run_preset_inpaint)
        self.titleBar.video_translator_trigger.connect(self.on_video_translator)
        self.titleBar.subtitle_file_translator_trigger.connect(self.on_subtitle_file_translator)
        self.titleBar.video_subtitle_editor_trigger.connect(self.on_video_subtitle_editor)
        self.titleBar.keyboard_shortcuts_trigger.connect(self.open_shortcuts_dialog)
        self.titleBar.theme_customizer_trigger.connect(self.open_theme_customizer)
        self.titleBar.context_menu_options_trigger.connect(self.shortcutContextMenuOptions)
        self.titleBar.help_doc_trigger.connect(self.on_help_documentation)
        self.titleBar.help_about_trigger.connect(self.on_help_about)
        self.titleBar.help_update_from_github_trigger.connect(self.on_update_from_github)

        sc = getattr(pcfg, "shortcuts", None) or {}
        self._shortcuts_list = []  # (action_id, default_key, QShortcut)

        def _mk_shortcut(action_id: str, default_key: str, slot, context: Qt.ShortcutContext = Qt.ShortcutContext.WindowShortcut):
            key = get_shortcut(action_id, sc) or default_key
            q = QShortcut(QKeySequence.fromString(key) if key else QKeySequence(), self)
            q.setContext(context)
            def _guarded_slot(fn=slot, shortcut=q):
                k = shortcut.key().toString()
                if is_single_key_sequence(k) and shortcut_should_ignore_text_input(QApplication.focusWidget()):
                    return
                return fn()
            q.activated.connect(_guarded_slot)
            self._shortcuts_list.append((action_id, default_key, q))
            return q

        _mk_shortcut("view.text_edit", "T", self.shortcutTextedit, Qt.ShortcutContext.ApplicationShortcut)
        _mk_shortcut("view.draw_board", "P", self.shortcutDrawboard, Qt.ShortcutContext.ApplicationShortcut)
        _mk_shortcut("go.prev_page_alt", "A", self.shortcutBefore, Qt.ShortcutContext.ApplicationShortcut)
        _mk_shortcut("go.prev_page", "PageUp", self.shortcutBefore, Qt.ShortcutContext.ApplicationShortcut)
        _mk_shortcut("go.next_page_alt", "D", self.shortcutNext, Qt.ShortcutContext.ApplicationShortcut)
        _mk_shortcut("go.next_page", "PageDown", self.shortcutNext, Qt.ShortcutContext.ApplicationShortcut)
        _mk_shortcut("canvas.textblock_mode", "W", self.shortcutTextblock)
        _mk_shortcut("canvas.zoom_in", "Ctrl++", self.canvas.gv.scale_up_signal.emit)
        _mk_shortcut("canvas.zoom_out", "Ctrl+-", self.canvas.gv.scale_down_signal.emit)
        _mk_shortcut("canvas.delete", "Ctrl+D", self.shortcutCtrlD)
        _mk_shortcut("canvas.space", "Space", self.shortcutSpace)
        sel_all_shortcut = _mk_shortcut("canvas.select_all", "Ctrl+A", self.shortcutSelectAll)
        sel_all_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        _mk_shortcut("canvas.escape", "Escape", self.shortcutEscape)
        _mk_shortcut("edit.omni_search", "Ctrl+P", self.shortcutOmniSearch)
        # Widget-level Ctrl+A when canvas view has focus: select all text blocks on canvas
        self._canvas_select_all_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.SelectAll), self.canvas.gv)
        self._canvas_select_all_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._canvas_select_all_shortcut.activated.connect(self.on_select_all_canvas)
        _mk_shortcut("format.bold", "Ctrl+B", self.shortcutBold)
        _mk_shortcut("format.italic", "Ctrl+I", self.shortcutItalic)
        _mk_shortcut("format.underline", "Ctrl+U", self.shortcutUnderline)
        _mk_shortcut("format.font_size_up", "Ctrl+Alt+Up", self.shortcutFontSizeUp)
        _mk_shortcut("format.font_size_down", "Ctrl+Alt+Down", self.shortcutFontSizeDown)
        _mk_shortcut("canvas.delete_line", "Delete", self.shortcutDelete)
        _mk_shortcut("canvas.create_textbox", "Ctrl+Shift+N", self.shortcutCreateTextbox)
        _mk_shortcut("view.context_menu_options", "Ctrl+Shift+O", self.shortcutContextMenuOptions)
        _mk_shortcut("format.apply", "", self.shortcutFormatApply)
        _mk_shortcut("format.layout", "", self.shortcutFormatLayout)
        _mk_shortcut("format.fit_to_bubble", "", self.shortcutFitToBubble)
        _mk_shortcut("format.auto_fit", "", self.shortcutFormatAutoFit)
        _mk_shortcut("format.auto_fit_binary", "", self.shortcutFormatAutoFitBinary)
        _mk_shortcut("format.re_auto_fit_selected", "", self.shortcutReAutoFitSelected)
        _mk_shortcut("format.re_auto_fit_page", "", self.shortcutReAutoFitCurrentPage)
        _mk_shortcut("format.re_auto_fit_all", "", self.shortcutReAutoFitAllPages)
        _mk_shortcut("format.balloon_shape_auto", "", self.shortcutBalloonShapeAuto)
        _mk_shortcut("format.resize_to_fit_content", "", self.shortcutResizeToFitContent)
        _mk_shortcut("format.layout_review_selected", "", self.shortcutLayoutReviewSelected)
        _mk_shortcut("format.layout_review_page", "", self.shortcutLayoutReviewPage)
        _mk_shortcut("format.layout_review_config", "", self.shortcutLayoutReviewConfig)
        _mk_shortcut("review.ocr_triage_page", "Ctrl+Shift+Y", self.on_open_ocr_triage_current_page)
        _mk_shortcut("review.translation_qa_page", "Ctrl+Shift+Q", self.on_translation_qa_report_current_page)
        _mk_shortcut("review.auto_extract_glossary_page", "Ctrl+Shift+G", self.on_auto_extract_glossary_current_page)

        drawpanel_shortcuts = [
            ("draw.hand", "H", "hand"),
            ("draw.inpaint", "J", "inpaint"),
            ("draw.pen", "B", "pen"),
            ("draw.text_eraser", "E", "textEraser"),
            ("draw.rect", "R", "rect"),
        ]
        for action_id, default_key, tool_name in drawpanel_shortcuts:
            key = get_shortcut(action_id, sc) or default_key
            shortcut = QShortcut(QKeySequence.fromString(key) if key else QKeySequence(), self)
            def _guarded_draw(name=tool_name, shortcut=shortcut):
                k = shortcut.key().toString()
                if is_single_key_sequence(k) and shortcut_should_ignore_text_input(QApplication.focusWidget()):
                    return
                return self.drawingPanel.shortcutSetCurrentToolByName(name)
            shortcut.activated.connect(_guarded_draw)
            self.drawingPanel.setShortcutTip(tool_name, key or default_key)
            self._shortcuts_list.append((action_id, default_key, shortcut))
        for action_id, default_key, slot in [
            ("draw.brush_size_up", "]", self.drawingPanel.on_incre_pensize),
            ("draw.brush_size_down", "[", self.drawingPanel.on_decre_pensize),
        ]:
            key = get_shortcut(action_id, sc) or default_key
            shortcut = QShortcut(QKeySequence.fromString(key) if key else QKeySequence(), self)
            def _guarded_brush(fn=slot, shortcut=shortcut):
                k = shortcut.key().toString()
                if is_single_key_sequence(k) and shortcut_should_ignore_text_input(QApplication.focusWidget()):
                    return
                return fn()
            shortcut.activated.connect(_guarded_brush)
            self._shortcuts_list.append((action_id, default_key, shortcut))
        self._draw_shortcut_tools = drawpanel_shortcuts

    def shortcutOmniSearch(self):
        """Open the top-bar omni search command palette."""
        try:
            tb = getattr(self, "titleBar", None)
            if tb is None:
                return

            setup = getattr(tb, "_setup_omni_search", None)
            if callable(setup):
                setup()

            show = getattr(tb, "_show_omni_search", None)
            if callable(show):
                show()
                return

            toggle = getattr(tb, "_toggle_omni_search", None)
            if callable(toggle):
                toggle()
                return

            box = getattr(tb, "omniSearch", None)
            if box is None:
                box = getattr(tb, "omniSearchEdit", None)
            if box is None:
                return
            box.show()
            if box.width() <= 8:
                box.setFixedWidth(320)
            box.setFocus(Qt.FocusReason.ShortcutFocusReason)
            try:
                box.selectAll()
            except Exception:
                pass
            comp = box.completer()
            if comp is not None:
                QTimer.singleShot(0, comp.complete)
        except Exception:
            return

    def apply_shortcuts(self, shortcuts_dict=None):
        """Apply keyboard shortcuts from config (action_id -> key string)."""
        sc = shortcuts_dict if shortcuts_dict is not None else getattr(pcfg, "shortcuts", None) or {}
        for action_id, default_key, qshortcut in self._shortcuts_list:
            key = sc.get(action_id) or default_key
            qshortcut.setKey(QKeySequence.fromString(key) if key else QKeySequence())
        for action_id, default_key, tool_name in self._draw_shortcut_tools:
            key = sc.get(action_id) or default_key
            self.drawingPanel.setShortcutTip(tool_name, key or default_key)
        self.leftBar.apply_shortcuts(sc)
        self.titleBar.apply_shortcuts(sc)

    def open_shortcuts_dialog(self):
        dlg = ShortcutsDialog(self)
        dlg.shortcuts_changed.connect(self.apply_shortcuts)
        dlg.show()

    def open_theme_customizer(self):
        from .theme_customizer_dialog import ThemeCustomizerDialog
        dlg = ThemeCustomizerDialog(self)
        dlg.theme_applied.connect(self._on_theme_customizer_applied)
        dlg.show()

    def _on_theme_customizer_applied(self, font_family: str, font_size: int):
        self.titleBar.themeLightAction.setChecked(not pcfg.darkmode)
        self.titleBar.themeDarkAction.setChecked(pcfg.darkmode)
        self.resetStyleSheet()
        # Font is applied in resetStyleSheet() -> _apply_app_font() (pcfg already saved by dialog)

    def on_help_documentation(self):
        """Open project README (#126 Help menu)."""
        root = osp.dirname(osp.dirname(osp.abspath(__file__)))
        readme = osp.join(root, 'README.md')
        if osp.isfile(readme):
            if sys.platform == 'win32':
                os.startfile(readme)
            elif sys.platform == 'darwin':
                subprocess.run(['open', readme], check=False)
            else:
                subprocess.run(['xdg-open', readme], check=False)
        else:
            create_info_dialog({'title': self.tr('Documentation'), 'text': self.tr('README.md not found.')})

    def on_help_about(self):
        """Show About dialog (#126 Help menu)."""
        from launch import VERSION
        QMessageBox.about(
            self,
            self.tr('About'),
            self.tr('BallonsTranslatorPro — community fork') + f' {VERSION}\n\n' + self.tr('Deep learning–assisted comic/manga translation.')
        )

    # --- Omni search jump helpers (TitleBar search box) ---
    def jump_to_config_item(self, idx0: int, idx1: int) -> None:
        """Open Config UI and focus the requested config sub-block."""
        try:
            # Do not toggle left-bar checkers directly here. Their custom signals can
            # bounce through setupImgTransUI/setupConfigUI and override the requested
            # view. Use the main-window routing method instead.
            self.setupConfigUI()

            cp = getattr(self, "configPanel", None)
            if cp is None:
                return
            try:
                cp.configTable.setCurrentItem(idx0, idx1)
            except Exception:
                pass
            try:
                cp.onTableItemPressed(idx0, idx1)
            except Exception:
                try:
                    cp.configTable.tableitem_pressed.emit(idx0, idx1)
                except Exception:
                    pass
        except Exception:
            return

    def jump_to_canvas_block(self, block_idx: int) -> None:
        """Ensure canvas/text UI is visible and select a text block by index."""
        try:
            if not self._has_open_project():
                self._show_welcome_screen()
                return
            self._show_main_content()

            if not getattr(self, "st_manager", None) or not getattr(self.st_manager, "textblk_item_list", None):
                return
            if block_idx < 0 or block_idx >= len(self.st_manager.textblk_item_list):
                return
            self.rightComicTransStackPanel.setHidden(False)
            self.rightComicTransStackPanel.setCurrentIndex(1)
            self.canvas.block_selection_signal = True
            self.canvas.clearSelection()
            blk_item = self.st_manager.textblk_item_list[block_idx]
            blk_item.setSelected(True)
            self.canvas.block_selection_signal = False
            try:
                self.st_manager.textEditList.set_selected_list([block_idx])
            except Exception:
                pass
        except Exception:
            return

    def jump_to_text_panel(self) -> None:
        """Switch right-side stack to the text/format panel and ensure main canvas view is visible."""
        try:
            if not self._has_open_project():
                self._show_welcome_screen()
                return
            self._show_main_content()
            try:
                if hasattr(self, "rightComicTransStackPanel"):
                    self.rightComicTransStackPanel.setHidden(False)
                    self.rightComicTransStackPanel.setCurrentIndex(1)
            except Exception:
                pass
        except Exception:
            return

    def _new_themed_message_box(self, parent=None) -> QMessageBox:
        box = QMessageBox(parent if parent is not None else self)
        box.setOption(QMessageBox.Option.DontUseNativeDialog, True)
        return box

    def _msg_information(self, title: str, text: str):
        box = self._new_themed_message_box(self)
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(text)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _msg_warning(self, title: str, text: str):
        box = self._new_themed_message_box(self)
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(text)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _msg_question(self, title: str, text: str, buttons, default_button):
        box = self._new_themed_message_box(self)
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(text)
        box.setStandardButtons(buttons)
        box.setDefaultButton(default_button)
        box.exec()
        return box.standardButton(box.clickedButton())

    def on_update_from_github(self):
        """Pull latest changes from GitHub (Help menu). Only updates tracked files; config and local files are preserved."""
        from .custom_widget import ProgressMessageBox
        progress = ProgressMessageBox(self.tr('Update from GitHub'))
        progress.setTaskName(self.tr('Checking for updates...'))
        progress.updateTaskProgress(0)
        progress.show()
        thread = GitUpdateThread(shared.PROGRAM_PATH)

        def on_finished(success: bool, message: str):
            progress.hide()
            progress.deleteLater()
            if hasattr(self, '_git_update_threads') and thread in self._git_update_threads:
                self._git_update_threads.remove(thread)
            if success:
                create_info_dialog({'title': self.tr('Update from GitHub'), 'text': message})
            else:
                create_error_dialog(RuntimeError(message), self.tr('Update from GitHub'), 'GitUpdateThread')

        thread.finished_with_result.connect(on_finished)
        thread.start()
        if not hasattr(self, '_git_update_threads'):
            self._git_update_threads = []
        self._git_update_threads.append(thread)

    def _spellcheck_get_blocks(self):
        """Return [(block_idx, text_lines, trans_lines), ...] for current page (PR #974).
        Uses the scene's block list so the panel sees the same text blocks as the canvas/text panel.
        """
        # Prefer blocks from scene manager (current page as displayed); fallback to project
        if getattr(self, 'st_manager', None) and getattr(self.st_manager, 'textblk_item_list', None):
            result = []
            for blk_item in self.st_manager.textblk_item_list:
                blk = getattr(blk_item, 'blk', None)
                if blk is None:
                    continue
                text_lines = getattr(blk, 'text', None)
                if text_lines is None:
                    text_lines = []
                elif isinstance(text_lines, str):
                    text_lines = text_lines.split('\n') if text_lines else []
                trans = getattr(blk, 'translation', None) or ''
                trans_lines = trans.split('\n') if isinstance(trans, str) else []
                result.append((blk_item.idx, text_lines, trans_lines))
            return result
        if not getattr(self, 'imgtrans_proj', None) or getattr(self.imgtrans_proj, 'is_empty', True):
            return []
        page = self.imgtrans_proj.pages.get(self.imgtrans_proj.current_img, [])
        return [
            (i, self._spellcheck_normalize_lines(getattr(blk, 'text', None)), (getattr(blk, 'translation', None) or '').split('\n'))
            for i, blk in enumerate(page)
        ]

    def _spellcheck_normalize_lines(self, text_attr):
        """Return a list of strings from block text (list or string)."""
        if text_attr is None:
            return []
        if isinstance(text_attr, str):
            return text_attr.split('\n') if text_attr else []
        return [line if isinstance(line, str) else str(line) for line in text_attr]

    def _spellcheck_apply_replacement(self, block_idx: int, line_idx: int, new_line: str, is_translation: bool):
        """Apply spell check replacement to block (PR #974)."""
        page = self.imgtrans_proj.pages.get(self.imgtrans_proj.current_img, [])
        if block_idx < 0 or block_idx >= len(page):
            return
        blk = page[block_idx]
        if is_translation:
            lines = (blk.translation or '').split('\n')
            if line_idx < len(lines):
                lines[line_idx] = new_line
                blk.translation = _merge_stub_lines_in_string('\n'.join(lines))
                if block_idx < len(self.st_manager.pairwidget_list):
                    self.st_manager.pairwidget_list[block_idx].e_trans.setPlainText(blk.translation)
        else:
            if hasattr(blk, 'text') and blk.text is not None and line_idx < len(blk.text):
                blk.text[line_idx] = new_line
                if block_idx < len(self.st_manager.pairwidget_list):
                    self.st_manager.pairwidget_list[block_idx].e_source.setPlainText(blk.get_text())
        self.canvas.setProjSaveState(False)

    def shortcutNext(self):
        sender: QShortcut = self.sender()
        if isinstance(sender, QShortcut):
            if sender.key() == QKEY.Key_D:
                if self.canvas.editing_textblkitem is not None:
                    return
        if self.centralStackWidget.currentIndex() == 1:
            focus_widget = self.app.focusWidget()
            if self.st_manager.is_editting():
                self.st_manager.on_switch_textitem(1)
            elif isinstance(focus_widget, (SourceTextEdit, TransTextEdit)):
                self.st_manager.on_switch_textitem(1, current_editing_widget=focus_widget)
            else:
                index = self.pageList.currentIndex()
                page_count = self.pageList.count()
                if index.isValid():
                    row = index.row()
                    row = (row + 1) % page_count
                    self.pageList.setCurrentRow(row)

    def shortcutBefore(self):
        sender: QShortcut = self.sender()
        if isinstance(sender, QShortcut):
            if sender.key() == QKEY.Key_A:
                if self.canvas.editing_textblkitem is not None:
                    return
        if self.centralStackWidget.currentIndex() == 1:
            focus_widget = self.app.focusWidget()
            if self.st_manager.is_editting():
                self.st_manager.on_switch_textitem(-1)
            elif isinstance(focus_widget, (SourceTextEdit, TransTextEdit)):
                self.st_manager.on_switch_textitem(-1, current_editing_widget=focus_widget)
            else:
                index = self.pageList.currentIndex()
                page_count = self.pageList.count()
                if index.isValid():
                    row = index.row()
                    row = (row - 1 + page_count) % page_count
                    self.pageList.setCurrentRow(row)

    def shortcutTextedit(self):
        if not self._has_open_project():
            return
        if self.centralStackWidget.currentIndex() != 1:
            self._show_main_content()
        self.bottomBar.setPipelineVisible(True)
        self.bottomBar.texteditChecker.click()

    def shortcutTextblock(self):
        if self.centralStackWidget.currentIndex() == 1:
            if self.bottomBar.texteditChecker.isChecked():
                self.bottomBar.textblockChecker.click()

    def shortcutDrawboard(self):
        if not self._has_open_project():
            return
        if self.centralStackWidget.currentIndex() != 1:
            self._show_main_content()
        self.bottomBar.setPipelineVisible(True)
        self.bottomBar.paintChecker.click()

    def shortcutSpellCheckPanel(self):
        """Show Spell check panel (PR #974)."""
        if not self._has_open_project():
            return
        if self.centralStackWidget.currentIndex() != 1:
            self._show_main_content()
        self._relax_right_panel_vertical_minimums()
        if self.rightComicTransStackPanel.isHidden():
            self.rightComicTransStackPanel.show()
        self.rightComicTransStackPanel.setCurrentIndex(2)
        self.bottomBar.spellCheckChecker.blockSignals(True)
        self.bottomBar.spellCheckChecker.setChecked(True)
        self.bottomBar._set_spellcheck_icon(True)
        self.bottomBar.spellCheckChecker.blockSignals(False)
        self._keep_bottom_bar_visible()
    def _on_spellcheck_panel_close(self):
        """Switch right panel from spell check back to text panel."""
        self.rightComicTransStackPanel.setCurrentIndex(1)
        self.bottomBar.spellCheckChecker.blockSignals(True)
        self.bottomBar.spellCheckChecker.setChecked(False)
        self.bottomBar.spellCheckChecker.blockSignals(False)

    def on_spellcheck_panel_toggled(self):
        """Show or hide Spell check panel from bottom bar toggle."""
        if self.bottomBar.spellCheckChecker.isChecked():
            if not self._has_open_project():
                self.bottomBar.spellCheckChecker.blockSignals(True)
                self.bottomBar.spellCheckChecker.setChecked(False)
                self.bottomBar._set_spellcheck_icon(False)
                self.bottomBar.spellCheckChecker.blockSignals(False)
                return
            if self.centralStackWidget.currentIndex() != 1:
                self._show_main_content()
            self._relax_right_panel_vertical_minimums()
            if self.rightComicTransStackPanel.isHidden():
                self.rightComicTransStackPanel.show()
            self.rightComicTransStackPanel.setCurrentIndex(2)
        else:
            self.rightComicTransStackPanel.setCurrentIndex(1)
        self._keep_bottom_bar_visible()
    def _on_spellcheck_focus_block(self, block_idx: int, line_idx: int, is_translation: bool):
        """After spell check replace: switch to text panel and select the block so the change is visible."""
        self.rightComicTransStackPanel.setCurrentIndex(1)
        if not self.st_manager.textblk_item_list or block_idx < 0 or block_idx >= len(self.st_manager.textblk_item_list):
            return
        self.canvas.block_selection_signal = True
        self.canvas.clearSelection()
        blk_item = self.st_manager.textblk_item_list[block_idx]
        blk_item.setSelected(True)
        self.canvas.block_selection_signal = False
        self.st_manager.textEditList.set_selected_list([block_idx])
        self.st_manager.formatpanel.set_textblk_item(blk_item)
        self.canvas.gv.ensureVisible(blk_item)
        if block_idx < len(self.st_manager.pairwidget_list):
            edit = self.st_manager.pairwidget_list[block_idx].e_trans if is_translation else self.st_manager.pairwidget_list[block_idx].e_source
            edit.setFocus()
            edit.blockSignals(True)
            lines = edit.toPlainText().split('\n')
            if line_idx < len(lines):
                from qtpy.QtGui import QTextCursor
                cursor = edit.textCursor()
                pos = 0
                for i, line in enumerate(lines):
                    if i == line_idx:
                        cursor.setPosition(pos)
                        cursor.movePosition(QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.KeepAnchor)
                        edit.setTextCursor(cursor)
                        break
                    pos += len(line) + 1
            edit.blockSignals(False)

    def shortcutCtrlD(self):
        if self.centralStackWidget.currentIndex() == 1:
            if self.drawingPanel.isVisible():
                if self.drawingPanel.currentTool == self.drawingPanel.rectTool:
                    self.drawingPanel.rectPanel.delete_btn.click()
            elif self.canvas.textEditMode():
                self.canvas.delete_textblks.emit(0)

    def shortcutSelectAll(self):
        """Ctrl+A: select all pages when page list has focus, else select all text blocks on canvas."""
        focus = QApplication.focusWidget()
        if focus is self.pageList or (focus and self.pageList.isAncestorOf(focus)):
            self.pageList.selectAll()
            return
        if self.centralStackWidget.currentIndex() == 1:
            self.on_select_all_canvas()

    def eventFilter(self, obj, event):
        """Intercept Ctrl+A when canvas (or its viewport) has focus; select all text blocks on canvas. Skip when editing a text block so Ctrl+A selects all text in that block."""
        if event.type() == QEvent.Type.KeyPress:
            e = event
            if e.key() == Qt.Key.Key_A and e.modifiers() in (Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.MetaModifier):
                if self.centralStackWidget.currentIndex() == 1:
                    if self.canvas.editing_textblkitem is not None:
                        return super().eventFilter(obj, event)
                    if obj is self.canvas.gv or obj is self.canvas.gv.viewport():
                        self.on_select_all_canvas()
                        return True
                    focus = QApplication.focusWidget()
                    on_canvas = (
                        focus is self.canvas.gv or (focus and isinstance(focus, QWidget) and self.canvas.gv.isAncestorOf(focus))
                    )
                    if on_canvas:
                        self.on_select_all_canvas()
                        return True
        return super().eventFilter(obj, event)

    def shortcutSpace(self):
        if self.centralStackWidget.currentIndex() == 1:
            if self.drawingPanel.isVisible():
                if self.drawingPanel.currentTool == self.drawingPanel.rectTool:
                    self.drawingPanel.rectPanel.inpaint_btn.click()

    def shortcutBold(self):
        if self.textPanel.formatpanel.isVisible():
            self.textPanel.formatpanel.formatBtnGroup.boldBtn.click()

    def shortcutDelete(self):
        if self.canvas.gv.isVisible():
            self.canvas.delete_textblks.emit(1)

    def shortcutItalic(self):
        if self.textPanel.formatpanel.isVisible():
            self.textPanel.formatpanel.formatBtnGroup.italicBtn.click()

    def shortcutUnderline(self):
        if self.textPanel.formatpanel.isVisible():
            self.textPanel.formatpanel.formatBtnGroup.underlineBtn.click()

    def shortcutFontSizeUp(self):
        """Increase font size for selected text blocks."""
        if not self.textPanel.formatpanel.isVisible():
            return
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        gf = self.textPanel.formatpanel.global_format
        from ui import fontformat_commands as FC
        FC.ffmt_change_rel_font_size('rel_font_size', 1.25, gf, True, blks, clip_size=True)
        self.st_manager.updateSceneTextitems()
        self.canvas.setProjSaveState(False)

    def shortcutFontSizeDown(self):
        """Decrease font size for selected text blocks."""
        if not self.textPanel.formatpanel.isVisible():
            return
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        gf = self.textPanel.formatpanel.global_format
        from ui import fontformat_commands as FC
        FC.ffmt_change_rel_font_size('rel_font_size', 0.75, gf, True, blks, clip_size=True)
        self.st_manager.updateSceneTextitems()
        self.canvas.setProjSaveState(False)

    def shortcutCreateTextbox(self):
        """Create a default-size text box at cursor (Canvas)."""
        if self.centralStackWidget.currentIndex() == 1 and self.canvas.gv.isVisible():
            self.canvas.create_textbox_at_cursor()

    def shortcutContextMenuOptions(self):
        """Open Context menu options dialog (View)."""
        dlg = ContextMenuConfigDialog(self)
        dlg.exec()

    def shortcutFormatApply(self):
        """Apply font formatting to selection (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 1 and self.canvas.gv.isVisible():
            self.canvas.format_textblks.emit()

    def shortcutFormatLayout(self):
        """Auto layout selected text blocks (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 1 and self.canvas.gv.isVisible() and self.canvas.selected_text_items():
            self.canvas.layout_textblks.emit()

    def shortcutFitToBubble(self):
        """Fit to bubble for selection (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 1 and self.canvas.gv.isVisible() and self.canvas.selected_text_items():
            self.canvas.layout_textblks.emit()

    def shortcutFormatAutoFit(self):
        """Auto fit font size to box (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 1 and self.canvas.gv.isVisible() and self.canvas.selected_text_items():
            self.canvas.auto_fit_font_signal.emit()

    def shortcutFormatAutoFitBinary(self):
        """Auto fit font size binary search (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 1 and self.canvas.gv.isVisible() and self.canvas.selected_text_items():
            self.canvas.auto_fit_binary_signal.emit()

    def shortcutReAutoFitSelected(self):
        """Re-auto-fit selected text boxes (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 1 and self.canvas.gv.isVisible() and self.canvas.selected_text_items():
            if hasattr(self.canvas, "re_auto_fit_selected_signal"):
                self.canvas.re_auto_fit_selected_signal.emit()

    def shortcutReAutoFitCurrentPage(self):
        """Re-auto-fit current page (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 1 and self.canvas.gv.isVisible():
            if hasattr(self.canvas, "re_auto_fit_page_signal"):
                self.canvas.re_auto_fit_page_signal.emit()

    def shortcutReAutoFitAllPages(self):
        """Re-auto-fit all pages (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 1 and self.canvas.gv.isVisible():
            if hasattr(self.canvas, "re_auto_fit_all_signal"):
                self.canvas.re_auto_fit_all_signal.emit()

    def shortcutBalloonShapeAuto(self):
        """Set balloon shape to Auto (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 1 and self.canvas.gv.isVisible() and self.canvas.selected_text_items():
            self.canvas.set_balloon_shape_signal.emit("auto")

    def shortcutResizeToFitContent(self):
        """Resize selected text box(es) to fit content (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 1 and self.canvas.gv.isVisible() and self.canvas.selected_text_items():
            self.canvas.resize_to_fit_content_signal.emit()

    def shortcutLayoutReviewSelected(self):
        """Run layout review/fix for selected textboxes with a reviewable report."""
        self._show_layout_review_report('selected')

    def shortcutLayoutReviewPage(self):
        """Run layout review/fix for all textboxes on current page with a reviewable report."""
        self._show_layout_review_report('page')

    def _layout_review_result_for_scope(self, mode: str):
        if self.centralStackWidget.currentIndex() != 1 or not self.canvas.gv.isVisible():
            raise RuntimeError('No editable page is visible.')
        mode = (mode or 'page').strip().lower()
        if mode == 'selected':
            if not self.canvas.selected_text_items():
                raise RuntimeError('No selected text boxes.')
            target_indices = None
        else:
            target_indices = [
                blk.idx for blk in self.st_manager.textblk_item_list
                if blk is not None
            ]
            if not target_indices:
                raise RuntimeError('No text boxes on this page.')
        cfg = self._get_layout_review_config()
        result = self.st_manager.review_selected_textboxes_with_provider(
            config=cfg,
            target_block_indices=target_indices,
        )
        return result, target_indices

    def _summarize_layout_review_result(self, result: PageReviewResult) -> dict:
        blocks = list(getattr(result, 'blocks', []) or [])
        issues = sum(len(getattr(b, 'issues', []) or []) for b in blocks)
        actions = sum(len(getattr(b, 'actions', []) or []) for b in blocks)
        score = (sum(float(getattr(b, 'score_after', 1.0) or 0.0) for b in blocks) / len(blocks)) if blocks else 1.0
        return {'blocks': len(blocks), 'issues': issues, 'actions': actions, 'score_after': max(0.0, min(1.0, score))}

    def _layout_review_result_with_actions(self, result: PageReviewResult, selected_actions: list) -> PageReviewResult:
        selected_ids = {id(action) for action in (selected_actions or [])}
        blocks = []
        for block in getattr(result, 'blocks', []) or []:
            actions = [a for a in (getattr(block, 'actions', []) or []) if id(a) in selected_ids]
            blocks.append(
                BlockReviewResult(
                    block_index=block.block_index,
                    score_before=block.score_before,
                    score_after=block.score_after if actions else block.score_before,
                    issues=list(getattr(block, 'issues', []) or []),
                    actions=actions,
                )
            )
        return PageReviewResult(page_name=result.page_name, blocks=blocks)

    def _show_layout_review_report(self, mode: str):
        try:
            result, target_indices = self._layout_review_result_for_scope(mode)
        except Exception as e:
            self.pipelineInsightsPanel.add_warning('LAYOUT_REVIEW', str(e))
            self._show_status_message(self.tr('Layout review unavailable: {0}').format(str(e)), 5000)
            return
        summary = self._summarize_layout_review_result(result)
        dlg = LayoutReviewReportDialog(result, self)
        exec_fn = getattr(dlg, 'exec', None) or dlg.exec_
        if int(exec_fn()) != int(QDialog.Accepted):
            self.pipelineInsightsPanel.add_event('LAYOUT', self.tr('Layout review closed without applying fixes.'))
            return
        selected_actions = dlg.selected_actions()
        apply_result = self._layout_review_result_with_actions(result, selected_actions)
        applied = self.st_manager.apply_review_result(apply_result, target_block_indices=target_indices)
        if applied > 0:
            if self.imgtrans_proj and self.imgtrans_proj.directory:
                self.imgtrans_proj.save()
            self.pipelineInsightsPanel.add_event('LAYOUT', self.tr('Applied {0} layout fix action(s).').format(applied))
            self._show_status_message(self.tr('Layout review applied {0} fix action(s).').format(applied), 5000)
        else:
            self.pipelineInsightsPanel.add_event('LAYOUT', self.tr('Layout review found no applicable fixes.'))
            self._show_status_message(self.tr('Layout review: no applicable fixes.'), 4000)
        if summary['issues'] > 0:
            self.pipelineInsightsPanel.add_warning('LAYOUT_REVIEW', self.tr('{0} issue(s), {1} proposed fix(es).').format(summary['issues'], summary['actions']))

    def _get_layout_review_config(self) -> ReviewModelConfig:
        return self._layout_review_config_from_pcfg()

    def shortcutLayoutReviewConfig(self):
        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr("Layout review agent settings"))
        dlg.setMinimumWidth(560)

        outer = QVBoxLayout(dlg)
        form = QFormLayout()
        outer.addLayout(form)

        provider = QComboBox(dlg)
        provider.addItems(["heuristic", "local_api", "cloud_api"])
        provider.setCurrentText(getattr(pcfg.module, "layout_review_provider", "heuristic"))

        quick_profile = QComboBox(dlg)
        quick_profile.addItem(self.tr("Manual / existing settings"), "manual")
        quick_profile.addItem(self.tr("Heuristic (no API key needed)"), "heuristic_easy")
        quick_profile.addItem(self.tr("Cloud API quick setup"), "cloud_easy")
        quick_profile.addItem(self.tr("Local API quick setup (Ollama/LLM Studio)"), "local_easy")
        quick_profile.setToolTip(self.tr("Choose a starter profile to reduce setup steps for the layout review agent."))

        use_translator = QCheckBox(
            self.tr("Reuse current LLM_API_Translator API settings when possible"),
            dlg,
        )
        use_translator.setChecked(bool(getattr(pcfg.module, "layout_review_use_translator_settings", True)))

        api_provider = QComboBox(dlg)
        api_provider.addItems(["OpenAI", "Google", "Grok", "OpenRouter", "LLM Studio", "Ollama"])
        api_provider.setCurrentText(getattr(pcfg.module, "layout_review_api_provider", "OpenAI"))

        endpoint_preset = QComboBox(dlg)
        endpoint_preset.addItems([
            self.tr("Auto (provider default)"),
            self.tr("OpenAI Cloud"),
            self.tr("OpenRouter"),
            self.tr("Google Gemini (OpenAI-compatible)"),
            self.tr("LM Studio (local)"),
            self.tr("Ollama (local)"),
        ])

        _api_key_fallback = getattr(pcfg.module, "layout_review_api_key", "")
        _api_key_loaded = get_secret("layout_review_api_key") or _api_key_fallback
        api_key = QLineEdit(_api_key_loaded, dlg)
        api_key.setEchoMode(QLineEdit.EchoMode.Password)

        endpoint = QLineEdit(getattr(pcfg.module, "layout_review_api_endpoint", ""), dlg)
        endpoint.setPlaceholderText(self.tr("Optional. Example: http://localhost:11434/v1"))

        override_model = QLineEdit(getattr(pcfg.module, "layout_review_override_model", ""), dlg)
        override_model.setPlaceholderText(self.tr("Optional custom model name"))

        model_name = QLineEdit(getattr(pcfg.module, "layout_review_model_name", ""), dlg)
        model_name.setPlaceholderText(self.tr("Optional. Blank = use selected/translator model"))

        temperature = QDoubleSpinBox(dlg)
        temperature.setRange(0.0, 2.0)
        temperature.setSingleStep(0.05)
        temperature.setDecimals(2)
        temperature.setValue(float(getattr(pcfg.module, "layout_review_temperature", 0.0)))

        top_p = QDoubleSpinBox(dlg)
        top_p.setRange(0.0, 1.0)
        top_p.setSingleStep(0.05)
        top_p.setDecimals(2)
        top_p.setValue(float(getattr(pcfg.module, "layout_review_top_p", 1.0)))

        max_tokens = QSpinBox(dlg)
        max_tokens.setRange(128, 32768)
        max_tokens.setSingleStep(128)
        max_tokens.setValue(int(getattr(pcfg.module, "layout_review_max_tokens", 2048)))

        include_screenshot = QCheckBox(self.tr("Include page screenshot for vision-capable models"), dlg)
        include_screenshot.setChecked(bool(getattr(pcfg.module, "layout_review_include_page_screenshot", True)))

        screenshot_max_side = QSpinBox(dlg)
        screenshot_max_side.setRange(256, 4096)
        screenshot_max_side.setSingleStep(128)
        screenshot_max_side.setValue(int(getattr(pcfg.module, "layout_review_screenshot_max_side", 1280)))

        prompt = QPlainTextEdit(dlg)
        prompt.setPlainText(getattr(pcfg.module, "layout_review_prompt", ""))
        prompt.setMinimumHeight(110)

        extra_params = QPlainTextEdit(dlg)
        extra_params.setPlainText(getattr(pcfg.module, "layout_review_extra_params_json", "{}") or "{}")
        extra_params.setMinimumHeight(80)
        extra_params.setPlaceholderText(self.tr('Optional JSON, e.g. {"reasoning_effort": "low"}'))

        conn_status = QLabel(self.tr("Connection: not tested"), dlg)
        btn_test_conn = QPushButton(self.tr("Test connection"), dlg)

        form.addRow(self.tr("Quick setup"), quick_profile)
        form.addRow(self.tr("Provider"), provider)
        form.addRow("", use_translator)
        form.addRow(self.tr("API provider"), api_provider)
        form.addRow(self.tr("API key"), api_key)
        cred_label = QLabel(self.tr("Credential backend: OS keyring") if has_keyring() else self.tr("Credential backend: config fallback"), dlg)
        form.addRow("", cred_label)
        form.addRow(self.tr("Endpoint"), endpoint)
        form.addRow(self.tr("Override model"), override_model)
        form.addRow(self.tr("Model"), model_name)
        form.addRow(self.tr("Temperature"), temperature)
        form.addRow(self.tr("Top P"), top_p)
        form.addRow(self.tr("Max tokens"), max_tokens)
        form.addRow("", include_screenshot)
        form.addRow(self.tr("Screenshot max side"), screenshot_max_side)
        form.addRow(self.tr("Prompt"), prompt)
        form.addRow(self.tr("Extra params JSON"), extra_params)
        form.addRow(self.tr("Endpoint preset"), endpoint_preset)
        form.addRow("", btn_test_conn)
        form.addRow("", conn_status)

        def _apply_endpoint_preset(label: str):
            label = (label or "").strip()
            provider_map = {
                self.tr("OpenAI Cloud"): "OpenAI",
                self.tr("OpenRouter"): "OpenRouter",
                self.tr("Google Gemini (OpenAI-compatible)"): "Google",
                self.tr("LM Studio (local)"): "LLM Studio",
                self.tr("Ollama (local)"): "Ollama",
            }
            provider_name = provider_map.get(label, api_provider.currentText())
            if provider_name and label != self.tr("Auto (provider default)"):
                api_provider.setCurrentText(provider_name)
                endpoint.setText(provider_endpoint_preset(provider_name) or endpoint.text())

        endpoint_preset.currentTextChanged.connect(_apply_endpoint_preset)

        def _test_conn():
            rst = check_provider_connection(
                provider=api_provider.currentText(),
                endpoint=endpoint.text().strip(),
                api_key=api_key.text().strip(),
                timeout_sec=max(2.0, float(getattr(pcfg, "runtime_http_timeout_sec", 60.0))),
            )
            if rst.ok:
                conn_status.setText(self.tr("Connection: OK ({0})").format(rst.detail or "ok"))
            else:
                conn_status.setText(self.tr("Connection: failed ({0})").format(rst.detail or str(rst.status_code)))

        btn_test_conn.clicked.connect(_test_conn)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        outer.addWidget(buttons)

        exec_fn = getattr(dlg, "exec", None) or dlg.exec_
        if int(exec_fn()) != int(QDialog.Accepted):
            return

        profile_id = str(quick_profile.currentData() or "manual")
        if profile_id == "heuristic_easy":
            provider.setCurrentText("heuristic")
            use_translator.setChecked(False)
            include_screenshot.setChecked(False)
            temperature.setValue(0.0)
            top_p.setValue(1.0)
        elif profile_id == "cloud_easy":
            provider.setCurrentText("cloud_api")
            use_translator.setChecked(True)
            include_screenshot.setChecked(True)
            if not endpoint.text().strip():
                endpoint.setText("https://api.openai.com/v1")
            if not api_provider.currentText().strip():
                api_provider.setCurrentText("OpenAI")
        elif profile_id == "local_easy":
            provider.setCurrentText("local_api")
            use_translator.setChecked(False)
            include_screenshot.setChecked(True)
            if not endpoint.text().strip():
                endpoint.setText("http://localhost:11434/v1")
            api_provider.setCurrentText("Ollama")
            if not model_name.text().strip():
                model_name.setText("qwen2.5vl:latest")

        pcfg.module.layout_review_provider = provider.currentText()
        pcfg.module.layout_review_use_translator_settings = use_translator.isChecked()
        pcfg.module.layout_review_api_provider = api_provider.currentText()
        _layout_key = api_key.text().strip()
        _saved_layout = set_secret("layout_review_api_key", _layout_key) if _layout_key else False
        if _saved_layout and not bool(getattr(pcfg, "credential_use_plaintext_fallback", False)):
            pcfg.module.layout_review_api_key = ""
        else:
            pcfg.module.layout_review_api_key = _layout_key
        pcfg.module.layout_review_api_endpoint = endpoint.text().strip()
        pcfg.module.layout_review_override_model = override_model.text().strip()
        pcfg.module.layout_review_model_name = model_name.text().strip()
        pcfg.module.layout_review_temperature = float(temperature.value())
        pcfg.module.layout_review_top_p = float(top_p.value())
        pcfg.module.layout_review_max_tokens = int(max_tokens.value())
        pcfg.module.layout_review_include_page_screenshot = include_screenshot.isChecked()
        pcfg.module.layout_review_screenshot_max_side = int(screenshot_max_side.value())
        pcfg.module.layout_review_prompt = prompt.toPlainText().strip()
        pcfg.module.layout_review_extra_params_json = extra_params.toPlainText().strip() or "{}"
        save_config()

    def _module_param_value(self, params: dict, key: str, default=None):
        if not isinstance(params, dict):
            return default
        value = params.get(key, default)
        if isinstance(value, dict) and "value" in value:
            return value.get("value", default)
        return value

    def _current_translator_params_for_layout_review(self) -> dict:
        all_params = getattr(pcfg.module, "translator_params", {}) or {}
        translator_name = getattr(pcfg.module, "translator", "") or ""

        if not isinstance(all_params, dict):
            return {}

        params = (
            all_params.get(translator_name)
            or all_params.get("LLM_API_Translator")
            or all_params
        )
        return params if isinstance(params, dict) else {}

    def _layout_review_config_from_pcfg(self) -> ReviewModelConfig:
        import json

        provider = getattr(pcfg.module, "layout_review_provider", "heuristic") or "heuristic"
        if provider not in ("heuristic", "local_api", "cloud_api"):
            provider = "heuristic"

        extra = {}
        raw_extra = getattr(pcfg.module, "layout_review_extra_params_json", "{}") or "{}"
        try:
            loaded = json.loads(raw_extra)
            if isinstance(loaded, dict):
                extra.update(loaded)
        except Exception:
            extra["raw_extra_params"] = raw_extra

        model_name = getattr(pcfg.module, "layout_review_model_name", "") or ""

        if bool(getattr(pcfg.module, "layout_review_use_translator_settings", True)):
            params = self._current_translator_params_for_layout_review()

            # Mirror the LLM_API_Translator / llm_ocr style settings.
            key_map = [
                ("provider", "api_provider"),
                ("apikey", "api_key"),
                ("api_key", "api_key"),
                ("multiple_keys", "multiple_keys"),

                # These are what _layout_review_http_provider reads.
                ("api_base_url", "api_base_url"),
                ("base_url", "api_base_url"),
                ("endpoint", "api_base_url"),
                ("api path", "api_path"),
                ("api_path", "api_path"),

                ("endpoint_preset", "endpoint_preset"),
                ("override model", "override_model"),
                ("override_model", "override_model"),
                ("proxy", "proxy"),
            ]
            for src_key, dst_key in key_map:
                val = self._module_param_value(params, src_key, None)
                if val not in (None, ""):
                    extra[dst_key] = val

            if not model_name:
                model_name = (
                    self._module_param_value(params, "model", "")
                    or self._module_param_value(params, "override model", "")
                    or self._module_param_value(params, "override_model", "")
                    or ""
                )
        else:
            extra.update(
                {
                    "api_provider": getattr(pcfg.module, "layout_review_api_provider", "OpenAI"),
                    "api_key": (get_secret("layout_review_api_key") or getattr(pcfg.module, "layout_review_api_key", "")),
                    "api_base_url": getattr(pcfg.module, "layout_review_api_endpoint", ""),
                    "api_path": "/v1/chat/completions",
                    "override_model": getattr(pcfg.module, "layout_review_override_model", ""),
                }
            )
            if not model_name:
                model_name = getattr(pcfg.module, "layout_review_override_model", "") or ""

        return ReviewModelConfig(
            provider=provider,
            model_name=model_name,
            prompt=getattr(pcfg.module, "layout_review_prompt", "") or "",
            temperature=float(getattr(pcfg.module, "layout_review_temperature", 0.0)),
            top_p=float(getattr(pcfg.module, "layout_review_top_p", 1.0)),
            max_tokens=int(getattr(pcfg.module, "layout_review_max_tokens", 2048)),
            include_page_screenshot=bool(getattr(pcfg.module, "layout_review_include_page_screenshot", True)),
            screenshot_max_side=int(getattr(pcfg.module, "layout_review_screenshot_max_side", 1280)),
            extra_params=extra,
        )

    def _layout_review_local_api_handler(self, page_name, blocks, config: ReviewModelConfig) -> PageReviewResult:
        return self._layout_review_http_provider(page_name, blocks, config, provider_name="local_api")

    def _layout_review_cloud_api_handler(self, page_name, blocks, config: ReviewModelConfig) -> PageReviewResult:
        return self._layout_review_http_provider(page_name, blocks, config, provider_name="cloud_api")

    def _layout_review_http_provider(self, page_name, blocks, config: ReviewModelConfig, provider_name: str) -> PageReviewResult:
        ep = config.extra_params if isinstance(config.extra_params, dict) else {}
        base_url = str(ep.get("api_base_url") or ep.get("endpoint") or "").rstrip("/")
        api_path = str(ep.get("api_path") or ep.get("api path") or "/v1/chat/completions").strip()
        api_key = str(ep.get("api_key") or ep.get("apikey") or "").strip()
        if not base_url:
            LOGGER.warning("layout review %s provider has no api_base_url; fallback to heuristic", provider_name)
            return self.st_manager._resolve_review_provider(ReviewModelConfig(provider="heuristic")).review(page_name, blocks, config)
        url = f"{base_url}{api_path if api_path.startswith('/') else '/' + api_path}"
        page_ctx = ep.get("page_context") or config.extra_params.get("page_context")
        payload = {
            "model": config.model_name or str(ep.get("override_model") or "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": config.prompt},
                {"role": "user", "content": json.dumps({
                    "page_name": page_name,
                    "blocks": [b.__dict__ for b in blocks],
                    "page_context": page_ctx,
                    "required_schema": {
                        "actions": [{"action": "move|resize|set_font_size|auto_fit|center_in_bubble|resize_to_fit_content", "block_index": 0, "args": {}, "reason": ""}]
                    }
                })},
            ],
            "temperature": float(config.temperature),
            "top_p": float(config.top_p),
            "max_tokens": int(config.max_tokens),
            "response_format": {"type": "json_object"},
        }
        req = urlrequest.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
        req.add_header("Content-Type", "application/json")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        retries = max(1, int(getattr(pcfg, "runtime_http_retries", 1)))
        timeout_sec = max(2.0, float(getattr(pcfg, "runtime_http_timeout_sec", 60.0)))
        last_error = None
        for attempt in range(retries):
            try:
                with urlrequest.urlopen(req, timeout=timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                parsed = json.loads(content) if isinstance(content, str) else content
                return self._build_review_result_from_provider_actions(page_name, blocks, parsed.get("actions", []))
            except (urlerror.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as e:
                last_error = e
                LOGGER.warning("layout review %s attempt %d/%d failed: %s", provider_name, attempt + 1, retries, e)
        self.pipelineInsightsPanel.add_warning('HTTP_RETRY_FAIL', f'Layout review provider failed after {retries} attempts ({provider_name}).')
        LOGGER.warning("layout review %s provider request failed after retries, fallback to heuristic: %s", provider_name, last_error)
        return self.st_manager._resolve_review_provider(ReviewModelConfig(provider="heuristic")).review(page_name, blocks, config)

    def _build_review_result_from_provider_actions(self, page_name: str, blocks, raw_actions) -> PageReviewResult:
        action_map = {}
        for ra in raw_actions if isinstance(raw_actions, list) else []:
            try:
                action = ReviewAction(
                    action=str(ra.get("action", "move")),
                    block_index=int(ra.get("block_index", -1)),
                    args=dict(ra.get("args", {}) or {}),
                    reason=str(ra.get("reason", "")),
                )
                if action.block_index < 0:
                    continue
                action_map.setdefault(action.block_index, []).append(action)
            except Exception:
                continue
        block_results = []
        for blk in blocks:
            block_results.append(
                BlockReviewResult(
                    block_index=blk.block_index,
                    score_before=1.0,
                    score_after=1.0 if blk.block_index not in action_map else 0.95,
                    issues=[ReviewIssue(code="provider_suggested_fix", severity="info", message="Provider suggested layout adjustments.", score_penalty=0.0)] if blk.block_index in action_map else [],
                    actions=action_map.get(blk.block_index, []),
                )
            )
        return PageReviewResult(page_name=page_name, blocks=block_results)

    def on_redo(self):
        log_diagnostic_event(
            "ui.redo",
            page_index=self.pageList.currentRow(),
            page_name=getattr(self.imgtrans_proj, 'current_img', None) if self.imgtrans_proj is not None else None,
            block_count=len(getattr(self.st_manager, 'textblk_item_list', [])) if self.st_manager is not None else 0,
        )
        self.canvas.redo()

    def on_undo(self):
        log_diagnostic_event(
            "ui.undo",
            page_index=self.pageList.currentRow(),
            page_name=getattr(self.imgtrans_proj, 'current_img', None) if self.imgtrans_proj is not None else None,
            block_count=len(getattr(self.st_manager, 'textblk_item_list', [])) if self.st_manager is not None else 0,
        )
        self.canvas.undo()

    def on_install_google_font(self):
        target_dir = osp.join(shared.PROGRAM_PATH, "fonts", "google")
        dlg = GoogleFontInstallDialog(self, target_dir=target_dir)
        dlg.font_installed.connect(self._on_google_font_installed)
        dlg.exec()

    def _on_google_font_installed(self, installed, registered, target_dir):
        # Refresh right-panel font selectors immediately while preserving the user's
        # current "show only custom fonts" preference.
        self.on_show_only_custom_font(bool(getattr(pcfg, 'let_show_only_custom_fonts_flag', False)))
        LOGGER.info("Installed Google font files to %s: %s; registered families: %s", target_dir, installed, registered)

    def on_page_search(self):
        if self.canvas.gv.isVisible():
            fo = self.app.focusObject()
            sel_text = ''
            tgt_edit = None
            blkitem = self.canvas.editing_textblkitem
            if fo == self.canvas.gv and blkitem is not None:
                sel_text = blkitem.textCursor().selectedText()
                tgt_edit = self.st_manager.pairwidget_list[blkitem.idx].e_trans
            elif isinstance(fo, QTextEdit) or isinstance(fo, QPlainTextEdit):
                sel_text = fo.textCursor().selectedText()
                if isinstance(fo, SourceTextEdit):
                    tgt_edit = fo
            se = self.canvas.search_widget.search_editor
            se.setFocus()
            if sel_text != '':
                se.setPlainText(sel_text)
                cursor = se.textCursor()
                cursor.select(QTextCursor.SelectionType.Document)
                se.setTextCursor(cursor)

            if self.canvas.search_widget.isHidden():
                self.canvas.search_widget.show()
            self.canvas.search_widget.setCurrentEditor(tgt_edit)

    def on_global_search(self):
        if self.canvas.gv.isVisible():
            if not self.leftBar.globalSearchChecker.isChecked():
                self.leftBar.globalSearchChecker.click()
            fo = self.app.focusObject()
            sel_text = ''
            blkitem = self.canvas.editing_textblkitem
            if fo == self.canvas.gv and blkitem is not None:
                sel_text = blkitem.textCursor().selectedText()
            elif isinstance(fo, QTextEdit) or isinstance(fo, QPlainTextEdit):
                sel_text = fo.textCursor().selectedText()
            se = self.global_search_widget.search_editor
            se.setFocus()
            if sel_text != '':
                se.setPlainText(sel_text)
                cursor = se.textCursor()
                cursor.select(QTextCursor.SelectionType.Document)
                se.setTextCursor(cursor)
                
                self.global_search_widget.commit_search()

    def on_texteditlist_contextmenu_requested(self, pos: QPoint, from_empty: bool = True, block_idx: int = -1):
        menu = QMenu(self)
        jump_act = None
        compare_tr_act = menu.addAction(self.tr("Compare translation providers"))
        compare_ocr_act = menu.addAction(self.tr("Compare OCR engines"))
        compare_det_act = menu.addAction(self.tr("Compare detectors"))
        compare_inp_act = menu.addAction(self.tr("Compare inpainters"))
        menu.addSeparator()
        open_assist_act = menu.addAction(self.tr("Open Translation Assist"))
        if not from_empty and block_idx >= 0:
            jump_act = menu.addAction(self.tr("Jump to block on canvas"))
        rst = menu.exec_(pos)
        if rst is None:
            return
        if rst == open_assist_act:
            self.on_open_translation_assist_dock()
            return
        if rst == jump_act:
            self.jump_to_canvas_block(int(block_idx))
            return
        if rst in (compare_tr_act, compare_ocr_act, compare_det_act, compare_inp_act):
            scope = 'translator'
            if rst == compare_ocr_act:
                scope = 'ocr'
            elif rst == compare_det_act:
                scope = 'detector'
            elif rst == compare_inp_act:
                scope = 'inpainter'
            self._run_text_block_compare_from_menu(scope=scope, block_idx=int(block_idx))

    def _run_text_block_compare_from_menu(self, scope: str, block_idx: int = -1):
        try:
            target_blocks = []
            if block_idx >= 0:
                self.jump_to_canvas_block(block_idx)
                target_blocks = [self._selected_assist_page_block()]
            else:
                target_blocks = list(self._selected_assist_page_blocks() or [])
            target_blocks = [b for b in target_blocks if b]
            if not target_blocks:
                raise RuntimeError(self.tr("Select a text block first."))
            rst = self._api_translation_assist_compare({
                'compare_scope': str(scope or 'translator'),
                'preset': 'high_quality',
                'max_candidates': int(getattr(pcfg.module, 'max_mt_candidates', 6) or 6),
                'preferred_assist_providers': list(getattr(pcfg.module, 'preferred_assist_providers', ['current_translator']) or []),
                'page': target_blocks[0]['page'],
                'block': target_blocks[0]['block'],
            })
            candidates = list(rst.get('candidates', []) or [])
            if not candidates:
                create_info_dialog(self.tr("No compare candidates available."), self.tr("Compare"), "CompareEmpty")
                return
            dlg = CompareResultsDialog(scope=scope, candidates=candidates, parent=self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            picked = dlg.selected_index()
            if picked < 0 or picked >= len(candidates):
                return
            row = candidates[picked]
            if scope == 'ocr':
                self._api_ocr_apply_compare_choice_ui({
                    'page': rst.get('page', getattr(self.imgtrans_proj, 'current_img', '')),
                    'index': (block_idx if block_idx >= 0 else self._selected_assist_page_block()['block']),
                    'text': str(row.get('text', '') or ''),
                    'engine': str(row.get('text', '') or ''),
                })
                self.pipelineInsightsPanel.add_event('OCR_COMPARE', self.tr('Applied OCR compare choice to selected block.'))
            elif scope in {'detector', 'inpainter'}:
                selected = str(row.get('text', '') or '').strip()
                if scope == 'detector':
                    pcfg.module.textdetector = selected
                    self.module_manager.setTextDetector(selected)
                else:
                    pcfg.module.inpainter = selected
                    self.module_manager.setInpainter(selected)
                self.pipelineInsightsPanel.add_event('MODULE_COMPARE', self.tr('Switched {0} to {1}.').format(scope, selected))
            else:
                # Translator-scope compare rows are assist candidates; apply text to each selected block.
                txt = str(row.get('text', '') or '').strip()
                for sel_block in target_blocks:
                    self._api_translation_assist_apply_text({'page': sel_block['page'], 'block': sel_block['block'], 'text': txt})
                self.refresh_translation_assist_for_selection()
            self.pipelineInsightsPanel.add_event('COMPARE', self.tr('Applied compare candidate #{0} for {1}.').format(int(picked + 1), str(scope)))
        except Exception as e:
            create_error_dialog(e, self.tr('Compare failed'))

    def show_pre_MT_keyword_window(self):
        self.mtPreSubWidget.show()

    def show_MT_keyword_window(self):
        self.mtSubWidget.show()

    def show_translation_context_dialog(self):
        """Open dialog to edit project series context path and glossary."""
        proj = self.imgtrans_proj
        if proj is None or getattr(proj, "directory", None) is None:
            create_info_dialog(
                self.tr("Open a project first (File → Open Folder / Open Project) to set project translation context."),
                self.tr("Translation context"),
                "TranslationContext",
            )
            return
        dlg = TranslationContextDialog(proj, self)
        dlg.exec()


    def show_OCR_keyword_window(self):
        self.ocrSubWidget.show()

    def on_open_merge_tool(self):
        """Open the region merge tool dialog."""
        if not hasattr(self, 'merge_dialog') or self.merge_dialog is None:
            from .merge_dialog import MergeDialog
            from qtpy.QtCore import QThread
            from qtpy.QtWidgets import QProgressDialog
            from utils import merger
            
            self.merge_dialog = MergeDialog(self)
            self.merge_dialog.preview_current_clicked.connect(lambda: self.run_merge_task(on_current=True, dry_run=True))
            self.merge_dialog.run_current_clicked.connect(lambda: self.run_merge_task(on_current=True, dry_run=False))
            self.merge_dialog.run_all_clicked.connect(lambda: self.run_merge_task(on_current=False, dry_run=False))
        
        if self.merge_dialog.isVisible():
            self.merge_dialog.raise_()
            self.merge_dialog.activateWindow()
        else:
            self.merge_dialog.show()

    def on_open_manga_source(self):
        """Open the Manga / Comic source dialog (search, chapters, download)."""
        if not hasattr(self, 'manga_source_dialog') or self.manga_source_dialog is None:
            from .manga_source_dialog import MangaSourceDialog
            self.manga_source_dialog = MangaSourceDialog(self)
            self.manga_source_dialog.open_folder_requested.connect(self.OpenProj)
        if self.manga_source_dialog.isVisible():
            self.manga_source_dialog.raise_()
            self.manga_source_dialog.activateWindow()
        else:
            self.manga_source_dialog.show()

    def on_open_translation_assist_dock(self):
        if not hasattr(self, '_translation_assist_dock') or self._translation_assist_dock is None:
            self._translation_assist_dock = TranslationAssistDock(self)
            if hasattr(self, 'addDockWidget'):
                self.addDockWidget(Qt.RightDockWidgetArea, self._translation_assist_dock)
            else:
                # FramelessWindow inherits QWidget (not QMainWindow), so fallback to floating tool window.
                self._translation_assist_dock.setFloating(True)
                self._translation_assist_dock.setAllowedAreas(Qt.NoDockWidgetArea)
            self._translation_assist_dock.set_callbacks(
                self.refresh_translation_assist_for_selection,
                self.apply_translation_assist_candidate_for_selection,
                self.add_translation_assist_candidate_to_tm,
                self.add_translation_assist_candidate_to_glossary,
                self.apply_translation_assist_text_for_selection,
            )
            self._translation_assist_dock.set_query_defaults({
                'auto_query_tm': getattr(pcfg.module, 'auto_query_tm', True),
                'auto_query_glossary': getattr(pcfg.module, 'auto_query_glossary', True),
                'auto_query_sfx': getattr(pcfg.module, 'auto_query_sfx', True),
                'auto_query_concordance': getattr(pcfg.module, 'auto_query_concordance', False),
                'max_mt_candidates': getattr(pcfg.module, 'max_mt_candidates', 6),
                'preferred_assist_providers': getattr(pcfg.module, 'preferred_assist_providers', ['current_translator']),
                'default_assist_prompt_profile': getattr(pcfg.module, 'default_assist_prompt_profile', 'dialogue'),
                'prompt_profiles': sorted(PROMPT_PROFILES.keys()),
            })
            # Refresh dynamic provider lists from module registry
            all_providers = ['TM', 'Glossary', 'Concordance', 'SFX'] + sorted(GET_VALID_TRANSLATORS())
            self._translation_assist_dock.refresh_provider_list(all_providers)
        self._translation_assist_dock.show()
        self._translation_assist_dock.raise_()
        self.refresh_translation_assist_for_selection()

    def _selected_assist_page_blocks(self):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            return []
        page = str(getattr(self.imgtrans_proj, 'current_img', '') or '').strip()
        if not page:
            return []
        blocks = list((self.imgtrans_proj.pages or {}).get(page, []) or [])
        sel = list(self.canvas.selected_text_items() or [])
        out = []
        for it in sel:
            blk = getattr(it, 'blk', None)
            if blk is None:
                continue
            for i, b in enumerate(blocks):
                if b is blk:
                    out.append({'page': page, 'block': i})
                    break
        return out

    def _selected_assist_page_block(self):
        rows = self._selected_assist_page_blocks()
        return rows[0] if rows else None

    def refresh_translation_assist_for_selection(self, opts: dict = None):
        if not hasattr(self, '_translation_assist_dock') or self._translation_assist_dock is None:
            return
        opts = dict(opts or {})
        pcfg.module.auto_query_tm = bool(opts.get('auto_query_tm', getattr(pcfg.module, 'auto_query_tm', True)))
        pcfg.module.auto_query_glossary = bool(opts.get('auto_query_glossary', getattr(pcfg.module, 'auto_query_glossary', True)))
        pcfg.module.auto_query_sfx = bool(opts.get('auto_query_sfx', getattr(pcfg.module, 'auto_query_sfx', True)))
        pcfg.module.auto_query_concordance = bool(opts.get('auto_query_concordance', getattr(pcfg.module, 'auto_query_concordance', False)))
        pcfg.module.max_mt_candidates = int(opts.get('max_mt_candidates', getattr(pcfg.module, 'max_mt_candidates', 6)) or 6)
        pcfg.module.preferred_assist_providers = list(opts.get('preferred_assist_providers', getattr(pcfg.module, 'preferred_assist_providers', ['current_translator'])) or [])
        pcfg.module.default_assist_prompt_profile = str(opts.get('default_assist_prompt_profile', getattr(pcfg.module, 'default_assist_prompt_profile', 'dialogue')) or 'dialogue')
        if bool(opts.get('clear_assist_cache', False)):
            self._api_translation_assist_cache({'action': 'clear'})
        sel = self._selected_assist_page_block()
        if not sel:
            self._translation_assist_dock.set_assist_payload({
                'source_text': '',
                'current_target_text': '',
                'candidates': [],
                'summary': self.tr('Select a text block first.'),
            })
            return
        base = self._api_translation_assist_block({'page': sel['page'], 'block': sel['block']})
        source = str(base.get('source_text', '') or '')
        target = str(base.get('current_target_text', '') or '')
        if bool(opts.get('run_compare', False)):
            scope = str(opts.get('compare_scope', 'translator') or 'translator').strip().lower()
            providers = list(opts.get('preferred_assist_providers', getattr(pcfg.module, 'preferred_assist_providers', ['current_translator'])) or [])
            self._translation_assist_dock.set_busy(True)
            worker = CompareWorker(
                scope=scope,
                providers=providers,
                source_text=source,
                lang_source=str(getattr(pcfg.module, 'translate_source', 'Auto') or 'Auto'),
                lang_target=str(getattr(pcfg.module, 'translate_target', '简体中文') or '简体中文'),
                current_module=str(getattr(pcfg.module, 'translator', '') or '') if scope == 'translator' else str(getattr(pcfg.module, {'ocr': 'ocr', 'detector': 'textdetector', 'inpainter': 'inpainter'}.get(scope, ''), '') or ''),
            )
            thread = QThread(self)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(lambda rst: self._on_compare_worker_finished(rst, sel['page'], sel['block'], source, target))
            worker.finished.connect(thread.quit)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.start()
            return
        max_n = max(1, int(getattr(pcfg.module, 'max_mt_candidates', 6) or 6))
        tm_hits = []
        glossary_hits = []
        sfx_hits = []
        concordance_hits = []
        telemetry = {}
        if bool(getattr(pcfg.module, 'auto_query_tm', True)):
            t0 = time.perf_counter()
            tm_hits = list(self._api_translation_assist_tm({'page': sel['page'], 'block': sel['block'], 'source_text': source}).get('hits', []) or [])
            telemetry["TM"] = {"latency_ms": int((time.perf_counter() - t0) * 1000), "source": "project_tm"}
        if bool(getattr(pcfg.module, 'auto_query_glossary', True)):
            t0 = time.perf_counter()
            glossary_hits = list(self._api_translation_assist_glossary({'page': sel['page'], 'block': sel['block'], 'source_text': source}).get('hits', []) or [])
            telemetry["Glossary"] = {"latency_ms": int((time.perf_counter() - t0) * 1000), "source": "project_glossary"}
        if bool(getattr(pcfg.module, 'auto_query_sfx', True)):
            t0 = time.perf_counter()
            sfx_hits = list(self._api_translation_assist_sfx({'page': sel['page'], 'block': sel['block'], 'query': source}).get('hits', []) or [])
            telemetry["SFX"] = {"latency_ms": int((time.perf_counter() - t0) * 1000), "source": "sfx_dictionary"}
        if bool(getattr(pcfg.module, 'auto_query_concordance', False)):
            t0 = time.perf_counter()
            concordance_hits = list(self._api_translation_assist_concordance({'page': sel['page'], 'block': sel['block'], 'query': source, 'limit': max_n}).get('hits', []) or [])
            telemetry["Concordance"] = {"latency_ms": int((time.perf_counter() - t0) * 1000), "source": "project_concordance"}
        providers = list(getattr(pcfg.module, 'preferred_assist_providers', ['current_translator']) or [])
        profile = str(getattr(pcfg.module, 'default_assist_prompt_profile', 'dialogue') or 'dialogue')
        cached = self._translation_assist_service().get_cached_candidates(source, profile, providers)
        candidates = cached if cached else build_candidates_from_sources(
            tm_hits=tm_hits,
            glossary_hits=glossary_hits,
            sfx_hits=sfx_hits,
            concordance_hits=concordance_hits,
            max_candidates=max_n,
            telemetry=telemetry,
        )
        if not cached:
            self._translation_assist_service().set_cached_candidates(source, profile, providers, candidates)
        cand_payload = self._api_translation_assist_candidates({
            'page': sel['page'], 'block': sel['block'],
            'source_text': source, 'current_target_text': target,
            'candidates': candidates,
        })
        self._translation_assist_dock.set_assist_payload({
            'source_text': source,
            'current_target_text': target,
            'candidates': list(cand_payload.get('candidates', []) or []),
            'summary': self.tr('Candidates: {0}').format(len(candidates)),
            'qa_warnings': self._translation_assist_block_qa_warnings(sel['page'], sel['block']),
        })

    def _on_compare_worker_finished(self, rst: dict, page: str, block: int, source: str, target: str):
        if not hasattr(self, '_translation_assist_dock') or self._translation_assist_dock is None:
            return
        self._translation_assist_dock.set_busy(False)
        scope = str(rst.get('scope', 'translator'))
        candidates = list(rst.get('candidates', []) or [])
        warnings = list(rst.get('warnings', []) or [])
        self._api_translation_assist_candidates({
            'page': page, 'block': block,
            'source_text': source, 'current_target_text': target,
            'candidates': candidates,
        })
        warn_texts = [str(w.get('message', '')) for w in warnings if str(w.get('message', '')).strip()]
        self._translation_assist_dock.set_assist_payload({
            'source_text': source,
            'current_target_text': target,
            'candidates': candidates,
            'summary': self.tr('Compare scope: {0} | candidates: {1}').format(scope, len(candidates)),
            'qa_warnings': self._translation_assist_block_qa_warnings(page, block) + warn_texts,
        })

    def on_translation_assist_selection_changed(self):
        if not hasattr(self, '_translation_assist_dock') or self._translation_assist_dock is None:
            return
        if not self._translation_assist_dock.isVisible():
            return
        if not bool(getattr(pcfg.module, 'enable_translation_assist', True)):
            return
        self.refresh_translation_assist_for_selection()

    def apply_translation_assist_candidate_for_selection(self, candidate_id: str):
        sel = self._selected_assist_page_block()
        if not sel:
            return
        rst = self._api_translation_assist_apply_candidate({
            'page': sel['page'],
            'block': sel['block'],
            'candidate_id': str(candidate_id or ''),
            'apply_to_project': False,
        })
        candidate_text = str((rst or {}).get('candidate_text', '') or '')
        if candidate_text:
            page = str(sel['page'])
            block = int(sel['block'])
            blk_items = list(self.st_manager.textblk_item_list or [])
            if 0 <= block < len(blk_items):
                blk_item = blk_items[block]
                trans_edit = None
                if 0 <= block < len(self.st_manager.pairwidget_list):
                    trans_edit = self.st_manager.pairwidget_list[block].e_trans
                self.canvas.push_undo_command(ApplyTranslationCandidateCommand(blk_item, trans_edit, candidate_text))
                self.canvas.setProjSaveState(False)
        self.refresh_translation_assist_for_selection()

    def add_translation_assist_candidate_to_tm(self, candidate_id: str):
        sel = self._selected_assist_page_block()
        if not sel:
            return
        rst = self._api_translation_assist_apply_candidate({
            'page': sel['page'],
            'block': sel['block'],
            'candidate_id': str(candidate_id or ''),
            'apply_to_project': False,
        })
        text = str((rst or {}).get('candidate_text', '') or '').strip()
        if not text:
            return
        self._api_translation_assist_add_to_tm({'page': sel['page'], 'block': sel['block'], 'text': text})
        self.refresh_translation_assist_for_selection()

    def add_translation_assist_candidate_to_glossary(self, candidate_id: str):
        sel = self._selected_assist_page_block()
        if not sel:
            return
        _p, _b, blk = self._resolve_assist_block(sel)
        source = (getattr(blk, 'get_text', lambda: '')() or '').strip()
        rst = self._api_translation_assist_apply_candidate({
            'page': sel['page'],
            'block': sel['block'],
            'candidate_id': str(candidate_id or ''),
            'apply_to_project': False,
        })
        target = str((rst or {}).get('candidate_text', '') or '').strip()
        if not source or not target:
            return
        self._api_translation_assist_add_to_glossary({'source': source, 'target': target})
        self.refresh_translation_assist_for_selection()

    def apply_translation_assist_text_for_selection(self, text: str):
        sel = self._selected_assist_page_block()
        if not sel:
            return
        rst = self._api_translation_assist_apply_text({'page': sel['page'], 'block': sel['block'], 'text': text})
        candidate_text = str((rst or {}).get('text', '') or '')
        if not candidate_text:
            return
        page = str(sel['page'])
        block = int(sel['block'])
        blk_items = list(self.st_manager.textblk_item_list or [])
        if 0 <= block < len(blk_items):
            blk_item = blk_items[block]
            trans_edit = None
            if 0 <= block < len(self.st_manager.pairwidget_list):
                trans_edit = self.st_manager.pairwidget_list[block].e_trans
            self.canvas.push_undo_command(ApplyTranslationCandidateCommand(blk_item, trans_edit, candidate_text))
            self.canvas.setProjSaveState(False)
        self.refresh_translation_assist_for_selection()

    def _translation_assist_block_qa_warnings(self, page: str, block: int):
        warnings = []
        try:
            qa = self._api_translation_assist_qa({'page': page, 'block': block})
            warnings.extend(list(qa.get('warnings', []) or []))
            _p, _b, blk = self._resolve_assist_block({'page': page, 'block': block})
            src = (getattr(blk, 'get_text', lambda: '')() or '').strip()
            tgt = str(getattr(blk, 'translation', '') or '').strip()
            if not tgt:
                warnings.append(self.tr('Target text is empty.'))
            if src and tgt and src == tgt:
                warnings.append(self.tr('Target equals source (untranslated risk).'))
            if src and tgt and len(tgt) < max(1, int(len(src) * 0.2)):
                warnings.append(self.tr('Target may be too short versus source.'))
        except Exception:
            return warnings
        # stable unique order
        seen = set()
        out = []
        for w in warnings:
            k = str(w).strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(w)
        return out

    def on_open_realtime_translator(self):
        """Open realtime screen translation dialog (project-less live mode)."""
        # Dev-only feature - guard against accidental use
        if not pcfg.dev_mode:
            from utils.message import create_info_dialog
            create_info_dialog(self, self.tr('Developer Mode Required'), self.tr('Realtime Screen Translator is a developer-only feature. Enable Developer Mode in Settings > General to use it.'))
            return
        if not hasattr(self, '_realtime_translator_dialog') or self._realtime_translator_dialog is None:
            self._realtime_translator_dialog = RealtimeTranslatorDialog(self, module_manager=self.module_manager)
        if self._realtime_translator_dialog.isVisible():
            self._realtime_translator_dialog.raise_()
            self._realtime_translator_dialog.activateWindow()
        else:
            self._realtime_translator_dialog.show()

    def on_open_batch_queue(self):
        """Open the Batch queue dialog (multiple folders, Pause/Cancel)."""
        if self._batch_queue_dialog is None:
            self._batch_queue_dialog = BatchQueueDialog(self)
            self._batch_queue_dialog.start_queue_requested.connect(self._on_batch_start)
            self._batch_queue_dialog.pause_requested.connect(self.module_manager.requestPausePipeline)
            self._batch_queue_dialog.resume_requested.connect(self.module_manager.requestResumePipeline)
            self._batch_queue_dialog.cancel_requested.connect(self._on_batch_cancel)
            self.batch_queue_empty.connect(self._batch_queue_dialog.set_queue_empty)
            self.batch_queue_cancelled.connect(self._batch_queue_dialog.set_queue_cancelled)
            self.batch_queue_item_started.connect(self._on_batch_item_started)
        if self._batch_queue_dialog.isVisible():
            self._batch_queue_dialog.raise_()
            self._batch_queue_dialog.activateWindow()
        else:
            self._batch_queue_dialog.show()

    def _on_batch_start(self, paths: list, skip_ignored_pages: bool = True):
        self.run_batch(paths, skip_ignored_pages=skip_ignored_pages)

    def _on_batch_cancel(self):
        self._batch_cancelled = True
        self.exec_dirs = []
        # Cleanup any extracted ZIP temp dirs
        try:
            self._zip_batch.cleanup_all()
        except Exception:
            pass
        self.module_manager.stopImgtransPipeline()

    def _on_batch_item_started(self, path: str):
        if self._batch_queue_dialog is not None:
            self._batch_queue_dialog.set_running_state(True, paused=False, current_path=path)
            if self._batch_queue_dialog.list_widget.count() > 0:
                self._batch_queue_dialog.list_widget.takeItem(0)

    def on_open_manage_models(self):
        """Open the Manage models dialog (check downloaded models, download selected)."""
        dlg = ModelManagerDialog(self)
        dlg.exec()

    def on_open_troubleshooting(self):
        """Open troubleshooting docs for model downloads and environment/network issues."""
        root = osp.dirname(osp.dirname(osp.abspath(__file__)))
        docs_path = osp.join(root, 'docs', 'TROUBLESHOOTING.md')
        if osp.isfile(docs_path):
            if sys.platform == 'win32':
                os.startfile(docs_path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', docs_path], check=False)
            else:
                subprocess.run(['xdg-open', docs_path], check=False)
        else:
            create_info_dialog({'title': self.tr('Troubleshooting'), 'text': self.tr('docs/TROUBLESHOOTING.md not found.')})

    def _persist_model_download_status(self, summary: dict, flow: str, status: str):
        payload = dict(summary or {})
        payload['flow'] = flow
        payload['status'] = status
        pcfg.model_download_last_status = payload
        save_config()
        LOGGER.info(
            "Persisted model download status | flow=%s | status=%s | package_ids=%s | module_count=%s | downloaded=%s | failed=%s | skipped=%s",
            flow,
            status,
            payload.get('package_ids', []),
            payload.get('module_count', 0),
            payload.get('downloaded', 0),
            payload.get('failed', 0),
            payload.get('skipped', 0),
        )

    def _collect_startup_diagnostics_text(self) -> str:
        from qtpy import API, QT_VERSION
        from utils.logger import NoisyThirdPartyFilter
        lines = [
            f"Version: {(QApplication.instance().applicationVersion() or 'unknown') if QApplication.instance() else 'unknown'}",
            f"Qt API: {API}",
            f"Qt Version: {QT_VERSION}",
            f"Python: {sys.version.splitlines()[0]}",
            f"Python executable: {sys.executable}",
            f"Platform: {sys.platform}",
            f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
            f"BT_GPU_PROFILE: {os.environ.get('BT_GPU_PROFILE', '<unset>')}",
            f"Detector: {getattr(pcfg.module, 'textdetector', '')}",
            f"OCR: {getattr(pcfg.module, 'ocr', '')}",
            f"Inpainter: {getattr(pcfg.module, 'inpainter', '')}",
            f"Translator: {getattr(pcfg.module, 'translator', '')}",
        ]
        suppressed = NoisyThirdPartyFilter.suppressed_summary()
        if suppressed:
            lines.append(f"Suppressed noisy warning groups: {sum(suppressed.values())}")
        last = getattr(pcfg, 'model_download_last_status', {}) or {}
        if last:
            lines.extend([
                "",
                "Last model download:",
                f"  flow={last.get('flow', 'unknown')}, status={last.get('status', 'unknown')}",
                f"  downloaded={last.get('downloaded', 0)}, failed={last.get('failed', 0)}, skipped={last.get('skipped', 0)}",
                f"  package_ids={last.get('package_ids', [])}",
            ])
            failed_items = last.get('failed_items', [])
            if failed_items:
                lines.append(f"  failed_items={failed_items}")
        stages = getattr(shared, "STARTUP_HEALTH", []) or []
        if stages:
            lines.append("")
            lines.append("Startup stages:")
            for item in stages:
                lines.append(f"  - {item.get('stage', '?')}: {item.get('status', 'ok')} ({item.get('details', '')})")
        try:
            from utils.gpu_runtime import build_gpu_install_plan, format_gpu_install_plan
            lines.append("")
            lines.append("GPU install plan:")
            lines.append(format_gpu_install_plan(build_gpu_install_plan(os.environ.get('BT_GPU_PROFILE', 'auto'))))
        except Exception as exc:
            lines.append(f"GPU install plan unavailable: {exc}")
        return "\n".join(lines)

    def _show_startup_health_overlay(self):
        if not getattr(pcfg, 'show_startup_health_dialog', True):
            return
        stages = getattr(shared, "STARTUP_HEALTH", []) or []
        if not stages:
            return
        labels = " \u2192 ".join([str(s.get("stage", "?")) for s in stages])
        summary = self.tr("Startup stages: {0}").format(labels)
        msg = QMessageBox(self)
        msg.setWindowTitle(self.tr("Startup health"))
        msg.setText(summary)
        msg.setDetailedText(self._collect_startup_diagnostics_text())
        copy_btn = msg.addButton(self.tr("Copy diagnostics"), QMessageBox.ButtonRole.ActionRole)
        pyqt5_btn = msg.addButton(self.tr("Relaunch PyQt5"), QMessageBox.ButtonRole.ActionRole)
        cpu_btn = msg.addButton(self.tr("Relaunch CPU-only"), QMessageBox.ButtonRole.ActionRole)
        msg.addButton(self.tr("Close"), QMessageBox.ButtonRole.AcceptRole)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == copy_btn:
            self.on_copy_startup_diagnostics()
        elif clicked == pyqt5_btn:
            self.on_relaunch_pyqt5_safe_mode()
        elif clicked == cpu_btn:
            self.on_relaunch_cpu_only_safe_mode()

    def on_copy_startup_diagnostics(self):
        text = self._collect_startup_diagnostics_text()
        app = QApplication.instance()
        if app is not None:
            app.clipboard().setText(text)
        create_info_dialog({'title': self.tr('Diagnostics copied'), 'text': self.tr('Startup diagnostics copied to clipboard.')})

    def _collect_runtime_resource_summary_text(self) -> str:
        lines = []
        run_state = self.tr("RUNNING") if self.imgtrans_thread.isRunning() else self.tr("IDLE")
        lines.append(self.tr("Pipeline state: {}").format(run_state))
        try:
            import psutil
            vm = psutil.virtual_memory()
            proc = psutil.Process(os.getpid())
            proc_mem = proc.memory_info().rss / (1024 ** 3)
            cpu_pct = psutil.cpu_percent(interval=0.1)
            lines.append(self.tr("CPU usage: {0:.1f}%").format(cpu_pct))
            lines.append(self.tr("System RAM: {0:.1f} / {1:.1f} GB ({2:.0f}%)").format(
                (vm.total - vm.available) / (1024 ** 3), vm.total / (1024 ** 3), vm.percent
            ))
            lines.append(self.tr("App RAM (RSS): {0:.2f} GB").format(proc_mem))
        except Exception as e:
            lines.append(self.tr("CPU/RAM stats unavailable: {0}").format(str(e)))

        try:
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                dev_idx = torch.cuda.current_device()
                total = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3)
                allocated = torch.cuda.memory_allocated(dev_idx) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(dev_idx) / (1024 ** 3)
                lines.append(self.tr("GPU ({0}) VRAM allocated/reserved/total: {1:.2f} / {2:.2f} / {3:.2f} GB").format(
                    torch.cuda.get_device_name(dev_idx), allocated, reserved, total
                ))
                if reserved / max(total, 1e-6) > 0.75:
                    lines.append(self.tr("Suggestion: VRAM usage is high; use low-VRAM mode, smaller models, CPU device, or quantized variants (4-bit/8-bit) when available."))
            else:
                lines.append(self.tr("GPU VRAM: CUDA not available in current runtime."))
        except Exception as e:
            lines.append(self.tr("GPU stats unavailable: {0}").format(str(e)))

        try:
            from modules.base import get_device_diagnostics_text
            lines.append("")
            lines.append(self.tr("Runtime device diagnostics:"))
            lines.append(get_device_diagnostics_text())
        except Exception as e:
            lines.append(self.tr("Runtime device diagnostics unavailable: {0}").format(str(e)))

        lines.append(self.tr("Quantization note: model quantization is module-dependent; some HF/LLM modules provide low-VRAM/4-bit/8-bit style options, but not every detector/OCR/inpainter supports quantized loading."))
        return "\n".join(lines)

    def on_runtime_resource_summary(self):
        text = self._collect_runtime_resource_summary_text()
        msg = QMessageBox(self)
        msg.setWindowTitle(self.tr("Runtime resource summary"))
        msg.setText(self.tr("Current resource usage snapshot."))
        msg.setDetailedText(text)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()

    def on_export_startup_report(self):
        text = self._collect_startup_diagnostics_text()
        default_name = f"startup_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        out_path = QFileDialog.getSaveFileName(self, self.tr('Export startup report'), default_name, self.tr('Text files (*.txt)'))[0]
        if not out_path:
            return
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(text)
            create_info_dialog({'title': self.tr('Startup report saved'), 'text': out_path})
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to export startup report.'))

    def on_open_log_folder(self):
        p = shared.LOGGING_PATH
        try:
            if sys.platform == 'win32':
                os.startfile(p)
            elif sys.platform == 'darwin':
                subprocess.run(['open', p], check=False)
            else:
                subprocess.run(['xdg-open', p], check=False)
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to open log folder.'))

    def on_show_model_download_diagnostics(self):
        last = getattr(pcfg, 'model_download_last_status', {}) or {}
        if not last:
            create_info_dialog({'title': self.tr('Model download diagnostics'), 'text': self.tr('No model download diagnostics available yet.')})
            return
        failed_items = last.get('failed_items', [])
        details = self.tr('Flow: {0}\nStatus: {1}\nDownloaded: {2}\nFailed: {3}\nSkipped: {4}\nPackage IDs: {5}').format(
            last.get('flow', 'unknown'),
            last.get('status', 'unknown'),
            last.get('downloaded', 0),
            last.get('failed', 0),
            last.get('skipped', 0),
            ', '.join(last.get('package_ids', []))
        )
        if failed_items:
            details += '\n' + self.tr('Failed items: {0}').format(', '.join(failed_items))
        create_info_dialog({'title': self.tr('Model download diagnostics'), 'text': details})

    def _relaunch_with_overrides(self, extra_args=None, env_overrides=None):
        extra_args = extra_args or []
        env_overrides = env_overrides or {}
        script_path = osp.abspath(osp.join(osp.dirname(__file__), '..', 'launch.py'))
        cmd = [sys.executable, script_path] + list(extra_args)
        env = os.environ.copy()
        env.update(env_overrides)
        subprocess.Popen(cmd, env=env, cwd=osp.dirname(script_path))
        QApplication.instance().quit()

    def on_relaunch_pyqt5_safe_mode(self):
        self._relaunch_with_overrides(extra_args=['--qt-api', 'pyqt5'])

    def on_relaunch_cpu_only_safe_mode(self):
        # Unload GPU models before switching to CPU-only to free VRAM.
        try:
            self.on_release_model_caches()
        except Exception:
            pass
        self._relaunch_with_overrides(env_overrides={'CUDA_VISIBLE_DEVICES': ''})

    def _recover_window_geometry_if_offscreen(self):
        try:
            frame = self.frameGeometry()
            if frame.width() <= 0 or frame.height() <= 0:
                return
            target = frame.adjusted(40, 40, -40, -40)
            for s in QGuiApplication.screens():
                if s.availableGeometry().intersects(target):
                    return
            primary = QGuiApplication.primaryScreen()
            if primary is None:
                return
            ag = primary.availableGeometry()
            w = min(max(frame.width(), 900), ag.width())
            h = min(max(frame.height(), 600), ag.height())
            x = ag.x() + max(0, (ag.width() - w) // 2)
            y = ag.y() + max(0, (ag.height() - h) // 2)
            self.setGeometry(x, y, w, h)
            LOGGER.warning("Recovered off-screen window geometry to primary screen.")
        except Exception as e:
            LOGGER.debug("Window geometry recovery skipped: %s", e)

    def _show_model_download_result_dialog(self, *, title: str, status: str, summary: dict, details: str = ''):
        if not getattr(pcfg, 'show_model_download_result_dialog', True):
            return
        downloaded = int((summary or {}).get('downloaded', 0))
        failed = int((summary or {}).get('failed', 0))
        skipped = int((summary or {}).get('skipped', 0))
        body = self.tr('Summary: downloaded {0}, failed {1}, skipped {2}.').format(downloaded, failed, skipped)
        if details:
            body += '\n\n' + details
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(body)
        manage_btn = msg.addButton(self.tr('Open Manage Models'), QMessageBox.ButtonRole.ActionRole)
        trouble_btn = msg.addButton(self.tr('Open Troubleshooting'), QMessageBox.ButtonRole.ActionRole)
        continue_btn = msg.addButton(self.tr('Continue with available modules'), QMessageBox.ButtonRole.AcceptRole)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == manage_btn:
            self.on_open_manage_models()
        elif clicked == trouble_btn:
            self.on_open_troubleshooting()
        elif clicked == continue_btn and status == 'success':
            self._reset_modules_to_core_defaults()

    def _reset_modules_to_core_defaults(self):
        """Set detector/OCR/inpainter/translator with core defaults when available, else fallback to available modules."""
        pcfg.module.textdetector = 'ctd'
        pcfg.module.ocr = 'mit48px'
        pcfg.module.inpainter = 'aot'
        pcfg.module.translator = 'google'
        save_config()
        self.module_manager.setTextDetector('ctd')
        self.module_manager.setOCR('mit48px')
        self.module_manager.setInpainter('aot')
        self.module_manager.setTranslator('google')

        applied = [('Detector', 'ctd'), ('OCR', 'mit48px'), ('Inpainter', 'aot'), ('Translator', 'google')]
        lines = []
        for label, key in applied:
            lines.append(f'• {label}: {key} (core default requested; runtime may fallback)')
        return '\n'.join(lines)

    def _run_deferred_model_download(self):
        """Run initial model download after window is shown (set by launch.py so app opens first)."""
        package_ids = getattr(pcfg, 'model_packages_enabled', None)
        if package_ids is not None and len(package_ids) == 0:
            LOGGER.info("Initial model download skipped (local-only mode selected).")
            log_diagnostic_event(
                "startup.offline_local_only",
                selected=True,
                package_ids=[],
            )
            return
        dlg = ModelDownloadProgressDialog(self)
        thread = ModelDownloadThread(self)
        dlg.set_thread(thread)

        def on_finished(status: str, message: str, summary: dict):
            dlg.accept()
            self._persist_model_download_status(summary, flow='deferred', status=status)
            tip = self.tr('If the connectivity check is slow or fails, set DISABLE_MODEL_SOURCE_CHECK=True and retry. See docs/TROUBLESHOOTING.md.')
            if status == 'success':
                self._show_model_download_result_dialog(
                    title=self.tr('Model package download complete'),
                    status=status,
                    summary=summary,
                )
            elif status == 'partial':
                failed_items = (summary or {}).get('failed_items', [])
                partial_details = self.tr('Some model packages failed. Failed items: {0}').format(', '.join(failed_items) if failed_items else self.tr('Unknown'))
                self._show_model_download_result_dialog(
                    title=self.tr('Model package download partially complete'),
                    status=status,
                    summary=summary,
                    details=partial_details + '\n\n' + tip
                )
            else:
                create_error_dialog(
                    Exception(message),
                    self.tr('Model download failed. You can retry from Tools → Retry model downloads.') + '\n\n' + tip,
                    'ModelDownloadThread'
                )

        thread.finished_with_result.connect(on_finished)
        thread.start()
        dlg.exec()

    def on_retry_model_downloads(self):
        """Retry downloading model packages (Tools → Retry model downloads). Useful when first-run download failed."""
        package_ids = getattr(pcfg, 'model_packages_enabled', None)
        if package_ids is not None and len(package_ids) == 0:
            log_diagnostic_event(
                "startup.offline_local_only",
                selected=True,
                retry_requested=True,
            )
            create_info_dialog({
                'title': self.tr('No model packages selected.'),
                'text': self.tr('First-run local-only mode is active. Use Tools → Manage models to import/download models, or edit config/config.json to enable packages and restart.'),
            })
            return
        dlg = ModelDownloadProgressDialog(self)
        thread = ModelDownloadThread(self)
        dlg.set_thread(thread)

        def on_finished(status: str, message: str, summary: dict):
            dlg.accept()
            self._persist_model_download_status(summary, flow='retry', status=status)
            tip = self.tr('If the connectivity check is slow or fails, set DISABLE_MODEL_SOURCE_CHECK=True and retry. See docs/TROUBLESHOOTING.md.')
            if status == 'success':
                self._show_model_download_result_dialog(
                    title=self.tr('Download complete.'),
                    status=status,
                    summary=summary,
                    details=self.tr('Model packages have been downloaded. You can use the pipeline now.')
                )
            elif status == 'partial':
                failed_items = (summary or {}).get('failed_items', [])
                partial_details = self.tr('Some model packages failed. Failed items: {0}').format(', '.join(failed_items) if failed_items else self.tr('Unknown'))
                self._show_model_download_result_dialog(
                    title=self.tr('Download partially complete.'),
                    status=status,
                    summary=summary,
                    details=partial_details + '\n\n' + tip
                )
            if status == "success":
                defaults_summary = self._reset_modules_to_core_defaults()
                create_info_dialog({
                    'title': self.tr('Download complete. Defaults updated.'),
                    'text': self.tr('Model packages have been downloaded. Applied modules:\n') + defaults_summary,
                })
            else:
                create_error_dialog(
                    Exception(message),
                    self.tr('Model download failed. You can retry from Tools → Retry model downloads.') + '\n\n' + tip,
                    'ModelDownloadThread'
                )

        thread.finished_with_result.connect(on_finished)
        thread.start()
        dlg.exec()

    def on_release_model_caches(self):
        """Unload all models and clear pipeline cache to free GPU/RAM (Tools → Release model caches)."""
        import gc
        if not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS):
            self.module_manager.unload_all_models()
            try:
                from utils.pipeline_cache import get_pipeline_cache
                cache = get_pipeline_cache(True)
                if cache is not None:
                    cache.clear()
            except Exception:
                pass
            gc.collect()
            self._reset_idle_unload_timer()

    def on_clear_pipeline_caches(self):
        """Clear in-session OCR/translation caches (Tools → Clear OCR and translation caches)."""
        try:
            from utils.pipeline_cache import get_pipeline_cache
            cache = get_pipeline_cache(True)
            if cache is not None:
                cache.clear()
        except Exception:
            pass
        try:
            from utils.pipeline_cache_manager import clear_block_level_caches
            clear_block_level_caches()
        except Exception:
            pass
        # Also clear translation strings on all text blocks so next run re-translates.
        try:
            if self.imgtrans_proj is not None:
                for page_name, blk_list in self.imgtrans_proj.pages.items():
                    for blk in blk_list:
                        blk.translation = ""
                # Update visible text items to reflect cleared translations.
                if hasattr(self, 'st_manager') and self.st_manager is not None:
                    for blkitem in self.st_manager.textblk_item_list:
                        blkitem.setPlainText("")
        except Exception:
            pass

    def run_merge_task(self, on_current=False, dry_run=False):
        """Run the region merge task."""
        from utils import merger
        from qtpy.QtWidgets import QMessageBox
        
        if self.imgtrans_proj.is_empty:
            self._msg_warning("Warning", "Please open a project first.")
            return
        
        config = self.merge_dialog.get_config()
        if bool(getattr(getattr(pcfg, "module", None), "merge_nearby_blocks_collision", False)) and mode in ("all_pages", "current_page"):
            self._show_status_message(self.tr("Detector collision-merge and Region merge are both enabled; Region merge runs as a second pass."), 6000)
        
        if on_current:
            # Run on current page — operate in memory, no file I/O
            from utils.textblock import TextBlock
            
            current_img = self.imgtrans_proj.current_img
            if not current_img:
                self._msg_warning("Warning", "No current page.")
                return

            # Get text blocks for current page from memory
            if current_img not in self.imgtrans_proj.pages:
                self._msg_warning("Warning", "Current page data not found.")
                return

            textblocks = self.imgtrans_proj.pages[current_img]
            if not textblocks:
                self._msg_warning("Info", "Current page has no text blocks.")
                return
            
            # Convert to dict format (merger expects dicts)
            initial_shapes = [blk.to_dict() for blk in textblocks]
            
            initial_count = len(initial_shapes)
            mode = config.get("MERGE_MODE", "NONE")
            total_merged = 0
            
            # Perform merge in memory
            if mode == "VERTICAL":
                final_shapes, count = merger.perform_merge(initial_shapes, "VERTICAL", config)
                total_merged += count
            elif mode == "HORIZONTAL":
                final_shapes, count = merger.perform_merge(initial_shapes, "HORIZONTAL", config)
                total_merged += count
            elif mode == "VERTICAL_THEN_HORIZONTAL":
                temp, count1 = merger.perform_merge(initial_shapes, "VERTICAL", config)
                final_shapes, count2 = merger.perform_merge(temp, "HORIZONTAL", config)
                total_merged += (count1 + count2)
            elif mode == "HORIZONTAL_THEN_VERTICAL":
                temp, count1 = merger.perform_merge(initial_shapes, "HORIZONTAL", config)
                final_shapes, count2 = merger.perform_merge(temp, "VERTICAL", config)
                total_merged += (count1 + count2)
            else:
                final_shapes = initial_shapes
            
            if total_merged > 0:
                final_count = len(final_shapes)
                if dry_run:
                    self._msg_information("Preview", f"Merge preview: {initial_count} -> {final_count} blocks (would reduce by {initial_count - final_count})")
                    return
                # Convert dicts back to TextBlock and update memory
                self.imgtrans_proj.pages[current_img] = [TextBlock(**blk_dict) for blk_dict in final_shapes]
                self.canvas.updateCanvas()
                self.st_manager.updateSceneTextitems()
                self._msg_information("Success", f"Merge done: {initial_count} -> {final_count} blocks (reduced by {initial_count - final_count})")
            else:
                raw_labels = set(s.get('label') for s in initial_shapes)
                label_list = sorted({str(lbl) for lbl in raw_labels if lbl})
                label_str = ", ".join(label_list) if label_list else "none"
                detail_msg = (
                    f"No blocks were merged.\n"
                    f"Total: {initial_count} text blocks.\n"
                    f"Label types: {label_str}\n\n"
                )
                detail_msg += "Suggestions:\n"
                detail_msg += "1. Try increasing max gap (e.g. 100–200)\n"
                detail_msg += "2. Lower min overlap ratio (e.g. 50–70%)\n"
                detail_msg += "3. Uncheck 'Enable exclude-from-merge labels'\n"
                detail_msg += "4. Check if labels are in the blacklist"
                self._msg_warning("Info", detail_msg)
        else:
            # Run on all pages
            img_list = list(self.imgtrans_proj.pages.keys())
            if not img_list:
                self._msg_warning("Warning", "Project has no images.")
                return

            json_path = self.imgtrans_proj.proj_path
            if not json_path or not osp.exists(json_path):
                self._msg_warning("Warning", f"Project JSON not found: {json_path}")
                return

            self.run_merge_all_async(json_path, img_list, config)
    
    def run_merge_all_async(self, json_path, img_list, config):
        """Run merge on all pages in a background thread."""
        from .io_thread import MergeThread
        
        # Create merge thread if needed
        if not hasattr(self, 'merge_thread'):
            self.merge_thread = MergeThread()
            self.merge_thread.progress_changed.connect(self.on_merge_progress)
            self.merge_thread.merge_finished.connect(self.on_merge_finished)
            self.merge_thread.progress_bar.stop_clicked.connect(self.on_merge_stop)
        
        # Start merge
        if self.merge_thread.runMerge(json_path, img_list, config):
            self.merge_thread.progress_bar.zero_progress()
            self.merge_thread.progress_bar.show()

    def on_merge_progress(self, current, total):
        """Update merge progress."""
        progress = int(current / total * 100)
        self.merge_thread.progress_bar.updateTaskProgress(progress, f' {current}/{total}')
    
    def on_merge_stop(self):
        """Stop merge."""
        if hasattr(self, 'merge_thread'):
            self.merge_thread.requestStop()
            self.merge_thread.progress_bar.hide()
    
    def on_merge_finished(self, success_count, fail_count):
        """Merge finished."""
        self.merge_thread.progress_bar.hide()
        
        # Reload project
        try:
            json_path = self.imgtrans_proj.proj_path
            current_img = self.imgtrans_proj.current_img
            self.imgtrans_proj.load_from_json(json_path)
            if current_img and current_img in self.imgtrans_proj.pages:
                self.imgtrans_proj.set_current_img(current_img)
                self.canvas.updateCanvas()
                self.st_manager.updateSceneTextitems()
        except Exception:
            pass
        
        # Show result
        total = success_count + fail_count
        self._msg_information("Done", f"Region merge finished\nSuccess: {success_count}/{total}\nFailed: {fail_count}/{total}")

    def on_re_run_detection_only(self):
        """Run pipeline with only detection enabled (OCR, translate, inpaint disabled)."""
        if self.imgtrans_proj.is_empty:
            return
        self._detection_only_restore = (
            pcfg.module.enable_detect,
            pcfg.module.enable_ocr,
            pcfg.module.enable_translate,
            pcfg.module.enable_inpaint,
        )
        pcfg.module.enable_detect = True
        pcfg.module.enable_ocr = False
        pcfg.module.enable_translate = False
        pcfg.module.enable_inpaint = False
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        self.on_run_imgtrans()

    def on_re_run_ocr_only(self):
        """Run pipeline with only OCR enabled (re-recognize text, keep detection boxes)."""
        if self.imgtrans_proj.is_empty:
            return
        self._detection_only_restore = (
            pcfg.module.enable_detect,
            pcfg.module.enable_ocr,
            pcfg.module.enable_translate,
            pcfg.module.enable_inpaint,
        )
        pcfg.module.enable_detect = False
        pcfg.module.enable_ocr = True
        pcfg.module.enable_translate = False
        pcfg.module.enable_inpaint = False
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        self.on_run_imgtrans()

    def _on_idle_unload_timeout(self):
        if getattr(pcfg, 'unload_after_idle_minutes', 0) <= 0:
            return
        self.module_manager.unload_all_models()
        LOGGER.info('Models unloaded after idle timeout.')

    def _reset_idle_unload_timer(self):
        self._idle_unload_timer.stop()
        mins = getattr(pcfg, 'unload_after_idle_minutes', 0)
        if mins > 0:
            self._idle_unload_timer.start(mins * 60 * 1000)

    def _sync_stage_actions_from_config(self):
        for idx, sa in enumerate(getattr(self.titleBar, 'stageActions', []) or []):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        if hasattr(self, '_update_run_button_tooltip'):
            self._update_run_button_tooltip()

    def on_apply_workflow_preset_requested(self, preset_id: str, run_after: bool = False):
        try:
            result = apply_workflow_preset(pcfg.module, preset_id)
            self._sync_stage_actions_from_config()
            save_config()
            label = str(result.get('label') or result.get('preset_id'))
            self.pipelineInsightsPanel.add_event('PRESET', self.tr('Workflow preset applied: {0}').format(label))
            if run_after:
                self.on_run_imgtrans()
        except Exception as e:
            self.pipelineInsightsPanel.add_warning('PRESET', self.tr('Failed to apply workflow preset: {0}').format(e))
            create_error_dialog(e, self.tr('Workflow preset failed'))

    def on_run_preset_full(self):
        self.on_apply_workflow_preset_requested('full')

    def on_run_preset_detect_ocr(self):
        self.on_apply_workflow_preset_requested('detect_ocr')

    def on_run_preset_translate(self):
        self.on_apply_workflow_preset_requested('translate')

    def on_run_preset_inpaint(self):
        self.on_apply_workflow_preset_requested('inpaint')

    def on_video_translator(self):
        """Open the Video translator dialog (Pipeline → Video translator...)."""
        from .video_translator_dialog import VideoTranslatorDialog
        # Keep a single instance so closing/hiding doesn't lose a running job's UI.
        dlg = getattr(self, "_video_translator_dlg", None)
        if dlg is None:
            dlg = VideoTranslatorDialog(self)
            self._video_translator_dlg = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def on_subtitle_file_translator(self):
        """Open subtitle file translator (Pipeline → Translate subtitle file…)."""
        from .subtitle_file_translator_dialog import SubtitleFileTranslatorDialog
        dlg = getattr(self, "_subtitle_file_translator_dlg", None)
        if dlg is None:
            dlg = SubtitleFileTranslatorDialog(self)
            self._subtitle_file_translator_dlg = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def on_video_subtitle_editor(self):
        """Open the Video Subtitle Editor (Pipeline → Video Subtitle Editor...)."""
        from .video_subtitle_editor import VideoSubtitleEditorWindow
        win = VideoSubtitleEditorWindow(self)
        win.show()

    def on_batch_export(self):
        """Export all pages as images (and optionally PDF) to a chosen folder."""
        if self.imgtrans_proj.is_empty:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project first.'))
            return
        out_dir = QFileDialog.getExistingDirectory(self, self.tr('Select output folder'))
        if not out_dir:
            return
        self._do_batch_export(out_dir, ext=None)

    def on_batch_export_as(self):
        """Export all pages with chosen format (#126 Export as)."""
        if self.imgtrans_proj.is_empty:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project first.'))
            return
        initial = self.imgtrans_proj.directory if self.imgtrans_proj.directory else None
        dlg = ExportFormatDialog(self, initial_dir=initial)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        export_as_zip = dlg.get_export_as_zip()
        archive_format = dlg.get_archive_format()
        zip_path = (dlg.get_zip_path() or '').strip()
        if export_as_zip and not zip_path:
            QMessageBox.information(self, self.tr('Export'), self.tr('Choose a path for the archive file.'))
            return
        out_dir = dlg.get_folder()
        if not export_as_zip and not out_dir:
            QMessageBox.information(self, self.tr('Export'), self.tr('Select an output folder.'))
            return
        temp_export_dir = None
        if export_as_zip:
            import tempfile
            temp_export_dir = tempfile.mkdtemp(prefix='ballons_export_')
            out_dir = temp_export_dir
        try:
            self._do_batch_export(
                out_dir,
                ext=dlg.get_extension(),
                also_pdf=dlg.get_also_pdf(),
                show_message=not export_as_zip,
                clean_after_export=dlg.get_clean_after_export() if not export_as_zip else False,
                include_intermediate=dlg.get_include_intermediate(),
                include_unrendered=dlg.get_include_unrendered(),
                filename_template=dlg.get_filename_template(),
            )
            if export_as_zip and zip_path and osp.isdir(out_dir):
                import zipfile
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for root, _dirs, files in os.walk(out_dir):
                        for f in files:
                            path = osp.join(root, f)
                            zf.write(path, osp.relpath(path, out_dir))
                msg = self.tr('Exported as {0}: {1}').format('CBZ' if archive_format == 'cbz' else 'ZIP', zip_path)
                if dlg.get_clean_after_export():
                    self.imgtrans_proj.clean_mask_and_inpainted_cache()
                    msg += '\n' + self.tr('Cache cleaned.')
                QMessageBox.information(self, self.tr('Export'), msg)
                self._open_export_folder_if_requested(zip_path)
        finally:
            if temp_export_dir and osp.isdir(temp_export_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_export_dir, ignore_errors=True)
                except Exception:
                    pass

    def _open_export_folder_if_requested(self, path: str):
        if not getattr(pcfg, 'export_open_folder_after_batch', False):
            return
        try:
            target = path if osp.isdir(path) else osp.dirname(path)
            if not target:
                return
            if sys.platform.startswith('win'):
                os.startfile(target)  # type: ignore[attr-defined]
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', target])
            else:
                subprocess.Popen(['xdg-open', target])
        except Exception as e:
            LOGGER.warning('Could not open export folder %s: %s', path, e)

    def _do_batch_export(self, out_dir: str, ext: str = None, also_pdf: bool = False, show_message: bool = True, clean_after_export: bool = False, include_intermediate: bool = False, include_unrendered: bool = False, filename_template: str = None):
        from utils.io_utils import imread, imwrite
        from utils.export_naming import render_export_filename
        try:
            os.makedirs(out_dir, exist_ok=True)
            result_dir = self.imgtrans_proj.result_dir()
            exported = 0
            missing = []
            exported_paths = []  # (pagename, path) in page order for PDF
            rendered_exported_paths = []
            helper_paths = []
            fallback_paths = []
            export_sources = {}
            used_export_names = set()
            page_order = list(self.imgtrans_proj.pages.keys())
            for i, pagename in enumerate(page_order):
                result_path = self.imgtrans_proj.get_result_path(pagename)
                source_path = result_path if osp.exists(result_path) else ''
                source_kind = 'rendered'
                if not source_path and include_unrendered:
                    try:
                        clean_path = self.imgtrans_proj.get_inpainted_path(pagename, get_last_modified=True)
                    except TypeError:
                        clean_path = self.imgtrans_proj.get_inpainted_path(pagename)
                    original_path = osp.join(self.imgtrans_proj.directory, pagename)
                    if clean_path and osp.exists(clean_path):
                        source_path = clean_path
                        source_kind = 'clean_fallback'
                    elif osp.exists(original_path):
                        source_path = original_path
                        source_kind = 'original_fallback'
                if source_path and osp.exists(source_path):
                    use_ext = ext if ext else osp.splitext(source_path)[1]
                    if use_ext not in ('.png', '.jpg', '.jpeg', '.webp', '.jxl'):
                        use_ext = '.png'
                    fname = render_export_filename(filename_template or getattr(pcfg, 'export_filename_template', '{index:03d}'), pagename, i + 1, use_ext, source_kind)
                    base_name, base_ext = osp.splitext(fname)
                    dedupe = 2
                    while fname.lower() in used_export_names:
                        fname = f"{base_name}_{dedupe}{base_ext}"
                        dedupe += 1
                    used_export_names.add(fname.lower())
                    dest = osp.join(out_dir, fname)
                    img = imread(source_path)
                    kw = {'ext': use_ext, 'quality': pcfg.imgsave_quality}
                    if use_ext == '.webp' and getattr(pcfg, 'imgsave_webp_lossless', False):
                        kw['webp_lossless'] = True
                    imwrite(dest, img, **kw)
                    exported += 1
                    exported_paths.append((pagename, dest))
                    export_sources[pagename] = source_kind
                    if source_kind == 'rendered':
                        rendered_exported_paths.append((pagename, dest))
                    else:
                        fallback_paths.append((pagename, source_kind, dest))
                    if include_intermediate:
                        import shutil
                        for kind, getter, subdir in [
                            ('clean', self.imgtrans_proj.get_inpainted_path, 'clean'),
                            ('mask', self.imgtrans_proj.get_mask_path, 'masks'),
                        ]:
                            try:
                                src = getter(pagename, get_last_modified=True)
                            except TypeError:
                                src = getter(pagename)
                            if src and osp.exists(src):
                                helper_dir = osp.join(out_dir, subdir)
                                os.makedirs(helper_dir, exist_ok=True)
                                helper_dest = osp.join(helper_dir, f"{i + 1:03d}{osp.splitext(src)[1] or use_ext}")
                                shutil.copy2(src, helper_dest)
                                helper_paths.append((pagename, kind, helper_dest))
                else:
                    missing.append(pagename)
            msg = self.tr('Exported {0} page(s) to {1}.').format(exported, out_dir)
            if also_pdf and exported_paths:
                pdf_path = osp.join(out_dir, 'exported.pdf')
                try:
                    import img2pdf
                    with open(pdf_path, 'wb') as f:
                        # img2pdf expects image paths; order by page order
                        img_paths = [p for _, p in exported_paths]
                        f.write(img2pdf.convert(img_paths))
                    msg += '\n' + self.tr('PDF saved: {}').format(pdf_path)
                except ImportError:
                    msg += '\n' + self.tr('PDF export skipped. Install img2pdf: pip install img2pdf')
                except Exception as e:
                    LOGGER.exception(e)
                    msg += '\n' + self.tr('PDF export failed: {}').format(str(e))
            from utils.export_manifest import mark_exported_pages, write_export_manifest
            marked_exported = mark_exported_pages(self.imgtrans_proj, rendered_exported_paths)
            manifest = write_export_manifest(
                self.imgtrans_proj,
                out_dir,
                exported_paths,
                missing,
                export_kind='rendered_images',
                options={'ext': ext or '', 'also_pdf': bool(also_pdf), 'clean_after_export': bool(clean_after_export), 'include_intermediate': bool(include_intermediate), 'include_unrendered': bool(include_unrendered), 'filename_template': filename_template or getattr(pcfg, 'export_filename_template', '{index:03d}'), 'helper_paths': helper_paths, 'fallback_paths': fallback_paths, 'export_sources': export_sources, 'renderer': collect_renderer_diagnostics()},
            )
            msg += '\n' + self.tr('Export manifest: {0}').format(manifest.get('manifest_path', ''))
            if marked_exported:
                msg += '\n' + self.tr('Marked {0} page(s) as Exported.').format(marked_exported)
                try:
                    self.imgtrans_proj.save()
                    self._update_page_list_state_style()
                except Exception as e:
                    LOGGER.warning('Failed to save exported page completion state: %s', e)
            if helper_paths:
                msg += '\n' + self.tr('Included {0} clean/mask helper image(s).').format(len(helper_paths))
            if fallback_paths:
                msg += '\n' + self.tr('Included {0} unrendered page fallback(s).').format(len(fallback_paths))
            if missing:
                msg += '\n' + self.tr('Missing result/source for {0} page(s). Run pipeline first or restore original files.').format(len(missing))
            if clean_after_export and self.imgtrans_proj is not None:
                self.imgtrans_proj.clean_mask_and_inpainted_cache()
                msg += '\n' + self.tr('Cache cleaned.')
            if show_message:
                QMessageBox.information(self, self.tr('Export'), msg)
                self._open_export_folder_if_requested(out_dir)
            return {'exported': exported, 'missing': missing, 'paths': exported_paths, 'rendered_paths': rendered_exported_paths, 'helper_paths': helper_paths, 'fallback_paths': fallback_paths, 'manifest': manifest, 'marked_exported': marked_exported}
        except Exception as e:
            LOGGER.exception(e)
            create_error_dialog(e, self.tr('Batch export failed'), 'BatchExport')
            return {'exported': 0, 'missing': [], 'paths': [], 'helper_paths': [], 'error': str(e)}

    def on_validate_project(self):
        """Check project: missing images, invalid JSON, duplicate/overlapping blocks."""
        if self.imgtrans_proj.is_empty:
            QMessageBox.information(self, self.tr('Check project'), self.tr('Open a project first.'))
            return
        report = []
        proj_dir = self.imgtrans_proj.directory
        json_path = self.imgtrans_proj.proj_path
        if not osp.exists(json_path):
            report.append(self.tr('Project JSON not found: {}').format(json_path))
        else:
            try:
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                pages = data.get('pages') or data.get('image_info') or {}
                if not pages:
                    report.append(self.tr('No pages in project JSON.'))
            except Exception as e:
                report.append(self.tr('Invalid project JSON: {}').format(str(e)))
        for pagename in list(self.imgtrans_proj.pages.keys()):
            img_path = osp.join(proj_dir, pagename)
            if not osp.exists(img_path):
                report.append(self.tr('Missing image: {}').format(pagename))

        # Duplicate/overlapping block check
        from utils.imgproc_utils import union_area
        overlap_report = []
        for pagename in list(self.imgtrans_proj.pages.keys()):
            blks = self.imgtrans_proj.pages.get(pagename) or []
            n = len(blks)
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    blk_i = blks[i]
                    blk_j = blks[j]
                    xyxy_i = getattr(blk_i, 'xyxy', None) if not isinstance(blk_i, dict) else blk_i.get('xyxy')
                    xyxy_j = getattr(blk_j, 'xyxy', None) if not isinstance(blk_j, dict) else blk_j.get('xyxy')
                    if xyxy_i is None and isinstance(blk_i, (list, tuple)) and len(blk_i) >= 4:
                        xyxy_i = list(blk_i)[:4]
                    if xyxy_j is None and isinstance(blk_j, (list, tuple)) and len(blk_j) >= 4:
                        xyxy_j = list(blk_j)[:4]
                    if xyxy_i and xyxy_j and len(xyxy_i) >= 4 and len(xyxy_j) >= 4:
                        inter = union_area(list(xyxy_i)[:4], list(xyxy_j)[:4])
                        if inter > 0:
                            pairs.append((i, j))
            if pairs:
                overlap_report.append(self.tr('Page "{}": {} overlapping block pair(s): {}').format(
                    pagename, len(pairs), ', '.join(f'({a},{b})' for a, b in pairs[:10]) + (' ...' if len(pairs) > 10 else '')
                ))
        if overlap_report:
            report.append('')
            report.append(self.tr('Duplicate/overlapping blocks:'))
            report.extend(overlap_report)

        if not report:
            report.append(self.tr('No issues found.'))
        msg = '\n'.join(report)
        dlg = QMessageBox(self)
        dlg.setWindowTitle(self.tr('Check project'))
        dlg.setText(msg)
        dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
        dlg.exec()


    def on_req_update_pagetext(self):
        if self.canvas.text_change_unsaved():
            self.st_manager.updateTextBlkList()

    def on_req_move_page(self, page_name: str, force_save=False):
        ori_save = self.save_on_page_changed
        self.save_on_page_changed = False
        current_img = self.imgtrans_proj.current_img
        if current_img == page_name and not force_save:
            return
        if current_img not in self.global_search_widget.page_set:
            if self.canvas.projstate_unsaved: 
                self.saveCurrentPage()
        else:
            self.saveCurrentPage(save_rst_only=True)
        self.pageList.setCurrentRow(self.imgtrans_proj.pagename2idx(page_name))
        self.save_on_page_changed = ori_save

    def on_search_result_item_clicked(self, pagename: str, blk_idx: int, is_src: bool, start: int, end: int):
        idx = self.imgtrans_proj.pagename2idx(pagename)
        self.pageList.setCurrentRow(idx)
        pw = self.st_manager.pairwidget_list[blk_idx]
        edit = pw.e_source if is_src else pw.e_trans
        edit.setFocus()
        edit.ensure_scene_visible.emit()
        cursor = QTextCursor(edit.document())
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        edit.setTextCursor(cursor)

    def shortcutEscape(self):
        if self.canvas.search_widget.isVisible():
            self.canvas.search_widget.hide()
        elif self.canvas.editing_textblkitem is not None and self.canvas.editing_textblkitem.isEditing():
            self.canvas.editing_textblkitem.endEdit()
        elif self.canvas.image_edit_mode == ImageEditMode.TextEraserTool:
            self.drawingPanel.setCurrentToolByName('hand')
            if hasattr(self.canvas, '_text_eraser_selected_blocks'):
                self.canvas._text_eraser_selected_blocks = None
        elif self.canvas.drawMode() and self.canvas.cancel_rect_selection():
            pass  # rect selection cancelled (#126)

    def setPaintMode(self):
        if self.bottomBar.paintChecker.isChecked():
            if self.rightComicTransStackPanel.isHidden():
                self.rightComicTransStackPanel.show()
            self.rightComicTransStackPanel.setCurrentIndex(0)
            self.canvas.setPaintMode(True)
            self.bottomBar.originalSlider.show()
            self.bottomBar.textlayerSlider.show()
            self.bottomBar.textblockChecker.hide()
        else:
            self.canvas.setPaintMode(False)
            self.rightComicTransStackPanel.setHidden(True)
        self.st_manager.setTextEditMode(False)

    def setTextEditMode(self):
        if self.bottomBar.texteditChecker.isChecked():
            if self.rightComicTransStackPanel.isHidden():
                self.rightComicTransStackPanel.show()
            self.bottomBar.textblockChecker.show()
            self.rightComicTransStackPanel.setCurrentIndex(1)
            self.st_manager.setTextEditMode(True)
            self.setTextBlockMode()
        else:
            self.bottomBar.textblockChecker.hide()
            self.rightComicTransStackPanel.setHidden(True)
            self.st_manager.setTextEditMode(False)
        self.canvas.setPaintMode(False)

    def setTextBlockMode(self):
        mode = self.bottomBar.textblockChecker.isChecked()
        self.canvas.setTextBlockMode(mode)
        pcfg.imgtrans_textblock = mode
        self.st_manager.showTextblkItemRect(mode)

    def manual_save(self):
        if self.leftBar.imgTransChecker.isChecked()\
            and self.imgtrans_proj.directory is not None:
            LOGGER.debug('Manually saving...')
            self.saveCurrentPage(update_scene_text=True, save_proj=True, restore_interface=True, save_rst_only=False)

    def close_project_and_show_welcome(self):
        """Close current project and switch to welcome screen. Prompts to save if unsaved."""
        if self.imgtrans_proj.is_empty:
            self._show_welcome_screen()
            return
        if self.canvas.projstate_unsaved:
            ret = QMessageBox.question(
                self,
                self.tr("Unsaved changes"),
                self.tr("Save project before closing?"),
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if ret == QMessageBox.StandardButton.Cancel:
                return
            if ret == QMessageBox.StandardButton.Save:
                self.saveCurrentPage(update_scene_text=True, save_proj=True, restore_interface=True, save_rst_only=False)
        self.imgtrans_proj.pages = {}
        self.imgtrans_proj._pagename2idx = {}
        self.imgtrans_proj._idx2pagename = {}
        self.imgtrans_proj.current_img = None
        self.imgtrans_proj.directory = None
        self.imgtrans_proj.proj_path = None
        self.imgtrans_proj.img_array = None
        self.imgtrans_proj.mask_array = None
        self.imgtrans_proj.inpainted_array = None
        self.st_manager.clearSceneTextitems()
        self.pageList.clear()
        self.titleBar.setTitleContent(proj_name="", page_name="")
        self.canvas.clear_undostack(update_saved_step=True)
        self._show_welcome_screen()

    def saveCurrentPage(self, update_scene_text=True, save_proj=True, restore_interface=False, save_rst_only=False, keep_exist_as_backup=False):
        
        if not self.imgtrans_proj.img_valid:
            return
        
        if restore_interface:
            set_canvas_focus = self.canvas.hasFocus()
            sel_textitem = self.canvas.selected_text_items()
            n_sel_textitems = len(sel_textitem)
            editing_textitem = None
            if n_sel_textitems == 1 and sel_textitem[0].isEditing():
                editing_textitem = sel_textitem[0]
        
        if update_scene_text:
            self.st_manager.updateTextBlkList()
        
        if self.rightComicTransStackPanel.isHidden():
            self.bottomBar.texteditChecker.click()

        restore_textblock_mode = False
        if pcfg.imgtrans_textblock:
            restore_textblock_mode = True
            self.bottomBar.textblockChecker.click()

        hide_tsc = False
        if self.st_manager.txtblkShapeControl.isVisible():
            hide_tsc = True
            self.st_manager.txtblkShapeControl.hide()

        if not osp.exists(self.imgtrans_proj.result_dir()):
            os.makedirs(self.imgtrans_proj.result_dir())

        if save_proj:
            try:
                self.imgtrans_proj.save(keep_exist_as_backup=keep_exist_as_backup)
                if not save_rst_only:
                    mask_path = self.imgtrans_proj.get_mask_path()
                    mask_array = self.imgtrans_proj.mask_array
                    if mask_array is not None:
                        self.imsave_thread.saveImg(mask_path, mask_array, save_params={'ext': pcfg.intermediate_imgsave_ext})
                    inpainted_path = self.imgtrans_proj.get_inpainted_path()
                    if self.canvas.drawingLayer.drawed():
                        inpainted = self.canvas.base_pixmap.copy()
                        painter = QPainter(inpainted)
                        painter.drawPixmap(0, 0, self.canvas.drawingLayer.get_drawed_pixmap())
                        painter.end()
                    else:
                        inpainted = self.imgtrans_proj.inpainted_array
                    if inpainted is not None:
                        self.imsave_thread.saveImg(inpainted_path, inpainted, save_params={'ext': pcfg.intermediate_imgsave_ext}, keep_alpha=self.imgtrans_proj.current_has_alpha())
            except Exception as e:
                LOGGER.error(f"Failed to save project files: {e}")

        # Render the final result image properly
        try:
            img = self.canvas.render_result_img()
            # Optional final upscale (Section 6: 2x output for nicer export)
            if getattr(pcfg.module, 'image_upscale_final', False):
                try:
                    factor = float(getattr(pcfg.module, 'image_upscale_final_factor', 2.0) or 2.0)
                    policy = (getattr(pcfg.module, 'upscale_policy_final', None) or 'lanczos').strip().lower()
                    if factor > 1.0 and policy != 'none':
                        from utils.image_upscale import apply_upscale_final
                        if hasattr(img, 'bits'):
                            img = pixmap2ndarray(img, keep_alpha=self.imgtrans_proj.current_has_alpha())
                        img = apply_upscale_final(img, factor=factor, policy=policy)
                except Exception as e:
                    LOGGER.warning("Final upscale failed: %s", e)
            # Optional: lightweight colorization for grayscale pages (Section 6+Colorization).
            if getattr(pcfg.module, 'enable_colorization', False):
                try:
                    if hasattr(img, 'bits'):
                        img = pixmap2ndarray(img, keep_alpha=self.imgtrans_proj.current_has_alpha())
                    strength = float(getattr(pcfg.module, 'colorization_strength', 0.6) or 0.6)
                    backend = getattr(pcfg.module, 'colorization_backend', 'simple') or 'simple'
                    img = apply_colorization(img, strength=strength, backend=backend)
                except Exception as e:
                    LOGGER.warning("Colorization failed: %s", e)
            imsave_path = self.imgtrans_proj.get_result_path(self.imgtrans_proj.current_img)
            save_params = {'ext': pcfg.imgsave_ext, 'quality': pcfg.imgsave_quality}
            if pcfg.imgsave_ext == '.webp' and getattr(pcfg, 'imgsave_webp_lossless', False):
                save_params['webp_lossless'] = True
            self.imsave_thread.saveImg(imsave_path, img, self.imgtrans_proj.current_img, save_params=save_params, keep_alpha=self.imgtrans_proj.current_has_alpha())
            self.canvas.setProjSaveState(False)
            self.canvas.update_saved_undostep()
        except Exception as e:
            LOGGER.error(f"Failed to render and save result image: {e}")

        if restore_interface:
            if restore_textblock_mode:
                self.bottomBar.textblockChecker.click()
            if hide_tsc:
                self.st_manager.txtblkShapeControl.show()
            if set_canvas_focus:
                self.canvas.setFocus()
            if n_sel_textitems > 0:
                self.canvas.block_selection_signal = True
                for blk in sel_textitem:
                    blk.setSelected(True)
                self.st_manager.on_incanvas_selection_changed()
                self.canvas.block_selection_signal = False
            if editing_textitem is not None:
                editing_textitem.startEdit()
        
    def to_trans_config(self):
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnTranslator()

    def to_inpaint_config(self):
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnInpaint()

    def to_ocr_config(self):
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnOCR()

    def to_detect_config(self):
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnDetect()

    def on_textdet_changed(self):
        module = self.bottomBar.textdet_selector.selector.currentText()
        tgt_selector = self.configPanel.detect_config_panel.module_combobox
        if tgt_selector.currentText() != module and module in GET_VALID_TEXTDETECTORS():
            tgt_selector.setCurrentText(module)

    def on_ocr_changed(self):
        module = self.bottomBar.ocr_selector.selector.currentText()
        tgt_selector = self.configPanel.ocr_config_panel.module_combobox
        if tgt_selector.currentText() != module and module in GET_VALID_OCR():
            tgt_selector.setCurrentText(module)

    def on_trans_changed(self):
        module = self.bottomBar.trans_selector.selector.currentText()
        tgt_selector = self.configPanel.trans_config_panel.module_combobox
        if tgt_selector.currentText() != module and module in GET_VALID_TRANSLATORS():
            tgt_selector.setCurrentText(module)

    def on_trans_src_changed(self):
        sender = self.sender()
        text = sender.currentText()
        key = lang_display_to_key(text)
        translator = self.module_manager.translator
        if translator is not None:
            translator.set_source(key)
        pcfg.module.translate_source = key
        self.schedule_config_save()
        try:
            idx = translator.supported_src_list.index(key) if translator else 0
        except (ValueError, AttributeError):
            idx = 0
        combobox = self.configPanel.trans_config_panel.source_combobox
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentIndex(idx)
            combobox.blockSignals(False)
        combobox = self.bottomBar.trans_selector.src_selector
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentIndex(idx)
            combobox.blockSignals(False)

    def on_trans_tgt_changed(self):
        sender = self.sender()
        text = sender.currentText()
        key = lang_display_to_key(text)
        translator = self.module_manager.translator
        if translator is not None:
            translator.set_target(key)
        pcfg.module.translate_target = key
        self.schedule_config_save()
        try:
            idx = translator.supported_tgt_list.index(key) if translator else 0
        except (ValueError, AttributeError):
            idx = 0
        combobox = self.configPanel.trans_config_panel.target_combobox
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentIndex(idx)
            combobox.blockSignals(False)
        combobox = self.bottomBar.trans_selector.tgt_selector
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentIndex(idx)
            combobox.blockSignals(False)

    def on_inpaint_changed(self):
        module = self.bottomBar.inpaint_selector.selector.currentText()
        tgt_selector = self.configPanel.inpaint_config_panel.module_combobox
        if tgt_selector.currentText() != module and module in GET_VALID_INPAINTERS():
            tgt_selector.setCurrentText(module)

    def on_transpagebtn_pressed(self, run_target: bool):
        page_key = self.imgtrans_proj.current_img
        if page_key is None:
            return

        blkitem_list = self.st_manager.textblk_item_list

        if len(blkitem_list) < 1:
            return
        
        self.translateBlkitemList(blkitem_list, -1)


    def translateBlkitemList(self, blkitem_list: List, mode: int) -> bool:

        tgt_img = self.imgtrans_proj.img_array
        if tgt_img is None:
            return False
        tgt_mask = self.imgtrans_proj.mask_array
        
        if len(blkitem_list) < 1:
            return False
        
        self.global_search_widget.set_document_edited()
        
        im_h, im_w = tgt_img.shape[:2]

        blk_list, blk_ids = [], []
        for blkitem in blkitem_list:
            blk: TextBlock = blkitem.blk
            blk._bounding_rect = blkitem.absBoundingRect()
            blk.text = self.st_manager.pairwidget_list[blkitem.idx].e_source.toPlainText()
            blk_ids.append(blkitem.idx)
            blk.set_lines_by_xywh(blk._bounding_rect, angle=-blk.angle, x_range=[0, im_w-1], y_range=[0, im_h-1], adjust_bbox=True)
            blk_list.append(blk)

        self.module_manager.runBlktransPipeline(blk_list, tgt_img, mode, blk_ids, tgt_mask = tgt_mask)
        return True


    def finishTranslatePage(self, page_key):
        if page_key == self.imgtrans_proj.current_img:
            self.st_manager.updateTranslation()

    def _auto_mark_translated_pages_after_run(self):
        if not getattr(pcfg, 'auto_mark_translated_pages', True):
            return
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            return
        pages = getattr(self, '_last_pages_to_process', None) or list(self.imgtrans_proj.pages.keys())
        changed = 0
        required_code = int(getattr(self, '_last_run_finish_code', 0) or getattr(pcfg.module, 'finish_code', 0) or 0)
        for page in pages:
            if page not in self.imgtrans_proj.pages:
                continue
            info = self.imgtrans_proj._ensure_image_info_entry(page)
            finish_code = int(info.get('finish_code', 0) or 0)
            if required_code and (finish_code & required_code) == required_code:
                if self.imgtrans_proj.get_page_completion_state(page) in ('todo', 'translated'):
                    self.imgtrans_proj.set_page_completion_state(page, 'translated')
                    changed += 1
        if changed:
            self.imgtrans_proj.save()
            self._update_page_list_state_style()
            self.pipelineInsightsPanel.add_event('PAGE_STATE', self.tr('Auto-marked {0} page(s) as translated.').format(changed))

    def on_imgtrans_pipeline_finished(self):
        if getattr(self, '_detection_only_restore', None) is not None:
            det, ocr, trans, inp = self._detection_only_restore
            pcfg.module.enable_detect = det
            pcfg.module.enable_ocr = ocr
            pcfg.module.enable_translate = trans
            pcfg.module.enable_inpaint = inp
            for idx, sa in enumerate(self.titleBar.stageActions):
                sa.setChecked(pcfg.module.stage_enabled(idx))
            self._detection_only_restore = None
        if getattr(self, '_run_stages_restore', None) is not None:
            det, ocr, trans, inp = self._run_stages_restore
            pcfg.module.enable_detect = det
            pcfg.module.enable_ocr = ocr
            pcfg.module.enable_translate = trans
            pcfg.module.enable_inpaint = inp
            for idx, sa in enumerate(self.titleBar.stageActions):
                sa.setChecked(pcfg.module.stage_enabled(idx))
            self._run_stages_restore = None
        self.backup_blkstyles.clear()
        self._run_imgtrans_wo_textstyle_update = False
        self.postprocess_mt_toggle = True
        self._auto_mark_translated_pages_after_run()
        if pcfg.module.empty_runcache and not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS):
            self.module_manager.unload_all_models()
            try:
                from utils.pipeline_cache import get_pipeline_cache
                cache = get_pipeline_cache(True)
                if cache is not None:
                    cache.clear()
            except Exception:
                pass
        elif getattr(pcfg, 'release_caches_after_batch', False) and not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS):
            import gc
            self.module_manager.unload_all_models()
            try:
                from utils.pipeline_cache import get_pipeline_cache
                cache = get_pipeline_cache(True)
                if cache is not None:
                    cache.clear()
            except Exception:
                pass
            gc.collect()
        if not (pcfg.module.empty_runcache or getattr(pcfg, 'release_caches_after_batch', False)):
            self._reset_idle_unload_timer()
        if shared.args.export_translation_txt:
            self.on_export_txt('translation')
        if shared.args.export_source_txt:
            self.on_export_txt('source')
        if shared.HEADLESS or shared.HEADLESS_CONTINUOUS or getattr(self, 'exec_dirs', None) is not None:
            # If this project dir was extracted from a ZIP, copy results out before moving on.
            try:
                if self._current_batch_dir:
                    handled, job, copied = self._zip_batch.on_project_finished(self._current_batch_dir)
                    if handled and job is not None:
                        LOGGER.info(
                            f'ZIP batch: copied {copied} file(s) for {self._current_batch_dir} -> {job.output_root}'
                        )
                        finished_jobs = self._zip_batch.finalize_finished_jobs()
                        for fj in finished_jobs:
                            LOGGER.info(
                                f'ZIP batch finished: {fj.zip_path} -> {fj.output_root}'
                            )
            except Exception as e:
                LOGGER.warning(f'ZIP batch output copy failed: {e}')
            self.run_next_dir()
            return
        self.maybe_auto_run_region_merge()
        if not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS):
            report = self.module_manager.get_last_batch_report()
            self._last_batch_report = report
            if report and report.get("skipped") and len(report["skipped"]) > 0:
                self._show_batch_report_dialog()
            self.titleBar.show_batch_report_action.setEnabled(
                bool(self._last_batch_report and len(self._last_batch_report.get("skipped", {})) > 0)
            )

    def _show_batch_report_dialog(self):
        """Show the batch report dialog with last report (if any)."""
        if not hasattr(self, '_batch_report_dialog') or self._batch_report_dialog is None:
            from .batch_report_dialog import BatchReportDialog
            self._batch_report_dialog = BatchReportDialog(self)
            self._batch_report_dialog.page_activated.connect(self._go_to_page_from_report)
        self._batch_report_dialog.set_report(self._last_batch_report)
        self._batch_report_dialog.show()
        self._batch_report_dialog.raise_()
        self._batch_report_dialog.activateWindow()

    def _go_to_page_from_report(self, page_key: str):
        """Switch to the given page (from batch report double-click)."""
        if self.imgtrans_proj.is_empty or page_key not in self.imgtrans_proj.pages:
            return
        self.imgtrans_proj.set_current_img(page_key)
        idx = self.imgtrans_proj.pagename2idx(page_key)
        if idx >= 0 and self.pageList.count() > idx:
            self.page_changing = True
            self.pageList.setCurrentRow(idx)
            self.page_changing = False
        self.canvas.clear_undostack(update_saved_step=True)
        self.canvas.updateCanvas()
        self.st_manager.updateSceneTextitems()
        self.titleBar.setTitleContent(page_name=self.imgtrans_proj.current_img)
        self.module_manager.handle_page_changed()
        self.drawingPanel.handle_page_changed()

    def maybe_auto_run_region_merge(self):
        """If configured, run Region merge after pipeline (all pages or current page)."""
        mode = getattr(pcfg, 'auto_region_merge_after_run', 'never')
        if mode == 'never' or self.imgtrans_proj.is_empty:
            return
        if not hasattr(self, 'merge_dialog') or self.merge_dialog is None:
            from .merge_dialog import MergeDialog
            self.merge_dialog = MergeDialog(self)
            self.merge_dialog.preview_current_clicked.connect(lambda: self.run_merge_task(on_current=True, dry_run=True))
            self.merge_dialog.run_current_clicked.connect(lambda: self.run_merge_task(on_current=True, dry_run=False))
            self.merge_dialog.run_all_clicked.connect(lambda: self.run_merge_task(on_current=False, dry_run=False))
        config = self.merge_dialog.get_config()
        if bool(getattr(getattr(pcfg, "module", None), "merge_nearby_blocks_collision", False)) and mode in ("all_pages", "current_page"):
            self._show_status_message(self.tr("Detector collision-merge and Region merge are both enabled; Region merge runs as a second pass."), 6000)
        if mode == 'all_pages':
            img_list = list(self.imgtrans_proj.pages.keys())
            if not img_list:
                return
            json_path = self.imgtrans_proj.proj_path
            if not json_path or not osp.exists(json_path):
                return
            self.run_merge_all_async(json_path, img_list, config)
        elif mode == 'current_page':
            self.run_merge_task(on_current=True)

    def postprocess_translations(self, blk_list: List[TextBlock]) -> None:
        src_is_cjk = is_cjk(pcfg.module.translate_source)
        tgt_is_cjk = is_cjk(pcfg.module.translate_target)
        if tgt_is_cjk:
            for blk in blk_list:
                if src_is_cjk:
                    blk.translation = full_len(blk.translation)
                else:
                    blk.translation = half_len(blk.translation)
                    blk.translation = re.sub(r'([?.!"])\s+', r'\1', blk.translation)    # remove spaces following punctuations
        else:
            for blk in blk_list:
                if blk.vertical:
                    blk.alignment = TextAlignment.Center
                blk.translation = half_len(blk.translation)
                blk.vertical = False

        for blk in blk_list:
            blk.translation = self.mtSubWidget.sub_text(blk.translation)
            if pcfg.let_uppercase_flag:
                blk.translation = locale_aware_upper(blk.translation, getattr(pcfg.module, 'translate_target', ''))

    def on_pagtrans_finished(self, page_index: int):
        blk_list = self.imgtrans_proj.get_blklist_byidx(page_index)
        detected_block_indices = [ii for ii, blk in enumerate(blk_list) if getattr(blk, 'font_size', -1) < 0]
        ffmt_list = None
        if len(self.backup_blkstyles) == self.imgtrans_proj.num_pages and len(self.backup_blkstyles[page_index]) == len(blk_list):
            ffmt_list: List[FontFormat] = self.backup_blkstyles[page_index]

        self.postprocess_translations(blk_list)
                
        # override font format if necessary
        override_fnt_size = pcfg.let_fntsize_flag == 1
        override_fnt_stroke = pcfg.let_fntstroke_flag == 1
        override_fnt_color = pcfg.let_fntcolor_flag == 1
        override_fnt_scolor = pcfg.let_fnt_scolor_flag == 1
        override_alignment = pcfg.let_alignment_flag == 1
        override_effect = pcfg.let_fnteffect_flag == 1
        override_writing_mode = pcfg.let_writing_mode_flag == 1
        override_font_family = pcfg.let_family_flag == 1
        gf = self.textPanel.formatpanel.global_format

        inpaint_only = pcfg.module.enable_inpaint
        inpaint_only = inpaint_only and not (pcfg.module.enable_detect or pcfg.module.enable_ocr or pcfg.module.enable_translate)
        
        if not inpaint_only:
            for ii, blk in enumerate(blk_list):
                if self._run_imgtrans_wo_textstyle_update and ffmt_list is not None:
                    blk.fontformat.merge(ffmt_list[ii])
                else:
                    if override_fnt_size or \
                        blk.font_size < 0:  # fall back to global font size if font size is not valid, it will be set to -1 for detected blocks
                        blk.font_size = gf.font_size
                    elif blk._detected_font_size > 0 and not pcfg.module.enable_detect:
                        blk.font_size = blk._detected_font_size
                    if override_fnt_stroke:
                        blk.stroke_width = gf.stroke_width
                    elif pcfg.module.enable_ocr:
                        blk.recalulate_stroke_width()
                    if override_fnt_color:
                        blk.set_font_colors(fg_colors=gf.frgb)
                    if override_fnt_scolor:
                        blk.set_font_colors(bg_colors=gf.srgb)
                    if override_alignment:
                        blk.alignment = gf.alignment
                    elif pcfg.module.enable_detect and not blk.src_is_vertical:
                        blk.recalulate_alignment()
                    if override_effect:
                        blk.opacity = gf.opacity
                        blk.shadow_color = gf.shadow_color
                        blk.shadow_radius = gf.shadow_radius
                        blk.shadow_strength = gf.shadow_strength
                        blk.shadow_offset = gf.shadow_offset
                    if override_writing_mode:
                        blk.vertical = gf.vertical
                    if override_font_family or blk.font_family is None:
                        blk.font_family = gf.font_family
                        if blk.rich_text:
                            blk.rich_text = set_html_family(blk.rich_text, gf.font_family)
                    
                    blk.line_spacing = gf.line_spacing
                    blk.letter_spacing = gf.letter_spacing
                    blk.italic = gf.italic
                    blk.bold = gf.bold
                    blk.underline = gf.underline
                    blk.strikethrough = getattr(gf, 'strikethrough', False)
                    sw = blk.stroke_width
                    if sw > 0 and pcfg.module.enable_ocr and pcfg.module.enable_detect and not override_fnt_size:
                        blk.font_size = blk.font_size / (1 + sw)

            self.st_manager.auto_textlayout_flag = should_enable_auto_textlayout(
                pcfg.let_autolayout_flag,
                pcfg.let_fntsize_flag,
                pcfg.module.enable_detect,
                pcfg.module.enable_ocr,
                pcfg.module.enable_translate,
            )
        else:
            self.st_manager.auto_textlayout_flag = False

        if page_index != self.pageList.currentIndex().row():
            self.pageList.setCurrentRow(page_index)
        else:
            self.imgtrans_proj.set_current_img_byidx(page_index)
            self.canvas.updateCanvas()
            self.st_manager.updateSceneTextitems()

        # Run auto layout (and balloon shape Auto) on all blocks so text boxes match bubble shape after Run.
        if self.st_manager.auto_textlayout_flag:
            self.st_manager.run_auto_layout_on_current_page_once()
            if should_run_post_detect_autofit(
                pcfg.let_autolayout_flag,
                pcfg.let_fntsize_flag,
                pcfg.module.enable_detect,
                pcfg.module.enable_ocr,
                pcfg.module.enable_translate,
            ):
                self.st_manager.run_detect_post_autofit_on_current_page_once(detected_block_indices)

        if not pcfg.module.enable_detect and pcfg.module.enable_translate:
            # Skip squeeze when auto layout ran; layout_textblk already set box size and squeeze would narrow/stretch boxes
            if not self.st_manager.auto_textlayout_flag:
                for blkitem in self.st_manager.textblk_item_list:
                    blkitem.squeezeBoundingRect()

        if page_index + 1 == self.imgtrans_proj.num_pages:
            self.st_manager.auto_textlayout_flag = False

        # save proj file on page trans finished
        self.imgtrans_proj.save()

        self.saveCurrentPage(False, False)

    def on_savestate_changed(self, unsaved: bool):
        save_state = self.tr('unsaved') if unsaved else self.tr('saved')
        self.titleBar.setTitleContent(save_state=save_state)

    def on_textstack_changed(self):
        if not self.page_changing:
            self.global_search_widget.set_document_edited()

    def on_run_blktrans(self, mode: int):
        blkitem_list = self.canvas.selected_text_items()
        self.translateBlkitemList(blkitem_list, mode)

    def on_blktrans_finished(self, mode: int, blk_ids: List[int]):

        if len(blk_ids) < 1:
            return
        
        blkitem_list = [self.st_manager.textblk_item_list[idx] for idx in blk_ids]

        pairw_list = []
        for blk in blkitem_list:
            pairw_list.append(self.st_manager.pairwidget_list[blk.idx])
        self.canvas.push_undo_command(RunBlkTransCommand(self.canvas, blkitem_list, pairw_list, mode))

    def on_imgtrans_progressbox_showed(self):
        msg_size = self.module_manager.progress_msgbox.size()
        size = self.size()
        p = self.mapToGlobal(QPoint(size.width() - msg_size.width(),
                                    size.height() - msg_size.height()))
        self.module_manager.progress_msgbox.move(p)

    def on_closebtn_clicked(self):
        if self.imsave_thread.isRunning():
            self.imsave_thread.finished.connect(self.close)
            mb = FrameLessMessageBox()
            mb.setText(self.tr('Saving image...'))
            self.imsave_thread.finished.connect(mb.close)
            mb.exec()
            return
        self.close()

    def on_display_lang_changed(self, lang: str):
        if lang != pcfg.display_lang:
            pcfg.display_lang = lang
            self.set_display_lang(lang)

    def on_dev_mode_changed(self):
        """Refresh detector/OCR/translator dropdowns after dev_mode toggle."""
        # Update realtime translator menu item visibility
        if hasattr(self.titleBar, 'realtime_translator_action'):
            self.titleBar.realtime_translator_action.setVisible(pcfg.dev_mode)
        valid_det = GET_VALID_TEXTDETECTORS()
        valid_ocr = GET_VALID_OCR()
        valid_trans = GET_VALID_TRANSLATORS()
        self.bottomBar.textdet_selector.selector.clear()
        self.bottomBar.textdet_selector.selector.addItems(valid_det)
        self.bottomBar.textdet_selector.selector.setCurrentText(
            pcfg.module.textdetector if pcfg.module.textdetector in valid_det else (valid_det[0] if valid_det else '')
        )
        self.bottomBar.ocr_selector.selector.clear()
        self.bottomBar.ocr_selector.selector.addItems(valid_ocr)
        self.bottomBar.ocr_selector.selector.setCurrentText(
            pcfg.module.ocr if pcfg.module.ocr in valid_ocr else (valid_ocr[0] if valid_ocr else '')
        )
        self.bottomBar.trans_selector.selector.clear()
        self.bottomBar.trans_selector.selector.addItems(valid_trans)
        self.bottomBar.trans_selector.selector.setCurrentText(
            pcfg.module.translator if pcfg.module.translator in valid_trans else (valid_trans[0] if valid_trans else '')
        )
        if hasattr(self, 'module_manager') and self.module_manager is not None:
            self.module_manager.refresh_module_dropdowns()
    
    def run_imgtrans(self):
        if not self.imgtrans_proj.is_all_pages_no_text and not pcfg.module.keep_exist_textlines:
            if getattr(pcfg, 'confirm_before_run', True):
                msgBox = QMessageBox(self)
                msgBox.setIcon(QMessageBox.Question)
                msgBox.setWindowTitle(self.tr('Confirmation'))
                msgBox.setText(self.tr('"Run" will clear previous results, "Continue" will try to run from previous progress'))

                restart_btn = msgBox.addButton(self.tr('Run'), QMessageBox.YesRole)
                continue_btn = msgBox.addButton(self.tr('Continue'), QMessageBox.AcceptRole)
                cancel_btn = msgBox.addButton(self.tr('Cancel'), QMessageBox.RejectRole)

                msgBox.setDefaultButton(continue_btn)
                msgBox.exec_()

                clicked_button = msgBox.clickedButton()
                if clicked_button == cancel_btn:
                    return
                elif clicked_button == continue_btn:
                    self.on_run_imgtrans(continue_mode=True)
                    return
        self.on_run_imgtrans()

    def run_imgtrans_wo_textstyle_update(self):
        self._run_imgtrans_wo_textstyle_update = True
        self.run_imgtrans()

    def on_run_imgtrans(self, continue_mode=False, pages_to_process=None, skip_ignored_in_batch=None):
        self._idle_unload_timer.stop()
        self.backup_blkstyles.clear()

        if self.bottomBar.textblockChecker.isChecked():
            self.bottomBar.textblockChecker.click()
        self.postprocess_mt_toggle = False

        all_disabled = pcfg.module.all_stages_disabled()
        skip_ignored = skip_ignored_in_batch if skip_ignored_in_batch is not None else getattr(pcfg, 'skip_ignored_in_run', True)

        # 继续模式：先检查哪些页面需要处理
        if continue_mode:
            pages_to_process = []
            for page_name in self.imgtrans_proj.pages:
                if not self.imgtrans_proj.get_page_progress(page_name):
                    pages_to_process.append(page_name)
            if skip_ignored:
                pages_to_process = [p for p in pages_to_process if not self.imgtrans_proj.is_page_ignored(p)]
            if len(pages_to_process) == 0:
                return
        elif pages_to_process is None:
            if getattr(pcfg, 'manual_mode', False) and self.imgtrans_proj.current_img:
                pages_to_process = [self.imgtrans_proj.current_img]
                self.imgtrans_proj.set_page_progress(pages_to_process[0], 0)
            else:
                pages_to_process = list(self.imgtrans_proj.pages.keys())
                if skip_ignored:
                    pages_to_process = [p for p in pages_to_process if not self.imgtrans_proj.is_page_ignored(p)]
                if getattr(pcfg, 'skip_satisfied_pipeline_steps', False):
                    before_count = len(pages_to_process)
                    pages_to_process = [p for p in pages_to_process if not self.imgtrans_proj.get_page_progress(p)]
                    skipped_count = before_count - len(pages_to_process)
                    if skipped_count > 0 and hasattr(self, 'pipelineInsightsPanel'):
                        self.pipelineInsightsPanel.add_event('SKIP', self.tr('Skipped {0} already-satisfied page(s).').format(skipped_count))
                if not pages_to_process:
                    self.statusBar().showMessage(self.tr('All pages already satisfy the enabled pipeline stages.'), 5000)
                    return
                for page_name in pages_to_process:
                    self.imgtrans_proj.set_page_progress(page_name, 0)
        else:
            for page_name in pages_to_process:
                if page_name in self.imgtrans_proj.pages:
                    self.imgtrans_proj.set_page_progress(page_name, 0)
        
        if pcfg.module.enable_detect:
            for page in self.imgtrans_proj.pages:
                if not pcfg.module.keep_exist_textlines:
                    if not pages_to_process:
                        # 没有指定pages_to_process，清空所有页面
                        self.imgtrans_proj.pages[page].clear()
        else:
            self.st_manager.updateTextBlkList()
            textblk: TextBlock = None
            for page_name, blklist in self.imgtrans_proj.pages.items():
                # 如果指定了pages_to_process，跳过不需要处理的页面
                if pages_to_process and page_name not in pages_to_process:
                    continue
                    
                ffmt_list = []
                self.backup_blkstyles.append(ffmt_list)
                for textblk in blklist:
                    if not pcfg.module.enable_detect:
                        ffmt_list.append(textblk.fontformat.deepcopy())
                    # 继续模式且没有指定pages_to_process时：跳过已有文本的文本块
                    if continue_mode and not pages_to_process and textblk.text and len(textblk.text) > 0:
                        continue
                    if pcfg.module.enable_ocr:
                        textblk.text = []
                        textblk.set_font_colors((0, 0, 0), (0, 0, 0))
                    if pcfg.module.enable_translate or (all_disabled and not self._run_imgtrans_wo_textstyle_update) or pcfg.module.enable_ocr:
                        textblk.rich_text = ''
                    textblk.vertical = textblk.src_is_vertical
        
        # 如果有指定pages_to_process或者是continue_mode，则传递页面列表
        self._last_pages_to_process = list(pages_to_process) if pages_to_process else None
        self._last_run_finish_code = int(getattr(pcfg.module, 'finish_code', 0) or 0)
        self.module_manager.runImgtransPipeline(pages_to_process if (pages_to_process or continue_mode) else None)

    def on_transpanel_changed(self):
        self.canvas.editor_index = self.rightComicTransStackPanel.currentIndex()
        if self.rightComicTransStackPanel.currentIndex() != 0:
            self.canvas.restoreDefaultCursor()
            self.canvas.clearDrawingStroke()
        if not self.canvas.textEditMode() and self.canvas.search_widget.isVisible():
            self.canvas.search_widget.hide()
        self.canvas.updateLayers()

    def import_tstyles(self):
        ddir = osp.dirname(pcfg.text_styles_path)
        p = QFileDialog.getOpenFileName(self, self.tr("Import Text Styles"), ddir, None, "(.json)")
        if not isinstance(p, str):
            p = p[0]
        if p == '':
            return
        try:
            load_textstyle_from(p, raise_exception=True)
            save_config()
            self.textPanel.formatpanel.textstyle_panel.setStyles(text_styles)
            self.textPanel.formatpanel.textstyle_panel.set_active_style_by_name(
                getattr(pcfg, 'default_text_style_name', '') or ''
            )
        except Exception as e:
            create_error_dialog(e, self.tr(f'Failed to load from {p}'))

    def export_tstyles(self):
        ddir = osp.dirname(pcfg.text_styles_path)
        savep = QFileDialog.getSaveFileName(self, self.tr("Save Text Styles"), ddir, None, "(.json)")
        if not isinstance(savep, str):
            savep = savep[0]
        if savep == '':
            return
        suffix = Path(savep).suffix
        if suffix != '.json':
            if suffix == '':
                savep = savep + '.json'
            else:
                savep = savep.replace(suffix, '.json')
        oldp = pcfg.text_styles_path
        try:
            pcfg.text_styles_path = savep
            save_text_styles(raise_exception=True)
            save_config()
        except Exception as e:
            create_error_dialog(e, self.tr(f'Failed save to {savep}'))
            pcfg.text_styles_path = oldp

    def fold_textarea(self, fold: bool):
        pcfg.fold_textarea = fold
        self.textPanel.textEditList.setFoldTextarea(fold)

    def show_source_text(self, show: bool):
        pcfg.show_source_text = show
        self.textPanel.textEditList.setSourceVisible(show)

    def show_trans_text(self, show: bool):
        pcfg.show_trans_text = show
        self.textPanel.textEditList.setTransVisible(show)

    def on_export_doc(self):
        if self.canvas.text_change_unsaved():
            self.st_manager.updateTextBlkList()
        self.export_doc_thread.exportAsDoc(self.imgtrans_proj)

    def on_export_current_page_as(self):
        """Export current page as image: result if available, else inpainted, else original (#1134)."""
        self._export_or_download_page(mode=None)

    def on_download_page_requested(self, mode: str):
        """Handle canvas right-click Download image: result | inpainted | original."""
        self._export_or_download_page(mode=mode)

    def _export_or_download_page(self, mode: str = None):
        """Export or download current page. mode: None = result→inpainted→original; 'result'|'inpainted'|'original' = canvas Download menu."""
        if self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project and select a page first.'))
            return
        from utils.io_utils import imread, imwrite
        current_img = self.imgtrans_proj.current_img
        result_path = self.imgtrans_proj.get_result_path(current_img)
        inpainted_path = self.imgtrans_proj.get_inpainted_path(current_img)
        orig_path = osp.join(self.imgtrans_proj.directory, current_img)
        img = None
        source_kind = None

        def get_inpainted():
            if self.imgtrans_proj.inpainted_valid and self.imgtrans_proj.inpainted_array is not None:
                arr = np.array(self.imgtrans_proj.inpainted_array)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                elif arr.shape[-1] == 4:
                    arr = arr[:, :, :3].copy()
                return arr
            if inpainted_path and osp.exists(inpainted_path):
                arr = imread(inpainted_path)
                if arr is not None and arr.ndim == 3 and arr.shape[-1] == 4:
                    arr = arr[:, :, :3].copy()
                return arr
            return None

        def get_original():
            if orig_path and osp.exists(orig_path):
                return imread(orig_path)
            return None

        if mode == 'original':
            img = get_original()
            source_kind = self.tr('original')
        elif mode == 'inpainted':
            img = get_inpainted()
            source_kind = self.tr('inpainted')
        elif mode == 'result':
            if osp.exists(result_path):
                img = imread(result_path)
                source_kind = self.tr('result')
            elif self.imgtrans_proj.inpainted_valid and self.imgtrans_proj.inpainted_array is not None:
                try:
                    qimg = self.canvas.render_result_img()
                    if qimg is not None and not qimg.isNull():
                        img = pixmap2ndarray(qimg, keep_alpha=False)
                        if img is not None and img.ndim == 3 and img.shape[-1] == 4:
                            img = img[:, :, :3].copy()
                        source_kind = self.tr('result')
                except Exception as e:
                    LOGGER.exception(e)
            if img is None:
                img = get_inpainted()
                source_kind = self.tr('inpainted (no text yet)')
            if img is None:
                img = get_original()
                source_kind = self.tr('original')
        else:
            # mode is None: result → inpainted → original
            if osp.exists(result_path):
                img = imread(result_path)
                source_kind = self.tr('result')
            if img is None:
                img = get_inpainted()
                source_kind = self.tr('inpainted')
            if img is None:
                img = get_original()
                source_kind = self.tr('original')

        if img is None:
            QMessageBox.information(self, self.tr('Export'), self.tr('No image available for current page.'))
            return
        flt = 'PNG (*.png);;JPEG (*.jpg);;WebP (*.webp);;JXL (*.jxl)'
        default_name = osp.splitext(current_img)[0] + '.png'
        path, sel = QFileDialog.getSaveFileName(self, self.tr('Export current page as'), default_name, flt)
        if not path:
            return
        try:
            ext = osp.splitext(path)[1].lower()
            if ext not in ('.png', '.jpg', '.jpeg', '.webp', '.jxl'):
                ext = '.png'
            kw = {'ext': ext, 'quality': pcfg.imgsave_quality}
            if ext == '.webp' and getattr(pcfg, 'imgsave_webp_lossless', False):
                kw['webp_lossless'] = True
            imwrite(path, img, **kw)
            msg = self.tr('Saved to {}').format(path)
            if source_kind:
                msg += ' (' + source_kind + ')'
            QMessageBox.information(self, self.tr('Export'), msg)
        except Exception as e:
            LOGGER.exception(e)
            create_error_dialog(e, self.tr('Export failed'))

    def on_import_doc(self):
        self.import_doc_thread.importDoc(self.imgtrans_proj)

    def on_export_txt(self, dump_target, suffix='.txt'):
        try:
            self.imgtrans_proj.dump_txt(dump_target=dump_target, suffix=suffix)
            create_info_dialog(self.tr('Text file exported to ') + self.imgtrans_proj.dump_txt_path(dump_target, suffix))
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to export as TEXT file'))




    def _selected_page_names(self) -> List[str]:
        try:
            return [(item.data(Qt.ItemDataRole.UserRole) or item.text()) for item in self.pageList.selectedItems()]
        except Exception:
            return []

    def on_lettering_workflow_current_page(self):
        self.on_lettering_workflow_pages(self._selected_page_names())

    def on_open_auto_format_qa(self):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            create_info_dialog({'title': self.tr('Auto-format QA'), 'text': self.tr('Open a project and select a page first.')})
            return
        selected = self.canvas.selected_text_items()
        block_items = selected if selected else list(getattr(self.st_manager, 'textblk_item_list', []) or [])
        if not block_items:
            create_info_dialog({'title': self.tr('Auto-format QA'), 'text': self.tr('No text boxes on current page.')})
            return

        def _apply(indices, profile):
            changed = self.st_manager.auto_format_textboxes(indices=indices, push_undo=True, profile=profile)
            if changed and self.imgtrans_proj and self.imgtrans_proj.directory:
                self.imgtrans_proj.save()
            self.pipelineInsightsPanel.add_event('TYPO_QA', self.tr('Auto-format QA applied changes to {0} text box(es)').format(changed))

        dlg = AutoFormatQADialog(self, block_items, _apply)
        dlg.exec()

    def on_lettering_workflow_pages(self, page_names: List[str] = None):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            create_info_dialog({'title': self.tr('Lettering workflow'), 'text': self.tr('Open a project and select a page first.')})
            return
        try:
            dlg = LetteringWorkflowDialog(self.imgtrans_proj, self.imgtrans_proj.current_img, page_names or [], pcfg, self)
            if not dlg.exec():
                return
            result = self._api_lettering_workflow_ui({
                'pages': dlg.selected_pages(),
                'apply': dlg.should_apply(),
                'export_proof': dlg.should_export_proof(),
                'render': dlg.should_render(),
                'include_final': True,
            })
            plan = result.get('plan', {}) or {}
            msg = [self.tr('Processed {pages} page(s); planned {steps} step(s); applied {applied} formatting fix(es).').format(
                pages=len(result.get('pages', []) or []),
                steps=len(plan.get('steps', []) or []),
                applied=result.get('applied_actions', 0),
            )]
            if result.get('proof_manifests'):
                msg.append(self.tr('Proof packs exported: {0}').format(len(result.get('proof_manifests', []))))
            if result.get('rendered'):
                msg.append(self.tr('Rendered current page: {0}').format((result.get('rendered') or {}).get('path', '')))
            for warning in result.get('warnings', [])[:5]:
                msg.append(self.tr('Warning: ') + str(warning))
            QMessageBox.information(self, self.tr('Lettering workflow'), "\n".join(msg))
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            create_error_dialog(e, self.tr('Lettering workflow failed'))

    def on_next_rendering_issue(self):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            create_info_dialog({'title': self.tr('Rendering issue'), 'text': self.tr('Open a project and select a page first.')})
            return
        try:
            result = self._api_next_rendering_issue_ui({'after_selection': True, 'select': True})
            if not result.get('found'):
                self.pipelineInsightsPanel.add_event('TYPO_QA', self.tr('No rendering issues on current page.'))
                QMessageBox.information(self, self.tr('Rendering issue'), self.tr('No rendering issues found on the current page.'))
                return
            warnings = ', '.join((result.get('issue') or {}).get('warnings', []) or [])
            self.pipelineInsightsPanel.add_event('TYPO_QA', self.tr('Selected rendering issue #{0}: {1}').format(result.get('index'), warnings))
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            create_error_dialog(e, self.tr('Find rendering issue failed'))


    def on_export_lettering_proof_pack(self):
        if not self.imgtrans_proj or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project and select a page first.'))
            return
        default_dir = osp.join(self.imgtrans_proj.directory, 'lettering_proofs')
        out_dir = QFileDialog.getExistingDirectory(self, self.tr('Export lettering proof pack'), default_dir)
        if not out_dir:
            return
        try:
            qimg = self.canvas.render_result_img()
            final_arr = pixmap2ndarray(qimg, keep_alpha=False) if qimg is not None and not qimg.isNull() else None
            manifest = build_lettering_proof_pack(self.imgtrans_proj, self.imgtrans_proj.current_img, out_dir, final_image=final_arr, config_obj=pcfg)
            QMessageBox.information(self, self.tr('Lettering proof'), self.tr('Lettering proof pack exported to {0}').format(manifest.get('page_dir', out_dir)))
            self.pipelineInsightsPanel.add_event('TYPO_QA', self.tr('Exported lettering proof pack: {0}').format(manifest.get('page_dir', out_dir)))
        except Exception as e:
            LOGGER.exception(e)
            create_error_dialog(e, self.tr('Failed to export lettering proof pack'))

    def on_export_svg_text_handoff(self):
        if not self.imgtrans_proj or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project and select a page first.'))
            return
        default_dir = osp.join(self.imgtrans_proj.directory, 'svg_handoff')
        out_dir = QFileDialog.getExistingDirectory(self, self.tr('Export SVG text handoff'), default_dir)
        if not out_dir:
            return
        try:
            manifest = build_svg_text_handoff(self.imgtrans_proj, self.imgtrans_proj.current_img, out_dir)
            warn = '\n'.join(manifest.get('warnings', []) or [])
            msg = self.tr('SVG text handoff exported to {0}.').format(out_dir)
            msg += '\n' + self.tr('SVG: {0}').format(manifest.get('svg_path', ''))
            msg += '\n' + self.tr('Manifest: {0}').format(manifest.get('manifest_path', ''))
            if warn:
                msg += '\n\n' + self.tr('Warnings:') + '\n' + warn
            QMessageBox.information(self, self.tr('Export'), msg)
            self.pipelineInsightsPanel.add_event('EXPORT', self.tr('SVG text handoff exported for {0}.').format(self.imgtrans_proj.current_img))
        except Exception as e:
            LOGGER.exception(e)
            create_error_dialog(e, self.tr('SVG text handoff export failed'))

    def on_export_layered_psd_handoff(self):
        if not self.imgtrans_proj or self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project and select a page first.'))
            return
        default_dir = osp.join(self.imgtrans_proj.directory, 'psd_handoff')
        out_dir = QFileDialog.getExistingDirectory(self, self.tr('Export layered PSD handoff'), default_dir)
        if not out_dir:
            return
        try:
            final_arr = None
            qimg = self.canvas.render_result_img()
            if qimg is not None and not qimg.isNull():
                final_arr = pixmap2ndarray(qimg, keep_alpha=False)
            manifest = build_layered_psd_handoff(self.imgtrans_proj, self.imgtrans_proj.current_img, out_dir, final_image=final_arr)
            warn = '\n'.join(manifest.get('warnings', []) or [])
            msg = self.tr('Layered PSD handoff exported to {0}.').format(out_dir)
            msg += '\n' + self.tr('Manifest: {0}').format(manifest.get('manifest_path', ''))
            msg += '\n' + self.tr('Photoshop script: {0}').format(manifest.get('photoshop_jsx_path', ''))
            if warn:
                msg += '\n\n' + self.tr('Warnings:') + '\n' + warn
            QMessageBox.information(self, self.tr('Export'), msg)
            self.pipelineInsightsPanel.add_event('EXPORT', self.tr('Layered PSD handoff exported for {0}.').format(self.imgtrans_proj.current_img))
        except Exception as e:
            LOGGER.exception(e)
            create_error_dialog(e, self.tr('Layered PSD handoff export failed'))


    def on_export_xliff(self):
        if not self.imgtrans_proj or self.imgtrans_proj.is_empty:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project first.'))
            return
        default_path = osp.join(self.imgtrans_proj.directory, 'translation_export.xliff')
        savep = QFileDialog.getSaveFileName(self, self.tr('Export XLIFF'), default_path, self.tr('XLIFF (*.xliff *.xlf)'))
        if not isinstance(savep, str):
            savep = savep[0]
        if not savep:
            return
        try:
            with open(savep, 'w', encoding='utf-8') as f:
                f.write(export_project_xliff(self.imgtrans_proj))
            create_info_dialog(self.tr('XLIFF exported to: ') + savep)
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to export XLIFF'))

    def on_import_xliff(self):
        if not self.imgtrans_proj or self.imgtrans_proj.is_empty:
            QMessageBox.information(self, self.tr('Import'), self.tr('Open a project first.'))
            return
        dialog = QFileDialog()
        selected_file = str(dialog.getOpenFileUrl(self.parent(), self.tr('Import XLIFF'), filter='*.xliff *.xlf')[0].toLocalFile())
        if not osp.exists(selected_file):
            return
        try:
            with open(selected_file, 'r', encoding='utf-8') as f:
                xml_text = f.read()
            all_matched, match_rst = import_project_xliff(self.imgtrans_proj, xml_text)
            if self.imgtrans_proj.current_img in (match_rst.get('matched_pages') or set()):
                self.canvas.clear_undostack(update_saved_step=True)
                self.st_manager.updateSceneTextitems()
            create_info_dialog(self.tr('XLIFF import done. Matched: {0}, Unmatched: {1}, Missing: {2}').format(len(match_rst.get('matched_pages') or []), len(match_rst.get('unmatched_pages') or []), len(match_rst.get('missing_pages') or [])))
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to import XLIFF'))


    def on_export_translation_json(self):
        if not self.imgtrans_proj or self.imgtrans_proj.is_empty:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project first.'))
            return
        default_path = osp.join(self.imgtrans_proj.directory, 'translation_export.json')
        savep = QFileDialog.getSaveFileName(self, self.tr('Export translation JSON'), default_path, self.tr('JSON (*.json)'))
        if not isinstance(savep, str):
            savep = savep[0]
        if not savep:
            return
        try:
            payload = export_translation_json(self.imgtrans_proj)
            with open(savep, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            create_info_dialog(self.tr('Translation JSON exported to: ') + savep)
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to export translation JSON'))

    def on_import_translation_json(self):
        if not self.imgtrans_proj or self.imgtrans_proj.is_empty:
            QMessageBox.information(self, self.tr('Import'), self.tr('Open a project first.'))
            return
        dialog = QFileDialog()
        selected_file = str(dialog.getOpenFileUrl(self.parent(), self.tr('Import translation JSON'), filter='*.json')[0].toLocalFile())
        if not osp.exists(selected_file):
            return
        try:
            with open(selected_file, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            all_matched, match_rst = import_translation_json(self.imgtrans_proj, payload if isinstance(payload, dict) else {})
            if self.imgtrans_proj.current_img in (match_rst.get('matched_pages') or set()):
                self.canvas.clear_undostack(update_saved_step=True)
                self.st_manager.updateSceneTextitems()
            create_info_dialog(self.tr('Translation JSON import done. Matched: {0}, Unmatched: {1}, Missing: {2}').format(len(match_rst.get('matched_pages') or []), len(match_rst.get('unmatched_pages') or []), len(match_rst.get('missing_pages') or [])))
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to import translation JSON'))


    def on_export_translation_csv(self):
        if not self.imgtrans_proj or self.imgtrans_proj.is_empty:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project first.'))
            return
        default_path = osp.join(self.imgtrans_proj.directory, 'translation_export.csv')
        savep = QFileDialog.getSaveFileName(self, self.tr('Export translation CSV'), default_path, self.tr('CSV (*.csv)'))
        if not isinstance(savep, str):
            savep = savep[0]
        if not savep:
            return
        try:
            with open(savep, 'w', encoding='utf-8', newline='') as f:
                f.write(export_translation_csv_text(self.imgtrans_proj))
            create_info_dialog(self.tr('Translation CSV exported to: ') + savep)
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to export translation CSV'))

    def on_import_translation_csv(self):
        if not self.imgtrans_proj or self.imgtrans_proj.is_empty:
            QMessageBox.information(self, self.tr('Import'), self.tr('Open a project first.'))
            return
        dialog = QFileDialog()
        selected_file = str(dialog.getOpenFileUrl(self.parent(), self.tr('Import translation CSV'), filter='*.csv')[0].toLocalFile())
        if not osp.exists(selected_file):
            return
        try:
            with open(selected_file, 'r', encoding='utf-8') as f:
                text = f.read()
            all_matched, match_rst = import_translation_csv_text(self.imgtrans_proj, text)
            if self.imgtrans_proj.current_img in (match_rst.get('matched_pages') or set()):
                self.canvas.clear_undostack(update_saved_step=True)
                self.st_manager.updateSceneTextitems()
            create_info_dialog(self.tr('Translation CSV import done. Matched: {0}, Unmatched: {1}, Missing: {2}').format(len(match_rst.get('matched_pages') or []), len(match_rst.get('unmatched_pages') or []), len(match_rst.get('missing_pages') or [])))
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to import translation CSV'))


    def on_batch_find_replace_current_project(self):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            create_info_dialog({'title': self.tr('No project loaded'), 'text': self.tr('Open a project first.')})
            return
        pattern, ok = QInputDialog.getText(self, self.tr('Batch find/replace'), self.tr('Find (regex supported):'))
        if not ok or not str(pattern).strip():
            return
        replacement, ok2 = QInputDialog.getText(self, self.tr('Batch find/replace'), self.tr('Replace with:'))
        if not ok2:
            return
        preview = preview_batch_find_replace(self.imgtrans_proj, str(pattern), str(replacement), use_regex=True, case_sensitive=False, target='translation')
        if preview.get('count', 0) <= 0:
            create_info_dialog({'title': self.tr('Batch find/replace'), 'text': self.tr('No matches found.')})
            return
        sample = []
        for hit in preview.get('hits', [])[:12]:
            sample.append(f"{hit['page']}#{hit['index']+1}: {hit['before'][:36]} -> {hit['after'][:36]}")
        msg = self.tr('Preview matches: {0}\nApply replacements?').format(preview.get('count', 0)) + '\n\n' + '\n'.join(sample)
        yn = QMessageBox.question(self, self.tr('Batch find/replace preview'), msg, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if yn != QMessageBox.StandardButton.Yes:
            return
        changed, _applied = apply_batch_find_replace(self.imgtrans_proj, preview)
        if changed > 0:
            self.canvas.updateTextBlkList()
            self.canvas.setProjSaveState(True)
        create_info_dialog({'title': self.tr('Batch find/replace'), 'text': self.tr('Applied changes: {0}').format(changed)})

    def on_export_structured_ocr_json(self):
        if not self.imgtrans_proj or self.imgtrans_proj.is_empty:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project first.'))
            return
        default_path = osp.join(self.imgtrans_proj.directory, 'structured_ocr_export.json')
        savep = QFileDialog.getSaveFileName(self, self.tr('Export structured OCR JSON'), default_path, self.tr('JSON (*.json)'))
        if not isinstance(savep, str):
            savep = savep[0]
        if not savep:
            return
        if Path(savep).suffix.lower() != '.json':
            savep += '.json'
        try:
            payload = build_structured_ocr_export(self.imgtrans_proj)
            with open(savep, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            create_info_dialog({'title': self.tr('Structured OCR exported'), 'text': self.tr('Exported {0} page(s) to {1}').format(len(payload.get('pages', [])), savep)})
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to export structured OCR JSON'))

    def on_export_lptxt(self):
        """Export translations to LPtxt format (for auto-labeling tools)."""
        if not self.imgtrans_proj or not self.imgtrans_proj.directory:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project first.'))
            return
        reply = QMessageBox.question(
            self,
            self.tr('Export translation to LPtxt'),
            self.tr('Include font and size info in the LPtxt output?'),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return
        include_font_info = reply == QMessageBox.StandardButton.Yes
        try:
            path = self.imgtrans_proj.dump_lptxt(include_font_info=include_font_info)
            create_info_dialog(self.tr('LPtxt exported to: ') + path)
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to export LPtxt'))

    def on_import_trans_txt(self):
        try:
            selected_file = ''
            dialog = QFileDialog()
            selected_file = str(dialog.getOpenFileUrl(self.parent(), self.tr('Import *.md/*.txt'), filter="*.txt *.md *.TXT *.MD")[0].toLocalFile())
            if not osp.exists(selected_file):
                return

            all_matched, match_rst = self.imgtrans_proj.load_translation_from_txt(selected_file)
            matched_pages = match_rst['matched_pages']

            if self.imgtrans_proj.current_img in matched_pages:
                self.canvas.clear_undostack(update_saved_step=True)
                self.st_manager.updateSceneTextitems()

            if all_matched:
                msg = self.tr('Translation imported and matched successfully.')
            else:
                msg = self.tr('Imported txt file not fully matched with current project, please make sure source txt file structured like results from \"export TXT/markdown\"')
                if len(match_rst['missing_pages']) > 0:
                    msg += '\n' + self.tr('Missing pages: ') + '\n'
                    msg += '\n'.join(match_rst['missing_pages'])
                if len(match_rst['unexpected_pages']) > 0:
                    msg += '\n' + self.tr('Unexpected pages: ') + '\n'
                    msg += '\n'.join(match_rst['unexpected_pages'])
                if len(match_rst['unmatched_pages']) > 0:
                    msg += '\n' + self.tr('Unmatched pages: ') + '\n'
                    msg += '\n'.join(match_rst['unmatched_pages'])
                msg = msg.strip()

            for pagename in matched_pages:
                for blk in self.imgtrans_proj.pages[pagename]:
                    blk.translation = self.mtSubWidget.sub_text(blk.translation)
            
            create_info_dialog(msg)

        except Exception as e:
            create_error_dialog(e, self.tr('Failed to import translation from ') + selected_file)

    def on_reveal_file(self):
        current_img_path = self.imgtrans_proj.current_img_path()
        if sys.platform == 'win32':
            # qprocess seems to fuck up with "\""
            p = "\""+str(Path(current_img_path))+"\""
            subprocess.Popen("explorer.exe /select,"+p, shell=True)
        elif sys.platform == 'darwin':
            p = "\""+current_img_path+"\""
            subprocess.Popen("open -R "+p, shell=True)

    def on_translate_images(self, image_names: list):
        if not image_names:
            return
        self.on_run_imgtrans(pages_to_process=image_names)

    def on_run_ocr_images(self, image_names: list):
        if not image_names:
            return
        self._run_stages_restore = (
            pcfg.module.enable_detect,
            pcfg.module.enable_ocr,
            pcfg.module.enable_translate,
            pcfg.module.enable_inpaint,
        )
        pcfg.module.enable_detect = False
        pcfg.module.enable_ocr = True
        pcfg.module.enable_translate = False
        pcfg.module.enable_inpaint = False
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        self.on_run_imgtrans(pages_to_process=image_names)

    def on_run_detect_images(self, image_names: list):
        if not image_names:
            return
        self._run_stages_restore = (
            pcfg.module.enable_detect,
            pcfg.module.enable_ocr,
            pcfg.module.enable_translate,
            pcfg.module.enable_inpaint,
        )
        pcfg.module.enable_detect = True
        pcfg.module.enable_ocr = False
        pcfg.module.enable_translate = False
        pcfg.module.enable_inpaint = False
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        self.on_run_imgtrans(pages_to_process=image_names)

    def on_run_translate_images(self, image_names: list):
        if not image_names:
            return
        self._run_stages_restore = (
            pcfg.module.enable_detect,
            pcfg.module.enable_ocr,
            pcfg.module.enable_translate,
            pcfg.module.enable_inpaint,
        )
        pcfg.module.enable_detect = False
        pcfg.module.enable_ocr = False
        pcfg.module.enable_translate = True
        pcfg.module.enable_inpaint = False
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        self.on_run_imgtrans(pages_to_process=image_names)

    def on_run_inpaint_images(self, image_names: list):
        if not image_names:
            return
        self._run_stages_restore = (
            pcfg.module.enable_detect,
            pcfg.module.enable_ocr,
            pcfg.module.enable_translate,
            pcfg.module.enable_inpaint,
        )
        pcfg.module.enable_detect = False
        pcfg.module.enable_ocr = False
        pcfg.module.enable_translate = False
        pcfg.module.enable_inpaint = True
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        self.on_run_imgtrans(pages_to_process=image_names)

    def on_toggle_page_ignored(self, pagenames: list, ignored: bool):
        """Mark selected pages as ignored (skip in full run) or include in run."""
        if not pagenames or self.imgtrans_proj.is_empty:
            return
        for name in pagenames:
            if name in self.imgtrans_proj.pages:
                self.imgtrans_proj.set_page_ignored(name, ignored)
        if self.imgtrans_proj.directory:
            self.imgtrans_proj.save()
            self.canvas.setProjSaveState(False)
        self._update_page_list_state_style()

    def on_set_page_completion_state(self, pagenames: list, state: str):
        """Persist user-visible completion state for selected pages."""
        if not pagenames or self.imgtrans_proj.is_empty:
            return
        state = str(state or 'todo').strip().lower()
        if state not in {'todo', 'translated', 'reviewed', 'exported'}:
            state = 'todo'
        for name in pagenames:
            self.imgtrans_proj.set_page_completion_state(name, state)
        if self.imgtrans_proj.directory:
            self.imgtrans_proj.save()
            self.canvas.setProjSaveState(False)
        self._update_page_list_state_style()
        self.pipelineInsightsPanel.add_event('PAGE_STATE', self.tr('Set {0} page(s) to {1}.').format(len(pagenames), state))

    def _update_page_list_state_style(self):
        """Update page list items for ignored + completion state visibility."""
        default_fg = self.pageList.palette().brush(self.pageList.foregroundRole())
        default_bg = self.pageList.palette().brush(self.pageList.backgroundRole())
        if getattr(pcfg, 'darkmode', False):
            ignored_fg = QBrush(QColor(120, 120, 120))
            ignored_bg = QBrush(QColor(45, 45, 48))
            state_bg = {
                'translated': QBrush(QColor(35, 58, 74)),
                'reviewed': QBrush(QColor(43, 74, 50)),
                'exported': QBrush(QColor(74, 61, 36)),
            }
        else:
            ignored_fg = QBrush(QColor(100, 100, 100))
            ignored_bg = QBrush(QColor(232, 232, 234))
            state_bg = {
                'translated': QBrush(QColor(225, 244, 255)),
                'reviewed': QBrush(QColor(224, 246, 228)),
                'exported': QBrush(QColor(255, 241, 214)),
            }
        labels = {
            'todo': self.tr('Needs work'),
            'translated': self.tr('Translated'),
            'reviewed': self.tr('Reviewed'),
            'exported': self.tr('Exported'),
        }
        for i in range(self.pageList.count()):
            item = self.pageList.item(i)
            if item is None:
                continue
            name = item.data(Qt.ItemDataRole.UserRole) or item.text()
            is_ignored = self.imgtrans_proj.is_page_ignored(name) if not self.imgtrans_proj.is_empty else False
            state = self.imgtrans_proj.get_page_completion_state(name) if not self.imgtrans_proj.is_empty else 'todo'
            state_prefix = {'todo': '◻', 'translated': '🟦', 'reviewed': '🟩', 'exported': '🟨'}.get(state, '◻')
            display_name = f"{state_prefix} {name}"
            if is_ignored:
                display_name = f"⏭ {display_name}"
            if item.text() != display_name:
                item.setText(display_name)
            tips = [self.tr('Completion: {0}').format(labels.get(state, state))]
            if is_ignored:
                tips.append(self.tr('Ignored in run (right-click → Include in run to process)'))
                item.setForeground(ignored_fg)
                item.setBackground(ignored_bg)
            else:
                item.setForeground(default_fg)
                item.setBackground(state_bg.get(state, default_bg))
            item.setToolTip('\n'.join(tips))

    def on_remove_images(self, image_names: list):
        if not image_names:
            return
        if len(image_names) == 1:
            confirm_msg = self.tr('Are you sure you want to remove "{}" from the project?').format(image_names[0])
        else:
            confirm_msg = self.tr('Are you sure you want to remove {} images from the project?').format(len(image_names))
        reply = QMessageBox.question(
            self, self.tr('Confirm Removal'), confirm_msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        current_img = self.imgtrans_proj.current_img
        need_switch = current_img in image_names
        next_img = None
        for img_name in self.imgtrans_proj.pages.keys():
            if img_name not in image_names:
                next_img = img_name
                break
        if need_switch and next_img:
            if self.canvas.projstate_unsaved:
                self.saveCurrentPage()
            self.imgtrans_proj.set_current_img(next_img)
        for img_name in image_names:
            if img_name in self.imgtrans_proj.pages:
                del self.imgtrans_proj.pages[img_name]
            if img_name in self.imgtrans_proj._pagename2idx:
                idx = self.imgtrans_proj._pagename2idx[img_name]
                del self.imgtrans_proj._pagename2idx[img_name]
                if idx in self.imgtrans_proj._idx2pagename:
                    del self.imgtrans_proj._idx2pagename[idx]
        self.imgtrans_proj._pagename2idx = {}
        self.imgtrans_proj._idx2pagename = {}
        for ii, imgname in enumerate(self.imgtrans_proj.pages.keys()):
            self.imgtrans_proj._pagename2idx[imgname] = ii
            self.imgtrans_proj._idx2pagename[ii] = imgname
        if len(self.imgtrans_proj.pages) == 0:
            self.imgtrans_proj.set_current_img(None)
            self.canvas.clear_undostack(update_saved_step=True)
            self.canvas.drop_files = []
            self.canvas.drop_folder = None
            self.canvas.updateCanvas()
            self.st_manager.clearSceneTextitems()
            self.titleBar.setTitleContent(page_name="")
        elif need_switch and next_img:
            self.canvas.clear_undostack(update_saved_step=True)
            self.canvas.drop_files = []
            self.canvas.drop_folder = None
            self.canvas.updateCanvas()
            self.st_manager.updateSceneTextitems()
            self.titleBar.setTitleContent(page_name=self.imgtrans_proj.current_img)
        self.imgtrans_proj.save()
        self.updatePageList()

    def on_set_gsearch_widget(self):
        setup = self.leftBar.globalSearchChecker.isChecked()
        if setup and not self._has_open_project():
            # Global project search has nothing to search without a project. Keep the
            # welcome screen stable instead of showing an empty side panel.
            try:
                self.leftBar.globalSearchChecker.blockSignals(True)
                self.leftBar.globalSearchChecker.setChecked(False)
            finally:
                self.leftBar.globalSearchChecker.blockSignals(False)
            self._show_welcome_screen()
            return

        if setup:
            if self.leftStackWidget.isHidden():
                self.leftStackWidget.show()
            self.leftBar.showPageListLabel.setChecked(False)
            self.leftStackWidget.setCurrentWidget(self.global_search_widget)
        else:
            self.leftStackWidget.hide()

    def on_fin_export_doc(self):
        msg = QMessageBox()
        msg.setText(self.tr('Export to ') + self.imgtrans_proj.doc_path())
        msg.exec_()

    def on_fin_import_doc(self):
        self.st_manager.updateSceneTextitems()

    def on_global_replace_finished(self):
        rt = self.global_search_widget.replace_thread
        self.canvas.push_text_command(
            GlobalRepalceAllCommand(rt.sceneitem_list, rt.background_list, rt.target_text, self.imgtrans_proj)
        )
        rt.sceneitem_list = None
        rt.background_list = None

    def _on_theme_light_triggered(self):
        pcfg.darkmode = False
        self.titleBar.themeLightAction.setChecked(True)
        self.titleBar.themeDarkAction.setChecked(False)
        self.resetStyleSheet()

    def _on_theme_dark_triggered(self):
        pcfg.darkmode = True
        self.titleBar.themeLightAction.setChecked(False)
        self.titleBar.themeDarkAction.setChecked(True)
        self.resetStyleSheet()

    def on_config_darkmode_changed(self, checked: bool):
        self.titleBar.themeLightAction.setChecked(not checked)
        self.titleBar.themeDarkAction.setChecked(checked)
        self.resetStyleSheet()
        self.save_config()

    def _on_config_custom_cursor_changed(self):
        self._apply_custom_cursor()
        self.save_config()

    def ocr_postprocess(self, textblocks: List[TextBlock], img, ocr_module=None, **kwargs):
        for blk in textblocks:
            text = blk.get_text()
            blk.text = self.ocrSubWidget.sub_text(text)

        # 字体检测：在 OCR 完成后按配置执行（按需导入以减少启动开销）
        try:
            if pcfg.module.ocr_font_detect:
                try:
                    from utils import font_detect
                    for blk in textblocks:
                        try:
                            name, conf = font_detect.detect_font_from_block(img, blk)
                            blk._detected_font_name = name
                            blk._detected_font_confidence = float(conf)
                        except Exception:
                            # don't break the pipeline on detector errors
                            blk._detected_font_name = ''
                            blk._detected_font_confidence = 0.0
                except Exception:
                    # failed to import or run detector
                    pass
        except Exception:
            pass

    def translate_preprocess(self, translations: List[str] = None, textblocks: List[TextBlock] = None, translator = None, source_text:list = None):
        if source_text is None:
            source_text = []
        for i in range(len(source_text)):
            source_text[i] = self.mtPreSubWidget.sub_text(source_text[i])

    def translate_postprocess(self, translations: List[str] = None, textblocks: List[TextBlock] = None, translator = None):
        if not self.postprocess_mt_toggle:
            return

        glossary_map = getattr(pcfg, 'llm_glossary_map', {}) or {}
        for ii, tr in enumerate(translations):
            text = self.mtSubWidget.sub_text(tr)
            if getattr(pcfg, 'enable_glossary_enforcement', True) and glossary_map:
                text = enforce_glossary(text, glossary_map).text
            if getattr(pcfg, 'enable_text_normalization', False):
                text = normalize_text(text, getattr(pcfg, 'text_normalization_profile', 'balanced'))
            translations[ii] = text

        if getattr(pcfg, 'enable_back_translation_qa', False) and textblocks:
            try:
                for i, blk in enumerate(textblocks[:len(translations)]):
                    src = blk.get_text() if hasattr(blk, 'get_text') else ''
                    drift = back_translation_drift_score(src, translations[i])
                    if drift > float(getattr(pcfg, 'back_translation_drift_threshold', 0.58)):
                        LOGGER.warning('Translation drift warning blk=%s drift=%.3f', i, drift)
                        self.pipelineInsightsPanel.add_warning('MT_DRIFT', f'Block {i} drift={drift:.3f}')
            except Exception:
                LOGGER.debug('Back-translation QA scoring skipped.', exc_info=True)

    def on_copy_src(self):
        blks = self.canvas.selected_text_items()
        if len(blks) == 0:
            return
        
        if isinstance(self.module_manager.translator, GPTTranslator):
            src_list = [self.st_manager.pairwidget_list[blk.idx].e_source.toPlainText() for blk in blks]
            src_txt = ''
            for (prompt, num_src) in self.module_manager.translator._assemble_prompts(src_list, max_tokens=4294967295):
                src_txt += prompt
            src_txt = src_txt.strip()
        else:
            src_list = [self.st_manager.pairwidget_list[blk.idx].e_source.toPlainText().strip().replace('\n', ' ') for blk in blks]
            src_txt = '\n'.join(src_list)

        self.st_manager.app_clipborad.setText(src_txt, QClipboard.Mode.Clipboard)

    def on_paste_src(self):
        blks = self.canvas.selected_text_items()
        if len(blks) == 0:
            return

        src_widget_list = [self.st_manager.pairwidget_list[blk.idx].e_source for blk in blks]
        text_list = self.st_manager.app_clipborad.text().split('\n')
        
        n_paragraph = min(len(src_widget_list), len(text_list))
        if n_paragraph < 1:
            return
        
        src_widget_list = src_widget_list[:n_paragraph]
        text_list = text_list[:n_paragraph]

        self.canvas.push_undo_command(PasteSrcItemsCommand(src_widget_list, text_list))

    def on_copy_trans(self):
        blks = self.canvas.selected_text_items()
        if len(blks) == 0:
            return
        trans_list = [self.st_manager.pairwidget_list[blk.idx].e_trans.toPlainText().strip().replace('\n', ' ') for blk in blks]
        self.st_manager.app_clipborad.setText('\n'.join(trans_list), QClipboard.Mode.Clipboard)

    def on_paste_trans(self):
        blks = self.canvas.selected_text_items()
        if len(blks) == 0:
            return
        trans_widget_list = [self.st_manager.pairwidget_list[blk.idx].e_trans for blk in blks]
        text_list = self.st_manager.app_clipborad.text().split('\n')
        n_paragraph = min(len(trans_widget_list), len(text_list))
        if n_paragraph < 1:
            return
        trans_widget_list = trans_widget_list[:n_paragraph]
        text_list = text_list[:n_paragraph]
        for w, t in zip(trans_widget_list, text_list):
            w.setPlainText(t)
        self.canvas.setProjSaveState(False)

    def on_clear_src(self):
        blks = self.canvas.selected_text_items()
        for blk in blks:
            self.st_manager.pairwidget_list[blk.idx].e_source.setPlainText('')
        if blks:
            self.canvas.setProjSaveState(False)

    def on_clear_trans(self):
        blks = self.canvas.selected_text_items()
        for blk in blks:
            self.st_manager.pairwidget_list[blk.idx].e_trans.setPlainText('')
        if blks:
            self.canvas.setProjSaveState(False)

    def on_select_all_canvas(self):
        # Comic trans (canvas) view is centralStackWidget index 1; index 0 = welcome, 2 = config
        if self.centralStackWidget.currentIndex() != 1:
            return
        self.st_manager.set_blkitems_selection(True)

    def on_spell_check_src(self):
        """Open spell check panel and run check on source text so misspelled words are listed."""
        try:
            from utils.ocr_spellcheck import _init_enchant
            if not _init_enchant():
                raise RuntimeError("enchant not available")
        except Exception:
            QMessageBox.warning(
                self,
                self.tr("Spell check"),
                self.tr("Spell check requires pyenchant. Install it and a system dictionary (e.g. en_US)."),
            )
            return
        if self.centralStackWidget.currentIndex() != 1:
            return
        if self.rightComicTransStackPanel.isHidden():
            self.rightComicTransStackPanel.show()
            self.bottomBar.paintChecker.setChecked(True)
        self.rightComicTransStackPanel.setCurrentIndex(2)
        self.spellCheckPanel.run_check(source_not_translation=True)

    def on_spell_check_trans(self):
        """Open spell check panel and run check on translation so misspelled words are listed."""
        try:
            from utils.ocr_spellcheck import _init_enchant
            if not _init_enchant():
                raise RuntimeError("enchant not available")
        except Exception:
            QMessageBox.warning(
                self,
                self.tr("Spell check"),
                self.tr("Spell check requires pyenchant. Install it and a system dictionary (e.g. en_US)."),
            )
            return
        if self.centralStackWidget.currentIndex() != 1:
            return
        if self.rightComicTransStackPanel.isHidden():
            self.rightComicTransStackPanel.show()
            self.bottomBar.paintChecker.setChecked(True)
        self.rightComicTransStackPanel.setCurrentIndex(2)
        self.spellCheckPanel.run_check(source_not_translation=False)

    def on_trim_whitespace(self):
        """Trim leading/trailing whitespace from each line in selected blocks (source and translation)."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        for blkitem in blks:
            blk = blkitem.blk
            pw = self.st_manager.pairwidget_list[blkitem.idx]
            src = pw.e_source.toPlainText()
            trans = pw.e_trans.toPlainText()
            src_trimmed = "\n".join(line.strip() for line in src.split("\n"))
            trans_trimmed = "\n".join(line.strip() for line in trans.split("\n"))
            if getattr(blk, "text", None) and isinstance(blk.text, list):
                blk.text = [line.strip() for line in blk.text]
            blk.translation = trans_trimmed
            pw.e_source.setPlainText(src_trimmed)
            pw.e_trans.setPlainText(trans_trimmed)
        self.canvas.setProjSaveState(False)

    def on_to_uppercase(self):
        """Convert source and translation of selected blocks to uppercase."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        for blkitem in blks:
            blk = blkitem.blk
            pw = self.st_manager.pairwidget_list[blkitem.idx]
            src = locale_aware_upper(pw.e_source.toPlainText(), getattr(pcfg.module, 'translate_source', ''))
            trans = locale_aware_upper(pw.e_trans.toPlainText(), getattr(pcfg.module, 'translate_target', ''))
            if getattr(blk, "text", None) and isinstance(blk.text, list):
                blk.text = [locale_aware_upper(line, getattr(pcfg.module, 'translate_source', '')) for line in blk.text]
            blk.translation = trans
            pw.e_source.setPlainText(src)
            pw.e_trans.setPlainText(trans)
        self.canvas.setProjSaveState(False)

    def on_to_lowercase(self):
        """Convert source and translation of selected blocks to lowercase."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        for blkitem in blks:
            blk = blkitem.blk
            pw = self.st_manager.pairwidget_list[blkitem.idx]
            src = pw.e_source.toPlainText().lower()
            trans = pw.e_trans.toPlainText().lower()
            if getattr(blk, "text", None) and isinstance(blk.text, list):
                blk.text = [line.lower() for line in blk.text]
            blk.translation = trans
            pw.e_source.setPlainText(src)
            pw.e_trans.setPlainText(trans)
        self.canvas.setProjSaveState(False)

    def on_toggle_strikethrough(self):
        """Toggle strikethrough on selected text blocks."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        gf = self.textPanel.formatpanel.global_format
        new_val = not getattr(blks[0].fontformat, 'strikethrough', False)
        from ui import fontformat_commands as FC
        FC.ffmt_change_strikethrough('strikethrough', new_val, gf, True, blks)
        self.textPanel.formatpanel.formatBtnGroup.strikethroughBtn.setChecked(new_val)
        self.st_manager.updateSceneTextitems()
        self.canvas.setProjSaveState(False)

    def on_set_gradient_type(self, gradient_type: int):
        """Set gradient type (0=Linear, 1=Radial) on selected text blocks."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        gf = self.textPanel.formatpanel.global_format
        from ui import fontformat_commands as FC
        FC.ffmt_change_gradient_type('gradient_type', gradient_type, gf, True, blks)
        self.st_manager.updateSceneTextitems()
        self.canvas.setProjSaveState(False)

    def on_set_text_on_path(self, mode: int):
        """Set text on path (0=None, 1=Circular, 2=Arc) on selected text blocks."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        gf = self.textPanel.formatpanel.global_format
        from ui import fontformat_commands as FC
        FC.ffmt_change_text_on_path('text_on_path', [mode] * len(blks), gf, True, blks)
        self.st_manager.updateSceneTextitems()
        self.canvas.setProjSaveState(False)

    def on_run_detect_region(self, rect, replace_page: bool = False):
        if not self.imgtrans_proj.img_valid or self.imgtrans_proj.img_array is None:
            return
        page_name = self.imgtrans_proj.current_img
        if not page_name:
            return
        self.module_manager.run_detect_region(rect, self.imgtrans_proj.img_array, page_name, replace_if_full_page=replace_page)

    def on_detect_region_finished(self, page_name: str, blk_list: List, replace: bool = False):
        if page_name != self.imgtrans_proj.current_img:
            return
        if replace:
            # Replace all blocks on the page (e.g. "Detect text on page")
            old_blks = self.imgtrans_proj.pages.get(page_name, [])
            self.imgtrans_proj.pages[page_name] = list(blk_list)
            # Old blocks are TextBlocks (no .idx); widgets and canvas items are in same order as page, so use position index.
            n_old = len(old_blks)
            to_remove = [self.st_manager.textblk_item_list[i] for i in range(n_old) if i < len(self.st_manager.textblk_item_list)]
            to_remove_widgets = [self.st_manager.pairwidget_list[i] for i in range(n_old) if i < len(self.st_manager.pairwidget_list)]
            if to_remove:
                self.st_manager.deleteTextblkItemList(to_remove, to_remove_widgets)
            im_h, im_w = self.imgtrans_proj.img_array.shape[:2]
            for blk in blk_list:
                if getattr(blk, 'lines', None) and len(blk.lines) > 0:
                    examine_textblk(blk, im_w, im_h, sort=True)
            for blk in blk_list:
                self.st_manager.addTextBlock(blk)
        elif blk_list:
            im_h, im_w = self.imgtrans_proj.img_array.shape[:2]
            for blk in blk_list:
                if getattr(blk, 'lines', None) and len(blk.lines) > 0:
                    examine_textblk(blk, im_w, im_h, sort=True)
            self.imgtrans_proj.pages[page_name].extend(blk_list)
            for blk in blk_list:
                self.st_manager.addTextBlock(blk)
        self.canvas.setProjSaveState(False)
        self.global_search_widget.set_document_edited()

    def on_merge_selected_blocks(self):
        blks = self.canvas.selected_text_items()
        if len(blks) < 2:
            return
        self.canvas.push_undo_command(MergeBlkItemsCommand(blks, self.st_manager))

    def on_split_selected_regions(self):
        """Try to split each selected region into multiple regions using image gaps (e.g. two bubbles in one box)."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        page_name = self.imgtrans_proj.current_img
        if not page_name or not self.imgtrans_proj.img_valid or self.imgtrans_proj.img_array is None:
            return
        img = self.imgtrans_proj.img_array
        if img.ndim == 3:
            crop_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            crop_img = img
        h_img, w_img = img.shape[:2]
        # Process in descending index so indices stay valid
        for blk_item in sorted(blks, key=lambda b: b.idx, reverse=True):
            blk = blk_item.blk
            x1, y1, x2, y2 = blk.xyxy
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = crop_img[y1:y2, x1:x2]
            if crop.size < 100:
                continue
            try:
                span_list, _ = split_textblock(crop, crop_ratio=0.2, shrink=True, discard=True, recheck=False)
            except Exception:
                continue
            if not span_list or len(span_list) <= 1:
                continue
            new_blks = []
            for span in span_list:
                if span.top is None or span.bottom is None or span.left is None or span.right is None:
                    continue
                ox1 = x1 + max(0, span.left)
                oy1 = y1 + max(0, span.top)
                ox2 = x1 + min(crop.shape[1], span.right)
                oy2 = y1 + min(crop.shape[0], span.bottom)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                new_blk = copy.copy(blk)
                new_blk.xyxy = [ox1, oy1, ox2, oy2]
                new_blk.lines = [[[ox1, oy1], [ox2, oy1], [ox2, oy2], [ox1, oy2]]]
                new_blk.text = [""]
                new_blk.translation = ""
                new_blks.append(new_blk)
            if len(new_blks) <= 1:
                continue
            idx = blk_item.idx
            self.imgtrans_proj.pages[page_name].pop(idx)
            self.st_manager.deleteTextblkItemList([blk_item], [self.st_manager.pairwidget_list[idx]])
            for i, nb in enumerate(new_blks):
                self.imgtrans_proj.pages[page_name].insert(idx + i, nb)
            self.st_manager.insertTextBlocksAt(idx, new_blks)
        self.canvas.setProjSaveState(False)
        self.global_search_widget.set_document_edited()

    def on_move_blocks_up(self):
        blks = self.canvas.selected_text_items()
        if len(blks) != 1:
            return
        idx = blks[0].idx
        if idx <= 0:
            return
        self.st_manager.swap_block_positions(idx - 1, idx)
        self.canvas.setProjSaveState(False)

    def on_move_blocks_down(self):
        blks = self.canvas.selected_text_items()
        if len(blks) != 1:
            return
        idx = blks[0].idx
        n = len(self.st_manager.textblk_item_list)
        if idx >= n - 1:
            return
        self.st_manager.swap_block_positions(idx, idx + 1)
        self.canvas.setProjSaveState(False)

    def on_import_image_to_blk(self):
        """PR #1070: Import image as foreground overlay for the single selected empty block."""
        if self.imgtrans_proj.is_empty or not self.imgtrans_proj.directory:
            return
        sel = self.canvas.selected_text_items()
        if len(sel) != 1 or not sel[0].document().isEmpty():
            return
        item = sel[0]
        blk = getattr(item, 'blk', None)
        if blk is None:
            return
        flt = self.tr("Images") + " (*.png *.jpg *.jpeg *.bmp *.webp);;" + self.tr("All files") + " (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, self.tr("Import Image"), None, flt)
        if not path or not osp.isfile(path):
            return
        proj_dir = self.imgtrans_proj.directory
        overlay_dir = osp.join(proj_dir, "overlays")
        try:
            os.makedirs(overlay_dir, exist_ok=True)
        except OSError:
            create_error_dialog(self, self.tr("Could not create overlays folder."))
            return
        base = osp.splitext(osp.basename(self.imgtrans_proj.current_img or "page"))[0]
        ext = osp.splitext(path)[1] or ".png"
        dest_name = f"{base}_blk{item.idx}{ext}"
        dest_path = osp.join(overlay_dir, dest_name)
        try:
            shutil.copy2(path, dest_path)
        except OSError as e:
            create_error_dialog(self, self.tr("Could not copy image: ") + str(e))
            return
        rel_path = osp.join("overlays", dest_name)
        blk.foreground_image_path = rel_path
        item.invalidate_foreground_cache()
        item.update()
        self.imgtrans_proj.save()
        self.canvas.setProjSaveState(False)

    def on_clear_overlay(self):
        """PR #1070: Clear foreground image overlay from the selected block."""
        sel = self.canvas.selected_text_items()
        if len(sel) != 1:
            return
        item = sel[0]
        blk = getattr(item, 'blk', None)
        if blk is None or not getattr(blk, 'foreground_image_path', None):
            return
        blk.foreground_image_path = None
        item.invalidate_foreground_cache()
        item.update()
        if not self.imgtrans_proj.is_empty and self.imgtrans_proj.directory:
            self.imgtrans_proj.save()
            self.canvas.setProjSaveState(False)
    
    def run_batch(self, exec_dirs: Union[List, str], **kwargs):
        if not isinstance(exec_dirs, List):
            exec_dirs = [x.strip() for x in exec_dirs.split(',') if x.strip()]
        valid_dirs = []
        from utils.validation import normalize_zip_file_input, validate_batch_input_path
        for d in exec_dirs:
            d = normalize_zip_file_input(d)
            if not d:
                continue
            try:
                path, kind = validate_batch_input_path(d)
            except ValueError as e:
                LOGGER.warning(str(e))
                continue
            if kind == "zip":
                if not osp.isfile(path) or not path.lower().endswith(".zip"):
                    continue
                try:
                    job = self._zip_batch.add_zip(path)
                    if not job.project_dirs:
                        LOGGER.warning(f'ZIP contains no images: {path}')
                        continue
                    LOGGER.info(
                        f'ZIP batch: extracted {osp.basename(path)} -> {job.extracted_root} '
                        f'({len(job.project_dirs)} folder(s) with images), output -> {job.output_root}'
                    )
                    valid_dirs.extend(job.project_dirs)
                except Exception as e:
                    LOGGER.warning(f'Failed to extract ZIP {path}: {e}')
                    continue
            else:
                valid_dirs.append(path)
        self.exec_dirs = valid_dirs
        self._batch_cancelled = False
        self._batch_skip_ignored = kwargs.get('skip_ignored_pages', getattr(pcfg, 'skip_ignored_in_run', True))
        self._batch_run_indices = None  # after first item, use same page indices for rest of batch
        self.run_next_dir()

    def run_next_dir(self):
        if len(self.exec_dirs) == 0:
            while self.imsave_thread.isRunning():
                time.sleep(0.1)
            LOGGER.info('finished translating all dirs.')
            try:
                self._zip_batch.cleanup_all()
            except Exception:
                pass
            if self._batch_cancelled:
                self.batch_queue_cancelled.emit()
                self._batch_cancelled = False
            else:
                self.batch_queue_empty.emit()
            self._batch_run_indices = None
            if shared.HEADLESS_CONTINUOUS:
                LOGGER.info('Enter next dirs to translate (comma-separated), or "exit" to quit.')
                try:
                    new_exec_dirs = input()
                except (EOFError, KeyboardInterrupt):
                    new_exec_dirs = 'exit'
                if new_exec_dirs.strip().lower() == 'exit':
                    LOGGER.info('Exiting.')
                    self.app.quit()
                    return
                self.run_batch(new_exec_dirs)
                return
            if shared.HEADLESS:
                self.app.quit()
            return
        d = self.exec_dirs.pop(0)
        self._current_batch_dir = d
        self.batch_queue_item_started.emit(d)
        LOGGER.info(f'translating {d} ...')
        self.openDir(d)
        all_pages = list(self.imgtrans_proj.pages.keys())
        skip_ignored = getattr(self, '_batch_skip_ignored', True)
        if getattr(self, '_batch_run_indices', None) is not None:
            # Use same page indices as first item (e.g. skip page 1 in every chapter)
            pages_to_process = [all_pages[i] for i in self._batch_run_indices if i < len(all_pages)]
        else:
            # First item: compute from this project's ignore state, then store indices for rest of batch
            if skip_ignored:
                pages_to_process = [p for p in all_pages if not self.imgtrans_proj.is_page_ignored(p)]
            else:
                pages_to_process = list(all_pages)
            self._batch_run_indices = [all_pages.index(p) for p in pages_to_process]
        shared.pbar = {}
        npages = len(pages_to_process)
        if npages > 0:
            if pcfg.module.enable_detect:
                shared.pbar['detect'] = tqdm(range(npages), desc="Text Detection")
            if pcfg.module.enable_ocr:
                shared.pbar['ocr'] = tqdm(range(npages), desc="OCR")
            if pcfg.module.enable_translate:
                shared.pbar['translate'] = tqdm(range(npages), desc="Translation")
            if pcfg.module.enable_inpaint:
                shared.pbar['inpaint'] = tqdm(range(npages), desc="Inpaint")
            self.on_run_imgtrans(
                skip_ignored_in_batch=skip_ignored,
                pages_to_process=pages_to_process,
            )

    def on_environment_doctor(self):
        from utils.environment_doctor import run_environment_doctor
        checks = run_environment_doctor()
        lines = [f"{name}: {status} ({detail})" for name, status, detail in checks]
        create_info_dialog({'title': self.tr('Environment doctor report'), 'text': '\n'.join(lines)})

    def on_translation_qa_report_current_page(self):
        if self.imgtrans_proj is None or not getattr(self.imgtrans_proj, 'current_img', None):
            create_info_dialog({'title': self.tr('No page loaded'), 'text': self.tr('Open a project/page first.')})
            return
        page = self.imgtrans_proj.current_img
        blocks = self.imgtrans_proj.pages.get(page, []) or []
        glossary = getattr(self.imgtrans_proj, 'translation_glossary', None) or []
        profile = getattr(pcfg.module, 'translation_prompt_profile_default', 'dialogue')
        retry_threshold = int(getattr(pcfg.module, 'translation_qa_retry_issue_threshold', 2) or 2)
        report = build_translation_qa_report(blocks, glossary, profile=profile, retry_issue_threshold=retry_threshold)
        from utils.translation_review import extract_glossary_candidates
        src_texts = [getattr(b, 'text', '') or '' for b in blocks]
        trans_texts = [getattr(b, 'translation', '') or '' for b in blocks]
        candidates = extract_glossary_candidates(src_texts, trans_texts, min_freq=2)[:20]
        lines = [self.tr('Page: ') + str(page), self.tr('Profile: ') + str(profile), self.tr('Blocks: ') + str(report['total_blocks'])]
        lines.append(self.tr('Issue blocks: ') + str(report['issue_blocks']))
        lines.append(self.tr('Retry candidates (>= issue threshold): ') + ', '.join(str(i+1) for i in report.get('retry_candidates', [])) if report.get('retry_candidates') else self.tr('none'))
        for row in report.get('rows', [])[:80]:
            for issue in row.get('issues', []):
                lines.append(f"#{row['index']+1}: {issue}")
        if candidates:
            lines.append(self.tr('Glossary candidates (source -> target):'))
            lines.extend([f"- {c.get('source','')} -> {c.get('target','')}" for c in candidates])
        create_info_dialog({'title': self.tr('Translation QA report'), 'text': '\n'.join(lines)})

    def on_save_run_profile_snapshot(self):
        if self.imgtrans_proj is None:
            return
        from qtpy.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, self.tr('Save run profile'), self.tr('Profile name:'))
        if not ok or not str(name).strip():
            return
        cfg = pcfg.module
        snap = {
            'textdetector': getattr(cfg, 'textdetector', ''),
            'ocr': getattr(cfg, 'ocr', ''),
            'inpainter': getattr(cfg, 'inpainter', ''),
            'translator': getattr(cfg, 'translator', ''),
        }
        if not isinstance(getattr(self.imgtrans_proj, 'run_profiles', None), dict):
            self.imgtrans_proj.run_profiles = {}
        self.imgtrans_proj.run_profiles[str(name).strip()] = snap
        self.imgtrans_proj.save()
        create_info_dialog({'title': self.tr('Saved'), 'text': self.tr('Run profile saved to project.')})

    def on_apply_run_profile_snapshot(self):
        if self.imgtrans_proj is None:
            return
        profiles = getattr(self.imgtrans_proj, 'run_profiles', None) or {}
        if not profiles:
            create_info_dialog({'title': self.tr('No profiles'), 'text': self.tr('No run profiles found in this project.')})
            return
        from qtpy.QtWidgets import QInputDialog
        names = sorted(list(profiles.keys()))
        name, ok = QInputDialog.getItem(self, self.tr('Apply run profile'), self.tr('Select profile:'), names, 0, False)
        if not ok or not name:
            return
        snap = profiles.get(name) or {}
        if snap.get('textdetector'):
            self.module_manager.setTextDetector(snap['textdetector'])
        if snap.get('ocr'):
            self.module_manager.setOCR(snap['ocr'])
        if snap.get('inpainter'):
            self.module_manager.setInpainter(snap['inpainter'])
        if snap.get('translator'):
            self.module_manager.setTranslator(snap['translator'])
        create_info_dialog({'title': self.tr('Applied'), 'text': self.tr('Run profile applied.')})

    def on_open_ocr_triage_current_page(self):
        if self.imgtrans_proj is None or not getattr(self.imgtrans_proj, 'current_img', None):
            create_info_dialog({'title': self.tr('No page loaded'), 'text': self.tr('Open a project/page first.')})
            return
        from utils.translation_review import check_translation_guardrails
        from ui.ocr_triage_dialog import OcrTriageDialog
        page = self.imgtrans_proj.current_img
        blks = self.imgtrans_proj.pages.get(page, []) or []
        glossary = getattr(self.imgtrans_proj, 'translation_glossary', None) or []
        rows = []
        seen_rows = set()
        triage_state = (self.imgtrans_proj._image_info.get(page, {}) or {}).get('triage_flags', {})
        for i, b in enumerate(blks):
            src = (getattr(b, 'text', '') or '').strip()
            trn = (getattr(b, 'translation', '') or '').strip()

            def _add_issue(issue_text: str):
                key = (i, issue_text)
                if key in seen_rows:
                    return
                seen_rows.add(key)
                rows.append({'block': i + 1, 'source': src, 'translation': trn, 'issue': issue_text})

            state = triage_state.get(str(i), 'none')
            if state != 'none':
                _add_issue(self.tr('Triage state: ') + state)
            if not src:
                _add_issue(self.tr('Empty OCR source text'))
            for issue in check_translation_guardrails(src, trn, glossary=glossary):
                _add_issue(issue)
        if not rows:
            create_info_dialog({'title': self.tr('OCR triage worklist'), 'text': self.tr('No triage issues found on current page.')})
            return
        dlg = OcrTriageDialog(rows, self)
        dlg.open_block_requested.connect(self.jump_to_canvas_block)
        dlg.mark_open_requested.connect(self.on_add_selected_to_triage)
        dlg.mark_reviewed_requested.connect(self.on_mark_selected_triage_reviewed)
        dlg.exec()

    def on_auto_extract_glossary_current_page(self):
        if self.imgtrans_proj is None or not getattr(self.imgtrans_proj, 'current_img', None):
            create_info_dialog({'title': self.tr('No page loaded'), 'text': self.tr('Open a project/page first.')})
            return
        from utils.translation_review import extract_glossary_candidates
        page = self.imgtrans_proj.current_img
        blks = self.imgtrans_proj.pages.get(page, []) or []
        src_texts = [getattr(b, 'text', '') or '' for b in blks]
        candidates = extract_glossary_candidates(src_texts, None, min_freq=2)
        if not candidates:
            create_info_dialog({'title': self.tr('Auto-extract glossary'), 'text': self.tr('No glossary candidates found on current page.')})
            return
        existing = getattr(self.imgtrans_proj, 'translation_glossary', None) or []
        seen = {str(g.get('source', '')).strip() for g in existing if isinstance(g, dict)}
        added = []
        for c in candidates:
            src = str(c.get('source', '')).strip()
            if not src or src in seen:
                continue
            existing.append({'source': src, 'target': ''})
            seen.add(src)
            added.append(src)
        self.imgtrans_proj.translation_glossary = existing
        self.imgtrans_proj.save()
        preview = ', '.join(added[:20]) if added else self.tr('None (all candidates already existed).')
        create_info_dialog({'title': self.tr('Auto-extract glossary complete'), 'text': self.tr('Added terms: {0}').format(preview)})

    def on_add_selected_to_triage(self, idx_list: list):
        if self.imgtrans_proj is None or not getattr(self.imgtrans_proj, 'current_img', None):
            return
        page = self.imgtrans_proj.current_img
        self.imgtrans_proj.add_triage_flags(page, idx_list)
        self.imgtrans_proj.save()
        create_info_dialog({'title': self.tr('Triage updated'), 'text': self.tr('Selected blocks added to triage worklist.')})

    def on_mark_selected_triage_reviewed(self, idx_list: list):
        if self.imgtrans_proj is None or not getattr(self.imgtrans_proj, 'current_img', None):
            return
        page = self.imgtrans_proj.current_img
        self.imgtrans_proj.mark_triage_reviewed(page, idx_list)
        self.imgtrans_proj.save()
        create_info_dialog({'title': self.tr('Triage updated'), 'text': self.tr('Selected blocks marked as reviewed.')})

    def setupRegisterWidget(self):
        self.titleBar.viewMenu.addSeparator()
        for cfg_name in shared.config_name_to_view_widget:
            d = shared.config_name_to_view_widget[cfg_name]
            widget: ViewWidget = d['widget']
            action = QAction(widget.action_name, self.titleBar)
            action.setCheckable(True)
            visible = getattr(pcfg, cfg_name)
            action.setChecked(visible)
            action.triggered.connect(self.action_set_view_visible)
            self.titleBar.viewMenu.addAction(action)
            d['action'] = action
            shared.action_to_view_config_name[action] = cfg_name
            widget.set_expend_area(expend=getattr(pcfg, widget.config_expand_name), set_config=False)
            widget.view_hide_btn_clicked.connect(self.on_hide_view_widget)
            widget.setVisible(visible)

        if hasattr(self, 'textPanel') and hasattr(self.textPanel, 'formatpanel'):
            self.textPanel.formatpanel.showGlobalFontFormatBtn.setVisible(
                not getattr(pcfg, 'show_text_style_preset', True)
            )

        # Section 10: Canvas view mode (original / debug / translated / normal) + Fit to width
        self.titleBar.viewMenu.addSeparator()
        from qtpy.QtGui import QActionGroup
        self._canvas_view_group = QActionGroup(self)
        self._canvas_view_group.setExclusive(True)
        mode_actions = []
        for mode, label in [
            ("normal", self.tr("Canvas: Normal")),
            ("original", self.tr("Canvas: Original")),
            ("debug", self.tr("Canvas: Debug (boxes/masks)")),
            ("translated", self.tr("Canvas: Translated")),
        ]:
            act = QAction(label, self.titleBar)
            act.setCheckable(True)
            act.setChecked(getattr(pcfg, "canvas_view_mode", "normal") == mode)
            act.setProperty("canvas_view_mode", mode)
            act.triggered.connect(self._on_canvas_view_mode_triggered)
            self._canvas_view_group.addAction(act)
            self.titleBar.viewMenu.addAction(act)
            mode_actions.append((mode, act))
        self._canvas_view_mode_actions = dict(mode_actions)
        fit_action = QAction(self.tr("Canvas: Fit to width"), self.titleBar)
        fit_action.setToolTip(self.tr("Zoom canvas so image width fits the view."))
        fit_action.triggered.connect(self._on_canvas_fit_to_width)
        self.titleBar.viewMenu.addAction(fit_action)

    def _on_canvas_view_mode_triggered(self):
        action = self.sender()
        if not isinstance(action, QAction):
            return
        mode = action.property("canvas_view_mode")
        if mode:
            pcfg.canvas_view_mode = mode
            self.canvas.updateLayers()

    def _on_canvas_fit_to_width(self):
        self.canvas.fitToWidth()

    def action_set_view_visible(self):
        action: QAction = self.sender()
        show = action.isChecked()
        cfg_name = shared.action_to_view_config_name[action]
        widget: ViewWidget = shared.config_name_to_view_widget[cfg_name]['widget']
        widget.setVisible(show)
        setattr(pcfg, cfg_name, show)
        if cfg_name == 'show_text_style_preset' and hasattr(self, 'textPanel'):
            self.textPanel.formatpanel.showGlobalFontFormatBtn.setVisible(not show)
        if cfg_name == 'text_advanced_format_panel' and hasattr(self, 'textPanel'):
            self.textPanel.formatpanel.showAdvancedFontFormatBtn.setVisible(not show)

    def on_hide_view_widget(self, cfg_name: str):
        d = shared.config_name_to_view_widget[cfg_name]
        widget: ViewWidget = d['widget']
        widget.setVisible(False)
        action: QAction = d['action']
        action.setChecked(False)
        setattr(pcfg, cfg_name, False)
        if cfg_name == 'show_text_style_preset' and hasattr(self, 'textPanel'):
            self.textPanel.formatpanel.showGlobalFontFormatBtn.setVisible(True)
        if cfg_name == 'text_advanced_format_panel' and hasattr(self, 'textPanel'):
            self.textPanel.formatpanel.showAdvancedFontFormatBtn.setVisible(True)


    def on_concordance_search_current_project(self):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            create_info_dialog({'title': self.tr('Concordance search'), 'text': self.tr('Open a project first.')})
            return
        text, ok = QInputDialog.getText(self, self.tr('Concordance search'), self.tr('Find in source/target:'))
        if not ok or not str(text).strip():
            return
        rows = build_concordance_from_project(self.imgtrans_proj)
        hits = query_concordance(rows, text, in_target=True, limit=200)
        if not hits:
            create_info_dialog({'title': self.tr('Concordance search'), 'text': self.tr('No matches found.')})
            return
        lines = [self.tr('Matches: {0}').format(len(hits))]
        for h in hits[:50]:
            lines.append(f"[{h.get('page','')}] {h.get('source','')} -> {h.get('target','')}")
        if len(hits) > 50:
            lines.append(self.tr('...and {0} more').format(len(hits)-50))
        create_info_dialog({'title': self.tr('Concordance search'), 'text': '\n'.join(lines)})


    def on_import_glossary_with_preview(self):
        if self.imgtrans_proj is None or self.imgtrans_proj.is_empty:
            QMessageBox.information(self, self.tr('Import'), self.tr('Open a project first.'))
            return
        in_path, _ = QFileDialog.getOpenFileName(self, self.tr('Import glossary'), self.imgtrans_proj.directory, self.tr('Glossary (*.json *.csv)'))
        if not in_path:
            return
        mode, ok = QInputDialog.getItem(self, self.tr('Import mode'), self.tr('Mode'), [self.tr('merge'), self.tr('replace')], 0, False)
        if not ok:
            return
        mode_v = 'replace' if str(mode).lower().startswith('replace') else 'merge'
        try:
            preview = self._api_glossary_import_preview_ui({'path': in_path, 'mode': mode_v})
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to preview glossary import'))
            return
        text = self.tr('Incoming: {0}\nWill add: {1}\nSkipped: {2}\nResult total: {3}').format(preview.get('incoming_count', 0), preview.get('added_count', 0), preview.get('skipped_count', 0), preview.get('result_count', 0))
        yn = QMessageBox.question(self, self.tr('Glossary import preview'), text + '\n\n' + self.tr('Apply now?'), QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if yn != QMessageBox.StandardButton.Yes:
            return
        try:
            rst = self._api_glossary_import_ui({'path': in_path, 'mode': mode_v})
            create_info_dialog({'title': self.tr('Glossary import'), 'text': self.tr('Added entries: {0} (total: {1})').format(rst.get('added', 0), rst.get('entries', 0))})
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to import glossary'))
