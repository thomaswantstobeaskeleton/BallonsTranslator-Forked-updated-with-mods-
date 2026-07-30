import json
import time
import threading
import copy
from typing import Union, List, Dict, Callable
import os.path as osp

import cv2
import numpy as np
from qtpy.QtCore import QThread, Signal, QObject, QLocale, QTimer, QMutex, QWaitCondition
from qtpy.QtWidgets import QFileDialog, QMessageBox, QApplication
from qtpy.QtGui import QClipboard

from .funcmaps import get_maskseg_method
from utils.logger import logger as LOGGER
from utils.imgproc_utils import enlarge_window, union_area, get_block_mask
from utils.registry import Registry
from modules.inpaint.base import _clip_xyxy_to_image, _block_mask_polygon, _block_mask_polygons, build_mask_with_resolved_overlaps, _apply_block_text_mask_to_mask
from utils.io_utils import imread, text_is_empty
from modules.translators import MissingTranslatorParams
from modules.translators.exceptions import CriticalTranslationError
from modules.base import BaseModule, soft_empty_cache, GPUINTENSIVE_SET
from modules import INPAINTERS, TRANSLATORS, TEXTDETECTORS, OCR, \
    GET_VALID_TRANSLATORS, GET_VALID_TEXTDETECTORS, GET_VALID_INPAINTERS, GET_VALID_OCR, \
    BaseTranslator, InpainterBase, TextDetectorBase, OCRBase, merge_config_module_params
from modules.ocr.base import normalize_block_text, ensure_blocks_have_lines
import modules
modules.translators.SYSTEM_LANG = QLocale.system().name()
from utils.textblock import TextBlock, sort_regions, examine_textblk, remove_contained_boxes, deduplicate_primary_boxes
from modules.textdetector.outside_text_processor import OSB_LABELS, filter_osb_overlapping_bubbles
from utils import shared
from utils.message import create_error_dialog, create_info_dialog
from utils.translator_test import test_translator
from utils.series_context_store import DEFAULT_SERIES_ID, get_series_context_dir, ensure_series_dir
from .custom_widget import ImgtransProgressMessageBox, ParamComboBox
from .configpanel import ConfigPanel
from utils.proj_imgtrans import ProjImgTrans
from utils.config import pcfg, RunStatus, log_diagnostic_event
from utils.image_upscale import (
    apply_initial_upscale,
    downscale_to_size,
    processing_scale as get_processing_scale,
)
from utils.cancellation import CancelFlag
cfg_module = pcfg.module


def _model_missing_guidance(module_label: str, module_name: str = "") -> str:
    suffix = f" ({module_name})" if module_name else ""
    return (
        f"{module_label}{suffix} is not ready: required model files are missing or incompatible.\n"
        "Open Tools → Manage models to check/download/import local models, then select an available module."
    )


class ModuleThread(QThread):

    finish_set_module = Signal()
    _failed_set_module_msg = 'Failed to set module.'
    module_thread_stopped = Signal()

    def __init__(self, module_key: str, MODULE_REGISTER: Registry, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.job = None
        self.module: Union[TextDetectorBase, BaseTranslator, InpainterBase, OCRBase] = None
        self.module_register = MODULE_REGISTER
        self.module_key = module_key

        self.pipeline_pagekey_queue = []
        self.finished_counter = 0
        self.num_process_pages = 0
        self.imgtrans_proj: ProjImgTrans = None
        self.stop_requested = False
        # True while the loaded module is a substitute for an unavailable one; the
        # user's saved selection must not be overwritten in that case (issue #152).
        self.module_fallback_used = False

    def _set_module(self, module_name: str, is_fallback: bool = False):
        old_module = self.module
        self.module_fallback_used = is_fallback
        register = self.module_register
        valid_keys = None
        if self.module_key == 'textdetector':
            valid_keys = GET_VALID_TEXTDETECTORS()
        elif self.module_key == 'ocr':
            valid_keys = GET_VALID_OCR()
        elif self.module_key == 'translator':
            valid_keys = GET_VALID_TRANSLATORS()
        if module_name not in register.module_dict or (valid_keys is not None and module_name not in valid_keys):
            fallback = None
            if self.module_key == 'textdetector':
                valid = GET_VALID_TEXTDETECTORS()
                fallback = "ctd" if "ctd" in valid else (valid[0] if valid else None)
            elif self.module_key == 'ocr':
                valid = GET_VALID_OCR()
                fallback = valid[0] if valid else None
            elif self.module_key == 'translator':
                valid = GET_VALID_TRANSLATORS()
                fallback = "google" if "google" in valid else (valid[0] if valid else None)
            elif self.module_key == 'inpainter':
                valid = GET_VALID_INPAINTERS()
                fallback = valid[0] if valid else None
            if fallback:
                # Keep the configured module in cfg_module: it may become available again
                # once its model files are downloaded (issue #152).
                self.module_fallback_used = True
                LOGGER.warning(
                    "Module '%s' (%s) is not available (e.g. failed to load). Using '%s'.",
                    module_name, self.module_key, fallback,
                )
                module_name = fallback
            else:
                create_error_dialog(
                    KeyError(module_name),
                    self._failed_set_module_msg + " " + self.tr("No fallback available.") + "\n"
                    + self.tr("Open Tools → Manage models to check or import local model files."),
                )
                self.finish_set_module.emit()
                return
        try:
            module: Union[TextDetectorBase, BaseTranslator, InpainterBase, OCRBase] \
                = register.module_dict[module_name]
            params = cfg_module.get_params(self.module_key).get(module_name)
            if params is not None:
                self.module = module(**params)
            else:
                self.module = module()
            if not pcfg.module.load_model_on_demand:
                self.module.load_model()
            if old_module is not None:
                del old_module
        except Exception as e:
            self.module = old_module
            create_error_dialog(e, self._failed_set_module_msg)

        self.finish_set_module.emit()

    def pipeline_finished(self):
        if self.imgtrans_proj is None:
            return True
        elif self.finished_counter >= self.num_process_pages:
            return True
        return False

    def initImgtransPipeline(self, proj: ProjImgTrans):
        if self.isRunning():
            self.terminate()
        self.imgtrans_proj = proj
        self.finished_counter = 0
        self.pipeline_pagekey_queue.clear()

    def requestStop(self):
        self.stop_requested = True

    def run(self):
        if self.job is not None:
            self.job()
        self.job = None


class InpaintThread(ModuleThread):

    finish_inpaint = Signal(dict)
    inpainting = False    
    inpaint_failed = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__('inpainter', INPAINTERS, *args, **kwargs)

    @property
    def inpainter(self) -> InpainterBase:
        return self.module

    def setInpainter(self, inpainter: str, is_fallback: bool = False):
        self.job = lambda : self._set_module(inpainter, is_fallback)
        self.start()

    def inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect=None):
        self.job = lambda : self._inpaint(img, mask, img_key, inpaint_rect)
        self.start()
    
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect=None):
        inpaint_dict = {}
        self.inpainting = True
        try:
            inpainted = self.inpainter.inpaint(img, mask)
            inpaint_dict = {
                'inpainted': inpainted,
                'img': img,
                'mask': mask,
                'img_key': img_key,
                'inpaint_rect': inpaint_rect
            }
            self.finish_inpaint.emit(inpaint_dict)
        except Exception as e:
            create_error_dialog(e, self.tr('Inpainting Failed.'), 'InpaintFailed')
            self.inpainting = False
            self.inpaint_failed.emit()
        self.inpainting = False


class TextDetectThread(ModuleThread):
    
    finish_detect_page = Signal(str)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__('textdetector', TEXTDETECTORS, *args, **kwargs)

    def setTextDetector(self, textdetector: str, is_fallback: bool = False):
        self.job = lambda : self._set_module(textdetector, is_fallback)
        self.start()

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.module


class OCRThread(ModuleThread):

    finish_ocr_page = Signal(str)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__('ocr', OCR, *args, **kwargs)

    def setOCR(self, ocr: str, is_fallback: bool = False):
        self.job = lambda : self._set_module(ocr, is_fallback)
        self.start()
    
    @property
    def ocr(self) -> OCRBase:
        return self.module


class TranslateThread(ModuleThread):

    finish_translate_page = Signal(str)
    progress_changed = Signal(int)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__('translator', TRANSLATORS, *args, **kwargs)
        self.translator: BaseTranslator = self.module

    def _set_translator(self, translator: str, is_fallback: bool = False):

        old_translator = self.translator
        self.module_fallback_used = is_fallback
        source, target = cfg_module.translate_source, cfg_module.translate_target
        if self.translator is not None:
            if self.translator.name == translator:
                return

        if translator not in TRANSLATORS.module_dict:
            valid = GET_VALID_TRANSLATORS()
            fallback = 'google' if 'google' in valid else (valid[0] if valid else None)
            if fallback:
                # Keep cfg_module.translator: the selection stays valid once the
                # translator becomes available again (issue #152).
                self.module_fallback_used = True
                LOGGER.warning(
                    "Translator '%s' is not available (e.g. failed to load). Using '%s'.",
                    translator, fallback,
                )
                translator = fallback
            else:
                create_error_dialog(KeyError(translator), self.tr('Failed to set translator. No fallback available.'))
                self.finish_set_module.emit()
                return
        try:
            params = cfg_module.translator_params.get(translator)
            translator_module: BaseTranslator = TRANSLATORS.module_dict[translator]
            if params is not None:
                self.translator = translator_module(source, target, raise_unsupported_lang=False, **params)
            else:
                self.translator = translator_module(source, target, raise_unsupported_lang=False)
            cfg_module.translate_source = self.translator.lang_source
            cfg_module.translate_target = self.translator.lang_target
            if not self.module_fallback_used:
                cfg_module.translator = self.translator.name
        except Exception as e:
            if old_translator is None:
                valid = GET_VALID_TRANSLATORS()
                fallback_name = 'google' if 'google' in valid else (valid[0] if valid else None)
                if fallback_name:
                    old_translator = TRANSLATORS.module_dict[fallback_name](
                        '简体中文', 'English', raise_unsupported_lang=False
                    )
            self.translator = old_translator
            msg = self.tr('Failed to set translator ') + translator
            create_error_dialog(e, msg, 'FailedSetTranslator')

        self.module = self.translator
        self.finish_set_module.emit()

    def setTranslator(self, translator: str, is_fallback: bool = False):
        if translator in ['Sugoi']:
            self._set_translator(translator, is_fallback)
        else:
            self.job = lambda : self._set_translator(translator, is_fallback)
            self.start()

    def _translate_page(self, page_dict, page_key: str, emit_finished=True):
        page = page_dict[page_key]
        proj = getattr(self, "imgtrans_proj", None)
        series_path = ""
        if proj is not None:
            series_path = (getattr(proj, "series_context_path", None) or "").strip()
        if not series_path and getattr(self.translator, "params", None) and "series_context_path" in self.translator.params:
            series_path = (self.translator.get_param_value("series_context_path") or "").strip()
        if not series_path:
            series_path = DEFAULT_SERIES_ID
        # Ensure data/translation_context/default (or chosen series) exists so translators don't fail (Issue #6).
        if series_path and hasattr(self.translator, "set_translation_context"):
            dir_path = get_series_context_dir(series_path)
            if dir_path:
                ensure_series_dir(dir_path)
        if proj is not None and hasattr(self.translator, "set_translation_context"):
            ordered = list(proj.pages.keys())
            if page_key in ordered:
                idx = ordered.index(page_key)
                n = getattr(
                    self.translator,
                    "context_previous_pages_count",
                    0,
                )
                prev_keys = ordered[max(0, idx - n) : idx]
                previous_pages = []
                for k in prev_keys:
                    blks = proj.pages.get(k, [])
                    previous_pages.append({
                        "sources": [blk.get_text() for blk in blks],
                        "translations": [getattr(blk, "translation", "") or "" for blk in blks],
                    })
                next_page = None
                if idx + 1 < len(ordered):
                    next_key = ordered[idx + 1]
                    blks = proj.pages.get(next_key, [])
                    if blks:
                        next_page = {"sources": [blk.get_text() for blk in blks]}
                self.translator.set_translation_context(
                    previous_pages,
                    getattr(proj, "translation_glossary", None) or [],
                    series_context_path=series_path or None,
                    next_page=next_page,
                )
        if proj is not None and hasattr(self.translator, "set_translation_project"):
            self.translator.set_translation_project(proj, page_key)
        skip = (
            getattr(cfg_module, 'skip_already_translated', False)
            and page
            and all(getattr(b, 'translation', None) and str(b.translation).strip() for b in page)
        )
        if skip:
            if series_path and hasattr(self.translator, "append_page_to_series_context"):
                sources = [blk.get_text() for blk in page]
                translations = [getattr(blk, "translation", "") or "" for blk in page]
                self.translator.append_page_to_series_context(series_path, sources, translations)
        else:
            trans_from_cache = False
            if getattr(cfg_module, 'translation_cache_enabled', False) and page and self.imgtrans_proj is not None:
                try:
                    from utils.pipeline_cache_manager import (
                        get_pipeline_cache_manager,
                        _generate_image_hash,
                        _settings_hash,
                    )
                    img = self.imgtrans_proj.read_img(page_key)
                    if img is not None:
                        cm = get_pipeline_cache_manager()
                        image_hash = _generate_image_hash(img)
                        trans_name = getattr(cfg_module, 'translator', '') or ''
                        src_lang = getattr(cfg_module, 'translate_source', '') or ''
                        tgt_lang = getattr(cfg_module, 'translate_target', '') or ''
                        ctx_dict = dict(getattr(cfg_module, 'translator_params', {}).get(trans_name, {}))
                        ctx_dict["_series_context_path"] = series_path
                        ctx_hash = _settings_hash(ctx_dict)
                        trans_key = cm.get_translation_cache_key(image_hash, trans_name, src_lang, tgt_lang, ctx_hash)
                        if cm.can_serve_all_blocks_from_translation_cache(trans_key, page):
                            cm.apply_cached_translation_to_blocks(trans_key, page)
                            trans_from_cache = True
                except Exception as _e:
                    LOGGER.debug("Translation cache check failed: %s", _e)
            if not trans_from_cache:
                try:
                    setattr(self.translator, '_current_page_key', page_key)
                    _page_img = self.imgtrans_proj.read_img(page_key) if self.imgtrans_proj else None
                    setattr(self.translator, '_current_page_image', _page_img)
                    self.translator.translate_textblk_lst(page)
                except CriticalTranslationError as e:
                    create_error_dialog(e, self.tr('Translation Failed.') + f' (page: {page_key})', 'TranslationFailed')
                    self.stop_requested = True
                except Exception as e:
                    if not getattr(cfg_module, "translation_soft_failure_continue", True):
                        create_error_dialog(e, self.tr('Translation Failed.') + f' (page: {page_key})', 'TranslationFailed')
                        self.stop_requested = True
                    else:
                        LOGGER.warning("Translation failed (soft) for page %s: %s. Using placeholders and continuing.", page_key, e)
                        placeholder = "[Translation failed]"
                        for blk in page:
                            if getattr(blk, "translation", None) is None or str(blk.translation).strip() == "":
                                blk.translation = placeholder
                else:
                    if getattr(cfg_module, 'translation_cache_enabled', False) and page and self.imgtrans_proj is not None:
                        try:
                            from utils.pipeline_cache_manager import (
                                get_pipeline_cache_manager,
                                _generate_image_hash,
                                _settings_hash,
                            )
                            img = self.imgtrans_proj.read_img(page_key)
                            if img is not None:
                                cm = get_pipeline_cache_manager()
                                image_hash = _generate_image_hash(img)
                                trans_name = getattr(cfg_module, 'translator', '') or ''
                                src_lang = getattr(cfg_module, 'translate_source', '') or ''
                                tgt_lang = getattr(cfg_module, 'translate_target', '') or ''
                                ctx_dict = dict(getattr(cfg_module, 'translator_params', {}).get(trans_name, {}))
                                ctx_dict["_series_context_path"] = series_path
                                ctx_hash = _settings_hash(ctx_dict)
                                trans_key = cm.get_translation_cache_key(image_hash, trans_name, src_lang, tgt_lang, ctx_hash)
                                cm.cache_translation_results(trans_key, page)
                        except Exception as _e:
                            LOGGER.debug("Translation cache store failed: %s", _e)
                    if series_path and hasattr(self.translator, "append_page_to_series_context"):
                        sources = [blk.get_text() for blk in page]
                        translations = [getattr(blk, "translation", "") or "" for blk in page]
                        self.translator.append_page_to_series_context(series_path, sources, translations)
            elif series_path and hasattr(self.translator, "append_page_to_series_context"):
                sources = [blk.get_text() for blk in page]
                translations = [getattr(blk, "translation", "") or "" for blk in page]
                self.translator.append_page_to_series_context(series_path, sources, translations)
        if emit_finished:
            self.finish_translate_page.emit(page_key)

    def translatePage(self, page_dict, page_key: str):
        self.job = lambda: self._translate_page(page_dict, page_key)
        self.start()

    def push_pagekey_queue(self, page_key: str):
        self.pipeline_pagekey_queue.append(page_key)

    def runTranslatePipeline(self, imgtrans_proj: ProjImgTrans):
        self.initImgtransPipeline(imgtrans_proj)
        self.job = self._run_translate_pipeline
        self.start()


    def _run_translate_pipeline(self):
        delay = self.translator.delay()

        while not self.pipeline_finished():
            if self.stop_requested:
                self.module_thread_stopped.emit()
                self.stop_requested = False
                break

            if len(self.pipeline_pagekey_queue) == 0:
                time.sleep(0.1)
                continue
            
            page_key = self.pipeline_pagekey_queue.pop(0)
            self.blockSignals(True)
            trans_success = True
            try:
                setattr(self.translator, '_current_page_key', page_key)
                self._translate_page(self.imgtrans_proj.pages, page_key, emit_finished=False)
            except Exception as e:
                trans_success = False
                msg = self.tr('Translation Failed.')
                if isinstance(e, MissingTranslatorParams):
                    msg = msg + '\n' + str(e) + self.tr(' is required for ' + self.translator.name)
                self.blockSignals(False)
                manager = getattr(self, 'manager', None)
                if manager is not None:
                    create_error_dialog(e, msg + f' (page: {page_key})', 'TranslationFailed')
                    manager.translation_failure_request.emit(msg + f' (page: {page_key})', page_key)
                    manager._trans_failure_mutex.lock()
                    manager._trans_failure_condition.wait(manager._trans_failure_mutex)
                    choice = getattr(manager, '_trans_failure_choice', 'skip')
                    manager._trans_failure_choice = None
                    manager._trans_failure_mutex.unlock()
                    if choice == 'terminate':
                        self.stop_requested = True
                        self.pipeline_pagekey_queue.clear()
                        self.finished_counter += 1
                        self.progress_changed.emit(self.finished_counter)
                        break
                    elif choice == 'retry':
                        self.pipeline_pagekey_queue.insert(0, page_key)
                        self.blockSignals(True)
                        if not self.pipeline_finished() and delay > 0:
                            time.sleep(delay)
                        continue
                else:
                    create_error_dialog(e, msg + f' (page: {page_key})', 'TranslationFailed')
            self.blockSignals(False)
            self.finished_counter += 1
            if trans_success:
                # Record the target language so the page can seed LLM history only
                # while it matches the current target.
                self.imgtrans_proj.mark_translation_finished(
                    page_key, getattr(self.translator, 'lang_target', '')
                )
            self.progress_changed.emit(self.finished_counter)

            if not self.pipeline_finished() and delay > 0:
                time.sleep(delay)


class ImgtransThread(QThread):

    pipeline_stopped = Signal()
    update_detect_progress = Signal(int)
    update_ocr_progress = Signal(int)
    update_translate_progress = Signal(int)
    update_inpaint_progress = Signal(int)

    finish_blktrans_stage = Signal(str, int)
    finish_blktrans = Signal(int, list)
    unload_modules = Signal(list)

    detect_counter = 0
    ocr_counter = 0
    translate_counter = 0
    inpaint_counter = 0

    def __init__(self, 
                 textdetect_thread: TextDetectThread,
                 ocr_thread: OCRThread,
                 translate_thread: TranslateThread,
                 inpaint_thread: InpaintThread,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.textdetect_thread = textdetect_thread
        self.ocr_thread = ocr_thread
        self.translate_thread = translate_thread
        self.translate_thread.module_thread_stopped.connect(self.on_module_thread_stopped)
        self.inpaint_thread = inpaint_thread
        self.job = None
        self.imgtrans_proj: ProjImgTrans = None
        self.stop_requested = False
        self.pages_to_process = None  # 需要处理的页面列表（用于继续运行模式）
        self._pause_requested = False
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._last_batch_report = None

    def on_module_thread_stopped(self):
        while True:
            # might freeze UI
            if self.translate_thread.isRunning() or self.inpaint_thread.isRunning() or self.ocr_thread.isRunning() or self.textdetect_thread.isRunning():
                time.sleep(0.05)
                continue
            break

        self.pipeline_stopped.emit()

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.textdetect_thread.textdetector

    def _run_extra_detectors(self, img: np.ndarray, mask: np.ndarray, blk_list: List[TextBlock], im_w: int, im_h: int, extra_names: List[str]):
        """Run one or more extra detectors and merge blocks (IoU < 0.4). Used for secondary and tertiary."""
        if not extra_names or blk_list is None:
            blk_list = blk_list or []
        valid = GET_VALID_TEXTDETECTORS()
        primary_name = (getattr(cfg_module, 'textdetector', '') or '').strip()
        iou_threshold = 0.4
        outside_bubble_only = getattr(cfg_module, 'secondary_detector_outside_bubble_only', False)
        primary_blocks = list(blk_list) if outside_bubble_only else []

        def _iou(a: TextBlock, b: TextBlock) -> float:
            inter = union_area(a.xyxy, b.xyxy)
            if inter <= 0:
                return 0.0
            area_a = (a.xyxy[2] - a.xyxy[0]) * (a.xyxy[3] - a.xyxy[1])
            area_b = (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1])
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        for sec_name in extra_names:
            sec_name = (sec_name or '').strip()
            if not sec_name or sec_name not in valid or sec_name == primary_name:
                continue
            try:
                sec_class = TEXTDETECTORS.module_dict.get(sec_name)
                if sec_class is None:
                    continue
                merged_params = merge_config_module_params(
                    copy.deepcopy(cfg_module.get_params('textdetector')),
                    valid,
                    TEXTDETECTORS.get
                )
                params = merged_params.get(sec_name)
                if isinstance(params, dict):
                    params = {k: v for k, v in params.items() if not k.startswith('__')}
                sec_detector = sec_class(**params) if params and isinstance(params, dict) else sec_class()
                if not pcfg.module.load_model_on_demand:
                    sec_detector.load_model()
            except Exception as e:
                LOGGER.warning('Could not create extra detector %s: %s', sec_name, e)
                continue
            try:
                mask2, blk_list_2 = sec_detector.detect(img, self.imgtrans_proj)
            except Exception as e:
                LOGGER.warning('Extra detector %s failed: %s', sec_name, e)
                continue
            if blk_list_2 is None:
                blk_list_2 = []
            if outside_bubble_only and primary_blocks:
                blk_list_2 = filter_osb_overlapping_bubbles(blk_list_2, primary_blocks, iou_threshold=0.10)
            for blk in blk_list_2:
                safe_xyxy = _clip_xyxy_to_image(blk.xyxy, im_w, im_h)
                if safe_xyxy is None:
                    continue
                blk.xyxy = safe_xyxy
                if getattr(blk, 'lines', None) and len(blk.lines) > 0:
                    examine_textblk(blk, im_w, im_h, sort=True)
                if outside_bubble_only:
                    blk_list.append(blk)
                else:
                    max_iou = max((_iou(blk, p) for p in blk_list), default=0.0)
                    if max_iou < iou_threshold:
                        blk_list.append(blk)
            blk_list = sort_regions(blk_list)
            if mask is not None and mask2 is not None and mask.shape == mask2.shape:
                mask = np.bitwise_or(mask, mask2)
            elif mask is None and mask2 is not None:
                mask = mask2
        return mask, blk_list

    @staticmethod
    def _clip_detection_blocks_to_image(blk_list: List[TextBlock], im_w: int, im_h: int) -> List[TextBlock]:
        """Clip all block xyxy and lines to image bounds; drop blocks that are fully outside or degenerate.
        Prevents off-page or invalid boxes from detectors (e.g. rare bad page)."""
        if not blk_list:
            return blk_list
        out = []
        for blk in blk_list:
            safe_xyxy = _clip_xyxy_to_image(getattr(blk, "xyxy", None), im_w, im_h)
            if safe_xyxy is None:
                continue
            blk.xyxy = safe_xyxy
            lines = getattr(blk, "lines", None)
            if lines and len(lines) > 0:
                clipped_lines = []
                for line in lines:
                    if line is None or len(line) < 3:
                        continue
                    pts = []
                    for p in line:
                        try:
                            x, y = float(p[0]), float(p[1])
                            x = max(0.0, min(x, im_w))
                            y = max(0.0, min(y, im_h))
                            pts.append([x, y])
                        except (TypeError, ValueError, IndexError):
                            continue
                    if len(pts) >= 3:
                        clipped_lines.append(pts)
                if clipped_lines:
                    blk.lines = clipped_lines
                else:
                    # Fallback: one line from bbox so block remains valid
                    x1, y1, x2, y2 = safe_xyxy
                    blk.lines = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
            out.append(blk)
        if len(out) < len(blk_list):
            LOGGER.debug(
                "Clipped detection blocks to image: kept %d of %d (dropped %d out-of-bounds or degenerate)",
                len(out), len(blk_list), len(blk_list) - len(out),
            )
        return out

    def _run_dual_detect(self, img: np.ndarray, mask: np.ndarray, blk_list: List[TextBlock], im_w: int, im_h: int):
        """Section 14: Secondary (and optionally tertiary) detector merge. Builds extra list and calls _run_extra_detectors."""
        primary = (getattr(cfg_module, 'textdetector', '') or '').strip()
        secondary = (getattr(cfg_module, 'textdetector_secondary', '') or '').strip()
        tertiary = (getattr(cfg_module, 'textdetector_tertiary', '') or '').strip()
        extra = []
        if getattr(cfg_module, 'enable_dual_detect', False) and secondary and secondary != primary:
            extra.append(secondary)
        if getattr(cfg_module, 'enable_tertiary_detect', False) and tertiary and tertiary != primary and tertiary not in extra:
            extra.append(tertiary)
        if not extra:
            return mask, blk_list
        return self._run_extra_detectors(img, mask, blk_list, im_w, im_h, extra)

    @property
    def ocr(self) -> OCRBase:
        return self.ocr_thread.ocr
    
    @property
    def translator(self) -> BaseTranslator:
        return self.translate_thread.translator

    @property
    def inpainter(self) -> InpainterBase:
        return self.inpaint_thread.inpainter

    def runImgtransPipeline(self, imgtrans_proj: ProjImgTrans, pages_to_process=None):
        self.imgtrans_proj = imgtrans_proj
        self.pages_to_process = pages_to_process  # 保存需要处理的页面列表
        self.num_pages = len(self.imgtrans_proj.pages)
        self.stop_requested = False
        self.cancel_flag = CancelFlag()
        self.cancel_flag.reset()
        self._pause_requested = False
        self._resume_event.set()
        # 创建处理索引到实际页面索引的映射
        self.process_idx_to_page_idx = {}
        self.job = self._imgtrans_pipeline
        self.start()
    
    def requestStop(self):
        """请求停止当前任务"""
        if self.isRunning():
            self.stop_requested = True
        if getattr(self, 'cancel_flag', None) is not None:
            self.cancel_flag.request()
        self._pause_requested = False
        self._resume_event.set()
        # 同时停止翻译线程
        if self.translate_thread.isRunning():
            self.translate_thread.requestStop()

    def requestPause(self):
        """Pause the pipeline (batch queue); pipeline will wait until requestResume()."""
        self._pause_requested = True
        self._resume_event.clear()

    def requestResume(self):
        """Resume the pipeline after pause."""
        self._pause_requested = False
        self._resume_event.set()

    def runBlktransPipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int, blk_ids: List[int], tgt_mask):
        self.job = lambda : self._blktrans_pipeline(blk_list, tgt_img, mode, blk_ids, tgt_mask)
        self.start()

    @staticmethod
    def _cleaning_kwargs_for_block(blk_list: List[TextBlock], idx: int, x1: int, y1: int, x2: int, y2: int) -> dict:
        """Section 16: self_xyxy and neighbor_xyxy_list in crop coords for adaptive shrink near conjoined junctions."""
        blk = blk_list[idx]
        crop_w, crop_h = x2 - x1, y2 - y1
        try:
            bx1, by1, bx2, by2 = float(blk.xyxy[0]), float(blk.xyxy[1]), float(blk.xyxy[2]), float(blk.xyxy[3])
        except (IndexError, TypeError):
            return {}
        self_xyxy = [bx1 - x1, by1 - y1, bx2 - x1, by2 - y1]
        margin = 20
        neighbor_xyxy_list = []
        for j, other in enumerate(blk_list):
            if j == idx:
                continue
            try:
                ox1, oy1, ox2, oy2 = float(other.xyxy[0]), float(other.xyxy[1]), float(other.xyxy[2]), float(other.xyxy[3])
            except (IndexError, TypeError):
                continue
            if not (ox2 < x1 - margin or ox1 > x2 + margin or oy2 < y1 - margin or oy1 > y2 + margin):
                nc = [max(0.0, ox1 - x1), max(0.0, oy1 - y1), min(float(crop_w), ox2 - x1), min(float(crop_h), oy2 - y1)]
                if nc[2] > nc[0] and nc[3] > nc[1]:
                    neighbor_xyxy_list.append(nc)
        return {"self_xyxy": self_xyxy, "neighbor_xyxy_list": neighbor_xyxy_list}

    def _blktrans_pipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int, blk_ids: List[int], tgt_mask):
        if mode >= 0 and mode < 3:
            try:
                self.ocr_thread.module.run_ocr(tgt_img, blk_list, split_textblk=True)
            except Exception as e:
                create_error_dialog(e, self.tr('OCR Failed.'), 'OCRFailed')
            self.finish_blktrans.emit(mode, blk_ids)

        if mode != 0 and mode < 3:
            try:
                self.translate_thread.module.translate_textblk_lst(blk_list)
            except CriticalTranslationError as e:
                create_error_dialog(e, self.tr('Translation Failed.'), 'TranslationFailed')
            except Exception as e:
                if not getattr(pcfg.module, "translation_soft_failure_continue", True):
                    create_error_dialog(e, self.tr('Translation Failed.'), 'TranslationFailed')
                else:
                    LOGGER.warning("Translation failed (soft): %s. Using placeholders.", e)
                    for blk in blk_list:
                        if getattr(blk, "translation", None) is None or str(blk.translation).strip() == "":
                            blk.translation = "[Translation failed]"
            self.finish_blktrans.emit(mode, blk_ids)
        if mode > 1:
            im_h, im_w = tgt_img.shape[:2]
            progress_prod = 100. / len(blk_list) if len(blk_list) > 0 else 0
            for ii, blk in enumerate(blk_list):
                xyxy = enlarge_window(blk.xyxy, im_w, im_h)
                xyxy = np.array(xyxy)
                x1, y1, x2, y2 = xyxy.astype(np.int64)
                blk.region_inpaint_dict = None
                if y2 - y1 > 2 and x2 - x1 > 2:
                    im = np.copy(tgt_img[y1: y2, x1: x2])
                    maskseg_method = get_maskseg_method()
                    cleaning_kwargs = self._cleaning_kwargs_for_block(blk_list, ii, int(x1), int(y1), int(x2), int(y2))
                    inpaint_mask_array, ballon_mask, bub_dict = maskseg_method(
                        im, mask=tgt_mask[y1: y2, x1: x2], **cleaning_kwargs
                    )
                    # Dilate mask slightly so original text (e.g. Chinese) is fully covered and erased.
                    # Using a 7x7 kernel instead of 5x5 because tight detectors (e.g. YSGYOLO) can leave
                    # anti-aliased text edges un-masked, which then remain visible after inpaint (#935).
                    if inpaint_mask_array is not None and inpaint_mask_array.size > 0:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        inpaint_mask_array = cv2.dilate(
                            (inpaint_mask_array > 127).astype(np.uint8) * 255, kernel, iterations=1
                        )
                    mask = self.post_process_mask(inpaint_mask_array)
                    if mask is not None and mask.sum() > 0:
                        inpainted = self.inpaint_thread.inpainter.inpaint(im, mask)
                        blk.region_inpaint_dict = {'img': im, 'mask': mask, 'inpaint_rect': [x1, y1, x2, y2], 'inpainted': inpainted}
                    self.finish_blktrans_stage.emit('inpaint', int((ii+1) * progress_prod))
        self.finish_blktrans.emit(mode, blk_ids)

    def _set_translation_context_for_page(self, imgname: str, pages_to_iterate: list):
        """Set previous-page context and project glossary on translator before translating this page."""
        if not hasattr(self.translator, "set_translation_context"):
            return
        if imgname not in pages_to_iterate:
            return
        series_path = (getattr(self.imgtrans_proj, "series_context_path", None) or "").strip()
        if not series_path and getattr(self.translator, "params", None) and "series_context_path" in self.translator.params:
            series_path = (self.translator.get_param_value("series_context_path") or "").strip()
        if not series_path:
            series_path = DEFAULT_SERIES_ID
        dir_path = get_series_context_dir(series_path)
        if dir_path:
            ensure_series_dir(dir_path)
        idx = pages_to_iterate.index(imgname)
        n = getattr(self.translator, "context_previous_pages_count", 0)
        prev_keys = pages_to_iterate[max(0, idx - n) : idx]
        previous_pages = []
        for k in prev_keys:
            blks = self.imgtrans_proj.pages.get(k, [])
            previous_pages.append({
                "sources": [blk.get_text() for blk in blks],
                "translations": [getattr(blk, "translation", "") or "" for blk in blks],
            })
        next_page = None
        if idx + 1 < len(pages_to_iterate):
            next_key = pages_to_iterate[idx + 1]
            blks = self.imgtrans_proj.pages.get(next_key, [])
            if blks:
                next_page = {"sources": [blk.get_text() for blk in blks]}
        self.translator.set_translation_context(
            previous_pages,
            getattr(self.imgtrans_proj, "translation_glossary", None) or [],
            series_context_path=series_path or None,
            next_page=next_page,
        )
        # Context-aware LLM translation: snapshot the reusable history window and glossary
        # for this page (upstream dmMaze/BallonsTranslator llm_context).
        if hasattr(self.translator, "set_translation_project"):
            self.translator.set_translation_project(self.imgtrans_proj, imgname)

    def _append_page_to_series_context(self, imgname: str, blk_list: list):
        """Append the translated page to the series context store for cross-chapter consistency."""
        if not blk_list or not hasattr(self.translator, "append_page_to_series_context"):
            return
        series_path = (getattr(self.imgtrans_proj, "series_context_path", None) or "").strip()
        if not series_path and getattr(self.translator, "params", None) and "series_context_path" in self.translator.params:
            series_path = (self.translator.get_param_value("series_context_path") or "").strip()
        if not series_path:
            series_path = DEFAULT_SERIES_ID
        sources = [blk.get_text() for blk in blk_list]
        translations = [getattr(blk, "translation", "") or "" for blk in blk_list]
        self.translator.append_page_to_series_context(series_path, sources, translations)

    def _imgtrans_pipeline(self):
        self.detect_counter = 0
        self.ocr_counter = 0
        self.translate_counter = 0
        self.inpaint_counter = 0

        def _run_stage_with_retry(stage_name: str, fn, retries: int = 1):
            attempts = max(1, int(retries))
            last_err = None
            for i in range(attempts):
                try:
                    return fn()
                except Exception as e:
                    last_err = e
                    LOGGER.warning('%s failed (attempt %d/%d): %s', stage_name, i + 1, attempts, e)
            if last_err is not None:
                raise last_err

        # 如果指定了pages_to_process，只处理这些页面
        all_pages = list(self.imgtrans_proj.pages.keys())
        if self.pages_to_process is not None and len(self.pages_to_process) > 0:
            pages_to_iterate = self.pages_to_process
            self.num_pages = num_pages = len(self.pages_to_process)
            # 建立处理索引到实际页面索引的映射
            for process_idx, page_name in enumerate(pages_to_iterate):
                if page_name in all_pages:
                    self.process_idx_to_page_idx[process_idx] = all_pages.index(page_name)
            LOGGER.info(f'Processing specific pages: {len(pages_to_iterate)} pages')
        else:
            pages_to_iterate = all_pages
            self.num_pages = num_pages = len(self.imgtrans_proj.pages)
            # 处理索引等于实际页面索引
            for i in range(num_pages):
                self.process_idx_to_page_idx[i] = i
            LOGGER.info(f'Processing all {num_pages} pages')
        self.textdetect_thread.num_process_pages = self.num_pages
        self.ocr_thread.num_process_pages = self.num_pages
        self.inpaint_thread.num_process_pages = self.num_pages
        self.translate_thread.num_process_pages = self.num_pages

        try:
            from utils.batch_report import start_batch_report
            start_batch_report(pages_to_iterate)
        except Exception:
            pass

        # Auto OCR by source language: optionally use a different OCR module for this run
        self._auto_ocr_instance = None
        self._auto_ocr_key = None
        if cfg_module.enable_ocr and getattr(cfg_module, 'ocr_auto_by_language', False):
            try:
                from utils.ocr_lang_mapping import get_ocr_key_for_language
                from modules import OCR
                fallback = getattr(cfg_module, 'ocr', None) or 'mit48px'
                effective_key = get_ocr_key_for_language(
                    getattr(cfg_module, 'translate_source', '') or '', fallback
                )
                valid_ocr = GET_VALID_OCR()
                if effective_key != getattr(cfg_module, 'ocr', None) and effective_key in valid_ocr:
                    ocr_cls = OCR.module_dict.get(effective_key)
                    if ocr_cls is not None:
                        params = cfg_module.get_params('ocr').get(effective_key)
                        self._auto_ocr_instance = ocr_cls(**(params or {}))
                        self._auto_ocr_instance.load_model()
                        self._auto_ocr_key = effective_key
                        LOGGER.info('Auto OCR by language: using %s for source %s', effective_key, getattr(cfg_module, 'translate_source', ''))
            except Exception as e:
                LOGGER.warning('Auto OCR by language failed: %s. Using default OCR.', e)

        low_vram_trans = False
        if self.translator is not None:
            low_vram_trans = self.translator.low_vram_mode
            self.parallel_trans = not self.translator.is_computational_intensive() and not low_vram_trans
        else:
            self.parallel_trans = False
        if self.parallel_trans and cfg_module.enable_translate:
            self.translate_thread.runTranslatePipeline(self.imgtrans_proj)

        page_num = 0
        for imgname in pages_to_iterate:
            page_num += 1
            # 检查是否请求停止
            if self.stop_requested or (getattr(self, 'cancel_flag', None) is not None and self.cancel_flag.is_set()):
                LOGGER.info('Image translation pipeline stopped by user')
                break
            while self._pause_requested and not self.stop_requested and (getattr(self, 'cancel_flag', None) is None or not self.cancel_flag.is_set()):
                self._resume_event.wait(timeout=0.3)
            if self.stop_requested or (getattr(self, 'cancel_flag', None) is not None and self.cancel_flag.is_set()):
                break
            LOGGER.info(f'Page {page_num}/{len(pages_to_iterate)} ({imgname}): starting')
            log_diagnostic_event(
                "pipeline.page_start",
                page_index=page_num - 1,
                page_name=imgname,
                total_pages=len(pages_to_iterate),
            )
            if cfg_module.enable_ocr and hasattr(self.ocr, 'restore_to_device'):
                try:
                    self.ocr.restore_to_device()
                except Exception as e:
                    LOGGER.warning("OCR restore_to_device failed (will try to continue): %s", e)
            img = self.imgtrans_proj.read_img(imgname)
            orig_h, orig_w = img.shape[:2]
            scale = 1.0
            if getattr(cfg_module, 'image_upscale_initial', False):
                try:
                    factor = float(getattr(cfg_module, 'image_upscale_initial_factor', 2.0) or 2.0)
                    policy = (getattr(cfg_module, 'upscale_policy_initial', None) or 'lanczos').strip().lower()
                    if factor > 1.0 and policy != 'none':
                        img = apply_initial_upscale(img, factor=factor, policy=policy)
                        scale = factor
                except Exception as e:
                    LOGGER.warning("Initial upscale failed: %s", e)
            mask = blk_list = None
            need_save_mask = False
            blk_removed: List[TextBlock] = []
            if cfg_module.enable_detect:
                log_diagnostic_event(
                    "pipeline.detect_start",
                    page_index=page_num - 1,
                    page_name=imgname,
                    block_count=len(blk_list) if blk_list is not None else len(self.imgtrans_proj.pages.get(imgname, []) or []),
                )
                if self.textdetector is None:
                    create_error_dialog(
                        RuntimeError("Text detector module is not set or failed to load."),
                        self.tr(_model_missing_guidance("Text detector", getattr(cfg_module, "textdetector", ""))),
                        "DetectorNotSet",
                    )
                    self.stop_requested = True
                    break
                try:
                    mask, blk_list = self.textdetector.detect(img, self.imgtrans_proj)
                    need_save_mask = True
                    im_h, im_w = img.shape[:2]
                    blk_list = self._clip_detection_blocks_to_image(blk_list, im_w, im_h)
                    for blk in blk_list:
                        if getattr(blk, 'lines', None) and len(blk.lines) > 0:
                            examine_textblk(blk, im_w, im_h, sort=True)
                    # Optional: collision-based merge (Dango-style) for word-level or many small blocks.
                    # Skip when block count is low (typical bubble layout) to avoid merging separate bubbles and losing boxes.
                    if getattr(cfg_module, 'merge_nearby_blocks_collision', False) and blk_list:
                        try:
                            from utils.ocr_result_merge import merge_blocks_horizontal, merge_blocks_vertical
                            # Only merge when we have many blocks (likely word-level or scattered); typical manga has ~5–15 bubbles per page.
                            merge_min_blocks = int(getattr(cfg_module, 'merge_nearby_blocks_min_blocks', 18) or 18)
                            if len(blk_list) >= merge_min_blocks:
                                gap = float(getattr(cfg_module, 'merge_nearby_blocks_gap_ratio', 1.5) or 1.5)
                                vertical_count = sum(1 for b in blk_list if getattr(b, 'vertical', False) or getattr(getattr(b, 'fontformat', None), 'vertical', False))
                                if vertical_count > len(blk_list) / 2:
                                    blk_list = merge_blocks_vertical(blk_list, gap_ratio=gap)
                                else:
                                    blk_list = merge_blocks_horizontal(blk_list, gap_ratio=gap, add_space_between=False)
                                if mask is not None and blk_list:
                                    mask = build_mask_with_resolved_overlaps(blk_list, im_w, im_h)
                                    need_save_mask = True
                        except Exception as e:
                            LOGGER.warning('Collision merge failed: %s', e)
                    # Section 14: primary detection deduplication and nested-box removal
                    blk_list = remove_contained_boxes(blk_list)
                    blk_list = deduplicate_primary_boxes(blk_list, iou_threshold=0.5)
                    if (getattr(cfg_module, 'enable_dual_detect', False) and
                            getattr(cfg_module, 'textdetector_secondary', '').strip() and
                            cfg_module.textdetector_secondary != cfg_module.textdetector) or \
                            (getattr(cfg_module, 'enable_tertiary_detect', False) and
                            getattr(cfg_module, 'textdetector_tertiary', '').strip() and
                            cfg_module.textdetector_tertiary != cfg_module.textdetector):
                        try:
                            mask, blk_list = self._run_dual_detect(img, mask, blk_list, im_w, im_h)
                        except Exception as e:
                            LOGGER.warning('Dual text detection failed: %s', e)
                except Exception as e:
                    LOGGER.error("Text detection failed for page: %s", imgname, exc_info=True)
                    try:
                        from utils.batch_report import register_batch_skip
                        register_batch_skip(imgname, "detection", str(e))
                    except Exception:
                        pass
                    create_error_dialog(
                        e,
                        self.tr('Text Detection Failed.') + f' (page: {imgname})\n'
                        + self.tr(_model_missing_guidance("Text detector", getattr(cfg_module, "textdetector", ""))),
                        'TextDetectFailed',
                    )
                    blk_list = []
                self.detect_counter += 1
                if pcfg.module.keep_exist_textlines:
                    blk_list = self.imgtrans_proj.pages[imgname] + blk_list
                    blk_list = sort_regions(blk_list)
                    existed_mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                    if existed_mask is not None:
                        mask = np.bitwise_or(mask, existed_mask)
                # Optional: outside-speech-bubble (OSB) processing for detectors that set blk.label (e.g. HF object det "text_free").
                if getattr(cfg_module, "enable_osb_pipeline", False):
                    try:
                        from modules.textdetector.outside_text_processor import (
                            split_blocks_by_label,
                            filter_osb_overlapping_bubbles,
                            group_osb_blocks,
                            apply_osb_style_defaults,
                            expand_bubble_boxes_with_osb,
                        )

                        bubble_blks, osb_blks, other_blks = split_blocks_by_label(blk_list)
                        osb_blks = filter_osb_overlapping_bubbles(
                            osb_blks,
                            bubble_blks,
                            iou_threshold=float(getattr(cfg_module, "osb_exclude_bubble_iou", 0.10) or 0.10),
                        )
                        # Section 14: expand bubble boxes to contain overlapping OSB text (so SAM/mask includes all text)
                        if cfg_module.osb_expand_bubbles_with_osb:
                            expand_bubble_boxes_with_osb(bubble_blks, osb_blks, im_w, im_h)
                        if getattr(cfg_module, "osb_group_nearby", True):
                            gap_px = int(getattr(cfg_module, "osb_group_gap_px", 24) or 24)
                            if getattr(cfg_module, "processing_scale_enabled", True):
                                try:
                                    pscale = get_processing_scale(im_w, im_h)
                                    gap_px = max(4, int(round(gap_px * pscale)))
                                except Exception:
                                    pass
                            osb_blks = group_osb_blocks(
                                osb_blks,
                                gap_px=gap_px,
                            )
                        if getattr(cfg_module, "osb_style_probe", False):
                            apply_osb_style_defaults(img, osb_blks)
                        blk_list = bubble_blks + other_blks + osb_blks
                        blk_list = sort_regions(blk_list)
                        # Keep mask consistent with updated blocks for "detect-only" runs.
                        if mask is not None:
                            im_h, im_w = img.shape[:2]
                            if getattr(cfg_module, "resolve_mask_overlaps_bisector", True):
                                mask = build_mask_with_resolved_overlaps(
                                    blk_list, im_w, im_h,
                                    text_blocks_for_nudge=osb_blks if osb_blks else None,
                                )
                            else:
                                mask = np.zeros((im_h, im_w), dtype=np.uint8)
                                for blk in blk_list:
                                    for pts in _block_mask_polygons(blk, im_w, im_h):
                                        if pts is not None and len(pts) >= 3:
                                            cv2.fillPoly(mask, [pts], 255)
                            need_save_mask = True
                    except Exception as e:
                        LOGGER.warning("OSB processing failed: %s", e)
                # Optional: panel-aware ordering (improves translation context order and rendering sequence)
                if getattr(cfg_module, "enable_panel_order", False):
                    try:
                        from modules.textdetector.panel_finder import reorder_textblocks_by_panels

                        blk_list = reorder_textblocks_by_panels(
                            img,
                            blk_list,
                            reading_direction=getattr(cfg_module, "panel_reading_direction", "auto"),
                        )
                    except Exception as e:
                        LOGGER.warning("Panel reorder failed: %s", e)
                # Final clip so no block is saved off-page (safety for any detector/merge path)
                im_h, im_w = img.shape[:2]
                blk_list = self._clip_detection_blocks_to_image(blk_list, im_w, im_h)
                # Don't overwrite existing blocks with empty detection result (preserves boxes when detector fails or returns nothing)
                existing = self.imgtrans_proj.pages.get(imgname) or []
                if blk_list or len(existing) == 0:
                    self.imgtrans_proj.pages[imgname] = blk_list
                else:
                    LOGGER.warning("Detection returned no blocks for page %s; keeping %d existing block(s)", imgname, len(existing))
                    blk_list = existing  # run OCR/translate/inpaint on existing blocks

                if mask is not None and not cfg_module.enable_ocr:
                    mask_to_save = downscale_to_size(mask, orig_w, orig_h) if scale > 1 else mask
                    self.imgtrans_proj.save_mask(imgname, mask_to_save)
                    need_save_mask = False
                    
                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_DET)
                self.update_detect_progress.emit(self.detect_counter)
                LOGGER.info(f'Page {page_num}/{len(pages_to_iterate)}: detection done')
                log_diagnostic_event(
                    "pipeline.detect_finish",
                    page_index=page_num - 1,
                    page_name=imgname,
                    block_count=len(blk_list) if blk_list is not None else 0,
                )

            # Replace translation mode: load translated image, detect+OCR on it, match blocks, set raw blk.translation (manga-translator-ui style)
            if (getattr(cfg_module, "replace_translation_mode", False) and
                getattr(cfg_module, "replace_translation_translated_dir", "").strip() and
                blk_list and mask is not None and self.textdetector is not None and self.ocr is not None):
                try:
                    trans_dir = (cfg_module.replace_translation_translated_dir or "").strip()
                    if osp.isabs(trans_dir):
                        trans_img_path = osp.join(trans_dir, imgname)
                    else:
                        trans_img_path = osp.join(self.imgtrans_proj.directory, trans_dir, imgname)
                    if not osp.isfile(trans_img_path):
                        LOGGER.warning("Replace translation: translated image not found: %s", trans_img_path)
                    else:
                        img_trans = imread(trans_img_path)
                        if img_trans is not None:
                            if img_trans.shape[:2] != img.shape[:2]:
                                img_trans = cv2.resize(img_trans, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                            mask_trans, blk_list_trans = self.textdetector.detect(img_trans, self.imgtrans_proj)
                            self.ocr.run_ocr(img_trans, blk_list_trans)
                            def _box_area(xyxy):
                                if not xyxy or len(xyxy) != 4:
                                    return 0.0
                                return max(0.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
                            def _iou(a_xyxy, b_xyxy):
                                inter = union_area(a_xyxy, b_xyxy)
                                if inter <= 0:
                                    return 0.0
                                sa = _box_area(a_xyxy)
                                sb = _box_area(b_xyxy)
                                return inter / (sa + sb - inter) if (sa + sb - inter) > 0 else 0.0
                            for raw_blk in blk_list:
                                ra = getattr(raw_blk, "xyxy", None)
                                if not ra or len(ra) != 4:
                                    continue
                                best_iou, best_text = 0.0, ""
                                for trans_blk in blk_list_trans:
                                    tb = getattr(trans_blk, "xyxy", None)
                                    if not tb or len(tb) != 4:
                                        continue
                                    iou = _iou(ra, tb)
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_text = (trans_blk.get_text() or "").strip() or getattr(trans_blk, "translation", "") or ""
                                if best_iou > 0.2 and best_text:
                                    raw_blk.translation = best_text
                            self.imgtrans_proj.pages[imgname] = blk_list
                            LOGGER.info("Replace translation: applied translations from %s", trans_img_path)
                        else:
                            LOGGER.warning("Replace translation: failed to load %s", trans_img_path)
                except Exception as e:
                    LOGGER.warning("Replace translation failed for page %s: %s", imgname, e)

            if blk_list is None:
                blk_list = self.imgtrans_proj.pages[imgname] if imgname in self.imgtrans_proj.pages else []

            # When image is upscaled but detection was skipped, blocks from project are in original resolution.
            # Ensure every block has lines (from xyxy if missing), then scale block coordinates to upscaled image.
            if scale > 1.0 and blk_list and not cfg_module.enable_detect:
                ensure_blocks_have_lines(blk_list)
                for blk in blk_list:
                    if getattr(blk, 'xyxy', None):
                        blk.xyxy = [x * scale for x in blk.xyxy]
                    if getattr(blk, 'lines', None) and len(blk.lines) > 0:
                        blk.lines = [[[p[0] * scale, p[1] * scale] for p in line] for line in blk.lines]

            if cfg_module.enable_ocr:
                LOGGER.info(f'Page {page_num}/{len(pages_to_iterate)}: OCR running')
                log_diagnostic_event(
                    "pipeline.ocr_start",
                    page_index=page_num - 1,
                    page_name=imgname,
                    block_count=len(blk_list) if blk_list is not None else 0,
                )
                ocr_runner = getattr(self, '_auto_ocr_instance', None) if getattr(cfg_module, 'ocr_auto_by_language', False) else None
                if ocr_runner is None:
                    ocr_runner = self.ocr
                ocr_name_for_cache = getattr(self, '_auto_ocr_key', None) or getattr(cfg_module, 'ocr', '') or ''
                try:
                    ocr_ran = False
                    if getattr(cfg_module, 'ocr_cache_enabled', True) and blk_list:
                        try:
                            from utils.pipeline_cache_manager import (
                                get_pipeline_cache_manager,
                                _generate_image_hash,
                            )
                            cm = get_pipeline_cache_manager()
                            image_hash = _generate_image_hash(img)
                            source_lang = getattr(cfg_module, 'translate_source', '') or ''
                            device = getattr(ocr_runner, 'device', '') or '' if ocr_runner else ''
                            ocr_cache_key = cm.get_ocr_cache_key(image_hash, ocr_name_for_cache, source_lang, device)
                            if cm.can_serve_all_blocks_from_ocr_cache(ocr_cache_key, blk_list):
                                cm.apply_cached_ocr_to_blocks(ocr_cache_key, blk_list)
                                for blk in blk_list:
                                    normalize_block_text(blk)
                                ocr_ran = True
                        except Exception as _e:
                            LOGGER.debug("OCR cache check failed: %s", _e)
                    if not ocr_ran:
                        # Optional one-step VLM translation: OCR module writes blk.translation directly from image.
                        if (
                            getattr(cfg_module, "translation_mode", "two_step") == "one_step_vlm"
                            and getattr(cfg_module, "enable_translate", False)
                            and hasattr(ocr_runner, "run_ocr_translate")
                        ):
                            try:
                                _run_stage_with_retry(
                                    'OCR+Translate',
                                    lambda: ocr_runner.run_ocr_translate(img, blk_list, cfg_module.translate_source, cfg_module.translate_target),
                                    getattr(pcfg, 'pipeline_retry_ocr', 1),
                                )
                                # Ensure source text exists for later editing (keep original OCR empty)
                                for blk in blk_list:
                                    if getattr(blk, "text", None) is None:
                                        blk.text = [""]
                            except NotImplementedError:
                                _run_stage_with_retry('OCR', lambda: ocr_runner.run_ocr(img, blk_list), getattr(pcfg, 'pipeline_retry_ocr', 1))
                        else:
                            _run_stage_with_retry('OCR', lambda: ocr_runner.run_ocr(img, blk_list), getattr(pcfg, 'pipeline_retry_ocr', 1))
                        if getattr(cfg_module, 'ocr_cache_enabled', True) and blk_list:
                            try:
                                from utils.pipeline_cache_manager import (
                                    get_pipeline_cache_manager,
                                    _generate_image_hash,
                                )
                                cm = get_pipeline_cache_manager()
                                image_hash = _generate_image_hash(img)
                                source_lang = getattr(cfg_module, 'translate_source', '') or ''
                                device = getattr(ocr_runner, 'device', '') or '' if ocr_runner else ''
                                ocr_cache_key = cm.get_ocr_cache_key(image_hash, ocr_name_for_cache, source_lang, device)
                                cm.cache_ocr_results(ocr_cache_key, blk_list)
                            except Exception as _e:
                                LOGGER.debug("OCR cache store failed: %s", _e)
                except Exception as e:
                    LOGGER.error("OCR failed for page: %s", imgname, exc_info=True)
                    try:
                        from utils.batch_report import register_batch_skip
                        register_batch_skip(imgname, "ocr", str(e))
                    except Exception:
                        pass
                    create_error_dialog(
                        e,
                        self.tr('OCR Failed.') + f' (page: {imgname})\n'
                        + self.tr(_model_missing_guidance("OCR", getattr(cfg_module, "ocr", ""))),
                        'OCRFailed',
                    )
                self.ocr_counter += 1

                if pcfg.restore_ocr_empty:
                    blk_list_updated = []
                    for blk in blk_list:
                        text = blk.get_text()
                        if text_is_empty(text):
                            blk_removed.append(blk)
                        else:
                            blk_list_updated.append(blk)

                    if len(blk_removed) > 0:
                        LOGGER.info(
                            "Restore empty OCR: removed %d block(s) with empty OCR result (page %s). Turn off 'Restore empty OCR' in Config to keep all boxes.",
                            len(blk_removed),
                            imgname,
                        )
                        blk_list.clear()
                        blk_list += blk_list_updated
                        
                        if mask is None:
                            mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                        if mask is not None:
                            inpainted = None
                            if not cfg_module.enable_inpaint:
                                inpainted = self.imgtrans_proj.load_inpainted_by_imgname(imgname)
                            for blk in blk_removed:
                                xywh = blk.bounding_rect()
                                blk_mask, xyxy = get_block_mask(xywh, mask, blk.angle)
                                x1, y1, x2, y2 = xyxy
                                if blk_mask is not None:
                                    mask[y1: y2, x1: x2] = 0
                                    if inpainted is not None:
                                        mskpnt = np.where(blk_mask)
                                        inpainted[y1: y2, x1: x2][mskpnt] = img[y1: y2, x1: x2][mskpnt]
                                    need_save_mask = True
                            if inpainted is not None and need_save_mask:
                                inpainted_to_save = downscale_to_size(inpainted, orig_w, orig_h) if scale > 1 else inpainted
                                self.imgtrans_proj.save_inpainted(imgname, inpainted_to_save)
                            if need_save_mask:
                                mask_to_save = downscale_to_size(mask, orig_w, orig_h) if scale > 1 else mask
                                self.imgtrans_proj.save_mask(imgname, mask_to_save)
                                need_save_mask = False

                # Optional: after OCR, filter margin page numbers for OSB blocks.
                if getattr(cfg_module, "osb_page_number_filter", False):
                    try:
                        im_h, im_w = img.shape[:2]
                        from modules.textdetector.outside_text_processor import filter_page_number_blocks_after_ocr

                        blk_list_new, removed = filter_page_number_blocks_after_ocr(
                            blk_list,
                            im_w=im_w,
                            im_h=im_h,
                            margin_ratio=float(getattr(cfg_module, "osb_page_number_margin_ratio", 0.08) or 0.08),
                        )
                        if removed:
                            blk_list.clear()
                            blk_list += blk_list_new
                            # Clear their mask so they won't be inpainted.
                            if mask is None:
                                mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                            if mask is not None:
                                for blk in removed:
                                    xywh = blk.bounding_rect()
                                    blk_mask, xyxy = get_block_mask(xywh, mask, blk.angle)
                                    x1, y1, x2, y2 = xyxy
                                    if blk_mask is not None:
                                        mask[y1: y2, x1: x2] = 0
                                        need_save_mask = True
                            if need_save_mask and mask is not None:
                                mask_to_save = downscale_to_size(mask, orig_w, orig_h) if scale > 1 else mask
                                self.imgtrans_proj.save_mask(imgname, mask_to_save)
                                need_save_mask = False
                            # Ensure page JSON uses filtered list
                            self.imgtrans_proj.pages[imgname] = blk_list
                    except Exception as e:
                        LOGGER.warning("OSB page-number filter failed: %s", e)

                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_OCR)
                self.update_ocr_progress.emit(self.ocr_counter)
                LOGGER.info(f'Page {page_num}/{len(pages_to_iterate)}: OCR done')
                log_diagnostic_event(
                    "pipeline.ocr_finish",
                    page_index=page_num - 1,
                    page_name=imgname,
                    block_count=len(blk_list) if blk_list is not None else 0,
                )
                if cfg_module.enable_inpaint and getattr(self.ocr, 'device', None) in GPUINTENSIVE_SET and hasattr(self.ocr, 'offload_to_cpu'):
                    self.ocr.offload_to_cpu()
                    soft_empty_cache()

            if need_save_mask and mask is not None:
                mask_to_save = downscale_to_size(mask, orig_w, orig_h) if scale > 1 else mask
                self.imgtrans_proj.save_mask(imgname, mask_to_save)
                need_save_mask = False

            if cfg_module.enable_translate:
                log_diagnostic_event(
                    "pipeline.translate_start",
                    page_index=page_num - 1,
                    page_name=imgname,
                    block_count=len(blk_list) if blk_list is not None else 0,
                )
                # Skip translator stage for one-step VLM flow (already wrote blk.translation).
                if getattr(cfg_module, "translation_mode", "two_step") == "one_step_vlm":
                    self.translate_counter += 1
                    self.update_translate_progress.emit(self.translate_counter)
                    log_diagnostic_event(
                        "pipeline.translate_finish",
                        page_index=page_num - 1,
                        page_name=imgname,
                        block_count=len(blk_list) if blk_list is not None else 0,
                    )
                elif getattr(cfg_module, 'skip_already_translated', False) and blk_list and all(
                    getattr(b, 'translation', None) and str(b.translation).strip() for b in blk_list
                ):
                    self.translate_counter += 1
                    self.update_translate_progress.emit(self.translate_counter)
                    log_diagnostic_event(
                        "pipeline.translate_finish",
                        page_index=page_num - 1,
                        page_name=imgname,
                        block_count=len(blk_list) if blk_list is not None else 0,
                    )
                elif self.parallel_trans:
                    self.translate_thread.push_pagekey_queue(imgname)
                    log_diagnostic_event(
                        "pipeline.translate_queued",
                        page_index=page_num - 1,
                        page_name=imgname,
                        block_count=len(blk_list) if blk_list is not None else 0,
                    )
                elif not low_vram_trans:
                    self._set_translation_context_for_page(imgname, pages_to_iterate)
                    _series_path = (getattr(self.imgtrans_proj, "series_context_path", None) or "").strip()
                    if not _series_path and getattr(self.translator, "params", None) and "series_context_path" in self.translator.params:
                        _series_path = (self.translator.get_param_value("series_context_path") or "").strip()
                    if not _series_path:
                        _series_path = DEFAULT_SERIES_ID
                    trans_from_cache = False
                    if getattr(cfg_module, 'translation_cache_enabled', False) and blk_list:
                        try:
                            from utils.pipeline_cache_manager import (
                                get_pipeline_cache_manager,
                                _generate_image_hash,
                                _settings_hash,
                            )
                            cm = get_pipeline_cache_manager()
                            image_hash = _generate_image_hash(img)
                            trans_name = getattr(cfg_module, 'translator', '') or ''
                            src_lang = getattr(cfg_module, 'translate_source', '') or ''
                            tgt_lang = getattr(cfg_module, 'translate_target', '') or ''
                            ctx_dict = dict(getattr(cfg_module, 'translator_params', {}).get(trans_name, {}))
                            ctx_dict["_series_context_path"] = _series_path
                            ctx_hash = _settings_hash(ctx_dict)
                            trans_key = cm.get_translation_cache_key(image_hash, trans_name, src_lang, tgt_lang, ctx_hash)
                            if cm.can_serve_all_blocks_from_translation_cache(trans_key, blk_list):
                                cm.apply_cached_translation_to_blocks(trans_key, blk_list)
                                trans_from_cache = True
                        except Exception as _e:
                            LOGGER.debug("Translation cache check failed: %s", _e)
                    if not trans_from_cache:
                        try:
                            setattr(self.translator, '_current_page_key', imgname)
                            setattr(self.translator, '_current_page_image', img)
                            self.translator.translate_textblk_lst(blk_list)
                        except CriticalTranslationError as e:
                            create_error_dialog(e, self.tr('Translation Failed.') + f' (page: {imgname})', 'TranslationFailed')
                            self.stop_requested = True
                            break
                        except Exception as e:
                            if not getattr(cfg_module, "translation_soft_failure_continue", True):
                                create_error_dialog(e, self.tr('Translation Failed.') + f' (page: {imgname})', 'TranslationFailed')
                                self.stop_requested = True
                                break
                            LOGGER.warning("Translation failed (soft) for page %s: %s. Using placeholders and continuing.", imgname, e)
                            try:
                                from utils.batch_report import register_batch_skip
                                register_batch_skip(imgname, "translation", str(e))
                            except Exception:
                                pass
                            for blk in blk_list:
                                if getattr(blk, "translation", None) is None or str(blk.translation).strip() == "":
                                    blk.translation = "[Translation failed]"
                        if getattr(cfg_module, 'translation_cache_enabled', False) and blk_list:
                            try:
                                from utils.pipeline_cache_manager import (
                                    get_pipeline_cache_manager,
                                    _generate_image_hash,
                                    _settings_hash,
                                )
                                cm = get_pipeline_cache_manager()
                                image_hash = _generate_image_hash(img)
                                trans_name = getattr(cfg_module, 'translator', '') or ''
                                src_lang = getattr(cfg_module, 'translate_source', '') or ''
                                tgt_lang = getattr(cfg_module, 'translate_target', '') or ''
                                ctx_dict = dict(getattr(cfg_module, 'translator_params', {}).get(trans_name, {}))
                                ctx_dict["_series_context_path"] = _series_path
                                ctx_hash = _settings_hash(ctx_dict)
                                trans_key = cm.get_translation_cache_key(image_hash, trans_name, src_lang, tgt_lang, ctx_hash)
                                cm.cache_translation_results(trans_key, blk_list)
                            except Exception as _e:
                                LOGGER.debug("Translation cache store failed: %s", _e)
                    self._append_page_to_series_context(imgname, blk_list)
                    self.translate_counter += 1
                    self.update_translate_progress.emit(self.translate_counter)
                    log_diagnostic_event(
                        "pipeline.translate_finish",
                        page_index=page_num - 1,
                        page_name=imgname,
                        block_count=len(blk_list) if blk_list is not None else 0,
                    )
                        
            if cfg_module.enable_inpaint:
                LOGGER.info(f'Page {page_num}/{len(pages_to_iterate)}: inpainting (loading model if needed)')
                if mask is None:
                    mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                
                im_h, im_w = img.shape[:2]
                # When image was upscaled but we didn't run detection, blk_list is from project (original resolution).
                # Scale block coordinates only if we didn't already scale them before OCR (when OCR is enabled).
                if scale > 1.0 and blk_list and not cfg_module.enable_detect and not getattr(cfg_module, 'enable_ocr', True):
                    for blk in blk_list:
                        if getattr(blk, 'xyxy', None):
                            blk.xyxy = [x * scale for x in blk.xyxy]
                        if getattr(blk, 'lines', None) and len(blk.lines) > 0:
                            blk.lines = [[[p[0] * scale, p[1] * scale] for p in line] for line in blk.lines]
                # Build or rebuild mask from blk_list so mask and blocks are from the same source.
                # When mask was never saved (e.g. inpaint-only re-run), build from blocks so inpainting can run.
                if blk_list:
                    if getattr(cfg_module, "resolve_mask_overlaps_bisector", True):
                        text_for_nudge = [b for b in blk_list if (getattr(b, "label", None) or "").strip().lower() in OSB_LABELS]
                        mask = build_mask_with_resolved_overlaps(
                            blk_list, im_w, im_h,
                            text_blocks_for_nudge=text_for_nudge if text_for_nudge else None,
                        )
                    else:
                        mask = np.zeros((im_h, im_w), dtype=np.uint8)
                        for blk in blk_list:
                            for pts in _block_mask_polygons(blk, im_w, im_h):
                                if pts is not None and len(pts) >= 3:
                                    cv2.fillPoly(mask, [pts], 255)
                            _apply_block_text_mask_to_mask(mask, blk, im_w, im_h)
                elif mask is not None:
                    if mask.shape[0] != im_h or mask.shape[1] != im_w:
                        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_NEAREST)
                    if mask.ndim == 3:
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                
                # Per-block needs blocks; if no blocks, fall back to full-image so mask is still inpainted.
                blk_list_arg = None if getattr(cfg_module, 'inpaint_full_image', False) or not blk_list else blk_list
                    
                if mask is not None:
                    try:
                        inpainted = self.inpainter.inpaint(img, mask, blk_list_arg)
                        inpainted_to_save = downscale_to_size(inpainted, orig_w, orig_h) if scale > 1 else inpainted
                        self.imgtrans_proj.save_inpainted(imgname, inpainted_to_save)
                    except Exception as e:
                        LOGGER.error("Inpainting failed for page: %s", imgname, exc_info=True)
                        create_error_dialog(e, self.tr('Inpainting Failed.') + f' (page: {imgname})', 'InpaintFailed')
                    
                self.inpaint_counter += 1
                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_INPAINT)
                self.update_inpaint_progress.emit(self.inpaint_counter)
                LOGGER.info(f'Page {page_num}/{len(pages_to_iterate)}: inpainting done')
            else:
                if len(blk_removed) > 0:
                    self.imgtrans_proj.load_mask_by_imgname
            
            # Scale back block coordinates to original resolution when initial upscale was used
            if scale > 1.0 and blk_list:
                for blk in blk_list:
                    if getattr(blk, 'xyxy', None):
                        blk.xyxy = [x / scale for x in blk.xyxy]
                    if getattr(blk, 'lines', None) and len(blk.lines) > 0:
                        blk.lines = [[[p[0] / scale, p[1] / scale] for p in line] for line in blk.lines]

        # Release auto-OCR override so we don't hold an extra model
        self._auto_ocr_instance = None
        self._auto_ocr_key = None

        if cfg_module.enable_translate and low_vram_trans:
            unload_modules(self, ['textdetector', 'inpainter', 'ocr'])
            for imgname in pages_to_iterate:
                # 检查是否请求停止
                if self.stop_requested or (getattr(self, 'cancel_flag', None) is not None and self.cancel_flag.is_set()):
                    LOGGER.info('Translation stopped by user')
                    break
                while self._pause_requested and not self.stop_requested and (getattr(self, 'cancel_flag', None) is None or not self.cancel_flag.is_set()):
                    self._resume_event.wait(timeout=0.3)
                if self.stop_requested or (getattr(self, 'cancel_flag', None) is not None and self.cancel_flag.is_set()):
                    break
                self._set_translation_context_for_page(imgname, pages_to_iterate)
                blk_list = self.imgtrans_proj.pages[imgname]
                translation_failed = False
                skip = (
                    getattr(cfg_module, 'skip_already_translated', False)
                    and blk_list
                    and all(getattr(b, 'translation', None) and str(b.translation).strip() for b in blk_list)
                )
                if not skip:
                    try:
                        setattr(self.translator, '_current_page_key', imgname)
                        _page_img = self.imgtrans_proj.read_img(imgname) if self.imgtrans_proj else None
                        setattr(self.translator, '_current_page_image', _page_img)
                        self.translator.translate_textblk_lst(blk_list)
                    except CriticalTranslationError as e:
                        create_error_dialog(e, self.tr('Translation Failed.') + f' (page: {imgname})', 'TranslationFailed')
                        self.stop_requested = True
                        break
                    except Exception as e:
                        if not getattr(cfg_module, "translation_soft_failure_continue", True):
                            create_error_dialog(e, self.tr('Translation Failed.') + f' (page: {imgname})', 'TranslationFailed')
                            self.stop_requested = True
                            break
                        LOGGER.warning("Translation failed (soft) for page %s: %s. Using placeholders and continuing.", imgname, e)
                        try:
                            from utils.batch_report import register_batch_skip
                            register_batch_skip(imgname, "translation", str(e))
                        except Exception:
                            pass
                        translation_failed = True
                        for blk in blk_list:
                            if getattr(blk, "translation", None) is None or str(blk.translation).strip() == "":
                                blk.translation = "[Translation failed]"
                self._append_page_to_series_context(imgname, blk_list)
                self.translate_counter += 1
                if translation_failed:
                    self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_TRANSLATE)
                else:
                    # Record the target language so the page can seed LLM history only
                    # while it matches the current target.
                    self.imgtrans_proj.mark_translation_finished(
                        imgname, getattr(self.translator, 'lang_target', '')
                    )
                self.update_translate_progress.emit(self.translate_counter)
                log_diagnostic_event(
                    "pipeline.translate_finish",
                    page_index=page_num - 1,
                    page_name=imgname,
                    block_count=len(blk_list) if blk_list is not None else 0,
                )

        try:
            from utils.batch_report import finalize_batch_report
            self._last_batch_report = finalize_batch_report(
                self.stop_requested or (getattr(self, 'cancel_flag', None) is not None and self.cancel_flag.is_set())
            )
        except Exception:
            self._last_batch_report = None

        if (self.stop_requested or (getattr(self, 'cancel_flag', None) is not None and self.cancel_flag.is_set())) and (not cfg_module.enable_translate or not self.parallel_trans):
            self.pipeline_stopped.emit()

    def detect_finished(self) -> bool:
        if self.imgtrans_proj is None:
            return True
        return self.detect_counter == self.num_pages or not cfg_module.enable_detect

    def ocr_finished(self) -> bool:
        if self.imgtrans_proj is None:
            return True
        return self.ocr_counter == self.num_pages or not cfg_module.enable_ocr

    def translate_finished(self) -> bool:
        if self.imgtrans_proj is None \
            or not cfg_module.enable_ocr \
            or not cfg_module.enable_translate:
            return True
        if self.parallel_trans:
            # 检查翻译计数器是否达到需要处理的页面数
            return self.translate_thread.finished_counter >= self.num_pages
        return self.translate_counter == self.num_pages or not cfg_module.enable_translate

    def inpaint_finished(self) -> bool:
        if self.imgtrans_proj is None or not cfg_module.enable_inpaint:
            return True
        return self.inpaint_counter == self.num_pages or not cfg_module.enable_inpaint

    def run(self):
        if self.job is not None:
            self.job()
        self.job = None

    def recent_finished_index(self, ref_counter: int) -> int:
        if cfg_module.enable_detect:
            ref_counter = min(ref_counter, self.detect_counter)
        if cfg_module.enable_ocr:
            ref_counter = min(ref_counter, self.ocr_counter)
        if cfg_module.enable_inpaint:
            ref_counter = min(ref_counter, self.inpaint_counter)
        if cfg_module.enable_translate:
            if self.parallel_trans:
                ref_counter = min(ref_counter, self.translate_thread.finished_counter)
            else:
                ref_counter = min(ref_counter, self.translate_counter)

        process_idx = ref_counter - 1
        # 将处理索引转换为实际页面索引
        if hasattr(self, 'process_idx_to_page_idx') and process_idx in self.process_idx_to_page_idx:
            return self.process_idx_to_page_idx[process_idx]
        return process_idx


def unload_modules(self, module_names):
    model_deleted = False
    for module in module_names:
        module: BaseModule = getattr(self, module)
        model_deleted = model_deleted or module.unload_model()
    if model_deleted:
        soft_empty_cache()


class ModuleManager(QObject):
    imgtrans_proj: ProjImgTrans = None

    finish_translate_page = Signal(str)
    canvas_inpaint_finished = Signal(dict)
    inpaint_th_finished = Signal()

    imgtrans_pipeline_finished = Signal()
    blktrans_pipeline_finished = Signal(int, list)
    page_trans_finished = Signal(int)
    detect_region_finished = Signal(str, list, bool)  # page_name, new_blk_list, replace (replace page blocks)
    pipeline_job_started = Signal(str)
    pipeline_stage_event = Signal(str, int, str)  # stage_name, progress, page_name
    pipeline_event_logged = Signal(str, str)  # code, message
    engine_registry_updated = Signal(dict)

    # Translation failure in batch: user choice (retry/skip/terminate)
    translation_failure_request = Signal(str, str)  # msg, page_key

    run_canvas_inpaint = False
    is_waiting_th = False
    block_set_inpainter = False

    def __init__(self, 
                 imgtrans_proj: ProjImgTrans,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imgtrans_proj = imgtrans_proj
        self.check_inpaint_fin_timer = QTimer(self)
        self.check_inpaint_fin_timer.timeout.connect(self.check_inpaint_th_finished)
        self._trans_failure_mutex = QMutex()
        self._trans_failure_condition = QWaitCondition()
        self._trans_failure_choice = None  # 'retry' | 'skip' | 'terminate'
        self._pipeline_job_seq = 0

    def setupThread(self, config_panel: ConfigPanel, imgtrans_progress_msgbox: ImgtransProgressMessageBox, ocr_postprocess: Callable = None, translate_preprocess: Callable = None, translate_postprocess: Callable = None):
        self.textdetect_thread = TextDetectThread()

        self.ocr_thread = OCRThread()
        
        self.translate_thread = TranslateThread()
        self.translate_thread.manager = self
        self.translate_thread.progress_changed.connect(self.on_update_translate_progress)
        self.translate_thread.finish_translate_page.connect(self.on_finish_translate_page)
        self.translation_failure_request.connect(self.on_translation_failure_request)  

        self.inpaint_thread = InpaintThread()
        self.inpaint_thread.finish_inpaint.connect(self.on_finish_inpaint)

        self.progress_msgbox = imgtrans_progress_msgbox
        self.progress_msgbox.stop_clicked.connect(self.stopImgtransPipeline)
        self.progress_msgbox.force_stop_clicked.connect(self.forceStopImgtransPipeline)

        self.imgtrans_thread = ImgtransThread(self.textdetect_thread, self.ocr_thread, self.translate_thread, self.inpaint_thread)
        self.imgtrans_thread.update_detect_progress.connect(self.on_update_detect_progress)
        self.imgtrans_thread.update_ocr_progress.connect(self.on_update_ocr_progress)
        self.imgtrans_thread.update_translate_progress.connect(self.on_update_translate_progress)
        self.imgtrans_thread.update_inpaint_progress.connect(self.on_update_inpaint_progress)
        self.imgtrans_thread.finish_blktrans_stage.connect(self.on_finish_blktrans_stage)
        self.imgtrans_thread.finish_blktrans.connect(self.on_finish_blktrans)
        self.imgtrans_thread.pipeline_stopped.connect(self.on_imgtrans_thread_stopped)

        self.translator_panel = translator_panel = config_panel.trans_config_panel        
        translator_params = merge_config_module_params(cfg_module.translator_params, GET_VALID_TRANSLATORS(), TRANSLATORS.get)
        # Chain translator: show dropdown of available translators for chain_translators param
        if 'Chain' in translator_params and isinstance(translator_params.get('Chain'), dict):
            chain_params = translator_params['Chain']
            if isinstance(chain_params.get('chain_translators'), dict):
                chain_params['chain_translators'] = dict(chain_params['chain_translators'])
                chain_params['chain_translators']['type'] = 'selector'
                chain_params['chain_translators']['options'] = list(GET_VALID_TRANSLATORS())
                chain_params['chain_translators']['editable'] = True
        translator_panel.addModulesParamWidgets(translator_params)
        translator_panel.translator_changed.connect(self.setTranslator)
        translator_panel.paramwidget_edited.connect(self.on_translatorparam_edited)
        translator_panel.translateByTextblockBox.checker_changed.connect(self.on_translatebyblock_checker_changed)
        translator_panel.translateByTextblockBox.checker.setChecked(cfg_module.translate_by_textblock)
        BaseTranslator.translate_by_textblock = cfg_module.translate_by_textblock
        translator_panel.test_translator_clicked.connect(self.on_test_translator)
        translator_panel.copy_manual_prompt_requested.connect(self.on_copy_manual_prompt)
        translator_panel.paste_manual_response_requested.connect(self.on_paste_manual_response)
        from modules.translators.hooks import chs2cht
        BaseTranslator.register_preprocess_hooks({'keyword_sub': translate_preprocess})
        BaseTranslator.register_postprocess_hooks({'chs2cht': chs2cht, 'keyword_sub': translate_postprocess})

        self.inpaint_panel = inpainter_panel = config_panel.inpaint_config_panel
        inpainter_params = merge_config_module_params(cfg_module.inpainter_params, GET_VALID_INPAINTERS(), INPAINTERS.get)
        inpainter_panel.addModulesParamWidgets(inpainter_params)
        inpainter_panel.paramwidget_edited.connect(self.on_inpainterparam_edited)
        inpainter_panel.inpainter_changed.connect(self.setInpainter)
        inpainter_panel.needInpaintChecker.checker_changed.connect(self.on_inpainter_checker_changed)
        inpainter_panel.needInpaintChecker.checker.setChecked(cfg_module.check_need_inpaint)

        self.textdetect_panel = textdetector_panel = config_panel.detect_config_panel
        textdetector_params = merge_config_module_params(cfg_module.textdetector_params, GET_VALID_TEXTDETECTORS(), TEXTDETECTORS.get)
        textdetector_panel.addModulesParamWidgets(textdetector_params)
        textdetector_panel.paramwidget_edited.connect(self.on_textdetectorparam_edited)
        textdetector_panel.detector_changed.connect(self.setTextDetector)

        self.ocr_panel = ocr_panel = config_panel.ocr_config_panel
        ocr_params = merge_config_module_params(cfg_module.ocr_params, GET_VALID_OCR(), OCR.get)
        ocr_panel.addModulesParamWidgets(ocr_params)
        ocr_panel.paramwidget_edited.connect(self.on_ocrparam_edited)
        ocr_panel.ocr_changed.connect(self.setOCR)
        OCRBase.register_postprocess_hooks(ocr_postprocess)

        config_panel.unload_models.connect(self.unload_all_models)


    def refresh_module_dropdowns(self):
        """Repopulate detector/OCR/translator dropdowns from GET_VALID_* (e.g. after dev_mode toggle)."""
        if not hasattr(self, 'textdetect_panel'):
            return
        self.textdetect_panel.clearModuleList()
        textdetector_params = merge_config_module_params(cfg_module.textdetector_params, GET_VALID_TEXTDETECTORS(), TEXTDETECTORS.get)
        self.textdetect_panel.addModulesParamWidgets(textdetector_params)
        self.textdetect_panel.setModule(cfg_module.textdetector if cfg_module.textdetector in GET_VALID_TEXTDETECTORS() else (GET_VALID_TEXTDETECTORS()[0] if GET_VALID_TEXTDETECTORS() else ''))

        self.ocr_panel.clearModuleList()
        ocr_params = merge_config_module_params(cfg_module.ocr_params, GET_VALID_OCR(), OCR.get)
        self.ocr_panel.addModulesParamWidgets(ocr_params)
        self.ocr_panel.setModule(cfg_module.ocr if cfg_module.ocr in GET_VALID_OCR() else (GET_VALID_OCR()[0] if GET_VALID_OCR() else ''))

        self.translator_panel.clearModuleList()
        translator_params = merge_config_module_params(cfg_module.translator_params, GET_VALID_TRANSLATORS(), TRANSLATORS.get)
        self.translator_panel.addModulesParamWidgets(translator_params)
        self.translator_panel.setModule(cfg_module.translator if cfg_module.translator in GET_VALID_TRANSLATORS() else (GET_VALID_TRANSLATORS()[0] if GET_VALID_TRANSLATORS() else ''))
        self.engine_registry_updated.emit(self.get_engine_registry_snapshot())

    def get_engine_registry_snapshot(self) -> dict:
        return {
            "detector": {
                "selected": getattr(cfg_module, "textdetector", ""),
                "loaded": getattr(self.textdetect_thread.textdetector, "name", "") if hasattr(self, "textdetect_thread") and self.textdetect_thread.textdetector else "",
            },
            "ocr": {
                "selected": getattr(cfg_module, "ocr", ""),
                "loaded": getattr(self.ocr_thread.ocr, "name", "") if hasattr(self, "ocr_thread") and self.ocr_thread.ocr else "",
            },
            "translator": {
                "selected": getattr(cfg_module, "translator", ""),
                "loaded": getattr(self.translate_thread.translator, "name", "") if hasattr(self, "translate_thread") and self.translate_thread.translator else "",
            },
            "inpainter": {
                "selected": getattr(cfg_module, "inpainter", ""),
                "loaded": getattr(self.inpaint_thread.inpainter, "name", "") if hasattr(self, "inpaint_thread") and self.inpaint_thread.inpainter else "",
            },
        }

    def unload_all_models(self):
        unload_modules(self, {'textdetector', 'inpainter', 'ocr', 'translator'})

    @property
    def translator(self) -> BaseTranslator:
        return self.translate_thread.translator

    @property
    def inpainter(self) -> InpainterBase:
        return self.inpaint_thread.inpainter

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.textdetect_thread.textdetector

    @property
    def ocr(self) -> OCRBase:
        return self.ocr_thread.ocr

    def translatePage(self, run_target: bool, page_key: str):
        if not run_target:
            if self.translate_thread.isRunning():
                LOGGER.warning('Terminating a running translation thread.')
                self.translate_thread.terminate()
            return
        self.translate_thread.translatePage(self.imgtrans_proj.pages, page_key)

    def on_translation_failure_request(self, msg: str, page_key: str):
        """Show Retry/Skip/Terminate dialog when translation fails in batch; set choice and wake waiting thread."""
        from qtpy.QtWidgets import QApplication
        box = QMessageBox(QApplication.activeWindow() if QApplication.instance() else None)
        box.setWindowTitle(self.tr('Translation Failed'))
        box.setText(msg)
        box.setStandardButtons(QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Ignore | QMessageBox.StandardButton.Abort)
        box.setDefaultButton(QMessageBox.StandardButton.Retry)
        box.button(QMessageBox.StandardButton.Ignore).setText(self.tr('Skip'))
        box.button(QMessageBox.StandardButton.Abort).setText(self.tr('Terminate'))
        result = box.exec()
        self._trans_failure_mutex.lock()
        if result == QMessageBox.StandardButton.Retry:
            self._trans_failure_choice = 'retry'
        elif result == QMessageBox.StandardButton.Ignore:
            self._trans_failure_choice = 'skip'
        else:
            self._trans_failure_choice = 'terminate'
        self._trans_failure_condition.wakeAll()
        self._trans_failure_mutex.unlock()

    def inpainterBusy(self):
        return self.inpaint_thread.isRunning()

    def inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect = None, **kwargs):
        if self.inpaint_thread.isRunning():
            LOGGER.warning('Waiting for inpainting to finish')
            return
        self.inpaint_thread.inpaint(img, mask, img_key, inpaint_rect)

    def terminateRunningThread(self):
        if self.textdetect_thread.isRunning():
            self.textdetect_thread.quit()
        if self.ocr_thread.isRunning():
            self.ocr_thread.quit()
        if self.inpaint_thread.isRunning():
            self.inpaint_thread.quit()
        if self.translate_thread.isRunning():
            self.translate_thread.quit()

    def check_inpaint_th_finished(self):
        if self.inpaint_thread.isRunning():
            return
        self.block_set_inpainter = False
        self.check_inpaint_fin_timer.stop()
        self.inpaint_th_finished.emit()

    def runImgtransPipeline(self, pages_to_process=None):
        if self.imgtrans_proj.is_empty:
            LOGGER.info('proj file is empty, nothing to do')
            self.progress_msgbox.hide()
            return
        self.last_finished_index = -1
        self.terminateRunningThread()
        
        if cfg_module.all_stages_disabled() and self.imgtrans_proj is not None and self.imgtrans_proj.num_pages > 0:
            for ii in range(self.imgtrans_proj.num_pages):
                self.page_trans_finished.emit(ii)
            self.imgtrans_pipeline_finished.emit()
            return
        
        self.progress_msgbox.detect_bar.setVisible(cfg_module.enable_detect)
        if cfg_module.enable_detect:
            sec = (getattr(cfg_module, 'textdetector_secondary', '') or '').strip()
            ter = (getattr(cfg_module, 'textdetector_tertiary', '') or '').strip()
            dual = getattr(cfg_module, 'enable_dual_detect', False) and sec
            tertiary = getattr(cfg_module, 'enable_tertiary_detect', False) and ter
            if dual or tertiary:
                parts = [cfg_module.textdetector]
                if dual:
                    parts.append(sec)
                if tertiary:
                    parts.append(ter)
                self.progress_msgbox.detect_bar.description = self.tr('Detecting ({}): ').format(' + '.join(parts))
            else:
                self.progress_msgbox.detect_bar.description = self.tr('Detecting: ')
        self.progress_msgbox.ocr_bar.setVisible(cfg_module.enable_ocr)
        self.progress_msgbox.translate_bar.setVisible(cfg_module.enable_translate)
        self.progress_msgbox.inpaint_bar.setVisible(cfg_module.enable_inpaint)
        self.progress_msgbox.zero_progress()
        self.progress_msgbox.show()
        self._pipeline_job_seq += 1
        job_id = f'job-{self._pipeline_job_seq:04d}'
        self.pipeline_job_started.emit(job_id)
        self.pipeline_event_logged.emit('JOB_START', f'Pipeline started: {job_id}')
        self.imgtrans_thread.runImgtransPipeline(self.imgtrans_proj, pages_to_process)
    
    def stopImgtransPipeline(self):
        """停止图像翻译流程"""
        LOGGER.info('Stopping image translation pipeline...')
        self.imgtrans_thread.requestStop()

    def forceStopImgtransPipeline(self):
        """Force-stop the image translation pipeline when the progress window is closed."""
        LOGGER.warning('Force-stopping image translation pipeline due to progress window close...')
        if self.imgtrans_thread.isRunning():
            self.imgtrans_thread.requestStop()
            # Give the pipeline a short time to exit cleanly (avoids crash from terminate() mid-run).
            if not self.imgtrans_thread.wait(3000):
                try:
                    self.imgtrans_thread.terminate()
                    self.imgtrans_thread.wait(1000)
                except Exception as e:
                    LOGGER.debug("Force stop terminate/wait: %s", e)
        try:
            self.terminateRunningThread()
        except Exception as e:
            LOGGER.debug("Force stop terminateRunningThread: %s", e)
        try:
            self.progress_msgbox.hide()
        except Exception:
            pass
        self.imgtrans_pipeline_finished.emit()

    def requestPausePipeline(self):
        """Pause the running pipeline (for batch queue). No-op if not running."""
        if self.imgtrans_thread.isRunning():
            self.imgtrans_thread.requestPause()

    def requestResumePipeline(self):
        """Resume the pipeline after pause."""
        self.imgtrans_thread.requestResume()

    def runBlktransPipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int, blk_ids: List[int], tgt_mask):
        self.terminateRunningThread()
        self.progress_msgbox.hide_all_bars()
        if mode >= 0 and mode < 3:
            self.progress_msgbox.ocr_bar.show()
        if mode >= 2:
            self.progress_msgbox.inpaint_bar.show()
        if mode != 0 and mode < 3:
            self.progress_msgbox.translate_bar.show()
        self.progress_msgbox.zero_progress()
        self.progress_msgbox.show()
        self.imgtrans_thread.runBlktransPipeline(blk_list, tgt_img, mode, blk_ids, tgt_mask)

    def on_finish_blktrans_stage(self, stage: str, progress: int):
        if stage == 'ocr':
            self.progress_msgbox.updateOCRProgress(progress)
        elif stage == 'translate':
            self.progress_msgbox.updateTranslateProgress(progress)
        elif stage == 'inpaint':
            self.progress_msgbox.updateInpaintProgress(progress)
        else:
            raise NotImplementedError(f'Unknown stage: {stage}')
        
    def on_finish_blktrans(self, mode: int, blk_ids: List):
        self.blktrans_pipeline_finished.emit(mode, blk_ids)
        self.progress_msgbox.hide()

    def on_update_detect_progress(self, progress: int):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        if 'detect' in shared.pbar:
            shared.pbar['detect'].update(1)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        self.pipeline_stage_event.emit('detect', progress, '')
        if progress in (1, 100):
            self.pipeline_event_logged.emit('DETECT', f'Detect progress {progress}%')
        self.progress_msgbox.updateDetectProgress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def on_update_ocr_progress(self, progress: int):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        if 'ocr' in shared.pbar:
            shared.pbar['ocr'].update(1)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        self.pipeline_stage_event.emit('ocr', progress, '')
        if progress in (1, 100):
            self.pipeline_event_logged.emit('OCR', f'OCR progress {progress}%')
        self.progress_msgbox.updateOCRProgress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def on_update_translate_progress(self, progress: int):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        if 'translate' in shared.pbar:
            shared.pbar['translate'].update(1)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        self.pipeline_stage_event.emit('translate', progress, '')
        if progress in (1, 100):
            self.pipeline_event_logged.emit('TRANSLATE', f'Translate progress {progress}%')
        self.progress_msgbox.updateTranslateProgress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def on_update_inpaint_progress(self, progress: int):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        if 'inpaint' in shared.pbar:
            shared.pbar['inpaint'].update(1)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        self.pipeline_stage_event.emit('inpaint', progress, '')
        if progress in (1, 100):
            self.pipeline_event_logged.emit('INPAINT', f'Inpaint progress {progress}%')
        self.progress_msgbox.updateInpaintProgress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def progress(self):
        progress = {}
        num_pages = self.imgtrans_thread.num_pages
        if cfg_module.enable_detect:
            progress['detect'] = self.imgtrans_thread.detect_counter / num_pages
        if cfg_module.enable_ocr:
            progress['ocr'] = self.imgtrans_thread.ocr_counter / num_pages
        if cfg_module.enable_inpaint:
            progress['inpaint'] = self.imgtrans_thread.inpaint_counter / num_pages
        if cfg_module.enable_translate:
            progress['translate'] = self.imgtrans_thread.translate_counter / num_pages
        return progress

    def proj_finished(self):
        if self.imgtrans_thread.detect_finished() \
            and self.imgtrans_thread.ocr_finished() \
                and self.imgtrans_thread.translate_finished() \
                    and self.imgtrans_thread.inpaint_finished():
            return True
        return False

    def finishImgtransPipeline(self):
        if self.proj_finished():
            self.progress_msgbox.hide()
            self.imgtrans_pipeline_finished.emit()
    
    def on_imgtrans_thread_stopped(self):
        """线程完成时确保关闭进度对话框"""
        # 线程完成了，直接关闭窗口
        self.progress_msgbox.hide()
        self.imgtrans_pipeline_finished.emit()

    def get_last_batch_report(self):
        """Return the batch report from the last pipeline run (skipped pages, reasons). None if none."""
        return getattr(self.imgtrans_thread, '_last_batch_report', None)

    def setTranslator(self, translator: str = None):
        if translator is None:
            translator = cfg_module.translator
        is_fallback = False
        valid = GET_VALID_TRANSLATORS()
        if translator not in valid:
            if translator in TRANSLATORS.module_dict:
                LOGGER.info(
                    "Translator '%s' is registered but model files are not yet downloaded; "
                    "skipping initialization. Will auto-set after download completes.",
                    translator,
                )
                return
            fallback = 'google' if 'google' in valid else (valid[0] if valid else None)
            if fallback:
                LOGGER.warning(
                    "Translator '%s' is not available (e.g. not yet integrated). Using '%s'.",
                    translator, fallback,
                )
                translator = fallback
                is_fallback = True
            else:
                create_error_dialog(
                    ValueError(self.tr("No translator module available.")),
                    self.tr("Failed to set Translator.") + "\n" + self.tr(_model_missing_guidance("Translator", translator)),
                )
                return
        if self.translate_thread.isRunning():
            LOGGER.warning('Terminating a running translation thread.')
            self.translate_thread.terminate()
        self.translate_thread.setTranslator(translator, is_fallback)

    def setInpainter(self, inpainter: str = None):
        
        if self.block_set_inpainter:
            return
        
        if inpainter is None:
            inpainter = cfg_module.inpainter
        is_fallback = False
        valid = GET_VALID_INPAINTERS()
        if inpainter not in valid:
            if inpainter in INPAINTERS.module_dict:
                LOGGER.info(
                    "Inpainter '%s' is registered but model files are not yet downloaded; "
                    "skipping initialization. Will auto-set after download completes.",
                    inpainter,
                )
                return
            fallback = valid[0] if valid else None
            if fallback:
                LOGGER.warning(
                    "Inpainter '%s' is not available (e.g. not yet integrated). Using '%s'.",
                    inpainter, fallback,
                )
                inpainter = fallback
                is_fallback = True
            else:
                create_error_dialog(
                    ValueError(self.tr("No inpainter module available.")),
                    self.tr("Failed to set Inpainter.") + "\n" + self.tr(_model_missing_guidance("Inpainter", inpainter)),
                )
                return
        
        if self.inpaint_thread.isRunning():
            self.block_set_inpainter = True
            create_info_dialog(self.tr('Set Inpainter...'), modal=True, signal_slot_map_list=[{'signal': self.inpaint_th_finished, 'slot': 'done'}])
            self.check_inpaint_fin_timer.start(300)
            return

        self.inpaint_thread.setInpainter(inpainter, is_fallback)

    def setTextDetector(self, textdetector: str = None):
        if textdetector is None:
            textdetector = cfg_module.textdetector
        is_fallback = False
        valid = GET_VALID_TEXTDETECTORS()
        if textdetector not in valid:
            if textdetector in TEXTDETECTORS.module_dict:
                LOGGER.info(
                    "Text detector '%s' is registered but model files are not yet downloaded; "
                    "skipping initialization. Will auto-set after download completes.",
                    textdetector,
                )
                return
            fallback = "ctd" if "ctd" in valid else (valid[0] if valid else None)
            if fallback is None:
                create_error_dialog(
                    ValueError(self.tr("No text detector module available.")),
                    self.tr("Failed to set Text Detector.") + "\n" + self.tr(_model_missing_guidance("Text detector", textdetector)),
                )
                return
            LOGGER.warning(
                "Text detector '%s' is not available (e.g. not yet integrated). Using '%s'.",
                textdetector, fallback,
            )
            textdetector = fallback
            is_fallback = True
        if self.textdetect_thread.isRunning():
            LOGGER.warning('Terminating a running text detection thread.')
            self.textdetect_thread.terminate()
        self.textdetect_thread.setTextDetector(textdetector, is_fallback)

    def run_detect_region(self, rect, img_array, page_name: str, replace_if_full_page: bool = False):
        """Run text detection on a cropped region; emit detect_region_finished(page_name, blk_list, replace).
        When replace_if_full_page is True (e.g. "Detect text on page"), existing blocks are replaced."""
        import numpy as np
        h, w = img_array.shape[:2]
        x1 = max(0, int(rect.x()))
        y1 = max(0, int(rect.y()))
        x2 = min(w, int(rect.x() + rect.width()))
        y2 = min(h, int(rect.y() + rect.height()))
        if x2 <= x1 or y2 <= y1:
            self.detect_region_finished.emit(page_name, [], replace_if_full_page)
            return
        crop = np.ascontiguousarray(img_array[y1:y2, x1:x2])
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            self.detect_region_finished.emit(page_name, [], replace_if_full_page)
            return
        manager = self
        replace = replace_if_full_page
        def job():
            try:
                if manager.textdetector is None:
                    manager.detect_region_finished.emit(page_name, [], replace)
                    return
                mask, blk_list = manager.textdetector.detect(crop, None)
            except Exception as e:
                create_error_dialog(e, manager.tr('Detection in region failed.'), 'DetectRegion')
                manager.detect_region_finished.emit(page_name, [], replace)
                return
            for blk in blk_list:
                blk.xyxy = [blk.xyxy[0] + x1, blk.xyxy[1] + y1, blk.xyxy[2] + x1, blk.xyxy[3] + y1]
                if getattr(blk, 'lines', None):
                    for line in blk.lines:
                        for i, pt in enumerate(line):
                            line[i] = [pt[0] + x1, pt[1] + y1]
            manager.detect_region_finished.emit(page_name, blk_list, replace)
        self.textdetect_thread.job = job
        self.textdetect_thread.start()

    def setOCR(self, ocr: str = None):
        if ocr is None:
            ocr = cfg_module.ocr
        is_fallback = False
        valid = GET_VALID_OCR()
        if ocr not in valid:
            if ocr in OCR.module_dict:
                LOGGER.info(
                    "OCR '%s' is registered but model files are not yet downloaded; "
                    "skipping initialization. Will auto-set after download completes.",
                    ocr,
                )
                return
            fallback = valid[0] if valid else None
            if fallback:
                LOGGER.warning(
                    "OCR '%s' is not available (e.g. not yet integrated). Using '%s'.",
                    ocr, fallback,
                )
                ocr = fallback
                is_fallback = True
            else:
                create_error_dialog(
                    ValueError(self.tr("No OCR module available.")),
                    self.tr("Failed to set OCR.") + "\n" + self.tr(_model_missing_guidance("OCR", ocr)),
                )
                return
        if self.ocr_thread.isRunning():
            LOGGER.warning('Terminating a running OCR thread.')
            self.ocr_thread.terminate()
        self.ocr_thread.setOCR(ocr, is_fallback)

    def on_finish_translate_page(self, page_key: str):
        page_index = -1
        block_count = 0
        if self.imgtrans_thread.imgtrans_proj is not None:
            page_keys = list(self.imgtrans_thread.imgtrans_proj.pages.keys())
            if page_key in page_keys:
                page_index = page_keys.index(page_key)
            block_count = len(self.imgtrans_thread.imgtrans_proj.pages.get(page_key, []) or [])
        log_diagnostic_event(
            "pipeline.translate_finish",
            page_index=page_index,
            page_name=page_key,
            block_count=block_count,
        )
        self.finish_translate_page.emit(page_key)
    
    def on_finish_inpaint(self, inpaint_dict: dict):
        if self.run_canvas_inpaint:
            self.canvas_inpaint_finished.emit(inpaint_dict)
            self.run_canvas_inpaint = False

    def canvas_inpaint(self, inpaint_dict):
        self.run_canvas_inpaint = True
        self.inpaint(**inpaint_dict)
    
    def schedule_config_save(self, delay_ms: int = 1500):
        """
        Persist module parameters shortly after they change so they are not lost when
        the app is killed or crashes before a clean exit
        (upstream dmMaze/BallonsTranslator#913).
        """
        timer = getattr(self, '_config_save_timer', None)
        if timer is None:
            timer = self._config_save_timer = QTimer(self)
            timer.setSingleShot(True)

            def _save():
                try:
                    from utils.config import save_config
                    save_config()
                except Exception as e:
                    LOGGER.debug('Deferred config save failed: %s', e)

            timer.timeout.connect(_save)
        timer.start(delay_ms)

    def on_translatorparam_edited(self, param_key: str, param_content: dict):
        if self.translator is not None:
            self.updateModuleSetupParam(self.translator, param_key, param_content)
            cfg_module.translator_params[self.translator.name] = self.translator.params
            self.schedule_config_save()

    def on_test_translator(self):
        if self.translator is None:
            create_error_dialog(ValueError("No translator loaded"), self.tr("Test translator"))
            return
        btn = self.translator_panel.testTranslatorBtn
        btn.setEnabled(False)
        btn.setText(self.tr("Testing..."))
        def run():
            try:
                success, src_text, result = test_translator(self.translator)
                if success:
                    create_info_dialog(self.tr("Success.") + "\n\n" + result)
                else:
                    create_error_dialog(RuntimeError(result), self.tr("Test translator"))
            finally:
                btn.setEnabled(True)
                btn.setText(self.tr("Test translator"))
        QTimer.singleShot(50, run)

    def on_copy_manual_prompt(self):
        """Section 10: Copy manual translation prompt (JSON) for current page to clipboard."""
        proj = getattr(self, "imgtrans_proj", None)
        if proj is None or getattr(proj, "is_empty", True) or not proj.current_img:
            create_info_dialog(self.tr("Open a project and select a page with text blocks."), self.tr("Copy prompt"))
            return
        page_key = proj.current_img
        blks = proj.pages.get(page_key, [])
        src_list = []
        for blk in blks:
            text = getattr(blk, "text", None)
            if isinstance(text, list):
                src_list.append("\n".join(text).strip() if text else "")
            else:
                src_list.append(str(text or "").strip())
        if not src_list:
            create_info_dialog(self.tr("No source text on this page. Run OCR first."), self.tr("Copy prompt"))
            return
        lang_src = getattr(self.translator, "lang_source", "") if self.translator else (cfg_module.translate_source or "")
        lang_tgt = getattr(self.translator, "lang_target", "") if self.translator else (cfg_module.translate_target or "")
        prompt_obj = {
            "instruction": (
                "Translate each 'source' to the target language. Return JSON only.\n"
                "Accepted response formats:\n"
                "1) {\"translations\":[{\"id\":1,\"translation\":\"...\"}, ...]}\n"
                "2) {\"1\":\"...\",\"2\":\"...\"}\n"
                "3) [\"...\", \"...\", ...] (same order)\n"
            ),
            "source_language": lang_src,
            "target_language": lang_tgt,
            "items": [{"id": i + 1, "source": s} for i, s in enumerate(src_list)],
        }
        text = json.dumps(prompt_obj, ensure_ascii=False, indent=2)
        cb = QApplication.clipboard()
        if cb:
            cb.setText(text, QClipboard.Mode.Clipboard)
        create_info_dialog(self.tr("Prompt copied to clipboard. Paste it into your tool, then paste the response back and click Paste response."), self.tr("Copy prompt"))

    def on_paste_manual_response(self):
        """Section 10: Paste JSON from clipboard into manual translator response and optionally apply to blocks."""
        proj = getattr(self, "imgtrans_proj", None)
        if proj is None or getattr(proj, "is_empty", True) or not proj.current_img:
            create_info_dialog(self.tr("Open a project and select a page with text blocks."), self.tr("Paste response"))
            return
        cb = QApplication.clipboard()
        if not cb:
            create_error_dialog(RuntimeError("No clipboard available."), self.tr("Paste response"))
            return
        raw = (cb.text() or "").strip()
        if not raw:
            create_info_dialog(self.tr("Clipboard is empty. Copy the response from your translation tool first."), self.tr("Paste response"))
            return
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            create_error_dialog(e, self.tr("Paste response"))
            return
        page_key = proj.current_img
        blks = proj.pages.get(page_key, [])
        src_list = []
        for blk in blks:
            text = getattr(blk, "text", None)
            if isinstance(text, list):
                src_list.append("\n".join(text).strip() if text else "")
            else:
                src_list.append(str(text or "").strip())
        n = len(src_list)
        if n == 0:
            create_info_dialog(self.tr("No source text on this page."), self.tr("Paste response"))
            return
        out = [""] * n
        if isinstance(obj, list):
            if len(obj) != n:
                create_error_dialog(ValueError(self.tr("Response has %d items but page has %d blocks.") % (len(obj), n)), self.tr("Paste response"))
                return
            out = [str(x) if x is not None else "" for x in obj]
        elif isinstance(obj, dict):
            if "translations" in obj and isinstance(obj["translations"], list):
                for el in obj["translations"]:
                    if not isinstance(el, dict):
                        continue
                    try:
                        idx = int(el.get("id", 0)) - 1
                    except Exception:
                        continue
                    if 0 <= idx < n:
                        out[idx] = str(el.get("translation") or "")
            else:
                for k, v in obj.items():
                    try:
                        idx = int(k) - 1
                    except Exception:
                        continue
                    if 0 <= idx < n:
                        out[idx] = str(v) if v is not None else ""
        else:
            create_error_dialog(ValueError(self.tr("Response must be JSON object or array.")), self.tr("Paste response"))
            return
        if "trans_manual" not in cfg_module.translator_params:
            cfg_module.translator_params["trans_manual"] = {}
        if "response_json" not in cfg_module.translator_params["trans_manual"]:
            cfg_module.translator_params["trans_manual"]["response_json"] = {"type": "editor", "value": ""}
        cfg_module.translator_params["trans_manual"]["response_json"]["value"] = raw
        for i, blk in enumerate(blks):
            if i < len(out):
                blk.translation = out[i].split("\n") if out[i] else []
        self.translator_panel.paramwidget_edited.emit("response_json", {"content": raw})
        n = len(out)
        create_info_dialog(
            self.tr("Response applied to {} block(s). You can run Translate again or edit in the text panel.").format(n),
            self.tr("Paste response"),
        )

    def on_inpainterparam_edited(self, param_key: str, param_content: dict):
        if self.inpainter is not None:
            self.updateModuleSetupParam(self.inpainter, param_key, param_content)
            cfg_module.inpainter_params[self.inpainter.name] = self.inpainter.params
            self.schedule_config_save()

    def on_textdetectorparam_edited(self, param_key: str, param_content: dict):
        if self.textdetector is not None:
            self.updateModuleSetupParam(self.textdetector, param_key, param_content)
            cfg_module.textdetector_params[self.textdetector.name] = self.textdetector.params
            self.schedule_config_save()

    def on_ocrparam_edited(self, param_key: str, param_content: dict):
        if self.ocr is not None:
            self.updateModuleSetupParam(self.ocr, param_key, param_content)
            cfg_module.ocr_params[self.ocr.name] = self.ocr.params
            self.schedule_config_save()

    def updateModuleSetupParam(self,
                               module: Union[InpainterBase, BaseTranslator],
                               param_key: str, param_content: dict):
            
        if param_content.get('flush', False):
            if getattr(module, 'params', None) is not None and param_key in module.params:
                param_widget: ParamComboBox = param_content['widget']
                param_widget.blockSignals(True)
                current_item = param_widget.currentText()
                param_widget.clear()
                param_widget.addItems(module.flush(param_key))
                param_widget.setCurrentText(current_item)
                param_widget.blockSignals(False)
        elif param_content.get('select_path', False):
            if getattr(module, 'params', None) is not None and param_key in module.params:
                dialog = QFileDialog()
                f = module.params[param_key].get('path_filter', None)
                p = dialog.getOpenFileUrl(self.parent(), filter=f)[0].toLocalFile()
                if osp.exists(p):
                    param_widget: ParamComboBox = param_content['widget']
                    param_widget.setCurrentText(p)
        else:
            if getattr(module, 'params', None) is not None and param_key in module.params:
                module.updateParam(param_key, param_content['content'])
            else:
                LOGGER.debug(
                    "Module %s: skipping param update for unknown key %r",
                    getattr(module, 'name', type(module).__name__),
                    param_key,
                )

    def handle_page_changed(self):
        if not self.imgtrans_thread.isRunning():
            if self.inpaint_thread.inpainting:
                self.run_canvas_inpaint = False
                self.inpaint_thread.terminate()

    def on_inpainter_checker_changed(self, is_checked: bool):
        cfg_module.check_need_inpaint = is_checked
        InpainterBase.check_need_inpaint = is_checked

    def on_translatebyblock_checker_changed(self, is_checked: bool):
        cfg_module.translate_by_textblock = is_checked
        BaseTranslator.translate_by_textblock = is_checked
