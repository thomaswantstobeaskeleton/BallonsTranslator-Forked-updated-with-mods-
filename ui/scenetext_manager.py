
from typing import List, Union, Tuple, Optional, Set
import numpy as np
import copy
import base64

from qtpy.QtWidgets import QApplication, QWidget, QGraphicsItem, QScrollArea, QSizePolicy
from qtpy.QtCore import QObject, QRectF, Qt, Signal, QPointF, QPoint
from qtpy.QtGui import QKeyEvent, QTextCursor, QFontMetricsF, QFont, QTextCharFormat, QClipboard
try:
    from qtpy.QtWidgets import QUndoCommand
except ImportError:
    from qtpy.QtGui import QUndoCommand

from .textitem import TextBlkItem, TextBlock
from .canvas import Canvas
from .textedit_area import TransTextEdit, SourceTextEdit, TransPairWidget, SelectTextMiniMenu, TextEditListScrollArea, QVBoxLayout, Widget
from utils.fontformat import FontFormat, pt2px
from .textedit_commands import propagate_user_edit, TextEditCommand, ReshapeItemCommand, MoveBlkItemsCommand, AutoLayoutCommand, ApplyFontformatCommand, RotateItemCommand, WarpItemCommand, TextItemEditCommand, PageReplaceOneCommand, PageReplaceAllCommand, MultiPasteCommand, ResetAngleCommand, SqueezeCommand
from .text_panel import FontFormatPanel
from utils.config import pcfg
try:
    from utils.config import log_diagnostic_event
except ImportError:
    def log_diagnostic_event(*args, **kwargs):
        return None
from utils import shared
from utils.imgproc_utils import extract_ballon_region, rotate_polygons, get_block_mask, classify_bubble_shape_from_mask, mask_centroid_in_crop
from utils.bubble_shape_model import get_bubble_shape_from_model as _bubble_shape_from_model_impl
from utils.box_size_check_model import check_box_size_from_model
from utils.text_processing import seg_text, is_cjk
from utils.text_layout import layout_text
from utils.auto_text_layout import auto_layout_effective_preset, auto_layout_preset_settings, auto_rendered_fit_scale, normalize_auto_layout_preset
from utils.text_rendering import (
    balance_lines,
    fallback_chain_for_text,
    fit_font_size_to_box,
    merge_font_fallback_chain,
    missing_glyphs_after_fallback,
    normalize_vertical_punctuation,
    resolve_fit_font_size_bounds,
    resolve_writing_mode,
    smart_fit_text_to_box,
    plan_typography_cleanup,
    plan_atomic_bubble_fit,
    normalize_atomic_fit_mode,
)
from utils.line_breaking import split_long_token_with_hyphenation
from utils.text_masking import bubble_safe_text_rect, masked_text_warnings, recommended_padding_for_mask, mask_effective_box
from utils.textbox_masking import centered_resize_xyxy, mask_aware_textbox_diagnostics
from utils.text_layout_auto_fit import AutoFitRequest, auto_fit_text

from utils.layout_review_agent import (
    BlockSnapshot,
    ExternalReviewProvider,
    HeuristicReviewProvider,
    LayoutReviewPlanner,
    ReviewModelConfig,
    ReviewProvider,
    ReviewAction,
    collect_actions_from_list,
    filter_actions_by_block_indices,
)
from utils.logger import logger as LOGGER
from modules.textdetector.outside_text_processor import OSB_LABELS
from utils.font_detect import detect_font_from_block

# Layout tuning: keep text inside bubbles and avoid overflow/tiny text (see layout_textblk)
# Defaults favor more lines and smaller base size so text fits without overflowing wide boxes.
LAYOUT_FIT_RATIO = 0.50  # fit text to this fraction of region (use more of bubble width/height)
LAYOUT_FIT_RATIO_MIN = 0.15  # minimum scale when fitting (allow smaller text when bubble is crowded)
LAYOUT_MIN_FONT_PT = 10.0  # fallback minimum font size; config layout_font_size_min/max override (2.1).
LAYOUT_MIN_BOX_W = 28  # minimum text box width (px) so at least one character is visible
LAYOUT_MIN_BOX_H = 20  # minimum text box height (px)
LAYOUT_HEIGHT_PADDING = 1.22  # multiplier for block height (bottom margin; prevent last-line/descender cropping)
LAYOUT_WIDTH_STROKE_FACTOR = 1.24  # width = max line width * this (stroke/outline clearance; avoid horizontal crop e.g. "loo")
LAYOUT_SCALE_UP_MAX = 1  # cap scale-up when bubble is large and text is short
LAYOUT_SCALE_UP_FILL = 0.68  # target fill ratio when scaling up in large bubbles
# Scale down final font so text is smaller and fits in more lines (narrower effective width).
LAYOUT_TEXT_SCALE = 0.68  # multiply final font size by this (smaller base = more lines in wide boxes)
LAYOUT_TEXT_SCALE_LONG = 0.58  # stronger scale-down for long paragraphs
LAYOUT_TEXT_SCALE_LONG_MIN_LINES = 3  # use LAYOUT_TEXT_SCALE_LONG when at least this many lines
LAYOUT_TEXT_SCALE_LONG_MIN_CHARS = 80  # or when at least this many characters
LAYOUT_MAX_LINE_WIDTH_FRAC = 0.92  # max line width = bubble width * this (higher = fuller lines, text fills box more)
# Extra crop padding (px) when extracting bubble region for balloon-shape classification so the mask sees the full bubble.
LAYOUT_BALLOON_SHAPE_CROP_PAD = 20
# Shapes that are resolved from "auto" (used to skip re-resolving when already set by a prior shape pass).
CONCRETE_BALLOON_SHAPES = frozenset(("round", "elongated", "narrow", "diamond", "square", "bevel", "pentagon", "point"))


def _auto_layout_preset():
    return normalize_auto_layout_preset(getattr(pcfg.module, "layout_auto_preset", "balanced"))


def _auto_lettering_page_scale(img, box_size: Tuple[float, float] = None) -> float:
    mod = getattr(pcfg, "module", None)
    if img is None or mod is None or not bool(getattr(mod, "layout_scale_font_by_page_size", True)):
        return 1.0
    try:
        h, w = img.shape[:2]
        area_mp = max(0.01, (float(w) * float(h)) / 1_000_000.0)
        ref_mp = max(0.25, float(getattr(mod, "layout_scale_reference_megapixels", 1.0) or 1.0))
        raw = (area_mp / ref_mp) ** 0.5
        min_factor = float(getattr(mod, "layout_scale_factor_min", 0.72) or 0.72)
        max_factor = float(getattr(mod, "layout_scale_factor_max", 1.58) or 1.58)
        scale = max(min_factor, min(max_factor, raw))
        if bool(getattr(mod, "layout_scale_use_box_area", True)) and box_size is not None:
            bw = max(1.0, float(box_size[0] if len(box_size) > 0 else 1.0))
            bh = max(1.0, float(box_size[1] if len(box_size) > 1 else 1.0))
            box_ratio = (bw * bh) / max(1.0, float(w) * float(h))
            ref_ratio = max(0.001, float(getattr(mod, "layout_scale_box_area_reference", 0.020) or 0.020))
            box_scale = (box_ratio / ref_ratio) ** 0.22
            scale *= max(0.85, min(1.18, box_scale))
        return scale
    except Exception:
        return 1.0


def _auto_layout_settings_for(text: str, box_w: float, box_h: float, balloon_shape: str = "auto", line_count: int = None):
    preset = auto_layout_effective_preset(_auto_layout_preset(), text, box_w, box_h, balloon_shape, line_count=line_count)
    return auto_layout_preset_settings(preset)


def _auto_width_profile_for_shape(width: float, height: float, line_count: int, balloon_shape: str = "auto") -> List[float]:
    line_count = max(1, int(line_count or 1))
    width = max(8.0, float(width))
    if balloon_shape not in ("round", "ellipse", "diamond", "point"):
        return [width] * line_count
    profile: List[float] = []
    for i in range(line_count):
        t = 0.0 if line_count <= 1 else (i / float(line_count - 1))
        d = abs(2.0 * t - 1.0)
        # narrower near top/bottom, wider near center
        factor = max(0.56, 0.98 - 0.40 * (d ** 1.6))
        profile.append(width * factor)
    return profile


def _auto_layout_text_scale(text: str, line_count: int, box_w: float, box_h: float) -> float:
    """Return an automatic final font scale for current text density."""
    clean_len = len((text or "").replace("\n", "").strip())
    area = max(1.0, float(box_w or 0.0) * float(box_h or 0.0))
    density = clean_len / max(1.0, np.sqrt(area))
    if line_count <= 1 and clean_len <= 18:
        scale = 0.92
    elif line_count <= 2 and clean_len <= 36:
        scale = 0.84
    elif line_count >= 5 or clean_len >= 120 or density >= 1.25:
        scale = LAYOUT_TEXT_SCALE_LONG
    elif line_count >= LAYOUT_TEXT_SCALE_LONG_MIN_LINES or clean_len >= LAYOUT_TEXT_SCALE_LONG_MIN_CHARS or density >= 0.85:
        scale = min(0.72, max(LAYOUT_TEXT_SCALE_LONG, LAYOUT_TEXT_SCALE))
    else:
        scale = LAYOUT_TEXT_SCALE
    settings = _auto_layout_settings_for(text, box_w, box_h, line_count=line_count)
    return float(np.clip(scale * settings.font_scale_multiplier, 0.48, 1.05))


def _auto_line_width_fraction(text: str, box_w: float, box_h: float, balloon_shape: str) -> float:
    """Choose max line width as a fraction of the safe bubble width."""
    shape = (balloon_shape or "auto").lower()
    base = {
        "round": 0.82,
        "diamond": 0.76,
        "narrow": 0.72,
        "point": 0.74,
        "square": 0.86,
        "bevel": 0.84,
        "pentagon": 0.82,
        "elongated": 0.92,
    }.get(shape, LAYOUT_MAX_LINE_WIDTH_FRAC)
    aspect = float(box_w or 1.0) / max(1.0, float(box_h or 1.0))
    clean_len = len((text or "").replace("\n", "").strip())
    if aspect >= 2.2:
        base = min(0.96, base + 0.05)
    elif aspect <= 0.72:
        base = max(0.62, base - 0.08)
    if clean_len >= 90:
        base = min(0.96, base + 0.04)
    settings = _auto_layout_settings_for(text, box_w, box_h, balloon_shape=balloon_shape)
    return float(np.clip(base + settings.line_width_delta, 0.52, 0.98))


def _shape_inner_inset(balloon_shape: str, text: str, rw: float, rh: float) -> float:
    """Safe-area inset for a bubble shape."""
    shape = (balloon_shape or "auto").lower()
    inset = {
        "round": 0.92,
        "narrow": 0.94,
        "point": 0.92,
        "diamond": 0.88,
        "square": 0.94,
        "bevel": 0.93,
        "pentagon": 0.92,
        "elongated": 0.96,
    }.get(shape, 0.94)
    clean_len = len((text or "").replace("\n", "").strip())
    if clean_len >= 100:
        inset += 0.03
    elif clean_len <= 18:
        inset -= 0.02
    if rw > 0 and rh > 0 and min(rw, rh) < 72:
        inset += 0.03
    settings = _auto_layout_settings_for(text, rw, rh, balloon_shape=balloon_shape)
    return float(np.clip(inset + settings.inner_inset_delta, 0.78, 0.99))


def _aspect_ratio_shape(rw: int, rh: int) -> str:
    """Aspect-ratio heuristic when no mask/contour: square, round, narrow, elongated."""
    if rw <= 0 or rh <= 0:
        return "round"
    ar = min(rw, rh) / max(rw, rh)
    # Prefer elongated (oval) for wide bubbles; only call square when nearly square
    if rw > rh * 1.2:
        return "elongated"
    if rh > rw * 1.5:
        return "narrow"
    if ar >= 0.9:
        return "square"
    if ar >= 0.55:
        return "round"
    return "round"


def _resolve_auto_balloon_shape(
    method: str,
    mask: np.ndarray,
    mask_xyxy: list,
    img: np.ndarray,
    rw: int,
    rh: int,
) -> str:
    """
    Resolve balloon shape when layout_balloon_shape is 'auto'.
    method: one of aspect_ratio, contour, model, model_contour, model_ratio, contour_ratio, model_contour_ratio.
    Returns first successful shape from the chosen chain.
    """
    method = (method or "contour_ratio").strip().lower()
    model_id = (getattr(pcfg.module, "layout_balloon_shape_model_id", None) or "").strip()

    def try_model() -> str:
        if not model_id:
            return None
        return _bubble_shape_from_model_impl(mask, mask_xyxy, img, model_id)

    def try_contour() -> str:
        return classify_bubble_shape_from_mask(mask) if mask is not None and mask.size > 0 else None

    def try_ratio() -> str:
        return _aspect_ratio_shape(rw, rh)

    steps = []
    if method == "aspect_ratio":
        steps = ["ratio"]
    elif method == "contour":
        steps = ["contour"]
    elif method == "model":
        steps = ["model"]
    elif method == "model_contour":
        steps = ["model", "contour"]
    elif method == "model_ratio":
        steps = ["model", "ratio"]
    elif method == "contour_ratio":
        steps = ["contour", "ratio"]
    else:
        steps = ["model", "contour", "ratio"]

    def _apply_allowed_filter(shape: str) -> str:
        allowed_str = (getattr(pcfg.module, "layout_balloon_shape_auto_allowed", None) or "").strip()
        if not allowed_str:
            return shape
        allowed = {s.strip().lower() for s in allowed_str.split(",") if s.strip()}
        if not allowed:
            return shape
        return shape if shape in allowed else "square"

    for step in steps:
        if step == "model":
            s = try_model()
        elif step == "contour":
            s = try_contour()
        else:
            s = try_ratio()
        if s:
            return _apply_allowed_filter(s)
    return _apply_allowed_filter(try_ratio())


def _word_count(line: str) -> int:
    """Number of non-empty tokens after split (strip each so 'the ' or ' the ' counts as 1)."""
    return len([w for w in line.split() if w.strip()])


def _merge_stub_lines_in_text(text: str) -> str:
    """Merge short lines so 'the', 'and', 'Realm.' never sit alone. First all 1-word lines, then 2-5 word stubs."""
    if not text or not text.strip():
        return text
    prev = ""
    while prev != text:
        prev = text
        lines = [ln.strip() for ln in text.splitlines() if ln is not None and ln.strip()]
        if len(lines) <= 1:
            return text
        # Eliminate every 0- or 1-word line (repeat until none left)
        while True:
            merged = []
            i = 0
            any_one = False
            while i < len(lines):
                line = lines[i]
                nw = _word_count(line)
                if nw <= 1 and i + 1 < len(lines):
                    merged.append((line + " " + lines[i + 1]).strip())
                    i += 2
                    any_one = True
                    continue
                if nw <= 1 and i == len(lines) - 1 and merged:
                    merged[-1] = (merged[-1] + " " + line).strip()
                    i += 1
                    any_one = True
                    continue
                merged.append(line)
                i += 1
            if not any_one:
                break
            lines = merged
            if len(lines) <= 1:
                break
        # Merge 2-5 word stubs
        merged = []
        i = 0
        while i < len(lines):
            line = lines[i]
            nw = _word_count(line)
            if nw <= 5 and i + 1 < len(lines):
                merged.append((line + " " + lines[i + 1]).strip())
                i += 2
                continue
            if nw <= 5 and i == len(lines) - 1 and merged:
                merged[-1] = (merged[-1] + " " + line).strip()
                i += 1
                continue
            merged.append(line)
            i += 1
        text = "\n".join(merged)
    # Final pass: merge any remaining 0-4 word line until none left (catches 'stage.', 'in', 'the', etc.)
    while True:
        lines = [ln.strip() for ln in text.splitlines() if ln is not None and ln.strip()]
        if len(lines) <= 1:
            break
        out = []
        i = 0
        any_stub = False
        while i < len(lines):
            line = lines[i]
            nw = _word_count(line)
            if nw <= 4 and i + 1 < len(lines):
                out.append((line + " " + lines[i + 1]).strip())
                i += 2
                any_stub = True
                continue
            if nw <= 4 and i == len(lines) - 1 and out:
                out[-1] = (out[-1] + " " + line).strip()
                i += 1
                any_stub = True
                continue
            out.append(line)
            i += 1
        if not any_stub:
            break
        text = "\n".join(out)
    # Last resort: merge any remaining 0-1 word line until none (ensures 'the' after 'years of' never stays alone)
    while True:
        lines = [ln.strip() for ln in text.splitlines() if ln is not None and ln.strip()]
        if len(lines) <= 1:
            break
        out = []
        i = 0
        any_one = False
        while i < len(lines):
            line = lines[i]
            nw = _word_count(line)
            if nw <= 1 and i + 1 < len(lines):
                out.append((line + " " + lines[i + 1]).strip())
                i += 2
                any_one = True
                continue
            if nw <= 1 and i == len(lines) - 1 and out:
                out[-1] = (out[-1] + " " + line).strip()
                i += 1
                any_one = True
                continue
            out.append(line)
            i += 1
        if not any_one:
            break
        text = "\n".join(out)
    return text


class CreateItemCommand(QUndoCommand):
    def __init__(self, blk_item: TextBlkItem, ctrl, page_name: str = None, mask_backup: tuple = None, parent=None):
        super().__init__(parent)
        self.blk_item = blk_item
        self.ctrl: SceneTextManager = ctrl
        self.op_count = -1
        self.page_name = page_name
        self.mask_backup = mask_backup  # (x1, y1, x2, y2, region_array) to restore mask on undo
        self.ctrl.addTextBlock(self.blk_item)
        self.pairw = self.ctrl.pairwidget_list[self.blk_item.idx]
        self.ctrl.txtblkShapeControl.setBlkItem(self.blk_item)

    def redo(self):
        if self.op_count < 0:
            self.op_count += 1
            self.blk_item.setSelected(True)
            return
        self.ctrl.recoverTextblkItemList([self.blk_item], [self.pairw])

    def undo(self):
        if self.page_name and self.ctrl.canvas.imgtrans_proj.current_img == self.page_name:
            pages = self.ctrl.canvas.imgtrans_proj.pages.get(self.page_name, [])
            blk = self.blk_item.blk
            for i in range(len(pages)):
                if pages[i] is blk:
                    pages.pop(i)
                    break
            if self.mask_backup is not None:
                x1, y1, x2, y2, region = self.mask_backup
                mask = self.ctrl.canvas.imgtrans_proj.load_mask_by_imgname(self.page_name)
                if mask is not None and 0 <= y1 < y2 <= mask.shape[0] and 0 <= x1 < x2 <= mask.shape[1]:
                    mask[y1:y2, x1:x2] = region
                    self.ctrl.canvas.imgtrans_proj.save_mask(self.page_name, mask)
        self.ctrl.deleteTextblkItemList([self.blk_item], [self.pairw])


class EmptyCommand(QUndoCommand):
    def __init__(self, parent=None):
        super().__init__(parent=parent)


class DeleteBlkItemsCommand(QUndoCommand):
    def __init__(self, blk_list: List[TextBlkItem], mode: int, ctrl, parent=None):
        super().__init__(parent)
        self.op_counter = 0
        self.blk_list = []
        self.pwidget_list: List[TransPairWidget] = []
        self.ctrl: SceneTextManager = ctrl
        self.sw = self.ctrl.canvas.search_widget
        self.canvas: Canvas = ctrl.canvas
        self.mode = mode

        self.undo_img_list = []
        self.redo_img_list = []
        self.inpaint_rect_lst = []
        self.mask_pnts = []
        img_array = self.canvas.imgtrans_proj.inpainted_array
        mask_array = self.canvas.imgtrans_proj.mask_array
        original_array = self.canvas.imgtrans_proj.img_array

        self.search_rstedit_list: List[SourceTextEdit] = []
        self.search_counter_list = []
        self.highlighter_list = []
        self.old_counter_sum = self.sw.counter_sum
        self.sw_changed = False

        blk_list.sort(key=lambda blk: blk.idx)
        
        for blkitem in blk_list:
            if not isinstance(blkitem, TextBlkItem):
                continue
            self.blk_list.append(blkitem)
            pw: TransPairWidget = ctrl.pairwidget_list[blkitem.idx]
            self.pwidget_list.append(pw)

            if mode == 1:
                is_empty = False
                msk, xyxy = get_block_mask(blkitem.absBoundingRect(), mask_array, blkitem.rotation())
                if msk is None:
                    is_empty = True
                if is_empty:
                    self.undo_img_list.append(None)
                    self.redo_img_list.append(None)
                    self.inpaint_rect_lst.append(None)
                    self.mask_pnts.append(None)
                else:
                    x1, y1, x2, y2 = xyxy
                    self.mask_pnts.append(np.where(msk))
                    self.undo_img_list.append(np.copy(img_array[y1: y2, x1: x2]))
                    self.redo_img_list.append(np.copy(original_array[y1: y2, x1: x2]))
                    self.inpaint_rect_lst.append([x1, y1, x2, y2])

            rst_idx = self.sw.get_result_edit_index(pw.e_trans)
            if rst_idx != -1:
                self.sw_changed = True
                highlighter = self.sw.highlighter_list.pop(rst_idx)
                counter = self.sw.search_counter_list.pop(rst_idx)
                self.sw.counter_sum -= counter
                if self.sw.current_edit == pw.e_trans:
                    highlighter.set_current_span(-1, -1)
                self.search_rstedit_list.append(self.sw.search_rstedit_list.pop(rst_idx))
                self.search_counter_list.append(counter)
                self.highlighter_list.append(highlighter)

            rst_idx = self.sw.get_result_edit_index(pw.e_source)
            if rst_idx != -1:
                self.sw_changed = True
                highlighter = self.sw.highlighter_list.pop(rst_idx)
                counter = self.sw.search_counter_list.pop(rst_idx)
                self.sw.counter_sum -= counter
                if self.sw.current_edit == pw.e_trans:
                    highlighter.set_current_span(-1, -1)
                self.search_rstedit_list.append(self.sw.search_rstedit_list.pop(rst_idx))
                self.search_counter_list.append(counter)
                self.highlighter_list.append(highlighter)

        self.new_counter_sum = self.sw.counter_sum
        if self.sw_changed:
            if self.sw.counter_sum > 0:
                idx = self.sw.get_result_edit_index(self.sw.current_edit)
                if self.sw.current_cursor is not None and idx != -1:
                    self.sw.result_pos = self.sw.highlighter_list[idx].matched_map[self.sw.current_cursor.position()]
                    if idx > 0:
                        self.sw.result_pos += sum(self.sw.search_counter_list[: idx])
                    self.sw.updateCounterText()
                else:
                    self.sw.setCurrentEditor(self.sw.search_rstedit_list[0])
            else:
                self.sw.setCurrentEditor(None)

        # Sync project pages: remove deleted blocks so save/run/pipeline stay consistent (Issue 9).
        self.pages_restore: List[Tuple[int, object]] = []
        page_name = self.canvas.imgtrans_proj.current_img
        if page_name and page_name in self.canvas.imgtrans_proj.pages:
            for blkitem in reversed(self.blk_list):
                idx = blkitem.idx
                removed = self.canvas.imgtrans_proj.pages[page_name].pop(idx)
                self.pages_restore.append((idx, removed))
            self.pages_restore.sort(key=lambda x: x[0])

        self.ctrl.deleteTextblkItemList(self.blk_list, self.pwidget_list)
        self.canvas.updateLayers()

    def redo(self):

        if self.mode == 1:
            self.canvas.saved_drawundo_step -= 1
            img_array = self.canvas.imgtrans_proj.inpainted_array
            mask_array = self.canvas.imgtrans_proj.mask_array
            for mskpnt, inpaint_rect, redo_img in zip(self.mask_pnts, self.inpaint_rect_lst, self.redo_img_list):
                if mskpnt == None:
                    continue
                x1, y1, x2, y2 = inpaint_rect
                img_array[y1: y2, x1: x2][mskpnt] = redo_img[mskpnt]
                mask_array[y1: y2, x1: x2][mskpnt] = 0
            self.canvas.updateLayers()

        if self.op_counter == 0:
            self.op_counter += 1
            return

        # Sync project pages again when re-doing delete (blocks were restored on undo).
        if getattr(self, 'pages_restore', None):
            page_name = self.canvas.imgtrans_proj.current_img
            if page_name and page_name in self.canvas.imgtrans_proj.pages:
                for idx, _ in sorted(self.pages_restore, key=lambda x: -x[0]):
                    self.canvas.imgtrans_proj.pages[page_name].pop(idx)

        self.ctrl.deleteTextblkItemList(self.blk_list, self.pwidget_list)
        self.canvas.updateLayers()
        if self.sw_changed:
            self.sw.counter_sum = self.new_counter_sum
            cursor_removed = False
            for edit in self.search_rstedit_list:
                idx = self.sw.get_result_edit_index(edit)
                if idx != -1:
                    self.sw.search_rstedit_list.pop(idx)
                    self.sw.search_counter_list.pop(idx)
                    self.sw.highlighter_list.pop(idx)
                if edit == self.sw.current_edit:
                    cursor_removed = True
            if cursor_removed:
                if self.sw.counter_sum > 0:
                    self.sw.setCurrentEditor(self.sw.search_rstedit_list[0])
                else:
                    self.sw.setCurrentEditor(None)

    def undo(self):

        if self.mode == 1:
            self.canvas.saved_drawundo_step += 1
            img_array = self.canvas.imgtrans_proj.inpainted_array
            mask_array = self.canvas.imgtrans_proj.mask_array
            for mskpnt, inpaint_rect, undo_img in zip(self.mask_pnts, self.inpaint_rect_lst, self.undo_img_list):
                if mskpnt == None:
                    continue
                x1, y1, x2, y2 = inpaint_rect
                img_array[y1: y2, x1: x2][mskpnt] = undo_img[mskpnt]
                mask_array[y1: y2, x1: x2][mskpnt] = 255
            self.canvas.updateLayers()

        # Restore blocks to project pages so delete/recover stays in sync (Issue 9).
        if getattr(self, 'pages_restore', None):
            page_name = self.canvas.imgtrans_proj.current_img
            if page_name and page_name in self.canvas.imgtrans_proj.pages:
                for idx, blk in self.pages_restore:
                    self.canvas.imgtrans_proj.pages[page_name].insert(idx, blk)

        self.ctrl.recoverTextblkItemList(self.blk_list, self.pwidget_list)
        self.canvas.updateLayers()
        if self.sw_changed:
            self.sw.counter_sum = self.old_counter_sum
            self.sw.search_rstedit_list += self.search_rstedit_list
            self.sw.search_counter_list += self.search_counter_list
            self.sw.highlighter_list += self.highlighter_list
            self.sw.updateCounterText()


class PasteBlkItemsCommand(QUndoCommand):
    def __init__(self, blk_list: List[TextBlkItem], pwidget_list: List[TransPairWidget], ctrl, parent=None):
        super().__init__(parent)
        self.op_counter = 0
        self.blk_list = blk_list
        self.ctrl:SceneTextManager = ctrl
        blk_list.sort(key=lambda blk: blk.idx)

        self.ctrl.canvas.block_selection_signal = True
        for blkitem in blk_list:
            blkitem.setSelected(True)
        self.ctrl.on_incanvas_selection_changed()
        self.ctrl.canvas.block_selection_signal = False
        self.pwidget_list = pwidget_list
        

    def redo(self):
        if self.op_counter == 0:
            self.op_counter += 1
            return
        self.ctrl.recoverTextblkItemList(self.blk_list, self.pwidget_list)

    def undo(self):
        self.ctrl.deleteTextblkItemList(self.blk_list, self.pwidget_list)


class PasteSrcItemsCommand(QUndoCommand):
    def __init__(self, src_list: List[SourceTextEdit], paste_list: List[str]):
        super().__init__()
        self.src_list = src_list
        self.paste_list = paste_list
        self.ori_text_list = [src.toPlainText() for src in src_list]

    def redo(self):
        for src, text in zip(self.src_list, self.paste_list):
            src.setPlainText(text)

    def undo(self):
        for src, text in zip(self.src_list, self.ori_text_list):
            src.setPlainText(text)


class RearrangeBlksCommand(QUndoCommand):

    def __init__(self, rmap: Tuple, ctrl, parent=None):
        super().__init__(parent)
        self.ctrl: SceneTextManager = ctrl
        self.src_ids, self.tgt_ids = rmap[0], rmap[1]

        self.nr = len(self.src_ids)
        self.src2tgt = {}
        self.tgt2src = {}
        for s, t in zip(self.src_ids, self.tgt_ids):
            self.src2tgt[s] = t
            self.tgt2src[t] = s
        self.visible_ = None
        self.redo_visible_idx = self.undo_visible_idx = None
        if len(rmap) > 2:
            self.redo_visible_idx, self.undo_visible_idx = rmap[2]

    def redo(self):
        self.rearange_blk_ids(self.src_ids, self.tgt_ids, self.redo_visible_idx)

    def undo(self):
        self.rearange_blk_ids(self.tgt_ids, self.src_ids, self.undo_visible_idx)

    def rearange_blk_ids(self, src_ids, tgt_ids, visible_idx = None):
        src_ids = np.array(src_ids)
        tgt_ids = np.array(tgt_ids)
        src_order_ids = np.argsort(src_ids)[::-1]

        src_ids = src_ids[src_order_ids]
        tgt_ids = tgt_ids[src_order_ids]
        
        blks: List[TextBlkItem] = []
        pws: List[TransPairWidget] = []
        for pos, pos_tgt in zip(src_ids, tgt_ids):
            pw = self.ctrl.pairwidget_list.pop(pos)
            if visible_idx == pos_tgt:
                pw.hide()
            blk = self.ctrl.textblk_item_list.pop(pos)
            pws.append(pw)
            blks.append(blk)

        tgt_order_ids = np.argsort(tgt_ids)
        for ii in tgt_order_ids:
            pos = tgt_ids[ii]
            self.ctrl.textblk_item_list.insert(pos, blks[ii])
            
            self.ctrl.textEditList.insertPairWidget(pws[ii], pos)
            self.ctrl.pairwidget_list.insert(pos, pws[ii])

        self.ctrl.updateTextBlkItemIdx(set(tgt_ids))
        if visible_idx is not None:
            pw_ct = self.ctrl.pairwidget_list[visible_idx]
            pw_ct.show()
            self.ctrl.textEditList.ensureWidgetVisible(pw_ct, yMargin=pw.height())


class TextPanel(Widget):
    def __init__(self, app: QApplication, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.textEditList = TextEditListScrollArea(self)
        self.formatpanel = FontFormatPanel(app, self)
        self.formatPanelScroll = QScrollArea(self)
        self.formatPanelScroll.setObjectName("FormatPanelScroll")
        self.formatPanelScroll.setWidget(self.formatpanel)
        self.formatPanelScroll.setWidgetResizable(True)
        self.formatPanelScroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.formatPanelScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.formatPanelScroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.formatPanelScroll.setMinimumHeight(240)
        self.formatPanelScroll.setMaximumHeight(420)
        self.formatPanelScroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        layout.addWidget(self.formatPanelScroll)
        layout.addWidget(self.textEditList, 1)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)


class SceneTextManager(QObject):
    new_textblk = Signal(int)
    def __init__(self, 
                 app: QApplication,
                 mainwindow: QWidget,
                 canvas: Canvas, 
                 textpanel: TextPanel, 
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.app = app     
        self.mainwindow = mainwindow
        self.canvas = canvas
        canvas.switch_text_item.connect(self.on_switch_textitem)
        self.selectext_minimenu: SelectTextMiniMenu = None
        self.canvas.scalefactor_changed.connect(self.adjustSceneTextRect)
        self.canvas.end_create_textblock.connect(self.onEndCreateTextBlock)
        self.canvas.paste2selected_textitems.connect(self.on_paste2selected_textitems)
        self.canvas.delete_textblks.connect(self.onDeleteBlkItems)
        self.canvas.copy_textblks.connect(self.onCopyBlkItems)
        self.canvas.paste_textblks.connect(self.onPasteBlkItems)
        self.canvas.format_textblks.connect(self.onFormatTextblks)
        self.canvas.layout_textblks.connect(self.onAutoLayoutTextblks)
        self.canvas.auto_fit_font_signal.connect(self.onAutoFitFontToBox)
        if hasattr(self.canvas, "smart_auto_fit_signal"):
            self.canvas.smart_auto_fit_signal.connect(self.onSmartAutoFitTextToBox)
        if hasattr(self.canvas, "atomic_bubble_fit_signal"):
            self.canvas.atomic_bubble_fit_signal.connect(self.onAtomicBubbleFitText)
        if hasattr(self.canvas, "atomic_bubble_fit_profile_signal"):
            self.canvas.atomic_bubble_fit_profile_signal.connect(self.onAtomicBubbleFitProfileText)
        if hasattr(self.canvas, "polish_typography_signal"):
            self.canvas.polish_typography_signal.connect(self.onPolishTypography)
        self.canvas.auto_fit_binary_signal.connect(self.onAutoFitBinarySearch)
        if hasattr(self.canvas, "re_auto_fit_selected_signal"):
            self.canvas.re_auto_fit_selected_signal.connect(self.onReAutoFitSelected)
        if hasattr(self.canvas, "re_auto_fit_page_signal"):
            self.canvas.re_auto_fit_page_signal.connect(self.onReAutoFitCurrentPage)
        if hasattr(self.canvas, "re_auto_fit_all_signal"):
            self.canvas.re_auto_fit_all_signal.connect(self.onReAutoFitAllPages)
        self.canvas.set_balloon_shape_signal.connect(self.onSetBalloonShape)
        self.canvas.resize_to_fit_content_signal.connect(self.onResizeToFitContent)
        self.canvas.center_in_bubble_signal.connect(self.onCenterInBubble)
        if hasattr(self.canvas, "set_writing_mode_signal"):
            self.canvas.set_writing_mode_signal.connect(self.onSetWritingMode)
        if hasattr(self.canvas, "recenter_text_in_box_signal"):
            self.canvas.recenter_text_in_box_signal.connect(self.onRecenterTextInBox)
        if hasattr(self.canvas, "apply_mask_safe_padding_signal"):
            self.canvas.apply_mask_safe_padding_signal.connect(self.onApplyMaskSafePadding)
        if hasattr(self.canvas, "fit_to_mask_safe_box_signal"):
            self.canvas.fit_to_mask_safe_box_signal.connect(self.onFitToMaskSafeBox)
        self.canvas.reset_angle.connect(self.onResetAngle)
        self.canvas.squeeze_blk.connect(self.onSqueezeBlk)
        self.canvas.incanvas_selection_changed.connect(self.on_incanvas_selection_changed)
        self.txtblkShapeControl = canvas.txtblkShapeControl
        self.textpanel = textpanel
        self.textEditList = textpanel.textEditList
        self.textEditList.focus_out.connect(self.on_textedit_list_focusout)
        self.textEditList.textpanel_contextmenu_requested.connect(mainwindow.on_texteditlist_contextmenu_requested)
        self.textEditList.selection_changed.connect(self.on_transwidget_selection_changed)
        self.textEditList.rearrange_blks.connect(self.on_rearrange_blks)
        self.formatpanel = textpanel.formatpanel
        self.formatpanel.textstyle_panel.apply_fontfmt.connect(self.onFormatTextblks)
        self.formatpanel.apply_global_to_all_blocks_requested.connect(self.apply_global_format_to_all_blocks)

        self.imgtrans_proj = self.canvas.imgtrans_proj
        self.textblk_item_list: List[TextBlkItem] = []
        self.pairwidget_list: List[TransPairWidget] = self.textEditList.pairwidget_list

        self.auto_textlayout_flag = False
        self.hovering_transwidget : TransTextEdit = None

        self.prev_blkitem: TextBlkItem = None
        self._moving_blur_item: TextBlkItem = None  # reserved for optional drag effect cleanup


    def onSetWritingMode(self, mode: str):
        from ui import fontformat_commands as FC
        blkitems = self.canvas.selected_text_items()
        if not blkitems:
            return
        FC.ffmt_change_writing_mode('writing_mode', mode, self.formatpanel.global_format, False, blkitems)
        self.updateSceneTextitems()
        self.canvas.setProjSaveState(False)

    def onRecenterTextInBox(self):
        for blkitem in self.canvas.selected_text_items():
            rect = blkitem.boundingRect()
            doc_size = blkitem.document().size()
            dx = (rect.width() - doc_size.width()) / 2.0
            dy = (rect.height() - doc_size.height()) / 2.0
            try:
                blkitem.document().setDocumentMargin(max(0.0, blkitem.document().documentMargin() + min(dx, dy, 8.0)))
            except Exception:
                pass
            blkitem.update()
        self.canvas.setProjSaveState(False)

    def onApplyMaskSafePadding(self):
        changed = 0
        for blkitem in self.canvas.selected_text_items():
            blk = getattr(blkitem, 'blk', None)
            if blk is None or getattr(blk, 'text_mask', None) is None:
                continue
            current = float(getattr(blkitem.fontformat, 'text_padding', blkitem.padding()) or 0.0)
            padding = recommended_padding_for_mask(getattr(blk, 'text_mask', None), current_padding=current)
            if padding <= current + 0.01:
                continue
            blkitem.fontformat.text_padding = padding
            blkitem.setPadding(padding)
            blk.fontformat.text_padding = padding
            changed += 1
            blkitem.update()
        if changed:
            self.canvas.setProjSaveState(True)
            if hasattr(self.mainwindow, 'pipelineInsightsPanel'):
                self.mainwindow.pipelineInsightsPanel.add_event('TYPO_QA', self.tr('Applied mask-safe padding to {0} text box(es)').format(changed))

    def on_switch_textitem(self, switch_delta: int, key_event: QKeyEvent = None, current_editing_widget: Union[SourceTextEdit, TransTextEdit] = None):
        n_blk = len(self.textblk_item_list)
        if n_blk < 1:
            return
        
        editing_blk = None
        if current_editing_widget is None:
            editing_blk = self.editingTextItem()
            if editing_blk is not None:
                tgt_idx = editing_blk.idx + switch_delta
            else:
                sel_blks = self.canvas.selected_text_items(sort=False)
                if len(sel_blks) == 0:
                    return
                sel_blk = sel_blks[0]
                tgt_idx = sel_blk.idx + switch_delta
        else:
            tgt_idx = current_editing_widget.idx + switch_delta

        if tgt_idx < 0:
            tgt_idx += n_blk
        elif tgt_idx >= n_blk:
            tgt_idx -= n_blk
        blk = self.textblk_item_list[tgt_idx]

        if current_editing_widget is None:
            if editing_blk is None:
                self.canvas.block_selection_signal = True
                self.canvas.clearSelection()
                blk.setSelected(True)
                self.canvas.block_selection_signal = False
                self.canvas.gv.ensureVisible(blk)
                self.txtblkShapeControl.setBlkItem(blk)
                edit = self.pairwidget_list[tgt_idx].e_trans
                self.changeHoveringWidget(edit)
                self.textEditList.set_selected_list([blk.idx])
            else:
                editing_blk.endEdit()
                editing_blk.setSelected(False)
                self.txtblkShapeControl.setBlkItem(blk)
                blk.setSelected(True)
                blk.startEdit()
                self.canvas.gv.ensureVisible(blk)
        else:
            self.textblk_item_list[current_editing_widget.idx].setSelected(False)
            current_pw = self.pairwidget_list[tgt_idx]
            is_trans = isinstance(current_editing_widget, TransTextEdit)
            if is_trans:
                w = current_pw.e_trans
            else:
                w = current_pw.e_source

            self.changeHoveringWidget(w)
            w.setFocus()

        if key_event is not None:
            key_event.accept()

    def setTextEditMode(self, edit: bool = False):
        if edit:
            self.textpanel.show()
            self.canvas.textLayer.show()
        else:
            self.txtblkShapeControl.setBlkItem(None)
            self.textpanel.hide()
            self.textpanel.formatpanel.set_textblk_item()
            self.canvas.textLayer.hide()

    def adjustSceneTextRect(self):
        self.txtblkShapeControl.updateBoundingRect()

    def clearSceneTextitems(self):
        self.hovering_transwidget = None
        self.txtblkShapeControl.setBlkItem(None)
        for blkitem in self.textblk_item_list:
            self.canvas.removeItem(blkitem)
            # Explicitly schedule deletion to free QTextDocument + pixmap memory (#issue-9.5)
            try:
                blkitem.deleteLater()
            except Exception:
                pass
        self.textblk_item_list.clear()
        self.textEditList.clearAllSelected()
        for textwidget in self.pairwidget_list:
            self.textEditList.removeWidget(textwidget)
        self.pairwidget_list.clear()

    def updateSceneTextitems(self):
        self.hovering_transwidget = None
        self.txtblkShapeControl.setBlkItem(None)
        self.clearSceneTextitems()
        # Flag used when adding new blocks from Run (to run layout there). Do not re-run layout
        # when loading a page from project, so saved text box size/format is preserved on reopen.
        self.auto_textlayout_flag = (
            pcfg.let_autolayout_flag
            and (pcfg.let_fntsize_flag == 0)
            and (pcfg.module.enable_detect or pcfg.module.enable_ocr or pcfg.module.enable_translate)
        )
        for textblock in self.imgtrans_proj.current_block_list():
            if textblock.font_family is None or textblock.font_family.strip() == '':
                textblock.font_family = self.formatpanel.familybox.currentText()
            blk_item = self.addTextBlock(textblock)

    def addTextBlock(self, blk: Union[TextBlock, TextBlkItem] = None) -> TextBlkItem:
        if isinstance(blk, TextBlkItem):
            blk_item = blk
            blk_item.idx = len(self.textblk_item_list)
        else:
            # Apply global stroke default to new blocks when they have no stroke (#1143)
            if blk.fontformat.stroke_width == 0 and getattr(pcfg.global_fontformat, 'stroke_width', 0) > 0:
                blk.fontformat.stroke_width = pcfg.global_fontformat.stroke_width
                blk.fontformat.srgb = list(getattr(pcfg.global_fontformat, 'srgb', [0, 0, 0]))
            if getattr(blk.fontformat, 'secondary_stroke_width', 0.0) == 0 and getattr(pcfg.global_fontformat, 'secondary_stroke_width', 0) > 0:
                blk.fontformat.secondary_stroke_width = pcfg.global_fontformat.secondary_stroke_width
                blk.fontformat.secondary_srgb = list(getattr(pcfg.global_fontformat, 'secondary_srgb', [255, 255, 255]))
            # Apply global text box corner radius default (shape) when present: 0 = rectangle, >0 = rounded / circle-like.
            gb_radius = float(getattr(pcfg.global_fontformat, "text_box_corner_radius", 0.0) or 0.0)
            if gb_radius > 0 and getattr(blk.fontformat, "text_box_corner_radius", 0.0) == 0.0:
                blk.fontformat.text_box_corner_radius = gb_radius
            # Load block as-is (saved layout). Auto layout is only run explicitly after Run or via menu.
            if self.auto_textlayout_flag and bool(getattr(pcfg, 'render_auto_polish_on_ocr', True)):
                try:
                    self._polish_textblock_model(blk)
                except Exception as e:
                    LOGGER.warning('Auto typography polish skipped: %s', e)
            blk_item = TextBlkItem(blk, len(self.textblk_item_list), show_rect=self.canvas.textblock_mode)
        self.addTextBlkItem(blk_item)

        pair_widget = TransPairWidget(blk, len(self.pairwidget_list), pcfg.fold_textarea)
        self.pairwidget_list.append(pair_widget)
        self.textEditList.addPairWidget(pair_widget)
        pair_widget.e_source.setPlainText(blk_item.blk.get_text())
        pair_widget.e_source.focus_in.connect(self.on_transwidget_focus_in)
        pair_widget.e_source.ensure_scene_visible.connect(self.on_ensure_textitem_svisible)
        pair_widget.e_source.push_undo_stack.connect(self.on_push_edit_stack)
        pair_widget.e_source.redo_signal.connect(self.on_textedit_redo)
        pair_widget.e_source.undo_signal.connect(self.on_textedit_undo)
        pair_widget.e_source.show_select_menu.connect(self.on_show_select_menu)
        pair_widget.e_source.focus_out.connect(self.on_pairw_focusout)
        pair_widget.e_source.select_all_blocks_requested.connect(lambda: self.set_blkitems_selection(True))
        pair_widget.e_trans.select_all_blocks_requested.connect(lambda: self.set_blkitems_selection(True))

        pair_widget.e_trans.setPlainText(blk_item.toPlainText())
        pair_widget.e_trans.focus_in.connect(self.on_transwidget_focus_in)
        pair_widget.e_trans.propagate_user_edited.connect(self.on_propagate_transwidget_edit)
        pair_widget.e_trans.ensure_scene_visible.connect(self.on_ensure_textitem_svisible)
        pair_widget.e_trans.push_undo_stack.connect(self.on_push_edit_stack)
        pair_widget.e_trans.redo_signal.connect(self.on_textedit_redo)
        pair_widget.e_trans.undo_signal.connect(self.on_textedit_undo)
        pair_widget.e_trans.show_select_menu.connect(self.on_show_select_menu)
        pair_widget.e_trans.focus_out.connect(self.on_pairw_focusout)
        pair_widget.drag_move.connect(self.textEditList.handle_drag_pos)
        pair_widget.pw_drop.connect(self.textEditList.on_pw_dropped)
        pair_widget.idx_edited.connect(self.textEditList.on_idx_edited)

        self.new_textblk.emit(blk_item.idx)
        return blk_item

    def insertTextBlocksAt(self, idx: int, blk_list: List[TextBlock]) -> List[TextBlkItem]:
        """Insert new text blocks at index idx (used e.g. after splitting a region). Caller must update imgtrans_proj.pages[page_name] before calling."""
        if idx < 0 or idx > len(self.textblk_item_list) or not blk_list:
            return []
        new_items = []
        new_widgets = []
        for k, blk in enumerate(blk_list):
            blk_item = TextBlkItem(blk, idx + k, show_rect=self.canvas.textblock_mode)
            pair_widget = TransPairWidget(blk, idx + k, pcfg.fold_textarea)
            pair_widget.e_source.setPlainText(blk_item.blk.get_text())
            pair_widget.e_trans.setPlainText(blk_item.toPlainText())
            pair_widget.e_source.focus_in.connect(self.on_transwidget_focus_in)
            pair_widget.e_source.ensure_scene_visible.connect(self.on_ensure_textitem_svisible)
            pair_widget.e_source.push_undo_stack.connect(self.on_push_edit_stack)
            pair_widget.e_source.redo_signal.connect(self.on_textedit_redo)
            pair_widget.e_source.undo_signal.connect(self.on_textedit_undo)
            pair_widget.e_source.show_select_menu.connect(self.on_show_select_menu)
            pair_widget.e_source.focus_out.connect(self.on_pairw_focusout)
            pair_widget.e_source.select_all_blocks_requested.connect(lambda: self.set_blkitems_selection(True))
            pair_widget.e_trans.select_all_blocks_requested.connect(lambda: self.set_blkitems_selection(True))
            pair_widget.e_trans.focus_in.connect(self.on_transwidget_focus_in)
            pair_widget.e_trans.propagate_user_edited.connect(self.on_propagate_transwidget_edit)
            pair_widget.e_trans.ensure_scene_visible.connect(self.on_ensure_textitem_svisible)
            pair_widget.e_trans.push_undo_stack.connect(self.on_push_edit_stack)
            pair_widget.e_trans.redo_signal.connect(self.on_textedit_redo)
            pair_widget.e_trans.undo_signal.connect(self.on_textedit_undo)
            pair_widget.e_trans.show_select_menu.connect(self.on_show_select_menu)
            pair_widget.e_trans.focus_out.connect(self.on_pairw_focusout)
            pair_widget.drag_move.connect(self.textEditList.handle_drag_pos)
            pair_widget.pw_drop.connect(self.textEditList.on_pw_dropped)
            pair_widget.idx_edited.connect(self.textEditList.on_idx_edited)
            new_items.append(blk_item)
            new_widgets.append(pair_widget)
        for k in range(len(blk_list)):
            self.textblk_item_list.insert(idx + k, new_items[k])
            self.pairwidget_list.insert(idx + k, new_widgets[k])
            self.textEditList.insertPairWidget(new_widgets[k], idx + k)
        for blk_item in new_items:
            blk_item.setParentItem(self.canvas.textLayer)
            blk_item.begin_edit.connect(self.onTextBlkItemBeginEdit)
            blk_item.end_edit.connect(self.onTextBlkItemEndEdit)
            blk_item.hover_enter.connect(self.onTextBlkItemHoverEnter)
            blk_item.leftbutton_pressed.connect(self.onLeftbuttonPressed)
            blk_item.moving.connect(self.onTextBlkItemMoving)
            blk_item.moved.connect(self.onTextBlkItemMoved)
            blk_item.reshaped.connect(self.onTextBlkItemReshaped)
            blk_item.rotated.connect(self.onTextBlkItemRotated)
            blk_item.warped.connect(self.onTextBlkItemWarped)
            blk_item.push_undo_stack.connect(self.on_push_textitem_undostack)
            blk_item.undo_signal.connect(self.on_textedit_undo)
            blk_item.redo_signal.connect(self.on_textedit_redo)
            blk_item.propagate_user_edited.connect(self.on_propagate_textitem_edit)
            blk_item.doc_size_changed.connect(self.onTextBlkItemSizeChanged)
            blk_item.pasted.connect(self.onBlkitemPaste)
        self.updateTextBlkItemIdx()
        for k in range(len(blk_list)):
            self.new_textblk.emit(idx + k)
        return new_items

    def addTextBlkItem(self, textblk_item: TextBlkItem) -> TextBlkItem:
        self.textblk_item_list.append(textblk_item)
        textblk_item.setParentItem(self.canvas.textLayer)
        textblk_item.begin_edit.connect(self.onTextBlkItemBeginEdit)
        textblk_item.end_edit.connect(self.onTextBlkItemEndEdit)
        textblk_item.hover_enter.connect(self.onTextBlkItemHoverEnter)
        textblk_item.leftbutton_pressed.connect(self.onLeftbuttonPressed)
        textblk_item.moving.connect(self.onTextBlkItemMoving)
        textblk_item.moved.connect(self.onTextBlkItemMoved)
        textblk_item.reshaped.connect(self.onTextBlkItemReshaped)
        textblk_item.rotated.connect(self.onTextBlkItemRotated)
        textblk_item.warped.connect(self.onTextBlkItemWarped)
        textblk_item.push_undo_stack.connect(self.on_push_textitem_undostack)
        textblk_item.undo_signal.connect(self.on_textedit_undo)
        textblk_item.redo_signal.connect(self.on_textedit_redo)
        textblk_item.propagate_user_edited.connect(self.on_propagate_textitem_edit)
        textblk_item.doc_size_changed.connect(self.onTextBlkItemSizeChanged)
        textblk_item.pasted.connect(self.onBlkitemPaste)
        return textblk_item

    def deleteTextblkItemList(self, blkitem_list: List[TextBlkItem], p_widget_list: List[TransPairWidget]):
        page_name = getattr(self.canvas.imgtrans_proj, 'current_img', None) if self.canvas.imgtrans_proj is not None else None
        page_index = -1
        if self.canvas.imgtrans_proj is not None and page_name in self.canvas.imgtrans_proj.pages:
            page_keys = list(self.canvas.imgtrans_proj.pages.keys())
            page_index = page_keys.index(page_name)
        log_diagnostic_event(
            "ui.blocks_delete",
            page_name=page_name,
            page_index=page_index,
            block_ids=[blkitem.idx for blkitem in blkitem_list],
            block_count=len(blkitem_list),
        )
        selection_changed = False
        for blkitem, p_widget in zip(blkitem_list, p_widget_list):
            if blkitem.isSelected():
                selection_changed = True
            self.canvas.removeItem(blkitem) # removeItem itself will block incanvas_selection_changed
            self.textblk_item_list.remove(blkitem)
            self.pairwidget_list.remove(p_widget)
            self.textEditList.removeWidget(p_widget)
        self.updateTextBlkItemIdx()
        self.txtblkShapeControl.setBlkItem(None)
        if selection_changed:
            # it must be called after updateTextBlkItemIdx if blk.idx changed
            self.on_incanvas_selection_changed()

    def swap_block_positions(self, i: int, j: int):
        """Swap two text blocks (and their pair widgets) at indices i and j. Updates project page list and scene."""
        if i == j or i < 0 or j < 0 or i >= len(self.textblk_item_list) or j >= len(self.textblk_item_list):
            return
        page_name = self.canvas.imgtrans_proj.current_img
        if not page_name or page_name not in self.canvas.imgtrans_proj.pages:
            return
        pages = self.canvas.imgtrans_proj.pages[page_name]
        if len(pages) != len(self.textblk_item_list):
            return
        pages[i], pages[j] = pages[j], pages[i]
        self.textblk_item_list[i], self.textblk_item_list[j] = self.textblk_item_list[j], self.textblk_item_list[i]
        self.pairwidget_list[i], self.pairwidget_list[j] = self.pairwidget_list[j], self.pairwidget_list[i]
        self.textblk_item_list[i].idx = i
        self.textblk_item_list[j].idx = j
        self.pairwidget_list[i].idx = i
        self.pairwidget_list[j].idx = j
        self.textEditList.removeWidget(self.pairwidget_list[i])
        self.textEditList.removeWidget(self.pairwidget_list[j])
        self.textEditList.insertPairWidget(self.pairwidget_list[i], i)
        self.textEditList.insertPairWidget(self.pairwidget_list[j], j)

    def recoverTextblkItemList(self, blkitem_list: List[TextBlkItem], p_widget_list: List[TransPairWidget]):
        page_name = getattr(self.canvas.imgtrans_proj, 'current_img', None) if self.canvas.imgtrans_proj is not None else None
        page_index = -1
        if self.canvas.imgtrans_proj is not None and page_name in self.canvas.imgtrans_proj.pages:
            page_keys = list(self.canvas.imgtrans_proj.pages.keys())
            page_index = page_keys.index(page_name)
        log_diagnostic_event(
            "ui.blocks_recover",
            page_name=page_name,
            page_index=page_index,
            block_ids=[blkitem.idx for blkitem in blkitem_list],
            block_count=len(blkitem_list),
        )
        self.canvas.block_selection_signal = True
        for blkitem, p_widget in zip(blkitem_list, p_widget_list):
            self.textblk_item_list.insert(blkitem.idx, blkitem)
            blkitem.setParentItem(self.canvas.textLayer)
            self.pairwidget_list.insert(p_widget.idx, p_widget)
            self.textEditList.insertPairWidget(p_widget, p_widget.idx)
            if self.txtblkShapeControl.blk_item is not None and blkitem.isSelected():
                blkitem.setSelected(False)
        self.updateTextBlkItemIdx()
        self.on_incanvas_selection_changed()
        self.canvas.block_selection_signal = False
        
    def onTextBlkItemSizeChanged(self, idx: int):
        blk_item = self.textblk_item_list[idx]
        if not self.txtblkShapeControl.reshaping:
            if self.txtblkShapeControl.blk_item == blk_item:
                self.txtblkShapeControl.updateBoundingRect()

    @property
    def app_clipborad(self) -> QClipboard:
        return self.app.clipboard()

    def onBlkitemPaste(self, idx: int):
        blk_item = self.textblk_item_list[idx]
        text = self.app_clipborad.text()
        cursor = blk_item.textCursor()
        cursor.insertText(text)

    def onTextBlkItemBeginEdit(self, blk_id: int):
        blk_item = self.textblk_item_list[blk_id]
        self.txtblkShapeControl.setBlkItem(blk_item)
        self.canvas.editing_textblkitem = blk_item
        self.formatpanel.set_textblk_item(blk_item)
        self.txtblkShapeControl.startEditing()
        e_trans = self.pairwidget_list[blk_item.idx].e_trans
        self.changeHoveringWidget(e_trans)

    def changeHoveringWidget(self, edit: SourceTextEdit):
        if self.hovering_transwidget is not None and self.hovering_transwidget != edit:
            self.hovering_transwidget.setHoverEffect(False)
        self.hovering_transwidget = edit
        if edit is not None:
            pw = self.pairwidget_list[edit.idx]
            h = pw.height()
            if shared.USE_PYSIDE6:
                self.textEditList.ensureWidgetVisible(pw, ymargin=h)
            else:
                self.textEditList.ensureWidgetVisible(pw, yMargin=h)
            edit.setHoverEffect(True)

    def onLeftbuttonPressed(self, blk_id: int):
        blk_item = self.textblk_item_list[blk_id]
        self.txtblkShapeControl.setBlkItem(blk_item)
        selections: List[TextBlkItem] = self.canvas.selectedItems()
        if len(selections) > 1:
            for item in selections:
                item.oldPos = item.pos()
        self.changeHoveringWidget(self.pairwidget_list[blk_id].e_trans)
        # Select the corresponding pair widget on the right bar (no focus in source/translation)
        self.textEditList.set_selected_list([blk_id])

    def onTextBlkItemEndEdit(self, blk_id: int):
        self.canvas.editing_textblkitem = None
        self.textblk_item_list[blk_id].setSelected(True)
        self.txtblkShapeControl.endEditing()

    def editingTextItem(self) -> TextBlkItem:
        if self.txtblkShapeControl.isVisible() and self.canvas.editing_textblkitem is not None:
            return self.canvas.editing_textblkitem
        return None

    def savePrevBlkItem(self, blkitem: TextBlkItem):
        self.prev_blkitem = blkitem
        self.prev_textCursor = QTextCursor(self.prev_blkitem.textCursor())

    def is_editting(self):
        blk_item = self.txtblkShapeControl.blk_item
        return blk_item is not None and blk_item.is_editting()

    def onTextBlkItemHoverEnter(self, blk_id: int):
        if self.is_editting():
            return
        blk_item = self.textblk_item_list[blk_id]
        if not blk_item.hasFocus():
            self.txtblkShapeControl.setBlkItem(blk_item)

    def onTextBlkItemMoving(self, item: TextBlkItem):
        # Skip QGraphicsBlurEffect during drag to avoid "A paint device can only be painted by one painter at a time" / Painter not active spam
        self.txtblkShapeControl.updateBoundingRect()

    def onTextBlkItemMoved(self):
        if self._moving_blur_item is not None:
            self._moving_blur_item.setGraphicsEffect(None)
            self._moving_blur_item = None
        selected_blks = self.canvas.selected_text_items()
        if len(selected_blks) > 0:
            self.canvas.push_undo_command(MoveBlkItemsCommand(selected_blks, self.txtblkShapeControl))
        
    def onTextBlkItemReshaped(self, item: TextBlkItem):
        self.canvas.push_undo_command(ReshapeItemCommand(item))

    def onTextBlkItemRotated(self, new_angle: float):
        blk_item = self.txtblkShapeControl.blk_item
        if blk_item:
            self.canvas.push_undo_command(RotateItemCommand(blk_item, new_angle, self.txtblkShapeControl))

    def onTextBlkItemWarped(self, before: dict, after: dict):
        """PR #1105: Push undo command for quad warp change."""
        item = self.sender()
        if isinstance(item, TextBlkItem):
            self.canvas.push_undo_command(WarpItemCommand(item, before, after, self.txtblkShapeControl))

    def onDeleteBlkItems(self, mode: int):
        selected_blks = self.canvas.selected_text_items()
        if len(selected_blks) == 0 and self.txtblkShapeControl.blk_item is not None:
            selected_blks.append(self.txtblkShapeControl.blk_item)
        if len(selected_blks) > 0:
            self.canvas.push_undo_command(DeleteBlkItemsCommand(selected_blks, mode, self))

    def onCopyBlkItems(self):
        selected_blks = self.canvas.selected_text_items()
        if len(selected_blks) == 0 and self.txtblkShapeControl.blk_item is not None:
            selected_blks.append(self.txtblkShapeControl.blk_item)

        if len(selected_blks) == 0:            
            return

        self.canvas.clipboard_blks.clear()
        if self.canvas.text_change_unsaved():
            self.updateTextBlkList()

        pos = selected_blks[0].blk.bounding_rect()
        pos_x = int(pos[0] + pos[2] / 2)
        pos_y = int(pos[1] + pos[3] / 2)

        textlist = []
        for blkitem in selected_blks:
            blk = copy.deepcopy(blkitem.blk)
            blk.adjust_pos(-pos_x, -pos_y)
            self.canvas.clipboard_blks.append(blk)
            textlist.append(blkitem.toPlainText().strip())
        textlist = '\n'.join(textlist)
        self.app_clipborad.setText(textlist, QClipboard.Mode.Clipboard)


    def onPasteBlkItems(self, pos: QPointF):
        if pos is None:
            pos_x, pos_y = 0, 0
        else:
            pos_x, pos_y = pos.x(), pos.y()
            pos_x = int(pos_x / self.canvas.scale_factor)
            pos_y = int(pos_y / self.canvas.scale_factor)
        blkitem_list, pair_widget_list = [], []
        for blk in self.canvas.clipboard_blks:
            blk = copy.deepcopy(blk)
            blk.adjust_pos(pos_x, pos_y)
            blkitem = self.addTextBlock(blk)
            pairw = self.pairwidget_list[-1]
            blkitem_list.append(blkitem)
            pair_widget_list.append(pairw)
        if len(blkitem_list) > 0:
            self.canvas.clearSelection()
            self.canvas.push_undo_command(PasteBlkItemsCommand(blkitem_list, pair_widget_list, self))
            if len(blkitem_list) == 1:
                self.formatpanel.set_textblk_item(blkitem_list[0])
            else:
                self.formatpanel.set_textblk_item(multi_select=True)

    def onFormatTextblks(self, fmt: FontFormat = None):
        if fmt is None:
            fmt = self.formatpanel.global_format
        self.apply_fontformat(fmt)

    def _resolve_balloon_shape_for_block(self, blkitem: TextBlkItem) -> None:
        """Resolve and set balloon shape (Auto → round/diamond/etc.) for one block only when it has a detected bubble.
        Skip for free-standing text (no mask) so we don't run contour detection or move boxes."""
        if getattr(blkitem.fontformat, "vertical", False):
            return
        img = self.imgtrans_proj.img_array
        if img is None:
            return
        im_h, im_w = img.shape[:2]
        br = blkitem.absBoundingRect(max_h=im_h, max_w=im_w)
        if not isinstance(br, list):
            br = [br.x(), br.y(), br.width(), br.height()]
        if br[2] <= 0 or br[3] <= 0:
            return
        # Only resolve shape when block has a detected bubble (mask); free-standing text stays "auto"
        page_name = getattr(self.imgtrans_proj, "current_img", None)
        if page_name:
            try:
                mask_array = self.imgtrans_proj.load_mask_by_imgname(page_name)
                if mask_array is None or mask_array.size == 0 or mask_array.shape[:2] != (im_h, im_w):
                    blkitem.fontformat.text_box_shape = "auto"
                    return
                mask, _ = get_block_mask(br, mask_array, blkitem.rotation())
                if mask is None or mask.size == 0:
                    blkitem.fontformat.text_box_shape = "auto"
                    return
            except Exception:
                blkitem.fontformat.text_box_shape = "auto"
                return
        else:
            blkitem.fontformat.text_box_shape = "auto"
            return
        try:
            cx, cy = br[0] + br[2] / 2, br[1] + br[3] / 2
            enlarge_ratio = min(max(br[2] / br[3], br[3] / br[2]) * 1.5, 3.0)
            mask, _area, mask_xyxy, region_rect = extract_ballon_region(
                img, br, enlarge_ratio=enlarge_ratio, cal_region_rect=True, margin_px=LAYOUT_BALLOON_SHAPE_CROP_PAD,
                seed_point_abs=(cx, cy),
            )
        except Exception:
            blkitem.fontformat.text_box_shape = "auto"
            return
        shape_cfg = getattr(pcfg.module, "layout_balloon_shape", "auto") or "auto"
        if shape_cfg != "auto":
            blkitem.fontformat.text_box_shape = shape_cfg
            return
        if region_rect is None or len(region_rect) < 4 or region_rect[2] <= 0 or region_rect[3] <= 0:
            blkitem.fontformat.text_box_shape = "auto"
            return
        rw, rh = region_rect[2], region_rect[3]
        balloon_shape = _resolve_auto_balloon_shape(
            getattr(pcfg.module, "layout_balloon_shape_auto_method", "contour_ratio"),
            mask, mask_xyxy, img, rw, rh,
        )
        blkitem.fontformat.text_box_shape = balloon_shape or "auto"

    def _collect_autolayout_targets(self, *, selected_only: bool = False, include_user_adjusted: bool = False) -> List[TextBlkItem]:
        src = self.canvas.selected_text_items() if selected_only else self.textblk_item_list
        out: List[TextBlkItem] = []
        skip_user_adjusted = bool(getattr(getattr(pcfg, 'module', None), 'layout_skip_user_adjusted', True))
        for blkitem in src:
            if getattr(blkitem.fontformat, 'vertical', False):
                continue
            if not bool(getattr(blkitem.fontformat, 'auto_fit_enabled', True)):
                continue
            if (not include_user_adjusted) and skip_user_adjusted and bool(getattr(blkitem.fontformat, 'user_adjusted', False)):
                continue
            out.append(blkitem)
        return out

    def run_auto_layout_on_current_page_once(self, include_user_adjusted: bool = False):
        """Run auto layout on all blocks of the current page (e.g. after opening project so first page is formatted)."""
        from utils.config import pcfg
        all_blks = self._collect_autolayout_targets(selected_only=False, include_user_adjusted=include_user_adjusted)
        if not all_blks:
            return
        old_autolayout = getattr(pcfg, 'let_autolayout_flag', True)
        old_flag = self.auto_textlayout_flag
        try:
            pcfg.let_autolayout_flag = True
            self.auto_textlayout_flag = True
            # Resolve balloon shape first so layout can use correct shape for line breaking and box outline.
            for blkitem in all_blks:
                self._resolve_balloon_shape_for_block(blkitem)
            for blkitem in all_blks:
                self.layout_textblk(blkitem)
                self.layout_textblk(blkitem)
            for blkitem in all_blks:
                if blkitem.idx < len(self.pairwidget_list):
                    merged = _merge_stub_lines_in_text(blkitem.toPlainText())
                    blkitem.setPlainText(merged)
                    self.pairwidget_list[blkitem.idx].e_trans.setPlainText(merged)
            # Optionally center each box in its bubble, skipping boxes that are close to another (combined bubbles).
            if getattr(pcfg.module, "layout_center_in_bubble_after_autolayout", True):
                img = self.imgtrans_proj.img_array
                if img is not None:
                    im_h, im_w = img.shape[:2]
                    gap = float(getattr(pcfg.module, "layout_center_in_bubble_min_gap_px", 40) or 40)
                    for blkitem in all_blks:
                        if self._is_blk_close_to_others(blkitem, all_blks, gap):
                            continue
                        self._center_blkitem_in_bubble(blkitem, img, im_w, im_h)
            self.canvas.setProjSaveState(True)
        finally:
            pcfg.let_autolayout_flag = old_autolayout
            self.auto_textlayout_flag = old_flag

    def re_auto_fit_selected_textboxes(self, include_user_adjusted: bool = True):
        selected_blks = self._collect_autolayout_targets(selected_only=True, include_user_adjusted=include_user_adjusted)
        if not selected_blks:
            return 0
        old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
        for blkitem in selected_blks:
            old_html_lst.append(blkitem.toHtml())
            old_rect_lst.append(blkitem.absBoundingRect(qrect=True))
            trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
            self._resolve_balloon_shape_for_block(blkitem)
            self._apply_smart_fit_to_item(blkitem)
            self.layout_textblk(blkitem)
        self.canvas.push_undo_command(AutoLayoutCommand(selected_blks, old_rect_lst, old_html_lst, trans_widget_lst))
        self.canvas.setProjSaveState(True)
        return len(selected_blks)

    def re_auto_fit_current_page(self, include_user_adjusted: bool = False, push_undo: bool = True):
        targets = self._collect_autolayout_targets(selected_only=False, include_user_adjusted=include_user_adjusted)
        if not targets:
            return 0
        old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
        for blkitem in targets:
            old_html_lst.append(blkitem.toHtml())
            old_rect_lst.append(blkitem.absBoundingRect(qrect=True))
            trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
            self._resolve_balloon_shape_for_block(blkitem)
            self._apply_smart_fit_to_item(blkitem)
            self.layout_textblk(blkitem)
        if push_undo:
            self.canvas.push_undo_command(AutoLayoutCommand(targets, old_rect_lst, old_html_lst, trans_widget_lst))
        self.canvas.setProjSaveState(True)
        return len(targets)

    def re_auto_fit_all_pages(self, include_user_adjusted: bool = False):
        # Current project view keeps one page loaded in scene; for now this API
        # re-fits the active page and is safe to call from future multi-page loop.
        return self.re_auto_fit_current_page(include_user_adjusted=include_user_adjusted, push_undo=True)

    def onReAutoFitSelected(self):
        self.re_auto_fit_selected_textboxes(include_user_adjusted=True)

    def onReAutoFitCurrentPage(self):
        self.re_auto_fit_current_page(include_user_adjusted=False)

    def onReAutoFitAllPages(self):
        self.re_auto_fit_all_pages(include_user_adjusted=False)

    def run_detect_post_autofit_on_current_page_once(self, block_indices: List[int]):
        """Apply the same auto-fit routine as shortcut Format→Auto fit to selected detected blocks."""
        from utils.config import pcfg
        if not block_indices:
            return
        selected_blks = []
        for idx in block_indices:
            if 0 <= idx < len(self.textblk_item_list):
                blkitem = self.textblk_item_list[idx]
                if not getattr(blkitem.fontformat, 'vertical', False):
                    selected_blks.append(blkitem)
        if not selected_blks:
            return
        old_autolayout = getattr(pcfg, 'let_autolayout_flag', True)
        old_flag = self.auto_textlayout_flag
        try:
            pcfg.let_autolayout_flag = True
            self.auto_textlayout_flag = True
            for blkitem in selected_blks:
                self.layout_textblk(blkitem)
                self.layout_textblk(blkitem)
                self._scale_font_to_fit_box(blkitem)
            self.canvas.setProjSaveState(True)
        finally:
            pcfg.let_autolayout_flag = old_autolayout
            self.auto_textlayout_flag = old_flag

    def onAutoLayoutTextblks(self):
        selected_blks = self._collect_autolayout_targets(selected_only=True, include_user_adjusted=True)
        old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
        if len(selected_blks) > 0:
            for blkitem in selected_blks:
                old_html_lst.append(blkitem.toHtml())
                old_rect_lst.append(blkitem.absBoundingRect(qrect=True))
                trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
                # Resolve balloon shape before layout so lines and box match the bubble.
                self._resolve_balloon_shape_for_block(blkitem)
                # Run auto layout twice: second pass uses the new box geometry to refine line breaks.
                self.layout_textblk(blkitem)
                self.layout_textblk(blkitem)
            # Optionally center each box in its bubble, skipping boxes close to another (combined bubbles).
            if getattr(pcfg.module, "layout_center_in_bubble_after_autolayout", True):
                img = self.imgtrans_proj.img_array
                if img is not None:
                    im_h, im_w = img.shape[:2]
                    gap = float(getattr(pcfg.module, "layout_center_in_bubble_min_gap_px", 40) or 40)
                    for blkitem in selected_blks:
                        if self._is_blk_close_to_others(blkitem, selected_blks, gap):
                            continue
                        self._center_blkitem_in_bubble(blkitem, img, im_w, im_h)
            self.canvas.push_undo_command(AutoLayoutCommand(selected_blks, old_rect_lst, old_html_lst, trans_widget_lst))

    def onAutoFitFontToBox(self):
        """Run layout (stub-aware line breaks) then scale font so text fits the selected box(es). Avoids Qt reflow introducing single-word lines."""
        selected_blks = self.canvas.selected_text_items()
        selected_blks = [blk for blk in selected_blks if not blk.fontformat.vertical]
        if not selected_blks:
            return
        old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
        for blkitem in selected_blks:
            old_html_lst.append(blkitem.toHtml())
            old_rect_lst.append(blkitem.absBoundingRect(qrect=True))
            trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
            # Establish line breaks with our layout (stub penalties) before scaling, so scaling doesn't reflow into stubs.
            self.layout_textblk(blkitem)
            self.layout_textblk(blkitem)
            self._scale_font_to_fit_box(blkitem)
        self.canvas.push_undo_command(AutoLayoutCommand(selected_blks, old_rect_lst, old_html_lst, trans_widget_lst))


    def _polish_textblock_model(self, blk: TextBlock) -> bool:
        ff = blk.fontformat
        xyxy = list(getattr(blk, 'xyxy', []) or [])
        if len(xyxy) >= 4:
            box = (max(1.0, float(xyxy[2]) - float(xyxy[0])), max(1.0, float(xyxy[3]) - float(xyxy[1])))
        else:
            box = (1.0, 1.0)
        text = getattr(blk, 'translation', '') or blk.get_text()
        result = plan_typography_cleanup(
            text, float(getattr(ff, 'font_size', 24.0) or 24.0), box,
            getattr(ff, 'writing_mode', getattr(pcfg, 'render_default_writing_mode', 'auto')),
            getattr(ff, 'fit_mode', getattr(pcfg, 'render_default_fit_mode', 'shrink')),
            getattr(ff, 'line_break_strategy', getattr(pcfg, 'render_default_line_break_strategy', 'auto')),
            line_spacing=float(getattr(ff, 'line_spacing', 1.15) or 1.15),
            letter_spacing=float(getattr(ff, 'letter_spacing', 1.0) or 1.0),
            text_padding=float(getattr(ff, 'text_padding', 0.0) or 0.0),
            font_family=getattr(ff, 'font_family', ''),
            fallback_font_chain=getattr(ff, 'fallback_font_chain', ''),
            config_obj=pcfg,
        )
        changed = False
        if result.text and result.text != getattr(blk, 'translation', ''):
            blk.translation = result.text
            changed = True
        for attr, value in [
            ('writing_mode', result.writing_mode),
            ('fit_mode', result.fit_mode),
            ('line_break_strategy', result.line_break_strategy),
            ('line_spacing', result.line_spacing),
            ('letter_spacing', result.letter_spacing),
            ('text_padding', result.text_padding),
            ('fallback_font_chain', result.fallback_font_chain),
        ]:
            if hasattr(ff, attr) and value is not None and getattr(ff, attr) != value:
                setattr(ff, attr, value)
                changed = True
        ff.vertical = result.resolved_writing_mode == 'vertical_rl'
        return changed

    def _apply_typography_cleanup_to_item(self, blkitem: TextBlkItem) -> bool:
        rect = blkitem.absBoundingRect(qrect=True)
        ff = blkitem.fontformat
        result = plan_typography_cleanup(
            blkitem.toPlainText(), float(getattr(ff, 'font_size', 24.0) or 24.0), (rect.width(), rect.height()),
            getattr(ff, 'writing_mode', 'auto'), getattr(ff, 'fit_mode', 'shrink'), getattr(ff, 'line_break_strategy', 'auto'),
            line_spacing=float(getattr(ff, 'line_spacing', 1.15) or 1.15),
            letter_spacing=float(getattr(ff, 'letter_spacing', 1.0) or 1.0),
            text_padding=float(getattr(ff, 'text_padding', 0.0) or 0.0),
            font_family=getattr(ff, 'font_family', ''),
            fallback_font_chain=getattr(ff, 'fallback_font_chain', ''),
            config_obj=pcfg,
        )
        changed = False
        if result.text and result.text != blkitem.toPlainText():
            blkitem.setPlainText(result.text)
            if blkitem.blk is not None:
                blkitem.blk.translation = result.text
            changed = True
        for attr, value in [
            ('writing_mode', result.writing_mode),
            ('fit_mode', result.fit_mode),
            ('line_break_strategy', result.line_break_strategy),
            ('line_spacing', result.line_spacing),
            ('letter_spacing', result.letter_spacing),
            ('text_padding', result.text_padding),
            ('fallback_font_chain', result.fallback_font_chain),
        ]:
            if hasattr(ff, attr) and value is not None and getattr(ff, attr) != value:
                setattr(ff, attr, value)
                changed = True
        vertical = result.resolved_writing_mode == 'vertical_rl'
        if getattr(ff, 'vertical', False) != vertical:
            ff.vertical = vertical
            changed = True
        if changed:
            blkitem.set_fontformat(ff)
            if blkitem.blk is not None:
                blkitem.blk.fontformat = ff.deepcopy()
        return changed

    def polish_typography_textboxes(self, indices: List[int] = None, push_undo: bool = True) -> int:
        if indices is None:
            target_blks = self.canvas.selected_text_items()
        else:
            wanted = set(int(i) for i in indices)
            target_blks = [blk for blk in self.textblk_item_list if blk.idx in wanted]
        if not target_blks:
            return 0
        old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
        changed = 0
        for blkitem in target_blks:
            old_html_lst.append(blkitem.toHtml())
            old_rect_lst.append(blkitem.absBoundingRect(qrect=True))
            trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
            if self._apply_typography_cleanup_to_item(blkitem):
                changed += 1
        if changed:
            if push_undo:
                self.canvas.push_undo_command(AutoLayoutCommand(target_blks, old_rect_lst, old_html_lst, trans_widget_lst))
            self.canvas.setProjSaveState(True)
            if hasattr(self.mainwindow, 'pipelineInsightsPanel'):
                self.mainwindow.pipelineInsightsPanel.add_event('TYPO_QA', self.tr('Polished typography for {0} text box(es)').format(changed))
        return changed

    def onPolishTypography(self):
        self.polish_typography_textboxes(indices=None, push_undo=True)

    def _apply_smart_fit_to_item(self, blkitem: TextBlkItem) -> bool:
        rect = blkitem.absBoundingRect(qrect=True)
        ff = blkitem.fontformat
        text = blkitem.toPlainText()
        mask = getattr(getattr(blkitem, 'blk', None), 'text_mask', None)
        effective = mask_effective_box(mask, (rect.width(), rect.height()), getattr(ff, 'text_padding', 0.0))
        style_min, style_max = resolve_fit_font_size_bounds(
            getattr(ff, 'fit_font_size_min', 0.0),
            getattr(ff, 'fit_font_size_max', 0.0),
            getattr(getattr(pcfg, 'module', None), 'layout_font_size_min', 6.0),
            getattr(getattr(pcfg, 'module', None), 'layout_font_size_max', 96.0),
        )
        page_scale = _auto_lettering_page_scale(getattr(self.imgtrans_proj, 'img_array', None), (rect.width(), rect.height()))
        style_min = max(4.0, style_min * page_scale)
        style_max = max(style_min + 1.0, style_max * page_scale)
        box_w = float(effective.get('width', rect.width())) if effective.get('uses_mask') else float(rect.width())
        box_h = float(effective.get('height', rect.height())) if effective.get('uses_mask') else float(rect.height())
        shape = str(getattr(ff, 'text_box_shape', '') or '')
        initial_line_count = max(1, text.count('\n') + 1)
        width_profile = _auto_width_profile_for_shape(box_w, box_h, initial_line_count + 3, shape)
        auto_req = AutoFitRequest(
            text=text,
            box=(0.0, 0.0, box_w, box_h),
            min_font_size=float(style_min),
            max_font_size=float(style_max),
            writing_mode='vertical' if bool(getattr(ff, 'vertical', False)) else 'horizontal',
            line_spacing=float(getattr(ff, 'line_spacing', 1.15) or 1.15),
            letter_spacing=float(getattr(ff, 'letter_spacing', 1.0) or 1.0),
            padding=float(getattr(ff, 'text_padding', 0.0) or 0.0),
            stroke_width=float(getattr(ff, 'stroke_width', 0.0) or 0.0) + float(getattr(ff, 'secondary_stroke_width', 0.0) or 0.0),
            shadow_radius=float(getattr(ff, 'shadow_radius', 0.0) or 0.0),
            shadow_offset=tuple((getattr(ff, 'shadow_offset', [0.0, 0.0]) or [0.0, 0.0])[:2]),
            width_profile=width_profile,
        )
        auto_res = auto_fit_text(auto_req)
        result = smart_fit_text_to_box(
            "\n".join(auto_res.lines) if auto_res.lines else text,
            float(auto_res.font_size or getattr(ff, 'font_size', 24.0) or 24.0),
            (rect.width(), rect.height()),
            getattr(ff, 'writing_mode', 'auto'),
            getattr(ff, 'fit_mode', 'shrink'),
            min_font_size=style_min,
            max_font_size=style_max,
            line_spacing=float(getattr(ff, 'line_spacing', 1.15) or 1.15),
            letter_spacing=float(getattr(ff, 'letter_spacing', 1.0) or 1.0),
            padding=float(getattr(ff, 'text_padding', 0.0) or 0.0),
            stroke_width=float(getattr(ff, 'stroke_width', 0.0) or 0.0),
            secondary_stroke_width=float(getattr(ff, 'secondary_stroke_width', 0.0) or 0.0),
            line_break_strategy=getattr(ff, 'line_break_strategy', 'auto'),
            shadow_radius=float(getattr(ff, 'shadow_radius', 0.0) or 0.0),
            shadow_offset=getattr(ff, 'shadow_offset', [0.0, 0.0]) or [0.0, 0.0],
            effective_box_size=(box_w, box_h),
        )
        changed = False
        if result.text and result.text != text:
            blkitem.setPlainText(result.text)
            if blkitem.blk is not None:
                blkitem.blk.translation = result.text
            changed = True
        for attr, value in [
            ('font_size', result.font_size),
            ('line_spacing', result.line_spacing),
            ('letter_spacing', result.letter_spacing),
            ('writing_mode', result.writing_mode),
            ('fit_mode', result.fit_mode),
            ('line_break_strategy', (result.diagnostics or {}).get('line_break_strategy', getattr(ff, 'line_break_strategy', 'auto'))),
        ]:
            if hasattr(ff, attr) and getattr(ff, attr) != value:
                setattr(ff, attr, value)
                changed = True
        if result.resolved_writing_mode == 'vertical_rl' and not getattr(ff, 'vertical', False):
            ff.vertical = True
            changed = True
        elif result.resolved_writing_mode != 'vertical_rl' and getattr(ff, 'vertical', False):
            ff.vertical = False
            changed = True
        if changed:
            blkitem.set_fontformat(ff)
            fit_diag = (result.diagnostics or {}).get('fit', {})
            if blkitem.blk is not None and hasattr(blkitem.blk, 'fontformat'):
                blkitem.blk.fontformat.last_layout_score = float(auto_res.score)
                blkitem.blk.fontformat.layout_warnings = list(auto_res.warnings)
                blkitem.blk.fontformat.layout_version = "v2"
            changed = self._postprocess_auto_lettering_item(blkitem, fit_diag=fit_diag) or changed
            if blkitem.blk is not None:
                blkitem.blk.fontformat = ff.deepcopy()
        return changed

    def _postprocess_auto_lettering_item(self, blkitem: TextBlkItem, fit_diag: dict = None) -> bool:
        """Final automatic safety pass after smart/atomic formatting.

        Grows a too-small box from renderer diagnostics, runs the rendered-size
        safety loop, and recenters in the detected bubble when available.  This
        keeps one-click formatting from leaving text clipped or visibly off-center.
        """
        changed = False
        rect = blkitem.absBoundingRect(qrect=True)
        diag = fit_diag or {}
        rec = diag.get('recommended_box_size') or []
        overflow = bool(diag.get('overflow'))
        if overflow and len(rec) >= 2:
            try:
                new_w = max(rect.width(), float(rec[0]))
                new_h = max(rect.height(), float(rec[1]))
                # Avoid runaway boxes from bad diagnostics; fitting should prefer
                # font/line changes before modest centered growth.
                new_w = min(new_w, rect.width() * 1.65)
                new_h = min(new_h, rect.height() * 1.65)
                img = getattr(self.imgtrans_proj, 'img_array', None)
                if img is not None:
                    im_h, im_w = img.shape[:2]
                    new_w = min(new_w, float(im_w))
                    new_h = min(new_h, float(im_h))
                if new_w > rect.width() + 0.5 or new_h > rect.height() + 0.5:
                    x1, y1, x2, y2 = centered_resize_xyxy([rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height()], [new_w, new_h])
                    blkitem.setRect([x1, y1, x2 - x1, y2 - y1], repaint=True)
                    changed = True
            except Exception:
                pass
        try:
            before_size = float(getattr(blkitem.fontformat, 'font_size', 0.0) or 0.0)
            self._auto_refine_rendered_fit(blkitem)
            after_size = float(blkitem.minFontSize())
            if after_size > 0 and abs(after_size - before_size) > 0.05:
                blkitem.fontformat.font_size = after_size
                changed = True
        except Exception:
            pass
        img = getattr(self.imgtrans_proj, 'img_array', None)
        if img is not None and getattr(getattr(pcfg, 'module', None), 'layout_center_in_bubble_after_autolayout', True):
            try:
                im_h, im_w = img.shape[:2]
                changed = self._center_blkitem_in_bubble(blkitem, img, im_w, im_h) or changed
            except Exception:
                pass
        if changed and blkitem.blk is not None:
            blkitem.blk._bounding_rect = blkitem.absBoundingRect()
        return changed

    def _apply_atomic_bubble_fit_to_item(self, blkitem: TextBlkItem, profile: str = None) -> bool:
        rect = blkitem.absBoundingRect(qrect=True)
        ff = blkitem.fontformat
        text = blkitem.toPlainText()
        mask = getattr(getattr(blkitem, 'blk', None), 'text_mask', None)
        effective = mask_effective_box(mask, (rect.width(), rect.height()), getattr(ff, 'text_padding', 0.0))
        style_min, style_max = resolve_fit_font_size_bounds(
            getattr(ff, 'fit_font_size_min', 0.0),
            getattr(ff, 'fit_font_size_max', 0.0),
            getattr(getattr(pcfg, 'module', None), 'layout_font_size_min', 6.0),
            getattr(getattr(pcfg, 'module', None), 'layout_font_size_max', 96.0),
        )
        page_scale = _auto_lettering_page_scale(getattr(self.imgtrans_proj, 'img_array', None), (rect.width(), rect.height()))
        style_min = max(4.0, style_min * page_scale)
        style_max = max(style_min + 1.0, style_max * page_scale)
        result = plan_atomic_bubble_fit(
            text,
            float(getattr(ff, 'font_size', 24.0) or 24.0),
            (rect.width(), rect.height()),
            getattr(ff, 'writing_mode', 'auto'),
            getattr(ff, 'fit_mode', 'shrink'),
            getattr(ff, 'line_break_strategy', 'auto'),
            line_spacing=float(getattr(ff, 'line_spacing', 1.15) or 1.15),
            letter_spacing=float(getattr(ff, 'letter_spacing', 1.0) or 1.0),
            padding=float(getattr(ff, 'text_padding', 0.0) or 0.0),
            stroke_width=float(getattr(ff, 'stroke_width', 0.0) or 0.0),
            secondary_stroke_width=float(getattr(ff, 'secondary_stroke_width', 0.0) or 0.0),
            shadow_radius=float(getattr(ff, 'shadow_radius', 0.0) or 0.0),
            shadow_offset=getattr(ff, 'shadow_offset', [0.0, 0.0]) or [0.0, 0.0],
            min_font_size=style_min,
            max_font_size=style_max,
            bubble_box_size=(float(effective.get('width', rect.width())), float(effective.get('height', rect.height()))) if effective.get('uses_mask') else (rect.width(), rect.height()),
            target_fill=float(getattr(pcfg, 'render_atomic_fit_target_fill', 0.78) or 0.78),
            max_expand_ratio=float(getattr(pcfg, 'render_atomic_fit_max_expand', 1.22) or 1.22),
            profile=normalize_atomic_fit_mode(profile or getattr(pcfg, 'render_atomic_fit_profile', 'balanced')),
        )
        changed = False
        if result.text and result.text != text:
            blkitem.setPlainText(result.text)
            if blkitem.blk is not None:
                blkitem.blk.translation = result.text
            changed = True
        for attr, value in [
            ('font_size', result.font_size),
            ('line_spacing', result.line_spacing),
            ('letter_spacing', result.letter_spacing),
            ('text_padding', result.text_padding),
            ('writing_mode', result.writing_mode),
            ('fit_mode', result.fit_mode),
            ('line_break_strategy', result.line_break_strategy),
            ('alignment', result.alignment),
        ]:
            if hasattr(ff, attr) and value is not None and getattr(ff, attr) != value:
                setattr(ff, attr, value)
                changed = True
        vertical = result.resolved_writing_mode == 'vertical_rl'
        if getattr(ff, 'vertical', False) != vertical:
            ff.vertical = vertical
            changed = True
        if changed:
            blkitem.set_fontformat(ff)
            if result.alignment in (0, 1, 2):
                blkitem.setAlignment(int(result.alignment), restore_cursor=False)
            fit_diag = (result.diagnostics or {}).get('fit', {})
            changed = self._postprocess_auto_lettering_item(blkitem, fit_diag=fit_diag) or changed
            if blkitem.blk is not None:
                blkitem.blk.fontformat = ff.deepcopy()
                blkitem.blk.vertical = vertical
            try:
                blkitem.repaint_background()
            except Exception:
                pass
        return changed

    def atomic_bubble_fit_textboxes(self, indices: List[int] = None, push_undo: bool = True, profile: str = None) -> int:
        if indices is None:
            target_blks = self.canvas.selected_text_items()
        else:
            wanted = set(int(i) for i in indices)
            target_blks = [blk for blk in self.textblk_item_list if blk.idx in wanted]
        if not target_blks:
            return 0
        old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
        changed = 0
        for blkitem in target_blks:
            old_html_lst.append(blkitem.toHtml())
            old_rect_lst.append(blkitem.absBoundingRect(qrect=True))
            trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
            if self._apply_atomic_bubble_fit_to_item(blkitem, profile=profile):
                changed += 1
        if changed:
            if push_undo:
                self.canvas.push_undo_command(AutoLayoutCommand(target_blks, old_rect_lst, old_html_lst, trans_widget_lst))
            self.canvas.setProjSaveState(True)
            if hasattr(self.mainwindow, 'pipelineInsightsPanel'):
                self.mainwindow.pipelineInsightsPanel.add_event('TYPO_QA', self.tr('Atomically fit {0} text box(es) inside bubbles').format(changed))
        return changed

    def onAtomicBubbleFitText(self):
        self.atomic_bubble_fit_textboxes(indices=None, push_undo=True)

    def onAtomicBubbleFitProfileText(self, profile: str):
        self.atomic_bubble_fit_textboxes(indices=None, push_undo=True, profile=profile)

    def smart_fit_textboxes(self, indices: List[int] = None, push_undo: bool = True) -> int:
        if indices is None:
            target_blks = self.canvas.selected_text_items()
        else:
            wanted = set(int(i) for i in indices)
            target_blks = [blk for blk in self.textblk_item_list if blk.idx in wanted]
        if not target_blks:
            return 0
        old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
        changed = 0
        for blkitem in target_blks:
            old_html_lst.append(blkitem.toHtml())
            old_rect_lst.append(blkitem.absBoundingRect(qrect=True))
            trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
            if self._apply_smart_fit_to_item(blkitem):
                changed += 1
        if changed:
            if push_undo:
                self.canvas.push_undo_command(AutoLayoutCommand(target_blks, old_rect_lst, old_html_lst, trans_widget_lst))
            self.canvas.setProjSaveState(True)
            if hasattr(self.mainwindow, 'pipelineInsightsPanel'):
                self.mainwindow.pipelineInsightsPanel.add_event('TYPO_QA', self.tr('Smart-fitted {0} text box(es)').format(changed))
        return changed

    def onSmartAutoFitTextToBox(self):
        self.smart_fit_textboxes(indices=None, push_undo=True)

    def auto_lettering_assist_textboxes(self, indices: List[int] = None, push_undo: bool = True) -> int:
        """
        One-click lettering assist for production teams:
        - auto detect likely font from source region (when detector/model available),
        - switch writing mode heuristically from content/geometry,
        - run smart fit + postprocess to keep text inside bubble.
        """
        if indices is None:
            target_blks = self.canvas.selected_text_items()
            if not target_blks:
                target_blks = [b for b in self.textblk_item_list if b is not None]
        else:
            wanted = set(int(i) for i in indices)
            target_blks = [blk for blk in self.textblk_item_list if blk.idx in wanted]
        if not target_blks:
            return 0

        old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
        changed = 0
        img = getattr(getattr(self.imgtrans_proj, "img_array", None), "copy", lambda: None)()
        for blkitem in target_blks:
            old_html_lst.append(blkitem.toHtml())
            old_rect_lst.append(blkitem.absBoundingRect(qrect=True))
            trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
            ff = blkitem.fontformat
            item_changed = False

            text_for_mode = blkitem.toPlainText()
            rect = blkitem.absBoundingRect(qrect=True)
            auto_mode = resolve_writing_mode("auto", text_for_mode, (rect.width(), rect.height()))
            if getattr(ff, "writing_mode", "auto") != auto_mode:
                ff.writing_mode = auto_mode
                ff.vertical = auto_mode == "vertical_rl"
                item_changed = True

            if img is not None and getattr(blkitem, "blk", None) is not None:
                try:
                    font_name, conf = detect_font_from_block(img, blkitem.blk)
                    if conf >= 0.66 and font_name and font_name != "UNKNOWN":
                        if getattr(ff, "font_family", "") != font_name:
                            ff.font_family = str(font_name)
                            item_changed = True
                except Exception as e:
                    LOGGER.debug("auto lettering assist: font detection failed for idx=%s: %s", blkitem.idx, e)

            if item_changed:
                blkitem.set_fontformat(ff)
                if blkitem.blk is not None:
                    blkitem.blk.fontformat = ff.deepcopy()

            if self._apply_smart_fit_to_item(blkitem):
                changed += 1
            elif item_changed:
                changed += 1

        if changed:
            if push_undo:
                self.canvas.push_undo_command(AutoLayoutCommand(target_blks, old_rect_lst, old_html_lst, trans_widget_lst))
            self.canvas.setProjSaveState(True)
            if hasattr(self.mainwindow, 'pipelineInsightsPanel'):
                self.mainwindow.pipelineInsightsPanel.add_event('TYPO_QA', self.tr('Auto lettering assist updated {0} text box(es)').format(changed))
        return changed

    def auto_format_textboxes(self, indices: List[int] = None, push_undo: bool = True, profile: str = "balanced") -> int:
        """High-level auto formatter: smart-fit first, then atomic bubble-fit rescue for difficult boxes."""
        changed = self.smart_fit_textboxes(indices=indices, push_undo=push_undo)
        # Follow up with atomic fit to smooth uneven lines / reduce overflow where smart-fit still struggles.
        rescued = self.atomic_bubble_fit_textboxes(indices=indices, push_undo=False, profile=profile)
        return int(changed) + int(rescued)

    def onAutoLetteringAssist(self):
        self.auto_lettering_assist_textboxes(indices=None, push_undo=True)

    def _layout_review_snapshot_for_item(self, blk: TextBlkItem) -> BlockSnapshot:
        """Build a renderer-aware snapshot for the layout review agent."""
        abr = blk.absBoundingRect(qrect=True)
        doc = blk.document()
        est = None
        if doc is not None:
            ds = doc.size()
            est = (max(0.0, float(ds.width())), max(0.0, float(ds.height())))
        text = blk.toPlainText()
        ff = blk.fontformat
        box_size = (max(1.0, float(abr.width())), max(1.0, float(abr.height())))
        try:
            _fit_size, _fit_text, fit_diag = fit_font_size_to_box(
                text,
                max(1.0, float(getattr(ff, 'font_size', blk.font().pointSizeF() or blk.font().pointSize() or 1.0) or 1.0)),
                box_size,
                getattr(ff, 'fit_mode', 'shrink'),
                getattr(ff, 'writing_mode', 'auto'),
                padding=float(getattr(ff, 'text_padding', 0.0) or 0.0),
                stroke_width=float(getattr(ff, 'stroke_width', 0.0) or 0.0),
                secondary_stroke_width=float(getattr(ff, 'secondary_stroke_width', 0.0) or 0.0),
                line_spacing=float(getattr(ff, 'line_spacing', 1.2) or 1.2),
                letter_spacing=float(getattr(ff, 'letter_spacing', 1.0) or 1.0),
                line_break_strategy=getattr(ff, 'line_break_strategy', 'auto'),
                shadow_radius=float(getattr(ff, 'shadow_radius', 0.0) or 0.0),
                shadow_offset=getattr(ff, 'shadow_offset', [0.0, 0.0]) or [0.0, 0.0],
            )
            fit_dict = fit_diag.to_dict()
        except Exception:
            fit_dict = {}
        effective_box = mask_effective_box(
            getattr(getattr(blk, 'blk', None), 'text_mask', None),
            box_size,
            getattr(ff, 'text_padding', 0.0),
        )

        typography_cleanup = plan_typography_cleanup(
            text,
            max(1.0, float(getattr(ff, 'font_size', blk.font().pointSizeF() or blk.font().pointSize() or 1.0) or 1.0)),
            box_size,
            getattr(ff, 'writing_mode', 'auto'),
            getattr(ff, 'fit_mode', 'shrink'),
            getattr(ff, 'line_break_strategy', 'auto'),
            line_spacing=float(getattr(ff, 'line_spacing', 1.2) or 1.2),
            letter_spacing=float(getattr(ff, 'letter_spacing', 1.0) or 1.0),
            text_padding=float(getattr(ff, 'text_padding', 0.0) or 0.0),
            font_family=getattr(ff, 'font_family', ''),
            fallback_font_chain=getattr(ff, 'fallback_font_chain', ''),
            config_obj=pcfg,
        )

        smart = smart_fit_text_to_box(
            text,
            max(1.0, float(getattr(ff, 'font_size', blk.font().pointSizeF() or blk.font().pointSize() or 1.0) or 1.0)),
            box_size,
            getattr(ff, 'writing_mode', 'auto'),
            getattr(ff, 'fit_mode', 'shrink'),
            line_spacing=float(getattr(ff, 'line_spacing', 1.2) or 1.2),
            letter_spacing=float(getattr(ff, 'letter_spacing', 1.0) or 1.0),
            padding=float(getattr(ff, 'text_padding', 0.0) or 0.0),
            stroke_width=float(getattr(ff, 'stroke_width', 0.0) or 0.0),
            secondary_stroke_width=float(getattr(ff, 'secondary_stroke_width', 0.0) or 0.0),
            line_break_strategy=getattr(ff, 'line_break_strategy', 'auto'),
            shadow_radius=float(getattr(ff, 'shadow_radius', 0.0) or 0.0),
            shadow_offset=getattr(ff, 'shadow_offset', [0.0, 0.0]) or [0.0, 0.0],
            effective_box_size=(
                float(effective_box.get('width', box_size[0])),
                float(effective_box.get('height', box_size[1])),
            ) if effective_box.get('uses_mask') else None,
        )

        smart_dict = smart.to_dict() if hasattr(smart, 'to_dict') else {}

        measured_bounds = (
            fit_dict.get('measured_bounds')
            or smart_dict.get('measured_bounds')
            or est
            or (0.0, 0.0)
        )

        mask_diag = mask_aware_textbox_diagnostics(
            box_size,
            (
                float(measured_bounds[0]),
                float(measured_bounds[1]),
            ),
            getattr(getattr(blk, 'blk', None), 'text_mask', None),
            effect_margin=float(
                fit_dict.get(
                    'effect_margin',
                    smart_dict.get('effect_margin', 0.0),
                ) or 0.0
            ),
            padding=float(getattr(ff, 'text_padding', 0.0) or 0.0),
        )
        style = {
            'font_family': getattr(ff, 'font_family', ''),
            'fallback_chain': ', '.join(merge_font_fallback_chain(getattr(ff, 'font_family', ''), text, pcfg, getattr(ff, 'fallback_font_chain', ''))),
            'fit_mode': getattr(ff, 'fit_mode', 'shrink'),
            'stroke_width': getattr(ff, 'stroke_width', 0.0),
            'secondary_stroke_width': getattr(ff, 'secondary_stroke_width', 0.0),
            'secondary_srgb': getattr(ff, 'secondary_srgb', [255, 255, 255]),
            'text_padding': getattr(ff, 'text_padding', 0.0),
            'line_spacing': getattr(ff, 'line_spacing', 1.2),
            'letter_spacing': getattr(ff, 'letter_spacing', 1.0),
            'line_break_strategy': getattr(ff, 'line_break_strategy', 'auto'),
            'alignment': getattr(ff, 'alignment', 0),
            'recommended_box_size': fit_dict.get('recommended_box_size', []),
            'box_scale_hint': fit_dict.get('box_scale_hint', 1.0),
            'ink_clip_risk': fit_dict.get('ink_clip_risk', False),
            'preset_suggestion': fit_dict.get('preset_suggestion', ''),
            'safe_inner_bounds': fit_dict.get('safe_inner_bounds', []),
            'line_break_quality': fit_dict.get('line_break_quality', {}),
            'mask_coverage': mask_diag.get('coverage', 1.0),
            'mask_safe_insets': mask_diag.get('safe_insets', []),
            'mask_warning': mask_diag.get('warning', ''),
            'mask_recommended_padding': mask_diag.get('recommended_padding', 0.0),
            'mask_effective_box': effective_box,
            'smart_fit': smart_dict,
            'typography_cleanup': typography_cleanup.to_dict(),
            'mask_diagnostics': mask_diag,
        }
        return BlockSnapshot(
            block_index=blk.idx,
            xyxy=(abr.x(), abr.y(), abr.x() + abr.width(), abr.y() + abr.height()),
            text=text,
            font_size=max(1.0, float(blk.font().pointSizeF() or blk.font().pointSize() or 1.0)),
            est_text_size=est,
            bubble_center=self._estimate_block_bubble_center(blk),
            writing_mode=getattr(ff, 'writing_mode', 'auto'),
            resolved_writing_mode=resolve_writing_mode(getattr(ff, 'writing_mode', 'auto'), text, box_size),
            overflow_status=bool(est and (est[0] > abr.width() * 0.98 or est[1] > abr.height() * 0.98)),
            measured_bounds=est,
            text_style=style,
            font_fallback_warning=', '.join(missing_glyphs_after_fallback(getattr(ff, 'font_family', ''), text, pcfg, getattr(ff, 'fallback_font_chain', ''))),
            quality_score=float(fit_dict.get('quality_score', 1.0) or 1.0),
            mask_coverage=float(mask_diag.get('coverage', 1.0) or 1.0),
            mask_safe_insets=tuple(mask_diag.get('safe_insets', [0, 0, 0, 0]) or [0, 0, 0, 0]),
            mask_warning=str(mask_diag.get('warning', '') or ''),
        )

    def onFitToMaskSafeBox(self):
        """Grow selected text boxes when text eraser masks leave too little visible lettering area."""
        selected = self.canvas.selected_text_items()
        if not selected:
            return
        actions = []
        for blkitem in selected:
            try:
                snap = self._layout_review_snapshot_for_item(blkitem)
                mask_diag = (snap.text_style or {}).get('mask_diagnostics') or {}
                if not mask_diag.get('mask_overflow'):
                    continue
                rec = mask_diag.get('recommended_box_size') or []
                if len(rec) < 2:
                    continue
                actions.append(ReviewAction(
                    action='resize_to_mask_safe_box',
                    block_index=blkitem.idx,
                    args={'width': float(rec[0]), 'height': float(rec[1])},
                    reason='Fit selected textbox to mask-visible safe area.',
                ))
            except Exception as e:
                LOGGER.debug('fit-to-mask-safe-box skipped block idx=%s: %s', getattr(blkitem, 'idx', -1), e)
        if actions:
            self._apply_layout_review_actions(actions, selected)
            self.canvas.setProjSaveState(False)

    def review_and_fix_selected_textboxes(
        self,
        max_iters: int = 2,
        target_block_indices: Optional[List[int]] = None,
    ) -> int:
        """Iteratively review selected blocks and apply conservative fixes.

        Returns the number of iterations used. This reuses existing operations
        (auto-fit, resize-to-fit-content, center-in-bubble) to preserve behavior.
        """
        selected = self._get_review_target_blocks(target_block_indices)
        if not selected:
            return 0
        planner = LayoutReviewPlanner()
        loops = 0
        for i in range(max(1, int(max_iters))):
            snapshots: List[BlockSnapshot] = []
            for blk in selected:
                snapshots.append(self._layout_review_snapshot_for_item(blk))
            result = planner.review_page(getattr(self.imgtrans_proj, "current_img", ""), snapshots)
            pending_actions = [a for block_result in result.blocks for a in block_result.actions]
            if not pending_actions:
                return loops
            if target_block_indices:
                pending_actions = filter_actions_by_block_indices(pending_actions, target_block_indices)
                if not pending_actions:
                    return loops
            self._apply_layout_review_actions(pending_actions, selected)
            loops = i + 1
        return loops

    def review_selected_textboxes_dry_run(
        self, target_block_indices: Optional[List[int]] = None
    ):
        """Run planner only and return PageReviewResult without applying fixes."""
        selected = self._get_review_target_blocks(target_block_indices)
        planner = LayoutReviewPlanner()
        snapshots: List[BlockSnapshot] = []
        for blk in selected:
            snapshots.append(self._layout_review_snapshot_for_item(blk))
        return planner.review_page(getattr(self.imgtrans_proj, "current_img", ""), snapshots)

    def review_selected_textboxes_with_provider(
        self,
        config: Optional[ReviewModelConfig] = None,
        target_block_indices: Optional[List[int]] = None,
        provider: Optional[ReviewProvider] = None,
    ):
        """Review selected/targeted boxes with heuristic/local/cloud provider and return proposals."""
        cfg = config or ReviewModelConfig()
        cfg = copy.deepcopy(cfg)
        self._attach_review_page_context(cfg)
        selected = self._get_review_target_blocks(target_block_indices)
        snapshots: List[BlockSnapshot] = []
        for blk in selected:
            snapshots.append(self._layout_review_snapshot_for_item(blk))
        provider_impl = provider or self._resolve_review_provider(cfg)
        return provider_impl.review(getattr(self.imgtrans_proj, "current_img", ""), snapshots, cfg)

    def _attach_review_page_context(self, config: ReviewModelConfig) -> None:
        """Attach page-level screenshot/context for provider-backed review.

        External provider handlers can consume this through:
        `config.extra_params["page_context"]`.
        """
        if not getattr(config, "include_page_screenshot", True):
            return
        img = getattr(self.imgtrans_proj, "img_array", None)
        if img is None or getattr(img, "size", 0) == 0:
            return
        try:
            import cv2
            h, w = img.shape[:2]
            max_side = max(1, int(getattr(config, "screenshot_max_side", 1280) or 1280))
            scale = min(1.0, float(max_side) / float(max(h, w)))
            if scale < 1.0:
                resized = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
            else:
                resized = img
            ok, enc = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return
            payload = {
                "image_data_url": "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii"),
                "image_width": int(resized.shape[1]),
                "image_height": int(resized.shape[0]),
                "source_width": int(w),
                "source_height": int(h),
            }
            if not isinstance(config.extra_params, dict):
                config.extra_params = {}
            config.extra_params["page_context"] = payload
        except Exception as e:
            LOGGER.debug("layout review: failed to attach page screenshot context: %s", e)

    def apply_review_result(
        self,
        review_result,
        target_block_indices: Optional[List[int]] = None,
    ) -> int:
        """Apply actions from a precomputed review result and return number of actions applied.

        Provider/page review can target blocks that are not currently selected.  The
        low-level auto-fit/center helpers intentionally operate on selection, so
        this method temporarily selects the requested targets to make API/menu
        invocations behave exactly like user-driven edits and remain undoable.
        """
        pending_actions = [a for block_result in review_result.blocks for a in block_result.actions]
        if target_block_indices:
            pending_actions = filter_actions_by_block_indices(pending_actions, target_block_indices)
        if not pending_actions:
            return 0

        selected_before = list(self.canvas.selected_text_items())
        was_editing = self.canvas.editing_textblkitem
        try:
            if target_block_indices:
                idx_set = set(target_block_indices)
                for blk in self.textblk_item_list:
                    if blk is not None:
                        blk.setSelected(blk.idx in idx_set)
            selected = self._get_review_target_blocks(target_block_indices)
            if not selected:
                return 0
            self._apply_layout_review_actions(pending_actions, selected)
            return len(pending_actions)
        finally:
            if target_block_indices:
                for blk in self.textblk_item_list:
                    if blk is not None:
                        blk.setSelected(False)
                for blk in selected_before:
                    try:
                        blk.setSelected(True)
                    except Exception:
                        continue
                if was_editing is not None and was_editing in self.textblk_item_list:
                    self.canvas.editing_textblkitem = was_editing

    def review_and_fix_selected_textboxes_with_provider(
        self,
        config: Optional[ReviewModelConfig] = None,
        target_block_indices: Optional[List[int]] = None,
        provider: Optional[ReviewProvider] = None,
        max_iters: int = 2,
    ) -> int:
        """Provider-driven iterative review/apply loop."""
        loops = 0
        for i in range(max(1, int(max_iters))):
            rst = self.review_selected_textboxes_with_provider(config=config, target_block_indices=target_block_indices, provider=provider)
            applied = self.apply_review_result(rst, target_block_indices=target_block_indices)
            if applied <= 0:
                return loops
            loops = i + 1
        return loops

    def _resolve_review_provider(self, config: ReviewModelConfig) -> ReviewProvider:
        """Return configured review provider.

        For `local_api` / `cloud_api`, attach a callable at runtime via:
        - `self.layout_review_local_api_handler`
        - `self.layout_review_cloud_api_handler`
        """
        if config.provider == "heuristic":
            return HeuristicReviewProvider()
        attr = "layout_review_local_api_handler" if config.provider == "local_api" else "layout_review_cloud_api_handler"
        handler = getattr(self, attr, None)
        if callable(handler):
            return ExternalReviewProvider(handler)
        LOGGER.warning("layout review provider '%s' requested but handler is not configured; falling back to heuristic", config.provider)
        return HeuristicReviewProvider()

    def review_all_textboxes_on_page_dry_run(self):
        """Run planner against all non-vertical blocks without mutating layout."""
        target_indices = [
            blk.idx
            for blk in self.textblk_item_list
            if blk is not None and not getattr(blk.fontformat, "vertical", False)
        ]
        return self.review_selected_textboxes_dry_run(target_indices)

    def review_and_fix_all_textboxes_on_page(self, max_iters: int = 2) -> int:
        """Review/fix all non-vertical text boxes on the current page."""
        target_indices = [
            blk.idx
            for blk in self.textblk_item_list
            if blk is not None and not getattr(blk.fontformat, "vertical", False)
        ]
        if not target_indices:
            return 0
        return self._run_layout_review_with_temporary_selection(target_indices, max_iters=max_iters)

    def _run_layout_review_with_temporary_selection(self, target_indices: List[int], max_iters: int = 2) -> int:
        """Run review while temporarily selecting requested targets, then restore selection."""
        selected_before = list(self.canvas.selected_text_items())
        was_editing = self.canvas.editing_textblkitem
        try:
            for blk in self.textblk_item_list:
                if blk is not None:
                    blk.setSelected(False)
            idx_set = set(target_indices)
            for blk in self.textblk_item_list:
                if blk is None:
                    continue
                blk.setSelected(blk.idx in idx_set)
            return self.review_and_fix_selected_textboxes(
                max_iters=max_iters,
                target_block_indices=target_indices,
            )
        finally:
            for blk in self.textblk_item_list:
                if blk is not None:
                    blk.setSelected(False)
            for blk in selected_before:
                try:
                    blk.setSelected(True)
                except Exception:
                    continue
            if was_editing is not None and was_editing in self.textblk_item_list:
                self.canvas.editing_textblkitem = was_editing

    def _get_review_target_blocks(self, target_block_indices: Optional[List[int]] = None) -> List[TextBlkItem]:
        """Resolve review targets deterministically by block index.

        If target_block_indices is None, currently selected blocks are used.
        """
        if not target_block_indices:
            return [blk for blk in self.canvas.selected_text_items() if not blk.fontformat.vertical]
        all_blocks = [
            blk for blk in self.textblk_item_list
            if blk is not None and not getattr(blk.fontformat, "vertical", False)
        ]
        selected_by_idx = {blk.idx: blk for blk in all_blocks}
        targets: List[TextBlkItem] = []
        for idx in target_block_indices:
            blk = selected_by_idx.get(idx)
            if blk is None:
                continue
            targets.append(blk)
        return targets

    def _estimate_block_bubble_center(self, blkitem: TextBlkItem) -> Optional[Tuple[float, float]]:
        """Estimate bubble center for a block so off-center review checks can run."""
        img = self.imgtrans_proj.img_array
        if img is None:
            return None
        abr = blkitem.absBoundingRect(qrect=True)
        br = [int(abr.x()), int(abr.y()), int(abr.width()), int(abr.height())]
        if br[2] <= 0 or br[3] <= 0:
            return None
        try:
            enlarge_ratio = min(max(br[2] / max(br[3], 1), br[3] / max(br[2], 1)) * 1.5, 3.0)
            mask, _area, mask_xyxy, _region_rect = extract_ballon_region(
                img, br, enlarge_ratio=enlarge_ratio, cal_region_rect=True, margin_px=LAYOUT_BALLOON_SHAPE_CROP_PAD,
                seed_point_abs=(abr.x() + abr.width() / 2, abr.y() + abr.height() / 2),
            )
            cx_crop, cy_crop = mask_centroid_in_crop(mask)
            return (float(mask_xyxy[0] + cx_crop), float(mask_xyxy[1] + cy_crop))
        except Exception as e:
            LOGGER.debug("layout review: failed to estimate bubble center for block idx=%s: %s", getattr(blkitem, "idx", -1), e)
            return None

    def _normalize_review_actions(self, actions: List[ReviewAction], target_indices: Set[int]) -> List[ReviewAction]:
        """Sanitize + dedupe review actions before applying."""
        normalized: List[ReviewAction] = []
        seen = set()
        for action in actions:
            if action.block_index not in target_indices:
                continue
            if action.action == "move":
                dx = float(action.args.get("dx", 0.0) or 0.0)
                dy = float(action.args.get("dy", 0.0) or 0.0)
                if dx == 0.0 and dy == 0.0:
                    continue
            elif action.action == "resize":
                scale = float(action.args.get("scale", 1.0) or 1.0)
                if scale <= 0 or abs(scale - 1.0) < 1e-6:
                    continue
            elif action.action in {"resize_to_recommended_box", "resize_to_mask_safe_box"}:
                width = float(action.args.get("width", 0.0) or 0.0)
                height = float(action.args.get("height", 0.0) or 0.0)
                if width <= 0 or height <= 0:
                    continue
            elif action.action == "set_font_size":
                size_pt = float(action.args.get("font_size", 0.0) or 0.0)
                if size_pt <= 0:
                    continue
            key = (action.action, action.block_index, tuple(sorted((action.args or {}).items())))
            if key in seen:
                continue
            seen.add(key)
            normalized.append(action)
        return normalized

    def _apply_layout_review_actions(self, actions: List[ReviewAction], selected: List[TextBlkItem]) -> None:
        """Apply planner actions through existing commands whenever possible."""
        selected_by_idx = {blk.idx: blk for blk in selected}
        if not selected_by_idx:
            return
        actions = self._normalize_review_actions(actions, set(selected_by_idx.keys()))
        if not actions:
            return
        grouped = collect_actions_from_list(actions)
        undo_stack = self.canvas.get_active_undostack()
        if undo_stack is not None:
            undo_stack.beginMacro("Layout review agent")
        try:
            # Keep existing action handlers first to preserve current behavior.
            if grouped["auto_fit"] or grouped["shrink_to_fit"]:
                self.onAutoFitFontToBox()
            if grouped["resize_to_fit_content"]:
                self.onResizeToFitContent()
            for action in grouped["resize_to_recommended_box"] + grouped.get("resize_to_mask_safe_box", []):
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                rect = blkitem.absBoundingRect(qrect=True)
                new_w = max(1.0, float(action.args.get("width", rect.width()) or rect.width()))
                new_h = max(1.0, float(action.args.get("height", rect.height()) or rect.height()))
                if new_w <= rect.width() * 1.01 and new_h <= rect.height() * 1.01:
                    continue
                blkitem.oldRect = rect
                x1, y1, x2, y2 = centered_resize_xyxy([rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height()], [new_w, new_h])
                blkitem.setRect([x1, y1, x2 - x1, y2 - y1], repaint=True)
                self.canvas.push_undo_command(ReshapeItemCommand(blkitem))
            if grouped["center_in_bubble"] or grouped["recenter"]:
                self.onCenterInBubble()
            for action in grouped["normalize_vertical_punctuation"]:
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                new_text = normalize_vertical_punctuation(blkitem.toPlainText())
                if new_text and new_text != blkitem.toPlainText():
                    blkitem.setPlainText(new_text)
                    if blkitem.blk is not None:
                        blkitem.blk.translation = new_text
            for action in grouped["set_alignment"]:
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                alignment = int(action.args.get("alignment", 1) or 1)
                alignment = max(0, min(2, alignment))
                blkitem.fontformat.alignment = alignment
                blkitem.setAlignment(alignment, restore_cursor=False)
                if blkitem.blk is not None:
                    blkitem.blk.fontformat.alignment = alignment
            for action in grouped["apply_contrast_stroke"]:
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                stroke = max(0.0, float(action.args.get("stroke_width", 0.06) or 0.06))
                blkitem.fontformat.stroke_width = max(float(getattr(blkitem.fontformat, "stroke_width", 0.0) or 0.0), stroke)
                if blkitem.blk is not None:
                    blkitem.blk.fontformat.stroke_width = blkitem.fontformat.stroke_width
                blkitem.set_fontformat(blkitem.fontformat)
            for action in grouped.get("apply_double_outline", []):
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                back_width = max(0.0, float(action.args.get("secondary_stroke_width", 0.20) or 0.20))
                back_color = action.args.get("secondary_srgb", [255, 255, 255]) or [255, 255, 255]
                blkitem.fontformat.secondary_stroke_width = max(float(getattr(blkitem.fontformat, "secondary_stroke_width", 0.0) or 0.0), back_width)
                blkitem.fontformat.secondary_srgb = list(back_color)[:3]
                if blkitem.blk is not None:
                    blkitem.blk.fontformat.secondary_stroke_width = blkitem.fontformat.secondary_stroke_width
                    blkitem.blk.fontformat.secondary_srgb = blkitem.fontformat.secondary_srgb
                blkitem.set_fontformat(blkitem.fontformat)
                blkitem.repaint_background()
            for action in grouped.get("polish_typography", []):
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                self._apply_typography_cleanup_to_item(blkitem)
            for action in grouped.get("smart_fit", []):
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                self._apply_smart_fit_to_item(blkitem)
            for action in grouped["switch_writing_mode"]:
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                mode = str(action.args.get("writing_mode", "auto") or "auto")
                blkitem.fontformat.writing_mode = mode
                resolved = resolve_writing_mode(mode, blkitem.toPlainText(), (blkitem.absBoundingRect(qrect=True).width(), blkitem.absBoundingRect(qrect=True).height()))
                blkitem.setVertical(resolved == "vertical_rl")
                if resolved == "rtl":
                    blkitem.setAlignment(2, restore_cursor=False)
                if blkitem.blk is not None:
                    blkitem.blk.fontformat.writing_mode = mode
                    blkitem.blk.fontformat.vertical = resolved == "vertical_rl"
                    blkitem.blk.vertical = resolved == "vertical_rl"
            for action in grouped["increase_padding"]:
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                padding = max(0.0, float(action.args.get("padding", 2.0) or 2.0))
                blkitem.fontformat.text_padding = max(float(getattr(blkitem.fontformat, 'text_padding', 0.0) or 0.0), padding)
                blkitem.setPadding(blkitem.fontformat.text_padding)
                if blkitem.blk is not None:
                    blkitem.blk.fontformat.text_padding = blkitem.fontformat.text_padding
            for action in grouped["set_line_break_strategy"]:
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                strategy = str(action.args.get("line_break_strategy", "auto") or "auto")
                blkitem.fontformat.line_break_strategy = strategy
                if blkitem.blk is not None:
                    blkitem.blk.fontformat.line_break_strategy = strategy
            for action in grouped["tighten_letter_spacing"]:
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                spacing = max(0.80, min(float(getattr(blkitem.fontformat, 'letter_spacing', 1.0) or 1.0), float(action.args.get("letter_spacing", 0.94) or 0.94)))
                blkitem.fontformat.letter_spacing = spacing
                if blkitem.blk is not None:
                    blkitem.blk.fontformat.letter_spacing = spacing
                blkitem.set_fontformat(blkitem.fontformat)
            for action in grouped["balance_lines"]:
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                rect = blkitem.absBoundingRect(qrect=True)
                fm = QFontMetricsF(blkitem.font())
                avg_w = max(1.0, fm.horizontalAdvance('漢A') / 2.0)
                balanced = balance_lines(blkitem.toPlainText().replace('\n', ''), max(2, int(rect.width() / avg_w)))
                if balanced and balanced != blkitem.toPlainText():
                    blkitem.setPlainText(balanced)
                    if blkitem.blk is not None:
                        blkitem.blk.translation = balanced
            for action in grouped["apply_manga_preset"]:
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                preset_id = str(action.args.get("preset", "default_manga_bubble") or "default_manga_bubble")
                from utils.text_rendering import manga_presets
                from utils.config import pcfg
                preset = manga_presets(pcfg).get(preset_id)
                if not preset:
                    continue
                for key, value in preset.items():
                    if key != 'label' and hasattr(blkitem.fontformat, key):
                        setattr(blkitem.fontformat, key, value)
                blkitem.fontformat.manga_preset = preset_id
                blkitem.set_fontformat(blkitem.fontformat)
                if blkitem.blk is not None:
                    blkitem.blk.fontformat = blkitem.fontformat.deepcopy()
            for action in grouped["apply_font_fallback"]:
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                chain = str(action.args.get("fallback_chain", "") or "")
                if chain:
                    existing = str(getattr(blkitem.fontformat, 'fallback_font_chain', '') or '')
                    if not existing:
                        blkitem.fontformat.fallback_font_chain = chain
                blkitem.set_fontformat(blkitem.fontformat)
                if blkitem.blk is not None:
                    blkitem.blk.fontformat = blkitem.fontformat.deepcopy()
            for action in grouped["move"]:
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                dx = float(action.args.get("dx", 0.0) or 0.0)
                dy = float(action.args.get("dy", 0.0) or 0.0)
                if dx == 0.0 and dy == 0.0:
                    continue
                rect = blkitem.absBoundingRect(qrect=True)
                blkitem.oldRect = rect
                blkitem.setRect([rect.x() + dx, rect.y() + dy, rect.width(), rect.height()], repaint=True)
                self.canvas.push_undo_command(ReshapeItemCommand(blkitem))
            for action in grouped["resize"]:
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                scale = float(action.args.get("scale", 1.0) or 1.0)
                if scale <= 0:
                    continue
                rect = blkitem.absBoundingRect(qrect=True)
                if rect.width() <= 0 or rect.height() <= 0:
                    continue
                new_w = max(1.0, rect.width() * scale)
                new_h = max(1.0, rect.height() * scale)
                cx = rect.x() + rect.width() / 2.0
                cy = rect.y() + rect.height() / 2.0
                blkitem.oldRect = rect
                blkitem.setRect([cx - new_w / 2.0, cy - new_h / 2.0, new_w, new_h], repaint=True)
                self.canvas.push_undo_command(ReshapeItemCommand(blkitem))
            for action in grouped["set_font_size"]:
                blkitem = selected_by_idx.get(action.block_index)
                if blkitem is None:
                    continue
                size_pt = float(action.args.get("font_size", 0.0) or 0.0)
                if size_pt <= 0:
                    continue
                pairw = self.pairwidget_list[blkitem.idx] if 0 <= blkitem.idx < len(self.pairwidget_list) else None
                if pairw is None:
                    continue
                fmt = blkitem.get_fontformat()
                fmt.font_size = pt2px(size_pt)
                self.canvas.push_undo_command(ApplyFontformatCommand([blkitem], [pairw.e_trans], fmt))
        finally:
            if undo_stack is not None:
                undo_stack.endMacro()

    def onSetBalloonShape(self, shape: str):
        """Set global balloon shape and run layout on selected blocks."""
        if shape not in ("auto", "round", "elongated", "narrow", "diamond", "square", "bevel", "pentagon", "point"):
            return
        pcfg.module.layout_balloon_shape = shape
        selected_blks = self.canvas.selected_text_items()
        selected_blks = [blk for blk in selected_blks if not blk.fontformat.vertical]
        if not selected_blks:
            return
        old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
        for blkitem in selected_blks:
            blkitem.fontformat.text_box_shape = shape
            old_html_lst.append(blkitem.toHtml())
            old_rect_lst.append(blkitem.absBoundingRect(qrect=True))
            trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
            self.layout_textblk(blkitem)
            self.layout_textblk(blkitem)
        self.canvas.push_undo_command(AutoLayoutCommand(selected_blks, old_rect_lst, old_html_lst, trans_widget_lst))

    def onAutoFitBinarySearch(self):
        """Run layout with binary search for font size on selected blocks (temporarily enable layout_font_binary_search)."""
        selected_blks = self.canvas.selected_text_items()
        selected_blks = [blk for blk in selected_blks if not blk.fontformat.vertical]
        if not selected_blks:
            return
        old_binary = getattr(pcfg.module, "layout_font_binary_search", False)
        try:
            pcfg.module.layout_font_binary_search = True
            old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
            for blkitem in selected_blks:
                old_html_lst.append(blkitem.toHtml())
                old_rect_lst.append(blkitem.absBoundingRect(qrect=True))
                trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
                self.layout_textblk(blkitem)
                self.layout_textblk(blkitem)
            self.canvas.push_undo_command(AutoLayoutCommand(selected_blks, old_rect_lst, old_html_lst, trans_widget_lst))
        finally:
            pcfg.module.layout_font_binary_search = old_binary

    def onResizeToFitContent(self):
        """Resize selected text box(es) so the full text is visible, respecting current line breaks (not one long line)."""
        selected_blks = self.canvas.selected_text_items()
        if not selected_blks:
            return
        min_w, min_h = 24.0, 24.0
        for blkitem in selected_blks:
            text = blkitem.toPlainText().strip()
            blkitem.oldRect = blkitem.absBoundingRect(qrect=True)
            pad = blkitem.padding()
            if not text:
                new_w = max(min_w, blkitem.oldRect.width())
                new_h = max(min_h, blkitem.oldRect.height())
            else:
                # Use the document layout's block rects so we respect line breaks and wrapping (avoid "straight line").
                doc = blkitem.document()
                layout = doc.documentLayout() if doc else None
                if layout is not None:
                    content_w, content_h = 0.0, 0.0
                    block = doc.firstBlock()
                    while block.isValid():
                        r = layout.blockBoundingRect(block)
                        content_w = max(content_w, r.width())
                        content_h += r.height()
                        block = block.next()
                    if content_w > 0 and content_h > 0:
                        new_w = max(min_w, content_w + 2 * pad)
                        new_h = max(min_h, content_h + 2 * pad)
                    else:
                        new_w = max(min_w, blkitem.oldRect.width())
                        new_h = max(min_h, blkitem.oldRect.height())
                else:
                    ff = blkitem.fontformat
                    font = QFont(ff.font_family or "Arial", max(1, int(round(ff.font_size))))
                    if getattr(ff, "bold", False):
                        font.setBold(True)
                    if getattr(ff, "italic", False):
                        font.setItalic(True)
                    fm = QFontMetricsF(font)
                    br = fm.boundingRect(text)
                    new_w = max(min_w, br.width() + 2 * pad)
                    new_h = max(min_h, br.height() + 2 * pad)
            blkitem.set_size(new_w, new_h, set_layout_maxsize=True, set_blk_size=True, preserve_topleft=False)
            self.canvas.push_undo_command(ReshapeItemCommand(blkitem))

    def _center_blkitem_in_bubble(self, blkitem: TextBlkItem, img: np.ndarray, im_w: int, im_h: int) -> bool:
        """Move this text box so its center aligns with the bubble centroid. No undo. Returns True if position changed.
        Skips blocks that have no detected bubble (project mask) so free-standing text is not moved."""
        abr = blkitem.absBoundingRect(qrect=True)
        bx, by, bw, bh = abr.x(), abr.y(), abr.width(), abr.height()
        if bw <= 0 or bh <= 0:
            return False
        # Only center when block has a detected bubble (project mask); skip free-standing text
        page_name = getattr(self.imgtrans_proj, "current_img", None)
        if page_name:
            try:
                mask_array = self.imgtrans_proj.load_mask_by_imgname(page_name)
                if mask_array is not None and mask_array.size > 0 and mask_array.shape[:2] == img.shape[:2]:
                    br_list = [abr.x(), abr.y(), abr.width(), abr.height()]
                    block_mask, _ = get_block_mask(br_list, mask_array, blkitem.rotation())
                    if block_mask is None or block_mask.size == 0:
                        return False
            except Exception:
                return False
        else:
            return False
        cx, cy = bx + bw / 2, by + bh / 2
        if bw > 0.5 * im_w or bh > 0.5 * im_h:
            search_w = min(bw, max(80, int(0.25 * im_w)))
            search_h = min(bh, max(60, int(0.25 * im_h)))
            br_x = max(0, min(int(cx - search_w / 2), im_w - search_w))
            br_y = max(0, min(int(cy - search_h / 2), im_h - search_h))
            br = [br_x, br_y, search_w, search_h]
        else:
            br = [int(abr.x()), int(abr.y()), int(abr.width()), int(abr.height())]
        try:
            enlarge_ratio = min(max(br[2] / br[3], br[3] / br[2]) * 1.5, 3.0)
            mask, _area, mask_xyxy, region_rect = extract_ballon_region(
                img, br, enlarge_ratio=enlarge_ratio, cal_region_rect=True, margin_px=LAYOUT_BALLOON_SHAPE_CROP_PAD,
                seed_point_abs=(cx, cy),
            )
        except Exception:
            return False
        if region_rect is None or len(region_rect) < 4 or region_rect[2] <= 0 or region_rect[3] <= 0:
            return False
        cx_crop, cy_crop = mask_centroid_in_crop(mask)
        im_cx = mask_xyxy[0] + cx_crop
        im_cy = mask_xyxy[1] + cy_crop
        new_x = im_cx - bw / 2
        new_y = im_cy - bh / 2
        bubble_x = mask_xyxy[0] + region_rect[0]
        bubble_y = mask_xyxy[1] + region_rect[1]
        bubble_w, bubble_h = region_rect[2], region_rect[3]
        # Guard against contour extraction locking onto a neighboring bubble: the
        # selected text center must be inside (or very near) the detected bubble,
        # and the requested move must be plausible for this bubble size.
        pad_x = max(10.0, bubble_w * 0.15)
        pad_y = max(10.0, bubble_h * 0.15)
        if not (bubble_x - pad_x <= cx <= bubble_x + bubble_w + pad_x and bubble_y - pad_y <= cy <= bubble_y + bubble_h + pad_y):
            return False
        if max(abs(im_cx - cx), abs(im_cy - cy)) > max(80.0, max(bubble_w, bubble_h) * 0.9):
            return False
        min_x, min_y = bubble_x, bubble_y
        max_x = bubble_x + bubble_w - bw
        max_y = bubble_y + bubble_h - bh
        if max_x >= min_x:
            new_x = min(max(new_x, min_x), max_x)
        if max_y >= min_y:
            new_y = min(max(new_y, min_y), max_y)
        new_x = max(0, min(new_x, im_w - bw))
        new_y = max(0, min(new_y, im_h - bh))
        if abs(new_x - bx) < 0.5 and abs(new_y - by) < 0.5:
            return False
        blkitem.setRect([new_x, new_y, bw, bh], repaint=True)
        if blkitem.blk is not None:
            blkitem.blk._bounding_rect = blkitem.absBoundingRect()
        return True

    def _is_blk_close_to_others(self, blkitem: TextBlkItem, other_blks: List, gap_px: float) -> bool:
        """True if this block's rect is within gap_px of any other block (combined/overlapping bubbles)."""
        r = blkitem.absBoundingRect(qrect=True)
        for other in other_blks:
            if other is blkitem:
                continue
            ro = other.absBoundingRect(qrect=True)
            # Inflate other by gap_px so "close" = within gap_px edge-to-edge
            expanded = QRectF(
                ro.x() - gap_px, ro.y() - gap_px,
                ro.width() + 2 * gap_px, ro.height() + 2 * gap_px
            )
            if r.intersects(expanded):
                return True
        return False

    def _text_box_overflows_bubble(
        self, blkitem: TextBlkItem, mask_xyxy: List, region_rect: List
    ) -> bool:
        """True if the text box is larger than the bubble or extends outside the bubble rect (no mask)."""
        if not region_rect or len(region_rect) < 4 or region_rect[2] <= 0 or region_rect[3] <= 0:
            return False
        bubble_x = mask_xyxy[0] + region_rect[0]
        bubble_y = mask_xyxy[1] + region_rect[1]
        bubble_w, bubble_h = region_rect[2], region_rect[3]
        abr = blkitem.absBoundingRect(qrect=True)
        bx, by, bw, bh = abr.x(), abr.y(), abr.width(), abr.height()
        if bw <= 0 or bh <= 0:
            return False
        if bw > bubble_w or bh > bubble_h:
            return True
        if bx < bubble_x or by < bubble_y:
            return True
        if bx + bw > bubble_x + bubble_w or by + bh > bubble_y + bubble_h:
            return True
        return False

    def _lines_outside_bubble_mask(
        self, blkitem: TextBlkItem, mask: np.ndarray, mask_xyxy: List
    ) -> bool:
        """True if any document line (block) extends outside the bubble mask (e.g. oval). Geometry only, no model."""
        if mask is None or mask.size == 0 or len(mask_xyxy) < 4:
            return False
        doc = blkitem.document()
        if doc is None:
            return False
        layout = doc.documentLayout()
        if layout is None:
            return False
        pad = blkitem.padding()
        item_pos = blkitem.pos()
        x0 = item_pos.x() + pad
        y0 = item_pos.y() + pad
        m_h, m_w = mask.shape[0], mask.shape[1]
        mx1, my1 = int(mask_xyxy[0]), int(mask_xyxy[1])
        block = doc.firstBlock()
        while block.isValid():
            r = layout.blockBoundingRect(block)
            if r.width() < 1 and r.height() < 1:
                block = block.next()
                continue
            # Line rect in image coords (scene = image)
            lx = x0 + r.x()
            ly = y0 + r.y()
            lw, lh = r.width(), r.height()
            # Sample center and corners of the line rect
            for sx, sy in [
                (lx + lw / 2, ly + lh / 2),
                (lx, ly),
                (lx + lw, ly),
                (lx + lw, ly + lh),
                (lx, ly + lh),
            ]:
                cx = int(round(sx - mx1))
                cy = int(round(sy - my1))
                if 0 <= cx < m_w and 0 <= cy < m_h:
                    if mask[cy, cx] <= 0:
                        return True
            block = block.next()
        return False

    def _clamp_rect_to_original(self, new_x: float, new_y: float, new_w: float, new_h: float, orig_rect: List[float]) -> Tuple[float, float, float, float]:
        """Clamp new rect so it stays within the original text box area (no moving or growing past it)."""
        if not orig_rect or len(orig_rect) < 4 or orig_rect[2] <= 0 or orig_rect[3] <= 0:
            return new_x, new_y, new_w, new_h
        ox, oy, ow, oh = orig_rect[0], orig_rect[1], orig_rect[2], orig_rect[3]
        new_w = min(new_w, ow)
        new_h = min(new_h, oh)
        new_x = max(ox, min(new_x, ox + ow - new_w))
        new_y = max(oy, min(new_y, oy + oh - new_h))
        return new_x, new_y, new_w, new_h

    def _fix_overflow_after_layout(
        self,
        blkitem: TextBlkItem,
        mask: np.ndarray,
        mask_xyxy: List,
        region_rect: List,
        is_osb: bool,
        img: Optional[np.ndarray] = None,
        original_rect: Optional[List[float]] = None,
    ) -> None:
        """If box is too large or lines go outside the bubble, shrink box and/or scale font. Optional model check. Clamps to original_rect when given."""
        if is_osb or not region_rect or len(region_rect) < 4 or region_rect[2] <= 0 or region_rect[3] <= 0:
            return
        if not getattr(pcfg.module, "layout_check_overflow_after_layout", True):
            return
        bubble_x = mask_xyxy[0] + region_rect[0]
        bubble_y = mask_xyxy[1] + region_rect[1]
        bubble_w, bubble_h = region_rect[2], region_rect[3]
        abr = blkitem.absBoundingRect(qrect=True)
        bx, by, bw, bh = abr.x(), abr.y(), abr.width(), abr.height()
        img_area = (img.shape[0] * img.shape[1]) if img is not None and img.size > 0 else 0
        bubble_area = bubble_w * bubble_h
        bubble_suspiciously_large = img_area > 0 and (bubble_area > 0.5 * img_area or bubble_area > 2.5 * (bw * bh))

        def _apply_rect(nx: float, ny: float, nw: float, nh: float) -> None:
            if original_rect and len(original_rect) >= 4:
                nx, ny, nw, nh = self._clamp_rect_to_original(nx, ny, nw, nh, original_rect)
            blkitem.setRect([nx, ny, nw, nh], repaint=True)
            if blkitem.blk is not None:
                blkitem.blk._bounding_rect = blkitem.absBoundingRect()

        # 0) Optional model: secondary check if detection box is too large or too small for the bubble
        model_id = (getattr(pcfg.module, "layout_box_size_check_model_id", None) or "").strip()
        if model_id and img is not None and img.size > 0 and not bubble_suspiciously_large:
            try:
                result = check_box_size_from_model(img, mask_xyxy, model_id)
                if result == "too_large":
                    _apply_rect(bubble_x, bubble_y, bubble_w, bubble_h)
                    self._scale_font_to_fit_box(blkitem)
                    self.canvas.setProjSaveState(True)
                    return
                if result == "too_small" and (bw < bubble_w or bh < bubble_h):
                    new_w = min(bubble_w, 2 * bw) if bw > 0 else bubble_w
                    new_h = min(bubble_h, 2 * bh) if bh > 0 else bubble_h
                    _apply_rect(bubble_x, bubble_y, new_w, new_h)
                    self._scale_font_to_fit_box(blkitem)
                    self.canvas.setProjSaveState(True)
                    return
            except Exception:
                pass

        # 1) Box larger than bubble: shrink box to bubble and scale font to fit (skip if bubble is suspiciously huge)
        if not bubble_suspiciously_large and (bw > bubble_w or bh > bubble_h):
            _apply_rect(bubble_x, bubble_y, bubble_w, bubble_h)
            self._scale_font_to_fit_box(blkitem)
            return

        # 2) Box extends outside bubble rect: move to fit (skip if bubble is suspiciously huge); clamp to original area
        if not bubble_suspiciously_large and (bx < bubble_x or by < bubble_y or bx + bw > bubble_x + bubble_w or by + bh > bubble_y + bubble_h):
            new_x = max(bubble_x, min(bx, bubble_x + bubble_w - bw))
            new_y = max(bubble_y, min(by, bubble_y + bubble_h - bh))
            _apply_rect(new_x, new_y, bw, bh)

        # 3) Lines outside bubble mask (e.g. oval): scale font down so text stays inside (skip if bubble suspicious)
        if not bubble_suspiciously_large and mask is not None and mask.size > 0 and self._lines_outside_bubble_mask(blkitem, mask, mask_xyxy):
            for _ in range(2):
                pt = blkitem.font().pointSizeF()
                if pt < 6:
                    break
                new_pt = max(6.0, pt * 0.88)
                blkitem.setFontSize(new_pt)
                if blkitem.blk is not None and hasattr(blkitem.blk, "fontformat"):
                    try:
                        blkitem.blk.fontformat.font_size = pt2px(new_pt)
                    except Exception:
                        pass
                blkitem.layout.reLayoutEverything()
                blkitem.repaint_background()
                if not self._lines_outside_bubble_mask(blkitem, mask, mask_xyxy):
                    break
        self.canvas.setProjSaveState(True)

    def onCenterInBubble(self):
        """Move selected text box(es) so their center aligns with the bubble centroid (no resize, no model)."""
        selected_blks = self.canvas.selected_text_items()
        selected_blks = [b for b in selected_blks if not getattr(b.fontformat, "vertical", False)]
        if not selected_blks:
            return
        img = self.imgtrans_proj.img_array
        if img is None:
            return
        im_h, im_w = img.shape[:2]
        page_name = getattr(self.imgtrans_proj, "current_img", None)
        for blkitem in selected_blks:
            abr = blkitem.absBoundingRect(qrect=True)
            bx, by, bw, bh = abr.x(), abr.y(), abr.width(), abr.height()
            if bw <= 0 or bh <= 0:
                continue
            # Skip blocks with no detected bubble (project mask) so free-standing text is not moved
            if page_name:
                try:
                    mask_array = self.imgtrans_proj.load_mask_by_imgname(page_name)
                    if mask_array is None or mask_array.size == 0 or mask_array.shape[:2] != (im_h, im_w):
                        continue
                    block_mask, _ = get_block_mask([bx, by, bw, bh], mask_array, blkitem.rotation())
                    if block_mask is None or block_mask.size == 0:
                        continue
                except Exception:
                    continue
            else:
                continue
            # When the box is very large (bigger than the bubble), using the full box as the search
            # region makes extract_ballon_region find the wrong contour and the centroid moves the box
            # far away. Use a search rect centered on the box center with a capped size so we always
            # look for the bubble near the current box.
            cx, cy = bx + bw / 2, by + bh / 2
            if bw > 0.5 * im_w or bh > 0.5 * im_h:
                search_w = min(bw, max(80, int(0.25 * im_w)))
                search_h = min(bh, max(60, int(0.25 * im_h)))
                br_x = max(0, min(int(cx - search_w / 2), im_w - search_w))
                br_y = max(0, min(int(cy - search_h / 2), im_h - search_h))
                br = [br_x, br_y, search_w, search_h]
            else:
                br = [int(abr.x()), int(abr.y()), int(abr.width()), int(abr.height())]
            try:
                enlarge_ratio = min(max(br[2] / br[3], br[3] / br[2]) * 1.5, 3.0)
                mask, _area, mask_xyxy, region_rect = extract_ballon_region(
                    img, br, enlarge_ratio=enlarge_ratio, cal_region_rect=True, margin_px=LAYOUT_BALLOON_SHAPE_CROP_PAD,
                    seed_point_abs=(cx, cy),
                )
            except Exception:
                continue
            if region_rect is None or len(region_rect) < 4 or region_rect[2] <= 0 or region_rect[3] <= 0:
                continue
            cx_crop, cy_crop = mask_centroid_in_crop(mask)
            im_cx = mask_xyxy[0] + cx_crop
            im_cy = mask_xyxy[1] + cy_crop
            new_x = im_cx - bw / 2
            new_y = im_cy - bh / 2
            bubble_x = mask_xyxy[0] + region_rect[0]
            bubble_y = mask_xyxy[1] + region_rect[1]
            bubble_w, bubble_h = region_rect[2], region_rect[3]
            # Guard against centering to an unrelated contour/bubble. This keeps
            # Format → Center in bubble from jumping dramatically across the page.
            pad_x = max(10.0, bubble_w * 0.15)
            pad_y = max(10.0, bubble_h * 0.15)
            if not (bubble_x - pad_x <= cx <= bubble_x + bubble_w + pad_x and bubble_y - pad_y <= cy <= bubble_y + bubble_h + pad_y):
                continue
            if max(abs(im_cx - cx), abs(im_cy - cy)) > max(80.0, max(bubble_w, bubble_h) * 0.9):
                continue
            min_x = bubble_x
            max_x = bubble_x + bubble_w - bw
            min_y = bubble_y
            max_y = bubble_y + bubble_h - bh
            if max_x >= min_x:
                new_x = min(max(new_x, min_x), max_x)
            if max_y >= min_y:
                new_y = min(max(new_y, min_y), max_y)
            new_x = max(0, min(new_x, im_w - bw))
            new_y = max(0, min(new_y, im_h - bh))
            blkitem.oldRect = abr
            blkitem.setRect([new_x, new_y, bw, bh], repaint=True)
            if blkitem.blk is not None:
                blkitem.blk._bounding_rect = blkitem.absBoundingRect()
            self.canvas.push_undo_command(ReshapeItemCommand(blkitem))
        self.canvas.setProjSaveState(True)

    def _scale_font_to_fit_box(self, blkitem: TextBlkItem):
        """Scale this block's font so its text fits inside the current box (no layout/box change)."""
        br = blkitem.absBoundingRect(qrect=True)
        box_w = br.width()
        box_h = br.height()
        if box_w < 1 or box_h < 1:
            return
        doc = blkitem.document()
        if doc is None:
            return
        doc_w = doc.size().width()
        doc_h = doc.size().height()
        if doc_w < 1 and doc_h < 1:
            return
        if doc_w < 1:
            doc_w = 1
        if doc_h < 1:
            doc_h = 1
        scale = min(box_w / doc_w, box_h / doc_h)
        current_pt = max(1.0, blkitem.font().pointSizeF() or 1.0)
        ff = getattr(blkitem.blk, 'fontformat', None) if blkitem.blk is not None else None
        min_pt, max_pt = resolve_fit_font_size_bounds(
            getattr(ff, 'fit_font_size_min', 0.0) if ff else 0.0,
            getattr(ff, 'fit_font_size_max', 0.0) if ff else 0.0,
            getattr(getattr(pcfg, 'module', None), 'layout_font_size_min', 6.0),
            getattr(getattr(pcfg, 'module', None), 'layout_font_size_max', 96.0),
        )
        min_scale = max(0.1, min_pt / current_pt)
        max_scale = min(3.0, max_pt / current_pt)
        scale = max(min_scale, min(scale, max_scale))
        if abs(scale - 1.0) < 0.01:
            return
        blkitem.setRelFontSize(scale, repaint_background=True)
        if blkitem.blk is not None and hasattr(blkitem.blk, 'fontformat'):
            try:
                new_pt = blkitem.font().pointSizeF()
                if new_pt > 0:
                    blkitem.blk.fontformat.font_size = pt2px(new_pt)
            except Exception:
                pass
        self.canvas.setProjSaveState(True)


    def _auto_refine_rendered_fit(
        self,
        blkitem: TextBlkItem,
        mask: np.ndarray = None,
        mask_xyxy: List = None,
        region_rect: List = None,
        is_osb: bool = False,
        max_passes: int = 4,
    ) -> None:
        """Final rendered-size polish after line breaking and textbox placement.

        This is intentionally small and deterministic: it expands only when the
        rendered document badly under-fills the box, shrinks when it overflows,
        and does an extra mask-corner shrink for curved bubbles.
        """
        if is_osb:
            return
        doc = blkitem.document()
        if doc is None:
            return
        preset = _auto_layout_preset()
        for _ in range(max(1, int(max_passes))):
            br = blkitem.absBoundingRect(qrect=True)
            box_w, box_h = br.width(), br.height()
            if box_w <= 1 or box_h <= 1:
                return
            doc_size = doc.size()
            mask_outside = bool(
                mask is not None
                and mask_xyxy is not None
                and region_rect is not None
                and self._lines_outside_bubble_mask(blkitem, mask, mask_xyxy)
            )
            scale = auto_rendered_fit_scale(
                doc_size.width(),
                doc_size.height(),
                box_w,
                box_h,
                preset,
                allow_expand=not mask_outside,
            )
            if mask_outside:
                scale = min(scale, 0.90)
            if abs(scale - 1.0) < 0.012:
                break
            blkitem.setRelFontSize(scale, repaint_background=True)
            if blkitem.blk is not None and hasattr(blkitem.blk, 'fontformat'):
                try:
                    new_pt = blkitem.font().pointSizeF()
                    if new_pt > 0:
                        blkitem.blk.fontformat.font_size = pt2px(new_pt)
                except Exception:
                    pass
            blkitem.layout.reLayoutEverything()
            doc = blkitem.document()
            if doc is None:
                break
        if blkitem.blk is not None:
            blkitem.blk._bounding_rect = blkitem.absBoundingRect()
        blkitem.repaint_background()
        self.canvas.setProjSaveState(True)

    def onResetAngle(self):
        selected_blks = self.canvas.selected_text_items()
        if len(selected_blks) > 0:
            self.canvas.push_undo_command(ResetAngleCommand(selected_blks, self.txtblkShapeControl))

    def onSqueezeBlk(self):
        selected_blks = self.canvas.selected_text_items()
        if len(selected_blks) > 0:
            self.canvas.push_undo_command(SqueezeCommand(selected_blks, self.txtblkShapeControl))

    def on_incanvas_selection_changed(self):
        if self.canvas.textEditMode():
            textitems = self.canvas.selected_text_items()
            self.textEditList.set_selected_list([t.idx for t in textitems])
            if len(textitems) == 1:
                self.formatpanel.set_textblk_item(textitems[-1])
            else:
                self.formatpanel.set_textblk_item(multi_select=bool(textitems))
            if hasattr(self.mainwindow, "on_translation_assist_selection_changed"):
                self.mainwindow.on_translation_assist_selection_changed()

    def layout_textblk(self, blkitem: TextBlkItem, text: str = None, mask: np.ndarray = None, bounding_rect: List = None, region_rect: List = None):
        '''
        Auto text layout (vertical writing not supported).
        Layout tuning is read from pcfg.module at runtime: layout_optimal_breaks, layout_hyphenation,
        layout_short_line_penalty, layout_height_overflow_penalty, optimize_line_breaks,
        layout_constrain_to_bubble, layout_collision_*. Font scaling runs when Auto layout is on
        and Font size = "decide by program" (or block has auto_fit_font_size).
        '''
        is_osb = (getattr(blkitem.blk, "label", None) or "").strip().lower() in OSB_LABELS
        if is_osb:
            setattr(blkitem.blk, "restore_original_region", False)

        img = self.imgtrans_proj.img_array
        if img is None:
            return
        im_h, im_w = img.shape[:2]
        img_area = im_h * im_w

        src_is_cjk = is_cjk(pcfg.module.translate_source)
        tgt_is_cjk = is_cjk(pcfg.module.translate_target)

        old_br_q = blkitem.absBoundingRect(qrect=True)
        old_br = [old_br_q.x(), old_br_q.y(), old_br_q.width(), old_br_q.height()]
        if old_br[2] < 1:
            return

        # Resolve per-textbox writing mode before deciding whether the horizontal auto-layout path applies.
        raw_text_for_mode = text if text is not None else blkitem.toPlainText()
        resolved_mode = resolve_writing_mode(getattr(blkitem.fontformat, 'writing_mode', 'auto'), raw_text_for_mode, (old_br[2], old_br[3]))
        blkitem.fontformat.vertical = resolved_mode == 'vertical_rl'
        if blkitem.blk is not None:
            blkitem.blk.fontformat.writing_mode = getattr(blkitem.fontformat, 'writing_mode', 'auto')
            blkitem.blk.fontformat.vertical = blkitem.fontformat.vertical
            blkitem.blk.vertical = blkitem.fontformat.vertical
        if resolved_mode == 'rtl':
            blkitem.setAlignment(2, restore_cursor=False)
        if resolved_mode == 'vertical_rl':
            new_text = normalize_vertical_punctuation(raw_text_for_mode)
            if new_text != raw_text_for_mode:
                blkitem.setPlainText(new_text)
                if blkitem.blk is not None:
                    blkitem.blk.translation = new_text
            blkitem.setVertical(True)
            return

        blk_font = blkitem.font()
        if blk_font.pointSizeF() <= 0 or blk_font.pointSize() <= 0:
            blk_font.setPointSizeF(10.0)
            blk_font.setPointSize(10)
        # Enforce a minimum readable font size whenever auto layout / auto-fit is driving font changes.
        # This prevents tiny (e.g. 3–4 pt) text for short lines like "That is also good.".
        if (
            ((self.auto_textlayout_flag and pcfg.let_fntsize_flag == 0 and pcfg.let_autolayout_flag)
             or getattr(blkitem.fontformat, 'auto_fit_font_size', False))
            and blk_font.pointSizeF() < LAYOUT_MIN_FONT_PT
        ):
            blk_font.setPointSizeF(LAYOUT_MIN_FONT_PT)
            blk_font.setPointSize(int(max(LAYOUT_MIN_FONT_PT, blk_font.pointSize() or 0)))
        fmt = blkitem.get_fontformat()
        blk_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, fmt.letter_spacing * 100)
        text_size_func = lambda text: get_text_size(QFontMetricsF(blk_font), text)

        restore_charfmts = False
        if text is None:
            text = blkitem.toPlainText()
            restore_charfmts = True

        if not text.strip():
            return
        fit_mode = getattr(fmt, 'fit_mode', 'shrink')
        if fit_mode in ('shrink', 'expand', 'preserve', 'balance'):
            fitted_size, fitted_text, fit_diag = fit_font_size_to_box(
                text, blk_font.pointSizeF(), (old_br[2], old_br[3]), fit_mode,
                getattr(fmt, 'writing_mode', 'auto'),
                min_font_size=float(getattr(fmt, 'fit_font_size_min', 0.0) or getattr(pcfg.module, 'layout_font_size_min', 8.0)),
                max_font_size=float(getattr(fmt, 'fit_font_size_max', 0.0) or getattr(pcfg.module, 'layout_font_size_max', 72.0)),
                line_spacing=float(getattr(fmt, 'line_spacing', 1.15) or 1.15),
                letter_spacing=float(getattr(fmt, 'letter_spacing', 1.0) or 1.0),
                padding=float(getattr(fmt, 'text_padding', 0.0) or 0.0),
                stroke_width=float(getattr(fmt, 'stroke_width', 0.0) or 0.0),
                secondary_stroke_width=float(getattr(fmt, 'secondary_stroke_width', 0.0) or 0.0),
                line_break_strategy=getattr(fmt, 'line_break_strategy', 'auto'),
            )
            if fit_mode in ('shrink', 'expand') and abs(fitted_size - blk_font.pointSizeF()) > 0.2:
                blk_font.setPointSizeF(max(1.0, fitted_size))
                blk_font.setPointSize(max(1, int(round(fitted_size))))
            if fitted_text != text:
                text = fitted_text

        # When mask not passed, try to get block's mask from project so we only run bubble/balloon logic when a bubble was detected
        if mask is None and img is not None:
            page_name = getattr(self.imgtrans_proj, "current_img", None)
            if page_name:
                try:
                    mask_array = self.imgtrans_proj.load_mask_by_imgname(page_name)
                    if mask_array is not None and mask_array.size > 0 and mask_array.shape[:2] == (im_h, im_w):
                        br = blkitem.absBoundingRect(max_h=im_h, max_w=im_w)
                        br_list = [br.x(), br.y(), br.width(), br.height()] if not isinstance(br, list) else br
                        mask, _ = get_block_mask(br_list, mask_array, blkitem.rotation())
                        if mask is None or mask.size == 0:
                            mask = None
                    else:
                        mask = None
                except Exception:
                    mask = None
            else:
                mask = None

        original_block_area = None  # used later to cap resize for huge blocks
        if mask is None:
            # No detected bubble (block had no mask): use only the text box area; do not run bubble detection or balloon/overflow logic.
            bounding_rect = blkitem.absBoundingRect(max_h=im_h, max_w=im_w)
            if bounding_rect[2] <= 0 or bounding_rect[3] <= 0:
                text = _merge_stub_lines_in_text(text)
                blkitem.setPlainText(text)
                if len(self.pairwidget_list) > blkitem.idx:
                    self.pairwidget_list[blkitem.idx].e_trans.setPlainText(text)
                return
            block_area = bounding_rect[2] * bounding_rect[3]
            original_block_area = block_area
            if img_area > 0 and block_area > 0.5 * img_area:
                max_w = max(80, int(0.5 * im_w))
                max_h = max(60, int(0.5 * im_h))
                cx = bounding_rect[0] + bounding_rect[2] / 2
                cy = bounding_rect[1] + bounding_rect[3] / 2
                bounding_rect = [
                    int(cx - max_w / 2),
                    int(cy - max_h / 2),
                    max_w,
                    max_h,
                ]
                bounding_rect[0] = max(0, min(bounding_rect[0], im_w - max_w))
                bounding_rect[1] = max(0, min(bounding_rect[1], im_h - max_h))
            region_rect = None
            mask_xyxy = [bounding_rect[0], bounding_rect[1], bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3]]
            balloon_shape = getattr(pcfg.module, "layout_balloon_shape", "auto") or "auto"
            # Synthetic rectangular mask so layout_text can run and wrap to box width (avoids one long line for free-standing text)
            bw, bh = max(1, int(bounding_rect[2])), max(1, int(bounding_rect[3]))
            mask = np.full((bh, bw), 255, dtype=np.uint8)
        else:
            # Detected bubble (block has mask): get bubble region and optionally shape from contour/model.
            bounding_rect = bounding_rect or blkitem.absBoundingRect(max_h=im_h, max_w=im_w)
            if bounding_rect[2] <= 0 or bounding_rect[3] <= 0:
                text = _merge_stub_lines_in_text(text)
                blkitem.setPlainText(text)
                if len(self.pairwidget_list) > blkitem.idx:
                    self.pairwidget_list[blkitem.idx].e_trans.setPlainText(text)
                return
            block_area = bounding_rect[2] * bounding_rect[3]
            original_block_area = block_area
            if tgt_is_cjk:
                max_enlarge_ratio = 2.5
            else:
                max_enlarge_ratio = 3
            enlarge_ratio = min(max(bounding_rect[2] / bounding_rect[3], bounding_rect[3] / bounding_rect[2]) * 1.5, max_enlarge_ratio)
            mask, ballon_area, mask_xyxy, region_rect = extract_ballon_region(
                img, bounding_rect, enlarge_ratio=enlarge_ratio, cal_region_rect=True, margin_px=LAYOUT_BALLOON_SHAPE_CROP_PAD
            )
            if region_rect is not None and len(region_rect) >= 4 and img_area > 0:
                region_area = region_rect[2] * region_rect[3]
                if region_area > 2.5 * block_area or region_area > 0.5 * img_area:
                    region_rect = None
            if region_rect is not None and len(region_rect) >= 4:
                rw, rh = region_rect[2], region_rect[3]
                if rw > 0 and rh > 0:
                    shape_cfg = getattr(pcfg.module, "layout_balloon_shape", "auto") or "auto"
                    if shape_cfg == "auto":
                        already = (getattr(blkitem.fontformat, "text_box_shape", None) or "").strip().lower()
                        if already in CONCRETE_BALLOON_SHAPES:
                            balloon_shape = already
                        else:
                            balloon_shape = _resolve_auto_balloon_shape(
                                getattr(pcfg.module, "layout_balloon_shape_auto_method", "contour_ratio"),
                                mask, mask_xyxy, img, rw, rh,
                            )
                    else:
                        balloon_shape = shape_cfg
                    inset = _shape_inner_inset(balloon_shape, text, rw, rh)
                    new_w, new_h = rw * inset, rh * inset
                    inset_rect = [
                        region_rect[0] + (rw - new_w) / 2,
                        region_rect[1] + (rh - new_h) / 2,
                        new_w,
                        new_h,
                    ]
                    if bool(getattr(pcfg.module, "layout_use_mask_safe_area", True)):
                        preset_settings = _auto_layout_settings_for(text, rw, rh, balloon_shape=balloon_shape)
                        safe = bubble_safe_text_rect(
                            mask,
                            inset_rect,
                            min_coverage=preset_settings.mask_min_coverage,
                            min_edge_coverage=preset_settings.mask_min_edge_coverage,
                        )
                        if safe.get("used_mask") and safe.get("rect"):
                            region_rect = safe["rect"]
                        else:
                            region_rect = inset_rect
                    else:
                        region_rect = inset_rect
                else:
                    balloon_shape = getattr(pcfg.module, "layout_balloon_shape", "auto") or "auto"
            else:
                balloon_shape = getattr(pcfg.module, "layout_balloon_shape", "auto") or "auto"
                mask_xyxy = [bounding_rect[0], bounding_rect[1], bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3]]

        # Store balloon shape on the block so the text box outline is drawn to match (ellipse, diamond, etc.).
        if getattr(blkitem.fontformat, "text_box_shape", None) != balloon_shape:
            blkitem.fontformat.text_box_shape = balloon_shape

        words, delimiter = seg_text(text, pcfg.module.translate_target)
        if len(words) < 1:
            return

        # Special-case very short English bubbles so they stay on a single line and expand horizontally.
        # Examples: "Mom", "Sing", "remind", "Short message", "TSK,".
        # Skip this path when the resulting box would overflow the bubble (so we wrap instead and avoid cropped text).
        if not tgt_is_cjk:
            clean_chars = ''.join(ch for ch in text if ch.isalnum())
            if 1 <= len(clean_chars) <= 16 and len(words) <= 4:
                single = text.replace('\n', ' ').strip()
                if single:
                    orig_pt = blk_font.pointSizeF()
                    fshort = QFontMetricsF(blk_font)
                    base_w = fshort.horizontalAdvance(single)
                    base_h = int(round(fmt.line_spacing * fshort.height()))
                    if base_w > 0 and base_h > 0:
                        # Prefer to fit inside bubble horizontally/vertically if region is known.
                        if region_rect is not None and len(region_rect) >= 4:
                            avail_w = max(8, region_rect[2] * 0.9)
                            avail_h = max(8, region_rect[3] * 0.9)
                        else:
                            avail_w = max(8, bounding_rect[2] * 0.9)
                            avail_h = max(8, bounding_rect[3] * 0.9)
                        scale = min(avail_w / base_w, avail_h / base_h, 1.0)
                        scale = max(scale, 0.25)
                        new_size = max(LAYOUT_MIN_FONT_PT, blk_font.pointSizeF() * scale)
                        short_scale = _auto_layout_text_scale(single, 1, avail_w, avail_h)
                        if short_scale != 1.0:
                            new_size = max(LAYOUT_MIN_FONT_PT, new_size * short_scale)
                        blkitem.setFontSize(new_size)
                        blk_font.setPointSizeF(new_size)
                        fshort = QFontMetricsF(blk_font)
                        base_w = fshort.horizontalAdvance(single)
                        base_h = int(round(fmt.line_spacing * fshort.height()))
                        layout_w = base_w * LAYOUT_WIDTH_STROKE_FACTOR
                        layout_h = int(base_h * LAYOUT_HEIGHT_PADDING)
                        # If bubble is known and this single-line box would overflow it, use normal layout so text wraps and fits.
                        if region_rect is not None and len(region_rect) >= 4 and region_rect[2] > 0 and region_rect[3] > 0:
                            if layout_w > region_rect[2] or layout_h > region_rect[3]:
                                blkitem.setFontSize(orig_pt)
                                blk_font.setPointSizeF(orig_pt)
                                pass  # fall through to multi-line layout
                            else:
                                layout_w = max(layout_w, LAYOUT_MIN_BOX_W)
                                layout_h = max(layout_h, LAYOUT_MIN_BOX_H)
                                blkitem.set_size(layout_w, layout_h, set_layout_maxsize=True)
                                blkitem.setPlainText(single)
                                blkitem._ensure_transparent_document_background()
                                if len(self.pairwidget_list) > blkitem.idx:
                                    self.pairwidget_list[blkitem.idx].e_trans.setPlainText(single)
                                return True
                        else:
                            layout_w = max(layout_w, LAYOUT_MIN_BOX_W)
                            layout_h = max(layout_h, LAYOUT_MIN_BOX_H)
                            blkitem.set_size(layout_w, layout_h, set_layout_maxsize=True)
                            blkitem.setPlainText(single)
                            blkitem._ensure_transparent_document_background()
                            if len(self.pairwidget_list) > blkitem.idx:
                                self.pairwidget_list[blkitem.idx].e_trans.setPlainText(single)
                            return True

        wl_list = get_words_length_list(QFontMetricsF(blk_font), words)
        text_w, text_h = text_size_func(text)
        text_area = text_w * text_h
        if tgt_is_cjk:
            line_height = int(round(fmt.line_spacing * text_size_func('X木')[1]))
        else:
            line_height = int(round(fmt.line_spacing * text_size_func('X')[1]))
        delimiter_len = text_size_func(delimiter)[0]
 
        ref_src_lines = False
        if not blkitem.blk.src_is_vertical:
            ref_src_lines = blkitem.blk.line_coord_valid(old_br)

        adaptive_fntsize = False
        resize_ratio = 1
        force_fit = getattr(fmt, 'auto_fit_font_size', False)
        # 2.1 Font scaling: configurable min/max and fit-to-bubble (used when clamping font size below).
        layout_font_fit_bubble = bool(getattr(pcfg.module, "layout_font_fit_bubble", True))
        layout_min_pt = max(LAYOUT_MIN_FONT_PT, float(getattr(pcfg.module, "layout_font_size_min", 8.0)))
        layout_max_pt = max(layout_min_pt, float(getattr(pcfg.module, "layout_font_size_max", 72.0)))
        # When Auto layout is on (and font size = "decide by program"): scale font to fit balloon area and avoid overflow.
        # When off: we still wrap to balloon and may shrink if overflow (post_resize_ratio), but no initial balloon-based scaling.
        if (self.auto_textlayout_flag and pcfg.let_fntsize_flag == 0 and pcfg.let_autolayout_flag) or force_fit:
            if blkitem.blk.src_is_vertical and blkitem.blk.vertical != blkitem.blk.src_is_vertical:
                adaptive_fntsize = True
                area_ratio = ballon_area / text_area
                ballon_area_thresh = 1.7
                downscale_constraint = 0.6
                region_w = (region_rect[2] if region_rect is not None and len(region_rect) >= 4 else bounding_rect[2])
                resize_ratio = np.clip(min(area_ratio / ballon_area_thresh, region_w / max(wl_list)), downscale_constraint, 1.0)

            else:
                if not src_is_cjk:
                    # Stricter scale-down so text stays inside bubble with margin (avoid overflow)
                    resize_ratio_ballon = max(ballon_area / 1.70 / text_area, 0.4)
                    if ref_src_lines:
                        _, src_width = blkitem.blk.normalizd_width_list(normalize=False)
                        resize_ratio_src = src_width / (sum(wl_list) + max((len(wl_list) - 1 - len(blkitem.blk.lines_array())), 0) * delimiter_len)
                        resize_ratio = min(resize_ratio_ballon, resize_ratio_src)
                    else:
                        resize_ratio = resize_ratio_ballon
                elif not blkitem.blk.src_is_vertical and ref_src_lines:
                    _, src_width = blkitem.blk.normalizd_width_list(normalize=False)
                    resize_ratio_src = src_width / (sum(wl_list) + max((len(wl_list) - 1 - len(blkitem.blk.lines_array())), 0) * delimiter_len)
                    resize_ratio = max(resize_ratio_src * 1.5, 0.4)
                # Minimum scale so text fits but isn't too small; allow smaller text
                # for genuinely small bubbles, short translations, and very long sentences.
                base_min = 0.55
                char_count = sum(len(w) for w in words if w.strip())
                if region_rect is not None and len(region_rect) >= 4:
                    region_area = region_rect[2] * region_rect[3]
                    # Small bubble (by area) → permit smaller font
                    if img_area > 0 and region_area < 0.025 * img_area:
                        base_min = 0.35
                    # Very long sentence: allow even smaller font so it stays inside bubbles.
                    if len(words) >= 18 or char_count >= 80:
                        base_min = min(base_min, 0.30)
                    # Extremely long sentence: permit more shrink so it doesn't overflow tall/narrow bubbles.
                    if char_count >= 120:
                        base_min = min(base_min, 0.25)
                resize_ratio = min(max(resize_ratio, base_min), 1.0)
                area_for_cap = (original_block_area if original_block_area is not None else bounding_rect[2] * bounding_rect[3])
                if img_area > 0 and area_for_cap > 0.5 * img_area:
                    resize_ratio = min(resize_ratio, 0.5)
                # When region is much larger than text: scale UP so short text in big bubbles is readable.
                # Skip scale-up for long text so our scale-down (LAYOUT_TEXT_SCALE_LONG) can actually shrink it.
                is_long_text = (len(words) >= 12 or char_count >= LAYOUT_TEXT_SCALE_LONG_MIN_CHARS)
                if not is_long_text and region_rect is not None and len(region_rect) >= 4 and text_area > 0:
                    region_area = region_rect[2] * region_rect[3]
                    if region_area > 2.5 * text_area:
                        scale_up = np.sqrt(LAYOUT_SCALE_UP_FILL * region_area / text_area)
                        scale_up_max = LAYOUT_SCALE_UP_MAX * (1.15 if getattr(pcfg.module, "optimize_line_breaks", False) else 1.0)
                        scale_up = min(scale_up, scale_up_max)
                        resize_ratio = max(resize_ratio, min(scale_up, scale_up_max))

        if resize_ratio != 1:
            new_font_size = max(1.0, blk_font.pointSizeF() * resize_ratio)
            new_font_size = np.clip(new_font_size, layout_min_pt, layout_max_pt)
            blk_font.setPointSizeF(new_font_size)
            wl_list = (np.array(wl_list, np.float64) * resize_ratio).astype(np.int32).tolist()
            line_height = int(line_height * resize_ratio)
            text_w = int(text_w * resize_ratio)
            delimiter_len = int(delimiter_len * resize_ratio)

        max_central_width = np.inf
        if fmt.alignment == 1:
            if len(blkitem.blk) > 0:
                centroid = blkitem.blk.center().astype(np.int64).tolist()
                centroid[0] -= mask_xyxy[0]
                centroid[1] -= mask_xyxy[1]
            else:
                centroid = [bounding_rect[2] // 2, bounding_rect[3] // 2]
        else:
            max_central_width = np.inf
            centroid = [0, 0]
            abs_centroid = [bounding_rect[0], bounding_rect[1]]
            if len(blkitem.blk) > 0:
                blkitem.blk.lines[0]
                abs_centroid = blkitem.blk.lines[0][0]
                centroid[0] = int(abs_centroid[0] - mask_xyxy[0])
                centroid[1] = int(abs_centroid[1] - mask_xyxy[1])
        # Limit line width so layout wraps instead of one long line (avoids text overflowing bubble).
        if region_rect is not None and len(region_rect) >= 4 and region_rect[2] > 0:
            width_frac = _auto_line_width_fraction(text, region_rect[2], region_rect[3] if len(region_rect) >= 4 else bounding_rect[3], balloon_shape)
            max_central_width = float(region_rect[2] * width_frac)
        elif bounding_rect[2] > 0:
            # No bubble region: cap line width to bounding rect so text wraps; use stricter fraction so more lines fit (e.g. free-standing notices)
            no_bubble_frac = float(getattr(pcfg.module, "layout_max_line_width_frac_no_bubble", 0.78) or 0.78)
            max_central_width = float(bounding_rect[2] * no_bubble_frac)
        # Ensure minimum line width so short text (e.g. "pluck!") is not broken into 2-char lines.
        min_lw = float(getattr(pcfg.module, "layout_min_line_width_px", 80.0) or 80.0)
        if max_central_width < min_lw:
            max_central_width = min_lw

        # Optional: hyphenation for non-CJK text (improves ugly wraps for long tokens).
        # We no longer pre-force line breaks here; instead we let layout_text() score candidates.
        # IMPORTANT: do NOT hyphenate very short texts (1–2 words), so they can stay on a single line.
        ffmt = QFontMetricsF(blk_font)
        try:
            if (
                not tgt_is_cjk
                and getattr(pcfg.module, "layout_optimal_breaks", True)
                and region_rect is not None
                and len(region_rect) >= 4
            ):
                # Only hyphenate when the overall text is reasonably long; short bubbles should stay as simple lines.
                char_count = sum(len(w) for w in words if w.strip())
                if len(words) > 2 and char_count > 8:
                    maxw_px = int(max(32, region_rect[2] * 1.05))

                    def measure_token(s: str) -> int:
                        return int(ffmt.horizontalAdvance(s))

                    expanded_words = []
                    expanded_wl = []
                    for w0 in words:
                        parts = split_long_token_with_hyphenation(
                            w0,
                            measure=measure_token,
                            max_width=maxw_px,
                            hyphenate=bool(getattr(pcfg.module, "layout_hyphenation", True)),
                        )
                        for p, pw in parts:
                            expanded_words.append(p)
                            expanded_wl.append(int(pw))
                    if expanded_words:
                        words = expanded_words
                        wl_list = expanded_wl
        except Exception:
            pass

        # True binary search for font size: find largest size in [min_pt, max_pt] such that layout fits in bubble.
        layout_font_binary_search = bool(getattr(pcfg.module, "layout_font_binary_search", False))
        if (
            layout_font_binary_search
            and layout_font_fit_bubble
            and region_rect is not None
            and len(region_rect) >= 4
            and region_rect[2] > 0
            and region_rect[3] > 0
            and not tgt_is_cjk
        ):
            rw, rh = region_rect[2], region_rect[3]
            low_pt = layout_min_pt
            high_pt = layout_max_pt
            best_pt = low_pt
            for _ in range(12):
                mid_pt = (low_pt + high_pt) * 0.5
                blk_font.setPointSizeF(mid_pt)
                ffmt_bs = QFontMetricsF(blk_font)
                wl_list_bs = [int(ffmt_bs.horizontalAdvance(w)) for w in words]
                line_height_bs = int(round(fmt.line_spacing * ffmt_bs.height()))
                delimiter_len_bs = int(ffmt_bs.horizontalAdvance(delimiter)) or 1
                words_cpy = list(words)
                wl_cpy = list(wl_list_bs)
                try:
                    txt_bs, xywh_bs, _, _ = layout_text(
                        blkitem.blk,
                        mask,
                        mask_xyxy,
                        centroid,
                        words_cpy,
                        wl_cpy,
                        delimiter,
                        delimiter_len_bs,
                        line_height_bs,
                        0,
                        max_central_width,
                        src_is_cjk=src_is_cjk,
                        tgt_is_cjk=tgt_is_cjk,
                        ref_src_lines=ref_src_lines,
                        forced_lines=None,
                        forced_wl_lines=None,
                        collision_check=bool(getattr(pcfg.module, "layout_collision_check", True)),
                        collision_min_mask_ratio=float(getattr(pcfg.module, "layout_collision_min_mask_ratio", 0.85) or 0.85),
                        collision_max_retries=int(getattr(pcfg.module, "layout_collision_max_retries", 3) or 3),
                        target_box_height=int(rh) if rh > 0 else None,
                        optimize_for_fewer_lines=bool(getattr(pcfg.module, "optimize_line_breaks", False)),
                        balloon_shape=balloon_shape,
                    )
                except Exception:
                    high_pt = mid_pt
                    continue
                if not (txt_bs and txt_bs.strip()):
                    high_pt = mid_pt
                    continue
                # Fit with 2% margin
                if xywh_bs[2] <= rw * 1.02 and xywh_bs[3] <= rh * 1.02:
                    best_pt = mid_pt
                    low_pt = mid_pt
                else:
                    high_pt = mid_pt
            blk_font.setPointSizeF(best_pt)
            ffmt = QFontMetricsF(blk_font)
            wl_list = [int(ffmt.horizontalAdvance(w)) for w in words]
            line_height = int(round(fmt.line_spacing * ffmt.height()))
            delimiter_len = int(ffmt.horizontalAdvance(delimiter)) or 1

        try:
            new_text, xywh, start_from_top, adjust_xy = layout_text(
            blkitem.blk,
            mask, 
            mask_xyxy, 
            centroid, 
            words, 
            wl_list, 
            delimiter, 
            delimiter_len, 
            line_height, 
            0, 
            max_central_width,
            src_is_cjk=src_is_cjk,
            tgt_is_cjk=tgt_is_cjk,
            ref_src_lines=ref_src_lines,
            forced_lines=None,
            forced_wl_lines=None,
            collision_check=bool(getattr(pcfg.module, "layout_collision_check", True)),
            collision_min_mask_ratio=float(getattr(pcfg.module, "layout_collision_min_mask_ratio", 0.85) or 0.85),
            collision_max_retries=int(getattr(pcfg.module, "layout_collision_max_retries", 3) or 3),
            target_box_height=int(region_rect[3]) if region_rect is not None and len(region_rect) >= 4 and region_rect[3] > 0 else None,
            optimize_for_fewer_lines=bool(getattr(pcfg.module, "optimize_line_breaks", False)),
            balloon_shape=balloon_shape,
        )
            if not (new_text and new_text.strip()):
                return
            # Safety net: merge any 1-2 word lines in the result (in case layout layer didn't merge)
            new_text = _merge_stub_lines_in_text(new_text)

            # Anti-cropping: if content would overflow bubble height when constrained, scale font down
            constrain_to_bubble = bool(getattr(pcfg.module, "layout_constrain_to_bubble", True)) and region_rect is not None and len(region_rect) >= 4 and region_rect[2] > 0 and region_rect[3] > 0
            # Additional safety: if the bubble is very small, shrink font more aggressively so text fits vertically.
            if region_rect is not None and len(region_rect) >= 4 and region_rect[3] > 0:
                n_lines = len(new_text.split('\n'))
                needed_h = n_lines * line_height * LAYOUT_HEIGHT_PADDING
                if needed_h > region_rect[3] and needed_h > 0:
                    scale_down = region_rect[3] / needed_h
                    # Allow more aggressive shrink for tiny bubbles, but keep a lower bound so text is still readable.
                    scale_down = max(scale_down, 0.35)
                    resize_ratio *= scale_down

            # 2.1 Font size post adjustment: scale to fit bubble when layout_font_fit_bubble; then clamp to min/max.
            post_resize_ratio = 1
            if layout_font_fit_bubble:
                if adaptive_fntsize:
                    downscale_constraint = 0.5
                    w = xywh[2]
                    region_w = (region_rect[2] if region_rect is not None and len(region_rect) >= 4 else w)
                    post_resize_ratio = np.clip(max(region_w / w, downscale_constraint), 0, 1)
                    resize_ratio *= post_resize_ratio
                elif region_rect is not None and len(region_rect) >= 4:
                    rw, rh = region_rect[2], region_rect[3]
                    w, h = xywh[2], xywh[3]
                    if rw > 0 and rh > 0 and (w > rw or h > rh):
                        fit_ratio = min(rw / w, rh / h) * LAYOUT_FIT_RATIO
                        fit_ratio = max(fit_ratio, LAYOUT_FIT_RATIO_MIN)
                        post_resize_ratio = fit_ratio
                        resize_ratio *= post_resize_ratio

            if post_resize_ratio != 1:
                cx, cy = xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2
                w, h = xywh[2] * post_resize_ratio, xywh[3] * post_resize_ratio
                xywh = [int(cx - w / 2), int(cy - h / 2), int(w), int(h)]

            if resize_ratio != 1:
                new_font_size = max(1.0, blkitem.font().pointSizeF() * resize_ratio)
                new_font_size = np.clip(new_font_size, layout_min_pt, layout_max_pt)
                resize_ratio = new_font_size / max(1.0, blkitem.font().pointSizeF())
                blkitem.textCursor().clearSelection()
                blkitem.setFontSize(new_font_size)
                blk_font.setPointSizeF(new_font_size)
                ffmt = QFontMetricsF(blk_font)
            # Font size we used for layout (for scale-down below); after setPlainText item font can be stale
            layout_font_pt = blk_font.pointSizeF()

            if restore_charfmts:
                char_fmts = blkitem.get_char_fmts()
            new_text = _merge_stub_lines_in_text(new_text)  # final pass so displayed text has no single-word lines
            maxw = max([ffmt.horizontalAdvance(t) for t in new_text.split('\n')]) if new_text.strip() else 0
            # Height: ensure minimum (single-line/descenders) and add bottom padding so last line doesn't sit on edge
            layout_h = max(int(xywh[3]), line_height)
            layout_h = int(layout_h * LAYOUT_HEIGHT_PADDING)
            layout_w = maxw * LAYOUT_WIDTH_STROKE_FACTOR
            # Constrain text box to detected bubble: use bubble region as exact box (no size change, keep centered)
            if constrain_to_bubble:
                # Use bubble region size and position so box stays fixed and centered in bubble
                # region_rect from extract_ballon_region is in mask/crop coords; convert position to image coords
                layout_w = region_rect[2]
                layout_h = region_rect[3]
            elif region_rect is not None and len(region_rect) >= 4 and region_rect[2] > 0 and region_rect[3] > 0:
                # Even when not fully constraining: cap height to bubble so box never extends past, and don't squeeze narrower than bubble
                layout_h = min(layout_h, region_rect[3])
                layout_w = max(layout_w, region_rect[2])
            elif region_rect is None and original_block_area is not None:
                # No bubble (free-standing text): cap box to original detected size so layout matches source area and uses more lines
                orig_w = old_br[2] if len(old_br) >= 4 else bounding_rect[2]
                orig_h = old_br[3] if len(old_br) >= 4 else bounding_rect[3]
                if orig_w > 0 and orig_h > 0:
                    layout_w = min(layout_w, max(orig_w, LAYOUT_MIN_BOX_W))
                    layout_h = min(layout_h, max(orig_h, LAYOUT_MIN_BOX_H))
            # So the bubble text box shows the same line breaks as the right panel: scale font so each of our
            # lines fits in the box width (2.1: clamp to layout_font_size_min/max). When box is constrained to
            # bubble, allow font to go below layout_min_pt so text fits instead of being cropped (e.g. small bubbles).
            if layout_w > 0 and maxw > layout_w:
                fit_scale = layout_w / maxw
                fit_scale = min(fit_scale, 1.0)
                fit_min_pt = layout_min_pt
                if constrain_to_bubble and region_rect is not None and (region_rect[2] * region_rect[3]) < (im_w * im_h * 0.04):
                    fit_min_pt = max(5.0, layout_min_pt * 0.6)  # allow smaller font so text fits in small bubble
                fit_pt = np.clip(blk_font.pointSizeF() * fit_scale, fit_min_pt, layout_max_pt)
                blk_font.setPointSizeF(fit_pt)
                blkitem.setFontSize(fit_pt)
                layout_font_pt = fit_pt
                ffmt = QFontMetricsF(blk_font)
                if blkitem.blk is not None:
                    blkitem.blk.fontformat.font_size = pt2px(fit_pt)
            layout_w = max(layout_w, LAYOUT_MIN_BOX_W)
            layout_h = max(layout_h, LAYOUT_MIN_BOX_H)
            blkitem.set_size(layout_w, layout_h, set_layout_maxsize=True)
            new_text = _merge_stub_lines_in_text(new_text)  # merge again right before set so 'stage.', 'in', 'the' never appear alone
            blkitem.setPlainText(new_text)
            if blkitem.blk is not None:
                blkitem.blk.translation = new_text  # keep block in sync so updateTranslation etc. don't overwrite with old text
            # Scale down only the text inside the box (same layout, same box size).
            # Use layout_font_pt (font size we used for layout). Applied before restore_charfmts;
            # restore_charfmts overwrites per-char formats with pre-scale size, so we re-apply after.
            scale = 1.0
            final_pt = 0.0
            if layout_font_pt > 0:
                n_lines = len(new_text.split('\n'))
                scale = _auto_layout_text_scale(new_text, n_lines, layout_w, layout_h)
                if scale != 1.0:
                    final_pt = np.clip(layout_font_pt * scale, layout_min_pt, layout_max_pt)
                    blkitem.setFontSize(final_pt)
                    doc = blkitem.document()
                    if doc is not None:
                        default_font = doc.defaultFont()
                        if default_font.pointSizeF() != final_pt:
                            default_font.setPointSizeF(final_pt)
                            doc.setDefaultFont(default_font)
                    # Persist scaled size to block so save/load and second pass use it
                    if blkitem.blk is not None:
                        blkitem.blk.fontformat.font_size = pt2px(final_pt)
            if len(self.pairwidget_list) > blkitem.idx:
                self.pairwidget_list[blkitem.idx].e_trans.setPlainText(new_text)
            if restore_charfmts:
                self.restore_charfmts(blkitem, text, new_text, char_fmts)
            # Re-apply scale-down after restore_charfmts; it overwrites char formats with pre-scale size.
            if final_pt > 0:
                blkitem.setFontSize(final_pt)
                doc = blkitem.document()
                if doc is not None:
                    default_font = doc.defaultFont()
                    if default_font.pointSizeF() != final_pt:
                        default_font.setPointSizeF(final_pt)
                        doc.setDefaultFont(default_font)
                # Persist again so block and item stay in sync after restore_charfmts
                if blkitem.blk is not None:
                    blkitem.blk.fontformat.font_size = pt2px(final_pt)
                # Ensure layout/repaint uses the new size
                blkitem.layout.reLayoutEverything()
                blkitem.repaint_background()
            blkitem._ensure_transparent_document_background()
            rw, rh = layout_w, layout_h  # keep for final clamp when constrain is on
            if constrain_to_bubble:
                # Set box to exact bubble region in image coords (mask_xyxy = crop top-left in image)
                im_x = mask_xyxy[0] + region_rect[0]
                im_y = mask_xyxy[1] + region_rect[1]
                rw, rh = region_rect[2], region_rect[3]
                # Layout judge: inset box so text stays away from bubble corners (when enabled)
                if not is_osb and bool(getattr(pcfg.module, "layout_judge_enabled", True)):
                    margin_ratio = float(getattr(pcfg.module, "layout_judge_margin_ratio", 0.06) or 0.06)
                    margin = margin_ratio * min(rw, rh)
                    if margin > 0 and rw > 2 * margin and rh > 2 * margin:
                        im_x = im_x + margin
                        im_y = im_y + margin
                        rw = max(1, int(rw - 2 * margin))
                        rh = max(1, int(rh - 2 * margin))
                # Clamp to image bounds so box never goes outside the image (handles bad region_rect or upscale mismatch)
                im_x = max(0, min(im_x, im_w - 1))
                im_y = max(0, min(im_y, im_h - 1))
                rw = max(1, min(rw, im_w - im_x))
                rh = max(1, min(rh, im_h - im_y))
                blkitem.setRect([im_x, im_y, rw, rh], padding=False, repaint=True)
                if blkitem.blk is not None:
                    blkitem.blk._bounding_rect = blkitem.absBoundingRect()
            elif region_rect is None or len(region_rect) < 4:
                blkitem.squeezeBoundingRect()
            # Clamp block to image bounds so text never draws outside the panel
            abr = blkitem.absBoundingRect(qrect=True)
            x, y, w, h = abr.x(), abr.y(), abr.width(), abr.height()
            if constrain_to_bubble:
                # Keep bubble size; only move position if needed so we don't resize the box
                w, h = rw, rh
            if w > 0 and h > 0:
                x2 = max(0, min(x, im_w - w))
                y2 = max(0, min(y, im_h - h))
                if x2 != x or y2 != y:
                    blkitem.setRect([x2, y2, w, h], repaint=True)
                # For non-constrained layout with a known bubble region, gently clamp the box so it stays inside the bubble.
                if (not constrain_to_bubble
                        and region_rect is not None and len(region_rect) >= 4
                        and region_rect[2] > 0 and region_rect[3] > 0
                        and not is_osb):
                    bubble_x = mask_xyxy[0] + region_rect[0]
                    bubble_y = mask_xyxy[1] + region_rect[1]
                    bubble_w = region_rect[2]
                    bubble_h = region_rect[3]
                    abr2 = blkitem.absBoundingRect(qrect=True)
                    bx, by, bw, bh = abr2.x(), abr2.y(), abr2.width(), abr2.height()
                    new_x, new_y = bx, by
                    if bw > 0 and bh > 0 and bw <= bubble_w * 1.05 and bh <= bubble_h * 1.05:
                        min_x = bubble_x
                        max_x = bubble_x + bubble_w - bw
                        min_y = bubble_y
                        max_y = bubble_y + bubble_h - bh
                        if max_x >= min_x:
                            new_x = min(max(bx, min_x), max_x)
                        if max_y >= min_y:
                            new_y = min(max(by, min_y), max_y)
                    if abs(new_x - bx) > 0.5 or abs(new_y - by) > 0.5:
                        blkitem.setRect([new_x, new_y, bw, bh], repaint=True)
                if constrain_to_bubble and blkitem.blk is not None:
                    blkitem.blk._bounding_rect = blkitem.absBoundingRect()
            if bool(getattr(pcfg.module, "layout_auto_final_fit_pass", True)):
                self._auto_refine_rendered_fit(blkitem, mask, mask_xyxy, region_rect, is_osb)
            # Post-layout overflow check: shrink box or scale font if text/box extends outside bubble; clamp to original box area
            if region_rect is not None and len(region_rect) >= 4 and mask_xyxy is not None:
                self._fix_overflow_after_layout(blkitem, mask, mask_xyxy, region_rect, is_osb, img, original_rect=old_br)
            return True
        except Exception:
            if is_osb:
                if getattr(pcfg.module, "osb_layout_fallbacks_enabled", True):
                    try:
                        self._layout_osb_vertical_fallback(blkitem, text, im_w, im_h, blk_font)
                        return
                    except Exception:
                        setattr(blkitem.blk, "restore_original_region", True)
                else:
                    setattr(blkitem.blk, "restore_original_region", True)
            else:
                raise
    
    def _layout_osb_vertical_fallback(self, blkitem: TextBlkItem, text: str, im_w: int, im_h: int, blk_font: QFont) -> None:
        """OSB last-chance fallback: stack lines vertically. Used when normal layout fails."""
        blk = blkitem.blk
        xyxy = getattr(blk, "xyxy", None)
        if not xyxy or len(xyxy) != 4:
            raise ValueError("Block has no valid xyxy")
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        x1, x2 = max(0, min(x1, im_w)), max(0, min(x2, im_w))
        y1, y2 = max(0, min(y1, im_h)), max(0, min(y2, im_h))
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        lines = [s.strip() for s in (text or "").split("\n") if s.strip()]
        if not lines:
            return
        ffmt = QFontMetricsF(blk_font)
        line_height = int(round(ffmt.height() * 1.2))
        max_line_w = max(ffmt.horizontalAdvance(ln) for ln in lines)
        total_h = len(lines) * line_height
        scale = 1.0
        if total_h > box_h or max_line_w > box_w:
            scale = min(box_h / max(total_h, 1), box_w / max(max_line_w, 1), 1.0)
            scale = max(scale, 0.4)
        if scale != 1.0:
            new_pt = max(LAYOUT_MIN_FONT_PT, blk_font.pointSizeF() * scale)
            blk_font.setPointSizeF(new_pt)
            ffmt = QFontMetricsF(blk_font)
            line_height = int(round(ffmt.height() * 1.2))
            max_line_w = max(ffmt.horizontalAdvance(ln) for ln in lines)
            total_h = len(lines) * line_height
        new_text = "\n".join(lines)
        layout_w = int(max_line_w * LAYOUT_WIDTH_STROKE_FACTOR)
        layout_h = int(total_h * LAYOUT_HEIGHT_PADDING)
        blkitem.set_size(layout_w, layout_h, set_layout_maxsize=True)
        blkitem.setPlainText(new_text)
        blkitem._ensure_transparent_document_background()
        if len(self.pairwidget_list) > blkitem.idx:
            self.pairwidget_list[blkitem.idx].e_trans.setPlainText(new_text)
        blkitem.squeezeBoundingRect()
    
    def restore_charfmts(self, blkitem: TextBlkItem, text: str, new_text: str, char_fmts: List[QTextCharFormat]):
        cursor = blkitem.textCursor()
        cpos = 0
        num_text = len(new_text)
        num_fmt = len(char_fmts)
        blkitem.layout.relayout_on_changed = False
        blkitem.repaint_on_changed = False
        if num_text >= num_fmt:
            for fmt_i in range(num_fmt):
                fmt = char_fmts[fmt_i]
                ori_char = text[fmt_i].strip()
                if ori_char == '':
                    continue
                else:
                    if cursor.atEnd():   
                        break
                    matched = False
                    while cpos < num_text:
                        if new_text[cpos] == ori_char:
                            matched = True
                            break
                        cpos += 1
                    if matched:
                        cursor.clearSelection()
                        cursor.setPosition(cpos)
                        cursor.setPosition(cpos+1, QTextCursor.MoveMode.KeepAnchor)
                        cursor.setCharFormat(fmt)
                        cursor.setBlockCharFormat(fmt)
                        cpos += 1
        blkitem.repaint_on_changed = True
        blkitem.layout.relayout_on_changed = True
        blkitem.layout.reLayout()
        blkitem.repaint_background()

    def onEndCreateTextBlock(self, rect: QRectF):
        xyxy = np.array([rect.x(), rect.y(), rect.right(), rect.bottom()])
        xyxy = np.round(xyxy).astype(np.int32)
        block = TextBlock(xyxy)
        xywh = np.copy(xyxy)
        xywh[[2, 3]] -= xywh[[0, 1]]
        block.set_lines_by_xywh(xywh)
        block.src_is_vertical = self.formatpanel.global_format.vertical
        blk_item = TextBlkItem(block, len(self.textblk_item_list), set_format=False, show_rect=True)
        blk_item.set_fontformat(self.formatpanel.global_format)

        page_name = self.canvas.imgtrans_proj.current_img
        mask_backup = None
        if page_name:
            self.canvas.imgtrans_proj.pages.setdefault(page_name, []).append(block)
            if self.canvas.imgtrans_proj.img_valid and self.canvas.imgtrans_proj.img_array is not None:
                img = self.canvas.imgtrans_proj.img_array
                im_h, im_w = img.shape[:2]
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                x1, x2 = max(0, min(x1, im_w)), max(0, min(x2, im_w))
                y1, y2 = max(0, min(y1, im_h)), max(0, min(y2, im_h))
                if x2 > x1 and y2 > y1:
                    mask = self.canvas.imgtrans_proj.load_mask_by_imgname(page_name)
                    if mask is None or mask.shape[0] != im_h or mask.shape[1] != im_w:
                        mask = np.full((im_h, im_w), 255, dtype=np.uint8)
                    region_backup = mask[y1:y2, x1:x2].copy()
                    mask[y1:y2, x1:x2] = 0
                    self.canvas.imgtrans_proj.save_mask(page_name, mask)
                    mask_backup = (x1, y1, x2, y2, region_backup)

        self.canvas.push_undo_command(CreateItemCommand(blk_item, self, page_name, mask_backup))

    def on_paste2selected_textitems(self):
        blkitems = self.canvas.selected_text_items()
        text = self.app_clipborad.text()

        num_blk = len(blkitems)
        if num_blk < 1:
            return
        
        if num_blk > 1:
            text_list = text.rstrip().split('\n')
            num_text = len(text_list)
            if num_text > 1:
                if num_text > num_blk:
                    text_list = text_list[:num_blk]
                elif num_text < num_blk:
                    text_list = text_list + [text_list[-1]] * (num_blk - num_text)
                text = text_list
        
        etrans = [self.pairwidget_list[blkitem.idx].e_trans for blkitem in blkitems]
        self.canvas.push_undo_command(MultiPasteCommand(text, blkitems, etrans))

    def onRotateTextBlkItem(self, item: TextBlock):
        self.canvas.push_undo_command(RotateItemCommand(item))
    
    def on_transwidget_focus_in(self, idx: int):
        if self.is_editting():
            textitm = self.editingTextItem()
            textitm.endEdit()
            self.pairwidget_list[textitm.idx].e_trans.setHoverEffect(False)
            self.textEditList.clearAllSelected()

        if idx < len(self.textblk_item_list):
            # Disabling the cache here broke the text item background rendering
            # (upstream dmMaze/BallonsTranslator@6d87898).
            blk_item = self.textblk_item_list[idx]
            self.canvas.gv.ensureVisible(blk_item)
            self.txtblkShapeControl.setBlkItem(blk_item)

    def on_textedit_redo(self):
        self.canvas.redo_textedit()

    def on_textedit_undo(self):
        self.canvas.undo_textedit()

    def on_show_select_menu(self, pos: QPoint, selected_text: str):
        if pcfg.textselect_mini_menu:
            if not selected_text:
                if self.selectext_minimenu.isVisible():
                    self.selectext_minimenu.hide()
            else:
                self.selectext_minimenu.show()
                self.selectext_minimenu.move(self.mainwindow.mapFromGlobal(pos))
                self.selectext_minimenu.selected_text = selected_text

    def on_block_current_editor(self, block: bool):
        w: SourceTextEdit = self.app.focusWidget()
        if isinstance(w, SourceTextEdit) or isinstance(w, TextBlkItem):
            w.block_all_input = block

    def on_pairw_focusout(self, idx: int):
        if self.selectext_minimenu.isVisible():
            self.selectext_minimenu.hide()
        sender = self.sender()
        if isinstance(sender, TransTextEdit) and idx < len(self.textblk_item_list):
            blk_item = self.textblk_item_list[idx]
            blk_item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

    def on_push_textitem_undostack(self, num_steps: int, is_formatting: bool):
        blkitem: TextBlkItem = self.sender()
        e_trans = self.pairwidget_list[blkitem.idx].e_trans if not is_formatting else None
        self.canvas.push_undo_command(TextItemEditCommand(blkitem, e_trans, num_steps, self.textpanel.formatpanel), update_pushed_step=is_formatting)

    def on_push_edit_stack(self, num_steps: int):
        edit: Union[TransTextEdit, SourceTextEdit] = self.sender()
        is_trans = type(edit) == TransTextEdit
        blkitem = self.textblk_item_list[edit.idx] if is_trans else None
        self.canvas.push_undo_command(TextEditCommand(edit, num_steps, blkitem), update_pushed_step=not is_trans)

    def on_propagate_textitem_edit(self, pos: int, added_text: str, joint_previous: bool):
        blk_item: TextBlkItem = self.sender()
        edit = self.pairwidget_list[blk_item.idx].e_trans
        propagate_user_edit(blk_item, edit, pos, added_text, joint_previous)
        self.canvas.push_text_command(command=None, update_pushed_step=True)

    def on_propagate_transwidget_edit(self, pos: int, added_text: str, joint_previous: bool):
        edit: TransTextEdit = self.sender()
        blk_item = self.textblk_item_list[edit.idx]
        if blk_item.isEditing():
            blk_item.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        propagate_user_edit(edit, blk_item, pos, added_text, joint_previous)
        self.canvas.push_text_command(command=None, update_pushed_step=True)

    def apply_fontformat(self, fontformat: FontFormat):
        selected_blks = self.canvas.selected_text_items()
        trans_widget_list = []
        for blk in selected_blks:
            trans_widget_list.append(self.pairwidget_list[blk.idx].e_trans)
        if len(selected_blks) > 0:
            def layout_after_apply(items):
                for blk in items:
                    if not getattr(blk.fontformat, 'vertical', False):
                        self.layout_textblk(blk)
                        self.layout_textblk(blk)
                        # For very short text (<=3 chars), scale font to fit box so full text is visible (e.g. "ah" not cut to "a").
                        text_len = len(blk.toPlainText().strip())
                        if text_len > 0 and text_len <= 3:
                            self._scale_font_to_fit_box(blk)
                    if blk.idx < len(self.pairwidget_list):
                        self.pairwidget_list[blk.idx].e_trans.setPlainText(_merge_stub_lines_in_text(blk.toPlainText()))
            layout_after = layout_after_apply if getattr(pcfg, 'let_autolayout_flag', True) else None
            self.canvas.push_undo_command(ApplyFontformatCommand(selected_blks, trans_widget_list, fontformat, layout_after=layout_after))
            if self.formatpanel.global_mode():
                if id(self.formatpanel.active_text_style_format()) != id(fontformat):
                    self.formatpanel.deactivate_style_label()
                self.formatpanel.on_active_textstyle_label_changed()
            else:
                self.formatpanel.set_active_format(fontformat)

    def apply_global_format_to_all_blocks(self):
        """Apply current global font format to every text block on this page."""
        if len(self.textblk_item_list) == 0:
            return
        all_blks = list(self.textblk_item_list)
        trans_widget_list = [self.pairwidget_list[blk.idx].e_trans for blk in all_blks]

        def layout_after_apply(items):
            for blk in items:
                if not getattr(blk.fontformat, 'vertical', False):
                    self.layout_textblk(blk)
                    self.layout_textblk(blk)
                    text_len = len(blk.toPlainText().strip())
                    if text_len > 0 and text_len <= 3:
                        self._scale_font_to_fit_box(blk)
                if blk.idx < len(self.pairwidget_list):
                    self.pairwidget_list[blk.idx].e_trans.setPlainText(_merge_stub_lines_in_text(blk.toPlainText()))
        layout_after = layout_after_apply if getattr(pcfg, 'let_autolayout_flag', True) else None
        self.canvas.push_undo_command(ApplyFontformatCommand(all_blks, trans_widget_list, self.formatpanel.global_format, layout_after=layout_after))
        self.formatpanel.on_active_textstyle_label_changed()

    def on_transwidget_selection_changed(self):
        selitems = self.canvas.selected_text_items()
        selset = {pw.idx: pw for pw in self.textEditList.checked_list}
        self.canvas.block_selection_signal = True
        for blkitem in selitems:
            if blkitem.idx not in selset:
                blkitem.setSelected(False)
            else:
                selset.pop(blkitem.idx)
        for idx in selset:
            self.textblk_item_list[idx].setSelected(True)
        self.canvas.block_selection_signal = False

    def on_textedit_list_focusout(self):
        fw = self.app.focusWidget()
        focusing_edit = isinstance(fw, (SourceTextEdit, TransTextEdit))
        if fw == self.canvas.gv or focusing_edit:
            self.textEditList.clearDrag()
        if focusing_edit:
            self.textEditList.clearAllSelected()

    def on_rearrange_blks(self, mv_map: Tuple[np.ndarray]):
        self.canvas.push_undo_command(RearrangeBlksCommand(mv_map, self))

    def updateTextBlkItemIdx(self, sel_ids: set = None):
        for ii, blk_item in enumerate(self.textblk_item_list):
            if sel_ids is not None and ii not in sel_ids:
                continue
            blk_item.idx = ii
            self.pairwidget_list[ii].updateIndex(ii)
        cl = self.textEditList.checked_list
        if len(cl) != 0:
            cl.sort(key=lambda x: x.idx)

    def updateTextBlkList(self):
        cbl = self.imgtrans_proj.current_block_list()
        if cbl is None:
            return
        cbl.clear()
        for blk_item, trans_pair in zip(self.textblk_item_list, self.pairwidget_list):
            if not blk_item.document().isEmpty():
                blk_item.blk.rich_text = blk_item.toHtml()
                blk_item.blk.translation = _merge_stub_lines_in_text(blk_item.toPlainText())
            else:
                blk_item.blk.rich_text = ''
                blk_item.blk.translation = ''
            blk_item.blk.text = [trans_pair.e_source.toPlainText()]
            blk_item.blk._bounding_rect = blk_item.absBoundingRect()
            blk_item.updateBlkFormat()
            cbl.append(blk_item.blk)

    def updateTranslation(self):
        for blk_item, transwidget in zip(self.textblk_item_list, self.pairwidget_list):
            raw = blk_item.blk.translation or ""
            merged = _merge_stub_lines_in_text(raw)  # avoid single-word lines when syncing from block
            if blk_item.blk is not None:
                blk_item.blk.translation = merged
            transwidget.e_trans.setPlainText(merged)
            blk_item.setPlainText(merged)
        self.canvas.clear_text_stack()

    def showTextblkItemRect(self, draw_rect: bool):
        for blk_item in self.textblk_item_list:
            blk_item.draw_rect = draw_rect
            blk_item.update()

    def set_blkitems_selection(self, selected: bool, blk_items: List[TextBlkItem] = None):
        self.canvas.block_selection_signal = True
        if blk_items is None:
            blk_items = self.textblk_item_list
        for blk_item in blk_items:
            blk_item.setSelected(selected)
        self.canvas.block_selection_signal = False
        self.on_incanvas_selection_changed()

    def on_ensure_textitem_svisible(self):
        edit: Union[TransTextEdit, SourceTextEdit] = self.sender()
        self.changeHoveringWidget(edit)
        self.canvas.gv.ensureVisible(self.textblk_item_list[edit.idx])
        self.txtblkShapeControl.setBlkItem(self.textblk_item_list[edit.idx])

    def on_page_replace_one(self):
        self.canvas.push_undo_command(PageReplaceOneCommand(self.canvas.search_widget))

    def on_page_replace_all(self):
        self.canvas.push_undo_command(PageReplaceAllCommand(self.canvas.search_widget))

def get_text_size(fm: QFontMetricsF, text: str) -> Tuple[int, int]:
    brt = fm.tightBoundingRect(text)
    br = fm.boundingRect(text)
    return int(np.ceil(fm.horizontalAdvance(text))), int(np.ceil(brt.height()))
    
def get_words_length_list(fm: QFontMetricsF, words: List[str]) -> List[int]:
    return [int(np.ceil(fm.horizontalAdvance(word))) for word in words]
