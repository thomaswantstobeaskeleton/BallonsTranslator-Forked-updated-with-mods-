import numpy as np
import cv2
from typing import Dict, List
from collections import OrderedDict
import sys
import os
import tempfile

from utils.registry import Registry
from utils.textblock_mask import extract_ballon_mask, classify_bubble_colored
from utils.imgproc_utils import enlarge_window, rotate_polygons, xywh2xyxypoly
from utils.config import pcfg
from utils.io_utils import imread, imwrite

from ..base import BaseModule, DEFAULT_DEVICE, soft_empty_cache, DEVICE_SELECTOR, GPUINTENSIVE_SET, TORCH_DTYPE_MAP, BF16_SUPPORTED
from ..textdetector import TextBlock

INPAINTERS = Registry('inpainters')
register_inpainter = INPAINTERS.register_module


def _feather_weight_2d(h: int, w: int, feather_px: int) -> np.ndarray:
    """Weight mask for blending: 1 in center, smooth falloff to 0 at edges. Reduces visible paste seams."""
    if feather_px <= 0 or h < 3 or w < 3:
        return np.ones((h, w), dtype=np.float32)
    fy = np.linspace(0, 1, h, dtype=np.float32)
    fx = np.linspace(0, 1, w, dtype=np.float32)
    # Raised cosine: 1 in center, 0 at edges over feather_px
    def fade(edge0: int, edge1: int, n: int) -> np.ndarray:
        out = np.ones(n, dtype=np.float32)
        if edge0 > 0:
            t = np.linspace(1, 0, edge0, dtype=np.float32)
            out[:edge0] = 0.5 * (1 + np.cos(np.pi * t))
        if edge1 < n:
            t = np.linspace(0, 1, n - edge1, dtype=np.float32)
            out[edge1:] = 0.5 * (1 + np.cos(np.pi * t))
        return out
    wy = fade(feather_px, h - feather_px, h)
    wx = fade(feather_px, w - feather_px, w)
    w2d = wy[:, np.newaxis] * wx[np.newaxis, :]
    return w2d


def _resample_inpainted_brightness(result_crop: np.ndarray, text_mask: np.ndarray, ballon_mask: np.ndarray) -> None:
    """
    Section 17: Re-sample brightness of the inpainted region to match surrounding balloon
    for better text contrast. Modifies result_crop in-place.
    """
    if result_crop is None or result_crop.size == 0 or text_mask is None or ballon_mask is None:
        return
    if result_crop.shape[:2] != text_mask.shape or text_mask.shape != ballon_mask.shape:
        return
    if result_crop.ndim != 3 or result_crop.shape[2] < 3:
        return
    interior = (ballon_mask > 127) & (text_mask <= 127)
    inpainted = (text_mask > 127) & (ballon_mask > 127)
    n_surround = int(np.sum(interior))
    n_inpaint = int(np.sum(inpainted))
    if n_surround < 16 or n_inpaint < 16:
        return
    rgb = result_crop[:, :, :3].astype(np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    target_lum = float(np.median(gray[interior]))
    mean_inpaint = float(np.mean(gray[inpainted]))
    if mean_inpaint <= 0:
        return
    scale = target_lum / mean_inpaint
    scale = max(0.5, min(2.0, scale))
    for c in range(3):
        result_crop[:, :, c] = np.where(
            inpainted,
            np.clip(rgb[:, :, c] * scale, 0, 255).astype(np.uint8),
            result_crop[:, :, c],
        )


def expand_block_inpaint_mask_vertical(msk: np.ndarray) -> np.ndarray:
    """
    Widen the per-block inpaint mask (after polygon fill) mostly in Y and slightly downward.

    Detectors (e.g. CTD) often fit quads tightly; anti-aliased edges, outlines, and descenders
    sit outside the polygon. Per-block inpainting ignores a separately dilated full-frame mask,
    so we must expand each crop mask here or the bottom of glyphs stays un-inpainted / looks
    corrupted after neural fill.
    """
    if msk is None or msk.size == 0:
        return msk
    m = (msk > 127).astype(np.uint8)
    if not np.any(m):
        return msk
    ch, cw = m.shape[:2]
    k0 = max(2, min(8, min(ch, cw) // 20))
    v_h = max(5, min(33, k0 * 2 + 9))
    v_w = max(3, min(25, k0 * 2 + 5))
    if v_h % 2 == 0:
        v_h += 1
    if v_w % 2 == 0:
        v_w += 1
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (v_w, v_h))
    m2 = cv2.dilate(m, kv, iterations=1)
    shift = max(1, min(6, ch // 45))
    m_down = np.zeros_like(m2)
    m_down[shift:, :] = m2[:-shift, :]
    m2 = np.maximum(m2, m_down)
    return ((m2 > 0).astype(np.uint8) * 255).astype(np.uint8)


def _clip_xyxy_to_image(xyxy, im_w: int, im_h: int):
    """Return [x1, y1, x2, y2] as integers clipped to image bounds; None if degenerate.
    Use for per-block inpainting so dual-detection merged blocks (possibly float or OOB) are safe."""
    if xyxy is None or len(xyxy) < 4:
        return None
    try:
        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    except (TypeError, ValueError):
        return None
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1 = max(0, min(int(round(x1)), im_w))
    x2 = max(0, min(int(round(x2)), im_w))
    y1 = max(0, min(int(round(y1)), im_h))
    y2 = max(0, min(int(round(y2)), im_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _apply_block_text_mask_to_mask(mask: np.ndarray, blk, im_w: int, im_h: int) -> None:
    """Punch holes in mask where blk.text_mask is 0 (e.g. text eraser). Modifies mask in place."""
    text_mask = getattr(blk, "text_mask", None)
    if text_mask is None or text_mask.size == 0:
        return
    xyxy = _clip_xyxy_to_image(getattr(blk, "xyxy", None), im_w, im_h)
    if xyxy is None:
        return
    x1, y1, x2, y2 = xyxy
    th, tw = text_mask.shape[:2]
    bh, bw = y2 - y1, x2 - x1
    if bh <= 0 or bw <= 0:
        return
    if (th, tw) != (bh, bw):
        text_mask = cv2.resize(text_mask, (bw, bh), interpolation=cv2.INTER_NEAREST)
    mask[y1:y2, x1:x2] = np.where(text_mask > 127, mask[y1:y2, x1:x2], 0).astype(np.uint8)


def _block_mask_polygon(blk, im_w: int, im_h: int):
    """Return polygon for block as (n, 2) int32 for fillPoly; None if degenerate.
    Block coordinates must be in page space (same as image size im_w x im_h).
    Uses blk.lines[0] when valid (e.g. HF object det, CTD), else xyxy bbox.
    Clips all points to [0, im_w-1] x [0, im_h-1] for cv2.fillPoly."""
    xyxy = _clip_xyxy_to_image(getattr(blk, 'xyxy', None), im_w, im_h)
    if xyxy is None:
        return None
    x1, y1, x2, y2 = xyxy
    # fillPoly expects indices in range; clamp so we never exceed (im_w-1, im_h-1)
    x2 = min(x2, im_w - 1)
    y2 = min(y2, im_h - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    lines = getattr(blk, 'lines', None)
    if lines and len(lines) > 0:
        try:
            pts = np.array(lines[0], dtype=np.int32)
            if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
                pts = None
            else:
                pts = np.clip(pts, [0, 0], [im_w - 1, im_h - 1])
        except (TypeError, ValueError, IndexError):
            pts = None
    else:
        pts = None
    if pts is None:
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    return pts


def _block_mask_polygons(blk, im_w: int, im_h: int):
    """Return list of polygons for block, each (n, 2) int32 for fillPoly.
    Uses ALL blk.lines (like Paddle/CTD mask build); HF has one line per block.
    Ensures mask rebuild and per-block inpainting use the same shapes."""
    xyxy = _clip_xyxy_to_image(getattr(blk, 'xyxy', None), im_w, im_h)
    if xyxy is None:
        return []
    x1, y1, x2, y2 = xyxy
    default_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    lines = getattr(blk, 'lines', None)
    if not lines or len(lines) == 0:
        return [default_pts]
    out = []
    for line in lines:
        try:
            pts = np.array(line, dtype=np.int32)
            if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
                out.append(default_pts)
            else:
                pts = np.clip(pts, [0, 0], [im_w - 1, im_h - 1])
                if pts.shape[0] >= 3:
                    out.append(pts)
        except (TypeError, ValueError, IndexError):
            out.append(default_pts)
    return out if out else [default_pts]


def _rect_mask_from_blocks(textblock_list: List[TextBlock], im_w: int, im_h: int) -> np.ndarray:
    """Build conservative rectangular mask from block xyxy only."""
    m = np.zeros((im_h, im_w), dtype=np.uint8)
    for blk in textblock_list or []:
        xyxy = _clip_xyxy_to_image(getattr(blk, "xyxy", None), im_w, im_h)
        if xyxy is None:
            continue
        x1, y1, x2, y2 = xyxy
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 255
    return m


def filter_mask_by_bboxes(mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
    """
    Keep only mask pixels inside detected text block rects, with a small margin.
    Enabled by Config -> Inpainter -> "Filter mask by text boxes"
    (upstream dmMaze/BallonsTranslator#1213).
    """
    if mask is None or not textblock_list:
        return mask

    rect_mask = np.zeros_like(mask)
    for blk in textblock_list:
        x1, y1, bbox_w, bbox_h = np.array(blk.bounding_rect()).astype(np.int64)
        if bbox_w <= 0 or bbox_h <= 0:
            continue
        rect = xywh2xyxypoly(np.array([[x1, y1, bbox_w, bbox_h]]))
        if blk.angle != 0:
            rect = rotate_polygons([x1 + bbox_w / 2, y1 + bbox_h / 2], rect, -blk.angle)
        rect = rect.reshape(-1, 4, 2).astype(np.int32)
        cv2.fillPoly(rect_mask, [rect], 255)
        cv2.polylines(rect_mask, [rect], True, 255, 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    rect_mask = cv2.dilate(rect_mask, kernel)
    return cv2.bitwise_and(mask, rect_mask)


def _block_center_xyxy(xyxy) -> tuple:
    """(cx, cy) from xyxy."""
    if not xyxy or len(xyxy) < 4:
        return (0.0, 0.0)
    return ((xyxy[0] + xyxy[2]) * 0.5, (xyxy[1] + xyxy[3]) * 0.5)


def _bisector_side(px: float, py: float, ci: tuple, cj: tuple, nudge: float = 0.0) -> int:
    """Return 0 if (px,py) is on side of center i (relative to bisector between i and j), else 1. With nudge, line is shifted toward j by nudge (positive = more pixels go to j)."""
    mx = (ci[0] + cj[0]) * 0.5
    my = (ci[1] + cj[1]) * 0.5
    vx = cj[0] - ci[0]
    vy = cj[1] - ci[1]
    # (px,py) on i's side when (p - M) · (cj - ci) <= 0 (i.e. dot <= 0). Add nudge: use dot <= nudge so positive nudge gives more to j.
    dot = (px - mx) * vx + (py - my) * vy
    return 0 if dot <= nudge else 1


def _compute_bisector_nudge(
    center_i: tuple,
    center_j: tuple,
    text_boxes_xyxy: List,
    im_w: int,
    im_h: int,
) -> float:
    """
    Section 15: Nudge bisector so text bbox corners do not fall on the wrong bubble side.
    Returns a nudge value (added to dot threshold): positive = shift bisector toward j so more overlap goes to j.
    """
    if not text_boxes_xyxy:
        return 0.0
    mx = (center_i[0] + center_j[0]) * 0.5
    my = (center_i[1] + center_j[1]) * 0.5
    vx = center_j[0] - center_i[0]
    vy = center_j[1] - center_i[1]
    nudge = 0.0
    for box in text_boxes_xyxy:
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        # Which bubble owns this text box? Use center of text box and assign to nearer bubble center.
        tx = (x1 + x2) * 0.5
        ty = (y1 + y2) * 0.5
        di = (tx - center_i[0]) ** 2 + (ty - center_i[1]) ** 2
        dj = (tx - center_j[0]) ** 2 + (ty - center_j[1]) ** 2
        owner_is_i = di <= dj
        dot_center = (tx - mx) * vx + (ty - my) * vy
        # If text center is on the "wrong" side of bisector (owner i but dot > 0, or owner j but dot < 0), nudge so it ends on correct side.
        if owner_is_i and dot_center > 0:
            nudge = max(nudge, dot_center)
        elif not owner_is_i and dot_center < 0:
            nudge = min(nudge, dot_center)
    return nudge


def build_mask_with_resolved_overlaps(
    blk_list: List,
    im_w: int,
    im_h: int,
    text_blocks_for_nudge: List = None,
) -> np.ndarray:
    """
    Section 15: Build a single mask from block polygons, resolving overlaps by bisector split
    (and optional nudge so text boxes stay on the correct bubble side). Prevents double-cleaning
    and improves per-bubble typesetting masks.
    """
    im_h, im_w = int(im_h), int(im_w)
    if im_h <= 0 or im_w <= 0 or not blk_list:
        return np.zeros((im_h, im_w), dtype=np.uint8)
    claim = np.full((im_h, im_w), -1, dtype=np.int32)
    centers = []
    for blk in blk_list:
        xyxy = getattr(blk, "xyxy", None)
        centers.append(_block_center_xyxy(xyxy) if xyxy and len(xyxy) >= 4 else (0.0, 0.0))
    text_boxes = None
    if text_blocks_for_nudge:
        text_boxes = [getattr(b, "xyxy", None) for b in text_blocks_for_nudge]
        text_boxes = [b for b in text_boxes if b is not None and len(b) >= 4]
    for idx, blk in enumerate(blk_list):
        polys = _block_mask_polygons(blk, im_w, im_h)
        if not polys:
            continue
        tmp = np.zeros((im_h, im_w), dtype=np.uint8)
        for pts in polys:
            if pts is not None and len(pts) >= 3:
                cv2.fillPoly(tmp, [np.asarray(pts, dtype=np.int32)], 255)
        # Apply per-block text_mask (e.g. from text eraser) so holes are preserved (Issue 9 incomplete masking).
        text_mask = getattr(blk, "text_mask", None)
        if text_mask is not None and text_mask.size > 0:
            xyxy = _clip_xyxy_to_image(getattr(blk, "xyxy", None), im_w, im_h)
            if xyxy is not None:
                x1, y1, x2, y2 = xyxy
                th, tw = text_mask.shape[:2]
                bh, bw = y2 - y1, x2 - x1
                if bh > 0 and bw > 0:
                    if (th, tw) != (bh, bw):
                        text_mask = cv2.resize(
                            text_mask, (bw, bh), interpolation=cv2.INTER_NEAREST
                        )
                    tmp[y1:y2, x1:x2] = np.where(
                        text_mask > 127, tmp[y1:y2, x1:x2], 0
                    ).astype(np.uint8)
        ys, xs = np.where(tmp > 0)
        ci = centers[idx]
        for y, x in zip(ys, xs):
            px, py = float(x), float(y)
            prev = int(claim[y, x])
            if prev < 0:
                claim[y, x] = idx
                continue
            cj = centers[prev]
            nudge = 0.0
            if text_boxes:
                nudge = _compute_bisector_nudge(ci, cj, text_boxes, im_w, im_h)
            side = _bisector_side(px, py, ci, cj, nudge)
            if side == 0:
                claim[y, x] = idx
            # else keep claim[y,x] = prev
    mask = (claim >= 0).astype(np.uint8) * 255
    return mask


def inpaint_handle_alpha_channel(original_alpha, mask):
    '''
    perhaps a better idea is to feed the alpha into inpainting model, but it'll double the cost  
    for now it just return the original alpha
    '''

    result_alpha = original_alpha.copy()

    # Analyze the alpha values around the original mask to determine appropriate transparency
    mask_dilated = cv2.dilate((mask > 127).astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1)
    surrounding_mask = mask_dilated - (mask > 127).astype(np.uint8)

    if np.any(surrounding_mask > 0):
        surrounding_alpha = original_alpha[surrounding_mask > 0]
        if len(surrounding_alpha) > 0:
            median_surrounding_alpha = np.median(surrounding_alpha)
            # If surrounding area is mostly transparent (median alpha < 128),
            # make inpainted areas transparent too
            if median_surrounding_alpha < 128:
                inpainted_mask = (mask > 127)
                result_alpha[inpainted_mask] = median_surrounding_alpha

    return result_alpha

class InpainterBase(BaseModule):

    inpaint_by_block = True
    check_need_inpaint = True

    _postprocess_hooks = OrderedDict()
    _preprocess_hooks = OrderedDict()

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in INPAINTERS.module_dict:
            if INPAINTERS.module_dict[key] == self.__class__:
                self.name = key
                break
    
    def memory_safe_inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        '''
        Handle cuda OOM (fallback to CPU). Tiling is disabled for all inpainters to prevent
        horizontal/vertical band and grid artifacts; use full-image mode or smaller inpaint_size if OOM.
        '''
        import torch

        def _is_oom(exc):
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
            if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
                return True
            return False

        def do_inpaint(im, msk, blk_list=None):
            return self._inpaint(im, msk, blk_list)

        try:
            return do_inpaint(img, mask, textblock_list)
        except Exception as e:
            if DEFAULT_DEVICE != 'cuda' or not _is_oom(e):
                raise e
            soft_empty_cache()
            try:
                return self._inpaint(img, mask, textblock_list)
            except Exception as ee:
                if not _is_oom(ee):
                    raise ee
                self.logger.warning(
                    'CUDA out of memory while calling %s, fall back to cpu. '
                    'If this happens often, set Inpainter device to CPU or lower inpaint_size.',
                    self.name,
                )
                self.moveToDevice('cpu')
                try:
                    inpainted = self._inpaint(img, mask, textblock_list)
                finally:
                    precision = getattr(self, 'precision', None)
                    self.moveToDevice('cuda', precision)
                return inpainted

    def inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None, check_need_inpaint: bool = False) -> np.ndarray:
        """
        Inpaint image using mask and optional per-block list (e.g. from HF object det).
        Contract: mask and textblock_list must be in page coordinates (same size as img).
        Mask is normalized to shape (im_h, im_w), dtype uint8, values 0 or 255.
        """
        if not self.all_model_loaded():
            self.load_model()

        im_h, im_w = img.shape[:2]
        # Regulate mask to actual page size: same shape as image, 2D, binary 0/255
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[0] != im_h or mask.shape[1] != im_w:
            mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_NEAREST)
        mask = np.copy(mask)
        
        # Binary mask: 255 = region to inpaint (hole); force strict 0/255 for C backends (OpenCV, PatchMatch)
        binary = (mask > 127).astype(np.uint8) * 255
        mask = np.where(binary > 0, 255, 0).astype(np.uint8)

        # Guard against malformed detector polygons causing huge random inpaint areas.
        # If incoming mask is far larger than block rectangles, fallback to xyxy-rect mask.
        if textblock_list:
            try:
                frame_area = float(im_h * im_w)
                mask_area = float(np.count_nonzero(mask > 127))
                rect_area = 0.0
                for blk in textblock_list:
                    xyxy = _clip_xyxy_to_image(getattr(blk, "xyxy", None), im_w, im_h)
                    if xyxy is None:
                        continue
                    x1, y1, x2, y2 = xyxy
                    rect_area += float(max(0, x2 - x1) * max(0, y2 - y1))
                too_large_vs_rects = rect_area > 0 and mask_area > (rect_area * 3.0)
                too_large_vs_frame = frame_area > 0 and (mask_area / frame_area) > 0.35
                if too_large_vs_rects and too_large_vs_frame:
                    self.logger.debug(
                        "Inpaint mask fallback: suspicious mask area %.0f vs rect area %.0f (blocks=%d).",
                        mask_area,
                        rect_area,
                        int(len(textblock_list)),
                    )
                    mask = _rect_mask_from_blocks(textblock_list, im_w, im_h)
            except Exception:
                pass
        if getattr(pcfg.module, 'filter_mask_by_bboxes', False):
            mask = filter_mask_by_bboxes(mask, textblock_list)

        # Subclasses (e.g. LamaLarge) can set mask_dilation_iterations if they need a small margin
        dilate_iter = getattr(self, 'mask_dilation_iterations', 0)
        if dilate_iter > 0:
            k = getattr(self, 'mask_dilation_kernel_size', 2)
            k = max(1, min(5, int(k)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        mask = np.ascontiguousarray(mask)
        
        # Warn if mask is suspiciously large (often causes dark blobs with Lama; try opencv-telea or patchmatch, or re-run detection)
        mask_ratio = np.sum(mask > 127) / (im_h * im_w)
        if mask_ratio > 0.5:
            self.logger.warning(
                f'Inpaint mask covers {mask_ratio*100:.0f}% of the image. If you see dark blobs, try Config → Inpainting → opencv-telea or patchmatch, '
                'or reduce detector "mask dilate size" and re-run detection.'
            )
        
        # Handle RGBA images by preserving alpha channel
        original_alpha = None
        if len(img.shape) == 3 and img.shape[2] == 4:
            original_alpha = img[:, :, 3:4]  # Keep alpha channel
            img_rgb = img[:, :, :3]  # Use only RGB for inpainting
        else:
            img_rgb = img
        if mask_ratio <= 0:
            self.logger.warning(
                'Inpainting skipped: mask is empty (no region to inpaint). '
                'If you use initial upscale, ensure detection was run at least once so block coordinates match the image size.'
            )
            return np.concatenate([img_rgb, original_alpha], axis=2) if original_alpha is not None else np.ascontiguousarray(img_rgb.copy())
        
        if not self.inpaint_by_block or textblock_list is None:
            if check_need_inpaint:
                ballon_msk, non_text_msk = extract_ballon_mask(img_rgb, mask)
                if ballon_msk is not None and non_text_msk is not None:
                    non_text_region = np.where(non_text_msk > 0)
                    non_text_px = img_rgb[non_text_region]
                    average_bg_color = np.median(non_text_px, axis=0).astype(np.uint8)
                    std_rgb = np.std(non_text_px - average_bg_color, axis=0)
                    std_max = np.max(std_rgb)
                    inpaint_thresh = 7 if np.std(std_rgb) > 1 else 10
                    ballon_area = np.sum(ballon_msk > 0)
                    min_ballon_area_for_median = 40000
                    if std_max < inpaint_thresh and ballon_area >= min_ballon_area_for_median:
                        is_colored = (
                            classify_bubble_colored(img_rgb, ballon_msk, mask, min_interior_pixels=64)
                            if getattr(pcfg.module, 'colored_bubble_handling', True) else False
                        )
                        if not is_colored:
                            result_rgb = img_rgb.copy()
                            if np.all(average_bg_color >= 220):
                                average_bg_color = np.array([255, 255, 255], dtype=np.uint8)
                            result_rgb[np.where(ballon_msk > 0)] = average_bg_color
                            if original_alpha is not None:
                                return np.concatenate([result_rgb, original_alpha], axis=2)
                            return result_rgb
            img_rgb = np.ascontiguousarray(img_rgb)
            result_rgb = self.memory_safe_inpaint(img_rgb, mask, textblock_list)
            # Recombine with alpha if original was RGBA
            if original_alpha is not None:
                result_alpha = inpaint_handle_alpha_channel(original_alpha, mask)
                return np.concatenate([result_rgb, result_alpha], axis=2)
            return result_rgb
        else:
            im_h, im_w = img_rgb.shape[:2]
            inpainted = np.copy(img_rgb)
            
            # Preserve original mask for transparency analysis
            original_mask = mask.copy()
            
            # Optional: exclude blocks by detector label (e.g. scene text); off by default so all blocks are inpainted
            if getattr(pcfg.module, 'inpaint_exclude_labels_enabled', False):
                exclude_str = (getattr(pcfg.module, 'inpaint_exclude_labels', None) or '').strip()
                if exclude_str:
                    excluded_labels = {s.strip().lower() for s in exclude_str.split(',') if s.strip()}
                    textblock_list = [b for b in textblock_list if (getattr(b, 'label', None) or '').strip().lower() not in excluded_labels]
            
            # Crop enlargement: larger ratio = more context, helps cover small missed detections (e.g. punctuation)
            enlarge_ratio = getattr(self, 'inpaint_enlarge_ratio', 2.0)
            spill_after_blocks = int(getattr(pcfg.module, 'inpaint_spill_to_disk_after_blocks', 0) or 0)
            temp_spill_path = None
            if spill_after_blocks > 0 and len(textblock_list) > spill_after_blocks:
                try:
                    fd = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    temp_spill_path = fd.name
                    fd.close()
                except Exception:
                    temp_spill_path = None
            for idx, blk in enumerate(textblock_list):
                xyxy = _clip_xyxy_to_image(blk.xyxy, im_w, im_h)
                if xyxy is None:
                    continue
                xyxy_e = enlarge_window(xyxy, im_w, im_h, ratio=enlarge_ratio)
                # Skip blocks with degenerate crop (e.g. zero-area bbox or invalid enlarge_window result)
                crop_w = xyxy_e[2] - xyxy_e[0]
                crop_h = xyxy_e[3] - xyxy_e[1]
                if crop_w < 2 or crop_h < 2:
                    continue
                im = inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
                # Use a crop mask that contains ONLY this block. Draw ALL polygons of the block
                # (HF = 1 polygon, Paddle/CTD = multiple lines) so mask matches rebuild.
                crop_x0, crop_y0 = xyxy_e[0], xyxy_e[1]
                polygons = _block_mask_polygons(blk, im_w, im_h)
                if not polygons:
                    continue
                msk = np.zeros((crop_h, crop_w), dtype=np.uint8)
                for pts in polygons:
                    pts_crop = pts - np.array([crop_x0, crop_y0], dtype=np.int32)
                    pts_crop[:, 0] = np.clip(pts_crop[:, 0], 0, crop_w - 1)
                    pts_crop[:, 1] = np.clip(pts_crop[:, 1], 0, crop_h - 1)
                    cv2.fillPoly(msk, [pts_crop], 255)
                if bool(getattr(pcfg.module, "inpaint_block_mask_vertical_expand", True)):
                    msk = expand_block_inpaint_mask_vertical(msk)
                # Skip if this block has no visible area in crop (degenerate after clip)
                if np.sum(msk > 127) == 0:
                    continue
                need_inpaint = True
                is_colored_bubble = False
                ballon_msk = None
                # Optional OSB fast-fill: for outside text (text_free) regions with low-variance background,
                # fill the masked region with median surrounding color instead of calling heavy models.
                osb_fast_fill = bool(getattr(pcfg.module, 'enable_osb_pipeline', False)) and bool(getattr(pcfg.module, 'osb_fast_fill', False))
                blk_label = (getattr(blk, 'label', None) or '').strip().lower()
                is_osb = blk_label in {'text_free', 'osb', 'outside', 'caption', 'sfx'}
                if osb_fast_fill and is_osb:
                    try:
                        # sample a ring around the mask inside the enlarged crop
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                        dil = cv2.dilate((msk > 127).astype(np.uint8), kernel, iterations=1)
                        ring = (dil > 0) & (msk <= 127)
                        ys, xs = np.where(ring)
                        if ys.size > 64:
                            px = im[ys, xs].astype(np.float32)
                            bg = np.median(px, axis=0).astype(np.uint8)
                            std = float(np.max(np.std(px - bg.astype(np.float32), axis=0)))
                            # If background is close to solid, fill directly.
                            if std < 8.0:
                                im[np.where(msk > 127)] = bg
                                need_inpaint = False
                    except Exception:
                        pass
                if self.check_need_inpaint or check_need_inpaint:
                    ballon_msk, non_text_msk = extract_ballon_mask(im, msk)
                    if ballon_msk is not None and non_text_msk is not None:
                        non_text_region = np.where(non_text_msk > 0)
                        non_text_px = im[non_text_region]
                        average_bg_color = np.median(non_text_px, axis=0).astype(np.uint8)
                        std_rgb = np.std(non_text_px - average_bg_color, axis=0)
                        std_max = np.max(std_rgb)
                        inpaint_thresh = 7 if np.std(std_rgb) > 1 else 10
                        ballon_area = np.sum(ballon_msk > 0)
                        # Skip median fill for small balloons so they get proper inpainting
                        min_ballon_area_for_median = 40000  # ~200x200
                        if std_max < inpaint_thresh and ballon_area >= min_ballon_area_for_median:
                            # Section 17: if colored/gradient bubble, inpaint text-only instead of median-fill
                            if getattr(pcfg.module, 'colored_bubble_handling', True):
                                is_colored_bubble = classify_bubble_colored(im, ballon_msk, msk, min_interior_pixels=64)
                            else:
                                is_colored_bubble = False
                            if not is_colored_bubble:
                                need_inpaint = False
                                # Use pure white for speech bubbles when median is already near white
                                if np.all(average_bg_color >= 220):
                                    average_bg_color = np.array([255, 255, 255], dtype=np.uint8)
                                im[np.where(ballon_msk > 0)] = average_bg_color
                    # cv2.imshow('im', im)
                    # cv2.imshow('ballon', ballon_msk)
                    # cv2.imshow('non_text', non_text_msk)
                    # cv2.waitKey(0)
                
                if need_inpaint:
                    try:
                        result_crop = self.memory_safe_inpaint(im, msk)
                    except Exception:
                        # Fallback: fill with median surrounding pixels (prevents total failure on OSB-heavy pages)
                        try:
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                            dil = cv2.dilate((msk > 127).astype(np.uint8), kernel, iterations=1)
                            ring = (dil > 0) & (msk <= 127)
                            ys, xs = np.where(ring)
                            if ys.size > 32:
                                px = im[ys, xs].astype(np.float32)
                                bg = np.median(px, axis=0).astype(np.uint8)
                            else:
                                bg = np.array([255, 255, 255], dtype=np.uint8)
                            result_crop = im.copy()
                            result_crop[np.where(msk > 127)] = bg
                        except Exception:
                            raise
                    # Ensure result matches crop size (some models return different size due to stride/pad)
                    ch, cw = im.shape[:2]
                    if result_crop.shape[0] != ch or result_crop.shape[1] != cw:
                        result_crop = cv2.resize(result_crop, (cw, ch), interpolation=cv2.INTER_LINEAR)
                    # Section 17: re-sample brightness of inpainted region for better text contrast on colored bubbles
                    if is_colored_bubble and ballon_msk is not None and getattr(pcfg.module, 'colored_bubble_resample_brightness', True):
                        _resample_inpainted_brightness(result_crop, msk, ballon_msk)
                    # Feather blend at crop edges to avoid visible rectangular seams (striped/grid artifacts)
                    feather_px = min(6, ch // 4, cw // 4)
                    if feather_px > 0:
                        w = _feather_weight_2d(ch, cw, feather_px)
                        if result_crop.ndim == 3:
                            w = w[:, :, np.newaxis]
                        roi = inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
                        blended = (w * result_crop.astype(np.float32) + (1 - w) * roi.astype(np.float32)).astype(np.uint8)
                        inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]] = blended
                    else:
                        inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]] = result_crop

                # Clear this block's region in the mask (all polygons) so we don't double-process.
                # Must match how the mask was built (all lines per block).
                for pts in _block_mask_polygons(blk, im_w, im_h):
                    if pts is not None and len(pts) >= 3:
                        cv2.fillPoly(mask, [pts], 0)

                # Temp-file spill to disk (Section 7): reduce peak RAM/VRAM on long pages with many regions
                if temp_spill_path is not None and spill_after_blocks > 0 and (idx + 1) % spill_after_blocks == 0 and (idx + 1) < len(textblock_list):
                    try:
                        imwrite(temp_spill_path, inpainted, ext='.png')
                        inpainted = imread(temp_spill_path)
                    except Exception:
                        pass
            
            if temp_spill_path is not None and os.path.isfile(temp_spill_path):
                try:
                    os.remove(temp_spill_path)
                except Exception:
                    pass
            
            # Recombine with alpha if original was RGBA
            if original_alpha is not None:
                result_alpha = inpaint_handle_alpha_channel(original_alpha, original_mask)
                return np.concatenate([inpainted, result_alpha], axis=2)
            return inpainted

    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        raise NotImplementedError
    
    def moveToDevice(self, device: str, precision: str = None):
        raise not NotImplementedError


def _opencv_classic_inpaint(
    img: np.ndarray,
    mask: np.ndarray,
    radius: int,
    dilate_px: int,
    passes: int,
    cv2_flag: int,
) -> np.ndarray:
    """
    Shared path for cv2.INPAINT_TELEA / INPAINT_NS. Optional mask dilation catches outlines/halos
    that detectors miss; larger radius samples a wider neighborhood (slower but fewer gaps).
    """
    if img is None or mask is None or img.size == 0:
        return img
    m = np.asarray(mask, dtype=np.uint8)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.shape[:2] != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = ((m > 127).astype(np.uint8) * 255)
    dp = max(0, min(32, int(dilate_px)))
    if dp > 0:
        k = 2 * dp + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, ker, iterations=1)
    r = max(1, min(64, int(radius)))
    p = max(1, min(3, int(passes)))
    out = cv2.inpaint(img, m, r, cv2_flag)
    for _ in range(1, p):
        out = cv2.inpaint(out, m, max(1, r // 2), cv2_flag)
    return out


_OPENCV_INPAINT_RADIUS_OPTS = [2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 19, 23]
_OPENCV_MASK_DILATE_OPTS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12]


@register_inpainter('opencv-tela')
class OpenCVInpainter(InpainterBase):

    params = {
        'inpaint_radius': {
            'type': 'selector',
            'options': list(_OPENCV_INPAINT_RADIUS_OPTS),
            'value': 5,
            'description': (
                'cv2.inpaint neighborhood radius (px). Default was 3; 5–9 helps thick subs / soft edges '
                '(slower).'
            ),
        },
        'mask_dilate_px': {
            'type': 'selector',
            'options': list(_OPENCV_MASK_DILATE_OPTS),
            'value': 2,
            'description': (
                'Extra mask dilation before inpaint. Fills missed halos/outlines; 0 = off if mask is already generous.'
            ),
        },
        'inpaint_passes': {
            'type': 'selector',
            'options': [1, 2, 3],
            'value': 1,
            'description': (
                'Extra cv2.inpaint passes on the same mask can reduce leftover fringe (slower).'
            ),
        },
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)

    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        r = int(self.get_param_value('inpaint_radius'))
        d = int(self.get_param_value('mask_dilate_px'))
        p = int(self.get_param_value('inpaint_passes'))
        return _opencv_classic_inpaint(img, mask, r, d, p, cv2.INPAINT_NS)

    def is_computational_intensive(self) -> bool:
        return True

    def is_cpu_intensive(self) -> bool:
        return True


@register_inpainter('opencv-telea')
class OpenCVTeleaInpainter(InpainterBase):
    """OpenCV Telea inpainting (#126). Fast, CPU-only, no model download."""

    params = {
        'inpaint_radius': {
            'type': 'selector',
            'options': list(_OPENCV_INPAINT_RADIUS_OPTS),
            'value': 5,
            'description': (
                'cv2.inpaint neighborhood radius (px). Default was 3; 5–9 helps thick subs / soft edges '
                '(slower).'
            ),
        },
        'mask_dilate_px': {
            'type': 'selector',
            'options': list(_OPENCV_MASK_DILATE_OPTS),
            'value': 2,
            'description': (
                'Extra mask dilation before inpaint. Fills missed halos/outlines; 0 = off if mask is already generous.'
            ),
        },
        'inpaint_passes': {
            'type': 'selector',
            'options': [1, 2, 3],
            'value': 1,
            'description': (
                'Extra cv2.inpaint passes on the same mask can reduce leftover fringe (slower).'
            ),
        },
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)

    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        r = int(self.get_param_value('inpaint_radius'))
        d = int(self.get_param_value('mask_dilate_px'))
        p = int(self.get_param_value('inpaint_passes'))
        return _opencv_classic_inpaint(img, mask, r, d, p, cv2.INPAINT_TELEA)

    def is_computational_intensive(self) -> bool:
        return True

    def is_cpu_intensive(self) -> bool:
        return True
    


@register_inpainter('patchmatch')
class PatchmatchInpainter(InpainterBase):

    if sys.platform == 'darwin':
        download_file_list = [{
                'url': 'https://github.com/dmMaze/PyPatchMatchInpaint/releases/download/v1.0/macos_arm64_patchmatch_libs.7z',
                'sha256_pre_calculated': ['843704ab096d3afd8709abe2a2c525ce3a836bb0a629ed1ee9b8f5cee9938310', '849ca84759385d410c9587d69690e668822a3fc376ce2219e583e7e0be5b5e9a'],
                'files': ['macos_libopencv_world.4.8.0.dylib', 'macos_libpatchmatch_inpaint.dylib'],
                'save_dir': 'data/libs',
                'archived_files': 'macos_patchmatch_libs.7z',
                'archive_sha256_pre_calculated': '9f332c888be0f160dbe9f6d6887eb698a302e62f4c102a0f24359c540d5858ea'
        }]
    elif sys.platform == 'win32':
        download_file_list = [{
                'url': 'https://github.com/dmMaze/PyPatchMatchInpaint/releases/download/v1.0/windows_patchmatch_libs.7z',
                'sha256_pre_calculated': ['3b7619caa29dc3352b939de4e9981217a9585a13a756e1101a50c90c100acd8d', '0ba60cfe664c97629daa7e4d05c0888ebfe3edcb3feaf1ed5a14544079c6d7af'],
                'files': ['opencv_world455.dll', 'patchmatch_inpaint.dll'],
                'save_dir': 'data/libs',
                'archived_files': 'windows_patchmatch_libs.7z',
                'archive_sha256_pre_calculated': 'c991ff61f7cb3efaf8e75d957e62d56ba646083bc25535f913ac65775c16ca65'
        }]

    def __init__(self, **params) -> None:
        super().__init__(**params)
        from . import patch_match
        # patch_size 3 is too small and can cause dark/blocky artifacts; use 7 for better quality (paper uses ~8-15)
        self.inpaint_method = lambda img, mask, *args, **kwargs: patch_match.inpaint(img, mask, patch_size=7)
    
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        return self.inpaint_method(img, mask)

    def is_computational_intensive(self) -> bool:
        return True
    
    def is_cpu_intensive(self) -> bool:
        return True


import torch
from utils.imgproc_utils import resize_keepasp
from .aot import AOTGenerator, load_aot_model


def _maybe_torch_compile_inpaint_model(model, device: str, logger):
    """
    Optionally wrap PyTorch inpaint models with torch.compile (module-level inpaint_torch_compile).
    CUDA only; dynamic=True for variable per-block crop sizes. First inference incurs compile cost.
    """
    if model is None:
        return None
    if not bool(getattr(pcfg.module, "inpaint_torch_compile", False)):
        return model
    dev = str(device or "").lower()
    if dev != "cuda":
        logger.info("inpaint_torch_compile: skipped (device is not cuda).")
        return model
    try:
        compiled = torch.compile(model, dynamic=True, mode="default")
        logger.info("inpaint_torch_compile: torch.compile enabled for this inpainter (CUDA).")
        return compiled
    except Exception as e:
        logger.warning("inpaint_torch_compile failed (%s); using eager model.", e)
        return model


@register_inpainter('aot')
class AOTInpainter(InpainterBase):

    params = {
        'inpaint_size': {
            'type': 'selector',
            'options': [
                1024, 
                2048
            ], 
            'value': 2048
        }, 
        'device': DEVICE_SELECTOR(),
        'description': 'manga-image-translator inpainter'
    }

    device = DEFAULT_DEVICE
    inpaint_size = 2048
    model: AOTGenerator = None
    _load_model_keys = {'model'}

    download_file_list = [{
            'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/aot_inpainter.ckpt',
            'sha256_pre_calculated': '878d541c68648969bc1b042a6e997f3a58e49b6c07c5636ad55130736977149f',
            'files': 'data/models/aot_inpainter.ckpt',
    }]

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.inpaint_size = int(self.params['inpaint_size']['value'])
        self.model: AOTGenerator = None
        
    def _load_model(self):
        AOTMODEL_PATH = 'data/models/aot_inpainter.ckpt'
        self.model = load_aot_model(AOTMODEL_PATH, self.device)
        self.model = _maybe_torch_compile_inpaint_model(self.model, self.device, self.logger)

    def moveToDevice(self, device: str, precision: str = None):
        self.model.to(device)
        self.device = device

    def inpaint_preprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:

        img_original = np.copy(img)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        new_shape = self.inpaint_size if max(img.shape[0: 2]) > self.inpaint_size else None

        img = resize_keepasp(img, new_shape, stride=None)
        mask = resize_keepasp(mask, new_shape, stride=None)

        im_h, im_w = img.shape[:2]
        pad_bottom = 128 - im_h if im_h < 128 else 0
        pad_right = 128 - im_w if im_w < 128 else 0
        mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 127.5 - 1.0
        mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1

        if self.device != 'cpu':
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)
        img_torch *= (1 - mask_torch)
        return img_torch, mask_torch, img_original, mask_original, pad_bottom, pad_right

    @torch.no_grad()
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:

        im_h, im_w = img.shape[:2]
        img_torch, mask_torch, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)
        img_inpainted_torch = self.model(img_torch, mask_torch)
        img_inpainted = ((img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() + 1.0) * 127.5)
        img_inpainted = (np.clip(np.round(img_inpainted), 0, 255)).astype(np.uint8)
        if pad_bottom > 0:
            img_inpainted = img_inpainted[:-pad_bottom]
        if pad_right > 0:
            img_inpainted = img_inpainted[:, :-pad_right]
        new_shape = img_inpainted.shape[:2]
        if new_shape[0] != im_h or new_shape[1] != im_w :
            img_inpainted = cv2.resize(img_inpainted, (im_w, im_h), interpolation = cv2.INTER_LINEAR)
        img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        
        return img_inpainted

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key == 'device':
            param_device = self.params['device']['value']
            if self.model is not None:
                self.model.to(param_device)
            self.device = param_device

        elif param_key == 'inpaint_size':
            self.inpaint_size = int(self.params['inpaint_size']['value'])


from .lama import LamaFourier, load_lama_mpe

@register_inpainter('lama_mpe')
class LamaInpainterMPE(InpainterBase):

    params = {
        'inpaint_size': {
            'type': 'selector',
            'options': [
                256,
                384,
                512,
                768,
                1024,
                1536,
                2048,
            ],
            'value': 2048,
            'description': (
                'Max side (px) for inpaint input; image is resized with stride 64. '
                'Smaller values use less VRAM—often enough for subtitle-sized regions.'
            ),
        },
        'inpaint_enlarge_ratio': {
            'type': 'selector',
            'options': [1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.2, 2.5],
            'value': 2.0,
            'description': (
                'Per-block crop margin ratio (1.1–2.5). Larger = more context around each block for LaMa. '
                'Same as lama_large_512px; helps punctuation/halos at block edges.'
            ),
        },
        'device': DEVICE_SELECTOR(not_supported=['privateuseone'])
    }

    download_file_list = [{
            'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/lama_mpe.ckpt',
            'sha256_pre_calculated': 'd625aa1b3e0d0408acfd6928aa84f005867aa8dbb9162480346a4e20660786cc',
            'files': 'data/models/lama_mpe.ckpt',
    }]
    _load_model_keys = {'model'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.inpaint_size = int(self.params['inpaint_size']['value'])
        self.precision = 'fp32'
        self.model: LamaFourier = None
        self._update_inpaint_enlarge_ratio()

    def _update_inpaint_enlarge_ratio(self):
        val = self.params.get('inpaint_enlarge_ratio', {}).get('value', 2.0)
        try:
            self.inpaint_enlarge_ratio = float(val) if val is not None else 2.0
        except (TypeError, ValueError):
            self.inpaint_enlarge_ratio = 2.0

    def _load_model(self):
        self.model = load_lama_mpe(r'data/models/lama_mpe.ckpt', self.device)
        self.model = _maybe_torch_compile_inpaint_model(self.model, self.device, self.logger)

    def inpaint_preprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:

        img_original = np.copy(img)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        max_side = max(img.shape[0:2])
        if max_side > self.inpaint_size:
            new_shape = self.inpaint_size
        elif max_side < 400:
            # Small bubbles: normalize to 512 max to reduce over-strong inpainting and artifacts
            new_shape = min(self.inpaint_size, 512)
        else:
            new_shape = None
        # high resolution input could produce cloudy artifacts
        img = resize_keepasp(img, new_shape, stride=64)
        mask = resize_keepasp(mask, new_shape, stride=64)

        im_h, im_w = img.shape[:2]
        longer = max(im_h, im_w)
        pad_bottom = longer - im_h if im_h < longer else 0
        pad_right = longer - im_w if im_w < longer else 0
        mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 255.0
        mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1
        rel_pos, _, direct = self.model.load_masked_position_encoding(mask_torch[0][0].numpy())
        rel_pos = torch.LongTensor(rel_pos).unsqueeze_(0)
        direct = torch.LongTensor(direct).unsqueeze_(0)

        if self.device != 'cpu':
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)
            rel_pos = rel_pos.to(self.device)
            direct = direct.to(self.device)
        img_torch *= (1 - mask_torch)
        return img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right

    @torch.no_grad()
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:

        im_h, im_w = img.shape[:2]
        img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)
        
        precision = TORCH_DTYPE_MAP[self.precision]
        if self.device in {'cuda'}:
            try:
                with torch.autocast(device_type=self.device, dtype=precision):
                    img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
            except Exception as e:
                is_oom = (
                    isinstance(e, torch.cuda.OutOfMemoryError)
                    or (isinstance(e, RuntimeError) and "out of memory" in str(e).lower())
                )
                if is_oom:
                    raise
                self.logger.error(e)
                self.logger.error(f'{precision} inference is not supported for this device, use fp32 instead.')
                img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
        else:
            img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)

        img_inpainted = (img_inpainted_torch.to(device='cpu', dtype=torch.float32).squeeze_(0).permute(1, 2, 0).numpy() * 255)
        img_inpainted = (np.clip(np.round(img_inpainted), 0, 255)).astype(np.uint8)
        if pad_bottom > 0:
            img_inpainted = img_inpainted[:-pad_bottom]
        if pad_right > 0:
            img_inpainted = img_inpainted[:, :-pad_right]
        new_shape = img_inpainted.shape[:2]
        if new_shape[0] != im_h or new_shape[1] != im_w :
            img_inpainted = cv2.resize(img_inpainted, (im_w, im_h), interpolation = cv2.INTER_LINEAR)
        img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        
        return img_inpainted

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key == 'device':
            param_device = self.params['device']['value']
            if self.model is not None:
                self.model.to(param_device)
            self.device = param_device

        elif param_key == 'inpaint_size':
            self.inpaint_size = int(self.params['inpaint_size']['value'])

        elif param_key == 'inpaint_enlarge_ratio':
            self._update_inpaint_enlarge_ratio()

        elif param_key == 'precision':
            p = self.params.get('precision')
            if isinstance(p, dict) and 'value' in p:
                self.precision = p['value']

    def moveToDevice(self, device: str, precision: str = None):
        self.model.to(device)
        self.device = device
        if precision is not None:
            self.precision = precision

@register_inpainter('lama_large_512px')
class LamaLarge(LamaInpainterMPE):

    mask_dilation_iterations = 2
    mask_dilation_kernel_size = 2  # 2×2 = gentler expansion per iteration; 3×3 = stronger
    # Always run LaMa; skip median fill to avoid "weird box of a certain color" in speech bubbles
    check_need_inpaint = False

    params = {
        'inpaint_size': {
            'type': 'selector',
            'options': [
                256,
                384,
                512,
                768,
                1024,
                1536,
                1920,
                2048,
            ],
            'value': 1024,
            'description': (
                'Max side (px) for each bubble crop before feeding to model (stride 64). '
                'Match detect_max_side (e.g. 1920) for high-res pages. '
                '256–512 use less VRAM—often enough for subtitle-sized regions.'
            ),
        },
        'mask_dilation': {
            'type': 'selector',
            'options': [0, 1, 2, 3, 4, 5],
            'value': 2,
            'description': 'Mask dilation iterations (0–5). Expands the inpainting mask to cover small missed detections (e.g. punctuation). 0 = no expansion. 2 = default, good for dots/punctuation near text. Higher = more coverage, may soften bubble edges.',
        },
        'mask_dilation_kernel': {
            'type': 'selector',
            'options': [1, 2, 3, 4, 5],
            'value': 2,
            'description': 'Dilation kernel size (1–5). 1 = minimal expansion; 2 = gentle; 5 = strongest per iteration. Use with mask_dilation iterations.',
        },
        'inpaint_enlarge_ratio': {
            'type': 'selector',
            'options': [1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.2, 2.5],
            'value': 2.0,
            'description': (
                'Crop margin ratio (1.1–2.5). Larger = more context around each block; helps cover small '
                'missed detections (e.g. punctuation) when combined with mask dilation. 2.0 = default. '
                '(Same option as lama_mpe.)'
            ),
        },
        'device': DEVICE_SELECTOR(not_supported=['privateuseone']),
        'precision': {
            'type': 'selector',
            'options': [
                'fp32',
                'bf16'
            ],
            'value': 'bf16' if BF16_SUPPORTED == 'cuda' else 'fp32'
        },
    }

    download_file_list = [{
            'url': 'https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt',
            'sha256_pre_calculated': '11d30fbb3000fb2eceae318b75d9ced9229d99ae990a7f8b3ac35c8d31f2c935',
            'files': 'data/models/lama_large_512px.ckpt',
    }]

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.precision = self.params['precision']['value']
        self._update_mask_dilation()
        self._update_mask_dilation_kernel()

    def _update_mask_dilation(self):
        self.mask_dilation_iterations = int(self.params.get('mask_dilation', {}).get('value', 2))

    def _update_mask_dilation_kernel(self):
        self.mask_dilation_kernel_size = int(self.params.get('mask_dilation_kernel', {}).get('value', 2))

    def _load_model(self):
        device = self.params['device']['value']
        if not (device and str(device).strip()):
            from ..base import DEFAULT_DEVICE
            device = DEFAULT_DEVICE
        precision = self.params['precision']['value']

        self.model = load_lama_mpe(r'data/models/lama_large_512px.ckpt', device='cpu', use_mpe=False, large_arch=True)
        self.moveToDevice(device, precision=precision)
        self.model = _maybe_torch_compile_inpaint_model(self.model, self.device, self.logger)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'mask_dilation':
            self._update_mask_dilation()
        elif param_key == 'mask_dilation_kernel':
            self._update_mask_dilation_kernel()


# LAMA_ORI: LamaFourier = None
# @register_inpainter('lama_ori')
# class LamaInpainterORI(InpainterBase):

#     params = {
#         'inpaint_size': {
#             'type': 'selector',
#             'options': [
#                 1024, 
#                 2048
#             ], 
#             'value': 2048
#         }, 
#         'device': {
#             'type': 'selector',
#             'options': [
#                 'cpu',
#                 'cuda'
#             ],
#             'value': DEFAULT_DEVICE
#         }
#     }

#     device = DEFAULT_DEVICE
#     inpaint_size = 2048

#     def setup_inpainter(self):
#         global LAMA_ORI

#         self.device = self.params['device']['value']
#         if LAMA_ORI is None:
#             self.model = LAMA_ORI = load_lama_mpe(r'data/models/lama_org.ckpt', self.device, False)
#         else:
#             self.model = LAMA_ORI
#             self.model.to(self.device)
#         self.inpaint_by_block = True if self.device == 'cuda' else False
#         self.inpaint_size = int(self.params['inpaint_size']['value'])

#     def inpaint_preprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:

#         img_original = np.copy(img)
#         mask_original = np.copy(mask)
#         mask_original[mask_original < 127] = 0
#         mask_original[mask_original >= 127] = 1
#         mask_original = mask_original[:, :, None]

#         new_shape = self.inpaint_size if max(img.shape[0: 2]) > self.inpaint_size else None
#         # high resolution input could produce cloudy artifacts
#         img = resize_keepasp(img, new_shape, stride=64)
#         mask = resize_keepasp(mask, new_shape, stride=64)

#         im_h, im_w = img.shape[:2]
#         longer = max(im_h, im_w)
#         pad_bottom = longer - im_h if im_h < longer else 0
#         pad_right = longer - im_w if im_w < longer else 0
#         mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
#         img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

#         img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 255.0
#         mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
#         mask_torch[mask_torch < 0.5] = 0
#         mask_torch[mask_torch >= 0.5] = 1
#         rel_pos, _, direct = self.model.load_masked_position_encoding(mask_torch[0][0].numpy())
#         rel_pos = torch.LongTensor(rel_pos).unsqueeze_(0)
#         direct = torch.LongTensor(direct).unsqueeze_(0)

#         if self.device == 'cuda':
#             img_torch = img_torch.cuda()
#             mask_torch = mask_torch.cuda()
#             rel_pos = rel_pos.cuda()
#             direct = direct.cuda()
#         img_torch *= (1 - mask_torch)
#         return img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right

#     @torch.no_grad()
#     def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:

#         im_h, im_w = img.shape[:2]
#         img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)
#         img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
        
#         img_inpainted = (img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#         if pad_bottom > 0:
#             img_inpainted = img_inpainted[:-pad_bottom]
#         if pad_right > 0:
#             img_inpainted = img_inpainted[:, :-pad_right]
#         new_shape = img_inpainted.shape[:2]
#         if new_shape[0] != im_h or new_shape[1] != im_w :
#             img_inpainted = cv2.resize(img_inpainted, (im_w, im_h), interpolation = cv2.INTER_LINEAR)
#         img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        
#         return img_inpainted

#     def updateParam(self, param_key: str, param_content):
#         super().updateParam(param_key, param_content)

#         if param_key == 'device':
#             param_device = self.params['device']['value']
#             self.model.to(param_device)
#             self.device = param_device
#             if param_device == 'cuda':
#                 self.inpaint_by_block = False
#             else:
#                 self.inpaint_by_block = True

#         elif param_key == 'inpaint_size':
#             self.inpaint_size = int(self.params['inpaint_size']['value'])