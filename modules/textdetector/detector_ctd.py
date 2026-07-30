import numpy as np
import cv2
import os
import torch
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEFAULT_DEVICE, DEVICE_SELECTOR, ProjImgTrans
from .box_utils import expand_blocks, shrink_blocks
from .ctd import CTDModel

CTD_ONNX_PATH = 'data/models/comictextdetector.pt.onnx'
CTD_TORCH_PATH = 'data/models/comictextdetector.pt'

def load_ctd_model(model_path, device, detect_size=1024) -> CTDModel:
    model = CTDModel(model_path, detect_size=detect_size, device=device)
    
    return model

@register_textdetectors('ctd')
class ComicTextDetector(TextDetectorBase):

    params = {
        'detect_size': {
            'type': 'selector',
            'options': [896, 1024, 1152, 1280, 1536, 1792, 2048, 2400],
            'value': 1536
        }, 
        'det_rearrange_max_batches': {
            'type': 'selector',
            'options': [1, 2, 4, 6, 8, 12, 16, 24, 32], 
            'value': 4
        },
        'device': DEVICE_SELECTOR(),
        'description': 'ComicTextDetector',
        'font size multiplier': 1.,
        'font size max': -1,
        'font size min': -1,
        'mask dilate size': 5,
        'merge font size tolerance': {
            'type': 'line_editor',
            'value': 3.0,
            'description': 'Legacy fallback for merge tolerances. If horizontal/vertical values are set, those take priority.',
        },
        'merge font size tolerance horizontal': {
            'type': 'line_editor',
            'value': 3.0,
            'description': 'Merge tolerance for horizontal lines. Higher = fewer boxes per bubble.',
        },
        'merge font size tolerance vertical': {
            'type': 'line_editor',
            'value': 3.0,
            'description': 'Merge tolerance for vertical lines. Higher = fewer boxes per bubble.',
        },
        'box score threshold': {
            'type': 'line_editor',
            'value': 0.45,
            'description': 'Min confidence (0.35–0.7). Lower = more boxes (e.g. small out-of-bubble text).',
        },
        'min box area': {
            'type': 'line_editor',
            'value': 0,
            'description': 'Drop regions smaller than this (px²). 0=off. Use 100–300 to remove tiny false boxes.',
        },
        'custom_onnx_path': {
            'type': 'line_editor',
            'value': '',
            'description': 'Optional: path to a custom ONNX model (e.g. mayocream/comic-text-detector-onnx). Leave empty to use built-in CTD ONNX. Used when device is CPU or when forcing ONNX.',
        },
        'box_padding': {
            'type': 'line_editor',
            'value': 5,
            'description': 'Pixels to add around each detected box (all sides). Reduces clipped punctuation (?, !) and character edges. Recommended 4–6.',
        },
        'box_shrink_px': {
            'type': 'line_editor',
            'value': 0,
            'description': 'Shrink each box inward by this many pixels (0=off). Use when CTD boxes are too large (e.g. 4–12).',
        },
        'det_invert': {
            'type': 'checkbox',
            'value': False,
            'description': 'Invert image before detection (manga-image-translator style). Can improve detection on light text on dark.',
        },
        'det_gamma_correct': {
            'type': 'checkbox',
            'value': False,
            'description': 'Apply gamma correction before detection (manga-image-translator style). Can improve contrast for detection.',
        },
        'det_rotate': {
            'type': 'checkbox',
            'value': False,
            'description': 'Rotate image 90° before detection, then rotate results back. Use when text is mainly vertical.',
        },
        'det_auto_rotate': {
            'type': 'checkbox',
            'value': False,
            'description': 'If majority of detected text is horizontal, re-run detection with 90° rotation (manga-image-translator style). Use when orientation is unknown.',
        },
        'det_min_image_side': {
            'type': 'line_editor',
            'value': 0,
            'description': 'Add border so smallest side is at least this (px). 0=off. 400=like manga-image-translator for small images. Reduces detection issues on tiny pages.',
        },
    }
    _load_model_keys = {'model'}
    download_file_list = [
        {
            'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/comictextdetector.pt',
            'files': 'data/models/comictextdetector.pt',
            'sha256_pre_calculated': '1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb',
        },
        {
            'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/comictextdetector.pt.onnx',
            'files': 'data/models/comictextdetector.pt.onnx',
            'sha256_pre_calculated': '1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f',
        },
    ]

    device = DEFAULT_DEVICE
    detect_size = 1024
    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model: CTDModel = None

    @property
    def device(self):
        return self.params['device']['value']
    
    @property
    def detect_size(self):
        return int(self.params['detect_size']['value'])

    def _load_model(self):
        custom_onnx = (self.params.get('custom_onnx_path') or {}).get('value', '') or ''
        custom_onnx = custom_onnx.strip() if isinstance(custom_onnx, str) else ''
        if self.device != 'cpu':
            self.model = load_ctd_model(CTD_TORCH_PATH, self.device, self.detect_size)
        else:
            onnx_path = custom_onnx if custom_onnx and os.path.isfile(custom_onnx) else CTD_ONNX_PATH
            self.model = load_ctd_model(onnx_path, self.device, self.detect_size)

    def _detect(self, img: np.ndarray, proj: ProjImgTrans) -> Tuple[np.ndarray, List[TextBlock]]:
        h_img, w_img = img.shape[:2]
        work = img.copy()
        det_rotate = bool((self.params.get('det_rotate') or {}).get('value', False))
        det_invert = bool((self.params.get('det_invert') or {}).get('value', False))
        det_gamma = bool((self.params.get('det_gamma_correct') or {}).get('value', False))
        det_auto_rotate = bool((self.params.get('det_auto_rotate') or {}).get('value', False))
        det_min_side = 0
        try:
            v = (self.params.get('det_min_image_side') or {}).get('value', 0)
            det_min_side = max(0, int(v) if v not in (None, '') else 0)
        except (TypeError, ValueError):
            pass
        # Add border for small images (manga-image-translator style)
        border_added = False
        old_h, old_w = h_img, w_img
        if det_min_side > 0 and min(h_img, w_img) < det_min_side:
            new_side = max(h_img, w_img, det_min_side)
            work = np.zeros((new_side, new_side, 3), dtype=np.uint8)
            work[:h_img, :w_img] = img.copy()
            border_added = True
            h_img, w_img = work.shape[:2]
        if det_rotate:
            work = np.rot90(work, k=-1)
        if det_invert:
            work = cv2.bitwise_not(work)
        if det_gamma:
            gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
            mid = 0.5
            mean = np.mean(gray)
            if mean > 1e-6:
                gamma = np.log(mid * 255) / np.log(mean)
                work = np.power(work.astype(np.float32) / 255.0, gamma).clip(0, 1)
                work = (work * 255).astype(np.uint8)

        legacy_merge_tol = self.get_param_value('merge font size tolerance')
        try:
            legacy_merge_tol = float(legacy_merge_tol) if legacy_merge_tol not in (None, '') else 3.0
        except (TypeError, ValueError):
            legacy_merge_tol = 3.0

        merge_tol_hor = self.get_param_value('merge font size tolerance horizontal')
        try:
            merge_tol_hor = float(merge_tol_hor) if merge_tol_hor not in (None, '') else legacy_merge_tol
        except (TypeError, ValueError):
            merge_tol_hor = legacy_merge_tol

        merge_tol_ver = self.get_param_value('merge font size tolerance vertical')
        try:
            merge_tol_ver = float(merge_tol_ver) if merge_tol_ver not in (None, '') else legacy_merge_tol
        except (TypeError, ValueError):
            merge_tol_ver = legacy_merge_tol

        # Passed to model and used in group_output() when merging lines by font size (see ctd/inference.py).
        self.model.merge_fntsize_tol_hor = merge_tol_hor
        self.model.merge_fntsize_tol_ver = merge_tol_ver
        box_thresh = self.get_param_value('box score threshold')
        try:
            box_thresh = float(box_thresh) if box_thresh not in (None, '') else 0.45
            box_thresh = max(0.35, min(0.8, box_thresh))
        except (TypeError, ValueError):
            box_thresh = 0.45
        self.model.box_thresh = box_thresh
        min_area = self.get_param_value('min box area')
        try:
            min_area = int(min_area) if min_area not in (None, '') else 0
            min_area = max(0, min_area)
        except (TypeError, ValueError):
            min_area = 0
        self.model.min_box_area = min_area
        try:
            _, mask, blk_list = self.model(work)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_msg = str(e).lower()
            if "out of memory" not in err_msg:
                raise
            self.logger.warning("CTD detector hit GPU OOM. Falling back to CPU for this run.")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cpu_model = load_ctd_model(CTD_TORCH_PATH, "cpu", self.detect_size)
                cpu_model.merge_fntsize_tol_hor = getattr(self.model, 'merge_fntsize_tol_hor', 2.0)
                cpu_model.merge_fntsize_tol_ver = getattr(self.model, 'merge_fntsize_tol_ver', 1.7)
                cpu_model.box_thresh = getattr(self.model, 'box_thresh', 0.45)
                cpu_model.min_box_area = getattr(self.model, 'min_box_area', 0)
                _, mask, blk_list = cpu_model(work)
                del cpu_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e2:
                self.logger.error("CTD det failed after CPU fallback: %s", e2)
                return np.zeros((h_img, w_img), dtype=np.uint8), []

        work_h, work_w = work.shape[:2]
        # det_auto_rotate: if majority of text is horizontal, re-run with 90° rotation (manga-image-translator style)
        if det_auto_rotate and not det_rotate and len(blk_list) > 0:
            n_hor = sum(1 for b in blk_list if (b.xyxy[2] - b.xyxy[0]) > (b.xyxy[3] - b.xyxy[1]))
            if n_hor > len(blk_list) / 2:
                work_orig = work
                work = np.rot90(work_orig, k=-1)
                _, mask, blk_list = self.model(work)
                work_h, work_w = work.shape[:2]
                h_img, w_img = work_orig.shape[0], work_orig.shape[1]
                det_rotate = True
        shrink_val = 0
        sp = self.params.get('box_shrink_px', {})
        if isinstance(sp, dict):
            v = sp.get('value', 0)
            try:
                shrink_val = max(0, min(50, int(v) if v not in (None, '') else 0))
            except (TypeError, ValueError):
                pass
        if shrink_val > 0:
            blk_list = shrink_blocks(blk_list, shrink_val, work_w, work_h)
        
        fnt_rsz = self.get_param_value('font size multiplier')
        fnt_max = self.get_param_value('font size max')
        fnt_min = self.get_param_value('font size min')
        for blk in blk_list:
            sz = blk._detected_font_size * fnt_rsz
            if fnt_max > 0:
                sz = min(fnt_max, sz)
            if fnt_min > 0:
                sz = max(fnt_min, sz)
            blk.font_size = sz
            blk._detected_font_size = sz

        pad_val = 0
        bp = self.params.get('box_padding', {})
        if isinstance(bp, dict):
            v = bp.get('value', 5)
            try:
                pad_val = max(0, min(24, int(v) if v not in (None, '') else 5))
            except (TypeError, ValueError):
                pad_val = 5
        if pad_val > 0:
            blk_list = expand_blocks(blk_list, pad_val, work_w, work_h)

        ksize = self.get_param_value('mask dilate size')
        if ksize > 0:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1),(ksize, ksize))
            mask = cv2.dilate(mask, element)

        if det_rotate:
            mask = np.ascontiguousarray(np.rot90(mask, k=1))
            for blk in blk_list:
                rx1, ry1, rx2, ry2 = blk.xyxy
                x1 = h_img - 1 - ry2
                x2 = h_img - 1 - ry1
                y1 = rx1
                y2 = rx2
                blk.xyxy = [max(0, x1), max(0, y1), min(h_img, x2), min(w_img, y2)]
                new_lines = []
                for line in blk.lines:
                    pts = np.array(line, dtype=np.float64)
                    if pts.ndim == 2 and pts.shape[0] >= 3:
                        ox = h_img - 1 - pts[:, 1]
                        oy = pts[:, 0]
                        ox = np.clip(ox, 0, h_img - 1)
                        oy = np.clip(oy, 0, w_img - 1)
                        new_lines.append(np.stack([ox, oy], axis=1).astype(np.int32).tolist())
                    else:
                        new_lines.append(line)
                blk.lines = new_lines

        # Remove border: crop mask and blocks back to original image size
        if border_added:
            mask = mask[0:old_h, 0:old_w].copy()
            new_blk_list = []
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                x1, x2 = max(0, min(x1, old_w)), max(0, min(x2, old_w))
                y1, y2 = max(0, min(y1, old_h)), max(0, min(y2, old_h))
                if x2 <= x1 or y2 <= y1:
                    continue
                blk.xyxy = [x1, y1, x2, y2]
                new_lines = []
                for line in blk.lines:
                    pts = np.array(line, dtype=np.float64)
                    if pts.ndim == 2 and pts.shape[0] >= 3:
                        pts[:, 0] = np.clip(pts[:, 0], 0, old_w - 1)
                        pts[:, 1] = np.clip(pts[:, 1], 0, old_h - 1)
                        new_lines.append(pts.astype(np.int32).tolist())
                    else:
                        new_lines.append(line)
                blk.lines = new_lines
                new_blk_list.append(blk)
            blk_list = new_blk_list

        return mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.device
        if self.model is not None:
            if self.model.device != device:
                self.model.device = device
                if device != 'cpu':
                    self.model.load_model(CTD_TORCH_PATH)
                else:
                    custom_onnx = (self.params.get('custom_onnx_path') or {}).get('value', '') or ''
                    custom_onnx = custom_onnx.strip() if isinstance(custom_onnx, str) else ''
                    onnx_path = custom_onnx if custom_onnx and os.path.isfile(custom_onnx) else CTD_ONNX_PATH
                    self.model.load_model(onnx_path)
            self.model.detect_size = self.detect_size
