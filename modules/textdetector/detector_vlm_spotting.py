"""
VLM-based text detectors (spotting): OlmOCR 7B, Callisto-OCR3-2B, Qwen2-VL-OCR-2B.
Full-image spotting: prompt the VLM to detect and recognize all text and return a JSON array
of {box, text}. Pair with none_ocr to keep spotter text, or with any OCR to re-recognize crops.
Reuses the same models as olm_ocr, callisto_ocr, qwen2_vl_ocr_2b (OCR modules).
Requires: pip install transformers torch pillow accelerate
"""
import os
import re
import json
import tempfile
import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, ProjImgTrans
from ..base import DEVICE_SELECTOR

try:
    from transformers import AutoProcessor, AutoModelForImageTextToText, GenerationConfig
    from PIL import Image
    import torch
    _VLM_SPOT_AVAILABLE = True
except ImportError:
    _VLM_SPOT_AVAILABLE = False
    import logging
    logging.getLogger("BallonsTranslator").debug(
        "VLM spotting detectors not available. Install: pip install transformers torch pillow accelerate"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def _parse_spotting_output(raw: str) -> List[Tuple[List[Tuple[int, int]], str]]:
    """
    Parse VLM spotting output into list of (polygon_points, text).
    Supports JSON array of {box/poly/bbox, text/content}; markdown code block; malformed array fix.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    out: List[Tuple[List[Tuple[int, int]], str]] = []

    def _parse_one_item(item: dict) -> bool:
        box = item.get("box", item.get("poly", item.get("bbox", item.get("points"))))
        text = item.get("text", item.get("content", ""))
        if isinstance(text, (list, tuple)):
            text = " ".join(str(t) for t in text)
        if box is None or not isinstance(box, (list, tuple)):
            return False
        pts = []
        for p in box:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((int(p[0]), int(p[1])))
            elif isinstance(p, (int, float)):
                if len(box) >= 4 and len(pts) < 2:
                    pts = [(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))]
                elif len(box) >= 8 and len(pts) == 0:
                    pts = [
                        (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                        (int(box[4]), int(box[5])), (int(box[6]), int(box[7])),
                    ]
                break
        if len(pts) >= 2:
            if len(pts) == 2:
                x1, y1, x2, y2 = pts[0][0], pts[0][1], pts[1][0], pts[1][1]
                pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            out.append((pts, str(text)))
            return True
        return False

    def _parse_json_array(s: str) -> bool:
        try:
            data = json.loads(s)
            if not isinstance(data, list):
                return False
            for item in data:
                if isinstance(item, dict):
                    _parse_one_item(item)
            return len(out) > 0
        except (json.JSONDecodeError, TypeError):
            return False

    # Strip markdown code block if present
    for pattern in (r"```(?:json)?\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"):
        m = re.search(pattern, raw, re.IGNORECASE)
        if m:
            if _parse_json_array(m.group(1).strip()):
                return out
            out.clear()

    # Fix common model typo: "}], {" instead of "}, {" (extra ] breaks JSON array)
    normalized = raw.replace('"}], {"', '"}, {"').replace('"}],{"', '"},{"')
    if _parse_json_array(normalized):
        return out
    out.clear()
    if _parse_json_array(raw):
        return out
    out.clear()

    # Try to extract and parse each object separately (handles truncated or malformed array)
    obj_pattern = re.compile(
        r'\{\s*"box"\s*:\s*(\[[^\]]+(?:\[[^\]]*\][^\]]*)*\])\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        re.DOTALL
    )
    for m in obj_pattern.finditer(raw):
        try:
            box_json = m.group(1)
            text = m.group(2).replace('\\"', '"')
            box = json.loads(box_json)
            if isinstance(box, list) and len(box) >= 2:
                pts = []
                for p in box:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        pts.append((int(p[0]), int(p[1])))
                if len(pts) >= 2:
                    if len(pts) == 2:
                        x1, y1, x2, y2 = pts[0][0], pts[0][1], pts[1][0], pts[1][1]
                        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    out.append((pts, text))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    if out:
        return out

    # Fallback: find array in string and try parse (greedy match may include too much)
    match = re.search(r'\[[\s\S]*\]', raw)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list) and data:
                for item in data:
                    if isinstance(item, dict):
                        _parse_one_item(item)
                if out:
                    return out
        except (json.JSONDecodeError, TypeError):
            pass
        out.clear()
    for m in re.finditer(r'\[[\d\s,]+\][\s:,\-]*["\']?([^"\']*)["\']?', raw):
        box_str = m.group(0).split(']')[0] + ']'
        text = m.group(1).strip().strip('"\'')
        try:
            coords = json.loads(box_str)
            if isinstance(coords, list) and len(coords) >= 4:
                arr = np.array(coords, dtype=np.int32)
                if arr.ndim == 1:
                    x1, y1, x2, y2 = int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])
                    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    out.append((pts, text))
                elif arr.ndim == 2 and arr.shape[0] >= 2:
                    pts = [(int(arr[i, 0]), int(arr[i, 1])) for i in range(min(4, len(arr)))]
                    if len(pts) == 2:
                        pts = [(pts[0][0], pts[0][1]), (pts[1][0], pts[0][1]), (pts[1][0], pts[1][1]), (pts[0][0], pts[1][1])]
                    out.append((pts, text))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return out


# Spotting prompt that asks for JSON box+text (same idea as HunyuanOCR json_en)
SPOTTING_PROMPT_JSON = (
    "Detect and recognize all text in the image. "
    "Output only a JSON array, nothing else. "
    "Each element: {\"box\": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], \"text\": \"recognized text\"}. "
    "box is four corner points in pixel coordinates (clockwise or counterclockwise)."
)

# Default max long side for input image to avoid CUDA OOM on full manga pages (e.g. 11GB GPU)
DEFAULT_MAX_IMAGE_SIZE = 1280


def _resize_pil_to_max_size(pil_img: "Image.Image", max_size: int):
    """Resize so longest side <= max_size; return (resized_pil, scale_x, scale_y) for scaling boxes back."""
    w, h = pil_img.size
    if max_size <= 0 or (w <= max_size and h <= max_size):
        return pil_img, 1.0, 1.0
    scale = min(max_size / w, max_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized, w / new_w, h / new_h


def _scale_parsed_boxes(parsed: List[Tuple[List[Tuple[int, int]], str]], scale_x: float, scale_y: float):
    """Scale polygon points from resized image coords back to original image."""
    if scale_x == 1.0 and scale_y == 1.0:
        return parsed
    out = []
    for pts, text in parsed:
        scaled_pts = [(int(round(x * scale_x)), int(round(y * scale_y))) for x, y in pts]
        out.append((scaled_pts, text))
    return out


def _make_vlm_spot_detector(
    key: str,
    model_id: str,
    processor_id: str,
    description_short: str,
    low_vram_default: bool = False,
):
    """Factory for VLM spotting detectors (OlmOCR, Callisto, Qwen2-VL-OCR-2B)."""

    class _VLMSpotDetector(TextDetectorBase):
        params = {
            "device": DEVICE_SELECTOR(),
            "max_image_size": {
                "type": "line_editor",
                "value": DEFAULT_MAX_IMAGE_SIZE,
                "description": "Max length of the longest side (px) before spotting. Resize to avoid OOM on 11GB GPUs; 0 = no resize.",
            },
            "max_new_tokens": {
                "type": "line_editor",
                "value": 2048,
                "description": "Max tokens for full-page spotting (increase for dense pages).",
            },
            "spotting_prompt": {
                "type": "line_editor",
                "value": SPOTTING_PROMPT_JSON,
                "description": "Prompt for spotting (ask for JSON box+text for best parsing).",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 when available.",
            },
            "low_vram": {
                "type": "checkbox",
                "value": low_vram_default,
                "description": "Use device_map=auto (CPU offload). Reduces OOM on 11GB.",
            },
            "description": description_short,
        }
        _load_model_keys = {"processor", "model"}
        _model_id = model_id
        _processor_id = processor_id

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = (self.params.get("device") or {}).get("value", "cpu")
            self.processor = None
            self.model = None
            self._device_for_inputs = None

        def _load_model(self):
            dev = (self.params.get("device") or {}).get("value", "cpu")
            if dev in ("cuda", "gpu") and torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = "cpu"
            if self.processor is not None and self.model is not None:
                if not getattr(self, "_device_for_inputs", None) and hasattr(self.model, "to"):
                    try:
                        self.model.to(dev)
                    except Exception:
                        pass
                self.device = dev
                return
            self.device = dev
            use_bf16 = self.params.get("use_bf16", {}).get("value", True)
            dtype = torch.bfloat16
            if not (torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
                dtype = torch.float16
            if not use_bf16:
                dtype = torch.float16
            low_vram = self.params.get("low_vram", {}).get("value", low_vram_default)
            model_id = self._model_id
            proc_id = self._processor_id or model_id
            self.processor = AutoProcessor.from_pretrained(proc_id)
            load_kw = {"torch_dtype": dtype}
            if low_vram:
                load_kw["device_map"] = "auto"
                self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kw)
                self._device_for_inputs = next(self.model.parameters()).device
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kw)
                self.model.to(dev)
                self._device_for_inputs = None
            self.model.eval()

        def _run_spotting(self, pil_img: "Image.Image"):
            if pil_img.size[0] == 0 or pil_img.size[1] == 0:
                return "", None, None
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pil_img.save(tmp_path)
                img_ref = os.path.abspath(tmp_path)
                prompt = (self.params.get("spotting_prompt") or {}).get("value", SPOTTING_PROMPT_JSON) or SPOTTING_PROMPT_JSON
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": img_ref},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs.pop("token_type_ids", None)
                inp_device = self._device_for_inputs if self._device_for_inputs is not None else self.model.device
                inputs = {k: (v.to(inp_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
                max_tokens = 2048
                mt = self.params.get("max_new_tokens", {})
                if isinstance(mt, dict):
                    try:
                        max_tokens = max(512, min(8192, int(mt.get("value", 2048))))
                    except (TypeError, ValueError):
                        pass
                # Use GenerationConfig with only supported args to avoid "temperature/top_p/top_k not valid" warning
                gen_config = GenerationConfig(
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=getattr(self.processor.tokenizer, "pad_token_id", None) or getattr(self.processor.tokenizer, "eos_token_id"),
                )
                with torch.inference_mode():
                    out = self.model.generate(**inputs, generation_config=gen_config)
                input_len = inputs["input_ids"].shape[1]
                gen = out[0, input_len:]
                text = self.processor.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                # Return model input size (H, W) for coordinate scaling; processor may resize our image
                model_h, model_w = None, None
                if "pixel_values" in inputs and hasattr(inputs["pixel_values"], "shape"):
                    sh = inputs["pixel_values"].shape
                    if len(sh) >= 4:
                        model_h, model_w = int(sh[-2]), int(sh[-1])
                return text, model_h, model_w
            except Exception as e:
                self.logger.warning(f"VLM spotting failed: {e}")
                return "", None, None
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            blk_list: List[TextBlock] = []
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            pil_img = _cv2_to_pil_rgb(img)
            max_size = DEFAULT_MAX_IMAGE_SIZE
            mis = self.params.get("max_image_size", {})
            if isinstance(mis, dict):
                try:
                    max_size = max(0, int(mis.get("value", DEFAULT_MAX_IMAGE_SIZE)))
                except (TypeError, ValueError):
                    pass
            pil_for_model, scale_x, scale_y = _resize_pil_to_max_size(pil_img, max_size)
            raw, model_h, model_w = self._run_spotting(pil_for_model)
            if not raw:
                self.logger.warning(
                    "VLM spotter returned no output. Try another detector or check model/device. "
                    "On 11GB GPU enable low_vram and/or lower max_image_size (e.g. 1024)."
                )
                return mask, blk_list
            parsed = _parse_spotting_output(raw)
            # Scale from model input space to original image; prefer actual model input size from processor
            if model_h is not None and model_w is not None and model_h > 0 and model_w > 0:
                scale_x = w / model_w
                scale_y = h / model_h
            parsed = _scale_parsed_boxes(parsed, scale_x, scale_y)
            if not parsed:
                self.logger.warning(
                    "Could not parse any boxes from VLM output. Try custom spotting_prompt asking for JSON box+text. Raw (first 400 chars): %s",
                    raw[:400] if len(raw) > 400 else raw,
                )
                return mask, blk_list
            for pts, text in parsed:
                if len(pts) < 3:
                    continue
                arr = np.array(pts, dtype=np.int32)
                x1 = int(arr[:, 0].min())
                y1 = int(arr[:, 1].min())
                x2 = int(arr[:, 0].max())
                y2 = int(arr[:, 1].max())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts])
                blk._detected_font_size = max(y2 - y1, 12)
                blk.text = [text] if text else [""]
                blk_list.append(blk)
                cv2.fillPoly(mask, [arr], 255)
            return mask, blk_list

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key == "device":
                self.device = (self.params.get("device") or {}).get("value", "cpu")
                if self.model is not None and not getattr(self, "_device_for_inputs", None):
                    try:
                        self.model.to(self.device)
                    except Exception:
                        pass
            elif param_key in ("use_bf16", "low_vram"):
                self.processor = None
                self.model = None
                self._device_for_inputs = None

    _VLMSpotDetector._model_id = model_id
    _VLMSpotDetector._processor_id = processor_id
    return _VLMSpotDetector


if _VLM_SPOT_AVAILABLE:
    OlmOCRDetector = _make_vlm_spot_detector(
        "olm_ocr_det",
        "allenai/olmOCR-2-7B-1025",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "OlmOCR 7B as full-page spotter (det+rec). Pair with none_ocr or any OCR.",
        low_vram_default=True,
    )
    register_textdetectors("olm_ocr_det")(OlmOCRDetector)

    CallistoOCRDetector = _make_vlm_spot_detector(
        "callisto_ocr_det",
        "prithivMLmods/Callisto-OCR3-2B-Instruct",
        "prithivMLmods/Callisto-OCR3-2B-Instruct",
        "Callisto-OCR3-2B as full-page spotter (det+rec). Pair with none_ocr or any OCR.",
        low_vram_default=True,
    )
    register_textdetectors("callisto_ocr_det")(CallistoOCRDetector)

    Qwen2VLOCR2BDetector = _make_vlm_spot_detector(
        "qwen2_vl_ocr_2b_det",
        "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
        "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
        "Qwen2-VL-OCR-2B as full-page spotter (det+rec). Pair with none_ocr or any OCR.",
        low_vram_default=True,
    )
    register_textdetectors("qwen2_vl_ocr_2b_det")(Qwen2VLOCR2BDetector)
