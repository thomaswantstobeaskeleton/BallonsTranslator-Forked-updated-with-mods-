"""
PaddleOCR-VL For Manga – manga/comics-tuned OCR (jzhang533/PaddleOCR-VL-For-Manga).
Uses local/data model; chat-style image + "OCR:" -> text. Compatible with transformers 5.x via shims.

Comparison with paddleocr_vl_hf: This module uses a manga-specific model (smaller, tuned for comics).
paddleocr_vl_hf uses general PaddleOCR-VL 0.9B from HF (109 languages, document parsing) and offers
model id, bf16, crop padding, and chinese_only prompt option.
"""
import numpy as np
import torch
from typing import List

from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

MODEL_PATH = 'data/models/PaddleOCR-VL-For-Manga'


def _ensure_sliding_window_cache():
    """Inject SlidingWindowCache into transformers.cache_utils if missing (e.g. transformers 5.x moved/renamed it)."""
    import transformers.cache_utils as cache_utils
    if getattr(cache_utils, 'SlidingWindowCache', None) is not None:
        return
    try:
        from transformers.cache_utils import SlidingWindowCache  # noqa: F401
        return
    except ImportError:
        pass
    # Stub class so the model's import works. We must NOT use DynamicCache here: the model does
    # target_length = past_key_values.get_max_cache_shape() when isinstance(cache, SlidingWindowCache),
    # and DynamicCache.get_max_cache_shape() returns -1, causing torch.full(..., [N, -1]). The model
    # uses DynamicCache() at runtime, so with a stub SlidingWindowCache, isinstance is False and
    # target_length is taken from attention_mask instead.
    from transformers.cache_utils import Cache

    class _SlidingWindowCacheStub(Cache):
        def get_max_cache_shape(self):
            return 0

        def lazy_initialization(self, key_states, value_states):
            pass

        def update(self, key_states, value_states, cache_kwargs=None):
            return key_states, value_states

        def get_mask_sizes(self, cache_position):
            return (0, 0)

        def get_seq_length(self):
            return 0

    cache_utils.SlidingWindowCache = _SlidingWindowCacheStub


def _ensure_check_model_inputs():
    """Inject check_model_inputs into transformers.utils.generic if missing (removed in transformers 5.x)."""
    import transformers.utils.generic as generic
    if getattr(generic, 'check_model_inputs', None) is not None:
        return
    def check_model_inputs(f):
        return f  # no-op decorator for forward(); model works without input checks
    generic.check_model_inputs = check_model_inputs


def _ensure_rope_default():
    """Inject 'default' into ROPE_INIT_FUNCTIONS if missing (transformers 5.x only has scaling types)."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(config, device=None, seq_len=None, **kwargs):
        # Support both new (rope_parameters) and old (rope_theta on config) formats (e.g. PaddleOCR-VL).
        if hasattr(config, "rope_parameters") and config.rope_parameters is not None:
            rp = config.rope_parameters
            base = rp.get("rope_theta", getattr(config, "rope_theta", 10000.0))
            partial = rp.get("partial_rotary_factor", 1.0)
        else:
            base = getattr(config, "rope_theta", 10000.0)
            partial = 1.0
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial)
        # Use a real device for creating tensors; "meta" device has no storage (lazy loading)
        if device is None or str(device) == "meta":
            device = torch.device("cpu")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _ensure_rope_init_patch():
    """Patch PreTrainedModel._init_weights so RotaryEmbedding without compute_default_rope_parameters uses ROPE_INIT_FUNCTIONS['default']."""
    from transformers.modeling_utils import PreTrainedModel
    from transformers import initialization as init
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if getattr(PreTrainedModel._init_weights, "_paddle_ocr_rope_patched", False):
        return

    _orig_init_weights = PreTrainedModel._init_weights

    def _patched_init_weights(self, module):
        if (
            "RotaryEmbedding" in module.__class__.__name__
            and hasattr(module, "original_inv_freq")
            and hasattr(module, "rope_type")
            and module.rope_type == "default"
            and not hasattr(module, "compute_default_rope_parameters")
        ):
            rope_fn = ROPE_INIT_FUNCTIONS.get("default")
            if rope_fn is not None:
                buffer_value, _ = rope_fn(module.config)
                init.copy_(module.inv_freq, buffer_value)
                init.copy_(module.original_inv_freq, buffer_value)
                return
        _orig_init_weights(self, module)

    _patched_init_weights._paddle_ocr_rope_patched = True  # type: ignore[attr-defined]
    PreTrainedModel._init_weights = _patched_init_weights


@register_OCR('PaddleOCRVLManga')
class PaddleOCRVLManga(OCRBase):
    params = {
        'device': DEVICE_SELECTOR(),
        "language_hint": {
            "type": "selector",
            "options": ["Default (OCR:)", "Chinese (OCR: 请识别图中的文字)", "Japanese (OCR: 画像の文字を認識)"],
            "value": "Default (OCR:)",
            "description": "Prompt hint to reduce misrecognition (e.g. Chinese text read as Japanese). Use Chinese for manhua.",
        },
        "max_new_tokens": {
            "value": 512,
            "description": "Max generation tokens"
        }
    }
    device = DEFAULT_DEVICE

    download_file_list = [{
        'url': 'https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga/resolve/main/',
        'save_dir': 'data/models/PaddleOCR-VL-For-Manga',
        'files': [
            'generation_config.json',
            'modeling_paddleocr_vl.py',
            'model.safetensors',
            'preprocessor_config.json',
            'tokenizer_config.json',
            'added_tokens.json',
            'image_processing.py',
            'tokenizer.model',
            'special_tokens_map.json',
            'tokenizer.json',
            'processor_config.json',
            'chat_template.jinja',
            'configuration_paddleocr_vl.py',
            'config.json',
            'processing_paddleocr_vl.py',
            'README.md'
        ],
        'sha256_pre_calculated': [
            'cf202f984e003e92dceaa27e749b60b4e6e1b566a1df8486b5b41adf1d016cea',
            '351269f534882b2df200192adc3af6117910e9a7446caf0d6706366b1ed45d9f',
            '71fcee0e3618582d4c8acc705242aa79b471b6134e7023bf3820642ba638b602',
            'f417a7f977820dfe6828f3ec2e461c027fdb0662f25cae4e841ec1028e0b988a',
            '67c651ba09c22151a1fff31e8773a24f7607aef1541aa2f200b48552ed30e894',
            'f59f889088e0fe21c523e7cf121bb6dca3b0bb148cb7159fbb4572c74dfc5644',
            '242b36a14461d81fba2913a2e736d9f7a26422c6f2cdf484a5e5a17037f78147',
            '34ef7db83df785924fb83d7b887b6e822a031c56e15cff40aaf9b982988180df',
            '215bf3a1b155fafe3497f8790bedf280af92d29c2f0286c2f87a5c78baff8f7c',
            'f90f04fd8e5eb6dfa380f37d10c87392de8438dccb6768a2486b5a96ee76dba6',
            '1568858960a9760c54431dae693a6152e601ff55cdf6d2eab97a4a99958faea0',
            '344fea8b69546526a00996468f86f583fd65441582a36f2fa4abc794aa94094c',
            '753dd93654c3a9c8c85a3eaee1e3092dd12591b0f2dce0305e1abfb7a41ff160',
            '928aaf78567a273cb73ede3671253ec4e38eb60c27a30e945bcd13b4131a0147',
            'e29cb1e5f275f2bd3ce051bd5c9983a33894e693b2823a0e13d4c07c8c4f9e13',
            None
        ],
        'concatenate_url_filename': 1
    }]
    _load_model_keys = {'model', 'processor'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.model = None
        self.processor = None

    def ocr_img(self, img: np.ndarray) -> str:
        hint = self.get_param_value('language_hint') or "Default (OCR:)"
        if hint.startswith("Chinese"):
            prompt_text = "OCR: 请识别图中的文字"
        elif hint.startswith("Japanese"):
            prompt_text = "OCR: 画像の文字を認識"
        else:
            prompt_text = "OCR:"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        inputs = {
            k: (v.to(self.model.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

        # Generate text
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.get_param_value('max_new_tokens'),
                do_sample=False,
                use_cache=True
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated[:, input_length:]
        answer = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return answer.split('\n')

    def _load_model(self):
        if self.model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoProcessor
            except ImportError as e:
                raise ImportError(
                    f"PaddleOCRVLManga requires transformers. Install: pip install transformers. {e}"
                ) from e
            _ensure_sliding_window_cache()
            _ensure_check_model_inputs()
            _ensure_rope_default()
            _ensure_rope_init_patch()
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Unrecognized keys in.*rope_parameters")
                    warnings.filterwarnings("ignore", message=".*layers were not sharded")
                    warnings.filterwarnings("ignore", message=".*AttentionMaskConverter.*deprecated")
                    model = AutoModelForCausalLM.from_pretrained(
                        MODEL_PATH,
                        trust_remote_code=True,
                        dtype=torch.float16 if self.device == "cuda" else torch.float32
                    ).to(self.device).eval()

                    # sentencepiece on Windows is sensitive to being first loaded/used from a
                    # PyQt QThread; add_prefix_space=None bypasses loading it completely.
                    # (upstream dmMaze/BallonsTranslator#1203)
                    processor = AutoProcessor.from_pretrained(
                        MODEL_PATH, trust_remote_code=True, use_fast=True,
                        add_prefix_space=None
                    )
            except ImportError as e:
                if "SlidingWindowCache" in str(e):
                    try:
                        import transformers
                        ver = getattr(transformers, "__version__", "unknown")
                    except Exception:
                        ver = "unknown"
                    raise ImportError(
                        f"PaddleOCRVLManga's model code needs SlidingWindowCache from transformers. "
                        f"Your version: {ver}. Try: pip install --upgrade \"transformers>=4.47\""
                    ) from e
                raise

            # Set pad_token_id to avoid warning during generation
            if model.generation_config.pad_token_id is None:
                model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

            self.model = model
            self.processor = processor

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            x1 = max(0, min(int(round(float(x1))), im_w - 1))
            y1 = max(0, min(int(round(float(y1))), im_h - 1))
            x2 = max(x1 + 1, min(int(round(float(x2))), im_w))
            y2 = max(y1 + 1, min(int(round(float(y2))), im_h))
            if y2 <= im_h and x2 <= im_w and x1 >= 0 and y1 >= 0 and x1 < x2 and y1 < y2: 
                # Extract region and convert RGBA to RGB if necessary for model input
                region = img[y1:y2, x1:x2]
                blk.text = self.ocr_img(region)
            else:
                self.logger.warning('invalid textbbox to target img')
                blk.text = ['']

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.params['device']['value']
        if self.device != device and self.model is not None:
            self.model.to(device)


