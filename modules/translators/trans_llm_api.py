import re
import time
import json
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Type, Tuple

import httpx
import openai
from pydantic import BaseModel, Field, ValidationError

from utils.llm_provider import (
    effective_provider,
    is_local_provider,
    normalize_model_name,
    resolve_endpoint,
    ENDPOINT_PRESETS,
)
from utils.proxy_utils import create_httpx_client
from utils.series_context_store import (
    get_series_context_dir,
    load_series_glossary,
    load_recent_context,
    append_page_to_series_context as store_append_page,
    merge_glossary_no_dupes,
    append_to_series_glossary,
)
from .base import BaseTranslator, register_translator
from .exceptions import CriticalTranslationError
from modules.context.errors import ContextLengthError, is_context_length_error
from modules.context.glossary import GlossaryError
from modules.context.token_usage import format_completion_token_usage


_MAX_VIDEO_OCR_CORRECT_CACHE = 256  # dedupe per-frame OCR LLM calls (e.g. static title cards)
# Video: merged glossary can be 100k+ chars; system prompt must stay bounded for local LLMs.
_MAX_VIDEO_GLOSSARY_SYSTEM_CHARS = 12000
_MAX_VIDEO_GLOSSARY_TERMS = 50


class InvalidNumTranslations(Exception):
    """Exception raised when the number of translations does not match the number of sources."""

    pass


class TranslationElement(BaseModel):
    id: int = Field(..., description="The original numeric ID of the text snippet.")
    translation: str = Field(
        ..., description="The translated text corresponding to the id."
    )


class TranslationResponse(BaseModel):
    translations: List[TranslationElement] = Field(
        ..., description="A list of all translated elements."
    )
    revised_previous: Optional[List[str]] = Field(
        default=None,
        description="Optional (video only): improved translations for the previous N subtitles, in same order, for better flow. Omit if not needed.",
    )


@register_translator("LLM_API_Translator")
class LLM_API_Translator(BaseTranslator):
    concate_text = False
    cht_require_convert = True

    @staticmethod
    def _page_key_subtitle_batch_pipeline(page_key: str) -> bool:
        """Chunked NLP, strict alignment fallback, per-line fixes (video dialog + subtitle file tool)."""
        pk = page_key or ""
        return pk.startswith("video_") or pk.startswith("subtitle_file")
    params: Dict = {
        "provider": {
            "type": "selector",
            "options": ["OpenAI", "Google", "Grok", "OpenRouter", "LLM Studio", "Ollama"],
            "value": "OpenAI",
            "description": "Select the LLM provider.",
        },
        "apikey": {
            "value": "",
            "description": "Single API key to use if multiple keys are not provided.",
        },
        "multiple_keys": {
            "type": "editor",
            "value": "",
            "description": "API keys separated by semicolons (;). Requests will rotate through these keys.",
        },
        "model": {
            "type": "selector",
            "options": [
                "OAI: gpt-4o",
                "OAI: gpt-4-turbo",
                "OAI: gpt-3.5-turbo",
                "GGL: gemini-1.5-pro-latest",
                "GGL: gemini-2.5-flash",
                "GGL: gemini-2.5-flash-lite",
                "XAI: grok-4",
                "XAI: grok-3",
                "XAI: grok-3-mini",
                # OpenRouter free text models (https://openrouter.ai/models?fmt=cards&max_price=0&order=most-popular&output_modalities=text&input_modalities=text)
                "OpenRouter: openrouter/free",
                "OpenRouter: meta-llama/llama-3.2-3b-instruct:free",
                "OpenRouter: meta-llama/llama-3.3-70b-instruct:free",
                "OpenRouter: google/gemma-3n-e2b-it:free",
                "OpenRouter: google/gemma-3n-e4b-it:free",
                "OpenRouter: stepfun/step-3.5-flash:free",
                "OpenRouter: qwen/qwen3-4b:free",
                "OpenRouter: qwen/qwen3-next-80b-a3b-instruct:free",
                "OpenRouter: qwen/qwen3-235b-a22b-thinking-2507",
                "OpenRouter: nvidia/nemotron-nano-9b-v2:free",
                "OpenRouter: nvidia/nemotron-3-nano-30b-a3b:free",
                "OpenRouter: liquid/lfm-2.5-1.2b-instruct:free",
                "OpenRouter: liquid/lfm-2.5-1.2b-thinking:free",
                "OpenRouter: z-ai/glm-4.5-air:free",
                "OpenRouter: arcee-ai/trinity-mini:free",
                "OpenRouter: arcee-ai/trinity-large-preview:free",
                "OpenRouter: nousresearch/hermes-3-llama-3.1-405b:free",
                "OpenRouter: upstage/solar-pro-3:free",
                "OpenRouter: openai/gpt-oss-20b:free",
                "OpenRouter: openai/gpt-oss-120b:free",
                "OpenRouter: qwen/qwen3-coder:free",
                "OpenRouter: cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
                "LLMS: (override model field)",
                "OLL: (override model field)",
            ],
            "value": "OAI: gpt-4o",
            "description": "Select a model that supports JSON Mode for structured output.",
        },
        "override model": {
            "value": "",
            "description": "Specify a custom model name to override the selected model.",
        },
        "endpoint": {
            "value": "",
            "description": "Base URL for the API. Leave empty for provider default.",
        },
        "endpoint_preset": {
            "type": "selector",
            "options": list(ENDPOINT_PRESETS.keys()),
            "value": "Auto (provider default)",
            "description": "Quick preset for endpoint/provider routing (e.g. Ollama, LM Studio, OpenRouter).",
        },
        "system_prompt": {
            "type": "editor",
            "value": (
                "You are an expert translator for comics and dialogue. Translate the given text snippets naturally and concisely. "
                "Use consistent terminology for the same concept (e.g. stick to one term for recurring ideas). "
                "Cohesion and styling: keep dialogue tone consistent and match character voice across snippets. "
                "Output MUST be valid JSON only: a single object with key 'translations', a list of objects with 'id' (integer) and 'translation' (string). "
                "No markdown, no code fences, no extra text.\n"
                "Example: {\"translations\": [{\"id\": 1, \"translation\": \"Translated text.\"}]}"
            ),
            "description": "Main translator system prompt (JSON mode, comics/pages). Used for regular image/page translation.",
        },
        "video_system_prompt": {
            "type": "editor",
            "value": (
                "You are an expert subtitle translator for Chinese video subtitles.\n"
                "- You will receive the last few subtitle lines (source and translation) as context. "
                "Your new translation MUST flow naturally from them—read as one continuous dialogue.\n"
                "- Translate each new input line into fluent, natural English suitable for on-screen subtitles.\n"
                "- HARD RULES: Do not add new facts/plot. Do not invent names, places, events, or backstory. If unsure, stay close to the source.\n"
                "- Use appropriate punctuation: ? for questions, ! for exclamations or strong reactions; add or fix when the tone clearly calls for it.\n"
                "- Continuations: when a line is clearly the first half of a sentence that continues in the next subtitle, end it with a comma (e.g. \"Around thirty years old,\"). If a continuation line would be a fragment, you may add the subject (e.g. \"I was taken away...\") when the speaker is first person.\n"
                "- Preserve speaker perspective: when the preceding context is first person (I / I'm / my) or the source implies 我, use first person (my not his, I not he).\n"
                "- NEVER output placeholders like \"[region 1]\" or \"Region 1\". If input is unreadable/noise, output an empty string.\n"
                "- Keep terminology, names, and tone consistent with the previous subtitles and the series.\n"
                "- If the first few previous subtitles read as fragments or break the flow, you MAY output an optional \"revised_previous\" array in your JSON: one improved translation per previous subtitle (same order), so the whole sequence reads smoothly. Only do this when it significantly improves flow; omit to save tokens and avoid rate limits.\n"
                "- When it helps clarity or tone, you may use *italic* for emphasis or off-screen dialogue and **bold** for strong emphasis (rendered on-screen). Use sparingly. No other markdown or code.\n"
                "- OUTPUT FORMAT (strict): Return ONLY a single JSON object. No markdown, no code fences, no commentary. JSON keys: \"translations\" (list of {id:int, translation:str}) and optionally \"revised_previous\" (list of strings, same count/order as provided previous lines). Do not add any other keys.\n"
                "Example: {\"translations\": [{\"id\": 1, \"translation\": \"Translated line.\"}], \"revised_previous\": [\"Improved line 1.\", \"Improved line 2.\"]}"
            ),
            "description": "System prompt for Video translator. Asks for JSON with 'translations' and optional 'revised_previous' for flow. Use video_allow_revised_previous to disable revised_previous to save tokens.",
        },
        "video_allow_revised_previous": {
            "type": "checkbox",
            "value": True,
            "description": "When translating for Video translator: allow model to return optional revised_previous (improved earlier lines for flow). Turn off to reduce tokens and avoid rate limits.",
        },
        "translation_glossary": {
            "type": "editor",
            "value": "",
            "description": "Terminology for consistency (one per line). Format: source -> target or source = target. E.g. 丹田 -> dantian. Used on every page.",
        },
        "context_previous_pages": {
            "type": "line_editor",
            "value": 1,
            "description": "Include this many previous pages (source+translation) as context for consistency. 0 = off. 1–2 recommended.",
        },
        "context_next_page": {
            "type": "checkbox",
            "value": False,
            "description": "Include the next page's source text as context (helps with continuity; upstream #1142). Off by default.",
        },
        "hint_original_regions": {
            "type": "checkbox",
            "value": False,
            "description": "Prepend [Original regions: N] to the prompt (manga-translator-ui AI line-break style). Tells the model to output exactly N translations for better 1:1 region matching. No extra API call.",
        },
        "series_context_prompt": {
            "type": "editor",
            "value": "",
            "description": "Optional. E.g. 'This is a cultivation manhua. Keep cultivation terms and character names consistent.'",
        },
        "series_context_path": {
            "type": "line_editor",
            "value": "default",
            "description": "Folder or series ID for cross-chapter consistency (e.g. urban_immortal_cultivator). Uses data/translation_context/<id>/glossary.txt and recent_context.json. Default: 'default'.",
        },
        "context_max_chars": {
            "type": "line_editor",
            "value": 2000,
            "description": "Max characters for previous-context block (to fit model context limit). 0 = no limit. When over limit: if 'summarize when over limit' is on, the model summarizes context; otherwise recent context is kept and the rest is dropped.",
        },
        "context_trim_mode": {
            "type": "selector",
            "options": ["full", "compact"],
            "value": "compact",
            "description": "full = use all lines per page (up to limit). compact = use at most 2 lines per page to save tokens.",
        },
        "summarize_context_when_over_limit": {
            "type": "checkbox",
            "value": True,
            "description": "When context exceeds context_max_chars, ask the model to summarize it (preserving key terms and style) instead of truncating. Uses one extra API call.",
        },
        "keyword_replacements": {
            "type": "editor",
            "value": "",
            "description": "Post-translation word substitutions (one per line). Format: original -> replacement. Whole-word only, e.g. 'powers' becomes 'abilities' but 'superpowers' is unchanged. Add more lines as needed.",
        },
        "invalid repeat count": {
            "value": 2,
            "description": "Number of retries if the count of translations mismatches the source count (check_br_and_retry style).",
        },
        "post_translation_check": {
            "type": "checkbox",
            "value": False,
            "description": "Validate output: repetition (e.g. 20+ same chars) and target-language ratio. Retry up to post_check_max_retries on failure.",
        },
        "post_check_repetition_chars": {
            "type": "line_editor",
            "value": 20,
            "description": "Treat as hallucination when this many consecutive identical characters appear in a translation (post_translation_check).",
        },
        "post_check_target_ratio": {
            "type": "line_editor",
            "value": 0.4,
            "description": "Minimum fraction of output that must be in target language (post_translation_check). 0 = disabled. Lower if translations with names/terms in source script trigger retries.",
        },
        "post_check_max_retries": {
            "type": "line_editor",
            "value": 2,
            "description": "Max retries when post-translation check fails (repetition or low target-language ratio).",
        },
        "extract_glossary": {
            "type": "checkbox",
            "value": False,
            "description": "After each translated batch, call the model once to extract terms (names, places, etc.) and append to series glossary (series_context_path). Uses one extra API call per batch.",
        },
        "max requests per minute": {
            "value": 10,
            "description": "Max requests per minute per API key. OpenRouter free-tier limit is 20 RPM; use 6–10 to stay safe.",
        },
        "delay": {
            "value": 3.5,
            "description": "Seconds between each API request. OpenRouter free tier allows 20 RPM (min 3s). Use 3.5–5 for :free models; paid keys can use 0.3–1.",
        },
        "max tokens": {
            "value": 4096,
            "description": "Maximum tokens for the response (upper bound). Translation requests use a smaller effective cap from batch size so values like 20000 do not slow local servers.",
        },
        "enable_reasoning": {
            "type": "checkbox",
            "value": False,
            "description": "Enable model reasoning/thinking when supported. Safe fallback is automatic: if unsupported, request retries without reasoning params.",
        },
        "reasoning_effort": {
            "type": "selector",
            "options": ["low", "medium", "high"],
            "value": "medium",
            "description": "Reasoning effort level when reasoning is enabled. Higher can improve difficult lines but may be slower/costlier.",
        },
        "temperature": {
            "value": 0.1,
            "description": "Sampling temperature. Lower values are recommended for structured output.",
        },
        "top p": {
            "value": 1.0,
            "description": "Top P for sampling.",
        },
        "top k": {
            "value": 0,
            "description": "Top K for sampling. 0 disables/lets provider default. Supported by OpenRouter, Ollama, and many local OpenAI-compatible servers.",
        },
        "retry attempts": {
            "value": 3,
            "description": "Number of retry attempts on API connection or parsing failures.",
        },
        "retry timeout": {
            "value": 15,
            "description": "Timeout between retry attempts (seconds).",
        },
        "rate limit delay": {
            "value": 60,
            "description": "Seconds to wait when API returns 429 before retrying. Free models often hit upstream provider limits; 60–90s helps.",
        },
        "proxy": {
            "value": "",
            "description": "Proxy address (e.g., http(s)://user:password@host:port or socks4/5://user:password@host:port)",
        },
        "frequency penalty": {
            "value": 0.0,
            "description": "Frequency penalty (OpenAI).",
        },
        "presence penalty": {"value": 0.0, "description": "Presence penalty (OpenAI)."},
        "low vram mode": {
            "type": "checkbox",
            "value": False,
            "description": "Enable for local/OpenAI-compatible providers on low-VRAM devices to disable parallel page translation and avoid OOM.",
        },
        "include_page_image": {
            "type": "checkbox",
            "value": False,
            "description": "Include the current page image as context for the model (vision-capable models only). Helps with layout and style.",
        },
        "reflection_translation": {
            "type": "checkbox",
            "value": False,
            "description": "After translating, ask the model to review and improve the translation (VideoCaptioner-style). One extra API call per batch; improves naturalness at higher cost.",
        },
        "correct_ocr_with_llm": {
            "type": "checkbox",
            "value": False,
            "description": "Before translating, ask the model to correct OCR output (typos, punctuation, spacing). Used by video translator when enabled. One extra API call per keyframe.",
        },
        "correct_asr_with_llm": {
            "type": "checkbox",
            "value": False,
            "description": "Correct ASR output with LLM before translating (punctuation, casing, typos). Used by video translator when Source = Audio. One extra API call per run.",
        },
    }

    def _setup_translator(self):
        self.lang_map = {
            "简体中文": "Simplified Chinese",
            "繁體中文": "Traditional Chinese",
            "日本語": "Japanese",
            "English": "English",
            "한국어": "Korean",
            "Tiếng Việt": "Vietnamese",
            "čeština": "Czech",
            "Nederlands": "Dutch",
            "Français": "French",
            "Deutsch": "German",
            "magyar nyelv": "Hungarian",
            "Italiano": "Italian",
            "Polski": "Polish",
            "Português": "Portuguese",
            "Brazilian Portuguese": "Brazilian Portuguese",
            "limba română": "Romanian",
            "русский язык": "Russian",
            "Español": "Spanish",
            "Türk dili": "Turkish",
            "украї́нська мо́ва": "Ukrainian",
            "Thai": "Thai",
            "Arabic": "Arabic",
            "Malayalam": "Malayalam",
            "Tamil": "Tamil",
            "Hindi": "Hindi",
        }
        self.token_count = 0
        self.token_count_last = 0
        self._token_count_lock = threading.Lock()
        self._client_init_lock = threading.Lock()
        self.current_key_index = 0
        self.last_request_time = 0
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.key_usage = {}
        self.client = None
        self._client_signature = None
        self._logged_lm_studio_default_endpoint = False
        self._translation_context_previous_pages = None
        self._translation_project_glossary = None
        self._translation_series_context_path = ""
        # Context-aware translation state (see history_mode_enabled).
        self._llm_request_context = None
        self._history_window = None

    def _get_series_context_path(self) -> str:
        """Resolve series context path (from param or set via set_translation_context)."""
        path = getattr(self, "_translation_series_context_path", None) or ""
        if (path or "").strip():
            return get_series_context_dir(path)
        param = (self.get_param_value("series_context_path") or "").strip()
        if param:
            return get_series_context_dir(param)
        return ""

    def _get_glossary_entries(self) -> List[tuple]:
        """Parse translation_glossary param into list of (source, target). Same format as keyword_replacements."""
        raw = self.get_param_value("translation_glossary") or ""
        out = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in ("->", "→", "=", ":"):
                if sep in line:
                    parts = line.split(sep, 1)
                    if len(parts) == 2:
                        k, v = parts[0].strip(), parts[1].strip()
                        if k and v:
                            out.append((k, v))
                        break
        return out

    def set_translation_context(self, previous_pages=None, project_glossary=None, series_context_path=None, next_page=None):
        """Set previous pages (list of {sources, translations}), optional next_page {sources: [...]}, and project glossary for consistency."""
        self._translation_context_previous_pages = previous_pages or []
        self._translation_context_next_page = next_page if next_page and isinstance(next_page, dict) else None
        self._translation_project_glossary = project_glossary or []
        if series_context_path is not None:
            self._translation_series_context_path = (series_context_path or "").strip()
        # Also store for base cache key
        try:
            super().set_translation_context(previous_pages, project_glossary, series_context_path, next_page)
        except Exception:
            pass

    def append_page_to_series_context(self, series_context_path: str, sources: List[str], translations: List[str]) -> None:
        """Append one translated page to the series context store for cross-chapter consistency."""
        path = get_series_context_dir((series_context_path or "").strip())
        if not path or not sources:
            return
        n_prev = max(5, self.context_previous_pages_count * 3)
        store_append_page(path, sources, translations, max_stored_pages=n_prev)

    def _build_system_prompt(self) -> str:
        """System prompt plus optional glossary (translator + project + series) and series context."""
        # Choose base prompt: dedicated video prompt for video translator contexts,
        # otherwise the main system_prompt.
        page_key = getattr(self, "_current_page_key", None) or ""
        if page_key.startswith("video_"):
            base_prompt = (self.get_param_value("video_system_prompt") or "").strip() or self.system_prompt
        else:
            base_prompt = self.system_prompt
        parts = [base_prompt]
        translator_glossary = self._get_glossary_entries()
        proj_glossary = getattr(self, "_translation_project_glossary", None) or []
        proj_as_tuples = [
            (g["source"], g["target"])
            for g in proj_glossary
            if isinstance(g, dict) and g.get("source") and g.get("target")
        ]
        series_path = self._get_series_context_path()
        series_glossary = load_series_glossary(series_path) if series_path else []
        # Series glossary first so path-based glossary (e.g. urban_immortal_cultivator) wins over translator param.
        glossary_entries = merge_glossary_no_dupes(
            series_glossary,
            proj_as_tuples,
            translator_glossary,
        )
        # Log when series context path is active (count only; actual used/corrected terms are logged in _check_glossary_terms_in_translations).
        series_id = (getattr(self, "_translation_series_context_path", None) or "").strip() or (self.get_param_value("series_context_path") or "").strip()
        if series_id and series_glossary:
            n_terms = len(series_glossary)
            last_key = getattr(self, "_last_series_glossary_log_key", None)
            cur_key = (series_id, n_terms)
            if last_key != cur_key:
                self.logger.info(
                    "Translation series context (path=%s): %d glossary terms loaded",
                    series_id,
                    n_terms,
                )
                self._last_series_glossary_log_key = cur_key
            else:
                self.logger.debug(
                    "Translation series context (path=%s): %d glossary terms (same as previous request)",
                    series_id,
                    n_terms,
                )
        if glossary_entries:
            # Section 18: Rosetta-style models get glossary as JSON list for systematic injection
            model_name = (self.override_model or self.model or "").strip()
            if ": " in model_name:
                model_name = model_name.split(": ", 1)[1]
            is_video_ctx = self._page_key_subtitle_batch_pipeline(page_key)
            if is_video_ctx:
                current_sources = getattr(self, "_video_current_sources", None) or []
                glossary_entries = self._select_video_glossary_entries(
                    glossary_entries,
                    current_sources,
                    max_terms=_MAX_VIDEO_GLOSSARY_TERMS,
                )
            if "rosetta" in model_name.lower():
                if is_video_ctx:
                    budget = _MAX_VIDEO_GLOSSARY_SYSTEM_CHARS - 80
                    acc: List[Dict[str, str]] = []
                    for s, t in glossary_entries:
                        item = {"source": s, "target": t}
                        trial = json.dumps(acc + [item], ensure_ascii=False)
                        if len(trial) > budget:
                            if not acc:
                                acc.append(item)  # keep at least one term
                            break
                        acc.append(item)
                    gl_list = json.dumps(acc, ensure_ascii=False)
                else:
                    gl_list = json.dumps(
                        [{"source": s, "target": t} for s, t in glossary_entries], ensure_ascii=False
                    )
                parts.append(f"\n\nGlossary (use these exact translations when the source term appears):\n{gl_list}")
            else:
                gl_str = "; ".join(f"{s} -> {t}" for s, t in glossary_entries)
                if is_video_ctx and len(gl_str) > _MAX_VIDEO_GLOSSARY_SYSTEM_CHARS:
                    gl_str = gl_str[: _MAX_VIDEO_GLOSSARY_SYSTEM_CHARS - 24].rstrip() + " ... [truncated]"
                parts.append(f"\n\nUse these exact translations for terms when they appear: {gl_str}")
        series = (self.get_param_value("series_context_prompt") or "").strip()
        if series:
            parts.append(f"\n\nContext: {series}")
        video_glossary = (getattr(self, "_video_glossary_hint", None) or "").strip()
        if video_glossary:
            vg = video_glossary
            if self._page_key_subtitle_batch_pipeline(page_key) and len(vg) > _MAX_VIDEO_GLOSSARY_SYSTEM_CHARS:
                vg = vg[: _MAX_VIDEO_GLOSSARY_SYSTEM_CHARS - 24].rstrip() + " ... [truncated]"
            parts.append(
                f"\n\nVideo translator context/glossary (use for terminology and consistency):\n{vg}"
            )
        if page_key.startswith("video_"):
            parts.append(
                "\nPreserve speaker perspective: when the source or the preceding subtitles are first person (e.g. I'm back, I, my; 我, 俺 in Chinese), "
                "translate as first person. Use my not his, I not he when the speaker is referring to themselves (e.g. 'Because of his talent' → 'Because of my talent' when the protagonist is speaking). Do not use third person for the speaker."
            )
            parts.append(
                "\nMaintain tense continuity with nearby subtitles: do not randomly switch between present and past. "
                "If surrounding lines are in past narrative (was/were/did), keep past unless the source clearly indicates present; "
                "if surrounding lines are immediate dialogue in present, keep present."
            )
            parts.append(
                "\nPronoun consistency rule: keep the same speaker viewpoint as recent context. "
                "Do not switch I/me/my to he/him/his (or vice versa) unless the source explicitly changes subject. "
                "Resolve omitted subjects from context to avoid ambiguous or wrong pronouns."
            )
            if not self.get_param_value("video_allow_revised_previous"):
                parts.append("\nDo not output revised_previous; translate only the current line(s).")
        return "\n".join(parts)

    def _select_video_glossary_entries(
        self,
        entries: List[Tuple[str, str]],
        current_sources: List[str],
        max_terms: int = _MAX_VIDEO_GLOSSARY_TERMS,
    ) -> List[Tuple[str, str]]:
        """
        For video_* requests, keep glossary compact:
        - prioritize entries whose source/target appears in current subtitle sources
        - cap final glossary size to max_terms
        """
        if not entries:
            return []
        max_terms = max(1, int(max_terms or _MAX_VIDEO_GLOSSARY_TERMS))
        if len(entries) <= max_terms:
            return entries

        haystack = " ".join((s or "").strip().lower() for s in (current_sources or []) if (s or "").strip())
        if not haystack:
            return entries[:max_terms]

        matched: List[Tuple[str, str]] = []
        unmatched: List[Tuple[str, str]] = []
        for src, tgt in entries:
            s = (src or "").strip().lower()
            t = (tgt or "").strip().lower()
            if not s and not t:
                continue
            # Ignore single-char tokens for fuzzy matching to reduce accidental hits.
            s_hit = len(s) >= 2 and s in haystack
            t_hit = len(t) >= 2 and t in haystack
            if s_hit or t_hit:
                matched.append((src, tgt))
            else:
                unmatched.append((src, tgt))

        out = matched[:max_terms]
        if len(out) < max_terms:
            unmatched_budget = max_terms - len(out)
            # When nothing in the merged glossary matches current cues, padding to max_terms
            # with arbitrary series entries wastes prompt tokens (~15–25 tokens/term) and makes
            # LM Studio's prompt-cache path churn on every slightly different user message.
            if not matched and haystack:
                unmatched_budget = min(unmatched_budget, 10)
            out.extend(unmatched[:unmatched_budget])

        try:
            if len(entries) > len(out):
                self.logger.debug(
                    "Video glossary filter: selected %d/%d terms (%d matched current sources, cap=%d)",
                    len(out), len(entries), min(len(matched), max_terms), max_terms,
                )
        except Exception:
            pass
        return out

    def _get_effective_glossary_entries(self) -> List[Tuple[str, str]]:
        """Return merged glossary (series + project + translator) for checks."""
        translator_glossary = self._get_glossary_entries()
        proj_glossary = getattr(self, "_translation_project_glossary", None) or []
        proj_as_tuples = [
            (g["source"], g["target"])
            for g in proj_glossary
            if isinstance(g, dict) and g.get("source") and g.get("target")
        ]
        series_path = self._get_series_context_path()
        series_glossary = load_series_glossary(series_path) if series_path else []
        return merge_glossary_no_dupes(
            series_glossary,
            proj_as_tuples,
            translator_glossary,
        )

    # Video/subtitle: known wrong phrasings for glossary terms (source -> wrong alternatives to replace with glossary target).
    _VIDEO_GLOSSARY_WRONG_ALTERNATIVES: Dict[str, List[str]] = {
        "心魔劫": ["mental demons", "mental demon", "inner demons", "battle against the mental demons"],
    }

    def _wrong_alternatives_for_glossary_term(self, term_tgt: str, term_src: str = "") -> List[str]:
        """Return possible wrong translations to replace with term_tgt (heuristics + video-specific list)."""
        t = (term_tgt or "").strip()
        out = []
        # Video: use known wrong alternatives for common terms (e.g. 心魔劫 -> "heart demon tribulation" vs "mental demons").
        if term_src and term_src in self._VIDEO_GLOSSARY_WRONG_ALTERNATIVES:
            out.extend(self._VIDEO_GLOSSARY_WRONG_ALTERNATIVES[term_src])
        if not t:
            return out
        # Heuristic: single-word proper noun often mistranslated as "x province" (e.g. Chuzhou -> chu province)
        if " " not in t and len(t) > 1:
            low = t[0].lower() + t[1:]
            out.extend([low + " province", t[0] + t[1:].lower() + " province"])
        return out

    # Common 2-word phrases we must not replace (would break the sentence).
    _STOPLIST_2W = frozenset(
        s.lower() for s in (
            "this is", "is not", "it is", "it was", "that is", "that was", "there is", "there was",
            "to be", "of the", "in the", "on the", "at the", "for the", "with the", "as a", "can be",
            "will be", "would be", "could be", "have been", "has been", "we are", "they are",
            "you are", "i am", "he is", "she is", "what is", "who is", "how is", "not a", "is a",
            "was a", "are a", "were a", "has a", "have a", "had a", "been a", "into the", "from the",
        )
    )

    def _find_auto_wrong_phrase(self, trans_text: str, target: str) -> Optional[str]:
        """Find a phrase in trans_text that looks like a wrong variant of target (same last word).
        E.g. target 'mental demons' -> find 'heart demon' or 'inner demons' and return that substring."""
        t = (target or "").strip()
        if not t or not trans_text:
            return None
        words = t.split()
        if len(words) < 2:
            return None
        last_word = words[-1]
        # Match "X lastword" or "X lastword_without_s" (e.g. demon when target is demons)
        last_esc = re.escape(last_word)
        if len(last_word) > 1 and last_word.endswith("s"):
            alt = last_word[:-1]
            pattern = r"\b\w+\s+(?:" + last_esc + r"|" + re.escape(alt) + r")\b"
        else:
            pattern = r"\b\w+\s+" + last_esc + r"\b"
        m = re.search(pattern, trans_text, re.IGNORECASE)
        return m.group(0) if m else None

    def _find_any_n_word_phrase_to_replace(
        self, trans_text: str, target: str, n: int
    ) -> Optional[str]:
        """When source had the glossary term but translation used something else (e.g. 'bird dog'
        instead of 'mental demons'), find an n-word phrase that shares at least one word with target.
        Skips stoplist phrases. Used as fallback when _find_auto_wrong_phrase finds nothing."""
        if not trans_text or not target or n < 1:
            return None
        t = (target or "").strip()
        if t.lower() in trans_text.lower():
            return None
        target_words = set(t.lower().split())
        # Build regex for exactly n words: \b\w+(\s+\w+){n-1}\b
        pattern = r"\b\w+" + (r"\s+\w+" * (n - 1)) + r"\b" if n > 1 else r"\b\w+\b"
        stoplist = self._STOPLIST_2W if n == 2 else frozenset()
        for m in re.finditer(pattern, trans_text, re.IGNORECASE):
            phrase = m.group(0)
            if phrase.lower() == t.lower():
                continue
            if n == 2 and phrase.lower() in stoplist:
                continue
            # Skip article + noun (e.g. "a bird") so we replace content phrases like "bird dog"
            if n == 2 and (phrase.lower().startswith("a ") or phrase.lower().startswith("the ")):
                continue
            # Only replace if phrase shares at least one word with target (avoids "Having fought" -> "Immortal Realm")
            phrase_words = set(re.findall(r"\w+", phrase.lower()))
            if not (phrase_words & target_words):
                continue
            return phrase
        return None

    def _allow_glossary_replacement(
        self,
        phrase: str,
        term_tgt: str,
        term_src: str,
        glossary: List[Tuple[str, str]],
        trans_text: Optional[str] = None,
        pos: Optional[int] = None,
    ) -> bool:
        """Return False if we must not replace phrase with term_tgt (e.g. phrase is another glossary target or same-source alternate).
        When trans_text and pos are given, only skip 'phrase is substring of other target' if the full other target
        actually appears at pos (so we don't block e.g. 'heart demon' -> 'mental demons' when the text is just
        'heart demon' and not 'heart demon tribulation')."""
        if not phrase or not term_tgt:
            return False
        pl = phrase.strip().lower()
        # Skip garbage/error phrases (e.g. "Heart no match", placeholders)
        if "no match" in pl or (len(pl) <= 3 and not pl.replace("'", "").isalnum()):
            return False
        # Phrase must not be any other source's target (would overwrite a correct term, e.g. Heavenly Tribulation)
        other_targets = {t.strip().lower() for s, t in glossary if s != term_src and t}
        if pl in other_targets:
            return False
        # Phrase must not be a substring of another source's target *when that full target appears here*
        # (e.g. don't replace "heart demon" with "mental demons" when text at pos is "heart demon tribulation")
        # If we don't have pos/trans_text, keep conservative: skip whenever phrase is substring of any other target
        if trans_text is not None and pos is not None:
            for ot in other_targets:
                if pl in ot and len(ot) > len(pl):
                    end = pos + len(ot)
                    if end <= len(trans_text) and trans_text[pos:end].lower() == ot:
                        return False
        else:
            if any(pl in ot for ot in other_targets):
                return False
        # Phrase must not be another accepted target for this same source (glossary can list multiple targets per term)
        same_src_targets = {t.strip().lower() for s, t in glossary if s == term_src and t}
        if pl in same_src_targets:
            return False
        return True

    def _translation_effectively_empty(self, s: str) -> bool:
        """True if translation is empty or only punctuation/whitespace (no point glossary-checking)."""
        t = s.strip()
        if not t or len(t) < 2:
            return True
        return not any(c.isalnum() for c in t)

    def _translation_contains_target_words_in_order(self, trans_norm: str, tgt_stripped: str) -> bool:
        """True if all words of target appear in translation in order (case-insensitive, as substrings)."""
        words = tgt_stripped.strip().lower().split()
        if not words:
            return True
        t = trans_norm.lower()
        pos = 0
        for w in words:
            idx = t.find(w, pos)
            if idx == -1:
                return False
            pos = idx + len(w)
        return True

    def _check_glossary_terms_in_translations(
        self, src_list: List[str], trans_list: List[str]
    ) -> None:
        """Check and correct: when a glossary term was in source but translation doesn't use the expected target,
        try to replace wrong alternatives so layout/auto-layout see the correct terms."""
        glossary = self._get_effective_glossary_entries()
        if not glossary:
            return
        page_key = getattr(self, "_current_page_key", None)
        # Video: merged series glossary can be 1000+ entries; scanning all per cue is slow and redundant
        # with the capped glossary already injected into the system prompt.
        if page_key and self._page_key_subtitle_batch_pipeline(str(page_key)):
            glossary = self._select_video_glossary_entries(
                glossary, src_list, max_terms=_MAX_VIDEO_GLOSSARY_TERMS
            )
            if not glossary:
                return
        # Process longer source terms first so e.g. 心魔劫 is applied before 心魔 (avoids "mental heart demon tribulation")
        glossary = sorted(glossary, key=lambda x: len((x[0] or "").strip()), reverse=True)
        page_suffix = f" (page: {page_key})" if page_key else ""
        for i, (src_text, trans_text) in enumerate(zip(src_list, trans_list)):
            if not src_text or not isinstance(trans_text, str):
                continue
            trans_norm = trans_text.strip()
            if self._translation_effectively_empty(trans_norm):
                continue
            # When the whole block source is exactly one glossary term, use glossary target if model returned something else (e.g. name -> "I'm back.")
            src_stripped = src_text.strip()
            exact_override = False
            for term_src, term_tgt in glossary:
                if (term_src or "").strip() == src_stripped:
                    primary = ((term_tgt or "").strip().split("|")[0].strip()
                               if "|" in (term_tgt or "") else (term_tgt or "").strip())
                    if primary and primary.lower() not in trans_norm.lower():
                        trans_list[i] = primary
                        self.logger.info(
                            "Glossary term applied (exact block source): '%s' -> '%s'%s",
                            term_src, primary, page_suffix,
                        )
                        exact_override = True
                    break
            if exact_override:
                continue
            new_trans = trans_text
            for term_src, term_tgt in glossary:
                if not term_src or not term_tgt:
                    continue
                if term_src not in src_text:
                    continue
                # If this source term is part of a longer glossary source that also appears in src_text,
                # and that longer term's target is already present in the translation, treat this term as satisfied.
                # Example: source has both '心魔' -> 'mental demons' and '心魔劫' -> 'heart demon tribulation'.
                # When the line actually contains '心魔劫' and the translation has 'heart demon tribulation',
                # we should not warn that '心魔' is missing.
                covered_by_longer = False
                for other_src, other_tgt in glossary:
                    if other_src == term_src or not other_src or not other_tgt:
                        continue
                    if term_src in other_src and other_src in src_text:
                        other_tgt_stripped = other_tgt.strip()
                        if not other_tgt_stripped:
                            continue
                        other_no_sp = other_tgt_stripped.lower().replace(" ", "")
                        tn_lower = trans_norm.lower()
                        if (
                            other_tgt_stripped in trans_norm
                            or other_tgt_stripped.lower() in tn_lower
                            or (other_no_sp and other_no_sp in tn_lower.replace(" ", ""))
                        ):
                            covered_by_longer = True
                            break
                if covered_by_longer:
                    continue
                tgt_stripped = term_tgt.strip()
                # Allow pipe-separated alternates in glossary (e.g. "Chu Province | Chuzhou"); 楚州: accept Chuzhou as correct
                acceptable = [s.strip() for s in tgt_stripped.split("|") if s.strip()]
                if not acceptable:
                    acceptable = [tgt_stripped]
                if term_src.strip() == "楚州" and "Chuzhou" not in [a for a in acceptable if a]:
                    acceptable = list(acceptable) + ["Chuzhou"]
                if term_src.strip() == "天劫":
                    for alt in ("Tribulation", "Tribulation stage"):
                        if alt not in acceptable:
                            acceptable = list(acceptable) + [alt]
                if term_src.strip() == "仙界" and "realm of immortality" not in acceptable:
                    acceptable = list(acceptable) + ["realm of immortality"]
                satisfied = False
                for alt in acceptable:
                    if not alt:
                        continue
                    if alt in trans_norm or alt.lower() in trans_norm.lower():
                        satisfied = True
                        break
                    alt_no_sp = alt.lower().replace(" ", "")
                    if alt_no_sp and alt_no_sp in trans_norm.lower().replace(" ", ""):
                        satisfied = True
                        break
                    if self._translation_contains_target_words_in_order(trans_norm, alt):
                        satisfied = True
                        break
                if satisfied:
                    continue
                # Use first acceptable target when replacing wrong forms (e.g. "Chu Province | Chuzhou" -> use first)
                primary_tgt = acceptable[0] if acceptable else tgt_stripped
                # Try to replace wrong forms: first heuristics (e.g. "x province"), then video-specific (e.g. 心魔劫), then auto-detect
                replaced = False
                for wrong in self._wrong_alternatives_for_glossary_term(primary_tgt, term_src or ""):
                    if not wrong:
                        continue
                    if wrong.lower() in new_trans.lower():
                        idx = new_trans.lower().find(wrong.lower())
                        if idx >= 0 and not self._allow_glossary_replacement(
                            wrong, primary_tgt, term_src, glossary, new_trans, idx
                        ):
                            continue
                        if idx >= 0:
                            new_trans = new_trans[:idx] + primary_tgt + new_trans[idx + len(wrong):]
                            replaced = True
                            self.logger.info(
                                "Glossary term corrected: '%s' -> '%s' (replaced '%s' in translation)%s",
                                term_src, primary_tgt, wrong, page_suffix,
                            )
                            break
                if not replaced:
                    # Auto-detect: find phrase that matches target's last word (e.g. "heart demon" for "mental demons")
                    auto_phrase = self._find_auto_wrong_phrase(new_trans, primary_tgt)
                    # Skip when phrase and target have same word count and same last word (e.g. "true disciple" vs "official disciple") - treat as valid alternate
                    same_shape = False
                    if auto_phrase and primary_tgt:
                        aw = auto_phrase.strip().lower().split()
                        tw = primary_tgt.strip().lower().split()
                        same_shape = len(aw) == len(tw) and aw and tw and aw[-1] == tw[-1]
                    pos_auto = new_trans.lower().find(auto_phrase.lower()) if auto_phrase else -1
                    if auto_phrase and not same_shape and pos_auto >= 0 and self._allow_glossary_replacement(
                        auto_phrase, primary_tgt, term_src, glossary, new_trans, pos_auto
                    ):
                        pos = pos_auto
                        if pos >= 0:
                            actual = new_trans[pos : pos + len(auto_phrase)]
                            new_trans = new_trans[:pos] + primary_tgt + new_trans[pos + len(actual):]
                            replaced = True
                            self.logger.info(
                                "Glossary term corrected: '%s' -> '%s' (replaced '%s' in translation)%s",
                                term_src, term_tgt, actual, page_suffix,
                            )
                if not replaced:
                    # Fallback: only replace n-word phrase that shares a word with target (avoids "Having fought" -> "Immortal Realm")
                    n = len(primary_tgt.split())
                    if 2 <= n <= 3:
                        any_phrase = self._find_any_n_word_phrase_to_replace(new_trans, primary_tgt, n)
                        pos_fb = new_trans.lower().find(any_phrase.lower()) if any_phrase else -1
                        if any_phrase and pos_fb >= 0 and self._allow_glossary_replacement(
                            any_phrase, primary_tgt, term_src, glossary, new_trans, pos_fb
                        ):
                            pos = pos_fb
                            if pos >= 0:
                                actual = new_trans[pos : pos + len(any_phrase)]
                                new_trans = new_trans[:pos] + primary_tgt + new_trans[pos + len(actual):]
                                replaced = True
                                self.logger.info(
                                    "Glossary term corrected: '%s' -> '%s' (replaced '%s' in translation)%s",
                                    term_src, primary_tgt, actual, page_suffix,
                                )
                if replaced:
                    trans_list[i] = new_trans
                    trans_norm = new_trans.strip()
                else:
                    snippet = (trans_norm[:80] + "…") if len(trans_norm) > 80 else trans_norm
                    self.logger.warning(
                        "Glossary term not applied: source '%s' expected '%s' in translation; got: %s%s",
                        term_src, term_tgt, snippet, page_suffix,
                    )

    @property
    def context_previous_pages_count(self) -> int:
        try:
            return max(0, min(5, int(self.get_param_value("context_previous_pages"))))
        except (TypeError, ValueError):
            return 0

    def _initialize_client(self, api_key_to_use: str, provider_override: Optional[str] = None) -> bool:
        provider = provider_override or self.provider
        endpoint = resolve_endpoint(provider, self.endpoint)

        proxy = self.proxy
        # Reuse existing client when connection config is unchanged.
        # Serialize creation so parallel video NLP workers do not double-init or race httpx/OpenAI client.
        sig = (provider, endpoint, proxy or "", api_key_to_use)
        lock = getattr(self, "_client_init_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._client_init_lock = lock
        with lock:
            if getattr(self, "client", None) is not None and getattr(self, "_client_signature", None) == sig:
                return True

            timeout = httpx.Timeout(30.0, read=120.0)
            try:
                http_client = create_httpx_client(proxy, timeout=timeout)
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize proxy '{proxy}': {e}. Proceeding without proxy."
                )
                http_client = httpx.Client(timeout=timeout)

            masked_key = (
                api_key_to_use[:4] + "..." + api_key_to_use[-4:]
                if len(api_key_to_use) > 8
                else api_key_to_use
            )
            self.logger.debug(
                f"Initializing client for {provider} with key {masked_key} at endpoint {endpoint}"
            )

            try:
                self.client = openai.OpenAI(
                    api_key=api_key_to_use, base_url=endpoint, http_client=http_client
                )
                self._client_signature = sig
                return True
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
                self._client_signature = None
                return False

    def is_deterministic(self) -> bool:
        try:
            return float(self.temperature) <= 0.0
        except Exception:
            return False

    def _effective_provider_for_model(self) -> str:
        """
        If user picked a model with a provider prefix (e.g. "GGL: ...") but provider selector mismatches,
        route to the implied provider to avoid sending Gemini model names to OpenAI endpoints, etc.
        """
        try:
            return effective_provider(self.provider, self.model, self.override_model)
        except ValueError:
            return self.provider
        except Exception:
            return self.provider

    def _build_generation_config(self, provider: str, model_name: str) -> dict:
        """
        Section 18: Central (provider, model) -> API param names, reasoning toggles, max token caps.
        Returns dict to merge into chat completion api_args.
        """
        cfg = {}
        if provider == "OpenAI":
            cfg["frequency_penalty"] = self.frequency_penalty
            cfg["presence_penalty"] = self.presence_penalty
        if int(self.top_k or 0) > 0 and provider in {"OpenRouter", "Ollama", "LLM Studio", "Grok"}:
            extra = cfg.get("extra_body")
            if not isinstance(extra, dict):
                extra = {}
            extra["top_k"] = int(self.top_k)
            cfg["extra_body"] = extra
        # Do not set max_tokens here.
        # _request_translation() already computes a tighter per-request cap via
        # _translation_completion_max_tokens(provider, expected_count). Overriding it
        # with the global user cap (often very high, e.g. 20k) slows local servers.
        return cfg

    def _reasoning_effort(self) -> str:
        v = (self.get_param_value("reasoning_effort") or "medium").strip().lower()
        if v not in {"low", "medium", "high"}:
            return "medium"
        return v

    def _apply_reasoning_controls(self, api_args: dict, provider: str, model_name: str) -> None:
        """
        Add opt-in reasoning controls for providers/models that may support them.
        These keys are best-effort: callers should retry without them if rejected.
        """
        if not bool(self.get_param_value("enable_reasoning")):
            return
        effort = self._reasoning_effort()
        try:
            if provider in {"OpenAI", "Grok", "OpenRouter", "LLM Studio", "Ollama"}:
                # Use extra_body passthrough for broad OpenAI SDK compatibility.
                extra = api_args.get("extra_body")
                if not isinstance(extra, dict):
                    extra = {}
                extra["reasoning"] = {"effort": effort}
                api_args["extra_body"] = extra
            if provider == "OpenRouter":
                # Ask OpenRouter for reasoning-enabled mode when available.
                extra = api_args.get("extra_body")
                if not isinstance(extra, dict):
                    extra = {}
                extra["include_reasoning"] = True
                api_args["extra_body"] = extra
        except Exception:
            pass

    @staticmethod
    def _adapt_unsupported_params(api_args: dict, err: Exception):
        """
        Rewrite request arguments a model rejects, or return None when the error is
        about something else.

        Newer OpenAI-style models (GPT-5 and later) only accept
        ``max_completion_tokens`` and refuse a non-default ``temperature``/``top_p``,
        which otherwise makes them unusable
        (upstream dmMaze/BallonsTranslator#1251).
        """
        msg = str(err).lower()
        if not any(
            marker in msg
            for marker in ("unsupported parameter", "unsupported value", "is not supported with this model",
                           "does not support", "only the default")
        ):
            return None

        adapted = dict(api_args or {})
        changed = []
        if "max_tokens" in msg and "max_tokens" in adapted and "max_completion_tokens" not in adapted:
            adapted["max_completion_tokens"] = adapted.pop("max_tokens")
            changed.append("max_tokens -> max_completion_tokens")
        for key in ("temperature", "top_p"):
            if key in msg and key in adapted:
                adapted.pop(key, None)
                changed.append(f"dropped {key}")
        if not changed:
            return None
        return adapted, ", ".join(changed)

    def _create_completion_adapting_params(self, api_args: dict, max_attempts: int = 3):
        """chat.completions.create, retrying with parameters the model actually accepts."""
        attempt = 0
        while True:
            try:
                return self.client.chat.completions.create(**api_args), api_args
            except Exception as e:
                attempt += 1
                adapted = self._adapt_unsupported_params(api_args, e) if attempt < max_attempts else None
                if adapted is None:
                    raise
                api_args, changed = adapted
                self.logger.warning("Model rejected request parameters (%s); retrying with %s.", e, changed)

    def _is_reasoning_param_error(self, err: Exception) -> bool:
        s = str(err).lower()
        return (
            "reasoning" in s
            or "include_reasoning" in s
            or "unknown argument" in s
            or "unknown field" in s
            or "extra inputs are not permitted" in s
            or "unexpected keyword argument" in s
        )

    def _strip_reasoning_args(self, api_args: dict) -> dict:
        clean = dict(api_args or {})
        extra = clean.get("extra_body")
        if isinstance(extra, dict):
            extra = dict(extra)
            extra.pop("reasoning", None)
            extra.pop("include_reasoning", None)
            if extra:
                clean["extra_body"] = extra
            else:
                clean.pop("extra_body", None)
        return clean

    # --- Property getters ---
    @property
    def provider(self) -> str:
        return self.get_param_value("provider")

    @property
    def apikey(self) -> str:
        return self.get_param_value("apikey")

    @property
    def multiple_keys_list(self) -> List[str]:
        keys_str = self.get_param_value("multiple_keys")
        if not isinstance(keys_str, str):
            return []
        return [
            key.strip()
            for key in keys_str.strip().replace("\n", ";").split(";")
            if key.strip()
        ]

    @property
    def model(self) -> str:
        return self.get_param_value("model")

    @property
    def override_model(self) -> Optional[str]:
        return self.get_param_value("override model") or None

    @property
    def endpoint(self) -> Optional[str]:
        return self.get_param_value("endpoint") or None

    @property
    def temperature(self) -> float:
        return float(self.get_param_value("temperature"))

    @property
    def top_p(self) -> float:
        return float(self.get_param_value("top p"))

    @property
    def top_k(self) -> int:
        return int(self.get_param_value("top k") or 0)

    @property
    def max_tokens(self) -> int:
        return int(self.get_param_value("max tokens"))

    @property
    def retry_attempts(self) -> int:
        return int(self.get_param_value("retry attempts"))

    @property
    def retry_timeout(self) -> int:
        return int(self.get_param_value("retry timeout"))

    @property
    def rate_limit_delay(self) -> int:
        return max(30, int(self.get_param_value("rate limit delay") or 60))

    @property
    def proxy(self) -> str:
        return self.get_param_value("proxy")

    @property
    def system_prompt(self) -> str:
        return self.get_param_value("system_prompt")

    @property
    def invalid_repeat_count(self) -> int:
        return int(self.get_param_value("invalid repeat count"))

    def _has_repetition_hallucination(self, texts: List[str], threshold: int) -> bool:
        """True if any translation has threshold+ consecutive identical characters (e.g. 啊啊啊...)."""
        if threshold <= 0:
            return False
        for t in texts:
            if not t or len(t) < threshold:
                continue
            run = 1
            for i in range(1, len(t)):
                if t[i] == t[i - 1]:
                    run += 1
                    if run >= threshold:
                        return True
                else:
                    run = 1
        return False

    def _target_lang_ratio(self, texts: List[str]) -> float:
        """Return fraction of characters that look like target language. Heuristic only."""
        target = (self.lang_target or "").strip()
        if not target or not texts:
            return 1.0
        full = "".join(texts)
        if not full:
            return 1.0
        total = len(full)
        if total == 0:
            return 1.0
        # CJK ranges (simplified)
        def is_cjk(c):
            o = ord(c)
            return (
                0x4E00 <= o <= 0x9FFF
                or 0x3040 <= o <= 0x309F
                or 0x30A0 <= o <= 0x30FF
                or 0xAC00 <= o <= 0xD7AF
            )
        def is_letter(c):
            return c.isalpha()
        if target in ("简体中文", "繁體中文", "日本語", "한국어"):
            in_lang = sum(1 for c in full if is_cjk(c) or (c.isalnum() and not c.isascii()))
        else:
            in_lang = sum(1 for c in full if is_letter(c) or c.isspace() or c in ".,?!;:'\"-")
        return in_lang / total if total else 1.0

    def _post_check_fail(self, ordered_translations: List[str]) -> bool:
        """True if we should retry due to repetition or low target-language ratio."""
        if not self.get_param_value("post_translation_check"):
            return False
        try:
            rep_chars = int(self.get_param_value("post_check_repetition_chars") or 20)
        except (TypeError, ValueError):
            rep_chars = 20
        if self._has_repetition_hallucination(ordered_translations, rep_chars):
            self.logger.warning("Post-check failed: repetition (hallucination) detected.")
            return True
        try:
            ratio = float(self.get_param_value("post_check_target_ratio") or 0)
        except (TypeError, ValueError):
            ratio = 0
        if ratio <= 0:
            return False
        r = self._target_lang_ratio(ordered_translations)
        if r < ratio:
            self.logger.warning("Post-check failed: target-language ratio %.2f < %.2f.", r, ratio)
            return True
        return False

    def _line_post_check_fail(self, line: str) -> bool:
        """True if a single subtitle line fails repetition or target-language ratio (video per-line retry)."""
        if not self.get_param_value("post_translation_check"):
            return False
        if not (line or "").strip():
            return False
        try:
            rep_chars = int(self.get_param_value("post_check_repetition_chars") or 20)
        except (TypeError, ValueError):
            rep_chars = 20
        if self._has_repetition_hallucination([line], rep_chars):
            return True
        try:
            ratio = float(self.get_param_value("post_check_target_ratio") or 0)
        except (TypeError, ValueError):
            ratio = 0
        if ratio <= 0:
            return False
        r = self._target_lang_ratio([line])
        return r < ratio

    def _apply_per_line_post_check_retries(
        self,
        src_list: List[str],
        ordered_translations: List[str],
        to_lang: str,
        num_src: int,
    ) -> List[str]:
        """
        For video_* page keys: re-translate only lines that still fail post-check after batch retries.
        Avoids whole-batch-only recovery when one cue is wrong language or garbled.
        """
        if getattr(self, "_inside_per_line_post_check", False):
            return ordered_translations
        page_key = getattr(self, "_current_page_key", "") or ""
        if not self._page_key_subtitle_batch_pipeline(page_key):
            return ordered_translations
        if not self.get_param_value("post_translation_check"):
            return ordered_translations
        try:
            max_post = int(self.get_param_value("post_check_max_retries") or 0)
        except (TypeError, ValueError):
            max_post = 0
        if max_post <= 0:
            return ordered_translations

        max_per_line = 2
        max_total_calls = min(max(num_src * 2, max_post * 4), 48)
        total_calls = 0
        line_retries: dict = {}
        out = list(ordered_translations)

        for _sweep in range(4):
            bad = [i for i in range(num_src) if i < len(out) and self._line_post_check_fail(out[i])]
            if not bad:
                break
            for i in bad:
                if total_calls >= max_total_calls:
                    return out
                if line_retries.get(i, 0) >= max_per_line:
                    continue
                line_retries[i] = line_retries.get(i, 0) + 1
                total_calls += 1
                src_one = src_list[i] if i < len(src_list) else ""
                self.logger.info(
                    "Post-check per-line retry for cue %d/%d (video).",
                    i + 1,
                    num_src,
                )
                time.sleep(self._subrequest_delay())
                self._inside_per_line_post_check = True
                try:
                    new_lines = self._translate_single_item_fallback(
                        [src_one],
                        to_lang,
                        out[i] if i < len(out) else "",
                    )
                finally:
                    self._inside_per_line_post_check = False
                if new_lines and len(new_lines) >= 1 and (new_lines[0] or "").strip():
                    while len(out) <= i:
                        out.append("")
                    out[i] = new_lines[0]
        return out

    def _source_has_cjk_jp_kr(self, src: str) -> bool:
        """True if text contains CJK, Japanese kana, or Korean syllables (common untranslated-echo cases)."""
        for c in src or "":
            o = ord(c)
            if 0x4E00 <= o <= 0x9FFF:
                return True
            if 0x3040 <= o <= 0x30FF:
                return True
            if 0xAC00 <= o <= 0xD7AF:
                return True
        return False

    def _likely_untranslated_echo(self, src: str, trans: str) -> bool:
        """Model sometimes returns the source line unchanged; re-translate when target should differ."""
        s = (src or "").strip()
        t = (trans or "").strip()
        if not s or not t:
            return False
        if s == t and self._source_has_cjk_jp_kr(s):
            return True
        return False

    def _apply_video_per_line_quality_fixes(
        self,
        src_list: List[str],
        ordered_translations: List[str],
        to_lang: str,
        num_src: int,
    ) -> List[str]:
        """
        For video_*: fix empty translations and obvious source-echo lines with a few per-cue API calls.
        Runs before per-line post-check retries (which need post_translation_check).
        """
        if getattr(self, "_inside_per_line_post_check", False):
            return ordered_translations
        page_key = getattr(self, "_current_page_key", "") or ""
        if not self._page_key_subtitle_batch_pipeline(page_key):
            return ordered_translations
        try:
            from utils.config import pcfg as _pc

            if not bool(
                getattr(_pc.module, "video_translator_llm_per_line_quality_fix", True)
            ):
                return ordered_translations
        except Exception:
            pass

        out = list(ordered_translations)
        max_fixes = min(24, max(4, num_src))
        fixes = 0
        for i in range(num_src):
            if fixes >= max_fixes:
                break
            src_one = src_list[i] if i < len(src_list) else ""
            if not (src_one or "").strip():
                continue
            cur = out[i] if i < len(out) else ""
            need = not (cur or "").strip() or self._likely_untranslated_echo(src_one, cur)
            if not need:
                continue
            fixes += 1
            self.logger.info(
                "Video per-line quality fix for cue %d/%d (empty or source-echo).",
                i + 1,
                num_src,
            )
            time.sleep(self._subrequest_delay())
            self._inside_per_line_post_check = True
            try:
                new_lines = self._translate_single_item_fallback(
                    [src_one],
                    to_lang,
                    cur if (cur or "").strip() else "",
                )
            finally:
                self._inside_per_line_post_check = False
            if new_lines and len(new_lines) >= 1 and (new_lines[0] or "").strip():
                while len(out) <= i:
                    out.append("")
                out[i] = new_lines[0]
        return out

    def _extract_and_append_glossary(
        self, sources: List[str], translations: List[str]
    ) -> None:
        """Optional: one extra API call to extract terms from source->translation pairs and append to series glossary."""
        if not sources or not translations or len(sources) != len(translations):
            return
        series_path = self._get_series_context_path()
        if not series_path:
            return
        pairs = [
            f"  {i+1}. [{s.strip()}] -> [{t.strip()}]"
            for i, (s, t) in enumerate(zip(sources, translations))
            if (s or "").strip() and (t or "").strip()
        ]
        if not pairs:
            return
        to_lang = self.lang_map.get(self.lang_target, self.lang_target)
        user = (
            "From the following source text and translation pairs, extract important terms "
            "that should stay consistent (names, places, skills, items, etc.). "
            "Output valid JSON only, no other text: {\"terms\": [{\"source\": \"original\", \"target\": \"translation\"}]}.\n\n"
            f"Target language: {to_lang}.\n\nPairs:\n" + "\n".join(pairs[:30])
        )
        messages = [
            {"role": "system", "content": "You extract terminology for a glossary. Output JSON only."},
            {"role": "user", "content": user},
        ]
        try:
            raw = self._request_raw_completion(messages, max_tokens=1024)
            if not raw or not raw.strip():
                return
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(raw[start : end + 1])
                terms = data.get("terms") or data.get("translations") or []
            else:
                return
            entries = []
            for t in terms:
                if isinstance(t, dict):
                    s, tg = t.get("source"), t.get("target")
                    if s and tg and isinstance(s, str) and isinstance(tg, str):
                        entries.append((s.strip(), tg.strip()))
            if entries:
                append_to_series_glossary(series_path, entries)
                self.logger.info("Extracted %d terms and appended to series glossary.", len(entries))
        except Exception as e:
            self.logger.debug("Glossary extraction failed (non-fatal): %s", e)

    @property
    def frequency_penalty(self) -> float:
        return float(self.get_param_value("frequency penalty"))

    @property
    def presence_penalty(self) -> float:
        return float(self.get_param_value("presence penalty"))

    @property
    def max_rpm(self) -> int:
        return int(self.get_param_value("max requests per minute"))

    @property
    def global_delay(self) -> float:
        return float(self.get_param_value("delay"))

    def _subrequest_delay(self) -> float:
        """Sleep between rapid follow-up LLM calls (per-line video fix, single-item fallback loop).

        Cloud APIs benefit from ``delay``; a local LM Studio server typically does not, and sleeping
        here makes video runs feel much slower when many cues need a fix.
        """
        if getattr(self, "provider", None) in {"LLM Studio", "Ollama"}:
            return 0.0
        return max(0.0, self.global_delay)

    # ------------------------------------------------------------------
    # Context-aware translation (prior pages as cache-friendly message pairs)
    # Ported from upstream dmMaze/BallonsTranslator (llm_context series).
    # ------------------------------------------------------------------

    def history_mode_enabled(self) -> bool:
        """True when prior pages are sent as real message pairs instead of prompt text."""
        from utils.config import LLMTranslateContext, pcfg
        return getattr(pcfg.module, 'llm_translate_context', LLMTranslateContext.PAGE) == LLMTranslateContext.HISTORY

    def _history_model_name(self) -> str:
        model = (getattr(self, "override_model", None) or self.model or "").strip()
        if ": " in model:
            model = model.split(": ", 1)[1]
        return model

    def render_history_user_prompt(self, sources: Tuple[str, ...]) -> str:
        """Stable user prompt for a history page.

        Only the parts that stay identical across requests are rendered, so the
        provider sees an unchanged prefix and can serve it from its prompt cache.
        """
        from_lang = self.lang_map.get(self.lang_source, self.lang_source)
        to_lang = self.lang_map.get(self.lang_target, self.lang_target)
        input_json_str = json.dumps(
            [{"id": i + 1, "source": s} for i, s in enumerate(sources)],
            ensure_ascii=False,
            indent=2,
        )
        return (
            f"Please translate the following text snippets from {from_lang} to {to_lang}. "
            f"The input is provided as a JSON array. Respond with a JSON object in the specified format.\n\n"
            f"INPUT:\n{input_json_str}"
        )

    @staticmethod
    def render_history_assistant_response(translations: Tuple[str, ...]) -> str:
        payload = {
            "translations": [
                {"id": index + 1, "translation": translation}
                for index, translation in enumerate(translations)
            ]
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _snapshot_history_page(self, project, page_key: str, target_language: str):
        """Copy one finished page, or None when it must not seed later requests."""
        from modules.context.history import HistoryPage
        from utils.config import RunStatus

        pages = getattr(project, "pages", None)
        image_info = getattr(project, "_image_info", None)
        if not isinstance(pages, dict) or page_key not in pages:
            return None
        if not isinstance(image_info, dict):
            return None
        info = image_info.get(page_key, {})
        if not isinstance(info, dict) or not (int(info.get("finish_code", 0)) & RunStatus.FIN_TRANSLATE):
            return None
        # Pages translated into another language must not be reused as context.
        # Missing metadata stays compatible with projects made before this feature.
        if "translation_target" in info and info["translation_target"] != target_language:
            return None

        blocks = pages[page_key]
        sources = []
        translations = []
        for block in blocks:
            source = block.get_text()
            if not source or not source.strip():
                continue
            translation = getattr(block, "translation", "")
            if not translation or not str(translation).strip():
                # A page is indivisible: never seed a partially translated one.
                return None
            sources.append(source)
            translations.append(str(translation))
        if not translations:
            return None
        return HistoryPage(
            page_key=str(page_key),
            sources=tuple(sources),
            translations=tuple(translations),
        )

    def _render_history_page(self, page, model: str):
        from modules.context.history import RenderedHistoryPage
        from modules.context.token_usage import messages_token_count

        messages = [
            {"role": "user", "content": self.render_history_user_prompt(page.sources)},
            {"role": "assistant", "content": self.render_history_assistant_response(page.translations)},
        ]
        return RenderedHistoryPage(
            snapshot=page,
            messages=tuple((str(m["role"]), str(m["content"])) for m in messages),
            token_count=messages_token_count(messages, model),
        )

    def set_translation_project(self, project, page_key: str):
        """
        Freeze the glossary and eligible page history for the page about to be
        translated. Called by the pipeline before each page.
        """
        try:
            self._llm_request_context = self._snapshot_request_context(project, page_key)
        except GlossaryError as e:
            self._llm_request_context = None
            self.logger.warning("Glossary disabled for this request: %s", e)
        except Exception as e:
            self._llm_request_context = None
            self.logger.debug("Could not snapshot translation context: %s", e, exc_info=True)

    def _snapshot_request_context(self, project, page_key):
        from utils.config import pcfg
        from modules.context.glossary import load_glossary
        from modules.context.history import (
            ContextAction,
            ContextDiagnostic,
            ContextReason,
            HistoryWindowKey,
            RequestContext,
            eligible_history_for_request,
            window_rebuild_reason,
        )

        use_history = self.history_mode_enabled()
        history_budget = max(0, int(getattr(pcfg.module, "llm_prior_context_token_budget", 4096) or 0))
        glossary_path = str(getattr(pcfg.module, "llm_glossary_path", "") or "")
        glossary_mode = getattr(pcfg.module, "llm_glossary_mode", "matching")

        if not use_history and not glossary_path:
            self._history_window = None
            return None

        glossary = load_glossary(glossary_path)
        if not use_history:
            # A glossary works on its own, but must not keep a stale history window.
            self._history_window = None

        history = ()
        window_key = None
        diagnostic = ContextDiagnostic(
            page_key=str(page_key or ""),
            action=ContextAction.DISABLED if not use_history else ContextAction.EMPTY,
            page_count=0,
            token_count=0,
            token_budget=history_budget,
            rebuild_reason=(
                ContextReason.HISTORY_DISABLED if not use_history else ContextReason.MISSING_PROJECT_PAGE
            ),
        )

        if use_history and project is not None and page_key is not None:
            model = self._history_model_name()
            window_key = HistoryWindowKey(
                load_identity=getattr(project, "load_identity", None),
                settings=(
                    ("source_language", str(self.lang_source)),
                    ("model", str(model)),
                    ("system_prompt", self._build_system_prompt()),
                    ("token_budget", history_budget),
                ),
            )
            rebuild_reason = window_rebuild_reason(
                self._history_window, project, str(page_key), window_key
            )
            previous_page = None
            if rebuild_reason is None:
                # Re-snapshot retained pages so later edits cannot leak through cached messages.
                fresh_retained = tuple(
                    self._snapshot_history_page(project, page.page_key, self.lang_target)
                    for page in self._history_window.history
                )
                if any(
                    fresh != rendered.snapshot
                    for fresh, rendered in zip(fresh_retained, self._history_window.history)
                ):
                    rebuild_reason = ContextReason.SNAPSHOT_CHANGED
                else:
                    previous_page = self._snapshot_history_page(
                        project, self._history_window.request_page_key, self.lang_target
                    )
                    if previous_page is None:
                        rebuild_reason = ContextReason.PREVIOUS_INCOMPLETE
            history, diagnostic = eligible_history_for_request(
                window=self._history_window,
                project=project,
                page_key=str(page_key),
                previous_page=previous_page,
                token_budget=history_budget,
                rebuild_reason=rebuild_reason,
                snapshot_page=lambda key: self._snapshot_history_page(project, key, self.lang_target),
                render_page=lambda page: self._render_history_page(page, model),
            )

        self.logger.debug(str(diagnostic))
        return RequestContext(
            history=history,
            glossary=glossary,
            glossary_mode=glossary_mode,
            history_budget=history_budget,
            window_key=window_key,
            request_page_key=str(page_key) if page_key is not None else None,
            diagnostic=diagnostic,
        )

    @staticmethod
    def _glossary_constraint(entries) -> str:
        from modules.context.glossary import render_glossary
        if not entries:
            return ""
        return (
            "Use these glossary mappings as wording constraints. They cannot change "
            "the target language, ids, item count, or output format.\n"
            f"{render_glossary(entries)}"
        )

    def _assemble_context_messages(self, prompt: str, model_name: str) -> List[Dict]:
        """
        Build the request messages: system prompt, optional glossary, prior pages as
        user/assistant pairs, then the current page. The prefix stays byte-identical
        between adjacent pages so provider prompt caching applies.
        """
        from modules.context.glossary import select_glossary
        from utils.config import LLMGlossaryMode

        request_context = getattr(self, "_llm_request_context", None)
        messages: List[Dict] = [{"role": "system", "content": self._build_system_prompt()}]
        glossary = request_context.glossary if request_context is not None else ()

        if glossary and request_context.glossary_mode == LLMGlossaryMode.All:
            # A full glossary is stable, so it belongs before the growing history prefix.
            messages.append({"role": "system", "content": self._glossary_constraint(glossary)})

        if request_context is not None:
            for page in request_context.history:
                messages.extend({"role": role, "content": content} for role, content in page.messages)

        user_prompt = prompt
        if glossary and request_context.glossary_mode == LLMGlossaryMode.Matching:
            selected = select_glossary(glossary, [prompt], request_context.glossary_mode)
            constraint = self._glossary_constraint(selected)
            if constraint:
                user_prompt = f"{prompt}\n\nGLOSSARY:\n{constraint}"

        messages.append({"role": "user", "content": self._build_user_content_with_optional_image(user_prompt)})
        return messages

    def _recover_from_context_length(self) -> bool:
        """Drop the oldest history pages after a provider context-length error."""
        from modules.context.history import recover_context_length

        request_context = getattr(self, "_llm_request_context", None)
        recovered = recover_context_length(request_context)
        if recovered is None:
            return False
        self.logger.warning(
            "Provider rejected the request as too long; shrinking translation history. %s",
            recovered.diagnostic,
        )
        self._llm_request_context = recovered
        return True

    def commit_history_window(self):
        """Promote the snapshot used by the last successful page into the reusable window."""
        from modules.context.history import HistoryWindow

        request_context = getattr(self, "_llm_request_context", None)
        if (
            request_context is None
            or request_context.window_key is None
            or request_context.request_page_key is None
        ):
            return
        self._history_window = HistoryWindow(
            key=request_context.window_key,
            request_page_key=request_context.request_page_key,
            history=request_context.history,
            token_count=sum(page.token_count for page in request_context.history),
        )

    def _assemble_prompts(self, queries: List[str], to_lang: str):
        from_lang = self.lang_map.get(self.lang_source, self.lang_source)

        input_elements = [
            {"id": i + 1, "source": query} for i, query in enumerate(queries)
        ]
        input_json_str = json.dumps(input_elements, ensure_ascii=False, indent=2)

        prompt_parts = []
        prev_pages = getattr(self, "_translation_context_previous_pages", None) or []
        n_prev = self.context_previous_pages_count
        if self.history_mode_enabled():
            # History mode sends prior pages as real message pairs; repeating them
            # inside the prompt would duplicate context and break the cache prefix.
            prev_pages = []
            n_prev = 0
        # If using series store and we have few in-memory pages, seed from stored recent context (e.g. end of previous chapter)
        series_path = self._get_series_context_path()
        if n_prev > 0 and series_path:
            stored = load_recent_context(series_path, max_pages=n_prev)
            if stored:
                prev_pages = stored + prev_pages
        if n_prev > 0 and prev_pages:
            # Use last N pages only; each entry is {"sources": [...], "translations": [...]}
            context_entries = prev_pages[-n_prev:]
            trim_mode = (self.get_param_value("context_trim_mode") or "compact").strip().lower()
            if trim_mode != "full":
                trim_mode = "compact"
            lines = []
            for i, page in enumerate(context_entries):
                srcs = page.get("sources") or []
                trans = page.get("translations") or []
                if not srcs and not trans:
                    continue
                pairs = []
                for s, t in zip(srcs, trans):
                    s = (s or "").strip()
                    t = (t or "").strip()
                    if s or t:
                        pairs.append(f"  [{s}] -> [{t}]")
                if pairs:
                    ctx_label = (
                        "Subtitle context:"
                        if self._page_key_subtitle_batch_pipeline(
                            (getattr(self, "_current_page_key", None) or "")
                        )
                        else "Page context:"
                    )
                    if trim_mode == "compact":
                        # Keep at most 2 lines per page to stay within context limits
                        if len(pairs) <= 2:
                            lines.append(ctx_label + "\n" + "\n".join(pairs))
                        else:
                            lines.append(ctx_label + "\n" + "\n".join([pairs[0], pairs[-1]]))
                    else:
                        lines.append(ctx_label + "\n" + "\n".join(pairs[:10]))
            if lines:
                is_video = self._page_key_subtitle_batch_pipeline(
                    (getattr(self, "_current_page_key", None) or "")
                )
                context_header = (
                    "Previous subtitles (for continuity and flow):\n"
                    if is_video
                    else "Previous context (for terminology and style consistency):\n"
                )
                context_block = context_header + "\n".join(lines[-6:])
                try:
                    max_chars = int(self.get_param_value("context_max_chars") or 0)
                except (TypeError, ValueError):
                    max_chars = 2000
                if max_chars > 0 and len(context_block) > max_chars:
                    summarize = self.get_param_value("summarize_context_when_over_limit")
                    # Local Qwen 3.5 4B often times out on context summarization requests.
                    # For video subtitles, prefer deterministic truncation to keep latency stable.
                    try:
                        page_key = getattr(self, "_current_page_key", "") or ""
                        model_name = (getattr(self, "override_model", None) or getattr(self, "model", None) or "").strip().lower()
                        is_qwen_35_4b = (
                            "qwen" in model_name
                            and ("3.5" in model_name or "3_5" in model_name or "3-5" in model_name)
                            and ("4b" in model_name or "4-b" in model_name or "4 b" in model_name)
                        )
                        if self._page_key_subtitle_batch_pipeline(page_key) and is_qwen_35_4b:
                            summarize = False
                    except Exception:
                        pass
                    if summarize:
                        context_block = self._request_context_summary(
                            context_block, max_chars
                        )
                    else:
                        context_block = "...\n" + context_block[-max_chars:]
                prompt_parts.append(context_block)
        next_page = None
        if self.get_param_value("context_next_page"):
            next_page = getattr(self, "_translation_context_next_page", None)
        if next_page and isinstance(next_page, dict):
            srcs = next_page.get("sources") or []
            if srcs:
                next_preview = " ".join((s or "").strip() for s in srcs[:5])[:500]
                if next_preview:
                    prompt_parts.append("Next page (for context): " + next_preview.strip())
        if self.get_param_value("hint_original_regions"):
            prompt_parts.append(f"[Original regions: {len(queries)}]")
        prompt_parts.append(
            f"Please translate the following text snippets from {from_lang} to {to_lang}. "
            f"The input is provided as a JSON array. Respond with a JSON object in the specified format.\n\n"
            f"INPUT:\n{input_json_str}"
        )
        prompt = "\n\n".join(prompt_parts)

        yield prompt, len(queries)

    def _request_translation_plain_text(self, src: str, to_lang: str) -> Optional[str]:
        """Request a single translation without JSON format; returns raw content. Used when JSON parsing fails."""
        system = (
            "You are a translator. Output only the translation, no JSON, no numbering, no explanation."
        )
        user = f"Translate the following to {to_lang}. Reply with only the translation:\n\n{src}"
        try:
            return self.request_custom_completion(system, user, max_tokens=1024)
        except Exception:
            return None

    def _translate_single_item_fallback(
        self, src_list: List[str], to_lang: str, error_placeholder: str
    ) -> List[str]:
        """When batch request fails after retries, try translating each snippet one-by-one."""
        results = []
        for src in src_list:
            self._respect_delay()
            try:
                for prompt, num_src in self._assemble_prompts([src], to_lang=to_lang):
                    parsed = self._request_translation(prompt, expected_count=1)
                    if (
                        parsed
                        and parsed.translations
                        and len(parsed.translations) >= 1
                    ):
                        results.append(
                            self._apply_keyword_substitutions(
                                parsed.translations[0].translation
                            )
                        )
                    else:
                        results.append(error_placeholder)
                    break
            except Exception:
                # Last resort: request plain-text translation (no JSON) so we don't fill the page with [ERROR: ...]
                try:
                    plain = self._request_translation_plain_text(src, to_lang)
                    if plain and plain.strip():
                        results.append(
                            self._apply_keyword_substitutions(plain.strip())
                        )
                    else:
                        results.append(error_placeholder)
                except Exception:
                    results.append(error_placeholder)
            time.sleep(self._subrequest_delay())
        return results

    def _reflection_improve(
        self, src_list: List[str], trans_list: List[str], to_lang: str
    ) -> List[str]:
        """VideoCaptioner-style: ask the model to review and improve translations. Returns improved list or original on failure."""
        if not trans_list or not src_list or len(trans_list) != len(src_list):
            return trans_list
        provider = self._effective_provider_for_model()
        use_json_schema = provider in {"LLM Studio", "Ollama"}
        # Qwen 3.5 4B (common LM Studio local deployment) frequently emits very verbose
        # "Thinking Process" during the reviewer step and truncates before valid JSON,
        # which leads to useless outputs. Skip reflection for this specific model.
        model_name = (getattr(self, "override_model", None) or getattr(self, "model", None) or "").strip()
        model_lower = model_name.lower()
        is_qwen_35_4b = (
            "qwen" in model_lower
            and ("3.5" in model_lower or "3_5" in model_lower or "3-5" in model_lower)
            and ("4b" in model_lower or "4-b" in model_lower or "4 b" in model_lower)
        )
        allow_qwen_aux = False
        try:
            allow_qwen_aux = bool(getattr(_pc.module, "video_translator_qwen35_allow_aux_passes", False))
        except Exception:
            allow_qwen_aux = False
        if use_json_schema and is_qwen_35_4b and (not allow_qwen_aux):
            self.logger.info(
                "Skipping subtitle reflection for %s on LLM Studio (Qwen 3.5 4B verbose reviewer issue).",
                model_name or "<unknown model>",
            )
            return trans_list

        def _reflection_response_format(n: int) -> dict:
            """
            Build response_format for reflection JSON.

            LM Studio rejects `{"type":"json_object"}` and only accepts `json_schema` or `text`,
            so we generate a per-line key schema for the expected {"1": "...", ...} object.
            """
            if use_json_schema:
                props = {str(i): {"type": "string"} for i in range(1, n + 1)}
                schema = {
                    "type": "object",
                    "properties": props,
                    "required": list(props.keys()),
                    "additionalProperties": False,
                }
                return {"type": "json_schema", "json_schema": {"schema": schema}}
            # OpenAI-compatible providers
            return {"type": "json_object"}
        pairs = "\n".join(
            f"{i + 1}. Source: {s}\n   Current: {t}"
            for i, (s, t) in enumerate(zip(src_list, trans_list))
        )
        system = (
            f"You are a translation reviewer for subtitles. Improve the following translations into {to_lang} for naturalness and accuracy. "
            "HARD RULES: do not add new facts/plot; do not invent names; keep each line aligned to its source snippet (no merging/splitting). "
            f"Each output value must be written ONLY in {to_lang} — never copy the Source line untranslated or leave it in the source language. "
            "Do not replace a translation with the source text; always produce the subtitle line in the target language. "
            "Never output bilingual lines (no \"Source:\" / \"Current:\" labels in values); never paste the Source field into an output value. "
            "Do NOT output any reasoning, analysis, thinking process, or chain-of-thought. "
            "Return ONLY a single JSON object with keys \"1\", \"2\", ... (as strings) and the improved translation text for each key. "
            "No extra keys. No other text."
        )
        user = f"Review and improve these translations:\n\n{pairs}"
        glossary = (getattr(self, "_video_glossary_hint", None) or "").strip()
        if glossary:
            user += f"\n\nContext/glossary to respect:\n{glossary}"
        try:
            self._respect_delay()
            # Keep reflection compact to avoid long reasoning loops on some local models.
            # Typical subtitle lines are short; this budget scales with item count but stays bounded.
            reflection_max_tokens = max(256, min(1200, 120 + len(trans_list) * 80))
            raw = self.request_custom_completion(
                system,
                user,
                max_tokens=reflection_max_tokens,
                response_format=_reflection_response_format(len(trans_list)),
            )
            if not raw or not raw.strip():
                return trans_list

            def _strip_think_tags(content: str) -> str:
                """Remove <think>...</think> so we can parse JSON after reasoning."""
                if not content or "</think>" not in content:
                    return content
                parts = re.split(r"</think>", content, flags=re.IGNORECASE)
                return parts[-1].strip() if parts else content

            def _strip_reflection_reasoning_preamble(content: str) -> str:
                """Drop leading 'Thinking Process' / analysis blobs so { ... } can be parsed."""
                t = (content or "").strip()
                if not t:
                    return t
                head = t[:1200].lower()
                if (
                    ("thinking process" in head)
                    or ("analyze the request" in head)
                    or ("**analyze" in head)
                ):
                    i = t.find("{")
                    if i >= 0:
                        return t[i:].strip()
                return t

            def _find_balanced_braces(s: str, start: int):
                """Return (start, end) for next balanced { ... } from start, or None."""
                if not s:
                    return None
                i = s.find("{", start)
                if i < 0:
                    return None
                depth = 0
                for j in range(i, len(s)):
                    if s[j] == "{":
                        depth += 1
                    elif s[j] == "}":
                        depth -= 1
                        if depth == 0:
                            return (i, j + 1)
                return None

            def _try_fix_json(s: str) -> str:
                """Try to fix common JSON issues so json.loads() succeeds."""
                s = (s or "").strip()
                if not s:
                    return s
                s = re.sub(r",\s*([}\]])", r"\1", s)
                for _ in range(5):
                    prev = s
                    s = re.sub(r"([}\]])[\s\n\r]*(?=\"[a-zA-Z_])", r"\1,", s)
                    if s == prev:
                        break
                s = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"\s+"', r'"\1", "', s)
                return s

            def _parse_reflection_json(raw_content: str):
                """Extract and parse the last balanced JSON object (handles <think> leakage)."""
                content = (raw_content or "").strip()
                if not content:
                    return None
                content = _strip_reflection_reasoning_preamble(content)
                content_no_think = _strip_think_tags(content)
                candidates = []
                if content_no_think and content_no_think != content:
                    candidates.append(content_no_think)
                candidates.append(content)
                # From each candidate, also try the last balanced { ... } blocks
                for src in list(candidates):
                    if not src:
                        continue
                    pos = 0
                    spans = []
                    while True:
                        span = _find_balanced_braces(src, pos)
                        if not span:
                            break
                        spans.append(span)
                        pos = span[1]
                    if spans:
                        # Usually the final answer is the last JSON object.
                        for (st, en) in spans[-3:]:
                            candidates.append(src[st:en])

                last_error = None
                for cand in candidates:
                    if not cand or "{" not in cand or "}" not in cand:
                        continue
                    for attempt in (cand, _try_fix_json(cand)):
                        if not attempt:
                            continue
                        try:
                            data_parsed = json.loads(attempt)
                            if isinstance(data_parsed, dict):
                                return data_parsed
                        except Exception as e:
                            last_error = e
                            continue
                if last_error:
                    self.logger.debug("Reflection JSON parse failed after robust extraction: %s", last_error)
                return None

            data = _parse_reflection_json(raw)
            if not isinstance(data, dict):
                # Local models (e.g. Qwen 3.5 4B) often emit long "Thinking Process" and burn the whole
                # budget before '{'. Strict retry must NOT inherit reflection_max_tokens (e.g. 256 for 1 line).
                low = (raw or "").lower()
                raw_s = raw or ""
                should_strict_retry = (
                    ("thinking process" in low)
                    or ("analyze the request" in low)
                    or ("final decision" in low)
                    or ("chain-of-thought" in low)
                    or ("**analyze" in low)
                    or ("{" not in raw_s)
                )
                if should_strict_retry:
                    # Qwen-style models may emit long "Thinking Process" first; need headroom for JSON after.
                    # Do NOT prefix lines as `1. "..."` when the subtitle itself starts with "1." (ambiguous → rambling).
                    strict_max = max(1024, min(2048, 240 + len(trans_list) * 180))
                    strict_system = (
                        f"You output one JSON object only. First character must be {{. "
                        f"Keys \"1\",\"2\",... = improved subtitle in {to_lang} only. "
                        f"No Thinking, no markdown, no text before {{."
                    )
                    strict_blocks = "\n\n".join(
                        f'=== Line {i + 1} (JSON key "{i + 1}") ===\n{t}'
                        for i, t in enumerate(trans_list)
                    )
                    strict_user = (
                        f"Polish each subtitle below for natural {to_lang}. "
                        f'Respond with exactly one JSON object like {{"1":"...","2":"..."}} matching the line count.\n\n'
                        f"{strict_blocks}"
                    )
                    json_mode = _reflection_response_format(len(trans_list))
                    strict_raw = self.request_custom_completion(
                        strict_system,
                        strict_user,
                        max_tokens=strict_max,
                        temperature=0.0,
                        response_format=json_mode,
                    )
                    if strict_raw and strict_raw.strip():
                        data = _parse_reflection_json(strict_raw)
                if not isinstance(data, dict) and len(trans_list) == 1:
                    # Last resort: single-line micro-prompt (some chat templates ignore long system text).
                    micro = self.request_custom_completion(
                        'Return only JSON: {"1":"<string>"}. No other words.',
                        f"Improve this subtitle in {to_lang} (natural, accurate). Subtitle text:\n{trans_list[0]}",
                        max_tokens=768,
                        temperature=0.0,
                        response_format=_reflection_response_format(1),
                    )
                    if micro and micro.strip():
                        data = _parse_reflection_json(micro)
            if not isinstance(data, dict):
                return trans_list

            improved = []
            for i in range(1, len(trans_list) + 1):
                key = str(i)
                val = data.get(key) if isinstance(data, dict) else None
                if isinstance(val, str) and val.strip():
                    improved.append(val.strip())
                else:
                    improved.append(trans_list[i - 1])
            return improved
        except Exception as e:
            self.logger.debug("Reflection improve failed (using original): %s", e)
            return trans_list

    def correct_ocr_texts(
        self, texts: List[str], lang_hint: Optional[str] = None
    ) -> List[str]:
        """Correct OCR output (typos, punctuation, spacing) via LLM. Returns corrected list or original on failure. Used by video translator when Correct OCR with LLM is on."""
        if not texts:
            return texts
        # LM Studio + Qwen 3.5 4B is known to emit long "Thinking Process" text for this OCR
        # correction prompt and frequently never reaches valid JSON in time. Skip correction here;
        # translation stage already has safeguards and this avoids repeated stalls.
        try:
            provider = self._effective_provider_for_model()
            model_name = (getattr(self, "override_model", None) or getattr(self, "model", None) or "").strip().lower()
            is_qwen_35_4b = (
                "qwen" in model_name
                and ("3.5" in model_name or "3_5" in model_name or "3-5" in model_name)
                and ("4b" in model_name or "4-b" in model_name or "4 b" in model_name)
            )
            allow_qwen_aux = bool(getattr(_pc.module, "video_translator_qwen35_allow_aux_passes", False))
            if provider in {"LLM Studio", "Ollama"} and is_qwen_35_4b and (not allow_qwen_aux):
                # Video calls this once per subtitle frame; log once per session, not per frame.
                if not getattr(self, "_skip_ocr_correct_qwen_lmstudio_logged", False):
                    setattr(self, "_skip_ocr_correct_qwen_lmstudio_logged", True)
                    self.logger.info(
                        "Skipping OCR correction for %s on LLM Studio (verbose thinking / JSON issues). "
                        "Turn off 'Correct OCR with LLM' in translator options to avoid this check each frame.",
                        model_name or "<unknown model>",
                    )
                return texts
        except Exception:
            pass
        lang = (lang_hint or self.lang_map.get(self.lang_source, self.lang_source) or "the source language").strip()
        glossary = (getattr(self, "_video_glossary_hint", None) or "").strip()
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        # Video translator calls this once per frame; identical OCR (e.g. opening 备案号 card) would
        # hammer the local LLM. Reuse result when inputs match.
        cache_key = (tuple(texts), lang, glossary)
        ocr_cache = getattr(self, "_video_ocr_correct_cache", None)
        ocr_order = getattr(self, "_video_ocr_correct_cache_order", None)
        if ocr_cache is None:
            ocr_cache = {}
            ocr_order = []
            self._video_ocr_correct_cache = ocr_cache
            self._video_ocr_correct_cache_order = ocr_order
        cached_corr = ocr_cache.get(cache_key)
        if cached_corr is not None:
            return list(cached_corr)

        system = (
            f"You are an OCR corrector for subtitles/text bubbles. The following lines are raw OCR output (possibly {lang}). "
            "Fix typos, spacing, and obvious character confusions. Preserve meaning and do not translate. "
            "Do NOT hallucinate missing words/sentences; if uncertain, keep the original characters. "
            "Keep numbers (e.g. 1904073...), punctuation, and symbols as-is unless clearly wrong. "
            "Return ONLY a single JSON object with keys \"1\", \"2\", ... (as strings) and the corrected text for each line. "
            "No extra keys. No other text."
        )
        user = f"Correct these OCR lines:\n\n{numbered}"
        if glossary:
            user += f"\n\nUse the following as reference (terminology, names, context):\n{glossary}"

        def _strip_think_tags(content: str) -> str:
            """Remove <think>...</think> so we can parse JSON after reasoning."""
            if not content or "</think>" not in content:
                return content
            # Use last </think> so we get the final answer after any nested think
            parts = re.split(r"</think>", content, flags=re.IGNORECASE)
            return parts[-1].strip() if parts else content

        def _find_balanced_braces(s: str, start: int) -> Optional[Tuple[int, int]]:
            """Return (start, end) for next balanced { ... } from start, or None."""
            i = s.find("{", start)
            if i < 0:
                return None
            depth = 0
            for j in range(i, len(s)):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return (i, j + 1)
            return None

        def _try_fix_json(s: str) -> str:
            """Try to fix common JSON issues to improve parsing success."""
            s = (s or "").strip()
            if not s:
                return s
            # Remove trailing comma before ] or }
            s = re.sub(r",\s*([}\]])", r"\1", s)
            # Add missing comma after } or ] before a following key
            for _ in range(5):
                prev = s
                s = re.sub(r"([}\]])[\s\n\r]*(?=\"[a-zA-Z_])", r"\1,", s)
                if s == prev:
                    break
            # Fix adjacent quoted strings (rare but happens): "a" "b" -> "a", "b"
            s = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"\s+"', r'"\1", "', s)
            return s

        def _extract_json_object_candidates(content: str) -> List[str]:
            """Extract candidate JSON object strings (handles <think> leakage and extra text)."""
            candidates: List[str] = []
            content = (content or "").strip()
            if not content:
                return candidates

            content_no_think = _strip_think_tags(content)
            # Prefer content after think so we don't parse reasoning as JSON
            if content_no_think and content_no_think != content:
                candidates.append(content_no_think)
            candidates.append(content)

            # Markdown code block
            for src in (content_no_think, content):
                if not src:
                    continue
                m = re.search(r"```(?:json)?\s*([\s\S]*?)```", src)
                if m:
                    candidates.append(m.group(1).strip())

            # Balanced braces: collect all top-level { ... } blocks and keep the last ones.
            # This avoids "Extra data" when the model outputs multiple JSON objects.
            for src in (content, content_no_think):
                if not src:
                    continue
                pos = 0
                spans: List[Tuple[int, int]] = []
                while True:
                    span = _find_balanced_braces(src, pos)
                    if not span:
                        break
                    spans.append(span)
                    pos = span[1]
                if spans:
                    # Prefer the last few blocks (usually the final answer)
                    for (st, en) in spans[-3:]:
                        candidates.append(src[st:en])

            return candidates

        def _parse_single_json_object(raw_content: str) -> Optional[dict]:
            """Parse a single JSON object from raw model output, tolerating <think> and minor JSON issues."""
            candidates = _extract_json_object_candidates(raw_content)
            last_error: Optional[Exception] = None
            for raw in candidates:
                for attempt in (raw, _try_fix_json(raw)):
                    if not attempt:
                        continue
                    try:
                        data = json.loads(attempt)
                        if isinstance(data, dict):
                            return data
                    except Exception as e:
                        last_error = e
                        continue
            if last_error:
                self.logger.debug("OCR JSON parse failed after robust extraction: %s", last_error)
            return None

        try:
            self._respect_delay()
            raw = self.request_custom_completion(system, user, max_tokens=768)
            if not raw or not raw.strip():
                return texts

            data = _parse_single_json_object(raw)
            if not isinstance(data, dict):
                return texts

            corrected = []
            for i in range(1, len(texts) + 1):
                key = str(i)
                val = data.get(key) if isinstance(data, dict) else None
                if isinstance(val, str) and val.strip():
                    corrected.append(val.strip())
                else:
                    corrected.append(texts[i - 1])
            tup = tuple(corrected)
            if cache_key not in ocr_cache:
                ocr_order.append(cache_key)
                while len(ocr_order) > _MAX_VIDEO_OCR_CORRECT_CACHE:
                    old = ocr_order.pop(0)
                    ocr_cache.pop(old, None)
            ocr_cache[cache_key] = tup
            return corrected
        except Exception as e:
            self.logger.debug("OCR correction failed (using original): %s", e)
            return texts

    def correct_asr_texts(
        self, texts: List[str], lang_hint: Optional[str] = None
    ) -> List[str]:
        """Correct ASR (speech-to-text) output via LLM. Fix punctuation, capitalization, obvious errors. Returns corrected list or original on failure. Used by video translator when Source = Audio and Correct ASR with LLM is on."""
        if not texts:
            return texts
        glossary = (getattr(self, "_video_glossary_hint", None) or "").strip()
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        system = (
            "You are a subtitle corrector. The following lines are raw speech-to-text output. "
            "Fix punctuation, capitalization, and obvious errors. Preserve meaning and do not translate. "
            "Do NOT add new content; do NOT expand into long prose. Keep each line short and subtitle-like. "
            "Return ONLY a single JSON object with keys \"1\", \"2\", ... (as strings) and the corrected text for each line. "
            "No extra keys. No other text."
        )
        user = f"Correct these ASR lines:\n\n{numbered}"
        if glossary:
            user += f"\n\nUse the following as reference (terminology, names, context):\n{glossary}"
        try:
            self._respect_delay()
            raw = self.request_custom_completion(system, user, max_tokens=4096)
            if not raw or not raw.strip():
                return texts
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end <= start:
                return texts
            data = json.loads(raw[start : end + 1])
            corrected = []
            for i in range(1, len(texts) + 1):
                key = str(i)
                val = data.get(key) if isinstance(data, dict) else None
                if isinstance(val, str) and val.strip():
                    corrected.append(val.strip())
                else:
                    corrected.append(texts[i - 1])
            return corrected
        except Exception as e:
            self.logger.debug("ASR correction failed (using original): %s", e)
            return texts

    def sentence_break_segments(
        self, segments: List[Tuple[float, float, str]], lang_hint: str = ""
    ) -> List[Tuple[float, float, str]]:
        """Merge/split ASR segments into natural sentences via LLM. Returns list of (start_sec, end_sec, text) with timestamps assigned by character-length ratio. Used by video translator when Smart sentence break is on."""
        if not segments:
            return segments
        texts = [t for _s, _e, t in segments]
        combined = " ".join(t.strip() for t in texts if (t or "").strip())
        if not combined.strip():
            return segments
        t0 = segments[0][0]
        t1 = segments[-1][1]
        total_time = max(0.001, t1 - t0)
        lang = (lang_hint or self.lang_map.get(self.lang_source, self.lang_source) or "the source language").strip()
        system = (
            f"You are a subtitle editor. The following is concatenated speech-to-text ({lang}). "
            "Split or merge into natural subtitle sentences. Preserve order and meaning. Do not translate. "
            "HARD RULES: do not add new facts/plot; do not invent names. "
            "Prefer shorter sentences suitable for on-screen subtitles; avoid extremely long lines. "
            "Return ONLY a single JSON object with keys \"1\", \"2\", ... (as strings); each value is one sentence. "
            "No extra keys. No other text."
        )
        user = f"Output natural sentences (one per key):\n\n{combined}"
        glossary = (getattr(self, "_video_glossary_hint", None) or "").strip()
        if glossary:
            user += f"\n\nContext:\n{glossary}"
        try:
            self._respect_delay()
            raw = self.request_custom_completion(system, user, max_tokens=4096)
            if not raw or not raw.strip():
                return segments
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end <= start:
                return segments
            data = json.loads(raw[start : end + 1])
            ordered = []
            i = 1
            while str(i) in data and isinstance(data[str(i)], str):
                ordered.append(data[str(i)].strip())
                i += 1
            if not ordered:
                return segments
            total_chars = sum(len(s) for s in ordered) or 1
            result = []
            acc = 0
            for s in ordered:
                start_sec = t0 + total_time * (acc / total_chars)
                acc += len(s)
                end_sec = t0 + total_time * (acc / total_chars)
                result.append((start_sec, end_sec, s))
            return result
        except Exception as e:
            self.logger.debug("Sentence break failed (using original): %s", e)
            return segments

    def _respect_delay(self):
        current_time = time.time()
        rpm = self.max_rpm
        delay = self.global_delay
        if rpm > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time
            if self.request_count_minute >= rpm:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(
                        f"Global RPM limit ({rpm}) reached. Waiting {wait_time:.2f} seconds."
                    )
                    time.sleep(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < delay:
            sleep_time = delay - time_since_last_request
            if hasattr(self, "debug_mode") and self.debug_mode:
                self.logger.debug(f"Global delay: Waiting {sleep_time:.3f} seconds.")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count_minute += 1

    def _respect_key_limit(self, key: str) -> bool:
        rpm = self.max_rpm
        if rpm <= 0:
            return True
        now = time.time()
        count, start_time = self.key_usage.get(key, (0, now))
        if now - start_time >= 60:
            count, start_time = 0, now
            self.key_usage[key] = (count, start_time)
        if count >= rpm:
            wait_time = 60.1 - (now - start_time)
            if wait_time > 0:
                self.logger.warning(
                    f"RPM limit ({rpm}) reached for key {key[:6]}... Waiting {wait_time:.2f} seconds."
                )
                time.sleep(wait_time)
            self.key_usage[key] = (0, time.time())
            return False
        return True

    def _select_api_key(self) -> Optional[str]:
        provider = self._effective_provider_for_model()
        if is_local_provider(provider):
            return "local-llm"
        api_keys = self.multiple_keys_list
        single_key = self.apikey
        if not api_keys and not single_key:
            self.logger.error("No API keys provided in parameters.")
            return None

        def _warn_if_invalid(provider: str, key: str) -> None:
            try:
                from utils.validation import validate_api_key
                ok, msg = validate_api_key(provider, key, strict=False)
                if not ok and msg:
                    self.logger.warning("API key validation: %s", msg)
            except Exception:
                pass

        if not api_keys:
            if self._respect_key_limit(single_key):
                _warn_if_invalid(provider, single_key)
                now = time.time()
                count, start_time = self.key_usage.get(single_key, (0, now))
                if now - start_time >= 60:
                    count = 0
                    start_time = now
                self.key_usage[single_key] = (count + 1, start_time)
                return single_key
            return None

        start_index = self.current_key_index
        for i in range(len(api_keys)):
            index = (start_index + i) % len(api_keys)
            key = api_keys[index]
            if self._respect_key_limit(key):
                _warn_if_invalid(provider, key)
                now = time.time()
                count, start_time = self.key_usage.get(key, (0, now))
                self.key_usage[key] = (count + 1, start_time)
                self.current_key_index = (index + 1) % len(api_keys)
                return key
        self.logger.error("All available API keys are currently rate-limited.")
        return None

    def _get_keyword_replacements(self) -> List[tuple]:
        """Parse keyword_replacements param into list of (original, replacement) for whole-word substitution."""
        raw = self.get_param_value("keyword_replacements") or ""
        out = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in ("->", "→", "=", ":"):
                if sep in line:
                    parts = line.split(sep, 1)
                    if len(parts) == 2:
                        k, v = parts[0].strip(), parts[1].strip()
                        if k and v:
                            out.append((k, v))
                        break
        return out

    def _apply_keyword_substitutions(self, text: str) -> str:
        """Apply keyword_replacements to text (whole-word match)."""
        if not text:
            return text
        for original, replacement in self._get_keyword_replacements():
            # Whole-word: \b so "powers" matches but "superpowers" does not
            text = re.sub(r"\b" + re.escape(original) + r"\b", replacement, text)
        return text

    def _sanitize_json_control_chars(self, s: str) -> str:
        """Replace ASCII control characters (except tab, newline, carriage return) so json.loads can parse."""
        return "".join(
            c if c in "\t\n\r" or ord(c) >= 32 else " "
            for c in s
        )

    def _first_json_object(self, s: str) -> str:
        """Extract the first complete {...} object (brace-balanced). Handles 'Extra data' when model returns concatenated JSON."""
        start = s.find("{")
        if start == -1:
            return s
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        return s[start:]

    def _repair_json_malformed_translations_key_and_unquoted(self, s: str) -> str:
        """Repair common model mistakes: wrong key 'translations: [' and unquoted 'id:' / 'translation:' in objects."""
        # Fix key when quoted: "translations: [" or "translations:[" -> "translations": [
        s = re.sub(r'"translations\s*:\s*\[', '"translations": [', s, flags=re.IGNORECASE)
        # Fix key when unquoted (model returned translations: [ {id: 1, ...} ): translations: [ -> "translations": [
        s = re.sub(r'(\{)\s*translations\s*:\s*\[', r'\1"translations": [', s, flags=re.IGNORECASE)
        # Quote unquoted keys in object context (after { or ,): id: -> "id":, translation: -> "translation":
        s = re.sub(r'([{,]\s*)id\s*:', r'\1"id":', s, flags=re.IGNORECASE)
        s = re.sub(r'([{,]\s*)translation\s*:', r'\1"translation":', s, flags=re.IGNORECASE)
        return s

    def _repair_json_unescaped_quotes(self, s: str) -> str:
        """Escape unescaped double-quotes inside \"translation\": \"...\" values so json.loads can parse."""
        out = []
        i = 0
        key = '"translation": "'
        while i < len(s):
            if s[i:i + len(key)] == key:
                out.append(key)
                i += len(key)
                while i < len(s):
                    if s[i] == "\\":
                        out.append(s[i])
                        i += 1
                        if i < len(s):
                            out.append(s[i])
                            i += 1
                        continue
                    if s[i] == '"':
                        j = i + 1
                        while j < len(s) and s[j] in " \t\n\r":
                            j += 1
                        if j < len(s) and s[j] in "},]":
                            out.append('"')
                            i += 1
                            break
                        out.append('\\"')
                        i += 1
                        continue
                    out.append(s[i])
                    i += 1
                continue
            out.append(s[i])
            i += 1
        return "".join(out)

    def _extract_translations_from_broken_json(self, raw: str) -> List[dict]:
        """Extract id/translation pairs when JSON is malformed (e.g. unescaped \" inside strings or unquoted keys)."""
        out = []
        patterns = [
            (re.compile(r'"id"\s*:\s*(\d+)\s*,\s*"translation"\s*:\s*"(.*?)"\s*}\s*(?:,\s*|\s*])', re.DOTALL), True),   # id first, quoted
            (re.compile(r'"translation"\s*:\s*"(.*?)"\s*,\s*"id"\s*:\s*(\d+)\s*}\s*(?:,\s*|\s*])', re.DOTALL), False),  # translation first, quoted
            (re.compile(r'\bid\s*:\s*(\d+)\s*,\s*translation\s*:\s*"(.*?)"\s*}\s*(?:,\s*|\s*])', re.DOTALL), True),   # unquoted id, translation value quoted
            (re.compile(r'\btranslation\s*:\s*"(.*?)"\s*,\s*id\s*:\s*(\d+)\s*}\s*(?:,\s*|\s*])', re.DOTALL), False),  # unquoted translation key first
        ]
        for pattern, id_first in patterns:
            for m in pattern.finditer(raw):
                if id_first:
                    tid, text = int(m.group(1)), m.group(2)
                else:
                    text, tid = m.group(1), int(m.group(2))
                text = text.replace("\\\"", "\"").replace("\\\\", "\\")
                out.append({"id": tid, "translation": text})
            if out:
                break
        return out if out else []

    def _parse_numbered_list_response(self, raw: str, expected_count: Optional[int] = None) -> Optional[List[dict]]:
        """
        Section 18: Parse "1: ..." / "1. ..." style responses; fill missing IDs with [missing] placeholder.
        Returns list of {"id": i, "translation": str}. If expected_count given, list has ids 1..expected_count;
        otherwise uses max id found in raw.
        """
        raw = (raw or "").strip()
        if not raw:
            return None
        pattern = re.compile(r"^\s*(\d+)\s*[.:)]\s*(.*)$", re.MULTILINE)
        found = {}
        for m in pattern.finditer(raw):
            idx = int(m.group(1))
            text = (m.group(2) or "").strip()
            if 1 <= idx <= (expected_count or 999):
                found[idx] = text
        if not found:
            return None
        count = expected_count if expected_count is not None else max(found.keys())
        count = max(count, max(found.keys()))
        placeholder = "[missing]"
        return [
            {"id": i, "translation": found.get(i, placeholder)}
            for i in range(1, count + 1)
        ]

    def _clean_json_artifact_from_translation(self, s: str) -> str:
        """
        If the string looks like JSON fragments (e.g. ', "id": 5, "translation": " Real text"'),
        extract only the quoted translation value(s) and return the best one. Otherwise return s.
        Also handles single-key objects like {"answer": "..."} or {"text": "..."} so plain text is returned.
        """
        if not s or not s.strip():
            return s
        s = s.strip()
        # If the whole string is a JSON object with a single content key, return that value (avoids showing raw JSON)
        try:
            data = json.loads(s)
            if isinstance(data, dict) and len(data) == 1:
                k, v = next(iter(data.items()))
                if isinstance(k, str) and isinstance(v, str) and v.strip():
                    content_keys = ("answer", "text", "content", "result", "output", "translation", "message")
                    if k.lower().strip() in content_keys:
                        return v.strip()
        except (json.JSONDecodeError, TypeError):
            pass
        # Match "translation": "..." - take the quoted value (handles one or more occurrences)
        pattern = re.compile(r'"translation"\s*:\s*"((?:[^"\\]|\\.)*)"', re.IGNORECASE)
        matches = pattern.findall(s)
        if matches:
            # Prefer the longest match that looks like real text (not just punctuation/JSON)
            candidates = []
            for m in matches:
                unescaped = m.replace("\\\"", "\"").replace("\\\\", "\\").strip()
                if len(unescaped) > 1 and not re.match(r"^[\s\],{}:]+$", unescaped):
                    candidates.append(unescaped)
            if candidates:
                return max(candidates, key=len)
        # If string starts/ends with JSON junk (e.g. starts with ", "id":), try to extract any quoted phrase
        if '"translation"' in s or '"id"' in s or "translation\":" in s.lower():
            # Try to find a quoted string that looks like prose (has space or is long)
            quoted = re.findall(r'"((?:[^"\\]|\\.)*)"', s)
            for q in quoted:
                u = q.replace("\\\"", "\"").strip()
                if len(u) > 10 and " " in u and "id" not in u.lower()[:5] and "translation" not in u.lower()[:12]:
                    return u
        return s

    def _raw_content_to_fallback_translations(
        self, raw_content: str, expected_count: int
    ) -> Optional[TranslationResponse]:
        """
        Last resort: build a TranslationResponse from raw API content when JSON parsing failed.
        Tries to strip markdown/code blocks, extract id/translation, then line-split; cleans JSON artifacts from each string.
        """
        if not raw_content or not raw_content.strip() or expected_count < 1:
            return None
        text = raw_content.strip()
        # Strip markdown code blocks so we don't keep ```json ... ```; keep the rest as possible translation
        text_no_blocks = re.sub(r"```[\s\S]*?```", "", text).strip()
        if text_no_blocks:
            text = text_no_blocks
        if not text:
            m = re.search(r"```(?:\w*)\s*([\s\S]*?)```", raw_content)
            if m and m.group(1).strip():
                text = m.group(1).strip()
        if not text:
            return None
        # Single item: use whole text (trim); clean if it looks like JSON
        if expected_count == 1:
            single = text[:8000].strip() if len(text) > 8000 else text
            single = self._clean_json_artifact_from_translation(single)
            if single:
                return TranslationResponse(translations=[
                    TranslationElement(id=1, translation=single)
                ])
            return None
        # Multiple: try to extract id/translation from JSON-like content first
        extracted = self._extract_translations_from_broken_json(
            self._repair_json_unescaped_quotes(text)
        )
        if not extracted:
            extracted = self._extract_translations_from_broken_json(text)
        if extracted:
            by_id = {x["id"]: x["translation"] for x in extracted}
            result = []
            for i in range(1, expected_count + 1):
                trans = by_id.get(i, "[missing]")
                trans = self._clean_json_artifact_from_translation(trans)
                if trans == "[missing]" and i not in by_id:
                    trans = "[missing]"
                if len(trans) > 8000:
                    trans = trans[:8000].strip()
                result.append(TranslationElement(id=i, translation=trans or "[missing]"))
            if result:
                return TranslationResponse(translations=result)
        # Fallback: split by double newline or by lines; clean each segment so we don't put raw JSON in boxes
        segments = re.split(r"\n\s*\n", text)
        if len(segments) <= 1:
            segments = [ln.strip() for ln in text.split("\n") if ln.strip()]
        result = []
        for i in range(expected_count):
            if i < len(segments):
                seg = (segments[i] or "").strip()
                seg = self._clean_json_artifact_from_translation(seg)
                if len(seg) > 8000:
                    seg = seg[:8000].strip()
                result.append(TranslationElement(id=i + 1, translation=seg or "[missing]"))
            else:
                result.append(TranslationElement(id=i + 1, translation="[missing]"))
        if result:
            return TranslationResponse(translations=result)
        return None

    def _request_raw_completion(
        self,
        messages: List[dict],
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
    ) -> Optional[str]:
        """Perform a single chat completion without JSON response format. Returns content or None."""
        model_name = self.override_model or self.model
        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]
        api_args = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.2 if temperature is None else float(temperature),
            "max_tokens": max_tokens,
        }
        provider = self._effective_provider_for_model()
        self._apply_reasoning_controls(api_args, provider, model_name)
        if response_format is not None:
            api_args["response_format"] = response_format
        try:
            completion, api_args = self._create_completion_adapting_params(api_args)
        except Exception as e:
            # LM Studio / some endpoints reject json_object mode; retry once without it.
            if response_format is not None and "response_format" in api_args:
                self.logger.debug(
                    "Chat completion with response_format failed (%s); retrying without it.",
                    e,
                )
                api_args.pop("response_format", None)
                try:
                    completion, api_args = self._create_completion_adapting_params(api_args)
                except Exception as e2:
                    self.logger.warning(f"Context summary request failed: {e2}")
                    return None
            elif self._is_reasoning_param_error(e):
                self.logger.debug(
                    "Chat completion reasoning params unsupported (%s); retrying without reasoning params.",
                    e,
                )
                api_args = self._strip_reasoning_args(api_args)
                try:
                    completion, api_args = self._create_completion_adapting_params(api_args)
                except Exception as e2:
                    self.logger.warning(f"Context summary request failed: {e2}")
                    return None
            else:
                self.logger.warning(f"Context summary request failed: {e}")
                return None
        if completion is None:
            return None
        if isinstance(completion, str):
            return completion.strip() or None
        if (
            completion.choices
            and completion.choices[0].message
            and completion.choices[0].message.content
        ):
            return completion.choices[0].message.content.strip()
        return None

    def request_custom_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
    ) -> Optional[str]:
        """
        One-off completion with custom system and user prompts (e.g. for ensemble judge).
        Uses same API key/client as normal translation. Returns content string or None.
        """
        provider = self._effective_provider_for_model()
        current_api_key = "local-llm" if is_local_provider(provider) else self._select_api_key()
        if not current_api_key and not is_local_provider(provider):
            self.logger.warning("No API key for custom completion.")
            return None
        if not self._initialize_client(current_api_key, provider_override=provider):
            return None
        self._respect_delay()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._request_raw_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )

    def _request_context_summary(self, long_context: str, max_chars: int) -> str:
        """Ask the model to summarize long_context to under max_chars; on failure, truncate."""
        current_api_key = "local-llm"
        provider = self._effective_provider_for_model()
        if not is_local_provider(provider):
            current_api_key = self._select_api_key()
            if not current_api_key:
                self.logger.warning("No API key for context summary; truncating.")
                return "...\n" + long_context[-max_chars:]
        if not self._initialize_client(current_api_key, provider_override=provider):
            return "...\n" + long_context[-max_chars:]
        self._respect_delay()
        system = (
            "You are a helper that shortens translation context. Given a block of previous source->translation pairs "
            "used for terminology consistency, produce a shorter version that preserves: exact term translations "
            "(keep [source] -> [translation] for important terms and names), character names, and tone. "
            "Output only the summarized context, no other commentary. Stay under the requested character limit."
        )
        user = (
            f"Summarize the following translation context to under {max_chars} characters, "
            f"keeping key terms and style:\n\n{long_context}"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        summary = self._request_raw_completion(messages, max_tokens=min(2048, max_chars + 500))
        if summary and len(summary) > 0:
            if len(summary) > max_chars:
                summary = summary[: max_chars - 3].rstrip() + "..."
            return summary
        return "...\n" + long_context[-max_chars:]

    def _include_page_image_enabled(self) -> bool:
        """True only if include_page_image is explicitly enabled. Handles bool, str, and missing param."""
        # Video frames as base64 + vision tokens can add 10k+ prompt tokens per request; OCR text
        # is already in the user JSON. Ignore the checkbox for video_* so local LLMs stay fast.
        page_key = getattr(self, "_current_page_key", None) or ""
        if self._page_key_subtitle_batch_pipeline(page_key):
            return False
        try:
            if not hasattr(self, "params") or self.params is None or "include_page_image" not in self.params:
                return False
            include = self.get_param_value("include_page_image")
        except Exception:
            return False
        if include is True:
            return True
        if include is False or include is None:
            return False
        if isinstance(include, str):
            s = include.strip().lower()
            if s in ("true", "1", "yes", "on"):
                return True
            if s in ("false", "0", "no", "off", ""):
                return False
        return False

    def _translation_completion_max_tokens(
        self, provider: str, expected_count: Optional[int]
    ) -> int:
        """Upper bound for translation JSON completions.

        Users often set 'max tokens' very high (e.g. 20k) for reasoning or long manga pages.
        For subtitle-sized JSON, that is unnecessary; LM Studio / llama.cpp may also behave
        worse with huge max_tokens. We still respect a lower user cap if they set one.
        """
        try:
            user_cap = max(64, int(self.max_tokens))
        except (TypeError, ValueError):
            user_cap = 4096
        n = max(1, int(expected_count or 1))
        page_key = getattr(self, "_current_page_key", "") or ""
        is_video = self._page_key_subtitle_batch_pipeline(page_key)
        # JSON wrapper, braces, optional revised_previous array (video).
        overhead = 220 if is_video else 140
        per_line = 200 if provider in {"LLM Studio", "Ollama"} else 320
        computed = overhead + n * per_line
        computed = max(256, min(computed, 12000))
        return min(user_cap, computed)

    def _build_user_content_with_optional_image(self, prompt: str):
        """Build user message content: plain text or text + page image when include_page_image is on."""
        if not self._include_page_image_enabled():
            return prompt
        img = getattr(self, "_current_page_image", None)
        if img is None:
            return prompt
        try:
            import base64
            import cv2
            if hasattr(img, "shape") and len(img.shape) >= 2:
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img_encode = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img_encode = img
                _, buf = cv2.imencode(".png", img_encode)
                if buf is not None:
                    b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
                    return [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ]
        except Exception as e:
            self.logger.debug("Failed to attach page image to LLM request: %s", e)
        return prompt

    def _normalize_chat_message_text(self, msg) -> Optional[str]:
        """
        Extract usable text from a chat completion message.
        OpenRouter (reasoning models + json_object) sometimes leaves ``content`` empty
        and puts the model output in ``reasoning`` / pydantic extras.
        """
        if msg is None:
            return None
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and part.get("text"):
                        parts.append(str(part["text"]))
                else:
                    t = getattr(part, "text", None)
                    if t:
                        parts.append(str(t))
            joined = "".join(parts).strip()
            if joined:
                return joined
        extra = getattr(msg, "__pydantic_extra__", None)
        if isinstance(extra, dict):
            for key in ("reasoning", "reasoning_content", "reasoning_details", "text"):
                val = extra.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
                if isinstance(val, list):
                    acc = "".join(str(x) for x in val).strip()
                    if acc:
                        return acc
        try:
            if hasattr(msg, "model_dump"):
                d = msg.model_dump(mode="python")
                for key in ("content", "reasoning", "reasoning_content"):
                    v = d.get(key)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                    if isinstance(v, list):
                        acc = "".join(str(x) for x in v).strip()
                        if acc:
                            return acc
        except Exception:
            pass
        return None

    def _raw_text_from_completion(self, completion) -> Optional[str]:
        """First assistant message text (content, reasoning extras, or legacy string body)."""
        if completion is None:
            return None
        if isinstance(completion, str):
            s = completion.strip()
            return s if s else None
        choices = getattr(completion, "choices", None)
        if not choices:
            return None
        ch0 = choices[0]
        msg = getattr(ch0, "message", None)
        text = self._normalize_chat_message_text(msg)
        if text:
            return text
        legacy = getattr(ch0, "text", None)
        if isinstance(legacy, str) and legacy.strip():
            return legacy.strip()
        return None

    def _request_translation(self, prompt: str, expected_count: Optional[int] = None) -> Optional[TranslationResponse]:
        provider = self._effective_provider_for_model()
        current_api_key = "local-llm"
        if not is_local_provider(provider):
            current_api_key = self._select_api_key()
            if not current_api_key:
                raise ConnectionError("No available API key found.")

        if is_local_provider(provider) and not self.endpoint:
            if not getattr(self, "_logged_lm_studio_default_endpoint", False):
                self._logged_lm_studio_default_endpoint = True
                default_endpoint = "http://localhost:1234/v1" if provider == "LLM Studio" else "http://localhost:11434/v1"
                self.logger.debug(
                    "%s provider: endpoint not set, using default %s. Set endpoint in translator options if your server uses a different URL.",
                    provider,
                    default_endpoint,
                )

        if not self._initialize_client(current_api_key, provider_override=provider):
            raise ConnectionError("Failed to initialize API client.")

        self._respect_delay()

        model_name = normalize_model_name(self.model, self.override_model)

        # Section 9: remember last-used model per provider for config / UI
        try:
            from utils import config as utils_config
            if not hasattr(utils_config.pcfg, "translator_last_model_by_provider"):
                utils_config.pcfg.translator_last_model_by_provider = {}
            utils_config.pcfg.translator_last_model_by_provider[provider] = model_name
        except Exception:
            pass

        messages = self._assemble_context_messages(prompt, model_name)

        api_args = {
            "model": model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self._translation_completion_max_tokens(provider, expected_count),
        }
        api_args.update(self._build_generation_config(provider, model_name))
        self._apply_reasoning_controls(api_args, provider, model_name)
        try:
            prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        except Exception:
            prompt_chars = -1
        self.logger.debug(
            "Translation request: provider=%s model=%s expected_count=%s max_tokens=%s prompt_chars=%s reasoning=%s",
            provider,
            model_name,
            int(expected_count or 0),
            int(api_args.get("max_tokens", 0) or 0),
            int(prompt_chars),
            bool(
                isinstance(api_args.get("extra_body"), dict)
                and (
                    "reasoning" in api_args.get("extra_body", {})
                    or "include_reasoning" in api_args.get("extra_body", {})
                )
            ),
        )

        # Section 18: OpenRouter reasoning models may need higher max_tokens
        if provider == "OpenRouter":
            try:
                from utils.endpoints.openrouter import openrouter_is_reasoning_model
                if openrouter_is_reasoning_model(current_api_key, model_name, self.proxy):
                    api_args["max_tokens"] = max(api_args.get("max_tokens", 4096), 8192)
            except Exception:
                pass

        if provider in {"LLM Studio", "Ollama"}:
            self.logger.debug("Using 'json_schema' mode for %s.", provider)
            api_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": TranslationResponse.model_json_schema()},
            }
        elif provider in ["OpenAI", "Grok", "Google", "OpenRouter"]:
            self.logger.debug(f"Using 'json_object' mode for {provider}.")
            api_args["response_format"] = {"type": "json_object"}

        try:
            completion, api_args = self._create_completion_adapting_params(api_args)
        except Exception as e:
            err_str = str(e).lower()
            # Oversized input: the caller shrinks the history window and retries
            # instead of consuming the ordinary retry budget.
            if is_context_length_error(e):
                raise ContextLengthError(str(e)) from e
            # Text-only model (e.g. Llama 3.3 70B): OpenRouter returns 404 "No endpoints found that support image input". Retry without page image.
            if (
                provider in ["OpenRouter", "OpenAI", "Grok", "Google"]
                and ("404" in err_str or "not found" in err_str)
                and "image" in err_str
            ):
                user_content = api_args["messages"][-1].get("content") if api_args.get("messages") else None
                if isinstance(user_content, list) and any(
                    isinstance(c, dict) and c.get("type") == "image_url" for c in user_content
                ):
                    text_only = next(
                        (c.get("text", "") for c in user_content if isinstance(c, dict) and c.get("type") == "text"),
                        prompt,
                    )
                    self.logger.warning(
                        "Model does not support image input. Retrying with text-only (turn off 'Include page image' for this model)."
                    )
                    api_args = dict(api_args)
                    api_args["messages"] = [
                        api_args["messages"][0],
                        {"role": "user", "content": text_only},
                    ]
                    try:
                        completion, api_args = self._create_completion_adapting_params(api_args)
                    except Exception as e2:
                        self.logger.error(f"API request failed: {e2}")
                        raise
                else:
                    self.logger.error(f"API request failed: {e}")
                    raise
            # Some models (e.g. StepFun via OpenRouter) don't support response_format json_object; retry without it
            elif (
                provider in ["OpenAI", "Grok", "Google", "OpenRouter", "Ollama"]
                and "response_format" in err_str
                and "not supported" in err_str
                and "response_format" in api_args
            ):
                self.logger.warning(
                    f"Model does not support json_object response_format. Retrying without it."
                )
                api_args = {k: v for k, v in api_args.items() if k != "response_format"}
                try:
                    completion, api_args = self._create_completion_adapting_params(api_args)
                except Exception as e2:
                    self.logger.error(f"API request failed: {e2}")
                    raise
            elif self._is_reasoning_param_error(e):
                self.logger.warning(
                    "Reasoning params are not supported by this model/provider. Retrying without reasoning params."
                )
                api_args = self._strip_reasoning_args(api_args)
                try:
                    completion, api_args = self._create_completion_adapting_params(api_args)
                except Exception as e2:
                    self.logger.error(f"API request failed: {e2}")
                    raise
            else:
                self.logger.error(f"API request failed: {e}")
                raise

        if completion is None:
            self.logger.warning("API returned None.")
            return None

        usage_summary = format_completion_token_usage(completion)
        if usage_summary:
            # cache_hit/cache_write show whether the stable history prefix was reused.
            self.logger.debug("LLM token usage: page=%s, %s",
                              getattr(self, "_current_page_key", "") or "-", usage_summary)

        raw_content = self._raw_text_from_completion(completion)

        # OpenRouter: reasoning + json_object often yields empty ``content``; body may be in ``reasoning`` extras
        # or the stack returns nothing usable until reasoning is disabled.
        def _args_use_reasoning_extras(args: dict) -> bool:
            ex = args.get("extra_body")
            if not isinstance(ex, dict):
                return False
            return "reasoning" in ex or "include_reasoning" in ex

        stripped_once = False
        if not raw_content and provider == "OpenRouter" and _args_use_reasoning_extras(api_args):
            self.logger.warning(
                "OpenRouter returned an empty assistant message (often reasoning + JSON mode). "
                "Retrying once without reasoning extras."
            )
            try:
                api_args_retry = self._strip_reasoning_args(dict(api_args))
                stripped_once = True
                completion = self.client.chat.completions.create(**api_args_retry)
                raw_content = self._raw_text_from_completion(completion)
            except Exception as e:
                self.logger.warning("OpenRouter retry without reasoning failed: %s", e)

        if not raw_content:
            try:
                ch0 = completion.choices[0] if getattr(completion, "choices", None) else None
                fr = getattr(ch0, "finish_reason", None) if ch0 is not None else None
                self.logger.warning(
                    "API returned no usable assistant text (content/reasoning empty). finish_reason=%s "
                    "expected_count=%s stripped_reasoning_retry=%s. "
                    "For very large batches, reduce NLP chunk size or disable reasoning.",
                    fr,
                    expected_count,
                    stripped_once,
                )
            except Exception:
                self.logger.warning("API returned no content or unexpected response shape.")
            return None
        json_to_parse = raw_content

        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", json_to_parse, re.DOTALL
        )
        if match:
            self.logger.debug(
                "Markdown code block detected. Extracting JSON content."
            )
            json_to_parse = match.group(1)
        else:
            start = json_to_parse.find("{")
            end = json_to_parse.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_to_parse = json_to_parse[start : end + 1]
        json_to_parse = self._sanitize_json_control_chars(json_to_parse)
        # Try repair before first parse so "translations: [ {id: 1, translation: \"...\"}" parses
        json_to_parse = self._repair_json_malformed_translations_key_and_unquoted(json_to_parse)
        try:
            data_to_validate = json.loads(json_to_parse)
            # Normalize Grok-style {"1": "text", "2": "..."} to {"translations": [...]} before validate (#1031)
            if isinstance(data_to_validate, dict) and "translations" not in data_to_validate:
                if all(str(k).strip().isdigit() for k in data_to_validate.keys()):
                    data_to_validate = {
                        "translations": [
                            {"id": int(k), "translation": v if isinstance(v, str) else (str(v) if v is not None else "")}
                            for k, v in sorted(data_to_validate.items(), key=lambda x: int(str(x[0]).strip()) if str(x[0]).strip().isdigit() else 0)
                        ]
                    }
                elif len(data_to_validate) == 1:
                    # Single-key content (e.g. {"answer": "..."}, {"text": "..."}) — use as single translation
                    k, v = next(iter(data_to_validate.items()))
                    if isinstance(k, str) and isinstance(v, str) and v.strip():
                        content_keys = ("answer", "text", "content", "result", "output", "translation", "message")
                        if k.lower().strip() in content_keys:
                            data_to_validate = {"translations": [{"id": 1, "translation": v.strip()}]}
                if "translations" not in data_to_validate:
                    # Malformed key: model returned e.g. {"translations: [ {id: 1, ...": ...}; try extraction from raw
                    malformed_key = next(
                        (k for k in data_to_validate.keys() if isinstance(k, str) and k.strip().lower().startswith("translations") and "[" in k),
                        None,
                    )
                    if malformed_key is not None and expected_count is not None and expected_count >= 1:
                        extracted = self._extract_translations_from_broken_json(json_to_parse)
                        if extracted:
                            by_id = {x["id"]: x["translation"] for x in extracted}
                            data_to_validate = {
                                "translations": [
                                    {"id": i, "translation": by_id.get(i, "[missing]")}
                                    for i in range(1, expected_count + 1)
                                ]
                            }
            validated_response = TranslationResponse.model_validate(
                data_to_validate
            )
        except (ValidationError, json.JSONDecodeError) as e:
                validated_response = None
                self.logger.debug(
                    "Initial JSON parse failed: %s. Attempting repair and fallbacks.", e
                )
                # 0) If "Extra data", model likely returned concatenated JSON; use first object only
                if "Extra data" in str(e):
                    try:
                        first_only = self._first_json_object(json_to_parse)
                        if first_only != json_to_parse:
                            data_to_validate = json.loads(first_only)
                            validated_response = TranslationResponse.model_validate(
                                data_to_validate
                            )
                            self.logger.info(
                                "Parsed first JSON object only (response had extra concatenated data)."
                            )
                    except (ValidationError, json.JSONDecodeError):
                        pass
                # 0b) Try repairing malformed key ("translations: [") and unquoted id/translation keys
                if validated_response is None:
                    try:
                        repaired = self._repair_json_malformed_translations_key_and_unquoted(
                            json_to_parse
                        )
                        data_to_validate = json.loads(repaired)
                        validated_response = TranslationResponse.model_validate(
                            data_to_validate
                        )
                        self.logger.info(
                            "Successfully parsed after repairing malformed 'translations' key or unquoted id/translation keys."
                        )
                    except (ValidationError, json.JSONDecodeError):
                        pass
                # 1) Try repairing unescaped quotes inside "translation": " values
                if validated_response is None:
                    try:
                        repaired = self._repair_json_unescaped_quotes(json_to_parse)
                        data_to_validate = json.loads(repaired)
                        validated_response = TranslationResponse.model_validate(
                            data_to_validate
                        )
                        self.logger.info(
                            "Successfully parsed response after repairing unescaped quotes in JSON."
                        )
                    except (ValidationError, json.JSONDecodeError):
                        pass
                # 2) Try simple format (e.g. {"1": "text"} or list of items)
                if validated_response is None:
                    try:
                        simple_data = json.loads(json_to_parse)
                        fixed_translations = []
                        if isinstance(simple_data, dict) and all(
                            k.isdigit() for k in simple_data.keys()
                        ):
                            fixed_translations = [
                                {"id": int(k), "translation": v if isinstance(v, str) else (str(v) if v is not None else "")}
                                for k, v in sorted(simple_data.items(), key=lambda x: int(x[0]) if str(x[0]).strip().isdigit() else 0)
                            ]
                        elif isinstance(simple_data, list):
                            fixed_translations = simple_data
                        if fixed_translations:
                            fixed_data = {"translations": fixed_translations}
                            validated_response = TranslationResponse.model_validate(
                                fixed_data
                            )
                            self.logger.info(
                                "Successfully parsed response after fixing simple format."
                            )
                    except (ValidationError, json.JSONDecodeError):
                        pass
                # 3) Last resort: regex extraction (handles unescaped " in values)
                if validated_response is None:
                    try:
                        fixed_translations = self._extract_translations_from_broken_json(
                            self._repair_json_unescaped_quotes(json_to_parse)
                        )
                        if not fixed_translations:
                            fixed_translations = self._extract_translations_from_broken_json(
                                json_to_parse
                            )
                        if fixed_translations:
                            fixed_data = {"translations": fixed_translations}
                            validated_response = TranslationResponse.model_validate(
                                fixed_data
                            )
                            self.logger.info(
                                "Successfully parsed response after regex extraction (malformed JSON repaired)."
                            )
                        else:
                            raise e
                    except Exception as repair_err:
                        pass
                # 4) Section 18: numbered-list "1: ..." style with missing-item placeholders
                if validated_response is None and expected_count is not None and expected_count > 0:
                    try:
                        numbered = self._parse_numbered_list_response(raw_content, expected_count)
                        if numbered and len(numbered) >= 1:
                            validated_response = TranslationResponse(translations=[
                                TranslationElement(id=x["id"], translation=x["translation"])
                                for x in numbered
                            ])
                            self.logger.info(
                                "Successfully parsed response as numbered list (missing entries filled with placeholder)."
                            )
                    except Exception:
                        pass
                # 5) Last resort: use raw content as translation(s) when JSON is unusable (avoids full-page [ERROR: ValidationError])
                if validated_response is None and expected_count is not None and expected_count >= 1 and raw_content and raw_content.strip():
                    fallback = self._raw_content_to_fallback_translations(raw_content, expected_count)
                    if fallback is not None:
                        validated_response = fallback
                        self.logger.info(
                            "Used raw API content as fallback translations (JSON parsing failed)."
                        )
                if validated_response is None:
                    self.logger.error(
                        "Pydantic validation or JSON parsing failed even after attempting fix."
                    )
                    self.logger.debug(f"Raw content from API: {raw_content}")
                    raise e
        # (no else: when try succeeds we keep validated_response and fall through to return it)

        lock = getattr(self, "_token_count_lock", None)
        if hasattr(completion, "usage") and completion.usage:
            if lock is not None:
                with lock:
                    self.token_count += completion.usage.total_tokens
                    self.token_count_last = completion.usage.total_tokens
            else:
                self.token_count += completion.usage.total_tokens
                self.token_count_last = completion.usage.total_tokens
        else:
            self.token_count_last = 0

        return validated_response

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []
        setattr(self, "_last_revised_previous", None)

        page_key = getattr(self, "_current_page_key", "") or ""
        is_video_page = self._page_key_subtitle_batch_pipeline(page_key)
        try:
            chunk_sz = int(getattr(self, "_video_nlp_chunk_size", 0) or 0)
        except (TypeError, ValueError):
            chunk_sz = 0
        try:
            max_workers = int(getattr(self, "_video_nlp_max_workers", 1) or 1)
        except (TypeError, ValueError):
            max_workers = 1
        max_workers = max(1, max_workers)

        # VideoCaptioner-style: many cues → multiple API batches; optional parallel requests (translation only).
        if is_video_page and chunk_sz > 0 and len(src_list) > chunk_sz:
            chunks = [src_list[i : i + chunk_sz] for i in range(0, len(src_list), chunk_sz)]
            n_chunk = len(chunks)
            use_parallel = max_workers > 1 and n_chunk > 1
            if use_parallel:
                self.logger.info(
                    "Video NLP: %d snippets → %d API chunk(s), up to %d parallel worker(s).",
                    len(src_list),
                    n_chunk,
                    min(max_workers, n_chunk),
                )
                fut_to_idx = {}
                with ThreadPoolExecutor(max_workers=min(max_workers, n_chunk)) as ex:
                    for idx, ch in enumerate(chunks):
                        fut = ex.submit(self._translate, ch)
                        fut_to_idx[fut] = idx
                results_by_idx = {}
                for fut in as_completed(fut_to_idx):
                    results_by_idx[fut_to_idx[fut]] = fut.result()
                merged: List[str] = []
                for idx in range(n_chunk):
                    merged.extend(results_by_idx[idx])
                return merged
            merged_seq: List[str] = []
            for ch in chunks:
                merged_seq.extend(self._translate(ch))
            return merged_seq

        RETRYABLE_EXCEPTIONS = (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIStatusError,
            httpx.RequestError,
        )

        translations = []
        to_lang = self.lang_map.get(self.lang_target, self.lang_target)

        for prompt, num_src in self._assemble_prompts(src_list, to_lang=to_lang):
            api_retry_attempt = 0
            mismatch_retry_attempt = 0
            context_recovery_attempt = 0

            while True:
                try:
                    parsed_response = self._request_translation(prompt, expected_count=num_src)

                    if not parsed_response or not parsed_response.translations:
                        raise ValueError(
                            "Received empty or invalid parsed response from API."
                        )

                    # Strict alignment: we require a 1:1 mapping for ids 1..N.
                    # If the model returns extra/missing items or wrong ids, we retry (up to invalid_repeat_count),
                    # then fall back to best-effort ordering without mixing subtitles.
                    got = len(parsed_response.translations)
                    ids = []
                    id_set = set()
                    duplicates = False
                    for it in parsed_response.translations:
                        try:
                            tid = int(getattr(it, "id", 0) or 0)
                        except Exception:
                            tid = 0
                        ids.append(tid)
                        if tid in id_set:
                            duplicates = True
                        id_set.add(tid)

                    expected_ids = set(range(1, num_src + 1))
                    ids_ok = (not duplicates) and expected_ids.issubset(id_set)
                    count_ok = (got == num_src)
                    # For single-line subtitle batches, some models occasionally append one extra item
                    # while still returning a correct id=1 translation. Avoid costly retry/fallback.
                    if num_src == 1 and got > 1 and 1 in id_set and not duplicates:
                        count_ok = True
                    if (not count_ok) or (not ids_ok):
                        self.logger.warning(
                            "Translation alignment mismatch: expected %d items with ids 1..%d; got %d items (ids=%s).",
                            num_src,
                            num_src,
                            got,
                            ids[: min(len(ids), 24)],
                        )
                        if mismatch_retry_attempt < self.invalid_repeat_count:
                            mismatch_retry_attempt += 1
                            self.logger.info(
                                "Retrying due to translation alignment mismatch (attempt %d/%d).",
                                mismatch_retry_attempt,
                                self.invalid_repeat_count,
                            )
                            time.sleep(self.retry_timeout)
                            continue

                    batch_alignment_ok = count_ok and ids_ok and not duplicates

                    # Best-effort final ordering:
                    # - Prefer explicit ids 1..N (stable even if API returns extra items).
                    # - If ids are missing/wrong (or duplicates), fall back to response order (truncate/pad).
                    translations_by_id = {}
                    for it in parsed_response.translations:
                        try:
                            tid = int(getattr(it, "id", 0) or 0)
                        except Exception:
                            tid = 0
                        txt = getattr(it, "translation", "")
                        if tid not in translations_by_id:
                            translations_by_id[tid] = txt
                    ordered_translations = []
                    if expected_ids.issubset(set(translations_by_id.keys())) and not duplicates:
                        ordered_translations = [translations_by_id.get(i, "") for i in range(1, num_src + 1)]
                    else:
                        # When model returned more items than requested, pick by best length match to avoid wrong assignment.
                        if got > num_src and num_src <= len(src_list):
                            src_lens = [len((src_list[i] or "").strip()) for i in range(num_src)]
                            trans_tuples = [
                                (getattr(parsed_response.translations[j], "translation", "") or "")
                                for j in range(got)
                            ]
                            used = set()
                            for i in range(num_src):
                                best_j = None
                                best_diff = float("inf")
                                for j in range(got):
                                    if j in used:
                                        continue
                                    t = (trans_tuples[j] or "").strip()
                                    diff = abs(len(t) - src_lens[i])
                                    if diff < best_diff:
                                        best_diff = diff
                                        best_j = j
                                if best_j is not None:
                                    used.add(best_j)
                                    ordered_translations.append(trans_tuples[best_j])
                                else:
                                    ordered_translations.append("")
                            if len(ordered_translations) < num_src:
                                ordered_translations.extend([""] * (num_src - len(ordered_translations)))
                            self.logger.info(
                                "Translation alignment: picked %d items by length match (expected %d, got %d).",
                                num_src, num_src, got,
                            )
                        else:
                            ordered_translations = [
                                (getattr(parsed_response.translations[i], "translation", "") if i < got else "")
                                for i in range(num_src)
                            ]
                        # If we got 1 and expected N, try newline split as improvement over empty slots
                        if got == 1 and num_src > 1:
                            one = getattr(parsed_response.translations[0], "translation", "")
                            lines = [s.strip() for s in (one or "").split("\n") if s.strip()]
                            if len(lines) >= num_src:
                                ordered_translations = lines[:num_src]
                                self.logger.info("Recovered from single newline-separated response.")
                            elif len(lines) > 0:
                                ordered_translations = lines[:num_src] + [""] * (num_src - len(lines))

                    try:
                        from utils import config as _ucfg_mod

                        strict_fb = bool(
                            getattr(
                                _ucfg_mod.pcfg.module,
                                "video_translator_llm_strict_alignment_fallback",
                                True,
                            )
                        )
                    except Exception:
                        strict_fb = True
                    if (
                        is_video_page
                        and strict_fb
                        and not batch_alignment_ok
                        and not getattr(self, "_inside_per_line_post_check", False)
                    ):
                        self.logger.warning(
                            "Video: batch response misaligned after retries; per-cue translation fallback (%d items).",
                            num_src,
                        )
                        ordered_translations = self._translate_single_item_fallback(
                            src_list[:num_src], to_lang, ""
                        )

                    # Post-translation check: repetition / target-language ratio; retry up to N times
                    post_check_retry = 0
                    try:
                        max_post = int(self.get_param_value("post_check_max_retries") or 0)
                    except (TypeError, ValueError):
                        max_post = 0
                    while (
                        not getattr(self, "_inside_per_line_post_check", False)
                        and self._post_check_fail(ordered_translations)
                        and post_check_retry < max_post
                    ):
                        post_check_retry += 1
                        self.logger.info(
                            f"Post-check retry {post_check_retry}/{max_post}. Re-requesting translation."
                        )
                        time.sleep(self.retry_timeout)
                        parsed_response = self._request_translation(prompt, expected_count=num_src)
                        if not parsed_response or not parsed_response.translations:
                            break
                        # Re-apply strict alignment after post-check retry; if still broken, keep last ordered_translations.
                        try:
                            translations_by_id = {int(item.id): item.translation for item in parsed_response.translations if getattr(item, "id", None) is not None}
                            if set(range(1, num_src + 1)).issubset(set(translations_by_id.keys())):
                                ordered_translations = [translations_by_id.get(i, "") for i in range(1, num_src + 1)]
                        except Exception:
                            pass

                    if not getattr(self, "_inside_per_line_post_check", False):
                        page_key_pc = getattr(self, "_current_page_key", "") or ""
                        if self._page_key_subtitle_batch_pipeline(page_key_pc):
                            ordered_translations = self._apply_video_per_line_quality_fixes(
                                src_list[:num_src],
                                ordered_translations,
                                to_lang,
                                num_src,
                            )
                            ordered_translations = self._apply_per_line_post_check_retries(
                                src_list[:num_src],
                                ordered_translations,
                                to_lang,
                                num_src,
                            )

                    # Optional reflection pass (VideoCaptioner-style: review and improve)
                    if self.get_param_value("reflection_translation") and len(ordered_translations) == num_src:
                        ordered_translations = self._reflection_improve(
                            src_list[:num_src], ordered_translations, to_lang
                        )
                    # Apply keyword substitutions (e.g. powers -> abilities)
                    ordered_translations = [
                        self._apply_keyword_substitutions(t) for t in ordered_translations
                    ]
                    self._check_glossary_terms_in_translations(src_list[:num_src], ordered_translations)
                    translations.extend(ordered_translations)
                    # Video: expose optional revised_previous so dialog can update context/SRT/cache
                    page_key = getattr(self, "_current_page_key", None) or ""
                    if page_key.startswith("video_") and getattr(parsed_response, "revised_previous", None):
                        rev = parsed_response.revised_previous
                        if isinstance(rev, list) and rev:
                            setattr(self, "_last_revised_previous", [str(s).strip() for s in rev if s])
                    if self.get_param_value("extract_glossary"):
                        self._extract_and_append_glossary(src_list[:num_src], ordered_translations)
                    self.logger.info(
                        f"Successfully translated batch of {num_src}. Tokens used: {self.token_count_last}"
                    )
                    break

                except ContextLengthError as e:
                    # Provider tokenization can exceed our estimate: drop whole history
                    # pages and retry without consuming the ordinary retry budget.
                    context_recovery_attempt += 1
                    max_recoveries = len(getattr(getattr(self, "_llm_request_context", None), "history", ()) or ())
                    if context_recovery_attempt > max_recoveries or not self._recover_from_context_length():
                        raise CriticalTranslationError(
                            "The request exceeds the model's context length. Reduce the history token budget "
                            "in Config → Translator, or use a model with a larger context window.",
                            cause=e,
                        )

                except RETRYABLE_EXCEPTIONS as e:
                    api_retry_attempt += 1
                    err_msg = str(e)
                    # OpenRouter 404 "model not found" / "model moved" → do not retry; user must change model in settings.
                    if self.provider == "OpenRouter" and "404" in err_msg:
                        if "data policy" in err_msg.lower():
                            self.logger.info(
                                "OpenRouter 404: enable 'Free model publication' (or your model) at https://openrouter.ai/settings/privacy"
                            )
                        elif "find it here" in err_msg.lower() or "was a stealth" in err_msg.lower() or "mimo-v2" in err_msg.lower():
                            raise CriticalTranslationError(
                                "OpenRouter returned 404: this model was removed or renamed. Change the model in Translator → LLM API (e.g. to xiaomi/mimo-v2-pro if you used Hunter Alpha). Check the log or OpenRouter docs for the new model ID.",
                                cause=e,
                            )
                    self.logger.warning(
                        f"API Error (retryable): {type(e).__name__} - {e}. Attempt {api_retry_attempt}/{self.retry_attempts}."
                    )
                    if isinstance(e, (openai.APIConnectionError, httpx.RequestError)):
                        self.logger.info(
                            "Connection error: check firewall/proxy, VPN, or DNS. If behind a proxy, set 'Proxy' in translator settings (e.g. http://proxy:port). Test: curl -I https://openrouter.ai"
                        )
                    is_rate_limit = self.provider == "OpenRouter" and "429" in err_msg and ("rate limit" in err_msg.lower() or "Rate limit" in err_msg)
                    if is_rate_limit:
                        self.logger.info(
                            "Rate limit (429). Waiting %s s then retrying (%s/%s).",
                            self.rate_limit_delay, api_retry_attempt, self.retry_attempts,
                        )
                    if api_retry_attempt >= self.retry_attempts:
                        raise CriticalTranslationError(
                            f"Failed to connect to API after {self.retry_attempts} attempts.",
                            cause=e,
                        )
                    # Use longer delay for rate limit so upstream can recover
                    delay = self.rate_limit_delay if is_rate_limit else self.retry_timeout
                    time.sleep(delay)

                except json.JSONDecodeError as e:
                    api_retry_attempt += 1
                    self.logger.warning(
                        f"Malformed JSON from API (retrying). Attempt {api_retry_attempt}/{self.retry_attempts}."
                    )
                    if api_retry_attempt >= self.retry_attempts:
                        self.logger.warning(
                            "Malformed JSON after retries. Trying each snippet one-by-one (single-item fallback)."
                        )
                        batch_src = src_list[len(translations) : len(translations) + num_src]
                        fallback = self._translate_single_item_fallback(
                            batch_src, to_lang, "[ERROR: Invalid JSON]"
                        )
                        translations.extend(fallback)
                        break
                    time.sleep(self.retry_timeout)

                except ValueError as e:
                    if "empty or invalid parsed response" in str(e):
                        api_retry_attempt += 1
                        self.logger.warning(
                            f"Empty or invalid API response. Attempt {api_retry_attempt}/{self.retry_attempts}."
                        )
                        if api_retry_attempt >= self.retry_attempts:
                            raise CriticalTranslationError(
                                "Empty or invalid API response after retries.",
                                cause=e,
                            )
                        time.sleep(self.retry_timeout)
                    else:
                        page = getattr(self, '_current_page_key', None)
                        self.logger.error(
                            f"Fatal Error: An unrecoverable error occurred: {type(e).__name__} - {e}"
                            + (f" (page: {page})" if page else "")
                        )
                        self.logger.debug(traceback.format_exc())
                        translations.extend([f"[ERROR: {type(e).__name__}]"] * num_src)
                        break

                except (
                    ValidationError,
                    openai.BadRequestError,
                    openai.AuthenticationError,
                ) as e:
                    err_msg = str(e).lower()
                    is_auth = isinstance(e, openai.AuthenticationError)
                    is_quota = "quota" in err_msg or "limit exceeded" in err_msg
                    if is_auth or is_quota:
                        raise CriticalTranslationError(
                            "Authentication or quota error; cannot continue.",
                            cause=e,
                        )
                    # Retry on ValidationError / BadRequest (e.g. malformed API response) before giving up
                    api_retry_attempt += 1
                    page = getattr(self, "_current_page_key", None)
                    self.logger.warning(
                        "Validation/API error (retrying): %s - %s. Attempt %s/%s%s.",
                        type(e).__name__,
                        e,
                        api_retry_attempt,
                        self.retry_attempts,
                        f" (page: {page})" if page else "",
                    )
                    if api_retry_attempt >= self.retry_attempts:
                        self.logger.warning(
                            "Validation error after retries. Trying each snippet one-by-one (single-item fallback)."
                        )
                        batch_src = src_list[len(translations) : len(translations) + num_src]
                        fallback = self._translate_single_item_fallback(
                            batch_src, to_lang, f"[ERROR: {type(e).__name__}]"
                        )
                        translations.extend(fallback)
                        break
                    time.sleep(self.retry_timeout)
                    continue

        # Growth/eviction stays speculative until every batch of the page parsed.
        self.commit_history_window()
        return translations

    def unload_model(self, empty_cache=False):
        # Rendered history belongs to one loaded model/profile; drop it on unload.
        self._history_window = None
        self._llm_request_context = None
        return super().unload_model(empty_cache=empty_cache)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key == "endpoint_preset":
            preset_name = str(param_content or "Auto (provider default)")
            provider, endpoint = ENDPOINT_PRESETS.get(preset_name, (None, ""))
            if provider:
                self.params["provider"]["value"] = provider
            self.params["endpoint"]["value"] = endpoint

        if param_key in ["proxy", "multiple_keys", "apikey", "provider", "endpoint"]:
            self.client = None
            if param_key == "apikey" or param_key == "provider":
                try:
                    from utils.endpoints.openrouter import openrouter_clear_models_cache
                    openrouter_clear_models_cache()
                except Exception:
                    pass
