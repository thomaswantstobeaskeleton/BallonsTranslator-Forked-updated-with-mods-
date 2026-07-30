"""
Gemini / Mistral translators via neverliie-ai-sdk.

Implements EasyScanlate-style Gemini + Mistral translation using the
neverliie-ai-sdk unified client, as described in docs/EASYSCANLATE_INTEGRATION.md.

Requires:
    pip install neverliie-ai-sdk
"""
from typing import List, Dict

from .base import BaseTranslator, register_translator
from .exceptions import MissingTranslatorParams
from utils.io_utils import text_is_empty

try:
    from neverliie_ai_sdk import Google as NeverliieGoogle, Mistral as NeverliieMistral
    _NEVERLIIE_AVAILABLE = True
except Exception:
    NeverliieGoogle = None
    NeverliieMistral = None
    _NEVERLIIE_AVAILABLE = False


def _build_lang_map() -> Dict[str, str]:
    """
    Shared language display map for Gemini/Mistral translators.
    Matches the human-readable names used in other translators.

    Every language of ``LANGMAP_GLOBAL`` is listed here: the entries with a
    non-empty value are the ones offered in the source/target selectors, so a
    missing entry silently hides that language (issue #151: Hungarian).
    """
    return {
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
        "Hindi": "Hindi",
        "Malayalam": "Malayalam",
        "Tamil": "Tamil",
    }


class _NeverliieBaseTranslator(BaseTranslator):
    """
    Small helper base to share common neverliie-ai-sdk behavior.
    Subclasses must set:
        _provider_name: str (for error messages)
        _default_model: str (provider-specific default model)
        _client_cls: neverliie_ai_sdk client class
    """

    concate_text = False
    cht_require_convert = True
    _client_cls = None
    _provider_name = "LLM"
    _default_model = ""

    params: Dict = {
        "api_key": {
            "value": "",
            "description": "API key for the provider (stored in config.json, which is gitignored).",
        },
        "model": {
            "type": "line_editor",
            "value": "",
            "description": "Optional model name. Leave empty to use the provider default "
                           "(e.g. gemini-1.5-flash or mistral-small-latest).",
        },
        "delay": {
            "value": 0.2,
            "description": "Global delay in seconds between requests.",
        },
    }

    def _setup_translator(self):
        if not _NEVERLIIE_AVAILABLE or self._client_cls is None:
            raise MissingTranslatorParams(
                f"{self._provider_name} translator requires neverliie-ai-sdk.\n"
                f"Install with: pip install neverliie-ai-sdk"
            )
        api_key = (self.get_param_value("api_key") or "").strip()
        if not api_key:
            raise MissingTranslatorParams(
                f"{self._provider_name} translator requires an API key.\n"
                f"Set it in Config → Translator → {self.name} → api_key."
            )
        self.client = self._client_cls(api_key=api_key)
        # Human-readable language names for prompts
        self.lang_map = _build_lang_map()

    @property
    def _model_name(self) -> str:
        m = (self.get_param_value("model") or "").strip()
        return m or self._default_model

    def _translate_single(self, text: str) -> str:
        """Translate a single string; subclasses use same behavior."""
        if text_is_empty(text):
            return ""
        from_lang = self.lang_map.get(self.lang_source, self.lang_source)
        to_lang = self.lang_map.get(self.lang_target, self.lang_target)
        prompt = (
            f"Translate the following comic / manga dialogue from {from_lang} to {to_lang}.\n"
            f"- Preserve tone and character voice.\n"
            f"- Keep it concise and natural.\n"
            f"- Output ONLY the translated text, no explanations or prefixes.\n\n"
            f"{text}"
        )
        try:
            response = self.client.chat(messages=prompt, model=self._model_name or None)
        except Exception as e:
            self.logger.error("%s translation failed: %s", self._provider_name, e)
            return ""
        try:
            choice = (response or {}).get("choices", [])[0]
            message = (choice or {}).get("message", {})
            content = message.get("content", "")
            # neverliie normalized responses use string content; be robust if list/dict appears
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        parts.append(str(part.get("text", "")))
                    else:
                        parts.append(str(part))
                content = "".join(parts)
            return str(content).strip()
        except Exception as e:
            self.logger.error("%s: failed to parse response: %s", self._provider_name, e)
            return ""

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []
        # Lazy client init if updateParam invalidated it
        if not hasattr(self, "client") or self.client is None:
            self.setup_translator()
        out: List[str] = []
        for src in src_list:
            out.append(self._translate_single(src))
        return out

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ("api_key", "model"):
            # Force lazy re-init on next use
            if hasattr(self, "client"):
                self.client = None


if _NEVERLIIE_AVAILABLE:

    @register_translator("Gemini_neverliie")
    class GeminiNeverliieTranslator(_NeverliieBaseTranslator):
        """Google Gemini translation via neverliie-ai-sdk (EasyScanlate-style)."""

        _provider_name = "Gemini (neverliie-ai-sdk)"
        _default_model = "gemini-1.5-flash"
        _client_cls = NeverliieGoogle


    @register_translator("Mistral_neverliie")
    class MistralNeverliieTranslator(_NeverliieBaseTranslator):
        """Mistral translation via neverliie-ai-sdk."""

        _provider_name = "Mistral (neverliie-ai-sdk)"
        _default_model = "mistral-small-latest"
        _client_cls = NeverliieMistral

