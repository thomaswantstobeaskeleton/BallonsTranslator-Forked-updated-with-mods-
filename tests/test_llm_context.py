"""Context-aware LLM translation: rolling history window, glossary, recovery.

Ported from upstream dmMaze/BallonsTranslator's llm_context work; these tests cover
the parts our fork wires into its own translator and pipeline.
"""
import json
import sys
import types

import pytest

cv2_stub = sys.modules.get("cv2") or types.ModuleType("cv2")
for _name, _value in (
    ("IMREAD_COLOR", 1), ("IMREAD_GRAYSCALE", 0), ("COLOR_GRAY2RGB", 0),
    ("INTER_LINEAR", 1), ("INTER_AREA", 3), ("INTER_NEAREST", 0), ("INTER_CUBIC", 2),
    ("INTER_LANCZOS4", 4), ("BORDER_CONSTANT", 0), ("BORDER_REFLECT", 2), ("BORDER_REPLICATE", 1),
):
    if not hasattr(cv2_stub, _name):
        setattr(cv2_stub, _name, _value)
cv2_stub.cvtColor = getattr(cv2_stub, "cvtColor", lambda img, code: img)
sys.modules["cv2"] = cv2_stub

from modules.context.errors import ContextLengthError, is_context_length_error  # noqa: E402
from modules.context.glossary import (  # noqa: E402
    GlossaryEntry,
    GlossaryError,
    load_glossary,
    render_glossary,
    select_glossary,
)
from modules.context.history import (  # noqa: E402
    ContextAction,
    ContextReason,
    HistoryPage,
    HistoryWindow,
    HistoryWindowKey,
    RenderedHistoryPage,
    RequestContext,
    eligible_history_for_request,
    recover_context_length,
    window_rebuild_reason,
)


def _rendered(page_key, tokens, sources=('s',), translations=('t',)):
    snapshot = HistoryPage(page_key, tuple(sources), tuple(translations))
    return RenderedHistoryPage(snapshot, (('user', 'u'), ('assistant', 'a')), tokens)


class _Project:
    def __init__(self, page_keys):
        self.pages = {key: [] for key in page_keys}
        self.load_identity = object()


# --------------------------------------------------------------------------
# glossary
# --------------------------------------------------------------------------

def test_glossary_formats(tmp_path):
    json_path = tmp_path / 'g.json'
    json_path.write_text(json.dumps([
        {"src": "勇者", "dst": "Hero", "info": "title"},
        {"src": "魔王", "dst": "Demon Lord"},
    ]), encoding='utf-8')
    assert load_glossary(json_path) == (
        GlossaryEntry('勇者', 'Hero', 'title'),
        GlossaryEntry('魔王', 'Demon Lord'),
    )

    txt_path = tmp_path / 'g.txt'
    txt_path.write_text("# comment\n勇者->Hero #title\n", encoding='utf-8')
    assert load_glossary(txt_path) == (GlossaryEntry('勇者', 'Hero', 'title'),)

    tsv_path = tmp_path / 'g.tsv'
    tsv_path.write_text("勇者\tHero\ttitle\n", encoding='utf-8')
    assert load_glossary(tsv_path) == (GlossaryEntry('勇者', 'Hero', 'title'),)

    assert load_glossary('') == ()


def test_glossary_reports_conflicts_and_missing_files(tmp_path):
    path = tmp_path / 'g.txt'
    path.write_text("Hero->勇者\nhero->英雄\n", encoding='utf-8')
    with pytest.raises(GlossaryError):
        load_glossary(path)
    with pytest.raises(GlossaryError):
        load_glossary(tmp_path / 'missing.txt')


def test_glossary_selection_and_rendering():
    entries = (GlossaryEntry('Hero', '勇者'), GlossaryEntry('Mage', '魔法使'))
    assert select_glossary(entries, ['A HERO appears.'], 'matching') == (entries[0],)
    assert select_glossary(entries, ['nothing'], 'all') == entries
    assert render_glossary([entries[0]]) == (
        '{"glossary":[{"source":"Hero","translation":"勇者","note":""}]}'
    )
    assert render_glossary([]) == ''


# --------------------------------------------------------------------------
# history window
# --------------------------------------------------------------------------

def test_window_rebuild_reasons():
    key = HistoryWindowKey(load_identity=object(), settings=(('model', 'm'),))
    assert window_rebuild_reason(None, None, '2.png', key) is ContextReason.WINDOW_EMPTY

    project = _Project(['1.png', '2.png', '3.png'])
    window_key = HistoryWindowKey(load_identity=project.load_identity, settings=(('model', 'm'),))
    window = HistoryWindow(window_key, '1.png', (), 0)
    assert window_rebuild_reason(window, project, '2.png', window_key) is None
    # a gap between the window's page and this one
    assert window_rebuild_reason(window, project, '3.png', window_key) is ContextReason.NON_ADJACENT
    # a reloaded project gets a new identity
    other_key = HistoryWindowKey(load_identity=object(), settings=(('model', 'm'),))
    assert window_rebuild_reason(window, project, '2.png', other_key) is ContextReason.PROJECT_CHANGED
    # changed settings (e.g. another model) must not reuse rendered pages
    changed = HistoryWindowKey(load_identity=project.load_identity, settings=(('model', 'other'),))
    assert window_rebuild_reason(window, project, '2.png', changed) is ContextReason.SETTINGS_CHANGED


def test_history_grows_then_evicts_in_bulk():
    project = _Project(['1.png', '2.png'])
    key = HistoryWindowKey(project.load_identity, ())
    window = HistoryWindow(key, '1.png', (_rendered('0.png', 30),), 30)

    history, diagnostic = eligible_history_for_request(
        window=window,
        project=project,
        page_key='2.png',
        previous_page=HistoryPage('1.png', ('s',), ('t',)),
        token_budget=100,
        rebuild_reason=None,
        snapshot_page=lambda key_: None,
        render_page=lambda page: _rendered(page.page_key, 30),
    )
    assert diagnostic.action is ContextAction.GROW
    assert [page.page_key for page in history] == ['0.png', '1.png']

    full_window = HistoryWindow(key, '1.png', (_rendered('a', 40), _rendered('b', 40)), 80)
    history, diagnostic = eligible_history_for_request(
        window=full_window,
        project=project,
        page_key='2.png',
        previous_page=HistoryPage('1.png', ('s',), ('t',)),
        token_budget=100,
        rebuild_reason=None,
        snapshot_page=lambda key_: None,
        render_page=lambda page: _rendered(page.page_key, 40),
    )
    assert diagnostic.action is ContextAction.EVICT
    assert diagnostic.evicted >= 1
    assert diagnostic.token_count <= 100


def test_rebuild_leaves_headroom_for_the_next_page():
    project = _Project(['1.png', '2.png', '3.png', '4.png'])
    history, diagnostic = eligible_history_for_request(
        window=None,
        project=project,
        page_key='4.png',
        previous_page=None,
        token_budget=100,
        rebuild_reason=ContextReason.WINDOW_EMPTY,
        snapshot_page=lambda key_: HistoryPage(key_, ('s',), ('t',)),
        render_page=lambda page: _rendered(page.page_key, 30),
    )
    assert diagnostic.action is ContextAction.REBUILD
    # low water is 60% of the budget, so only two 30-token pages are taken
    assert diagnostic.token_count <= 60
    assert [page.page_key for page in history] == ['2.png', '3.png']


def test_context_recovery_drops_whole_pages():
    context = RequestContext(
        history=(_rendered('a', 40), _rendered('b', 40), _rendered('c', 40)),
        history_budget=100,
        request_page_key='4.png',
    )
    recovered = recover_context_length(context)
    assert recovered is not None
    assert len(recovered.history) < 3
    assert recovered.diagnostic.action is ContextAction.CONTEXT_RECOVERY
    assert recover_context_length(RequestContext(())) is None


def test_context_length_error_detection():
    assert is_context_length_error(RuntimeError('maximum context length exceeded'))
    assert is_context_length_error(RuntimeError('prompt is too long'))
    assert not is_context_length_error(RuntimeError('rate limit reached'))
    assert not is_context_length_error(RuntimeError("Unsupported parameter: 'max_tokens'"))
    assert issubclass(ContextLengthError, RuntimeError)


# --------------------------------------------------------------------------
# config + project plumbing
# --------------------------------------------------------------------------

def test_context_settings_are_validated():
    from utils.config import LLMGlossaryMode, LLMTranslateContext, ModuleConfig

    assert ModuleConfig().llm_translate_context == LLMTranslateContext.PAGE
    assert ModuleConfig(llm_translate_context='bogus').llm_translate_context == LLMTranslateContext.PAGE
    assert ModuleConfig(llm_translate_context='history').llm_translate_context == LLMTranslateContext.HISTORY
    assert ModuleConfig(llm_glossary_mode='nope').llm_glossary_mode == LLMGlossaryMode.Matching
    assert ModuleConfig(llm_prior_context_token_budget=0).llm_prior_context_token_budget == 4096
    assert ModuleConfig(llm_prior_context_token_budget=8192).llm_prior_context_token_budget == 8192


def test_project_tracks_load_identity_and_translation_target():
    from utils.config import RunStatus
    from utils.proj_imgtrans import ProjImgTrans

    project = ProjImgTrans()
    assert project.load_identity is project.load_identity
    project.pages = {'001.png': []}
    project._image_info = {'001.png': {'finish_code': 0}}

    project.mark_translation_finished('001.png', 'English')
    info = project._image_info['001.png']
    assert info['finish_code'] & RunStatus.FIN_TRANSLATE
    assert info['translation_target'] == 'English'

    # Re-running detection invalidates the translated marker and its target.
    project.invalidate_translation('001.png')
    assert not (project._image_info['001.png']['finish_code'] & RunStatus.FIN_TRANSLATE)
    assert 'translation_target' not in project._image_info['001.png']


# --------------------------------------------------------------------------
# translator integration (fake provider, no network)
# --------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)
        self.finish_reason = 'stop'


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13,
                      'prompt_cache_hit_tokens': 8}


class _FakeClient:
    """Records the messages of each request and replies with valid JSON."""

    def __init__(self, fail_first_with=None):
        self.requests = []
        self.fail_first_with = fail_first_with

        outer = self

        class _Completions:
            def create(self, **api_args):
                outer.requests.append(api_args)
                if outer.fail_first_with is not None and len(outer.requests) == 1:
                    error = outer.fail_first_with
                    outer.fail_first_with = None
                    raise error
                count = api_args.get('_expected', 0) or _sources_in(api_args)
                payload = {'translations': [
                    {'id': i + 1, 'translation': f'T{i + 1}'} for i in range(count)
                ]}
                return _FakeCompletion(json.dumps(payload, ensure_ascii=False))

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _sources_in(api_args) -> int:
    user = api_args['messages'][-1]['content']
    if isinstance(user, list):
        user = ' '.join(str(part.get('text', '')) for part in user if isinstance(part, dict))
    return user.count('"id":') or user.count('"id": ')


def _make_translator(monkeypatch, client):
    from modules.translators.trans_llm_api import LLM_API_Translator
    from utils.config import pcfg

    translator = LLM_API_Translator.__new__(LLM_API_Translator)
    translator.lang_source = '日本語'
    translator.lang_target = 'English'
    translator.lang_map = {'日本語': 'Japanese', 'English': 'English'}
    translator.name = 'LLM_API_Translator'
    translator._llm_request_context = None
    translator._history_window = None
    translator._current_page_key = ''
    translator.client = client
    monkeypatch.setattr(type(translator), 'logger', _NullLogger(), raising=False)
    monkeypatch.setattr(type(translator), '_build_system_prompt', lambda self: 'SYSTEM', raising=False)
    monkeypatch.setattr(type(translator), '_build_user_content_with_optional_image',
                        lambda self, prompt: prompt, raising=False)
    monkeypatch.setattr(pcfg.module, 'llm_translate_context', 'history', raising=False)
    monkeypatch.setattr(pcfg.module, 'llm_prior_context_token_budget', 4096, raising=False)
    monkeypatch.setattr(pcfg.module, 'llm_glossary_path', '', raising=False)
    monkeypatch.setattr(pcfg.module, 'llm_glossary_mode', 'matching', raising=False)
    return translator


class _NullLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _translated_project(page_keys, target='English'):
    from utils.config import RunStatus

    class _Block:
        def __init__(self, text, translation):
            self._text = text
            self.translation = translation

        def get_text(self):
            return self._text

    project = _Project(page_keys)
    for key in page_keys:
        project.pages[key] = [_Block(f'src-{key}', f'dst-{key}')]
    project._image_info = {
        key: {'finish_code': RunStatus.FIN_TRANSLATE, 'translation_target': target}
        for key in page_keys
    }
    return project


def test_history_pages_are_sent_as_message_pairs(monkeypatch):
    client = _FakeClient()
    translator = _make_translator(monkeypatch, client)
    project = _translated_project(['1.png', '2.png', '3.png'])

    translator.set_translation_project(project, '3.png')
    messages = translator._assemble_context_messages('CURRENT PROMPT', 'gpt-test')

    roles = [m['role'] for m in messages]
    assert roles[0] == 'system'
    assert roles[-1] == 'user'
    # prior pages arrive as real user/assistant turns, not as prompt text
    assert roles.count('assistant') >= 1
    assert any('dst-1.png' in str(m['content']) for m in messages)
    assert messages[-1]['content'] == 'CURRENT PROMPT'


def test_adjacent_pages_keep_a_stable_cache_prefix(monkeypatch):
    client = _FakeClient()
    translator = _make_translator(monkeypatch, client)
    project = _translated_project(['1.png', '2.png', '3.png'])

    translator.set_translation_project(project, '2.png')
    first = translator._assemble_context_messages('P2', 'gpt-test')
    translator.commit_history_window()

    translator.set_translation_project(project, '3.png')
    second = translator._assemble_context_messages('P3', 'gpt-test')

    # everything before the current user turn is unchanged and only grows
    assert second[:len(first) - 1] == first[:-1]
    assert len(second) > len(first)


def test_history_is_not_reused_for_another_target_language(monkeypatch):
    client = _FakeClient()
    translator = _make_translator(monkeypatch, client)
    project = _translated_project(['1.png', '2.png'], target='Français')

    translator.set_translation_project(project, '2.png')
    messages = translator._assemble_context_messages('CURRENT', 'gpt-test')
    assert [m['role'] for m in messages] == ['system', 'user']


def test_glossary_is_attached_to_the_request(monkeypatch, tmp_path):
    from utils.config import pcfg

    glossary = tmp_path / 'g.txt'
    glossary.write_text('src-1.png->GLOSSED\n', encoding='utf-8')

    client = _FakeClient()
    translator = _make_translator(monkeypatch, client)
    monkeypatch.setattr(pcfg.module, 'llm_glossary_path', str(glossary), raising=False)
    monkeypatch.setattr(pcfg.module, 'llm_glossary_mode', 'all', raising=False)
    project = _translated_project(['1.png', '2.png'])

    translator.set_translation_project(project, '2.png')
    messages = translator._assemble_context_messages('CURRENT src-1.png', 'gpt-test')
    assert any('GLOSSED' in str(m['content']) for m in messages if m['role'] == 'system')


def test_context_length_error_shrinks_history(monkeypatch):
    client = _FakeClient()
    translator = _make_translator(monkeypatch, client)
    project = _translated_project(['1.png', '2.png', '3.png', '4.png'])

    translator.set_translation_project(project, '4.png')
    before = len(translator._llm_request_context.history)
    assert before >= 2
    assert translator._recover_from_context_length() is True
    assert len(translator._llm_request_context.history) < before


def test_window_is_committed_only_after_the_page_parsed(monkeypatch):
    """A page advances the reusable window when (and only when) its batches parsed."""
    from modules.translators.trans_llm_api import LLM_API_Translator, TranslationElement, TranslationResponse

    client = _FakeClient()
    translator = _make_translator(monkeypatch, client)
    project = _translated_project(['1.png', '2.png', '3.png'])

    monkeypatch.setattr(
        type(translator), '_assemble_prompts',
        lambda self, queries, to_lang: iter([('PROMPT', len(queries))]), raising=False)
    monkeypatch.setattr(
        type(translator), '_request_translation',
        lambda self, prompt, expected_count=None: TranslationResponse(
            translations=[TranslationElement(id=i + 1, translation=f'T{i + 1}') for i in range(expected_count)]),
        raising=False)
    for attr, value in (
        ('invalid_repeat_count', 0), ('retry_attempts', 1), ('retry_timeout', 0),
        ('rate_limit_delay', 0), ('token_count_last', 0), ('_video_nlp_chunk_size', 0),
        ('_video_nlp_max_workers', 1),
    ):
        monkeypatch.setattr(type(translator), attr, value, raising=False)
    monkeypatch.setattr(type(translator), '_apply_keyword_substitutions', lambda self, t: t, raising=False)
    monkeypatch.setattr(type(translator), 'get_param_value', lambda self, key: 0, raising=False)

    translator.set_translation_project(project, '3.png')
    assert translator._history_window is None

    result = translator._translate(['source line'])
    assert result == ['T1']
    assert translator._history_window is not None
    assert translator._history_window.request_page_key == '3.png'
    assert [p.page_key for p in translator._history_window.history] == ['1.png', '2.png']
