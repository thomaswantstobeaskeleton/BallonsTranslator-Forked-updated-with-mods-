from typing import List, Callable

from modules import GET_VALID_INPAINTERS, GET_VALID_TEXTDETECTORS, GET_VALID_TRANSLATORS, GET_VALID_OCR, \
    BaseTranslator, DEFAULT_DEVICE, GPUINTENSIVE_SET
from modules.translators.base import lang_display_label, lang_display_to_key
from utils.logger import logger as LOGGER
from .custom_widget import ConfigComboBox, ParamComboBox, NoBorderPushBtn, ParamNameLabel
from .custom_widget.hover_animation import install_hover_opacity_animation, install_hover_scale_animation
from utils.shared import CONFIG_COMBOBOX_LONG, size2width, CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_HEIGHT
from utils.config import pcfg
from utils.module_tiers import format_module_tier_tooltip

from qtpy.QtWidgets import QPlainTextEdit, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QCheckBox, QLineEdit, QGridLayout, QPushButton, QMessageBox, QSpinBox, QDoubleSpinBox
from qtpy.QtCore import Qt, Signal, QTimer
from qtpy.QtGui import QDoubleValidator


class ParamCheckGroup(QWidget):

    paramwidget_edited = Signal(str, dict)

    def __init__(self, param_key, check_group: dict, parent=None) -> None:
        super().__init__(parent=parent)
        self.param_key = param_key
        layout = QHBoxLayout(self)
        self.label2widget = {}
        for k, v in check_group.items():
            checker = QCheckBox(text=k, parent=self)
            checker.setChecked(v)
            layout.addWidget(checker)
            self.label2widget[k] = checker
            checker.clicked.connect(self.on_checker_clicked)

    def on_checker_clicked(self):
        new_state_dict = {}
        w = QCheckBox()
        for k, w in self.label2widget.items():
            new_state_dict[k] = w.isChecked()
        self.paramwidget_edited.emit(self.param_key, new_state_dict)


class ParamLineEditor(QLineEdit):
    
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, force_digital, size='short', *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key
        self.setFixedWidth(size2width(size))
        self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.textChanged.connect(self.on_text_changed)

        if force_digital:
            validator = QDoubleValidator()
            self.setValidator(validator)

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.text())

class ParamEditor(QPlainTextEdit):
    
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key

        if param_key == 'chat sample':
            self.setFixedWidth(int(CONFIG_COMBOBOX_LONG * 1.2))
            self.setFixedHeight(200)
        else:
            self.setFixedWidth(CONFIG_COMBOBOX_LONG)
            self.setFixedHeight(100)
        # self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.textChanged.connect(self.on_text_changed)

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.text())

    def setText(self, text: str):
        self.setPlainText(text)

    def text(self):
        return self.toPlainText()


class ParamCheckerBox(QWidget):
    checker_changed = Signal(bool)
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.checker = QCheckBox()
        name_label = ParamNameLabel(param_key)
        hlayout = QHBoxLayout(self)
        hlayout.addWidget(name_label)
        hlayout.addWidget(self.checker)
        hlayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.checker.stateChanged.connect(self.on_checker_changed)

    def on_checker_changed(self):
        is_checked = self.checker.isChecked()
        self.checker_changed.emit(is_checked)
        checked = 'true' if is_checked else 'false'
        self.paramwidget_edited.emit(self.param_key, checked)


class ParamCheckBox(QCheckBox):
    paramwidget_edited = Signal(str, bool)
    def __init__(self, param_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.stateChanged.connect(self.on_checker_changed)

    def on_checker_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.isChecked())


def get_param_display_name(param_key: str, param_dict: dict = None):
    if param_dict is not None and isinstance(param_dict, dict):
        if 'display_name' in param_dict:
            return param_dict['display_name']
    return param_key


class ParamPushButton(QPushButton):
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, param_dict: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.setText(get_param_display_name(param_key, param_dict))
        self.clicked.connect(self.on_clicked)

    def on_clicked(self):
        self.paramwidget_edited.emit(self.param_key, '')


class ParamWidget(QWidget):

    paramwidget_edited = Signal(str, dict)
    def __init__(self, params, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layout = QHBoxLayout(self)
        self.param_layout = param_layout = QGridLayout()
        param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(param_layout)
        layout.addStretch(-1)

        if 'description' in params:
            self.setToolTip(params['description'])

        for ii, param_key in enumerate(params):
            if param_key == 'description' or param_key.startswith('__'):
                continue
            display_param_name = param_key

            require_label = True
            is_str = isinstance(params[param_key], str)
            is_digital = isinstance(params[param_key], float) or isinstance(params[param_key], int)
            param_widget = None

            if isinstance(params[param_key], bool):
                param_widget = ParamCheckBox(param_key)
                val = params[param_key]
                param_widget.setChecked(val)
                param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

            elif is_str or is_digital:
                param_widget = ParamLineEditor(param_key, force_digital=is_digital)
                val = params[param_key]
                if is_digital:
                    val = str(val)
                param_widget.setText(val)
                param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

            elif isinstance(params[param_key], dict):
                param_dict = params[param_key]
                display_param_name = get_param_display_name(param_key, param_dict)
                value = params[param_key]['value']
                param_widget = None  # Ensure initialization
                param_type = param_dict['type'] if 'type' in param_dict else 'line_editor'
                flush_btn = param_dict.get('flush_btn', False)
                path_selector = param_dict.get('path_selector', False)
                param_size = param_dict.get('size', 'short')
                if param_type == 'selector':
                    if 'url' in param_key:
                        size = size2width('median')
                    else:
                        size = size2width(param_size)

                    options_list = list(param_dict.get('options', []))
                    if param_key == 'device':
                        default_label = self.tr('Default')
                        if default_label not in options_list:
                            options_list = [default_label] + options_list
                        effective_default = (getattr(pcfg, 'default_device', None) or '').strip() or DEFAULT_DEVICE
                        if not str(value).strip() or value == effective_default:
                            value = default_label
                    param_widget = ParamComboBox(
                        param_key, options_list, size=size, scrollWidget=scrollWidget, flush_btn=flush_btn, path_selector=path_selector)

                    if param_key == 'device' and DEFAULT_DEVICE == 'cpu' and str(param_dict.get('value', '')).strip() == '':
                        # Keep CPU as the implicit default when runtime did not detect GPU,
                        # but do not disable GPU choices in UI: users may still run with
                        # CUDA/XPU/MPS in environments where torch detection is incomplete.
                        param_dict['value'] = 'cpu'
                    param_widget.setCurrentText(str(value))
                    param_widget.setEditable(param_dict.get('editable', False))

                elif param_type == 'editor':
                    param_widget = ParamEditor(param_key)
                    param_widget.setText(value)

                elif param_type == 'checkbox':
                    param_widget = ParamCheckBox(param_key)
                    if isinstance(value, str):
                        value = value.lower().strip() == 'true'
                        params[param_key]['value'] = value
                    param_widget.setChecked(value)

                elif param_type == 'pushbtn':
                    param_widget = ParamPushButton(param_key, param_dict)
                    require_label = False

                elif param_type == 'line_editor':
                    param_widget = ParamLineEditor(param_key, force_digital=is_digital)
                    param_widget.setText(str(value))

                elif param_type == 'text':
                    param_widget = ParamLineEditor(param_key, force_digital=False, size=param_dict.get('size', 'long'))
                    param_widget.setText(str(value))

                elif param_type == 'check_group':
                    param_widget = ParamCheckGroup(param_key, check_group=value)

                if param_widget is not None:
                    param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)
                    if 'description' in param_dict:
                        param_widget.setToolTip(param_dict['description'])

                # Fix #904: fallback for unknown/malformed param types so a missing option
                # list or unrecognised param_type doesn't crash the module config UI.
                if param_widget is None:
                    param_widget = ParamLineEditor(param_key, force_digital=is_digital)
                    param_widget.setText(str(value))
                    param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

            widget_idx = 0
            if require_label:
                param_label = ParamNameLabel(display_param_name)
                param_layout.addWidget(param_label, ii, 0)
                widget_idx = 1
            if param_widget is not None:
                pw_lo = None
                if hasattr(param_widget, 'flush_btn') or hasattr(param_widget, 'path_select_btn'):
                    pw_lo = QHBoxLayout()
                    pw_lo.addWidget(param_widget)
                if hasattr(param_widget, 'flush_btn'):
                    pw_lo.addWidget(param_widget.flush_btn)
                    param_widget.flushbtn_clicked.connect(self.on_flushbtn_clicked)
                if hasattr(param_widget, 'path_select_btn'):
                    pw_lo.addWidget(param_widget.path_select_btn)
                    param_widget.pathbtn_clicked.connect(self.on_pathbtn_clicked)
                if pw_lo is None:
                    param_layout.addWidget(param_widget, ii, widget_idx)
                else:
                    param_layout.addLayout(pw_lo, ii, widget_idx)
            else:
                v = params[param_key]
                raise ValueError(f"Failed to initialize widget for key-value pair: {param_key}-{v}")
            
    def on_flushbtn_clicked(self):
        paramw: ParamComboBox = self.sender()
        content_dict = {'content': '', 'widget': paramw, 'flush': True}
        self.paramwidget_edited.emit(paramw.param_key, content_dict)

    def on_pathbtn_clicked(self):
        paramw: ParamComboBox = self.sender()
        content_dict = {'content': '', 'widget': paramw, 'select_path': True}
        self.paramwidget_edited.emit(paramw.param_key, content_dict)

    def on_paramwidget_edited(self, param_key, param_content):
        if param_key == 'device' and param_content == self.tr('Default'):
            param_content = (getattr(pcfg, 'default_device', None) or '').strip() or DEFAULT_DEVICE
        content_dict = {'content': param_content}
        self.paramwidget_edited.emit(param_key, content_dict)

class ModuleParseWidgets(QWidget):
    def addModulesParamWidgets(self, ocr_instance):
        self.params = ocr_instance.get_params()
        self.on_module_changed()

    def on_module_changed(self):
        self.updateModuleParamWidget()

    def updateModuleParamWidget(self):
        widget = ParamWidget(self.params, scrollWidget=self)
        layout = QVBoxLayout()
        layout.addWidget(widget)
        self.setLayout(layout)

class ModuleConfigParseWidget(QWidget):
    module_changed = Signal(str)
    paramwidget_edited = Signal(str, dict)
    def __init__(self, module_name: str, get_valid_module_keys: Callable, scrollWidget: QWidget, add_from: int = 1, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.get_valid_module_keys = get_valid_module_keys
        self.module_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.params_layout = QHBoxLayout()
        self.params_layout.setContentsMargins(0, 0, 0, 0)

        p_layout = QHBoxLayout()
        p_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.module_label = ParamNameLabel(module_name)
        p_layout.addWidget(self.module_label)
        p_layout.addWidget(self.module_combobox)
        p_layout.addStretch(-1)
        self.p_layout = p_layout

        layout = QVBoxLayout(self)
        self.param_widget_map = {}
        layout.addLayout(p_layout) 
        layout.addLayout(self.params_layout)
        layout.setSpacing(30)
        self.vlayout = layout

        self.visibleWidget: QWidget = None
        self.module_dict: dict = {}

    def clearModuleList(self):
        """Clear combobox, param widgets, and map. Use before repopulating (e.g. on dev_mode toggle)."""
        try:
            self.module_combobox.currentTextChanged.disconnect(self.on_module_changed)
        except (TypeError, AttributeError):
            pass
        self.module_combobox.blockSignals(True)
        self.module_combobox.clear()
        self.module_combobox.blockSignals(False)
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.param_widget_map.clear()
        self.visibleWidget = None

    def addModulesParamWidgets(self, module_dict: dict):
        invalid_module_keys = []
        valid_modulekeys = self.get_valid_module_keys()

        num_widgets_before = len(self.param_widget_map)

        for module in module_dict:
            if module not in valid_modulekeys:
                invalid_module_keys.append(module)
                continue

            if module in self.param_widget_map:
                LOGGER.warning(f'duplicated module key: {module}')
                continue

            self.module_combobox.addItem(module)
            tip = format_module_tier_tooltip(
                module,
                show_badge=bool(getattr(pcfg, "show_module_tier_badges_in_tooltips", True)),
            )
            if tip:
                self.module_combobox.setItemData(
                    self.module_combobox.count() - 1,
                    tip,
                    Qt.ItemDataRole.ToolTipRole,
                )
            params = module_dict[module]
            if params is not None:
                self.param_widget_map[module] = None

        if len(invalid_module_keys) > 0:
            LOGGER.warning(F'Invalid module keys: {invalid_module_keys}')
            for ik in invalid_module_keys:
                module_dict.pop(ik)

        self.module_dict = module_dict

        num_widgets_after = len(self.param_widget_map)
        if num_widgets_before == 0 and num_widgets_after > 0:
            self.on_module_changed()
            self.module_combobox.currentTextChanged.connect(self.on_module_changed)

    def setModule(self, module: str):
        self.blockSignals(True)
        self.module_combobox.setCurrentText(module)
        self.updateModuleParamWidget()
        self.blockSignals(False)

    def updateModuleParamWidget(self):
        module = self.module_combobox.currentText()
        if self.visibleWidget is not None:
            self.visibleWidget.hide()
        if module in self.param_widget_map:
            widget: QWidget = self.param_widget_map[module]
            if widget is None:
                # lazy load widgets
                params = self.module_dict[module]
                widget = ParamWidget(params, scrollWidget=self)
                widget.paramwidget_edited.connect(self.paramwidget_edited)
                self.param_widget_map[module] = widget
                self.params_layout.addWidget(widget)
            else:
                widget.show()
            self.visibleWidget = widget

    def on_module_changed(self):
        self.updateModuleParamWidget()
        self.module_changed.emit(self.module_combobox.currentText())


class TranslatorConfigPanel(ModuleConfigParseWidget):

    show_pre_MT_keyword_window = Signal()
    show_MT_keyword_window = Signal()
    show_OCR_keyword_window = Signal()
    show_translation_context_requested = Signal()
    test_translator_clicked = Signal()
    copy_manual_prompt_requested = Signal()
    paste_manual_response_requested = Signal()

    def __init__(self, module_name, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_TRANSLATORS, scrollWidget=scrollWidget, *args, **kwargs)
        self.translator_changed = self.module_changed
    
        self.source_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.target_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.testTranslatorBtn = QPushButton(self.tr("Test translator"), self)
        self.testTranslatorBtn.setToolTip(self.tr("Test if the current translator is available (e.g. API key, network)."))
        self.testTranslatorBtn.clicked.connect(self.test_translator_clicked.emit)
        install_hover_opacity_animation(self.testTranslatorBtn, duration_ms=100, normal_opacity=0.9)
        install_hover_scale_animation(self.testTranslatorBtn, duration_ms=80, size_delta=(3, 2))
        self.replacePreMTkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for machine translation source text"), self)
        self.replacePreMTkeywordBtn.clicked.connect(self.show_pre_MT_keyword_window)
        self.replacePreMTkeywordBtn.setFixedWidth(500)
        self.replaceMTkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for machine translation"), self)
        self.replaceMTkeywordBtn.clicked.connect(self.show_MT_keyword_window)
        self.replaceMTkeywordBtn.setFixedWidth(500)
        self.replaceOCRkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for source text"), self)
        self.replaceOCRkeywordBtn.clicked.connect(self.show_OCR_keyword_window)
        self.replaceOCRkeywordBtn.setFixedWidth(500)
        self.translateByTextblockBox = ParamCheckerBox(self.tr('Translate each text block individually'))
        self.translationContextBtn = NoBorderPushBtn(self.tr("Translation context (project)..."), self)
        self.translationContextBtn.setToolTip(self.tr("Set series path and glossary for this project (cross-chapter consistency)."))
        self.translationContextBtn.clicked.connect(self.show_translation_context_requested.emit)
        self.translationContextBtn.setFixedWidth(500)

        # Section 19: continue batch on soft translation failure (vs stop and show dialog)
        self.soft_failure_continue_checker = QCheckBox(self.tr("Continue batch on soft translation failure"))
        self.soft_failure_continue_checker.setToolTip(self.tr(
            "When checked (default), non-critical translation failures (e.g. timeout, parse error) use a placeholder and continue the batch. "
            "When unchecked, show an error dialog and stop like critical errors (auth, quota)."))
        self.soft_failure_continue_checker.setChecked(getattr(pcfg.module, "translation_soft_failure_continue", True))
        self.soft_failure_continue_checker.clicked.connect(self._on_soft_failure_continue_changed)
        self.vlayout.addWidget(self.soft_failure_continue_checker)

        # Section 10: manual translation clipboard helper (visible only when translator is "manual")
        self.manual_helper_widget = QWidget(self)
        manual_hl = QHBoxLayout(self.manual_helper_widget)
        manual_hl.setContentsMargins(0, 4, 0, 4)
        self.copyPromptBtn = QPushButton(self.tr("Copy prompt"), self.manual_helper_widget)
        self.copyPromptBtn.setToolTip(self.tr("Copy the translation prompt (JSON with source texts) to clipboard. Paste into your tool, then paste the response back and click Paste response."))
        self.copyPromptBtn.clicked.connect(self.copy_manual_prompt_requested.emit)
        install_hover_opacity_animation(self.copyPromptBtn, duration_ms=100, normal_opacity=0.9)
        install_hover_scale_animation(self.copyPromptBtn, duration_ms=80, size_delta=(3, 2))
        self.pasteResponseBtn = QPushButton(self.tr("Paste response"), self.manual_helper_widget)
        self.pasteResponseBtn.setToolTip(self.tr("Paste JSON from clipboard into the response box and apply to blocks. Run Translate to use it."))
        self.pasteResponseBtn.clicked.connect(self.paste_manual_response_requested.emit)
        install_hover_opacity_animation(self.pasteResponseBtn, duration_ms=100, normal_opacity=0.9)
        install_hover_scale_animation(self.pasteResponseBtn, duration_ms=80, size_delta=(3, 2))
        manual_hl.addWidget(self.copyPromptBtn)
        manual_hl.addWidget(self.pasteResponseBtn)
        manual_hl.addStretch()
        self.manual_helper_widget.setVisible(False)

        st_layout = QHBoxLayout()
        st_layout.setSpacing(15)
        st_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        st_layout.addWidget(ParamNameLabel(self.tr('Source')))
        st_layout.addWidget(self.source_combobox)
        st_layout.addWidget(ParamNameLabel(self.tr('Target')))
        st_layout.addWidget(self.target_combobox)
        
        self.vlayout.insertLayout(1, st_layout)
        self.vlayout.addWidget(self.manual_helper_widget)
        self.vlayout.addWidget(self.testTranslatorBtn)
        self.vlayout.addWidget(self.translateByTextblockBox)
        self.vlayout.addWidget(self.replaceOCRkeywordBtn)
        self.vlayout.addWidget(self.translationContextBtn)
        self.vlayout.addWidget(self.replacePreMTkeywordBtn)
        self.vlayout.addWidget(self.replaceMTkeywordBtn)
        self._add_llm_context_widgets(scrollWidget)

    def _add_llm_context_widgets(self, scrollWidget: QWidget = None):
        """Context-aware LLM translation settings (prior pages + glossary file)."""
        from utils.config import LLMGlossaryMode, LLMTranslateContext

        self.vlayout.addWidget(ParamNameLabel(self.tr('LLM context')))

        ctx_hl = QHBoxLayout()
        ctx_hl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        ctx_hl.addWidget(QLabel(self.tr('Prior context:')))
        self._llm_context_modes = (LLMTranslateContext.PAGE, LLMTranslateContext.HISTORY)
        self.llm_context_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.llm_context_combobox.addItems([self.tr('Current page only'), self.tr('Previously translated pages')])
        self.llm_context_combobox.setToolTip(self.tr(
            'Send previously translated pages of the project as prior conversation turns so names, terminology '
            'and tone stay consistent. The prefix is kept stable between adjacent pages, so providers that cache '
            'prompts charge less for it.'))
        mode = getattr(pcfg.module, 'llm_translate_context', LLMTranslateContext.PAGE)
        self.llm_context_combobox.setCurrentIndex(
            self._llm_context_modes.index(mode) if mode in self._llm_context_modes else 0)
        self.llm_context_combobox.currentIndexChanged.connect(self._on_llm_context_mode_changed)
        ctx_hl.addWidget(self.llm_context_combobox)
        ctx_hl.addWidget(QLabel(self.tr('Token budget:')))
        self.llm_context_budget_spin = QSpinBox()
        self.llm_context_budget_spin.setRange(256, 262144)
        self.llm_context_budget_spin.setSingleStep(1024)
        self.llm_context_budget_spin.setValue(int(getattr(pcfg.module, 'llm_prior_context_token_budget', 4096) or 4096))
        self.llm_context_budget_spin.setToolTip(self.tr(
            'How many tokens of prior pages may be sent. Older pages are dropped once the budget is reached.'))
        self.llm_context_budget_spin.valueChanged.connect(self._on_llm_context_budget_changed)
        ctx_hl.addWidget(self.llm_context_budget_spin)
        self.vlayout.addLayout(ctx_hl)

        glossary_hl = QHBoxLayout()
        glossary_hl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        glossary_hl.addWidget(QLabel(self.tr('Glossary file:')))
        self.llm_glossary_edit = QLineEdit()
        self.llm_glossary_edit.setPlaceholderText(self.tr('optional .json / .txt / .tsv'))
        self.llm_glossary_edit.setText(str(getattr(pcfg.module, 'llm_glossary_path', '') or ''))
        self.llm_glossary_edit.setToolTip(self.tr(
            'Terms the model must follow. Formats: JSON [{"src": "...", "dst": "...", "info": "..."}], '
            '"source->translation #note" lines, or tab-separated columns.'))
        self.llm_glossary_edit.textChanged.connect(self._on_llm_glossary_path_changed)
        glossary_hl.addWidget(self.llm_glossary_edit)
        self._llm_glossary_modes = (LLMGlossaryMode.Matching, LLMGlossaryMode.All)
        self.llm_glossary_mode_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.llm_glossary_mode_combobox.addItems([self.tr('Matching terms only'), self.tr('Whole glossary')])
        self.llm_glossary_mode_combobox.setToolTip(self.tr(
            'Matching sends only the terms found on the page; whole glossary sends every entry once as a stable prefix.'))
        glossary_mode = getattr(pcfg.module, 'llm_glossary_mode', LLMGlossaryMode.Matching)
        self.llm_glossary_mode_combobox.setCurrentIndex(
            self._llm_glossary_modes.index(glossary_mode) if glossary_mode in self._llm_glossary_modes else 0)
        self.llm_glossary_mode_combobox.currentIndexChanged.connect(self._on_llm_glossary_mode_changed)
        glossary_hl.addWidget(self.llm_glossary_mode_combobox)
        self.vlayout.addLayout(glossary_hl)

    def _on_llm_context_mode_changed(self, index: int):
        if 0 <= index < len(self._llm_context_modes):
            pcfg.module.llm_translate_context = self._llm_context_modes[index]

    def _on_llm_context_budget_changed(self, value: int):
        pcfg.module.llm_prior_context_token_budget = int(value)

    def _on_llm_glossary_path_changed(self, text: str):
        pcfg.module.llm_glossary_path = (text or '').strip()

    def _on_llm_glossary_mode_changed(self, index: int):
        if 0 <= index < len(self._llm_glossary_modes):
            pcfg.module.llm_glossary_mode = self._llm_glossary_modes[index]

    def finishSetTranslator(self, translator: BaseTranslator):
        self.source_combobox.blockSignals(True)
        self.target_combobox.blockSignals(True)
        self.module_combobox.blockSignals(True)

        self.source_combobox.clear()
        self.target_combobox.clear()
        for key in translator.supported_src_list:
            self.source_combobox.addItem(lang_display_label(key))
        for key in translator.supported_tgt_list:
            self.target_combobox.addItem(lang_display_label(key))
        self.module_combobox.setCurrentText(translator.name)
        self._set_combobox_current_by_key(self.source_combobox, translator.supported_src_list, translator.lang_source)
        self._set_combobox_current_by_key(self.target_combobox, translator.supported_tgt_list, translator.lang_target)
        self.updateModuleParamWidget()
        self.source_combobox.blockSignals(False)
        self.target_combobox.blockSignals(False)
        self.module_combobox.blockSignals(False)

    def _set_combobox_current_by_key(self, combobox, key_list: list, key: str):
        """Set combobox current item by internal key (items are display labels)."""
        try:
            idx = key_list.index(key)
            combobox.setCurrentIndex(idx)
        except (ValueError, IndexError):
            combobox.setCurrentIndex(0)

    def updateModuleParamWidget(self):
        super().updateModuleParamWidget()
        module = self.module_combobox.currentText()
        self.manual_helper_widget.setVisible(module == "manual")

    def _on_soft_failure_continue_changed(self):
        pcfg.module.translation_soft_failure_continue = self.soft_failure_continue_checker.isChecked()


class InpaintConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_INPAINTERS, scrollWidget = scrollWidget, *args, **kwargs)
        self.inpainter_changed = self.module_changed
        self.setInpainter = self.setModule
        self.needInpaintChecker = ParamCheckerBox(self.tr('Let the program decide whether it is necessary to use the selected inpaint method.'))
        self.vlayout.addWidget(self.needInpaintChecker)
        # Tile size and overlap for tiled inpainting (reduces VRAM on large images)
        tile_hl = QHBoxLayout()
        tile_hl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        tile_hl.addWidget(QLabel(self.tr('Inpaint tile size:')))
        self.inpaint_tile_size_spin = QSpinBox()
        self.inpaint_tile_size_spin.setRange(0, 2048)
        self.inpaint_tile_size_spin.setSingleStep(256)
        self.inpaint_tile_size_spin.setSpecialValueText(self.tr('Off'))
        self.inpaint_tile_size_spin.setValue(getattr(pcfg.module, 'inpaint_tile_size', 0))
        self.inpaint_tile_size_spin.setToolTip(self.tr('Off = best quality. Use 512–1024 only if you get out-of-memory errors; tiling can cause grey or blurry bubbles.'))
        tile_hl.addWidget(self.inpaint_tile_size_spin)
        tile_hl.addWidget(QLabel(self.tr('Overlap (px):')))
        self.inpaint_tile_overlap_spin = QSpinBox()
        self.inpaint_tile_overlap_spin.setRange(0, 512)
        self.inpaint_tile_overlap_spin.setValue(getattr(pcfg.module, 'inpaint_tile_overlap', 64))
        self.inpaint_tile_overlap_spin.setToolTip(self.tr('Overlap between tiles for seamless blending. Ignored when tile size is Off.'))
        tile_hl.addWidget(self.inpaint_tile_overlap_spin)
        self.vlayout.addLayout(tile_hl)
        self.inpaint_tile_size_spin.valueChanged.connect(self._on_inpaint_tile_size_changed)
        self.inpaint_tile_overlap_spin.valueChanged.connect(self._on_inpaint_tile_overlap_changed)

        # Optional: exclude blocks by detector label from inpainting (off by default)
        self.exclude_labels_checker = QCheckBox(self.tr('Exclude certain labels from inpainting'))
        self.exclude_labels_checker.setChecked(getattr(pcfg.module, 'inpaint_exclude_labels_enabled', False))
        self.exclude_labels_checker.setToolTip(self.tr(
            'When enabled, text blocks whose detector label is in the list below will not be inpainted (e.g. leave scene text as-is). '
            'Labels are case-insensitive. Requires a detector that sets block labels (e.g. YSG YOLO).'))
        self.exclude_labels_checker.clicked.connect(self._on_exclude_labels_checker_changed)
        self.vlayout.addWidget(self.exclude_labels_checker)
        exclude_hl = QHBoxLayout()
        exclude_hl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        exclude_hl.addWidget(QLabel(self.tr('Labels to exclude (comma-separated):')))
        self.exclude_labels_edit = QLineEdit()
        self.exclude_labels_edit.setPlaceholderText(self.tr('e.g. other, scene'))
        self.exclude_labels_edit.setText(getattr(pcfg.module, 'inpaint_exclude_labels', '') or '')
        self.exclude_labels_edit.setEnabled(self.exclude_labels_checker.isChecked())
        self.exclude_labels_edit.textChanged.connect(self._on_exclude_labels_edit_changed)
        self.exclude_labels_edit.setToolTip(self.tr('Comma-separated detector labels to exclude from inpainting.'))
        exclude_hl.addWidget(self.exclude_labels_edit)
        self.vlayout.addLayout(exclude_hl)

        # Full-image inpainting: bypass per-block crops (use if Lama/per-block gives bad results)
        self.full_image_checker = QCheckBox(self.tr('Inpaint full image (no per-block crops)'))
        self.full_image_checker.setChecked(getattr(pcfg.module, 'inpaint_full_image', False))
        self.full_image_checker.setToolTip(self.tr(
            'Process the whole image at once instead of cropping per text block. Uses more VRAM and can be slower, '
            'but avoids crop/mask issues that cause bad results with some inpainting models (e.g. Lama). Try this if inpainting looks wrong.'))
        self.full_image_checker.clicked.connect(self._on_full_image_checker_changed)
        self.vlayout.addWidget(self.full_image_checker)

        # Clip the mask to the detected text boxes (upstream dmMaze/BallonsTranslator#1213)
        self.filter_mask_by_bboxes_checker = QCheckBox(self.tr('Filter mask by text boxes'))
        self.filter_mask_by_bboxes_checker.setChecked(getattr(pcfg.module, 'filter_mask_by_bboxes', False))
        self.filter_mask_by_bboxes_checker.setToolTip(self.tr(
            'Only inpaint mask pixels that fall inside a detected text box. Use it when the mask spills over '
            'artwork outside the text areas.'))
        self.filter_mask_by_bboxes_checker.clicked.connect(self._on_filter_mask_by_bboxes_changed)
        self.vlayout.addWidget(self.filter_mask_by_bboxes_checker)

    def _on_filter_mask_by_bboxes_changed(self):
        pcfg.module.filter_mask_by_bboxes = self.filter_mask_by_bboxes_checker.isChecked()

    def _on_full_image_checker_changed(self):
        pcfg.module.inpaint_full_image = self.full_image_checker.isChecked()

    def _on_exclude_labels_checker_changed(self):
        enabled = self.exclude_labels_checker.isChecked()
        pcfg.module.inpaint_exclude_labels_enabled = enabled
        self.exclude_labels_edit.setEnabled(enabled)

    def _on_exclude_labels_edit_changed(self, text: str):
        pcfg.module.inpaint_exclude_labels = (text or '').strip()

    def _on_inpaint_tile_size_changed(self, value: int):
        pcfg.module.inpaint_tile_size = value

    def _on_inpaint_tile_overlap_changed(self, value: int):
        pcfg.module.inpaint_tile_overlap = value

    def showEvent(self, e) -> None:
        self.p_layout.insertWidget(1, self.module_combobox)
        super().showEvent(e)

    def hideEvent(self, e) -> None:
        self.p_layout.removeWidget(self.module_combobox)
        return super().hideEvent(e)

class TextDetectConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_TEXTDETECTORS, scrollWidget = scrollWidget, *args, **kwargs)
        self.detector_changed = self.module_changed
        self.setDetector = self.setModule
        self.keep_existing_checker = QCheckBox(text=self.tr('Keep Existing Lines'))
        self.p_layout.insertWidget(2, self.keep_existing_checker)
        self.dual_detect_checker = QCheckBox(self.tr('Run second detector (dual detect)'))
        self.dual_detect_checker.setToolTip(self.tr('Run a second text detector and merge results to catch more regions (e.g. one good at bubbles, one at captions).'))
        self.dual_detect_checker.setChecked(getattr(pcfg.module, 'enable_dual_detect', False))
        self.dual_detect_checker.clicked.connect(self._on_dual_detect_changed)
        self.secondary_detector_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.secondary_detector_combobox.addItem('')
        self.secondary_detector_combobox.addItems(GET_VALID_TEXTDETECTORS())
        self.secondary_detector_combobox.setCurrentText(getattr(pcfg.module, 'textdetector_secondary', '') or '')
        self.secondary_detector_combobox.currentTextChanged.connect(self._on_secondary_detector_changed)
        dual_hl = QHBoxLayout()
        dual_hl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        dual_hl.addWidget(self.dual_detect_checker)
        dual_hl.addWidget(QLabel(self.tr('Secondary:')))
        dual_hl.addWidget(self.secondary_detector_combobox)
        self.secondary_outside_bubble_only_checker = QCheckBox(self.tr('Outside bubbles only'))
        self.secondary_outside_bubble_only_checker.setToolTip(self.tr(
            'Only add secondary detector boxes that are outside primary (bubble) regions. Use when primary is good at bubbles (e.g. YSGYOLO) and secondary is for signs/captions (e.g. EasyOCR).'))
        self.secondary_outside_bubble_only_checker.setChecked(getattr(pcfg.module, 'secondary_detector_outside_bubble_only', False))
        self.secondary_outside_bubble_only_checker.clicked.connect(self._on_secondary_outside_bubble_only_changed)
        dual_hl.addWidget(self.secondary_outside_bubble_only_checker)
        self.vlayout.addLayout(dual_hl)
        self.tertiary_detect_checker = QCheckBox(self.tr('Run third detector'))
        self.tertiary_detect_checker.setToolTip(self.tr('Run a third text detector and merge results (e.g. primary + secondary + tertiary for maximum coverage).'))
        self.tertiary_detect_checker.setChecked(getattr(pcfg.module, 'enable_tertiary_detect', False))
        self.tertiary_detect_checker.clicked.connect(self._on_tertiary_detect_changed)
        self.tertiary_detector_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.tertiary_detector_combobox.addItem('')
        self.tertiary_detector_combobox.addItems(GET_VALID_TEXTDETECTORS())
        self.tertiary_detector_combobox.setCurrentText(getattr(pcfg.module, 'textdetector_tertiary', '') or '')
        self.tertiary_detector_combobox.currentTextChanged.connect(self._on_tertiary_detector_changed)
        tertiary_hl = QHBoxLayout()
        tertiary_hl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        tertiary_hl.addWidget(self.tertiary_detect_checker)
        tertiary_hl.addWidget(QLabel(self.tr('Tertiary:')))
        tertiary_hl.addWidget(self.tertiary_detector_combobox)
        self.vlayout.addLayout(tertiary_hl)
        # Section 19: OSB layout fallbacks
        self.osb_layout_fallbacks_checker = QCheckBox(self.tr("OSB layout fallbacks (vertical stack + restore original)"))
        self.osb_layout_fallbacks_checker.setToolTip(self.tr(
            "When checked (default), if layout fails for outside-speech-bubble (OSB) blocks, retry with vertical stacking; if that fails, restore the original image region. "
            "Uncheck to only mark failed OSB blocks and restore original (no vertical retry)."))
        self.osb_layout_fallbacks_checker.setChecked(getattr(pcfg.module, "osb_layout_fallbacks_enabled", True))
        self.osb_layout_fallbacks_checker.clicked.connect(self._on_osb_layout_fallbacks_changed)
        self.vlayout.addWidget(self.osb_layout_fallbacks_checker)
        # Allow detection to set box rotation for slanted (horizontal) text; threshold = min degrees to apply.
        self.allow_detection_box_rotation_checker = QCheckBox(self.tr('Allow box rotation for slanted text'))
        self.allow_detection_box_rotation_checker.setToolTip(self.tr(
            'When enabled, text detection can set the rotation angle of horizontal boxes for slanted text. '
            'Only angles at or above the threshold are applied; smaller slants stay 0°.'))
        self.allow_detection_box_rotation_checker.setChecked(getattr(pcfg.module, 'allow_detection_box_rotation', False))
        self.allow_detection_box_rotation_checker.clicked.connect(self._on_allow_detection_box_rotation_changed)
        rot_hl = QHBoxLayout()
        rot_hl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        rot_hl.addWidget(self.allow_detection_box_rotation_checker)
        rot_hl.addWidget(QLabel(self.tr('Min angle (degrees):')))
        self.detection_rotation_threshold_spinbox = QSpinBox()
        self.detection_rotation_threshold_spinbox.setRange(1, 45)
        self.detection_rotation_threshold_spinbox.setValue(int(getattr(pcfg.module, 'detection_rotation_threshold_degrees', 10.0)))
        self.detection_rotation_threshold_spinbox.setSuffix('°')
        self.detection_rotation_threshold_spinbox.valueChanged.connect(self._on_detection_rotation_threshold_changed)
        rot_hl.addWidget(self.detection_rotation_threshold_spinbox)
        self.vlayout.addLayout(rot_hl)
        # Secondary detector params (shown when dual detect is on and a secondary is selected)
        self.secondary_params_container = QWidget(self)
        secondary_params_outer = QVBoxLayout(self.secondary_params_container)
        secondary_params_outer.setContentsMargins(0, 0, 0, 0)
        secondary_params_outer.addWidget(QLabel(self.tr('Secondary detector params:')))
        self.secondary_params_layout = QHBoxLayout()
        self.secondary_params_layout.setContentsMargins(0, 0, 0, 0)
        secondary_params_outer.addLayout(self.secondary_params_layout)
        self.secondary_param_widget_map = {}
        self.secondary_visible_widget = None
        self.vlayout.addWidget(self.secondary_params_container)
        self._update_secondary_params_visibility()
        # Tertiary detector params (shown when tertiary detect is on and a tertiary is selected)
        self.tertiary_params_container = QWidget(self)
        tertiary_params_outer = QVBoxLayout(self.tertiary_params_container)
        tertiary_params_outer.setContentsMargins(0, 0, 0, 0)
        tertiary_params_outer.addWidget(QLabel(self.tr('Tertiary detector params:')))
        self.tertiary_params_layout = QHBoxLayout()
        self.tertiary_params_layout.setContentsMargins(0, 0, 0, 0)
        tertiary_params_outer.addLayout(self.tertiary_params_layout)
        self.tertiary_param_widget_map = {}
        self.tertiary_visible_widget = None
        self.vlayout.addWidget(self.tertiary_params_container)
        self._update_tertiary_params_visibility()

        # Collision-based post-merge (word-level OCR cleanup)
        self.merge_nearby_blocks_checker = QCheckBox(self.tr('Merge nearby detected blocks (word-level cleanup)'))
        self.merge_nearby_blocks_checker.setToolTip(self.tr(
            'When enabled, nearby text boxes can be merged after detection to fix word-level fragmentation. '
            'Recommended when one bubble is split into many tiny boxes.'))
        self.merge_nearby_blocks_checker.setChecked(bool(getattr(pcfg.module, 'merge_nearby_blocks_collision', False)))
        self.merge_nearby_blocks_checker.clicked.connect(self._on_merge_nearby_blocks_changed)
        merge_hl = QHBoxLayout()
        merge_hl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        merge_hl.addWidget(self.merge_nearby_blocks_checker)
        merge_hl.addWidget(QLabel(self.tr('Min blocks:')))
        self.merge_nearby_min_blocks_spinbox = QSpinBox()
        self.merge_nearby_min_blocks_spinbox.setRange(4, 200)
        self.merge_nearby_min_blocks_spinbox.setValue(int(getattr(pcfg.module, 'merge_nearby_blocks_min_blocks', 18) or 18))
        self.merge_nearby_min_blocks_spinbox.setToolTip(self.tr('Only run merge when at least this many boxes are detected on a page.'))
        self.merge_nearby_min_blocks_spinbox.valueChanged.connect(self._on_merge_nearby_min_blocks_changed)
        merge_hl.addWidget(self.merge_nearby_min_blocks_spinbox)
        merge_hl.addWidget(QLabel(self.tr('Gap ratio:')))
        self.merge_nearby_gap_ratio_spinbox = QDoubleSpinBox()
        self.merge_nearby_gap_ratio_spinbox.setRange(0.5, 4.0)
        self.merge_nearby_gap_ratio_spinbox.setSingleStep(0.1)
        self.merge_nearby_gap_ratio_spinbox.setDecimals(2)
        self.merge_nearby_gap_ratio_spinbox.setValue(float(getattr(pcfg.module, 'merge_nearby_blocks_gap_ratio', 1.5) or 1.5))
        self.merge_nearby_gap_ratio_spinbox.setToolTip(self.tr('Higher values merge more aggressively across nearby boxes.'))
        self.merge_nearby_gap_ratio_spinbox.valueChanged.connect(self._on_merge_nearby_gap_ratio_changed)
        merge_hl.addWidget(self.merge_nearby_gap_ratio_spinbox)
        self.vlayout.addLayout(merge_hl)

    def _on_dual_detect_changed(self):
        pcfg.module.enable_dual_detect = self.dual_detect_checker.isChecked()
        self._update_secondary_params_visibility()

    def _on_osb_layout_fallbacks_changed(self):
        pcfg.module.osb_layout_fallbacks_enabled = self.osb_layout_fallbacks_checker.isChecked()

    def _on_allow_detection_box_rotation_changed(self):
        pcfg.module.allow_detection_box_rotation = self.allow_detection_box_rotation_checker.isChecked()

    def _on_detection_rotation_threshold_changed(self, value: int):
        pcfg.module.detection_rotation_threshold_degrees = float(value)

    def _on_secondary_detector_changed(self, name: str):
        pcfg.module.textdetector_secondary = (name or '').strip()
        self._update_secondary_params_visibility()

    def _on_secondary_outside_bubble_only_changed(self):
        pcfg.module.secondary_detector_outside_bubble_only = self.secondary_outside_bubble_only_checker.isChecked()

    def _on_tertiary_detect_changed(self):
        pcfg.module.enable_tertiary_detect = self.tertiary_detect_checker.isChecked()
        self._update_tertiary_params_visibility()


    def _on_tertiary_detector_changed(self, name: str):
        pcfg.module.textdetector_tertiary = (name or '').strip()
        self._update_tertiary_params_visibility()


    def _update_tertiary_params_visibility(self):
        ter_enabled = self.tertiary_detect_checker.isChecked()
        ter = (self.tertiary_detector_combobox.currentText() or '').strip()
        if ter_enabled and ter and ter in getattr(self, 'module_dict', {}):
            self.tertiary_params_container.show()
            self._update_tertiary_param_widget()
        else:
            self.tertiary_params_container.hide()

    def _update_tertiary_param_widget(self):
        ter_name = (self.tertiary_detector_combobox.currentText() or '').strip()
        if self.tertiary_visible_widget is not None:
            self.tertiary_visible_widget.hide()
            self.tertiary_params_layout.removeWidget(self.tertiary_visible_widget)
            self.tertiary_visible_widget = None
        if not ter_name or not getattr(self, 'module_dict', None) or ter_name not in self.module_dict:
            return
        if ter_name in self.tertiary_param_widget_map:
            widget = self.tertiary_param_widget_map[ter_name]
        else:
            params = self.module_dict[ter_name]
            if params is None:
                return
            widget = ParamWidget(params, scrollWidget=self)
            widget.paramwidget_edited.connect(self._on_tertiary_param_edited)
            self.tertiary_param_widget_map[ter_name] = widget
        self.tertiary_params_layout.addWidget(widget)
        widget.show()
        self.tertiary_visible_widget = widget


    def _on_merge_nearby_blocks_changed(self):
        enabled = self.merge_nearby_blocks_checker.isChecked()
        pcfg.module.merge_nearby_blocks_collision = enabled

    def _on_merge_nearby_min_blocks_changed(self, value: int):
        pcfg.module.merge_nearby_blocks_min_blocks = int(value)

    def _on_merge_nearby_gap_ratio_changed(self, value: float):
        pcfg.module.merge_nearby_blocks_gap_ratio = float(value)

    def _on_tertiary_param_edited(self, param_key: str, param_content: dict):
        ter_name = (self.tertiary_detector_combobox.currentText() or '').strip()
        if not ter_name:
            return
        tparams = getattr(pcfg.module, 'textdetector_params', None) or {}
        if ter_name not in tparams or param_key not in tparams[ter_name]:
            return
        val = param_content.get('content')
        if val is None:
            return
        tparams[ter_name][param_key] = val
        pcfg.module.textdetector_params = tparams

    def _update_secondary_params_visibility(self):
        dual = self.dual_detect_checker.isChecked()
        sec = (self.secondary_detector_combobox.currentText() or '').strip()
        if dual and sec and sec in getattr(self, 'module_dict', {}):
            self.secondary_params_container.show()
            self._update_secondary_param_widget()
        else:
            self.secondary_params_container.hide()

    def _update_secondary_param_widget(self):
        sec_name = (self.secondary_detector_combobox.currentText() or '').strip()
        if self.secondary_visible_widget is not None:
            self.secondary_visible_widget.hide()
            self.secondary_params_layout.removeWidget(self.secondary_visible_widget)
            self.secondary_visible_widget = None
        if not sec_name or not getattr(self, 'module_dict', None) or sec_name not in self.module_dict:
            return
        if sec_name in self.secondary_param_widget_map:
            widget = self.secondary_param_widget_map[sec_name]
        else:
            params = self.module_dict[sec_name]
            if params is None:
                return
            widget = ParamWidget(params, scrollWidget=self)
            widget.paramwidget_edited.connect(self._on_secondary_param_edited)
            self.secondary_param_widget_map[sec_name] = widget
        self.secondary_params_layout.addWidget(widget)
        widget.show()
        self.secondary_visible_widget = widget

    def _on_secondary_param_edited(self, param_key: str, param_content: dict):
        sec_name = (self.secondary_detector_combobox.currentText() or '').strip()
        if not sec_name:
            return
        tparams = getattr(pcfg.module, 'textdetector_params', None) or {}
        if sec_name not in tparams or param_key not in tparams[sec_name]:
            return
        val = param_content.get('content')
        if val is None:
            return
        params = tparams[sec_name]
        if isinstance(params.get(param_key), dict):
            params[param_key]['value'] = val
        else:
            params[param_key] = val

    def addModulesParamWidgets(self, module_dict: dict):
        super().addModulesParamWidgets(module_dict)
        self._update_secondary_params_visibility()

class OCRConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_OCR, scrollWidget = scrollWidget, *args, **kwargs)
        self.ocr_changed = self.module_changed
        self.setOCR = self.setModule
        self.restoreEmptyOCRChecker = QCheckBox(self.tr("Delete and restore region where OCR return empty string."), self)
        self.restoreEmptyOCRChecker.setToolTip(
            self.tr("When enabled, text boxes whose OCR result is empty are removed from the page and the mask is restored. "
                    "Disable this to keep all boxes even when OCR returns nothing or fails (e.g. to avoid boxes disappearing)."))
        self.restoreEmptyOCRChecker.clicked.connect(self.on_restore_empty_ocr)
        self.vlayout.addWidget(self.restoreEmptyOCRChecker)
        # 字体检测选项
        self.fontDetectChecker = QCheckBox(self.tr("Font Detection"), self)
        self.fontDetectChecker.setToolTip(self.tr("Detect font properties (e.g. bold, italic) from the image after OCR. Useful for manga/comics."))
        self.fontDetectChecker.setChecked(pcfg.module.ocr_font_detect)
        self.fontDetectChecker.clicked.connect(self.on_fontdetect_changed)
        self.vlayout.addWidget(self.fontDetectChecker)

        # Letter case applied to every OCR result (upstream dmMaze/BallonsTranslator#1248)
        from utils.config import OCRTextPostprocess
        case_hl = QHBoxLayout()
        case_hl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        case_hl.addWidget(QLabel(self.tr('Letter case:')))
        self.letter_case_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self._letter_case_modes = (OCRTextPostprocess.NONE, OCRTextPostprocess.CAPITALIZE, OCRTextPostprocess.UPPERCASE)
        self.letter_case_combobox.addItems([self.tr('Keep as detected'), self.tr('Capitalize sentences'), self.tr('UPPERCASE')])
        self.letter_case_combobox.setToolTip(self.tr(
            'Applied to OCR output: "Capitalize sentences" turns ALL CAPS comic lettering into normal sentences.'))
        mode = getattr(pcfg.module, 'ocr_text_postprocess', OCRTextPostprocess.NONE)
        self.letter_case_combobox.setCurrentIndex(
            self._letter_case_modes.index(mode) if mode in self._letter_case_modes else 0)
        self.letter_case_combobox.currentIndexChanged.connect(self.on_letter_case_changed)
        case_hl.addWidget(self.letter_case_combobox)
        self.vlayout.addLayout(case_hl)

    def on_letter_case_changed(self, index: int):
        if 0 <= index < len(self._letter_case_modes):
            pcfg.module.ocr_text_postprocess = self._letter_case_modes[index]

    def on_restore_empty_ocr(self):
        pcfg.restore_ocr_empty = self.restoreEmptyOCRChecker.isChecked()

    def on_fontdetect_changed(self):
        pcfg.module.ocr_font_detect = self.fontDetectChecker.isChecked()
