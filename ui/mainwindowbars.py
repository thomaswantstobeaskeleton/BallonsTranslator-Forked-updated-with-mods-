import os.path as osp
from pathlib import Path
from typing import List, Union

from qtpy.QtWidgets import (
    QMainWindow,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QLabel,
    QSizePolicy,
    QToolBar,
    QMenu,
    QSpacerItem,
    QPushButton,
    QCheckBox,
    QToolButton,
    QMessageBox,
    QWidget,
    QScrollArea,
    QLineEdit,
    QCompleter,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
)
from qtpy.QtCore import Qt, Signal, QPoint, QEvent, QSize, QSortFilterProxyModel, QModelIndex, QRegularExpression, QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup, QTimer
from qtpy.QtGui import QMouseEvent, QKeySequence, QActionGroup, QIcon, QWheelEvent, QStandardItemModel, QStandardItem

from modules.translators import BaseTranslator
from modules.translators.base import lang_display_label
from .custom_widget import Widget, PaintQSlider, SmallComboBox, ConfigClickableLabel, sanitize_font
from .custom_widget.hover_animation import install_hover_opacity_animation, install_hover_scale_animation, install_button_animations
from utils.shared import TITLEBAR_HEIGHT, WINDOW_BORDER_WIDTH, BOTTOMBAR_HEIGHT, LEFTBAR_WIDTH, LEFTBTN_WIDTH
from .framelesswindow import FramelessMoveResize
from utils.config import pcfg
from utils import shared as C
from utils.shortcuts import get_shortcut
if C.FLAG_QT6:
    from qtpy.QtGui import QAction
else:
    from qtpy.QtWidgets import QAction

class ShowPageListChecker(QCheckBox):
    pass


class OpenBtn(QToolButton):
    pass


class StatusButton(QPushButton):
    pass


class BottomModeToolButton(QToolButton):
    pass


class TitleBarToolBtn(QToolButton):
    pass


class StateChecker(QCheckBox):
    checked = Signal(str)
    unchecked = Signal(str)
    def __init__(self, checker_type: str, uncheckable: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.checker_type = checker_type
        self.uncheckable = uncheckable

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.isChecked():
                self.setChecked(True)
            elif self.uncheckable:
                self.setChecked(False)
                
    def setChecked(self, check: bool) -> None:
        check_state = self.isChecked()
        super().setChecked(check)
        if check_state != check:
            if check:
                self.checked.emit(self.checker_type)
            else:
                self.unchecked.emit(self.checker_type)

class LeftBar(Widget):
    recent_proj_list = []
    imgTransChecked = Signal()
    configChecked = Signal()
    run_clicked = Signal()
    close_project_requested = Signal()
    open_dir = Signal(str)
    open_json_proj = Signal(str)
    save_proj = Signal()
    save_config = Signal()
    def __init__(self, mainwindow, *args, **kwargs) -> None:
        super().__init__(mainwindow, *args, **kwargs)
        self.mainwindow: QMainWindow = mainwindow

        self.setFixedWidth(LEFTBAR_WIDTH)
        self.showPageListLabel = ShowPageListChecker()

        self.globalSearchChecker = QCheckBox()
        self.globalSearchChecker.setObjectName('GlobalSearchChecker')
        self.globalSearchChecker.setToolTip(self.tr('Global Search (Ctrl+G)'))

        self.imgTransChecker = StateChecker('imgtrans')
        self.imgTransChecker.setObjectName('ImgTransChecker')
        self.imgTransChecker.checked.connect(self.stateCheckerChanged)
        
        self.configChecker = StateChecker('config', uncheckable=True)
        self.configChecker.setObjectName('ConfigChecker')
        self.configChecker.checked.connect(self.stateCheckerChanged)
        self.configChecker.unchecked.connect(self.stateCheckerChanged)

        self.runBtn = QPushButton()
        self.runBtn.setObjectName('LeftBarRunBtn')
        self.runBtn.setFixedSize(28, 28)
        run_icon_path = osp.join(C.PROGRAM_PATH, 'icons', 'leftbar_run.svg')
        if not osp.isfile(run_icon_path):
            run_icon_path = osp.join(C.PROGRAM_PATH, 'icons', 'bottombar_translate_activate.svg')
        if not osp.isfile(run_icon_path):
            run_icon_path = osp.join(C.PROGRAM_PATH, 'icons', 'bottombar_translate.svg')
        if osp.isfile(run_icon_path):
            self.runBtn.setIcon(QIcon(run_icon_path))
        self.runBtn.setIconSize(QSize(20, 20))
        self.runBtn.setToolTip(self.tr('Run pipeline (same as Pipeline → Run).'))
        self.runBtn.clicked.connect(self.run_clicked.emit)

        actionOpenFolder = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'openbtn.svg')), self.tr("Open Folder ..."), self)
        actionOpenFolder.triggered.connect(self.onOpenFolder)
        actionOpenFolder.setShortcut(QKeySequence.fromString(get_shortcut("file.open_folder", getattr(pcfg, "shortcuts", None))))

        actionOpenProj = QAction(self.tr("Open Project ... *.json"), self)
        actionOpenProj.triggered.connect(self.onOpenProj)

        actionOpenImages = QAction(self.tr("Open Images ..."), self)
        actionOpenImages.triggered.connect(self.onOpenImages)

        actionOpenACBFCBZ = QAction(self.tr("Open ACBF/CBZ ..."), self)
        actionOpenACBFCBZ.setToolTip(self.tr("Open a comic book archive (.cbz/.zip); extracts to a folder and opens as project."))
        actionOpenACBFCBZ.triggered.connect(self.onOpenACBFCBZ)

        actionOpenCBR = QAction(self.tr("Open CBR ..."), self)
        actionOpenCBR.setToolTip(self.tr("Open a comic book RAR archive (.cbr/.rar). Requires: pip install rarfile; WinRAR/7-Zip in PATH."))
        actionOpenCBR.triggered.connect(self.onOpenCBR)

        actionSaveProj = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'save_activate.svg')), self.tr("Save Project"), self)
        self.save_proj = actionSaveProj.triggered
        actionSaveProj.setShortcut(QKeySequence.fromString(get_shortcut("file.save_proj", getattr(pcfg, "shortcuts", None))))

        self._shortcut_actions_left = [
            ("file.open_folder", "Ctrl+O", actionOpenFolder),
            ("file.save_proj", "Ctrl+S", actionSaveProj),
        ]

        actionExportAsDoc = QAction(self.tr("Export as Doc"), self)
        self.export_doc = actionExportAsDoc.triggered
        actionExportCurrentPageAs = QAction(self.tr("Export current page as..."), self)
        actionExportCurrentPageAs.setToolTip(self.tr("Save current page result image in chosen format (PNG, JPEG, WebP, JXL)."))
        self.export_current_page_as = actionExportCurrentPageAs.triggered
        actionImportFromDoc = QAction(self.tr("Import from Doc"), self)
        self.import_doc = actionImportFromDoc.triggered

        actionExportSrcTxt = QAction(self.tr("Export source text as TXT"), self)
        self.export_src_txt = actionExportSrcTxt.triggered
        actionExportTranslationTxt = QAction(self.tr("Export translation as TXT"), self)
        self.export_trans_txt = actionExportTranslationTxt.triggered

        actionExportBothTxt = QAction(self.tr("Export source + translation as TXT"), self)
        actionExportBothTxt.setToolTip(self.tr("One file with each block's source text followed by its translation."))
        self.export_both_txt = actionExportBothTxt.triggered

        actionExportSrcMD = QAction(self.tr("Export source text as markdown"), self)
        self.export_src_md = actionExportSrcMD.triggered
        actionExportTranslationMD = QAction(self.tr("Export translation as markdown"), self)
        self.export_trans_md = actionExportTranslationMD.triggered

        actionImportTranslationTxt = QAction(self.tr("Import translation from TXT/markdown"), self)
        self.import_trans_txt = actionImportTranslationTxt.triggered

        self.openBtn = OpenBtn()

        openMenu = QMenu(self.openBtn)
        # Keep submenu ownership aligned with the visual popup chain for Wayland
        # (upstream dmMaze/BallonsTranslator@1feb09b).
        self.recentMenu = QMenu(self.tr("Open Recent"), openMenu)
        openMenu.addActions([actionOpenFolder, actionOpenImages, actionOpenACBFCBZ, actionOpenCBR, actionOpenProj])
        openMenu.addMenu(self.recentMenu)
        openMenu.addSeparator()
        actionCloseProject = QAction(self.tr("Close project and go to welcome screen"), self)
        actionCloseProject.triggered.connect(self.close_project_requested.emit)
        openMenu.addAction(actionCloseProject)
        openMenu.addSeparator()
        openMenu.addAction(actionSaveProj)
        self.openBtn.setFixedSize(28, 28)
        open_icon_path = osp.join(C.PROGRAM_PATH, 'icons', 'openbtn.svg')
        if osp.isfile(open_icon_path):
            self.openBtn.setIcon(QIcon(open_icon_path))
        self.openBtn.setIconSize(QSize(26, 26))
        self.openBtn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.openBtn.setAutoRaise(True)
        self.openBtn.setMenu(openMenu)
        self.openBtn.setPopupMode(QToolButton.InstantPopup)

        vlayout = QVBoxLayout(self)
        vlayout.setContentsMargins(4, 8, 4, 8)
        vlayout.setSpacing(6)
        vlayout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Top section: open + page list toggle + search
        vlayout.addWidget(self.openBtn)
        vlayout.addWidget(self.showPageListLabel)
        vlayout.addWidget(self.globalSearchChecker)
        vlayout.addSpacing(8)

        # Middle section: primary mode switchers (imgtrans / config)
        vlayout.addWidget(self.imgTransChecker)
        vlayout.addWidget(self.configChecker)
        vlayout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Bottom section: run pipeline
        vlayout.addWidget(self.runBtn)
        self.setGeometry(0, 0, 300, 500)
        self.setMouseTracking(True)

        install_button_animations(
            self.openBtn,
            normal_opacity=0.9,
            press_opacity=0.74,
            with_scale=True,
        )
        for w in (self.showPageListLabel, self.globalSearchChecker, self.imgTransChecker, self.configChecker):
            install_hover_opacity_animation(w, duration_ms=100, normal_opacity=0.88, press_opacity=0.74)
        # Run button: scale animation persists even when Bubbly UI is off
        install_hover_scale_animation(self.runBtn, duration_ms=80, size_delta=(3, 2))

        # Gentle breathing animation to spotlight the primary Run action in the refreshed UI.
        self._run_btn_breathe = QSequentialAnimationGroup(self)
        self._run_btn_pulse_up = QPropertyAnimation(self.runBtn, b"iconSize", self)
        self._run_btn_pulse_up.setDuration(900)
        self._run_btn_pulse_up.setEasingCurve(QEasingCurve.InOutQuad)
        self._run_btn_pulse_down = QPropertyAnimation(self.runBtn, b"iconSize", self)
        self._run_btn_pulse_down.setDuration(900)
        self._run_btn_pulse_down.setEasingCurve(QEasingCurve.InOutQuad)
        self._run_btn_breathe.addAnimation(self._run_btn_pulse_up)
        self._run_btn_breathe.addAnimation(self._run_btn_pulse_down)
        self._refresh_run_button_breathe_animation()
        self._run_btn_breathe.setLoopCount(-1)
        QTimer.singleShot(300, self._run_btn_breathe.start)

    def _refresh_run_button_breathe_animation(self):
        icon_sz = self.runBtn.iconSize()
        if icon_sz.width() <= 0 or icon_sz.height() <= 0:
            icon_sz = QSize(20, 20)
            self.runBtn.setIconSize(icon_sz)

        up_sz = QSize(icon_sz.width() + 2, icon_sz.height() + 2)

        self._run_btn_pulse_up.setStartValue(icon_sz)
        self._run_btn_pulse_up.setEndValue(up_sz)
        self._run_btn_pulse_down.setStartValue(up_sz)
        self._run_btn_pulse_down.setEndValue(icon_sz)

    def apply_shortcuts(self, shortcuts_dict):
        """Apply keyboard shortcuts from config (action_id -> key string)."""
        from qtpy.QtGui import QKeySequence
        for action_id, default_key, action in self._shortcut_actions_left:
            key = (shortcuts_dict or {}).get(action_id) or default_key
            action.setShortcut(QKeySequence.fromString(key) if key else QKeySequence())

    def initRecentProjMenu(self, proj_list: List[str]):
        self.recent_proj_list = proj_list
        for proj in proj_list:
            action = QAction(proj, self)
            self.recentMenu.addAction(action)
            action.triggered.connect(self.recentActionTriggered)

    def updateRecentProjList(self, proj_list: Union[str, List[str]]):
        if len(proj_list) == 0:
            return
        if isinstance(proj_list, str):
            proj_list = [proj_list]
        if self.recent_proj_list == proj_list:
            return

        actionlist = self.recentMenu.actions()
        if len(self.recent_proj_list) == 0:
            self.recent_proj_list.append(proj_list.pop())
            topAction = QAction(self.recent_proj_list[-1], self)
            topAction.triggered.connect(self.recentActionTriggered)
            self.recentMenu.addAction(topAction)
        else:
            topAction = actionlist[0]
        for proj in proj_list[::-1]:
            try:    # remove duplicated
                idx = self.recent_proj_list.index(proj)
                if idx == 0:
                    continue
                del self.recent_proj_list[idx]
                self.recentMenu.removeAction(self.recentMenu.actions()[idx])
                if len(self.recent_proj_list) == 0:
                    topAction = QAction(proj, self)
                    self.recentMenu.addAction(topAction)
                    topAction.triggered.connect(self.recentActionTriggered)
                    continue
            except ValueError:
                pass
            newTop = QAction(proj, self)
            self.recentMenu.insertAction(topAction, newTop)
            newTop.triggered.connect(self.recentActionTriggered)
            self.recent_proj_list.insert(0, proj)
            topAction = newTop

        MAXIUM_RECENT_PROJ_NUM = getattr(pcfg, 'recent_proj_list_max', 14)
        actionlist = self.recentMenu.actions()
        num_to_remove = len(actionlist) - MAXIUM_RECENT_PROJ_NUM
        if num_to_remove > 0:
            actions_to_remove = actionlist[-num_to_remove:]
            for action in actions_to_remove:
                self.recentMenu.removeAction(action)
                self.recent_proj_list.pop()

        self.save_config.emit()

    def recentActionTriggered(self):
        path = self.sender().text()
        if osp.exists(path):
            self.updateRecentProjList(path)
            self.open_dir.emit(path)
        else:
            self.recent_proj_list.remove(path)
            self.recentMenu.removeAction(self.sender())
        
    def onOpenFolder(self) -> None:
        
        d = None
        if len(self.recent_proj_list) > 0:
            for projp in self.recent_proj_list:
                if not osp.isdir(projp):
                    projp = osp.dirname(projp)
                if osp.exists(projp):
                    d = projp
                    break
        
        dialog = QFileDialog()
        folder_path = str(dialog.getExistingDirectory(self, self.tr("Select Directory"), d))
        if folder_path and osp.exists(folder_path):
            self.updateRecentProjList(folder_path)
            self.open_dir.emit(folder_path)

    def onOpenACBFCBZ(self) -> None:
        d = None
        if len(self.recent_proj_list) > 0:
            for projp in self.recent_proj_list:
                if not osp.isdir(projp):
                    projp = osp.dirname(projp)
                if osp.exists(projp):
                    d = projp
                    break
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter(self.tr("Comic archives (*.cbz *.zip)"))
        if d:
            dialog.setDirectory(d)
        if not dialog.exec():
            return
        selected = dialog.selectedFiles()
        if not selected or not osp.isfile(selected[0]):
            return
        cbz_path = str(selected[0])
        try:
            from utils.cbz_acbf_utils import extract_cbz_to_folder
            out_dir = osp.join(osp.dirname(cbz_path), Path(cbz_path).stem)
            extract_cbz_to_folder(cbz_path, out_dir=out_dir)
            self.updateRecentProjList(out_dir)
            self.open_dir.emit(out_dir)
        except Exception as e:
            QMessageBox.warning(
                self,
                self.tr("Open ACBF/CBZ"),
                self.tr("Failed to extract archive: {}").format(str(e)),
            )

    def onOpenCBR(self) -> None:
        d = None
        if len(self.recent_proj_list) > 0:
            for projp in self.recent_proj_list:
                if not osp.isdir(projp):
                    projp = osp.dirname(projp)
                if osp.exists(projp):
                    d = projp
                    break
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter(self.tr("Comic RAR archives (*.cbr *.rar)"))
        if d:
            dialog.setDirectory(d)
        if not dialog.exec():
            return
        selected = dialog.selectedFiles()
        if not selected or not osp.isfile(selected[0]):
            return
        cbr_path = str(selected[0])
        try:
            from utils.archive_utils import extract_cbr_to_folder
            out_dir = osp.join(osp.dirname(cbr_path), Path(cbr_path).stem)
            extract_cbr_to_folder(cbr_path, out_dir=out_dir)
            self.updateRecentProjList(out_dir)
            self.open_dir.emit(out_dir)
        except RuntimeError as e:
            QMessageBox.warning(
                self,
                self.tr("Open CBR"),
                str(e),
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                self.tr("Open CBR"),
                self.tr("Failed to extract archive: {}").format(str(e)),
            )

    def onOpenProj(self):
        dialog = QFileDialog()
        json_path = str(dialog.getOpenFileUrl(self.parent(), self.tr('Import *.docx'), filter="*.json")[0].toLocalFile())
        if osp.exists(json_path):
            self.open_json_proj.emit(json_path)

    def onOpenImages(self) -> None:
        d = None
        if len(self.recent_proj_list) > 0:
            for projp in self.recent_proj_list:
                if not osp.isdir(projp):
                    projp = osp.dirname(projp)
                if osp.exists(projp):
                    d = projp
                    break
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Images (*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif *.gif)")
        if d:
            dialog.setDirectory(d)
        if dialog.exec_():
            selected = dialog.selectedFiles()
            if not selected:
                return
            image_files = [str(p) for p in selected if osp.isfile(p) and osp.splitext(str(p))[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif', '.gif'}]
            if image_files:
                self.mainwindow.dropOpenFiles(image_files)

    def stateCheckerChanged(self, checker_type: str):
        if checker_type == 'imgtrans':
            self.configChecker.setChecked(False)
            self.imgTransChecked.emit()
        elif checker_type == 'config':
            if self.configChecker.isChecked():
                self.imgTransChecker.setChecked(False)
                self.configChecked.emit()
            else:
                self.imgTransChecker.setChecked(True)
                

    def needleftStackWidget(self) -> bool:
        return self.showPageListLabel.isChecked() or self.globalSearchChecker.isChecked()


class _ProjectChip(QWidget):
    """Small rounded chip showing the current project name with a dirty-state dot."""

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._dirty = False
        self.setFixedHeight(22)
        self.setMinimumWidth(60)
        self.setMaximumWidth(280)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(4)

        self._dot = QLabel(self)
        self._dot.setFixedSize(6, 6)
        self._dot.setStyleSheet("background-color: transparent; border-radius: 3px;")

        self._label = QLabel(self)
        self._label.setObjectName("ProjectChipLabel")
        self._label.setStyleSheet("color: inherit; background: transparent;")
        self._label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self._dot)
        layout.addWidget(self._label, 1)
        self.setLayout(layout)

    def set_project(self, name: str, dirty: bool = False) -> None:
        self._dirty = dirty
        self._label.setText(name or "")
        if dirty:
            self._dot.setStyleSheet("background-color: #FF9800; border-radius: 3px;")
        else:
            self._dot.setStyleSheet("background-color: #4CAF50; border-radius: 3px;")
        self.setVisible(bool(name))

    def set_dirty(self, dirty: bool) -> None:
        if dirty != self._dirty:
            self.set_project(self._label.text(), dirty)


class TitleBar(Widget):

    closebtn_clicked = Signal()
    display_lang_changed = Signal(str)
    enable_module = Signal(int, bool)

    def __init__(self, parent, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)
        self.mainwindow : QMainWindow = parent
        self.mainwindow.installEventFilter(self)
        self.mPos: QPoint = None
        self.normalsize = False
        self.proj_name = ''
        self.page_name = ''
        self.save_state = ''
        self.setFixedHeight(TITLEBAR_HEIGHT)
        self.setMouseTracking(True)

        self.editToolBtn = TitleBarToolBtn(self)
        self.editToolBtn.setText(self.tr('Edit'))

        undoAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'arrow-left.svg')), self.tr('Undo'), self)
        self.undo_trigger = undoAction.triggered
        undoAction.setShortcut(QKeySequence.fromString(get_shortcut("edit.undo", getattr(pcfg, "shortcuts", None))))
        redoAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'arrow-right.svg')), self.tr('Redo'), self)
        self.redo_trigger = redoAction.triggered
        redoAction.setShortcut(QKeySequence.fromString(get_shortcut("edit.redo", getattr(pcfg, "shortcuts", None))))
        pageSearchAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'search.svg')), self.tr('Search'), self)
        self.page_search_trigger = pageSearchAction.triggered
        pageSearchAction.setShortcut(QKeySequence.fromString(get_shortcut("edit.page_search", getattr(pcfg, "shortcuts", None))))
        globalSearchAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'search_activate.svg')), self.tr('Global Search'), self)
        self.global_search_trigger = globalSearchAction.triggered
        globalSearchAction.setShortcut(QKeySequence.fromString(get_shortcut("edit.global_search", getattr(pcfg, "shortcuts", None))))

        replacePreMTkeyword = QAction(self.tr("Keyword substitution for machine translation source text"), self)
        self.replacePreMTkeyword_trigger = replacePreMTkeyword.triggered
        replaceMTkeyword = QAction(self.tr("Keyword substitution for machine translation"), self)
        self.replaceMTkeyword_trigger = replaceMTkeyword.triggered
        replaceOCRkeyword = QAction(self.tr("Keyword substitution for source text"), self)
        self.replaceOCRkeyword_trigger = replaceOCRkeyword.triggered

        translationContextAction = QAction(self.tr("Translation context (project)..."), self)
        self.translation_context_trigger = translationContextAction.triggered

        keywordSubMenu = QMenu(self.tr("Keyword substitution"), self)
        keywordSubMenu.addAction(replaceOCRkeyword)
        keywordSubMenu.addAction(replacePreMTkeyword)
        keywordSubMenu.addAction(replaceMTkeyword)

        editMenu = QMenu(self.editToolBtn)
        editMenu.addActions([undoAction, redoAction])
        editMenu.addSeparator()
        editMenu.addActions([pageSearchAction, globalSearchAction])
        editMenu.addSeparator()
        editMenu.addMenu(keywordSubMenu)
        editMenu.addAction(translationContextAction)
        self.editToolBtn.setMenu(editMenu)
        self.editToolBtn.setPopupMode(QToolButton.InstantPopup)

        self.viewToolBtn = TitleBarToolBtn(self)
        self.viewToolBtn.setText(self.tr('View'))

        # Keep submenu ownership aligned with the visual popup chain for Wayland
        # (upstream dmMaze/BallonsTranslator@1feb09b).
        self.viewMenu = viewMenu = QMenu(self.viewToolBtn)
        self.displayLanguageMenu = QMenu(self.tr("Display Language"), viewMenu)
        self.lang_ac_group = lang_ac_group = QActionGroup(self)
        lang_ac_group.setExclusive(True)
        lang_actions = []
        for lang, lang_code in C.DISPLAY_LANGUAGE_MAP.items():
            la = QAction(lang, self)
            if lang_code == pcfg.display_lang:
                la.setChecked(True)
            la.triggered.connect(self.on_displaylang_triggered)
            la.setCheckable(True)
            lang_ac_group.addAction(la)
            lang_actions.append(la)
        self.displayLanguageMenu.addActions(lang_actions)

        drawBoardAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'drawingtools_pen.svg')), self.tr('Drawing Board'), self)
        texteditAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'bottombar_textedit.svg')), self.tr('Text Editor'), self)

        # MainWindow owns the actual keyboard shortcuts for these actions.
        # Keeping the same keys on these QActions causes Qt ambiguous shortcut overloads
        # (for example P/T doing nothing because both QAction and QShortcut own the key).
        drawBoardAction.setShortcut(QKeySequence())
        texteditAction.setShortcut(QKeySequence())
        draw_key = get_shortcut("view.draw_board", getattr(pcfg, "shortcuts", None)) or "P"
        text_key = get_shortcut("view.text_edit", getattr(pcfg, "shortcuts", None)) or "T"
        drawBoardAction.setToolTip(self.tr('Drawing Board') + f" ({draw_key})")
        texteditAction.setToolTip(self.tr('Text Editor') + f" ({text_key})")
        importTextStyles = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'menu_import_styles.svg')), self.tr('Import Text Styles'), self)
        exportTextStyles = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'menu_export_styles.svg')), self.tr('Export Text Styles'), self)
        spellCheckPanelAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'menu_spellcheck.svg')), self.tr('Spell check panel'), self)
        spellCheckPanelAction.setToolTip(self.tr('Show spell check panel (PR #974).'))
        self.spellcheck_panel_trigger = spellCheckPanelAction.triggered
        keyboardShortcutsAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'menu_keyboard.svg')), self.tr('Keyboard Shortcuts...'), self)
        keyboardShortcutsAction.setShortcut(QKeySequence.fromString(get_shortcut("view.keyboard_shortcuts", getattr(pcfg, "shortcuts", None))))
        self.keyboard_shortcuts_trigger = keyboardShortcutsAction.triggered
        contextMenuOptionsAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'menu_context_menu.svg')), self.tr('Context menu options...'), self)
        # MainWindow owns the Ctrl+Shift+O QShortcut; keep this QAction click-only to avoid ambiguity.
        contextMenuOptionsAction.setShortcut(QKeySequence())
        context_menu_key = get_shortcut("view.context_menu_options", getattr(pcfg, "shortcuts", None)) or "Ctrl+Shift+O"
        contextMenuOptionsAction.setToolTip(
            self.tr('Show or hide canvas right-click menu actions by category.') + f" ({context_menu_key})"
        )
        self.context_menu_options_trigger = contextMenuOptionsAction.triggered
        themeLightAction = QAction(self.tr('Light'), self)
        themeLightAction.setToolTip(self.tr('Use light theme (Eva Light).'))
        themeDarkAction = QAction(self.tr('Dark'), self)
        themeDarkAction.setToolTip(self.tr('Use dark theme (Eva Dark).'))
        themeGroup = QActionGroup(self)
        themeGroup.setExclusive(True)
        themeLightAction.setCheckable(True)
        themeDarkAction.setCheckable(True)
        themeGroup.addAction(themeLightAction)
        themeGroup.addAction(themeDarkAction)
        self.theme_light_trigger = themeLightAction.triggered
        self.theme_dark_trigger = themeDarkAction.triggered
        self.themeLightAction = themeLightAction
        self.themeDarkAction = themeDarkAction

        helpMenu = QMenu(self.tr('Help'), self)
        docAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'search.svg')), self.tr('Documentation'), self)
        docAction.setToolTip(self.tr('Open project README (installation and usage).'))
        aboutAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'menu_about.svg')), self.tr('About'), self)
        aboutAction.setToolTip(self.tr('Show application version and info.'))
        updateFromGitHubAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'menu_update_github.svg')), self.tr('Update from GitHub'), self)
        updateFromGitHubAction.setToolTip(self.tr('Pull latest changes from GitHub. Keeps your config and local files unchanged.'))
        helpMenu.addAction(docAction)
        helpMenu.addAction(aboutAction)
        helpMenu.addSeparator()
        helpMenu.addAction(updateFromGitHubAction)

        viewMenu.addMenu(self.displayLanguageMenu)
        viewMenu.addSeparator()
        # Phase 2: group per-panel toggles into a Panels submenu so the flat View list is shorter.
        panelsMenu = QMenu(self.tr('Panels'), self)
        panelsMenu.addAction(drawBoardAction)
        panelsMenu.addAction(texteditAction)
        panelsMenu.addAction(spellCheckPanelAction)
        viewMenu.addMenu(panelsMenu)
        viewMenu.addSeparator()
        viewMenu.addAction(keyboardShortcutsAction)
        viewMenu.addAction(contextMenuOptionsAction)
        viewMenu.addSeparator()
        viewMenu.addAction(importTextStyles)
        viewMenu.addAction(exportTextStyles)
        viewMenu.addSeparator()
        # Phase 2: group Theme + Theme customizer into an Appearance submenu.
        appearanceMenu = QMenu(self.tr('Appearance'), self)
        themeMenu = QMenu(self.tr('Theme'), self)
        themeMenu.addAction(themeLightAction)
        themeMenu.addAction(themeDarkAction)
        appearanceMenu.addMenu(themeMenu)
        themeCustomizerAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'menu_theme_customizer.svg')), self.tr('Theme and UI customizer...'), self)
        themeCustomizerAction.setToolTip(self.tr('Change accent color, app font, light/dark, simple vs advanced UI.'))
        appearanceMenu.addAction(themeCustomizerAction)
        viewMenu.addMenu(appearanceMenu)
        self.theme_customizer_trigger = themeCustomizerAction.triggered
        viewMenu.addSeparator()
        viewMenu.addMenu(helpMenu)
        self.viewToolBtn.setMenu(viewMenu)
        self.viewToolBtn.setPopupMode(QToolButton.InstantPopup)
        self.textedit_trigger = texteditAction.triggered
        self.drawboard_trigger = drawBoardAction.triggered
        self.importtstyle_trigger = importTextStyles.triggered
        self.exporttstyle_trigger = exportTextStyles.triggered
        self.help_doc_trigger = docAction.triggered
        self.help_about_trigger = aboutAction.triggered
        self.help_update_from_github_trigger = updateFromGitHubAction.triggered

        mergeToolAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'merge-tool.svg')), self.tr('Region merge tool'), self)
        mergeToolAction.setShortcut(QKeySequence.fromString(get_shortcut("edit.merge_tool", getattr(pcfg, "shortcuts", None))))

        self._shortcut_actions_title = [
            ("edit.undo", "Ctrl+Z", undoAction),
            ("edit.redo", "Ctrl+Shift+Z", redoAction),
            ("edit.page_search", "Ctrl+F", pageSearchAction),
            ("edit.global_search", "Ctrl+G", globalSearchAction),
            ("view.draw_board", "P", drawBoardAction),
            ("view.text_edit", "T", texteditAction),
            ("view.keyboard_shortcuts", "Ctrl+K", keyboardShortcutsAction),
            ("view.context_menu_options", "Ctrl+Shift+O", contextMenuOptionsAction),
            ("edit.merge_tool", "Ctrl+Shift+M", mergeToolAction),
        ]

        self.goToolBtn = TitleBarToolBtn(self)
        self.goToolBtn.setText(self.tr('Go'))
        prevPageAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'arrow-left.svg')), self.tr('Previous Page'), self)
        nextPageAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'arrow-right.svg')), self.tr('Next Page'), self)
        prevPageAltAction = QAction(self.tr('Previous Page (alternate)'), self)
        nextPageAltAction = QAction(self.tr('Next Page (alternate)'), self)

        # Page navigation shortcuts are registered as MainWindow QShortcuts.
        # Keep menu QActions click-only; duplicate QAction shortcuts cause:
        # QAction::event: Ambiguous shortcut overload: PgUp/PgDown/A/D
        for _act in (prevPageAction, nextPageAction, prevPageAltAction, nextPageAltAction):
            _act.setShortcut(QKeySequence())
        prevPageAction.setToolTip(self.tr('Previous Page') + " (PageUp)")
        nextPageAction.setToolTip(self.tr('Next Page') + " (PageDown)")
        prevPageAltAction.setToolTip(self.tr('Previous Page (alternate)') + " (A)")
        nextPageAltAction.setToolTip(self.tr('Next Page (alternate)') + " (D)")
        goMenu = QMenu(self.goToolBtn)
        goMenu.addActions([prevPageAction, nextPageAction])
        goMenu.addSeparator()
        goMenu.addActions([prevPageAltAction, nextPageAltAction])
        self.goToolBtn.setMenu(goMenu)
        self.goToolBtn.setPopupMode(QToolButton.InstantPopup)
        self.prevpage_trigger = prevPageAction.triggered
        self.nextpage_trigger = nextPageAction.triggered
        prevPageAltAction.triggered.connect(self.prevpage_trigger)
        nextPageAltAction.triggered.connect(self.nextpage_trigger)

        # Tools menu
        self.toolsToolBtn = TitleBarToolBtn(self)
        self.toolsToolBtn.setText(self.tr('Tools'))

        # Region merge tool
        self.merge_tool_trigger = mergeToolAction.triggered

        reRunDetectAction = QAction(self.tr('Re-run detection only'), self)
        self.re_run_detection_only_trigger = reRunDetectAction.triggered
        reRunOCRAction = QAction(self.tr('Re-run OCR only'), self)
        self.re_run_ocr_only_trigger = reRunOCRAction.triggered
        batchExportAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'save_activate.svg')), self.tr('Export all pages'), self)
        batchExportAction.setShortcut(QKeySequence.fromString(get_shortcut("file.export_all_pages", getattr(pcfg, "shortcuts", None))))
        self.batch_export_trigger = batchExportAction.triggered
        self._shortcut_actions_title.append(("file.export_all_pages", "Ctrl+Shift+S", batchExportAction))
        batchExportAsAction = QAction(self.tr('Export all pages as...'), self)
        batchExportAsAction.setToolTip(self.tr('Choose output folder and format (PNG, JPEG, WebP, JXL).'))
        self.batch_export_as_trigger = batchExportAsAction.triggered
        validateProjAction = QAction(self.tr('Check project'), self)
        self.validate_project_trigger = validateProjAction.triggered

        showBatchReportAction = QAction(self.tr('Show last batch report'), self)
        showBatchReportAction.setToolTip(self.tr('Open the report of skipped pages from the last pipeline run.'))
        showBatchReportAction.setEnabled(False)
        self.show_batch_report_trigger = showBatchReportAction.triggered
        self.show_batch_report_action = showBatchReportAction

        mangaSourceAction = QAction(self.tr('Manga / Comic source...'), self)
        self.manga_source_trigger = mangaSourceAction.triggered
        realtimeTranslatorAction = QAction(self.tr('Realtime Screen Translator (dev)...'), self)
        realtimeTranslatorAction.setToolTip(self.tr('Live OCR + translation for a selected screen/window region without opening a full project. (Developer mode only)'))
        self.realtime_translator_trigger = realtimeTranslatorAction.triggered
        self.realtime_translator_action = realtimeTranslatorAction
        # Only visible in dev mode
        realtimeTranslatorAction.setVisible(pcfg.dev_mode)
        translationAssistAction = QAction(self.tr('Translation Assist (beta)...'), self)
        translationAssistAction.setToolTip(self.tr('Open Translation Assist sidebar for side-by-side candidates and glossary/TM hints.'))
        self.translation_assist_trigger = translationAssistAction.triggered

        batchQueueAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'arrow-right.svg')), self.tr('Batch queue...'), self)
        batchQueueAction.setToolTip(self.tr('Process multiple folders in sequence with Pause/Cancel (issue #1020).'))
        self.batch_queue_trigger = batchQueueAction.triggered

        manageModelsAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'leftbar_config.svg')), self.tr('Manage models...'), self)
        manageModelsAction.setToolTip(self.tr('Check which models are downloaded and download selected models.'))
        self.manage_models_trigger = manageModelsAction.triggered
        installGoogleFontAction = QAction(self.tr('Install Google Font...'), self)
        installGoogleFontAction.setToolTip(self.tr('Download and register a Google Font family (e.g. Anime Ace-like alternatives, Bangers, Noto Sans JP).'))
        self.install_google_font_trigger = installGoogleFontAction.triggered

        retryModelsAction = QAction(self.tr('Retry model downloads'), self)
        retryModelsAction.setToolTip(self.tr('Retry downloading model packages (e.g. after a failed first install).'))
        self.retry_models_trigger = retryModelsAction.triggered
        showDownloadDiagAction = QAction(self.tr('Show model download diagnostics'), self)
        showDownloadDiagAction.setToolTip(self.tr('Show last model download summary and failed items.'))
        self.show_model_download_diag_trigger = showDownloadDiagAction.triggered
        copyStartupDiagAction = QAction(self.tr('Copy startup diagnostics'), self)
        copyStartupDiagAction.setToolTip(self.tr('Copy startup and environment diagnostics to clipboard.'))
        self.copy_startup_diag_trigger = copyStartupDiagAction.triggered
        runtimeResourceSummaryAction = QAction(self.tr('Runtime resource summary'), self)
        runtimeResourceSummaryAction.setToolTip(self.tr('Show current CPU/RAM/VRAM usage and quick quantization guidance.'))
        self.runtime_resource_summary_trigger = runtimeResourceSummaryAction.triggered
        exportStartupDiagAction = QAction(self.tr('Export startup report...'), self)
        exportStartupDiagAction.setToolTip(self.tr('Save startup diagnostics to a text file for support.'))
        self.export_startup_diag_trigger = exportStartupDiagAction.triggered
        openLogFolderAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'openbtn.svg')), self.tr('Open log folder'), self)
        openLogFolderAction.setToolTip(self.tr('Open logs directory in your file explorer.'))
        self.open_log_folder_trigger = openLogFolderAction.triggered
        relaunchPyQt5Action = QAction(self.tr('Relaunch with PyQt5 (safe mode)'), self)
        relaunchPyQt5Action.setToolTip(self.tr('Restart app using --qt-api pyqt5 for compatibility troubleshooting.'))
        self.relaunch_pyqt5_trigger = relaunchPyQt5Action.triggered
        relaunchCpuOnlyAction = QAction(self.tr('Relaunch CPU-only (safe mode)'), self)
        relaunchCpuOnlyAction.setToolTip(self.tr('Restart app with CUDA_VISIBLE_DEVICES empty to disable GPU for this launch.'))
        self.relaunch_cpu_only_trigger = relaunchCpuOnlyAction.triggered

        environmentDoctorAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'doctor.svg')), self.tr('Environment doctor...'), self)
        environmentDoctorAction.setToolTip(self.tr('Run dependency and auth checks and show a report.'))
        self.environment_doctor_trigger = environmentDoctorAction.triggered

        ocrTriageAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'doctor.svg')), self.tr('OCR triage worklist (current page)...'), self)
        ocrTriageAction.setToolTip(self.tr('Show blocks that likely need OCR/translation review and triage.'))
        self.ocr_triage_trigger = ocrTriageAction.triggered

        autoExtractGlossaryAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'add.svg')), self.tr('Auto-extract glossary (current page)...'), self)
        autoExtractGlossaryAction.setToolTip(self.tr('Extract frequent source terms and append to project glossary.'))
        self.auto_extract_glossary_trigger = autoExtractGlossaryAction.triggered

        translationQaAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'search.svg')), self.tr('Translation QA report (current page)...'), self)
        translationQaAction.setToolTip(self.tr('Run glossary candidate extraction and guardrail checks on current page.'))
        self.translation_qa_report_trigger = translationQaAction.triggered

        concordanceSearchAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'search.svg')), self.tr('Concordance search (project)...'), self)
        concordanceSearchAction.setToolTip(self.tr('Search previous source/target lines with page provenance.'))
        self.concordance_search_trigger = concordanceSearchAction.triggered

        saveRunProfileAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'save_activate.svg')), self.tr('Save run profile snapshot...'), self)
        saveRunProfileAction.setToolTip(self.tr('Save current module selections as a project-local run profile.'))
        self.save_run_profile_trigger = saveRunProfileAction.triggered

        layoutReviewSelectedAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'menu_layout_review.svg')), self.tr('Layout review selected textboxes'), self)
        layoutReviewSelectedAction.setToolTip(self.tr('Review/fix layout for selected text boxes on current page.'))
        self.layout_review_selected_trigger = layoutReviewSelectedAction.triggered

        layoutReviewPageAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'menu_layout_review.svg')), self.tr('Layout review entire page'), self)
        layoutReviewPageAction.setToolTip(self.tr('Review/fix layout for all non-vertical text boxes on current page.'))
        self.layout_review_page_trigger = layoutReviewPageAction.triggered

        layoutReviewConfigAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'leftbar_config.svg')), self.tr('Layout review settings...'), self)
        layoutReviewConfigAction.setToolTip(self.tr('Configure provider/model/prompt for page review agent.'))
        self.layout_review_config_trigger = layoutReviewConfigAction.triggered

        letteringWorkflowAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'menu_text_style.svg')), self.tr('Lettering workflow...'), self)
        letteringWorkflowAction.setToolTip(self.tr('Plan/apply typography polish, smart fit, layout review, proof export, and render steps for current, selected, or all pages.'))
        self.lettering_workflow_trigger = letteringWorkflowAction.triggered
        autoFormatQaAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'search.svg')), self.tr('Auto-format QA (before/after)...'), self)
        autoFormatQaAction.setToolTip(self.tr('Live preview per-block auto-format score before and after applying bubble-aware formatting.'))
        self.auto_format_qa_trigger = autoFormatQaAction.triggered

        nextRenderingIssueAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'search-stop.svg')), self.tr('Next rendering issue'), self)
        nextRenderingIssueAction.setToolTip(self.tr('Select the next textbox with overflow, glyph, mask, or writing-mode warnings on the current page.'))
        self.next_rendering_issue_trigger = nextRenderingIssueAction.triggered

        applyRunProfileAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'openbtn.svg')), self.tr('Apply run profile snapshot...'), self)
        applyRunProfileAction.setToolTip(self.tr('Apply a previously saved project-local run profile.'))
        self.apply_run_profile_trigger = applyRunProfileAction.triggered

        releaseCachesAction = QAction(self.tr('Release model caches'), self)
        releaseCachesAction.setToolTip(self.tr('Unload detector/OCR/inpainter/translator models and free GPU memory. Use after a batch to reduce RAM.'))
        self.release_model_caches_trigger = releaseCachesAction.triggered

        clearPipelineCachesAction = QAction(self.tr('Clear OCR and translation caches'), self)
        clearPipelineCachesAction.setToolTip(self.tr('Clear in-session OCR and translation caches (current project). Next run will recompute.'))
        self.clear_pipeline_caches_trigger = clearPipelineCachesAction.triggered

        toolsMenu = QMenu(self.toolsToolBtn)
        projectMenu = QMenu(self.tr('Project'), self)
        projectMenu.addAction(mergeToolAction)
        projectMenu.addAction(reRunDetectAction)
        projectMenu.addAction(reRunOCRAction)
        projectMenu.addAction(validateProjAction)
        projectMenu.addAction(showBatchReportAction)
        projectMenu.addSeparator()
        projectMenu.addAction(saveRunProfileAction)
        projectMenu.addAction(applyRunProfileAction)
        projectMenu.addSeparator()
        projectMenu.addAction(layoutReviewSelectedAction)
        projectMenu.addAction(layoutReviewPageAction)
        projectMenu.addAction(layoutReviewConfigAction)
        projectMenu.addAction(letteringWorkflowAction)
        projectMenu.addAction(autoFormatQaAction)
        projectMenu.addAction(nextRenderingIssueAction)
        projectMenu.addSeparator()
        projectMenu.addAction(translationQaAction)
        projectMenu.addAction(ocrTriageAction)
        projectMenu.addAction(autoExtractGlossaryAction)
        projectMenu.addAction(concordanceSearchAction)
        batchFindReplaceAction = QAction(self.tr('Batch find/replace translations...'), self)
        batchFindReplaceAction.setToolTip(self.tr('Preview regex find/replace across project translations before applying.'))
        self.batch_find_replace_trigger = batchFindReplaceAction.triggered
        projectMenu.addAction(batchFindReplaceAction)
        projectMenu.addSeparator()
        projectMenu.addAction(realtimeTranslatorAction)
        projectMenu.addAction(translationAssistAction)
        exportMenuTools = QMenu(self.tr('Export'), self)
        exportMenuTools.addAction(batchExportAction)
        exportMenuTools.addAction(batchExportAsAction)
        exportLptxtAction = QAction(self.tr('Export translation to LPtxt...'), self)
        exportLptxtAction.setToolTip(self.tr('Export translations in LPtxt format for auto-labeling tools (e.g. 气泡翻译器自动打标).'))
        self.export_lptxt_trigger = exportLptxtAction.triggered
        exportMenuTools.addAction(exportLptxtAction)
        layeredPsdHandoffAction = QAction(self.tr('Export layered PSD handoff...'), self)
        layeredPsdHandoffAction.setToolTip(self.tr('Export original/inpainted/mask/final helper layers plus editable text metadata and a Photoshop JSX rebuild script.'))
        self.export_layered_psd_handoff_trigger = layeredPsdHandoffAction.triggered
        exportMenuTools.addAction(layeredPsdHandoffAction)
        svgTextHandoffAction = QAction(self.tr('Export SVG text handoff...'), self)
        svgTextHandoffAction.setToolTip(self.tr('Export current page as an SVG with editable translated text elements for vector editors.'))
        self.export_svg_text_handoff_trigger = svgTextHandoffAction.triggered
        exportMenuTools.addAction(svgTextHandoffAction)
        importGlossaryPreviewAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'openbtn.svg')), self.tr('Import glossary with preview...'), self)
        importGlossaryPreviewAction.setToolTip(self.tr('Preview merge/replace impact before importing glossary entries.'))
        self.import_glossary_preview_trigger = importGlossaryPreviewAction.triggered
        exportTransCsvAction = QAction(self.tr('Export translation CSV...'), self)
        exportTransCsvAction.setToolTip(self.tr('Export source/translation rows as CSV for spreadsheet and CAT workflows.'))
        self.export_translation_csv_trigger = exportTransCsvAction.triggered
        exportMenuTools.addAction(exportTransCsvAction)
        importTransCsvAction = QAction(self.tr('Import translation CSV...'), self)
        importTransCsvAction.setToolTip(self.tr('Import translation CSV by page + stable block ID with index fallback.'))
        self.import_translation_csv_trigger = importTransCsvAction.triggered
        exportMenuTools.addAction(importTransCsvAction)
        exportMenuTools.addAction(importGlossaryPreviewAction)
        exportTransJsonAction = QAction(self.tr('Export translation JSON...'), self)
        exportTransJsonAction.setToolTip(self.tr('Export source/translation with stable block IDs for external translation workflows.'))
        self.export_translation_json_trigger = exportTransJsonAction.triggered
        exportMenuTools.addAction(exportTransJsonAction)
        importTransJsonAction = QAction(self.tr('Import translation JSON...'), self)
        importTransJsonAction.setToolTip(self.tr('Import translation JSON back by page and stable block ID (with index fallback).'))
        self.import_translation_json_trigger = importTransJsonAction.triggered
        exportMenuTools.addAction(importTransJsonAction)
        exportXliffAction = QAction(self.tr('Export XLIFF...'), self)
        exportXliffAction.setToolTip(self.tr('Export source/translation blocks in XLIFF 1.2 for CAT-tool workflows.'))
        self.export_xliff_trigger = exportXliffAction.triggered
        exportMenuTools.addAction(exportXliffAction)
        importXliffAction = QAction(self.tr('Import XLIFF...'), self)
        importXliffAction.setToolTip(self.tr('Import translated XLIFF back into current project by page and block index.'))
        self.import_xliff_trigger = importXliffAction.triggered
        exportMenuTools.addAction(importXliffAction)
        structuredOcrExportAction = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'structured_ocr_export.svg')), self.tr('Export structured OCR JSON...'), self)
        structuredOcrExportAction.setToolTip(self.tr('Export page/block geometry, OCR text, translations, completion state, and font hints as JSON for LLM/tooling workflows.'))
        self.export_structured_ocr_trigger = structuredOcrExportAction.triggered
        exportMenuTools.addAction(structuredOcrExportAction)
        sourcesMenu = QMenu(self.tr('Sources'), self)
        sourcesMenu.addAction(mangaSourceAction)
        queueMenu = QMenu(self.tr('Queue'), self)
        queueMenu.addAction(batchQueueAction)
        modelsMenu = QMenu(self.tr('Models'), self)
        modelsMenu.addAction(manageModelsAction)
        modelsMenu.addAction(installGoogleFontAction)
        modelsMenu.addAction(retryModelsAction)
        modelsMenu.addAction(environmentDoctorAction)
        modelsMenu.addAction(showDownloadDiagAction)
        modelsMenu.addSeparator()
        modelsMenu.addAction(copyStartupDiagAction)
        modelsMenu.addAction(runtimeResourceSummaryAction)
        modelsMenu.addAction(exportStartupDiagAction)
        modelsMenu.addAction(openLogFolderAction)
        modelsMenu.addAction(relaunchPyQt5Action)
        modelsMenu.addAction(relaunchCpuOnlyAction)
        modelsMenu.addSeparator()
        modelsMenu.addAction(releaseCachesAction)
        modelsMenu.addAction(clearPipelineCachesAction)
        toolsMenu.addMenu(projectMenu)
        toolsMenu.addMenu(exportMenuTools)
        toolsMenu.addMenu(sourcesMenu)
        toolsMenu.addMenu(queueMenu)
        toolsMenu.addMenu(modelsMenu)
        self.toolsToolBtn.setMenu(toolsMenu)
        self.toolsToolBtn.setPopupMode(QToolButton.InstantPopup)

        self.runToolBtn = TitleBarToolBtn(self)
        self.runToolBtn.setObjectName('PipelineRunBtn')
        self.runToolBtn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        run_icon_path = osp.join(C.PROGRAM_PATH, 'icons', 'bottombar_translate.svg')
        if osp.isfile(run_icon_path):
            self.runToolBtn.setIcon(QIcon(run_icon_path))
        self.runToolBtn.setText(self.tr('Pipeline'))

        self.stageActions = stageActions = [
            QAction(self.tr('Enable Text Dection'), self),
            QAction(self.tr('Enable OCR'), self),
            QAction(self.tr('Enable Translation'), self),
            QAction(self.tr('Enable Inpainting'), self)
        ]
        for idx, sa in enumerate(stageActions):
            sa.setCheckable(True)
            sa.setChecked(pcfg.module.stage_enabled(idx))
            sa.triggered.connect(self.stageEnableStateChanged)

        runAction = QAction(self.tr('Run'), self)
        runWoUpdateTextStyle = QAction(self.tr('Run without update textstyle'), self)
        translatePageAction = QAction(self.tr('Translate page'), self)
        presetFullAc = QAction(self.tr('Preset: Full'), self)
        presetDetectOCRAc = QAction(self.tr('Preset: Detect + OCR only'), self)
        presetTranslateAc = QAction(self.tr('Preset: Translate only'), self)
        presetInpaintAc = QAction(self.tr('Preset: Inpaint only'), self)
        self.run_preset_full_trigger = presetFullAc.triggered
        self.run_preset_detect_ocr_trigger = presetDetectOCRAc.triggered
        self.run_preset_translate_trigger = presetTranslateAc.triggered
        self.run_preset_inpaint_trigger = presetInpaintAc.triggered
        runMenu = QMenu(self.runToolBtn)
        runMenu.addActions(stageActions)
        runMenu.addSeparator()
        presetMenu = QMenu(self.tr('Pipeline presets'), self)
        presetMenu.addActions([presetFullAc, presetDetectOCRAc, presetTranslateAc, presetInpaintAc])
        runMenu.addMenu(presetMenu)
        runMenu.addSeparator()
        runMenu.addActions([runAction, runWoUpdateTextStyle, translatePageAction])
        runMenu.addSeparator()
        videoTranslatorAc = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'video_render.svg')), self.tr('Video translator...'), self)
        videoTranslatorAc.setToolTip(self.tr('Translate hardcoded subtitles in video: detect, OCR, translate, inpaint per frame.'))
        runMenu.addAction(videoTranslatorAc)
        self.video_translator_trigger = videoTranslatorAc.triggered
        subtitleFileTranslatorAc = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'subtitle_export.svg')), self.tr('Translate subtitle file…'), self)
        subtitleFileTranslatorAc.setToolTip(
            self.tr('Translate a standalone .srt or timestamped .txt using the configured translator (OpenRouter via LLM API).')
        )
        runMenu.addAction(subtitleFileTranslatorAc)
        self.subtitle_file_translator_trigger = subtitleFileTranslatorAc.triggered
        videoSubtitleEditorAc = QAction(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'video_subtitle_editor.svg')), self.tr('Video Subtitle Editor...'), self)
        videoSubtitleEditorAc.setToolTip(self.tr('Edit video captions: cut, edit text, frame-accurate timing, export SRT/ASS/VTT, render with burned-in subtitles.'))
        runMenu.addAction(videoSubtitleEditorAc)
        self.video_subtitle_editor_trigger = videoSubtitleEditorAc.triggered
        self.runToolBtn.setMenu(runMenu)
        self.runToolBtn.setPopupMode(QToolButton.InstantPopup)
        self.run_trigger = runAction.triggered
        self.run_woupdate_textstyle_trigger = runWoUpdateTextStyle.triggered
        self.translate_page_trigger = translatePageAction.triggered

        self.iconLabel = QLabel(self)
        if not C.ON_MACOS:
            self.iconLabel.setFixedWidth(LEFTBAR_WIDTH - 12)
        else:
            self.iconLabel.setFixedWidth(LEFTBAR_WIDTH + 8)

        self.fileToolBtn = TitleBarToolBtn(self)
        self.fileToolBtn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        file_icon_path = osp.join(C.PROGRAM_PATH, 'icons', 'openbtn.svg')
        if osp.isfile(file_icon_path):
            self.fileToolBtn.setIcon(QIcon(file_icon_path))
        self.fileToolBtn.setText(self.tr('File'))
        self.fileToolBtn.setPopupMode(QToolButton.InstantPopup)

        self.titleLabel = QLabel(self.tr('BallonsTranslatorPro'))
        self.titleLabel.setObjectName('TitleLabel')
        self.titleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Omni search (menus + settings + canvas): compact icon toggle + animated QLineEdit.
        self._searchExpanded = False
        self._searchTargetWidth = 320
        self._searchToggleBtn = QToolButton(self)
        self._searchToggleBtn.setObjectName("OmniSearchToggleBtn")
        self._searchToggleBtn.setIcon(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'search.svg')))
        self._searchToggleBtn.setToolTip(self.tr("Open search"))
        self._searchToggleBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._searchToggleBtn.clicked.connect(self._toggle_omni_search)

        self.omniSearch = QLineEdit(self)
        self.omniSearch.setObjectName("OmniSearch")
        self.omniSearch.setPlaceholderText(self.tr("Search: menus, settings, canvas…"))
        # Width is dynamically adjusted in resizeEvent so it stays compact windowed,
        # but becomes wider in fullscreen/maximized.
        self.omniSearch.setMinimumWidth(260)
        self.omniSearch.setMaximumWidth(1200)
        self.omniSearch.textEdited.connect(self._on_omni_search_text_edited)
        self.omniSearch.returnPressed.connect(self._on_omni_search_return_pressed)
        self.omniSearch.installEventFilter(self)

        self._omni_model = QStandardItemModel(self.omniSearch)
        self._omni_proxy = QSortFilterProxyModel(self.omniSearch)
        self._omni_proxy.setSourceModel(self._omni_model)
        self._omni_proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._omni_proxy.setFilterKeyColumn(0)

        self._omni_completer = QCompleter(self._omni_proxy, self.omniSearch)
        self._omni_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._omni_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        try:
            # Prefer contains-style matching in the popup list.
            self._omni_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        except Exception:
            pass
        # QCompleter activation differs between Qt/PyQt/PySide versions:
        # sometimes the signal provides a QModelIndex, sometimes the display string.
        # Connect both forms and resolve the payload robustly in the handlers below.
        try:
            self._omni_completer.activated[QModelIndex].connect(self._on_omni_search_index_activated)
        except Exception:
            pass
        try:
            self._omni_completer.activated[str].connect(self._on_omni_search_text_activated)
        except Exception:
            pass
        self.omniSearch.setCompleter(self._omni_completer)
        self._omni_actions_cache = None  # lazily built list of (label, QAction)
        
        # Center container: keep title + search aligned to the canvas area (between left/right panes).
        # Use two rows so the title stays visually centered and the search doesn't "steal" its space.
        self._centerContainer = QWidget(self)
        self._centerLayout = QVBoxLayout(self._centerContainer)
        self._centerLayout.setContentsMargins(0, 0, 0, 0)
        self._centerLayout.setSpacing(0)

        # Row 1: title / project / page label
        self._centerLayout.addWidget(self.titleLabel)

        # Row 2: omni search + dropdown button
        self._searchRow = QWidget(self._centerContainer)
        self._searchRowLayout = QHBoxLayout(self._searchRow)
        self._searchRowLayout.setContentsMargins(0, 0, 0, 0)
        self._searchRowLayout.setSpacing(6)

        # Add a small dropdown button (QLineEdit has no built-in arrow).
        self._omniDropBtn = QToolButton(self._searchRow)
        self._omniDropBtn.setObjectName("OmniSearchDropBtn")
        self._omniDropBtn.setText("▾")
        self._omniDropBtn.setToolTip(self.tr("Show search results"))
        self._omniDropBtn.setCursor(Qt.CursorShape.ArrowCursor)
        self._omniDropBtn.clicked.connect(self._on_omni_drop_clicked)

        # Slightly shorter height so title + search fit cleanly.
        try:
            # Keep small so it doesn't collide with title text.
            self.omniSearch.setFixedHeight(24)
            self._omniDropBtn.setFixedHeight(24)
            self._omniDropBtn.setFixedWidth(24)
            self._searchToggleBtn.setFixedHeight(24)
            self._searchToggleBtn.setFixedWidth(24)
        except Exception:
            pass
        self.omniSearch.setMaximumWidth(0)
        self.omniSearch.hide()
        self._omniDropBtn.hide()

        self._searchRowLayout.addStretch()
        self._searchRowLayout.addWidget(self._searchToggleBtn, 0)
        self._searchRowLayout.addWidget(self.omniSearch, 0)
        self._searchRowLayout.addWidget(self._omniDropBtn, 0)
        self._searchRowLayout.addStretch()

        self._centerLayout.addWidget(self._searchRow)
        self._centerContainer.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.projectChip = _ProjectChip(self)
        self.projectChip.setObjectName("ProjectChip")
        self.projectChip.hide()

        hlayout = QHBoxLayout(self)
        hlayout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hlayout.addWidget(self.iconLabel)
        hlayout.addWidget(self.fileToolBtn)
        hlayout.addWidget(self.editToolBtn)
        hlayout.addWidget(self.viewToolBtn)
        hlayout.addWidget(self.goToolBtn)
        hlayout.addWidget(self.runToolBtn)
        hlayout.addWidget(self.toolsToolBtn)
        hlayout.addWidget(self.projectChip)
        hlayout.addStretch()
        hlayout.addWidget(self._centerContainer)
        hlayout.addStretch()
        hlayout.setContentsMargins(0, 0, 0, 0)

        if not C.ON_MACOS:
            self.minBtn = QPushButton()
            self.minBtn.setObjectName('minBtn')
            self.minBtn.clicked.connect(self.onMinBtnClicked)
            self.maxBtn = QCheckBox()
            self.maxBtn.setObjectName('maxBtn')
            self.maxBtn.clicked.connect(self.onMaxBtnClicked)
            self.maxBtn.setFixedSize(48, 27)
            self.closeBtn = QPushButton()
            self.closeBtn.setObjectName('closeBtn')
            self.closeBtn.clicked.connect(self.closebtn_clicked)
            hlayout.addWidget(self.minBtn)
            hlayout.addWidget(self.maxBtn)
            hlayout.addWidget(self.closeBtn)
            hlayout.setContentsMargins(0, 0, 0, 0)
            hlayout.setSpacing(0)

        for btn in (self.fileToolBtn, self.editToolBtn, self.viewToolBtn, self.goToolBtn, self.runToolBtn, self.toolsToolBtn):
            install_button_animations(btn, normal_opacity=0.9, press_opacity=0.74, with_scale=True)

    def resizeEvent(self, event) -> None:
        """Scale center container to canvas width (fullscreen-friendly)."""
        try:
            # Prefer actual canvas width (splitter middle pane).
            canvas_w = None
            try:
                mw = getattr(self, "mainwindow", None)
                splitter = getattr(mw, "comicTransSplitter", None)
                if splitter is not None and hasattr(splitter, "sizes"):
                    sizes = splitter.sizes()
                    if isinstance(sizes, (list, tuple)) and len(sizes) >= 2:
                        canvas_w = int(sizes[1])
            except Exception:
                canvas_w = None
            if canvas_w is None or canvas_w <= 0:
                canvas_w = int(self.width() * 0.45)
            center_w = int(max(420, min(1400, canvas_w)))
            self._centerContainer.setFixedWidth(center_w)
            # Search takes ~45% of canvas area, clamped.
            search_w = int(max(280, min(950, center_w * 0.45)))
            self._searchTargetWidth = search_w
            if self._searchExpanded:
                self.omniSearch.setMaximumWidth(search_w)
        except Exception:
            pass
        return super().resizeEvent(event)


    def openOmniSearch(self):
        """Open/focus the omni search from MainWindow shortcuts such as Ctrl+P."""
        if not bool(getattr(self, '_searchExpanded', False)):
            self._toggle_omni_search()
        else:
            self.omniSearch.show()
            self._omniDropBtn.show()
            self.omniSearch.setFocus(Qt.FocusReason.ShortcutFocusReason)
            self.omniSearch.selectAll()

    def closeOmniSearch(self):
        """Close the omni search safely."""
        if bool(getattr(self, '_searchExpanded', False)):
            self._toggle_omni_search()
        else:
            try:
                self._omni_completer.popup().hide()
            except Exception:
                pass
            self.omniSearch.hide()
            self._omniDropBtn.hide()

    def _toggle_omni_search(self):
        expand = not bool(getattr(self, '_searchExpanded', False))
        self._searchExpanded = expand
        self._searchToggleBtn.setToolTip(self.tr("Close search") if expand else self.tr("Open search"))
        self._searchToggleBtn.setIcon(QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'search_activate.svg' if expand else 'search.svg')))
        if expand:
            self.omniSearch.show()
            self._omniDropBtn.show()
            self.omniSearch.setFocus()
            self.omniSearch.selectAll()
        else:
            self.omniSearch.clear()
            self._omni_model.clear()

        try:
            anim = QPropertyAnimation(self.omniSearch, b"maximumWidth", self)
            anim.setDuration(180)
            anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
            anim.setStartValue(self.omniSearch.maximumWidth())
            anim.setEndValue(self._searchTargetWidth if expand else 0)
            if not expand:
                def _hide_after():
                    self.omniSearch.hide()
                    self._omniDropBtn.hide()
                anim.finished.connect(_hide_after)
            self._search_anim = anim
            anim.start()
        except Exception:
            self.omniSearch.setMaximumWidth(self._searchTargetWidth if expand else 0)

    def _walk_menu_actions(self, menu: QMenu, prefix: str = ""):
        """Yield (label, QAction) for all actions in menu (including submenus)."""
        if menu is None:
            return
        for act in menu.actions() or []:
            try:
                if act is None:
                    continue
                if act.isSeparator():
                    continue
                text = (act.text() or "").replace("&", "").strip()
                if not text:
                    continue
                sub = act.menu()
                if sub is not None:
                    sub_prefix = f"{prefix}{text} > "
                    yield from self._walk_menu_actions(sub, prefix=sub_prefix)
                    continue
                tip = (act.toolTip() or "").strip()
                shortcut = act.shortcut().toString() if hasattr(act, "shortcut") else ""
                extra = ""
                if tip:
                    extra += f" ({tip})"
                if shortcut:
                    extra += f" [{shortcut}]"
                label = (f"Menu > {prefix}{text}" if prefix else f"Menu > {text}") + extra
                yield (label, act)
            except Exception:
                continue

    def _get_actions_cache(self, force_rebuild: bool = False):
        if self._omni_actions_cache is not None and not force_rebuild:
            return self._omni_actions_cache
        actions = []
        try:
            # Tool buttons hold menus (some are set later, but by the time user searches they should exist)
            for btn, name in (
                (getattr(self, "fileToolBtn", None), "File"),
                (getattr(self, "editToolBtn", None), "Edit"),
                (getattr(self, "viewToolBtn", None), "View"),
                (getattr(self, "goToolBtn", None), "Go"),
                (getattr(self, "runToolBtn", None), "Run"),
                (getattr(self, "toolsToolBtn", None), "Tools"),
            ):
                if btn is None:
                    continue
                m = btn.menu()
                if m is None:
                    continue
                actions.extend(list(self._walk_menu_actions(m, prefix=f"{name} > ")))
        except Exception:
            pass
        self._omni_actions_cache = actions
        return actions

    def _add_result_item(self, label: str, payload: dict):
        it = QStandardItem(label)
        it.setEditable(False)
        it.setData(payload, Qt.ItemDataRole.UserRole)
        self._omni_model.appendRow(it)

    def _on_omni_search_text_edited(self, text: str):
        q_raw = text or ""
        q = q_raw.strip()
        self._omni_model.clear()
        if not q:
            return
        q_low = q.lower()
        n_added = 0
        max_items = 40

        # 1) Menu actions
        for label, act in self._get_actions_cache():
            if q_low in label.lower():
                self._add_result_item(label, {"type": "action", "action": act})
                n_added += 1
                if n_added >= max_items:
                    break

        # 2) Settings (ConfigPanel sub-blocks)
        if n_added < max_items:
            try:
                mw = self.mainwindow
                cp = getattr(mw, "configPanel", None)
                content = getattr(cp, "configContent", None)
                blocks = getattr(content, "config_block_list", None) or []
                for b in blocks:
                    header = ""
                    try:
                        header = (getattr(b, "header", None) and getattr(b.header, "text", lambda: "")()) or ""
                    except Exception:
                        header = ""
                    for sb in getattr(b, "subblock_list", []) or []:
                        name = ""
                        try:
                            nl = getattr(sb, "name_label", None)
                            name = (nl.text() if nl is not None else "") or ""
                        except Exception:
                            name = ""
                        if not name:
                            continue
                        # Include current value text in search so numbers (e.g. 60, 4096) are searchable.
                        value_str = ""
                        w = getattr(sb, "widget", None)
                        try:
                            if isinstance(w, QLineEdit):
                                value_str = (w.text() or "").strip()
                            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                                value_str = str(w.value())
                            elif isinstance(w, QCheckBox):
                                value_str = "true" if w.isChecked() else "false"
                            elif isinstance(w, QComboBox):
                                value_str = (w.currentText() or "").strip()
                        except Exception:
                            value_str = ""
                        base_label = f"Settings > {header} > {name}" if header else f"Settings > {name}"
                        label = base_label
                        if value_str:
                            label = f"{base_label} = {value_str}"
                        hay = f"{base_label} {value_str}".lower()
                        if q_low in hay:
                            self._add_result_item(label, {"type": "config", "idx0": sb.idx0, "idx1": sb.idx1})
                            n_added += 1
                            if n_added >= max_items:
                                break
                    if n_added >= max_items:
                        break
            except Exception:
                pass

        # 3) Canvas (current page blocks)
        if n_added < max_items:
            try:
                mw = self.mainwindow
                st = getattr(mw, "st_manager", None)
                items = getattr(st, "textblk_item_list", None) or []
                for it in items:
                    blk = getattr(it, "blk", None)
                    if blk is None:
                        continue
                    src = (blk.get_text() if hasattr(blk, "get_text") else "") or ""
                    trans = (getattr(blk, "translation", None) or "") or ""
                    hay = f"{src} {trans}".lower()
                    if q_low in hay:
                        short = (trans.strip() or src.strip() or "").replace("\n", " ")
                        if len(short) > 80:
                            short = short[:80] + "…"
                        label = f"Canvas > Block {getattr(it, 'idx', '?')}: {short}"
                        self._add_result_item(label, {"type": "canvas", "block_idx": int(getattr(it, "idx", -1))})
                        n_added += 1
                        if n_added >= max_items:
                            break
            except Exception:
                pass

        # 4) Other UI text (format panel, bottom bar toggles, etc.)
        if n_added < max_items:
            try:
                mw = self.mainwindow
                # Bottom bar: text/tooltip of main toggles and actions
                bb = getattr(mw, "bottomBar", None)
                if bb is not None:
                    bottom_items = []
                    for attr_name in ("texteditChecker", "paintChecker", "textblockChecker", "spellcheckChecker"):
                        w = getattr(bb, attr_name, None)
                        if w is None:
                            continue
                        txt = (getattr(w, "text", lambda: "")() or "").strip()
                        tip = (getattr(w, "toolTip", lambda: "")() or "").strip()
                        label = txt or tip
                        if not label:
                            continue
                        bottom_items.append(label)
                    for lbl in bottom_items:
                        if q_low in lbl.lower():
                            self._add_result_item(f"UI > Bottom bar > {lbl}", {"type": "ui_text", "target": "bottom"})
                            n_added += 1
                            if n_added >= max_items:
                                break

                # Right text/format panel: labels and buttons
                tp = getattr(mw, "textPanel", None)
                if tp is not None and n_added < max_items:
                    from qtpy.QtWidgets import QLabel as _Lbl, QAbstractButton as _Btn

                    for lbl in tp.findChildren(_Lbl):
                        txt = (lbl.text() or "").strip()
                        if not txt:
                            continue
                        if q_low in txt.lower():
                            self._add_result_item(f"UI > Format/Text panel > {txt}", {"type": "ui_text", "target": "text_panel"})
                            n_added += 1
                            if n_added >= max_items:
                                break
                    # Also search button texts/tooltips for format panel
                    if n_added < max_items:
                        for btn in tp.findChildren(_Btn):
                            txt = (btn.text() or "").strip()
                            tip = (btn.toolTip() or "").strip()
                            label = txt or tip
                            if not label:
                                continue
                            if q_low in label.lower():
                                self._add_result_item(f"UI > Format/Text panel > {label}", {"type": "ui_text", "target": "text_panel"})
                                n_added += 1
                                if n_added >= max_items:
                                    break
            except Exception:
                pass

        # Filter proxy and show popup (command palette UX). This does not replace the typed text.
        try:
            # Use a "contains" regex, not fixed-string equality, so typing finds items like "Dark Mode".
            pat = QRegularExpression.escape(q_raw)
            self._omni_proxy.setFilterRegularExpression(QRegularExpression(pat, QRegularExpression.PatternOption.CaseInsensitiveOption))
            if self._omni_model.rowCount() > 0:
                self._omni_completer.complete(self.omniSearch.rect())
        except Exception:
            pass

    def _on_omni_search_return_pressed(self):
        """Activate current highlighted result, or the first result if popup is hidden."""
        try:
            cidx = self._omni_completer.currentIndex()
            if cidx is not None and cidx.isValid():
                self._on_omni_search_index_activated(cidx)
                return
            if self._omni_model.rowCount() > 0:
                src_idx = self._omni_model.index(0, 0)
                self._on_omni_search_index_activated(src_idx)
        except Exception:
            pass

    def _on_omni_drop_clicked(self):
        """Open the popup list on demand."""
        try:
            q_raw = (self.omniSearch.text() or "")
            if q_raw.strip():
                self._on_omni_search_text_edited(q_raw)
            else:
                # Show a short list of menu actions as a starting point.
                self._omni_model.clear()
                n = 0
                for label, act in self._get_actions_cache():
                    self._add_result_item(label, {"type": "action", "action": act})
                    n += 1
                    if n >= 30:
                        break
                self._omni_proxy.setFilterRegularExpression(QRegularExpression(""))
            self.omniSearch.setFocus(Qt.FocusReason.OtherFocusReason)
            self._omni_completer.complete(self.omniSearch.rect())
        except Exception:
            pass

    def _payload_for_omni_label(self, label: str):
        """Find the stored payload for a visible completion label."""
        label = str(label or "").strip()
        if not label:
            return None

        for row in range(self._omni_model.rowCount()):
            item = self._omni_model.item(row, 0)
            if item is None:
                continue
            if str(item.text() or "").strip() == label:
                payload = item.data(Qt.ItemDataRole.UserRole)
                return payload if isinstance(payload, dict) else None
        return None

    def _payload_for_omni_index(self, index: QModelIndex):
        """Resolve a completer activation index to the original QStandardItem payload.

        QCompleter can emit indexes from its private completion model instead of
        the source model/proxy. The old code returned early for those indexes,
        so clicking a search result appeared to do nothing.
        """
        try:
            if index is None or not index.isValid():
                return None

            model = index.model()

            if model is self._omni_model:
                item = self._omni_model.itemFromIndex(index)
                payload = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
                return payload if isinstance(payload, dict) else None

            if model is self._omni_proxy:
                src_index = self._omni_proxy.mapToSource(index)
                item = self._omni_model.itemFromIndex(src_index) if src_index.isValid() else None
                payload = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
                return payload if isinstance(payload, dict) else None

            # Completion model or unknown proxy: resolve by displayed text.
            label = str(model.data(index, Qt.ItemDataRole.DisplayRole) or "").strip()
            return self._payload_for_omni_label(label)
        except Exception:
            return None

    def _on_omni_search_text_activated(self, text: str):
        """Triggered by Qt variants that emit the selected completion text."""
        payload = self._payload_for_omni_label(str(text or "").strip())
        self._activate_omni_payload(payload)

    def _on_omni_search_index_activated(self, proxy_index: QModelIndex):
        """Triggered when the user explicitly selects a completion by click/enter."""
        payload = self._payload_for_omni_index(proxy_index)
        self._activate_omni_payload(payload)

    def _activate_omni_payload(self, payload: dict):
        if not isinstance(payload, dict):
            return

        def _run_payload():
            try:
                t = payload.get("type")
                if t == "action":
                    act = payload.get("action")
                    if act is not None and hasattr(act, "trigger"):
                        act.trigger()
                elif t == "config":
                    idx0, idx1 = payload.get("idx0"), payload.get("idx1")
                    if idx0 is not None and idx1 is not None:
                        self.mainwindow.jump_to_config_item(int(idx0), int(idx1))
                elif t == "canvas":
                    bi = int(payload.get("block_idx", -1))
                    if bi >= 0:
                        self.mainwindow.jump_to_canvas_block(bi)
                elif t == "ui_text":
                    target = payload.get("target") or ""
                    if target == "text_panel":
                        if hasattr(self.mainwindow, "jump_to_text_panel"):
                            self.mainwindow.jump_to_text_panel()
                    elif target == "bottom":
                        if hasattr(self.mainwindow, "setupImgTransUI"):
                            self.mainwindow.setupImgTransUI()
            except Exception:
                pass

        try:
            self.omniSearch.clear()
            self.omniSearch.clearFocus()
            self._omni_model.clear()
            self._omni_proxy.setFilterRegularExpression(QRegularExpression(""))
            if bool(getattr(self, '_searchExpanded', False)):
                self._toggle_omni_search()
        except Exception:
            pass

        # Run after popup/search collapse so dialogs and menus can take focus.
        QTimer.singleShot(0, _run_payload)

    def setLeftBar(self, leftBar):
        """Build File menu and connect to left bar (open/save/export/import). Call from mainwindow after both are created."""
        fileMenu = QMenu(self.fileToolBtn)
        actionOpenFolder = QAction(self.tr("Open Folder ..."), self)
        actionOpenFolder.triggered.connect(leftBar.onOpenFolder)
        actionOpenImages = QAction(self.tr("Open Images ..."), self)
        actionOpenImages.triggered.connect(leftBar.onOpenImages)
        actionOpenProj = QAction(self.tr("Open Project ... *.json"), self)
        actionOpenProj.triggered.connect(leftBar.onOpenProj)
        fileMenu.addActions([actionOpenFolder, actionOpenImages, actionOpenProj])
        fileMenu.addMenu(leftBar.recentMenu)
        fileMenu.addSeparator()
        actionCloseProject = QAction(self.tr("Close project and go to welcome screen"), self)
        actionCloseProject.triggered.connect(leftBar.close_project_requested.emit)
        fileMenu.addAction(actionCloseProject)
        fileMenu.addSeparator()
        actionSave = QAction(self.tr("Save Project"), self)
        actionSave.triggered.connect(leftBar.save_proj.emit)
        fileMenu.addAction(actionSave)
        fileMenu.addSeparator()
        exportMenu = QMenu(self.tr("Export"), self)
        actionExportDoc = QAction(self.tr("Export as Doc"), self)
        actionExportDoc.triggered.connect(leftBar.export_doc.emit)
        actionExportCurrentPage = QAction(self.tr("Export current page as..."), self)
        actionExportCurrentPage.triggered.connect(leftBar.export_current_page_as.emit)
        actionExportSrcTxt = QAction(self.tr("Export source text as TXT"), self)
        actionExportSrcTxt.triggered.connect(leftBar.export_src_txt.emit)
        actionExportTransTxt = QAction(self.tr("Export translation as TXT"), self)
        actionExportTransTxt.triggered.connect(leftBar.export_trans_txt.emit)
        actionExportSrcMD = QAction(self.tr("Export source text as markdown"), self)
        actionExportSrcMD.triggered.connect(leftBar.export_src_md.emit)
        actionExportTransMD = QAction(self.tr("Export translation as markdown"), self)
        actionExportTransMD.triggered.connect(leftBar.export_trans_md.emit)
        actionExportBothTxt = QAction(self.tr("Export source + translation as TXT"), self)
        actionExportBothTxt.triggered.connect(leftBar.export_both_txt.emit)
        exportMenu.addActions([actionExportDoc, actionExportCurrentPage])
        exportMenu.addSeparator()
        exportMenu.addActions([actionExportSrcTxt, actionExportTransTxt, actionExportBothTxt, actionExportSrcMD, actionExportTransMD])
        fileMenu.addMenu(exportMenu)
        importMenu = QMenu(self.tr("Import"), self)
        actionImportDoc = QAction(self.tr("Import from Doc"), self)
        actionImportDoc.triggered.connect(leftBar.import_doc.emit)
        actionImportTransTxt = QAction(self.tr("Import translation from TXT/markdown"), self)
        actionImportTransTxt.triggered.connect(leftBar.import_trans_txt.emit)
        importMenu.addActions([actionImportDoc, actionImportTransTxt])
        fileMenu.addMenu(importMenu)
        self.fileToolBtn.setMenu(fileMenu)
        self._omni_actions_cache = None

    def apply_shortcuts(self, shortcuts_dict):
        """Apply shortcuts to title-bar menu actions without duplicating MainWindow shortcuts.

        MainWindow uses QShortcut for high-frequency canvas/navigation mode keys.
        If these QAction menu entries also own the same keys, Qt reports
        "QAction::event: Ambiguous shortcut overload" and the shortcut may do nothing.
        """
        mainwindow_owned_shortcuts = {
            "view.text_edit",
            "view.draw_board",
            "go.prev_page",
            "go.next_page",
            "go.prev_page_alt",
            "go.next_page_alt",
            "canvas.textblock_mode",
            "canvas.zoom_in",
            "canvas.zoom_out",
            "canvas.delete",
            "canvas.space",
            "canvas.select_all",
            "canvas.escape",
            "canvas.delete_line",
            "canvas.create_textbox",
            "view.context_menu_options",
            "edit.omni_search",
        }

        shortcuts_dict = shortcuts_dict or {}

        for action_id, default_key, action in self._shortcut_actions_title:
            key = shortcuts_dict.get(action_id) or default_key

            if action_id in mainwindow_owned_shortcuts:
                # QAction remains clickable from the menu, but keyboard ownership stays with MainWindow.
                action.setShortcut(QKeySequence())
                if key:
                    action.setToolTip(action.text().replace("&", "") + f" ({key})")
                continue

            action.setShortcut(QKeySequence.fromString(key) if key else QKeySequence())

    def eventFilter(self, obj, e):
        if obj == getattr(self, "omniSearch", None):
            if e.type() == QEvent.Type.KeyPress:
                if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    self._on_omni_search_return_pressed()
                    e.accept()
                    return True
                if e.key() == Qt.Key.Key_Escape:
                    self.closeOmniSearch()
                    e.accept()
                    return True

        if obj == self.mainwindow:
            if e.type() == QEvent.Type.WindowStateChange and not C.ON_MACOS:
                self.maxBtn.setChecked(self.mainwindow.isMaximized())
                return False

        return super().eventFilter(obj, e)

    def stageEnableStateChanged(self):
        sender = self.sender()
        idx= self.stageActions.index(sender)
        checked = sender.isChecked()
        self.enable_module.emit(idx, checked)

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        super().mouseDoubleClickEvent(e)
        FramelessMoveResize.toggleMaxState(self.mainwindow)

    def onMaxBtnClicked(self):
        FramelessMoveResize.toggleMaxState(self.mainwindow)

    def onMinBtnClicked(self):
        self.mainwindow.showMinimized()

    def on_displaylang_triggered(self):
        ac = self.lang_ac_group.checkedAction()
        self.display_lang_changed.emit(C.DISPLAY_LANGUAGE_MAP[ac.text()])
    

    def mousePressEvent(self, event: QMouseEvent) -> None:

        if C.FLAG_QT6:
            g_pos = event.globalPosition().toPoint()
        else:
            g_pos = event.globalPos()
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.mainwindow.isMaximized() and \
                event.pos().y() < WINDOW_BORDER_WIDTH:
                pass
            else:
                self.mPos = event.pos()
                self.mPosGlobal = g_pos
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.mPos = None
        return super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.mPos is not None:
            if C.FLAG_QT6:
                g_pos = event.globalPosition().toPoint()
            else:
                g_pos = event.globalPos()
            FramelessMoveResize.startSystemMove(self.window(), g_pos)

    def hideEvent(self, e) -> None:
        self.mPos = None
        return super().hideEvent(e)

    def leaveEvent(self, e) -> None:
        self.mPos = None
        return super().leaveEvent(e)

    def setTitleContent(self, proj_name: str = None, page_name: str = None, save_state: str = None):
        max_proj_len = 50
        max_page_len = 50
        dirty = False
        if proj_name is not None:
            if len(proj_name) > max_proj_len:
                proj_name = proj_name[:max_proj_len-3] + '...'
            self.proj_name = proj_name
        if page_name is not None:
            if len(page_name) > max_page_len:
                page_name = page_name[:max_page_len-3] + '...'
            self.page_name = page_name
        if save_state is not None:
            self.save_state = save_state
            dirty = (save_state != '')
        # Update project chip (left side, visible when project is open)
        self.projectChip.set_project(self.proj_name, dirty=dirty)
        # Keep titleLabel clean: just page name, centered
        self.titleLabel.setText(self.page_name)


class SmallConfigPutton(QPushButton):
    pass


CFG_ICON = QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'leftbar_config_activate.svg'))
CFG_ICON_NORMAL = QIcon(osp.join(C.PROGRAM_PATH, 'icons', 'leftbar_config.svg'))


class SelectionWithConfigWidget(Widget):

    cfg_clicked = Signal()

    def __init__(self, selector_name: str, add_cfg_btn=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        label = ConfigClickableLabel(text=selector_name)
        label.clicked.connect(self.cfg_clicked)
        
        self.selector = SmallComboBox()
        self.selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.selector.setMinimumContentsLength(10)
        if add_cfg_btn:
            self.cfg_btn = SmallConfigPutton()
            self.cfg_btn.setObjectName('BottomBarCfgBtn')
            self.cfg_btn.setToolTip(self.tr('Open module settings'))
            self.cfg_btn.setIconSize(QSize(18, 18))
            if osp.isfile(osp.join(C.PROGRAM_PATH, 'icons', 'leftbar_config.svg')):
                self.cfg_btn.setIcon(CFG_ICON_NORMAL)
            self.cfg_btn.clicked.connect(self.cfg_clicked)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label)
        layout2 = QHBoxLayout()
        layout2.setSpacing(0)
        layout2.addWidget(self.selector)
        layout2.addWidget(self.cfg_btn)
        layout2.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(layout2)

    def enterEvent(self, event: QEvent) -> None:
        if self.cfg_btn is not None and osp.isfile(osp.join(C.PROGRAM_PATH, 'icons', 'leftbar_config_activate.svg')):
            self.cfg_btn.setIcon(CFG_ICON)
        return super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        if self.cfg_btn is not None and osp.isfile(osp.join(C.PROGRAM_PATH, 'icons', 'leftbar_config.svg')):
            self.cfg_btn.setIcon(CFG_ICON_NORMAL)
        return super().leaveEvent(event)
    
    def blockSignals(self, block: bool):
        self.selector.blockSignals(block)
        super().blockSignals(block)
    
    def setSelectedValue(self, value: str, block_signals=True):
        if block_signals:
            self.blockSignals(True)
        self.selector.setCurrentText(value)
        if block_signals:
            self.blockSignals(False)
    

class TranslatorSelectionWidget(Widget):

    cfg_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        label = ConfigClickableLabel(text=self.tr('Translate'))
        label.clicked.connect(self.cfg_clicked)
        label_src = ConfigClickableLabel(text=self.tr('Source'))
        label_src.clicked.connect(self.cfg_clicked)
        label_tgt = ConfigClickableLabel(text=self.tr('Target'))
        label_tgt.clicked.connect(self.cfg_clicked)
        
        self.selector = SmallComboBox()
        self.selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.selector.setMinimumContentsLength(10)
        self.src_selector = SmallComboBox()
        self.src_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.src_selector.setMinimumContentsLength(8)
        self.tgt_selector = SmallComboBox()
        self.tgt_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.tgt_selector.setMinimumContentsLength(8)
        self.cfg_btn = SmallConfigPutton()
        self.cfg_btn.setObjectName('BottomBarCfgBtn')
        self.cfg_btn.setToolTip(self.tr('Open module settings'))
        self.cfg_btn.setIconSize(QSize(18, 18))
        if osp.isfile(osp.join(C.PROGRAM_PATH, 'icons', 'leftbar_config.svg')):
            self.cfg_btn.setIcon(CFG_ICON_NORMAL)
        self.cfg_btn.clicked.connect(self.cfg_clicked)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label)
        layout.addWidget(self.selector)
        layout.addWidget(label_src)
        layout.addWidget(self.src_selector)
        layout.addWidget(label_tgt)
        layout.addWidget(self.tgt_selector)
        layout.addWidget(self.cfg_btn)
        layout.setSpacing(1)

    def enterEvent(self, event: QEvent) -> None:
        if self.cfg_btn is not None and osp.isfile(osp.join(C.PROGRAM_PATH, 'icons', 'leftbar_config_activate.svg')):
            self.cfg_btn.setIcon(CFG_ICON)
        return super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        if self.cfg_btn is not None and osp.isfile(osp.join(C.PROGRAM_PATH, 'icons', 'leftbar_config.svg')):
            self.cfg_btn.setIcon(CFG_ICON_NORMAL)
        return super().leaveEvent(event)
    
    def blockSignals(self, block: bool):
        self.src_selector.blockSignals(block)
        self.tgt_selector.blockSignals(block)
        self.selector.blockSignals(block)
        super().blockSignals(block)
    
    def finishSetTranslator(self, translator: BaseTranslator):
        self.blockSignals(True)
        self.src_selector.clear()
        self.tgt_selector.clear()
        for key in translator.supported_src_list:
            self.src_selector.addItem(lang_display_label(key))
        for key in translator.supported_tgt_list:
            self.tgt_selector.addItem(lang_display_label(key))
        self.selector.setCurrentText(translator.name)
        self._set_lang_selector_by_key(self.src_selector, translator.supported_src_list, translator.lang_source)
        self._set_lang_selector_by_key(self.tgt_selector, translator.supported_tgt_list, translator.lang_target)
        self.blockSignals(False)

    def _set_lang_selector_by_key(self, combobox, key_list: list, key: str):
        try:
            idx = key_list.index(key)
            combobox.setCurrentIndex(idx)
        except (ValueError, IndexError):
            combobox.setCurrentIndex(0)


class HorizontalWheelScrollArea(QScrollArea):
    """Scroll area that maps vertical wheel to horizontal scroll (for bottom bar pipeline strip)."""
    def wheelEvent(self, event: QWheelEvent) -> None:
        hbar = self.horizontalScrollBar()
        if hbar.isVisible():
            delta = event.angleDelta().y() if hasattr(event, 'angleDelta') else event.delta()
            hbar.setValue(hbar.value() - delta)
            event.accept()
        else:
            super().wheelEvent(event)


class BottomBar(Widget):
    
    textedit_checkchanged = Signal()
    paintmode_checkchanged = Signal()
    textblock_checkchanged = Signal()
    run_clicked = Signal()
    spellcheck_checkchanged = Signal()

    def __init__(self, mainwindow: QMainWindow, *args, **kwargs) -> None:
        super().__init__(mainwindow, *args, **kwargs)
        self.setFixedHeight(BOTTOMBAR_HEIGHT)
        self.setMouseTracking(True)
        self.mainwindow = mainwindow
        
        self.textdet_selector = SelectionWithConfigWidget(self.tr('Text Detector'))
        self.ocr_selector = SelectionWithConfigWidget(self.tr('OCR'))
        self.inpaint_selector = SelectionWithConfigWidget(self.tr('Inpaint'))
        self.trans_selector = TranslatorSelectionWidget()

        self.runBtn = QPushButton(self.tr('Run'))
        self.runBtn.setObjectName('BottomBarRunBtn')
        self.runBtn.setEnabled(True)
        self.runBtn.setFixedHeight(26)
        self.runBtn.setMinimumWidth(50)
        self.runBtn.setToolTip(self.tr('Run pipeline (same as Pipeline → Run).'))
        self.runBtn.clicked.connect(self.run_clicked.emit)
        self.runBtn.setFont(sanitize_font(self.runBtn.font()))
        install_button_animations(self.runBtn, normal_opacity=0.88, press_opacity=0.72, with_scale=True)
        runBtnWrapper = QWidget(self)
        runBtnLayout = QVBoxLayout(runBtnWrapper)
        runBtnLayout.setContentsMargins(0, 3, 0, 0)
        runBtnLayout.setSpacing(0)
        runBtnLayout.addWidget(self.runBtn)
        self.runBtnWrapper = runBtnWrapper

        for w in (self.textdet_selector, self.ocr_selector, self.inpaint_selector):
            w.selector.setFont(sanitize_font(w.selector.font()))
        self.trans_selector.selector.setFont(sanitize_font(self.trans_selector.selector.font()))
        self.trans_selector.src_selector.setFont(sanitize_font(self.trans_selector.src_selector.font()))
        self.trans_selector.tgt_selector.setFont(sanitize_font(self.trans_selector.tgt_selector.font()))

        self.hlayout = QHBoxLayout(self)
        self.paintChecker = QCheckBox()
        self.paintChecker.setObjectName('PaintChecker')
        self.paintChecker.setToolTip(self.tr('Drawing board'))
        self.paintChecker.clicked.connect(self.onPaintCheckerPressed)
        self.texteditChecker = QCheckBox()
        self.texteditChecker.setObjectName('TexteditChecker')
        self.texteditChecker.setToolTip(self.tr('Text editor'))
        self.texteditChecker.clicked.connect(self.onTextEditCheckerPressed)
        self.spellCheckChecker = BottomModeToolButton(self)
        self.spellCheckChecker.setObjectName('SpellCheckChecker')

        # Keep the spell-check bottom-bar icon centered like the neighboring
        # QCheckBox indicator buttons. Without fixed button/icon metrics,
        # QToolButton style metrics can shift this icon differently in the
        # settings stack vs the normal canvas stack.
        self.spellCheckChecker.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.spellCheckChecker.setFixedSize(44, 28)
        self.spellCheckChecker.setIconSize(QSize(28, 28))
        self.spellCheckChecker.setCheckable(True)
        self.spellCheckChecker.setAutoRaise(True)
        self.spellCheckChecker.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.spellCheckChecker.setToolTip(self.tr('Spell check panel: review OCR/translation spelling issues'))
        self.spellCheckChecker.setAccessibleName(self.tr('Spell check panel'))
        self.spellCheckChecker.setAccessibleDescription(self.tr('Toggle the spell check side panel.'))
        self._set_spellcheck_icon(False)
        self.spellCheckChecker.clicked.connect(self.onSpellCheckCheckerPressed)
        self.textblockChecker = BottomModeToolButton(self)
        self.textblockChecker.setObjectName('TextblockChecker')
        self.textblockChecker.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.textblockChecker.setFixedSize(44, 28)
        self.textblockChecker.setIconSize(QSize(28, 28))
        self.textblockChecker.setCheckable(True)
        self.textblockChecker.setAutoRaise(True)
        self.textblockChecker.setToolTip(self.tr('Show text box controls'))
        self.textblockChecker.setAccessibleName(self.tr('Text box controls'))
        self.textblockChecker.setAccessibleDescription(self.tr('Toggle text box handles and controls.'))
        self._set_textblock_icon(False)
        self.textblockChecker.clicked.connect(self.onTextblockCheckerClicked)

        # Keep all bottom-bar mode icons in one fixed, centered row.
        # QCheckBox mode icons otherwise expand inside the QHBoxLayout and
        # push the QToolButton-based spell-check icon toward the textblock icon.
        self._normalize_bottom_mode_button_geometry()
        
        self.originalSlider = PaintQSlider(self.tr("Original image opacity"), Qt.Orientation.Horizontal, self)
        self.originalSlider.setFixedWidth(150)
        self.originalSlider.setRange(0, 100)

        self.textlayerSlider = PaintQSlider(self.tr("Text layer opacity"), Qt.Orientation.Horizontal, self)
        self.textlayerSlider.setFixedWidth(150)
        self.textlayerSlider.setValue(100)
        self.textlayerSlider.setRange(0, 100)
        
        # Pipeline modules in a horizontal scroll area so they don't get squished
        self.pipelineContainer = QWidget(self)
        self.pipelineContainer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        pipelineLayout = QHBoxLayout(self.pipelineContainer)
        pipelineLayout.setContentsMargins(0, 0, 0, 0)
        pipelineLayout.setSpacing(8)
        pipelineLayout.addStretch(1)
        pipelineLayout.addWidget(self.textdet_selector)
        pipelineLayout.addWidget(self.ocr_selector)
        pipelineLayout.addWidget(self.inpaint_selector)
        pipelineLayout.addWidget(self.trans_selector)
        self.pipelineScroll = HorizontalWheelScrollArea(self)
        self.pipelineScroll.setWidget(self.pipelineContainer)
        self.pipelineScroll.setWidgetResizable(True)
        self.pipelineScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.pipelineScroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.pipelineScroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.pipelineScroll.setMaximumHeight(BOTTOMBAR_HEIGHT)
        self.pipelineScroll.setMinimumWidth(120)
        self.pipelineScroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.pipelineScroll.setStyleSheet("""
            QScrollBar:horizontal {
                height: 4px;
                margin: 0;
                border: none;
                background: transparent;
            }
            QScrollBar::handle:horizontal {
                min-width: 24px;
                border-radius: 2px;
                background: rgba(128, 128, 128, 0.4);
            }
            QScrollBar::handle:horizontal:hover {
                background: rgba(128, 128, 128, 0.6);
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
        """)
        
        # Layout: Pipeline (stretches from left to Run) | Run | sliders | canvas mode
        self.hlayout.addWidget(self.pipelineScroll, 1)
        self.hlayout.addWidget(self.runBtnWrapper)
        self.hlayout.addWidget(self.textlayerSlider)
        self.hlayout.addWidget(self.originalSlider)
        self.hlayout.addWidget(self.paintChecker)
        self.hlayout.addWidget(self.texteditChecker)
        self.hlayout.addWidget(self.spellCheckChecker)
        self.hlayout.addSpacing(6)
        self.hlayout.addWidget(self.textblockChecker)
        self.hlayout.setContentsMargins(0, 0, 10, WINDOW_BORDER_WIDTH)
        self.hlayout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Hover + press animations for bottom bar controls (runBtn already has install_button_animations above)
        for chk in (self.paintChecker, self.texteditChecker, self.spellCheckChecker, self.textblockChecker):
            install_hover_opacity_animation(chk, duration_ms=100, normal_opacity=0.88, press_opacity=0.74)



    def _normalize_bottom_mode_button_geometry(self):
        # Bottom bar height is 32px; its layout keeps a 4px bottom border margin.
        # Keep every mode button inside the remaining 28px content row.
        mode_buttons = (
            getattr(self, "paintChecker", None),
            getattr(self, "texteditChecker", None),
            getattr(self, "spellCheckChecker", None),
            getattr(self, "textblockChecker", None),
        )
        for btn in mode_buttons:
            if btn is None:
                continue
            try:
                btn.setFixedSize(44, 28)
                btn.setMinimumSize(44, 28)
                btn.setMaximumSize(44, 28)
                btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                btn.setContentsMargins(0, 0, 0, 0)
            except Exception:
                pass

        for btn in (getattr(self, "spellCheckChecker", None), getattr(self, "textblockChecker", None)):
            if btn is None:
                continue
            try:
                btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
                btn.setIconSize(QSize(28, 28))
                btn.setAutoRaise(True)
            except Exception:
                pass

        try:
            self.hlayout.setSpacing(6)
            self.hlayout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        except Exception:
            pass

    def _set_spellcheck_icon(self, checked: bool = None):
        """Use a real icon for the bottom-bar spell-check mode button.

        The other bottom-bar mode buttons are primarily styled by objectName in
        QSS.  SpellCheckChecker was added later and may not have a matching QSS
        image rule in older themes, so give it an explicit SVG fallback here.
        """
        if checked is None:
            checked = bool(self.spellCheckChecker.isChecked())

        icon_name = 'bottombar_spellcheck_activate.svg' if checked else 'bottombar_spellcheck.svg'
        icon_path = osp.join(C.PROGRAM_PATH, 'icons', icon_name)

        # Fall back to the title-bar spell-check icon if the new bottombar icon
        # was not copied yet. This keeps older checkouts usable.
        if not osp.isfile(icon_path):
            icon_path = osp.join(C.PROGRAM_PATH, 'icons', 'search-stop.svg')

        if osp.isfile(icon_path):
            self.spellCheckChecker.setIcon(QIcon(icon_path))
            # Match the visual weight of the neighboring bottom-bar checkbox icons.
            self.spellCheckChecker.setIconSize(QSize(28, 28))


    def _set_textblock_icon(self, checked: bool = None):
        """Use the same fixed-size button/icon path as SpellCheckChecker.

        TextblockChecker used to be a QCheckBox painted through ::indicator.
        That has different size/baseline metrics from QToolButton, which made
        the textbox icon sit lower than the spell-check icon.
        """
        if checked is None:
            checked = bool(self.textblockChecker.isChecked())

        icon_name = 'bottombar_textblock_activate.svg' if checked else 'bottombar_textblock.svg'
        icon_path = osp.join(C.PROGRAM_PATH, 'icons', icon_name)

        if osp.isfile(icon_path):
            self.textblockChecker.setIcon(QIcon(icon_path))
            self.textblockChecker.setIconSize(QSize(28, 28))

    def onPaintCheckerPressed(self):
        checked = self.paintChecker.isChecked()
        if checked:
            self.texteditChecker.setChecked(False)
            self.spellCheckChecker.setChecked(False)
            self._set_spellcheck_icon(False)
        pcfg.imgtrans_paintmode = checked
        self.paintmode_checkchanged.emit()

    def onTextEditCheckerPressed(self):
        checked = self.texteditChecker.isChecked()
        if checked:
            self.paintChecker.setChecked(False)
            self.spellCheckChecker.setChecked(False)
            self._set_spellcheck_icon(False)
        pcfg.imgtrans_textedit = checked
        self.textedit_checkchanged.emit()

    def onSpellCheckCheckerPressed(self):
        checked = self.spellCheckChecker.isChecked()
        if checked:
            self.paintChecker.setChecked(False)
            self.texteditChecker.setChecked(False)
        self._set_spellcheck_icon(checked)
        self.spellcheck_checkchanged.emit()

    def setPipelineVisible(self, visible: bool):
        """Show or hide pipeline strip. When showing, respect each stage's enable flag (Issue #18)."""
        if not visible:
            self.pipelineScroll.setVisible(False)
            self.textdet_selector.setVisible(False)
            self.ocr_selector.setVisible(False)
            self.inpaint_selector.setVisible(False)
            self.trans_selector.setVisible(False)
            self.runBtnWrapper.setVisible(False)
        else:
            self.pipelineScroll.setVisible(True)
            self.textdet_selector.setVisible(pcfg.module.enable_detect)
            self.ocr_selector.setVisible(pcfg.module.enable_ocr)
            self.trans_selector.setVisible(pcfg.module.enable_translate)
            self.inpaint_selector.setVisible(pcfg.module.enable_inpaint)
            self.runBtnWrapper.setVisible(True)
            self.runBtn.setEnabled(True)

    def onTextblockCheckerClicked(self):
        self._set_textblock_icon(self.textblockChecker.isChecked())
        self.textblock_checkchanged.emit()
    def _refresh_run_button_breathe_animation(self):
        icon_sz = self.runBtn.iconSize()
        if icon_sz.width() <= 0 or icon_sz.height() <= 0:
            icon_sz = QSize(20, 20)
            self.runBtn.setIconSize(icon_sz)
        up_sz = QSize(icon_sz.width() + 2, icon_sz.height() + 2)
        self._run_btn_pulse_up.setStartValue(icon_sz)
        self._run_btn_pulse_up.setEndValue(up_sz)
        self._run_btn_pulse_down.setStartValue(up_sz)
        self._run_btn_pulse_down.setEndValue(icon_sz)
