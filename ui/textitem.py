import math, re
import os.path as osp
import numpy as np
import cv2
from typing import List, Union, Tuple

from qtpy.QtWidgets import QGraphicsItem, QWidget, QGraphicsSceneHoverEvent, QGraphicsTextItem, QStyleOptionGraphicsItem, QStyle, QGraphicsSceneMouseEvent
from qtpy.QtCore import Qt, QRect, QRectF, QPointF, Signal, QSizeF
from qtpy.QtGui import (QGradient, QKeyEvent, QFont, QTextCursor, QPixmap, QPainterPath, QTextDocument, 
                       QInputMethodEvent, QPainter, QPen, QColor, QTextCharFormat, QTextDocument, QLinearGradient, QRadialGradient,
                       QBrush, QPalette, QAbstractTextDocumentLayout, QTextFrameFormat, QImage, QBitmap, QRegion, QTransform, QPolygonF)

from utils.textblock import TextBlock, FontFormat, TextAlignment, LineSpacingType
from utils.text_layout import _merge_stub_lines_in_string
from utils.imgproc_utils import xywh2xyxypoly, rotate_polygons
from utils.fontformat import FontFormat, px2pt, pt2px
from utils.config import pcfg
from utils.text_rendering import estimate_text_bounds, first_missing_glyphs, font_fallback_runs, merge_font_fallback_chain, missing_glyphs_after_fallback, resolve_writing_mode
from utils.text_masking import mask_safe_rect
from .misc import td_pattern, table_pattern, pixmap2ndarray, ndarray2pixmap
from .image_edit import ImageEditMode
from .scene_textlayout import VerticalTextDocumentLayout, HorizontalTextDocumentLayout, SceneTextLayout
from .text_graphical_effect import apply_shadow_effect

TEXTRECT_SHOW_COLOR = QColor(30, 147, 229, 170)
TEXTRECT_SELECTED_COLOR = QColor(248, 64, 147, 170)

# Blend mode (fontformat.blend_mode) to QPainter.CompositionMode
def _blend_mode_qt(mode: int):
    modes = [
        QPainter.CompositionMode.CompositionMode_SourceOver,
        getattr(QPainter.CompositionMode, 'CompositionMode_Multiply', QPainter.CompositionMode.CompositionMode_SourceOver),
        getattr(QPainter.CompositionMode, 'CompositionMode_Screen', QPainter.CompositionMode.CompositionMode_SourceOver),
        getattr(QPainter.CompositionMode, 'CompositionMode_Overlay', QPainter.CompositionMode.CompositionMode_SourceOver),
        getattr(QPainter.CompositionMode, 'CompositionMode_Darken', QPainter.CompositionMode.CompositionMode_SourceOver),
        getattr(QPainter.CompositionMode, 'CompositionMode_Lighten', QPainter.CompositionMode.CompositionMode_SourceOver),
    ]
    return modes[mode] if 0 <= mode < len(modes) else modes[0]


class TextBlkItem(QGraphicsTextItem):

    begin_edit = Signal(int)
    end_edit = Signal(int)
    hover_enter = Signal(int)
    hover_move = Signal(int)
    moved = Signal()
    moving = Signal(QGraphicsTextItem)
    rotated = Signal(float)
    reshaped = Signal(QGraphicsTextItem)
    leftbutton_pressed = Signal(int)
    doc_size_changed = Signal(int)
    pasted = Signal(int)
    redo_signal = Signal()
    undo_signal = Signal()
    push_undo_stack = Signal(int, bool)
    propagate_user_edited = Signal(int, str, bool)
    warped = Signal(object, object)  # PR #1105: (before_state_dict, after_state_dict) when quad/mesh warp changes

    def __init__(self, blk: TextBlock = None, idx: int = 0, set_format=True, show_rect=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_editing = False
        self.blk: TextBlock = None
        self.fontformat: FontFormat = None
        self.repainting = False
        self.reshaping = False
        self.under_ctrl = False
        self.draw_rect = show_rect
        self._display_rect: QRectF = QRectF(0, 0, 1, 1)
        self.old_ffmt_values = None
        
        self.idx = idx
        
        self.background_pixmap: QPixmap = None
        self.stroke_qcolor = QColor(0, 0, 0)
        self.oldPos = QPointF()
        self.oldRect = QRectF()
        self.repaint_on_changed = True

        self.is_formatting = False
        self.old_undo_steps = 0
        self.in_redo_undo = False
        self.change_from: int = 0
        self.change_added: int = 0
        self.input_method_from = -1
        self.input_method_text = ''
        self.block_all_input = False
        self.block_change_signal = False

        self.layout: Union[VerticalTextDocumentLayout, HorizontalTextDocumentLayout] = None
        self.document().setDocumentMargin(0)
        self._ensure_transparent_document_background()
        self.initTextBlock(blk, set_format=set_format)
        self.setBoundingRegionGranularity(0)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self._foreground_pixmap: QPixmap = None  # PR #1070 image overlay cache

    def inputMethodEvent(self, e: QInputMethodEvent):
        if self.pre_editing == False:
            cursor = self.textCursor()
            self.input_method_from = cursor.selectionStart()
        if e.preeditString() == '':
            self.pre_editing = False
            self.input_method_text = e.commitString()
        else:
            self.pre_editing = True
        super().inputMethodEvent(e)
        
    def is_editting(self):
        return self.textInteractionFlags() == Qt.TextInteractionFlag.TextEditorInteraction

    def _ensure_transparent_document_background(self):
        """Force document and root frame to transparent so no gray box appears behind text."""
        doc = self.document()
        doc.setDefaultStyleSheet("body, html { background-color: transparent; }")
        try:
            root = doc.rootFrame()
            fmt = root.frameFormat()
            fmt.setBackground(QBrush(Qt.GlobalColor.transparent))
            root.setFrameFormat(fmt)
        except Exception:
            pass

    def on_content_changed(self):
        if (self.hasFocus() or self.is_formatting) and not self.pre_editing and not self.block_change_signal:   
            # self.content_changed.emit(self)
            if not self.in_redo_undo:
                undo_steps = self.document().availableUndoSteps()
                new_steps = undo_steps - self.old_undo_steps
                joint_previous = new_steps == 0

                if not self.is_formatting:
                    change_from = self.change_from
                    added_text = ''
                    if self.input_method_from != -1:
                        added_text = self.input_method_text
                        change_from = self.input_method_from
                        self.input_method_from = -1

                    elif self.change_added > 0:
                        cursor = self.textCursor()
                        cursor.setPosition(change_from)
                        cursor.setPosition(change_from + self.change_added, QTextCursor.MoveMode.KeepAnchor)
                        added_text = cursor.selectedText()

                    self.propagate_user_edited.emit(change_from, added_text, joint_previous)
                    self.change_added = 0

                if new_steps > 0:
                    self.old_undo_steps = undo_steps
                    self.push_undo_stack.emit(new_steps, self.is_formatting)

        if not (self.hasFocus() and self.pre_editing):
            if self.repaint_on_changed:
                if not self.repainting:
                    self.repaint_background()
            self.update()

    def paint_stroke(self, painter: QPainter, stroke_width: float = None, stroke_color=None):
        doc = QTextDocument()
        doc.setUndoRedoEnabled(False)
        doc.setDocumentMargin(self.document().documentMargin())
        doc.setDefaultStyleSheet("body, html { background-color: transparent; }")
        try:
            root = doc.rootFrame()
            fmt = root.frameFormat()
            fmt.setBackground(QBrush(Qt.GlobalColor.transparent))
            root.setFrameFormat(fmt)
        except Exception:
            pass
        default_font = self.document().defaultFont()
        if default_font.pointSizeF() <= 0:
            default_font.setPointSizeF(1.0)
        doc.setDefaultFont(default_font)
        doc.setHtml(self.document().toHtml())
        doc.setDefaultTextOption(self.document().defaultTextOption())
        cursor = QTextCursor(doc)
        block = doc.firstBlock()
        stroke_qcolor = QColor(*(stroke_color or self.fontformat.stroke_color())) if stroke_color is not None else self.stroke_qcolor
        stroke_pen = QPen(stroke_qcolor, 0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        letter_spacing = self.fontformat.letter_spacing * 100
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                cfmt = fragment.charFormat()
                sw = pt2px(cfmt.fontPointSize()) * (self.fontformat.stroke_width if stroke_width is None else float(stroke_width or 0.0))
                if getattr(self.fontformat, 'stroke_outline_outside_only', False):
                    sw = sw * 2  # fill drawn on top covers inner half → stroke only outside (EasyScanlate-style)
                stroke_pen.setWidthF(sw)
                pos1 = fragment.position()
                pos2 = pos1 + fragment.length()
                cursor.setPosition(pos1)
                cursor.setPosition(pos2, QTextCursor.MoveMode.KeepAnchor)
                cfmt.setTextOutline(stroke_pen)
                if letter_spacing != 100 and not self.fontformat.vertical:
                    cfmt.setFontLetterSpacingType(QFont.SpacingType.PercentageSpacing)
                    cfmt.setFontLetterSpacing(letter_spacing)
                cursor.mergeCharFormat(cfmt)
                it += 1
            block = block.next()

        layout = VerticalTextDocumentLayout(doc, self.fontformat) if self.fontformat.vertical \
            else HorizontalTextDocumentLayout(doc, self.fontformat)
        layout._draw_offset = self.layout._draw_offset
        layout._is_painting_stroke = True
        layout.setMaxSize(self.layout.max_width, self.layout.max_height, False)
        doc.setDocumentLayout(layout)

        layout.relayout_on_changed = False
        doc.drawContents(painter)

    def _paint_text_on_path(self, painter: QPainter, draw_stroke: bool = False, draw_fill: bool = True, stroke_width: float = None, stroke_color=None) -> None:
        """Draw text along a circular or arc path (for text_on_path fontformat)."""
        ff = self.fontformat
        if ff.text_on_path == 0:
            return
        doc = self.document()
        text = doc.toPlainText().strip()
        if not text:
            return
        br = self.boundingRect()
        w, h = br.width(), br.height()
        cx, cy = br.center().x(), br.center().y()
        r = min(w, h) / 2.0 * 0.85
        if r <= 0:
            return
        font = doc.defaultFont()
        if font.pointSizeF() <= 0:
            font.setPointSizeF(max(1.0, pt2px(ff.font_size)))
        fg = QColor(*ff.foreground_color())
        stroke_pen = QPen(QColor(*(stroke_color or ff.stroke_color())), 0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        active_width = ff.stroke_width if stroke_width is None else float(stroke_width or 0.0)
        stroke_px = pt2px(font.pointSizeF()) * active_width if draw_stroke and active_width > 0 else 0
        if stroke_px > 0 and getattr(ff, 'stroke_outline_outside_only', False):
            stroke_px = stroke_px * 2  # fill drawn on top → stroke only outside (EasyScanlate-style)
        if stroke_px > 0:
            stroke_pen.setWidthF(stroke_px)

        if ff.text_on_path == 1:
            # Full circle: start at bottom (270°), clockwise
            start_deg = 270.0
            arc_span_deg = 360.0
        else:
            # Arc: top semicircle; arc_degrees from fontformat
            arc_span_deg = max(30.0, min(360.0, ff.text_on_path_arc_degrees))
            start_deg = 180.0 - (arc_span_deg / 2.0)

        n = len(text)
        angle_step = arc_span_deg / n if n else 0
        to_rad = math.pi / 180.0

        for i, ch in enumerate(text):
            angle_deg = start_deg + (i + 0.5) * angle_step
            angle_rad = angle_deg * to_rad
            x = cx + r * math.cos(angle_rad)
            y = cy + r * math.sin(angle_rad)
            rotation = angle_deg + 90.0

            painter.save()
            painter.translate(x, y)
            painter.rotate(rotation)
            painter.setFont(font)

            if draw_stroke and stroke_px > 0:
                per_char = getattr(ff, 'stroke_rgb_per_char', None)
                if per_char and i < len(per_char) and len(per_char[i]) >= 3:
                    stroke_pen.setColor(QColor(per_char[i][0], per_char[i][1], per_char[i][2]))
                painter.setPen(stroke_pen)
                painter.setBrush(Qt.GlobalColor.transparent)
                painter.drawText(0, 0, ch)
            if draw_fill:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(fg)
                painter.drawText(0, 0, ch)
            painter.restore()

    def repaint_background(self):
        empty = self.document().isEmpty()
        if self.repainting:
            return

        use_path = getattr(self.fontformat, 'text_on_path', 0) != 0
        secondary_stroke_width = max(0.0, float(getattr(self.fontformat, "secondary_stroke_width", 0.0) or 0.0))
        paint_secondary_stroke = secondary_stroke_width > 0
        paint_stroke = self.fontformat.stroke_width > 0
        paint_shadow = self.fontformat.shadow_radius > 0 and self.fontformat.shadow_strength > 0
        warp_style = getattr(self.fontformat, 'warp_style', 0)
        perspective_x = float(getattr(self.fontformat, 'perspective_x', 0.0) or 0.0)
        perspective_y = float(getattr(self.fontformat, 'perspective_y', 0.0) or 0.0)
        extrusion_depth = max(0.0, float(getattr(self.fontformat, 'extrusion_depth', 0.0) or 0.0))
        use_3d = abs(perspective_x) > 1e-4 or abs(perspective_y) > 1e-4 or extrusion_depth > 1e-4

        if (not use_path and not paint_shadow and not paint_stroke and not paint_secondary_stroke and not warp_style and not use_3d) or empty:
            self.background_pixmap = None
            return

        self.repainting = True
        try:
            font_size = self.layout.max_font_size(to_px=True)
            br = self.boundingRect()
            br_size = br.size().toSize()
            if br_size.width() < 1 or br_size.height() < 1:
                self.background_pixmap = None
                return

            # Issue #105 fix:
            # boundingRect() may have a negative top-left because stroke/shadow
            # expands the item bounds. _draw_accessories() later draws this
            # pixmap at br.toRect(), so any document-space drawing inside the
            # pixmap must be translated by -br.topLeft().
            #
            # Also, for normal text, do not draw document fill into the visible
            # effect pixmap. super().paint(...) already draws the QTextDocument
            # fill. Drawing fill here too creates the visible duplicated /
            # ghosted translation. We still draw a hidden silhouette map for
            # shadow generation.
            draw_offset = -br.topLeft()
            outline_only = getattr(self.fontformat, 'outline_only', False)

            effect_map = QPixmap(br_size)
            effect_map.fill(Qt.GlobalColor.transparent)
            effect_painter = QPainter(effect_map)
            effect_painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            effect_painter.translate(draw_offset)

            silhouette_map = QPixmap(br_size)
            silhouette_map.fill(Qt.GlobalColor.transparent)
            silhouette_painter = QPainter(silhouette_map)
            silhouette_painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            silhouette_painter.translate(draw_offset)

            try:
                if use_path:
                    # Text-on-path is not drawn by the base QGraphicsTextItem
                    # paint path, so keep both fill and effects in this cached
                    # layer for path text only.
                    if paint_secondary_stroke:
                        self._paint_text_on_path(
                            effect_painter,
                            draw_stroke=True,
                            draw_fill=False,
                            stroke_width=secondary_stroke_width,
                            stroke_color=getattr(self.fontformat, "secondary_srgb", [255, 255, 255]),
                        )
                    self._paint_text_on_path(
                        effect_painter,
                        draw_stroke=paint_stroke,
                        draw_fill=not outline_only,
                    )
                    self._paint_text_on_path(
                        silhouette_painter,
                        draw_stroke=paint_stroke or paint_secondary_stroke,
                        draw_fill=True,
                        stroke_width=max(float(getattr(self.fontformat, "stroke_width", 0.0) or 0.0), secondary_stroke_width),
                    )
                else:
                    if paint_secondary_stroke:
                        self.paint_stroke(effect_painter, stroke_width=secondary_stroke_width, stroke_color=getattr(self.fontformat, "secondary_srgb", [255, 255, 255]))
                        self.paint_stroke(silhouette_painter, stroke_width=secondary_stroke_width)
                    if paint_stroke:
                        self.paint_stroke(effect_painter)
                        self.paint_stroke(silhouette_painter)

                    if not outline_only:
                        # Shadow silhouette only. Do not draw this into
                        # effect_map, otherwise normal text is painted twice.
                        self.document().drawContents(silhouette_painter)
            finally:
                silhouette_painter.end()

            if paint_shadow:
                r = int(round(self.fontformat.shadow_radius * font_size))
                xoffset = int(self.fontformat.shadow_offset[0] * font_size)
                yoffset = int(self.fontformat.shadow_offset[1] * font_size)
                shadow_map, _img_array = apply_shadow_effect(
                    silhouette_map,
                    self.fontformat.shadow_color,
                    self.fontformat.shadow_strength,
                    r,
                )
                cm = effect_painter.compositionMode()
                effect_painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOver)
                effect_painter.drawPixmap(xoffset, yoffset, shadow_map)
                effect_painter.setCompositionMode(cm)

            effect_painter.end()

            if warp_style and effect_map.width() > 2 and effect_map.height() > 2:
                effect_map = self._apply_warp_preset(
                    effect_map,
                    warp_style,
                    getattr(self.fontformat, 'warp_strength', 0.5),
                )
            if use_3d and effect_map.width() > 2 and effect_map.height() > 2:
                effect_map = self._apply_3d_text_effect(effect_map, perspective_x, perspective_y, extrusion_depth)

            self.background_pixmap = effect_map
        finally:
            self.repainting = False

    def _apply_3d_text_effect(self, src: QPixmap, perspective_x: float, perspective_y: float, extrusion_depth: float) -> QPixmap:
        w, h = src.width(), src.height()
        if w < 2 or h < 2:
            return src
        font_px = max(1.0, self.layout.max_font_size(to_px=True))
        depth_px = int(round(max(0.0, extrusion_depth) * font_px * 0.7))
        shift_x = float(perspective_x) * w * 0.18
        shift_y = float(perspective_y) * h * 0.18
        canvas_pad = int(max(4, abs(shift_x), abs(shift_y), depth_px) + 4)
        out = QPixmap(w + canvas_pad * 2, h + canvas_pad * 2)
        out.fill(Qt.GlobalColor.transparent)
        painter = QPainter(out)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        base_pos = QPointF(canvas_pad, canvas_pad)
        if depth_px > 0:
            alpha_step = max(8, int(80 / max(1, depth_px)))
            for i in range(depth_px, 0, -1):
                t = i / float(max(1, depth_px))
                layer = self._warp_pixmap(src, shift_x * t, shift_y * t)
                painter.setOpacity(min(0.8, alpha_step * i / 255.0))
                painter.drawPixmap(base_pos + QPointF(t * shift_x, t * shift_y), layer)
            painter.setOpacity(1.0)
        final = self._warp_pixmap(src, shift_x, shift_y)
        painter.drawPixmap(base_pos, final)
        painter.end()
        return out

    def _warp_pixmap(self, src: QPixmap, shift_x: float, shift_y: float) -> QPixmap:
        w, h = src.width(), src.height()
        src_poly = QPolygonF([QPointF(0, 0), QPointF(w, 0), QPointF(w, h), QPointF(0, h)])
        dst_poly = QPolygonF([
            QPointF(shift_x, shift_y),
            QPointF(w - shift_x, -shift_y),
            QPointF(w + shift_x, h + shift_y),
            QPointF(-shift_x, h - shift_y),
        ])
        transform = QTransform()
        ok = QTransform.quadToQuad(src_poly, dst_poly, transform)
        if not ok:
            return src
        out = QPixmap(w, h)
        out.fill(Qt.GlobalColor.transparent)
        p = QPainter(out)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setTransform(transform, False)
        p.drawPixmap(0, 0, src)
        p.end()
        return out
    def docSizeChanged(self):
        self.setCenterTransform()
        self.doc_size_changed.emit(self.idx)

    def initTextBlock(self, blk: TextBlock = None, set_format=True):
        self.blk = blk
        self.fontformat = blk.fontformat
        if blk is not None:
            self.fontformat.overlay_opacity = getattr(blk, 'overlay_opacity', 1.0)
            self.fontformat.skew_x = getattr(blk, 'skew_x', 0.0)
            self.fontformat.skew_y = getattr(blk, 'skew_y', 0.0)
        if blk is None:
            xyxy = [0, 0, 0, 0]
            blk = TextBlock(xyxy)
            blk.lines = [xyxy]
            bx1, by1, bx2, by2 = xyxy
            xywh = np.array([[bx1, by1, bx2-bx1, by2-by1]])
            blk.lines = xywh2xyxypoly(xywh).reshape(-1, 4, 2).tolist()
        self.setVertical(blk.vertical)
        self.setRect(blk.bounding_rect())
        
        if blk.angle != 0:
            self.setRotation(blk.angle)
        
        set_char_fmt = False
        if blk.translation:
            set_char_fmt = True

        font_fmt = blk.fontformat
        if set_format:
            self.set_fontformat(font_fmt, set_char_format=set_char_fmt, set_stroke_width=False, set_effect=False)

        if not blk.rich_text:
            if blk.translation:
                self.setPlainText(_merge_stub_lines_in_string(blk.translation))
        else:
            self.setHtml(blk.rich_text)
            self.setLetterSpacing(font_fmt.letter_spacing, repaint_background=False)
            cursor = self.textCursor()
            cursor.clearSelection()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            cfmt = cursor.charFormat()
            cursor.setCharFormat(cfmt)
            cursor.setBlockCharFormat(cfmt)
            self.setTextCursor(cursor)
        if self.fontformat.gradient_enabled:
            self.setGradientEnabled(True)
        self.setShadow(font_fmt, repaint=False)
        self.setStrokeWidth(font_fmt.stroke_width, repaint_background=False)
        self._ensure_transparent_document_background()
        self.repaint_background()

    def setCenterTransform(self):
        center = self.boundingRect().center()
        self.setTransformOriginPoint(center)

    def rect(self) -> QRectF:
        return QRectF(self.pos(), self.boundingRect().size())

    def startReshape(self):
        self.oldRect = self.absBoundingRect(qrect=True)
        self.reshaping = True

    def endReshape(self):
        self.reshaped.emit(self)
        self.reshaping = False

    def padRect(self, rect: QRectF) -> QRectF:
        p = self.padding()
        P = p * 2
        return QRectF(rect.x() - p, rect.y() - p, rect.width() + P, rect.height() + P)
    
    def unpadRect(self, rect: QRectF) -> QRectF:
        p = -self.padding()
        P = p * 2
        return QRectF(rect.x() - p, rect.y() - p, rect.width() + P, rect.height() + P)

    def _box_outline_rect_and_radius(self) -> Tuple[QRectF, float]:
        """Return (unpadded outline rect, corner radius in item coords). Radius 0 = rectangle."""
        br = self.boundingRect()
        r = self.unpadRect(br)
        radius_px = max(0.0, getattr(self.fontformat, 'text_box_corner_radius', 0.0) or 0.0)
        if radius_px <= 0:
            return r, 0.0
        scale = self.get_scale()
        if scale <= 0:
            scale = 1.0
        radius_item = radius_px / scale
        radius_item = min(radius_item, r.width() / 2, r.height() / 2)
        return r, max(0.0, radius_item)

    def _box_outline_path(self) -> QPainterPath:
        """Return a QPainterPath for the text box outline so the drawn shape can match the balloon (ellipse, diamond, etc.)."""
        path = QPainterPath()
        outline_rect, radius = self._box_outline_rect_and_radius()
        w = outline_rect.width()
        h = outline_rect.height()
        shape = (getattr(self.fontformat, "text_box_shape", None) or "").strip().lower()
        if shape in ("elongated", "narrow", "ellipse"):
            path.addEllipse(outline_rect)
        elif shape == "round":
            r = min(w, h) / 2
            path.addRoundedRect(outline_rect, r, r)
        elif shape == "diamond" or shape == "point":
            cx, cy = outline_rect.center().x(), outline_rect.center().y()
            top = QPointF(cx, outline_rect.top())
            right = QPointF(outline_rect.right(), cy)
            bottom = QPointF(cx, outline_rect.bottom())
            left = QPointF(outline_rect.left(), cy)
            path.moveTo(top)
            path.lineTo(right)
            path.lineTo(bottom)
            path.lineTo(left)
            path.closeSubpath()
        elif shape == "pentagon":
            cx, cy = outline_rect.center().x(), outline_rect.center().y()
            top = QPointF(cx, outline_rect.top())
            # Five points: top, top-right, bottom-right, bottom-left, top-left (inset)
            path.moveTo(top)
            path.lineTo(outline_rect.right() - w * 0.1, outline_rect.top() + h * 0.35)
            path.lineTo(outline_rect.right() - w * 0.2, outline_rect.bottom())
            path.lineTo(outline_rect.left() + w * 0.2, outline_rect.bottom())
            path.lineTo(outline_rect.left() + w * 0.1, outline_rect.top() + h * 0.35)
            path.closeSubpath()
        elif shape == "square":
            path.addRect(outline_rect)
        elif shape == "bevel":
            r = min(w, h) * 0.2
            path.addRoundedRect(outline_rect, r, r)
        else:
            # rectangle, auto, or empty: use corner radius if set
            if radius > 0:
                path.addRoundedRect(outline_rect, radius, radius)
            else:
                path.addRect(outline_rect)
        return path

    def setRect(self, rect: Union[List, QRectF], padding=True, repaint=True) -> None:

        def _xywh_to_xyxy(r):
            if isinstance(r, QRectF):
                x, y, w, h = r.x(), r.y(), r.width(), r.height()
            else:
                x, y, w, h = r[0], r[1], r[2], r[3]
            return [x, y, x + w, y + h]

        if isinstance(rect, List):
            rect = QRectF(*rect)
        if padding:
            rect = self.padRect(rect)
        self.setPos(rect.topLeft())
        self.prepareGeometryChange()
        self._display_rect = rect
        self.layout.setMaxSize(rect.width(), rect.height())
        self.setCenterTransform()
        self.blk.xyxy = _xywh_to_xyxy(rect)
        if repaint:
            self.repaint_background()

    def documentSize(self):
        return self.layout.documentSize()

    def boundingRect(self) -> QRectF:
        br = super().boundingRect()
        if self._display_rect is not None:
            br.setHeight(self._display_rect.height())
            br.setWidth(self._display_rect.width())
        if self.fontformat is not None:
            try:
                fs = self.layout.max_font_size(to_px=True) if self.layout is not None else pt2px(self.document().defaultFont().pointSizeF())
                stroke_pad = fs * max(0.0, float(getattr(self.fontformat, 'stroke_width', 0.0) or 0.0))
                shadow_pad = fs * max(0.0, float(getattr(self.fontformat, 'shadow_radius', 0.0) or 0.0))
                off = getattr(self.fontformat, 'shadow_offset', [0.0, 0.0]) or [0.0, 0.0]
                shadow_pad += fs * max(abs(float(off[0] if len(off) > 0 else 0.0)), abs(float(off[1] if len(off) > 1 else 0.0)))
                pad = max(0.0, stroke_pad + shadow_pad)
                if pad > 0:
                    br = br.adjusted(-pad, -pad, pad, pad)
            except Exception:
                pass
        return br

    def padding(self) -> float:
        return self.document().documentMargin()

    def setPadding(self, p: float):
        _p = self.padding()
        if _p >= p:
            return
        abr = self.absBoundingRect(qrect=True)
        self.layout.relayout_on_changed = False
        self.layout.updateDocumentMargin(p)
        self.layout.relayout_on_changed = True
        self.setRect(abr, repaint=False)

    def absBoundingRect(self, max_h=None, max_w=None, qrect=False) -> Union[List, QRectF]:
        br = self.boundingRect()
        P = 2 * self.padding()
        w, h = br.width() - P, br.height() - P
        pos = self.pos()
        x = pos.x() + self.padding()
        y = pos.y() + self.padding()
        if max_h is not None:
            y = min(max(0, y), max_h)
            y1 = y + h
            h = min(max_h, y1) - y
        if max_w is not None:
            x = min(max(0, x), max_w)
            x1 = x + w
            w = min(max_w, x1) - x
        if qrect:
            return QRectF(x, y, w, h)
        return [int(round(x)), int(round(y)), math.ceil(w), math.ceil(h)]

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        br = self.boundingRect()
        path.addRect(br)
        return path

    def setScale(self, scale: float) -> None:
        self.setTransformOriginPoint(0, 0)
        super().setScale(scale)
        self.setCenterTransform()

    @property
    def angle(self) -> int:
        return self.blk.angle

    def toTextBlock(self) -> TextBlock:
        raise NotImplementedError

    def setAngle(self, angle: int):
        self.setCenterTransform()
        if self.blk.angle != angle:
            self.setRotation(angle)
        self.blk.angle = angle

    def setVertical(self, vertical: bool):

        # Switching layout drops the text interaction state; remember it so the user
        # keeps editing at the same position (upstream dmMaze/BallonsTranslator@cdf5c6f).
        is_editing = self.is_editting()
        cursor_pos = None
        if is_editing:
            cursor = self.textCursor()
            cursor_pos = (cursor.position(), cursor.anchor())

        if self.fontformat is not None:
            self.fontformat.vertical = vertical

        valid_layout = True
        doc = self.document()
        if self.layout is not None:
            if isinstance(self.layout, VerticalTextDocumentLayout) == vertical:
                return
            self.layout.size_enlarged.disconnect(self.on_document_enlarged)
            self.layout.documentSizeChanged.disconnect(self.docSizeChanged)
        else:
            valid_layout = False
            doc.contentsChanged.connect(self.on_content_changed)
            doc.contentsChange.connect(self.on_content_changing)

        if valid_layout:
            rect = self.rect() if self.layout is not None else None
        
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        doc.documentLayout().blockSignals(True)
        if vertical:
            layout = VerticalTextDocumentLayout(doc, self.fontformat)
        else:
            layout = HorizontalTextDocumentLayout(doc, self.fontformat)

        # Letter spacing must be reset for the other layout type: the vertical layout
        # applies it itself, the horizontal one takes it from the document format
        # (upstream dmMaze/BallonsTranslator@64a7365).
        reset_spacing_val = 1 if vertical else self.fontformat.letter_spacing
        cursor = QTextCursor(doc)
        cursor.joinPreviousEditBlock()
        char_fmt = QTextCharFormat()
        char_fmt.setFontLetterSpacingType(QFont.SpacingType.PercentageSpacing)
        char_fmt.setFontLetterSpacing(reset_spacing_val * 100)
        cursor.select(QTextCursor.SelectionType.Document)
        self.set_cursor_cfmt(cursor, char_fmt, True)
        cursor.endEditBlock()

        self.layout = layout
        doc.setDocumentLayout(layout)
        layout.size_enlarged.connect(self.on_document_enlarged)
        layout.documentSizeChanged.connect(self.docSizeChanged)

        if valid_layout:
            layout.setMaxSize(rect.width(), rect.height())
            self.setCenterTransform()
            self.repaint_background()
        self.doc_size_changed.emit(self.idx)

        if is_editing:
            self.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
            self.setFocus()
            cursor = QTextCursor(doc)
            pos1, pos2 = cursor_pos
            cursor.setPosition(min(pos1, pos2))
            cursor.setPosition(max(pos1, pos2), QTextCursor.MoveMode.KeepAnchor)
            self.setTextCursor(cursor)

    def updateUndoSteps(self):
        self.old_undo_steps = self.document().availableUndoSteps()

    def on_content_changing(self, from_: int, removed: int, added: int):
        if not self.pre_editing:
            if self.hasFocus():
                self.change_from = from_
                self.change_added = added

    def keyPressEvent(self, e: QKeyEvent) -> None:

        if self.block_all_input:
            e.setAccepted(True)
            return

        if e.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if e.key() == Qt.Key.Key_Z:
                e.accept()
                self.undo_signal.emit()
                return
            elif e.key() == Qt.Key.Key_Y:
                e.accept()
                self.redo_signal.emit()
                return
            elif e.key() == Qt.Key.Key_V:
                if self.isEditing():
                    e.accept()
                    self.pasted.emit(self.idx)
                    return
        elif e.modifiers() == Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier:
            if e.key() == Qt.Key.Key_Z:
                e.accept()
                self.redo_signal.emit()
                return
        elif e.key() == Qt.Key.Key_Return:
            e.accept()
            self.textCursor().insertText('\n')
            return
        return super().keyPressEvent(e)

    def undo(self) -> None:
        self.in_redo_undo = True
        self.document().undo()
        self.in_redo_undo = False
        self.old_undo_steps = self.document().availableUndoSteps()

    def redo(self) -> None:
        self.in_redo_undo = True
        self.document().redo()
        self.in_redo_undo = False
        self.old_undo_steps = self.document().availableUndoSteps()

    def on_document_enlarged(self):
        """Layout wants to change size. Cap width at 1.4x and height at 1.3x; when cut-off, only allow width to grow (no full height)."""
        size = self.documentSize()
        cw = self._display_rect.width()
        ch = self._display_rect.height()
        nw = size.width()
        nh = size.height()
        # Cap width growth at 1.4x; never shrink (slightly wider lines to reduce vertical overflow)
        width_cap = cw * 1.4 if cw > 0 else nw
        use_w = max(nw, cw)
        use_w = min(use_w, width_cap)
        # Cap height growth at 1.3x; when cut-off (nh > cap) only widen, don't allow full height
        height_cap = ch * 1.3 if ch > 0 else nh
        use_h = min(nh, height_cap)
        if use_w != cw or use_h != ch:
            # The layout has already enlarged its internal max size and emitted size_enlarged;
            # avoid calling setMaxSize() again here to prevent recursive reLayout → size_enlarged loops.
            # Grow symmetrically around the center so the text box stays visually centered in the bubble.
            self.set_size(use_w, use_h, set_layout_maxsize=False, preserve_topleft=False)

    def get_scale(self) -> float:
        tl = self.topLevelItem()
        if tl is not None:
            return tl.scale()
        else:
            return self.scale()

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        # Clear any default gray/opaque background only when not editing (editing draws accessories first; clearing would wipe them)
        if not self.is_editting():
            br = self.boundingRect()
            painter.save()
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.fillRect(br, Qt.GlobalColor.transparent)
            painter.restore()
        blk = getattr(self, 'blk', None)
        skew_x = getattr(blk, 'skew_x', 0.0) or 0.0 if blk else 0.0
        skew_y = getattr(blk, 'skew_y', 0.0) or 0.0 if blk else 0.0
        if skew_x != 0 or skew_y != 0:
            painter.save()
            painter.shear(skew_x, skew_y)
        if self.is_editting():
            self._draw_accessories(painter)
        elif self._has_warp():
            # PR #1105: draw text+accessories as warped quad (no separate text/accessories)
            bm = getattr(self.fontformat, 'blend_mode', 0)
            painter.save()
            painter.setCompositionMode(_blend_mode_qt(bm))
            self._draw_warped_content(painter)
            painter.restore()
            if skew_x != 0 or skew_y != 0:
                painter.restore()
            br = self.boundingRect()
            draw_rect = self.draw_rect and not self.under_ctrl
            box_path = self._box_outline_path()
            if self.isSelected() and not self.is_editting():
                pen = QPen(TEXTRECT_SELECTED_COLOR, 3.5 / self.get_scale(), Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(QBrush(Qt.GlobalColor.transparent))
                painter.drawPath(box_path)
            elif draw_rect:
                pen = QPen(TEXTRECT_SHOW_COLOR, 3 / self.get_scale(), Qt.PenStyle.SolidLine)
                painter.setPen(pen)
                painter.setBrush(QBrush(Qt.GlobalColor.transparent))
                painter.drawPath(box_path)
            return
        option.state = QStyle.State_None
        # Prevent base class from filling with palette Base (e.g. brown/beige on some themes)
        painter.save()
        painter.setBrush(QBrush(Qt.GlobalColor.transparent))
        painter.setPen(Qt.PenStyle.NoPen)
        pal = QPalette(option.palette)
        pal.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.transparent)
        option.palette = pal
        bm = getattr(self.fontformat, 'blend_mode', 0)
        painter.setCompositionMode(_blend_mode_qt(bm))
        use_path = not self.is_editting() and getattr(self.fontformat, 'text_on_path', 0) != 0
        if not self.document().isEmpty():
            if use_path:
                # Path text is drawn entirely via background_pixmap (stroke+fill+shadow) in _draw_accessories
                pass
            else:
                if self._text_mask_all_zero():
                    pass
                else:
                    mask_region = self._text_mask_region() if not self.is_editting() else None
                    box_path = self._box_outline_path()
                    clip_region = QRegion(box_path.toFillPolygon().toPolygon())
                    if mask_region is not None:
                        clip_region = mask_region.intersected(clip_region) if clip_region is not None else mask_region
                    if clip_region is not None:
                        painter.setClipRegion(clip_region)
                    super().paint(painter, option, widget)
                    if clip_region is not None:
                        painter.setClipRegion(QRegion())
        painter.restore()

        if not self.is_editting():
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOver)
            self._draw_accessories(painter)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        if skew_x != 0 or skew_y != 0:
            painter.restore()


    def _render_diagnostics(self):
        ff = self.fontformat
        if ff is None:
            return None
        br = self.boundingRect()
        fs = self.layout.max_font_size(to_px=True) if self.layout is not None else pt2px(self.document().defaultFont().pointSizeF())
        mode = resolve_writing_mode(getattr(ff, 'writing_mode', 'auto'), self.toPlainText(), (br.width(), br.height()))
        measured = estimate_text_bounds(
            self.toPlainText(), fs, mode, br.width(), br.height(),
            line_spacing=float(getattr(ff, 'line_spacing', 1.15) or 1.15),
            letter_spacing=float(getattr(ff, 'letter_spacing', 1.0) or 1.0),
            padding=float(getattr(ff, 'text_padding', 0.0) or 0.0),
            stroke_width=float(getattr(ff, 'stroke_width', 0.0) or 0.0),
            shadow_radius=float(getattr(ff, 'shadow_radius', 0.0) or 0.0),
            shadow_offset=getattr(ff, 'shadow_offset', [0.0, 0.0]) or [0.0, 0.0],
            line_break_strategy=getattr(ff, 'line_break_strategy', 'auto'),
        )
        families = merge_font_fallback_chain(getattr(ff, 'font_family', ''), self.toPlainText(), pcfg, getattr(ff, 'fallback_font_chain', ''))
        missing = missing_glyphs_after_fallback(getattr(ff, 'font_family', ''), self.toPlainText(), pcfg, getattr(ff, 'fallback_font_chain', '')) if self.toPlainText() else []
        mask_diag = mask_safe_rect(getattr(getattr(self, 'blk', None), 'text_mask', None))
        return {
            'mode': mode,
            'measured': measured,
            'overflow': measured[0] > br.width() or measured[1] > br.height(),
            'fallback': ', '.join(families),
            'missing': ''.join(missing),
            'mask': mask_diag.to_dict(),
        }

    def _draw_renderer_diagnostics_overlay(self, painter: QPainter):
        if not bool(getattr(pcfg, 'render_diagnostics_overlay', False)):
            return
        diag = self._render_diagnostics()
        if diag is None:
            return
        br = self.boundingRect()
        ff = self.fontformat
        painter.save()
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setPen(QPen(QColor(255, 128, 0, 190) if diag['overflow'] else QColor(0, 180, 255, 160), 1.5 / max(0.01, self.get_scale()), Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(Qt.GlobalColor.transparent))
        painter.drawRect(br)
        measured_w, measured_h = diag['measured'][0], diag['measured'][1]
        painter.setPen(QPen(QColor(255, 64, 64, 170), 1 / max(0.01, self.get_scale())))
        painter.drawRect(QRectF(0, 0, measured_w, measured_h))
        mask_diag = diag.get('mask') or {}
        if mask_diag.get('narrow_safe_area') or mask_diag.get('fully_masked'):
            sr = mask_diag.get('safe_rect') or [0, 0, 0, 0]
            painter.setPen(QPen(QColor(180, 80, 255, 190), 1 / max(0.01, self.get_scale()), Qt.PenStyle.DotLine))
            painter.drawRect(QRectF(
                float(sr[0]),
                float(sr[1]),
                max(0.0, float(sr[2]) - float(sr[0])),
                max(0.0, float(sr[3]) - float(sr[1])),
            ))

        ff = getattr(self, 'fontformat', None)
        label = f"{diag['mode']} {getattr(ff, 'line_break_strategy', 'auto')} {measured_w:.0f}×{measured_h:.0f}"
        if diag['missing']:
            label += f" missing: {diag['missing']}"
        if (diag.get('mask') or {}).get('warning'):
            label += f" mask: {(diag.get('mask') or {}).get('coverage', 0):.0%}"
        painter.setPen(QColor(255, 128, 0) if diag['overflow'] else QColor(0, 128, 255))
        painter.drawText(QPointF(2, max(10.0, 10.0 / max(0.01, self.get_scale()))), label)
        painter.restore()

    def _text_mask_region(self):
        """Build QRegion from blk.text_mask for clipping (#1093 text eraser). Returns None if no mask or mask is full (no clip needed)."""
        blk = getattr(self, 'blk', None)
        if blk is None or getattr(blk, 'text_mask', None) is None:
            return None
        mask_np = blk.text_mask
        if mask_np.size == 0:
            return None
        if np.all(mask_np >= 255):
            return None
        if np.all(mask_np <= 0):
            return None
        br = self.boundingRect()
        tw, th = int(br.width()), int(br.height())
        if tw <= 0 or th <= 0:
            return None
        h, w = mask_np.shape[:2]
        if (w, h) != (tw, th):
            mask_np = cv2.resize(mask_np, (tw, th), interpolation=cv2.INTER_NEAREST)
            h, w = th, tw
        m = np.ascontiguousarray(mask_np, dtype=np.uint8)
        try:
            qimg = QImage(m.tobytes(), w, h, w, QImage.Format.Format_Alpha8)
        except Exception:
            qimg = QImage(m.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        bm = QBitmap.fromImage(qimg, Qt.ImageConversionFlag.ThresholdAlphaDither)
        return QRegion(bm)

    def _text_mask_all_zero(self):
        """True if blk has a text_mask that is all zeros (fully erased)."""
        blk = getattr(self, 'blk', None)
        if blk is None or getattr(blk, 'text_mask', None) is None:
            return False
        return np.all(blk.text_mask <= 0)

    def _get_foreground_pixmap(self) -> Union[QPixmap, None]:
        """Load and cache foreground overlay image (PR #1070). Returns None if none or load fails."""
        blk = getattr(self, 'blk', None)
        path = getattr(blk, 'foreground_image_path', None) if blk else None
        if not path or not path.strip():
            self._foreground_pixmap = None
            return None
        sc = self.scene()
        if sc is None or not getattr(sc, 'imgtrans_proj', None):
            return None
        proj_dir = getattr(sc.imgtrans_proj, 'directory', None)
        if not proj_dir:
            return None
        full_path = osp.normpath(osp.join(proj_dir, path))
        if not osp.isfile(full_path):
            self._foreground_pixmap = None
            return None
        if self._foreground_pixmap is not None and getattr(self, '_foreground_path', None) == full_path:
            return self._foreground_pixmap
        try:
            self._foreground_pixmap = QPixmap(full_path)
            self._foreground_path = full_path
            if self._foreground_pixmap.isNull():
                self._foreground_pixmap = None
            return self._foreground_pixmap
        except Exception:
            self._foreground_pixmap = None
            return None

    def invalidate_foreground_cache(self):
        """Call when blk.foreground_image_path changes (PR #1070)."""
        self._foreground_pixmap = None
        if hasattr(self, '_foreground_path'):
            del self._foreground_path

    def _has_warp(self) -> bool:
        """True if block has quad or mesh warp enabled (PR #1105)."""
        blk = getattr(self, 'blk', None)
        if blk is None:
            return False
        mode = getattr(blk, 'warp_mode', None)
        return mode in ('quad', 'mesh')

    def _warp_quad_points_px(self):
        """Return 4 corner points in item-local pixel coords from blk.warp_quad (PR #1105)."""
        blk = getattr(self, 'blk', None)
        if blk is None:
            return None
        quad = getattr(blk, 'warp_quad', None)
        if not quad or len(quad) != 4:
            return None
        br = self.boundingRect()
        w, h = br.width(), br.height()
        return [QPointF(q[0] * w, q[1] * h) for q in quad]

    def _apply_quad_warp(self, pixmap: QPixmap) -> Union[QPixmap, None]:
        """Apply quad warp to pixmap using blk.warp_quad; returns new pixmap (PR #1105)."""
        pts = self._warp_quad_points_px()
        if pts is None or pixmap.isNull():
            return pixmap
        w, h = pixmap.width(), pixmap.height()
        src_poly = QPolygonF([QPointF(0, 0), QPointF(w, 0), QPointF(w, h), QPointF(0, h)])
        dst_poly = QPolygonF(pts)
        tr = QTransform.quadToQuad(src_poly, dst_poly)
        if not tr.isInvertible():
            return pixmap
        out = QPixmap(w, h)
        out.fill(Qt.GlobalColor.transparent)
        painter = QPainter(out)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setTransform(tr)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        return out

    def _draw_warped_content(self, painter: QPainter):
        """Draw text+accessories as warped quad (PR #1105). Call when not editing and _has_warp()."""
        pixmap = self.toPixmap()
        if pixmap.isNull():
            return
        pts = self._warp_quad_points_px()
        if not pts:
            return
        w, h = pixmap.width(), pixmap.height()
        src_poly = QPolygonF([QPointF(0, 0), QPointF(w, 0), QPointF(w, h), QPointF(0, h)])
        dst_poly = QPolygonF(pts)
        tr = QTransform.quadToQuad(src_poly, dst_poly)
        if not tr.isInvertible():
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setTransform(tr, True)
        painter.drawPixmap(0, 0, pixmap)
        painter.restore()

    def _apply_warp_preset(self, pixmap: QPixmap, warp_style: int, strength: float) -> QPixmap:
        """Apply Photoshop-like warp presets (#1093): 1=Arc, 2=Arch, 3=Bulge, 4=Flag."""
        if warp_style < 1 or strength <= 0:
            return pixmap
        arr = pixmap2ndarray(pixmap, keep_alpha=True)
        if arr is None or arr.size == 0:
            return pixmap
        h, w = arr.shape[:2]
        strength = max(0.01, min(1.0, float(strength)))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        if warp_style == 1:  # Arc: top bulges up
            amp = strength * h * 0.12 * np.sin(np.pi * xx / max(1, w))
            map_x, map_y = xx, yy - amp
        elif warp_style == 2:  # Arch: top bulges down
            amp = strength * h * 0.12 * np.sin(np.pi * xx / max(1, w))
            map_x, map_y = xx, yy + amp
        elif warp_style == 3:  # Bulge: radial
            cx, cy = w * 0.5, h * 0.5
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            R = max(1, np.sqrt(cx ** 2 + cy ** 2))
            f = 1 + strength * 0.3 * (1 - np.minimum(r / R, 1.0))
            map_x = cx + (xx - cx) * f
            map_y = cy + (yy - cy) * f
        elif warp_style == 4:  # Flag: horizontal wave
            amp = strength * w * 0.06 * np.sin(2 * np.pi * yy / max(1, h))
            map_x, map_y = xx - amp, yy
        else:
            return pixmap
        out = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return ndarray2pixmap(out)

    def _draw_accessories(self, painter: QPainter):
        br = self.boundingRect()
        ff = self.fontformat
        painter.save()
        
        if self.background_pixmap is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            if not self._text_mask_all_zero():
                mask_region = self._text_mask_region()
                if mask_region is not None:
                    painter.setClipRegion(mask_region)
                painter.drawPixmap(br.toRect(), self.background_pixmap)
                if mask_region is not None:
                    painter.setClipRegion(QRegion())
            else:
                pass

        # PR #1070: foreground image overlay
        fg_pix = self._get_foreground_pixmap()
        if fg_pix is not None:
            painter.save()
            blk = getattr(self, 'blk', None)
            op = 1.0
            if blk is not None:
                op = max(0.0, min(1.0, getattr(blk, 'overlay_opacity', 1.0)))
            if op < 1.0:
                painter.setOpacity(op)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(br.toRect(), fg_pix)
            painter.restore()

        draw_rect = self.draw_rect and not self.under_ctrl
        box_path = self._box_outline_path()
        if self.isSelected() and not self.is_editting():
            pen = QPen(TEXTRECT_SELECTED_COLOR, 3.5 / self.get_scale(), Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(Qt.GlobalColor.transparent))
            painter.drawPath(box_path)
        elif draw_rect:
            pen = QPen(TEXTRECT_SHOW_COLOR, 3 / self.get_scale(), Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(Qt.GlobalColor.transparent))
            painter.drawPath(box_path)
        painter.restore()


    def startEdit(self, pos: QPointF = None) -> None:
        self.pre_editing = False
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
        self.setFocus()
        self.begin_edit.emit(self.idx)
        if pos is not None:
            hit = self.layout.hitTest(pos, None)
            cursor = self.textCursor()
            cursor.setPosition(hit)
            self.setTextCursor(cursor)

    def endEdit(self, keep_focus=True) -> None:
        self.end_edit.emit(self.idx)
        cursor = self.textCursor()
        cursor.clearSelection()
        self.setTextCursor(cursor)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        if keep_focus:
            self.setFocus()

    def isEditing(self) -> bool:
        return self.textInteractionFlags() == Qt.TextInteractionFlag.TextEditorInteraction

    def isMultiFontSize(self) -> bool:
        doc = self.document()
        block = doc.firstBlock()
        if block.isValid():
            it = block.begin()
            if it.atEnd():
                firstFontSize = block.charFormat().fontPointSize()
            else:
                # empty blocks causes frozen for pyside==6.8.1
                # also randomly freezes pyqt==6.6.1 https://github.com/dmMaze/BallonsTranslator/issues/736
                firstFontSize = it.fragment().charFormat().fontPointSize()
        else:
            return False
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                font_size = fragment.charFormat().fontPointSize()
                if not firstFontSize == font_size:
                    return True
                it += 1
            block = block.next()
        return False
    
    def minFontSize(self, to_px=True):
        doc = self.document()
        block = doc.firstBlock()
        min_font_size = self.textCursor().charFormat().fontPointSize()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                font_size = fragment.charFormat().fontPointSize()
                min_font_size = min(min_font_size, font_size)
                it += 1
            block = block.next()
        if to_px:
            min_font_size = pt2px(min_font_size)
        return min_font_size

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if not self.isEditing():
            self.startEdit(pos=event.pos())
        else:
            super().mouseDoubleClickEvent(event)
        
    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        super().mouseMoveEvent(event)  
        if self.textInteractionFlags() != Qt.TextInteractionFlag.TextEditorInteraction:
            self.moving.emit(self)

    # QT 5.15.x causing segmentation fault 
    def contextMenuEvent(self, event):
        return super().contextMenuEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            sc = self.scene()
            if sc is not None and getattr(sc, 'image_edit_mode', None) == ImageEditMode.TextEraserTool:
                event.ignore()
                return
            self.oldPos = self.pos()
            self.leftbutton_pressed.emit(self.idx)
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            sc = self.scene()
            if sc is not None and getattr(sc, 'image_edit_mode', None) == ImageEditMode.TextEraserTool:
                event.ignore()
                return
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            sc = self.scene()
            if sc is not None and getattr(sc, 'image_edit_mode', None) == ImageEditMode.TextEraserTool:
                event.ignore()
                return
            if self.oldPos != self.pos():
                self.moved.emit()
        return super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.hover_move.emit(self.idx)
        return super().hoverMoveEvent(event)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.hover_enter.emit(self.idx)
        return super().hoverEnterEvent(event)

    def toPixmap(self) -> QPixmap:
        pixmap = QPixmap(self.boundingRect().size().toSize())
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        doc = self.document()
        doc.drawContents(painter)
        painter.end()
        return pixmap

    def toHtml(self) -> str:
        html = super().toHtml()
        tables = table_pattern.findall(html)
        if tables:
            _, td = td_pattern.findall(html)[0]
            html = tables[0] + td + '</body></html>'

        return html.replace('>\n<', '><')

    def get_fontformat(self) -> FontFormat:
        fmt = self.textCursor().charFormat()
        font = fmt.font()
        color = fmt.foreground().color()
        fontformat = self.fontformat.deepcopy()
        fontformat.frgb = [color.red(), color.green(), color.blue()]
        fontformat.font_weight = font.weight()
        fontformat.font_family = font.family()
        if self.isEditing():
            fontformat.font_size = pt2px(font.pointSizeF())
        else:
            fontformat.font_size = self.minFontSize()
        fontformat.bold = font.bold()
        fontformat.underline = font.underline()
        fontformat.strikethrough = font.strikeOut()
        fontformat.italic = font.italic()
        # Preserve gradient settings
        fontformat.gradient_enabled = self.fontformat.gradient_enabled
        fontformat.gradient_type = getattr(self.fontformat, 'gradient_type', 0)
        fontformat.gradient_start_color = self.fontformat.gradient_start_color
        fontformat.gradient_end_color = self.fontformat.gradient_end_color
        fontformat.gradient_angle = self.fontformat.gradient_angle
        fontformat.gradient_size = self.fontformat.gradient_size
        return fontformat

    def set_fontformat(self, ffmat: FontFormat, set_char_format=False, set_stroke_width=True, set_effect=True):
        self.repainting = True
        try:
            from utils.text_rendering import normalize_vertical_punctuation, resolve_writing_mode
            rect = self.absBoundingRect(qrect=True)
            text = self.toPlainText()
            resolved_mode = resolve_writing_mode(getattr(ffmat, 'writing_mode', 'auto'), text, (rect.width(), rect.height()))
            ffmat.vertical = resolved_mode == 'vertical_rl'
            if ffmat.vertical:
                new_text = normalize_vertical_punctuation(text)
                if new_text != text:
                    self.setPlainText(new_text)
                    if getattr(self, 'blk', None) is not None:
                        self.blk.translation = new_text
        except Exception:
            pass
        if self.fontformat.vertical != ffmat.vertical:
            self.setVertical(ffmat.vertical)

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        format = cursor.charFormat()
        font = self.document().defaultFont()
        
        font.setFamily(ffmat.font_family)
        pt = ffmat.size_pt
        font.setPointSizeF(max(1.0, pt) if pt <= 0 else pt)
        font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias)

        fweight = ffmat.font_weight
        if fweight is  None:
            fweight = font.weight()
            ffmat.font_weight = fweight
        font.setBold(ffmat.bold)

        self.document().setDefaultFont(font)
        format.setFont(font)
        if ffmat.gradient_enabled:
            gradient = self.get_text_gradient(ffmat)
            format.setForeground(gradient)
        else:
            format.setForeground(QColor(*ffmat.foreground_color()))
        if not ffmat.bold:
            format.setFontWeight(fweight)
        format.setFontItalic(ffmat.italic)
        format.setFontUnderline(ffmat.underline)
        format.setFontStrikeOut(getattr(ffmat, 'strikethrough', False))
        if not ffmat.vertical:
            format.setFontLetterSpacingType(QFont.SpacingType.PercentageSpacing)
            format.setFontLetterSpacing(ffmat.letter_spacing * 100)
        cursor.setCharFormat(format)
        cursor.select(QTextCursor.SelectionType.Document)
        cursor.setBlockCharFormat(format)
        if set_char_format:
            cursor.setCharFormat(format)
        cursor.clearSelection()
        self._apply_font_fallback_runs(ffmat, format)
        # https://stackoverflow.com/questions/37160039/set-default-character-format-in-qtextdocument
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.setTextCursor(cursor)
        self.stroke_qcolor = QColor(*ffmat.stroke_color())

        if set_effect:
            self.setShadow(ffmat, repaint=False)
        if set_stroke_width:
            self.setStrokeWidth(ffmat.stroke_width, repaint_background=False)
        if float(getattr(ffmat, 'text_padding', 0.0) or 0.0) > 0:
            self.setPadding(float(getattr(ffmat, 'text_padding', 0.0) or 0.0))
        self.setOpacity(ffmat.opacity)
        
        alignment_qt_flag = [Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignCenter, Qt.AlignmentFlag.AlignRight][ffmat.alignment]
        doc = self.document()
        op = doc.defaultTextOption()
        resolved_mode = resolve_writing_mode(getattr(ffmat, 'writing_mode', 'auto'), self.toPlainText(), (self.boundingRect().width(), self.boundingRect().height()))
        if resolved_mode == 'rtl':
            op.setTextDirection(Qt.LayoutDirection.RightToLeft)
            if ffmat.alignment == 0:
                alignment_qt_flag = Qt.AlignmentFlag.AlignRight
        else:
            op.setTextDirection(Qt.LayoutDirection.LeftToRight)
        op.setAlignment(alignment_qt_flag)
        doc.setDefaultTextOption(op)
        
        if ffmat.vertical:
            self.setLetterSpacing(ffmat.letter_spacing)
        self.setLineSpacing(ffmat.line_spacing)
        
        # Preserve gradient properties
        self.fontformat.gradient_enabled = ffmat.gradient_enabled
        self.fontformat.gradient_type = getattr(ffmat, 'gradient_type', 0)
        self.fontformat.gradient_start_color = ffmat.gradient_start_color
        self.fontformat.gradient_end_color = ffmat.gradient_end_color
        self.fontformat.gradient_angle = ffmat.gradient_angle
        self.fontformat.gradient_size = ffmat.gradient_size
        
        self.fontformat.merge(ffmat)
        
        if self.fontformat.gradient_enabled:
            self.update()

        self.repainting = False
        if set_effect or set_stroke_width:
            self.repaint_background()


    def _apply_font_fallback_runs(self, ffmat: FontFormat, base_format: QTextCharFormat):
        """Apply per-character fallback fonts for glyphs missing in the primary family.

        Qt's default font substitution is platform-dependent and can still leave
        tofu boxes in mixed-script manga lettering. Applying explicit fallback
        char formats makes CJK/RTL/emoji fallback deterministic while preserving
        the primary font for supported glyphs.
        """
        text = self.toPlainText()
        if not text:
            return
        runs = font_fallback_runs(
            text,
            getattr(ffmat, 'font_family', ''),
            pcfg,
            getattr(ffmat, 'fallback_font_chain', ''),
        )
        if not runs:
            return
        doc = self.document()
        run_cursor = QTextCursor(doc)
        for start, end, family in runs:
            if end <= start or not family:
                continue
            cfmt = QTextCharFormat()
            cfmt.merge(base_format)
            cfont = cfmt.font()
            cfont.setFamily(family)
            cfmt.setFont(cfont)
            run_cursor.setPosition(start)
            run_cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            run_cursor.mergeCharFormat(cfmt)
        run_cursor.clearSelection()

    def updateBlkFormat(self):
        fmt = self.get_fontformat()
        self.blk.fontformat.merge(fmt)
        self.blk.overlay_opacity = getattr(fmt, 'overlay_opacity', 1.0)
        self.blk.skew_x = getattr(fmt, 'skew_x', 0.0)
        self.blk.skew_y = getattr(fmt, 'skew_y', 0.0)

    def set_cursor_cfmt(self, cursor: QTextCursor, cfmt: QTextCharFormat, merge_char: bool = False):
        doc_is_empty = self.document().isEmpty()
        if merge_char:
            self.block_change_signal = True
            cursor.mergeCharFormat(cfmt)
            self.block_change_signal = False
        cursor.mergeBlockCharFormat(cfmt)
        cursor.clearSelection()
        self.setTextCursor(cursor)
        if doc_is_empty:
            font = cursor.blockCharFormat().font()
            if font.pointSizeF() <= 0:
                font.setPointSizeF(1.0)
            self.document().setDefaultFont(font)

    def _before_set_ffmt(self, set_selected: bool, restore_cursor: bool):
        self.is_formatting = True
        cursor = self.textCursor()

        cursor_pos = None
        if restore_cursor:
            cursor_pos = (cursor.position(), cursor.anchor().__pos__()) if restore_cursor else None

        if set_selected:
            has_set_all = not cursor.hasSelection()
            if has_set_all:
                cursor.select(QTextCursor.SelectionType.Document)
        else:
            has_set_all = False
            cursor = QTextCursor(self.document())
            cursor.select(QTextCursor.SelectionType.Document)

        cursor.beginEditBlock()
        return cursor, dict(cursor_pos=cursor_pos, has_set_all=has_set_all)

    def _after_set_ffmt(self, cursor: QTextCursor, repaint_background: bool, restore_cursor: bool, cursor_pos: Tuple, has_set_all: bool):
        
        if restore_cursor:
            if cursor_pos is not None:
                pos1, pos2 = cursor_pos
                if has_set_all:
                    cursor.setPosition(pos1)
                else:
                    cursor.setPosition(min(pos1, pos2))
                    cursor.setPosition(max(pos1, pos2), QTextCursor.MoveMode.KeepAnchor)
                self.setTextCursor(cursor)

        if repaint_background:
            self.repaint_background()

        cursor.endEditBlock()
        self.is_formatting = False

    def setFontFamily(self, value: str, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        cursor, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        self.layout.relayout_on_changed = False
        self._doc_set_font_family(value, cursor)
        self.layout.relayout_on_changed = True
        self.layout.reLayoutEverything()
        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)

    def _doc_set_font_family(self, value: str, cursor: QTextCursor):
        doc = self.document()
        lastpos = doc.rootFrame().lastPosition()
        if cursor.selectionStart() == 0 and \
            cursor.selectionEnd() == lastpos:
            font = doc.defaultFont()
            font.setFamily(value)
            if font.pointSizeF() <= 0:
                font.setPointSizeF(1.0)
            doc.setDefaultFont(font)

        sel_start = cursor.selectionStart()
        sel_end = cursor.selectionEnd()
        block = doc.firstBlock()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                
                frag_start = fragment.position()
                frag_end = frag_start + fragment.length()
                pos2 = min(frag_end, sel_end)
                pos1 = max(frag_start, sel_start)
                if pos1 < pos2:
                    cfmt = fragment.charFormat()
                    under_line = cfmt.fontUnderline()
                    cfont = cfmt.font()
                    pt_int = cfont.pointSize()
                    pt_float = cfont.pointSizeF()
                    pt_int = max(1, pt_int) if pt_int <= 0 else pt_int
                    pt_float = max(1.0, pt_float) if pt_float <= 0 else pt_float
                    font = QFont(value, pt_int, cfont.weight(), cfont.italic())
                    font.setPointSizeF(pt_float)
                    font.setBold(font.bold())
                    font.setWordSpacing(cfont.wordSpacing())
                    font.setLetterSpacing(cfont.letterSpacingType(), cfont.letterSpacing())
                    cfmt.setFont(font)
                    cfmt.setFontUnderline(under_line)
                    cursor.setPosition(pos1)
                    cursor.setPosition(pos2, QTextCursor.MoveMode.KeepAnchor)
                    cursor.setCharFormat(cfmt)
                it += 1
            block = block.next()

        cfmt = cursor.charFormat()
        cfmt.setFontFamily(value)
        self.set_cursor_cfmt(cursor, cfmt)

    def setFontWeight(self, value: float, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        cursor, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        cfmt = QTextCharFormat()
        cfmt.setFontWeight(value)
        self.set_cursor_cfmt(cursor, cfmt, True)
        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)

    def setFontItalic(self, value: bool, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        cursor, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        cfmt = QTextCharFormat()
        cfmt.setFontItalic(value)
        self.set_cursor_cfmt(cursor, cfmt, True)
        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)

    def setFontUnderline(self, value: bool, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        cursor, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        cfmt = QTextCharFormat()
        cfmt.setFontUnderline(value)
        self.set_cursor_cfmt(cursor, cfmt, True)
        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)

    def setFontStrikethrough(self, value: bool, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        self.fontformat.strikethrough = value
        cursor, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        cfmt = QTextCharFormat()
        cfmt.setFontStrikeOut(value)
        self.set_cursor_cfmt(cursor, cfmt, True)
        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)

    def setGradientEnabled(self, value: bool, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        self.fontformat.gradient_enabled = value
        cursor, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        cfmt = QTextCharFormat()
        if value:
            gradient = self.get_text_gradient()
            cfmt.setForeground(gradient)
        else:
            cfmt.setForeground(QColor(*[int(c) for c in self.fontformat.frgb]))

        self.set_cursor_cfmt(cursor, cfmt, True)
        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)

    def get_text_gradient(self, fontformat: FontFormat = None):
        if fontformat is None:
            fontformat = self.fontformat
        gradient_type = getattr(fontformat, 'gradient_type', 0)
        rect = self.boundingRect()
        center = rect.center()
        radius = max(rect.width(), rect.height()) * fontformat.gradient_size * 0.5
        start_color = QColor(*fontformat.gradient_start_color)
        end_color = QColor(*fontformat.gradient_end_color)
        if gradient_type == 1:
            # Radial: center to edge
            gradient = QRadialGradient(center, radius, center)
            gradient.setColorAt(0, start_color)
            gradient.setColorAt(1, end_color)
            return gradient
        # Linear
        gradient = QLinearGradient()
        angle = fontformat.gradient_angle
        rad = math.radians(angle)
        dx = math.cos(rad)
        dy = math.sin(rad)
        r = max(rect.width(), rect.height()) * fontformat.gradient_size
        gradient.setStart(center.x() - dx * r, center.y() - dy * r)
        gradient.setFinalStop(center.x() + dx * r, center.y() + dy * r)
        gradient.setColorAt(0, start_color)
        gradient.setColorAt(1, end_color)
        return gradient

    def setLineSpacing(self, value: float, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        self.is_formatting = True
        self.fontformat.line_spacing = value
        self.layout.setLineSpacing(value)
        if repaint_background:
            self.repaint_background()
            self.update()
        self.is_formatting = False

    def setLineSpacingType(self, value: int, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        self.is_formatting = True
        self.fontformat.line_spacing_type = value
        self.layout.setLineSpacingType(value)
        if repaint_background:
            self.repaint_background()
            self.update()
        self.is_formatting = False

    def setLetterSpacing(self, value: float, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False, force=False):
        self.is_formatting = True
        self.fontformat.letter_spacing = value
        if self.fontformat.vertical:
            self.layout.setLetterSpacing(value)
        else:
            cursor = QTextCursor(self.document())
            char_fmt = QTextCharFormat()
            char_fmt.setFontLetterSpacingType(QFont.SpacingType.PercentageSpacing)
            char_fmt.setFontLetterSpacing(value * 100)
            cursor.select(QTextCursor.SelectionType.Document)
            self.set_cursor_cfmt(cursor, char_fmt, True)

        if repaint_background:
            self.repaint_background()
            self.update()

        self.is_formatting = False

    def setFontColor(self, value: Tuple, repaint_background: bool = False, set_selected: bool = False, restore_cursor: bool = False, force=False):
        cursor, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        cfmt = QTextCharFormat()
        cfmt.setForeground(QColor(*value))
        self.set_cursor_cfmt(cursor, cfmt, True)
        self._after_set_ffmt(cursor, repaint_background=repaint_background, restore_cursor=restore_cursor, **after_kwargs)

    def setStrokeColor(self, scolor, **kwargs):
        self.stroke_qcolor = scolor if isinstance(scolor, QColor) else QColor(*scolor)
        self.fontformat.srgb = [self.stroke_qcolor.red(), self.stroke_qcolor.green(), self.stroke_qcolor.blue()]
        self.repaint_background()
        self.update()

    def setStrokeWidth(self, stroke_width: float, padding=True, repaint_background=True, restore_cursor=False, **kwargs):
        
        cursor, after_kwargs = self._before_set_ffmt(set_selected=False, restore_cursor=restore_cursor)

        self.fontformat.stroke_width = stroke_width
        if stroke_width > 0 and padding:
            p = self.layout.max_font_size(to_px=True) * (stroke_width + 0.05) / 2
            self.setPadding(p)

        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)
        if repaint_background:
            self.update()

    def setRelFontSize(self, value: float, repaint_background: bool = False, set_selected: bool = False, restore_cursor: bool = False, clip_size: bool = False, **kwargs):
        self.layout.relayout_on_changed = False
        _, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        doc = self.document()
        cursor = QTextCursor(doc)
        block = doc.firstBlock()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                old_font_size = fragment.charFormat().fontPointSize()
                new_font_size = max(1.0, round(old_font_size * value, 2)) if old_font_size <= 0 or old_font_size * value <= 0 else round(old_font_size * value, 2)
                cfmt = fragment.charFormat()
                cfmt.setFontPointSize(new_font_size)
                pos1 = fragment.position()
                pos2 = pos1 + fragment.length()
                cursor.setPosition(pos1)
                cursor.setPosition(pos2, QTextCursor.MoveMode.KeepAnchor)
                cursor.mergeCharFormat(cfmt)
                it += 1
            block = block.next()
        self.layout.relayout_on_changed = True
        self.layout.reLayoutEverything()
        if clip_size:
            self.squeezeBoundingRect(True, repaint=False)

        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)
        

    def setFontSize(self, value: float, repaint_background: bool = False, set_selected: bool = False, restore_cursor: bool = False, clip_size: bool = False, **kwargs):
        '''
        value should be point size
        '''
        
        cursor, after_kwargs = self._before_set_ffmt(set_selected=set_selected, restore_cursor=restore_cursor)
        self.layout.relayout_on_changed = False
        if self.fontformat.stroke_width != 0:
            repaint_background = True
        if repaint_background:
            fs = pt2px(max(self.layout.max_font_size(), value))
            self.layout.relayout_on_changed = False
            if self.fontformat.stroke_width > 0:
                self.setPadding(fs * (self.fontformat.stroke_width + 0.05) / 2)
            self.layout.relayout_on_changed = True
        cfmt = QTextCharFormat()
        pt = max(1.0, value) if value <= 0 else value
        cfmt.setFontPointSize(pt)
        self.set_cursor_cfmt(cursor, cfmt, True)
        self.layout.relayout_on_changed = True
        self.layout.reLayoutEverything()
        if clip_size:
            self.squeezeBoundingRect(cond_on_alignment=True)

        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)

    def setAlignment(self, value, restore_cursor=False, repaint_background=True, *args, **kwargs):
        cursor, after_kwargs = self._before_set_ffmt(set_selected=False, restore_cursor=restore_cursor)
        if isinstance(value, int):
            qt_align_flag = [Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignCenter, Qt.AlignmentFlag.AlignRight][value]
        doc = self.document()
        op = doc.defaultTextOption()
        op.setAlignment(qt_align_flag)
        doc.setDefaultTextOption(op)
        if repaint_background:
            self.repaint_background()
            self.update()
        self.fontformat.alignment = value
        self._after_set_ffmt(cursor, repaint_background=False, restore_cursor=restore_cursor, **after_kwargs)

    def get_char_fmts(self) -> List[QTextCharFormat]:
        cursor = self.textCursor()
        
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        char_fmts = []
        while True:
            cursor.movePosition(QTextCursor.MoveOperation.NextCharacter)
            cursor.clearSelection()
            char_fmts.append(cursor.charFormat())
            if cursor.atEnd():
                break
        return char_fmts

    def setShadow(self, fmt: FontFormat, repaint=True):
        self.fontformat.shadow_radius = fmt.shadow_radius
        self.fontformat.shadow_strength = fmt.shadow_strength
        self.fontformat.shadow_color = fmt.shadow_color
        self.fontformat.shadow_offset = fmt.shadow_offset
        if self.fontformat.shadow_radius > 0:
            self.setPadding(self.layout.max_font_size(to_px=True))
        if repaint:
            self.repaint_background()

    def setBGAttribute(self, attr_name: str, value, repaint=True):
        setattr(self.fontformat, attr_name, value)
        if repaint:
            self.repaint_background()
            self.update()

    def setGradientAttribute(self, attr_name: str, value):
        self.old_ffmt_values = {}
        self.old_ffmt_values[attr_name] = self.fontformat[attr_name]
        setattr(self.fontformat, attr_name, value)
        self.setGradientEnabled(self.fontformat.gradient_enabled)
        self.old_ffmt_values = None

    def setOpacity(self, opacity: float):
        super().setOpacity(opacity)
        self.fontformat.opacity = opacity

    def setPlainTextAndKeepUndoStack(self, text: str):
        cursor = QTextCursor(self.document())
        cursor.select(QTextCursor.SelectionType.Document)
        cursor.insertText(text)

    def squeezeBoundingRect(self, cond_on_alignment: bool = False, repaint=True):
        mh, mw = self.layout.minSize()
        if mh == 0 or mw == 0:
            return
        br = self.absBoundingRect(qrect=True)
        br_w, br_h = br.width(), br.height()

        if self.fontformat.vertical:
            if cond_on_alignment:
                mh = br.height()
        else:
            if cond_on_alignment:
                mw = br.width()

        if np.abs(br_w - mw) > 0.001 or np.abs(br_h - mh) > 0.001:
            P = self.padding() * 2
            mh += P
            mw += P
            self.set_size(mw, mh, set_layout_maxsize=True, set_blk_size=True)
            if self.under_ctrl:
                self.doc_size_changed.emit(self.idx)
            if repaint:
                self.repaint_background()

    def scene_scale_factor(self):
        scale = 1
        if hasattr(self.scene(), 'scale_factor'):
            scale = self.scene().scale_factor
        return scale
            
    def set_size(self, w: float, h: float, set_layout_maxsize=False, set_blk_size=True, preserve_topleft=False):
        '''
        rotation invariant. When preserve_topleft=True, do not move the item (keep top-left fixed so box stays in bubble).
        '''

        if set_layout_maxsize:
            self.layout.setMaxSize(w, h)

        old_w = self._display_rect.width()
        old_h = self._display_rect.height()
        oc = self.sceneBoundingRect().center() if not preserve_topleft else None

        self._display_rect.setWidth(w)
        self._display_rect.setHeight(h)
        self.setCenterTransform()

        if not preserve_topleft and oc is not None:
            pos_shift = oc - self.sceneBoundingRect().center()
            pos_shift = pos_shift / self.scene_scale_factor()
            
            align_c = align_tl = align_tr = False
            if self.fontformat.vertical:
                align_tr = True
            else:
                alignment = self.fontformat.alignment
                if alignment == TextAlignment.Left:
                    align_tl = True
                elif alignment == TextAlignment.Right:
                    align_tr = True
                else:
                    align_c = True

            if align_c:
                pass
            else:
                dw, dh = (w - old_w) / 2, (h - old_h) / 2
                if align_tr:
                    dw = -dw
                rad = -np.deg2rad(self.rotation())
                c, s = np.cos(rad), np.sin(rad)
                dx = c * dw + s * dh
                dy = -s * dw + c * dh
                pos_shift = pos_shift + QPointF(dx, dy)

            self.setPos(self.pos() + pos_shift)
        if self.blk is not None and set_blk_size:
            self.blk._bounding_rect = self.absBoundingRect()
