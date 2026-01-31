#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PySide6.QtCore import Qt, QPointF, Signal, QPoint, QRectF
from PySide6.QtGui import (QImage, QColor, QPainter, QPixmap, QBrush,
                           QAction, QCursor)
from PySide6.QtWidgets import QWidget, QMenu, QApplication

from libs.shape import Shape
from libs.utils import distance

# PySide6 枚举定义
CURSOR_DEFAULT = Qt.CursorShape.ArrowCursor
CURSOR_POINT = Qt.CursorShape.PointingHandCursor
CURSOR_DRAW = Qt.CursorShape.CrossCursor
CURSOR_MOVE = Qt.CursorShape.ClosedHandCursor
CURSOR_GRAB = Qt.CursorShape.OpenHandCursor

class Canvas(QWidget):
    zoomRequest = Signal(int)
    lightRequest = Signal(int)
    scrollRequest = Signal(int, int)  # orientation 传入 int 即可
    newShape = Signal()
    selectionChanged = Signal(bool)
    shapeMoved = Signal()
    drawingPolygon = Signal(bool)

    CREATE, EDIT = list(range(2))

    # 点击容差，像素单位
    epsilon = 11.0

    def __init__(self, *args, **kwargs):
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.current = None
        self.selected_shape = None  # save the selected shape here
        self.selected_shape_copy = None
        self.drawing_line_color = QColor(0, 0, 255)
        self.drawing_rect_color = QColor(0, 0, 255)
        self.line = Shape(line_color=self.drawing_line_color)
        self.prev_point = QPointF()
        self.offsets = QPointF(), QPointF()
        self.scale = 1.0
        self.label_font_size = 8
        self.pixmap = QPixmap()
        self.visible = {}
        self._hide_background = False
        self.hide_backed = False
        self.h_shape = None
        self.h_vertex = None
        self._painter = QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        self.menus = (QMenu(), QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.verified = False
        self.draw_square = False
        # 当用户在第一次点击时按下了矩形模式（例如按住 Ctrl），
        # 即使在第二次点击前释放 Ctrl 也应保持矩形模式直到 finalise()。
        self._drawing_rect_mode = False
        self.pan_initial_pos = QPoint()

        # Initialise action states.
        self._create_mode = False
        self._fill_drawing = False
        self._fill_backgound = False
        self.overlay_color = None

    # ----------------------------------------------------------------------
    # 核心修复: 添加缺失的方法
    # ----------------------------------------------------------------------
    def set_drawing_shape_to_square(self, status):
        self.draw_square = status

    def enterEvent(self, ev):
        self.override_cursor(self._cursor)
        super(Canvas, self).enterEvent(ev)

    def leaveEvent(self, ev):
        self.restore_cursor()
        super(Canvas, self).leaveEvent(ev)

    def focusOutEvent(self, ev):
        self.restore_cursor()
        super(Canvas, self).focusOutEvent(ev)

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def set_editing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.un_highlight()
            self.de_select_shape()
        self.prev_point = QPointF()
        self.repaint()

    def un_highlight(self):
        if self.h_shape:
            self.h_shape.highlight_clear()
        self.h_vertex = None
        self.h_shape = None

    def selected_vertex(self):
        return self.h_vertex is not None

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        # PySide6: ev.position() 返回 QPointF
        pos = self.transform_pos(ev.position())

        # DEBUG: log mouse move state
        try:
            print(f"[CANVAS][mouseMoveEvent] pos=({pos.x():.1f},{pos.y():.1f}) draw_square={self.draw_square} _drawing_rect_mode={self._drawing_rect_mode} current_points={len(self.current.points) if self.current else 0}")
        except Exception:
            print(f"[CANVAS][mouseMoveEvent] pos=({pos.x():.1f},{pos.y():.1f})")

        # Update coordinates in main window
        window = self.parent().window()
        if hasattr(window, 'label_coordinates'):
            window.label_coordinates.setText(f'X: {pos.x():.0f}; Y: {pos.y():.0f}')

        # Polygon drawing.
        if self.drawing():
            self.override_cursor(CURSOR_DRAW)
            if self.current:
                # For rectangle mode, show width and height
                if (self.draw_square or self._drawing_rect_mode) and len(self.current.points) > 0:
                    current_width = abs(self.current.points[0].x() - pos.x())
                    current_height = abs(self.current.points[0].y() - pos.y())
                    if hasattr(window, 'label_coordinates'):
                        window.label_coordinates.setText(
                            f'Width: {current_width:.0f}, Height: {current_height:.0f} / X: {pos.x():.0f}; Y: {pos.y():.0f}')
                
                color = self.drawing_line_color
                if self.out_of_pixmap(pos):
                    # Don't allow the user to draw outside the pixmap.
                    # Project the point to the pixmap's boundaries.
                    if len(self.current.points) > 0:
                        pos = self.intersection_point(self.current.points[-1], pos)
                elif len(self.current.points) > 1 and self.close_enough(pos, self.current.points[0]):
                    # Attract line to starting point and colorise to alert the user:
                    pos = self.current.points[0]
                    color = self.current.line_color
                    self.override_cursor(CURSOR_POINT)
                    self.current.highlight_vertex(0, Shape.NEAR_VERTEX)

                # Update preview line for rectangle mode. 这里使用临时状态
                # `_drawing_rect_mode` 以保证在两次点击之间即使释放 modifier
                # 也能保持矩形预览行为。
                if self.draw_square or self._drawing_rect_mode:
                    self.line[0] = self.current.points[0]
                    self.line[1] = pos
                else:
                    self.line[1] = pos

                self.line.line_color = color
                self.prev_point = QPointF()
                self.current.highlight_clear()
            else:
                self.prev_point = pos
            self.repaint()
            return

        # Polygon copy moving.
        if Qt.MouseButton.RightButton & ev.buttons():
            if self.selected_shape_copy and not self.prev_point.isNull():
                self.override_cursor(CURSOR_MOVE)
                self.bounded_move_shape(self.selected_shape_copy, pos)
                self.repaint()
            elif self.selected_shape:
                self.selected_shape_copy = self.selected_shape.copy()
                self.repaint()
            return

        # Polygon/Vertex moving.
        if Qt.MouseButton.LeftButton & ev.buttons():
            if self.selected_vertex():
                self.bounded_move_vertex(pos)
                self.shapeMoved.emit()
                self.repaint()
            elif self.selected_shape and not self.prev_point.isNull():
                self.override_cursor(CURSOR_MOVE)
                self.bounded_move_shape(self.selected_shape, pos)
                self.shapeMoved.emit()
                self.repaint()
            else:
                # Pan Mode
                delta = ev.position().toPoint() - self.pan_initial_pos
                self.scrollRequest.emit(delta.x(), Qt.Orientation.Horizontal)
                self.scrollRequest.emit(delta.y(), Qt.Orientation.Vertical)
                self.update()
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        self.setToolTip("Image")
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearest_vertex(pos, self.epsilon)
            if index is not None:
                if self.selected_vertex():
                    self.h_shape.highlight_clear()
                self.h_vertex, self.h_shape = index, shape
                shape.highlight_vertex(index, Shape.MOVE_VERTEX)
                self.override_cursor(CURSOR_POINT)
                self.setToolTip("Click & drag to move point")
                self.setStatusTip(self.toolTip())
                self.update()
                return
            elif shape.contains_point(pos):
                if self.selected_vertex():
                    self.h_shape.highlight_clear()
                self.h_vertex, self.h_shape = None, shape
                self.setToolTip(f"Click & drag to move shape '{shape.label}'")
                self.setStatusTip(self.toolTip())
                self.override_cursor(CURSOR_GRAB)
                self.update()
                return
        else:  # Nothing found, clear highlights, reset state.
            if self.h_shape:
                self.h_shape.highlight_clear()
                self.update()
            self.h_vertex, self.h_shape = None, None
            self.override_cursor(CURSOR_DEFAULT)

    def mousePressEvent(self, ev):
        # PySide6: ev.position()
        pos = self.transform_pos(ev.position())

        # DEBUG: log mouse press
        mods = ev.modifiers()
        try:
            mods_val = int(mods)
        except Exception:
            mods_val = 0
        print(f"[CANVAS][mousePressEvent] button={ev.button()} pos=({pos.x():.1f},{pos.y():.1f}) draw_square={self.draw_square} _drawing_rect_mode={self._drawing_rect_mode} current_exists={self.current is not None} mods={mods_val}")

        if ev.button() == Qt.MouseButton.LeftButton:
            if self.drawing():
                # If Ctrl is held at the moment of click, treat this as rectangle start.
                if mods & Qt.KeyboardModifier.ControlModifier:
                    # Temporarily enable rectangle drawing mode for this interaction.
                    self._drawing_rect_mode = True
                    print("[CANVAS][mousePressEvent] Ctrl detected -> enabling _drawing_rect_mode")
                self.handle_drawing(pos)
            else:
                selection = self.select_shape_point(pos)
                self.prev_point = pos
                if selection is None:
                    # Start Pan
                    QApplication.setOverrideCursor(Qt.CursorShape.OpenHandCursor)
                    self.pan_initial_pos = ev.position().toPoint()

        elif ev.button() == Qt.MouseButton.RightButton and self.editing():
            self.select_shape_point(pos)
            self.prev_point = pos
            self.handle_right_click(pos, ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            menu = self.menus[bool(self.selected_shape_copy)]
            self.restore_cursor()
            # PySide6: exec requires global pos
            if not menu.exec(self.mapToGlobal(ev.position().toPoint())) and self.selected_shape_copy:
                self.selected_shape_copy = None
                self.repaint()

        elif ev.button() == Qt.MouseButton.LeftButton:
            if self.selected_shape:
                if self.selected_vertex():
                    self.override_cursor(CURSOR_POINT)
                else:
                    self.override_cursor(CURSOR_GRAB)
            elif not self.drawing():
                QApplication.restoreOverrideCursor()

    def handle_drawing(self, pos):
        # Rectangle mode: 2-click drawing (first point, then second point)
        print(f"[CANVAS][handle_drawing] pos=({pos.x():.1f},{pos.y():.1f}) draw_square={self.draw_square} _drawing_rect_mode={self._drawing_rect_mode} current_exists={self.current is not None}")

        if self.draw_square or self._drawing_rect_mode:
            if not self.current:
                # First click: create shape and record first point
                self.current = Shape()
                self.current.add_point(pos)
                self.line.points = [pos, pos]
                # 锁定矩形模式直到 finalise()
                self._drawing_rect_mode = True
                self.set_hiding(False)
                self.drawingPolygon.emit(True)
                self.update()
                return
            else:
                # Second click: complete the rectangle with 4 points
                p1 = self.current.points[0]
                p2 = pos
                # Calculate rectangle corners
                min_x = min(p1.x(), p2.x())
                max_x = max(p1.x(), p2.x())
                min_y = min(p1.y(), p2.y())
                max_y = max(p1.y(), p2.y())
                # Set all 4 corners
                self.current.points = [
                    QPointF(min_x, min_y),  # top-left
                    QPointF(min_x, max_y),  # bottom-left
                    QPointF(max_x, max_y),  # bottom-right
                    QPointF(max_x, min_y)   # top-right
                ]
                self.finalise()
                return
        
        # Polygon mode: multi-click drawing
        if self.current and self.current.reach_max_points() is False:
            target_pos = self.line[1]
            self.current.add_point(target_pos)
            self.line[0] = target_pos
            # 如果新添加的点与起点相同或足够接近，则视为闭合并 finalise()
            if len(self.current.points) > 2 and (
                    self.current.points[-1] == self.current.points[0] or
                    self.close_enough(self.current.points[-1], self.current.points[0])):
                # 强制使用相同起点以保证 finalise() 的第一项相等分支生效
                self.current.points[-1] = self.current.points[0]
                self.finalise()
            elif self.current.reach_max_points():
                # 达到最大点数（例如 4）也终止绘制
                self.finalise()
        elif not self.out_of_pixmap(pos):
            self.current = Shape()
            self.current.add_point(pos)
            self.line.points = [pos, pos]
            self.set_hiding(False)
            self.drawingPolygon.emit(True)
            self.update()

    def handle_right_click(self, pos, ev):
        if self.drawing():
            if len(self.current.points) > 1:
                self.current.pop_point()
                self.line[0] = self.current.points[-1]
                self.line[1] = pos
                self.repaint()
            elif len(self.current.points) == 1:
                self.reset_all_lines()
                self.drawingPolygon.emit(False)

    def end_move(self, copy=False):
        assert self.selected_shape and self.selected_shape_copy
        shape = self.selected_shape_copy
        # del shape.fill_color
        # del shape.line_color
        if copy:
            self.shapes.append(shape)
            self.selected_shape.selected = False
            self.selected_shape = shape
            self.repaint()
        else:
            self.selected_shape.points = [p for p in shape.points]
        self.selected_shape_copy = None

    def hide_backround_shapes(self, value):
        self.hide_backed = value
        if self.selected_shape:
            self.set_hiding(True)
            self.repaint()

    def set_hiding(self, enable=True):
        self._hide_background = self.hide_backed if enable else False

    def can_close_shape(self):
        return self.drawing() and self.current and len(self.current) > 2

    def select_shape_point(self, pos):
        """Select the shape or vertex at the given position."""
        self.de_select_shape()
        if self.selected_vertex():  # A vertex is already highlighted
            index, shape = self.h_vertex, self.h_shape
            shape.highlight_vertex(index, Shape.MOVE_VERTEX)
            self.select_shape(shape)
            return self.h_vertex
        
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            if self.isVisible(shape) and shape.contains_point(pos):
                self.select_shape(shape)
                self.calculate_offsets(shape, pos)
                return self.selected_shape
        return None

    def calculate_offsets(self, shape, point):
        rect = shape.bounding_rect()
        x1 = rect.x() - point.x()
        y1 = rect.y() - point.y()
        x2 = (rect.x() + rect.width()) - point.x()
        y2 = (rect.y() + rect.height()) - point.y()
        self.offsets = QPointF(x1, y1), QPointF(x2, y2)

    def select_shape(self, shape):
        self.de_select_shape()
        shape.selected = True
        self.selected_shape = shape
        self.set_hiding()
        self.selectionChanged.emit(True)
        self.update()

    def set_shape_visible(self, shape, value):
        self.visible[shape] = value
        self.repaint()

    def override_cursor(self, cursor):
        self._cursor = cursor
        QApplication.setOverrideCursor(cursor)

    def restore_cursor(self):
        QApplication.restoreOverrideCursor()

    def reset_state(self):
        self.restore_cursor()
        self.pixmap = None
        self.update()

    def set_drawing_color(self, qColor):
        self.drawing_line_color = qColor
        self.drawing_rect_color = qColor

    def de_select_shape(self):
        if self.selected_shape:
            self.selected_shape.selected = False
            self.selected_shape = None
            self.set_hiding(False)
            self.selectionChanged.emit(False)
            self.update()

    def delete_selected(self):
        if self.selected_shape:
            shape = self.selected_shape
            self.shapes.remove(self.selected_shape)
            self.selected_shape = None
            self.update()
            return shape

    def copy_selected_shape(self):
        if self.selected_shape:
            shape = self.selected_shape.copy()
            self.de_select_shape()
            self.shapes.append(shape)
            shape.selected = True
            self.selected_shape = shape
            self.bounded_shift_shape(shape)
            return shape

    def bounded_shift_shape(self, shape):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shape.points[0]
        offset = QPointF(2.0, 2.0)
        self.calculate_offsets(shape, point)
        self.prev_point = point
        if not self.bounded_move_shape(shape, point - offset):
            self.bounded_move_shape(shape, point + offset)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offset_to_center())

        p.drawPixmap(0, 0, self.pixmap)
        
        # Draw shapes
        Shape.scale = self.scale
        Shape.label_font_size = self.label_font_size
        for shape in self.shapes:
            if (shape.selected or not self._hide_background) and self.isVisible(shape):
                shape.fill = shape.selected or shape == self.h_shape
                shape.paint(p)

        if self.current:
            # For rectangle mode, only draw the preview rect, not the polygon points
            if not (self.draw_square or self._drawing_rect_mode):
                self.current.paint(p)
                self.line.paint(p)

        if self.selected_shape_copy:
            self.selected_shape_copy.paint(p)

        # Paint rect preview (for rectangle mode)
        if (self.draw_square or self._drawing_rect_mode) and self.current is not None and len(self.line) == 2:
            left_top = self.line[0]
            right_bottom = self.line[1]
            rect_width = right_bottom.x() - left_top.x()
            rect_height = right_bottom.y() - left_top.y()
            p.setPen(self.drawing_rect_color)
            brush = p.brush()
            brush.setColor(self.drawing_rect_color)
            brush.setStyle(Qt.BrushStyle.Dense7Pattern)
            p.setBrush(brush)
            p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)
        elif self.current is not None and len(self.line) == 2 and not (self.draw_square or self._drawing_rect_mode):
            # Paint rect for polygon mode (legacy)
            left_top = self.line[0]
            right_bottom = self.line[1]
            rect_width = right_bottom.x() - left_top.x()
            rect_height = right_bottom.y() - left_top.y()
            p.setPen(self.drawing_rect_color)
            brush = p.brush()
            brush.setColor(self.drawing_rect_color)
            brush.setStyle(Qt.BrushStyle.Dense7Pattern)
            p.setBrush(brush)
            p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)

        if self.drawing() and not self.prev_point.isNull() and not self.out_of_pixmap(self.prev_point):
            p.setPen(QColor(0, 0, 0))
            p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), self.pixmap.height())
            p.drawLine(0, int(self.prev_point.y()), self.pixmap.width(), int(self.prev_point.y()))

        self.setAutoFillBackground(True)
        if self.verified:
            pal = self.palette()
            pal.setColor(self.backgroundRole(), QColor(184, 239, 38, 128))
            self.setPalette(pal)
        else:
            pal = self.palette()
            pal.setColor(self.backgroundRole(), QColor(232, 232, 232, 255))
            self.setPalette(pal)

        p.end()

    def transform_pos(self, point):
        """Convert from widget-logical coordinates to painter-logical coordinates."""
        return point / self.scale - self.offset_to_center()

    def offset_to_center(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QPointF(x, y)

    def out_of_pixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w and 0 <= p.y() <= h)

    def finalise(self):
        assert self.current
        if self.current.points[0] == self.current.points[-1]:
            self.current = None
            self.drawingPolygon.emit(False)
            self.update()
            return

        # 绘制完成后，释放矩形临时模式标志
        self._drawing_rect_mode = False

        self.current.close()
        self.shapes.append(self.current)
        self.current = None
        self.set_hiding(False)
        self.newShape.emit()
        self.update()

    def close_enough(self, p1, p2):
        return distance(p1 - p2) < self.epsilon

    def intersection_point(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        size = self.pixmap.size()
        points = [(0, 0),
                  (size.width(), 0),
                  (size.width(), size.height()),
                  (0, size.height())]
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersecting_edges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QPointF(x, y)

    def intersecting_edges(self, x1y1, x2y2, points):
        x1, y1 = x1y1
        x2, y2 = x2y2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                continue
            ua = nua / denom
            ub = nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = distance(m - QPointF(x2, y2))
                yield d, i, (x, y)

    def len(self):
        return len(self.shapes)

    def current_cursor(self):
        cursor = QApplication.overrideCursor()
        if cursor is not None:
            cursor = cursor.shape()
        return cursor

    def reset_all_lines(self):
        self.current = None
        # 取消任何临时矩形模式
        self._drawing_rect_mode = False
        self.drawingPolygon.emit(False)
        self.update()

    def load_pixmap(self, pixmap):
        self.pixmap = pixmap
        self.shapes = []
        self.repaint()

    def load_shapes(self, shapes):
        self.shapes = list(shapes)
        self.current = None
        self.repaint()

    def current_item(self):
        return self.current

    def undoLastLine(self):
        pass

    def set_last_label(self, text, line_color=None, fill_color=None):
        assert text
        self.shapes[-1].label = text
        if line_color:
            self.shapes[-1].line_color = line_color
        if fill_color:
            self.shapes[-1].fill_color = fill_color
        return self.shapes[-1]

    def undoLastPoint(self):
        if self.current and self.current.points:
            self.current.points.pop()
            self.update()

    def bounded_move_vertex(self, pos):
        index, shape = self.h_vertex, self.h_shape
        point = shape.points[index]
        if self.out_of_pixmap(pos):
            pos = self.intersection_point(point, pos)

        shape.points[index] = pos
        return True

    def bounded_move_shape(self, shape, pos):
        if self.out_of_pixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.out_of_pixmap(o1):
            pos -= QPointF(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.out_of_pixmap(o2):
            pos += QPointF(min(0, self.pixmap.width() - o2.x()),
                           min(0, self.pixmap.height() - o2.y()))
        
        dp = pos - self.prev_point
        if dp:
            shape.move(dp)
            self.prev_point = pos
            return True
        return False

    def snap_point_to_canvas(self, x, y):
        """
        Moves a point x,y to within the boundaries of the canvas.
        :return: (x,y,snapped) where snapped is True if x or y were changed, False if not.
        """
        if x < 0 or x > self.pixmap.width() or y < 0 or y > self.pixmap.height():
            x = max(x, 0)
            y = max(y, 0)
            x = min(x, self.pixmap.width())
            y = min(y, self.pixmap.height())
            return x, y, True

        return x, y, False

    def wheelEvent(self, ev):
        delta = ev.angleDelta()
        h_delta = delta.x()
        v_delta = delta.y()

        mods = ev.modifiers()
        if (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier) == mods and v_delta:
            self.lightRequest.emit(v_delta)
        elif Qt.KeyboardModifier.ControlModifier == mods and v_delta:
            self.zoomRequest.emit(v_delta)
        else:
            if v_delta: self.scrollRequest.emit(v_delta, Qt.Orientation.Vertical)
            if h_delta: self.scrollRequest.emit(h_delta, Qt.Orientation.Horizontal)
        ev.accept()

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == Qt.Key.Key_Escape:
            if self.current:
                self.current = None
                self.drawingPolygon.emit(False)
                self.update()
        elif key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            if self.can_close_shape():
                self.finalise()
        elif key == Qt.Key.Key_Left or key == Qt.Key.Key_Right or key == Qt.Key.Key_Up or key == Qt.Key.Key_Down:
            self.move_shape(key)

    def move_shape(self, key):
        index = self.selected_shape
        if self.selected_shape:
            # 简化版移动逻辑，如果需要像素级微调可参考原版
            pass