#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа 3. Базовые растровые алгоритмы (PyQt6).

Алгоритмы:
  1. Пошаговый алгоритм (y = k*x + b)
  2. ЦДА (DDA)
  3. Брезенхем (отрезок, все октанты)
  4. Брезенхем (окружность, midpoint circle)
  + (доп.) Кастл–Питвей (алгоритм Бронса/Питвея, строковая перезапись)
  + (доп.) Сглаживание отрезка по Ву (Xiaolin Wu)

Запуск:  pip install PyQt6  &&  python raster_lab3.py
"""
import math
import sys
from time import perf_counter

from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF, QBrush, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QComboBox, QSpinBox, QPushButton, QLabel, QCheckBox, QSlider,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QPlainTextEdit,
    QHeaderView, QFileDialog, QMessageBox, QAbstractItemView,
)


def rnd(v):
    """Округление «половина вверх» (round() в Python — банковское)."""
    return math.floor(v + 0.5)


# ======================================================================
#                          АЛГОРИТМЫ РАСТЕРИЗАЦИИ
# Каждая функция возвращает (точки, заголовки_таблицы, текст_пояснения).
# Если log is not None — в него добавляются строки таблицы вычислений.
# ======================================================================

def step_by_step(x1, y1, x2, y2, log=None):
    """Пошаговый алгоритм: y = k*x + b, шаг по более «длинной» оси."""
    dx, dy = x2 - x1, y2 - y1
    pts = []
    if dx == 0 and dy == 0:
        pts.append((x1, y1))
        if log is not None:
            log.append((0, x1, y1, "—"))
        return pts, ["№", "x", "y", "—"], "Вырожденный отрезок: одна точка."
    if abs(dx) >= abs(dy):
        k = dy / dx
        b = y1 - k * x1
        st = 1 if dx > 0 else -1
        hdr = ["№", "x", "y = k·x + b", "y (округл.)"]
        info = (f"Δx = {dx}, Δy = {dy}\nk = Δy/Δx = {k:.6f}\n"
                f"b = y1 − k·x1 = {b:.6f}\n"
                f"Шаг по x от {x1} до {x2} (шаг {st}); y = round(k·x + b)") if log is not None else ""
        for i, x in enumerate(range(x1, x2 + st, st)):
            yf = k * x + b
            y = rnd(yf)
            pts.append((x, y))
            if log is not None:
                log.append((i, x, f"{yf:.4f}", y))
    else:
        k = dx / dy
        b = x1 - k * y1
        st = 1 if dy > 0 else -1
        hdr = ["№", "y", "x = k·y + b", "x (округл.)"]
        info = (f"Δx = {dx}, Δy = {dy}  (|Δy| > |Δx| — шагаем по y)\n"
                f"k = Δx/Δy = {k:.6f}\nb = x1 − k·y1 = {b:.6f}\n"
                f"Шаг по y от {y1} до {y2} (шаг {st}); x = round(k·y + b)") if log is not None else ""
        for i, y in enumerate(range(y1, y2 + st, st)):
            xf = k * y + b
            x = rnd(xf)
            pts.append((x, y))
            if log is not None:
                log.append((i, y, f"{xf:.4f}", x))
    return pts, hdr, info


def dda(x1, y1, x2, y2, log=None):
    """ЦДА: L = max(|Δx|,|Δy|), приращения Δx/L и Δy/L."""
    dx, dy = x2 - x1, y2 - y1
    L = max(abs(dx), abs(dy))
    sx = dx / L if L else 0.0
    sy = dy / L if L else 0.0
    x, y = float(x1), float(y1)
    pts = []
    for i in range(L + 1):
        px, py = rnd(x), rnd(y)
        pts.append((px, py))
        if log is not None:
            log.append((i, f"{x:.4f}", f"{y:.4f}", px, py))
        x += sx
        y += sy
    info = (f"Δx = {dx}, Δy = {dy}\nL = max(|Δx|,|Δy|) = {L}\n"
            f"шаг по x: Δx/L = {sx:.6f}\nшаг по y: Δy/L = {sy:.6f}\n"
            f"Точки: (round(x), round(y)), затем x += Δx/L, y += Δy/L") if log is not None else ""
    return pts, ["№", "x", "y", "X = round(x)", "Y = round(y)"], info


def bresenham(x1, y1, x2, y2, log=None):
    """Брезенхем для отрезка (целочисленный, все октанты)."""
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = 1 if x2 >= x1 else -1
    sy = 1 if y2 >= y1 else -1
    steep = dy > dx
    if steep:
        dx, dy = dy, dx
    e = 2 * dy - dx
    e0 = e
    x, y = x1, y1
    pts = []
    for i in range(dx + 1):
        pts.append((x, y))
        if log is not None:
            log.append((i, x, y, e, "диагональный шаг" if e >= 0 else "шаг по главной оси"))
        if e >= 0:
            if steep:
                x += sx
            else:
                y += sy
            e -= 2 * dx
        if steep:
            y += sy
        else:
            x += sx
        e += 2 * dy
    info = (f"|Δx| = {abs(x2 - x1)}, |Δy| = {abs(y2 - y1)}; "
            f"{'главная ось — y (отрезок крутой)' if steep else 'главная ось — x'}\n"
            f"Δmax = {dx}, Δmin = {dy}\n"
            f"e₀ = 2·Δmin − Δmax = {e0}\n"
            f"Если e ≥ 0: шаг и по главной, и по второй оси, e −= 2·Δmax;\n"
            f"всегда: шаг по главной оси, e += 2·Δmin") if log is not None else ""
    return pts, ["№", "x", "y", "e", "следующий шаг"], info


def bresenham_circle(cx, cy, r, log=None):
    """Брезенхем (midpoint) для окружности, симметрия по 8 октантам."""
    r = max(0, r)
    x, y, d = 0, r, 1 - r
    pts, seen = [], set()
    i = 0
    while x <= y:
        for px, py in ((x, y), (y, x), (-x, y), (-y, x),
                       (x, -y), (y, -x), (-x, -y), (-y, -x)):
            p = (cx + px, cy + py)
            if p not in seen:
                seen.add(p)
                pts.append(p)
        if log is not None:
            log.append((i, x, y, d, f"({cx + x}, {cy + y})",
                        "d < 0: y не меняется" if d < 0 else "d ≥ 0: y := y − 1"))
        if d < 0:
            d += 2 * x + 3
        else:
            d += 2 * (x - y) + 5
            y -= 1
        x += 1
        i += 1
    info = (f"Центр ({cx}, {cy}), R = {r}\n"
            f"Старт: x = 0, y = R = {r}, d₀ = 1 − R = {1 - r}\n"
            f"d < 0: d += 2x + 3;  d ≥ 0: d += 2(x − y) + 5, y −= 1;  затем x += 1\n"
            f"Считается 1-й октант (x ≤ y), остальные 7 получаются симметрией:\n"
            f"(±x,±y), (±y,±x) относительно центра") if log is not None else ""
    return pts, ["№", "x", "y", "d", "точка (1-й октант)", "решение"], info


def castle_pitteway(x1, y1, x2, y2, log=None):
    """
    Алгоритм Кастла–Питвея (Brons/Pitteway): отрезок задаётся строкой шагов
    'x' (шаг по главной оси) и 'd' (диагональный шаг). Строка строится как
    «алгоритм Евклида» над строками:
        m1 = a − b (число 'x'),  m2 = b (число 'd'),  s1 = 'x', s2 = 'd'
        пока m1 ≠ m2:
            если m1 > m2: m1 −= m2; s2 = s1 + s2
            иначе:        m2 −= m1; s1 = s1 + s2
        период = s1 + s2 (при m1 = m2 = 1), затем циклический сдвиг
        (чтобы округлять к ближайшему пикселю), и повтор g = НОД(a, b) раз
    Умножений и делений в цикле растеризации нет — только копирование строки.
    """
    dx, dy = x2 - x1, y2 - y1
    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    a, b = abs(dx), abs(dy)
    steep = b > a
    if steep:
        a, b = b, a
    trace = []
    g = 1
    if b == 0 or a == b:
        s = ('x' if b == 0 else 'd') * a
    else:
        g = math.gcd(a, b)
        aa, bb = a // g, b // g
        m1, m2 = aa - bb, bb
        s1, s2 = 'x', 'd'
        if log is not None:
            trace.append(f"старт: m1={m1}, m2={m2}, s1='{s1}', s2='{s2}'")
        while m1 != m2:
            if m1 > m2:
                m1 -= m2
                s2 = s1 + s2
            else:
                m2 -= m1
                s1 = s1 + s2
            if log is not None:
                trace.append(f"m1={m1}, m2={m2}, s1='{s1[:30]}', s2='{s2[:30]}'")
        # Построенная строка (s1+s2) — «нижняя» аппроксимация (путь идёт под прямой).
        # Чтобы получить округление к ближайшему пикселю, строку циклически
        # сдвигаем на t: t·b' ≡ ⌊a'/2⌋ (mod a'), где a' = a/НОД, b' = b/НОД.
        w = s1 + s2
        shift = (aa // 2) * pow(bb, -1, aa) % aa
        w = w[shift:] + w[:shift]
        s = w * g
        if log is not None:
            trace.append(f"период (до сдвига): {s1 + s2}; сдвиг t = {shift}; период после сдвига: {w}")
    x, y = x1, y1
    pts = [(x, y)]
    if log is not None:
        log.append((0, "старт", x, y))
    for i, ch in enumerate(s, 1):
        if steep:
            y += sy
            if ch == 'd':
                x += sx
        else:
            x += sx
            if ch == 'd':
                y += sy
        pts.append((x, y))
        if log is not None:
            log.append((i, ch, x, y))
    info = ""
    if log is not None:
        info = (f"a = {a} (главная ось), b = {b}, НОД = {g}\n"
                f"'x' — шаг по главной оси, 'd' — диагональный шаг\n"
                + "\n".join(trace)
                + f"\nИтоговая строка шагов: {s[:120]}{'…' if len(s) > 120 else ''}")
    return pts, ["№", "шаг", "x", "y"], info


def wu(x1, y1, x2, y2, log=None):
    """Сглаженный отрезок по Ву: два пикселя в каждом столбце, яркость ~ покрытию."""
    steep = abs(y2 - y1) > abs(x2 - x1)
    if steep:
        x1, y1, x2, y2 = y1, x1, y2, x2
    if x1 > x2:
        x1, x2, y1, y2 = x2, x1, y2, y1
    dx, dy = x2 - x1, y2 - y1
    grad = dy / dx if dx else 0.0
    y = float(y1)
    pts = []
    for i, x in enumerate(range(x1, x2 + 1)):
        yi = math.floor(y)
        f = y - yi
        if steep:
            pts.append((yi, x, 1 - f))
            if f > 1e-9:
                pts.append((yi + 1, x, f))
        else:
            pts.append((x, yi, 1 - f))
            if f > 1e-9:
                pts.append((x, yi + 1, f))
        if log is not None:
            log.append((i, x, f"{y:.4f}", yi, f"{1 - f:.3f}", yi + 1, f"{f:.3f}"))
        y += grad
    info = (f"{'Крутой отрезок: оси x и y поменяны местами' if steep else 'Пологий отрезок'}\n"
            f"Градиент k = {grad:.6f}\n"
            f"В каждом шаге: y_int = ⌊y⌋, f = y − y_int;\n"
            f"пиксель (x, y_int) получает яркость 1 − f, пиксель (x, y_int+1) — f.\n"
            f"Яркость 1 = полностью закрашен, 0 = фон (на экране — прозрачность)."
            f"\n(таблица — в преобразованных координатах)") if log is not None else ""
    return pts, ["№", "x*", "y*", "y_int", "I(y_int)", "y_int+1", "I(y_int+1)"], info


ALGOS = {
    "Пошаговый алгоритм": (step_by_step, "line"),
    "Алгоритм ЦДА (DDA)": (dda, "line"),
    "Алгоритм Брезенхема (отрезок)": (bresenham, "line"),
    "Алгоритм Брезенхема (окружность)": (bresenham_circle, "circle"),
    "Кастл–Питвей (доп.)": (castle_pitteway, "line"),
    "Сглаживание Ву (доп.)": (wu, "line"),
}

MAPPING_TEXT = (
    "Привязка к сетке: пиксель с целыми координатами (x, y) — это клетка, "
    "ЦЕНТР которой лежит в точке (x, y). Линии сетки проходят через полуцелые "
    "значения (x ± 0,5; y ± 0,5), а подписи на осях стоят под центрами клеток. "
    "Красная линия — идеальная геометрическая фигура; растровый алгоритм "
    "выбирает клетки, центры которых ближе всего к ней."
)


# ======================================================================
#                          ПОЛОТНО С СЕТКОЙ
# ======================================================================

class Canvas(QWidget):
    scaleChanged = pyqtSignal(float)
    hovered = pyqtSignal(str)

    PIXEL_COLOR = QColor(46, 110, 230)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.scale = 30.0
        self.off = QPointF(0, 0)       # смещение начала координат от центра виджета (px)
        self.pts = []
        self.ideal = None
        self.has_alpha = False
        self.show_ideal = True
        self.show_grid = True
        self.show_nums = True
        self._drag = None

    # --- преобразования координат
    def sx(self, x):
        return self.width() / 2 + self.off.x() + x * self.scale

    def sy(self, y):
        return self.height() / 2 + self.off.y() - y * self.scale

    def to_world(self, px, py):
        return ((px - self.width() / 2 - self.off.x()) / self.scale,
                -(py - self.height() / 2 - self.off.y()) / self.scale)

    # --- управление
    def set_data(self, pts, ideal):
        self.pts = pts
        self.ideal = ideal
        self.has_alpha = any(len(p) > 2 for p in pts)
        self.fit()

    def fit(self):
        if self.pts:
            xs = [p[0] for p in self.pts]
            ys = [p[1] for p in self.pts]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            w = max(x1 - x0 + 5, 8)
            h = max(y1 - y0 + 5, 8)
            s = max(4.0, min(90.0, min(self.width() / w, self.height() / h)))
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            self.scale = s
            self.off = QPointF(-cx * s, cy * s)
            self.scaleChanged.emit(s)
        self.update()

    def reset_view(self):
        self.off = QPointF(0, 0)
        self.update()

    def zoom_to(self, new_scale, anchor=None):
        new_scale = max(4.0, min(120.0, float(new_scale)))
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)
        wx, wy = self.to_world(anchor.x(), anchor.y())
        self.scale = new_scale
        self.off = QPointF(anchor.x() - self.width() / 2 - wx * new_scale,
                           anchor.y() - self.height() / 2 + wy * new_scale)
        self.scaleChanged.emit(new_scale)
        self.update()

    # --- события мыши
    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.zoom_to(self.scale * f, e.position())

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.position()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._drag is not None and (e.buttons() & Qt.MouseButton.LeftButton):
            self.off += pos - self._drag
            self._drag = pos
            self.update()
        wx, wy = self.to_world(pos.x(), pos.y())
        self.hovered.emit(f"Клетка: ({rnd(wx)}, {rnd(wy)})    "
                          f"точка плоскости: ({wx:.2f}; {wy:.2f})")

    # --- отрисовка
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h, s = self.width(), self.height(), self.scale
        p.fillRect(self.rect(), QColor(255, 255, 255))

        xmin = math.floor(self.to_world(0, 0)[0]) - 1
        xmax = math.ceil(self.to_world(w, 0)[0]) + 1
        ymax = math.ceil(self.to_world(0, 0)[1]) + 1
        ymin = math.floor(self.to_world(0, h)[1]) - 1

        # линии сетки (на границах клеток: k + 0.5)
        if self.show_grid and s >= 6:
            p.setPen(QPen(QColor(210, 210, 210), 1))
            for k in range(xmin, xmax + 1):
                X = self.sx(k + 0.5)
                p.drawLine(QPointF(X, 0), QPointF(X, h))
            for k in range(ymin, ymax + 1):
                Y = self.sy(k + 0.5)
                p.drawLine(QPointF(0, Y), QPointF(w, Y))

        # пиксели
        p.setPen(Qt.PenStyle.NoPen)
        for pt in self.pts:
            x, y = pt[0], pt[1]
            if x < xmin or x > xmax or y < ymin or y > ymax:
                continue
            c = QColor(self.PIXEL_COLOR)
            if len(pt) > 2:
                c.setAlpha(max(0, min(255, int(pt[2] * 255))))
            p.setBrush(QBrush(c))
            p.drawRect(QRectF(self.sx(x) - s / 2, self.sy(y) - s / 2, s, s))

        # порядковые номера пикселей
        if self.show_nums and not self.has_alpha and s >= 30 and len(self.pts) <= 400:
            f = QFont(p.font())
            f.setPointSizeF(max(6.0, min(11.0, s / 4)))
            p.setFont(f)
            p.setPen(QColor(255, 255, 255))
            for i, pt in enumerate(self.pts):
                r = QRectF(self.sx(pt[0]) - s / 2, self.sy(pt[1]) - s / 2, s, s)
                p.drawText(r, Qt.AlignmentFlag.AlignCenter, str(i))
            p.setFont(QFont())

        # идеальная фигура
        if self.show_ideal and self.ideal:
            p.setPen(QPen(QColor(225, 40, 40), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            if self.ideal[0] == "line":
                _, a, b, c, d = self.ideal
                p.drawLine(QPointF(self.sx(a), self.sy(b)), QPointF(self.sx(c), self.sy(d)))
            else:
                _, a, b, r = self.ideal
                p.drawEllipse(QPointF(self.sx(a), self.sy(b)), r * s, r * s)

        # оси
        ax, ay = self.sx(0), self.sy(0)
        p.setPen(QPen(QColor(0, 0, 0), 1.6))
        p.drawLine(QPointF(0, ay), QPointF(w, ay))
        p.drawLine(QPointF(ax, 0), QPointF(ax, h))
        p.setBrush(QColor(0, 0, 0))
        ayc = min(max(ay, 8), h - 8)
        axc = min(max(ax, 8), w - 8)
        p.drawPolygon(QPolygonF([QPointF(w - 2, ayc), QPointF(w - 12, ayc - 4), QPointF(w - 12, ayc + 4)]))
        p.drawPolygon(QPolygonF([QPointF(axc, 2), QPointF(axc - 4, 12), QPointF(axc + 4, 12)]))

        # подписи
        fm = p.fontMetrics()
        step = 1
        for st in (1, 2, 5, 10, 20, 50, 100, 200, 500):
            step = st
            if st * s >= 34:
                break
        p.setPen(QColor(0, 0, 0))
        ly = min(max(ay + 15, 14), h - 4)
        for k in range(xmin, xmax + 1):
            if k % step == 0:
                X = self.sx(k)
                if s >= 10:
                    p.drawLine(QPointF(X, ay - 3), QPointF(X, ay + 3))
                if k != 0:
                    t = str(k)
                    p.drawText(QPointF(X - fm.horizontalAdvance(t) / 2, ly), t)
        for k in range(ymin, ymax + 1):
            if k % step == 0:
                Y = self.sy(k)
                if s >= 10:
                    p.drawLine(QPointF(ax - 3, Y), QPointF(ax + 3, Y))
                if k != 0:
                    t = str(k)
                    tw = fm.horizontalAdvance(t)
                    tx = min(max(ax - 7 - tw, 2), w - tw - 2)
                    p.drawText(QPointF(tx, Y + 4), t)
        t = "0"
        p.drawText(QPointF(min(max(ax - 7 - fm.horizontalAdvance(t), 2), w - 14), ly), t)
        bf = QFont(p.font())
        bf.setBold(True)
        p.setFont(bf)
        p.drawText(QPointF(w - 16, max(ayc - 8, 12)), "x")
        p.drawText(QPointF(min(axc + 10, w - 12), 14), "y")
        p.end()


# ======================================================================
#                          ГЛАВНОЕ ОКНО
# ======================================================================

class MainWindow(QMainWindow):
    BENCH_SIZES = [10, 100, 1000, 10000]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа 3 — Базовые растровые алгоритмы")
        self.resize(1350, 880)
        self.bench_text = ""
        self.cur_report = ""

        self.canvas = Canvas()
        self.canvas.hovered.connect(lambda t: self.statusBar().showMessage(t))
        self.canvas.scaleChanged.connect(self.sync_slider)

        # ---------- левая панель ----------
        self.combo = QComboBox()
        self.combo.addItems(ALGOS.keys())
        self.combo.currentTextChanged.connect(self.on_algo_changed)

        def spin(v, lo=-500, hi=500):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(v)
            return s

        self.x1, self.y1 = spin(-6), spin(-3)
        self.x2, self.y2 = spin(8), spin(5)
        self.rr = spin(8, 0, 500)
        self.lbl1 = QLabel("Начало X1:")
        self.lbl2 = QLabel("Начало Y1:")
        self.lbl3 = QLabel("Конец X2:")
        self.lbl4 = QLabel("Конец Y2:")
        self.lbl5 = QLabel("Радиус R:")
        form = QFormLayout()
        for lab, w in ((self.lbl1, self.x1), (self.lbl2, self.y1), (self.lbl3, self.x2),
                       (self.lbl4, self.y2), (self.lbl5, self.rr)):
            form.addRow(lab, w)
        g_par = QGroupBox("Параметры (целые координаты)")
        g_par.setLayout(form)

        g_alg = QGroupBox("Алгоритм")
        la = QVBoxLayout()
        la.addWidget(self.combo)
        g_alg.setLayout(la)

        btn_draw = QPushButton("Построить")
        btn_draw.setStyleSheet("font-weight:bold; padding:6px;")
        btn_draw.clicked.connect(self.run)
        btn_fit = QPushButton("Вписать фигуру в окно")
        btn_fit.clicked.connect(lambda: self.canvas.fit())
        btn_origin = QPushButton("К началу координат")
        btn_origin.clicked.connect(self.canvas.reset_view)
        btn_save = QPushButton("Сохранить отчёт (.txt)")
        btn_save.clicked.connect(self.save_report)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(4, 120)
        self.slider.setValue(30)
        self.slider.valueChanged.connect(lambda v: self.canvas.zoom_to(v))
        self.scale_lbl = QLabel("Масштаб: 30 px/клетка (колесо мыши; ЛКМ — сдвиг)")
        self.scale_lbl.setWordWrap(True)

        chk_ideal = QCheckBox("Идеальная линия/окружность")
        chk_ideal.setChecked(True)
        chk_ideal.toggled.connect(lambda v: self._set("show_ideal", v))
        chk_grid = QCheckBox("Линии сетки")
        chk_grid.setChecked(True)
        chk_grid.toggled.connect(lambda v: self._set("show_grid", v))
        chk_num = QCheckBox("Номера пикселей (при крупном масштабе)")
        chk_num.setChecked(True)
        chk_num.toggled.connect(lambda v: self._set("show_nums", v))

        g_view = QGroupBox("Вид")
        lv = QVBoxLayout()
        for w in (self.scale_lbl, self.slider, chk_ideal, chk_grid, chk_num, btn_fit, btn_origin):
            lv.addWidget(w)
        g_view.setLayout(lv)

        expl = QLabel(MAPPING_TEXT)
        expl.setWordWrap(True)
        expl.setStyleSheet("color:#333; background:#f4f6fa; padding:6px; border:1px solid #ccd;")

        left = QWidget()
        ll = QVBoxLayout(left)
        for w in (g_alg, g_par, btn_draw, g_view, btn_save, expl):
            ll.addWidget(w)
        ll.addStretch(1)
        left.setFixedWidth(310)

        # ---------- нижние вкладки ----------
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(170)
        self.info.setFont(QFont("Consolas", 10))
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        tab1 = QWidget()
        t1 = QVBoxLayout(tab1)
        t1.addWidget(self.info)
        t1.addWidget(self.table)

        self.bench_table = QTableWidget()
        self.bench_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        btn_bench = QPushButton("Измерить время всех алгоритмов")
        btn_bench.clicked.connect(self.run_benchmark)
        self.bench_lbl = QLabel(
            "Время одного вызова функции растеризации (без отрисовки и логирования), "
            "мкс, усреднено по многим запускам. Отрезок: (0,0)→(n, 0.6n); окружность: R = n.")
        self.bench_lbl.setWordWrap(True)
        tab2 = QWidget()
        t2 = QVBoxLayout(tab2)
        t2.addWidget(self.bench_lbl)
        t2.addWidget(btn_bench)
        t2.addWidget(self.bench_table)

        tabs = QTabWidget()
        tabs.addTab(tab1, "Сопроводительные вычисления")
        tabs.addTab(tab2, "Временные характеристики")

        vs = QSplitter(Qt.Orientation.Vertical)
        vs.addWidget(self.canvas)
        vs.addWidget(tabs)
        vs.setStretchFactor(0, 3)
        vs.setStretchFactor(1, 2)

        main = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(left)
        main.addWidget(vs)
        main.setStretchFactor(1, 1)
        self.setCentralWidget(main)

        self.on_algo_changed(self.combo.currentText())
        QTimer.singleShot(80, self.run)

    # ---------- служебное ----------
    def _set(self, name, val):
        setattr(self.canvas, name, val)
        self.canvas.update()

    def sync_slider(self, s):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(s)))
        self.slider.blockSignals(False)
        self.scale_lbl.setText(f"Масштаб: {s:.0f} px/клетка (колесо мыши; ЛКМ — сдвиг)")

    def on_algo_changed(self, name):
        kind = ALGOS[name][1]
        circle = kind == "circle"
        self.lbl1.setText("Центр X:" if circle else "Начало X1:")
        self.lbl2.setText("Центр Y:" if circle else "Начало Y1:")
        for w in (self.lbl3, self.x2, self.lbl4, self.y2):
            w.setEnabled(not circle)
        for w in (self.lbl5, self.rr):
            w.setEnabled(circle)

    # ---------- построение ----------
    def run(self):
        name = self.combo.currentText()
        func, kind = ALGOS[name]
        if kind == "line":
            args = (self.x1.value(), self.y1.value(), self.x2.value(), self.y2.value())
            ideal = ("line",) + args
        else:
            args = (self.x1.value(), self.y1.value(), self.rr.value())
            ideal = ("circle",) + args

        log = []
        pts, hdr, info = func(*args, log=log)

        reps = 300
        t0 = perf_counter()
        for _ in range(reps):
            func(*args)
        us = (perf_counter() - t0) / reps * 1e6

        self.canvas.set_data(pts, ideal)

        head = f"{name}\nПараметры: {args}\n" + "-" * 50 + "\n"
        tail = (f"\n{'-' * 50}\nВсего пикселей (клеток): {len(pts)}\n"
                f"Время работы алгоритма: {us:.2f} мкс "
                f"(среднее по {reps} запускам, без вывода и логирования)")
        self.info.setPlainText(head + info + tail)

        shown = log[:5000]
        self.table.clear()
        self.table.setColumnCount(len(hdr))
        self.table.setHorizontalHeaderLabels(hdr)
        self.table.setRowCount(len(shown))
        for r, row in enumerate(shown):
            for c, v in enumerate(row):
                it = QTableWidgetItem(str(v))
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, c, it)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        lines = ["\t".join(map(str, hdr))] + ["\t".join(map(str, r)) for r in log]
        self.cur_report = head + info + tail + "\n\nТаблица вычислений:\n" + "\n".join(lines)

    # ---------- замеры времени ----------
    def run_benchmark(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            names = list(ALGOS.keys())
            self.bench_table.clear()
            self.bench_table.setRowCount(len(names))
            self.bench_table.setColumnCount(len(self.BENCH_SIZES))
            self.bench_table.setHorizontalHeaderLabels([f"n = {n}" for n in self.BENCH_SIZES])
            self.bench_table.setVerticalHeaderLabels(names)
            lines = ["Алгоритм\t" + "\t".join(f"n={n}" for n in self.BENCH_SIZES)]
            for r, name in enumerate(names):
                func, kind = ALGOS[name]
                row = [name]
                for c, n in enumerate(self.BENCH_SIZES):
                    args = (0, 0, n, int(n * 0.6)) if kind == "line" else (0, 0, n)
                    reps = max(5, 100000 // n)
                    t0 = perf_counter()
                    for _ in range(reps):
                        func(*args)
                    us = (perf_counter() - t0) / reps * 1e6
                    txt = f"{us:.1f} мкс"
                    it = QTableWidgetItem(txt)
                    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.bench_table.setItem(r, c, it)
                    row.append(txt)
                    QApplication.processEvents()
                lines.append("\t".join(row))
            self.bench_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.bench_text = "\n".join(lines)
        finally:
            QApplication.restoreOverrideCursor()

    # ---------- отчёт ----------
    def save_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчёт", "report_lab3.txt",
                                              "Текст (*.txt)")
        if not path:
            return
        parts = ["ЛАБОРАТОРНАЯ РАБОТА 3. БАЗОВЫЕ РАСТРОВЫЕ АЛГОРИТМЫ", "",
                 MAPPING_TEXT, "", self.cur_report]
        if self.bench_text:
            parts += ["", "Временные характеристики (мкс на один вызов):", self.bench_text]
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(parts))
            self.statusBar().showMessage(f"Отчёт сохранён: {path}", 5000)
        except OSError as e:
            QMessageBox.warning(self, "Ошибка", str(e))


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()