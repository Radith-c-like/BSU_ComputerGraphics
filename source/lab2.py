"""
Лабораторная работа 2, вариант 8.

1. Высокочастотные фильтры (увеличение резкости):
     - Лапласиан гауссиана (LoG):      g = f - a * sigma^2 * Лапласиан(Гаусс(f))
     - Нерезкое маскирование (Unsharp): g = f + a * (f - Гаусс(f))
2. Локальная пороговая обработка (сравнение двух методов):
     - Бернсен:  T(x,y) = (min + max) / 2 в окне; если контраст (max-min) < порога -
                 область однородна и классифицируется по (min+max)/2 >= 128
     - Ниблэк:   T(x,y) = m(x,y) + k * s(x,y), где m, s - среднее и СКО в окне
"""
import os
import time

import sys

import cv2
import numpy as np

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

from PyQt6 import QtCore, QtGui, QtWidgets


# ====================================================================
#                          Алгоритмы
# ====================================================================
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def clip8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def log_sharpen(img, sigma, amount):
    """Лапласиан гауссиана. Сначала сглаживаем (Гаусс), затем берём Лапласиан;
    вычитание Лапласиана усиливает границы. Множитель sigma^2 нормирует отклик,
    чтобы 'amount' примерно не зависел от sigma."""
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), sigma)
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=1)  # ядро [[0,1,0],[1,-4,1],[0,1,0]]
    return clip8(f - amount * sigma ** 2 * lap)


def unsharp_mask(img, sigma, amount, thr):
    """Нерезкое маскирование: к оригиналу добавляется 'маска' f - Гаусс(f).
    thr - маска применяется только там, где |маска| >= thr (не усиливает слабый шум)."""
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), sigma)
    mask = f - blur
    if thr > 0:
        mask = np.where(np.abs(mask) >= thr, mask, 0)
    return clip8(f + amount * mask)


def bernsen(gray, win, c_min):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    mn = cv2.erode(gray, k, borderType=cv2.BORDER_REFLECT).astype(np.float32)
    mx = cv2.dilate(gray, k, borderType=cv2.BORDER_REFLECT).astype(np.float32)
    mid = (mn + mx) / 2.0
    contrast = mx - mn
    g = gray.astype(np.float32)
    uniform = np.where(mid >= 128, 255, 0)
    out = np.where(contrast < c_min, uniform, np.where(g > mid, 255, 0))
    return out.astype(np.uint8)


def niblack(gray, win, k):
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (win, win), borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(g * g, -1, (win, win), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    return np.where(g > mean + k * std, 255, 0).astype(np.uint8)


def sharpness(img):
    """Простая мера резкости: дисперсия Лапласиана полутонового изображения."""
    return float(cv2.Laplacian(to_gray(img), cv2.CV_64F).var())


# ====================================================================
#                   Генерация базы тестовых изображений
# ====================================================================
def imwrite(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    cv2.imencode(ext, img)[1].tofile(path)


def make_test_images(folder):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(1)
    h, w = 360, 640
    doc = np.full((h, w), 235, np.uint8)
    for i, t in enumerate(["Local thresholding", "Niblack vs Bernsen", "Sharpen: LoG / USM"]):
        cv2.putText(doc, t, (25, 90 + i * 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (30,), 2, cv2.LINE_AA)
    cv2.circle(doc, (560, 60), 35, (40,), -1)
    cv2.rectangle(doc, (520, 250), (610, 330), (60,), -1)

    yy, xx = np.mgrid[0:h, 0:w]
    light = 0.25 + 0.75 * (xx / w) * (0.6 + 0.4 * (1 - yy / h))
    f = doc.astype(np.float64)
    blur = lambda a, s: cv2.GaussianBlur(a, (0, 0), s)
    items = {
        "01_document_clean": f,
        "02_uneven_light": f * light,
        "03_uneven_light_noise": f * light + rng.normal(0, 12, f.shape),
        "04_blurred": blur(f, 2.5),
        "05_blurred_uneven": blur(f, 2.0) * light,
        "06_blurred_noise": blur(f, 2.0) + rng.normal(0, 8, f.shape),
        "07_low_contrast": 110 + (f - 30) * 0.3,
    }
    for name, arr in items.items():
        imwrite(os.path.join(folder, name + ".png"), clip8(arr))
    return len(items)


# ====================================================================
#                          Интерфейс (PyQt6)
# ====================================================================
def bgr_to_qimage(img):
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h, w, _ = rgb.shape
    data = rgb.tobytes()
    return QtGui.QImage(data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()


class ImageView(QtWidgets.QWidget):
    """Панель: заголовок, изображение (с сохранением пропорций) и подпись."""

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.title = QtWidgets.QLabel("")
        self.title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.img = QtWidgets.QLabel("Нет изображения")
        self.img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.img.setMinimumSize(120, 120)
        self.img.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        self.img.setStyleSheet("background:#2b2b2b; color:#aaa;")
        self.info = QtWidgets.QLabel("")
        self.info.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.info.setMinimumHeight(60)
        lay.addWidget(self.title)
        lay.addWidget(self.img, 1)
        lay.addWidget(self.info)
        self._pix = None

    def set_content(self, title, img, info):
        self.title.setText(title)
        self.info.setText(info)
        self._pix = QtGui.QPixmap.fromImage(bgr_to_qimage(img))
        self._rescale()

    def _rescale(self):
        if self._pix is not None:
            self.img.setPixmap(self._pix.scaled(
                self.img.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()


class SliderRow(QtWidgets.QWidget):
    """Подпись со значением + горизонтальный ползунок с вещественным шагом."""
    changed = QtCore.pyqtSignal()

    def __init__(self, text, lo, hi, init, step, fmt="{:.1f}"):
        super().__init__()
        self.text, self.lo, self.step, self.fmt = text, lo, step, fmt
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(0)
        self.label = QtWidgets.QLabel()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, int(round((hi - lo) / step)))
        self.slider.setValue(int(round((init - lo) / step)))
        lay.addWidget(self.label)
        lay.addWidget(self.slider)
        self.slider.valueChanged.connect(self._on_change)
        self._update_label()

    def value(self):
        return round(self.lo + self.slider.value() * self.step, 4)

    def _update_label(self):
        self.label.setText(f"{self.text}: {self.fmt.format(self.value())}")

    def _on_change(self, _):
        self._update_label()
        self.changed.emit()


def group_label(text):
    lbl = QtWidgets.QLabel(text)
    lbl.setStyleSheet("font-weight: bold; margin-top: 10px;")
    return lbl


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная №2 - Вариант 8 (Обработка изображений)")
        self.resize(1400, 780)
        self.cv_img = None
        self.results = []

        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.run)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QWidget()
        left.setFixedWidth(320)
        ll = QtWidgets.QVBoxLayout(left)
        title = QtWidgets.QLabel("Методы\nВариант 8")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        ll.addWidget(title)
        b_load = QtWidgets.QPushButton("📂 Загрузить фото")
        b_base = QtWidgets.QPushButton("🧪 Создать тестовую базу")
        b_load.clicked.connect(self.load_image)
        b_base.clicked.connect(self.make_base)
        ll.addWidget(b_load)
        ll.addWidget(b_base)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_sharpen_tab(), "Резкость")
        self.tabs.addTab(self._build_threshold_tab(), "Локальный порог")
        self.tabs.currentChanged.connect(self.run)
        ll.addWidget(self.tabs, 1)

        b_save = QtWidgets.QPushButton("💾 Сохранить результат")
        b_save.setStyleSheet("background:#2e7d32; color:white; padding:6px;")
        b_save.clicked.connect(self.save_image)
        ll.addWidget(b_save)

        right = QtWidgets.QHBoxLayout()
        self.views = [ImageView() for _ in range(3)]
        for v in self.views:
            right.addWidget(v, 1)

        root.addWidget(left)
        root.addLayout(right, 1)

    def _slider(self, *args, **kw):
        s = SliderRow(*args, **kw)
        s.changed.connect(self.timer.start)
        return s

    def _build_sharpen_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(group_label("Лапласиан гауссиана (LoG)"))
        self.s_log_sigma = self._slider("σ", 0.5, 5, 1.5, 0.1)
        self.s_log_amount = self._slider("Сила", 0, 5, 1.0, 0.1)
        lay.addWidget(self.s_log_sigma)
        lay.addWidget(self.s_log_amount)
        lay.addWidget(group_label("Unsharp Masking"))
        self.s_us_sigma = self._slider("σ", 0.5, 10, 1.5, 0.1)
        self.s_us_amount = self._slider("Сила", 0, 5, 1.5, 0.1)
        self.s_us_thr = self._slider("Порог |маски|", 0, 50, 0, 1, "{:.0f}")
        for s in (self.s_us_sigma, self.s_us_amount, self.s_us_thr):
            lay.addWidget(s)
        lay.addStretch(1)
        return w

    def _build_threshold_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.s_win = self._slider("Размер окна", 3, 101, 25, 2, "{:.0f}")
        lay.addWidget(self.s_win)
        lay.addWidget(group_label("Бернсен"))
        self.s_bern = self._slider("Мин. контраст", 0, 100, 15, 1, "{:.0f}")
        lay.addWidget(self.s_bern)
        lay.addWidget(group_label("Ниблэк"))
        self.s_nib = self._slider("k", -1.0, 1.0, -0.2, 0.05, "{:.2f}")
        lay.addWidget(self.s_nib)
        lay.addStretch(1)
        return w

    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Выберите изображение", "",
            "Изображения (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QtWidgets.QMessageBox.critical(self, "Ошибка", "Не удалось открыть файл.")
            return
        self.cv_img = img
        self.run()

    def make_base(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Куда сохранить тестовую базу")
        if not folder:
            return
        folder = os.path.join(folder, "test_images")
        n = make_test_images(folder)
        QtWidgets.QMessageBox.information(self, "Готово", f"Создано {n} изображений:\n{folder}")

    def save_image(self):
        if not self.results:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала обработайте изображение!")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить результат", "result.png", "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        stem, ext = os.path.splitext(path)
        names = ["_log", "_usm"] if self.tabs.currentIndex() == 0 else ["_bernsen", "_niblack"]
        for name, img in zip(names, self.results):
            imwrite(stem + name + (ext or ".png"), img)
        QtWidgets.QMessageBox.information(
            self, "Успех", f"Сохранено 2 файла:\n{stem}{names[0]}{ext}\n{stem}{names[1]}{ext}")

    def run(self, *_):
        if self.cv_img is None:
            return
        img = self.cv_img
        orig_info = f"Резкость (var Лапласиана): {sharpness(img):.1f}"
        if self.tabs.currentIndex() == 0:
            ls, la = self.s_log_sigma.value(), self.s_log_amount.value()
            us, ua, ut = self.s_us_sigma.value(), self.s_us_amount.value(), self.s_us_thr.value()
            t0 = time.perf_counter()
            r1 = log_sharpen(img, ls, la)
            t1 = time.perf_counter()
            r2 = unsharp_mask(img, us, ua, ut)
            t2 = time.perf_counter()
            panels = [
                ("Оригинал", img, orig_info),
                (f"LoG (σ={ls}, a={la})", r1, f"Резкость: {sharpness(r1):.1f}\n{(t1 - t0) * 1000:.1f} мс"),
                (f"Unsharp Masking (σ={us}, a={ua})", r2,
                 f"Резкость: {sharpness(r2):.1f}\n{(t2 - t1) * 1000:.1f} мс"),
            ]
        else:
            win = int(self.s_win.value()) | 1
            c = int(self.s_bern.value())
            k = self.s_nib.value()
            gray = to_gray(img)
            t0 = time.perf_counter()
            b1 = bernsen(gray, win, c)
            t1 = time.perf_counter()
            b2 = niblack(gray, win, k)
            t2 = time.perf_counter()
            bgr = lambda b: cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
            r1, r2 = bgr(b1), bgr(b2)
            panels = [
                ("Оригинал", img, f"Окно: {win}×{win}"),
                (f"Бернсен (контраст ≥ {c})", r1,
                 f"Белых: {100 * np.mean(b1 > 0):.1f}%\n{(t1 - t0) * 1000:.1f} мс"),
                (f"Ниблэк (k={k})", r2,
                 f"Белых: {100 * np.mean(b2 > 0):.1f}%\n{(t2 - t1) * 1000:.1f} мс\n"
                 f"Отличие от Бернсена: {100 * np.mean(b1 != b2):.1f}% пикс."),
            ]
        self.results = [r1, r2]
        for view, (title, im, info) in zip(self.views, panels):
            view.set_content(title, im, info)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()