import sys
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def rgb_to_cmyk(r: int, g: int, b: int):
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    k = 1.0 - max(r_norm, g_norm, b_norm)
    if k == 1.0:
        return 0.0, 0.0, 0.0, 100.0
    c = (1.0 - r_norm - k) / (1.0 - k)
    m = (1.0 - g_norm - k) / (1.0 - k)
    y = (1.0 - b_norm - k) / (1.0 - k)
    return c * 100.0, m * 100.0, y * 100.0, k * 100.0


def cmyk_to_rgb(c: float, m: float, y: float, k: float):
    c_norm, m_norm, y_norm, k_norm = c / 100.0, m / 100.0, y / 100.0, k / 100.0
    r = round(255.0 * (1.0 - c_norm) * (1.0 - k_norm))
    g = round(255.0 * (1.0 - m_norm) * (1.0 - k_norm))
    b = round(255.0 * (1.0 - y_norm) * (1.0 - k_norm))
    return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))


def rgb_to_hls(r: int, g: int, b: int):
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    max_c = max(r_norm, g_norm, b_norm)
    min_c = min(r_norm, g_norm, b_norm)
    delta = max_c - min_c

    l = (max_c + min_c) / 2.0

    if delta == 0:
        s = 0.0
        h = 0.0
    else:
        s = delta / (1.0 - abs(2.0 * l - 1.0))

        if max_c == r_norm:
            h = ((g_norm - b_norm) / delta) % 6.0
        elif max_c == g_norm:
            h = ((b_norm - r_norm) / delta) + 2.0
        else:
            h = ((r_norm - g_norm) / delta) + 4.0
        h *= 60.0
        if h < 0:
            h += 360.0

    return h, l * 100.0, s * 100.0


def hls_to_rgb(h: float, l: float, s: float):
    h_deg = h % 360.0
    l_norm = l / 100.0
    s_norm = s / 100.0

    c = (1.0 - abs(2.0 * l_norm - 1.0)) * s_norm
    x = c * (1.0 - abs((h_deg / 60.0) % 2.0 - 1.0))
    m = l_norm - c / 2.0

    if 0 <= h_deg < 60:
        r_p, g_p, b_p = c, x, 0.0
    elif 60 <= h_deg < 120:
        r_p, g_p, b_p = x, c, 0.0
    elif 120 <= h_deg < 180:
        r_p, g_p, b_p = 0.0, c, x
    elif 180 <= h_deg < 240:
        r_p, g_p, b_p = 0.0, x, c
    elif 240 <= h_deg < 300:
        r_p, g_p, b_p = x, 0.0, c
    else:
        r_p, g_p, b_p = c, 0.0, x

    r = round((r_p + m) * 255.0)
    g = round((g_p + m) * 255.0)
    b = round((b_p + m) * 255.0)

    return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))


class ColorConverterApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа 1 — CMYK / RGB / HLS")
        self.setFixedSize(850, 520)

        self.is_updating = False

        self.r = 255
        self.g = 0
        self.b = 0

        self.init_ui()
        self.update_all_from_rgb()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        left_layout = QVBoxLayout()

        self.rgb_widgets = self.create_model_group(
            "RGB Model",
            [
                ("R (0..255):", 0, 255, 1, 0),
                ("G (0..255):", 0, 255, 1, 0),
                ("B (0..255):", 0, 255, 1, 0),
            ],
            self.on_rgb_changed,
        )

        self.cmyk_widgets = self.create_model_group(
            "CMYK Model",
            [
                ("C (0..100%):", 0, 100, 0.1, 1),
                ("M (0..100%):", 0, 100, 0.1, 1),
                ("Y (0..100%):", 0, 100, 0.1, 1),
                ("K (0..100%):", 0, 100, 0.1, 1),
            ],
            self.on_cmyk_changed,
        )

        self.hls_widgets = self.create_model_group(
            "HLS Model",
            [
                ("H (0..360°):", 0, 360, 0.1, 1),
                ("L (0..100%):", 0, 100, 0.1, 1),
                ("S (0..100%):", 0, 100, 0.1, 1),
            ],
            self.on_hls_changed,
        )

        left_layout.addWidget(self.rgb_widgets["group_box"])
        left_layout.addWidget(self.cmyk_widgets["group_box"])
        left_layout.addWidget(self.hls_widgets["group_box"])

        right_layout = QVBoxLayout()

        self.color_preview = QFrame()
        self.color_preview.setFrameShape(QFrame.Shape.StyledPanel)
        self.color_preview.setMinimumSize(220, 220)

        self.btn_palette = QPushButton("Открыть палитру...")
        self.btn_palette.setFixedHeight(40)
        self.btn_palette.clicked.connect(self.open_color_dialog)

        right_layout.addWidget(QLabel("<b>Предпросмотр цвета:</b>"))
        right_layout.addWidget(self.color_preview)
        right_layout.addWidget(self.btn_palette)
        right_layout.addStretch()

        main_layout.addLayout(left_layout, stretch=2)
        main_layout.addLayout(right_layout, stretch=1)
        self.setCentralWidget(main_widget)

    def create_model_group(self, title, params, model_callback):
        group_box = QGroupBox(title)
        grid = QGridLayout()

        controls = []
        for row, (name, min_v, max_v, step, decimals) in enumerate(params):
            label = QLabel(name)

            spin = QDoubleSpinBox()
            spin.setRange(min_v, max_v)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)

            scale = 10**decimals
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(min_v * scale), int(max_v * scale))

            slider.valueChanged.connect(
                lambda val, sp=spin, sc=scale: self.on_slider_moved(
                    val, sp, sc, model_callback
                )
            )

            spin.valueChanged.connect(
                lambda val, sl=slider, sc=scale: self.on_spin_changed(
                    val, sl, sc, model_callback
                )
            )

            grid.addWidget(label, row, 0)
            grid.addWidget(spin, row, 1)
            grid.addWidget(slider, row, 2)

            controls.append({"spin": spin, "slider": slider, "scale": scale})

        group_box.setLayout(grid)
        return {"group_box": group_box, "controls": controls}

    def on_slider_moved(self, slider_val, spin, scale, model_callback):
        if self.is_updating:
            return
        target_val = slider_val / scale
        if abs(spin.value() - target_val) > 1e-4:
            spin.blockSignals(True)
            spin.setValue(target_val)
            spin.blockSignals(False)
            model_callback()

    def on_spin_changed(self, spin_val, slider, scale, model_callback):
        if self.is_updating:
            return
        target_slider_val = int(round(spin_val * scale))
        if slider.value() != target_slider_val:
            slider.blockSignals(True)
            slider.setValue(target_slider_val)
            slider.blockSignals(False)
        model_callback()

    def on_rgb_changed(self):
        if self.is_updating:
            return
        self.r = int(self.rgb_widgets["controls"][0]["spin"].value())
        self.g = int(self.rgb_widgets["controls"][1]["spin"].value())
        self.b = int(self.rgb_widgets["controls"][2]["spin"].value())
        self.update_all_from_rgb(skip_model="RGB")

    def on_cmyk_changed(self):
        if self.is_updating:
            return
        c = self.cmyk_widgets["controls"][0]["spin"].value()
        m = self.cmyk_widgets["controls"][1]["spin"].value()
        y = self.cmyk_widgets["controls"][2]["spin"].value()
        k = self.cmyk_widgets["controls"][3]["spin"].value()

        self.r, self.g, self.b = cmyk_to_rgb(c, m, y, k)
        self.update_all_from_rgb(skip_model="CMYK")

    def on_hls_changed(self):
        if self.is_updating:
            return
        h = self.hls_widgets["controls"][0]["spin"].value()
        l = self.hls_widgets["controls"][1]["spin"].value()
        s = self.hls_widgets["controls"][2]["spin"].value()

        self.r, self.g, self.b = hls_to_rgb(h, l, s)
        self.update_all_from_rgb(skip_model="HLS")

    def open_color_dialog(self):
        color = QColorDialog.getColor(QColor(self.r, self.g, self.b), self)
        if color.isValid():
            self.r = color.red()
            self.g = color.green()
            self.b = color.blue()
            self.update_all_from_rgb()

    def update_all_from_rgb(self, skip_model=None):
        self.is_updating = True

        color_str = f"rgb({self.r}, {self.g}, {self.b})"
        self.color_preview.setStyleSheet(
            f"background-color: {color_str}; border: 1px solid #777;"
        )

        if skip_model != "RGB":
            for i, val in enumerate([self.r, self.g, self.b]):
                ctrl = self.rgb_widgets["controls"][i]
                ctrl["spin"].setValue(val)
                ctrl["slider"].setValue(int(val * ctrl["scale"]))

        if skip_model != "CMYK":
            cmyk = rgb_to_cmyk(self.r, self.g, self.b)
            for i, val in enumerate(cmyk):
                ctrl = self.cmyk_widgets["controls"][i]
                ctrl["spin"].setValue(val)
                ctrl["slider"].setValue(int(round(val * ctrl["scale"])))

        if skip_model != "HLS":
            hls = rgb_to_hls(self.r, self.g, self.b)
            for i, val in enumerate(hls):
                ctrl = self.hls_widgets["controls"][i]
                ctrl["spin"].setValue(val)
                ctrl["slider"].setValue(int(round(val * ctrl["scale"])))

        self.is_updating = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ColorConverterApp()
    window.show()
    sys.exit(app.exec())