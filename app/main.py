"""EquityVista desktop app: frameless neon window around the Predict chart page. The app trains nothing: models are
trained from the command line (`python train_all.py`, see its docstring) and the app only reads runs/ and models/active.json.

Run: python app/main.py
EDUCATIONAL ONLY, NOT FINANCIAL ADVICE.
"""
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pages import predict as predict_page  # noqa: E402  (imports QtWebEngine first; must precede QApplication)
from PySide6.QtCore import QEvent, QRectF, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QColor, QPainter, QPen, QRegion  # noqa: E402
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QMainWindow, QWidget  # noqa: E402

from app.core.job_manager import JobManager  # noqa: E402
from app.theme import BG, CYAN, GLOW_PX, QSS, RED, SHELL_R, VIOLET, GlowFrame, WinButton  # noqa: E402


class Backdrop(QWidget):
    """Transparent root: paints the outer glow ring around the rounded shell (only the margin repaints on tick)."""

    def __init__(self):
        super().__init__(objectName="root")
        self.phase, self.shell, self.square = 0.0, None, False  # square = maximized: flat opaque fill, no glow

    def tick(self, phase):
        self.phase = phase
        if not self.square:
            self.update(QRegion(self.rect()) - QRegion(self.shell.geometry()))

    def paintEvent(self, _):
        p = QPainter(self)
        if self.square:
            p.fillRect(self.rect(), QColor(BG))
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(Qt.BrushStyle.NoBrush)
        k = 0.5 + 0.5 * math.sin(self.phase * 0.7)  # cyan <-> violet drift
        a, b = QColor(CYAN), QColor(VIOLET)
        c = QColor(int(a.red() + (b.red() - a.red()) * k), int(a.green() + (b.green() - a.green()) * k),
                   int(a.blue() + (b.blue() - a.blue()) * k))
        r, g = QRectF(self.shell.geometry()), GLOW_PX
        for i in range(1, g):
            c.setAlpha(int(52 * (1 - i / g) ** 2 * (0.75 + 0.25 * math.sin(self.phase * 1.3))))
            p.setPen(QPen(c, 1.2))
            p.drawRoundedRect(r.adjusted(-i, -i, i, i), SHELL_R + i, SHELL_R + i)


class MainWindow(QMainWindow):
    def __init__(self, manager):
        super().__init__()
        self.setWindowTitle("EquityVista")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)  # no OS title bar
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # rounded shell shows through
        self.resize(1280 + 2 * GLOW_PX, 780 + 2 * GLOW_PX)
        self._drag = None
        self.manager = manager

        self.root = Backdrop()
        self.setCentralWidget(self.root)
        outer = QHBoxLayout(self.root)
        outer.setContentsMargins(GLOW_PX, GLOW_PX, GLOW_PX, GLOW_PX)
        self.shell = self.root.shell = QFrame(objectName="shell")  # the whole UI lives in this rounded container
        outer.addWidget(self.shell)
        body = QHBoxLayout(self.shell)
        body.setContentsMargins(1, 1, 1, 1)  # keep the 1px border visible
        body.setSpacing(0)

        self.predict = predict_page.PredictPage(manager)
        body.addWidget(self.predict, 1)
        self.controls = self._controls(self.shell)  # floats top-right: minimize / maximize / close
        self.shell.installEventFilter(self)
        QTimer.singleShot(0, self._place_controls)
        self.controls.animated = True
        self.t = 0.0
        self.anim = QTimer(self, interval=90)  # ~11 fps; repaints island, controls and the glow margin only
        self.anim.timeout.connect(self._tick)
        self.anim.start()

    def _controls(self, parent):
        box = GlowFrame(CYAN, VIOLET, g=10, pill=True)
        box.setParent(parent)
        row = QHBoxLayout(box)
        row.setContentsMargins(10 + 8, 10 + 4, 10 + 8, 10 + 4)
        row.setSpacing(2)
        self.b_max = WinButton("max", "Maximize", CYAN)
        for b, fn in ((WinButton("min", "Minimize", VIOLET), self.showMinimized), (self.b_max, self._toggle_max),
                      (WinButton("close", "Close", RED), self.close)):
            b.clicked.connect(fn)
            row.addWidget(b)
        box.adjustSize()
        return box

    def _tick(self):
        self.t += 0.09
        self.predict.tick(self.t)
        self.controls.phase = self.t
        self.controls.update()
        self.root.tick(self.t)

    def _toggle_max(self):
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def changeEvent(self, e):
        if e.type() == QEvent.Type.WindowStateChange:  # also fires for Win+Up / snap
            m = self.isMaximized() or self.isFullScreen()
            self.b_max.set_kind("restore" if m else "max")
            self.b_max.setToolTip("Restore" if m else "Maximize")
            g = 0 if m else GLOW_PX  # drop the glow margin when filling the screen
            self.root.layout().setContentsMargins(g, g, g, g)
            self.root.square = m  # Backdrop: opaque BG, no glow ring
            self.shell.setStyleSheet(f"QFrame#shell {{ background: {BG}; border: none; border-radius: 0; }}" if m else "")
            self.shell.layout().setContentsMargins(*[0 if m else 1] * 4)
            self.root.update()
            self.anim.stop() if self.isMinimized() else self.anim.start()  # 0 cost while minimized
        super().changeEvent(e)

    # Drag the window by any background area: presses on buttons / inputs / chart are accepted by those
    # widgets and never reach here; labels, panels and the island ignore them so they bubble up.
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and not self.isMaximized():
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._drag = None
        super().mouseReleaseEvent(e)

    def _place_controls(self):
        self.controls.adjustSize()
        self.controls.move(self.shell.width() - self.controls.width() - 14, 8)
        self.controls.raise_()

    def eventFilter(self, obj, ev):
        if obj is self.shell and ev.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._place_controls)
        return super().eventFilter(obj, ev)


def run() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(QSS)
    manager = JobManager(restore=False)  # read-only view of runs/: a training started from the command line may be running
    win = MainWindow(manager)
    win.show()
    win.predict.autostart()  # after show(): the chart needs a visible, sized webview
    if os.environ.get("EV_QUIT_MS"):  # smoke tests: open, wait, close cleanly
        QTimer.singleShot(int(os.environ["EV_QUIT_MS"]), app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
