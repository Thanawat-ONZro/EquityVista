"""EquityVista desktop app: neon-glow TradingView-style dashboard (floating island, candlestick chart, glowing stat cards).

Run: python app/main.py
EDUCATIONAL ONLY, NOT FINANCIAL ADVICE.
"""
import math
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402
from lightweight_charts.widgets import QtChart  # noqa: E402  (imports QtWebEngine; must precede QApplication)
from PySide6.QtCore import (QByteArray, QEvent, QObject, QRectF, QRunnable, QSize, Qt, QThread,  # noqa: E402
                            QThreadPool, QTimer, Signal)
from PySide6.QtGui import (QBrush, QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPen,  # noqa: E402
                           QPixmap, QRegion)
from PySide6.QtSvg import QSvgRenderer  # noqa: E402
from PySide6.QtWidgets import (QApplication, QFrame, QGraphicsDropShadowEffect, QGridLayout,  # noqa: E402
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QVBoxLayout, QWidget)

# TradingView-style canvas + neon accents (up = green, down = red: TradingView/Binance convention)
SHELL_R, GLOW_PX = 24, 14  # window corner radius, transparent margin that holds the outer glow
BG, CARD, FG, DIM, EDGE ="#0b0e11", "#131722", "#d1d4dc", "#787b86", "#1e2433"
CYAN, VIOLET, GREEN, RED = "#00e5ff", "#7c5cff", "#0ecb81", "#f6465d"
DISCLAIMER = "Educational, not financial advice; measured: no edge after costs"
NO_MODELS = "models not found — run training first"
MIN_BARS, ZOOM_BARS = 5, 35  # fewer bars than MIN_BARS = unusable; zoom to last ZOOM_BARS only when history is comfortably longer

QSS = f"""
* {{ font-family: 'Segoe UI Variable Display', 'Segoe UI', 'Inter', sans-serif; font-size: 13px; color: {FG}; }}
QMainWindow, QWidget#root, QWidget#panel {{ background: transparent; }}
QFrame#shell {{ background: {BG}; border: 1px solid #262d40; border-radius: {SHELL_R}px; }}
QWidget#panel {{ border-left: 1px solid {EDGE}; }}
QLabel {{ background: transparent; }}
QLineEdit {{ background: {BG}; border: 1px solid #2a3142; border-radius: 12px; padding: 6px 12px;
            font-weight: 700; selection-background-color: {VIOLET}; }}
QLineEdit:focus {{ border: 1px solid {CYAN}; background: #0e1219; }}
QPushButton {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {CYAN}, stop:1 {VIOLET});
              color: #05080c; border: none; border-radius: 12px; padding: 7px 20px; font-weight: 700; }}
QPushButton:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5af0ff, stop:1 #9a82ff); }}
QPushButton:pressed {{ background: {VIOLET}; }}
QPushButton:disabled {{ background: #1b2130; color: {DIM}; }}
QLabel#status {{ color: {DIM}; font-size: 12px; }}
QLabel#head {{ color: {CYAN}; font-size: 11px; font-weight: 700; }}
QLabel#key {{ color: {DIM}; font-size: 11px; font-weight: 600; }}
QLabel#val {{ color: #aab0bd; font-size: 15px; font-weight: 600; }}
QLabel#big {{ font-size: 38px; font-weight: 800; }}
QLabel#disc {{ color: #565a66; font-size: 11px; }}
QToolTip {{ background: {CARD}; color: {FG}; border: 1px solid {EDGE}; }}
"""


def glow(widget, color, blur=24):
    """Soft neon halo around a widget/label (drop-shadow with zero offset)."""
    fx = QGraphicsDropShadowEffect(widget)
    fx.setOffset(0, 0)
    fx.setBlurRadius(blur)
    fx.setColor(QColor(color))
    widget.setGraphicsEffect(fx)
    return fx


class GlowFrame(QFrame):
    """Rounded translucent-gradient surface with a glowing edge (halo is painted inside its `g` px margin)."""

    def __init__(self, accent=CYAN, accent2=None, radius=14, g=8, pill=False):
        super().__init__()
        self.accent, self.accent2, self.radius, self.g, self.pill = accent, accent2, radius, g, pill
        self.phase, self.animated = 0.0, False  # driven by MainWindow's single timer

    def set_accent(self, color):
        self.accent = color
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        g = self.g
        r = QRectF(self.rect()).adjusted(g, g, -g, -g)
        rad = r.height() / 2 if self.pill else self.radius
        halo = QColor(self.accent)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(g, 0, -1):  # outer glow: faint concentric rings fading outwards
            halo.setAlpha(int(64 * (1 - i / g) ** 2 * (0.8 + 0.2 * math.sin(self.phase * 1.3))))  # slow breathing
            p.setPen(QPen(halo, 1.2))
            p.drawRoundedRect(r.adjusted(-i, -i, i, i), rad + i, rad + i)
        fill = QLinearGradient(r.topLeft(), r.bottomRight())
        fill.setColorAt(0, QColor(24, 29, 44, 238))
        fill.setColorAt(1, QColor(11, 14, 20, 238))
        edge = QColor(self.accent)
        edge.setAlpha(170)
        pen = QPen(edge, 1.3)
        if self.accent2:  # cyan -> violet sweep along the edge
            sweep = QLinearGradient(r.topLeft(), r.topRight())
            sweep.setColorAt(0, QColor(self.accent))
            sweep.setColorAt(1, QColor(self.accent2))
            pen = QPen(QBrush(sweep), 1.4)
        p.setPen(pen)
        p.setBrush(fill)
        p.drawRoundedRect(r, rad, rad)
        if self.animated:  # soft white shine gliding along the edge every ~6s
            s = (self.phase * 0.25) % 1.6 - 0.3
            shine = QLinearGradient(r.topLeft(), r.topRight())
            clear = QColor(255, 255, 255, 0)
            shine.setColorAt(0, clear)
            shine.setColorAt(1, clear)
            for pos, a in ((s - 0.18, 0), (s, 170), (s + 0.18, 0)):
                shine.setColorAt(min(1.0, max(0.0, pos)), QColor(255, 255, 255, a))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QBrush(shine), 1.6))
            p.drawRoundedRect(r, rad, rad)


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


class GlowButton(QPushButton):
    """Gradient button whose cyan halo swells on hover."""

    def __init__(self, text):
        super().__init__(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fx = glow(self, CYAN, 8)

    def enterEvent(self, e):
        self.fx.setBlurRadius(34)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.fx.setBlurRadius(8)
        super().leaveEvent(e)


_S = 'fill="none" stroke="COLOR" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"'
ICONS = {  # 16x16 viewBox, COLOR is substituted per state
    "min": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M3.5 8h9" {_S}/></svg>',
    "max": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect x="3.5" y="3.5" width="9" height="9" rx="2.2" {_S}/></svg>',
    "restore": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect x="3" y="5.5" width="7.5" height="7.5" rx="2" {_S}/>'
               f'<path d="M6 3.5h4.5A2.5 2.5 0 0 1 13 6v4.5" {_S}/></svg>',
    "close": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M4 4l8 8M12 4l-8 8" {_S}/></svg>',
}


class WinButton(QPushButton):
    """Flat window-control icon (inline SVG): dim at rest, neon glow + tint in `color` on hover."""

    def __init__(self, kind, tip, color):
        super().__init__()
        self.color = color
        self.setToolTip(tip)
        self.setFixedSize(30, 24)
        self.setIconSize(QSize(16, 16))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        r, g, b = QColor(color).red(), QColor(color).green(), QColor(color).blue()
        self.setStyleSheet(f"QPushButton {{ background: transparent; border-radius: 12px; padding: 0; }}"
                           f"QPushButton:hover {{ background: rgba({r},{g},{b},0.16); }}"
                           f"QPushButton:pressed {{ background: rgba({r},{g},{b},0.32); }}")
        self.fx = glow(self, color, 0)
        self.fx.setEnabled(False)  # no effect (0 cost) until hovered
        self.set_kind(kind)

    def set_kind(self, kind):
        self.kind = kind
        self._paint_icon(self.color if self.underMouse() else DIM)  # clicked while hovered: keep the tint

    def _paint_icon(self, color):
        dpr = self.devicePixelRatioF()
        pm = QPixmap(int(16 * dpr), int(16 * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        QSvgRenderer(QByteArray(ICONS[self.kind].replace("COLOR", color).encode())).render(p, QRectF(0, 0, 16, 16))
        p.end()
        self.setIcon(QIcon(pm))

    def enterEvent(self, e):
        self.fx.setEnabled(True)
        self.fx.setBlurRadius(18)
        self._paint_icon(self.color)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.fx.setEnabled(False)
        self._paint_icon(DIM)
        super().leaveEvent(e)


# Middle-mouse drag pans the chart (lightweight-charts itself only starts a pan on the left button)
MIDDLE_PAN_JS = """
(function () {
    const el = document.getElementById('container'), ts = %s.chart.timeScale();
    let x = null;
    el.addEventListener('mousedown', e => { if (e.button === 1) { e.preventDefault(); x = e.clientX; } });
    window.addEventListener('mouseup', e => { if (e.button === 1) x = null; });
    window.addEventListener('mousemove', e => {
        if (x === null) return;
        const r = ts.getVisibleLogicalRange();
        if (!r) return;
        const d = (e.clientX - x) * (r.to - r.from) / el.clientWidth;  // bars per pixel
        x = e.clientX;
        ts.setVisibleLogicalRange({from: r.from - d, to: r.to - d});
    });
})();
"""


class Signals(QObject):
    settled = Signal(object, object, object)  # (fn, result, error message or None) - always emitted exactly once


class Job(QRunnable):
    """Run fn(ticker) on the thread pool; report via signals (emitted back on the GUI thread)."""

    def __init__(self, fn, ticker):
        super().__init__()
        self.fn, self.ticker, self.signals = fn, ticker, Signals()

    def run(self):
        res = err = None
        try:
            res = self.fn(self.ticker)
        except BaseException as e:  # noqa: BLE001 - any failure (even SystemExit from a lib) goes to the UI, never kills the pool thread
            err = str(e) or type(e).__name__
        self.signals.settled.emit(self.fn, res, err)  # outside the try: the counter in MainWindow must always get its tick


def fetch_ohlcv(ticker: str) -> pd.DataFrame:
    d = yf.Ticker(ticker).history(period="5y", auto_adjust=True)
    if d.empty:
        raise ValueError(f"no price data for {ticker!r} (bad symbol or delisted)")
    d = d.reset_index().rename(columns=str.lower).rename(columns={"date": "time"})
    d["time"] = pd.to_datetime(d["time"]).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")  # force ns: lib divides by 1e9 (pandas 3.0 defaults to [s] which breaks it)
    d = d[["time", "open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])
    if len(d) < MIN_BARS:  # a near-empty frame must never reach the chart (QtWebEngine can abort the process)
        raise ValueError(f"insufficient data for {ticker}")
    return d


class MainThreadCNN(QObject):
    """Keras model proxy: .predict() called from ANY thread runs on the GUI thread (TF stays single-threaded,
    avoids `_PyThreadState_Attach` fatals from TF inside a pool thread). Caller blocks, the GUI loop doesn't."""
    _call = Signal(object)

    def __init__(self, model):  # must be constructed on the GUI thread (slot affinity)
        super().__init__()
        self.model = model
        self._call.connect(self._run)  # emitted from a worker -> queued onto this object's (GUI) thread

    def _run(self, job):
        try:
            job["r"] = self.model.predict(*job["a"], **job["k"])
        except BaseException as e:  # noqa: BLE001 - handed back to the caller thread
            job["e"] = e
        job["ev"].set()

    def predict(self, *a, **k):
        if QThread.currentThread() is self.thread():
            return self.model.predict(*a, **k)
        job = {"a": a, "k": k, "ev": threading.Event()}
        self._call.emit(job)
        if not job["ev"].wait(120):
            raise TimeoutError("CNN-LSTM predict timed out")
        if "e" in job:
            raise job["e"]
        return job["r"]


_cnn_proxy = _load_err = None  # proxy kept alive; startup load error text


def load_models_once() -> str | None:
    """GUI thread, before any worker starts: import TF, load the CNN once, warm it up. Returns error text or None."""
    global _cnn_proxy
    from src.models import cnn_lstm_model as cnn  # pulls in tensorflow (slow)
    from src.models.lgbm_model import MODEL_DIR
    if not ((MODEL_DIR / "lgbm_y.txt").exists() and cnn.MODEL_PATH.exists()):
        return NO_MODELS
    try:
        import numpy as np
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)  # single-threaded TF: no attach race
        tf.config.threading.set_inter_op_parallelism_threads(1)
        from src import predict as P
        m = P._cnn()  # the one and only load, cached in P._models
        m.predict(np.zeros((1, cnn.W, len(cnn.FEATURES)), "float32"), verbose=0)  # warm-up on the GUI thread
        _cnn_proxy = P._models["cnn"] = MainThreadCNN(m)
    except Exception as e:  # noqa: BLE001
        return f"model load failed: {e}"
    return None


def run_predict(ticker: str) -> dict:
    if _cnn_proxy is None:  # startup load failed / models missing: never lazily load TF in a worker
        raise FileNotFoundError(_load_err or NO_MODELS)
    from src.predict import predict_ticker
    return predict_ticker(ticker)  # fetch+LGBM on this worker; the CNN call hops to the GUI thread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EquityVista — Stock Forecast")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)  # no OS title bar
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # rounded shell shows through
        self.resize(1280 + 2 * GLOW_PX, 780 + 2 * GLOW_PX)
        self._drag = None
        self.pool, self.pending, self.jobs = QThreadPool.globalInstance(), 0, []

        self.root = Backdrop()
        self.setCentralWidget(self.root)
        outer = QHBoxLayout(self.root)
        outer.setContentsMargins(GLOW_PX, GLOW_PX, GLOW_PX, GLOW_PX)
        self.shell = self.root.shell = QFrame(objectName="shell")  # the whole UI lives in this rounded container
        outer.addWidget(self.shell)
        root = self.shell
        body = QHBoxLayout(root)  # no header bar: chart fills the shell, results panel flush right
        body.setContentsMargins(1, 1, 1, 1)  # keep the 1px border visible
        body.setSpacing(0)

        self.chart = QtChart(root)
        self.web = self.chart.get_webview()
        self.web.setStyleSheet(f"background: {BG};")
        self.chart.layout(background_color=BG, text_color=FG, font_size=12, font_family="Segoe UI")
        self.chart.grid(color="rgba(120,123,134,0.07)")
        self.chart.candle_style(up_color=GREEN, down_color=RED)
        self.chart.crosshair(vert_color="rgba(0,229,255,0.5)", horz_color="rgba(0,229,255,0.5)")
        self.chart.legend(visible=True, font_family="Segoe UI", color=FG)
        # wheel zoom + left-drag pan stay on; middle-drag pan comes from MIDDLE_PAN_JS
        self.chart.run_script(f"{self.chart.id}.chart.applyOptions({{handleScroll: {{mouseWheel: true, "
                              f"pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false}}}})")
        self.chart.run_script(MIDDLE_PAN_JS % self.chart.id)
        self.connector = None  # solid direction line, created per prediction in on_pred
        self.last_ticker = self.run_ticker = None  # last fully successful ticker (idempotent Predict)
        body.addWidget(self.web, 1)
        body.addWidget(self._panel())

        self.island = self._island(root)  # floats over the chart, top-centre
        self.controls = self._controls(root)  # floats top-right: minimize / maximize / close
        self.web.installEventFilter(self)
        QTimer.singleShot(0, self._place_island)
        self.island.animated = self.controls.animated = True
        self.t = 0.0
        self.anim = QTimer(self, interval=90)  # ~11 fps; repaints island, controls and the glow margin only
        self.anim.timeout.connect(self._tick)
        self.anim.start()

        self.btn.clicked.connect(self.start)
        self.ticker.returnPressed.connect(self.start)

    def _island(self, parent):
        isl = GlowFrame(CYAN, VIOLET, g=14, pill=True)
        isl.setParent(parent)
        row = QHBoxLayout(isl)
        row.setContentsMargins(14 + 18, 14 + 7, 14 + 8, 14 + 7)
        row.setSpacing(12)
        brand = QLabel(f'<span style="font-size:15px;font-weight:800;color:{FG}">Equity</span>'
                       f'<span style="font-size:15px;font-weight:800;color:{CYAN}">Vista</span>')
        glow(brand, CYAN, 16)
        self.ticker = QLineEdit(placeholderText="AAPL", maxLength=10)
        self.ticker.setFixedWidth(92)
        self.btn = GlowButton("Predict")
        self.dot = QLabel()
        self.dot.setFixedSize(9, 9)
        self.dot_fx = glow(self.dot, DIM, 12)
        self.status = QLabel(objectName="status")
        self.status.setFixedWidth(150)
        for w in (brand, self.ticker, self.btn, self.dot, self.status):
            row.addWidget(w, 0, Qt.AlignmentFlag.AlignVCenter)
        self._status("Enter a US ticker", DIM)
        isl.adjustSize()
        return isl

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
        for w in (self.island, self.controls):
            w.phase = self.t
            w.update()
        self.root.tick(self.t)

    def _round_web(self):
        """Clip the opaque webview to the shell's rounded left corners (right side stays square: panel is there)."""
        if self.root.square:
            self.web.clearMask()
            return
        r = SHELL_R - 1
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.web.width() + r, self.web.height()), r, r)
        self.web.setMask(QRegion(path.toFillPolygon().toPolygon()))

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
            self._round_web()
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

    def _status(self, text, color):
        self.status.setText(self.status.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, 148))
        self.status.setToolTip(text)
        self.dot.setStyleSheet(f"background: {color}; border-radius: 4px;")
        self.dot_fx.setColor(QColor(color))

    def _place_island(self):
        self.island.adjustSize()
        self.island.move(self.web.x() + (self.web.width() - self.island.width()) // 2, 8)
        self.island.raise_()
        self.controls.adjustSize()
        self.controls.move(self.shell.width() - self.controls.width() - 14, 8)
        self.controls.raise_()

    def eventFilter(self, obj, ev):
        if obj is self.web and ev.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._place_island)
            self._round_web()
        return super().eventFilter(obj, ev)

    def _card(self, key, big=False):
        g = 8 if big else 2  # hero card keeps the neon halo; secondary cards are calm, dim-edged
        c = GlowFrame(VIOLET if big else "#2a3142", radius=18 if big else 12, g=g)
        v = QVBoxLayout(c)
        v.setContentsMargins(g + 16, g + (14 if big else 9), g + 16, g + (14 if big else 9))
        v.setSpacing(2)
        k = QLabel(key.upper(), objectName="key")
        f = k.font()
        f.setLetterSpacing(f.SpacingType.AbsoluteSpacing, 1.4)  # spaced uppercase label
        k.setFont(f)
        v.addWidget(k)
        lab = QLabel("—", objectName="big" if big else "val")
        v.addWidget(lab)
        self.vals[key] = lab
        return c

    def _panel(self):
        p = QWidget(objectName="panel")
        p.setFixedWidth(340)
        v = QVBoxLayout(p)
        v.setContentsMargins(14, 56, 14, 18)  # top clears the floating window controls
        v.setSpacing(0)
        head = QLabel("FORECAST", objectName="head")
        hf = head.font()
        hf.setLetterSpacing(hf.SpacingType.AbsoluteSpacing, 2.4)
        head.setFont(hf)
        head.setContentsMargins(10, 0, 0, 10)
        v.addWidget(head)
        self.vals = {}
        self.today = ("—", FG)
        self.dir_card = self._card("Direction", big=True)
        self.dir_fx = glow(self.vals["Direction"], DIM, 28)
        g = QGridLayout()
        g.setSpacing(2)
        g.setRowMinimumHeight(1, 0)
        g.addWidget(self.dir_card, 0, 0, 1, 2)
        g.addWidget(self._card("Next-day return"), 1, 0)
        g.addWidget(self._card("Confidence"), 1, 1)
        g.addWidget(self._card("Last close"), 2, 0)
        g.addWidget(self._card("Today"), 2, 1)
        g.addWidget(self._card("Price view"), 3, 0)
        g.addWidget(self._card("95% interval"), 3, 1)
        g.addWidget(self._card("CNN-LSTM view"), 4, 0, 1, 2)
        v.addLayout(g)
        self.note = QLabel("", objectName="disc", wordWrap=True)
        self.note.setContentsMargins(12, 6, 12, 0)
        v.addWidget(self.note)
        v.addStretch(1)
        d = QLabel(DISCLAIMER, objectName="disc", wordWrap=True)
        d.setContentsMargins(12, 0, 12, 0)
        v.addWidget(d)
        self._set({})
        return p

    def _set(self, vals: dict, color: str = FG):
        for k, lab in self.vals.items():
            lab.setText(vals.get(k, "—"))
            lab.setStyleSheet("")
        txt, col = self.today  # live session change comes from the chart job, independent of the forecast
        self.vals["Today"].setText(txt)
        self.vals["Today"].setStyleSheet(f"color: {col};")
        has_dir = "Direction" in vals
        if has_dir:
            self.vals["Direction"].setStyleSheet(f"color: {color};")
        self.dir_fx.setColor(QColor(color if has_dir else DIM))
        self.dir_fx.setBlurRadius(30 if has_dir else 0)
        self.dir_card.set_accent(color if has_dir else VIOLET)

    def start(self):
        if not self.btn.isEnabled():  # busy: Enter in the field bypasses the disabled button; never overlap runs
            return
        t = self.ticker.text().strip().upper() or "AAPL"
        self.ticker.setText(t)
        if t == self.last_ticker:  # already predicted OK: keep chart/connector/panel as is
            self._status(f"Already predicted: {t}", GREEN)
            return
        self.last_ticker = None  # set again in finish() only if this run fully succeeds
        self.run_ticker = t
        self.btn.setEnabled(False)
        self._status("Predicting…", CYAN)
        self.note.setText("")
        self.today = ("—", FG)  # clear stale change from the previous ticker
        self._set({})
        self._drop_connector()
        self.pending = 2
        self.errors = []
        self.handlers = {fetch_ohlcv: self.on_chart, run_predict: self.on_pred}
        for fn in self.handlers:
            j = Job(fn, t)
            j.signals.settled.connect(self._settle)  # bound QObject method -> queued onto the GUI thread
            self.jobs.append(j)  # keep Python refs (signals objects) alive until done
            self.pool.start(j)

    def _settle(self, fn, res, err):
        """One slot for both jobs: a crash in one handler is reported, and the counter ticks regardless."""
        try:
            if err is None:
                self.handlers[fn](res)
        except Exception as e:  # noqa: BLE001 - handler bug must not leave the button disabled or kill the other job
            err = str(e) or type(e).__name__
        finally:
            if err is not None:
                self.on_fail(fn, err)
            self.finish()

    def on_chart(self, df):
        if df is None or len(df) < MIN_BARS:
            raise ValueError("insufficient data")
        price = df.drop(columns=["volume"])
        self.chart.set(price)
        n = len(price)
        if n >= ZOOM_BARS + 5:  # default zoom: last ~35 bars (+3 right padding); short history: leave the auto-fit view
            lo = max(n - ZOOM_BARS, 0)  # never negative: a bad logical range can abort QtWebEngine
            self.chart.run_script(f"{self.chart.id}.chart.timeScale().setVisibleLogicalRange({{from: {lo}, to: {n + 3}}})")
        else:
            self.chart.fit()
        self.last_time = df["time"].iloc[-1]
        last, prev = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
        if math.isfinite(last) and math.isfinite(prev) and prev:
            chg = last / prev - 1
            self.today = (("▲ " if chg >= 0 else "▼ ") + f"{chg:+.2%}", GREEN if chg >= 0 else RED)
            self.vals["Today"].setText(self.today[0])
            self.vals["Today"].setStyleSheet(f"color: {self.today[1]};")

    def _drop_connector(self):
        if self.connector is not None:
            self.connector.delete()
            self.connector = None

    def on_pred(self, r):
        up = r["direction"] == "UP"
        c = r["cnn_lstm"]
        iv = r.get("interval_95")  # residual-based, None if models/interval.json is missing
        iv_ok = isinstance(iv, (list, tuple)) and len(iv) == 2 and all(math.isfinite(x) for x in iv)
        self._set({
            "Last close": f"{r['last_close']:.2f}",
            "Next-day return": f"{r['next_day_return_pred']:+.2%}",
            "Direction": f"{'▲' if up else '▼'} {r['direction']} {r['next_day_return_pred']:+.2%}",
            "Confidence": f"{r['confidence']:.2f}",
            "Price view": f"{r['price_pred']:.2f}",
            "95% interval": f"{iv[0]:.2f} – {iv[1]:.2f}" if iv_ok else "n/a",
            "CNN-LSTM view": f"{c['next_day_return_pred']:+.2%} {c['direction']}",
        }, GREEN if up else RED)
        if r.get("stale_days", 0) > 5:
            self.note.setText(f"Warning: last bar is {r['stale_days']} days old")
        lc, pp = r.get("last_close"), r.get("price_pred")
        if all(isinstance(x, (int, float)) and math.isfinite(x) for x in (lc, pp)) and r.get("date") and r.get("next_date"):
            df = pd.DataFrame({"time": pd.to_datetime([r["date"], r["next_date"]]).astype("datetime64[ns]"),
                               "Forecast": [lc, pp]})
            col = GREEN if up else RED  # solid connector last close -> forecast, coloured by direction
            self._drop_connector()
            self.connector = self.chart.create_line("Trend", color=col, style="solid", width=3)
            self.connector.set(df.rename(columns={"Forecast": "Trend"}))
            self.connector.marker(time=df["time"].iloc[-1], position="above" if up else "below",
                                 shape="arrow_up" if up else "arrow_down", color=col,
                                 text=f"{'▲' if up else '▼'} {r['next_day_return_pred']:+.2%}")  # signed % on the line tip

    def on_fail(self, fn, msg):
        if fn is run_predict:  # predict failure only annotates the panel; the chart (fetch) stays as drawn
            self.note.setText(msg)
            self._set({})
        self.errors.append(msg)  # any failure -> "Error" status and no last_ticker cache

    def finish(self, *_):
        self.pending -= 1
        if self.pending > 0:
            return
        self.jobs.clear()
        self.btn.setEnabled(True)
        if self.errors:
            self._status(f"Error: {self.errors[0]}", RED)
        else:
            self.last_ticker = self.run_ticker
            self._status("Done", GREEN)


def run() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(QSS)
    global _load_err
    _load_err = load_models_once()  # TF + CNN load on the main thread, before any worker exists
    win = MainWindow()
    win.show()
    # auto-load a default ticker on startup so the dashboard isn't empty
    win.ticker.setText("AAPL")
    win.start()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
