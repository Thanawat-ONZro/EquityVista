"""Predict page: candlestick chart + floating island (ticker search, forecast days) + forecast panel, using the ACTIVE
model (Logistic Regression) of the chosen horizon (models/active.json). No model for that horizon -> a plain status line.
Window chrome (frame, glow ring, drag, min/max/close) lives in the shell.

EDUCATIONAL ONLY, NOT FINANCIAL ADVICE.
"""
import math

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402
from lightweight_charts.widgets import QtChart  # noqa: E402  (imports QtWebEngine; must precede QApplication)
from PySide6.QtCore import QEvent, QObject, Qt, QThreadPool, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QColor, QPainter, QPen  # noqa: E402
from PySide6.QtWidgets import (QGridLayout, QCheckBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,  # noqa: E402
                               QScrollArea, QSizePolicy, QSpinBox, QSplitter, QVBoxLayout, QWidget)

from app.core import explain, registry, settings  # noqa: E402
from app.theme import AMBER, BG, CYAN, DIM, EDGE, FG, GREEN, RED, VIOLET, GlowButton, GlowFrame, glow  # noqa: E402
from src import active  # noqa: E402
from src import predict as P  # noqa: E402
from src.split import TRAIN_END  # noqa: E402

DISCLAIMER = "Educational, not financial advice; measured: little or no edge after costs"
MIN_BARS, VIEW_BARS, RIGHT_PAD = 5, 120, 12  # fewer than MIN_BARS = unusable; view = last VIEW_BARS candles + forecast + RIGHT_PAD empty slots
DEFAULT_DAYS = 21
NO_MODEL = "No active model for this horizon (python train_all.py)"
PANEL_W, PANEL_MIN, PANEL_MAX = 340, 280, 700  # forecast panel width: default / drag limits (the user's width persists in settings)
PANEL_KEY = "predict_panel_w"
HIDE = 60  # Backtest: trading days hidden from the end of the ticker

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
    settled = Signal(object, object, object)  # (fn, result, exception or None) - always emitted exactly once


def _work(sig, fn, args):
    """Thread-pool body. `sig` is owned by the page (GUI thread) and outlives every worker, so nothing a worker
    touches is ever destroyed under it (a per-run QRunnable + Signals pair was freed from the GUI thread mid-emit)."""
    res = err = None
    try:
        res = fn(*args)
    except BaseException as e:  # noqa: BLE001 - any failure (even SystemExit from a lib) goes to the UI, never kills the pool thread
        err = e
    sig.settled.emit(fn, res, err)  # outside the try: the counter in the page must always get its tick


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


def run_predict(ticker: str, horizon: int, hide: int = 0) -> dict:
    return P.predict_ticker(ticker, horizon, hide)


def clamp_width(w) -> int:
    try:
        return max(PANEL_MIN, min(PANEL_MAX, int(w)))
    except (TypeError, ValueError):
        return PANEL_W


VERDICT_COLOR = {"beats": GREEN, "useful": GREEN, "equal": AMBER, "weak": AMBER, "not proven": AMBER,
                 "worse": RED, "none": RED, "n/a": DIM}


class Meter(QWidget):
    """3-segment skill meter: none / weak / useful (segments up to `level` are lit)."""
    def __init__(self):
        super().__init__()
        self.level, self.lit = 0, False
        self.setFixedSize(54, 8)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        col = QColor(VERDICT_COLOR[explain.IC_LEVELS[self.level]])
        for i in range(3):
            p.setBrush(col if self.lit and i <= self.level else QColor(EDGE))
            p.drawRoundedRect(i * 19, 0, 16, 8, 3, 3)


class CompareChart(QWidget):
    """Horizontal bars per model type: test IC, then direction accuracy with the 'always UP' baseline line."""
    ROW, LBL = 17, 78

    def __init__(self):
        super().__init__()
        self.rows = []
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(1)

    def set_rows(self, rows):
        self.rows = rows
        self.setFixedHeight(2 * (len(rows) * self.ROW + 36) if rows else 1)
        self.update()

    def _group(self, p, y0, title, vals, lo, hi, ref, ref_txt, fmt):
        w0, w1 = self.LBL, self.width() - 44
        sc = lambda v: w0 + (min(max(v, lo), hi) - lo) / (hi - lo) * (w1 - w0)  # noqa: E731
        p.setPen(QColor(DIM))
        p.drawText(0, y0, self.width(), 14, Qt.AlignmentFlag.AlignLeft, title)
        top = y0 + 18
        for i, (r, v) in enumerate(vals):
            y = top + i * self.ROW
            p.setPen(QColor(CYAN if r["active"] else FG))
            p.drawText(0, y, w0 - 4, self.ROW, Qt.AlignmentFlag.AlignVCenter, r["model"])
            if v is None:
                continue
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(CYAN if r["active"] else "#3a4358"))
            x0, x1 = sorted((sc(max(lo, 0) if lo < 0 else lo), sc(v)))
            p.drawRoundedRect(int(x0), y + 3, max(2, int(x1 - x0)), self.ROW - 7, 3, 3)
            p.setPen(QColor(FG))
            p.drawText(int(w1) + 4, y, 40, self.ROW, Qt.AlignmentFlag.AlignVCenter, fmt(v))
        x = int(sc(ref))
        p.setPen(QPen(QColor(AMBER), 1, Qt.PenStyle.DashLine))
        p.drawLine(x, top, x, top + len(vals) * self.ROW)
        p.setPen(QColor(AMBER))
        p.drawText(max(0, x - 100), top + len(vals) * self.ROW, 200, 14, Qt.AlignmentFlag.AlignHCenter, ref_txt)

    def paintEvent(self, _):
        if not self.rows:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = p.font()
        f.setPixelSize(11)
        p.setFont(f)
        ics = [r["ic"] for r in self.rows if r["ic"] is not None]
        self._group(p, 0, "Ranking skill (rank-IC; 0 = no skill)", [(r, r["ic"]) for r in self.rows],
                    min([0, *ics]) - 0.005, max([0.05, *ics]), 0, "baseline: no skill", lambda v: f"{v:+.3f}")
        accs = [r["acc"] for r in self.rows if r["acc"] is not None]
        bases = [r["base"] for r in self.rows if r["base"] is not None]
        base = sum(bases) / len(bases) if bases else 0.5
        self._group(p, len(self.rows) * self.ROW + 36, "Direction accuracy", [(r, r["acc"]) for r in self.rows],
                    min([0.45, *accs, base]) - 0.01, max([0.6, *accs, base]) + 0.01, base,
                    f"baseline: always UP {base:.0%}" if bases else "baseline: coin flip 50%", lambda v: f"{v:.1%}")


class PredictPage(QWidget):
    def __init__(self, mgr):
        super().__init__()
        self.mgr, self._reload_pending = mgr, False
        self.pool, self.pending = QThreadPool.globalInstance(), 0
        self.sig = Signals(self)
        self.sig.settled.connect(self._settle)  # queued onto the GUI thread
        body = QHBoxLayout(self)  # no header bar: chart fills the page, results panel flush right
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.chart = QtChart(self)
        self.web = self.chart.get_webview()
        self.web.setStyleSheet(f"background: {BG};")
        self.chart.layout(background_color=BG, text_color=FG, font_size=12, font_family="Segoe UI")
        self.chart.grid(color="rgba(120,123,134,0.07)")
        self.chart.candle_style(up_color=GREEN, down_color=RED)
        self.chart.crosshair(vert_color="rgba(0,229,255,0.5)", horz_color="rgba(0,229,255,0.5)")
        self.chart.legend(visible=True, lines=False, font_family="Segoe UI", color=FG)  # lines off: a forecast line's legend value is a point on its path, not the target
        # wheel zoom + left-drag pan stay on; middle-drag pan comes from MIDDLE_PAN_JS
        self.chart.run_script(f"{self.chart.id}.chart.applyOptions({{handleScroll: {{mouseWheel: true, "
                              f"pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false}}}})")
        self.chart.run_script(MIDDLE_PAN_JS % self.chart.id)
        self.forecast = []  # every series drawn for the current forecast (the marker's reference line); wiped before each Predict
        self.pred, self.n_bars, self.no_model = None, 0, False
        self.full = self.hidden = None  # Backtest: all bars / the hidden ones, until "Reveal"
        self.last_key = self.run_key = None  # (ticker, days, backtest) of the last fully successful / the running Predict
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(3)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStyleSheet(f"QSplitter::handle {{ background: {EDGE}; }} "
                                    f"QSplitter::handle:hover, QSplitter::handle:pressed {{ background: {CYAN}; }}")
        self.panel_w = clamp_width(settings.ui_get(PANEL_KEY, PANEL_W))
        self.splitter.addWidget(self.web)
        self.splitter.addWidget(self._panel())
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)  # window resize changes the chart, the panel keeps its width
        self.splitter.setSizes([max(1, 900 - self.panel_w), self.panel_w])
        self.splitter.splitterMoved.connect(self._moved)
        self._save_timer = QTimer(self, singleShot=True, interval=300)
        self._save_timer.timeout.connect(self.save_width)
        body.addWidget(self.splitter)

        self.island = self._island(self)  # floats over the chart, top-centre
        self.web.installEventFilter(self)
        QTimer.singleShot(0, self._place_island)
        self.island.animated = True

        self.btn.clicked.connect(self.start)
        self.ticker.returnPressed.connect(self.start)
        self.days.valueChanged.connect(self._header)
        self.reveal_btn.clicked.connect(self.reveal)
        mgr.active_changed.connect(self.reload_model)
        for sig in (mgr.finished, mgr.added):
            sig.connect(lambda *_: self._header())
        self._header()

    def _moved(self, *_):
        self.panel_w = self.splitter.sizes()[1]
        self._save_timer.start()  # debounce: one write after the drag stops

    def save_width(self):
        self.panel_w = clamp_width(self.panel_w)
        settings.ui_set(PANEL_KEY, self.panel_w)

    def showEvent(self, ev):
        super().showEvent(ev)
        if not getattr(self, "_sized", False):  # first show: the real width is known now, apply the saved panel width
            self._sized = True
            self.splitter.setSizes([max(1, self.width() - self.panel_w), self.panel_w])

    # ---- header: horizon, active model ----
    def _header(self, *_):
        h = self.days.value()
        name = registry.horizon_name(h)
        self.head.setText(f"FORECAST · {name.upper()}")
        rid = active.run_id(h) if active.path(h) else None
        ic = self.mgr.jobs.get(rid, {}).get("metrics", {}).get(registry.KEY) if rid else None
        self.model_lbl.setText(f"Model: {rid}" + (f" · rank-IC {ic:.3f}" if ic is not None else "") if rid
                               else f"No model for {h} days yet.")
        for k, q in self.tips.items():
            q.setToolTip(explain.tip(k, h))
        self._helps()
        self._quality(h, rid)

    def reload_model(self):
        """active.json changed: refresh the header and re-run the last ticker (deferred while a prediction runs)."""
        self._header()
        if not self.btn.isEnabled():
            self._reload_pending = True
            return
        self._reload_pending = False
        if self.run_key:
            self.last_key = None
            self.start()

    def tick(self, phase):
        """Driven by the shell's single animation timer."""
        self.island.phase = phase
        self.island.update()

    def autostart(self):
        """Auto-load a default ticker on startup so the dashboard isn't empty."""
        self.ticker.setText("AAPL")
        self._status("Loading AAPL…", CYAN)
        QTimer.singleShot(150, self.start)  # let the page paint first; start() itself only queues pool work

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
        self.days = QSpinBox(minimum=1, maximum=252, value=DEFAULT_DAYS, suffix=" days")
        self.days.setToolTip("Forecast horizon in trading days (21 trading days = 1 month, about 30 calendar days)")
        self.days.setFixedWidth(96)
        self.bt = QCheckBox("Backtest")
        self.bt.setToolTip(f"Hide the last {HIDE} trading days, predict from that date, then compare with what happened")
        self.btn = GlowButton("Predict")
        self.dot = QLabel()
        self.dot.setFixedSize(9, 9)
        self.dot_fx = glow(self.dot, DIM, 12)
        self.status = QLabel(objectName="status")
        self.status.setFixedWidth(150)
        for w in (brand, self.ticker, self.days, self.bt, self.btn, self.dot, self.status):
            row.addWidget(w, 0, Qt.AlignmentFlag.AlignVCenter)
        self._status("Enter a US ticker", DIM)
        isl.adjustSize()
        return isl

    def _status(self, text, color):
        self.status.setText(self.status.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, 148))
        self.status.setToolTip(text)
        self.dot.setStyleSheet(f"background: {color}; border-radius: 4px;")
        self.dot_fx.setColor(QColor(color))

    def _place_island(self):
        self.island.adjustSize()
        self.island.move(self.web.x() + (self.web.width() - self.island.width()) // 2, 8)
        self.island.raise_()

    def eventFilter(self, obj, ev):
        if obj is self.web and ev.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._place_island)
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
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(k)
        q = QLabel("?", objectName="help")  # hover = longer explanation
        q.setCursor(Qt.CursorShape.WhatsThisCursor)
        q.setStyleSheet(f"color: {CYAN}; border: 1px solid {CYAN}; border-radius: 7px; font-size: 10px; "
                        f"font-weight: 700; min-width: 14px; max-width: 14px; min-height: 14px; max-height: 14px;")
        q.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tips[key] = q
        top.addWidget(q)
        top.addStretch(1)
        v.addLayout(top)
        lab = QLabel("—", objectName="big" if big else "val")
        v.addWidget(lab)
        self.vals[key] = lab
        hp = QLabel(objectName="disc", wordWrap=True)  # plain-words meaning, always visible
        hp.setStyleSheet("color: #8a90a0;")
        v.addWidget(hp)
        self.helplbl[key] = hp
        return c

    def _panel(self):
        outer = QScrollArea(objectName="panelscroll")  # content taller than the window scrolls vertically
        outer.setWidgetResizable(True)
        outer.setFrameShape(QFrame.Shape.NoFrame)
        outer.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.setMinimumWidth(PANEL_MIN)
        outer.setStyleSheet("QScrollArea#panelscroll, QScrollArea#panelscroll > QWidget > QWidget { background: transparent; }")
        outer.viewport().setAutoFillBackground(False)
        p = QWidget(objectName="panel")
        outer.setWidget(p)
        v = QVBoxLayout(p)
        v.setContentsMargins(14, 56, 14, 18)  # top clears the floating window controls
        v.setSpacing(0)
        self.head = QLabel("FORECAST", objectName="head")
        hf = self.head.font()
        hf.setLetterSpacing(hf.SpacingType.AbsoluteSpacing, 2.4)
        self.head.setFont(hf)
        self.head.setContentsMargins(10, 0, 0, 4)
        v.addWidget(self.head)
        self.model_lbl = QLabel(objectName="status", wordWrap=True)
        self.model_lbl.setContentsMargins(10, 0, 0, 4)
        v.addWidget(self.model_lbl)
        v.addSpacing(8)
        self.vals, self.tips, self.helplbl = {}, {}, {}
        self.today = ("—", FG)
        self.dir_card = self._card("Direction", big=True)
        self.dir_fx = glow(self.vals["Direction"], DIM, 28)
        g = QGridLayout()
        g.setSpacing(2)
        g.setRowMinimumHeight(1, 0)
        g.addWidget(self.dir_card, 0, 0, 1, 2)
        g.addWidget(self._card("Confidence"), 1, 0)
        g.addWidget(self._card("Last close"), 1, 1)
        g.addWidget(self._card("Today"), 2, 0)
        g.addWidget(self._card("Horizon"), 2, 1)
        v.addLayout(g)
        self._quality_block(v)
        self.note = QLabel("", objectName="disc", wordWrap=True)
        self.note.setContentsMargins(12, 6, 12, 0)
        v.addWidget(self.note)
        self.reveal_btn = QPushButton("Reveal actual prices")
        self.reveal_btn.hide()
        v.addWidget(self.reveal_btn)
        v.addStretch(1)
        d = QLabel(DISCLAIMER, objectName="disc", wordWrap=True)
        d.setContentsMargins(12, 0, 12, 0)
        v.addWidget(d)
        self._set({})
        return outer

    # ---- "How good is this model?" + "Compared with other models" ----
    def _section(self, v, title):
        v.addSpacing(14)
        h = QLabel(title.upper(), objectName="head")
        h.setContentsMargins(10, 0, 0, 4)
        v.addWidget(h)

    def _quality_block(self, v):
        self._section(v, "How good is this model?")
        self.q_info = QLabel(objectName="status", wordWrap=True)
        self.q_info.setContentsMargins(10, 0, 10, 4)
        v.addWidget(self.q_info)
        self.q_rows = {}
        for key, title in (("acc", "Direction accuracy"), ("ic", "Ranking skill (rank-IC)"),
                           ("unseen", "Works on unseen stocks?")):
            c = GlowFrame("#2a3142", radius=12, g=2)
            r = QVBoxLayout(c)
            r.setContentsMargins(18, 8, 18, 8)
            r.setSpacing(2)
            top = QHBoxLayout()
            top.addWidget(QLabel(title.upper(), objectName="key"))
            top.addStretch(1)
            meter, chip = Meter(), QLabel()
            meter.hide()
            top.addWidget(meter)
            top.addWidget(chip)
            r.addLayout(top)
            val = QLabel(objectName="val", wordWrap=True)
            note = QLabel(objectName="disc", wordWrap=True)
            note.setStyleSheet("color: #8a90a0;")
            r.addWidget(val)
            r.addWidget(note)
            v.addSpacing(2)
            v.addWidget(c)
            self.q_rows[key] = (val, note, chip, meter)
        self._section(v, "Compared with other models")
        self.cmp_chart = CompareChart()
        v.addWidget(self.cmp_chart)
        self.cmp_note = QLabel(objectName="disc", wordWrap=True)
        self.cmp_note.setContentsMargins(10, 0, 10, 0)
        v.addWidget(self.cmp_note)

    @staticmethod
    def _chip(chip, verdict):
        col = VERDICT_COLOR[verdict]
        chip.setText(verdict.upper())
        chip.setStyleSheet(f"color: {col}; border: 1px solid {col}; border-radius: 8px; padding: 0 7px; "
                           f"font-size: 10px; font-weight: 700;")

    def _quality(self, h, rid):
        q = explain.quality(rid, self.mgr.jobs) if rid else None
        rows = self.q_rows
        for val, _, chip, meter in rows.values():
            chip.setText("")
            chip.setStyleSheet("")
            meter.hide()
            val.setText("—")
        for k, t in explain.QUALITY_HELP.items():
            rows[k][1].setText(t)
        if q is None:
            self.q_info.setText("No trained model for this horizon, so there are no quality numbers yet.")
        else:
            yrs = f"{q['start_year']}-{TRAIN_END[:4]}" if q["start_year"] else "?"
            self.q_info.setText(f"{q['model']} · run {q['run']} · trained on {yrs} · forecasts {registry.horizon_name(h)} ahead "
                                f"({h} trading days)")
            f = lambda x, fmt: "n/a" if x is None else format(x, fmt)  # noqa: E731
            rows["acc"][0].setText(f"{f(q['acc'], '.1%')} right vs {f(q['base'], '.1%')} for always-guess-UP")
            self._chip(rows["acc"][2], q["acc_verdict"])
            rows["ic"][0].setText(f"rank-IC {f(q['ic'], '+.3f')}")
            self._chip(rows["ic"][2], q["ic_verdict"])
            rows["ic"][3].level, rows["ic"][3].lit = q["ic_level"], q["ic"] is not None
            rows["ic"][3].show()
            rows["unseen"][0].setText(f"rank-IC {f(q['unseen_ic'], '+.3f')}")
            self._chip(rows["unseen"][2], q["unseen_verdict"])
        cmp = explain.comparison(self.mgr.jobs, h, rid)
        self.cmp_chart.set_rows(cmp)
        self.cmp_note.setText("Same horizon only; best run of each model type. Cyan = the model used here."
                              if any(not r["active"] for r in cmp) else "Train the other models (python train_all.py) to compare")

    def _helps(self):
        h = self.days.value() if hasattr(self, "days") else DEFAULT_DAYS
        for k, lab in self.helplbl.items():
            lab.setText(explain.helper(k, self.pred, h))

    def _set(self, vals: dict, color: str = FG):
        self._helps()
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

    # ---- one Predict run ----
    def start(self):
        if not self.btn.isEnabled():  # busy: Enter in the field bypasses the disabled button; never overlap runs
            return
        t, h, bt = self.ticker.text().strip().upper() or "AAPL", self.days.value(), self.bt.isChecked()
        self.ticker.setText(t)
        if (t, h, bt) == self.last_key:  # already predicted OK: keep chart/forecast/panel as is
            self._status(f"Already predicted: {t}", GREEN)
            return
        self.last_key = None  # set again in finish() only if this run fully succeeds
        self.run_key = (t, h, bt)
        self.btn.setEnabled(False)
        self._status(f"Loading {t}…", CYAN)
        self.note.setText("")
        self.today = ("—", FG)  # clear stale change from the previous ticker
        self.pred = None
        self._set({})
        self._clear_forecast()
        self.pred, self.n_bars, self.errors = None, 0, []
        self.full = self.hidden = None
        self.reveal_btn.hide()
        self.handlers, args = {fetch_ohlcv: self.on_chart}, {fetch_ohlcv: (t,)}
        self.no_model = active.path(h) is None
        if self.no_model:
            self.note.setText(f"No active model for the {h}-day horizon.")
            self._status(NO_MODEL, "#f0b90b")
        else:
            self.handlers[run_predict], args[run_predict] = self.on_pred, (t, h, HIDE if bt else 0)
        self.pending = len(self.handlers)
        for fn in self.handlers:
            self.pool.start(lambda fn=fn: _work(self.sig, fn, args[fn]))

    def _settle(self, fn, res, err):
        """One slot for both jobs: a crash in one handler is reported, and the counter ticks regardless."""
        try:
            if err is None:
                self.handlers[fn](res)
        except Exception as e:  # noqa: BLE001 - handler bug must not leave the button disabled or kill the other job
            err = e
        finally:
            if err is not None:
                self.on_fail(fn, err)
            self.finish()

    def on_chart(self, df):
        if df is None or len(df) < MIN_BARS:
            raise ValueError("insufficient data")
        self.full = df.drop(columns=["volume"])
        if self.run_key[2] and len(df) < HIDE + 100:
            raise ValueError(f"not enough history to hide {HIDE} days")
        if not self.run_key[2]:  # Backtest: drawn in finish(), once the cutoff date is known
            self._show_bars(self.full)

    def _show_bars(self, price):
        self.chart.set(price)
        self.n_bars = len(price)
        last, prev = float(price["close"].iloc[-1]), float(price["close"].iloc[-2])
        if math.isfinite(last) and math.isfinite(prev) and prev:
            chg = last / prev - 1
            self.today = (("▲ " if chg >= 0 else "▼ ") + f"{chg:+.2%}", GREEN if chg >= 0 else RED)
            self.vals["Today"].setText(self.today[0])
            self.vals["Today"].setStyleSheet(f"color: {self.today[1]};")

    def on_pred(self, r):
        if r.get("ticker", self.run_key[0]) != self.run_key[0]:  # stale result for another ticker: never draw it
            return
        self.pred = r
        up = r["direction"] == "UP"
        p = r["confidence"]
        self._set({
            "Direction": f"{'▲' if up else '▼'} {r['direction']}",
            "Confidence": f"{max(p, 1 - p):.0%}" if r.get("calibrated", True) else "not reliable",
            "Last close": f"{r['last_close']:.2f}",
            "Horizon": f"{registry.horizon_name(r['horizon'])} · {r['horizon']}d",
        }, GREEN if up else RED)
        if self.run_key[2]:
            seen = str(r["date"]) <= TRAIN_END
            self.note.setText(f"Backtest from {r['date']} ({HIDE} days hidden)."
                              + (" Note: the model may have seen this period in training." if seen else ""))
        elif r.get("stale_days", 0) > 5:
            self.note.setText(f"Warning: last bar is {r['stale_days']} days old")

    def on_fail(self, fn, exc):
        msg = str(exc) or type(exc).__name__
        if fn is run_predict:  # predict failure only annotates the panel; the chart (fetch) stays as drawn
            self.note.setText(msg)
            self._set({})
        self.errors.append(msg)  # any failure -> "Error" status and no last_key cache

    # ---- chart: exactly one forecast at a time ----
    def _clear_forecast(self):
        """Remove EVERY previous forecast series with its legend row and markers. The library's Line.delete() throws on
        `legend.div.removeChild(row)` (the row lives in seriesContainer) BEFORE reaching removeSeries, so the series stayed."""
        c = self.chart.id
        for s in self.forecast:
            self.chart.run_script(f"""(function () {{
                const s = {s.id}; if (!s) return;
                const it = {c}.legend._lines.find(l => l.series === s.series);
                if (it) {{ {c}.legend._lines = {c}.legend._lines.filter(l => l !== it); it.row.remove(); }}
                s.series.setMarkers([]);
                {c}.chart.removeSeries(s.series);
                {c}._seriesList = {c}._seriesList.filter(x => x !== s.series);
                {s.id} = undefined;
            }})()""")
            if s in self.chart._lines:
                self.chart._lines.remove(s)
        self.forecast = []
        self.chart.clear_markers()

    def _line(self, name, color, style, width, t, v, labels=False):
        s = self.chart.create_line(name, color=color, style=style, width=width, price_line=False, price_label=labels)
        s.set(pd.DataFrame({"time": t, name: v}))
        self.forecast.append(s)
        return s

    def _draw_forecast(self):
        r = self.pred
        if r is None or not math.isfinite(r["last_close"]):
            return
        self._clear_forecast()  # belt and braces: never two forecasts on screen
        up = r["direction"] == "UP"
        col = GREEN if up else RED
        # one point per business day: the chart's time axis is index-based; with H+1 points the tip sits exactly H
        # business days after the last bar
        t = pd.bdate_range(r["date"], periods=r["horizon"] + 1).astype("datetime64[ns]")
        lc = r["last_close"]
        base = self._line("Last close", col, "dotted", 1, t, np.full(len(t), lc))  # the model gives a direction, not a size: no sloped line, no range
        p = r["confidence"]
        chance = f" {max(p, 1 - p):.0%}" if r.get("calibrated", True) else ""
        base.marker(time=t[-1], position="above" if up else "below", shape="arrow_up" if up else "arrow_down", color=col,
                    text=f"{'▲' if up else '▼'} {r['direction']}{chance} in {registry.horizon_name(r['horizon'])}")

    def reveal(self):
        """Backtest: draw the hidden days and compare the forecast with what really happened."""
        r, h = self.pred, self.hidden
        if r is None or h is None or h.empty:
            return
        self._show_bars(self.full)
        k = min(r["horizon"], len(h))  # the hidden window may be shorter than the horizon
        actual = float(h["close"].iloc[k - 1]) / r["last_close"] - 1
        ok = (actual >= 0) == (r["direction"] == "UP")
        self.note.setText(f"After {k} trading days: model said {r['direction']}, actual {actual:+.2%}. "
                          f"Direction {'RIGHT ✔' if ok else 'WRONG ✘'}.")
        self._status("Direction right" if ok else "Direction wrong", GREEN if ok else RED)
        self.reveal_btn.hide()

    def _fit(self):
        """Newest candle (and forecast tip) centre-right: last VIEW_BARS candles + forecast slots + RIGHT_PAD, price axis
        on autoscale. Deferred so the web view has its real size; a stale timer (ticker changed meanwhile) is a no-op."""
        n, key = self.n_bars, self.run_key
        if n < MIN_BARS:
            return
        tip = self.pred["horizon"] if self.pred else 0  # the forecast tip sits `horizon` slots after the last bar
        c = self.chart.id
        js = (f"{c}.chart.priceScale('right').applyOptions({{autoScale: true}});"
              f"{c}.chart.timeScale().setVisibleLogicalRange({{from: {max(0, n - VIEW_BARS)}, to: {n - 1 + tip + RIGHT_PAD}}});"
              f"{c}.chart.priceScale('right').applyOptions({{autoScale: true}})")  # from >= 0: a bad range can abort QtWebEngine
        QTimer.singleShot(250, lambda: self.run_key == key and self.chart.run_script(js))

    def finish(self, *_):
        self.pending -= 1
        if self.pending > 0:
            return
        try:
            if self.run_key[2] and self.full is not None:  # Backtest: cut the candles at the forecast's cutoff date
                cut = pd.Timestamp(self.pred["date"]) if self.pred else self.full["time"].iloc[-HIDE - 1]
                self.hidden = self.full[self.full["time"] > cut]
                self._show_bars(self.full[self.full["time"] <= cut])
                self.reveal_btn.setVisible(self.pred is not None and not self.hidden.empty)
        except Exception as e:  # noqa: BLE001
            self.errors.append(str(e) or type(e).__name__)
        try:
            self._draw_forecast()  # after BOTH jobs settled: candles first, then the single new forecast
        except Exception as e:  # noqa: BLE001 - a drawing problem must not leave the button disabled
            self.errors.append(str(e) or type(e).__name__)
        self._fit()
        self.btn.setEnabled(True)
        if self.errors:
            self._status(f"Error: {self.errors[0]}", RED)
        elif self.no_model:
            self._status(NO_MODEL, "#f0b90b")
        else:
            self.last_key = self.run_key
            self._status("Done", GREEN)
        if self._reload_pending:  # active model changed mid-run
            self.reload_model()
