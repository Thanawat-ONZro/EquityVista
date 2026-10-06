"""Shared look: TradingView-style canvas + neon accents, QSS, glow helpers and the glowing widgets."""
import math

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QLinearGradient, QPainter, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QPushButton

# TradingView-style canvas + neon accents (up = green, down = red: TradingView/Binance convention)
SHELL_R, GLOW_PX = 24, 14  # window corner radius, transparent margin that holds the outer glow
BG, CARD, FG, DIM, EDGE ="#0b0e11", "#131722", "#d1d4dc", "#787b86", "#1e2433"
CYAN, VIOLET, GREEN, RED = "#00e5ff", "#7c5cff", "#0ecb81", "#f6465d"
AMBER = "#f0b90b"
STATUS_COLOR = {"queued": DIM, "running": CYAN, "done": GREEN, "failed": RED, "cancelled": AMBER, "interrupted": AMBER}

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

QWidget#side {{ border-right: 1px solid {EDGE}; }}
QPushButton#nav {{ background: transparent; color: {DIM}; text-align: left; padding: 9px 14px; border-radius: 10px; }}
QPushButton#nav:hover {{ background: rgba(255,255,255,0.05); color: {FG}; }}
QPushButton#nav:checked {{ background: rgba(0,229,255,0.10); color: {CYAN}; }}
QPushButton#ghost {{ background: {CARD}; color: {FG}; border: 1px solid #2a3142; }}
QPushButton#ghost:hover {{ border: 1px solid {CYAN}; }}
QPushButton#ghost:disabled {{ background: {BG}; color: #3d4352; border: 1px solid {EDGE}; }}
QTableWidget {{ background: transparent; border: none; gridline-color: transparent;
               selection-background-color: rgba(124,92,255,0.25); }}
QTableWidget::item {{ border-bottom: 1px solid {EDGE}; padding: 4px 8px; }}
QHeaderView::section {{ background: {CARD}; color: {DIM}; border: none; border-bottom: 1px solid {EDGE};
                       padding: 6px 8px; font-size: 11px; font-weight: 700; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ padding: 3px 4px; }}
QListWidget::item:selected {{ background: rgba(124,92,255,0.25); }}
QPlainTextEdit {{ background: #0e1219; border: 1px solid {EDGE}; border-radius: 10px; padding: 6px;
                 font-family: 'Cascadia Mono', Consolas, monospace; font-size: 12px; }}
QComboBox {{ background: {BG}; border: 1px solid #2a3142; border-radius: 10px; padding: 5px 10px; }}
QComboBox QAbstractItemView {{ background: {CARD}; border: 1px solid {EDGE}; selection-background-color: rgba(124,92,255,0.35); }}
QProgressBar {{ background: {BG}; border: 1px solid {EDGE}; border-radius: 7px; text-align: center; font-size: 10px; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {CYAN}, stop:1 {VIOLET}); border-radius: 6px; }}
QSpinBox, QDoubleSpinBox {{ background: {BG}; border: 1px solid #2a3142; border-radius: 10px; padding: 5px 10px; font-weight: 600; }}
QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {CYAN}; }}
QSlider::groove:horizontal {{ height: 4px; background: {EDGE}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {CYAN}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; background: {FG}; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {DIM}; padding: 9px 18px; font-weight: 700; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {CYAN}; border-bottom: 2px solid {CYAN}; }}
QTabBar::tab:hover {{ color: {FG}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
QDialog {{ background: {CARD}; }}
QCheckBox {{ spacing: 6px; }}
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
