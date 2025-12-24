"""
IEU Engineering Curriculum Chat - Modern PyQt6 Implementation
Clean flat design without gradients and shadows
Author: Senior Frontend Engineer Refactor
"""

import sys
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QScrollArea, QFrame, QTextEdit, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QObject, QPropertyAnimation,
    QEasingCurve, QRect, QSize, pyqtSlot
)
from PyQt6.QtGui import (
    QFont, QColor, QPainter, QPixmap, QPainterPath,
    QBrush, QPalette, QKeyEvent
)

# Add parent directory for imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from rag.rag_app import FAISSRAG


# ============================================================================
# THEME SYSTEM
# ============================================================================

@dataclass
class ColorScheme:
    """Flat color scheme definition"""
    primary: str
    primary_hover: str
    secondary: str
    background: str
    surface: str
    surface_variant: str
    text_primary: str
    text_secondary: str
    border: str
    border_focus: str
    success: str
    warning: str
    error: str


class Theme:
    """Centralized theme management"""

    DARK = ColorScheme(
        primary="#e85a2a",  # Daha koyu turuncu
        primary_hover="#d14d20",  # Hover için daha da koyu
        secondary="#e85a2a",
        background="#0f172a",
        surface="#1e293b",
        surface_variant="#334155",
        text_primary="#f1f5f9",
        text_secondary="#94a3b8",
        border="#334155",
        border_focus="#e85a2a",
        success="#10b981",
        warning="#fbbf24",
        error="#ef4444"
    )

    @staticmethod
    def get_stylesheet(scheme: ColorScheme) -> str:
        """Generate flat stylesheet from color scheme"""
        return f"""
        QMainWindow {{ 
            background-color: {scheme.background};
        }}

        QLabel {{ 
            color: {scheme.text_primary}; 
            font-family: 'Segoe UI', 'San Francisco', 'Roboto', sans-serif; 
        }}

        QPushButton {{ 
            border: none; 
            border-radius: 12px; 
            padding: 12px 24px; 
            font-weight: 600; 
            font-size: 14px; 
            color: white;
        }}

        QPushButton#SendBtn {{ 
            background-color: {scheme.primary};
        }}

        QPushButton#SendBtn:hover {{ 
            background-color: {scheme.primary_hover};
        }}

        QPushButton#SendBtn:disabled {{ 
            background-color: {scheme.surface_variant}; 
            color: {scheme.text_secondary}; 
        }}

        QScrollArea {{ 
            border: none; 
            background-color: transparent; 
        }}

        QTextEdit {{
            background-color: {scheme.surface};
            color: {scheme.text_primary};
            border: 2px solid {scheme.border};
            border-radius: 12px;
            padding: 12px;
            font-size: 13px;
            font-family: 'Segoe UI', 'San Francisco', 'Roboto', sans-serif;
            selection-background-color: {scheme.primary};
        }}

        QTextEdit:focus {{
            border: 2px solid {scheme.border_focus};
        }}

        QScrollBar:vertical {{
            background: {scheme.surface};
            width: 12px;
            border-radius: 6px;
            margin: 0;
        }}

        QScrollBar::handle:vertical {{
            background: {scheme.surface_variant};
            border-radius: 6px;
            min-height: 30px;
        }}

        QScrollBar::handle:vertical:hover {{
            background: {scheme.text_secondary};
        }}

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        """


# ============================================================================
# SIGNAL HANDLER
# ============================================================================

class SignalHandler(QObject):
    """Thread-safe signal handler for UI updates"""
    message_signal = pyqtSignal(str, str)  # role, content
    status_signal = pyqtSignal(str, str, str)  # text, color, bg_color


# ============================================================================
# CUSTOM WIDGETS
# ============================================================================

class IEULogo(QWidget):
    """Flat IEU Logo"""

    def __init__(self, size: int = 60, parent=None):
        super().__init__(parent)
        self.logo_size = size
        self.setFixedSize(size, size)
        self.theme = Theme.DARK

    def paintEvent(self, event):
        """Custom paint with antialiasing"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Diamond path
        path = QPainterPath()
        center = self.logo_size / 2
        margin = 5

        path.moveTo(center, margin)
        path.lineTo(self.logo_size - margin, center)
        path.lineTo(center, self.logo_size - margin)
        path.lineTo(margin, center)
        path.closeSubpath()

        # Flat color fill
        painter.fillPath(path, QBrush(QColor(self.theme.primary)))

        # Text
        painter.setPen(Qt.GlobalColor.white)
        font = QFont("Segoe UI", int(self.logo_size * 0.25), QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "IEU")


class MessageBubble(QFrame):
    """Flat message bubble with timestamp"""

    def __init__(self, text: str, is_user: bool = False, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.full_text = text
        self.displayed_text = ""
        self.char_index = 0
        self.theme = Theme.DARK
        self.timestamp = datetime.now()

        # Responsive sizing
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.setMaximumWidth(650)

        self._setup_ui()
        self._apply_styles()

        # Typing animation for assistant
        if not is_user:
            self.typing_timer = QTimer()
            self.typing_timer.timeout.connect(self._animate_typing)
            self.typing_timer.start(8)
        else:
            self.content_label.setText(text)
            self.adjustSize()

    def _setup_ui(self):
        """Setup bubble layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Header with role and timestamp
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)

        # Role label
        self.role_label = QLabel("You" if self.is_user else "Assistant")
        self.role_label.setStyleSheet(f"""
            color: {'white' if self.is_user else self.theme.primary};
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.2px;
        """)
        header_layout.addWidget(self.role_label)

        # Timestamp
        color = 'white' if not self.is_user else self.theme.text_secondary
        self.time_label = QLabel(self.timestamp.strftime("%H:%M"))
        self.time_label.setStyleSheet(f"""
            color: {color};
            font-size: 9px;
            font-weight: 500;
        """)
        header_layout.addWidget(self.time_label)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Message content
        self.content_label = QLabel("")
        self.content_label.setWordWrap(True)
        self.content_label.setTextFormat(Qt.TextFormat.PlainText)
        self.content_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.content_label.setStyleSheet(f"""
            color: {'white' if self.is_user else self.theme.text_primary};
            font-size: 13px;
            line-height: 1.6;
            padding: 4px 0;
        """)
        layout.addWidget(self.content_label)

    def _apply_styles(self):
        """Apply flat bubble styling"""
        if self.is_user:
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {self.theme.primary};
                    border-radius: 16px;
                    padding: 4px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {self.theme.surface};
                    border-radius: 16px;
                    border: 1px solid {self.theme.border};
                    padding: 4px;
                }}
            """)

    @pyqtSlot()
    def _animate_typing(self):
        """Smooth typing animation"""
        if self.char_index < len(self.full_text):
            chunk_size = 2 if len(self.full_text) > 100 else 1
            end_idx = min(self.char_index + chunk_size, len(self.full_text))

            self.displayed_text = self.full_text[:end_idx]
            self.content_label.setText(self.displayed_text)
            self.char_index = end_idx
            self.adjustSize()
        else:
            self.typing_timer.stop()


class ChatContainer(QWidget):
    """Responsive scrollable chat container"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.setSpacing(16)
        self.layout.setContentsMargins(20, 20, 20, 20)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

    def add_message(self, text: str, is_user: bool = False) -> MessageBubble:
        """Add message with proper alignment"""
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        bubble = MessageBubble(text, is_user)

        if is_user:
            container_layout.addStretch()
            container_layout.addWidget(bubble)
        else:
            container_layout.addWidget(bubble)
            container_layout.addStretch()

        self.layout.addWidget(container)
        return bubble

    def resizeEvent(self, event):
        """Handle container resize"""
        super().resizeEvent(event)
        self.updateGeometry()


class ThinkingIndicator(QWidget):
    """Animated thinking indicator with dots"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme = Theme.DARK
        self.dot_count = 3
        self.current_dot = 0

        # Setup UI
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(8)

        self.text_label = QLabel("Thinking")
        self.text_label.setStyleSheet(f"""
            color: {self.theme.text_secondary};
            font-size: 13px;
            font-weight: 500;
        """)
        layout.addWidget(self.text_label)

        # Create dot labels
        self.dots = []
        for i in range(self.dot_count):
            dot = QLabel("●")
            dot.setStyleSheet(f"""
                color: {self.theme.text_secondary};
                font-size: 16px;
            """)
            self.dots.append(dot)
            layout.addWidget(dot)

        layout.addStretch()

        # Container styling
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.theme.surface};
                border-radius: 12px;
                border: 1px solid {self.theme.border};
            }}
        """)

        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._animate_dots)

    def start(self):
        """Start animation"""
        self.current_dot = 0
        self._update_dots()
        self.timer.start(400)
        self.show()

    def stop(self):
        """Stop animation"""
        self.timer.stop()
        self.hide()

    @pyqtSlot()
    def _animate_dots(self):
        """Animate dots"""
        self.current_dot = (self.current_dot + 1) % (self.dot_count + 1)
        self._update_dots()

    def _update_dots(self):
        """Update dot colors"""
        for i, dot in enumerate(self.dots):
            if i < self.current_dot:
                # Active dot
                dot.setStyleSheet(f"""
                    color: {self.theme.primary};
                    font-size: 16px;
                """)
            else:
                # Inactive dot
                dot.setStyleSheet(f"""
                    color: {self.theme.text_secondary};
                    font-size: 16px;
                """)


class StatusBadge(QLabel):
    """Flat status indicator"""

    def __init__(self, parent=None):
        super().__init__("● Initializing", parent)
        self.theme = Theme.DARK
        self._current_color = self.theme.warning

    def update_status(self, text: str, color: str, bg_color: str):
        """Update status with color"""
        self.setText(text)
        self._current_color = color

        self.setStyleSheet(f"""
            color: {color};
            font-size: 12px;
            font-weight: 600;
            padding: 8px 16px;
            background-color: {bg_color};
            border-radius: 20px;
            border: 1px solid {color};
        """)


# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainChatWindow(QMainWindow):
    """Professional chat interface with flat design"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IEU Engineering Curriculum Assistant")
        self.resize(1200, 850)
        self.setMinimumSize(800, 600)

        # State management
        self.theme = Theme.DARK
        self.rag: Optional[FAISSRAG] = None
        self.is_processing = False

        # Signal handler
        self.signals = SignalHandler()
        self.signals.message_signal.connect(self._add_message_to_ui)
        self.signals.status_signal.connect(self._update_status_ui)

        # Setup
        self.setStyleSheet(Theme.get_stylesheet(self.theme))
        self._setup_ui()
        self._initialize_rag()

    def _setup_ui(self):
        """Setup main UI with responsive layout"""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header
        main_layout.addWidget(self._create_header())

        # Chat area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setStyleSheet(
            f"QScrollArea {{ background-color: {self.theme.background}; }}"
        )

        self.chat_container = ChatContainer()
        self.scroll_area.setWidget(self.chat_container)
        main_layout.addWidget(self.scroll_area, stretch=1)

        # Thinking indicator
        thinking_container = QWidget()
        thinking_layout = QHBoxLayout(thinking_container)
        thinking_layout.setContentsMargins(20, 0, 20, 10)

        self.thinking_indicator = ThinkingIndicator()
        self.thinking_indicator.hide()
        thinking_layout.addWidget(self.thinking_indicator)
        thinking_layout.addStretch()

        main_layout.addWidget(thinking_container)

        # Input area
        main_layout.addWidget(self._create_input_area())

        # Welcome message
        self._add_welcome_message()

    def _create_header(self) -> QFrame:
        """Create flat header"""
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {self.theme.surface};
                border-bottom: 2px solid {self.theme.primary};
            }}
        """)
        header.setFixedHeight(100)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(20)

        # Logo
        logo = IEULogo(60)
        layout.addWidget(logo)

        # Title section
        title_container = QVBoxLayout()
        title_container.setSpacing(4)

        title = QLabel("IEU Engineering")
        title.setStyleSheet(f"""
            color: white;
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.5px;
        """)

        subtitle = QLabel("Curriculum Assistant • AI Powered")
        subtitle.setStyleSheet(f"""
            color: {self.theme.primary};
            font-size: 13px;
            font-weight: 600;
        """)

        title_container.addWidget(title)
        title_container.addWidget(subtitle)
        layout.addLayout(title_container)
        layout.addStretch()

        # Status badge
        self.status_badge = StatusBadge()
        self.status_badge.update_status(
            "● Initializing",
            self.theme.warning,
            "rgba(251, 191, 36, 0.1)"
        )
        layout.addWidget(self.status_badge)

        return header

    def _create_input_area(self) -> QFrame:
        """Create flat input area"""
        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background-color: {self.theme.surface};
                border-top: 1px solid {self.theme.border};
            }}
        """)
        container.setFixedHeight(140)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(30, 20, 30, 20)

        input_row = QHBoxLayout()
        input_row.setSpacing(15)

        # Text input
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText(
            "Ask about courses, requirements, ECTS credits..."
        )
        self.input_field.setFixedHeight(80)
        self.input_field.setFont(QFont("Segoe UI", 12))
        self.input_field.installEventFilter(self)
        input_row.addWidget(self.input_field)

        # Send button
        self.send_btn = QPushButton("Send ▶")
        self.send_btn.setObjectName("SendBtn")
        self.send_btn.setFixedSize(120, 80)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.clicked.connect(self._send_message)

        input_row.addWidget(self.send_btn)
        layout.addLayout(input_row)

        return container

    def _add_welcome_message(self):
        """Add welcome message"""
        welcome = """Hello! I can help you with information about courses, prerequisites, ECTS credits, and curriculum details across all engineering departments.

Try asking:
• What are the second year Software Engineering courses?
• Which courses cover machine learning?
• List mandatory courses in Computer Engineering"""

        self.chat_container.add_message(welcome, is_user=False)

    def _initialize_rag(self):
        """Initialize RAG system asynchronously"""

        def load_rag():
            try:
                self.rag = FAISSRAG()
                chunk_count = len(self.rag.data)
                self.signals.status_signal.emit(
                    f"● Ready ({chunk_count} chunks loaded)",
                    self.theme.success,
                    "rgba(16, 185, 129, 0.1)"
                )
            except Exception as e:
                self.signals.status_signal.emit(
                    "● Error",
                    self.theme.error,
                    "rgba(239, 68, 68, 0.1)"
                )
                self.signals.message_signal.emit(
                    "system",
                    f"Error loading RAG: {str(e)}"
                )

        thread = threading.Thread(target=load_rag, daemon=True)
        thread.start()

    @pyqtSlot()
    def _send_message(self):
        """Handle message sending"""
        if self.is_processing or not self.rag:
            return

        message = self.input_field.toPlainText().strip()
        if not message:
            return

        # Clear and add user message
        self.input_field.clear()
        self.chat_container.add_message(message, is_user=True)
        self._scroll_to_bottom()

        # Update UI state
        self.thinking_indicator.start()
        self.is_processing = True
        self.send_btn.setEnabled(False)
        self.send_btn.setText("Processing...")
        self.signals.status_signal.emit(
            "● Processing",
            self.theme.warning,
            "rgba(251, 191, 36, 0.1)"
        )

        # Get response asynchronously
        def get_response():
            try:
                response = self.rag.answer(message)
                self.signals.message_signal.emit("assistant", response)
            except Exception as e:
                self.signals.message_signal.emit(
                    "assistant",
                    f"Error: {str(e)}"
                )

        thread = threading.Thread(target=get_response, daemon=True)
        thread.start()

    @pyqtSlot(str, str)
    def _add_message_to_ui(self, role: str, content: str):
        """Thread-safe message addition"""
        self.thinking_indicator.stop()

        if role == "assistant":
            self.chat_container.add_message(content, is_user=False)

        self._scroll_to_bottom()

        # Reset UI state
        self.is_processing = False
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send ▶")
        self.signals.status_signal.emit(
            "● Ready",
            self.theme.success,
            "rgba(16, 185, 129, 0.1)"
        )
        self.input_field.setFocus()

    @pyqtSlot(str, str, str)
    def _update_status_ui(self, text: str, color: str, bg_color: str):
        """Thread-safe status update"""
        self.status_badge.update_status(text, color, bg_color)

    def _scroll_to_bottom(self):
        """Smooth scroll to bottom"""
        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def eventFilter(self, obj: QObject, event) -> bool:
        """Handle keyboard shortcuts"""
        if obj == self.input_field and event.type() == event.Type.KeyPress:
            key_event: QKeyEvent = event
            if (key_event.key() == Qt.Key.Key_Return and
                    not key_event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                self._send_message()
                return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        """Handle window resize"""
        super().resizeEvent(event)
        # Adjust max bubble width based on window size
        new_max_width = min(650, int(self.width() * 0.7))
        for i in range(self.chat_container.layout.count()):
            item = self.chat_container.layout.itemAt(i)
            if item and item.widget():
                container = item.widget()
                for j in range(container.layout().count()):
                    widget = container.layout().itemAt(j).widget()
                    if isinstance(widget, MessageBubble):
                        widget.setMaximumWidth(new_max_width)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Set application metadata
    app.setApplicationName("IEU Chat Assistant")
    app.setOrganizationName("IEU Engineering")
    app.setApplicationVersion("2.0.0")

    # Launch main window
    window = MainChatWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()