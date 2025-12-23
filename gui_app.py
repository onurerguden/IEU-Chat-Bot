"""
IEU Engineering Curriculum Chat - Desktop GUI
Modern PyQt5 interface with IEU branding
Author: Onur Ergüden inspired design
"""

import sys
import threading
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QScrollArea,
                             QFrame, QTextEdit, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QFont, QColor, QPainter, QPixmap, QPainterPath

# Add parent directory to path for imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from rag.rag_app import FAISSRAG


# Signal handler for thread-safe GUI updates
class SignalHandler(QObject):
    message_signal = pyqtSignal(str, str)  # role, content
    status_signal = pyqtSignal(str, str)  # text, color


STYLESHEET = """
QMainWindow { 
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #0f172a, stop:1 #1e293b);
}
QLabel { 
    color: #f1f5f9; 
    font-family: 'Segoe UI', Arial, sans-serif; 
}
QPushButton { 
    border: none; 
    border-radius: 12px; 
    padding: 12px 24px; 
    font-weight: bold; 
    font-size: 14px; 
    color: white; 
}
QPushButton#SendBtn { 
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #ff6b35, stop:1 #ff8c42);
}
QPushButton#SendBtn:hover { 
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #e85a2a, stop:1 #e67935);
}
QPushButton#SendBtn:disabled { 
    background-color: #475569; 
    color: #94a3b8; 
}
QScrollArea { 
    border: none; 
    background-color: transparent; 
}
QTextEdit {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 2px solid #334155;
    border-radius: 12px;
    padding: 12px;
    font-size: 13px;
    font-family: 'Segoe UI', Arial;
}
QTextEdit:focus {
    border: 2px solid #ff6b35;
}
"""


class IEULogo(QLabel):
    """Custom IEU Logo Widget - Diamond shape with gradient"""

    def __init__(self, size=60):
        super().__init__()
        self.setFixedSize(size, size)
        self.logo_size = size

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create diamond shape
        path = QPainterPath()
        center = self.logo_size / 2
        path.moveTo(center, 5)  # Top
        path.lineTo(self.logo_size - 5, center)  # Right
        path.lineTo(center, self.logo_size - 5)  # Bottom
        path.lineTo(5, center)  # Left
        path.closeSubpath()

        # Gradient fill
        from PyQt5.QtGui import QLinearGradient, QBrush
        gradient = QLinearGradient(0, 0, self.logo_size, self.logo_size)
        gradient.setColorAt(0, QColor("#ff6b35"))
        gradient.setColorAt(1, QColor("#ff8c42"))

        painter.fillPath(path, QBrush(gradient))

        # Draw "IEU" text
        painter.setPen(Qt.white)
        painter.setFont(QFont("Segoe UI", int(self.logo_size * 0.25), QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, "IEU")


class MessageBubble(QFrame):
    """Modern chat message bubble"""

    def __init__(self, text, is_user=False, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.full_text = text
        self.displayed_text = ""
        self.char_index = 0

        self.setMaximumWidth(650)

        if is_user:
            self.setStyleSheet("""
                QFrame {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                stop:0 #ff6b35, stop:1 #ff8c42);
                    border-radius: 16px;
                    padding: 4px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: #1e293b;
                    border-radius: 16px;
                    border: 1px solid #334155;
                    padding: 4px;
                }
            """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)

        # Role label
        role_label = QLabel("You" if is_user else "Assistant")
        role_label.setStyleSheet(f"""
            color: {'white' if is_user else '#ff6b35'};
            font-size: 10px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        """)
        layout.addWidget(role_label)

        # Message content
        self.content_label = QLabel("")
        self.content_label.setWordWrap(True)
        self.content_label.setTextFormat(Qt.PlainText)
        self.content_label.setStyleSheet(f"""
            color: {'white' if is_user else '#e2e8f0'};
            font-size: 13px;
            line-height: 1.6;
            padding: 4px 0;
        """)
        layout.addWidget(self.content_label)

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

        # For bot messages, animate typing
        if not is_user:
            self.typing_timer = QTimer()
            self.typing_timer.timeout.connect(self.animate_typing)
            self.typing_timer.start(10)  # Fast typing effect
        else:
            self.content_label.setText(text)
            self.adjustSize()

    def animate_typing(self):
        """Typing animation effect"""
        if self.char_index < len(self.full_text):
            self.displayed_text += self.full_text[self.char_index]
            self.content_label.setText(self.displayed_text)
            self.char_index += 1
            self.adjustSize()
        else:
            self.typing_timer.stop()


class ChatContainer(QWidget):
    """Scrollable chat container"""

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setSpacing(16)
        self.layout.setContentsMargins(20, 20, 20, 20)

    def add_message(self, text, is_user=False):
        """Add a message bubble"""
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        bubble = MessageBubble(text, is_user)

        if is_user:
            container_layout.addStretch()
            container_layout.addWidget(bubble)
        else:
            container_layout.addWidget(bubble)
            container_layout.addStretch()

        self.layout.addWidget(container)
        return bubble


class LoadingIndicator(QLabel):
    """Animated loading dots"""

    def __init__(self):
        super().__init__("Thinking")
        self.setStyleSheet("""
            color: #94a3b8;
            font-size: 13px;
            font-style: italic;
            padding: 12px 20px;
            background-color: #1e293b;
            border-radius: 12px;
            border: 1px solid #334155;
        """)
        self.dots = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_dots)

    def start(self):
        self.timer.start(400)
        self.show()

    def stop(self):
        self.timer.stop()
        self.hide()

    def update_dots(self):
        self.dots = (self.dots + 1) % 4
        self.setText("Thinking" + "." * self.dots)


class MainChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IEU Engineering Curriculum Assistant")
        self.resize(1200, 850)
        self.setStyleSheet(STYLESHEET)

        # Signal handler for thread-safe updates
        self.signals = SignalHandler()
        self.signals.message_signal.connect(self._add_message_to_ui)
        self.signals.status_signal.connect(self._update_status_ui)

        self.rag = None
        self.is_processing = False

        self.setup_ui()
        self.initialize_rag()

    def setup_ui(self):
        """Setup the main UI"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # === HEADER ===
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #1e293b, stop:1 #334155);
                border-bottom: 2px solid #ff6b35;
            }
        """)
        header.setFixedHeight(100)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 20, 30, 20)

        # Logo
        logo = IEULogo(60)
        header_layout.addWidget(logo)

        header_layout.addSpacing(20)

        # Title section
        title_container = QVBoxLayout()
        title_container.setSpacing(4)

        title = QLabel("IEU Engineering")
        title.setStyleSheet("""
            color: white;
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.5px;
        """)

        subtitle = QLabel("Curriculum Assistant • AI Powered")
        subtitle.setStyleSheet("""
            color: #ff6b35;
            font-size: 13px;
            font-weight: 600;
        """)

        title_container.addWidget(title)
        title_container.addWidget(subtitle)
        header_layout.addLayout(title_container)

        header_layout.addStretch()

        # Status indicator
        self.status_badge = QLabel("● Initializing")
        self.status_badge.setStyleSheet("""
            color: #fbbf24;
            font-size: 12px;
            font-weight: 600;
            padding: 8px 16px;
            background-color: rgba(251, 191, 36, 0.1);
            border-radius: 20px;
            border: 1px solid #fbbf24;
        """)
        header_layout.addWidget(self.status_badge)

        main_layout.addWidget(header)

        # === CHAT AREA ===
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("QScrollArea { background-color: #0f172a; }")

        self.chat_container = ChatContainer()
        self.scroll_area.setWidget(self.chat_container)

        main_layout.addWidget(self.scroll_area)

        # === LOADING INDICATOR ===
        loading_container = QWidget()
        loading_layout = QHBoxLayout(loading_container)
        loading_layout.setContentsMargins(20, 0, 20, 0)

        self.loading_indicator = LoadingIndicator()
        self.loading_indicator.hide()
        loading_layout.addWidget(self.loading_indicator)
        loading_layout.addStretch()

        main_layout.addWidget(loading_container)

        # === INPUT AREA ===
        input_container = QFrame()
        input_container.setStyleSheet("""
            QFrame {
                background-color: #1e293b;
                border-top: 1px solid #334155;
            }
        """)
        input_container.setFixedHeight(140)
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(30, 20, 30, 20)

        input_row = QHBoxLayout()
        input_row.setSpacing(15)

        # Text input
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Ask about courses, requirements, ECTS credits...")
        self.input_field.setFixedHeight(80)
        self.input_field.setFont(QFont("Segoe UI", 12))
        input_row.addWidget(self.input_field)

        # Send button
        self.send_btn = QPushButton("Send ▶")
        self.send_btn.setObjectName("SendBtn")
        self.send_btn.setFixedSize(120, 80)
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.clicked.connect(self.send_message)

        btn_shadow = QGraphicsDropShadowEffect()
        btn_shadow.setBlurRadius(20)
        btn_shadow.setColor(QColor(255, 107, 53, 100))
        btn_shadow.setOffset(0, 4)
        self.send_btn.setGraphicsEffect(btn_shadow)

        input_row.addWidget(self.send_btn)
        input_layout.addLayout(input_row)

        main_layout.addWidget(input_container)

        # Key bindings
        self.input_field.installEventFilter(self)

        # Welcome message
        self.add_welcome_message()

    def add_welcome_message(self):
        """Add initial welcome messages"""
        welcome = """Hello! I can help you with information about courses, prerequisites, ECTS credits, and curriculum details across all engineering departments.

Try asking:
• What are the second year Software Engineering courses?
• Which courses cover machine learning?
• List mandatory courses in Computer Engineering"""

        self.chat_container.add_message(welcome, is_user=False)

    def initialize_rag(self):
        """Initialize RAG system in background"""

        def load_rag():
            try:
                self.rag = FAISSRAG()
                self.signals.status_signal.emit(
                    f"● Ready ({len(self.rag.data)} chunks loaded)",
                    "#10b981"
                )
            except Exception as e:
                self.signals.status_signal.emit("● Error", "#ef4444")
                self.signals.message_signal.emit("system", f"Error loading RAG: {str(e)}")

        thread = threading.Thread(target=load_rag, daemon=True)
        thread.start()

    def send_message(self):
        """Handle send button click"""
        if self.is_processing or not self.rag:
            return

        message = self.input_field.toPlainText().strip()
        if not message:
            return

        # Clear input
        self.input_field.clear()

        # Add user message
        self.chat_container.add_message(message, is_user=True)
        self.scroll_to_bottom()

        # Show loading
        self.loading_indicator.start()
        self.is_processing = True
        self.send_btn.setEnabled(False)
        self.send_btn.setText("Processing...")
        self.signals.status_signal.emit("● Processing", "#fbbf24")

        # Get response in background
        def get_response():
            try:
                response = self.rag.answer(message)
                self.signals.message_signal.emit("assistant", response)
            except Exception as e:
                self.signals.message_signal.emit("assistant", f"Error: {str(e)}")

        thread = threading.Thread(target=get_response, daemon=True)
        thread.start()

    def _add_message_to_ui(self, role, content):
        """Thread-safe message addition"""
        self.loading_indicator.stop()

        if role == "assistant":
            self.chat_container.add_message(content, is_user=False)

        self.scroll_to_bottom()
        self.is_processing = False
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send ▶")
        self.signals.status_signal.emit("● Ready", "#10b981")
        self.input_field.setFocus()

    def _update_status_ui(self, text, color):
        """Thread-safe status update"""
        self.status_badge.setText(text)

        # Update border color based on status
        if color == "#10b981":  # Ready/Success
            bg_color = "rgba(16, 185, 129, 0.1)"
        elif color == "#fbbf24":  # Processing
            bg_color = "rgba(251, 191, 36, 0.1)"
        else:  # Error
            bg_color = "rgba(239, 68, 68, 0.1)"

        self.status_badge.setStyleSheet(f"""
            color: {color};
            font-size: 12px;
            font-weight: 600;
            padding: 8px 16px;
            background-color: {bg_color};
            border-radius: 20px;
            border: 1px solid {color};
        """)

    def scroll_to_bottom(self):
        """Scroll chat to bottom"""
        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def eventFilter(self, obj, event):
        """Handle key events"""
        if obj == self.input_field and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Return and not event.modifiers() & Qt.ShiftModifier:
                self.send_message()
                return True
        return super().eventFilter(obj, event)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    window = MainChatWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()