"""
Logging utilities for the Deep Particle Tracker GUI.

Contains the ConsoleLogHandler for redirecting log messages to a QTextEdit widget,
and the TrainingCompletionHandler for thread-safe UI updates.
"""

import logging
import time
import datetime

from PyQt5.QtCore import QTimer, QMutex, QObject, pyqtSlot


class TrainingCompletionHandler(QObject):
    """Helper class to handle training completion in the UI thread."""

    def __init__(self, training_tab):
        super().__init__()
        self.training_tab = training_tab

    @pyqtSlot(str, object)
    def on_training_completed(self, training_id, result):
        """Safely handle training completion in the UI thread."""
        self.training_tab.train_button.setEnabled(True)
        self.training_tab.stop_button.setEnabled(False)
        self.training_tab.status_label.setText(f"Training {training_id} completed")
        self.training_tab.progress_bar.setValue(100)
        self.training_tab.refresh_trainings()


class ConsoleLogHandler(logging.Handler):
    """Logging handler to redirect logs to a QTextEdit widget with crash prevention."""

    def __init__(self, console_widget, max_lines=1000, rate_limit=100):
        """
        Initialize the console log handler with crash prevention.

        Args:
            console_widget: QTextEdit widget to display logs
            max_lines: Maximum number of lines to keep in console
            rate_limit: Maximum number of log messages per second
        """
        super().__init__()
        self.console = console_widget
        self.max_lines = max_lines
        self.rate_limit = rate_limit
        self.message_count = 0
        self.last_flush_time = time.time()
        self.message_queue = []
        self.mutex = QMutex()

        self.flush_timer = QTimer()
        self.flush_timer.timeout.connect(self.periodic_flush)
        self.flush_timer.start(100)

        self.last_message = None
        self.repeat_count = 0
        self.limited_warnings = set()

        self.console.clear()
        self.console.append('<font color="blue">Logging initialized. System ready.</font>')

    def emit(self, record):
        """
        Queue a log record for display.

        Args:
            record: Log record
        """
        if (record.levelno == logging.WARNING and
            ('shape' in record.getMessage() or 'dimension' in record.getMessage())):
            warning_key = f"{record.name}_{record.funcName}_{record.lineno}"
            if warning_key in self.limited_warnings and len(self.limited_warnings) > 10:
                return
            self.limited_warnings.add(warning_key)

        msg = self.format(record)

        if msg == self.last_message:
            self.repeat_count += 1
            if self.repeat_count > 3:
                return
        else:
            self.last_message = msg
            self.repeat_count = 1

        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        level_name = record.levelname

        color = 'black'
        if record.levelno >= logging.ERROR:
            color = 'red'
        elif record.levelno >= logging.WARNING:
            color = 'orange'
        elif record.levelno >= logging.INFO:
            color = 'blue'

        formatted_msg = f'<font color="{color}">[{timestamp}] [{level_name}] {msg}</font>'

        self.mutex.lock()
        self.message_queue.append(formatted_msg)
        self.mutex.unlock()

        if len(self.message_queue) > 50:
            self.periodic_flush()

    def periodic_flush(self):
        """Periodically flush queued messages to the console."""
        if not self.message_queue:
            return

        current_time = time.time()
        elapsed = current_time - self.last_flush_time

        if elapsed < 0.1 and len(self.message_queue) < 10:
            return

        self.mutex.lock()
        messages = self.message_queue.copy()
        self.message_queue.clear()
        self.mutex.unlock()

        if len(messages) > self.rate_limit:
            tmp = messages[:10]
            tmp.append(f'<font color="purple">... {len(messages) - 20} messages skipped ...</font>')
            tmp.extend(messages[-10:])
            messages = tmp

        html = '<br>'.join(messages)
        self.console.append(html)

        self._trim_console()

        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        self.last_flush_time = current_time

    def _trim_console(self):
        """Trim console text to prevent it from getting too large."""
        text = self.console.toPlainText()
        lines = text.split('\n')

        if len(lines) > self.max_lines:
            lines_to_keep = lines[-self.max_lines:]

            scrollbar = self.console.verticalScrollBar()

            self.console.clear()
            self.console.append('<font color="purple">... earlier messages removed ...</font><br>')
            self.console.append('<br>'.join(lines_to_keep))
