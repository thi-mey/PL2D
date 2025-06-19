"""
Manage the scrollbar and the functions for the window size
"""
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QScrollArea, QVBoxLayout
from PyQt5.QtGui import QGuiApplication


class LayoutFrame:
    """Class for setting up layout and scroll area"""

    def __init__(self, main_window):
        self.main_window = main_window
        self.canvas_widget = None
        self.canvas_layout = None
        self.scroll_area = QScrollArea(main_window)

    def setup_layout(self, canvas_widget, canvas_layout):
        """Set up layout with scroll area and other settings"""
        self.canvas_widget = canvas_widget
        self.canvas_layout = canvas_layout

        # Set up the QScrollArea
        self.scroll_area.setWidget(self.canvas_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Set layout for the main window
        layout = QVBoxLayout(self.main_window)
        layout.addWidget(self.scroll_area)
        self.main_window.setLayout(layout)

    def set_max_window_size(self):
        """Set the maximum window size based on the screen size"""
        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
        max_width = screen_geometry.width()
        max_height = screen_geometry.height() - 50
        self.main_window.setMaximumSize(max_width, max_height)

    def position_window_top_left(self):
        """Position the window at the top-left corner of the screen"""
        screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
        self.main_window.move(screen_geometry.topLeft())

    def adjust_scroll_area_size(self):
        """Adjust the size of the window based on the content"""
        self.canvas_widget.adjustSize()
        optimal_size = self.canvas_widget.sizeHint()

        screen_size = QGuiApplication.primaryScreen().availableSize()
        new_size = QSize(min(optimal_size.width(), screen_size.width()) + 50,
                         min(optimal_size.height(), screen_size.height()) + 50)
        self.main_window.resize(new_size)
