"""
GUI for data visualization with a separate console
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QVBoxLayout
from PyQt5.QtCore import QTimer
from PL.Layout.main_window_att import LayoutFrame
from PL.Layout.frames_buttons import ButtonFrame
from PL.Plot.frame_attributes import PlotFrame
import multiprocessing


class MainWindow(QWidget):
    """Main window for data visualization"""

    def __init__(self):

        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("PL Data Visualization")
        self.setStyleSheet("background-color: #F5F5F5;")


        self.canvas_widget = QWidget(self)
        self.canvas_layout = QGridLayout(self.canvas_widget)
        
        self.layout_frame = LayoutFrame(self)
        self.layout_frame.setup_layout(self.canvas_widget, self.canvas_layout)

        self.button_frame = ButtonFrame(self.canvas_layout)
        self.plot_frame = PlotFrame(self.canvas_layout, self.button_frame)


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.layout_frame.adjust_scroll_area_size)
        self.timer.start(200)

        # Set/adapt the maximum window size
        self.layout_frame.set_max_window_size()
        self.layout_frame.position_window_top_left()

    def closeEvent(self, event):
        """Override closeEvent to log when the window is closed."""
        logging.info("Close window event triggered")
        super().closeEvent(event)



def main():
    """Launch the main application in the main process"""
    logging.info("Starting the application")
    multiprocessing.freeze_support()

    # Restore stdout/stderr to ensure they go to the console
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    try:
        logging.info("Running the Qt application")
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"Error while running the application: {e}")
    finally:
        logging.info("Application exited")

if __name__ == "__main__":
    main()
