"""
Module for the button styles
"""


def checkbox_style_num_slot():
    return """
        QCheckBox {
            font-size: 16px;
            spacing: 0px;
        }
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
        }
        QCheckBox::indicator:checked {
            background-color: #E6E6FA; 
            border: 2px solid black;
        }
        QCheckBox::indicator:unchecked {
            background-color: white;
            border: 2px solid #ccc;
        }
    """


def checkbox_style():
    return """
        QCheckBox {
            font-size: 16px;
            spacing: 0px;
        }
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
        }
        QCheckBox::indicator:checked {
            background-color: #f0ca41; 
            border: 2px solid black;
        }
        QCheckBox::indicator:unchecked {
            background-color: white;
            border: 2px solid #ccc;
        }
    """

def common_radiobutton_style():
    return """
        QRadioButton {
            spacing: 0px;
            font-size: 16px;
        }
        QRadioButton::indicator {
            width: 20px;
            height: 20px;
        }
        QRadioButton::indicator:checked {
            background-color: #f0ca41;
            border: 2px solid black;
        }
        QRadioButton::indicator:unchecked {
            background-color: white;
            border: 2px solid #ccc;
        }
    """

def settings_button_style():
    return """
        QPushButton {
            font-size: 16px;
            background-color: #b3e5fc; 
            border: 2px solid #8c8c8c;
            border-radius: 10px; 
            padding: 10px;
        }
        QPushButton:hover {
            background-color: #64b5f6; 
        }
    """

def toggle_button_style():
    """Apply a consistent style to toggle buttons."""
    return """
        QPushButton {
            font-size: 16px;
            background-color: #ccffcc;  /* Light Green */
            border: 2px solid #8c8c8c;
            border-radius: 10px;
            padding: 15px;
        }
        QPushButton:hover {
            background-color: #b2f2b2;  /* Slightly darker green when hovered */
        }
    """

def run_button_style():
    return """
        QPushButton {
            font-size: 16px;
            background-color: #ffcc80;
            border: 2px solid #8c8c8c;
            border-radius: 10px;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #ffb74d;
        }
    """

def group_box_style():
    return """
        QGroupBox {
            border: 1px solid black;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 20px;
            font-weight: bold;
        }
        QGroupBox::title {
            font-size: 14px; 
            font-weight: bold;
            subcontrol-origin: margin;
            subcontrol-position: top center;
        }
    """

def checkbox_style_default():
    return """
        QCheckBox {
            spacing: 0px;
            font-size: 16px;
        }
        QCheckBox::indicator {
            width: 25px;
            height: 25px;
            border: 2px solid #ccc;
            background-color: lightcoral; 
        }
        QCheckBox::indicator:checked {
            background-color: #ccffcc; 
            border: 2px solid black;
        }
        QCheckBox::indicator:unchecked {
            background-color: lightcoral;  
            border: 2px solid #ccc;
        }
    """

def checkbox_style_present():
    return """
        QCheckBox {
            spacing: 0px;
            font-size: 16px;
        }
        QCheckBox::indicator {
            width: 25px;
            height: 25px;
            border: 2px solid #ccc;
            background-color: lightblue;  
        }
        QCheckBox::indicator:checked {
            background-color: #ccffcc;  
            border: 2px solid black;
        }
        QCheckBox::indicator:unchecked {
            background-color: lightblue;  
            border: 2px solid #ccc;
        }
    """

def checkbox_style_absent():
    return """
        QCheckBox {
            spacing: 0px;
            font-size: 16px;
        }
        QCheckBox::indicator {
            width: 25px;
            height: 25px;
            border: 2px solid #ccc;
            background-color: lightcoral;  
        }
        QCheckBox::indicator:checked {
            background-color: #ccffcc;  
            border: 2px solid black;
        }
        QCheckBox::indicator:unchecked {
            background-color: lightcoral; 
            border: 2px solid #ccc;
        }
    """

def save_style():
    return"""
        QPushButton {
            background-color: #81C784;  /* Lighter Green */
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 10px;
            border: 2px solid black;  /* Black border */
            transition: background-color 0.3s ease;
        }
        QPushButton:hover {
            background-color: #66BB6A;  /* Slightly darker Green for hover */
        }
        QPushButton:pressed {
            background-color: #388E3C;  /* Darker Green when pressed */
        }
    """

def load_style():
    return"""
        QPushButton {
            background-color: #64B5F6;  /* Lighter Blue */
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 10px;
            border: 2px solid black;  /* Black border */
            transition: background-color 0.3s ease;
        }
        QPushButton:hover {
            background-color: #42A5F5;  /* Slightly darker Blue for hover */
        }
        QPushButton:pressed {
            background-color: #1565C0;  /* Darker Blue when pressed */
        }
    """