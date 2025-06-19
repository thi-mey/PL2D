"""
Module for the button styles
"""



def button_style_energy():
    return """
        QPushButton {
            font-size: 16px;
            background-color: #ffcccc;  
            border: 2px solid #8c8c8c;
            border-radius: 10px;
            padding: 5px;
            width: 35px;
            height: 20px;
        }
        QPushButton:hover {
            background-color: #ff9999; 
        }
    """

def button_style_intensity():
    return """
        QPushButton {
            font-size: 16px;
            background-color: #ccffcc; 
            border: 2px solid #8c8c8c;
            border-radius: 10px;
            padding: 5px;
            width: 35px;
            height: 20px;
        }
        QPushButton:hover {
            background-color: #99ff99; 
        }
    """

def button_style_area():
    return """
        QPushButton {
            font-size: 16px;
            background-color: #E0F7FA; 
            border: 2px solid #8c8c8c;
            border-radius: 10px;
            padding: 5px;
            width: 35px;
            height: 20px;
        }
        QPushButton:hover {
            background-color: #B2EBF2;  
        }
    """

def save_button_style():
    return """
        QPushButton {
            font-size: 16px;
            background-color: #e6ccff;  /* Light purple */
            border: 2px solid #8c8c8c;
            border-radius: 10px;
            padding: 10px;
        }
        QPushButton:hover {
            background-color: #d1b3ff;  /* Slightly darker purple */
        }
    """

def button_style_thickness():
    return """
        QPushButton {
            font-size: 16px;
            background-color: #E6E6FA;
            border: 2px solid #8c8c8c;
            border-radius: 10px;
            padding: 5px;
            width: 35px;
            height: 20px;
        }
        QPushButton:hover {
            background-color: #D8BFD8; 
        }
    """

def button_style_top_base():
    return """
        QPushButton {
            font-size: 16px;
            color: black;
            background-color: white;
            border: 2px solid #8c8c8c;
            border-radius: 10px;
            padding: 5px;
            width: 35px;
            height: 20px;
        }
        QPushButton:hover {
            background-color: #d3d3d3;
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