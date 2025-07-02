"""
UI style definitions and theming for the SEM/GDS alignment application.

This module provides consistent styling across all UI components using
Qt Style Sheets (QSS). The styling includes:
- Color schemes and palettes
- Widget-specific styling rules
- Layout and spacing standards
- Icon and image styling
"""

# Color palette for consistent theming
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Pink/Purple  
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red (for emphasis)
    'background': '#F5F5F5',   # Light gray
    'surface': '#FFFFFF',      # White
    'text_primary': '#212121', # Dark gray
    'text_secondary': '#757575', # Medium gray
    'border': '#E0E0E0',       # Light border
    'hover': '#E3F2FD'         # Light blue hover
}

# Main application stylesheet
MAIN_STYLE = f"""
QMainWindow {{
    background-color: {COLORS['background']};
    color: {COLORS['text_primary']};
}}

QMenuBar {{
    background-color: {COLORS['surface']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 4px;
}}

QMenuBar::item {{
    background-color: transparent;
    padding: 6px 12px;
    border-radius: 4px;
}}

QMenuBar::item:selected {{
    background-color: {COLORS['hover']};
}}

QStatusBar {{
    background-color: {COLORS['surface']};
    border-top: 1px solid {COLORS['border']};
    padding: 4px;
}}
"""

# Panel and group box styling
PANEL_STYLE = f"""
QGroupBox {{
    font-weight: bold;
    border: 2px solid {COLORS['border']};
    border-radius: 8px;
    margin: 8px 0px;
    padding-top: 8px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {COLORS['primary']};
}}

QFrame {{
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    background-color: {COLORS['surface']};
}}
"""

# Button styling
BUTTON_STYLE = f"""
QPushButton {{
    background-color: {COLORS['primary']};
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: #1976D2;
}}

QPushButton:pressed {{
    background-color: #1565C0;
}}

QPushButton:disabled {{
    background-color: {COLORS['border']};
    color: {COLORS['text_secondary']};
}}

QPushButton.secondary {{
    background-color: {COLORS['secondary']};
}}

QPushButton.secondary:hover {{
    background-color: #8E1A5C;
}}
"""

# Input control styling
INPUT_STYLE = f"""
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 4px 8px;
    background-color: {COLORS['surface']};
}}

QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
    border-color: {COLORS['primary']};
}}

QSlider::groove:horizontal {{
    border: 1px solid {COLORS['border']};
    height: 4px;
    background: {COLORS['background']};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {COLORS['primary']};
    border: 1px solid {COLORS['primary']};
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
"""

# List and tree styling
LIST_STYLE = f"""
QListWidget, QTreeWidget {{
    border: 1px solid {COLORS['border']};
    background-color: {COLORS['surface']};
    alternate-background-color: {COLORS['background']};
    border-radius: 4px;
}}

QListWidget::item, QTreeWidget::item {{
    padding: 4px;
    border-bottom: 1px solid {COLORS['border']};
}}

QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {COLORS['primary']};
    color: white;
}}

QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {COLORS['hover']};
}}
"""

# Combine all styles
COMPLETE_STYLESHEET = f"""
{MAIN_STYLE}
{PANEL_STYLE}
{BUTTON_STYLE}
{INPUT_STYLE}
{LIST_STYLE}
"""

def apply_theme(widget):
    """
    Apply the complete theme to a widget and its children.
    
    Args:
        widget: QWidget to apply styling to
    """
    widget.setStyleSheet(COMPLETE_STYLESHEET)

def get_color(color_name: str) -> str:
    """
    Get a color value from the theme palette.
    
    Args:
        color_name: Name of the color in the COLORS dictionary
        
    Returns:
        Hex color string
    """
    return COLORS.get(color_name, COLORS['text_primary'])
